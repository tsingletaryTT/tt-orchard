#!/usr/bin/env python3
"""Check a model's sidecar head against the device: do the chips give the head the same inputs the CPU does?

Some models are a known backbone plus an extra head in a file of its own (the first case is
Cloudflare/clef: a Qwen3.8-27B backbone and `joint_head.safetensors`). The weights-swap tests check the
backbone's text output. This script checks what the head needs: the backbone's final hidden state for
every token. The TT serving stack keeps only the last row, so this script runs the model's layer loop
itself (a copy of the loop in `Qwen36Model.prefill_tp`, over all rows), applies the model's own final
norm and reads the rows back. Nothing in tt-metal is changed.

The sidecar-parity skill copies this file into the stage directory. `prepare_parity.py` builds a launcher
(`parity-run.sh`) that runs it in the nearest bundle's python with the server's environment. It reads
`parity_config.json` from its own directory (the stage directory) and writes `evidence/` next to itself.

Steps:
(a) Hash check. `joint_schema_model.py` is code from the model repo. It is imported only after its sha256
    equals `code_sha256` (the value stage 0 recorded). A mismatch exits 5.
(b) CPU reference (phase `cpu`). The model's own Hugging Face backbone in bf16 on the CPU, and the head on
    the CPU, run on the fixed records below. The backbone's `last_hidden_state` and the head's logits are
    saved in `evidence/cpu-reference.pt` and reused when the records and layer limit are the same. The
    backbone then runs again with another torch thread count; the difference between the two runs is the
    noise floor of the reference itself.
(c) Device (phase `device`). Cache guard (exit 3, as in serve_and_compare.py), then the mesh is opened
    (exit 4 when it cannot be) and `Qwen36Model.from_pretrained` loads `model-dir`. For each record the
    layer loop runs over all rows, the model's final norm is applied, and one replica is read back.
(d) Wiring check. For record 0, `prefill_tp` gives the model's own last-row logits. The script applies the
    output-embedding matrix on the host to its own last normed row and takes the cosine with those logits.
    Below `wiring_cosine_min` the copied loop no longer matches the model and the script exits 6.
(e) The head runs on the host on the device's hidden states, and the probabilities are compared with the
    reference's. `evidence/sidecar-parity.json` holds the numbers.

Exit codes: 0 measured, 2 bad configuration, 3 cache guard, 4 the device could not be opened, 5 the code
hash does not match, 6 the wiring check failed (the evidence file is still written, with passed false).

The records are fixed in this file and text only. The agent never writes them. A real model's head may
behave differently on other inputs; the check says the device and the CPU agree on these.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

HERE_DIR = Path(__file__).resolve().parent
EXIT_OK, EXIT_BAD_CONFIG, EXIT_CACHE, EXIT_DEVICE, EXIT_HASH, EXIT_WIRING = 0, 2, 3, 4, 5, 6
MARKER = ".orchard-model"
PAD_MULTIPLE = 128                # the GDN chunk kernel wants the sequence padded to this
WIRING_COSINE_MIN = 0.99          # choice: the copied loop and prefill_tp should agree far above this
CPU_THREADS = 16                  # the reference's torch threads; the noise run uses NOISE_THREADS
NOISE_THREADS = 8
L1_SMALL_SIZE = 24576             # GDN_CONV1D_L1_SMALL_SIZE in the model's config
MIN_MAX_SEQ_LEN = 2048
REQUIRED_KEYS = ("run_dir", "model_snapshot", "head_file", "head_config", "code_file", "code_sha256", "tt_cache")


class CodeHashMismatch(ValueError):
    """The sidecar code file is missing or does not have the expected sha256."""


# ---- the fixed records ---------------------------------------------------------------------------

def build_records() -> list[dict]:
    """Six text-only records in the sidecar's input format. Fixed here so every run, on every model, asks the
    same questions: the three question types, short and long states, and options that are close in meaning."""
    return [
        {"id": "outage",
         "state": "Our checkout started returning errors about ten minutes ago and orders are blocked. "
                  "Customers see a 500 page when they press Pay.",
         "questions": {
             "department": {"type": "choice", "instructions": "Which team should handle this message?",
                            "criteria": {"billing": "Payments, invoices and refunds", "technical": "Bugs and outages",
                                         "sales": "New business and pricing"}},
             "urgency": {"type": "score", "instructions": "How urgent is this?",
                         "criteria": ["Can wait a week", "Handle this week", "Handle today", "Drop everything"]},
             "service_down": {"type": "noul", "instructions": "Is a service down?"}}},
        {"id": "invoice",
         "state": {"invoice": {"vendor": "Acme Tools", "total": 1250.0, "currency": "USD", "status": "overdue",
                               "days_late": 21}},
         "questions": {
             "status": {"type": "choice", "instructions": "What is the invoice status?",
                        "criteria": {"paid": "The invoice is paid.", "overdue": "The invoice is past due.",
                                     "draft": "The invoice has not been sent."}},
             "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"}}},
        {"id": "review",
         "state": "I bought the blue kettle in March. It boils fast and looks great, but the lid cracked after "
                  "two months and support never answered my emails. I would not buy it again.",
         "questions": {
             "sentiment": {"type": "score", "instructions": "How positive is this review?",
                           "criteria": ["Very negative", "Negative", "Mixed", "Positive", "Very positive"]},
             "mentions_support": {"type": "noul", "instructions": "Does the review mention customer support?"},
             "topic": {"type": "choice", "instructions": "What is the main complaint?",
                       "criteria": {"durability": "The product broke or wore out.",
                                    "service": "Support was slow or missing.",
                                    "price": "The product costs too much.",
                                    "performance": "The product does not do its job."}}}},
        {"id": "access",
         "state": {"request": {"user": "j.rivera", "resource": "payroll-db", "role": "read-write",
                               "reason": "Quarter close", "manager_approved": True, "mfa_enrolled": False}},
         "questions": {
             "decision": {"type": "choice", "instructions": "Should this access request be granted?",
                          "criteria": {"grant": "Approve as requested.", "deny": "Refuse the request.",
                                       "escalate": "Send to the security team for review."}},
             "needs_mfa": {"type": "noul", "instructions": "Does the user still need to enrol in MFA?"}}},
        {"id": "support-thread",
         "state": ("Customer: I changed my address last week but the package went to the old one. Agent: I am "
                   "sorry about that. Can you confirm the order number? Customer: It is 88213. Agent: Thank you. "
                   "I can see the address was updated after the label was printed. I will ask the carrier to "
                   "redirect it. Customer: How long will that take? Agent: Usually two to three business days. "
                   "Customer: That is too long, the gift is for a birthday on Friday. Agent: I understand. I can "
                   "send a replacement with express shipping at no charge. Customer: Please do. "
                   "Agent: Done. You will get a tracking number within the hour."),
         "questions": {
             "resolved": {"type": "noul", "instructions": "Did the agent resolve the customer's problem?"},
             "satisfaction": {"type": "score", "instructions": "How satisfied is the customer at the end?",
                              "criteria": ["Angry", "Unhappy", "Neutral", "Satisfied", "Delighted"]},
             "root_cause": {"type": "choice", "instructions": "What caused the problem?",
                            "criteria": {"address": "The address change came too late for the label.",
                                         "carrier": "The carrier lost the package.",
                                         "stock": "The item was out of stock."}}}},
        {"id": "deploy",
         "state": {"change": {"service": "search-api", "lines_changed": 48, "touches": ["ranking", "cache"],
                              "tests_passed": True, "peak_traffic_window": False}},
         "questions": {
             "risk": {"type": "score", "instructions": "How risky is this deployment?",
                      "criteria": ["Negligible", "Low", "Medium", "High"]},
             "needs_rollback_plan": {"type": "noul", "instructions": "Should the author write a rollback plan?"},
             "window": {"type": "choice", "instructions": "When should it ship?",
                        "criteria": {"now": "Ship immediately.", "off_peak": "Ship in the next off-peak window.",
                                     "freeze": "Hold until the freeze ends."}}}},
    ]


def validate_record(record: dict) -> list[str]:
    """The sidecar's own refusals, checked here so a bad fixed record fails in a test and not on the board."""
    problems = []
    if "state" not in record:
        problems.append("the record has no state")
    for media in ("images", "videos", "media_kwargs"):
        if media in record:
            problems.append(f"the record has {media}; these records are text only")
    questions = record.get("questions")
    if not isinstance(questions, dict) or not questions:
        problems.append("the record needs at least one question")
        return problems
    for qid, q in questions.items():
        if q.get("type") not in ("noul", "choice", "score"):
            problems.append(f"{qid}: type must be noul, choice or score")
        elif q["type"] != "noul" and not q.get("criteria"):
            problems.append(f"{qid}: criteria must not be empty")
        elif q["type"] == "choice" and not isinstance(q["criteria"], dict):
            problems.append(f"{qid}: choice criteria must be a mapping")
        elif q["type"] == "score" and not isinstance(q["criteria"], list):
            problems.append(f"{qid}: score criteria must be a list")
    return problems


def records_digest(records: list[dict], n_layers) -> str:
    """Identifies a set of records and a layer limit, so a saved CPU reference is reused only for the same."""
    blob = json.dumps({"records": records, "n_layers": n_layers}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---- the code hash and the cache guard ------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_code_hash(path, expected: str) -> None:
    if not expected:
        raise CodeHashMismatch("code_sha256 is empty; stage 0 must record the code file's sha256")
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise CodeHashMismatch(f"cannot read the sidecar code {path}: {exc}") from exc
    if actual != expected:
        raise CodeHashMismatch(f"{path} has sha256 {actual}, expected {expected}")


def guard_cache(cache: Path, model_id: str) -> None:
    """Exit 3 for a non-empty tensor cache whose marker is missing or names another model. A cache is keyed
    by layer name only, so another model's cache would be read without an error and serve that model's
    weights. A missing or empty cache is created and marked."""
    cache = Path(cache)
    if cache.is_dir() and any(cache.iterdir()):
        marker = cache / MARKER
        found = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
        if found != model_id:
            what = "has no marker file" if found is None else f"is marked for {found!r}"
            print(f"hidden_parity: the tensor cache {cache} is not empty and {what} ({MARKER} must contain "
                  f"{model_id!r}). Use a new empty directory for tt_cache.", file=sys.stderr)
            sys.exit(EXIT_CACHE)
        return
    cache.mkdir(parents=True, exist_ok=True)
    (cache / MARKER).write_text(model_id, encoding="utf-8")


# ---- the math ---------------------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def pcc(a, b) -> float:
    """Pearson correlation of two tensors of one shape, over all elements, in float64."""
    torch = _torch()
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"shapes differ: {tuple(a.shape)} and {tuple(b.shape)}")
    x, y = a.reshape(-1).to(torch.float64), b.reshape(-1).to(torch.float64)
    x, y = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denom == 0.0:
        return 1.0 if torch.equal(a.to(torch.float64), b.to(torch.float64)) else 0.0
    return float((x * y).sum()) / denom


def cosine(a, b) -> float:
    torch = _torch()
    x, y = a.reshape(-1).to(torch.float64), b.reshape(-1).to(torch.float64)
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum()) / denom if denom else 0.0


def probs_from_logits(logits_per_record) -> list[list[list[float]]]:
    """[record][question] logits -> [record][question] probabilities (softmax over each question's options)."""
    torch = _torch()
    return [[torch.softmax(q.float(), dim=-1).tolist() for q in record] for record in logits_per_record]


def max_abs_prob_diff(a, b) -> float:
    return max(abs(x - y) for ra, rb in zip(a, b) for qa, qb in zip(ra, rb) for x, y in zip(qa, qb))


def record_metrics(record_id: str, ref_hidden, dev_hidden, ref_probs, dev_probs) -> dict:
    agree = sum(1 for qa, qb in zip(ref_probs, dev_probs)
                if max(range(len(qa)), key=qa.__getitem__) == max(range(len(qb)), key=qb.__getitem__))
    return {"id": record_id, "tokens": int(ref_hidden.shape[0]), "hidden_pcc": pcc(ref_hidden, dev_hidden),
            "prob_max_abs_diff": max_abs_prob_diff([ref_probs], [dev_probs]), "top1_agree": agree,
            "n_questions": len(ref_probs)}


def _margin(probs: list[float]) -> float:
    """Top probability minus the runner-up (the top probability alone for a single option)."""
    ordered = sorted(probs, reverse=True)
    return ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)


def question_details(questions, ref_probs, dev_probs) -> list[dict]:
    """One entry per question: its options, the reference's and the device's probabilities, whether the top
    choice agrees and how clear each side's top choice was. A disagreement with a small margin on both sides is
    a near tie; one with a large margin on either side is not."""
    if not (len(questions) == len(ref_probs) == len(dev_probs)):
        raise ValueError("questions and probability lists have different lengths")
    out = []
    for q, ref, dev in zip(questions, ref_probs, dev_probs):
        top = lambda p: max(range(len(p)), key=p.__getitem__)
        out.append({"id": q.question_id, "options": list(q.option_ids), "ref": ref, "dev": dev,
                    "agree": top(ref) == top(dev), "ref_margin": _margin(ref), "dev_margin": _margin(dev)})
    return out


def pad_length(n: int) -> int:
    return -(-n // PAD_MULTIPLE) * PAD_MULTIPLE


def assemble_hidden(t, n_devices: int, hidden_size: int, valid_len: int):
    """The rows read back from the mesh as [n_devices, 1, T, d] -> [valid_len, hidden_size]. The model's
    DistributedNorm gathers after the norm, so each replica holds the full width (d == hidden_size) and one
    replica is taken. If instead each device holds a slice (d * n_devices == hidden_size) the slices are
    joined along the width. Any other width is an error."""
    torch = _torch()
    d = t.shape[-1]
    if d == hidden_size:
        return t[0, 0, :valid_len]
    if d * n_devices == hidden_size:
        return torch.cat([t[i, 0, :valid_len] for i in range(n_devices)], dim=-1)
    raise ValueError(f"the device returned width {d}; expected {hidden_size} or {hidden_size // n_devices}")


def wiring_check(hidden_row, weight, tt_logits, minimum: float) -> dict:
    """Cosine between the host's logits for the script's own last row and the model's own prefill logits."""
    torch = _torch()
    host_logits = weight.to(torch.float32) @ hidden_row.to(torch.float32)
    c = cosine(host_logits, tt_logits.reshape(-1)[: host_logits.shape[0]])
    return {"cosine": c, "passed": bool(c >= minimum), "minimum": minimum}


# ---- the result -------------------------------------------------------------------------------------

def aggregate(per_record: list[dict], noise_floor: dict, wiring: dict, meta: dict) -> dict:
    if not per_record:
        raise ValueError("no records were measured")
    pccs = [r["hidden_pcc"] for r in per_record]
    n_questions = sum(r["n_questions"] for r in per_record)
    out = {
        "measured": True, "n_records": len(per_record), "n_questions": n_questions,
        "hidden_pcc_min": min(pccs), "hidden_pcc_mean": sum(pccs) / len(pccs),
        "prob_max_abs_diff": max(r["prob_max_abs_diff"] for r in per_record),
        "top1_agree_fraction": sum(r["top1_agree"] for r in per_record) / n_questions,
        "noise_floor": dict(noise_floor), "wiring_check": dict(wiring),
        **meta, "per_record": per_record,
    }
    out["label"] = {k: "measured" for k in ("hidden_pcc_min", "hidden_pcc_mean", "prob_max_abs_diff",
                                           "top1_agree_fraction")}
    return out


def validate_result(d: dict) -> list[str]:
    problems = []

    def number(key, low=None, high=None):
        v = d.get(key)
        if key not in d:
            problems.append(f"{key} is missing")
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            problems.append(f"{key} must be a finite number, got {v!r}")
        elif (low is not None and v < low) or (high is not None and v > high):
            problems.append(f"{key} is out of range: {v!r}")

    if d.get("measured") is not True:
        problems.append("measured must be true")
    number("n_records", 1)
    number("n_questions", 1)
    number("hidden_pcc_min", -1.0, 1.0 + 1e-6)
    number("hidden_pcc_mean", -1.0, 1.0 + 1e-6)
    number("prob_max_abs_diff", 0.0, 1.0 + 1e-6)
    number("top1_agree_fraction", 0.0, 1.0)
    nf = d.get("noise_floor")
    if not isinstance(nf, dict) or not isinstance(nf.get("prob_max_abs_diff"), (int, float)):
        problems.append("noise_floor.prob_max_abs_diff is missing")
    w = d.get("wiring_check")
    if not isinstance(w, dict) or not isinstance(w.get("passed"), bool) or not isinstance(w.get("cosine"), (int, float)):
        problems.append("wiring_check needs a numeric cosine and a boolean passed")
    for key in ("code_sha256", "head_sha256"):
        if not isinstance(d.get(key), str) or len(d[key]) != 64:
            problems.append(f"{key} must be a 64-character sha256")
    ms = d.get("mesh_shape")
    if not (isinstance(ms, list) and len(ms) == 2 and all(isinstance(x, int) for x in ms)):
        problems.append("mesh_shape must be a list of two integers")
    if not isinstance(d.get("tt_metal_sha"), str) or not d.get("tt_metal_sha"):
        problems.append("tt_metal_sha must be a non-empty string")
    return problems


def write_evidence(evidence_dir, result: dict) -> Path:
    problems = validate_result(result)
    if problems:
        raise ValueError("the result does not match its schema: " + "; ".join(problems))
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / "sidecar-parity.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# ---- the output-embedding weight and the saved reference ---------------------------------------------

EMBEDDING_NAMES = ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")


def output_embedding_name(weight_map: dict, tie: bool) -> str:
    """The tensor the head takes as the output embedding: lm_head.weight, or the input embedding when the
    model ties them."""
    if not tie and "lm_head.weight" in weight_map:
        return "lm_head.weight"
    for name in EMBEDDING_NAMES:
        if name in weight_map:
            return name
    raise KeyError("neither lm_head.weight nor an embed_tokens weight is in the weight index")


def load_tensor(snapshot, name: str):
    from safetensors import safe_open
    snapshot = Path(snapshot)
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    if name not in weight_map:
        raise KeyError(f"{name} is not in the weight index of {snapshot}")
    with safe_open(str(snapshot / weight_map[name]), framework="pt") as f:
        return f.get_tensor(name)


def save_reference(path, digest: str, ref: dict) -> None:
    _torch().save({"digest": digest, **ref}, str(path))


def load_reference(path, digest: str):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = _torch().load(str(path), map_location="cpu", weights_only=False)
    except Exception:                       # a torn or foreign file is not a reference
        return None
    return data if isinstance(data, dict) and data.get("digest") == digest else None


# ---- the configuration --------------------------------------------------------------------------------

def model_label(snapshot) -> str:
    """`.../models--Cloudflare--clef/snapshots/<rev>` -> `Cloudflare/clef@<rev>`, the label stage 0 and the
    swap template put in the cache marker. Any other path is its own label."""
    p = Path(str(snapshot))
    if p.parent.name == "snapshots" and p.parent.parent.name.startswith("models--"):
        return p.parent.parent.name[len("models--"):].replace("--", "/", 1) + "@" + p.name
    return str(snapshot)


def _bad(message: str) -> None:
    print(f"hidden_parity: {message}", file=sys.stderr)
    sys.exit(EXIT_BAD_CONFIG)


def read_config(path) -> dict:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _bad(f"cannot read {path}: {exc}")
    for key in REQUIRED_KEYS:
        if key not in cfg:
            _bad(f"parity_config.json is missing {key}")
    for key in ("run_dir", "model_snapshot", "head_file", "head_config", "code_file", "tt_cache"):
        if not isinstance(cfg[key], str) or not cfg[key]:
            _bad(f"parity_config.json: {key} must be a non-empty string")
    h = cfg["code_sha256"]
    if not (isinstance(h, str) and len(h) == 64):
        _bad("parity_config.json: code_sha256 must be a 64-character sha256")
    ms = cfg.setdefault("mesh_shape", [1, 2])
    if not (isinstance(ms, list) and len(ms) == 2 and all(isinstance(x, int) and x > 0 for x in ms)):
        _bad("parity_config.json: mesh_shape must be a list of two positive integers, such as [1, 2]")
    n = cfg.setdefault("n_layers", None)
    if n is not None and (isinstance(n, bool) or not isinstance(n, int) or n < 1):
        _bad("parity_config.json: n_layers must be null or a positive integer")
    cfg.setdefault("model_id", model_label(cfg["model_snapshot"]))
    cfg.setdefault("wiring_cosine_min", WIRING_COSINE_MIN)
    cfg.setdefault("cpu_threads", CPU_THREADS)
    cfg.setdefault("noise_threads", NOISE_THREADS)
    return cfg


# ---- the parts that need the model, the CPU stack or the board ------------------------------------------

def import_sidecar(code_path: Path):
    """Import the sidecar's own code. Only called after the hash check."""
    spec = importlib.util.spec_from_file_location("joint_schema_model", str(code_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_head(sidecar, cfg: dict):
    from safetensors.torch import load_file
    torch = _torch()
    snap = Path(cfg["model_snapshot"])
    head = sidecar.JointSchemaHead(**json.loads((snap / cfg["head_config"]).read_text(encoding="utf-8")))
    head.load_state_dict(load_file(str(snap / cfg["head_file"])), strict=True)
    return head.to(dtype=torch.bfloat16).eval()


def say(message: str) -> None:
    print(f"hidden_parity: {message}", flush=True)


def compute_cpu_reference(cfg: dict, records: list[dict], sidecar, tokenizer) -> dict:
    """Backbone and head on the CPU in bf16, twice with different thread counts (the second run is the noise
    floor). Returns the dict saved in cpu-reference.pt."""
    torch = _torch()
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
    snap = Path(cfg["model_snapshot"])
    config = AutoConfig.from_pretrained(str(snap))
    tc = config.get_text_config()
    n = cfg["n_layers"]
    if n:
        tc.num_hidden_layers = n
        if getattr(tc, "layer_types", None):
            tc.layer_types = list(tc.layer_types)[:n]
    tie = bool(getattr(config, "tie_word_embeddings", False) or getattr(tc, "tie_word_embeddings", False))
    weight_map = json.loads((snap / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    emb_name = output_embedding_name(weight_map, tie)
    say(f"loading the CPU backbone in bf16 ({n or 'all'} layers)")
    t0 = time.time()
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(str(snap), config=config, dtype=torch.bfloat16)
    backbone.eval()
    say(f"loaded in {time.time() - t0:.0f} s")
    text_model = backbone.model.language_model if hasattr(backbone.model, "language_model") else backbone.model
    head = load_head(sidecar, cfg)
    weight = backbone.get_output_embeddings().weight.detach()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    encoded = [sidecar.encode_record(tokenizer, r) for r in records]

    def run(threads: int):
        torch.set_num_threads(threads)
        hidden, logits = [], []
        with torch.inference_mode():
            for r, enc in zip(records, encoded):
                batch = sidecar.collate_records([enc], pad_id, torch.device("cpu"))
                t1 = time.time()
                out = text_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                 use_cache=False, return_dict=True)
                h = out.last_hidden_state
                lg = head(h, batch["input_ids"], batch["attention_mask"], batch["records"], weight)[0]
                hidden.append(h[0].clone())
                logits.append([q.float().clone() for q in lg])
                say(f"  cpu {r['id']}: {h.shape[1]} tokens, {time.time() - t1:.0f} s, {threads} threads")
        return hidden, logits

    hidden, logits = run(cfg["cpu_threads"])
    _, noise_logits = run(cfg["noise_threads"])
    del backbone, text_model, weight
    gc.collect()
    return {"hidden": hidden, "logits": logits, "noise_logits": noise_logits, "embedding_name": emb_name,
            "input_ids": [list(e.input_ids) for e in encoded]}


def device_hidden(model, ttnn, torch, Mode, token_ids, valid_len: int, n_devices: int, hidden_size: int):
    """The model's layer loop over all rows, then its final norm. A copy of `Qwen36Model.prefill_tp` up to the
    one-hot row select, which is skipped. No lm head. token_ids is [1, T] with T a multiple of 128."""
    T = token_ids.shape[1]
    model.reset_tp()
    model._build_request_rope(token_ids[:, :valid_len], None)
    rep = ttnn.ReplicateTensorToMesh(model.device)
    tok = ttnn.from_torch(token_ids.to(torch.int32), dtype=ttnn.uint32, device=model.device, mesh_mapper=rep)
    x = model.embd(tok)
    x = model._scatter_vision_tokens(x, token_ids, None)
    x = ttnn.reshape(x, (1, 1, T, x.shape[-1]))
    cos_t, sin_t = model._rope_tp_cos_sin_torch(0, T)
    cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=model.device, mesh_mapper=rep)
    sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=model.device, mesh_mapper=rep)
    for layer in model.layers:
        x = layer.forward(x, cos=cos, sin=sin, mode="prefill", chunk_size=128, valid_len=valid_len)
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    h = model.norm(x, mode=Mode.PREFILL)
    t = ttnn.to_torch(h, mesh_composer=ttnn.ConcatMeshToTensor(model.device, dim=0))
    return assemble_hidden(t, n_devices, hidden_size, valid_len).clone()


def runtime_versions() -> dict:
    from importlib import metadata
    out = {}
    for pkg in ("ttnn", "torch", "transformers"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def run_device(cfg: dict, records: list[dict], ref: dict, sidecar, tokenizer, evidence: Path) -> int:
    torch = _torch()
    snap = Path(cfg["snapshot_dir"])                 # model-dir: the clean backbone directory
    guard_cache(Path(cfg["tt_cache"]), cfg["model_id"])
    try:
        import ttnn
        shape = cfg["mesh_shape"]
        if shape[0] * shape[1] > 1:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), l1_small_size=L1_SMALL_SIZE)
    except Exception as exc:
        print(f"hidden_parity: the mesh device could not be opened: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_DEVICE
    n_devices = shape[0] * shape[1]
    try:
        from models.demos.blackhole.qwen36.tt.model import Qwen36Model
        from models.tt_transformers.tt.common import Mode
        encoded = [sidecar.encode_record(tokenizer, r) for r in records]
        lengths = [len(e.input_ids) for e in encoded]
        max_seq = max(MIN_MAX_SEQ_LEN, -(-pad_length(max(lengths)) // 2048) * 2048)
        say(f"loading Qwen36Model on a {shape[0]}x{shape[1]} mesh (max_seq_len {max_seq})")
        t0 = time.time()
        model = Qwen36Model.from_pretrained(mesh, max_batch_size=1, max_seq_len=max_seq, n_layers=cfg["n_layers"])
        load_s = time.time() - t0
        say(f"model loaded in {load_s:.0f} s")
        hidden_size = model.args.dim
        head = load_head(sidecar, cfg)
        weight = load_tensor(snap, ref["embedding_name"]).to(torch.bfloat16)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        dev_hidden, device_s = [], []
        for r, enc, n in zip(records, encoded, lengths):
            padded = torch.full((1, pad_length(n)), pad_id, dtype=torch.long)
            padded[0, :n] = torch.tensor(enc.input_ids)
            t1 = time.time()
            dev_hidden.append(device_hidden(model, ttnn, torch, Mode, padded, n, n_devices, hidden_size))
            device_s.append(time.time() - t1)
            say(f"  device {r['id']}: {n} tokens, {device_s[-1]:.1f} s")
        # (d) wiring check on record 0: the model's own prefill logits against logits from our own row.
        n0 = lengths[0]
        padded0 = torch.full((1, pad_length(n0)), pad_id, dtype=torch.long)
        padded0[0, :n0] = torch.tensor(encoded[0].input_ids)
        model.reset_tp()
        tt_logits = model.prefill_tp(padded0, valid_len=n0)
        wiring = wiring_check(dev_hidden[0][n0 - 1], weight, tt_logits.float(), cfg["wiring_cosine_min"])
        say(f"wiring check: cosine {wiring['cosine']:.5f} (minimum {wiring['minimum']})")
        # (e) the head on the host, on the device's hidden states.
        dev_logits = []
        with torch.inference_mode():
            for enc, h in zip(encoded, dev_hidden):
                batch = sidecar.collate_records([enc], pad_id, torch.device("cpu"))
                dev_logits.append(head(h.unsqueeze(0).to(torch.bfloat16), batch["input_ids"],
                                       batch["attention_mask"], batch["records"], weight)[0])
        ref_probs, dev_probs = probs_from_logits(ref["logits"]), probs_from_logits(dev_logits)
        per = [record_metrics(r["id"], ref["hidden"][i].float(), dev_hidden[i].float(), ref_probs[i], dev_probs[i])
               for i, r in enumerate(records)]
        for i, row in enumerate(per):
            row["questions"] = question_details(encoded[i].questions, ref_probs[i], dev_probs[i])
        noise ={"prob_max_abs_diff": max_abs_prob_diff(ref_probs, probs_from_logits(ref["noise_logits"]))}
        meta = {"tt_metal_sha": runtime_versions()["ttnn"], "versions": runtime_versions(),
                "code_sha256": sha256_file(Path(cfg["model_snapshot"]) / cfg["code_file"]),
                "head_sha256": sha256_file(Path(cfg["model_snapshot"]) / cfg["head_file"]),
                "mesh_shape": list(shape), "n_layers": cfg["n_layers"], "model_load_s": round(load_s, 1),
                "device_seconds_per_record": [round(s, 2) for s in device_s],
                "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 1)}
        result = aggregate(per, noise, wiring, meta)
        path = write_evidence(evidence, result)
        say(f"wrote {path}")
        say(f"hidden_pcc min {result['hidden_pcc_min']:.5f} mean {result['hidden_pcc_mean']:.5f}; "
            f"prob_max_abs_diff {result['prob_max_abs_diff']:.5f} (noise floor {noise['prob_max_abs_diff']:.5f}); "
            f"top1 agreement {result['top1_agree_fraction']:.3f} over {result['n_questions']} questions")
        return EXIT_OK if wiring["passed"] else EXIT_WIRING
    finally:
        try:
            for sub in mesh.get_submeshes():
                ttnn.close_mesh_device(sub)
            ttnn.close_mesh_device(mesh)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception as exc:                         # closing is best effort; the lease release resets
            print(f"hidden_parity: closing the mesh: {type(exc).__name__}: {exc}", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=("all", "cpu", "device"), default="all")
    args = ap.parse_args(argv)
    cfg = read_config(HERE_DIR / "parity_config.json")
    snap = Path(cfg["model_snapshot"])
    try:
        verify_code_hash(snap / cfg["code_file"], cfg["code_sha256"])
    except CodeHashMismatch as exc:
        print(f"hidden_parity: {exc}", file=sys.stderr)
        return EXIT_HASH
    records = build_records()
    for r in records:
        problems = validate_record(r)
        if problems:
            _bad(f"record {r.get('id')}: {'; '.join(problems)}")
    evidence = HERE_DIR / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    cfg["snapshot_dir"] = str(HERE_DIR / "model-dir") if (HERE_DIR / "model-dir").is_dir() else str(snap)
    sidecar = import_sidecar(snap / cfg["code_file"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snap))
    digest = records_digest(records, cfg["n_layers"])
    ref = load_reference(evidence / "cpu-reference.pt", digest)
    if ref is None and args.phase in ("all", "cpu"):
        ref = compute_cpu_reference(cfg, records, sidecar, tokenizer)
        save_reference(evidence / "cpu-reference.pt", digest, ref)
        say("wrote evidence/cpu-reference.pt")
    elif ref is not None:
        say("reusing evidence/cpu-reference.pt")
    if args.phase == "cpu":
        return EXIT_OK
    if ref is None:
        _bad("phase device needs evidence/cpu-reference.pt for these records; run phase cpu first")
    return run_device(cfg, records, ref, sidecar, tokenizer, evidence)


if __name__ == "__main__":
    sys.exit(main())
