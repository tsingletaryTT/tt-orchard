#!/usr/bin/env python3
"""Build stage 1's CPU reference of the new model, check it, and draft reference.json.

The reference-gate skill copies this file into the stage directory and runs it with the
interpreter named in `reference_config.json` (next to this file):

    {"run_dir": "<the run directory>",
     "model_snapshot": "<the new model's Hugging Face snapshot directory>",
     "python": "<the interpreter that runs this script>"}

torch and transformers must be importable by that interpreter. If either is missing the script
exits 4 and names the package.

What it does, in order. Each step writes its evidence file before the next step starts, and every
step prints progress lines, because a 27B model on this CPU takes minutes:

1. Tokenizer round trip: 14 fixed strings (Latin, accented, emoji, CJK, Devanagari, Thai,
   punctuation, whitespace) are encoded and decoded back. -> evidence/tokenizer-roundtrip.json
2. Load: the weights load on CPU in bf16 with low_cpu_mem_usage, through the class the config's
   `architectures` names (AutoModelForCausalLM when transformers has no such class). The report
   lists missing, unexpected and mismatched keys. Checkpoint keys that match the class's
   `_keys_to_ignore_on_load_unexpected` patterns (for example `mtp.*`) are not counted as
   unexpected; the report lists them as ignored and the note says so. -> evidence/load-report.json
3. Decode: PROMPT is rendered through the chat template (the raw prompt when the tokenizer has
   none) and decoded greedily for 32 new tokens, one full forward pass per token with no KV
   cache. A separate forward pass over the prompt alone gives the argmax of the last prompt
   position, which must equal the first generated token. Each step's sequence must be the
   previous one with the chosen token appended at the end (`appended_at_end`).
   -> evidence/reference/prompt-ids.json (prompt_ids, prompt_text),
      evidence/reference/generated-ids.json (generated_ids, generated_text; rewritten after every
      step with "complete": false until all 32 tokens exist), evidence/reference/top5-logits.json,
      evidence/decode.txt
4. Thai: when the tokenizer has tokens that decode to Thai characters, THAI_PROMPT is decoded the
   same way. -> evidence/reference/thai-*.json and evidence/thai-decode.txt
5. Card check: a form check only. It records whether the generated text is non-empty and whether
   it begins with reasoning ("Let me", "We need", "<think"). The script does not read the card's
   claims. -> evidence/card-check.txt
6. evidence/summary.json and a DRAFT reference.json in the schema the stage 1 gate reads.

Stage 2 reads prompt-ids.json and generated-ids.json and needs 32 generated ids.

Exit codes: 0 when reference.json was written after a load (its verdict may be "fail"), 2 when
reference_config.json is missing or incomplete or the stage directory is not inside run_dir,
3 when the weights did not load (reference.json is still written, with verdict "fail"), 4 when
torch or transformers is not importable. The script writes only inside its own directory.
"""
from __future__ import annotations

import json
import os
import platform
import re
import struct
import sys
import time
from pathlib import Path

HERE_DIR = Path(__file__).resolve().parent
EVIDENCE = HERE_DIR / "evidence"
REFERENCE = EVIDENCE / "reference"
CONFIG_KEYS = ("run_dir", "model_snapshot", "python")

PROMPT = "Write the text I send my landlord about the broken boiler."
THAI_PROMPT = "เขียนข้อความถึงเจ้าของบ้านเรื่องหม้อต้มน้ำร้อนเสีย"
THAI_SAMPLE = "สวัสดีครับ ภาษาไทย"
N_NEW = 32
TOP_K = 5
REASONING_STARTS = ("Let me", "We need", "<think")

ROUNDTRIP_STRINGS = [
    "Hello, world.",
    "The quick brown fox jumps over the lazy dog.",
    "café naïve résumé façade Zürich",
    "I love it 😀👍🏽 ❤️",
    "你好，世界。我把合同发给房东了。",
    "日本語のテキストです。",
    "नमस्ते दुनिया, यह एक वाक्य है।",
    "สวัสดีครับ เขียนข้อความถึงเจ้าของบ้าน",
    "안녕하세요 세계",
    "Punctuation: (a), [b], {c}; \"d\" 'e' -- f... ?!",
    "Tabs\tand\nnewlines\r\nmixed",
    "   leading and trailing spaces   ",
    "Numbers 0 12 345 6789 3.14159 -42",
    "def main(argv=None) -> int:\n    return 0",
]


def say(message: str) -> None:
    print(f"reference_gate: {message}", flush=True)


def fail(message: str, code: int = 2) -> None:
    print(f"reference_gate: {message}", file=sys.stderr, flush=True)
    sys.exit(code)


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def load_config() -> dict:
    path = HERE_DIR / "reference_config.json"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        fail(f"cannot read {path}: {exc}")
    if not isinstance(cfg, dict):
        fail(f"{path} is not a JSON object")
    missing = [k for k in CONFIG_KEYS if not isinstance(cfg.get(k), str) or not cfg[k].strip()]
    if missing:
        fail(f"reference_config.json is missing {missing}")
    run_dir = os.path.realpath(cfg["run_dir"])
    if os.path.commonpath([run_dir, str(HERE_DIR)]) != run_dir:
        fail(f"this script's directory {HERE_DIR} is not inside run_dir {run_dir}; copy it into "
             "the stage directory and run it from there")
    return cfg


def import_stack():
    """(torch, transformers), or exit 4 naming the package that is missing."""
    try:
        import torch
    except ImportError as exc:
        fail(f"torch is not importable by {sys.executable} ({exc}). Run this script with an "
             "interpreter that has torch and transformers, and name it in reference_config.json.", 4)
    try:
        import transformers
    except ImportError as exc:
        fail(f"transformers is not importable by {sys.executable} ({exc}). Run this script with an "
             "interpreter that has torch and transformers, and name it in reference_config.json.", 4)
    return torch, transformers


# ---- pure helpers (tested with fakes) -------------------------------------------------------------

def appended_at_end(prompt_ids: list[int], steps: list[dict]) -> tuple[bool, str]:
    """Each step's sequence must continue from the previous step's and gain exactly its chosen
    token, at the end. `steps` holds {"chosen", "before", "after"} per step."""
    prev = list(prompt_ids)
    for i, s in enumerate(steps):
        if list(s["before"]) != prev:
            return False, f"step {i} did not start from the previous step's sequence"
        if list(s["after"]) != prev + [s["chosen"]]:
            return False, (f"step {i}: the sequence after the step is not the previous sequence with "
                           f"token {s['chosen']} appended at the end")
        prev = list(s["after"])
    return True, f"all {len(steps)} tokens were appended at the end, one per step"


def reasoning_start(text: str) -> str | None:
    """The reasoning opener the text begins with, or None."""
    stripped = text.lstrip()
    return next((r for r in REASONING_STARTS if stripped.startswith(r)), None)


def split_ignored(keys, patterns) -> tuple[list[str], list[str]]:
    """(keys matching any pattern, the other keys). transformers matches these with re.search."""
    pats = [re.compile(p) for p in (patterns or [])]
    ignored = [k for k in keys if any(p.search(k) for p in pats)]
    return ignored, [k for k in keys if k not in ignored]


def checkpoint_keys(snapshot: Path) -> list[str]:
    """Tensor names from the safetensors headers (8-byte length, then JSON). No data is read."""
    names = []
    for f in sorted(snapshot.glob("*.safetensors")):
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n).decode("utf-8"))
        names += [k for k in header if k != "__metadata__"]
    return sorted(names)


def version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


# ---- the steps ------------------------------------------------------------------------------------

def roundtrip(tokenizer) -> dict:
    rows = []
    for s in ROUNDTRIP_STRINGS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        back = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        rows.append({"text": s, "ids": ids, "decoded": back, "ok": back == s})
    ok = sum(r["ok"] for r in rows)
    return {"total": len(rows), "ok": ok, "pass": ok == len(rows), "add_special_tokens": False, "rows": rows}


def load_model(torch, transformers, snapshot: Path) -> tuple[object, dict]:
    """(model, load report). Raises when from_pretrained raises."""
    config = transformers.AutoConfig.from_pretrained(str(snapshot))
    arch = (getattr(config, "architectures", None) or [None])[0]
    cls = getattr(transformers, arch, None) if arch else None
    if cls is None:
        cls = transformers.AutoModelForCausalLM
    # transformers renamed torch_dtype to dtype in 4.56.
    dkey = "dtype" if version_tuple(transformers.__version__) >= (4, 56) else "torch_dtype"
    t0 = time.time()
    model, info = cls.from_pretrained(str(snapshot), low_cpu_mem_usage=True, output_loading_info=True,
                                      **{dkey: torch.bfloat16})
    seconds = round(time.time() - t0, 2)
    model.eval()
    patterns = list(getattr(cls, "_keys_to_ignore_on_load_unexpected", None) or []) + \
        list(getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])
    patterns = sorted(set(patterns))
    ckpt = checkpoint_keys(snapshot)
    ignored, counted = split_ignored(ckpt, patterns)
    missing = sorted(str(k) for k in info.get("missing_keys") or [])
    unexpected = sorted(str(k) for k in split_ignored(info.get("unexpected_keys") or [], patterns)[1])
    mismatched = [[str(x) for x in m] if isinstance(m, (list, tuple)) else [str(m)]
                  for m in info.get("mismatched_keys") or []]
    errors = [str(e) for e in info.get("error_msgs") or []]
    model_keys = set(model.state_dict().keys())
    report = {
        "class": cls.__name__, "architecture": arch, "seconds": seconds, "device": "cpu",
        "dtype": str(next(model.parameters()).dtype), "low_cpu_mem_usage": True,
        "missing_keys": missing, "unexpected_keys": unexpected, "mismatched_keys": mismatched,
        "error_msgs": errors, "ignore_patterns": patterns, "ignored_checkpoint_keys": ignored,
        "checkpoint_key_count": len(ckpt), "model_key_count": len(model_keys),
        "checkpoint_keys_not_in_model": sorted(set(counted) - model_keys)[:50],
        "model_keys_not_in_checkpoint": sorted(model_keys - set(ckpt))[:50],
        "note": ("checkpoint_keys_not_in_model and model_keys_not_in_checkpoint compare names as "
                 "written; a class that renames keys while loading shows its renames there. The "
                 "load check uses transformers' own missing, unexpected and mismatched lists."),
    }
    report["pass"] = not (missing or unexpected or mismatched or errors)
    return model, report


def render(tokenizer, prompt: str) -> tuple[str, bool]:
    """(prompt text, whether the chat template was used)."""
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        return text, True
    return prompt, False


def top(torch, tokenizer, logits) -> list[dict]:
    vals, ids = torch.topk(logits, TOP_K)
    return [{"id": int(i), "logit": round(float(v), 4), "text": tokenizer.decode([int(i)])}
            for v, i in zip(vals.tolist(), ids.tolist())]


def last_logits(torch, model, seq: list[int]):
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([seq], dtype=torch.long), use_cache=False)
    return out.logits[0, -1].float()


def decode(torch, model, tokenizer, prompt: str, tag: str) -> dict:
    """Greedy decode of N_NEW tokens with the checks, writing evidence as it goes. `tag` is ""
    for the main prompt and "thai-" for the Thai one."""
    text, templated = render(tokenizer, prompt)
    prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    write_json(REFERENCE / f"{tag}prompt-ids.json",
               {"prompt_ids": prompt_ids, "prompt_text": text, "user_prompt": prompt,
                "chat_template_used": templated})
    say(f"wrote evidence/reference/{tag}prompt-ids.json ({len(prompt_ids)} prompt tokens)")
    say("forward pass over the prompt alone (the first-token check)")
    pre = last_logits(torch, model, prompt_ids)
    prompt_top = top(torch, tokenizer, pre)
    prompt_argmax = int(torch.argmax(pre))
    seq, steps, records = list(prompt_ids), [], []
    t0 = time.time()
    for k in range(N_NEW):
        logits = last_logits(torch, model, seq)
        chosen = int(torch.argmax(logits))
        before = list(seq)
        seq = seq + [chosen]
        steps.append({"chosen": chosen, "before": before, "after": list(seq)})
        records.append({"step": k, "position": len(before) - 1, "chosen": chosen,
                        "top5": top(torch, tokenizer, logits)})
        generated = seq[len(prompt_ids):]
        write_json(REFERENCE / f"{tag}generated-ids.json",
                   {"generated_ids": generated, "generated_text": tokenizer.decode(generated),
                    "complete": k + 1 == N_NEW})
        say(f"{tag or 'main '}decode step {k + 1}/{N_NEW}: token {chosen} "
            f"({time.time() - t0:.1f} s so far)")
    generated = seq[len(prompt_ids):]
    gen_text = tokenizer.decode(generated)
    appended, append_why = appended_at_end(prompt_ids, steps)
    prefix_kept = seq[:len(prompt_ids)] == prompt_ids and len(generated) == N_NEW
    first_ok = prompt_argmax == generated[0]
    write_json(REFERENCE / f"{tag}top5-logits.json",
               {"prompt_last_position": {"position": len(prompt_ids) - 1, "argmax": prompt_argmax,
                                         "top5": prompt_top},
                "steps": records})
    lines = [f"user prompt: {prompt}", f"chat template used: {templated}",
             f"prompt tokens: {len(prompt_ids)}", f"rendered prompt: {text!r}",
             f"generated ids ({len(generated)}): {generated}", f"generated text: {gen_text!r}",
             f"argmax of the last prompt position (separate forward pass): {prompt_argmax}",
             f"first generated token: {generated[0]}",
             f"first-token check: {'pass' if first_ok else 'FAIL'}",
             f"append check: {'pass' if appended and prefix_kept else 'FAIL'} ({append_why})",
             f"decode seconds: {time.time() - t0:.1f}"]
    write_text(EVIDENCE / f"{tag}decode.txt", "\n".join(lines) + "\n")
    say(f"wrote evidence/{tag}decode.txt")
    return {"prompt_ids": prompt_ids, "generated": generated, "text": gen_text, "templated": templated,
            "prompt_argmax": prompt_argmax, "first_ok": first_ok, "appended": appended and prefix_kept,
            "append_why": append_why, "seconds": round(time.time() - t0, 1)}


def has_thai_tokens(tokenizer) -> bool:
    """True when some token of a Thai sample decodes to a whole Thai character."""
    ids = tokenizer(THAI_SAMPLE, add_special_tokens=False)["input_ids"]
    return any("฀" <= ch <= "๿" for i in ids for ch in tokenizer.decode([i]))


def card_check(snapshot: Path, text: str) -> tuple[bool, str]:
    readme = snapshot / "README.md"
    card = readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else None
    start = reasoning_start(text)
    nonempty = bool(text.strip())
    note = ("This is a form check only. The script does not read the card's claims, so it cannot say "
            "whether the output matches the behaviour the card describes. ")
    note += (f"The model card README.md has {len(card)} characters. " if card is not None
             else "The snapshot has no README.md. ")
    note += (f"The generated text is non-empty ({len(text)} characters). " if nonempty
             else "The generated text is empty. ")
    if start:
        note += (f"It begins with reasoning ('{start}'), so the 32 tokens show the model's reasoning "
                 "and do not reach the final answer the card describes.")
    else:
        note += "It does not begin with reasoning ('Let me', 'We need' or '<think')."
    return nonempty, note


def main() -> int:
    cfg = load_config()
    torch, transformers = import_stack()
    snapshot = Path(cfg["model_snapshot"])
    run_dir = os.path.realpath(cfg["run_dir"])
    if not snapshot.is_dir():
        fail(f"model_snapshot is not a directory: {snapshot}")

    def rel(p: Path) -> str:
        return os.path.relpath(p, run_dir)

    env = {"python": platform.python_version(), "interpreter": sys.executable,
           "configured_python": cfg["python"], "torch": torch.__version__,
           "transformers": transformers.__version__}
    if os.path.realpath(cfg["python"]) != os.path.realpath(sys.executable):
        say(f"note: running under {sys.executable}, and reference_config.json names {cfg['python']}")
    say(f"torch {torch.__version__}, transformers {transformers.__version__}")

    say("loading the tokenizer")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(snapshot))
    rt = roundtrip(tokenizer)
    write_json(EVIDENCE / "tokenizer-roundtrip.json", rt)
    say(f"wrote evidence/tokenizer-roundtrip.json ({rt['ok']} of {rt['total']} round trips exact)")

    say("loading the weights on CPU in bf16 (this takes minutes for a large model)")
    try:
        model, rep = load_model(torch, transformers, snapshot)
    except Exception as exc:  # noqa: BLE001 - the failure goes into the evidence and reference.json
        rep = {"pass": False, "error": f"{type(exc).__name__}: {exc}"[:4000]}
        write_json(EVIDENCE / "load-report.json", rep)
        note = f"from_pretrained raised {rep['error'][:600]}"
        checks = [{"name": "loads", "pass": False, "note": note, "evidence": [rel(EVIDENCE / "load-report.json")]}]
        for name in ("decodes forward", "matches the card"):
            checks.append({"name": name, "pass": False, "note": "not run: the weights did not load",
                           "evidence": [rel(EVIDENCE / "load-report.json")]})
        checks.insert(1, {"name": "tokenizer round trip", "pass": rt["pass"],
                          "note": f"{rt['ok']} of {rt['total']} strings round trip exactly.",
                          "evidence": [rel(EVIDENCE / "tokenizer-roundtrip.json")]})
        write_json(HERE_DIR / "reference.json", {"verdict": "fail", "environment": env, "checks": checks})
        say(f"the weights did not load: {rep['error'][:300]}")
        return 3
    write_json(EVIDENCE / "load-report.json", rep)
    say(f"wrote evidence/load-report.json ({rep['class']}, {rep['seconds']} s, {len(rep['missing_keys'])} "
        f"missing, {len(rep['unexpected_keys'])} unexpected, {len(rep['mismatched_keys'])} mismatched)")

    main_run = decode(torch, model, tokenizer, PROMPT, "")
    thai = has_thai_tokens(tokenizer)
    thai_run = decode(torch, model, tokenizer, THAI_PROMPT, "thai-") if thai else None
    if not thai:
        say("the tokenizer has no token that decodes to a Thai character; no Thai decode")

    card_ok, card_note = card_check(snapshot, main_run["text"])
    write_text(EVIDENCE / "card-check.txt",
               card_note + f"\n\ngenerated text: {main_run['text']!r}\n")
    say("wrote evidence/card-check.txt")

    ign = rep["ignored_checkpoint_keys"]
    load_note = (f"Weights load on CPU (bf16, low_cpu_mem_usage) through {rep['class']} in {rep['seconds']} s. "
                 f"{len(rep['missing_keys'])} missing, {len(rep['unexpected_keys'])} unexpected and "
                 f"{len(rep['mismatched_keys'])} mismatched keys; {len(rep['error_msgs'])} error messages.")
    for label, keys in (("Missing", rep["missing_keys"]), ("Unexpected", rep["unexpected_keys"]),
                        ("Mismatched", [m[0] for m in rep["mismatched_keys"]])):
        if keys:
            load_note += f" {label}: {', '.join(keys[:10])}" + (f" and {len(keys) - 10} more." if len(keys) > 10 else ".")
    if ign:
        load_note += (f" {len(ign)} checkpoint keys match the class's _keys_to_ignore_on_load_unexpected "
                      f"patterns ({', '.join(rep['ignore_patterns'])}) and are not counted as unexpected.")
    load_note += f" The checkpoint has {rep['checkpoint_key_count']} keys and the model {rep['model_key_count']}."

    decode_ok = main_run["first_ok"] and main_run["appended"] and (thai_run is None or (thai_run["first_ok"] and thai_run["appended"]))
    decode_note = (f"Greedy decode of the prompt {PROMPT!r} ({'through the chat template' if main_run['templated'] else 'without a chat template'}), "
                   f"{N_NEW} new tokens. The argmax of the last prompt position ({main_run['prompt_argmax']}) "
                   f"{'equals' if main_run['first_ok'] else 'does not equal'} the first generated token "
                   f"({main_run['generated'][0]}). Append check: {main_run['append_why']}"
                   f"{'' if main_run['appended'] else ' (FAILED)'}. Generated text: {main_run['text'][:300]!r}.")
    if thai_run:
        decode_note += (f" Thai prompt: first-token check {'pass' if thai_run['first_ok'] else 'FAIL'}, append check "
                        f"{'pass' if thai_run['appended'] else 'FAIL'}; text {thai_run['text'][:200]!r}.")
    dec_ev = [rel(EVIDENCE / "decode.txt"), rel(REFERENCE / "prompt-ids.json"),
              rel(REFERENCE / "generated-ids.json"), rel(REFERENCE / "top5-logits.json")]
    if thai_run:
        dec_ev.append(rel(EVIDENCE / "thai-decode.txt"))
    checks = [
        {"name": "loads", "pass": rep["pass"], "note": load_note, "evidence": [rel(EVIDENCE / "load-report.json")]},
        {"name": "tokenizer round trip", "pass": rt["pass"],
         "note": (f"{rt['ok']} of {rt['total']} test strings (Latin, accented, emoji, CJK, Devanagari, Thai, "
                  "Korean, punctuation, whitespace, leading and trailing spaces) encode and decode back to "
                  "the same text."),
         "evidence": [rel(EVIDENCE / "tokenizer-roundtrip.json")]},
        {"name": "decodes forward", "pass": decode_ok, "note": decode_note, "evidence": dec_ev},
        {"name": "matches the card", "pass": card_ok, "note": card_note,
         "evidence": [rel(EVIDENCE / "card-check.txt")]},
    ]
    verdict = "pass" if all(c["pass"] for c in checks) else "fail"
    summary = {"verdict": verdict, "checks": {c["name"]: c["pass"] for c in checks}, "environment": env,
               "load_seconds": rep["seconds"], "decode_seconds": main_run["seconds"],
               "thai_tokens": thai, "thai_decode_seconds": thai_run["seconds"] if thai_run else None,
               "reasoning_start": reasoning_start(main_run["text"])}
    write_json(EVIDENCE / "summary.json", summary)
    ref = {"verdict": verdict, "environment": env, "checks": checks,
           "draft": "written by reference_gate.py; the agent reviews each note",
           "reference_artifacts": {"prompt_ids": rel(REFERENCE / "prompt-ids.json"),
                                   "generated_ids": rel(REFERENCE / "generated-ids.json"),
                                   "top5_logits": rel(REFERENCE / "top5-logits.json")}}
    write_json(HERE_DIR / "reference.json", ref)
    say(f"wrote evidence/summary.json and reference.json: verdict {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
