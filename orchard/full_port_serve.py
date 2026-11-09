# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 5 of a full port (issue 329, stage 5 half): the port behind a chat server, with an acceptance corpus that the server replays at boot.

Stages 2 and 3 show that the port's tokens agree with the stage 1 reference when the check drives the port directly. Stage 5 shows the same
port when it sits behind an HTTP server and answers requests one after another. A server can fail in ways the direct check cannot see:
state carried from one request to the next, a prompt that is cut, a request that reaches the port with other positions than the check used.

What the stage does, with no agent step (the port was written and judged in stages 2 and 3):

1. Prepare (supervisor). It records the acceptance corpus (`build_corpus`) from the stage 1 reference and from the record of the last port
   stage that passed (stage 3, or stage 2 when stage 3 was skipped). It copies that stage's port files to `stages/5/port/`, the shipped
   templates `port_serve.py` and `port_check.py`, and a config. The ledger holds the sha256 of each.
2. Test (one command under the lease): `python3 -P -s stages/5/port_serve.py`. The template starts the port in a worker process (the same worker
   as stage 2's check), puts an OpenAI-style chat server in front of it on a loopback port, replays the corpus through the server, refuses to
   go on when the replay differs from the corpus in any position, prints the readiness line, and then asks the passkey and canary questions
   through the same server. It writes `evidence/port-serve.json` (RECORD_SCHEMA) and prints `PORT_CHECK_SHA256 <sha256 of the record>`.
3. Result (supervisor). `derive` writes `result.json` from the record. The gate (`check`) recomputes every figure in it.

The corpus. It holds two entries. The teacher-forced entry gives the prompt and the first N reference ids; the server answers the port's
argmax at each position while the reference ids are fed in (the way stage 2's check does it), and the answer must equal the ids the accepted
port gave then (`expected_ids`). The free-run entry gives the prompt and asks for the port's own greedy ids, which must equal the free-run
ids of the accepted record. The replay bar is every position: a port that passed stage 2 or 3 and then answers one position differently behind
the server has changed. The reference ids travel with the corpus, so the replay's top-1 agreement with the reference is recomputed too, with
the stage 2 bar. The corpus is rebuilt from the stage 1 files and the accepted record by the gate and must equal the file the test read.

Passkey and canary. The passkey check hides a short key in filler text at two or more prompt lengths and asks for it through the chat
endpoint. The canary asks one fixed prompt twice with a different request between the two and expects the same ids both times, equal to the
accepted record's free-run ids.

The rehearsal option (`--full-port-rehearsal`) waives the field `coherent`, which here means "the passkey was answered at every length". A model
with random weights cannot answer. Every other check applies: the corpus replay, the boot line, the canary, the passkey runs that must be accepted by
the server and give the same ids twice, the provenance checks and the port sources.

What these checks do not show. For a model with random weights, the corpus is token-level numeric agreement between two runs of the same port
and with the CPU reference at the positions stage 1 recorded. It does not show that the port is right at positions or prompt lengths stage 1
did not record, that a long prompt (the passkey lengths) agrees with the CPU reference (only that the server accepts it and answers the same
ids twice), that the port is fast, or that it serves more than one request at a time (the server runs one request at a time). The chat
template is the model's own only when `tokenizer_config.json` carries one and `jinja2` is installed; the record says which was used. The kernel-cache
figure is a count of files in the cache directory before the boot. The runtime's own hit counter is not read. This module holds no state and starts nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from orchard import full_port, safeio
from orchard.defaults import SWAP_MIN_TOKENS, SWAP_TOP1_MIN

STAGE = 5
RECORD_SCHEMA = "orchard-port-serve/1"
CORPUS_SCHEMA = "orchard-port-corpus/1"
SKILL = "full-port-serving"
TEMPLATE = Path(__file__).resolve().parent / "skills" / "full-port-templates" / "port_serve.py"
TEMPLATE_NAME = "port_serve.py"
WORKER = full_port.TEMPLATE                      # the shipped stage 2 check; its worker runs the port here
WORKER_NAME = full_port.TEMPLATE_NAME            # "port_check.py"
CONFIG_NAME = "port_serve_config.json"
CORPUS_NAME = "corpus.json"
RECORD_FILE = "evidence/port-serve.json"
OUTPUT_FILE = full_port.OUTPUT_FILE
COMMAND = "python3 -P -s stages/5/port_serve.py"
PORT_DIR = "stages/5/port/"
DEADLINE_S = 3600
TOP1_MIN = SWAP_TOP1_MIN
MIN_TOKENS = SWAP_MIN_TOKENS
CANARY_TOKENS = 16                               # choice: enough ids to show a repeat without a long wait on a slow port
PASSKEY_LENGTHS_MIN = 2                          # the skill asks for two or more prompt lengths
PASSKEY_KEY_DIGITS = 5
PASSKEY_CAP = 8192                               # choice: the longest passkey prompt, in tokens (a port that prefills one token per step takes minutes)
PASSKEY_MARGIN = 64                              # room left in the context for the question and the answer
PASSKEY_SHORT = 256
PASSKEY_FIT = 0.5                                # a prompt the tokenizer measured must reach this share of the requested length
DEFAULT_MAX_LEN = 2048                           # the context limit when config.json names none
EVIDENCE_NAMES = ("port-serve.json", "hw-test-output.txt", CORPUS_NAME)
JSON_MAX = 16 * 1024 * 1024
LIMITS = ("a model with random weights shows token-level agreement of two runs of the same port and with the CPU reference at the recorded positions; "
          "it does not show correctness at other positions or prompt lengths, speed, or concurrent requests")


def command() -> str:
    """The only command the supervisor runs for stage 5."""
    return COMMAND


def worker_sha256() -> str | None:
    """The sha256 of the shipped stage 2 check, whose worker the template runs the port in."""
    return full_port.template_sha256()


def template_sha256() -> str | None:
    try:
        return hashlib.sha256(safeio.read_bytes(TEMPLATE, max_bytes=JSON_MAX)).hexdigest()
    except (OSError, safeio.SafeReadError):
        return None


# ---- which stage's port is served ------------------------------------------------------------------------------------------------

def source_stage(entries: list[dict]) -> int | None:
    """The stage whose port the run serves: 3 when stage 3 ended with a pass, else 2 when stage 2 did, else None. Read from the supervisor's own
    ledger entries, the latest `stage_end` of each stage (an earlier failed attempt does not count)."""
    last: dict[int, str] = {}
    for e in entries:
        if isinstance(e, dict) and e.get("event") == "stage_end" and e.get("stage") in (2, 3):
            data = e.get("data")
            last[e["stage"]] = data.get("result") if isinstance(data, dict) else None
    for n in (3, 2):
        if last.get(n) == "pass":
            return n
    return None


def accepted_sha(entries: list[dict], stage: int, rel: str) -> str | None:
    """The sha256 the ledger holds for `rel` in the latest passing `stage_end` of `stage` (the file the stage's gate accepted), or None."""
    found = None
    for e in entries:
        if isinstance(e, dict) and e.get("event") == "stage_end" and e.get("stage") == stage:
            data = e.get("data") if isinstance(e.get("data"), dict) else {}
            found = None
            if data.get("result") == "pass":
                for item in data.get("evidence") or []:
                    if isinstance(item, dict) and item.get("path") == rel and isinstance(item.get("sha256"), str):
                        found = item["sha256"]
    return found


def record_path(stage: int) -> str:
    return f"stages/{stage}/{full_port.RECORD_FILE}"


# ---- the corpus ------------------------------------------------------------------------------------------------------------------

def _read(run_dir, rel: str, problems: list):
    f = full_port._inside(run_dir, rel)
    if f is None:
        problems.append(f"{rel} is not a file inside the run directory")
        return None, None
    try:
        raw = safeio.read_bytes(f, max_bytes=JSON_MAX)
        return safeio.parse_json_bytes(raw, max_bytes=JSON_MAX, expect=dict), hashlib.sha256(raw).hexdigest()
    except safeio.SafeReadError as exc:
        problems.append(f"{rel} is not readable JSON: {exc.reason}")
        return None, None


def build_corpus(run_dir, stage: int, *, accepted: str | None = None) -> tuple[dict | None, list[str]]:
    """(corpus, []) for the port that passed `stage`, or (None, the reasons it cannot be built). The sources are the stage 1 files and the
    accepted record of `stage` (`stages/<stage>/evidence/port-check.json`). `accepted`, when given, is the sha256 the ledger holds for that record;
    a record that differs from it is refused. The same inputs give the same corpus, so the gate builds it again and compares.

    Every id must be a whole number; the record's reference ids must be the first ids of the stage 1 file; the accepted run must have compared
    at least MIN_TOKENS positions and made a free run."""
    problems: list[str] = []
    ref_doc, ref_sha = _read(run_dir, full_port.REFERENCE_FILE, problems)
    prm_doc, prm_sha = _read(run_dir, full_port.PROMPT_FILE, problems)
    rec_rel = record_path(stage)
    rec, rec_sha = _read(run_dir, rec_rel, problems)
    if problems:
        return None, problems
    if accepted is not None and rec_sha != accepted:
        return None, [f"{rec_rel} is not the file stage {stage}'s gate accepted (the ledger's sha256 differs): it changed after the stage passed"]
    # The record names the stage 1 files its ids came from and their sha256 (stage 2's gate checked that when it passed). A stage 1 file that
    # changed since would be paired with the record's old outputs, and rebuilding the corpus would repeat the pairing (review of PR 490).
    for key, rel, sha in (("reference", full_port.REFERENCE_FILE, ref_sha), ("prompt", full_port.PROMPT_FILE, prm_sha)):
        named = rec.get(key)
        if not isinstance(named, dict) or named.get("path") != rel or named.get("sha256") != sha:
            problems.append(f"the accepted record's {key} is not {rel} as it is on disk now (the path or the sha256 differs): "
                            "it changed after the stage passed")
    if problems:
        return None, problems
    ref = ref_doc.get("generated_ids")
    prompt = prm_doc.get("prompt_ids")
    chip, free, rref = rec.get("chip_ids"), rec.get("free_run_ids"), rec.get("reference_ids")
    for name, ids in (("generated_ids", ref), ("prompt_ids", prompt), ("chip_ids", chip), ("free_run_ids", free), ("reference_ids", rref)):
        if not full_port._ids(ids) or not ids:
            problems.append(f"{name} is not a non-empty list of whole numbers")
    if problems:
        return None, problems
    if len(chip) < MIN_TOKENS:
        problems.append(f"the accepted record compared {len(chip)} positions; at least {MIN_TOKENS} are needed")
    if len(chip) != len(rref) or rref != ref[:len(rref)]:
        problems.append("the accepted record's reference_ids are not the first ids of the stage 1 reference")
    if problems:
        return None, problems
    n = len(chip)
    return {"schema": CORPUS_SCHEMA, "source_stage": stage,
            "reference": {"path": full_port.REFERENCE_FILE, "sha256": ref_sha},
            "prompt": {"path": full_port.PROMPT_FILE, "sha256": prm_sha},
            "accepted": {"path": rec_rel, "sha256": rec_sha},
            "entries": [
                {"id": "teacher-forced", "kind": "teacher-forced", "prompt_ids": list(prompt), "forced_ids": list(ref[:n - 1]),
                 "reference_ids": list(ref[:n]), "expected_ids": list(chip)},
                {"id": "free-run", "kind": "free-run", "prompt_ids": list(prompt), "expected_ids": list(free)}]}, []


def corpus_bytes(corpus: dict) -> bytes:
    """The bytes the supervisor writes for a corpus, so the gate and the supervisor hash the same text."""
    return (json.dumps(corpus, indent=2, sort_keys=True) + "\n").encode("utf-8")


def positions(corpus: dict) -> int:
    """How many token positions a replay of `corpus` checks."""
    return sum(len(e["expected_ids"]) for e in corpus["entries"])


# ---- passkeys --------------------------------------------------------------------------------------------------------------------

def plan_lengths(max_len: int) -> list[int]:
    """The passkey prompt lengths for a server context of `max_len` tokens: a short one and the longest that leaves room, at most PASSKEY_CAP. A
    context too small for two different lengths gives []."""
    cap = min(max_len - PASSKEY_MARGIN, PASSKEY_CAP)
    short = min(PASSKEY_SHORT, cap // 2)
    return sorted({short, cap}) if short >= 16 and cap > short else []


def plan_passkeys(corpus_sha: str, max_len: int) -> list[dict]:
    """[{"length", "key"}] for each planned length. The key is digits taken from the sha256 of the corpus and the length, so the supervisor, the
    template's config and the gate agree without a stored secret, and a reader can reproduce it."""
    out = []
    for length in plan_lengths(max_len):
        digest = hashlib.sha256(f"{corpus_sha}:{length}".encode()).hexdigest()
        key = str(int(digest[:12], 16) % (10 ** PASSKEY_KEY_DIGITS)).zfill(PASSKEY_KEY_DIGITS)
        out.append({"length": length, "key": key})
    return out


# What the template recorded about the model's thinking (issue 641). `template_thinks`: the chat template opens a thinking block. `switch`: the
# name of the template variable that turns it off. `off`: the passkey prompts were rendered with it off. `plan_max_tokens`: the plan's ceiling for the
# answer limit; each run's own `limit` is the ceiling clamped to the context, and is the allowance that was actually in force.
NO_THINKING = {"template_thinks": False, "switch": None, "off": False, "plan_max_tokens": 12}
THINK_CLOSE = "</think>"
_SWITCH_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,40}")


def thinking_of(section) -> dict:
    """The thinking facts of the record's passkey section, or NO_THINKING when the section lacks them or they are not of the right types (a record
    from before issue 641, or a damaged one, is read as a model that does not think, so a missing key still fails)."""
    got = section.get("thinking") if isinstance(section, dict) else None
    if not (isinstance(got, dict) and isinstance(got.get("template_thinks"), bool) and isinstance(got.get("off"), bool)
            and (got.get("switch") is None or (isinstance(got.get("switch"), str) and _SWITCH_NAME.fullmatch(got["switch"])))
            and _int(got.get("plan_max_tokens"), 1) and got["plan_max_tokens"] <= 4096):
        return dict(NO_THINKING)
    return {"template_thinks": got["template_thinks"], "switch": got["switch"], "off": got["off"], "plan_max_tokens": got["plan_max_tokens"]}


def _truncated(r) -> bool:
    """Evidence that a reply was cut by its answer limit: the server's finish reason is "length", or the reply holds as many tokens as the limit the
    request carried. A reply that stopped on its own (an end token) before the limit is a complete reply."""
    if not isinstance(r, dict):
        return False
    if r.get("finish_reason") == "length":
        return True
    ids, limit = r.get("ids_first"), r.get("limit")
    return full_port._ids(ids) and _int(limit, 1) and len(ids) >= limit


def passkey_run_statuses(runs, thinking) -> list[str]:
    """One status for each passkey run, recomputed from the record: "answered" (the key is in the clean text, and for a model read after its thinking
    block also after the closing tag of the raw text), "cut" (a thinking model's reply never closed its block AND was cut by its answer limit: it says
    nothing about the port) or "failed" (anything else: a complete reply without the key, a reply that stopped on its own inside the thinking
    block, or a model that does not think)."""
    reads = thinking.get("template_thinks") is True and thinking.get("off") is not True
    out = []
    for r in runs:
        # The clean text drops special tokens; the raw text keeps ones such as </think>. The key must be in the clean text (an edited answer that lost
        # its key fails), and for a model read after its thinking block it must also be after the closing tag of the raw text. A record without the raw
        # text reads the clean text for both.
        clean = r["answer_text"] if isinstance(r, dict) and isinstance(r.get("answer_text"), str) else ""
        raw = r.get("answer_raw") if isinstance(r, dict) and isinstance(r.get("answer_raw"), str) else clean
        key = r.get("key") if isinstance(r, dict) else None
        closed = THINK_CLOSE in raw
        body = raw.rsplit(THINK_CLOSE, 1)[1] if (reads and closed) else raw
        if isinstance(key, str) and key and not (reads and not closed) and key in clean and key in body:
            out.append("answered")
        elif reads and not closed and _truncated(r):
            out.append("cut")
        else:
            out.append("failed")
    return out


def passkey_answers(runs, thinking) -> tuple[list[bool], str]:
    """(answered, status) for the passkey runs. `status` is "answered" (every key found), "not-applicable" (every key not found sits in a reply that
    a thinking model cut inside its thinking block at the answer limit) or "failed"."""
    got = passkey_run_statuses(runs, thinking)
    answered = [g == "answered" for g in got]
    if got and all(answered):
        return answered, "answered"
    if got and all(g in ("answered", "cut") for g in got):
        return answered, "not-applicable"
    return answered, "failed"


def max_len_of(snapshot) -> int:
    """The server's context limit: `max_position_embeddings` from the snapshot's config.json (or its text_config), else DEFAULT_MAX_LEN."""
    try:
        # A Hugging Face cache snapshot holds config.json as a link into blobs/, which the bounded reader refuses by itself: the link is resolved
        # first, as the port facts do (review of PR 490).
        cfg = safeio.read_json(safeio.resolved(Path(snapshot) / "config.json"), max_bytes=1 << 20, expect=dict)
    except (safeio.SafeReadError, OSError):
        return DEFAULT_MAX_LEN
    for part in (cfg, cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}):
        v = part.get("max_position_embeddings")
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
    return DEFAULT_MAX_LEN


def config_for(snapshot: str, stage: int, corpus_sha: str) -> dict:
    """The template's config: what it reads and nothing the coder can write (stage 5 has no coder step)."""
    max_len = max_len_of(snapshot)
    return {"model_snapshot": str(snapshot), "source_stage": stage, "worker_sha256": worker_sha256(), "corpus_sha256": corpus_sha,
            "max_len": max_len, "canary_tokens": CANARY_TOKENS, "passkeys": plan_passkeys(corpus_sha, max_len)}


# ---- what the gate checks --------------------------------------------------------------------------------------------------------

def _file_sha(path) -> str | None:
    return full_port._sha(path) if Path(path).is_file() and not Path(path).is_symlink() else None


def _int(x, lo=0) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= lo


def check(stage_dir, run_dir, *, rehearsal: bool = False, waived: list | None = None, expected_chips: int | None = None,
          source: int | None = None, corpus_sha256: str | None = None) -> tuple[list[str], list[str]]:
    """(reasons, evidence paths) for a stage 5 directory of a full port. No reasons means the result passes.

    The caller (the supervisor) gives `source` (the stage whose port is served), `corpus_sha256` (the hash the ledger recorded when it wrote the
    corpus) and `expected_chips` (the chip count of that stage); nothing in the stage directory can set them. `rehearsal` waives `coherent` and
    nothing else; `waived` receives the field's name when it did."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    reasons: list[str] = []
    seen: list[str] = []
    if source not in (2, 3):
        return [f"the serving check needs the stage whose port it serves (2 or 3), got {source!r}"], seen
    marker = full_port._json(stage_dir / "test-result.json", "test-result.json", reasons)
    if isinstance(marker, dict):
        code, timed_out = marker.get("returncode"), marker.get("timed_out")
        if not (type(code) is int and code == 0 and timed_out is False):
            why = ("the hardware test ran past its deadline and was stopped" if timed_out else
                   "the hardware test has no exit code (it was refused or did not start)" if code is None else
                   f"the hardware test exited with code {code!r}")
            reasons.append(f"test-result.json: {why}")
        if marker.get("command") != command():
            reasons.append(f"test-result.json: the hardware test ran {marker.get('command')!r}; the supervisor's command for this stage is {command()!r}")
    result = full_port._json(stage_dir / "result.json", "result.json", reasons)
    record = full_port._json(stage_dir / RECORD_FILE, RECORD_FILE, reasons)
    if result is None or record is None:
        return reasons, seen
    # The evidence is the three files of this stage, at the run-relative paths `derive` writes: the same names in another directory would be
    # cited by the gate and the operator bundle although the gate judged these files (review of PR 490, round 4).
    exact = [os.path.relpath(stage_dir / RECORD_FILE, run_dir), os.path.relpath(stage_dir / OUTPUT_FILE, run_dir), os.path.relpath(stage_dir / CORPUS_NAME, run_dir)]
    if result.get("evidence") != exact:
        reasons.append(f"result.json evidence must be exactly {exact}, got {result.get('evidence')!r}")
    for rel in exact:
        if full_port._inside(run_dir, rel) is None:
            reasons.append(f"result.json: evidence {rel!r} is not a file inside the run directory")
        else:
            seen.append(rel)

    if record.get("schema") != RECORD_SCHEMA:
        reasons.append(f"{RECORD_FILE} schema must be {RECORD_SCHEMA!r}, got {record.get('schema')!r}")
    corpus = _corpus_reasons(record, stage_dir, run_dir, source, corpus_sha256, reasons)
    _template_reasons(record, stage_dir, source, reasons)
    _chip_reasons(record, marker, expected_chips, source, reasons)
    if isinstance(marker, dict):
        _output_reasons(stage_dir, run_dir, marker, reasons)
    full_port._source_reasons(record, stage_dir, run_dir, reasons, stage=STAGE)
    _source_match_reasons(record, run_dir, source, reasons)
    boot = _boot_reasons(record, stage_dir, reasons)
    replay = _replay_reasons(record, corpus, reasons) if corpus is not None else None
    passkey = _passkey_reasons(record, corpus_sha256, stage_dir, reasons)
    _canary_reasons(record, corpus, reasons)
    _result_reasons(result, boot, replay, passkey, rehearsal, waived, reasons, record)
    return reasons, seen


def _corpus_reasons(record, stage_dir, run_dir, source, corpus_sha256, reasons) -> dict | None:
    """The corpus the test read is the one the supervisor recorded, and it is the one the stage 1 files and the accepted record give."""
    want = f"stages/{STAGE}/{CORPUS_NAME}"
    entry = record.get("corpus")
    path = stage_dir / CORPUS_NAME
    on_disk = _file_sha(path)
    if not (isinstance(corpus_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", corpus_sha256)):
        reasons.append("the ledger holds no sha256 for the corpus the supervisor wrote, so the corpus cannot be tied to it")
    elif on_disk != corpus_sha256:
        reasons.append(f"{CORPUS_NAME} is not the corpus the supervisor recorded in the ledger (sha256 differs, or the file is missing)")
    if not isinstance(entry, dict) or entry.get("path") != want or entry.get("sha256") != on_disk:
        reasons.append(f"{RECORD_FILE} corpus must name {want} with the sha256 of the file on disk")
    try:
        doc = safeio.read_json(path, max_bytes=JSON_MAX, expect=dict)
    except safeio.SafeReadError as exc:
        reasons.append(f"{CORPUS_NAME} is not readable JSON: {exc.reason}")
        return None
    rebuilt, problems = build_corpus(run_dir, source)
    if rebuilt is None:
        reasons.extend(f"the corpus cannot be built again from the stage 1 files and the stage {source} record: {p}" for p in problems)
        return None
    if doc != rebuilt:
        reasons.append(f"{CORPUS_NAME} differs from the corpus the stage 1 reference and the stage {source} record give: it was edited, or one of them changed")
        return None
    return rebuilt


def _template_reasons(record, stage_dir, source, reasons) -> None:
    """The record came from the shipped template and the shipped worker, and the config is the file the record names."""
    shipped, worker = template_sha256(), worker_sha256()
    if shipped is None or worker is None:
        reasons.append("a shipped template cannot be read, so the record cannot be tied to it")
        return
    for name, want in ((TEMPLATE_NAME, shipped), (WORKER_NAME, worker)):
        mine = _file_sha(stage_dir / name)
        if mine is None:
            reasons.append(f"{name} is not in this stage directory as a regular file: the supervisor copies it there and the gate compares it with the shipped one")
        elif mine != want:
            reasons.append(f"{name} in this stage differs from the shipped file (sha256 {mine[:12]} against {want[:12]}), so the record is not its")
    if record.get("template_sha256") != shipped:
        reasons.append(f"{RECORD_FILE} template_sha256 {str(record.get('template_sha256'))[:12]} is not the shipped template's ({shipped[:12]})")
    if record.get("worker_sha256") != worker:
        reasons.append(f"{RECORD_FILE} worker_sha256 {str(record.get('worker_sha256'))[:12]} is not the shipped worker's ({worker[:12]})")
    cfg = record.get("config")
    on_disk = _file_sha(stage_dir / CONFIG_NAME)
    if not isinstance(cfg, dict) or cfg.get("sha256") != on_disk or on_disk is None:
        reasons.append(f"{CONFIG_NAME} is not the file the record names (missing, or changed after the test)")
    if record.get("source_stage") != source:
        reasons.append(f"{RECORD_FILE} source_stage {record.get('source_stage')!r} is not the stage the supervisor serves ({source})")


def _chip_reasons(record, marker, expected_chips, source, reasons) -> None:
    rc = record.get("chips")
    mc = marker.get("chips") if isinstance(marker, dict) else None
    ok = all(isinstance(c, list) and c and all(isinstance(x, str) for x in c) for c in (rc, mc))
    if not ok or sorted(rc) != sorted(mc):
        reasons.append(f"{RECORD_FILE} chips {rc!r} are not the chips the supervisor leased for the test {mc!r} (both must be non-empty lists of chip names)")
    elif not _int(expected_chips, 1):
        reasons.append(f"expected_chips must be a whole number of at least 1 (the chips stage {source} ran on), got {expected_chips!r}")
    elif len(rc) != expected_chips:
        reasons.append(f"the server ran on {len(rc)} chips; stage {source} showed the port on {expected_chips}, so this is not the port that was checked")


def _output_reasons(stage_dir: Path, run_dir: Path, marker: dict, reasons: list) -> None:
    """The supervisor's output file is the file it recorded, and the template printed this stage's record's sha256 into it, once. (The same checks
    as `full_port._output_reasons`, which names stage 2's record file.)"""
    out = stage_dir / OUTPUT_FILE
    rec = marker.get("output")
    if not isinstance(rec, dict) or rec.get("path") != os.path.relpath(out, run_dir) or rec.get("sha256") != full_port._sha(out):
        reasons.append(f"{OUTPUT_FILE} is not the output file the supervisor recorded in test-result.json")
        return
    try:
        text = safeio.read_bytes(out, max_bytes=JSON_MAX * 4).decode("utf-8", "replace")
    except safeio.SafeReadError as exc:
        reasons.append(f"{OUTPUT_FILE} is not readable: {exc.reason}")
        return
    mark = full_port.OUTPUT_MARK
    starts = [l for l in text.splitlines() if l.startswith(mark)]
    marks = re.findall(rf"^{mark} ([0-9a-f]{{64}})\s*$", text, re.M)
    if len(starts) != 1:
        reasons.append(f"{OUTPUT_FILE} has {len(starts)} lines that begin with {mark}; the template prints exactly one (and prefixes every line the port prints)")
    if not marks or marks[-1] != full_port._sha(stage_dir / RECORD_FILE):
        reasons.append(f"{OUTPUT_FILE} does not carry a {mark} line that matches {RECORD_FILE}, so the record was not written by the test the supervisor ran")


def _source_match_reasons(record, run_dir, source, reasons) -> None:
    """The files served are the files the accepted stage hashed: same names under the port directory, same sha256."""
    rec, problems = _read(run_dir, record_path(source), [])
    accepted = {}
    for s in (rec or {}).get("port_sources") if isinstance((rec or {}).get("port_sources"), list) else []:
        if isinstance(s, dict) and isinstance(s.get("path"), str) and s["path"].startswith(f"stages/{source}/port/"):
            accepted[s["path"].removeprefix(f"stages/{source}/port/")] = s.get("sha256")
    served = {}
    for s in record.get("port_sources") if isinstance(record.get("port_sources"), list) else []:
        if isinstance(s, dict) and isinstance(s.get("path"), str) and s["path"].startswith(PORT_DIR):
            served[s["path"].removeprefix(PORT_DIR)] = s.get("sha256")
    if not accepted:
        reasons.append(f"the stage {source} record lists no port sources, so the served port cannot be compared with the one that passed")
    elif served != accepted:
        extra = sorted(set(served) ^ set(accepted)) or sorted(k for k in served if served[k] != accepted.get(k))
        reasons.append(f"the port files served differ from the files stage {source} accepted ({extra[0]}): this is not the port that was checked")


def _boot_reasons(record, stage_dir, reasons) -> dict | None:
    boot = record.get("boot")
    if not isinstance(boot, dict):
        reasons.append(f"{RECORD_FILE} has no boot section")
        return None
    line = boot.get("ready_line")
    if not (isinstance(line, str) and line.startswith("READY ") and "\n" not in line):
        reasons.append("boot.ready_line must be the one line the server printed when the corpus replay passed (it begins with READY)")
    else:
        out = stage_dir / OUTPUT_FILE
        try:
            text = safeio.read_bytes(out, max_bytes=JSON_MAX * 4).decode("utf-8", "replace")
        except safeio.SafeReadError:
            text = ""
        if sum(1 for l in text.splitlines() if l == line) != 1:
            reasons.append("boot.ready_line does not appear exactly once in the test's output, so it is not the line the server printed")
        if not re.fullmatch(r"READY \S+ on 127\.0\.0\.1:\d+ pid \d+ corpus (\d+)/\1 positions", line):
            reasons.append("boot.ready_line does not say that every corpus position matched (READY <model> on 127.0.0.1:<port> pid <pid> corpus N/N positions)")
    for key in ("load_s", "ready_s"):
        if not full_port._num(boot.get(key)) or boot[key] <= 0:
            reasons.append(f"boot.{key} must be a positive number of seconds, got {boot.get(key)!r}")
    if full_port._num(boot.get("load_s")) and full_port._num(boot.get("ready_s")) and boot["ready_s"] < boot["load_s"]:
        reasons.append("boot.ready_s is shorter than boot.load_s: the readiness line cannot come before the load ends")
    cache = boot.get("kernel_cache")
    if not (isinstance(cache, dict) and _int(cache.get("entries_before")) and _int(cache.get("entries_after"))
            and cache.get("warm") is (cache["entries_before"] > 0)):
        reasons.append("boot.kernel_cache must give entries_before and entries_after (whole numbers) and warm, true exactly when entries_before is above 0")
    if boot.get("models") != ["orchard-port"]:
        reasons.append("boot.models must be the list the server's /v1/models answered (the one model it serves)")
    return boot


def _replay_reasons(record, corpus, reasons) -> dict | None:
    """Every corpus entry was replayed once and every position equals the corpus; the agreement with the reference is recomputed."""
    replay = record.get("replay")
    got = {}
    for e in (replay.get("entries") if isinstance(replay, dict) and isinstance(replay.get("entries"), list) else []):
        if isinstance(e, dict) and isinstance(e.get("id"), str) and full_port._ids(e.get("got_ids")):
            got.setdefault(e["id"], []).append(e["got_ids"])
    if not got:
        reasons.append(f"{RECORD_FILE} replay.entries is missing or malformed")
        return None
    total = matched = 0
    agree = ref_total = 0
    for e in corpus["entries"]:
        runs = got.get(e["id"], [])
        if len(runs) != 1:
            reasons.append(f"the replay holds {len(runs)} results for corpus entry {e['id']!r}; it needs exactly one")
            total += len(e["expected_ids"])
            continue
        ids = runs[0]
        total += len(e["expected_ids"])
        same = sum(1 for a, b in zip(ids, e["expected_ids"]) if a == b) if len(ids) == len(e["expected_ids"]) else 0
        matched += same
        if ids != e["expected_ids"]:
            first = next((i for i, (a, b) in enumerate(zip(ids, e["expected_ids"])) if a != b), min(len(ids), len(e["expected_ids"])))
            reasons.append(f"the replay of {e['id']!r} differs from the corpus at position {first} ({same} of {len(e['expected_ids'])} equal): "
                           "the server does not answer as the port that passed stage "
                           f"{corpus['source_stage']} did, so it is refused")
        if e["kind"] == "teacher-forced" and len(ids) == len(e["reference_ids"]):
            agree += sum(1 for a, b in zip(ids, e["reference_ids"]) if a == b)
            ref_total += len(ids)
    if set(got) - {e["id"] for e in corpus["entries"]}:
        reasons.append("the replay holds a result for an entry that is not in the corpus")
    agreement = agree / ref_total if ref_total else 0.0
    if ref_total < MIN_TOKENS:
        reasons.append(f"the replay compares {ref_total} positions with the reference; at least {MIN_TOKENS} are needed")
    elif agreement < TOP1_MIN:
        reasons.append(f"the replay's top-1 agreement with the reference is {agreement:.4f}, below the minimum of {TOP1_MIN}")
    return {"positions": total, "matched": matched, "agreement": agreement, "reference_positions": ref_total}


def _passkey_reasons(record, corpus_sha256, stage_dir, reasons) -> tuple[list[bool], bool, str] | None:
    """The passkey runs: the keys are the planned ones, each prompt was measured by the server, each ran twice with the same ids. Returns, for
    each run, whether the answer holds the key (recomputed from the answer text, read after the thinking block for a model that thinks), whether
    every run gave the same ids twice, and the status `passkey_answers` gives."""
    section = record.get("passkey")
    runs = section.get("runs") if isinstance(section, dict) else None
    if not isinstance(runs, list) or len(runs) < PASSKEY_LENGTHS_MIN:
        reasons.append(f"passkey.runs must hold at least {PASSKEY_LENGTHS_MIN} prompt lengths")
        return None
    cfg = full_port._json(stage_dir / CONFIG_NAME, CONFIG_NAME, [])
    planned = None
    # A config that cannot be read, or a context limit that is not a whole number of at least 1 (a string such as "2048" is one: the template
    # takes it through int()), would switch the planned-key and planned-length checks off. It is a reason, not a skipped check (review of PR 490).
    if not isinstance(cfg, dict) or not _int(cfg.get("max_len"), 1):
        reasons.append(f"{CONFIG_NAME} must be readable and hold max_len as a whole number of at least 1, so the planned passkeys can be checked")
    elif isinstance(corpus_sha256, str):
        planned = plan_passkeys(corpus_sha256, cfg["max_len"])
        if cfg.get("passkeys") != planned:
            reasons.append(f"{CONFIG_NAME} passkeys are not the ones the corpus's hash and the context limit give")
    lengths = []
    same_twice = True
    for i, r in enumerate(runs):
        if not (isinstance(r, dict) and _int(r.get("length"), 1) and isinstance(r.get("key"), str) and isinstance(r.get("answer_text"), str)
                and full_port._ids(r.get("ids_first")) and full_port._ids(r.get("ids_second")) and _int(r.get("prompt_tokens"), 1)):
            reasons.append(f"passkey.runs[{i}] needs length, key, answer_text, prompt_tokens, ids_first and ids_second")
            same_twice = False
            continue
        if "answer_raw" in r and not isinstance(r["answer_raw"], str):
            reasons.append(f"passkey.runs[{i}] answer_raw must be text when it is present")
        lengths.append(r["length"])
        if planned is not None and not any(p["length"] == r["length"] and p["key"] == r["key"] for p in planned):
            reasons.append(f"passkey.runs[{i}] length {r['length']} and key are not in the plan")
        if not PASSKEY_FIT * r["length"] <= r["prompt_tokens"] <= r["length"] + PASSKEY_MARGIN:
            reasons.append(f"passkey.runs[{i}]: the server counted {r['prompt_tokens']} prompt tokens for a requested {r['length']}")
        if not r["ids_first"] or r["ids_first"] != r["ids_second"]:
            same_twice = False
            reasons.append(f"passkey.runs[{i}] (length {r['length']}): the same prompt gave different ids the second time, or none")
    answered, status = passkey_answers([r if isinstance(r, dict) else {} for r in runs], thinking_of(section))
    if len(set(lengths)) < PASSKEY_LENGTHS_MIN:
        reasons.append(f"passkey.runs must cover at least {PASSKEY_LENGTHS_MIN} different lengths")
    # Two planned lengths are two lengths only when the server measured two different prompts. With a small context the filler comes out the
    # same size for both, and the plan's lengths alone would pass (review of PR 490).
    measured = {r["prompt_tokens"] for r in runs if isinstance(r, dict) and _int(r.get("prompt_tokens"), 1)}
    if len(measured) < PASSKEY_LENGTHS_MIN:
        reasons.append(f"passkey.runs: the server measured the same prompt length for different planned lengths ({sorted(measured)}); "
                       f"at least {PASSKEY_LENGTHS_MIN} different measured prompt lengths are needed")
    if planned is not None and sorted(lengths) != sorted(p["length"] for p in planned):
        reasons.append("passkey.runs do not cover the planned lengths")
    return answered, same_twice, status


def _canary_reasons(record, corpus, reasons) -> None:
    canary = record.get("canary")
    if not isinstance(canary, dict) or not all(full_port._ids(canary.get(k)) and canary.get(k) for k in ("first_ids", "between_ids", "second_ids")):
        reasons.append(f"{RECORD_FILE} canary needs first_ids, between_ids and second_ids")
        return
    if canary["first_ids"] != canary["second_ids"]:
        reasons.append("canary: the same prompt gave different ids after another request in between: state leaked from one request to the next")
    if corpus is not None:
        free = next(e for e in corpus["entries"] if e["kind"] == "free-run")["expected_ids"]
        n = len(canary["first_ids"])
        if canary["first_ids"] != free[:n] or n < min(CANARY_TOKENS, len(free)):
            reasons.append("canary: the ids are not the first ids of the accepted port's free run")
    free_prompt = next((e["prompt_ids"] for e in corpus["entries"] if e["kind"] == "free-run"), None) if corpus is not None else None
    if canary.get("prompt_ids") != free_prompt or not full_port._ids(canary.get("between_prompt_ids")) or canary.get("between_prompt_ids") == free_prompt:
        reasons.append("canary: the prompt asked twice must be the corpus's free-run prompt, and the request between the two must be a different prompt "
                       "(or it could not show state leaked from one request to the next)")


def _result_reasons(result, boot, replay, passkey, rehearsal, waived, reasons, record=None) -> None:
    """result.json is the supervisor's copy of the record: each field is recomputed here."""
    if result.get("serves") is not True:
        reasons.append(f"serves must be true (the server reached its readiness line after the corpus replay), got {result.get('serves')!r}")
    answered, same_twice, status = passkey if passkey is not None else (None, False, None)
    # `coherent` is true when every key was found, null when the passkey could not be judged (a thinking model's replies were all cut inside the
    # thinking block: issue 641) and false otherwise.
    want_coherent = None if status == "not-applicable" else (bool(answered) and all(answered))
    if result.get("coherent") is not want_coherent:
        reasons.append(f"result.json coherent {result.get('coherent')!r} differs from the {want_coherent!r} the gate recomputed from the passkey answers")
    if status is not None and result.get("passkey_status") not in (None, status):
        reasons.append(f"result.json passkey_status {result.get('passkey_status')!r} differs from the {status!r} the gate recomputed")
    if status == "not-applicable" and result.get("passkey_status") != status:
        reasons.append("result.json must say passkey_status 'not-applicable' when the replies were cut inside the thinking block")
    if want_coherent is False:
        runs_in = (record.get("passkey") or {}).get("runs") if isinstance(record, dict) and isinstance(record.get("passkey"), dict) else None
        per_run = passkey_run_statuses([r if isinstance(r, dict) else {} for r in runs_in or []], thinking_of(record.get("passkey")))
        lacking = [r.get("length") for r, st in zip(runs_in or [], per_run) if isinstance(r, dict) and st == "failed"]
        if rehearsal:
            if waived is not None:
                waived.append("coherent")
        else:
            reasons.append(f"coherent must be true (the passkey was answered at every length): at length(s) {lacking or 'unknown'} the answer is complete and "
                           "holds no key, so the server did not find a key in its own context")
    if isinstance(boot, dict):
        for key in ("load_s", "ready_s"):
            if result.get(key) != boot.get(key):
                reasons.append(f"result.json {key} {result.get(key)!r} differs from the record's {boot.get(key)!r}")
        if result.get("ready_line") != boot.get("ready_line"):
            reasons.append("result.json ready_line differs from the record's")
        cache = boot.get("kernel_cache")
        if isinstance(cache, dict) and result.get("kernel_cache_warm") is not cache.get("warm"):
            reasons.append("result.json kernel_cache_warm differs from the record's")
    if isinstance(replay, dict):
        for key, field in (("corpus_positions", "positions"), ("corpus_matched", "matched")):
            if result.get(key) != replay[field]:
                reasons.append(f"result.json {key} {result.get(key)!r} differs from the {replay[field]} the gate recomputed from the replay")
        agreement = result.get("corpus_reference_agreement")
        if not full_port._num(agreement) or abs(agreement - replay["agreement"]) > full_port.TOLERANCE:
            reasons.append(f"result.json corpus_reference_agreement {agreement!r} differs from the {replay['agreement']:.6f} the gate recomputed")
    section = record.get("passkey") if isinstance(record, dict) else None
    runs_in_record = section.get("runs") if isinstance(section, dict) else None
    if isinstance(runs_in_record, list) and result.get("passkey_lengths") != [r.get("length") for r in runs_in_record if isinstance(r, dict)]:
        reasons.append("result.json passkey_lengths differs from the lengths the record's passkey runs hold")
    if answered is not None:
        if result.get("passkey_answered") != answered:
            reasons.append(f"result.json passkey_answered {result.get('passkey_answered')!r} differs from the {answered!r} the gate recomputed from the answers")
    checks = result.get("checks")
    for name in ("boots", "corpus", "passkey", "canary"):
        c = checks.get(name) if isinstance(checks, dict) else None
        if not isinstance(c, dict) or not isinstance(c.get("pass"), bool) or not isinstance(c.get("evidence"), list) or not c["evidence"]:
            reasons.append(f"result.json checks.{name} needs pass (true or false) and evidence")
    _check_summary_reasons(result, checks, boot, replay, passkey, record, reasons)
    # A passing serving check records a boot and a canary that passed: the gate refuses a result that says otherwise, so the operator bundle
    # never reports failed checks from a stage that passed (review of PR 490).
    for name in ("boots", "canary"):
        if isinstance(checks, dict) and isinstance(checks.get(name), dict) and checks[name].get("pass") is not True:
            reasons.append(f"result.json checks.{name}.pass is not true, and a passing serving check boots and passes the canary")
    if isinstance(checks, dict) and isinstance(checks.get("corpus"), dict) and isinstance(replay, dict):
        if checks["corpus"].get("pass") is not (replay["matched"] == replay["positions"]):
            reasons.append("result.json checks.corpus.pass differs from whether every position matched")
    if isinstance(checks, dict) and isinstance(checks.get("passkey"), dict) and answered is not None:
        want = bool(answered) and (all(answered) or status == "not-applicable" or rehearsal) and same_twice
        if checks["passkey"].get("pass") is not want:
            reasons.append("result.json checks.passkey.pass differs from the answers (true when every key was answered, when the replies were cut "
                           "inside a thinking block, or under the rehearsal option)")
    if result.get("derived_by") != "supervisor":
        reasons.append("result.json is not the supervisor's derived file (derived_by)")


def _check_summary_reasons(result, checks, boot, replay, passkey, record, reasons) -> None:
    """Each entry of result.checks is compared with the entry `derive` would write from the record: the same evidence list as the result's own
    (so `[null]` and a short list are refused), and the same counts, lengths, answers and seconds. A summary with a changed number would
    otherwise reach the operator bundle next to a passing flag (review of PR 490, round 3)."""
    if not isinstance(checks, dict):
        return
    evidence = result.get("evidence")
    for name in ("boots", "corpus", "passkey", "canary"):
        c = checks.get(name)
        if isinstance(c, dict) and isinstance(c.get("evidence"), list) and c["evidence"] and c["evidence"] != evidence:
            reasons.append(f"result.json checks.{name}.evidence must be the same list of files as the result's own evidence")
    def differs(name, key, want):
        c = checks.get(name)
        if isinstance(c, dict) and c.get(key) != want:
            reasons.append(f"result.json checks.{name}.{key} {c.get(key)!r} differs from the {want!r} the gate recomputed from the record")
    if isinstance(boot, dict):
        differs("boots", "seconds", boot.get("ready_s"))
    if isinstance(replay, dict):
        differs("corpus", "positions", replay["positions"])
        differs("corpus", "matched", replay["matched"])
    if passkey is not None:
        differs("passkey", "answered", passkey[0])
        section = record.get("passkey") if isinstance(record, dict) else None
        runs = section.get("runs") if isinstance(section, dict) else None
        if isinstance(runs, list):
            differs("passkey", "lengths", [r.get("length") for r in runs if isinstance(r, dict)])


# ---- the result ------------------------------------------------------------------------------------------------------------------

def derive(stage_dir, run_dir, *, rehearsal: bool = False) -> tuple[dict | None, str]:
    """(result, why): `result.json` for a stage 5 directory whose test passed and whose record is complete, or (None, why not). Every number is the
    record's or is recomputed from the record's ids exactly as the gate does. The gate runs on the result afterwards and refuses what is not
    true: a passkey not answered gives `coherent` false, and the gate decides whether the rehearsal option waives it."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    problems: list[str] = []
    marker = full_port._json(stage_dir / "test-result.json", "test-result.json", problems)
    if not isinstance(marker, dict):
        return None, "test-result.json is missing or unreadable"
    code = marker.get("returncode")
    if not (type(code) is int and code == 0 and marker.get("timed_out") is False):
        return None, "the hardware test did not pass"
    record = full_port._json(stage_dir / RECORD_FILE, RECORD_FILE, problems)
    if not isinstance(record, dict):
        return None, f"{RECORD_FILE} is missing or unreadable"
    boot, replay = record.get("boot"), record.get("replay")
    corpus_entry = record.get("corpus")
    passkey, canary = record.get("passkey"), record.get("canary")
    if not (isinstance(boot, dict) and isinstance(replay, dict) and isinstance(corpus_entry, dict) and isinstance(passkey, dict) and isinstance(canary, dict)):
        return None, "the record lacks a boot, replay, corpus, passkey or canary section"
    try:
        corpus = safeio.read_json(stage_dir / CORPUS_NAME, max_bytes=JSON_MAX, expect=dict)
        entries = {e["id"]: e["got_ids"] for e in replay["entries"]}
        expected = {e["id"]: e for e in corpus["entries"]}
        total = sum(len(e["expected_ids"]) for e in expected.values())
        matched = sum(sum(1 for a, b in zip(entries[i], e["expected_ids"]) if a == b) for i, e in expected.items() if len(entries[i]) == len(e["expected_ids"]))
        tf = next(e for e in expected.values() if e["kind"] == "teacher-forced")
        if not (isinstance(tf["reference_ids"], list) and tf["reference_ids"]):
            raise ValueError("the teacher-forced reference is empty")  # agreement() divides by its length (review of PR 490, round 6)
        agreement = full_port.agreement(tf["reference_ids"], entries[tf["id"]]) if len(entries[tf["id"]]) == len(tf["reference_ids"]) else 0.0
        runs = passkey["runs"]
        answered, status = passkey_answers(runs, thinking_of(passkey))
        deterministic = all(r["ids_first"] == r["ids_second"] and r["ids_first"] for r in runs)
        free = next(e for e in expected.values() if e["kind"] == "free-run")["expected_ids"]
        canary_ok = canary["first_ids"] == canary["second_ids"] and canary["first_ids"] == free[:len(canary["first_ids"])] and bool(canary["first_ids"])
        cache = boot["kernel_cache"]
        if not isinstance(cache, dict) or not all(isinstance(r, dict) for r in runs):
            raise TypeError("kernel_cache or a passkey run is not an object")
        lengths = [r["length"] for r in runs]
    except (KeyError, TypeError, AttributeError, ValueError, StopIteration, safeio.SafeReadError):
        return None, "the record or the corpus does not have the shape the template writes"
    base = [os.path.relpath(stage_dir / RECORD_FILE, run_dir), os.path.relpath(stage_dir / OUTPUT_FILE, run_dir), os.path.relpath(stage_dir / CORPUS_NAME, run_dir)]
    all_answered = bool(answered) and all(answered)
    coherent = None if status == "not-applicable" else all_answered
    return ({"serves": True, "coherent": coherent, "passkey_status": status, "load_s": boot.get("load_s"), "ready_s": boot.get("ready_s"), "ready_line": boot.get("ready_line"),
             "kernel_cache_warm": cache.get("warm"), "corpus_positions": total, "corpus_matched": matched, "corpus_reference_agreement": agreement,
             "passkey_lengths": lengths, "passkey_answered": answered,
             "checks": {"boots": {"pass": True, "seconds": boot.get("ready_s"), "evidence": base},
                        "corpus": {"pass": matched == total, "positions": total, "matched": matched, "evidence": base},
                        "passkey": {"pass": bool(answered) and (all_answered or status == "not-applicable" or rehearsal) and deterministic, "lengths": lengths,
                                    "answered": answered, "evidence": base},
                        "canary": {"pass": canary_ok, "evidence": base}},
             "evidence": base, "derived_by": "supervisor", "limits": LIMITS,
             "labels": "load_s and ready_s are measured on one boot; kernel_cache_warm counts files in the cache directory before the boot"}, "derived")
