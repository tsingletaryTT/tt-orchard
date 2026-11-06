# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 of a full port (docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md): what the gate checks.

The class `full-port` means the model needs model code that does not exist yet. With `--full-port` the supervisor lets the local coder
write that code under `stages/2/port/`, and runs one hardware test that compares the chip's greedy tokens with the stage 1 CPU
reference. This module reads the files that run leaves and returns the reasons the result cannot be believed. It holds no
state and starts nothing, and it imports nothing from orchard/stages.py (which imports it).

Files, all under the stage directory:

    result.json               written by the finish step (the agent). The gate recomputes every number in it.
    evidence/port-check.json  written by the hardware command (RECORD_SCHEMA): the stage 1 reference ids, the chip's ids, the chips it
                              ran on, the model load time, the free-run text and the port's source files with their sha256.
    evidence/hw-test-output.txt  the command's output, written by the supervisor. The command prints `PORT_CHECK_SHA256 <hex>`
                              of port-check.json as its last act, so the record and the run belong together.
    test-result.json          written by the supervisor after the command ends (exit code, timeout, chips, the output file's sha256).

What the checks do and do not show. Slice 2 (issue 325): the comparison is made by a supervisor-owned template, skills/full-port-templates/
port_check.py, which the supervisor copies to stages/2/port_check.py and runs with a fixed command. The coder writes only the port module
(stages/2/port/port_model.py: load, prefill, decode_step, close). The template runs the port in a second process, computes agreement, the
reference-id match, the free-run text and the record itself, and prints the sha256 as its last line. The gate checks that the stage's copy
of the template is the shipped file, that the record carries the shipped file's sha256, that the supervisor's command was the template's
fixed command, and that the output holds one mark line that matches the record. They tie `result.json` to a record, the record to a test
the supervisor ran on leased chips, and the ids to the stage 1 file on disk. They do not prove the chip computed the ids: a port that
returns the reference's ids without computing them looks the same as one that computed them. The source checks (a port file must mention
ttnn and must not read the reference) make the plain forms visible.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from orchard import safeio
from orchard.defaults import SWAP_MIN_TOKENS, SWAP_TOP1_MIN

RECORD_SCHEMA = "orchard-port-check/1"
TOP1_MIN = SWAP_TOP1_MIN              # choice: the bar the weights-only stage 2 uses for the same measurement
MIN_TOKENS = SWAP_MIN_TOKENS
MIN_FREE_RUN_CHARS = 20               # choice: the same floor as the stock-runtime gate's free-run text
MIN_FREE_RUN_WORDS = 4                # choice: a free run of one repeated word is not text
REFERENCE_FILE = "stages/1/evidence/reference/generated-ids.json"
PROMPT_FILE = "stages/1/evidence/reference/prompt-ids.json"
PORT_DIR = "stages/2/port/"
RECORD_FILE = "evidence/port-check.json"
OUTPUT_FILE = "evidence/hw-test-output.txt"
OUTPUT_MARK = "PORT_CHECK_SHA256"
MODEL_ROLE = "model"                  # every file under stages/2/port/ is model code; the check is the template's
TEMPLATE = Path(__file__).resolve().parent / "skills" / "full-port-templates" / "port_check.py"      # the shipped template
TEMPLATE_NAME = "port_check.py"       # its name in stages/2/
TOOLS = TEMPLATE.parent / "stage_tools.py"      # the coder's fixed-name tools (issue 505): last-check, layer-diff, facts, show
TOOLS_NAME = "stage_tools.py"         # its name in the stage directory, next to the template
CHEAT_SOURCE = TEMPLATE.parent / "ttnn_cheat_sheet.json"      # the verified TTNN cheat sheet (issue 523; orchard/op_cheats.py)
CHEAT_NAME = "cheat_sheet.json"       # its data in the stage directory: `stage_tools.py cheat NAME` reads it
CHEAT_MD_NAME = "cheat_sheet.md"      # the rendered sheet the coder reads
CONFIG_NAME = "port_check_config.json"
# The only command the supervisor runs for this stage. -P and -s keep files the coder leaves in stages/2/ off the template's sys.path.
COMMAND = "python3 -P -s stages/2/port_check.py"
DEADLINE_S = 3600
PORT_MODULE = "stages/2/port/port_model.py"
STAGES = (2, 3)                       # the stages that run the template: 2 on one board, 3 on the configured chips (issue 326)
ENTRY_STAGE = 4                       # stage 4 runs it once per required chip count, each in its own directory stages/4/configs/<N>/ (issue 454)


def stage_path(stage: int = 2, entry: int | None = None) -> str:
    """The run-relative directory a run of the template lives in (no trailing slash): `stages/<stage>`, or for stage 4's entry for a chip
    count `stages/4/configs/<entry>`. The port, the template copy, its config, the record and the marker are all beneath it."""
    return f"stages/{stage}" if entry is None else f"stages/{stage}/configs/{entry}"


def port_dir(stage: int = 2, entry: int | None = None) -> str:
    """The directory the coder writes the port in, for stage `stage` (or its entry `entry`; run-relative, with the trailing slash)."""
    return f"{stage_path(stage, entry)}/port/"


def port_module(stage: int = 2, entry: int | None = None) -> str:
    """The file the template calls, for stage `stage` (or its entry `entry`)."""
    return f"{port_dir(stage, entry)}port_model.py"


def command(stage: int = 2, entry: int | None = None) -> str:
    """The only command the supervisor runs for stage `stage` (or its entry `entry`). The template finds the port, its config and its record
    next to itself, so the same shipped file serves every stage and every entry that runs it."""
    return f"python3 -P -s {stage_path(stage, entry)}/port_check.py"
EVIDENCE_NAMES = ("port-check.json", "hw-test-output.txt")     # result.json's evidence list must name both
JSON_MAX = 16 * 1024 * 1024
SOURCE_MAX = 2 * 1024 * 1024
TOLERANCE = 1e-6                      # the gate's recomputed agreement against the number result.json reports


def template_sha256() -> str | None:
    """The sha256 of the shipped template, or None when it cannot be read."""
    try:
        return hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    except OSError:
        return None


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _text(x) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _ids(x) -> bool:
    return isinstance(x, list) and all(isinstance(i, int) and not isinstance(i, bool) for i in x)


def _inside(run_dir, rel) -> Path | None:
    """The regular file `rel` names inside the run directory (links resolved), or None."""
    if not isinstance(rel, str) or not rel or os.path.isabs(rel) or "\0" in rel:
        return None
    try:
        root = os.path.realpath(run_dir)
        p = os.path.realpath(os.path.join(root, rel))
        if os.path.commonpath([root, p]) != root or not os.path.isfile(p):
            return None
    except (ValueError, OSError):
        return None
    return Path(p)


def _sha(path) -> str | None:
    try:
        return hashlib.sha256(safeio.read_bytes(path, max_bytes=JSON_MAX)).hexdigest()
    except safeio.SafeReadError:
        return None


def _json(path, what: str, reasons: list):
    """The JSON object in `path`, or None with a reason."""
    try:
        data = safeio.read_json(path, max_bytes=JSON_MAX, expect=dict)
    except safeio.SafeReadError as exc:
        reasons.append(f"{what} is missing" if isinstance(exc, safeio.ReadFailed) and "ENOENT" in exc.reason
                       else f"{what} is not readable JSON: {exc.reason}")
        return None
    return data


def file_record(run_dir, path) -> dict | None:
    """{"path": run-relative path, "sha256"} of the file at `path`, read through the bounded safe reader, or None when the reader refuses it
    (a FIFO, a device, a link, a file over the cap) or it is missing. The same shape as orchard/stages.py `evidence_record`, which opens
    the file without those limits and would block on a FIFO."""
    try:
        digest = hashlib.sha256(safeio.read_bytes(path, max_bytes=JSON_MAX)).hexdigest()
        return {"path": os.path.relpath(os.path.realpath(path), os.path.realpath(run_dir)), "sha256": digest}
    except (safeio.SafeReadError, OSError, ValueError):
        return None


def agreement(reference: list, chip: list) -> float:
    """The share of positions where the chip's id equals the reference id. Lists of equal, non-zero length."""
    return sum(1 for a, b in zip(reference, chip) if a == b) / len(reference)


def check(stage_dir, run_dir, *, rehearsal: bool = False, waived: list | None = None, stage: int = 2,
          expected_chips: int | None = None, entry: int | None = None) -> tuple[list[str], list[str]]:
    """(reasons, evidence paths) for a stage 2 or stage 3 directory of a full port. No reasons means the result passes.

    `entry`, when given, is the second form (issue 454): `stage_dir` is the directory of stage 4's entry for that chip count
    (`stages/4/configs/<entry>/`), `stage` must be 4, and the port directory, the command, the template copy, the config and the sources are
    all checked beneath that directory. The entry's chip count is the number of chips the test must have run on, so `expected_chips`
    must be that number (or None, which means it).

    `stage` is 2 (the port on one board) or 3 (the same port and the same template on the chip count the run is configured for, issue 326):
    it names the port directory and the command the supervisor's test ran. `expected_chips`, when the caller gives it (the supervisor does for
    stage 3, from the run's `--required-chips`), is the number of chips the test must have run on: the record's chips and the supervisor's
    marker must both list that many. Without it only the agreement of the two lists is checked, as for stage 2.

    `rehearsal` is the operator's word, given by the supervisor from the run's ledger (`--full-port-rehearsal`, issue 357). A rehearsal
    model has random weights, so its free-run text is not readable. The option waives the `coherent` field and no other check: top-1
    agreement, the token count, the text floor (characters and words), the provenance checks and the port sources all still apply. The
    argument is a keyword the caller sets; nothing in the stage directory can set it. `waived`, when given, receives the name of each
    field the option waived, taken from the same read of result.json that the checks use, so the caller never reopens the file."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    if entry is not None:
        if stage != ENTRY_STAGE or type(entry) is not int or entry < 1:
            return [f"an entry of the full-port check is a chip count of at least 1 in stage {ENTRY_STAGE}; got entry {entry!r} of stage {stage!r}"], []
        if expected_chips is not None and expected_chips != entry:
            return [f"the entry for {entry} chips cannot be checked against an expected count of {expected_chips!r}"], []
        expected_chips = entry          # the entry's own count: a record that lists another number of chips is not this entry's
    elif stage not in STAGES:
        return [f"the full-port check runs in stages {list(STAGES)}, not stage {stage!r}"], []
    reasons: list[str] = []
    seen: list[str] = []
    if expected_chips is not None and (type(expected_chips) is not int or expected_chips < 1):
        reasons.append(f"expected_chips must be a whole number of at least 1 (the supervisor's configured chip count), got {expected_chips!r}")
        expected_chips = None          # the other checks still run and name their own reasons
    result = _json(stage_dir / "result.json", "result.json", reasons)
    if result is None:
        return reasons, seen
    marker = _json(stage_dir / "test-result.json", "test-result.json", reasons)
    record = _json(stage_dir / RECORD_FILE, RECORD_FILE, reasons)

    # ---- the hardware test the supervisor ran ----
    if isinstance(marker, dict):
        code, timed_out = marker.get("returncode"), marker.get("timed_out")
        if not (type(code) is int and code == 0 and timed_out is False):
            if timed_out:
                why = "the hardware test ran past its deadline and was stopped"
            elif code is None:
                why = "the hardware test has no exit code (it was refused or did not start)"
            else:
                why = f"the hardware test exited with code {code!r}"
            reasons.append(f"test-result.json: {why}")
        if marker.get("command") != command(stage, entry):
            reasons.append(f"test-result.json: the hardware test ran {marker.get('command')!r}; the supervisor's command for this stage is {command(stage, entry)!r}")

    # ---- result.json: each field on its own ----
    if result.get("runs") is not True:
        reason = f"runs must be true (the port loaded and produced tokens on the chip), got {result.get('runs')!r}"
        failure = result.get("failure")
        if isinstance(failure, str) and failure.strip():
            reason += f"; the recorded failure: {failure.strip()[:500]}"
        reasons.append(reason)
    if result.get("coherent") is not True and rehearsal:
        if waived is not None:
            waived.append("coherent")
    elif result.get("coherent") is not True:
        reasons.append(f"coherent must be true (the free-run text is readable), got {result.get('coherent')!r}")
    load = result.get("load_s")
    if not _num(load) or load <= 0:
        reasons.append(f"load_s must be a positive number of seconds, got {load!r}")
    n = result.get("n_tokens")
    if isinstance(n, bool) or not isinstance(n, int) or n < MIN_TOKENS:
        reasons.append(f"n_tokens must be a whole number of at least {MIN_TOKENS}, got {n!r}")
    top1 = result.get("top1_agreement")
    if not _num(top1) or not 0 <= top1 <= 1:
        reasons.append(f"top1_agreement must be a fraction from 0 to 1, got {top1!r}")
    elif top1 < TOP1_MIN:
        reasons.append(f"top1_agreement {top1} is below the minimum of {TOP1_MIN}")
    listed = result.get("evidence")
    if not isinstance(listed, list) or not listed:
        reasons.append("result.json lists no evidence")
    else:
        for name in EVIDENCE_NAMES:
            if not any(isinstance(e, str) and Path(e).name == name for e in listed):
                reasons.append(f"result.json evidence must list {name}")
        for rel in listed:
            if _inside(run_dir, rel) is None:
                reasons.append(f"result.json: evidence {rel!r} is not a file inside the run directory")
            else:
                seen.append(rel)
    if record is None:
        return reasons, seen

    # ---- the record against the stage 1 reference on disk ----
    if record.get("schema") != RECORD_SCHEMA:
        reasons.append(f"{RECORD_FILE} schema must be {RECORD_SCHEMA!r}, got {record.get('schema')!r}")
    ref_ids = _reference_ids(record, run_dir, reasons)
    chip_ids = record.get("chip_ids")
    if not _ids(chip_ids) or not _ids(record.get("reference_ids")):
        reasons.append("reference_ids and chip_ids must be lists of whole numbers")
    elif len(chip_ids) != len(record["reference_ids"]) or not chip_ids:
        reasons.append("reference_ids and chip_ids must have the same non-zero length")
    else:
        if ref_ids is not None and record["reference_ids"] != ref_ids[:len(chip_ids)]:
            reasons.append("reference_ids are not the first ids of the stage 1 reference file (stages/1): "
                           "the comparison was made against ids the reference did not produce")
        got = agreement(record["reference_ids"], chip_ids)
        if len(chip_ids) < MIN_TOKENS:
            reasons.append(f"the record compares {len(chip_ids)} positions; at least {MIN_TOKENS} are needed")
        if isinstance(n, int) and not isinstance(n, bool) and n != len(chip_ids):
            reasons.append(f"n_tokens {n} differs from the {len(chip_ids)} positions the record compares")
        if _num(top1) and abs(top1 - got) > TOLERANCE:
            reasons.append(f"top1_agreement {top1} differs from the {got:.6f} the gate recomputed from the record's ids")
        if got < TOP1_MIN and not (_num(top1) and top1 < TOP1_MIN):
            reasons.append(f"top1_agreement recomputed from the record is {got:.4f}, below the minimum of {TOP1_MIN}")

    # ---- prefill starts a request afresh (issue 363) ----
    # The TT vLLM plugin gives a freed decode slot to a new request and expects prefill to initialise that request's state. The check
    # calls prefill on the same prompt twice, so the two first ids must be equal. A port that keeps state across prefills would be
    # rewritten at serving time (docs/superpowers/specs/2026-11-02-port-interface-vllm-plugin-mapping.md).
    free_ids = record.get("free_run_ids")
    if not _ids(free_ids) or not free_ids:
        reasons.append("free_run_ids must be a non-empty list of whole numbers")
    elif _ids(chip_ids) and chip_ids and free_ids[0] != chip_ids[0]:
        reasons.append(f"the port's second prefill of the same prompt gave id {free_ids[0]} where the first gave {chip_ids[0]}: prefill must "
                       "reset the model's state (the plugin reuses a slot for a new request and expects prefill to start it afresh)")

    # ---- the record against the supervisor's test ----
    if isinstance(marker, dict):
        rc, mc = record.get("chips"), marker.get("chips")
        chip_lists = all(isinstance(c, list) and c and all(isinstance(x, str) for x in c) for c in (rc, mc))
        if not chip_lists or sorted(rc) != sorted(mc):
            reasons.append(f"{RECORD_FILE} chips {rc!r} are not the chips the supervisor leased for the test {mc!r} "
                           "(both must be non-empty lists of chip names)")
        elif expected_chips is not None and (len(rc) != expected_chips or len(mc) != expected_chips):
            reasons.append(f"the hardware test ran on {len(mc)} chips and the record lists {len(rc)}; stage {stage} is configured for "
                           f"{expected_chips} (the run's --required-chips), so the port was not shown on the configured chips")
        _output_reasons(stage_dir, run_dir, marker, reasons)
    rec_load = record.get("load_s")
    if not _num(rec_load) or rec_load <= 0:
        reasons.append(f"{RECORD_FILE} load_s must be a positive number of seconds, got {rec_load!r}")
    elif _num(load) and abs(load - rec_load) > TOLERANCE:
        reasons.append(f"load_s {load} differs from the record's {rec_load}")

    # ---- the free-run text ----
    text = record.get("free_run_text")
    if result.get("free_run_text") != text:
        reasons.append(f"result.json free_run_text differs from the text in {RECORD_FILE}")
    if not _text(text) or len("".join(text.split())) < MIN_FREE_RUN_CHARS or len(set(text.split())) < MIN_FREE_RUN_WORDS:
        reasons.append(f"free_run_text must be at least {MIN_FREE_RUN_CHARS} non-space characters and {MIN_FREE_RUN_WORDS} different words")

    _template_reasons(record, stage_dir, run_dir, reasons)
    _source_reasons(record, stage_dir, run_dir, reasons, stage, entry)
    return reasons, seen


def _template_reasons(record: dict, stage_dir: Path, run_dir: Path, reasons: list) -> None:
    """The record came from the shipped template: the stage's copy is the shipped file, the record carries its sha256, and the config the
    template read is the file on disk. The record also says which positions the template made, so its own agreement figure must match."""
    shipped = template_sha256()
    if shipped is None:
        reasons.append(f"the shipped template {TEMPLATE.name} cannot be read, so the record cannot be tied to it")
        return
    copy = stage_dir / TEMPLATE_NAME
    mine = _sha(copy) if copy.is_file() and not copy.is_symlink() else None
    if mine is None:
        reasons.append(f"{TEMPLATE_NAME} is not in this stage directory as a regular file: the supervisor copies it there and the gate compares it with the shipped one")
    elif mine != shipped:
        reasons.append(f"{TEMPLATE_NAME} in this stage differs from the shipped template (sha256 {mine[:12]} against {shipped[:12]}), "
                       "so the record is not the template's")
    got = record.get("template_sha256")
    if got != shipped:
        reasons.append(f"{RECORD_FILE} template_sha256 {str(got)[:12]} is not the shipped template's ({shipped[:12]}): the record was not written by it")
    cfg = record.get("config")
    cfg_path = stage_dir / CONFIG_NAME
    if not isinstance(cfg, dict) or cfg.get("sha256") != (_sha(cfg_path) if cfg_path.is_file() else None):
        reasons.append(f"{CONFIG_NAME} is not the file the record names (missing, or changed after the test)")
    free = record.get("free_run_ids")
    if not _ids(free) or not free:
        reasons.append(f"{RECORD_FILE} free_run_ids must be a non-empty list of whole numbers")


def _reference_ids(record: dict, run_dir: Path, reasons: list) -> list | None:
    """The stage 1 generated ids, after checking that the record names the stage 1 files and their current sha256."""
    ids = None
    for key, want in (("reference", REFERENCE_FILE), ("prompt", PROMPT_FILE)):
        entry = record.get(key)
        if not isinstance(entry, dict) or entry.get("path") != want:
            reasons.append(f"{key}.path must be {want!r}, the stage 1 file, got {entry.get('path') if isinstance(entry, dict) else entry!r}")
            continue
        f = _inside(run_dir, want)
        if f is None:
            reasons.append(f"{key}: {want} is not a file in the run directory")
            continue
        if entry.get("sha256") != _sha(f):
            reasons.append(f"{key}.sha256 does not match {want} as it is on disk")
        if key == "reference":
            try:
                data = safeio.read_json(f, max_bytes=JSON_MAX, expect=dict)
            except safeio.SafeReadError as exc:
                reasons.append(f"{want} is not readable JSON: {exc.reason}")
                continue
            ids = data.get("generated_ids") if _ids(data.get("generated_ids")) else None
            if ids is None:
                reasons.append(f"{want} holds no list of generated_ids")
    return ids


def _output_reasons(stage_dir: Path, run_dir: Path, marker: dict, reasons: list, record_sha: str | None = None) -> None:
    """The supervisor's output file is the file it recorded, and the command printed the record's sha256 into it. With `record_sha` the printed
    digest is compared with that value (the digest of the bytes the caller parsed) and the record file is not read again."""
    out = stage_dir / OUTPUT_FILE
    rec = marker.get("output")
    want = os.path.relpath(out, run_dir)
    # One read: the digest compared with the supervisor's record and the text searched for the mark line come from the same bytes, so a file
    # replaced between two reads cannot pass with altered text (review of PR 498, round 5).
    try:
        raw = safeio.read_bytes(out, max_bytes=JSON_MAX * 4)
    except safeio.SafeReadError as exc:
        if not isinstance(rec, dict) or rec.get("path") != want:
            reasons.append(f"{OUTPUT_FILE} is not the output file the supervisor recorded in test-result.json")
        else:
            reasons.append(f"{OUTPUT_FILE} is not readable: {exc.reason}")
        return
    # The record's digest covers at most JSON_MAX bytes (`_sha`): a larger file never matched it.
    if (not isinstance(rec, dict) or rec.get("path") != want or len(raw) > JSON_MAX
            or rec.get("sha256") != hashlib.sha256(raw).hexdigest()):
        reasons.append(f"{OUTPUT_FILE} is not the output file the supervisor recorded in test-result.json")
        return
    text = raw.decode("utf-8", "replace")
    marks = re.findall(rf"^{OUTPUT_MARK} ([0-9a-f]{{64}})\s*$", text, re.M)
    starts = [l for l in text.splitlines() if l.startswith(OUTPUT_MARK)]
    if len(starts) != 1:
        reasons.append(f"{OUTPUT_FILE} has {len(starts)} lines that begin with {OUTPUT_MARK}; the template prints exactly one "
                       "(and prefixes every line the port prints), so something else wrote a line of its own")
    if not marks or marks[-1] != (record_sha if record_sha is not None else _sha(stage_dir / RECORD_FILE)):
        reasons.append(f"{OUTPUT_FILE} does not carry a {OUTPUT_MARK} line that matches {RECORD_FILE}, so the record was not "
                       "written by the test the supervisor ran")


def _source_reasons(record: dict, stage_dir: Path, run_dir: Path, reasons: list, stage: int = 2, entry: int | None = None) -> None:
    """The port's source files: named, unchanged since the record, and not obviously a way to avoid the chip."""
    PORT_DIR, PORT_MODULE = port_dir(stage, entry), port_module(stage, entry)          # this stage's (or entry's) own; the module constants are stage 2's
    sources = record.get("port_sources")
    if not isinstance(sources, list) or not sources:
        reasons.append(f"port_sources must list the port's source files ({PORT_DIR})")
        return
    model_text = []
    listed = set()
    for i, s in enumerate(sources):
        path = s.get("path") if isinstance(s, dict) else None
        role = s.get("role") if isinstance(s, dict) else None
        f = _inside(run_dir, path) if isinstance(path, str) and path.startswith(PORT_DIR) else None
        if f is None or role != MODEL_ROLE:
            reasons.append(f"port_sources[{i}] needs a file under {PORT_DIR} and a role of {MODEL_ROLE!r}")
            continue
        listed.add(path)
        if s.get("sha256") != _sha(f):
            reasons.append(f"port_sources[{i}] {path} changed after the record was written (sha256 differs)")
        if role == MODEL_ROLE:
            try:
                model_text.append((path, safeio.read_bytes(f, max_bytes=SOURCE_MAX).decode("utf-8", "replace")))
            except safeio.SafeReadError:
                reasons.append(f"port_sources[{i}] {path} is not readable")
    if not model_text:
        reasons.append(f"port_sources must include at least one file with the role {MODEL_ROLE!r}")
        return
    if PORT_MODULE not in listed:
        reasons.append(f"port_sources does not list {PORT_MODULE}, the port module the template calls")
    links = [os.path.join(r, n) for r, ds, fs in os.walk(stage_dir / "port") for n in ds + fs if os.path.islink(os.path.join(r, n))]
    if links:
        reasons.append(f"{os.path.relpath(links[0], run_dir)} is a link under {PORT_DIR}: its target is not hashed or read by the checks, "
                       "so files there must be regular files")
    on_disk = {os.path.relpath(f, run_dir) for f in (stage_dir / "port").rglob("*.py") if f.is_file()}
    if on_disk - listed:
        reasons.append(f"{sorted(on_disk - listed)[0]} is in {PORT_DIR} but not in port_sources: it was added after the record was written")
    if not any("ttnn" in t for _, t in model_text):
        reasons.append("no model source in port_sources mentions ttnn, so nothing shows the port runs on the chip")
    for path, t in model_text:
        if "generated-ids" in t or "evidence/reference" in t:
            reasons.append(f"model source {path} refers to the stage 1 reference files; model code does not read the reference, "
                           "and one that does could copy its ids")
