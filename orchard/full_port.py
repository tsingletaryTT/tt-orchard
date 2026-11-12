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
import math
import os
import re
from pathlib import Path

from orchard import safeio, ttnn_calls
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
IDIOMS_SOURCE = TEMPLATE.parent / "example_decoder_idioms.md"      # idioms from a port that passed the check (helper-vs-coder); advice, not code to copy
IDIOMS_NAME = "decoder_idioms.md"     # its name in the stage directory, next to the cheat sheet
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
TOP5_FILE = "stages/1/evidence/reference/top5-logits.json"      # the reference's log-probability of each token it chose (issue 731)
FIDELITY_TOLERANCE = 1e-6             # the gate's recomputed fidelity against the value in the record
TIE_GAP = 0.125            # equal to TIE_GAP in the port_check template (a test keeps them equal)
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

    if _ids(chip_ids) and chip_ids:
        _fidelity_reasons(record, run_dir, len(chip_ids), reasons)
        _length_check_reasons(record, reasons)
        if _ids(record.get("reference_ids")) and len(record["reference_ids"]) == len(chip_ids):
            _near_tie_reasons(record, run_dir, record["reference_ids"], chip_ids, reasons)

    # ---- a sparse attention the check did or did not exercise (issue 638) ----
    if stage == 2 and entry is None and _ids(chip_ids) and chip_ids:
        facts, prompt_n = _facts_of(stage_dir), _prompt_tokens(run_dir)
        if facts is not None and prompt_n is not None:
            reasons.extend(indexer_reasons(result, facts, prompt_tokens=prompt_n, generated=len(chip_ids)))

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
    _text_floor_reasons(text, rehearsal, waived, reasons)

    _template_reasons(record, stage_dir, run_dir, reasons)
    _source_reasons(record, stage_dir, run_dir, reasons, stage, entry)
    _device_ops_reasons(record, run_dir, reasons, stage_dir)
    return reasons, seen


WORDS_WAIVED = "free_run_words"      # the name `waived` receives when a rehearsal waived the word count of the free-run text (issue 568)


def _text_floor_reasons(text, rehearsal: bool, waived: list | None, reasons: list) -> None:
    """The floor on the free-run text: printable text, at least MIN_FREE_RUN_CHARS non-space characters, MIN_FREE_RUN_WORDS different words.
    In a rehearsal (the run's `--full-port-rehearsal`, from the ledger) the word count alone is waived: a random-weight model decodes token
    noise with no spaces, which is one long word. The character floor still applies, and `waived` receives WORDS_WAIVED so the supervisor
    can write the decision. Outside a rehearsal nothing changes."""
    if not _text(text) or len("".join(text.split())) < MIN_FREE_RUN_CHARS:
        reasons.append(f"free_run_text must be at least {MIN_FREE_RUN_CHARS} non-space characters and {MIN_FREE_RUN_WORDS} different words")
    elif len(set(text.split())) < MIN_FREE_RUN_WORDS:
        if rehearsal:
            if waived is not None and WORDS_WAIVED not in waived:
                waived.append(WORDS_WAIVED)
        else:
            reasons.append(f"free_run_text must be at least {MIN_FREE_RUN_CHARS} non-space characters and {MIN_FREE_RUN_WORDS} different words")


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
    elif cfg.get("path") != os.path.relpath(cfg_path, run_dir):
        # Issue 546: the hash alone does not say which file the record means. A record could keep the stage config's hash and name another
        # in-run config (one with a smaller layer count) as its path.
        reasons.append(f"{RECORD_FILE} config path {str(cfg.get('path'))[:120]!r} is not this stage's {os.path.relpath(cfg_path, run_dir)}: "
                       f"the record must name the {CONFIG_NAME} it hashed")
    free = record.get("free_run_ids")
    if not _ids(free) or not free:
        reasons.append(f"{RECORD_FILE} free_run_ids must be a non-empty list of whole numbers")


def sparse_budget(facts) -> tuple[int, str, int] | None:
    """(the length beyond which a sparse attention selects, its kind, the extra the rule adds) from the port facts' `sparse_attention`, or None
    for anything that is not a known kind with a plain positive number. For a query-sparse indexer the number is `budget` and the extra is the
    block size (it keeps the current block beyond the budget, so it selects only once the evaluated length exceeds budget + block size); for a
    sliding window it is `window` and the extra is 0. A `kind` that is not a string (a readable facts file may hold a list or an object) is
    unknown, not an error."""
    sparse = facts.get("sparse_attention") if isinstance(facts, dict) else None
    if not isinstance(sparse, dict):
        return None
    kind = sparse.get("kind")
    if not isinstance(kind, str):
        return None
    key = {"qsa-indexer": "budget", "sliding-window": "window"}.get(kind)
    value = sparse.get(key) if key else None
    if not (isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 10 ** 9):
        return None
    block = sparse.get("block_size") if kind == "qsa-indexer" else 0
    block = block if isinstance(block, int) and not isinstance(block, bool) and 0 <= block <= 10 ** 9 else 0
    return value, kind, block


def indexer_record(facts, *, prompt_tokens: int, generated: int) -> dict:
    """{"indexer_exercised", "indexer_basis"} for the stage 2 result, or {} when the facts carry no usable sparse attention (issue 638). The
    check feeds the prompt and every generated token but the last, so its longest forward pass evaluates prompt + generated - 1 inputs. The sparse
    part is exercised only when that length exceeds the budget plus the block (a query-sparse indexer) or the window; at or below it every key is
    selected, and a port that ignores the sparse part is indistinguishable from one that has it."""
    found = sparse_budget(facts)
    if found is None:
        return {}
    limit, kind, block = found
    evaluated = prompt_tokens + generated - 1
    name = "indexer" if kind == "qsa-indexer" else "sliding window"
    rule = f"the budget of {limit} plus the block of {block}" if (kind == "qsa-indexer" and block) else f"the {'budget' if kind == 'qsa-indexer' else 'window'} of {limit}"
    beyond = limit + block
    if evaluated > beyond:
        return {"indexer_exercised": True, "indexer_basis": f"the check's longest forward pass evaluated {evaluated} inputs, above {rule}"}
    return {"indexer_exercised": False,
            "indexer_basis": (f"{name} not exercised: the check's longest forward pass evaluated {evaluated} inputs, not above {rule}, so the {name} "
                              "selects every key and a port that ignores it agrees with the reference")}


def indexer_reasons(result: dict, facts, *, prompt_tokens: int, generated: int) -> list[str]:
    """Reasons when a stage 2 result's `indexer_exercised` and `indexer_basis` differ from what the port facts and the record give. Facts without a
    sparse attention give none (a run from before issue 638 has neither the field nor the facts)."""
    want = indexer_record(facts, prompt_tokens=prompt_tokens, generated=generated)
    if not want:
        return []
    out = []
    if result.get("indexer_exercised") is not want["indexer_exercised"]:
        out.append(f"result.json indexer_exercised must be {want['indexer_exercised']!r} (the check's longest context was "
                   f"{prompt_tokens + generated} tokens against the budget in the port facts), got {result.get('indexer_exercised')!r}")
    if result.get("indexer_basis") != want["indexer_basis"]:
        out.append("result.json indexer_basis differs from the one the gate recomputed from the port facts and the record")
    return out


def _prompt_tokens(run_dir, reasons=None) -> int | None:
    """The number of ids in the stage 1 prompt file, or None."""
    f = _inside(run_dir, PROMPT_FILE)
    if f is None:
        return None
    try:
        ids = safeio.read_json(f, max_bytes=JSON_MAX, expect=dict).get("prompt_ids")
    except safeio.SafeReadError:
        return None
    return len(ids) if _ids(ids) else None


def _facts_of(stage_dir) -> dict | None:
    """The port facts file of a stage 2 directory, or None when it is missing or unreadable. The file lives in the agent's directory, so a missing
    file removes only this note; it never makes a check pass that would otherwise fail."""
    try:
        data = safeio.read_json(Path(stage_dir) / "port_facts.json", max_bytes=JSON_MAX, expect=dict)
    except safeio.SafeReadError:
        return None
    return data


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


def fidelity_value(ref_lp: list, chip_lp: list, scale: float = 10.0) -> float:
    """10 * PPL_reference / PPL_port over the same tokens: scale * exp(mean(port log-probability) - mean(reference log-probability))."""
    n = len(ref_lp)
    return scale * math.exp(sum(chip_lp) / n - sum(ref_lp) / n)


def _lps(v, n: int) -> bool:
    return (isinstance(v, list) and len(v) == n
            and all(safeio.is_finite_number(x) and x <= 1e-6 for x in v))


def _fidelity_reasons(record: dict, run_dir: Path, positions: int, reasons: list) -> None:
    """The record's report-only `fidelity` block, when it has one: well formed, recomputed from its own log-probabilities, and its reference
    log-probabilities equal to the stage 1 file's. It never fails a run for being low (nothing gates on it yet); it fails a record that
    disagrees with itself or with the reference on disk, since that record is evidence."""
    block = record.get("fidelity")
    if block is None:
        return
    if not isinstance(block, dict):
        reasons.append("fidelity must be an object")
        return
    ref_lp, chip_lp = block.get("reference_logprobs"), block.get("chip_logprobs")
    if not _lps(ref_lp, positions) or not _lps(chip_lp, positions):
        reasons.append(f"fidelity reference_logprobs and chip_logprobs must each be {positions} finite log-probabilities (at most 0)")
        return
    value = block.get("value")
    if block.get("tokens") != positions:
        reasons.append(f"fidelity tokens {block.get('tokens')!r} differs from the {positions} positions the record compares")
    if not _num(value) or abs(value - fidelity_value(ref_lp, chip_lp, float(block.get("scale") or 10.0))) > FIDELITY_TOLERANCE * max(1.0, abs(value)):
        reasons.append(f"fidelity value {value!r} differs from the {fidelity_value(ref_lp, chip_lp):.6f} the gate recomputed from the record's log-probabilities")
    f = _inside(run_dir, TOP5_FILE)
    if f is None:
        reasons.append(f"fidelity: {TOP5_FILE} is not a file in the run directory, so the reference log-probabilities cannot be checked")
        return
    try:
        steps = safeio.read_json(f, max_bytes=JSON_MAX, expect=dict).get("steps")
    except safeio.SafeReadError as exc:
        reasons.append(f"{TOP5_FILE} is not readable JSON: {exc.reason}")
        return
    on_disk = [s.get("logprob") if isinstance(s, dict) else None for s in (steps if isinstance(steps, list) else [])[:positions]]
    if on_disk != ref_lp:
        reasons.append(f"fidelity reference_logprobs are not the first log-probabilities of the stage 1 reference ({TOP5_FILE}): "
                       "the score was made against numbers the reference did not produce")


def _near_tie_reasons(record: dict, run_dir: Path, ref_ids: list, chip_ids: list, reasons: list) -> None:
    """The record's report-only `near_ties` block, when it has one: recomputed here from the record's own ids and the stage 1 top-5 file. It never
    fails a run for the number of disagreements; it fails a block that does not match what the reference on disk says, since the block is
    evidence for reading the agreement."""
    block = record.get("near_ties")
    if block is None:
        return
    if not isinstance(block, dict):
        reasons.append("near_ties must be an object")
        return
    f = _inside(run_dir, TOP5_FILE)
    if f is None:
        reasons.append(f"near_ties: {TOP5_FILE} is not a file in the run directory, so the block cannot be checked")
        return
    try:
        steps = safeio.read_json(f, max_bytes=JSON_MAX, expect=dict).get("steps")
    except safeio.SafeReadError as exc:
        reasons.append(f"{TOP5_FILE} is not readable JSON: {exc.reason}")
        return
    n = len(chip_ids)
    if not isinstance(steps, list) or len(steps) < n or len(ref_ids) < n:
        reasons.append("near_ties: the stage 1 top-5 file has fewer steps than the record compares")
        return

    def logit(step, token):
        for e in (step.get("top5") if isinstance(step, dict) and isinstance(step.get("top5"), list) else []):
            if isinstance(e, dict) and e.get("id") == token and safeio.is_finite_number(e.get("logit")):
                return float(e["logit"])
        return None
    wrong = [i for i in range(n) if chip_ids[i] != ref_ids[i]]
    ties = []
    for i in wrong:
        a, b = logit(steps[i], ref_ids[i]), logit(steps[i], chip_ids[i])
        if a is not None and b is not None and abs(a - b) <= TIE_GAP:
            ties.append(i)
    if block.get("tie_gap") != TIE_GAP or block.get("disagreements") != len(wrong) or block.get("near_tie_positions") != ties \
            or block.get("near_ties") != len(ties):
        reasons.append(f"near_ties does not match what the gate recomputed from the record's ids and {TOP5_FILE}: "
                       f"{len(wrong)} disagreements, {len(ties)} near-ties at {ties} (gap {TIE_GAP})")
        return
    want = (n - len(wrong) + len(ties)) / n
    if not _num(block.get("agreement_excluding_near_ties")) or abs(block["agreement_excluding_near_ties"] - want) > 1e-9:
        reasons.append(f"near_ties agreement_excluding_near_ties {block.get('agreement_excluding_near_ties')!r} differs from the {want} the gate recomputed")


def _length_check_reasons(record: dict, reasons: list) -> None:
    """The record's report-only `length_check` block, when it has one: well formed and consistent with itself (each row's `agree` is whether its two
    ids are equal, `all_agree` is whether every row agrees). The block comes from the port's own prefill and decode, so the gate cannot recompute
    the ids; it never fails a run for a disagreement, and it fails a block that says something its own rows do not."""
    block = record.get("length_check")
    if block is None:
        return
    rows = block.get("lengths") if isinstance(block, dict) else None
    if not isinstance(rows, list):
        reasons.append("length_check must be an object with a list of lengths")
        return
    last = 0
    for r in rows:
        ok = (isinstance(r, dict) and _ids([r.get("length"), r.get("prefill_id"), r.get("decode_id")]) and isinstance(r.get("agree"), bool)
              and r["length"] > last)
        if not ok:
            reasons.append("length_check rows must each hold a length (rising), prefill_id, decode_id and a boolean agree")
            return
        last = r["length"]
        if r["agree"] != (r["prefill_id"] == r["decode_id"]):
            reasons.append(f"length_check at length {r['length']} says agree={r['agree']} but its ids are {r['prefill_id']} and {r['decode_id']}")
    if "error" in block:
        if not isinstance(block["error"], str) or "all_agree" in block:
            reasons.append("length_check with an error holds a string error and no all_agree")
    elif block.get("all_agree") is not all(r["agree"] for r in rows) or not isinstance(block.get("all_agree"), bool):
        reasons.append("length_check all_agree is not whether every row agrees")


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


HOST_SHARE_MAX = 0.5             # the template's bound: more than this share of the weight elements on the host fails (port_check.py)


def _snapshot_floor(stage_dir: Path):
    """The floor the checkpoint's own config gives (orchard/ttnn_calls.py `layer_floor`), read from the model snapshot that this stage's own
    `port_check_config.json` names (not from a path the record names: issue 546), or None when that cannot be read. The record's floor is
    the template's; this is the gate's own reading of the same input. A Hugging Face cache stores `config.json` as a link, so the link is
    resolved before the safe read (issue 548)."""
    try:
        snapshot = safeio.read_json(Path(stage_dir) / CONFIG_NAME, max_bytes=1 << 16, expect=dict).get("model_snapshot")
        if not isinstance(snapshot, str) or not snapshot or "\0" in snapshot:
            return None
        config = safeio.read_json(safeio.resolved(Path(snapshot) / "config.json"), max_bytes=1 << 20, expect=dict)
    except (safeio.SafeReadError, OSError, ValueError):
        return None
    return ttnn_calls.layer_floor(config)[0]


def _count(x) -> bool:
    return type(x) is int and 0 <= x <= 10 ** 12


def _device_ops_reasons(record: dict, run_dir: Path, reasons: list, stage_dir: Path) -> None:
    """The record's `device_ops` (written by the template from the calls its worker counted, issue 541) must show the port computed on the device:
    at least `floor` matmul-family ttnn calls in the first prefill and in some decode step (or trace replays that stand for them), and weights that
    are mostly ttnn tensors. A record without the field is refused: the check that wrote it did not count."""
    ops = record.get("device_ops")
    if not isinstance(ops, dict):
        reasons.append(f"{RECORD_FILE} has no device_ops: the check that wrote it did not count the port's ttnn calls, so nothing shows the port "
                       "computed on the device. Run the supervisor's check again")
        return
    floor = ops.get("floor")
    if type(floor) is not int or not 1 <= floor <= 10_000:
        reasons.append(f"{RECORD_FILE} device_ops.floor must be a whole number from 1 to 10000, got {floor!r}")
        return
    own = _snapshot_floor(stage_dir)
    if own is not None and floor < own:
        reasons.append(f"device_ops.floor is {floor}, below the {own} the checkpoint's config gives (one matmul-family call per layer)")
        floor = own
    prefill, decode = ops.get("prefill_matmul_calls"), ops.get("decode_matmul_calls_max")
    replays, total = ops.get("decode_trace_replays"), ops.get("matmul_calls_total")
    if not all(_count(x) for x in (prefill, decode, replays, total)):
        reasons.append(f"{RECORD_FILE} device_ops needs whole-number counts in prefill_matmul_calls, decode_matmul_calls_max, "
                       "decode_trace_replays and matmul_calls_total")
        return
    if prefill < floor:
        reasons.append(f"the first prefill made {prefill} matmul-family ttnn calls, below the floor of {floor} (one per layer): the port did not "
                       "compute its forward pass on the device. Move the computation onto the ttnn tensors (see stages/2/cheat_sheet.md)")
    if decode < floor and not (replays and total >= 2 * floor):
        reasons.append(f"no decode step made {floor} matmul-family ttnn calls (the most was {decode}, trace replays {replays}): the port did "
                       "not decode on the device")
    weights = ops.get("weights")
    if weights is not None and not (isinstance(weights, dict) and _count(weights.get("host_elements")) and _count(weights.get("device_elements"))):
        # Issue 549: absent weight evidence stays allowed (the scan did not run), but evidence that was supplied has to be well formed.
        reasons.append(f"{RECORD_FILE} device_ops.weights must be an object with whole-number host_elements and device_elements, got {str(weights)[:80]!r}")
    elif isinstance(weights, dict):
        host, dev = weights.get("host_elements"), weights.get("device_elements")
        if host + dev and host / (host + dev) > HOST_SHARE_MAX:
            reasons.append(f"the port's weights stayed on the host: {host} elements as torch or numpy tensors, {dev} as ttnn tensors. Upload them "
                           "with ttnn.from_torch(..., device=mesh_device) in load()")


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
    else:
        # A mention is not a call (issue 541): the 22 versions of run 18's port all had `import ttnn  # noqa` and no call. A source that parses must
        # have a reachable matmul-family ttnn call somewhere in the port.
        found = [ttnn_calls.static_calls(t) for _, t in model_text]
        if all(f["error"] is None for f in found) and not any(ttnn_calls.is_matmul(k) and ttnn_calls.is_compute(k) for f in found for k in f["calls"]):
            reasons.append("no model source in port_sources makes a reachable ttnn matmul-family call (ttnn.matmul, ttnn.linear and the like): "
                           "`import ttnn` is not a call, so the port computes on the host. Move the computation onto the device (see stages/2/cheat_sheet.md), "
                           "then run the check again")
    for path, t in model_text:
        if "generated-ids" in t or "evidence/reference" in t:
            reasons.append(f"model source {path} refers to the stage 1 reference files; model code does not read the reference, "
                           "and one that does could copy its ids")
