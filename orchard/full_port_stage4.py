# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4 of a full port, the supervisor's side (issue 454): run the port on each required chip count, one entry at a time.

`orchard/full_port_entries.py` says what the files are and judges them. This module drives the stage. `run` is the stage's body: the
supervisor calls it in place of the generic prepare, test and finish steps, and it returns what `Supervisor._stage_body` returns,
`(status, reasons, gate)`. It reaches the supervisor only through the methods the other stages also use (`_step`, `_pin_full_port`,
`_seed_stage_3_port`, `_run_listed`, `_check_gate`), so the lease rules are theirs: one board for a count of two or less, and for a larger
count the further board is leased first, the coder is parked, both boards go to the test and both come back (`_run_listed`).

For each entry, in order of chip count:

1. A count that stage 3 or stage 2 already showed (its marker, vouched for by the ledger, lists that many chips, and the stage passed) is
   written as `covered.json` with the path and sha256 of that stage's marker and record. No test runs.
2. Otherwise the supervisor copies the shipped template and its config into `configs/<N>/`, seeds `configs/<N>/port/` with the port that
   passed the stage before (stage 3's, or stage 2's when stage 3 was skipped), and gives the coder one prepare step (skill
   full-port-shrink) to adapt the port to the smaller mesh. The coder writes `hw_test.json` and `handoff.json`; the supervisor sets aside
   any result the coder wrote, puts the template back and rewrites `hw_test.json` to the entry's fixed command.
3. The test runs under a lease. Its marker is `configs/<N>/test-result.json` and the ledger holds that file's sha256.
4. The supervisor writes `configs/<N>/result.json` from the record (`orchard/full_port_result.py`) and runs the entry check. An entry
   that does not pass ends the stage: no later entry is started, so no chip time goes to a stage that has already failed.

The stage's own `result.json` is the summary the gate rebuilds. There is no finish step for an agent: nothing the agent could add would
change a record the template wrote. A resumed stage goes on at the first entry that has no vouched marker (and no covering).
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

from orchard import full_port, full_port_entries, full_port_result, safeio, swap_prepare
from orchard.defaults import CHIPS_PER_BOARD
from orchard.stages import GateResult, evidence_record, full_port_rehearsal, recorded_by_supervisor

COVERED_WHAT = "full-port entry covered"                 # the `what` of the ledger entry that vouches for a covered.json
ENTRY_RESULT_WHAT = "entry result derived by the supervisor"
STAGE_RESULT_WHAT = "stage result derived by the supervisor"
PLAN_WHAT = "hardware test list"                         # the same entry the weights-only stage 4 writes for tests/plan.json
PREPARED = "prepared.json"                               # written by the supervisor when an entry's prepare step is done (a resume skips it)
PREPARED_WHAT = "entry prepare step done"                # the `what` of the ledger entry that vouches for a prepared.json (review of PR 489)


class _Ended(Exception):
    """A step ended with a status that is not a failure of the entry: `abort`, `operator-pause`, `pause` or `escalate`. The stage ends with that
    status and the reasons, as the generic stage body does for its own steps (review of PR 489: every status had become a failed stage)."""

    def __init__(self, status: str, reasons: list[str]):
        super().__init__(status)
        self.status, self.reasons = status, reasons


ENDING_STATUSES = ("abort", "operator-pause", "pause", "escalate")      # the statuses a step can end with that the stage must pass on as they are


def _parts(sup, path) -> tuple[str, ...]:
    """The run-relative parts of `path`, for the operations that go through the pinned run directory."""
    return Path(os.path.relpath(path, sup.run_dir)).parts


def _write_json_beneath(sup, pin, path: Path, data) -> Path:
    """`full_port_entries.write_json` through the pinned run directory: a link standing in the entry's path is refused, never followed."""
    safeio.write_beneath(pin.fd, _parts(sup, path), (json.dumps(data, indent=2) + "\n").encode("utf-8"), mode=0o644)
    return path


# ---- what the ledger says ----------------------------------------------------------------------------------------------

def _latest_end(entries: list[dict], stage: int) -> str | None:
    """The result of the latest `stage_end` of `stage` ("pass", "fail", "skipped", ...), or None."""
    found = None
    for e in entries:
        if e.get("event") == "stage_end" and e.get("stage") == stage:
            found = (e.get("data") or {}).get("result")
    return found


def seed_stage(entries: list[dict]) -> int:
    """The stage whose passed port the entries start from: stage 3 when it ran and passed, else stage 2 (stage 3 was skipped because the run
    lists no count above one board)."""
    return 3 if _latest_end(entries, 3) == "pass" else 2


def covering_stage(ledger, run_dir, chips: int) -> dict | None:
    """The stage that already showed `chips` chips, or None. Stage 3 is asked first, then stage 2. A stage covers a count when its latest
    end is a pass, its marker (test-result.json) is a file the supervisor recorded in the ledger, and the marker shows an exit code of 0,
    no timeout and exactly `chips` chip names. Returns {"covered_by", "evidence", "record"} with the path and sha256 of the marker and of
    the record, read through the bounded reader."""
    run_dir = Path(run_dir)
    entries = ledger.read()
    for k in full_port_entries.COVERED_BY:
        if _latest_end(entries, k) != "pass":
            continue
        marker = full_port.file_record(run_dir, run_dir / "stages" / str(k) / full_port_entries.MARKER_FILE)
        record = full_port.file_record(run_dir, run_dir / "stages" / str(k) / full_port.RECORD_FILE)
        if marker is None or record is None or not recorded_by_supervisor(ledger, k, marker):
            continue
        if full_port_entries.marker_problem(run_dir / "stages" / str(k), chips) is None and not full_port_entries.chain_problems(run_dir, k):
            return {"covered_by": k, "evidence": marker, "record": record}
    return None


# ---- the gate's ledger half --------------------------------------------------------------------------------------------

def _vouched(ledger, run_dir, n: int, path, what: str, label: str, reasons: list) -> None:
    """A file the supervisor writes: a reason unless it is there as a regular file and the ledger holds an entry of kind `what` for stage `n`
    with its path and sha256."""
    rec = full_port.file_record(run_dir, path)
    if rec is None or not recorded_by_supervisor(ledger, n, rec, whats=(what,)):
        reasons.append(f"{label} is not the file the supervisor recorded (the ledger holds no `{what}` entry with its sha256): "
                       "a file the supervisor writes must match the ledger")


def vouch(sup, spec, stage_dir, gate: GateResult, *, waived_decision: str) -> GateResult:
    """The checks the gate cannot make because it has no ledger: the plan, each entry's marker or covering, each entry's result and the stage's
    result are the files the supervisor wrote (their sha256 is in the ledger), and a covering stage still shows a pass in the ledger. A
    passing gate also leaves the rehearsal decision when `coherent` was waived, as stages 2 and 3 do."""
    run_dir, ledger, n = sup.run_dir, sup.ledger, spec.number
    stage_dir = Path(stage_dir)
    reasons: list[str] = []
    _vouched(ledger, run_dir, n, stage_dir / full_port_entries.PLAN, PLAN_WHAT, "tests/plan.json", reasons)
    for c in sorted(set(sup.required_chips or ())):
        edir, where = full_port_entries.entry_dir(stage_dir, c), f"the {c}-chip entry"
        if (edir / full_port_entries.COVERED_FILE).exists() or (edir / full_port_entries.COVERED_FILE).is_symlink():
            _vouched(ledger, run_dir, n, edir / full_port_entries.COVERED_FILE, COVERED_WHAT, f"{where}: covered.json", reasons)
            covered = full_port_entries.read_covered(edir) or {}
            k = covered.get("covered_by")
            if k in full_port_entries.COVERED_BY and type(k) is int:
                marker = full_port.file_record(run_dir, run_dir / "stages" / str(k) / full_port_entries.MARKER_FILE)
                if _latest_end(ledger.read(), k) != "pass" or marker is None or not recorded_by_supervisor(ledger, k, marker):
                    reasons.append(f"{where}: stage {k} covers it, and the ledger shows no pass of stage {k} with a marker the supervisor recorded")
            continue
        rec = full_port.file_record(run_dir, edir / full_port_entries.MARKER_FILE)
        if rec is None or not recorded_by_supervisor(ledger, n, rec):
            reasons.append(f"{where}: test-result.json is not the file the supervisor recorded when the entry's test ended "
                           "(the ledger holds no matching entry), so its exit code cannot be trusted")
        _vouched(ledger, run_dir, n, edir / full_port_entries.RESULT_FILE, ENTRY_RESULT_WHAT, f"{where}: result.json", reasons)
    _vouched(ledger, run_dir, n, stage_dir / full_port_entries.RESULT_FILE, STAGE_RESULT_WHAT, "result.json", reasons)
    if reasons:
        return GateResult(False, tuple(gate.reasons) + tuple(r for r in reasons if r not in gate.reasons), gate.evidence, gate.waived, gate.unverified)
    if gate.ok and "coherent" in gate.waived:
        ledger.append("decision", n, decision=waived_decision, field="coherent", by="--full-port-rehearsal",
                      note=f"the model is a rehearsal with random weights; every other stage {n} check still applied")
    return gate


# ---- the coder's prepare step ------------------------------------------------------------------------------------------

def entry_note(*, chips: int, rel: str, seed: int, copied: bool) -> str:
    """What the supervisor adds to the coder's prepare task for one entry. The generic task names only hw_test.json and handoff.json, and its
    stage directory is the entry's."""
    command = full_port.command(full_port.ENTRY_STAGE, chips)
    parts = [f"This is stage 4 of a full port, entry {chips}. Your stage directory is `{rel}`. The port that passed stage {seed} was shown "
             f"on a larger mesh; this step adapts it to {chips} chip{'s' if chips != 1 else ''}"
             + (" (one device, no mesh)" if chips == 1 else f" (a mesh of 1 x {chips})") + ". "
             f"The check is the supervisor's: when you finish, it puts its own `{rel}/port_check.py` back and rewrites `hw_test.json` to its "
             f"fixed command (`{command}`). Your work is the port under `{rel}/port/`; write nothing under `stages/{seed}/` or under another "
             "entry's directory."]
    if copied:
        parts.append(f"`{rel}/port/` already holds the files of the stage {seed} port, copied by the supervisor. Change what the smaller mesh "
                     "needs: how each weight is placed (replicated, or sharded across fewer devices), where a collective is now unnecessary "
                     "or still required, and how `load`, `prefill` and `decode_step` use the mesh device they are given. Leave what already works.")
    else:
        parts.append(f"The supervisor could not copy the stage {seed} port (the ledger says why). Copy the files of `stages/{seed}/port/` into "
                     f"`{rel}/port/` yourself, then adapt them.")
    parts.append(f"When the port is ready, write `hw_test.json` as {{\"command\": \"{command}\", \"deadline_s\": {full_port.DEADLINE_S}}} and "
                 "`handoff.json`, then reply with a short summary. Do not run the check yourself.")
    return " ".join(parts)


# ---- the stage ---------------------------------------------------------------------------------------------------------

def _fail(sup, spec, stage_dir, entries, reasons: list[str]):
    _write_stage_marker(sup, stage_dir, entries)
    return "fail", reasons, None


def _write_stage_marker(sup, stage_dir: Path, entries) -> None:
    """The stage's own test-result.json: which entries have a marker or a covering. It lets a resumed, escalated attempt see an entry that
    failed (`Supervisor._run_stage`), as the weights-only stage 4's summary does.

    The marker is written through its own descriptor of the run directory, because `_fail` and the `_Ended` handler run outside the pin of
    `run`. A link standing in `stages/` or in the stage directory is refused, never followed (review of PR 489). A refusal must not hide the
    failure or the pause that led here, so it is written to the ledger and the caller goes on with its own status."""
    rows = []
    for e in entries:
        edir = e.directory(stage_dir)
        rows.append({"chips": e.chips, "covered": (edir / full_port_entries.COVERED_FILE).is_file(),
                     "marker": os.path.relpath(e.record_path(stage_dir), sup.run_dir) if e.record_path(stage_dir).is_file() else None})
    try:
        with safeio.pinned_root(sup.run_dir) as pin:
            _write_json_beneath(sup, pin, Path(stage_dir) / full_port_entries.MARKER_FILE, {"entries": rows})
    except (safeio.SafeReadError, OSError) as exc:
        sup.ledger.append("decision", _stage_number(stage_dir), decision="the stage's test-result.json was not written",
                          reason=getattr(exc, "reason", None) or str(exc))


def _stage_number(stage_dir) -> int:
    """The stage number from a stage directory named `stages/<n>`."""
    return int(Path(stage_dir).name)


def _set_aside_results(sup, pin, edir: Path) -> list[str]:
    """`swap_prepare.set_aside_results` through the pinned run directory: each result file the coder left in the entry moves to
    `<name>.rejected-from-prepare` with a descriptor-anchored rename. A link standing in the entry's directory or in `evidence/` is refused
    and the rename never leaves the run; a link at the file's own name is moved as a link (review of PR 489)."""
    moved = []
    base = _parts(sup, edir)
    for name in swap_prepare.RESULT_FILES:
        src = base + tuple(name.split("/"))
        dst = src[:-1] + (src[-1] + swap_prepare.REJECTED_SUFFIX,)
        if safeio.replace_beneath(pin.fd, src, dst):
            moved.append(name)
    return moved


def _have_boards(sup) -> int | None:
    try:
        return len({c.board for c in sup.adapter.status()})
    except Exception:           # the adapter is asked again by the lease; a failure here is not this stage's to name
        return None


def run(sup, spec, stage_dir, escalated: bool, resumed: bool):
    """Stage 4's body: the plan, then each entry. Returns (status, reasons, gate)."""
    n = spec.number
    stage_dir = Path(stage_dir)
    entries = None
    if resumed:
        entries, why = full_port_entries.load_plan(stage_dir)
        if entries is None and why and "missing" not in why:
            sup.ledger.append("decision", n, decision="hardware test plan refused, so the plan is written again", file=full_port_entries.PLAN, reason=why)
    if entries is not None and [e.chips for e in entries] != sorted(set(sup.required_chips or ())):
        sup.ledger.append("decision", n, decision="hardware test plan differs from --required-chips, so the plan is written again",
                          file=full_port_entries.PLAN)
        entries = None
    if entries is None:
        entries, problems = full_port_entries.plan_entries(sup.required_chips, max_chips=spec.boards * CHIPS_PER_BOARD, budget_s=spec.budget_s)
        if problems:
            return "fail", problems, None
        plan = full_port_entries.write_plan(stage_dir, entries)
        sup.ledger.append("evidence", n, what=PLAN_WHAT, configs=[e.chips for e in entries], **evidence_record(sup.run_dir, plan))
    # A test lease left from a block (a container that would not stop) goes back first, as in the weights-only stage 4.
    if not sup._release_test_lease():
        sup._block(n, "releasing a test lease left from an earlier test failed")
    have = _have_boards(sup) if sup.lab is None else None
    for entry in entries:
        if have is not None and entry.boards > have:
            sup._block(n, f"the {entry.chips}-chip entry needs {entry.boards * CHIPS_PER_BOARD} chips and this box has {have * CHIPS_PER_BOARD}",
                       need_chips=entry.boards * CHIPS_PER_BOARD)
    rehearsal = full_port_rehearsal(sup.ledger.read())
    try:
        # Every removal and write the stage makes in an entry goes through this descriptor of the run directory, so a link standing in
        # `configs/`, in an entry's directory or in its `evidence/` is refused and the operation never leaves the run (review of PR 489).
        with safeio.pinned_root(sup.run_dir) as pin:
            for entry in entries:
                try:
                    reasons = _one_entry(sup, spec, stage_dir, entry, escalated, resumed, rehearsal, pin)
                except safeio.SafeReadError as exc:
                    reasons = [f"the {entry.chips}-chip entry: a path in it was refused ({exc.reason}); the supervisor does not follow a link there"]
                if reasons:
                    return _fail(sup, spec, stage_dir, entries, reasons)
    except _Ended as ended:
        _write_stage_marker(sup, stage_dir, entries)
        return ended.status, ended.reasons, None
    except safeio.SafeReadError as exc:
        return _fail(sup, spec, stage_dir, entries, [f"the run directory could not be opened for the stage's entries: {exc.reason}"])
    summary = full_port_entries.summarize(stage_dir, sup.run_dir, [e.chips for e in entries])
    path = full_port_entries.write_json(stage_dir / full_port_entries.RESULT_FILE, summary)
    sup.ledger.append("evidence", n, what=STAGE_RESULT_WHAT, **evidence_record(sup.run_dir, path))
    _write_stage_marker(sup, stage_dir, entries)
    gate = sup._check_gate(spec, stage_dir)
    return ("pass", [], gate) if gate.ok else ("fail", list(gate.reasons), gate)


def _one_entry(sup, spec, stage_dir: Path, entry, escalated: bool, resumed: bool, rehearsal: bool, pin) -> list[str]:
    """Run one entry to the point where it passes (no reasons returned) or fails (the reasons)."""
    n, c = spec.number, entry.chips
    edir = entry.directory(stage_dir)
    rel = os.path.relpath(edir, sup.run_dir)
    where = f"the {c}-chip entry"
    ledger, run_dir = sup.ledger, sup.run_dir
    marker = entry.record_path(stage_dir)
    covered_path = edir / full_port_entries.COVERED_FILE
    safeio.mkdir_beneath(pin.fd, _parts(sup, edir))        # refuses a link in `configs/` or in the entry's directory before anything is read or removed
    # ---- a covering already written and vouched for: nothing to run ----
    if covered_path.is_file() and (rec := full_port.file_record(run_dir, covered_path)) and recorded_by_supervisor(ledger, n, rec, whats=(COVERED_WHAT,)):
        return []
    if safeio.unlink_beneath(pin.fd, _parts(sup, covered_path)):
        pass                           # a covered.json the ledger does not hold was not written by the supervisor
    # ---- the test: done and vouched for, or to be run ----
    tested = False
    if marker.is_file() and (rec := full_port.file_record(run_dir, marker)) and recorded_by_supervisor(ledger, n, rec):
        tested = True
    else:
        for stale in (marker, edir / full_port.RECORD_FILE, edir / full_port.OUTPUT_FILE, edir / full_port_entries.RESULT_FILE):
            if safeio.unlink_beneath(pin.fd, _parts(sup, stale)):
                pass                   # a marker the ledger does not hold (a kill between the file and its entry, or a file the coder wrote)
    if not tested:
        cover = covering_stage(ledger, run_dir, c)
        if cover is not None:
            path = _write_json_beneath(sup, pin, covered_path, {"chips": c, **cover})
            ledger.append("evidence", n, what=COVERED_WHAT, **evidence_record(run_dir, path))
            ledger.append("decision", n, decision="entry covered by an earlier stage", config=c, covered_by=cover["covered_by"],
                          evidence=cover["evidence"], record=cover["record"])
            return []
        problems, deadline = _prepare(sup, spec, stage_dir, entry, escalated, resumed, pin)
        if problems:
            return [f"{where}: {p}" for p in problems]
        # The deadline the supervisor's check pinned into hw_test.json (the coder's own, when it is shorter than the default) bounds the test; the
        # plan's deadline stays the longest it can be (review of PR 489).
        runnable = dataclasses.replace(entry, deadline_s=min(entry.deadline_s, float(deadline))) if deadline else entry
        sup._run_listed(spec, stage_dir, runnable)
    # ---- the entry's result, from the record ----
    why = full_port_entries.marker_problem(edir, c)
    if why:
        return [f"{where}: {why}"]
    path = edir / full_port_entries.RESULT_FILE
    result, because = full_port_result.derive(edir, run_dir, rehearsal=rehearsal)
    if result is None:
        return [f"{where}: no result could be derived from the check's record: {because}"]
    _write_json_beneath(sup, pin, path, result)
    reasons, _ = full_port.check(edir, run_dir, rehearsal=rehearsal, stage=full_port.ENTRY_STAGE, entry=c)
    if reasons:
        safeio.unlink_beneath(pin.fd, _parts(sup, path))
        ledger.append("decision", n, decision="entry result refused by the check", config=c, why="; ".join(reasons[:3])[:300])
        return [f"{where}: {r}" for r in reasons]
    ledger.append("evidence", n, what=ENTRY_RESULT_WHAT, **evidence_record(run_dir, path))
    ledger.append("decision", n, decision="entry passed", config=c, top1_agreement=result["top1_agreement"], n_tokens=result["n_tokens"],
                  coherent=result["coherent"], file=rel + "/result.json")
    return []


def _prepare(sup, spec, stage_dir: Path, entry, escalated: bool, resumed: bool, pin) -> tuple[list[str], float | None]:
    """The entry's prepare phase and the supervisor's own steps around it. Returns (the problems that stop the entry, the deadline of the
    entry's hw_test.json when it was read). A step that ends `abort`, `operator-pause`, `pause` or `escalate` raises `_Ended`."""
    n = spec.number
    edir = entry.directory(stage_dir)
    rel = os.path.relpath(edir, sup.run_dir)
    safeio.mkdir_beneath(pin.fd, _parts(sup, edir / "evidence"))
    if problems := sup._pin_full_port(spec, edir, entry=entry.chips):
        return problems, None
    seed = seed_stage(sup.ledger.read())
    sup._seed_stage_3_port(spec, edir, source=seed)
    prepared = edir / PREPARED
    # A prepared.json skips the coder's prepare step only when the ledger holds the supervisor's record of these bytes: a file the coder (or
    # anything else) wrote there has none, and the prepare step runs (review of PR 489).
    prepared_rec = full_port.file_record(sup.run_dir, prepared)
    if prepared_rec is None or not recorded_by_supervisor(sup.ledger, n, prepared_rec, whats=(PREPARED_WHAT,)):
        seeded = getattr(sup, "_port_seed", None) or {}
        note = entry_note(chips=entry.chips, rel=rel, seed=seed, copied=bool(seeded.get("copied") or seeded.get("kept")))
        out, step = sup._step(spec, "prepare", edir, escalated, resumed, extra=note)
        out = sup._wrap_up(spec, edir, step, out)
        if out.status != "done":
            why = [f"the prepare step ended: {out.status} {out.detail}".strip()]
            if out.status in ENDING_STATUSES:
                raise _Ended(out.status, [f"the {entry.chips}-chip entry: {why[0]}"])
            return why, None
        _write_json_beneath(sup, pin, prepared, {"chips": entry.chips, "seed_stage": seed})
        sup.ledger.append("evidence", n, what=PREPARED_WHAT, config=entry.chips, **evidence_record(sup.run_dir, prepared))
    # Nothing in the entry may claim a result before the test has run; the template and the fixed command go back as shipped.
    if moved := _set_aside_results(sup, pin, edir):
        sup.ledger.append("decision", n, decision="results written before the test were set aside", config=entry.chips, files=moved)
    if problems := sup._pin_full_port(spec, edir, command=True, entry=entry.chips):
        return problems, None
    sup._check_port_cache(spec, edir)
    test, problems = sup._read_test(edir)
    if not problems:
        # The park reads the coder's note from stages/<n>/handoff.json (`Supervisor._handoff`). The entry's note, which `_read_test` has just
        # checked, is copied there by the supervisor, so a wide entry can park the coder and the stand-in is told what the coder was doing.
        data = safeio.read_bytes(edir / "handoff.json", max_bytes=full_port_entries.JSON_MAX)
        safeio.atomic_write(Path(stage_dir) / "handoff.json", data, mode=0o644)
    deadline = test.get("deadline_s") if isinstance(test, dict) and not problems else None
    return problems, deadline
