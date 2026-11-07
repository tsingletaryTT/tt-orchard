# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4 of a full port (issue 454): the port on each required chip count, one record and one marker per entry.

Stage 2 shows the port on one board and stage 3 on the largest of the run's `--required-chips`. Stage 4 shows it on every other count the
operator listed. This module is the part of that stage that reads files and judges them. It starts nothing, holds no state, and imports
only `orchard.full_port` and `orchard.safeio`, so `orchard/stages.py` can build the stage's gate from it. The code that runs the entries
under leases is `orchard/full_port_stage4.py`.

Files, under the stage directory `stages/4/`:

    tests/plan.json               written by the supervisor from `--required-chips`: one entry per count, in order of chip count, in the
                                  shape of the weights-only stage 4's plan (`orchard/hwtests.py`). It is the stage's resume marker, and
                                  the ledger holds its sha256 ("hardware test list"). An agent never chooses a count.
    configs/<N>/                  one directory per entry. It holds that entry's copy of the shipped `port_check.py`, its config, its port
                                  (`port/`, seeded from the port that passed the stage before), its evidence and its marker:
        test-result.json          the marker, written by the supervisor when the entry's test ends (exit code, timeout, chips, output)
        evidence/port-check.json  the record the template wrote
        evidence/hw-test-output.txt  the command's output
        result.json               written by the supervisor from the record (`orchard/full_port_result.py`)
    configs/<N>/covered.json      written when the entry's count is one that stage 2 or stage 3 already showed. It names the
                                  stage and the path and sha256 of that stage's marker and record. No test runs for such an entry.
    result.json                   the stage's summary, written by the supervisor. The gate rebuilds it from the entries' records
                                  (`summarize`) and refuses a file that differs.
    test-result.json             the stage's list of entry markers, written by the supervisor at the end of the stage (it lets a resumed,
                                  escalated attempt see that an entry failed, as the weights-only stage 4's summary does).

Each entry is judged by the check of stages 2 and 3 with the entry's directory in place of the stage's (`full_port.check(..., entry=N)`):
the same record shape, the same provenance checks, and the same rehearsal rule, where `--full-port-rehearsal` waives the `coherent` field
and no other check. Every entry is one the operator required, so none is optional: one that does not pass ends the stage.

What this does not show: the checks tie each record to a test the supervisor ran on leased chips, and to the shipped template. They do
not prove the chip computed the ids (see `orchard/full_port.py`). A covered entry shows the count only as far as the covering stage's
record does. The ledger checks (a marker the supervisor wrote, a plan the supervisor wrote) are in `orchard/full_port_stage4.py`, because
they need the ledger.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from orchard import full_port, safeio
from orchard.defaults import CHIPS_PER_BOARD

STAGE = full_port.ENTRY_STAGE
PLAN = "tests/plan.json"                  # the same resume marker the weights-only stage 4 uses (orchard/stages.py, VOUCHED_MARKERS)
SCRIPT = "port_check.py"                  # the plan's `script` field: the shipped template, which the supervisor copies into each entry
COVERED_FILE = "covered.json"
MARKER_FILE = "test-result.json"
RESULT_FILE = "result.json"
COVERED_BY = (3, 2)                       # the stages whose chip count can cover an entry, the larger first
JSON_MAX = 16 * 1024 * 1024
MAX_ENTRIES = 16                          # choice: a run lists a few chip counts (the box has four chips); a longer list is a mistake
DERIVED_BY = "supervisor"
ENTRY_FIELDS = ("chips", "covered_by", "top1_agreement", "n_tokens", "load_s", "coherent", "evidence")


@dataclass(frozen=True)
class PortEntry:
    """One required chip count. The fields are those of `orchard.hwtests.HwTest` (so the plan has the same shape), and the methods are
    the ones the supervisor's test runner asks of a test: its command, and where its marker and output go."""
    chips: int
    script: str
    deadline_s: float
    cache: str = ""                       # the port converts nothing into a shared tensor cache the plan could name

    @property
    def boards(self) -> int:
        return -(-self.chips // CHIPS_PER_BOARD)

    def command(self, stage: int = STAGE) -> str:
        return full_port.command(stage, self.chips)

    def directory(self, stage_dir) -> Path:
        return entry_dir(stage_dir, self.chips)

    def record_path(self, stage_dir) -> Path:
        """The entry's marker: written by the supervisor when the entry's test ends."""
        return self.directory(stage_dir) / MARKER_FILE

    def output_path(self, stage_dir) -> Path:
        """The entry's test output. The template prints `PORT_CHECK_SHA256 <sha256 of the record>` as its last line."""
        return self.directory(stage_dir) / full_port.OUTPUT_FILE


def entry_dir(stage_dir, chips: int) -> Path:
    return Path(stage_dir) / "configs" / str(chips)


def plan_entries(required, *, max_chips: int, budget_s: float) -> tuple[list[PortEntry], list[str]]:
    """(entries, problems): one entry per distinct count in `required` (the operator's `--required-chips`), in order of chip count. A count
    that is not a whole number of at least 1, a count above `max_chips` (what the stage can lease), more than MAX_ENTRIES counts, or
    deadlines that add up to more than the stage's budget are problems; with a problem the supervisor runs nothing."""
    problems: list[str] = []
    if not isinstance(required, (list, tuple)) or not required:
        return [], ["the run lists no chip count (--required-chips), so stage 4 of a full port has nothing to run"]
    counts: list[int] = []
    for c in required:
        if type(c) is not int or c < 1:
            problems.append(f"--required-chips holds {c!r}; each count must be a whole number of at least 1")
        elif c > max_chips:
            problems.append(f"the {c}-chip entry needs more chips than stage 4 can lease ({max_chips})")
        elif c not in counts:
            counts.append(c)
    if len(counts) > MAX_ENTRIES:
        problems.append(f"--required-chips lists {len(counts)} distinct counts; at most {MAX_ENTRIES} are run")
    entries = [PortEntry(c, SCRIPT, float(full_port.DEADLINE_S)) for c in sorted(counts)]
    total = sum(e.deadline_s for e in entries)
    if total > budget_s:
        problems.append(f"the entries' deadlines add up to {total:g} s, more than the stage budget of {budget_s:g} s")
    return ([] if problems else entries), problems


def write_json(path, data) -> Path:
    """Whole or not at all: a crash never leaves half a file, and a FIFO or a link at the path is replaced, never opened."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safeio.atomic_write(path, (json.dumps(data, indent=2) + "\n").encode("utf-8"), mode=0o644)
    return path


def plan_data(entries: list[PortEntry]) -> dict:
    """The content of tests/plan.json for `entries`."""
    return {"tests": [{"chips": e.chips, "script": e.script, "deadline_s": e.deadline_s, "cache": e.cache} for e in entries]}


def write_plan(stage_dir, entries: list[PortEntry]) -> Path:
    return write_json(Path(stage_dir) / PLAN, plan_data(entries))


def load_plan(stage_dir) -> tuple[list[PortEntry] | None, str | None]:
    """(entries, why not): the entries in tests/plan.json, or None with the reason the file cannot be used. A plan that is not there says
    so; one that is there and refused (NaN, a link out of the run, over the cap, not a list of entries) says why."""
    got = safeio.read_json_checked(Path(stage_dir) / PLAN, max_bytes=JSON_MAX, expect=dict)
    if not got.ok:
        return None, (got.reason if got.refused else "tests/plan.json is missing")
    items = got.value.get("tests")
    if not isinstance(items, list) or not items or len(items) > MAX_ENTRIES:
        return None, "tests/plan.json holds no list of entries (or too long a one)"
    out = []
    for t in items:
        ok = (isinstance(t, dict) and type(t.get("chips")) is int and t["chips"] >= 1 and t.get("script") == SCRIPT
              and isinstance(t.get("deadline_s"), (int, float)) and not isinstance(t["deadline_s"], bool) and t["deadline_s"] > 0
              and t.get("cache") == "")
        if not ok:
            return None, f"tests/plan.json has an entry that is not one of the supervisor's ({str(t)[:80]})"
        out.append(PortEntry(t["chips"], t["script"], float(t["deadline_s"]), ""))
    if [e.chips for e in out] != sorted({e.chips for e in out}):
        return None, "tests/plan.json lists the chip counts out of order or more than once"
    return out, None


# ---- what one entry's files say ----------------------------------------------------------------------------------------------

def _read(path) -> dict | None:
    got = safeio.read_json_checked(Path(path), max_bytes=JSON_MAX, expect=dict)
    return got.value if got.ok else None


def read_covered(edir) -> dict | None:
    """The entry's covered.json as a dict, or None when there is none (or it cannot be read)."""
    return _read(Path(edir) / COVERED_FILE)


def marker_problem(edir, chips: int) -> str | None:
    """Why the entry's marker does not show a finished test on `chips` chips, or None. The marker is the supervisor's record: an exit
    code of 0, no timeout, the entry's command and as many chip names as the entry's count."""
    path = Path(edir) / MARKER_FILE
    got = safeio.read_json_checked(path, max_bytes=JSON_MAX, expect=dict)
    if not got.ok:
        return "the supervisor has no record of its test" if not got.refused else f"its marker is not readable: {got.reason}"
    m = got.value
    if m.get("timed_out") is not False:
        return f"its test did not finish before its deadline (timed_out {m.get('timed_out')!r})"
    if type(m.get("returncode")) is not int or m["returncode"] != 0:
        return f"its test exited {m.get('returncode')!r}"
    chips_ran = m.get("chips")
    if not isinstance(chips_ran, list) or len(chips_ran) != chips:
        return f"its test ran on {chips_ran!r}, which is not {chips} chips"
    return None


def failed(stage_dir, required) -> dict | None:
    """None when no entry that has a marker shows a failed test. Otherwise {"configs": [counts], "problem": one sentence per failure}. An
    entry with no marker yet has not failed. Used by the supervisor's resume rule for a stage whose test failed."""
    bad = []
    for c in sorted(set(required or ())):
        edir = entry_dir(stage_dir, c)
        if (edir / MARKER_FILE).is_file() or (edir / MARKER_FILE).is_symlink():
            why = marker_problem(edir, c)
            if why:
                bad.append((c, why))
    if not bad:
        return None
    return {"configs": [c for c, _ in bad], "problem": "; ".join(f"the {c}-chip entry: {w}" for c, w in bad)}


def chain_problems(run_dir, stage: int) -> list[str]:
    """Why the stage's marker, output and record do not hang together, or []. The marker names the supervisor's output file and its sha256;
    the output holds the one line the template printed, which carries the record's sha256. A covering rests on all three, so the record
    that covers an entry must be the one the test printed (review of PR 489: the hashes of the marker and of the record alone let a
    rewritten record cover an entry)."""
    run_dir = Path(run_dir)
    stage_dir = run_dir / "stages" / str(stage)
    marker = _read(stage_dir / MARKER_FILE)
    if marker is None:
        return [f"stage {stage}'s marker is not readable JSON"]
    reasons: list[str] = []
    full_port._output_reasons(stage_dir, run_dir, marker, reasons)
    return [f"stage {stage}: {r}" for r in reasons]


def _covered_reasons(edir: Path, run_dir: Path, chips: int, covered: dict) -> tuple[list[str], list[str]]:
    """The file-level checks of a covered entry. The ledger's part (the covering stage passed, its marker is the supervisor's) is in
    `orchard/full_port_stage4.py`."""
    reasons: list[str] = []
    seen: list[str] = []
    stage = covered.get("covered_by")
    if covered.get("chips") != chips or type(covered.get("chips")) is not int:
        reasons.append(f"{COVERED_FILE} is for {covered.get('chips')!r} chips and this entry is for {chips}")
    if stage not in COVERED_BY or type(stage) is not int:
        reasons.append(f"{COVERED_FILE} names stage {stage!r}; only stages {list(COVERED_BY)} cover an entry")
        return reasons, seen
    for key, want in (("evidence", f"stages/{stage}/{MARKER_FILE}"), ("record", f"stages/{stage}/{full_port.RECORD_FILE}")):
        ref = covered.get(key)
        if not isinstance(ref, dict) or ref.get("path") != want:
            reasons.append(f"{COVERED_FILE} {key}.path must be {want!r}, got {ref.get('path') if isinstance(ref, dict) else ref!r}")
            continue
        f = full_port._inside(run_dir, want)
        if f is None or full_port._sha(f) != ref.get("sha256"):
            reasons.append(f"{want} is missing or is not the file the covering of this entry rested on (its sha256 differs from {COVERED_FILE})")
        else:
            seen.append(want)
    marker = _read(run_dir / f"stages/{stage}/{MARKER_FILE}")
    if marker is None or type(marker.get("returncode")) is not int or marker["returncode"] != 0 or marker.get("timed_out") is not False:
        reasons.append(f"stage {stage}'s marker does not show a passing test, so it covers nothing")
    elif not isinstance(marker.get("chips"), list) or len(marker["chips"]) != chips:
        reasons.append(f"stage {stage}'s test ran on {marker.get('chips')!r}, which is not {chips} chips, so it does not cover this entry")
    reasons.extend(chain_problems(run_dir, stage))
    if (edir / MARKER_FILE).exists() or (edir / MARKER_FILE).is_symlink():
        reasons.append(f"the entry holds {COVERED_FILE} and a test marker; an entry is either covered or tested")
    return reasons, seen


def check_entry(edir, run_dir, chips: int, *, rehearsal: bool = False, waived: list | None = None) -> tuple[list[str], list[str]]:
    """(reasons, evidence paths) for one entry directory: a covered entry's file checks, or the full-port check of the entry."""
    edir, run_dir = Path(edir), Path(run_dir)
    if not edir.is_dir():
        return [f"the entry directory configs/{chips}/ does not exist"], []
    if (edir / COVERED_FILE).exists() or (edir / COVERED_FILE).is_symlink():
        covered = read_covered(edir)
        if covered is None:
            return [f"{COVERED_FILE} is not readable JSON"], []
        return _covered_reasons(edir, run_dir, chips, covered)
    return full_port.check(edir, run_dir, rehearsal=rehearsal, waived=waived, stage=STAGE, entry=chips)


# ---- the stage's summary ----------------------------------------------------------------------------------------------------

def _record_numbers(record: dict | None) -> dict:
    """top1_agreement and n_tokens recomputed from a record's ids, and its load time, or None for each that cannot be read."""
    out = {"top1_agreement": None, "n_tokens": None, "load_s": None}
    if not isinstance(record, dict):
        return out
    ref, chip = record.get("reference_ids"), record.get("chip_ids")
    if full_port._ids(ref) and full_port._ids(chip) and chip and len(ref) == len(chip):
        out["top1_agreement"] = full_port.agreement(ref, chip)
        out["n_tokens"] = len(chip)
    if full_port._num(record.get("load_s")):
        out["load_s"] = record["load_s"]
    return out


def summarize(stage_dir, run_dir, required) -> dict:
    """The stage's result.json, built from the entries' records alone: for each required count (in order) whether a stage covered it, the
    numbers recomputed from the record's ids (the entry's own, or the covering stage's), the entry's `coherent` and its evidence paths.
    The supervisor writes this file; the gate builds it again and requires the file on disk to equal it."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    rows = []
    for c in sorted(set(required or ())):
        edir = entry_dir(stage_dir, c)
        covered = read_covered(edir) if (edir / COVERED_FILE).is_file() else None
        if covered is not None and covered.get("covered_by") in COVERED_BY and type(covered.get("covered_by")) is int:
            k = covered["covered_by"]
            record = _read(run_dir / f"stages/{k}/{full_port.RECORD_FILE}")
            own = _read(run_dir / f"stages/{k}/{RESULT_FILE}") or {}
            evidence = [f"stages/{k}/{MARKER_FILE}", f"stages/{k}/{full_port.RECORD_FILE}"]
            by = k
        else:
            record = _read(edir / full_port.RECORD_FILE)
            own = _read(edir / RESULT_FILE) or {}
            evidence = [os.path.relpath(edir / full_port.RECORD_FILE, run_dir), os.path.relpath(edir / full_port.OUTPUT_FILE, run_dir)]
            by = None
        coherent = own.get("coherent")
        rows.append({"chips": c, "covered_by": by, **_record_numbers(record), "coherent": coherent if isinstance(coherent, bool) else None,
                     "evidence": evidence})
    return {"required": sorted(set(required or ())), "entries": rows, "derived_by": DERIVED_BY}


def check_stage(stage_dir, run_dir, *, required, rehearsal: bool = False) -> tuple[list[str], list[str], list[str]]:
    """(reasons, evidence paths, waived fields) for stage 4 of a full port. `required` is the run's `--required-chips`, the supervisor's
    own option; the gate takes the list of entries from it and never from a file. The plan must list exactly those counts, each entry
    must pass, and the stage's result.json must equal what `summarize` builds from the entries' records."""
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    reasons: list[str] = []
    seen: list[str] = []
    waived: list[str] = []
    counts = sorted(set(required)) if isinstance(required, (list, tuple)) and required and all(type(c) is int and c >= 1 for c in required) else None
    if counts is None:
        return [f"the run's chip counts (--required-chips) are {required!r}; stage 4 of a full port needs a list of whole numbers of at least 1"], seen, waived
    plan, why = load_plan(stage_dir)
    if plan is None:
        reasons.append(why or "tests/plan.json cannot be read")
    elif [e.chips for e in plan] != counts:
        reasons.append(f"tests/plan.json lists the counts {[e.chips for e in plan]}; the run's --required-chips are {counts}")
    else:
        seen.append(os.path.relpath(stage_dir / PLAN, run_dir))
    for c in counts:
        mine: list[str] = []
        got, evidence = check_entry(entry_dir(stage_dir, c), run_dir, c, rehearsal=rehearsal, waived=mine)
        reasons += [f"the {c}-chip entry: {r}" for r in got]
        seen += evidence
        for field in mine:
            if field not in waived:
                waived.append(field)
    result = _read(stage_dir / RESULT_FILE)
    if result is None:
        reasons.append("result.json is missing or is not a JSON object")
    elif result != json.loads(json.dumps(summarize(stage_dir, run_dir, counts))):
        reasons.append("result.json is not the summary the gate builds from the entries' records (a number, a count, a covering stage or an "
                       "evidence path differs): the supervisor writes this file and the gate rebuilds it")
    else:
        seen.append(os.path.relpath(stage_dir / RESULT_FILE, run_dir))
    return reasons, seen, waived
