# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Recompute the 32-token teacher-forced check from its raw records (issue 433).

The weights-swap template serves the new weights, feeds the CPU reference's tokens one at a time, and records for each position the
first token the server chose (`teacher_forced[k].first_id`). It also writes `matches`, `top1_agreement`, `n_tokens` and a list of the
mismatches, and copies the numbers into `result_draft`. The gate used to read those numbers as written. This module recomputes them
from the raw records and from stage 1's reference ids, so a swap-check.json and a result.json edited together cannot carry a
`top1_agreement` that the replies do not give.

Everything here is a pure function of its arguments: no file is read and no ledger is touched. `orchard/stages.py` reads stage 1's
`generated-ids.json` through `safeio`, checks it against the sha256 the supervisor put in the ledger, and passes the ids in.

Limits, stated so the PR and the gate do not claim more:
- `first_id` is taken as recorded. The template derived it by tokenizing the served text. Rebuilding it needs the tokenizer, which
  is the job of the reference-interpreter helper in PR 422 (`orchard/skip_mask_helper.py`).
- The skip list of PR 422 (`positions_skipped`) is not on main. A record that carries one is refused here, because there is no
  rebuilt mask to compare it with. PR 422 replaces that refusal with the rebuilt mask.
"""
from __future__ import annotations

import math
from typing import Any

# A recomputed fraction and a recorded one are compared to this absolute difference: the template wrote `matches / n` as a float.
FRACTION_TOLERANCE = 1e-9
# How many positions a reason names before it says "and N more": a reason is read by a person and written to the ledger.
LISTED_POSITIONS = 8


def _whole(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _positions(values) -> str:
    values = list(values)
    shown = ", ".join(str(v) for v in values[:LISTED_POSITIONS])
    return shown + (f" and {len(values) - LISTED_POSITIONS} more" if len(values) > LISTED_POSITIONS else "")


def recompute(forced: Any, reference_ids: Any) -> tuple[dict | None, list[str]]:
    """({n_tokens, matches, top1_agreement, mismatch_positions}, []) from the raw records, or (None, reasons) when the records cannot be
    read as a teacher-forced run against `reference_ids`. Position k matches when `forced[k]["first_id"]` equals `reference_ids[k]`; a
    missing first id (the server's reply tokenized to nothing) is a mismatch."""
    if not isinstance(forced, list) or not forced:
        return None, ["teacher_forced must be a non-empty list of the server's first token at each position"]
    if not isinstance(reference_ids, list) or not all(_whole(i) for i in reference_ids):
        return None, ["stage 1's reference ids are not a list of whole numbers"]
    n = len(forced)
    if len(reference_ids) < n:
        return None, [f"teacher_forced has {n} positions and stage 1's reference holds {len(reference_ids)} ids"]
    bad, mismatches = [], []
    for k, entry in enumerate(forced):
        if not isinstance(entry, dict) or entry.get("position") != k or isinstance(entry.get("position"), bool):
            bad.append(k)
            continue
        got = entry.get("first_id")
        if got is not None and not _whole(got):
            bad.append(k)
            continue
        if got != reference_ids[k]:
            mismatches.append(k)
    if bad:
        return None, [f"teacher_forced entries at positions {_positions(bad)} are not {{position, first_id}} records in order"]
    matches = n - len(mismatches)
    return {"n_tokens": n, "matches": matches, "top1_agreement": matches / n, "mismatch_positions": mismatches}, []


def _same_fraction(recorded: Any, expected: float) -> bool:
    return _number(recorded) and abs(float(recorded) - expected) <= FRACTION_TOLERANCE


def problems(report: Any, results: list[tuple[str, Any]], reference_ids: Any, where: str = "", *, require_records: bool = True) -> list[str]:
    """Reasons the numbers of one swap-check.json (`report`) and of each result that repeats them (`results`: (label, object) pairs,
    such as ("result_draft", ...) and ("result.json", ...)) are not what the report's raw records and `reference_ids` give. Empty when
    they agree. Each reason names the field, the recorded value and the recomputed one. A report with no `teacher_forced` records is refused
    when `require_records` is set (the reference ids are bound to the ledger, so the run is one the supervisor governed from the start) and
    judged as before otherwise (a run from before this check, whose reference ids nothing binds)."""
    if not isinstance(report, dict):
        return []
    # The template copies its numbers into the report's own result_draft; the finish step and the agent copy that again. It is judged with
    # the results the caller passes, so an edit to the draft alone is caught as well.
    if isinstance(report.get("result_draft"), dict):
        results = [("result_draft", report["result_draft"]), *results]
    forced = report.get("teacher_forced")
    if forced is None:
        if not require_records:
            return []
        return [f"{where}swap-check.json has no teacher_forced records, so its 32-token check cannot be recomputed from the server's replies"]
    got, why = recompute(forced, reference_ids)
    if got is None:
        return [f"{where}swap-check.json: {r}" for r in why]
    out = []
    n, matches, top1 = got["n_tokens"], got["matches"], got["top1_agreement"]
    truth = f"{matches} of {n} positions match stage 1's reference ids"
    # The skip list of PR 422 would hide positions from this count. Nothing on this branch can rebuild it, so it is refused.
    for label, obj in (("swap-check.json", report), *results):
        if isinstance(obj, dict) and obj.get("positions_skipped"):
            out.append(f"{where}{label} lists positions_skipped, and this gate has no rebuilt skip mask to check the list against, "
                       "so it cannot count fewer positions than the replies give")
    if report.get("n_tokens") != n or isinstance(report.get("n_tokens"), bool):
        out.append(f"{where}swap-check.json n_tokens {report.get('n_tokens')!r} is not the {n} records in teacher_forced")
    if report.get("matches") != matches or isinstance(report.get("matches"), bool):
        out.append(f"{where}swap-check.json matches {report.get('matches')!r} is not what teacher_forced gives ({truth})")
    if not _same_fraction(report.get("top1_agreement"), top1):
        out.append(f"{where}swap-check.json top1_agreement {report.get('top1_agreement')!r} is not what teacher_forced gives "
                   f"({top1:.6g}: {truth})")
    listed = report.get("mismatches")
    if not isinstance(listed, list):
        out.append(f"{where}swap-check.json mismatches must be a list (it has {len(got['mismatch_positions'])} positions the replies disagree on)")
    else:
        seen = [m.get("position") if isinstance(m, dict) else None for m in listed]
        if seen != got["mismatch_positions"]:
            hidden = sorted(set(got["mismatch_positions"]) - set(p for p in seen if _whole(p)))
            extra = sorted(p for p in set(seen) if not _whole(p) or p not in got["mismatch_positions"])
            detail = (f"it leaves out positions {_positions(hidden)}" if hidden else "") + (" and " if hidden and extra else "") \
                + (f"it lists positions that are not mismatches: {_positions(map(repr, extra))}" if extra else "")
            out.append(f"{where}swap-check.json mismatches are not the positions teacher_forced disagrees on ({detail or 'the order or a repeat differs'})")
    ids = report.get("reference_generated_ids")
    if ids != reference_ids[:n]:
        out.append(f"{where}swap-check.json reference_generated_ids are not the first {n} ids of stage 1's reference")
    for label, obj in results:
        if not isinstance(obj, dict):
            continue
        if obj.get("n_tokens") != n or isinstance(obj.get("n_tokens"), bool):
            out.append(f"{where}{label} n_tokens {obj.get('n_tokens')!r} is not the {n} records in swap-check.json teacher_forced")
        if not _same_fraction(obj.get("top1_agreement"), top1):
            out.append(f"{where}{label} top1_agreement {obj.get('top1_agreement')!r} is not what swap-check.json teacher_forced gives "
                       f"({top1:.6g}: {truth})")
    return out
