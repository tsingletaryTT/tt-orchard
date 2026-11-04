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

import heapq
import math
from typing import Any

# A recomputed fraction and a recorded one are compared to this absolute difference: the template wrote `matches / n` as a float.
FRACTION_TOLERANCE = 1e-9
# The positions the weights-swap template teacher-forces: its `N_TOKENS` (orchard/skills/weights-swap-templates/serve_and_compare.py). A run the
# supervisor bound to the ledger must carry all of them: the recompute trusts the number of records it is given, so without this count a
# failing suffix could be deleted and the summaries rewritten to the shorter record (review of PR 444, round 3). A test pins the two numbers
# together.
TEMPLATE_POSITIONS = 32
# How many positions a reason names before it says "and N more": a reason is read by a person and written to the ledger.
LISTED_POSITIONS = 8
# A reason is written to the ledger and into the continuation message the agent reads, and the ledger refuses to read a file over 16 MiB, so
# nothing the report holds may reach a reason at its own size (review of PR 444, limit round: one 9.4 MB position string made an 18.9 MB
# feedback entry). One report value is shown at most this many characters long; the rest is counted ("and N more characters").
MAX_SHOWN_CHARS = 80
# One reason is at most this long, with the count of what was cut. The field names and the recomputed values come first in every reason and
# the shown values are short, so an ordinary reason is far below it.
MAX_REASON_CHARS = 1000
# The supervisor writes at most this many reasons of one gate (orchard/supervisor.py `bound_reasons`), then says how many it left out.
MAX_REASONS_WRITTEN = 24


def _whole(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _number(v: Any) -> bool:
    """A finite number: an integer of any size, or a float that is neither NaN nor infinite. A boolean is not a number here. An integer is
    finite whatever its size, and `math.isfinite` would raise OverflowError on one that does not fit a float (JSON parses integers up to 4300
    digits, so an agent can write one; review of PR 444, limit round), so integers never reach it."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    return isinstance(v, float) and math.isfinite(v)


def show(value: Any) -> str:
    """One value from a report, as text of bounded length. Text is cut at `MAX_SHOWN_CHARS` with the count of what was cut; a list or an object
    is named by its size and never copied; an integer too long to print is named by its digits (its length comes from `bit_length`, so
    nothing builds the digits). Never raises."""
    if isinstance(value, str):
        cut = len(value) - MAX_SHOWN_CHARS
        return repr(value[:MAX_SHOWN_CHARS]) + (f" (and {cut} more characters)" if cut > 0 else "")
    if value is None or isinstance(value, (bool, float)):
        return repr(value)
    if isinstance(value, int):
        digits = value.bit_length() * 30103 // 100000 + 1
        return repr(value) if digits <= 40 else f"an integer of about {digits} digits"
    if isinstance(value, (list, tuple)):
        return f"a list of {len(value)} items"
    if isinstance(value, dict):
        return f"an object with {len(value)} keys"
    return f"a value of type {type(value).__name__}"


def cut_text(text: str, limit: int = MAX_REASON_CHARS) -> str:
    """`text` cut at `limit` characters, with the count of what was cut."""
    return text if len(text) <= limit else text[:limit] + f" ... ({len(text) - limit} more characters not shown)"


def _listed(shown: list[str], total: int) -> str:
    """The first few of `total` items, then how many more there are."""
    return ", ".join(shown) + (f" and {total - len(shown)} more" if total > len(shown) else "")


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
    """Whether a recorded fraction is `expected` (a float from 0 to 1) to within `FRACTION_TOLERANCE`. An integer is compared as an integer:
    it can only be 0 or 1 here, and it is never converted to a float (a 400-digit integer does not fit one)."""
    if not _number(recorded):
        return False
    if isinstance(recorded, int):
        return recorded in (0, 1) and abs(recorded - expected) <= FRACTION_TOLERANCE
    return abs(recorded - expected) <= FRACTION_TOLERANCE


def problems(report: Any, results: list[tuple[str, Any]], reference_ids: Any, where: str = "", *, require_records: bool = True,
             required_positions: int | None = None) -> list[str]:
    """`_problems`, with each reason cut at `MAX_REASON_CHARS` (a safety net: every value a reason shows is already bounded). The report is a
    file an agent wrote, so this never raises: an exception from the recompute (a value no one thought of) becomes one refusal that names
    the exception's class and not its message (the message can carry the report's own text). Every gate that calls it is total because of it."""
    try:
        found = _problems(report, results, reference_ids, where, require_records=require_records, required_positions=required_positions)
    except Exception as exc:
        return [f"{where}swap-check.json: the 32-token check could not read the report ({type(exc).__name__}), so its numbers cannot be "
                "recomputed from the server's replies"]
    return [cut_text(r) for r in found]


def _problems(report: Any, results: list[tuple[str, Any]], reference_ids: Any, where: str = "", *, require_records: bool = True,
              required_positions: int | None = None) -> list[str]:
    """Reasons the numbers of one swap-check.json (`report`) and of each result that repeats them (`results`: (label, object) pairs,
    such as ("result_draft", ...) and ("result.json", ...)) are not what the report's raw records and `reference_ids` give. Empty when
    they agree. Each reason names the field, the recorded value and the recomputed one. A report with no `teacher_forced` records is refused
    when `require_records` is set (the reference ids are bound to the ledger, so the run is one the supervisor governed from the start) and
    judged as before otherwise (a run from before this check, whose reference ids nothing binds). `required_positions`, when given, is the
    number of records `teacher_forced` must hold (a bound run passes `TEMPLATE_POSITIONS`): a record of another length is refused, because a
    deleted suffix and a shorter run give the same file."""
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
    if required_positions is not None and n != required_positions:
        out.append(f"{where}swap-check.json teacher_forced holds {n} records, but the template records {required_positions} positions, so a deleted "
                   "suffix cannot be told from a shorter run")
    truth = f"{matches} of {n} positions match stage 1's reference ids"
    # The skip list of PR 422 would hide positions from this count. Nothing on this branch can rebuild it, so it is refused.
    for label, obj in (("swap-check.json", report), *results):
        if isinstance(obj, dict) and obj.get("positions_skipped"):
            out.append(f"{where}{label} lists positions_skipped, and this gate has no rebuilt skip mask to check the list against, "
                       "so it cannot count fewer positions than the replies give")
    if report.get("n_tokens") != n or isinstance(report.get("n_tokens"), bool):
        out.append(f"{where}swap-check.json n_tokens {show(report.get('n_tokens'))} is not the {n} records in teacher_forced")
    if report.get("matches") != matches or isinstance(report.get("matches"), bool):
        out.append(f"{where}swap-check.json matches {show(report.get('matches'))} is not what teacher_forced gives ({truth})")
    if not _same_fraction(report.get("top1_agreement"), top1):
        out.append(f"{where}swap-check.json top1_agreement {show(report.get('top1_agreement'))} is not what teacher_forced gives "
                   f"({top1:.6g}: {truth})")
    listed = report.get("mismatches")
    if not isinstance(listed, list):
        out.append(f"{where}swap-check.json mismatches must be a list (it has {len(got['mismatch_positions'])} positions the replies disagree on)")
    else:
        seen = [m.get("position") if isinstance(m, dict) else None for m in listed]
        if seen != got["mismatch_positions"]:
            # The positions are JSON values. A list or an object is unhashable and a mix of null, text and numbers cannot be sorted, so only the
            # whole numbers are put in sets or sorted; every other value is named by its repr, once, in the order it first appears.
            # A report can hold a million of them, and each can be a huge string: only the first few are shown, each cut, and the rest are counted.
            whole = {p for p in seen if _whole(p)}
            hidden = sorted(set(got["mismatch_positions"]) - whole)
            odd_shown, odd_total = [], 0
            for p in seen:
                if not _whole(p):
                    odd_total += 1
                    if len(odd_shown) < LISTED_POSITIONS and show(p) not in odd_shown:
                        odd_shown.append(show(p))
            wrong = whole - set(got["mismatch_positions"])
            wrong_shown = [show(p) for p in heapq.nsmallest(LISTED_POSITIONS, wrong)]
            extra_shown = (odd_shown + wrong_shown)[:LISTED_POSITIONS]
            extra_total = odd_total + len(wrong)
            detail = (f"it leaves out positions {_positions(hidden)}" if hidden else "") + (" and " if hidden and extra_total else "") \
                + (f"it lists positions that are not mismatches: {_listed(extra_shown, extra_total)}" if extra_total else "")
            out.append(f"{where}swap-check.json mismatches are not the positions teacher_forced disagrees on ({detail or 'the order or a repeat differs'})")
    ids = report.get("reference_generated_ids")
    if ids != reference_ids[:n]:
        out.append(f"{where}swap-check.json reference_generated_ids are not the first {n} ids of stage 1's reference")
    for label, obj in results:
        if not isinstance(obj, dict):
            continue
        if obj.get("n_tokens") != n or isinstance(obj.get("n_tokens"), bool):
            out.append(f"{where}{label} n_tokens {show(obj.get('n_tokens'))} is not the {n} records in swap-check.json teacher_forced")
        if not _same_fraction(obj.get("top1_agreement"), top1):
            out.append(f"{where}{label} top1_agreement {show(obj.get('top1_agreement'))} is not what swap-check.json teacher_forced gives "
                       f"({top1:.6g}: {truth})")
    return out


# ---- the stage 7 boot check (issue 445) ----------------------------------------------------------------------------------------------

def boot_problems(report: Any, reference_ids: Any, where: str = "", *, require_records: bool = True) -> list[str]:
    """The reasons the numbers of the boot check's verify.json (`report`) are not what its raw `teacher_forced` records give against stage 1's
    reference ids. `verify_bundle.py` (orchard/package_templates/) writes, for each of its 32 positions, the first token the served model chose
    when it was fed the reference tokens before that position, plus `n_tokens`, `matches`, `top1_agreement` and the mismatches. The card's
    boot-check row cites those numbers, so they are recomputed here the way the stage 2 numbers are (`problems`).

    A report with no `teacher_forced` is refused when `require_records` is set (the supervisor hashed the file when the test ended, so the
    template that wrote it records them) and judged as before otherwise. Never raises: an exception becomes one refusal that names its class."""
    try:
        found = _boot_problems(report, reference_ids, where, require_records=require_records)
    except Exception as exc:
        return [f"{where}verify.json: the boot check could not be read ({type(exc).__name__}), so its numbers cannot be recomputed from the "
                "server's replies"]
    return [cut_text(r) for r in found]


def _boot_problems(report: Any, reference_ids: Any, where: str, *, require_records: bool) -> list[str]:
    if not isinstance(report, dict):
        return [f"{where}verify.json is not an object, so the boot check cannot be recomputed"] if require_records else []
    forced = report.get("teacher_forced")
    if forced is None:
        if not require_records:
            return []
        return [f"{where}verify.json has no teacher_forced records, so the boot check's top1 agreement cannot be recomputed from the "
                "server's replies"]
    got, why = recompute(forced, reference_ids)
    if got is None:
        return [f"{where}verify.json: {r}" for r in why]
    n, matches, top1 = got["n_tokens"], got["matches"], got["top1_agreement"]
    out = []
    if require_records and n != TEMPLATE_POSITIONS:
        out.append(f"{where}verify.json teacher_forced holds {n} records, but the boot check compares {TEMPLATE_POSITIONS} positions, so a "
                   "deleted suffix cannot be told from a shorter run")
    truth = f"{matches} of {n} positions match stage 1's reference ids"
    if report.get("n_tokens") != n or isinstance(report.get("n_tokens"), bool):
        out.append(f"{where}verify.json n_tokens {show(report.get('n_tokens'))} is not the {n} records in teacher_forced")
    if report.get("matches") != matches or isinstance(report.get("matches"), bool):
        out.append(f"{where}verify.json matches {show(report.get('matches'))} is not what teacher_forced gives ({truth})")
    if not _same_fraction(report.get("top1_agreement"), top1):
        out.append(f"{where}verify.json top1_agreement {show(report.get('top1_agreement'))} is not what teacher_forced gives "
                   f"({top1:.6g}: {truth})")
    listed = report.get("mismatches")
    if not isinstance(listed, list):
        out.append(f"{where}verify.json mismatches must be a list (it has {len(got['mismatch_positions'])} positions the replies disagree on)")
    else:
        seen = [m.get("position") if isinstance(m, dict) else None for m in listed]
        if seen != got["mismatch_positions"]:
            out.append(f"{where}verify.json mismatches are not the positions teacher_forced disagrees on (it should list "
                       f"{_positions(got['mismatch_positions']) or 'none'})")
    return out
