# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Recompute the sidecar parity numbers from the raw per-record measurements (issue 445).

`hidden_parity.py` (orchard/skills/sidecar-parity-templates/) writes `evidence/sidecar-parity.json`: a summary and, under `per_record`,
one entry for each fixed record it measured (`hidden_pcc`, `prob_max_abs_diff`, `top1_agree`, `n_questions`). The summary is
`aggregate()`'s fold of those entries. The agent copies the summary into result.json's `sidecar_parity`. The gate used to read the summary
as written; it now folds the records again and requires the summaries to match.

Pure functions: no file is read here. `orchard/stages.py` reads the evidence file through `raw_evidence` and passes the parsed value in.
A value nobody expected (a huge integer, NaN, a wrong type) is a reason and never an exception.

Not recomputed, and said so in the PR: `hidden_pcc` and `prob_max_abs_diff` of one record are the template's own comparison of two tensors on
the leased board, and the tensors are not kept. The recompute proves that the summary is the fold of the records. It does not prove that
the records came from the board.
"""
from __future__ import annotations

import math
from typing import Any

from orchard.token_check import MAX_REASON_CHARS, cut_text, show

# The summary fields the fold gives, and the tolerance for the floating-point ones: the template wrote `sum(...) / n` as a float.
FLOAT_FIELDS = ("hidden_pcc_min", "hidden_pcc_mean", "prob_max_abs_diff", "top1_agree_fraction")
WHOLE_FIELDS = ("n_records", "n_questions")
TOLERANCE = 1e-9
# Fields the gate reads from result.json's copy of the parity object without folding them from per_record. Each one must be the evidence
# file's own value (same type, same value), because the gate takes its bar (noise_floor) and its identities (head_sha256, code_sha256),
# its `measured` flag and its wiring check from that copy.
COPIED_FIELDS = ("measured", "noise_floor", "wiring_check", "head_sha256", "code_sha256")
# Bounds for the identity comparison, so a hostile value in a copied field cannot cost a deep recursion or a long walk.
IDENTICAL_MAX_DEPTH = 16
IDENTICAL_MAX_NODES = 10_000
# A record list longer than this is refused: the template measures six to eight fixed records, and a list of millions is a hostile file.
MAX_RECORDS = 10_000
MAX_QUESTIONS_PER_RECORD = 1_000_000


def _whole(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _finite(v: Any) -> bool:
    """A finite number that fits a float. An integer too large for a float is not finite here (math.isfinite would raise on it)."""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return abs(v) < 2 ** 1000
    return isinstance(v, float) and math.isfinite(v)


def recompute(per_record: Any) -> dict:
    """The summary `hidden_parity.aggregate` gives for `per_record`: n_records, n_questions, hidden_pcc_min, hidden_pcc_mean,
    prob_max_abs_diff, top1_agree_fraction. Raises ValueError (with a short message that holds no input text) when the records cannot be
    read as measurements; `safe_recompute` turns that into a reason."""
    if not isinstance(per_record, list) or not per_record:
        raise ValueError("per_record must be a non-empty list of the measured records")
    if len(per_record) > MAX_RECORDS:
        raise ValueError(f"per_record holds {len(per_record)} entries; the template measures a handful")
    pccs, diffs, agree, nq = [], [], 0, 0
    for i, r in enumerate(per_record):
        if not isinstance(r, dict):
            raise ValueError(f"per_record[{i}] is not an object")
        pcc, diff, a, n = r.get("hidden_pcc"), r.get("prob_max_abs_diff"), r.get("top1_agree"), r.get("n_questions")
        if not _finite(pcc) or not -1.0 <= pcc <= 1.0 + 1e-6:
            raise ValueError(f"per_record[{i}] hidden_pcc must be a finite number from -1 to 1, got {show(pcc)}")
        if not _finite(diff) or not 0 <= diff <= 1.0 + 1e-6:
            raise ValueError(f"per_record[{i}] prob_max_abs_diff must be a finite number from 0 to 1, got {show(diff)}")
        if not _whole(n) or not 1 <= n <= MAX_QUESTIONS_PER_RECORD:
            raise ValueError(f"per_record[{i}] n_questions must be a whole number of at least 1, got {show(n)}")
        if not _whole(a) or not 0 <= a <= n:
            raise ValueError(f"per_record[{i}] top1_agree must be a whole number from 0 to its n_questions, got {show(a)}")
        pccs.append(pcc)
        diffs.append(diff)
        agree += a
        nq += n
    return {"n_records": len(per_record), "n_questions": nq, "hidden_pcc_min": min(pccs), "hidden_pcc_mean": sum(pccs) / len(pccs),
            "prob_max_abs_diff": max(diffs), "top1_agree_fraction": agree / nq}


def safe_recompute(per_record: Any) -> tuple[dict | None, list[str]]:
    """(numbers, []) or (None, [reason]). Never raises."""
    try:
        return recompute(per_record), []
    except ValueError as exc:
        return None, [cut_text(str(exc))]
    except Exception as exc:                                    # a value nobody thought of: name the class, never the message
        return None, [f"per_record could not be read ({type(exc).__name__})"]


def identical(a: Any, b: Any) -> bool:
    """True when `a` and `b` have the same JSON shape, the same types and the same values. Never raises.

    Python's `==` treats True as 1 and 1 as 1.0, so it cannot say that a copied value is the evidence's own. Depth and node count are
    bounded; a value past either bound is not identical (a gate's own fields are small, so nothing real is lost)."""
    budget = [IDENTICAL_MAX_NODES]

    def walk(x: Any, y: Any, depth: int) -> bool:
        budget[0] -= 1
        if budget[0] < 0 or depth > IDENTICAL_MAX_DEPTH or type(x) is not type(y):
            return False
        if isinstance(x, dict):
            if len(x) != len(y):
                return False
            for k, v in x.items():
                if type(k) is not str or k not in y or not walk(v, y[k], depth + 1):
                    return False
            return True
        if isinstance(x, list):
            return len(x) == len(y) and all(walk(i, j, depth + 1) for i, j in zip(x, y))
        return x == y

    try:
        return walk(a, b, 0)
    except Exception:                                        # defensive: a gate must not crash on a value it cannot compare
        return False


def _same(recorded: Any, expected: float | int) -> bool:
    if not _finite(recorded):
        return False
    if isinstance(recorded, int) and isinstance(expected, int):
        return recorded == expected
    if isinstance(recorded, int):
        return False if abs(recorded) > 2 ** 52 else abs(recorded - expected) <= TOLERANCE
    return abs(recorded - expected) <= TOLERANCE


def problems(summaries: list[tuple[str, Any]], evidence: Any, *, bound: bool) -> tuple[list[str], bool]:
    """(reasons, verified). `evidence` is the parsed sidecar-parity.json; `summaries` are (label, object) pairs whose numbers must be the
    fold of its records: the evidence file's own summary and result.json's `sidecar_parity`. `verified` is True when the records were
    there and folded. A run that is not bound and has no records gives ([], False): judged as before, and the caller says the numbers are
    the agent's. A bound run with no readable records is refused."""
    try:
        return _problems(summaries, evidence, bound=bound)
    except Exception as exc:
        return [f"sidecar-parity.json: the parity numbers could not be recomputed ({type(exc).__name__})"], False


def _problems(summaries, evidence, *, bound: bool) -> tuple[list[str], bool]:
    records = evidence.get("per_record") if isinstance(evidence, dict) else None
    if records is None:
        if not bound:
            return [], False
        return ["sidecar-parity.json has no per_record list, though the supervisor hashed it when the hardware test ended, so the parity "
                "numbers cannot be recomputed from the measured records"], False
    got, why = safe_recompute(records)
    if got is None:
        return [f"sidecar-parity.json {r}" for r in why], False
    out = []
    for label, obj in summaries:
        if not isinstance(obj, dict):
            continue
        for field in WHOLE_FIELDS + FLOAT_FIELDS:
            if field in obj or label == "sidecar-parity.json":
                if not _same(obj.get(field), got[field]):
                    out.append(f"{label} {field} {show(obj.get(field))} is not what per_record gives ({got[field]:.6g}, folded from "
                               f"{got['n_records']} records)")
    # Copied fields (the bar, the identities, `measured`, the wiring check): each is one value in the evidence file, and the copy in result.json
    # must be that value, by type and value. The wiring check is the template's own comparison of the script's layer loop with the model's prefill,
    # so the file cannot say it failed while result.json says it passed; the noise floor sets the gate's bar, so result.json cannot raise it.
    if isinstance(evidence, dict):
        for label, obj in summaries:
            if not isinstance(obj, dict) or label == "sidecar-parity.json":
                continue
            for k in COPIED_FIELDS:
                if k in obj and not identical(obj[k], evidence.get(k)):
                    out.append(f"{label} {k} differs from the {k} in sidecar-parity.json (the file the hardware test wrote)")
    return [cut_text(r, MAX_REASON_CHARS) for r in out], True
