# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The drafter decision from a measurement record (issue 182).

`orchard/spec_break_even.py` does the arithmetic on three measured numbers. This module is the slice around it that needs no board:

  * `read_measure` reads `evidence/spec-measure.json` from stage 2's directory (bounded, strict JSON, no link). A hardware run that measures the plain step time,
    the verify pass time and the accepted tokens per pass writes it; no such run is built yet.
  * `decide` validates the record and gives the verdict by prompt kind (the lowest kind by default, so the drafter has to pay on the hardest text measured), whether
    the drafter should be on (only a gain turns it on), whether the measurement covers greedy decoding only, and the sentence for the card.
  * the supervisor writes `evidence/spec-decision.json` and a ledger decision (`Supervisor._spec_decision_record`), and `package.write_card` puts the sentence
    in the card's limits.

A record says where its prompts came from. Random-token prompts give misleading acceptance, so only `real-text` is accepted. A record that fails the checks is
refused with the field named; the supervisor writes that as a ledger decision and the stage still passes, because a measurement problem does not make the model
wrong. Nothing here runs a model or opens a device."""
from __future__ import annotations

import json
from pathlib import Path

from orchard import safeio, spec_break_even as sbe

SCHEMA = 1
MEASURE_FILE = "evidence/spec-measure.json"
DECISION_FILE = "evidence/spec-decision.json"
LEDGER_DECISION = "speculative decision"
REFUSED_DECISION = "speculative measurement refused"
MEASURE_MAX = 1024 * 1024
DECISION_MAX = 1024 * 1024
KIND_SHOWN = 6                     # kinds named in the card sentence; the record keeps them all
KIND_CHARS = 40
STATEMENT_MAX = 1500
PROMPT_SOURCE = "real-text"


class SpecRecordError(ValueError):
    """The measurement record is missing a field, has a field of the wrong kind, or does not hold what a decision needs."""


def _number(value, what: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not safeio.fits_float(value):
        raise SpecRecordError(f"{what} must be a finite number")
    if positive and value <= 0:
        raise SpecRecordError(f"{what} must be above 0")
    if value < 0:
        raise SpecRecordError(f"{what} must be at least 0")
    return float(value)             # finite: ok safeio.fits_float above refused a bool, a non-number, NaN, an infinity and an integer too large for a float


def _sampling(value) -> dict:
    if not isinstance(value, dict) or "temperature" not in value:
        raise SpecRecordError("sampling must be an object with a temperature (0 for greedy decoding)")
    out = {"temperature": _number(value["temperature"], "sampling.temperature")}
    if "top_p" in value:
        top_p = _number(value["top_p"], "sampling.top_p", positive=True)
        if top_p > 1:
            raise SpecRecordError("sampling.top_p must be at most 1")
        out["top_p"] = top_p
    return out


def _clean(record) -> dict:
    if not isinstance(record, dict):
        raise SpecRecordError("the measurement record must be a JSON object")
    for field in ("schema", "plain_step_ms", "verify_pass_ms", "prompt_source", "sampling", "acceptance_by_kind"):
        if field not in record:
            raise SpecRecordError(f"the measurement record has no {field}")
    if isinstance(record["schema"], bool) or record["schema"] != SCHEMA or not isinstance(record["schema"], int):
        raise SpecRecordError(f"schema must be the whole number {SCHEMA}")
    if record["prompt_source"] != PROMPT_SOURCE:
        raise SpecRecordError("prompt_source must be 'real-text': acceptance measured on random-token prompts is misleading, so a decision needs real text")
    kinds = record["acceptance_by_kind"]
    if not isinstance(kinds, dict) or not kinds:
        raise SpecRecordError("acceptance_by_kind must be a non-empty object of prompt kind to a list of accepted-per-pass values")
    clean_kinds = {}
    for kind, values in kinds.items():
        if not isinstance(kind, str) or not kind.strip():
            raise SpecRecordError("a prompt kind must be a non-empty string")
        if not isinstance(values, list) or not values:
            raise SpecRecordError(f"acceptance for {kind[:KIND_CHARS]!r} must be a non-empty list")
        clean_kinds[kind] = [_number(v, f"acceptance for {kind[:KIND_CHARS]!r}") for v in values]
    return {"plain_step_ms": _number(record["plain_step_ms"], "plain_step_ms", positive=True),
            "verify_pass_ms": _number(record["verify_pass_ms"], "verify_pass_ms", positive=True),
            "sampling": _sampling(record["sampling"]), "acceptance_by_kind": clean_kinds}


def _setting(sampling: dict) -> str:
    if sampling["temperature"] == 0:
        return "greedy decoding"
    text = f"temperature {sampling['temperature']:g}"
    return text + (f" and top_p {sampling['top_p']:g}" if "top_p" in sampling else "")


def _by_kind(per_kind: dict) -> str:
    rows = sorted(per_kind.items())
    shown = [f"{k[:KIND_CHARS]}: {v.accepted_per_pass:.2f} per pass, {v.ratio:.2f} times plain, {v.verdict}" for k, v in rows[:KIND_SHOWN]]
    more = f"; and {len(rows) - KIND_SHOWN} more kinds" if len(rows) > KIND_SHOWN else ""
    return "; ".join(shown) + more


def decide(record) -> dict:
    """{"verdict", "enable_drafter", "greedy_only", "sampling", "statement"} from a measurement record, or SpecRecordError naming what is wrong. `verdict` is
    `KindVerdict.as_dict()` for the lowest kind. The drafter is enabled only on a gain."""
    r = _clean(record)
    try:
        kv = sbe.verdict_by_prompt_kind(r["plain_step_ms"], r["verify_pass_ms"], r["acceptance_by_kind"], basis="lowest")
    except sbe.BreakEvenInputError as exc:
        raise SpecRecordError(str(exc)) from None
    enable = sbe.drafter_decision(kv)
    greedy = r["sampling"]["temperature"] == 0
    used = kv.result
    head = ("Speculative decoding is on: " if enable else "Speculative decoding is off: ")
    body = (f"measured on real text at {_setting(r['sampling'])}, the verdict used the {kv.kind_used[:KIND_CHARS]} prompts, which had the lowest acceptance. "
            f"{used.sentence()} By kind, {_by_kind(kv.per_kind)}.")
    tail = (" The drafter was measured at greedy decoding only, so a sampled request is not covered by this measurement." if greedy else "")
    statement = (head + body + tail)[:STATEMENT_MAX]
    return {"verdict": kv.as_dict(), "enable_drafter": enable, "greedy_only": greedy, "sampling": r["sampling"], "statement": statement}


def read_measure(stage_dir) -> dict | None:
    """The measurement record in `stage_dir/evidence/spec-measure.json`, or None when there is none. A file that is a link, is over its bound, is invalid JSON or
    is not one JSON object raises SpecRecordError with a reason that holds none of the file's content."""
    path = Path(stage_dir) / MEASURE_FILE
    if not path.exists() and not path.is_symlink():
        return None
    try:
        return safeio.read_json(path, max_bytes=MEASURE_MAX, expect=dict)
    except (OSError, ValueError, safeio.SafeReadError) as exc:
        raise SpecRecordError(f"the measurement file cannot be read ({type(exc).__name__})") from None


def statement_from_run(run_dir) -> str | None:
    """The card sentence of the decision stage 2 recorded (`stages/2/evidence/spec-decision.json`), or None when there is none or it cannot be read."""
    try:
        data = safeio.read_json(Path(run_dir) / "stages" / "2" / DECISION_FILE, max_bytes=DECISION_MAX, expect=dict)
    except (OSError, ValueError, safeio.SafeReadError):
        return None
    text = data.get("statement")
    return text if isinstance(text, str) and 0 < len(text) <= STATEMENT_MAX else None


def decision_json(decision: dict) -> str:
    return json.dumps(decision, indent=1, sort_keys=True) + "\n"
