# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Does a speculative drafter beat plain decode? A calculation from three measured times (issue 182).

A pass that verifies a draft emits `accepted_per_pass` tokens (counted the way the server reports it: tokens emitted per pass) and
takes `verify_pass_ms`. Plain decode emits one token per `plain_step_ms`. So the speed against plain decode is

    ratio = (accepted_per_pass x plain_step_ms) / verify_pass_ms

and the pass has to finish in `accepted_per_pass x plain_step_ms` to match plain decode (the break-even pass time). The published
example: 2.7 accepted per pass and a 4.6 ms plain step give a break-even pass of 12.4 ms.

The verdict compares the ratio with 1 inside a margin (DEFAULT_MARGIN), because both times are measured and a ratio of 1.01 is not a
finding. A ratio above 1 + margin is a gain, below 1 - margin a loss, and between them break-even. `drafter_decision` turns the drafter
on only for a gain.

Acceptance depends on the text. `verdict_by_prompt_kind` takes measured acceptance per prompt kind (code, chat, a non-English sample)
and states which kind the verdict used: the lowest by default, so the drafter is not enabled on the strength of the easiest prompts.
Acceptance has to come from real text at the sampling the card claims; this module only does the arithmetic on numbers it is given.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_MARGIN = 0.05
VERDICTS = ("gain", "break-even", "loss")


class BreakEvenInputError(ValueError):
    """A time or an acceptance is missing, is not a number, is not finite, or is out of range."""


def _positive(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        raise BreakEvenInputError(f"{what} must be a positive finite number")
    return float(x)


def _non_negative(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise BreakEvenInputError(f"{what} must be a finite number of at least 0")
    return float(x)


def _margin(m) -> float:
    if isinstance(m, bool) or not isinstance(m, (int, float)) or not math.isfinite(m) or not 0 <= m < 1:
        raise BreakEvenInputError("margin must be a number from 0 up to (not including) 1")
    return float(m)


@dataclass(frozen=True)
class BreakEven:
    plain_step_ms: float
    accepted_per_pass: float
    verify_pass_ms: float
    break_even_pass_ms: float
    ratio: float                 # speed against plain decode: above 1 is faster
    verdict: str
    margin: float

    def sentence(self) -> str:
        """The verdict in one sentence with its numbers and the margin it used."""
        return (f"{self.accepted_per_pass:.2f} accepted per pass, a {self.plain_step_ms:.1f} ms plain step and a "
                f"{self.verify_pass_ms:.1f} ms verify pass give {self.ratio:.2f} times plain decode, which is a {self.verdict} "
                f"(break-even pass time {self.break_even_pass_ms:.1f} ms; break-even means within {self.margin:.0%} of plain).")

    def as_dict(self) -> dict:
        return {"plain_step_ms": self.plain_step_ms, "accepted_per_pass": self.accepted_per_pass,
                "verify_pass_ms": self.verify_pass_ms, "break_even_pass_ms": self.break_even_pass_ms, "ratio": self.ratio,
                "verdict": self.verdict, "margin": self.margin}


def speculative_break_even(plain_step_ms, accepted_per_pass, verify_pass_ms, *, margin=DEFAULT_MARGIN) -> BreakEven:
    """The speed ratio against plain decode and a verdict (gain, break-even within `margin`, loss).

    `plain_step_ms` and `verify_pass_ms` must be positive and finite; `accepted_per_pass` finite and at least 0 (zero is a loss with
    a ratio of 0). Raises BreakEvenInputError on a bool, a string, NaN, an infinity or a negative."""
    plain = _positive(plain_step_ms, "plain_step_ms")
    accepted = _non_negative(accepted_per_pass, "accepted_per_pass")
    verify = _positive(verify_pass_ms, "verify_pass_ms")
    m = _margin(margin)
    break_even = accepted * plain
    ratio = break_even / verify
    verdict = "gain" if ratio > 1 + m else "loss" if ratio < 1 - m else "break-even"
    return BreakEven(plain, accepted, verify, break_even, ratio, verdict, m)


def drafter_decision(result) -> bool:
    """True when the drafter should be enabled: only a gain turns it on. Break-even and loss leave it off.
    `result` is a BreakEven or a KindVerdict."""
    r = result.result if isinstance(result, KindVerdict) else result
    return r.verdict == "gain"


@dataclass(frozen=True)
class KindVerdict:
    result: BreakEven            # the verdict that was used
    kind_used: str               # which prompt kind it used, or "pooled (all kinds)"
    basis: str                   # lowest, pooled, or the kind's name as asked for
    per_kind: dict               # kind -> BreakEven on that kind's mean acceptance, for every kind

    def sentence(self) -> str:
        others = "; ".join(f"{k}: {v.accepted_per_pass:.2f} per pass, {v.ratio:.2f} times, {v.verdict}"
                           for k, v in sorted(self.per_kind.items()))
        return f"The verdict used {self.kind_used} ({self.basis}). {self.result.sentence()} By kind, {others}."

    def as_dict(self) -> dict:
        return {"kind_used": self.kind_used, "basis": self.basis, "result": self.result.as_dict(),
                "per_kind": {k: v.as_dict() for k, v in self.per_kind.items()}}


def _clean_kinds(acceptance_by_kind) -> dict:
    if not isinstance(acceptance_by_kind, dict) or not acceptance_by_kind:
        raise BreakEvenInputError("acceptance must be a non-empty dict of prompt kind to a list of accepted-per-pass values")
    out = {}
    for kind, values in acceptance_by_kind.items():
        if not isinstance(kind, str) or not kind:
            raise BreakEvenInputError("a prompt kind must be a non-empty string")
        if not isinstance(values, (list, tuple)) or not values:
            raise BreakEvenInputError(f"acceptance for {kind!r} must be a non-empty list")
        out[kind] = [_non_negative(v, f"acceptance for {kind!r}") for v in values]
    return out


def verdict_by_prompt_kind(plain_step_ms, verify_pass_ms, acceptance_by_kind, *, basis="lowest", margin=DEFAULT_MARGIN) -> KindVerdict:
    """The verdict from measured acceptance per prompt kind, saying which kind it used.

    Each kind's acceptance is the mean of its list. `basis` is `lowest` (the kind with the lowest mean, the default: the drafter must
    pay on the hardest kind measured), `pooled` (the mean of every sample, which weights a kind by how many samples it has), or the
    name of one kind. Every kind's own verdict is returned in `per_kind` as well. Raises BreakEvenInputError on a bad input or an
    unknown basis."""
    kinds = _clean_kinds(acceptance_by_kind)
    means = {k: sum(v) / len(v) for k, v in kinds.items()}
    per_kind = {k: speculative_break_even(plain_step_ms, m, verify_pass_ms, margin=margin) for k, m in means.items()}
    if basis == "lowest":
        used = min(sorted(means), key=lambda k: means[k])
        return KindVerdict(per_kind[used], used, "lowest", per_kind)
    if basis == "pooled":
        samples = [x for v in kinds.values() for x in v]
        pooled = speculative_break_even(plain_step_ms, sum(samples) / len(samples), verify_pass_ms, margin=margin)
        return KindVerdict(pooled, "pooled (all kinds)", "pooled", per_kind)
    if isinstance(basis, str) and basis in kinds:
        return KindVerdict(per_kind[basis], basis, basis, per_kind)
    raise BreakEvenInputError(f"basis must be lowest, pooled or one of the measured kinds ({', '.join(sorted(kinds))})")
