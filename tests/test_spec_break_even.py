# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/spec_break_even.py: does a speculative drafter beat plain decode, from measured times (issue 182)."""
from __future__ import annotations

import json

import pytest

from orchard import spec_break_even as sb


def test_the_published_example_breaks_even_at_12_4_ms():
    r = sb.speculative_break_even(plain_step_ms=4.6, accepted_per_pass=2.7, verify_pass_ms=12.42)
    assert r.break_even_pass_ms == pytest.approx(12.42)
    assert round(r.break_even_pass_ms, 1) == 12.4
    assert r.ratio == pytest.approx(1.0)
    assert r.verdict == "break-even"


def test_a_faster_pass_is_a_gain():
    r = sb.speculative_break_even(4.6, 2.7, 8.0)
    assert r.verdict == "gain" and r.ratio == pytest.approx(2.7 * 4.6 / 8.0)


def test_a_slower_pass_is_a_loss():
    r = sb.speculative_break_even(4.6, 2.7, 20.0)
    assert r.verdict == "loss" and r.ratio < 1


def test_zero_acceptance_is_a_loss_with_zero_ratio():
    r = sb.speculative_break_even(4.6, 0, 12.0)
    assert r.ratio == 0.0 and r.verdict == "loss" and r.break_even_pass_ms == 0.0


def test_the_margin_decides_the_edges_both_ways():
    base = 4.6 * 2.0
    assert sb.speculative_break_even(4.6, 2.0, base / 1.04).verdict == "break-even"
    assert sb.speculative_break_even(4.6, 2.0, base / 1.06).verdict == "gain"
    assert sb.speculative_break_even(4.6, 2.0, base / 0.96).verdict == "break-even"
    assert sb.speculative_break_even(4.6, 2.0, base / 0.94).verdict == "loss"
    assert sb.speculative_break_even(4.6, 2.0, base / 1.2, margin=0.25).verdict == "break-even"
    assert sb.speculative_break_even(4.6, 2.0, base / 1.2, margin=0.1).verdict == "gain"
    assert sb.speculative_break_even(4.6, 2.0, base / 1.01, margin=0).verdict == "gain"


def test_a_ratio_exactly_on_the_margin_edge_is_break_even_on_both_sides():
    # exact binary fractions, so the comparison is exact: 3 tokens x 1 ms over a 2 ms pass is 1.5; 1 x 1 over 2 is 0.5
    assert sb.speculative_break_even(1, 3, 2, margin=0.5).verdict == "break-even"
    assert sb.speculative_break_even(1, 1, 2, margin=0.5).verdict == "break-even"


def test_the_margin_is_part_of_the_result_and_the_text_states_it():
    r = sb.speculative_break_even(4.6, 2.7, 12.42)
    assert r.margin == sb.DEFAULT_MARGIN
    assert f"{sb.DEFAULT_MARGIN:.0%}" in r.sentence()
    assert "12.4 ms" in r.sentence()


@pytest.mark.parametrize("args", [
    (0, 2.0, 5.0), (-1.0, 2.0, 5.0), (4.6, -0.1, 5.0), (4.6, 2.0, 0), (4.6, 2.0, -3.0),
    (float("nan"), 2.0, 5.0), (4.6, float("nan"), 5.0), (4.6, 2.0, float("nan")),
    (float("inf"), 2.0, 5.0), (4.6, float("inf"), 5.0), (4.6, 2.0, float("inf")),
    (True, 2.0, 5.0), (4.6, True, 5.0), (4.6, 2.0, True), ("4.6", 2.0, 5.0), (None, 2.0, 5.0)])
def test_bad_inputs_are_refused(args):
    with pytest.raises(sb.BreakEvenInputError):
        sb.speculative_break_even(*args)


@pytest.mark.parametrize("margin", [-0.1, 1.0, 2, float("nan"), float("inf"), True, "0.1"])
def test_bad_margin_is_refused(margin):
    with pytest.raises(sb.BreakEvenInputError, match="margin"):
        sb.speculative_break_even(4.6, 2.0, 5.0, margin=margin)


def test_result_is_plain_json():
    json.dumps(sb.speculative_break_even(4.6, 2.7, 8.0).as_dict(), allow_nan=False)


# ---- acceptance by prompt kind ---------------------------------------------------------------------

KINDS = {"code": [5.0, 6.0, 5.5], "chat": [2.0, 2.2], "thai": [0.9, 1.1]}


def test_the_verdict_names_the_kind_it_used_and_defaults_to_the_lowest():
    r = sb.verdict_by_prompt_kind(4.6, 8.0, KINDS)
    assert r.kind_used == "thai"                          # the lowest mean acceptance
    assert r.result.verdict == "loss"
    assert r.basis == "lowest"
    assert set(r.per_kind) == {"code", "chat", "thai"}
    assert r.per_kind["code"].verdict == "gain"           # every kind is reported, along with the kind used
    assert "thai" in r.sentence() and "code" in r.sentence()


def test_a_named_kind_is_used_when_asked():
    r = sb.verdict_by_prompt_kind(4.6, 8.0, KINDS, basis="code")
    assert r.kind_used == "code" and r.result.verdict == "gain"
    assert r.result.accepted_per_pass == pytest.approx(5.5)


def test_pooled_basis_averages_every_sample_and_says_so():
    r = sb.verdict_by_prompt_kind(4.6, 8.0, KINDS, basis="pooled")
    allv = [x for v in KINDS.values() for x in v]
    assert r.result.accepted_per_pass == pytest.approx(sum(allv) / len(allv))
    assert "pooled" in r.kind_used


@pytest.mark.parametrize("kinds", [{}, {"code": []}, {"code": [1.0, float("nan")]}, {"code": [-1.0]}, {"code": [True]},
                                   {"code": "5"}, [1.0], {"": [1.0]}, {3: [1.0]}, {"code": [float("inf")]}])
def test_bad_acceptance_input_is_refused(kinds):
    with pytest.raises(sb.BreakEvenInputError):
        sb.verdict_by_prompt_kind(4.6, 8.0, kinds)


def test_an_unknown_basis_is_refused():
    with pytest.raises(sb.BreakEvenInputError, match="basis"):
        sb.verdict_by_prompt_kind(4.6, 8.0, KINDS, basis="french")


def test_drafter_is_correctly_turned_off_when_the_targeted_kind_loses():
    # a plain step of 4.6 ms and a verify pass of 12.4 ms: 2.7 tokens per pass is break-even and 1.03 is a loss
    thai = sb.verdict_by_prompt_kind(4.6, 12.4, {"code": [5.57], "thai": [1.03]}, basis="thai")
    assert thai.result.verdict == "loss"
    assert sb.drafter_decision(thai) is False
    code = sb.verdict_by_prompt_kind(4.6, 8.0, {"code": [5.57]}, basis="code")
    assert sb.drafter_decision(code) is True
    edge = sb.speculative_break_even(4.6, 2.7, 12.42)
    assert sb.drafter_decision(edge) is False             # break-even does not turn the drafter on
