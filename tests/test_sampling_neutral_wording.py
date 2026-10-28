# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Third review of PR 114, finding 3 (issue 107): `no-variant-serves-purpose` also covers a missing speed, a variant that failed its checks and speeds taken from
different requests, so the card, the bundle risk text and the decision say only that a configuration is kept or selected. A speed relation is stated only when
both speeds were measured on the same basis."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator_bundle_sampling_choice import files, put  # noqa: E402
from test_operator_bundle_template import make_run, section  # noqa: E402
from test_sampling_choice import NEEDED, side, variant  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402


def numbers(cur, var, cb="sampled", vb="sampled"):
    return {"served_tok_s": cur, "variant_tok_s": var, "served_speed_basis": cb, "variant_speed_basis": vb}


@pytest.mark.parametrize("n,expect", [
    (numbers(5.0, 8.0), True), (numbers(5.0, 8.0, "greedy", "greedy"), True), (numbers(5.0, 8.0, "greedy", "sampled"), False),
    (numbers(None, 8.0), False), (numbers(5.0, None), False), (numbers(5.0, 8.0, None, None), False), ({}, False),
    (numbers(True, 8.0), False),
])
def test_two_speeds_are_comparable_only_when_both_exist_on_the_same_basis(n, expect):
    assert sc.comparable(n) is expect


# ---- the decision ----------------------------------------------------------------------------------------------------------------------

def test_speeds_from_different_requests_are_not_compared():
    cur = side(sampled=None, greedy=5.0)                     # a refused sampled request: the greedy speed stands in
    var = variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0)
    d = sc.decide(NEEDED, cur, var)
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose"
    assert "could not be compared" in d["reason"] and "faster" not in d["reason"]


def test_speeds_from_the_same_requests_are_compared_and_the_faster_one_is_chosen():
    d = sc.decide(NEEDED, side(sampled=None, greedy=5.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=None))
    assert d["choice"] == "drafter-off" and "faster of the two" in d["reason"]


def test_a_tie_does_not_claim_a_faster_original():
    d = sc.decide(NEEDED, side(greedy=8.0, sampled=8.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))
    assert d["choice"] == "keep" and "at least as fast" in d["reason"] and "faster of the two" not in d["reason"]


# ---- the card ---------------------------------------------------------------------------------------------------------------------------

def sentence(cur, var):
    return sc.card_sentence(sc.decide(NEEDED, cur, var))


def test_a_kept_configuration_is_called_kept_and_not_the_faster_one():
    s = sentence(side(greedy=59.0, sampled=59.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))
    assert "the configuration stage 2 served is kept" in s and "faster one is kept" not in s
    assert "decodes at least as fast as the drafter-off variant" in s


def test_a_kept_configuration_with_a_variant_that_failed_says_nothing_about_speed():
    s = sentence(side(), variant(served=False, greedy=None, sampled=None))
    assert "the configuration stage 2 served is kept" in s and "as fast" not in s and "faster" not in s


def test_a_kept_configuration_with_speeds_from_different_requests_says_nothing_about_speed():
    s = sentence(side(sampled=None, greedy=59.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))
    assert "as fast" not in s and "faster" not in s and "is kept" in s


def test_a_chosen_variant_that_is_faster_on_the_same_basis_says_so():
    s = sentence(side(greedy=5.0, sampled=5.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))
    assert "decodes faster than the configuration stage 2 first served" in s and "does not do what the model needs" in s


def test_a_chosen_variant_is_never_called_the_faster_of_the_two_without_the_numbers():
    d = sc.decide(NEEDED, side(greedy=5.0, sampled=5.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))
    d["numbers"] = {**d["numbers"], "served_speed_basis": "greedy"}          # the recorded bases no longer match
    s = sc.card_sentence(d)
    assert "faster" not in s and "this package serves the drafter-off configuration" in s


def test_no_card_sentence_says_the_faster_one_was_kept_or_chosen_as_the_faster_of_the_two():
    cases = [sc.decide(NEEDED, side(), variant(served=False, greedy=None, sampled=None)),
             sc.decide(NEEDED, side(greedy=59.0, sampled=59.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0)),
             sc.decide(NEEDED, side(greedy=5.0, sampled=5.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))]
    for d in cases:
        s = sc.card_sentence(d)
        assert "faster one is kept" not in s and "faster of the two" not in s


# ---- the bundle ---------------------------------------------------------------------------------------------------------------------------

def test_the_bundle_risk_says_selected_configuration_and_no_speed_winner(tmp_path):
    run = make_run(tmp_path)
    put(run, sc.decide(NEEDED, side(), variant(served=False, greedy=None, sampled=None)))
    results, risks = files(run)
    block = section(risks, "## Sampling choice")
    assert "ships the selected configuration" in block and "faster" not in block
    assert "faster" not in section(results, "## The sampling choice").split("Flag:")[1].split("\n")[0]
