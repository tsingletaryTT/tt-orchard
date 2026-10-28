# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fourth review of PR 114, finding D (issue 107): a speed relation ("faster", "at least as fast") is stated only for valid measurements. Both speeds must be
finite numbers above zero (a zero or negative speed is not a measurement) and both bases must be one of the two the probe records (`sampled`, `greedy`) and
equal. The decision, the card and the bundle all read the same `comparable`."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator_bundle_sampling_choice import files, put  # noqa: E402
from test_operator_bundle_template import make_run as make_bundle_run, section  # noqa: E402
from test_sampling_choice import NEEDED, side, variant  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402


def n(cur, var, cb="sampled", vb="sampled"):
    return {"served_tok_s": cur, "variant_tok_s": var, "served_speed_basis": cb, "variant_speed_basis": vb}


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, -0.001, math.nan, math.inf, -math.inf, True, False, "8.0", None, [8.0]])
def test_a_speed_that_is_not_a_positive_finite_number_is_not_a_measurement(bad):
    assert sc.comparable(n(bad, 8.0)) is False
    assert sc.comparable(n(5.0, bad)) is False


@pytest.mark.parametrize("cb,vb", [("sampled", "greedy"), ("greedy", "sampled"), ("sampled", "SAMPLED"), ("other", "other"), ("", ""), (None, None),
                                   (1, 1), ("sampled", None)])
def test_bases_must_be_the_two_recorded_ones_and_equal(cb, vb):
    assert sc.comparable(n(5.0, 8.0, cb, vb)) is False


@pytest.mark.parametrize("basis", ["sampled", "greedy"])
def test_two_positive_speeds_on_the_same_recorded_basis_are_comparable(basis):
    assert sc.comparable(n(5.0, 8.0, basis, basis)) is True
    assert sc.comparable(n(5, 8, basis, basis)) is True


def test_a_tiny_positive_speed_is_a_measurement():
    assert sc.comparable(n(1e-9, 2e-9)) is True


def test_a_decision_with_an_unrecorded_basis_never_names_a_winner():
    d = sc.decide(NEEDED, side(verdict="sampled-ignored", greedy=5.0, sampled=5.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))
    assert d["choice"] == "drafter-off" and "faster of the two" in d["reason"]           # valid speeds: the claim is allowed
    d["numbers"]["served_tok_s"] = 0
    assert "faster" not in sc.card_sentence(d)


@pytest.mark.parametrize("bad", [0, -3.0, math.nan])
def test_the_card_makes_no_speed_claim_from_a_forged_non_positive_speed(bad):
    d = sc.decide(NEEDED, side(verdict="sampled-ignored", greedy=5.0, sampled=5.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))
    d["numbers"]["served_tok_s"] = bad
    s = sc.card_sentence(d)
    assert "decodes faster" not in s and "as fast" not in s


@pytest.mark.parametrize("bad", [0, -3.0])
def test_the_kept_card_makes_no_speed_claim_from_a_forged_non_positive_speed(bad):
    d = sc.decide(NEEDED, side(verdict="sampled-ignored", greedy=59.0, sampled=59.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))
    assert d["choice"] == "keep"
    d["numbers"]["variant_tok_s"] = bad
    assert "as fast" not in sc.card_sentence(d)


def test_the_decision_does_not_compare_a_variant_with_no_valid_speed():
    d = sc.decide(NEEDED, side(verdict="sampled-ignored", greedy=5.0, sampled=5.0), variant(verdict="sampled-ignored", greedy=0.0, sampled=0.0))
    assert d["choice"] == "keep" and "could not be compared" in d["reason"]
