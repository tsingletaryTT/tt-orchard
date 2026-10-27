# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 114, findings 3 and 4 (issue 107). When neither configuration both applies sampling and clears the floor, the faster measured one is
the one used, and the decision and the card name that one. When a choice rests on a missing verdict or speed, the reason and the card sentence do not
claim a measurement that was not made."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sampling_choice import NEEDED, side, variant  # noqa: E402
from test_sampling_choice_files import make_run, report, save  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402
from orchard.card_use import PERF_IN_PROSE  # noqa: E402


# ---- finding 3: the faster configuration is the one used ------------------------------------------------------------------------------

def test_a_variant_that_applies_sampling_below_the_floor_beats_a_slower_ignoring_original():
    d = sc.decide(NEEDED, side(verdict="sampled-ignored", greedy=5.0, sampled=5.0), variant(greedy=8.0, sampled=8.0))
    assert d["choice"] == "drafter-off" and d["flag"] == "no-variant-serves-purpose"
    assert "8.0 tokens/s" in d["reason"] and "5.0 tokens/s" in d["reason"] and "faster" in d["reason"]


def test_when_both_ignore_sampling_the_faster_one_is_chosen():
    d = sc.decide(NEEDED, side(greedy=5.0, sampled=5.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))
    assert d["choice"] == "drafter-off" and d["flag"] == "no-variant-serves-purpose" and "faster" in d["reason"]


def test_a_variant_that_is_slower_than_the_original_leaves_the_original():
    d = sc.decide(NEEDED, side(greedy=59.0, sampled=59.0), variant(greedy=8.0, sampled=8.0))
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose" and "at least as fast" in d["reason"]


def test_equal_speeds_leave_the_original():
    d = sc.decide(NEEDED, side(greedy=8.0, sampled=8.0), variant(greedy=8.0, sampled=8.0))
    assert d["choice"] == "keep"


@pytest.mark.parametrize("cur,var,verdict", [((None, None), (8.0, 8.0), "sampled-full-speed"), ((5.0, 5.0), (None, None), "sampled-ignored")])
def test_a_missing_speed_leaves_the_original_and_says_the_speeds_could_not_be_compared(cur, var, verdict):
    d = sc.decide(NEEDED, side(greedy=cur[0], sampled=cur[1]), variant(verdict=verdict, greedy=var[0], sampled=var[1]))
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose" and "not measured" in d["reason"]


def test_a_variant_with_an_unknown_verdict_below_the_floor_that_is_faster_is_chosen_flagged():
    d = sc.decide(NEEDED, side(greedy=5.0, sampled=5.0), variant(verdict="inconclusive", greedy=8.0, sampled=8.0))
    assert d["choice"] == "drafter-off" and d["flag"] == "no-variant-serves-purpose"


def test_a_variant_that_did_not_serve_is_never_chosen_however_the_speeds_compare():
    d = sc.decide(NEEDED, side(greedy=5.0, sampled=5.0), variant(served=False, greedy=8.0, sampled=8.0))
    assert d["choice"] == "keep"


def test_the_gate_recomputes_the_faster_choice_from_the_files(tmp_path):
    run, s2 = make_run(tmp_path, stage2=report(ignored=True, seconds=19.2), variant=report(drafter="off", ignored=False, seconds=12.0))
    d = save(run, s2)
    assert d["choice"] == "drafter-off" and d["flag"] == "no-variant-serves-purpose"
    assert sc.decision_problems(s2, run) == []
    sc.write_decision(s2, {**d, "choice": "keep"}, trial=True)
    assert any("drafter-off" in p for p in sc.decision_problems(s2, run))


def test_the_card_sentence_names_the_faster_configuration_that_was_chosen():
    d = sc.decide(NEEDED, side(greedy=5.0, sampled=5.0), variant(greedy=8.0, sampled=8.0))
    s = sc.card_sentence(d)
    assert "chose the drafter-off configuration" in s and "faster" in s and "does not do what the model needs" in s
    assert "The drafter-off configuration applies sampling and decodes above" not in s
    assert "so the faster one is kept" not in s


# ---- finding 4: claims follow the evidence ----------------------------------------------------------------------------------------------

def test_a_verified_choice_claims_sampling_and_the_floor():
    s = sc.card_sentence(sc.decide(NEEDED, side(), variant()))
    assert "applies sampling and decodes above the usability floor" in s


@pytest.mark.parametrize("v", [variant(verdict="inconclusive", greedy=18.0, sampled=None), variant(greedy=None, sampled=None),
                               variant(verdict="not-run", greedy=None, sampled=None)])
def test_an_unverified_choice_makes_no_claim_about_sampling_or_the_floor(v):
    d = sc.decide(NEEDED, side(), v)
    assert d["choice"] == "drafter-off" and d["flag"] == "unverified"
    s = sc.card_sentence(d)
    assert "applies sampling" not in s and "decodes above" not in s and "Its speeds are in" not in s
    assert "documented host-sampling path" in s and "does not claim" in s
    assert not PERF_IN_PROSE.search(s)


@pytest.mark.parametrize("v,absent", [(variant(greedy=None, sampled=None), "at or above the floor"),
                                       (variant(verdict="inconclusive", greedy=18.0, sampled=None), "applies sampling")])
def test_the_reason_of_an_unverified_choice_claims_only_what_was_measured(v, absent):
    d = sc.decide(NEEDED, side(), v)
    assert absent not in d["reason"] and "documented host-sampling path" in d["reason"]


def test_a_variant_that_applies_sampling_with_no_speed_is_unverified_and_says_the_floor_was_not_checked():
    d = sc.decide(NEEDED, side(), variant(greedy=None, sampled=None))
    assert "applies sampling" in d["reason"] and "not measured" in d["reason"]


def test_the_reason_says_the_speed_when_it_was_measured():
    d = sc.decide(NEEDED, side(), variant(verdict="inconclusive", greedy=18.0, sampled=None))
    assert "18.0 tokens/s" in d["reason"] and "at or above the floor" in d["reason"]


@pytest.mark.parametrize("served_verdict,words", [("sampled-ignored", "ignored sampled requests"),
                                                   ("inconclusive", "no usable sampling verdict"),
                                                   ("not-run", "no usable sampling verdict"),
                                                   ("sampled-refused", "did not apply sampling")])
def test_the_card_describes_the_first_configuration_by_its_recorded_verdict(served_verdict, words):
    d = sc.decide(NEEDED, side(verdict=served_verdict), variant())
    assert d["choice"] == "drafter-off"
    s = sc.card_sentence(d)
    assert words in s
    if served_verdict != "sampled-ignored":
        assert "ignored sampled requests" not in s


@pytest.mark.parametrize("served_verdict", ["sampled-refused", "sampled-incoherent", "greedy-refused", "sampled-ignored"])
def test_a_measured_bad_first_configuration_with_a_measured_good_variant_is_a_verified_choice(served_verdict):
    d = sc.decide(NEEDED, side(verdict=served_verdict), variant())
    assert d["choice"] == "drafter-off" and d["flag"] is None


def test_a_first_configuration_with_no_measured_speed_is_not_given_one_in_the_reason():
    d = sc.decide(NEEDED, side(verdict="not-run", greedy=None, sampled=None), variant())
    assert d["choice"] == "drafter-off" and "unmeasured speed" not in d["reason"].split(";")[0]
