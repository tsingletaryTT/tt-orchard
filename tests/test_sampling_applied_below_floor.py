# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Operator decision (issue 107): when sampling is needed and no configuration both applies it and clears the speed floor, the configuration that applies
sampling wins even when it is slow and even when a faster one ignores sampling. The decision is flagged `sampling-applied-below-floor` so the operator sees it.
Only when neither applies sampling (or applying is not shown for either) does the faster measured one win."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator_bundle_sampling_choice import files, put  # noqa: E402
from test_operator_bundle_template import make_run as make_bundle_run, section  # noqa: E402
from test_sampling_choice import NEEDED, side, variant  # noqa: E402
from test_sampling_choice_files import make_run, report, save  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402
from orchard.card_use import PERF_IN_PROSE  # noqa: E402

SLOW = "sampling-applied-below-floor"


def slow_variant(**kw):
    return variant(greedy=8.0, sampled=8.0, **kw)


# ---- the rule ---------------------------------------------------------------------------------------------------------------------------

def test_a_slow_variant_that_applies_sampling_beats_a_fast_original_that_ignores_it():
    d = sc.decide(NEEDED, side(greedy=59.0, sampled=59.0), slow_variant())
    assert d["choice"] == "drafter-off" and d["flag"] == SLOW


@pytest.mark.parametrize("verdict", ["sampled-ignored", "sampled-matches-greedy", "inconclusive", "not-run", "sampled-refused"])
def test_it_wins_whatever_the_original_failed_to_show(verdict):
    d = sc.decide(NEEDED, side(verdict=verdict, greedy=59.0, sampled=59.0), slow_variant())
    assert d["choice"] == "drafter-off" and d["flag"] == SLOW


def test_it_wins_when_the_originals_speed_is_not_measured():
    d = sc.decide(NEEDED, side(greedy=None, sampled=None), slow_variant())
    assert d["choice"] == "drafter-off" and d["flag"] == SLOW


def test_the_reason_says_slow_gives_the_measured_speed_and_the_floor():
    r = sc.decide(NEEDED, side(greedy=59.0, sampled=59.0), slow_variant())["reason"]
    assert "slow" in r and "8.0 tokens/s" in r and "floor of 10 tokens/s" in r and "applies sampling" in r and "cannot do what the model is for" in r


def test_a_variant_at_or_above_the_floor_has_no_slow_flag():
    d = sc.decide(NEEDED, side(), variant(greedy=10.0, sampled=10.0))
    assert d["choice"] == "drafter-off" and d["flag"] is None


def test_a_variant_that_applies_sampling_with_no_speed_stays_unverified_not_slow():
    d = sc.decide(NEEDED, side(), variant(greedy=None, sampled=None))
    assert d["flag"] == "unverified"


def test_a_slow_original_that_applies_sampling_with_no_variant_is_kept_flagged_slow():
    d = sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=9.0, sampled=8.0))
    assert d["choice"] == "keep" and d["flag"] == SLOW and "slow" in d["reason"] and "8.0 tokens/s" in d["reason"]


def test_the_old_below_floor_flag_name_is_gone():
    assert "below-floor" not in sc.FLAGS and SLOW in sc.FLAGS


def test_neither_applying_still_lets_the_faster_measured_one_win():
    d = sc.decide(NEEDED, side(greedy=5.0, sampled=5.0), variant(verdict="sampled-ignored", greedy=8.0, sampled=8.0))
    assert d["choice"] == "drafter-off" and d["flag"] == "no-variant-serves-purpose"
    d = sc.decide(NEEDED, side(greedy=59.0, sampled=59.0), variant(verdict="sampled-matches-greedy", greedy=8.0, sampled=8.0))
    assert d["choice"] == "keep" and d["flag"] == "no-variant-serves-purpose"


def test_applying_unknown_for_both_lets_the_faster_measured_one_win():
    d = sc.decide(NEEDED, side(verdict="inconclusive", greedy=5.0, sampled=5.0), variant(verdict="inconclusive", greedy=8.0, sampled=8.0))
    assert d["choice"] == "drafter-off" and d["flag"] == "no-variant-serves-purpose"


def test_a_slow_variant_that_failed_its_checks_is_not_chosen():
    d = sc.decide(NEEDED, side(greedy=59.0, sampled=59.0), slow_variant(served=False))
    assert d["choice"] == "keep" and d["flag"] != SLOW


def test_a_model_that_does_not_need_sampling_never_gets_the_slow_flag():
    d = sc.decide({"needed": False, "source": "none", "override": "auto", "evidence": []}, side(greedy=59.0, sampled=59.0), slow_variant())
    assert d["choice"] == "keep" and d["flag"] is None


# ---- the gate ---------------------------------------------------------------------------------------------------------------------------

def test_the_gate_recomputes_the_slow_choice(tmp_path):
    run, s2 = make_run(tmp_path, stage2=report(ignored=True, seconds=1.6), variant=report(drafter="off", ignored=False, seconds=12.0))
    d = save(run, s2)
    assert d["choice"] == "drafter-off" and d["flag"] == SLOW
    assert sc.decision_problems(s2, run) == []
    sc.write_decision(s2, {**d, "choice": "keep", "flag": "no-variant-serves-purpose"}, trial=True)
    assert sc.decision_problems(s2, run)


# ---- the card ---------------------------------------------------------------------------------------------------------------------------

def test_the_card_says_the_chosen_configuration_applies_sampling_and_is_slow_with_no_number():
    s = sc.card_sentence(sc.decide(NEEDED, side(greedy=59.0, sampled=59.0), slow_variant()))
    assert "applies sampling but is slow" in s and "below the usability floor" in s and "chose the drafter-off configuration" in s
    assert "does not do what the model needs" not in s and "decodes above the usability floor" not in s
    assert "Read the decision file before relying on it." in s
    assert not PERF_IN_PROSE.search(s)


def test_the_card_says_a_kept_slow_configuration_is_slow():
    s = sc.card_sentence(sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=9.0, sampled=8.0)))
    assert "applies sampling but is slow" in s and "kept" in s and not PERF_IN_PROSE.search(s)


# ---- the bundle -------------------------------------------------------------------------------------------------------------------------

def test_the_bundle_risk_says_slow_and_names_the_flag(tmp_path):
    run = make_bundle_run(tmp_path)
    put(run, sc.decide(NEEDED, side(greedy=59.0, sampled=59.0), slow_variant()))
    results, risks = files(run)
    r = section(risks, "## Sampling choice")
    assert "slow" in r and "applies sampling" in r and "does not do what the model needs" not in r
    assert f"`{SLOW}`" in section(results, "## The sampling choice")
