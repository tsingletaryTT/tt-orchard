# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The graft trial's decision: when stage 0 found a graft candidate, stage 2 serves the grafted variant once, beside the safe
configuration, and the graft ships only when a measurement says it is better (orchard/graft_trial.py). No hardware: the
trial's numbers are written by hand here."""
import json

import pytest

from orchard import defaults
from orchard.graft_trial import (DECISION_FILE, TRIAL_DIR, TRIAL_RESULT, candidate, chosen, decide, decision_problems,
                                 read_decision, write_decision)


def trial(*, safe=20.0, variant=60.0, top1=0.96875, served=True, coherent=True, safe_ttft=0.3, variant_ttft=0.2, safe_failure=None,
          variant_failure=None):
    s = {"failure": safe_failure} if safe_failure else {"server_ready_s": 100.0, "decode_tok_s": safe, "ttft_s": safe_ttft}
    v = {"served": served, "coherent": coherent, "top1_agreement": top1, "n_tokens": 32, "server_ready_s": 240.0,
         "decode_tok_s": variant, "ttft_s": variant_ttft, "failure": variant_failure}
    return {"safe": s, "variant": v, "steps": []}


CANDIDATE = {"base_tensors": 15, "base_revision": "a" * 40, "base_shards": ["model-00018-of-00018.safetensors"], "dtypes": ["BF16"],
             "hidden_size": 5120, "shapes_match": True, "graftable": True}


# ---- the candidate ------------------------------------------------------------------------------------------------------------------

def test_a_vocabulary_trim_is_never_a_candidate():
    """The grafted head goes with the speculative drafter, which works on the nearest model's token ids (issue 56)."""
    assert candidate({"graft_candidate": CANDIDATE, "class": "vocab-trim"}) is None
    assert candidate({"graft_candidate": CANDIDATE, "class": "weights-only"}) == CANDIDATE


def test_a_graftable_candidate_whose_shapes_match_is_one():
    assert candidate({"graft_candidate": CANDIDATE}) == CANDIDATE


@pytest.mark.parametrize("change", [{"shapes_match": None}, {"shapes_match": False}, {"graftable": False}, {"graftable": None}])
def test_a_candidate_that_may_not_fit_is_not_tried(change):
    assert candidate({"graft_candidate": {**CANDIDATE, **change}}) is None


@pytest.mark.parametrize("delta", [{}, {"graft_candidate": None}, {"graft_candidate": "yes"}, None, [], "x"])
def test_no_candidate_is_no_trial(delta):
    assert candidate(delta) is None


# ---- the rule -----------------------------------------------------------------------------------------------------------------------

def test_a_fast_grafted_variant_that_agrees_is_chosen():
    d = decide(trial(), safe_top1=0.96875)
    assert d["choice"] == "graft"
    assert d["numbers"]["speedup"] == pytest.approx(3.0)
    assert d["numbers"]["safe_decode_tok_s"] == 20.0 and d["numbers"]["variant_decode_tok_s"] == 60.0
    assert d["numbers"]["safe_top1_agreement"] == 0.96875 and d["numbers"]["variant_top1_agreement"] == 0.96875
    assert "3.0 times" in d["reason"] and "60" in d["reason"] and "20" in d["reason"]


def test_the_thresholds_travel_with_the_decision():
    d = decide(trial(), safe_top1=0.9)
    assert d["thresholds"] == {"min_speedup": defaults.GRAFT_MIN_SPEEDUP, "top1_margin": defaults.GRAFT_TOP1_MARGIN,
                               "top1_min": defaults.SWAP_TOP1_MIN}


def test_a_speedup_exactly_at_the_bar_is_chosen_and_just_under_it_is_not():
    at = defaults.GRAFT_MIN_SPEEDUP * 20.0
    assert decide(trial(variant=at), safe_top1=0.9)["choice"] == "graft"
    under = decide(trial(variant=at - 0.01), safe_top1=0.9)
    assert under["choice"] == "safe" and "times" in under["reason"] and "only" in under["reason"]


def test_a_variant_that_is_not_faster_keeps_the_safe_configuration():
    d = decide(trial(variant=19.0), safe_top1=0.9)
    assert d["choice"] == "safe" and d["numbers"]["speedup"] == pytest.approx(0.95)


def test_a_variant_that_did_not_serve_keeps_the_safe_configuration_and_says_why():
    d = decide(trial(served=False, top1=None, variant=None, variant_failure="the server exited with code 1"), safe_top1=0.9)
    assert d["choice"] == "safe" and "did not serve" in d["reason"] and "exited with code 1" in d["reason"]


def test_an_incoherent_variant_keeps_the_safe_configuration():
    d = decide(trial(coherent=False), safe_top1=0.9)
    assert d["choice"] == "safe" and "not coherent" in d["reason"]


def test_a_variant_below_the_swap_bar_keeps_the_safe_configuration():
    d = decide(trial(top1=defaults.SWAP_TOP1_MIN - 0.01), safe_top1=None)
    assert d["choice"] == "safe" and "below" in d["reason"] and str(defaults.SWAP_TOP1_MIN) in d["reason"]


def test_a_variant_at_the_swap_bar_passes_it():
    assert decide(trial(top1=defaults.SWAP_TOP1_MIN), safe_top1=None)["choice"] == "graft"


def test_a_variant_that_loses_more_than_the_margin_against_the_safe_run_keeps_the_safe_configuration():
    lose = 0.99 - defaults.GRAFT_TOP1_MARGIN - 0.01
    d = decide(trial(top1=lose), safe_top1=0.99)
    assert d["choice"] == "safe" and "margin" in d["reason"]


def test_a_variant_that_loses_exactly_the_margin_is_allowed():
    assert decide(trial(top1=0.96875), safe_top1=0.96875 + defaults.GRAFT_TOP1_MARGIN)["choice"] == "graft"


def test_an_unknown_safe_agreement_skips_only_the_margin_test():
    assert decide(trial(), safe_top1=None)["choice"] == "graft"


@pytest.mark.parametrize("top1", [None, "0.9", True, float("nan")])
def test_a_variant_agreement_that_is_not_a_number_keeps_the_safe_configuration(top1):
    d = decide(trial(top1=top1), safe_top1=0.9)
    assert d["choice"] == "safe" and "agreement" in d["reason"]


def test_a_safe_speed_that_was_not_measured_keeps_the_safe_configuration():
    d = decide(trial(safe_failure="the server exited with code 4"), safe_top1=0.9)
    assert d["choice"] == "safe" and "safe configuration's speed was not measured" in d["reason"] and "code 4" in d["reason"]


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, None, True, "fast", float("nan")])
def test_a_variant_speed_that_is_not_a_positive_number_keeps_the_safe_configuration(bad):
    d = decide(trial(variant=bad), safe_top1=0.9)
    assert d["choice"] == "safe" and "speed" in d["reason"]


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, None, True, "fast"])
def test_a_safe_speed_that_is_not_a_positive_number_keeps_the_safe_configuration(bad):
    d = decide(trial(safe=bad), safe_top1=0.9)
    assert d["choice"] == "safe" and "speed" in d["reason"]


@pytest.mark.parametrize("t", [None, [], "x", {}, {"safe": {}}, {"variant": {}}])
def test_a_trial_with_no_result_keeps_the_safe_configuration(t):
    d = decide(t, safe_top1=0.9)
    assert d["choice"] == "safe" and d["reason"]


def test_the_first_reason_is_not_hidden_by_a_later_one():
    d = decide(trial(served=False, coherent=False, top1=None, variant=None, variant_failure="boom"), safe_top1=0.9)
    assert "did not serve" in d["reason"] and "boom" in d["reason"]


def test_serving_is_checked_before_speed():
    d = decide(trial(served=False, variant=999.0, variant_failure="x"), safe_top1=0.9)
    assert d["choice"] == "safe" and "did not serve" in d["reason"]


# ---- the decision file --------------------------------------------------------------------------------------------------------------

def stage2(tmp_path, *, top1=0.96875, with_trial=True, t=None):
    run = tmp_path / "run"
    sd = run / "stages" / "2"
    (sd / TRIAL_DIR).mkdir(parents=True)
    (sd / "result.json").write_text(json.dumps({"serves": True, "top1_agreement": top1}))
    if with_trial:
        (sd / TRIAL_DIR / TRIAL_RESULT).write_text(json.dumps(t if t is not None else trial()))
    return run, sd


def test_a_decision_is_written_whole_and_read_back(tmp_path):
    run, sd = stage2(tmp_path)
    d = decide(trial(), safe_top1=0.96875)
    path = write_decision(sd, d, candidate=CANDIDATE)
    assert path == sd / DECISION_FILE and not list(sd.glob("*.tmp"))
    back = read_decision(sd)
    assert back["choice"] == "graft" and back["candidate"] == CANDIDATE and back["trial"] == f"{TRIAL_DIR}/{TRIAL_RESULT}"
    assert back["numbers"]["speedup"] == pytest.approx(3.0)


def test_chosen_says_yes_only_for_a_graft_decision(tmp_path):
    run, sd = stage2(tmp_path)
    assert chosen(run) is False                                          # no decision file yet
    write_decision(sd, decide(trial(variant=21.0), safe_top1=0.9), candidate=CANDIDATE)
    assert chosen(run) is False                                          # the safe configuration was kept
    write_decision(sd, decide(trial(), safe_top1=0.9), candidate=CANDIDATE)
    assert chosen(run) is True


@pytest.mark.parametrize("text", ["not json", "[]", '{"choice": "maybe"}', '{"choice": "graft"', ""])
def test_a_malformed_decision_is_not_a_graft(tmp_path, text):
    run, sd = stage2(tmp_path)
    (sd / DECISION_FILE).write_text(text)
    assert chosen(run) is False and read_decision(sd) is None


# ---- the gate's check of the decision ------------------------------------------------------------------------------------------------

def test_no_decision_file_is_no_problem(tmp_path):
    run, sd = stage2(tmp_path)
    assert decision_problems(sd, run) == []


def test_a_consistent_graft_decision_has_no_problems(tmp_path):
    run, sd = stage2(tmp_path)
    write_decision(sd, decide(trial(), safe_top1=0.96875), candidate=CANDIDATE)
    assert decision_problems(sd, run) == []


def test_a_consistent_safe_decision_has_no_problems(tmp_path):
    run, sd = stage2(tmp_path)
    write_decision(sd, decide(trial(variant=21.0), safe_top1=0.96875), candidate=CANDIDATE)
    assert decision_problems(sd, run) == []


def test_a_graft_decision_with_no_trial_result_is_refused(tmp_path):
    run, sd = stage2(tmp_path, with_trial=False)
    write_decision(sd, decide(trial(), safe_top1=0.96875), candidate=CANDIDATE)
    assert any("trial" in p and TRIAL_RESULT in p for p in decision_problems(sd, run))


def test_a_graft_decision_the_trial_numbers_do_not_support_is_refused(tmp_path):
    """The file says graft; the trial file on disk says the variant was slower. The gate recomputes from the trial."""
    run, sd = stage2(tmp_path, t=trial(variant=19.0))
    write_decision(sd, decide(trial(), safe_top1=0.96875), candidate=CANDIDATE)
    problems = decision_problems(sd, run)
    assert problems and any("do not support" in p for p in problems)


def test_a_graft_decision_whose_variant_did_not_pass_is_refused(tmp_path):
    run, sd = stage2(tmp_path, t=trial(coherent=False))
    write_decision(sd, decide(trial(), safe_top1=0.96875), candidate=CANDIDATE)
    assert decision_problems(sd, run)


def test_a_graft_decision_is_checked_against_the_safe_run_in_result_json(tmp_path):
    run, sd = stage2(tmp_path, top1=0.99, t=trial(top1=0.90))                    # lost 0.09 against 0.99: more than the margin
    write_decision(sd, decide(trial(), safe_top1=0.9), candidate=CANDIDATE)
    assert any("do not support" in p for p in decision_problems(sd, run))


def test_a_safe_decision_with_no_reason_is_refused(tmp_path):
    run, sd = stage2(tmp_path)
    write_decision(sd, {**decide(trial(variant=21.0), safe_top1=0.9), "reason": "  "}, candidate=CANDIDATE)
    assert any("reason" in p for p in decision_problems(sd, run))


def test_a_decision_with_an_unknown_choice_is_refused(tmp_path):
    run, sd = stage2(tmp_path)
    (sd / DECISION_FILE).write_text(json.dumps({"choice": "maybe", "reason": "x"}))
    assert any("choice" in p for p in decision_problems(sd, run))


def test_an_unreadable_decision_file_is_refused(tmp_path):
    run, sd = stage2(tmp_path)
    (sd / DECISION_FILE).write_text("{")
    assert any(DECISION_FILE in p for p in decision_problems(sd, run))


def test_a_kept_safe_decision_needs_no_trial_file(tmp_path):
    """A trial that timed out leaves no result; the safe configuration stands and says so."""
    run, sd = stage2(tmp_path, with_trial=False)
    write_decision(sd, decide(None, safe_top1=0.9), candidate=CANDIDATE)
    assert decision_problems(sd, run) == []


def test_a_variant_agreement_above_one_is_not_a_fraction():
    d = decide(trial(top1=1.5), safe_top1=0.9)
    assert d["choice"] == "safe" and "fraction" in d["reason"]


def test_a_safe_agreement_that_is_not_finite_is_left_out_of_the_numbers_so_the_decision_file_stays_valid_json():
    d = decide(trial(), safe_top1=float("nan"))
    assert d["numbers"]["safe_top1_agreement"] is None
    json.dumps(d, allow_nan=False)                      # NaN would be written as a bare NaN, which is not JSON
