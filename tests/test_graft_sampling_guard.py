# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Order of the graft trial and the sampling choice (issue 107): a graft keeps the drafter on, so for a model that needs sampling it is chosen only when
the grafted variant's own sampling probe shows sampling applied. The trial still runs; the decision refuses the graft with the reason."""
import json

import pytest

from orchard import sampling_choice as sc
from orchard.graft_trial import DECISION_FILE, TRIAL_DIR, TRIAL_RESULT, decide, decision_problems, write_decision
from test_graft_trial import CANDIDATE, trial
from test_sampling_choice_files import report, write
from orchard.ledger import Ledger


def sampling(needed=True, verdict="sampled-full-speed"):
    return {"needed": needed, "verdict": verdict}


def test_a_graft_whose_variant_applies_sampling_is_chosen_when_sampling_is_needed():
    d = decide(trial(), safe_top1=0.96875, sampling=sampling(True, "sampled-full-speed"))
    assert d["choice"] == "graft"
    assert d["numbers"]["sampling_needed"] is True and d["numbers"]["variant_sampling_verdict"] == "sampled-full-speed"


@pytest.mark.parametrize("verdict", ["sampled-ignored", "sampled-refused", "sampled-incoherent", "inconclusive", "not-run", "malformed", None])
def test_a_graft_whose_variant_does_not_show_sampling_applied_is_not_chosen_when_sampling_is_needed(verdict):
    d = decide(trial(), safe_top1=0.96875, sampling=sampling(True, verdict))
    assert d["choice"] == "safe"
    assert "sampling" in d["reason"] and "needs" in d["reason"]


def test_sampled_slower_counts_as_applied():
    assert decide(trial(), safe_top1=0.96875, sampling=sampling(True, "sampled-slower"))["choice"] == "graft"


def test_a_graft_is_chosen_whatever_the_verdict_when_sampling_is_not_needed():
    d = decide(trial(), safe_top1=0.96875, sampling=sampling(False, "sampled-ignored"))
    assert d["choice"] == "graft"
    assert d["numbers"]["sampling_needed"] is False


def test_without_a_sampling_input_the_decision_is_the_old_one_and_the_numbers_stay_as_they_were():
    d = decide(trial(), safe_top1=0.96875)
    assert d["choice"] == "graft" and "sampling_needed" not in d["numbers"]


def test_a_speed_reason_comes_before_the_sampling_reason():
    d = decide(trial(variant=21.0), safe_top1=0.96875, sampling=sampling(True, "sampled-ignored"))
    assert d["choice"] == "safe" and "times as fast" in d["reason"]


# ---- the gate recomputes it from the files -------------------------------------------------------------------------------------------

def stage(tmp_path, *, needed, variant_report, override="auto"):
    run = tmp_path / "run"
    sd = run / "stages" / "2"
    write(run / "stages/0/evidence/genconfig-license.json",
          {"model_generation_config": {"temperature": 0.6} if needed else {}, "model_generation_config_present": needed,
           "card_sampling_hints": {}, "card_warns_greedy": []})
    write(sd / "result.json", {"serves": True, "top1_agreement": 0.96875})
    write(sd / TRIAL_DIR / TRIAL_RESULT, trial())
    write(sd / TRIAL_DIR / "variant/evidence/swap-check.json", variant_report)
    Ledger(run / "ledger.jsonl").append("run_start", None, model="x/y", sampling=override)
    return run, sd


def test_a_graft_decision_for_a_model_that_needs_sampling_is_refused_when_the_variants_probe_ignored_it(tmp_path):
    run, sd = stage(tmp_path, needed=True, variant_report=report(ignored=True))
    write_decision(sd, decide(trial(), safe_top1=0.96875), candidate=CANDIDATE)           # decided without the sampling input
    problems = decision_problems(sd, run)
    assert any("sampling" in p for p in problems)


def test_the_same_decision_is_fine_when_sampling_is_not_needed(tmp_path):
    run, sd = stage(tmp_path, needed=False, variant_report=report(ignored=True))
    write_decision(sd, decide(trial(), safe_top1=0.96875), candidate=CANDIDATE)
    assert decision_problems(sd, run) == []


def test_a_graft_decision_whose_variant_applies_sampling_passes(tmp_path):
    run, sd = stage(tmp_path, needed=True, variant_report=report(ignored=False, seconds=1.6))
    write_decision(sd, decide(trial(), safe_top1=0.96875, sampling=sampling(True, "sampled-full-speed")), candidate=CANDIDATE)
    assert decision_problems(sd, run) == []


def test_the_operators_override_is_what_the_gate_recomputes_with(tmp_path):
    run, sd = stage(tmp_path, needed=True, variant_report=report(ignored=True), override="not-needed")
    write_decision(sd, decide(trial(), safe_top1=0.96875), candidate=CANDIDATE)
    assert decision_problems(sd, run) == []


def test_helper_reads_the_variants_verdict_and_the_need(tmp_path):
    run, sd = stage(tmp_path, needed=True, variant_report=report(ignored=True))
    from orchard import graft_trial
    s = graft_trial.sampling_input(sd, run)
    assert s == {"needed": True, "verdict": "sampled-ignored"}
    (sd / TRIAL_DIR / "variant/evidence/swap-check.json").unlink()
    assert graft_trial.sampling_input(sd, run) == {"needed": True, "verdict": None}
