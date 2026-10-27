# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 8 bundle and stage 2's sampling decision (issue 107): RESULTS.md has a "The sampling choice" section that says which configuration the run
ships, why, and what the files showed; RISKS.md lists a flagged decision under "Sampling choice" so the operator reads it. A run with no decision file gets
the same bundle as before."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator_bundle_template import build, bundle_of, make_run, section, write_json  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402

HEAD = "## The sampling choice"
NEEDED = {"needed": True, "source": "generation_config", "override": "auto", "evidence": ["the model's own generation_config.json sets temperature 0.6"]}


def side(drafter="on", verdict="sampled-ignored", greedy=60.0, sampled=59.0, served=True):
    return {"drafter": drafter, "verdict": verdict, "greedy_tok_s": greedy, "sampled_tok_s": sampled, "served": served}


def put(run, decision, trial=True):
    write_json(run / "stages/2" / sc.DECISION_FILE, {**decision, "trial": f"{sc.TRIAL_DIR}/{sc.TRIAL_RESULT}" if trial else None})


def files(run):
    assert build(run).returncode == 0
    b = bundle_of(run)
    return (b / "RESULTS.md").read_text(), (b / "RISKS.md").read_text()


def test_a_run_without_a_decision_has_neither_section(tmp_path):
    run = make_run(tmp_path)
    results, risks = files(run)
    assert HEAD not in results and "## Sampling choice" not in risks


def test_a_drafter_off_choice_is_stated_with_the_reason_the_numbers_and_the_evidence_file(tmp_path):
    run = make_run(tmp_path)
    put(run, sc.decide(NEEDED, side(), side(drafter="off", verdict="sampled-full-speed", greedy=18.4, sampled=17.2, )))
    results, risks = files(run)
    block = section(results, HEAD)
    assert "chose the drafter-off configuration" in block and "generation_config.json" in block
    assert "59.0" in block and "17.2" in block and "10" in block                 # the speed the first configuration had, the variant's, the floor
    assert "sampled-ignored" in block and "sampled-full-speed" in block
    assert "stages/2/sampling-decision.json" in block and "measured" in block
    assert "## Sampling choice" not in risks                                       # no flag, nothing to warn about


def test_a_keep_choice_for_a_model_that_does_not_need_sampling_says_so(tmp_path):
    run = make_run(tmp_path)
    put(run, sc.decide({"needed": False, "source": "none", "override": "auto", "evidence": ["no evidence"]}, side()), trial=False)
    results, _ = files(run)
    block = section(results, HEAD)
    assert "kept the configuration it served" in block and "not needed" in block and "no evidence" in block


def test_a_flagged_choice_is_in_risks_in_plain_words(tmp_path):
    run = make_run(tmp_path)
    put(run, sc.decide(NEEDED, side(), side(drafter="off", verdict="sampled-ignored", greedy=18.0, sampled=18.0)))
    results, risks = files(run)
    assert "no-variant-serves-purpose" in section(results, HEAD)
    block = section(risks, "## Sampling choice")
    assert "does not do what the model needs" in block and "stages/2/sampling-decision.json" in block


def test_each_flag_gets_a_risk_and_an_unflagged_choice_gets_none(tmp_path):
    for variant, flag in ((side(drafter="off", verdict="sampled-full-speed", greedy=7.0, sampled=7.0), "sampling-applied-below-floor"),
                          (side(drafter="off", verdict="sampled-ignored", greedy=7.0, sampled=7.0), "no-variant-serves-purpose"),
                          (None, "variant-not-tried"), (side(drafter="off", verdict="inconclusive", greedy=18.0, sampled=None), "unverified")):
        run = make_run(tmp_path / flag)
        d = sc.decide(NEEDED, side(), variant, variant_note="skipped" if variant is None else None)
        assert d["flag"] == flag
        put(run, d, trial=variant is not None)
        _, risks = files(run)
        assert "## Sampling choice" in risks, flag


def test_the_below_floor_flag_is_a_risk(tmp_path):
    run = make_run(tmp_path)
    put(run, sc.decide(NEEDED, side(drafter="off", verdict="sampled-full-speed", greedy=9.0, sampled=8.0)))
    _, risks = files(run)
    assert "below the usability floor" in section(risks, "## Sampling choice")


def test_copied_text_goes_through_the_redactor(tmp_path):
    run = make_run(tmp_path)
    d = sc.decide(NEEDED, side(), side(drafter="off", verdict="sampled-full-speed", greedy=18.4, sampled=17.2))
    d["need_evidence"] = ["the card says /home/someone/secret/path"]
    put(run, d)
    results, _ = files(run)
    assert "/home/someone" not in results


def test_a_malformed_decision_file_adds_nothing(tmp_path):
    run = make_run(tmp_path)
    (run / "stages/2" / sc.DECISION_FILE).write_text("{not json")
    results, risks = files(run)
    assert HEAD not in results and "## Sampling choice" not in risks
    write_json(run / "stages/2" / sc.DECISION_FILE, {"choice": "maybe", "reason": "x"})
    results, _ = files(run)
    assert HEAD not in results
