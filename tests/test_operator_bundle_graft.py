# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 8 bundle and stage 2's graft decision (orchard/graft_trial.py): a RESULTS.md section that says which variant shipped, why, and shows
both measurements, and the disposition of the `drafter_graft` hazard stage 0 raises for a graft candidate. The fake run is the one the
operator-bundle tests use; the decision, trial and package records are written by hand."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator_bundle_template import (GOLDEN, build, bundle_of, make_run, section, write, write_json)  # noqa: E402

GRAFT_HAZARD = {"area": "drafter_graft", "finding": "The 15 mtp.* tensors of Qwen/Qwen3.8-27B could be grafted onto the new model.",
                "evidence": ["stages/0/evidence/tensor-compare.json"]}
NO_MTP = {"area": "drafter", "finding": "The new model has no mtp.* tensors; the nearest model has 15. A drafter that needs the MTP head "
          "cannot run on it.", "evidence": ["stages/0/evidence/tensor-compare.json"]}
CANDIDATE = {"base_tensors": 15, "base_revision": "a" * 40, "base_shards": ["x"], "dtypes": ["BF16"], "hidden_size": 5120,
             "shapes_match": True, "graftable": True}


def trial(*, safe=20.0, variant=60.0):
    return {"label": "measured", "steps": [], "safe": {"server_ready_s": 100.0, "decode_tok_s": safe, "ttft_s": 0.3},
            "variant": {"served": True, "coherent": True, "top1_agreement": 0.96875, "n_tokens": 32, "server_ready_s": 240.0,
                        "decode_tok_s": variant, "ttft_s": 0.2, "failure": None}}


def decision(choice, *, trial_ran=True, reason=None):
    numbers = {"safe_decode_tok_s": 20.0, "variant_decode_tok_s": 60.0 if choice == "graft" else 21.0, "speedup": 3.0 if choice == "graft" else 1.05,
               "safe_ttft_s": 0.3, "variant_ttft_s": 0.2, "safe_top1_agreement": 0.9375, "variant_top1_agreement": 0.96875,
               "variant_server_ready_s": 240.0}
    return {"choice": choice, "reason": reason or ("the grafted variant decodes 3.0 times as fast as the safe configuration"
                                                   if choice == "graft" else "the grafted variant decodes only 1.05 times as fast"),
            "numbers": numbers if trial_ran else {}, "thresholds": {"min_speedup": 1.25, "top1_margin": 0.05, "top1_min": 0.85},
            "candidate": CANDIDATE, "trial": "graft-trial/trial-result.json" if trial_ran else None}


def with_graft(tmp_path, choice=None, *, hazards=(GRAFT_HAZARD,), trial_ran=True, package_graft=False, config_graft=None, reason=None,
               stage7=True, **kw):
    """A finished run whose stage 0 raised the graft hazard, with stage 2's decision and what stages 4 and 7 built."""
    run = make_run(tmp_path, stage7=stage7, **kw)
    s = run / "stages"
    d = json.loads((s / "0" / "delta.json").read_text())
    d["hazards"] = [*d["hazards"], *hazards]
    write_json(s / "0" / "delta.json", d)
    if choice:
        write_json(s / "2" / "graft-decision.json", decision(choice, trial_ran=trial_ran, reason=reason))
        if trial_ran:
            write_json(s / "2" / "graft-trial" / "trial-result.json", trial(variant=60.0 if choice == "graft" else 21.0))
    grafted = (choice == "graft") if config_graft is None else config_graft
    for n in (2, 4):
        cfg = s / "4" / "configs" / str(n)
        write_json(cfg / "swap_config.json", {"bundle_dir": "/b", "tt_cache": "/c", **({"graft_mtp": True} if grafted else {})})
        write_json(cfg / "model-dir-sources.json", {"files": [], **({"mtp_graft": {"from": "x", "file": "f", "tensors": 15}} if grafted else {})})
    if package_graft and stage7:
        pk = json.loads((s / "7" / "package.json").read_text())
        pk["mtp_graft"] = {"repo": "Qwen/Qwen3.8-27B", "file": "f"}
        write_json(s / "7" / "package.json", pk)
    return run


def risks_block(run, area="drafter_graft", contains=""):
    assert build(run).returncode == 0
    text = (bundle_of(run) / "RISKS.md").read_text()
    return [b for b in section(text, "## Stage 0 hazards").split("### ") if b.startswith(area) and contains in b][0]


def results(run):
    assert build(run).returncode == 0
    return (bundle_of(run) / "RESULTS.md").read_text()


# ---- the hazard -------------------------------------------------------------------------------------------------------------------

def test_a_graft_the_trial_chose_and_the_package_and_stage_4_carry_is_dealt_with(tmp_path):
    block = risks_block(with_graft(tmp_path, "graft", package_graft=True))
    assert "Dealt with: yes" in block
    assert "graft-decision.json" in block and "trial-result.json" in block
    assert "grafted" in block


def test_a_safe_decision_with_nothing_grafted_is_dealt_with(tmp_path):
    block = risks_block(with_graft(tmp_path, "safe", package_graft=False))
    assert "Dealt with: yes" in block and "safe configuration" in block


def test_no_decision_file_is_not_shown_and_says_the_trial_did_not_run(tmp_path):
    block = risks_block(with_graft(tmp_path, None))
    assert "Dealt with: Not shown" in block and "graft-decision.json" in block and "not in this run" in block


def test_a_skipped_trial_is_not_shown_and_gives_its_reason(tmp_path):
    block = risks_block(with_graft(tmp_path, "safe", trial_ran=False, reason="the graft trial was skipped: --no-graft-trial"))
    assert "Dealt with: Not shown" in block and "skipped" in block and "--no-graft-trial" in block


def test_a_graft_decision_with_a_package_that_is_not_grafted_is_not_shown(tmp_path):
    block = risks_block(with_graft(tmp_path, "graft", package_graft=False))
    assert "Dealt with: Not shown" in block and "package" in block and "no graft" in block.lower()


def test_a_safe_decision_with_a_grafted_package_is_not_shown(tmp_path):
    block = risks_block(with_graft(tmp_path, "safe", package_graft=True))
    assert "Dealt with: Not shown" in block and "package" in block and "grafted" in block


def test_a_graft_decision_with_a_stage_4_bundle_configuration_built_without_it_is_not_shown(tmp_path):
    run = with_graft(tmp_path, "graft", package_graft=True)
    write_json(run / "stages/4/configs/2/model-dir-sources.json", {"files": []})
    block = risks_block(run)
    assert "Dealt with: Not shown" in block and "stage 4" in block and "2" in block


def test_a_graft_decision_with_no_stage_7_is_judged_on_stage_4_alone(tmp_path):
    block = risks_block(with_graft(tmp_path, "graft", stage7=False))
    assert "Dealt with: yes" in block


def test_a_decision_file_that_is_not_json_is_not_shown(tmp_path):
    run = with_graft(tmp_path, "graft", package_graft=True)
    write(run / "stages/2/graft-decision.json", "{not json")
    assert "Dealt with: Not shown" in risks_block(run)


def test_a_decision_with_an_unknown_choice_is_not_shown(tmp_path):
    run = with_graft(tmp_path, "safe", package_graft=False)            # everything built agrees with "safe"
    write_json(run / "stages/2/graft-decision.json", {"choice": "maybe", "reason": "x", "trial": "graft-trial/trial-result.json"})
    block = risks_block(run)
    assert "Dealt with: Not shown" in block and "no choice" in block


def test_a_container_configuration_is_not_held_to_the_decision(tmp_path):
    run = with_graft(tmp_path, "graft", package_graft=True)
    write_json(run / "stages/4/configs/4/swap_config.json", {"package": "x/y", "profile": "p", "tt_cache": "/c"})
    write_json(run / "stages/4/configs/4/model-dir-sources.json", {"files": []})
    assert "Dealt with: yes" in risks_block(run)


def test_stage_2_grafted_by_a_person_with_the_drafter_on_counts_as_a_grafted_check(tmp_path):
    run = with_graft(tmp_path, "graft", hazards=(NO_MTP,), package_graft=True, drafter="on (dflash2)", drafter_4="on (dflash2)")
    write_json(run / "stages/2/model-dir-sources.json", {"files": [], "mtp_graft": {"from": "x", "file": "f"}})
    assert "Dealt with: yes" in risks_block(run, "drafter", "no mtp.* tensors")


def test_stage_2_with_the_drafter_on_and_no_graft_record_is_not_dealt_with(tmp_path):
    run = with_graft(tmp_path, "graft", hazards=(NO_MTP,), package_graft=True, drafter="on (dflash2)", drafter_4="on (dflash2)")
    block = risks_block(run, "drafter", "no mtp.* tensors")
    assert "Dealt with: Not shown" in block and "stage 2" in block


# ---- the existing "no mtp.* tensors" drafter hazard -------------------------------------------------------------------------------

def test_the_no_mtp_hazard_is_dealt_with_when_the_drafter_is_on_only_where_a_graft_was_built(tmp_path):
    run = with_graft(tmp_path, "graft", hazards=(NO_MTP,), package_graft=True, drafter="off", drafter_4="on (dflash2)")
    block = risks_block(run, "drafter", "no mtp.* tensors")
    assert "Dealt with: yes" in block and "graft" in block


def test_the_no_mtp_hazard_is_not_dealt_with_when_a_drafter_served_a_configuration_with_no_graft_record(tmp_path):
    run = with_graft(tmp_path, "graft", hazards=(NO_MTP,), package_graft=True, drafter="off", drafter_4="on (dflash2)")
    write_json(run / "stages/4/configs/4/model-dir-sources.json", {"files": []})
    write_json(run / "stages/4/configs/4/swap_config.json", {"bundle_dir": "/b", "tt_cache": "/c"})
    assert "Dealt with: Not shown" in risks_block(run, "drafter", "no mtp.* tensors")


def test_the_no_mtp_hazard_keeps_its_old_rule_when_there_is_no_graft_decision(tmp_path):
    run = make_run(tmp_path, drafter="off")
    d = json.loads((run / "stages/0/delta.json").read_text())
    d["hazards"] = [*d["hazards"], NO_MTP]
    write_json(run / "stages/0/delta.json", d)
    assert "Dealt with: yes" in risks_block(run, "drafter", "no mtp.* tensors")


# ---- RESULTS.md ----------------------------------------------------------------------------------------------------------------------

def test_the_results_say_which_variant_shipped_why_and_show_both_measurements(tmp_path):
    text = results(with_graft(tmp_path, "graft", package_graft=True))
    sec = section(text, "## The graft trial")
    assert "grafted" in sec and "chose the graft" in sec.lower()
    assert "the grafted variant decodes 3.0 times as fast as the safe configuration" in sec
    rows = {ln.split("|")[1].strip(): ln for ln in sec.splitlines() if ln.startswith("| ") and not ln.startswith("| Measure") and "---" not in ln}
    safe_row = next(v for k, v in rows.items() if "decode" in k.lower())
    assert "20" in safe_row and "60" in safe_row
    assert "stages/2/graft-trial/trial-result.json" in sec and "stages/2/graft-decision.json" in sec
    assert "min_speedup" in sec or "1.25" in sec


def test_a_kept_safe_decision_is_shown_with_both_measurements(tmp_path):
    sec = section(results(with_graft(tmp_path, "safe")), "## The graft trial")
    assert "safe configuration" in sec and "kept" in sec.lower() and "only 1.05 times" in sec
    assert "21" in sec and "20" in sec


def test_a_skipped_trial_is_shown_as_skipped_with_no_measurements(tmp_path):
    sec = section(results(with_graft(tmp_path, "safe", trial_ran=False, reason="the graft trial was skipped: --no-graft-trial")),
                  "## The graft trial")
    assert "skipped" in sec and "--no-graft-trial" in sec and "| " not in sec


def test_a_run_with_no_decision_gets_no_graft_section_and_the_golden_text(tmp_path):
    run = make_run(tmp_path)
    text = results(run)
    assert "The graft trial" not in text and text == GOLDEN.read_text()


def test_the_section_sits_after_the_served_directory_and_before_stage_0(tmp_path):
    run = with_graft(tmp_path, "graft", package_graft=True)
    write_json(run / "stages/2/model-dir-sources.json", {"files": [], "sampling_defaults_from": "base"})
    text = results(run)
    assert text.index("## The served model directory") < text.index("## The graft trial") < text.index("## Stage 0")


def test_text_copied_from_the_decision_is_redacted(tmp_path):
    text = results(with_graft(tmp_path, "safe", reason="the variant failed at /home/someone/cache/x"))
    assert "The graft trial" in text and "/home/someone" not in text


@pytest.mark.parametrize("bad", ["{not json", "[]", '{"choice": "graft"}', '{"choice": "graft", "numbers": 5, "reason": 7}'])
def test_a_malformed_decision_cannot_break_the_build(tmp_path, bad):
    run = with_graft(tmp_path, "graft")
    write(run / "stages/2/graft-decision.json", bad)
    assert build(run).returncode == 0


# ---- a trial that left no result is not a measured comparison (review finding on PR 47) -------------------------------------------------
# A timed-out trial gets a safe decision whose `trial` still names trial-result.json, and whose numbers are empty. Before this was settled, the
# bundle called the hazard dealt with, said both configurations were measured, and cited a result file that does not exist.

TIMED_OUT = "the graft trial timed out and was stopped; the safe configuration stands"


def without_result(tmp_path, *, numbers=None, reason=TIMED_OUT, choice="safe"):
    run = with_graft(tmp_path, choice, package_graft=False, reason=reason)
    (run / "stages/2/graft-trial/trial-result.json").unlink()
    d = json.loads((run / "stages/2/graft-decision.json").read_text())
    d["numbers"] = {} if numbers is None else numbers
    write_json(run / "stages/2/graft-decision.json", d)
    return run


def test_a_timed_out_trial_with_no_result_is_not_shown_and_gives_the_recorded_reason(tmp_path):
    block = risks_block(without_result(tmp_path))
    assert "Dealt with: Not shown" in block
    assert "timed out" in block
    assert "measured both" not in block


def test_a_missing_result_file_is_not_shown_even_when_the_decision_still_holds_numbers(tmp_path):
    run = with_graft(tmp_path, "graft", package_graft=True)
    (run / "stages/2/graft-trial/trial-result.json").unlink()
    block = risks_block(run)
    assert "Dealt with: Not shown" in block and "trial-result.json" in block


def test_a_trial_whose_safe_speed_was_not_measured_is_not_shown(tmp_path):
    """The trial's safe measurement failed, so the variant was never tried: there is no comparison."""
    run = with_graft(tmp_path, "safe", package_graft=False, reason="the graft trial's safe speed was not measured")
    t = json.loads((run / "stages/2/graft-trial/trial-result.json").read_text())
    t["safe"] = {"failure": "measure_serving exited with code 1"}
    t["variant"] = {"served": False, "failure": "not tried: the safe configuration's speed was not measured"}
    write_json(run / "stages/2/graft-trial/trial-result.json", t)
    d = json.loads((run / "stages/2/graft-decision.json").read_text())
    d["numbers"] = {}
    write_json(run / "stages/2/graft-decision.json", d)
    block = risks_block(run)
    assert "Dealt with: Not shown" in block and "safe speed was not measured" in block
    assert "measured both" not in block


def test_a_variant_that_did_not_serve_is_a_trial_that_compared_nothing_but_the_safe_run_is_measured(tmp_path):
    """The variant failed to serve: both measurements do not exist, so the hazard is not dealt with and the failure is quoted."""
    run = with_graft(tmp_path, "safe", package_graft=False, reason="the grafted variant did not serve")
    t = json.loads((run / "stages/2/graft-trial/trial-result.json").read_text())
    t["variant"] = {"served": False, "failure": "the server exited"}
    write_json(run / "stages/2/graft-trial/trial-result.json", t)
    d = json.loads((run / "stages/2/graft-decision.json").read_text())
    d["numbers"] = {"safe_decode_tok_s": 20.0, "safe_ttft_s": 0.3}
    write_json(run / "stages/2/graft-decision.json", d)
    block = risks_block(run)
    assert "Dealt with: Not shown" in block and "did not serve" in block


@pytest.mark.parametrize("choice,packaged", [("safe", False), ("graft", True)])
def test_a_complete_trial_is_still_dealt_with(tmp_path, choice, packaged):
    assert "Dealt with: yes" in risks_block(with_graft(tmp_path, choice, package_graft=packaged))


def test_the_results_section_does_not_cite_a_result_file_that_is_not_there(tmp_path):
    text = results(without_result(tmp_path))
    part = text[text.index("## The graft trial"):]
    part = part[:part.index("\n## ", 5)] if "\n## " in part[5:] else part
    assert "trial-result.json" not in part and "graft-decision.json" in part
    assert "TODO" in part


def test_the_results_section_cites_the_result_file_when_it_is_there(tmp_path):
    text = results(with_graft(tmp_path, "graft", package_graft=True))
    part = text[text.index("## The graft trial"):]
    assert "trial-result.json" in part


@pytest.mark.parametrize("bad", ["fast", True, None, [20.0]])
def test_a_decision_number_that_is_not_a_number_is_not_a_measurement(tmp_path, bad):
    run = without_result(tmp_path, numbers={"safe_decode_tok_s": bad, "variant_decode_tok_s": 21.0})
    write_json(run / "stages/2/graft-trial/trial-result.json", trial(variant=21.0))      # the result file is there; the decision's number is not
    block = risks_block(run)
    assert "Dealt with: Not shown" in block and "decode speed" in block


def test_the_results_section_says_the_trial_left_no_result_when_it_left_none(tmp_path):
    text = results(without_result(tmp_path))
    part = text[text.index("## The graft trial"):]
    assert "left no result file" in part and "nothing was compared" in part
    assert "timed out" in part                                       # the recorded reason is quoted
