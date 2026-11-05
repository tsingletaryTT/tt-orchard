# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 6 of a full port (issue 329): the template runs for real as a subprocess on the CPU against a fake `ttnn`, and the gate
(orchard/full_port_quality.py) judges what it wrote. Each guard has a case that fails when the guard is gone (scripts/mutations/full_port_stage_6.py)."""
import json
import math

import pytest

from orchard import full_port_quality as fq
from orchard import resident
from port_check_fakes import CHIPS, HONEST
from quality_fakes import (DECLARED, HUGE_CONFIG, LOOPS, PRINTS_THE_MARK, READS_REFERENCE, REWRITES_ITSELF, SKEWED, check, derive_and_write, make_run,
                           run_template, sha)


def good(tmp_path, port=None, **kw):
    """A run with the template already run on the honest port (or `port`) and the supervisor's result.json beside its record. Returns (run, stage, seeded)."""
    run, stage, snap, seeded = make_run(tmp_path, *([port] if port else []), **kw)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout + done.stderr
    derive_and_write(run, stage, source=kw.get("source", 2), rehearsal=True)         # a result.json to judge; a test that needs another derives again
    return run, stage, seeded


def record(stage):
    return json.loads((stage / "evidence" / "port-quality.json").read_text())


def rewrite_record(stage, edit, *, remark=True):
    """Edit the record on disk and, by default, put the new record's sha256 in the output file and the marker, as a forger would have to."""
    rec = record(stage)
    edit(rec)
    data = (json.dumps(rec, indent=2) + "\n").encode()
    (stage / "evidence" / "port-quality.json").write_bytes(data)
    if remark:
        out = stage / "evidence" / "hw-test-output.txt"
        lines = [l for l in out.read_text().splitlines() if not l.startswith("PORT_QUALITY_SHA256")]
        out.write_text("\n".join(lines + [f"PORT_QUALITY_SHA256 {sha(data)}"]) + "\n")
        marker = json.loads((stage / "test-result.json").read_text())
        marker["output"]["sha256"] = sha(out.read_bytes())
        (stage / "test-result.json").write_text(json.dumps(marker))


def has(reasons, text):
    return any(text in r for r in reasons)


# ---- a pass, and what the record holds -----------------------------------------------------------------------------------------

def test_the_template_runs_and_the_derived_result_passes_the_gate(tmp_path):
    run, stage, seeded = good(tmp_path)
    result, why = derive_and_write(run, stage)
    assert why == "derived" and result["derived_by"] == "supervisor" and result["schema"] == fq.RESULT_SCHEMA
    reasons, seen = check(run, stage, seeded)
    assert reasons == [] and seen
    assert result["free_run"]["leading_match"] == 32 and result["perplexity"]["top5_gap_diff"] == pytest.approx(0.0)
    assert result["perplexity"]["perplexity"] == pytest.approx(math.exp(-sum(record(stage)["logprobs"]) / 32))


def test_the_record_holds_the_benchmark_by_the_rules(tmp_path):
    run, stage, _ = good(tmp_path)
    rec = record(stage)
    warm = [r for r in rec["bench"]["runs"] if r["warmup"]]
    measured = [r for r in rec["bench"]["runs"] if not r["warmup"]]
    assert len(warm) >= resident.WARMUP_RUNS and len(measured) >= resident.MIN_REPEATS
    assert [r["warmup"] for r in rec["bench"]["runs"]] == [True] * len(warm) + [False] * len(measured)      # warm-up first, in the same process
    assert all(r["steps"] >= fq.MIN_BENCH_STEPS and r["decode_s"] > 0 for r in measured)


def test_every_number_is_labelled_measured_or_todo_and_the_unmeasured_things_are_todo(tmp_path):
    run, stage, _ = good(tmp_path)
    result, _ = derive_and_write(run, stage)
    rows = result["numbers"]
    assert {r["label"] for r in rows} == {"measured", "TODO"}
    for r in rows:
        assert (r["label"] == "measured") == (isinstance(r["value"], (int, float)))        # a value is measured; no value is TODO
    names = {r["name"] for r in rows if r["label"] == "TODO"}
    assert names == set(fq.TODO_NUMBERS)
    assert any(r["name"].startswith("share of the estimated decode floor") and r["label"] == "measured" for r in rows)


def test_a_port_that_declares_its_dtype_policy_is_floored_with_it_and_one_that_does_not_is_assumed_bf16(tmp_path):
    run, stage, _ = good(tmp_path, port=DECLARED)
    b = derive_and_write(run, stage)[0]["benchmark"]
    assert b["dtype_policy"] == {"default": "bfp8"} and "declared by the port" in b["dtype_policy_source"]
    run2, stage2, _ = good(tmp_path / "b")
    b2 = derive_and_write(run2, stage2)[0]["benchmark"]
    assert b2["dtype_policy"] == {"default": "bf16"} and b2["dtype_policy_source"].startswith("assumed")
    assert b["floor"]["floor_tok_s"] > b2["floor"]["floor_tok_s"]            # fewer bytes per weight, so a higher floor in tokens per second


def test_the_floor_is_for_the_dies_the_test_ran_on(tmp_path):
    run, stage, _ = good(tmp_path)
    b = derive_and_write(run, stage)[0]["benchmark"]
    assert b["chips"] == len(CHIPS) and b["floor"]["inputs"]["dies"] == len(CHIPS) and b["floor"]["inputs"]["board"] == "p300"


# ---- the numbers are recomputed ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("edit", [
    lambda r: r["perplexity"].update(perplexity=1.0),
    lambda r: r["benchmark"].update(share_of_floor_median=0.5),
    lambda r: r["free_run"].update(leading_match=5),
    lambda r: r.update(numbers=[]),
])
def test_a_result_that_differs_from_what_the_record_gives_is_refused(tmp_path, edit):
    run, stage, seeded = good(tmp_path)
    result, _ = derive_and_write(run, stage)
    edit(result)
    (stage / "result.json").write_text(json.dumps(result))
    assert has(check(run, stage, seeded)[0], "is not what the record gives")


def test_a_record_edited_after_the_test_does_not_match_the_output_mark(tmp_path):
    run, stage, seeded = good(tmp_path)
    derive_and_write(run, stage)
    rewrite_record(stage, lambda r: r.update(logprobs=[-0.001] * 32), remark=False)
    assert has(check(run, stage, seeded)[0], "does not carry a PORT_QUALITY_SHA256 line that matches")


def test_logprobs_that_are_not_log_probabilities_are_refused(tmp_path):
    run, stage, seeded = good(tmp_path)
    rewrite_record(stage, lambda r: r.update(logprobs=[0.5] * 32))
    assert has(check(run, stage, seeded)[0], "logprobs must be 32 finite numbers at or below zero")


# ---- the benchmark rules -------------------------------------------------------------------------------------------------------

def test_fewer_than_three_measured_repeats_is_refused(tmp_path):
    run, stage, seeded = good(tmp_path)
    rewrite_record(stage, lambda r: r["bench"].update(runs=r["bench"]["runs"][:3]))          # one warm-up and two measured
    assert has(check(run, stage, seeded)[0], "2 measured run(s); the rule is 3")


def test_no_warm_up_run_is_refused(tmp_path):
    run, stage, seeded = good(tmp_path)
    rewrite_record(stage, lambda r: r["bench"].update(runs=[x for x in r["bench"]["runs"] if not x["warmup"]]))
    assert has(check(run, stage, seeded)[0], "0 warm-up run(s); the rule is 1")


def test_a_decode_speed_that_is_not_steps_over_time_is_refused(tmp_path):
    run, stage, seeded = good(tmp_path)

    def edit(r):
        r["bench"]["runs"][-1]["decode_tok_s"] *= 3
    rewrite_record(stage, edit)
    assert has(check(run, stage, seeded)[0], "its decode speed is not its steps over its decode time")


def test_a_share_above_the_bound_is_refused(tmp_path):
    run, stage, snap, seeded = make_run(tmp_path, config=HUGE_CONFIG)
    assert run_template(tmp_path, run, stage).returncode == 0
    reasons, _ = check(run, stage, seeded)
    result, why = derive_and_write(run, stage)
    reasons, _ = check(run, stage, seeded)
    assert result is not None and has(reasons, "is not a share between 0 and")


def test_a_model_config_changed_after_the_test_is_refused(tmp_path):
    run, stage, seeded = good(tmp_path)
    derive_and_write(run, stage)
    (tmp_path / "snapshot" / "config.json").write_text(json.dumps({**json.loads((tmp_path / "snapshot" / "config.json").read_text()), "num_hidden_layers": 40}))
    assert has(check(run, stage, seeded)[0], "config.json is not the file the record hashed")


# ---- quality against the reference -----------------------------------------------------------------------------------------------

def test_logit_gaps_that_differ_from_the_references_by_more_than_the_bar_fail(tmp_path):
    run, stage, seeded = good(tmp_path, port=SKEWED)
    result, why = derive_and_write(run, stage)
    assert result is not None and result["perplexity"]["top5_gap_diff"] == pytest.approx(5.0)
    assert has(check(run, stage, seeded)[0], f"the bar is {fq.GAP_MAX}")


def test_the_rehearsal_waiver_covers_coherent_and_nothing_else(tmp_path):
    run, stage, seeded = good(tmp_path, port=SKEWED)           # right ids, wrong gaps
    derive_and_write(run, stage, rehearsal=True)
    waived = []
    reasons, _ = check(run, stage, seeded, rehearsal=True, waived=waived)
    assert has(reasons, f"the bar is {fq.GAP_MAX}")             # the gap bar still applies under the option


def test_a_looping_free_run_is_incoherent_and_only_the_rehearsal_option_lets_it_through(tmp_path):
    run, stage, seeded = good(tmp_path, port=LOOPS)
    result, why = derive_and_write(run, stage)
    assert result is None and "readable floor" in why             # no rehearsal option: nothing is derived
    result, why = derive_and_write(run, stage, rehearsal=True)
    assert result["coherent"] is False and result["free_run"]["leading_match"] < 5
    assert has(check(run, stage, seeded)[0], "coherent must be true")        # the gate without the option refuses the same file
    waived = []
    reasons, _ = check(run, stage, seeded, rehearsal=True, waived=waived)
    assert reasons == [] and waived == ["coherent"]


def test_a_result_that_claims_coherent_for_a_loop_is_refused_even_under_the_option(tmp_path):
    run, stage, seeded = good(tmp_path, port=LOOPS)
    result, _ = derive_and_write(run, stage, rehearsal=True)
    result["coherent"] = True
    (stage / "result.json").write_text(json.dumps(result))
    assert has(check(run, stage, seeded, rehearsal=True)[0], "coherent is True; the supervisor's floor")


def test_a_model_the_port_gets_wrong_fails_the_top_1_bar(tmp_path):
    run, stage, seeded = good(tmp_path)
    derive_and_write(run, stage)
    rewrite_record(stage, lambda r: r.update(chip_ids=[100] + [0] * 31))
    assert has(check(run, stage, seeded)[0], "below the minimum")


def test_the_second_prefill_must_start_afresh(tmp_path):
    run, stage, seeded = good(tmp_path)
    rewrite_record(stage, lambda r: r["free_run_ids"].__setitem__(0, 7))
    assert has(check(run, stage, seeded)[0], "prefill must reset the model's state")


# ---- the stage 1 files -------------------------------------------------------------------------------------------------------------

def test_the_reference_top_five_in_the_record_must_be_the_stage_1_file(tmp_path):
    run, stage, seeded = good(tmp_path)
    rewrite_record(stage, lambda r: r["reference_top5"][3]["logits"].__setitem__(1, 0.25))
    assert has(check(run, stage, seeded)[0], "reference_top5[3] is not the top five in")


def test_a_stage_1_top_five_file_changed_after_the_test_is_refused(tmp_path):
    run, stage, seeded = good(tmp_path)
    p = run / "stages" / "1" / "evidence" / "reference" / "top5-logits.json"
    p.write_text(p.read_text().replace("1.0", "1.5"))
    assert has(check(run, stage, seeded)[0], "top5.sha256 does not match")


def test_stage_1_with_no_top_five_file_fails_the_template_with_a_named_reason(tmp_path):
    run, stage, snap, seeded = make_run(tmp_path, top5=False)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "top5-logits.json is missing" in done.stdout
    assert 'PORT_QUALITY_FAILURE {"kind": "setup"' in done.stdout
    assert fq.derive(stage, run, source_stage=2)[0] is None


def test_reference_ids_changed_after_the_test_are_refused(tmp_path):
    run, stage, seeded = good(tmp_path)
    p = run / "stages" / "1" / "evidence" / "reference" / "generated-ids.json"
    p.write_text(p.read_text().replace("100,", "999,", 1))
    assert has(check(run, stage, seeded)[0], "reference.sha256 does not match")


# ---- provenance ----------------------------------------------------------------------------------------------------------------------

def test_a_port_that_prints_the_mark_line_fails_the_run(tmp_path):
    run, stage, snap, seeded = make_run(tmp_path, port_src=PRINTS_THE_MARK)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "only this file prints that line" in done.stdout


def test_an_edited_template_is_refused(tmp_path):
    run, stage, seeded = good(tmp_path)
    t = stage / fq.TEMPLATE_NAME
    t.write_text(t.read_text() + "\n# edited\n")
    assert has(check(run, stage, seeded)[0], "differs from the shipped template")


def test_a_record_that_does_not_carry_the_shipped_templates_hash_is_refused(tmp_path):
    run, stage, seeded = good(tmp_path)
    rewrite_record(stage, lambda r: r.update(template_sha256="0" * 64))
    assert has(check(run, stage, seeded)[0], "template_sha256")


def test_a_command_other_than_the_fixed_one_is_refused(tmp_path):
    run, stage, seeded = good(tmp_path)
    marker = json.loads((stage / "test-result.json").read_text())
    marker["command"] = "python3 stages/6/port_quality.py"
    (stage / "test-result.json").write_text(json.dumps(marker))
    assert has(check(run, stage, seeded)[0], "the supervisor's command for this stage")


@pytest.mark.parametrize("edit,text", [(lambda m: m.update(returncode=1), "exited with code 1"), (lambda m: m.update(timed_out=True), "past its deadline")])
def test_a_failed_or_timed_out_test_is_refused(tmp_path, edit, text):
    run, stage, seeded = good(tmp_path)
    marker = json.loads((stage / "test-result.json").read_text())
    edit(marker)
    (stage / "test-result.json").write_text(json.dumps(marker))
    assert has(check(run, stage, seeded)[0], text)
    assert fq.derive(stage, run, source_stage=2)[0] is None


def test_the_test_must_have_run_on_the_chip_count_the_supervisor_expects(tmp_path):
    run, stage, seeded = good(tmp_path)
    derive_and_write(run, stage)
    assert has(check(run, stage, seeded, expected_chips=4)[0], "stage 6 measures the port on 4")


def test_a_chip_list_that_is_not_the_leased_one_is_refused(tmp_path):
    run, stage, seeded = good(tmp_path)
    marker = json.loads((stage / "test-result.json").read_text())
    marker["chips"] = ["somewhere-else"]
    (stage / "test-result.json").write_text(json.dumps(marker))
    assert has(check(run, stage, seeded)[0], "are not the chips the supervisor leased")


def test_a_port_that_changed_its_own_source_during_the_test_is_not_the_port_that_passed(tmp_path):
    run, stage, seeded = good(tmp_path, port=REWRITES_ITSELF, source_port_src=REWRITES_ITSELF)
    derive_and_write(run, stage)
    reasons, _ = check(run, stage, seeded)
    assert has(reasons, "are not the .py files the supervisor copied from stage 2")
    assert has(reasons, "not the ones stage 2's own record names")


def test_no_ledger_record_of_the_copy_is_a_refusal_not_a_skip(tmp_path):
    run, stage, seeded = good(tmp_path)
    derive_and_write(run, stage)
    assert has(check(run, stage, [])[0], "holds no record of the files copied")
    assert has(check(run, stage, None)[0], "holds no record of the files copied")


def test_a_port_that_differs_from_the_source_stages_record_is_refused(tmp_path):
    run, stage, snap, seeded = make_run(tmp_path, source_port_src=HONEST + "\n# the port stage 2 measured was another file\n")
    assert run_template(tmp_path, run, stage).returncode == 0
    derive_and_write(run, stage)
    assert has(check(run, stage, seeded)[0], "not the ones stage 2's own record names")


def test_a_port_from_stage_3_is_compared_with_stage_3s_record(tmp_path):
    run, stage, snap, seeded = make_run(tmp_path, source=3)
    assert run_template(tmp_path, run, stage).returncode == 0
    derive_and_write(run, stage, source=3)
    assert check(run, stage, seeded, source_stage=3)[0] == []
    assert has(check(run, stage, seeded, source_stage=2)[0], "stage 2")        # stage 2 has no record in this run


def test_a_model_source_that_reads_the_reference_is_refused(tmp_path):
    run, stage, seeded = good(tmp_path, port=READS_REFERENCE, source_port_src=READS_REFERENCE)
    derive_and_write(run, stage)
    assert has(check(run, stage, seeded)[0], "refers to the stage 1 reference files")


def test_a_link_in_the_port_directory_fails_the_template(tmp_path):
    run, stage, snap, seeded = make_run(tmp_path)
    (stage / "port" / "extra.py").symlink_to("port_model.py")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "is a link" in done.stdout


def test_an_unknown_source_stage_is_refused(tmp_path):
    run, stage, seeded = good(tmp_path)
    assert has(check(run, stage, seeded, source_stage=4)[0], "source_stage must be one of")


# ---- the helpers -------------------------------------------------------------------------------------------------------------------

def test_leading_match_counts_until_the_first_difference():
    assert fq.leading_match([1, 2, 3, 4], [1, 2, 9, 4]) == 2 and fq.leading_match([1, 2], [1, 2, 3]) == 2 and fq.leading_match([1], [2]) == 0


def test_the_gap_difference_ignores_a_shift_of_all_logits():
    top5 = [{"ids": [1, 2, 3, 4, 5], "logits": [3.0, 2.0, 1.0, 0.0, -1.0]}]
    assert fq.top5_gap_diff(top5, [[103.0, 102.0, 101.0, 100.0, 99.0]]) == pytest.approx(0.0)
    assert fq.top5_gap_diff(top5, [[3.0, 2.0, 1.0, 0.0, 1.0]]) == pytest.approx(0.5)       # one of four gaps is off by 2


def test_a_figure_with_no_value_is_a_todo():
    rows = fq.numbers({"leading_match": 3}, {"perplexity": None, "mean_nll": 1.0, "top1_agreement": 1.0, "top5_gap_diff": 0.1},
                      {"decode_tok_s": None, "prefill_s": None, "share_of_floor_median": None, "load_s": 2.0})
    by = {r["name"]: r for r in rows}
    assert by["teacher-forced perplexity of the reference text"]["label"] == "TODO" and by["decode speed, median"]["label"] == "TODO"
    assert by["model load time"]["label"] == "measured" and by["model load time"]["value"] == 2.0
