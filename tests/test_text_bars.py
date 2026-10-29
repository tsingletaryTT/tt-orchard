# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The text recorder and the chip bar set from CPU-only floors (issue 214)."""
import numpy as np
import pytest

from orchard import audio_reference as ar
from orchard import text_bars as tb
from orchard import text_reference as tr
from text_fixtures import N_DEC, N_PROMPT, base_arrays, load, perturb, write_run


def runs(tmp_path, floor_rel=0.01, chip_rel=0.03, chip_only=None):
    ref_arrays = base_arrays()
    ref = write_run(tmp_path / "ref", ref_arrays)
    floor = write_run(tmp_path / "floor", perturb(ref_arrays, floor_rel, seed=2), dtype="bfloat16")
    chip = write_run(tmp_path / "chip", perturb(ref_arrays, chip_rel, seed=3, only=chip_only), dtype="bfloat16")
    return ref_arrays, load(ref), {"bfloat16": load(floor)}, load(chip)


# ---- the recorder -----------------------------------------------------------------------------------------------------

def test_text_settings_map_to_the_audio_keys():
    s = tr.text_settings("hi", 7, 16, "bfloat16", device="chip 0")
    assert set(ar.MATCH_KEYS) <= set(s)
    assert (s["duration"], s["steps"], s["sampler"], s["cfg_scale"], s["sample_size"]) == (7, 16, "greedy", 0.0, 0)
    assert tr.settings_view(s) == {"prompt": "hi", "prompt_tokens": 7, "decode_tokens": 16, "sampler": "greedy", "dtype": "bfloat16", "device": "chip 0"}


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_text_settings_refuse_a_bad_count(bad):
    with pytest.raises(ar.ReferenceError):
        tr.text_settings("hi", bad, 4, "float32")


def test_make_recorder_writes_a_run_the_loader_reads(tmp_path):
    arrays = base_arrays()
    write_run(tmp_path / "r", arrays)
    run = load(tmp_path / "r")
    assert run.manifest["model"]["name"] == "example/text-model" and len(run.manifest["model"]["config_sha256"]) == 64
    assert run.settings["duration"] == N_PROMPT and run.settings["steps"] == N_DEC
    assert np.allclose(run.array("other", tr.TOKENS), arrays[("other", tr.TOKENS)])


# ---- the rule ---------------------------------------------------------------------------------------------------------------

def test_a_chip_run_a_few_times_the_floor_passes_where_the_default_first_divergence_margin_fails(tmp_path):
    _, ref, floors, chip = runs(tmp_path, floor_rel=0.01, chip_rel=0.05)         # 5 times the floor
    rep = tb.chip_report(ref, floors, chip)
    assert rep["ok"], rep["reasons"]
    assert 3.0 < rep["worst_ratio_to_floor"]["ratio"] < tb.CHIP_MARGIN





def test_a_chip_run_far_above_the_floor_leaves_at_the_first_bad_block(tmp_path):
    ref_arrays, ref, floors, _ = runs(tmp_path)
    bad = perturb(ref_arrays, 0.03, seed=3)
    bad[("layer", "layer_02")] = bad[("layer", "layer_02")] + 0.9 * np.abs(ref_arrays[("layer", "layer_02")])
    chip = load(write_run(tmp_path / "bad", bad, dtype="bfloat16"))
    rep = tb.chip_report(ref, floors, chip)
    assert not rep["ok"] and rep["first_block_leaving"] == "layer_02"
    assert any("layer_02" in r for r in rep["reasons"])


def test_the_operation_that_leaves_is_named(tmp_path):
    ref_arrays, ref, floors, _ = runs(tmp_path)
    bad = perturb(ref_arrays, 0.03, seed=3)
    bad[("layer", "layer_01.attn")] = bad[("layer", "layer_01.attn")] * 3.0
    bad[("layer", "layer_01")] = bad[("layer", "layer_01")] * 3.0
    chip = load(write_run(tmp_path / "bad", bad, dtype="bfloat16"))
    rep = tb.chip_report(ref, floors, chip)
    assert rep["first_block_leaving"] == "layer_01" and rep["first_op_leaving"] == "layer_01.attn"


def test_a_block_the_chip_run_lacks_leaves(tmp_path):
    ref_arrays, ref, floors, _ = runs(tmp_path)
    partial = {k: v for k, v in perturb(ref_arrays, 0.03, seed=3).items() if k[1] != "layer_03"}
    chip = load(write_run(tmp_path / "partial", partial, dtype="bfloat16"))
    rep = tb.chip_report(ref, floors, chip)
    assert not rep["ok"] and rep["first_block_leaving"] == "layer_03"


def test_an_array_no_floor_run_holds_cannot_have_a_bar(tmp_path):
    ref_arrays, ref, _, chip = runs(tmp_path)
    short = {k: v for k, v in perturb(ref_arrays, 0.01, seed=2).items() if k[1] != "layer_01"}
    floors = {"bfloat16": load(write_run(tmp_path / "short", short, dtype="bfloat16"))}
    rep = tb.chip_report(ref, floors, chip)
    assert not rep["ok"] and rep["first_block_leaving"] == "layer_01"
    assert "no floor run" in rep["blocks"][1]["why"]


def test_no_floor_run_is_an_error(tmp_path):
    _, ref, _, chip = runs(tmp_path)
    with pytest.raises(ar.ReferenceError):
        tb.chip_report(ref, {}, chip)


def test_a_zero_floor_is_lifted_so_the_bar_can_be_met_or_missed(tmp_path):
    ref_arrays = base_arrays()
    ref = load(write_run(tmp_path / "ref", ref_arrays))
    floor = {"bfloat16": load(write_run(tmp_path / "floor", ref_arrays, dtype="bfloat16"))}       # identical to the reference: floor 0
    chip = load(write_run(tmp_path / "chip", perturb(ref_arrays, 0.01, seed=3), dtype="bfloat16"))
    rep = tb.chip_report(ref, floor, chip)
    assert rep["blocks"][0]["tolerance"] == pytest.approx(tb.CHIP_MARGIN * tb.MIN_FLOOR)
    assert not rep["ok"]


def test_the_larger_of_two_floor_runs_sets_the_bar(tmp_path):
    ref_arrays, ref, floors, chip = runs(tmp_path, floor_rel=0.005, chip_rel=0.05)
    assert not tb.chip_report(ref, floors, chip)["ok"]                         # 10 times the only floor
    floors["native"] = load(write_run(tmp_path / "native", perturb(ref_arrays, 0.02, seed=9), dtype="bfloat16"))
    assert tb.chip_report(ref, floors, chip)["ok"]                             # 2.5 times the larger floor


# ---- agreement ------------------------------------------------------------------------------------------------------------------

def flip(arrays, name, token):
    a = arrays[("other", name)].copy()
    a[token] += 100
    arrays[("other", name)] = a


def test_agreement_counts_teacher_forced_matches(tmp_path):
    ref_arrays, ref, floors, chip = runs(tmp_path)
    ag = tb.agreement(ref, chip)
    assert (ag["matches"], ag["of"], ag["flips"]) == (N_DEC, N_DEC, [])


def test_flips_below_the_agreement_bar_fail_and_are_listed_with_their_margin(tmp_path):
    ref_arrays, ref, floors, _ = runs(tmp_path)
    bad = perturb(ref_arrays, 0.03, seed=3)
    flip(bad, tr.PREFILL_LOGITS, 20)
    flip(bad, tr.dec_logits(0), 21)
    chip = load(write_run(tmp_path / "bad", bad, dtype="bfloat16"))
    rep = tb.chip_report(ref, floors, chip)
    assert rep["agreement"]["matches"] == 2 and [f["step"] for f in rep["agreement"]["flips"]] == [0, 1]
    assert rep["agreement"]["flips"][0]["reference_margin"] > 0
    assert not rep["ok"] and any("agreement" in r for r in rep["reasons"])


def test_one_flip_in_four_is_below_the_bar_and_the_bar_is_stated():
    assert 3 / 4 < tb.AGREE_MIN_FRACTION <= 1.0


def test_a_chip_run_with_no_decode_logits_has_no_agreement(tmp_path):
    ref_arrays, ref, floors, _ = runs(tmp_path)
    partial = {k: v for k, v in perturb(ref_arrays, 0.03, seed=3).items() if k[1] != tr.dec_logits(1)}
    chip = load(write_run(tmp_path / "partial", partial, dtype="bfloat16"))
    ag = tb.agreement(ref, chip)
    assert ag["fraction"] == 0.0 and "dec_logits_1" in ag["problem"]


# ---- perplexity -----------------------------------------------------------------------------------------------------------------

def ppl(value, tokens=767, lps=None):
    return {"tokens": tokens, "perplexity": value, "mean_nll": float(np.log(value)), "logprobs": lps or [-1.0, -2.0]}


def test_perplexity_within_the_bar_passes():
    r = tb.perplexity_report(ppl(8.0574), ppl(8.0070, lps=[-1.1, -1.9]))
    assert r["ok"] and r["rel_diff"] == pytest.approx(-0.00625, abs=1e-4) and r["mean_abs_logprob_difference"] == pytest.approx(0.1)


def test_perplexity_outside_the_bar_fails():
    r = tb.perplexity_report(ppl(8.0), ppl(8.4))
    assert not r["ok"] and "5" in r["reasons"][0]


def test_perplexity_windows_must_match():
    assert not tb.perplexity_report(ppl(8.0, tokens=767), ppl(8.0, tokens=500))["ok"]


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), None, True])
def test_a_perplexity_that_is_not_a_positive_number_fails(bad):
    assert not tb.perplexity_report(ppl(8.0), {"tokens": 767, "perplexity": bad})["ok"]


# ---- the floor summary ------------------------------------------------------------------------------------------------------------

def test_floor_summary_reports_each_block_and_the_worst(tmp_path):
    _, ref, floors, _ = runs(tmp_path, floor_rel=0.02)
    s = tb.floor_summary(ref, floors)["bfloat16"]
    assert len(s["blocks"]) == 4 and s["worst"] == pytest.approx(max(s["blocks"].values())) and 0.01 < s["median"] < 0.03


def test_the_stated_limits_are_present():
    assert len(tb.CANNOT_SHOW) >= 4 and "chip arithmetic" in tb.CANNOT_SHOW[0]


def test_a_reference_perplexity_of_zero_is_refused_not_divided_by():
    r = tb.perplexity_report({"tokens": 5, "perplexity": 0}, {"tokens": 5, "perplexity": 0})
    assert not r["ok"] and "reference" in r["reasons"][0]
