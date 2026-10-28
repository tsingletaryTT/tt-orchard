# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 146: the gate's re-check of the audio class decision (orchard/audio_decision.py) after the review of PR 144.

An agent that can edit `delta.json` must not be able to turn a class its own evidence does not support into weights-only. Each test here edits a
delta.json that the shipped template wrote (so the stamp and the evidence are the template's) and expects the gate to refuse the edit.

  1. a delta stamped by the current template has to carry a decision;
  2. the decision's strict flag has to be the triage configuration's;
  3. each difference's category has to be what the allow-list gives for its key and values, or be recorded as an override;
  4. a path override has to be a single one, go to the decision's path, and carry its source;
  5. weights-only is refused when the comparison evidence holds a tensor or text-encoder difference or a read problem, even with an override;
  6. (template) a dangling audio_overrides.json link counts as an unreadable file;
  7. (template) a config key that holds a dot cannot pass for a nested key."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from orchard import audio_decision as ad
from test_audio_class_inference import (TABLE, UNKNOWN_KEY_CFG, decision, delta_of, edit_delta, gate, gate_reasons, serving_pair, strict, tm,  # noqa: F401
                                        write_overrides)
from test_delta_triage_audio import (TENSORS, audio_case, changed_model_config, model_snapshot, reasons, run_ok)  # noqa: F401
from test_delta_triage_template import stage_for, template_module  # noqa: F401


def to_weights_only(d):
    """The edit an agent would make: both class and path pairs set to weights-only."""
    d.update(path="weights-only", **{"class": "weights-only"})
    d["audio"]["decision"].update(path="weights-only", alternative_path=None, **{"class": "weights-only"})


def refused(stage, *needles):
    r = gate(stage)
    assert not r.ok, "the gate accepted the edited delta.json"
    text = " | ".join(r.reasons)
    for n in needles:
        assert n in text, (n, r.reasons)


# ---- 1. a decision is required from the current template ---------------------------------------------------------------------------

def test_deleting_the_decision_and_changing_class_and_path_together_is_refused(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(model__diffusion__config__depth=24)})        # a full port
    assert run_ok(stage)["class"] == "full-port"
    edit_delta(stage, lambda d: (d["audio"].pop("decision"), d.update(path="weights-only", **{"class": "weights-only"})))
    refused(stage, "no audio.decision")


def test_a_serving_only_delta_with_its_decision_deleted_is_refused_too(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"].pop("decision"))
    refused(stage, "no audio.decision")


def test_a_decision_set_to_null_is_refused(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"].update(decision=None))
    assert not gate(stage).ok


def test_an_audio_record_from_an_older_template_with_no_decision_has_nothing_to_check():
    """The legacy exception: before the decision existed no delta.json had one. The check applies to the current template's stamp only."""
    assert ad.decision_problems({"audio": {}}) == []
    assert ad.decision_problems({"audio": {}}, current_template=False) == []
    assert ad.decision_problems({"facts": {"modality": "audio"}}) == []
    assert ad.decision_problems({"audio": {}}, current_template=True)
    assert ad.decision_problems({}, current_template=True, audio_evidence=True)       # the stage's own files say it was an audio comparison
    assert ad.decision_problems({"facts": {"modality": "audio"}}, current_template=True) == []       # the agent can delete this field with the record


def test_deleting_the_audio_record_and_the_facts_does_not_hide_an_audio_comparison(audio_case):
    """The evidence file the audio branch wrote is still in the stage directory."""
    stage = audio_case(base_kw={"config": changed_model_config(model__diffusion__config__depth=24)})
    assert run_ok(stage)["class"] == "full-port"
    edit_delta(stage, lambda d: (d.pop("audio"), d.pop("facts"), d.update(path="weights-only", **{"class": "weights-only"})))
    refused(stage, "no audio.decision")


def test_the_audio_layout_of_the_configured_snapshot_shows_an_audio_comparison_without_the_evidence_file(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(model__diffusion__config__depth=24)})
    d = run_ok(stage)
    (stage / "evidence" / "audio-compare.json").unlink()
    d.pop("audio"), d.pop("facts")
    shipped = tm().THIS_SHA256
    assert ad.stage_problems(d, stage, shipped)
    assert any("no audio.decision" in p for p in ad.stage_problems(d, stage, shipped))


def test_the_audio_evidence_file_shows_an_audio_comparison_whatever_the_configured_snapshot_looks_like(tmp_path):
    stage = tmp_path / "stage"
    (stage / "evidence").mkdir(parents=True)
    (stage / "evidence" / "audio-compare.json").write_text("{}")
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (stage / "triage_config.json").write_text(json.dumps({"model_snapshot": str(snap)}))
    assert any("no audio.decision" in p for p in ad.stage_problems({"template_sha256": "x" * 64}, stage, "x" * 64))


def test_a_snapshot_with_both_config_files_is_a_text_model_and_shows_no_audio_comparison(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (snap / "model_config.json").write_text("{}")
    (stage / "triage_config.json").write_text(json.dumps({"model_snapshot": str(snap)}))
    assert ad.stage_problems({"template_sha256": "x" * 64}, stage, "x" * 64) == []


def test_a_text_comparison_of_the_current_template_is_not_mistaken_for_audio(tmp_path):
    stage = tmp_path / "stage"
    (stage / "evidence").mkdir(parents=True)
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (stage / "triage_config.json").write_text(json.dumps({"model_snapshot": str(snap)}))
    assert ad.stage_problems({"template_sha256": "x" * 64, "class": "full-port"}, stage, "x" * 64) == []


@pytest.mark.parametrize("delta", [{}, {"facts": {"modality": "text"}}, {"facts": 3}, {"class": "full-port"}])
def test_a_text_delta_from_the_current_template_needs_no_decision(delta):
    assert ad.decision_problems(delta, current_template=True) == []


# ---- 2. the strict flag is the configuration's -------------------------------------------------------------------------------------

def test_a_strict_configuration_cannot_be_turned_off_by_editing_the_decision(audio_case):
    stage = serving_pair(audio_case)
    strict(stage)                                      # the rule: any difference is a full port
    d = run_ok(stage)
    assert d["class"] == "full-port"
    def edit(d):
        to_weights_only(d)
        d["audio"]["decision"]["strict"] = False
    edit_delta(stage, edit)
    refused(stage, "strict")


def test_a_relaxed_configuration_cannot_be_claimed_strict(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"]["decision"].update(strict=True))
    refused(stage, "strict")


@pytest.mark.parametrize("value", ["yes", 1, [], {}])
def test_a_configuration_with_a_flag_that_is_not_a_boolean_means_relaxed_in_the_record(audio_case, value):
    """The template records strict false and class unknown for such a flag. The gate holds the record to that."""
    stage = serving_pair(audio_case)
    strict(stage, value)
    d = run_ok(stage)
    assert d["class"] == "unknown" and decision(d)["strict"] is False
    assert gate(stage).ok


def test_a_triage_configuration_that_cannot_be_read_is_refused_for_an_audio_delta(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    (stage / "triage_config.json").unlink()
    refused(stage, "triage_config.json")


def test_the_flag_is_not_compared_when_no_configuration_is_given():
    dec = {"class": "weights-only", "path": "weights-only", "alternative_path": "full-port", "confidence": "medium", "strict": True,
           "differences": [], "overrides": []}
    delta = {"class": "weights-only", "path": "weights-only", "audio": {"decision": dec}}
    assert ad.decision_problems(delta) == []
    assert any("strict" in p for p in ad.decision_problems(delta, expected_strict=False))
    assert ad.decision_problems(delta, expected_strict=True) == []


# ---- 3. a category has to follow from the key and the values ---------------------------------------------------------------------

def test_a_structural_difference_recorded_as_serving_with_class_and_path_changed_is_refused(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(sample_rate=48000)})
    assert run_ok(stage)["class"] == "full-port"
    def edit(d):
        to_weights_only(d)
        [diff] = d["audio"]["decision"]["differences"]
        diff["category"] = "serving"
    edit_delta(stage, edit)
    refused(stage, "sample_rate", "allow-list")


def test_an_unknown_key_recorded_as_serving_is_refused_without_an_override(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(**UNKNOWN_KEY_CFG)})
    run_ok(stage)
    def edit(d):
        to_weights_only(d)
        d["audio"]["decision"]["differences"][0]["category"] = "serving"
    edit_delta(stage, edit)
    refused(stage, "model.extra.a", "allow-list")


def test_a_serving_key_whose_values_the_rule_does_not_allow_cannot_be_recorded_as_serving(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(model__diffusion__diffusion_objective="v")})
    assert decision(run_ok(stage))["differences"][0]["category"] == "unknown"
    def edit(d):
        to_weights_only(d)
        d["audio"]["decision"]["differences"][0]["category"] = "serving"
    edit_delta(stage, edit)
    refused(stage, "diffusion_objective")


def real_override_stage(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(**UNKNOWN_KEY_CFG)})
    write_overrides(stage, {"categories": [{"key": "model.extra.a", "category": "serving", "source": "read foo.py: only a default"}]})
    assert run_ok(stage)["class"] == "weights-only" and gate(stage).ok
    return stage


def test_a_category_set_by_a_real_override_is_accepted(audio_case):
    real_override_stage(audio_case)


@pytest.mark.parametrize("edit,needle", [
    (lambda dec: dec["overrides"].clear(), "override"),                                                    # the difference says override, no record does
    (lambda dec: dec["overrides"][0].update(category="structural"), "override"),                          # the record names another category
    (lambda dec: dec["overrides"][0].update(key="model.other"), "override"),
    (lambda dec: dec["overrides"][0].update(source="another source"), "source"),
    (lambda dec: dec["overrides"][0].update(**{"from": "structural"}), "from"),                           # the allow-list says unknown
    (lambda dec: dec["overrides"].append(copy.deepcopy(dec["overrides"][0])), "override"),               # the same key overridden twice
    (lambda dec: dec["overrides"][0].pop("key"), "key"),
    (lambda dec: dec["overrides"][0].pop("category"), "category"),
])
def test_a_category_override_record_has_to_match_the_difference_it_explains(audio_case, edit, needle):
    stage = real_override_stage(audio_case)
    edit_delta(stage, lambda d: edit(d["audio"]["decision"]))
    refused(stage, needle)


def test_a_difference_claiming_an_override_with_no_matching_record_is_refused_even_with_a_path_override_present(audio_case):
    stage = real_override_stage(audio_case)
    def edit(d):
        dec = d["audio"]["decision"]
        dec["overrides"] = [{"kind": "path", "path": "weights-only", "from": "full-port", "source": "forged"}]
    edit_delta(stage, edit)
    assert not gate(stage).ok


def test_a_key_that_is_not_in_the_comparison_evidence_is_refused(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    def edit(d):
        d["audio"]["decision"]["differences"].append({"key": "sample_size_extra", "base": 1, "model": 2, "category": "unknown", "decided_by": "allow-list"})
    edit_delta(stage, edit)
    refused(stage, "sample_size_extra", "evidence")


def test_a_difference_the_evidence_lists_cannot_be_dropped_from_the_decision(audio_case):
    cfg = changed_model_config(sample_size=5324800, model__diffusion__config__depth=24)
    stage = audio_case(base_kw={"config": cfg})
    assert run_ok(stage)["class"] == "full-port"
    def edit(d):
        to_weights_only(d)
        dec = d["audio"]["decision"]
        dec["differences"] = [x for x in dec["differences"] if x["key"] == "sample_size"]        # the structural one is gone
    edit_delta(stage, edit)
    refused(stage, "model.diffusion.config.depth", "evidence")


def test_a_recorded_value_that_is_not_the_evidence_s_is_refused(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"]["decision"]["differences"][0].update(model=123456))
    refused(stage, "evidence")


def test_two_differences_with_one_key_are_refused(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"]["decision"]["differences"].append(copy.deepcopy(d["audio"]["decision"]["differences"][0])))
    refused(stage, "twice")


def test_a_difference_with_no_values_is_refused():
    dec = {"class": "weights-only", "path": "weights-only", "alternative_path": "full-port", "confidence": "medium", "strict": False,
           "differences": [{"key": "sample_size", "category": "serving", "decided_by": "allow-list"}], "overrides": []}
    assert any("base" in p and "model" in p for p in ad.decision_problems({"class": "weights-only", "path": "weights-only", "audio": {"decision": dec}}))


# ---- 4. a path override has to be one, to the decision's path, with a source ------------------------------------------------------

def structural_pair_with_override(audio_case, path):
    stage = audio_case(base_kw={"config": changed_model_config(model__diffusion__config__depth=24)})
    write_overrides(stage, {"path": {"path": path, "source": "operator: reason"}})
    return stage


def test_a_real_path_override_to_weights_only_is_accepted(audio_case):
    stage = structural_pair_with_override(audio_case, "weights-only")
    assert run_ok(stage)["path"] == "weights-only" and gate(stage).ok, gate(stage).reasons


def test_a_full_port_override_cannot_stay_in_place_while_both_pairs_say_weights_only(audio_case):
    stage = structural_pair_with_override(audio_case, "full-port")
    assert run_ok(stage)["class"] == "full-port"
    edit_delta(stage, to_weights_only)
    refused(stage, "override")


def test_a_path_override_with_no_path_is_refused(audio_case):
    stage = structural_pair_with_override(audio_case, "weights-only")
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"]["decision"]["overrides"][0].pop("path"))
    refused(stage, "path")


def test_a_path_override_to_a_path_that_is_not_the_decisions_is_refused(audio_case):
    stage = structural_pair_with_override(audio_case, "weights-only")
    run_ok(stage)
    def edit(d):
        d["audio"]["decision"]["overrides"][0]["path"] = "full-port"
    edit_delta(stage, edit)
    refused(stage, "override")


def test_two_path_overrides_are_refused(audio_case):
    stage = structural_pair_with_override(audio_case, "weights-only")
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"]["decision"]["overrides"].append(copy.deepcopy(d["audio"]["decision"]["overrides"][0])))
    refused(stage, "one path override")


def test_a_full_port_decision_with_a_weights_only_override_record_is_refused(audio_case):
    """An override that says weights-only cannot sit under a decision that went the other way: the record does not explain the decision."""
    stage = structural_pair_with_override(audio_case, "full-port")
    assert run_ok(stage)["class"] == "full-port"
    edit_delta(stage, lambda d: d["audio"]["decision"]["overrides"][0].update(path="weights-only"))
    refused(stage, "moves the path to weights-only")


def test_a_path_override_cannot_claim_an_origin_the_evidence_contradicts(audio_case):
    """Review of PR 191: the pre-override path is recomputed from strictness, the recorded categories and the hard evidence, and `from` has to match."""
    stage = structural_pair_with_override(audio_case, "weights-only")          # structural difference: the rule gave full-port
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"]["decision"]["overrides"][0].update(**{"from": "weights-only"}))
    refused(stage, "before the override")


def test_a_full_port_override_over_serving_differences_has_to_come_from_weights_only(audio_case):
    stage = serving_pair(audio_case)
    write_overrides(stage, {"path": {"path": "full-port", "source": "reason"}})
    assert run_ok(stage)["class"] == "full-port" and gate(stage).ok
    edit_delta(stage, lambda d: d["audio"]["decision"]["overrides"][0].update(**{"from": "full-port"}))
    refused(stage, "before the override")


def test_hard_evidence_makes_the_path_before_a_full_port_override_full_port_as_well(audio_case):
    """A tensor difference beside serving-only config differences: the rule gave full-port, so the override's `from` is full-port."""
    cfg = changed_model_config(sample_size=5324800)
    t = {**TENSORS, "model.model.layers.0.attn.weight": ("F32", [9, 8])}
    stage = audio_case(model_kw={"tensors": t}, base_kw={"config": cfg})
    write_overrides(stage, {"path": {"path": "full-port", "source": "reason"}})
    d = run_ok(stage)
    assert decision(d)["overrides"][0]["from"] == "full-port" and gate(stage).ok, gate(stage).reasons
    edit_delta(stage, lambda d: d["audio"]["decision"]["overrides"][0].update(**{"from": "weights-only"}))
    refused(stage, "before the override")


def test_the_strict_rule_changes_where_a_path_override_comes_from(audio_case):
    stage = serving_pair(audio_case)
    strict(stage)
    write_overrides(stage, {"path": {"path": "weights-only", "source": "reason"}})
    d = run_ok(stage)
    assert d["path"] == "weights-only" and decision(d)["overrides"][0]["from"] == "full-port" and gate(stage).ok, gate(stage).reasons


def test_a_path_override_with_an_unknown_path_is_refused(audio_case):
    stage = structural_pair_with_override(audio_case, "weights-only")
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"]["decision"]["overrides"][0].update(path="somewhere"))
    refused(stage, "path")


def test_a_path_override_without_a_from_is_refused(audio_case):
    stage = structural_pair_with_override(audio_case, "weights-only")
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"]["decision"]["overrides"][0].pop("from"))
    refused(stage, "from")


def test_a_weights_only_decision_with_a_structural_difference_and_no_override_is_still_refused(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(model__diffusion__config__depth=24)})
    run_ok(stage)
    edit_delta(stage, to_weights_only)
    refused(stage, "weights-only")


# ---- 5. weights-only needs the evidence to hold no hard difference or read problem ---------------------------------------------------

def shape_change():
    return {**TENSORS, "model.model.layers.0.attn.weight": ("F32", [8, 9])}


def test_a_different_tensor_shape_cannot_be_turned_into_weights_only_by_editing_the_pairs(audio_case):
    stage = audio_case(model_kw={"tensors": shape_change()})
    assert run_ok(stage)["class"] == "full-port"
    edit_delta(stage, to_weights_only)
    refused(stage, "tensor")


def test_a_tensor_that_has_another_shape_cannot_be_turned_into_weights_only(audio_case):
    """The shape of a tensor both models have: the added-tensor cases above leave this list empty."""
    t = {**TENSORS, "model.model.layers.0.attn.weight": ("F32", [8, 9])}
    del t["model.model.layers.0.attn.weight"]
    t["model.model.layers.0.attn.weight"] = ("F32", [9, 8])
    stage = audio_case(model_kw={"tensors": t})
    d = run_ok(stage)
    assert d["class"] == "full-port" and d["audio"]["tensors"]["shape_differences"] and not d["audio"]["tensors"]["only_in_model"]
    edit_delta(stage, to_weights_only)
    refused(stage, "shape_differences")


def test_a_tensor_with_another_integer_dtype_cannot_be_turned_into_weights_only(audio_case):
    t = {**TENSORS, "model.model.layers.0.attn.weight": ("I64", [8, 8])}
    stage = audio_case(model_kw={"tensors": t})
    assert run_ok(stage)["class"] == "full-port"
    edit_delta(stage, to_weights_only)
    refused(stage, "dtype_differences")


def test_a_cut_dtype_list_still_shows_an_incompatible_dtype_because_the_template_lists_it_first():
    """Review of PR 191: 51 tolerated differences in name order used to push a later incompatible one out of the 50 the record lists."""
    t = tm()
    mt = {f"a{i:03d}": {"dtype": "F32", "shape": [2]} for i in range(60)}
    bt = {name: {"dtype": "BF16", "shape": [2]} for name in mt}
    mt["z999"], bt["z999"] = {"dtype": "F32", "shape": [2]}, {"dtype": "I64", "shape": [2]}
    listed = t.compare_audio_tensors(mt, bt)["dtype_differences"][:t.AUDIO_MAX_LISTED]
    assert len(listed) == t.AUDIO_MAX_LISTED and listed[0]["name"] == "z999" and listed[0]["tolerated"] is False
    tensors = {"model_tensors": 61, "base_tensors": 61, "common": 61, "only_in_model": [], "only_in_base": [], "shape_differences": [],
               "dtype_differences": listed}
    assert "dtype_differences" in ad._tensor_findings(tensors)


def test_zero_tensors_on_every_side_is_not_a_comparison(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    def edit(d):
        d["audio"]["tensors"].update(model_tensors=0, base_tensors=0, common=0)
    edit_delta(stage, edit)
    refused(stage, "tensor counts")


@pytest.mark.parametrize("counts", [(0, 0, 0), (-1, -1, -1), (5, 5, 4), (5, 4, 5), (True, True, True), (5.0, 5.0, 5.0), (None, None, None)])
def test_tensor_counts_have_to_be_equal_and_positive_whole_numbers(counts):
    tensors = {"model_tensors": counts[0], "base_tensors": counts[1], "common": counts[2], "only_in_model": [], "only_in_base": [],
               "shape_differences": [], "dtype_differences": []}
    assert "tensor counts" in ad._tensor_findings(tensors)


def test_a_tolerated_float_dtype_difference_keeps_a_weights_only_decision_acceptable(audio_case):
    """The template tolerates a small-float dtype difference (bfloat16 against float32) and the gate does the same."""
    t = {**TENSORS, "model.model.layers.0.attn.weight": ("BF16", [8, 8])}
    stage = audio_case(model_kw={"tensors": t})
    d = run_ok(stage)
    assert d["class"] == "weights-only" and d["audio"]["tensors"]["dtype_differences"][0]["tolerated"] is True
    assert gate(stage).ok, gate(stage).reasons


def test_a_tensor_record_whose_lists_were_emptied_but_whose_counts_still_differ_is_refused(audio_case):
    stage = audio_case(model_kw={"tensors": {**TENSORS, "model.model.extra.weight": ("F32", [2, 2])}})
    run_ok(stage)
    def edit(d):
        to_weights_only(d)
        for k in ("only_in_model", "only_in_base", "shape_differences", "dtype_differences"):
            d["audio"]["tensors"][k] = []
    edit_delta(stage, edit)
    refused(stage, "tensor counts")


def test_a_forged_weights_only_path_override_does_not_explain_a_tensor_difference(audio_case):
    stage = audio_case(model_kw={"tensors": shape_change()})
    run_ok(stage)
    def edit(d):
        to_weights_only(d)
        d["audio"]["decision"]["overrides"] = [{"kind": "path", "path": "weights-only", "from": "full-port", "source": "forged"}]
    edit_delta(stage, edit)
    refused(stage, "tensor")


def test_a_text_encoder_difference_cannot_be_turned_into_weights_only(audio_case):
    stage = audio_case(model_kw={"text_config": {"architectures": ["x"]}})
    assert run_ok(stage)["class"] == "full-port"
    edit_delta(stage, to_weights_only)
    refused(stage, "text encoder")


def test_a_tensor_present_in_one_model_only_cannot_be_turned_into_weights_only(audio_case):
    t = {**TENSORS, "model.model.extra.weight": ("F32", [2, 2])}
    stage = audio_case(model_kw={"tensors": t})
    run_ok(stage)
    edit_delta(stage, to_weights_only)
    refused(stage, "tensor")


def test_unreadable_evidence_cannot_be_turned_into_weights_only(audio_case):
    stage = audio_case()
    (model_snapshot(stage) / "model.safetensors").write_bytes(b"\xff" * 16)
    assert run_ok(stage)["class"] == "unknown"
    edit_delta(stage, to_weights_only)
    refused(stage, "could not be read")


def test_a_decision_problem_list_that_is_not_empty_blocks_weights_only_even_when_the_counts_look_equal(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"]["decision"].update(problems=["a file could not be read"]))
    refused(stage, "problems")


def test_the_evidence_records_must_be_there_for_the_current_template(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"].pop("tensors"))
    refused(stage, "audio.tensors")


def test_the_text_encoder_record_must_be_there_for_the_current_template(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"].pop("text_encoders"))
    refused(stage, "text_encoders")


def test_a_decision_without_a_weights_only_class_is_not_held_to_the_evidence_check(audio_case):
    stage = audio_case(model_kw={"tensors": shape_change()})
    run_ok(stage)
    assert gate(stage).ok


# ---- 6. a dangling override link is an unreadable file --------------------------------------------------------------------------------

def test_a_dangling_override_link_is_an_unreadable_file_and_the_class_is_unknown(audio_case, tmp_path):
    stage = serving_pair(audio_case)
    (stage / "audio_overrides.json").symlink_to(tmp_path / "nowhere.json")
    d = run_ok(stage)
    assert d["class"] == "unknown" and "audio_overrides.json" in reasons(d)
    assert gate(stage).ok


def test_a_link_loop_in_place_of_the_override_file_is_unknown(audio_case):
    stage = serving_pair(audio_case)
    os.symlink("audio_overrides.json", stage / "audio_overrides.json")
    assert run_ok(stage)["class"] == "unknown"


def test_a_link_to_a_good_override_file_is_read(audio_case, tmp_path):
    stage = audio_case(base_kw={"config": changed_model_config(**UNKNOWN_KEY_CFG)})
    real = tmp_path / "blob"
    real.write_text(json.dumps({"categories": [{"key": "model.extra.a", "category": "serving", "source": "s"}]}))
    (stage / "audio_overrides.json").symlink_to(real)
    assert run_ok(stage)["class"] == "weights-only"


def test_a_directory_in_place_of_the_override_file_is_unknown(audio_case):
    stage = serving_pair(audio_case)
    (stage / "audio_overrides.json").mkdir()
    assert run_ok(stage)["class"] == "unknown"


# ---- 7. a dotted key is not a nested key -----------------------------------------------------------------------------------------------

def with_top_key(cfg, key, value):
    cfg = copy.deepcopy(cfg)
    cfg[key] = value
    return cfg


def test_a_top_level_key_named_like_a_serving_setting_is_not_that_setting(audio_case):
    base = with_top_key(changed_model_config(), "model.sampler_type", "euler")
    model = with_top_key(changed_model_config(), "model.sampler_type", "pingpong")
    d = run_ok(audio_case(model_kw={"config": model}, base_kw={"config": base}))
    assert d["class"] == "full-port"
    [diff] = decision(d)["differences"]
    assert diff["category"] == "unknown" and diff["key"] != "model.sampler_type"


def test_the_nested_setting_and_the_top_level_key_of_the_same_text_are_both_kept(audio_case):
    base, model = changed_model_config(), changed_model_config()
    base["model"]["sampler_type"], model["model"]["sampler_type"] = "euler", "pingpong"
    base["model.sampler_type"], model["model.sampler_type"] = "a", "b"
    d = run_ok(audio_case(model_kw={"config": model}, base_kw={"config": base}))
    diffs = {x["key"]: x for x in decision(d)["differences"]}
    assert len(diffs) == 2 and "model.sampler_type" in diffs and diffs["model.sampler_type"]["category"] == "serving"
    assert [x["category"] for k, x in diffs.items() if k != "model.sampler_type"] == ["unknown"]
    assert d["class"] == "full-port"
    config = d["audio"]["config"]
    assert len({**config["top_level_differences"], **config["model_differences"]}) == 2           # neither record replaced the other


def test_a_dotted_key_inside_the_model_section_is_one_quoted_component_after_the_section_name(audio_case):
    base, model = changed_model_config(), changed_model_config()
    base["model"]["diffusion.sampler_type"], model["model"]["diffusion.sampler_type"] = "euler", "pingpong"
    d = run_ok(audio_case(model_kw={"config": model}, base_kw={"config": base}))
    [diff] = decision(d)["differences"]
    assert diff["key"] == 'model["diffusion.sampler_type"]' and diff["category"] == "unknown" and d["class"] == "full-port"


def test_the_nested_setting_alone_is_still_a_serving_difference(audio_case):
    base, model = changed_model_config(), changed_model_config()
    base["model"]["sampler_type"], model["model"]["sampler_type"] = "euler", "pingpong"
    d = run_ok(audio_case(model_kw={"config": model}, base_kw={"config": base}))
    assert d["class"] == "weights-only" and [x["key"] for x in decision(d)["differences"]] == ["model.sampler_type"]


@pytest.mark.parametrize("key", ["a.b", "a[0]", "a]", "", 'a"b', "a\\b", "model.diffusion.config.depth"])
def test_a_key_with_a_dot_bracket_or_quote_is_written_as_one_quoted_component(key):
    [path] = tm()._audio_flatten({key: 1})
    assert path == "[" + json.dumps(key) + "]"


def test_flat_paths_of_plain_nested_keys_keep_their_form():
    t = tm()
    assert t._audio_flatten({"a": {"b": [1, {"c": 2}]}, "d": 3}) == {"a.b[0]": 1, "a.b[1].c": 2, "d": 3}


def test_two_different_paths_never_flatten_to_one_text():
    t = tm()
    docs = [{"a.b": 1}, {"a": {"b": 1}}, {"a": [{"b": 1}]}, {"a[0].b": 1}, {"a": {"b.c": 1}}, {"a.b": {"c": 1}}, {"a": {"b": {"c": 1}}}, {"": 1}, {"": {"": 1}}]
    seen = {}
    for doc in docs:
        [path] = t._audio_flatten(doc)
        assert path not in seen, (doc, seen[path])
        seen[path] = doc


# ---- the gate's copy of the rule is the template's rule ---------------------------------------------------------------------------------

MATRIX = [("sample_size", 5324800, 5292032), ("sample_size", 0, 5), ("sample_size", True, 5), ("sample_size", "<absent>", 6), ("sample_size", 5.5, 6),
          ("model.diffusion.diffusion_objective", "rectified_flow", "rf_denoiser"), ("model.diffusion.diffusion_objective", "v", "rf_denoiser"),
          ("model.diffusion.diffusion_objective", 3, "rf_denoiser"), ("model.sampler_type", "euler", "pingpong"), ("model.sampler_type", "", "pingpong"),
          ("model.default_steps", 50, 8), ("model.default_steps", 0, 8), ("model.default_cfg_scale", 7, 1), ("model.default_cfg_scale", True, 1.0),
          ("model.default_cfg_scale", 7.5, "<absent>"), ("sample_rate", 44100, 48000), ("audio_channels", 2, 1), ("model.io_channels", 1, 2),
          ("model.diffusion.config.depth", 20, 24), ("model.pretransform.config.x", 1, 2), ("model.conditioning.configs[0].type", "a", "b"),
          ("model.extra.a", "<absent>", 1), ("something_else", 1, 2), ("model.diffusion.config", 1, 2), ('["model.sampler_type"]', "a", "b")]


@pytest.mark.parametrize("key,base,model", MATRIX)
def test_the_gates_allow_list_gives_the_templates_category(key, base, model):
    rules = ad.category_rules()
    assert rules is not None
    assert ad.allow_list_category(key, base, model, rules) == tm().audio_category(key, base, model)[0]


@pytest.mark.parametrize("big", [10 ** 400, -(10 ** 400), 10 ** 30, 1e308, 1e-320])
def test_an_integer_too_large_for_a_float_is_a_number_and_never_a_crash(big):
    """Review of PR 191: math.isfinite raised OverflowError on a JSON integer of 400 digits, which the gate loads without the template's digit limit."""
    rules = ad.category_rules()
    assert ad.allow_list_category("model.default_cfg_scale", big, 1.0, rules) == "serving"
    if isinstance(big, int) and big > 0:
        assert ad.allow_list_category("sample_size", big, 5, rules) == "serving"


@pytest.mark.parametrize("value", [10 ** 400, -(10 ** 400), 1e308, [], {}, None, True, "x" * 100000, float("nan")])
@pytest.mark.parametrize("field", ["base", "model"])
def test_a_hostile_value_in_a_difference_never_raises(value, field):
    diff = {"key": "model.default_cfg_scale", "base": 7.0, "model": 1.0, "category": "serving", "decided_by": "allow-list"}
    diff[field] = value
    dec = {"class": "weights-only", "path": "weights-only", "alternative_path": "full-port", "confidence": "medium", "strict": False,
           "differences": [diff], "overrides": [], "problems": []}
    audio = {"decision": dec, "config": {"top_level_differences": {}, "model_differences": {"model.default_cfg_scale": {"base": diff["base"], "model": diff["model"]}}},
             "tensors": {"model_tensors": 1, "base_tensors": 1, "common": 1, "only_in_model": [], "only_in_base": [], "shape_differences": [], "dtype_differences": []},
             "text_encoders": []}
    problems = ad.decision_problems({"class": "weights-only", "path": "weights-only", "audio": audio}, current_template=True)
    assert isinstance(problems, list)


def test_a_decision_with_the_huge_integer_in_the_delta_file_returns_gate_feedback(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(model__default_cfg_scale=7.0)})
    stage_cfg = changed_model_config(model__default_cfg_scale=1.0)
    run_ok(stage)
    def edit(d):
        d["audio"]["decision"]["differences"][0]["model"] = 10 ** 400
    edit_delta(stage, edit)
    r = gate(stage)                                  # must return a result, not raise
    assert not r.ok


def test_the_gates_table_is_the_templates_table():
    assert list(ad.category_rules()) == tm().AUDIO_CATEGORY_RULES


def test_the_evidence_cut_the_gate_assumes_is_the_templates():
    assert ad.AUDIO_MAX_LISTED == tm().AUDIO_MAX_LISTED and ad.ABSENT == tm().ABSENT


@pytest.mark.parametrize("body", ["not json", "[]", '{"rules": 3}', '{"rules": [1]}', '{"rules": [{"key": "a"}]}', '{"rules": [{"category": "serving"}]}',
                                  '{"rules": [{"key": "a", "category": "mystery"}]}', '{"rules": NaN}', ""])
def test_a_table_that_is_not_a_list_of_rules_is_refused(tmp_path, body):
    from orchard import safeio
    p = tmp_path / "table.json"
    p.write_text(body)
    with pytest.raises(safeio.SafeReadError):
        ad.load_category_rules(p)


def test_a_decision_is_refused_when_the_table_cannot_be_read(audio_case, monkeypatch):
    stage = serving_pair(audio_case)
    run_ok(stage)
    monkeypatch.setattr(ad, "category_rules", lambda: None)
    refused(stage, "allow-list table")


def test_more_than_fifty_differences_in_a_section_are_held_to_the_first_half_of_the_evidence_check_only():
    """The template cuts each section of audio.config at 50 keys; the decision lists all of them."""
    n = ad.AUDIO_MAX_LISTED
    keys = {f"model.extra.k{i}": {"base": ad.ABSENT, "model": i} for i in range(n)}
    dec_diffs = [{"key": k, "base": v["base"], "model": v["model"], "category": "unknown", "decided_by": "allow-list"} for k, v in keys.items()]
    dec_diffs.append({"key": "model.extra.more", "base": ad.ABSENT, "model": 1, "category": "unknown", "decided_by": "allow-list"})
    dec = {"class": "full-port", "path": "full-port", "alternative_path": None, "confidence": "medium", "strict": False, "differences": dec_diffs,
           "overrides": [], "problems": []}
    audio = {"decision": dec, "config": {"top_level_differences": {}, "model_differences": keys}}
    assert ad.decision_problems({"class": "full-port", "path": "full-port", "audio": audio}, current_template=True) == []
    del dec["differences"][0]                                      # a listed key dropped from the decision
    assert any("lists the difference" in p for p in ad.decision_problems({"class": "full-port", "path": "full-port", "audio": audio}, current_template=True))


def test_an_override_naming_the_flat_text_does_not_match_the_top_level_key(audio_case):
    base = with_top_key(changed_model_config(), "model.default_steps", 50)
    model = with_top_key(changed_model_config(), "model.default_steps", 8)
    stage = audio_case(model_kw={"config": model}, base_kw={"config": base})
    d = run_ok(stage)
    assert d["class"] == "full-port"
    [diff] = decision(d)["differences"]
    assert diff["category"] == "unknown"           # the allow-list rule for model.default_steps is for the nested setting
