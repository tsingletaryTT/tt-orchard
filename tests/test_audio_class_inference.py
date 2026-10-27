# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The audio class decision (branch audio-class-inference): a difference in model_config.json is sorted into serving, structural or unknown by an allow-list
table, the audio branch of stage 0 returns a decision object that records the evidence for each difference, and only serving differences leave the path
weights-only. A hook (`audio_overrides.json`) lets a later inference step override a category or the path, and the override is recorded with its source.
The strict behaviour (any difference is a full port) stays reachable with `audio_strict` in triage_config.json."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from orchard import audio_decision
from orchard.stages import gate_delta
from test_delta_triage_audio import (BASE_ID, MODEL_ID, TENSORS, audio_case, changed_model_config, gate, make_audio, model_snapshot, reasons, run_ok,  # noqa: F401
                                     write_safetensors)
from test_delta_triage_template import stage_for, template_module, tokenizer_json  # noqa: F401

TABLE = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "delta-triage-templates" / "audio_config_categories.json"


def tm():
    return template_module()


# ---- the table ------------------------------------------------------------------------------------

def test_the_data_file_and_the_templates_copy_are_the_same_table():
    assert json.loads(TABLE.read_text())["rules"] == tm().AUDIO_CATEGORY_RULES


def test_every_rule_has_a_category_and_every_serving_rule_a_value_constraint_and_a_source():
    for rule in tm().AUDIO_CATEGORY_RULES:
        assert rule["category"] in ("serving", "structural") and bool(rule.get("key")) != bool(rule.get("prefix"))
        assert isinstance(rule.get("why"), str) and rule["why"]
        if rule["category"] == "serving":
            assert rule.get("values") and isinstance(rule.get("source"), str) and rule["source"]


def test_the_serving_allow_list_is_exactly_the_six_keys_the_operator_named_or_the_library_shows():
    serving = {r["key"] for r in tm().AUDIO_CATEGORY_RULES if r["category"] == "serving"}
    assert serving == {"sample_size", "model.diffusion.diffusion_objective", "model.sampler_type", "model.diffusion.sampler_type",
                       "model.default_steps", "model.default_cfg_scale"}


def test_the_objective_rule_cites_the_library_code_it_rests_on():
    rule = next(r for r in tm().AUDIO_CATEGORY_RULES if r.get("key") == "model.diffusion.diffusion_objective")
    assert "dit.py" in rule["source"] and "sampling.py" in rule["source"] and rule["values"] == ["rectified_flow", "rf_denoiser"]


@pytest.mark.parametrize("key,base,model,expected", [
    ("sample_size", 5324800, 5292032, "serving"), ("sample_size", 1, 2, "serving"),
    ("sample_size", 0, 5, "unknown"), ("sample_size", -4, 5, "unknown"), ("sample_size", True, 5, "unknown"), ("sample_size", "5", 6, "unknown"),
    ("sample_size", 5.5, 6, "unknown"), ("sample_size", "<absent>", 6, "unknown"), ("sample_size", 5, "<absent>", "unknown"),
    ("model.diffusion.diffusion_objective", "rectified_flow", "rf_denoiser", "serving"),
    ("model.diffusion.diffusion_objective", "rf_denoiser", "rectified_flow", "serving"),
    ("model.diffusion.diffusion_objective", "v", "rf_denoiser", "unknown"), ("model.diffusion.diffusion_objective", "rectified_flow", "v", "unknown"),
    ("model.diffusion.diffusion_objective", "<absent>", "rf_denoiser", "unknown"), ("model.diffusion.diffusion_objective", 3, "rf_denoiser", "unknown"),
    ("model.sampler_type", "euler", "pingpong", "serving"), ("model.sampler_type", 3, "pingpong", "unknown"), ("model.sampler_type", "", "pingpong", "unknown"),
    ("model.default_steps", 50, 8, "serving"), ("model.default_steps", 0, 8, "unknown"),
    ("model.default_cfg_scale", 7.0, 1.0, "serving"), ("model.default_cfg_scale", 7, 1, "serving"), ("model.default_cfg_scale", True, 1.0, "unknown"),
    ("model.default_cfg_scale", float("nan"), 1.0, "unknown"), ("model.default_cfg_scale", float("inf"), 1.0, "unknown"),
    ("sample_rate", 44100, 48000, "structural"), ("audio_channels", 2, 1, "structural"), ("model_type", "a", "b", "structural"),
    ("model.io_channels", 256, 128, "structural"), ("model.diffusion.config.depth", 20, 24, "structural"),
    ("model.pretransform.config.latent_dim", 1, 2, "structural"), ("model.conditioning.configs[0].type", "a", "b", "structural"),
    ("model.extra.a", "<absent>", 1, "unknown"), ("model.diffusion.type", "dit", "udit", "unknown"), ("something_else", 1, 2, "unknown"),
    ("model.diffusion.diffusion_objective_x", "a", "b", "unknown"), ("model.diffusion.config", 1, 2, "unknown"),
])
def test_each_difference_gets_the_category_the_table_gives(key, base, model, expected):
    category, rule = tm().audio_category(key, base, model)
    assert category == expected
    assert (rule is None) == (expected == "unknown" and not any(r.get("key") == key for r in tm().AUDIO_CATEGORY_RULES))


def test_an_exact_key_wins_over_a_prefix():
    rules = [{"prefix": "model.", "category": "structural", "why": "x"}, {"key": "model.sampler_type", "category": "serving", "values": "text", "why": "y"}]
    assert tm().audio_category("model.sampler_type", "a", "b", rules=rules)[0] == "serving"
    assert tm().audio_category("model.other", "a", "b", rules=rules)[0] == "structural"


# ---- a pair that differs only in serving settings ---------------------------------------------------------

def serving_pair(audio_case, **extra):
    cfg = changed_model_config(sample_size=5324800, model__diffusion__diffusion_objective="rectified_flow")
    return audio_case(base_kw={"config": cfg}, **extra)


def decision(d):
    return d["audio"]["decision"]


def test_only_serving_differences_leave_the_class_and_the_path_weights_only(audio_case):
    stage = serving_pair(audio_case)
    d = run_ok(stage)
    assert d["class"] == "weights-only" and d["path"] == "weights-only" and d["path_reasons"] == []
    assert gate(stage).ok, gate(stage).reasons


def test_the_decision_object_records_class_path_alternative_confidence_and_each_difference(audio_case):
    dec = decision(run_ok(serving_pair(audio_case)))
    assert dec["class"] == "weights-only" and dec["path"] == "weights-only" and dec["alternative_path"] == "full-port"
    assert dec["confidence"] == "medium" and dec["strict"] is False and dec["overrides"] == []
    diffs = {x["key"]: x for x in dec["differences"]}
    assert set(diffs) == {"sample_size", "model.diffusion.diffusion_objective"}
    size = diffs["sample_size"]
    assert (size["base"], size["model"], size["category"], size["decided_by"]) == (5324800, 5292032, "serving", "allow-list")
    assert "model.py" in size["source"]
    obj = diffs["model.diffusion.diffusion_objective"]
    assert (obj["base"], obj["model"]) == ("rectified_flow", "rf_denoiser") and "dit.py" in obj["source"]


def test_a_settings_hazard_names_the_keys_both_values_and_that_no_stage_checks_it(audio_case):
    d = run_ok(serving_pair(audio_case))
    [h] = [h for h in d["hazards"] if "serving" in h["finding"] and "sample_size" in h["finding"]]
    text = h["finding"]
    assert h["area"] == "other"
    for needle in ("sample_size", "5292032", "5324800", "model.diffusion.diffusion_objective", "rf_denoiser", "rectified_flow", "no stage checks"):
        assert needle in text, needle


def test_the_config_difference_names_the_categories(audio_case):
    d = run_ok(serving_pair(audio_case))
    finding = next(x for x in d["differences"] if x["area"] == "config")["finding"]
    assert "2 serving" in finding and "0 structural" in finding and "0 unknown" in finding


def test_a_pair_with_no_difference_is_weights_only_with_high_confidence_and_no_hazard(audio_case):
    d = run_ok(audio_case())
    dec = decision(d)
    assert d["class"] == "weights-only" and dec["confidence"] == "high" and dec["alternative_path"] is None and dec["differences"] == []
    assert not any("sample_size" in h["finding"] for h in d["hazards"])


def test_a_training_only_difference_is_not_a_difference_in_the_decision(audio_case):
    cfg = changed_model_config()
    cfg["training"]["arc"]["disc_update_interval"] = 9
    dec = decision(run_ok(audio_case(base_kw={"config": cfg})))
    assert dec["differences"] == []


# ---- anything else is a full port ---------------------------------------------------------------------

def test_a_structural_difference_beside_a_serving_one_is_a_full_port_with_high_confidence(audio_case):
    cfg = changed_model_config(sample_size=5324800, model__diffusion__config__depth=24)
    d = run_ok(audio_case(base_kw={"config": cfg}))
    dec = decision(d)
    assert d["class"] == "full-port" and d["path"] == "full-port" and dec["confidence"] == "high" and dec["alternative_path"] is None
    cats = {x["key"]: x["category"] for x in dec["differences"]}
    assert cats == {"sample_size": "serving", "model.diffusion.config.depth": "structural"}
    assert "model.diffusion.config.depth" in reasons(d) and "sample_size" not in reasons(d)


def test_an_unknown_key_is_a_full_port_with_medium_confidence(audio_case):
    cfg = changed_model_config(model__extra={"a": 1})
    d = run_ok(audio_case(base_kw={"config": cfg}))
    dec = decision(d)
    assert d["class"] == "full-port" and dec["confidence"] == "medium"
    assert [x["category"] for x in dec["differences"]] == ["unknown"] and "model.extra.a" in reasons(d)


def test_an_objective_the_table_does_not_cover_is_unknown_and_a_full_port(audio_case):
    cfg = changed_model_config(model__diffusion__diffusion_objective="v")
    d = run_ok(audio_case(base_kw={"config": cfg}))
    assert d["class"] == "full-port" and decision(d)["differences"][0]["category"] == "unknown"


def test_a_tensor_difference_beside_serving_differences_is_a_full_port(audio_case):
    cfg = changed_model_config(sample_size=5324800)
    t = {**TENSORS, "model.model.layers.0.attn.weight": ("F32", [8, 9])}
    d = run_ok(audio_case(model_kw={"tensors": t}, base_kw={"config": cfg}))
    assert d["class"] == "full-port" and decision(d)["confidence"] == "high" and "model.model.layers.0.attn.weight" in reasons(d)


def test_a_text_encoder_difference_beside_serving_differences_is_a_full_port(audio_case):
    cfg = changed_model_config(sample_size=5324800)
    d = run_ok(audio_case(model_kw={"text_config": {"architectures": ["x"]}}, base_kw={"config": cfg}))
    assert d["class"] == "full-port"


def test_unreadable_evidence_is_unknown_with_low_confidence_exactly_as_before(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(sample_size=5324800)})
    (model_snapshot(stage) / "model.safetensors").write_bytes(b"\xff" * 16)
    d = run_ok(stage)
    dec = decision(d)
    assert d["class"] == "unknown" and d["path"] == "full-port" and dec["class"] == "unknown" and dec["confidence"] == "low" and dec["alternative_path"] is None
    assert gate(stage).ok


# ---- the strict flag ---------------------------------------------------------------------------------

def strict(stage, value=True):
    p = stage / "triage_config.json"
    cfg = json.loads(p.read_text())
    cfg["audio_strict"] = value
    p.write_text(json.dumps(cfg))


def test_the_strict_flag_makes_any_difference_a_full_port_and_names_the_relaxed_alternative(audio_case):
    stage = serving_pair(audio_case)
    strict(stage)
    d = run_ok(stage)
    dec = decision(d)
    assert d["class"] == "full-port" and d["path"] == "full-port" and dec["strict"] is True and dec["alternative_path"] == "weights-only"
    assert "sample_size" in reasons(d) and "model.diffusion.diffusion_objective" in reasons(d)
    assert gate(stage).ok


def test_the_strict_flag_changes_nothing_for_a_pair_with_no_difference(audio_case):
    stage = audio_case()
    strict(stage)
    d = run_ok(stage)
    assert d["class"] == "weights-only" and decision(d)["strict"] is True and decision(d)["alternative_path"] is None


def test_the_strict_flag_off_is_the_default(audio_case):
    stage = serving_pair(audio_case)
    strict(stage, False)
    assert run_ok(stage)["class"] == "weights-only"


@pytest.mark.parametrize("value", ["yes", 1, [], {}])
def test_a_strict_flag_that_is_not_a_boolean_is_unknown(audio_case, value):
    stage = serving_pair(audio_case)
    strict(stage, value)
    d = run_ok(stage)
    assert d["class"] == "unknown" and "audio_strict" in reasons(d)


def test_no_settings_hazard_is_written_in_strict_mode(audio_case):
    stage = serving_pair(audio_case)
    strict(stage)
    assert not any("serving configuration" in h["finding"] for h in run_ok(stage)["hazards"])


# ---- the override hook ---------------------------------------------------------------------------------------

def write_overrides(stage, data):
    (stage / "audio_overrides.json").write_text(data if isinstance(data, str) else json.dumps(data))


UNKNOWN_KEY_CFG = dict(model__extra={"a": 1})


def test_a_category_override_with_a_source_moves_an_unknown_key_to_serving_and_is_recorded(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(**UNKNOWN_KEY_CFG)})
    write_overrides(stage, {"categories": [{"key": "model.extra.a", "category": "serving", "source": "read stable_audio_3/foo.py line 10: only a default"}]})
    d = run_ok(stage)
    dec = decision(d)
    assert d["class"] == "weights-only" and dec["confidence"] == "medium"
    [diff] = dec["differences"]
    assert (diff["category"], diff["decided_by"], diff["source"]) == ("serving", "override", "read stable_audio_3/foo.py line 10: only a default")
    assert dec["overrides"] == [{"kind": "category", "key": "model.extra.a", "category": "serving", "from": "unknown",
                                 "source": "read stable_audio_3/foo.py line 10: only a default"}]
    assert gate(stage).ok, gate(stage).reasons


def test_a_category_override_can_also_make_a_serving_key_structural(audio_case):
    stage = serving_pair(audio_case)
    write_overrides(stage, {"categories": [{"key": "sample_size", "category": "structural", "source": "the model's code indexes a table by it"}]})
    d = run_ok(stage)
    assert d["class"] == "full-port" and decision(d)["overrides"][0]["from"] == "serving"


@pytest.mark.parametrize("data", [
    "not json", "[]", '{"categories": "x"}', '{"categories": [1]}', '{"categories": [{"key": "a"}]}',
    '{"categories": [{"key": "model.extra.a", "category": "serving"}]}',
    '{"categories": [{"key": "model.extra.a", "category": "serving", "source": ""}]}',
    '{"categories": [{"key": "model.extra.a", "category": "serving", "source": "   "}]}',
    '{"categories": [{"key": "model.extra.a", "category": "serving", "source": 5}]}',
    '{"categories": [{"key": "model.extra.a", "category": "mystery", "source": "s"}]}',
    '{"categories": [{"key": 3, "category": "serving", "source": "s"}]}',
    '{"categories": [{"key": "not.a.differing.key", "category": "serving", "source": "s"}]}',
    '{"categories": [{"key": "model.extra.a", "category": "serving", "source": "s"}, {"key": "model.extra.a", "category": "structural", "source": "s"}]}',
    '{"path": "weights-only"}', '{"path": {"path": "weights-only"}}', '{"path": {"source": "s"}}', '{"path": {"path": "elsewhere", "source": "s"}}',
    '{"surprise": 1}',
])
def test_a_malformed_override_file_is_unknown_and_never_relaxes_anything(audio_case, data):
    stage = audio_case(base_kw={"config": changed_model_config(**UNKNOWN_KEY_CFG)})
    write_overrides(stage, data)
    d = run_ok(stage)
    assert d["class"] == "unknown" and "audio_overrides.json" in reasons(d)
    assert gate(stage).ok


def test_an_oversized_override_file_is_refused(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(**UNKNOWN_KEY_CFG)})
    write_overrides(stage, json.dumps({"categories": [], "pad": "x" * (2 << 20)}))
    assert run_ok(stage)["class"] == "unknown"


def test_no_override_file_is_not_a_problem(audio_case):
    assert run_ok(audio_case())["class"] == "weights-only"


def test_a_path_override_to_full_port_is_accepted_with_its_source(audio_case):
    stage = serving_pair(audio_case)
    write_overrides(stage, {"path": {"path": "full-port", "source": "the model's code reads sample_size into a position table"}})
    d = run_ok(stage)
    dec = decision(d)
    assert d["class"] == "full-port" and d["path"] == "full-port" and dec["alternative_path"] == "weights-only"
    assert dec["overrides"] == [{"kind": "path", "path": "full-port", "from": "weights-only",
                                 "source": "the model's code reads sample_size into a position table"}]


def test_a_path_override_to_weights_only_may_override_structural_config_evidence_and_is_recorded(audio_case):
    cfg = changed_model_config(model__diffusion__config__depth=24)
    stage = audio_case(base_kw={"config": cfg})
    write_overrides(stage, {"path": {"path": "weights-only", "source": "operator: the depth key is unused by the library"}})
    d = run_ok(stage)
    dec = decision(d)
    assert d["path"] == "weights-only" and dec["overrides"][0]["kind"] == "path" and dec["confidence"] == "medium" and dec["alternative_path"] == "full-port"
    assert gate(stage).ok, gate(stage).reasons


def test_a_path_override_to_weights_only_is_refused_against_tensor_evidence(audio_case):
    t = {**TENSORS, "model.model.layers.0.attn.weight": ("F32", [8, 9])}
    stage = audio_case(model_kw={"tensors": t})
    write_overrides(stage, {"path": {"path": "weights-only", "source": "s"}})
    d = run_ok(stage)
    assert d["class"] == "unknown" and "tensor" in reasons(d) and "override" in reasons(d)


def test_a_path_override_to_weights_only_is_refused_when_the_evidence_is_unreadable(audio_case):
    stage = audio_case()
    (model_snapshot(stage) / "model.safetensors").write_bytes(b"\xff" * 16)
    write_overrides(stage, {"path": {"path": "weights-only", "source": "s"}})
    assert run_ok(stage)["class"] == "unknown"


def test_overrides_are_applied_in_strict_mode_too_and_a_category_override_cannot_relax_it(audio_case):
    stage = serving_pair(audio_case)
    strict(stage)
    write_overrides(stage, {"categories": [{"key": "sample_size", "category": "serving", "source": "already serving"}]})
    assert run_ok(stage)["class"] == "full-port"


# ---- the gate re-checks the decision ----------------------------------------------------------------------

def delta_of(stage):
    return json.loads((stage / "delta.json").read_text())


def edit_delta(stage, fn):
    d = delta_of(stage)
    fn(d)
    (stage / "delta.json").write_text(json.dumps(d))


def test_the_gate_accepts_the_template_s_own_decisions(audio_case):
    for stage in (serving_pair(audio_case),):
        run_ok(stage)
        assert gate(stage).ok


def gate_reasons(stage):
    return " | ".join(gate(stage).reasons)


@pytest.mark.parametrize("edit,match", [
    (lambda d: d["audio"]["decision"].update(path="full-port"), "decision path"),
    (lambda d: d["audio"]["decision"].update(**{"class": "full-port"}), "decision class"),
    (lambda d: d["audio"]["decision"].update(confidence="certain"), "confidence"),
    (lambda d: d["audio"]["decision"].update(alternative_path="weights-only"), "alternative"),
    (lambda d: d["audio"]["decision"].update(alternative_path="nonsense"), "alternative"),
    (lambda d: d["audio"]["decision"].update(strict="no"), "strict"),
    (lambda d: d["audio"]["decision"].update(differences="x"), "differences"),
    (lambda d: d["audio"]["decision"].update(overrides="x"), "overrides"),
    (lambda d: d["audio"]["decision"]["differences"][0].update(category="mystery"), "category"),
    (lambda d: d["audio"]["decision"]["differences"][0].update(decided_by="guess"), "decided_by"),
    (lambda d: d["audio"]["decision"]["differences"][0].update(category="structural"), "weights-only"),
    (lambda d: d["audio"]["decision"]["differences"][0].update(category="unknown"), "weights-only"),
    (lambda d: d["audio"]["decision"]["differences"][0].pop("key"), "key"),
    (lambda d: d["audio"]["decision"]["differences"][0].update(decided_by="override"), "source"),
    (lambda d: d["audio"]["decision"].pop("class"), "class"),
])
def test_the_gate_refuses_an_inconsistent_or_malformed_decision(audio_case, edit, match):
    stage = serving_pair(audio_case)
    run_ok(stage)
    edit_delta(stage, edit)
    r = gate(stage)
    assert not r.ok and any(match in x for x in r.reasons), r.reasons


def test_the_gate_refuses_a_strict_decision_that_is_weights_only_with_differences(audio_case):
    stage = serving_pair(audio_case)
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"]["decision"].update(strict=True))
    assert not gate(stage).ok


def test_the_gate_refuses_an_override_without_a_source(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(**UNKNOWN_KEY_CFG)})
    write_overrides(stage, {"categories": [{"key": "model.extra.a", "category": "serving", "source": "s"}]})
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"]["decision"]["overrides"][0].update(source=""))
    assert not gate(stage).ok and "source" in gate_reasons(stage)


def test_the_gate_refuses_a_weights_only_decision_whose_override_does_not_explain_a_structural_difference(audio_case):
    stage = audio_case(base_kw={"config": changed_model_config(model__diffusion__config__depth=24)})
    run_ok(stage)
    edit_delta(stage, lambda d: (d.update(path="weights-only", **{"class": "weights-only"}), d["audio"]["decision"].update(path="weights-only", **{"class": "weights-only"})))
    assert not gate(stage).ok and "weights-only" in gate_reasons(stage)


def test_a_delta_with_no_decision_still_passes_the_gate(audio_case):
    stage = audio_case()
    run_ok(stage)
    edit_delta(stage, lambda d: d["audio"].pop("decision"))
    assert gate(stage).ok


def test_the_decision_check_is_importable_and_accepts_the_template_s_decision(audio_case):
    stage = serving_pair(audio_case)
    d = run_ok(stage)
    assert audio_decision.decision_problems(d) == []
    assert audio_decision.decision_problems({"audio": {}}) == []
    assert audio_decision.decision_problems({}) == []
    assert audio_decision.decision_problems({"audio": {"decision": 3}})


# ---- nothing runs on the relaxed class yet (spec of the audio stop) -----------------------------------------

def test_the_template_is_still_stamped_so_the_gate_accepts_it(audio_case):
    stage = serving_pair(audio_case)
    d = run_ok(stage)
    import hashlib
    assert d["template_sha256"] == hashlib.sha256((Path(__file__).resolve().parent.parent / "orchard/skills/delta-triage-templates/delta_triage.py").read_bytes()).hexdigest()
