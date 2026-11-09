# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 reads the nearest bundle's `orchard-capabilities.json` (Track C, item C1 of the Qwen4 drop-in plan).

A bundle whose model code reads its configuration at run time declares which text config keys may differ from the model it was built
with, the rule each new value must meet, the tensor dimensions that follow those keys, how layers are named (so a model with more layers
than the bundle's weights is not a pile of extra tensors), and tensors the code can do without. `decide_path` then sets aside exactly
those differences. Anything not declared stays a full-port reason, and with the file present every differing text config key must be
declared (the bundle serves the new model's own config.json, item C2). Without the file the triage is unchanged."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"


def template():
    spec = importlib.util.spec_from_file_location("delta_triage_capabilities_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = template()

H, HEADS, KV, D, INTER, VOCAB = 8, 2, 1, 4, 16, 32
LAYER = r"^model\.layers\.(?P<layer>\d+)\.(?P<rest>.+)$"


def cfg(layers=2, **change) -> dict:
    types = (["linear_attention", "full_attention"] * layers)[:layers]
    c = {"model_type": "qwen4_exp_text", "architectures": ["Qwen4ExpForCausalLM"], "num_hidden_layers": layers, "hidden_size": H,
         "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": D, "layer_types": types, "vocab_size": VOCAB,
         "intermediate_size": INTER, "rope_theta": 10000000, "hc_count": 4, "rms_norm_eps": 1e-6}
    c.update(change)
    return c


def layer_tensors(i: int, kind: str, *, kv=KV, inter=INTER) -> dict:
    p = f"model.layers.{i}."
    out = {p + "mlp.up_proj.weight": ("BF16", [inter, H]), p + "mlp.down_proj.weight": ("BF16", [H, inter]),
           p + "attn_hc.hc_norm.weight": ("BF16", [4 * H])}
    if kind == "full_attention":
        out.update({p + "self_attn.q_proj.weight": ("BF16", [2 * HEADS * D, H]), p + "self_attn.k_proj.weight": ("BF16", [kv * D, H])})
    else:
        out.update({p + "linear_attn.in_proj_qkvz.weight": ("BF16", [3 * H, H])})
    return out


def tensors(c: dict) -> dict:
    out = {"model.embed_tokens.weight": ("BF16", [c["vocab_size"], H]), "lm_head.weight": ("BF16", [c["vocab_size"], H]),
           "model.final_hc.hc_norm.weight": ("BF16", [4 * H])}
    for i, kind in enumerate(c["layer_types"]):
        out.update(layer_tensors(i, kind, kv=c["num_key_value_heads"], inter=c["intermediate_size"]))
    return out


def tmap(ts: dict) -> dict:
    return {k: {"dtype": dt, "shape": list(shape), "shard": "model.safetensors"} for k, (dt, shape) in ts.items()}


def caps(**change) -> dict:
    """The capabilities a Qwen4-capable bundle would declare for this toy model, in the release-day check's schema."""
    c = {"schema": 1, "bundle": "mando2222/qwen4-27b-dense-v6-thin",
         "model_types": ["qwen4_exp", "qwen4_exp_text"], "architectures": ["Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"],
         "keys": {
             "num_hidden_layers": {"min": 1},
             "num_key_value_heads": {"one_of": [1, 2]},
             "intermediate_size": {"multiple_of": 8},
             "rope_theta": {"min": 1},
             "hidden_size": {"equals": H},
         },
         "features": {"indexer": "exact-up-to-2051", "mtp": "ignored", "vision": "off"},
         "layers": {"pattern": LAYER, "count_key": "num_hidden_layers", "types_key": "layer_types",
                    "type_values": ["linear_attention", "full_attention"]},
         "shapes": [
             {"tensor": r"^model\.layers\.\d+\.mlp\.up_proj\.weight$", "dims": {"0": "intermediate_size"}},
             {"tensor": r"^model\.layers\.\d+\.mlp\.down_proj\.weight$", "dims": {"1": "intermediate_size"}},
             {"tensor": r"^model\.layers\.\d+\.self_attn\.k_proj\.weight$", "dims": {"0": "num_key_value_heads*head_dim"}},
         ],
         "optional_tensors": [{"tensor": r"^model\.layers\.\d+\.self_attn\.indexer\..+$", "feature": "qsa_indexer"}]}
    c.update(change)
    return c


def judge(mcfg, bcfg, mt=None, bt=None, capabilities="default"):
    """(reasons, review) for two configs and tensor maps, the way main() calls decide_path."""
    mt = tensors(mcfg) if mt is None else mt
    bt = tensors(bcfg) if bt is None else bt
    c = T.compare_config_dicts(mcfg, bcfg, [], "m/config.json", "b/config.json")
    t = T.compare_tensor_maps(tmap(mt), tmap(bt), mshards=["model.safetensors"], bshards=["model.safetensors"], merr=[], berr=[],
                              model_path="m", base_path="b")
    data = caps() if capabilities == "default" else capabilities
    review = None
    if data is not None:
        checked, problems = T.validate_capabilities(data)
        assert problems == [], problems
        review = T.review_capabilities(checked, c, t, mcfg, bcfg, tmap(mt), tmap(bt))
    return T.decide_path(c, t, review), review


# ---- the file itself ---------------------------------------------------------------------------------------------------------

def test_the_example_capabilities_are_valid():
    checked, problems = T.validate_capabilities(caps())
    assert problems == []
    assert set(checked["keys"]) >= {"num_hidden_layers", "num_key_value_heads"}


# The release-day check's example file (orchard/drop_check.py EXAMPLE_CAPABILITIES on the main session's branch, 2026-11-09): stage 0
# must accept the same file.
DROP_CHECK_EXAMPLE = {
    "schema": 1, "bundle": "mando2222/qwen4-27b-dense-v6-thin", "model_types": ["qwen4_exp", "qwen4_exp_text"],
    "architectures": ["Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM"],
    "keys": {"hidden_size": {"equals": 5120}, "num_attention_heads": {"equals": 24}, "head_dim": {"equals": 256},
             "num_key_value_heads": {"one_of": [2, 4]}, "intermediate_size": {"multiple_of": 128},
             "linear_num_key_heads": {"equals": 16}, "linear_num_value_heads": {"equals": 48},
             "linear_key_head_dim": {"equals": 128}, "linear_value_head_dim": {"equals": 128},
             "linear_conv_kernel_dim": {"equals": 4}, "vocab_size": {"equals": 248320}, "hc_count": {"equals": 4},
             "hc_lowrank": {"one_of": [320, 640]}, "output_gate_type": {"one_of": ["silu", "sigmoid"]},
             "hidden_act": {"equals": "silu"}},
    "features": {"indexer": "exact-up-to-2051", "mtp": "ignored", "vision": "off"},
}


def test_the_release_day_checks_example_file_is_valid_here_too():
    checked, problems = T.validate_capabilities(DROP_CHECK_EXAMPLE)
    assert problems == [] and checked["keys"]["num_key_value_heads"] == {"one_of": [2, 4]}


@pytest.mark.parametrize("change, words", [
    ({"schema": 2}, "schema"),
    ({"extra": 1}, "extra"),
    ({"keys": []}, "keys"),
    ({"keys": {"x": {"type": "int"}}}, "keys.x"),
    ({"keys": {"x": {}}}, "keys.x"),
    ({"keys": {"x": "int"}}, "keys.x"),
    ({"keys": {"x": {"one_of": 3}}}, "one_of"),
    ({"keys": {"x": {"multiple_of": 0}}}, "multiple_of"),
    ({"keys": {"x": {"min": True}}}, "min"),
    ({"keys": {"x": {"max": "8"}}}, "max"),
    ({"keys": {"a b": {"min": 1}}}, "dotted"),
    ({"model_types": []}, "model_types"),
    ({"architectures": "Qwen4ExpForCausalLM"}, "architectures"),
    ({"ignored_keys": "dtype"}, "ignored_keys"),
    ({"optional_keys": ["a b"]}, "optional_keys"),
    ({"features": {"indexer": "sometimes"}}, "indexer"),
    ({"features": []}, "features"),
    ({"bundle": "no-slash"}, "bundle"),
    ({"shapes": [{"tensor": "(", "dims": {"0": "a"}}]}, "regular expression"),
    ({"shapes": [{"tensor": "a", "dims": {"x": "a"}}]}, "dimension"),
    ({"shapes": [{"tensor": "a", "dims": {"0": "a-b"}}]}, "expression"),
    ({"shapes": [{"tensor": "a", "dims": {"0": "a**b"}}]}, "expression"),
    ({"shapes": [{"tensor": "a", "dims": {}}]}, "dims"),
    ({"shapes": {"tensor": "a"}}, "shapes"),
    ({"layers": {"pattern": r"^model\.layers\.(\d+)\.(.+)$", "count_key": "num_hidden_layers"}}, "layer"),
    ({"layers": {"pattern": LAYER}}, "count_key"),
    ({"layers": {"pattern": LAYER, "count_key": "num_hidden_layers", "type_values": ["full_attention"]}}, "type_values"),
    ({"optional_tensors": [{"tensor": "a"}]}, "feature"),
])
def test_a_malformed_file_is_refused_with_a_named_problem(change, words):
    _, problems = T.validate_capabilities(caps(**change))
    assert problems and any(words in p for p in problems), problems


def test_a_file_that_is_not_an_object_is_refused():
    assert T.validate_capabilities([1])[1]
    assert T.validate_capabilities(None)[1]


@pytest.mark.parametrize("missing", ["model_types", "architectures", "keys", "schema"])
def test_each_field_the_release_day_check_requires_is_required(missing):
    data = caps()
    del data[missing]
    assert T.validate_capabilities(data)[1]


# ---- the rules -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("rule, value, ok", [
    ({"equals": 4}, 4, True),
    ({"equals": 4}, 5, False),
    ({"equals": 1}, True, False),
    ({"equals": "silu"}, "silu", True),
    ({"one_of": [2, 4]}, 4, True),
    ({"one_of": [2, 4]}, 3, False),
    ({"one_of": ["silu", "sigmoid"]}, "swish", False),
    ({"multiple_of": 32}, 17408, True),
    ({"multiple_of": 32}, 17400, False),
    ({"multiple_of": 32}, "17408", False),
    ({"min": 1, "max": 8}, 0, False),
    ({"min": 1, "max": 8}, 9, False),
    ({"min": 1, "max": 8}, 8, True),
    ({"min": 1}, 1e7, True),
    ({"min": 1}, False, False),
    ({"min": 1}, None, False),
    ({"one_of": [2, 4], "max": 2}, 4, False),
    ({"min": 1}, T.ABSENT, False),
])
def test_rule_table(rule, value, ok):
    checked, problems = T.validate_capabilities(caps(keys={"k": rule}))
    assert problems == [], problems
    got = T.rule_problem(checked["keys"]["k"], value)
    assert (got is None) is ok, got


def test_key_value_reads_dotted_paths_and_nested_configs():
    nested = {"text_config": {"rope_parameters": {"rope_theta": 5.0}, "hidden_size": 8}, "model_type": "qwen4_exp"}
    assert T.key_value(nested, "rope_parameters.rope_theta") == 5.0
    assert T.key_value(nested, "rope_parameters.missing") == T.ABSENT
    assert T.key_value(nested, "hidden_size.x") == T.ABSENT
    assert T.key_value(nested, "model_type") == "qwen4_exp"


@pytest.mark.parametrize("expr, value", [("hidden_size", H), ("hc_count*hidden_size", 4 * H), ("2*num_attention_heads*head_dim", 2 * HEADS * D),
                                         ("num_attention_heads*head_dim+head_dim", HEADS * D + D), ("missing_key", None)])
def test_dimension_expressions(expr, value):
    assert T.dim_value(expr, cfg()) == value


def test_a_dimension_expression_on_a_boolean_or_a_huge_value_gives_nothing():
    assert T.dim_value("flag", cfg(flag=True)) is None
    assert T.dim_value("big*big", cfg(big=10 ** 20)) is None


# ---- decide_path with capabilities ---------------------------------------------------------------------------------------------

def test_identical_models_stay_weights_only_with_nothing_set_aside():
    reasons, review = judge(cfg(), cfg())
    assert reasons == []
    assert review["exempted"]["config_keys"] == [] and review["exempted"]["shapes"] == []


def test_without_capabilities_the_triage_is_unchanged():
    m, b = cfg(rope_theta=5), cfg()
    reasons, review = judge(m, b, capabilities=None)
    assert review is None and reasons == []          # rope_theta is not decisive: the old rule ignores it
    reasons, _ = judge(cfg(num_key_value_heads=2), cfg(), capabilities=None)
    assert any("num_key_value_heads" in r for r in reasons)


def test_a_declared_decisive_key_within_its_rule_and_the_shapes_that_follow_it_are_set_aside():
    reasons, review = judge(cfg(num_key_value_heads=2, intermediate_size=24), cfg())
    assert reasons == [], reasons
    keys = {x["key"] for x in review["exempted"]["config_keys"]}
    assert keys == {"num_key_value_heads", "intermediate_size"}
    assert {x["tensor"] for x in review["exempted"]["shapes"]} == {
        "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight", "model.layers.1.mlp.up_proj.weight",
        "model.layers.1.mlp.down_proj.weight", "model.layers.1.self_attn.k_proj.weight"}


def test_an_undeclared_decisive_key_stays_full_port():
    data = caps()
    del data["keys"]["num_key_value_heads"]
    reasons, _ = judge(cfg(num_key_value_heads=2), cfg(), capabilities=data)
    assert any("num_key_value_heads" in r and "does not declare" in r for r in reasons), reasons


def test_a_value_outside_the_rule_stays_full_port():
    reasons, _ = judge(cfg(num_key_value_heads=4), cfg())
    assert any("num_key_value_heads" in r and "one of" in r for r in reasons), reasons


def test_an_undeclared_other_key_stays_full_port_once_the_bundle_declares_capabilities():
    reasons, _ = judge(cfg(rms_norm_eps=1e-5), cfg())
    assert any("rms_norm_eps" in r and "does not declare" in r for r in reasons), reasons


def test_a_declared_other_key_within_its_rule_is_set_aside():
    reasons, review = judge(cfg(rope_theta=5000000), cfg())
    assert reasons == []
    assert [x["key"] for x in review["exempted"]["config_keys"]] == ["rope_theta"]


def test_a_new_key_the_bundle_does_not_declare_stays_full_port():
    reasons, _ = judge(cfg(indexer_topk=512), cfg())
    assert any("indexer_topk" in r for r in reasons), reasons


def test_a_bookkeeping_or_derivable_key_needs_no_declaration():
    m = cfg(transformers_version="5.19.0", _name_or_path="x")
    reasons, _ = judge(m, cfg())
    assert reasons == []


def test_a_rule_holds_for_a_key_that_does_not_differ_too():
    """The release-day check applies each rule to the release's value whether or not it differs; stage 0 does the same, so the two
    cannot disagree about a release the bundle's own weights would also break."""
    data = caps(keys={**caps()["keys"], "hidden_size": {"equals": 2 * H}})
    reasons, _ = judge(cfg(), cfg(), capabilities=data)
    assert any("hidden_size" in r and "outside the rule" in r for r in reasons), reasons


def test_a_model_type_outside_model_types_stays_full_port():
    reasons, _ = judge(cfg(model_type="qwen4_text"), cfg())
    assert any("model_type" in r for r in reasons), reasons
    reasons, review = judge(cfg(model_type="qwen4_text"), cfg(), capabilities=caps(model_types=["qwen4_exp_text", "qwen4_text"]))
    assert reasons == [] and "model_type" in {x["key"] for x in review["exempted"]["config_keys"]}


def test_a_model_type_is_checked_even_when_both_models_share_it():
    reasons, _ = judge(cfg(), cfg(), capabilities=caps(model_types=["qwen4"]))
    assert any("model_type" in r for r in reasons), reasons


def test_an_architecture_the_bundle_does_not_register_stays_full_port():
    reasons, _ = judge(cfg(architectures=["Qwen4ForCausalLM"]), cfg())
    assert any("Qwen4ForCausalLM" in r for r in reasons), reasons


def test_a_dotted_key_declares_one_value_inside_an_object():
    m, b = cfg(rope_parameters={"rope_theta": 5e6, "rope_type": "default"}), cfg(rope_parameters={"rope_theta": 1e7, "rope_type": "default"})
    reasons, _ = judge(m, b)
    assert any("rope_parameters" in r for r in reasons), reasons                 # not declared
    data = caps(keys={**caps()["keys"], "rope_parameters.rope_theta": {"min": 1}})
    reasons, review = judge(m, b, capabilities=data)
    assert reasons == [] and "rope_parameters" in {x["key"] for x in review["exempted"]["config_keys"]}
    m2 = cfg(rope_parameters={"rope_theta": 5e6, "rope_type": "yarn"})
    reasons, _ = judge(m2, b, capabilities=data)
    assert any("rope_parameters.rope_type" in r for r in reasons), reasons          # another sub-key differs, not declared
    m3 = cfg(rope_parameters={"rope_theta": 0.5, "rope_type": "default"})
    reasons, review = judge(m3, b, capabilities=data)
    assert any("rope_parameters.rope_theta" in r and "minimum" in r for r in reasons), reasons
    assert "rope_parameters" not in {x["key"] for x in review["exempted"]["config_keys"]}       # not recorded as set aside
    assert sum("rope_parameters.rope_theta" in r for r in reasons) == 1                          # and named once


def test_ignored_keys_accept_any_value_and_absence():
    data = caps(ignored_keys=["initializer_range"])
    assert judge(cfg(initializer_range=0.5), cfg(initializer_range=0.02), capabilities=data)[0] == []
    b = cfg(initializer_range=0.02)
    m = cfg()
    assert judge(m, b, capabilities=data)[0] == []
    assert judge(m, b)[0]                                                          # not ignored: a difference


def test_a_declared_key_the_new_model_lacks_needs_optional_keys():
    b, m = cfg(full_attention_interval=4), cfg()
    data = caps(keys={**caps()["keys"], "full_attention_interval": {"min": 1}})
    reasons, _ = judge(m, b, capabilities=data)
    assert any("full_attention_interval" in r and "optional_keys" in r for r in reasons), reasons
    assert judge(m, b, capabilities={**data, "optional_keys": ["full_attention_interval"]})[0] == []


def test_layer_types_outside_type_values_stay_full_port():
    m = cfg(layers=2, layer_types=["linear_attention", "indexed_attention"])
    mt = tensors(cfg(layers=2))
    reasons, _ = judge(m, cfg(layers=2), mt=mt)
    assert any("layer_types" in r and "type_values" in r for r in reasons), reasons


def test_layer_types_of_the_wrong_length_stay_full_port():
    m = cfg(layers=4)
    m["layer_types"] = m["layer_types"][:3]
    reasons, _ = judge(m, cfg(layers=2), mt=tensors(cfg(layers=4)))
    assert any("layer_types" in r for r in reasons), reasons


def test_a_shape_difference_along_an_undeclared_dimension_stays_full_port():
    m, b = cfg(), cfg()
    mt = tensors(m)
    mt["model.layers.0.mlp.up_proj.weight"] = ("BF16", [INTER, 2 * H])      # dim 1 is not declared for up_proj
    reasons, _ = judge(m, b, mt=mt)
    assert any("up_proj" in r for r in reasons), reasons


def test_a_shape_difference_on_a_tensor_no_rule_names_stays_full_port():
    m, b = cfg(), cfg()
    mt = tensors(m)
    mt["model.layers.0.attn_hc.hc_norm.weight"] = ("BF16", [8 * H])
    reasons, _ = judge(m, b, mt=mt)
    assert any("hc_norm" in r for r in reasons), reasons


def test_a_declared_dimension_whose_size_does_not_follow_the_key_stays_full_port():
    m, b = cfg(intermediate_size=24), cfg()
    mt = tensors(m)
    mt["model.layers.0.mlp.up_proj.weight"] = ("BF16", [32, H])            # the config says 24
    reasons, _ = judge(m, b, mt=mt)
    assert any("model.layers.0.mlp.up_proj.weight" in r for r in reasons), reasons


def test_a_rank_change_is_never_set_aside():
    m, b = cfg(intermediate_size=24), cfg()
    mt = tensors(m)
    mt["model.layers.0.mlp.up_proj.weight"] = ("BF16", [24, H, 1])
    reasons, _ = judge(m, b, mt=mt)
    assert any("model.layers.0.mlp.up_proj.weight" in r for r in reasons), reasons


def test_the_base_side_must_follow_the_declared_expression_too():
    m, b = cfg(intermediate_size=24), cfg()
    bt = tensors(b)
    bt["model.layers.0.mlp.up_proj.weight"] = ("BF16", [40, H])            # the base config says 16
    reasons, _ = judge(m, b, bt=bt)
    assert any("model.layers.0.mlp.up_proj.weight" in r for r in reasons), reasons


def test_an_extra_tensor_stays_full_port():
    m, b = cfg(), cfg()
    mt = tensors(m)
    mt["model.layers.0.mlp.gate_proj.weight"] = ("BF16", [INTER, H])
    reasons, _ = judge(m, b, mt=mt)
    assert any("gate_proj" in r and "only in the new model" in r for r in reasons), reasons


def test_a_missing_tensor_stays_full_port():
    m, b = cfg(), cfg()
    mt = tensors(m)
    del mt["model.layers.1.self_attn.q_proj.weight"]
    reasons, _ = judge(m, b, mt=mt)
    assert any("q_proj" in r and "only in the nearest model" in r for r in reasons), reasons


def test_an_optional_tensor_is_set_aside_and_named_by_its_feature():
    m, b = cfg(), cfg()
    mt = tensors(m)
    mt["model.layers.1.self_attn.indexer.index_qk_proj.weight"] = ("BF16", [5 * D, H])
    reasons, review = judge(m, b, mt=mt)
    assert reasons == []
    assert review["exempted"]["optional_tensors"]["features"] == {"qsa_indexer": 1}
    # and the same tensor missing from the new model where the nearest model has it
    reasons, review = judge(b, m, mt=tensors(b), bt=mt)
    assert reasons == []


def test_an_optional_tensor_with_another_shape_than_declared_stays_full_port():
    data = caps(optional_tensors=[{"tensor": r"^model\.layers\.\d+\.self_attn\.indexer\.index_qk_proj\.weight$", "feature": "qsa_indexer",
                                   "shape": ["head_dim+head_dim", "hidden_size"]}])
    m, b = cfg(), cfg()
    mt = tensors(m)
    mt["model.layers.1.self_attn.indexer.index_qk_proj.weight"] = ("BF16", [2 * D, H])
    assert judge(m, b, mt=mt, capabilities=data)[0] == []
    mt["model.layers.1.self_attn.indexer.index_qk_proj.weight"] = ("BF16", [3 * D, H])
    reasons, _ = judge(m, b, mt=mt, capabilities=data)
    assert any("index_qk_proj" in r for r in reasons), reasons


def test_more_layers_than_the_nearest_model_are_set_aside_when_the_count_is_declared():
    reasons, review = judge(cfg(layers=6), cfg(layers=2))
    assert reasons == [], reasons
    keys = {x["key"] for x in review["exempted"]["config_keys"]}
    assert keys == {"num_hidden_layers", "layer_types"}
    assert review["exempted"]["layer_tensors"]["count"] == sum(len(layer_tensors(i, k)) for i, k in enumerate(cfg(layers=6)["layer_types"]) if i >= 2)


def test_fewer_layers_than_the_nearest_model_are_set_aside_too():
    reasons, review = judge(cfg(layers=2), cfg(layers=4))
    assert reasons == [], reasons
    assert review["exempted"]["layer_tensors"]["count"] > 0


def test_extra_layers_are_not_set_aside_when_the_count_is_not_declared():
    data = caps()
    del data["keys"]["num_hidden_layers"]
    reasons, _ = judge(cfg(layers=4), cfg(layers=2), capabilities=data)
    assert any("num_hidden_layers" in r for r in reasons)
    assert any("only in the new model" in r for r in reasons), reasons


def test_an_extra_layer_of_a_type_the_nearest_model_lacks_stays_full_port():
    m = cfg(layers=3, layer_types=["linear_attention", "full_attention", "full_attention"])
    b = cfg(layers=2, layer_types=["linear_attention", "linear_attention"])
    reasons, _ = judge(m, b)
    assert any("model.layers.2." in r and "only in the new model" in r for r in reasons), reasons


def test_an_extra_layer_is_matched_only_with_a_layer_of_its_own_type():
    """Layer 2 is a full-attention layer by its config, but holds linear-attention tensors. The nearest model has those only at a
    linear-attention layer, so they have no analogue of the right type."""
    m = cfg(layers=3, layer_types=["linear_attention", "full_attention", "full_attention"])
    mt = tensors(cfg(layers=3, layer_types=["linear_attention", "full_attention", "linear_attention"]))
    reasons, _ = judge(m, cfg(layers=2), mt=mt)
    assert any("model.layers.2.linear_attn" in r for r in reasons), reasons


def test_an_extra_layer_tensor_with_another_dtype_than_its_analogue_stays_full_port():
    m, b = cfg(layers=4), cfg(layers=2)
    mt = tensors(m)
    mt["model.layers.3.attn_hc.hc_norm.weight"] = ("F32", [4 * H])
    reasons, _ = judge(m, b, mt=mt)
    assert any("model.layers.3.attn_hc.hc_norm.weight" in r for r in reasons), reasons


def test_an_extra_layer_whose_tensor_shape_differs_from_its_analogue_stays_full_port():
    m, b = cfg(layers=4), cfg(layers=2)
    mt = tensors(m)
    mt["model.layers.3.attn_hc.hc_norm.weight"] = ("BF16", [2 * H])
    reasons, _ = judge(m, b, mt=mt)
    assert any("model.layers.3.attn_hc.hc_norm.weight" in r for r in reasons), reasons


def test_a_tensor_missing_at_a_layer_both_models_have_is_not_a_layer_count_difference():
    m, b = cfg(layers=4), cfg(layers=2)
    mt = tensors(m)
    del mt["model.layers.1.mlp.up_proj.weight"]
    reasons, _ = judge(m, b, mt=mt)
    assert any("model.layers.1.mlp.up_proj.weight" in r for r in reasons), reasons


def test_an_extra_layer_beyond_the_new_models_own_count_stays_full_port():
    m, b = cfg(layers=4), cfg(layers=2)
    mt = tensors(m)
    mt["model.layers.7.mlp.up_proj.weight"] = ("BF16", [INTER, H])
    reasons, _ = judge(m, b, mt=mt)
    assert any("model.layers.7.mlp.up_proj.weight" in r for r in reasons), reasons


def test_a_dtype_difference_is_never_set_aside():
    m, b = cfg(), cfg()
    mt = tensors(m)
    mt["model.layers.0.mlp.up_proj.weight"] = ("I8", [INTER, H])
    reasons, _ = judge(m, b, mt=mt)
    assert any("dtype" in r for r in reasons), reasons


def test_the_qwen4_release_against_a_twenty_layer_franken_shape():
    """The release-day pair in miniature: more layers, other kv heads and feed-forward width, a new rope theta. Every difference is
    declared, so the path is weights-only and each set-aside item is listed."""
    m = cfg(layers=8, num_key_value_heads=2, intermediate_size=24, rope_theta=5000000)
    reasons, review = judge(m, cfg(layers=4))
    assert reasons == [], reasons
    assert {x["key"] for x in review["exempted"]["config_keys"]} == {
        "num_hidden_layers", "layer_types", "num_key_value_heads", "intermediate_size", "rope_theta"}


# ---- the script end to end ----------------------------------------------------------------------------------------------------

DTYPE_BYTES = {"BF16": 2, "F32": 4, "I8": 1}


def write_safetensors(path: Path, ts: dict) -> None:
    header, offset = {}, 0
    for name, (dtype, shape) in sorted(ts.items()):
        size = DTYPE_BYTES[dtype]
        for s in shape:
            size *= s
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)


TOKENIZER = json.dumps({"version": "1.0", "added_tokens": [], "normalizer": None, "pre_tokenizer": None, "post_processor": None,
                        "decoder": None, "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1, "[UNK]": 2}, "unk_token": "[UNK]"}})


def snapshot(root: Path, rev: str, c: dict) -> Path:
    snap = root / "snapshots" / rev
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps(c))
    (snap / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    (snap / "tokenizer.json").write_text(TOKENIZER)
    (snap / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 64}))
    (snap / "README.md").write_text("---\nlicense: apache-2.0\n---\n\n# A model\n")
    write_safetensors(snap / "model.safetensors", tensors(c))
    return snap


def stage(tmp_path: Path, mcfg: dict, bcfg: dict, *, capabilities=None, bundle: bool = True) -> tuple[Path, Path]:
    model = snapshot(tmp_path / "new", "1" * 40, mcfg)
    base = snapshot(tmp_path / "base", "2" * 40, bcfg)
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    if capabilities is not None:
        (bundle_dir / "orchard-capabilities.json").write_text(capabilities if isinstance(capabilities, str) else json.dumps(capabilities))
    st = tmp_path / "run" / "stages" / "0"
    st.mkdir(parents=True)
    shutil.copy(TEMPLATE, st)
    tc = {"run_dir": str(tmp_path / "run"), "model_id": "Qwen/Qwen4-27B", "model_snapshot": str(model),
          "nearest_model_id": "local/qwen4-franken", "nearest_snapshot": str(base)}
    if bundle:
        tc["nearest_bundle_dir"] = str(bundle_dir)
    (st / "triage_config.json").write_text(json.dumps(tc))
    return st, bundle_dir


def run(st: Path):
    r = subprocess.run([sys.executable, str(st / "delta_triage.py")], capture_output=True, text=True, timeout=300)
    return r, (json.loads((st / "delta.json").read_text()) if (st / "delta.json").exists() else None)


def test_end_to_end_a_declared_release_is_weights_only_and_the_record_names_the_file(tmp_path):
    data = caps()
    st, bundle_dir = stage(tmp_path, cfg(layers=4, num_key_value_heads=2), cfg(layers=2), capabilities=data)
    r, d = run(st)
    assert r.returncode == 0, r.stderr
    assert d["path"] == "weights-only" and d["class"] == "weights-only", d["path_reasons"]
    rec = d["capabilities"]
    assert rec["present"] is True and rec["applied"] is True
    assert rec["sha256"] == hashlib.sha256((bundle_dir / "orchard-capabilities.json").read_bytes()).hexdigest()
    assert rec["bundle_dir"] == str(bundle_dir)
    assert {x["key"] for x in rec["exempted"]["config_keys"]} == {"num_hidden_layers", "layer_types", "num_key_value_heads"}
    assert rec["features"] == {"indexer": "exact-up-to-2051", "mtp": "ignored", "vision": "off"}
    areas = [x for x in d["differences"] if x["area"] == "other" and "orchard-capabilities.json" in x["finding"]]
    assert areas, d["differences"]
    assert any("orchard-capabilities.json" in h["finding"] for h in d["hazards"])


def test_end_to_end_without_the_bundle_input_the_same_pair_is_full_port_and_has_no_record(tmp_path):
    st, _ = stage(tmp_path, cfg(layers=4), cfg(layers=2), capabilities=caps(), bundle=False)
    r, d = run(st)
    assert r.returncode == 0, r.stderr
    assert d["path"] == "full-port"
    assert "capabilities" not in d


def test_end_to_end_a_bundle_without_the_file_changes_nothing(tmp_path):
    st, _ = stage(tmp_path, cfg(layers=4), cfg(layers=2), capabilities=None)
    r, d = run(st)
    assert d["path"] == "full-port"
    assert d["capabilities"]["present"] is False and d["capabilities"]["applied"] is False


@pytest.mark.parametrize("text", ["{not json", json.dumps(caps(schema=9)), '{"schema": 1, "schema": 1}', '{"schema": NaN}', "[]"])
def test_end_to_end_a_broken_file_makes_the_path_full_port_and_says_why(tmp_path, text):
    st, _ = stage(tmp_path, cfg(), cfg(), capabilities=text)
    r, d = run(st)
    assert r.returncode == 0, r.stderr
    assert d["path"] == "full-port"
    assert any("orchard-capabilities.json" in x for x in d["path_reasons"]), d["path_reasons"]
    assert d["capabilities"]["applied"] is False and d["capabilities"]["problems"]


def test_end_to_end_a_linked_file_is_refused(tmp_path):
    st, bundle_dir = stage(tmp_path, cfg(), cfg(), capabilities=None)
    real = tmp_path / "elsewhere.json"
    real.write_text(json.dumps(caps()))
    os.symlink(real, bundle_dir / "orchard-capabilities.json")
    r, d = run(st)
    assert d["path"] == "full-port"
    assert any("link" in p for p in d["capabilities"]["problems"]), d["capabilities"]


def test_end_to_end_a_fifo_is_refused_without_blocking(tmp_path):
    st, bundle_dir = stage(tmp_path, cfg(), cfg(), capabilities=None)
    os.mkfifo(bundle_dir / "orchard-capabilities.json")
    r, d = run(st)
    assert r.returncode == 0, r.stderr
    assert d["path"] == "full-port"
    assert any("regular file" in p for p in d["capabilities"]["problems"]), d["capabilities"]


def test_end_to_end_an_oversized_file_is_refused(tmp_path):
    big = json.dumps({**caps(), "features": {"pad": "x" * (T.CAPABILITIES_MAX_BYTES + 10)}})
    st, _ = stage(tmp_path, cfg(), cfg(), capabilities=big)
    r, d = run(st)
    assert d["path"] == "full-port"
    assert any("bytes" in p for p in d["capabilities"]["problems"]), d["capabilities"]


def test_a_bundle_input_that_is_not_a_directory_exits_2(tmp_path):
    st, bundle_dir = stage(tmp_path, cfg(), cfg())
    shutil.rmtree(bundle_dir)
    r, _ = run(st)
    assert r.returncode == 2
    assert "nearest_bundle_dir" in r.stderr


@pytest.mark.parametrize("value", [3, [], {}, ""])
def test_a_bundle_input_of_the_wrong_type_exits_2(tmp_path, value):
    st, _ = stage(tmp_path, cfg(), cfg())
    tc = json.loads((st / "triage_config.json").read_text())
    tc["nearest_bundle_dir"] = value
    (st / "triage_config.json").write_text(json.dumps(tc))
    r, _ = run(st)
    assert r.returncode == 2
    assert "nearest_bundle_dir" in r.stderr
