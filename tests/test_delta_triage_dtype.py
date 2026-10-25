# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A fine-tune that keeps small tensors in F32 where the nearest model has BF16 is still a weights-only delta (issue 29).

sabicap/sabi-noema against Qwen3.8-27B: 96 tensors (48 `A_log` and 48 linear-attention norm weights, all small) are F32 in the
fine-tune and BF16 in the base; every shape matches. On 2026-10-24 the nearest model's 2-chip bundle served it with the weights
as they are: ready in 212 s, 31 of 32 tokens equal to the CPU reference, coherent, and all eight prompts of the suite clean.
The rule that sent it to full-port stays for everything that evidence does not cover: a quantized or integer dtype, a large
tensor, and a shape change.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"


def template():
    spec = importlib.util.spec_from_file_location("delta_triage_dtype_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def info(dtype, shape, shard="m.safetensors"):
    return {"dtype": dtype, "shape": list(shape), "shard": shard}


def compare(mt, bt):
    return template().compare_tensor_maps(mt, bt, mshards=["m"], bshards=["b"], merr=[], berr=[], model_path="m", base_path="b")


CFG = {"decisive_keys_differing": [], "text_key_values": {}}
BIG = [8192, 8192]


def pair(name, model_dtype, base_dtype, shape):
    return {name: info(model_dtype, shape)}, {name: info(base_dtype, shape)}


def reasons_for(model_dtype, base_dtype, shape):
    t = template()
    mt, bt = pair("model.layers.0.linear_attn.A_log", model_dtype, base_dtype, shape)
    return t.decide_path(CFG, compare(mt, bt))


def test_small_f32_against_bf16_is_weights_only():
    assert reasons_for("F32", "BF16", [48]) == []


def test_small_bf16_against_f32_is_weights_only():
    assert reasons_for("BF16", "F32", [128]) == []


def test_f16_against_bf16_on_a_small_tensor_is_weights_only():
    assert reasons_for("F16", "BF16", [64]) == []


def test_a_large_tensor_in_another_float_dtype_still_forces_a_full_port():
    r = reasons_for("F32", "BF16", BIG)
    assert len(r) == 1 and "dtype F32" in r[0] and "BF16" in r[0]


def test_the_size_limit_is_inclusive():
    t = template()
    n = t.DTYPE_TOLERANCE_ELEMENTS
    assert reasons_for("F32", "BF16", [n]) == []
    assert len(reasons_for("F32", "BF16", [n + 1])) == 1


def test_a_quantized_dtype_forces_a_full_port_even_when_small():
    for dtype in ("F8_E4M3", "I8", "U8", "I32", "BOOL"):
        r = reasons_for(dtype, "BF16", [48])
        assert len(r) == 1 and dtype in r[0], dtype


def test_a_base_that_is_quantized_forces_a_full_port_too():
    assert len(reasons_for("BF16", "F8_E4M3", [48])) == 1


def test_a_shape_difference_is_reported_whatever_the_dtype():
    t = template()
    mt, bt = {"x": info("F32", [48])}, {"x": info("BF16", [49])}
    r = t.decide_path(CFG, compare(mt, bt))
    assert any("has shape" in x for x in r)


def test_one_untolerated_difference_among_tolerated_ones_still_blocks():
    t = template()
    mt = {"a": info("F32", [48]), "b": info("F32", BIG), "c": info("F32", [8])}
    bt = {"a": info("BF16", [48]), "b": info("BF16", BIG), "c": info("BF16", [8])}
    r = t.decide_path(CFG, compare(mt, bt))
    assert len(r) == 1 and "tensor b " in r[0]


def test_the_comparison_marks_each_difference_tolerated_or_not():
    mt = {"a": info("F32", [48]), "b": info("F32", BIG)}
    bt = {"a": info("BF16", [48]), "b": info("BF16", BIG)}
    d = {x["name"]: x["tolerated"] for x in compare(mt, bt)["dtype_diffs_normalized"]}
    assert d == {"a": True, "b": False}


def test_a_tolerated_difference_leaves_a_hazard_that_names_what_was_and_was_not_tested():
    t = template()
    mt = {f"l{i}.A_log": info("F32", [48]) for i in range(3)} | {"same": info("BF16", [4])}
    bt = {f"l{i}.A_log": info("BF16", [48]) for i in range(3)} | {"same": info("BF16", [4])}
    h = t.tolerated_dtype_hazards(compare(mt, bt), lambda n: f"stages/0/evidence/{n}")
    assert len(h) == 1 and h[0]["area"] == "other"
    assert h[0]["evidence"] == ["stages/0/evidence/tensor-compare.json"]
    text = h[0]["finding"]
    assert "3 tensors" in text and "F32" in text and "BF16" in text
    assert "sabi-noema" in text and "31 of 32" in text
    assert "l0.A_log" in text


def test_no_hazard_when_every_dtype_matches():
    t = template()
    mt = {"x": info("BF16", [4])}
    assert t.tolerated_dtype_hazards(compare(mt, dict(mt)), lambda n: n) == []


def test_no_hazard_for_a_difference_that_is_not_tolerated():
    t = template()
    mt, bt = {"x": info("F32", BIG)}, {"x": info("BF16", BIG)}
    assert t.tolerated_dtype_hazards(compare(mt, bt), lambda n: n) == []


def test_the_tensors_finding_still_counts_every_dtype_difference():
    t = template()
    mt = {"a": info("F32", [48]), "b": info("F32", BIG)}
    bt = {"a": info("BF16", [48]), "b": info("BF16", BIG)}
    assert "2 in dtype" in t.tensors_finding(compare(mt, bt))


def test_the_class_is_weights_only_for_the_sabi_noema_shape_of_difference():
    t = template()
    mt = {"w": info("BF16", BIG), "n": info("F32", [128])}
    bt = {"w": info("BF16", BIG), "n": info("BF16", [128])}
    tens = compare(mt, bt)
    reasons = t.decide_path(CFG, tens)
    assert t.decide_class(CFG, tens, reasons, {"problems": [], "sidecars": []}) == "weights-only"


def test_a_dtype_difference_on_a_tensor_whose_shape_also_changed_is_not_tolerated():
    mt, bt = {"x": info("F32", [48])}, {"x": info("BF16", [49])}
    assert [x["tolerated"] for x in compare(mt, bt)["dtype_diffs_normalized"]] == [False]


def _flat(text):
    return " ".join(text.split())


def test_the_stage_0_skill_and_the_template_docstring_state_the_dtype_exception():
    """The skill is in the stage 0 agent's system message, so it must not say the old rule (Copilot, PR 58)."""
    skill = _flat((TEMPLATE.parents[1] / "delta-triage.md").read_text())
    doc = _flat(template().__doc__)
    for text in (skill, doc):
        assert "BF16, F16 or F32" in text and "equal shape" in text and "1,048,576" in text, text[:200]
    assert "same shape and dtype" not in skill and "same shape and dtype" not in doc
