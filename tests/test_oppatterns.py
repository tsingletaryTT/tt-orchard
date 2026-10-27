# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Subgraph recognition over the ordered op list (orchard/oppatterns.py).

The real-trace tests read a fixture made by the trace template on a tiny Qwen3 (torch.export on the meta device). The
synthetic tests build the same record shape by hand so each rule can be broken one condition at a time."""
import json
from pathlib import Path

import pytest

from orchard import oppatterns as op

FIX = Path(__file__).resolve().parent / "fixtures" / "ops-inventory" / "raw-qwen3-tiny-seq.json"
NAMES = {"rms_norm", "layer_norm", "swiglu", "transformer.scaled_dot_product_attention",
         "experimental.rotary_embedding_hf", "experimental.rotary_embedding_llama"}


def n(ident, opname, ins=(), const=(), module="model.layers.0.x"):
    return {"id": ident, "op": f"aten.{opname}", "in": list(ins), "module": module, "shape": [1, 8, 32],
            "const": list(const)}


def rms(prefix="r", x="x", eps=1e-6, exp=2, dim=(-1,), weight=True, module="model.layers.0.norm", cast=True):
    """The seven records of a Hugging Face RMSNorm, reading node `x`."""
    p = prefix
    recs = [n(f"{p}_pow", "pow.Tensor_Scalar", [x], [exp], module),
            n(f"{p}_mean", "mean.dim", [f"{p}_pow"], [list(dim), True], module),
            n(f"{p}_add", "add.Tensor", [f"{p}_mean"], [eps], module),
            n(f"{p}_rsqrt", "rsqrt.default", [f"{p}_add"], [], module),
            n(f"{p}_norm", "mul.Tensor", [x, f"{p}_rsqrt"], [], module)]
    if weight:
        last = f"{p}_norm"
        if cast:
            recs.append(n(f"{p}_cast", "to.dtype", [last], [], module))
            last = f"{p}_cast"
        recs.append(n(f"{p}_w", "mul.Tensor", [f"{p}_weight", last], [], module))
    return recs


def kinds(res):
    return {f["pattern"]: f for f in res["found"]}


# --- the real trace ------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    raw = json.loads(FIX.read_text())
    return raw, op.recognise(raw["sequence"], NAMES, raw["layers_in_config"])


def test_the_fixture_is_a_real_trace_with_data_flow(real):
    raw, _ = real
    assert raw["layers_in_config"] == 4 and len(raw["sequence"]) > 100
    ids = {s["id"] for s in raw["sequence"]}
    assert any(i in ids for s in raw["sequence"] for i in s["in"])


def test_rmsnorm_is_found_four_times_in_layer_zero_and_once_outside(real):
    _, res = real
    f = kinds(res)["rms_norm"]
    assert (f["occurrences"], f["in_layer"], f["outside"]) == (5, 4, 1)
    assert f["scaled_count"] == 4 * 4 + 1          # four per layer, four layers, plus the final norm
    assert f["ttnn"] == "rms_norm" and f["why"] == "pattern"
    assert f["nodes"] == 6                          # pow, mean, add, rsqrt, mul, weight mul
    assert f["ops"] == ["aten.add.Tensor", "aten.mean.dim", "aten.mul.Tensor", "aten.pow.Tensor_Scalar",
                        "aten.rsqrt.default"]


def test_silu_mul_and_rotary_are_found_and_sdpa_is_not_because_the_trace_has_the_fused_op(real):
    _, res = real
    f = kinds(res)
    assert f["silu_mul"]["occurrences"] == 1 and f["silu_mul"]["nodes"] == 2 and f["silu_mul"]["ttnn"] == "swiglu"
    assert f["rotary"]["occurrences"] == 2 and f["rotary"]["nodes"] == 7       # q and k
    assert f["rotary"]["ttnn"] == "experimental.rotary_embedding_hf"
    assert "sdpa" not in f and "layer_norm" not in f


def test_a_group_reports_the_shape_of_its_anchor_and_a_norm_its_epsilon(real):
    _, res = real
    f = kinds(res)
    assert f["rms_norm"]["eps"] == 1e-6 and f["rms_norm"]["shape"] == [1, 8, 1]
    assert f["silu_mul"]["eps"] is None and f["rotary"]["eps"] is None
    assert f["rotary"]["shape"] == [1, 2, 8, 16]


def test_modules_are_listed_without_more_than_three(real):
    _, res = real
    assert 1 <= len(kinds(res)["rms_norm"]["modules"]) <= 3


def test_nothing_is_found_without_a_sequence():
    assert op.recognise([], NAMES, 4) == {"found": [], "unavailable": []}
    assert op.recognise(None, NAMES) == {"found": [], "unavailable": []}


# --- RMSNorm, one broken condition at a time -----------------------------------------------------------------------

def test_a_synthetic_rmsnorm_matches():
    f = kinds(op.recognise(rms(), NAMES))["rms_norm"]
    assert f["occurrences"] == 1 and f["nodes"] == 6


def test_the_weight_multiply_is_optional():
    assert kinds(op.recognise(rms(weight=False), NAMES))["rms_norm"]["nodes"] == 5


def test_casts_between_the_steps_are_looked_through():
    recs = rms(cast=False)
    recs.insert(1, n("c1", "to.dtype", ["r_pow"]))
    recs[2]["in"] = ["c1"]
    assert "rms_norm" in kinds(op.recognise(recs, NAMES))


@pytest.mark.parametrize("change", [
    {"exp": 3}, {"exp": 1}, {"eps": 0.5}, {"eps": 0}, {"dim": (0,)}, {"dim": (1, 2)},
])
def test_a_changed_constant_is_not_an_rmsnorm(change):
    assert "rms_norm" not in kinds(op.recognise(rms(**change), NAMES))


def test_pow_of_two_as_a_float_matches():
    assert "rms_norm" in kinds(op.recognise(rms(exp=2.0), NAMES))


def test_square_and_x_times_x_are_squares_too():
    for variant in (n("r_pow", "square.default", ["x"]), n("r_pow", "mul.Tensor", ["x", "x"])):
        recs = rms()
        recs[0] = variant
        assert "rms_norm" in kinds(op.recognise(recs, NAMES))


def test_the_normalising_multiply_must_read_the_same_x():
    recs = rms()
    recs[4]["in"] = ["other", "r_rsqrt"]
    assert "rms_norm" not in kinds(op.recognise(recs, NAMES))


def test_a_chain_missing_a_step_is_not_an_rmsnorm():
    for drop in range(0, 4):
        recs = rms()
        del recs[drop]
        assert "rms_norm" not in kinds(op.recognise(recs, NAMES)), drop


def test_rsqrt_without_an_epsilon_add_is_not_an_rmsnorm():
    recs = rms()
    recs[3]["in"] = ["r_mean"]
    assert "rms_norm" not in kinds(op.recognise(recs, NAMES))


def test_two_norms_are_two_occurrences_and_the_layer_count_scales_only_the_ones_in_a_layer():
    recs = rms("a", module="model.layers.0.input_layernorm") + rms("b", x="x2", module="model.norm")
    f = kinds(op.recognise(recs, NAMES, layers_in_config=10))["rms_norm"]
    assert (f["occurrences"], f["in_layer"], f["outside"], f["scaled_count"]) == (2, 1, 1, 11)
    f = kinds(op.recognise(recs, NAMES))["rms_norm"]
    assert f["scaled_count"] == 2


def test_a_bool_is_not_the_exponent_two_or_an_epsilon():
    recs = rms(exp=True)
    assert "rms_norm" not in kinds(op.recognise(recs, NAMES))


# --- LayerNorm -----------------------------------------------------------------------------------------------------

def layer_norm(bias=True):
    m = "model.layers.0.ln"
    recs = [n("mu", "mean.dim", ["x"], [[-1], True], m), n("c", "sub.Tensor", ["x", "mu"], [], m)]
    recs += [n("l_pow", "pow.Tensor_Scalar", ["c"], [2], m), n("l_mean", "mean.dim", ["l_pow"], [[-1], True], m),
             n("l_add", "add.Tensor", ["l_mean"], [1e-5], m), n("l_rsqrt", "rsqrt.default", ["l_add"], [], m),
             n("l_norm", "mul.Tensor", ["c", "l_rsqrt"], [], m), n("l_w", "mul.Tensor", ["w", "l_norm"], [], m)]
    if bias:
        recs.append(n("l_b", "add.Tensor", ["l_w", "b"], [], m))
    return recs


def test_a_centred_normalisation_is_a_layernorm_and_not_an_rmsnorm():
    res = kinds(op.recognise(layer_norm(), NAMES))
    assert "rms_norm" not in res
    assert res["layer_norm"]["nodes"] == 9 and res["layer_norm"]["ttnn"] == "layer_norm"


def test_the_bias_add_is_optional():
    assert kinds(op.recognise(layer_norm(bias=False), NAMES))["layer_norm"]["nodes"] == 8


def test_a_sub_of_something_that_is_not_the_mean_of_x_is_an_rmsnorm_of_the_difference():
    recs = layer_norm()
    recs[0]["in"] = ["y"]
    res = kinds(op.recognise(recs, NAMES))
    assert "layer_norm" not in res and "rms_norm" in res


def test_variance_form_of_layernorm():
    m = "model.layers.0.ln"
    recs = [n("mu", "mean.dim", ["x"], [[-1], True], m), n("c", "sub.Tensor", ["x", "mu"], [], m),
            n("v", "var.correction", ["x"], [[-1], 0, True], m), n("v_add", "add.Tensor", ["v"], [1e-5], m),
            n("v_rsqrt", "rsqrt.default", ["v_add"], [], m), n("v_norm", "mul.Tensor", ["c", "v_rsqrt"], [], m)]
    res = kinds(op.recognise(recs, NAMES))
    assert res["layer_norm"]["nodes"] == 5        # var, add, rsqrt, the centring sub and the normalising mul
    recs[5]["in"] = ["z", "v_rsqrt"]
    assert "layer_norm" not in kinds(op.recognise(recs, NAMES))


# --- SiLU gate multiply --------------------------------------------------------------------------------------------

def test_silu_then_multiply_matches():
    recs = [n("a", "linear.default", ["x"]), n("s", "silu.default", ["a"]), n("b", "linear.default", ["x"]),
            n("m", "mul.Tensor", ["s", "b"])]
    f = kinds(op.recognise(recs, NAMES))["silu_mul"]
    assert f["nodes"] == 2 and f["ttnn"] == "swiglu"


def test_silu_without_a_multiply_or_with_a_multiply_by_itself_is_not_silu_mul():
    recs = [n("a", "linear.default", ["x"]), n("s", "silu.default", ["a"]), n("t", "add.Tensor", ["s", "b"])]
    assert "silu_mul" not in kinds(op.recognise(recs, NAMES))
    recs = [n("a", "linear.default", ["x"]), n("s", "silu.default", ["a"]), n("m", "mul.Tensor", ["s", "s"])]
    assert "silu_mul" not in kinds(op.recognise(recs, NAMES))


def test_silu_through_a_cast_still_matches():
    recs = [n("s", "silu.default", ["a"]), n("c", "to.dtype", ["s"]), n("m", "mul.Tensor", ["c", "b"])]
    assert "silu_mul" in kinds(op.recognise(recs, NAMES))


# --- unfused attention ---------------------------------------------------------------------------------------------

def attention(mask=True, scale="mul", second_matmul=True):
    recs = [n("t", "transpose.int", ["k"]), n("qk", "matmul.default", ["q", "t"]),
            n("sc", f"{scale}.Tensor", ["qk"], [0.25])]
    last = "sc"
    if mask:
        recs.append(n("mk", "add.Tensor", ["sc", "mask"]))
        last = "mk"
    recs.append(n("sm", "softmax.int", [last], [-1]))
    if second_matmul:
        recs.append(n("av", "matmul.default", ["sm", "v"]))
    return recs


@pytest.mark.parametrize("kw", [{}, {"mask": False}, {"scale": "div"}])
def test_unfused_attention_matches(kw):
    f = kinds(op.recognise(attention(**kw), NAMES))["sdpa"]
    assert f["ttnn"] == "transformer.scaled_dot_product_attention"
    # softmax, both matmuls, the scale and the mask add; the transpose of k is a pass-through and is not counted
    assert f["nodes"] == 5 - (not kw.get("mask", True))


def test_softmax_without_the_two_matmuls_is_not_attention():
    assert "sdpa" not in kinds(op.recognise(attention(second_matmul=False), NAMES))
    recs = [n("sm", "softmax.int", ["x"], [-1]), n("av", "matmul.default", ["sm", "v"])]
    assert "sdpa" not in kinds(op.recognise(recs, NAMES))


def test_scores_and_softmax_without_a_second_matmul_are_not_attention():
    recs = [n("qk", "matmul.default", ["q", "k"]), n("sm", "softmax.int", ["qk"], [-1]),
            n("t", "contiguous.default", ["sm"])]
    assert "sdpa" not in kinds(op.recognise(recs, NAMES))


def test_a_reshape_between_softmax_and_the_second_matmul_is_looked_through():
    recs = attention(second_matmul=False) + [n("vw", "view.default", ["sm"], [[1, 2]]), n("av", "matmul.default", ["vw", "v"])]
    assert "sdpa" in kinds(op.recognise(recs, NAMES))


# --- rotary embedding ----------------------------------------------------------------------------------------------

def rotary(lo_end=8, dim=3, swap=False):
    recs = [n("mc", "mul.Tensor", ["x", "cos"]), n("hi", "slice.Tensor", ["x"], [dim, 8, 99]),
            n("lo", "slice.Tensor", ["x"], [dim, 0, lo_end]), n("ng", "neg.default", ["hi"]),
            n("cat", "cat.default", ["lo", "ng"] if swap else ["ng", "lo"], [-1]),
            n("ms", "mul.Tensor", ["cat", "sin"]), n("ad", "add.Tensor", ["mc", "ms"])]
    return recs


def test_rotate_half_form_matches_with_seven_nodes():
    f = kinds(op.recognise(rotary(), NAMES))["rotary"]
    assert f["nodes"] == 7 and f["ttnn"] == "experimental.rotary_embedding_hf"


@pytest.mark.parametrize("kw", [{"lo_end": 7}, {"swap": True}])
def test_a_rotation_that_is_not_rotate_half_is_not_recognised(kw):
    assert "rotary" not in kinds(op.recognise(rotary(**kw), NAMES))


def test_the_plain_product_must_read_the_same_x():
    recs = rotary()
    recs[0]["in"] = ["y", "cos"]
    assert "rotary" not in kinds(op.recognise(recs, NAMES))


def test_a_cat_on_another_dimension_or_of_three_parts_is_not_rotate_half():
    recs = rotary()
    recs[4]["const"] = [1]
    assert "rotary" not in kinds(op.recognise(recs, NAMES))
    recs = rotary()
    recs[4]["in"] = ["ng", "lo", "lo"]
    assert "rotary" not in kinds(op.recognise(recs, NAMES))


def test_the_preferred_rotary_name_that_exists_is_chosen():
    f = kinds(op.recognise(rotary(), {"experimental.rotary_embedding_llama", "experimental.rotary_embedding"}))["rotary"]
    assert f["ttnn"] == "experimental.rotary_embedding_llama"


# --- the dump decides ----------------------------------------------------------------------------------------------

def test_a_group_whose_ttnn_op_is_not_in_the_dump_is_unavailable_and_not_found():
    res = op.recognise(rms(), {"add"})
    assert res["found"] == []
    assert res["unavailable"] == [{"pattern": "rms_norm", "wanted": ["rms_norm"], "occurrences": 1}]


def test_every_pattern_names_at_least_one_ttnn_op_and_the_names_exist_in_the_shipped_dump():
    dump = json.loads((Path(op.__file__).resolve().parent / "data" / "ttnn_ops-0.80.0.json").read_text())["names"]
    for kind, wanted in op.PATTERN_TTNN.items():
        assert wanted and any(w in dump for w in wanted), kind


def test_malformed_records_are_skipped_not_fatal():
    recs = [{"id": 1}, "x", {"op": "aten.add.Tensor"}] + rms()
    assert "rms_norm" in kinds(op.recognise(recs, NAMES))
