# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The gated delta net block of the port generator (issue 696, slice 2): the second entry of the block-family registry.

The reference comparison (`test_generated_gated_delta_net_matches_the_transformers_module`) runs the generated `_A_linear_attn` and
`_B_linear_attn` through the torch-backed fake ttnn (tests/fake_ttnn_compute.py) against the real transformers `Qwen4ExpTextGatedDeltaNet` at tiny
sizes (hidden 64, 2 key heads, 6 value heads, head width 32, kernel 4) with random weights. The fake's `chunk_gated_delta_rule` is a torch stand-in
(the probes' copy of the reference algorithm), so the comparison shows that the generated code reads the real tensor names, builds the causal
convolution, the decay and beta, the layouts and head counts the op's documentation asks for, the gated norm and the output projection in the order the
reference does. It does not show that the device kernel, `ttnn.softplus` or the row-major concatenation in the convolution run on a board."""
from __future__ import annotations

import ast

import pytest
import torch

from orchard import port_template as pt
import fake_ttnn_compute as fake
from test_port_template import _load, import_port, tiny_qwen4

GDN_TEST = "tests/test_port_template_gdn.py::test_generated_gated_delta_net_matches_the_transformers_module"
GDN_NAMES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d", "norm", "A_log", "dt_bias")


# ---- the registry and the plan ---------------------------------------------------------------------------------------------------------------

def test_gated_delta_net_is_a_registered_family_with_a_reference_test():
    fam = pt.BLOCK_FAMILIES["gated_delta_net"]
    assert callable(fam.recognize) and callable(fam.emit) and fam.reference_test == GDN_TEST and callable(fam.helpers) and not fam.model_level


def test_the_linear_attention_stubs_are_gone_and_the_rest_stay():
    text, p = tiny_qwen4()
    assert {s.function for s in p.stubs} == {"_layer_C", "_C_self_attn"}
    assert len(pt.stub_lines(text)) == 2
    assert "gated_delta_net" in p.summary()["generated"]


def test_the_placements_name_the_bare_parameters_and_the_convolution_taps():
    _text, p = tiny_qwen4()
    assert p.placements["model.layers.{i}.linear_attn.A_log"] == "neg_exp"
    assert p.placements["model.layers.{i}.linear_attn.conv1d.weight"] == "conv_taps"
    assert p.placements["model.layers.{i}.linear_attn.norm.weight"] == "norm"
    assert p.placements["model.layers.{i}.linear_attn.dt_bias"] == "tile"


def test_the_generated_function_carries_the_generated_mark_and_reads_only_names_the_checkpoint_has():
    text, _p = tiny_qwen4()
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_A_linear_attn")
    src = ast.get_source_segment(text, fn)
    assert pt.GENERATED_BLOCK_MARK in src and pt.STUB_MARK not in src and "model.layers.{i}.linear_attn" in src
    helper = ast.get_source_segment(text, next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_gated_delta_net"))
    for name in GDN_NAMES:
        assert name in helper, name
    assert "in_proj.weight" not in text and "in_proj\"" not in helper           # the name the coder guessed in arm E10


def test_a_model_without_a_gated_delta_net_gets_no_family_code():
    from test_port_template import std_generate
    text, p = std_generate()
    assert "_gated_delta_net" not in text and "gated_delta_net" not in p.summary()["generated"]


def test_the_composed_layer_calls_the_block_on_the_mixed_input_and_the_block_says_it_adds_no_residual():
    text, _ = tiny_qwen4()
    tree = ast.parse(text)
    layer = ast.get_source_segment(text, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_layer_A"))
    block = ast.get_source_segment(text, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_A_linear_attn"))
    assert "_A_linear_attn(model, i, mixed, ids)" in layer and "without the residual add" in block


def test_the_generated_port_still_passes_the_sheet_and_the_cpu_check():
    text, _ = tiny_qwen4()
    assert pt.sheet_problems(text) == []


# ---- the recognizer refuses what it cannot prove: the block stays a stub -----------------------------------------------------------------------

def _generate(mutate_tensors=None, mutate_construction=None):
    tensors, con = _load("tiny-qwen4-tensors.json"), _load("tiny-qwen4-construction.json")
    if mutate_tensors:
        mutate_tensors(tensors)
    if mutate_construction:
        mutate_construction(con)
    return pt.generate(config=_load("tiny-qwen4-config.json"), construction=con, tensors=tensors,
                       inventory=_load("tiny-qwen4-ops-inventory.json"), excerpts=None)


def _tensor(tensors, name):
    return next(t for t in tensors if t["name"] == name)


def _gdn_attrs(con):
    cls = next(c for c in con["classes"] if c["class"] == "Qwen4ExpTextGatedDeltaNet")
    return cls["variants"][0]["attrs"]


def _norm_attrs(con):
    cls = next(c for c in con["classes"] if c["class"] == "Qwen4ExpTextRMSNormGated")
    return cls["variants"][0]["attrs"]


def _reshape(name, fn):
    def mutate(tensors):
        t = _tensor(tensors, name)
        t["shape"] = fn(t["shape"])
    return mutate


def _drop(name):
    """Remove the tensor of every layer (the plan keys tensors by name pattern, so one layer alone would leave the pattern in place)."""
    import re
    pat = re.sub(r"(?<=\.)\d+(?=\.|$)", "{i}", name)
    return lambda tensors: tensors.__setitem__(slice(None), [t for t in tensors if re.sub(r"(?<=\.)\d+(?=\.|$)", "{i}", t["name"]) != pat])


def _reshape_all(suffix, fn):
    """Reshape the tensor of every layer, so that the shapes agree with a changed construction and only the guard under test can refuse."""
    def mutate(tensors):
        for t in tensors:
            if "linear_attn." in t["name"] and t["name"].endswith(suffix):
                t["shape"] = fn(t["shape"])
    return mutate


def _both(*mutations):
    return lambda tensors: [m(tensors) for m in mutations]


REFUSALS = {
    "a projection of the wrong height": dict(mutate_tensors=_reshape("model.layers.0.linear_attn.in_proj_qkv.weight", lambda s: [s[0] + 32, s[1]])),
    "a convolution with the wrong number of channels": dict(mutate_tensors=_reshape("model.layers.0.linear_attn.conv1d.weight", lambda s: [s[0] - 32, 1, s[2]])),
    "a convolution that is not depthwise": dict(mutate_tensors=_reshape("model.layers.0.linear_attn.conv1d.weight", lambda s: [s[0], 2, s[2]])),
    "no decay parameter": dict(mutate_tensors=_drop("model.layers.0.linear_attn.A_log")),
    "no time-step bias": dict(mutate_tensors=_drop("model.layers.0.linear_attn.dt_bias")),
    "a norm weight of the wrong width": dict(mutate_tensors=_reshape("model.layers.0.linear_attn.norm.weight", lambda s: [s[0] + 32])),
    # The shapes below agree with the changed construction, so only the divisibility guard can refuse them.
    "value heads that the key heads do not divide": dict(
        mutate_construction=lambda con: _gdn_attrs(con).update(num_k_heads=4),
        mutate_tensors=_both(_reshape_all("in_proj_qkv.weight", lambda s: [448, s[1]]), _reshape_all("conv1d.weight", lambda s: [448, 1, s[2]]))),
    "a key head width that is not a tile multiple": dict(
        mutate_construction=lambda con: _gdn_attrs(con).update(head_k_dim=16),
        mutate_tensors=_both(_reshape_all("in_proj_qkv.weight", lambda s: [256, s[1]]), _reshape_all("conv1d.weight", lambda s: [256, 1, s[2]]))),
    "a convolution activation the sheet does not hold": dict(mutate_construction=lambda con: _gdn_attrs(con).update(activation="gelu")),
    "a gate activation the generator does not write": dict(mutate_construction=lambda con: _norm_attrs(con).update(activation="gelu")),
    "a child module the generator does not know": dict(mutate_construction=lambda con: _gdn_attrs(con)["children"].update(extra_proj="Linear")),
    "no epsilon": dict(mutate_construction=lambda con: [_norm_attrs(con).pop(k, None) for k in ("variance_epsilon", "eps", "epsilon")]),
}


@pytest.mark.parametrize("why", sorted(REFUSALS))
def test_the_recognizer_refuses_what_it_cannot_prove(why):
    _text, p = _generate(**REFUSALS[why])
    assert "_A_linear_attn" in {s.function for s in p.stubs}, why


def test_the_unmutated_construction_is_recognized():
    """The refusals above start from this plan: without it they would pass for any reason."""
    _text, p = _generate()
    assert "_A_linear_attn" not in {s.function for s in p.stubs}


# ---- the reference comparison -----------------------------------------------------------------------------------------------------------------

def _gdn_checkpoint(tmp_path):
    """Random weights at a scale where the block's output is of order one (the tolerance below is relative to that)."""
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(11)
    weights = {}
    for t in _load("tiny-qwen4-tensors.json"):
        if t["dtype"] != "BF16":
            weights[t["name"]] = torch.arange(int(torch.tensor(t["shape"]).prod()), dtype=torch.int64).reshape(t["shape"])
            continue
        w = torch.randn(*t["shape"], generator=g) * 0.3
        if t["name"].endswith("linear_attn.A_log"):
            w = torch.randn(*t["shape"], generator=g) * 0.5
        elif t["name"].endswith("linear_attn.dt_bias"):
            w = torch.randn(*t["shape"], generator=g) * 0.5
        elif t["name"].endswith("linear_attn.norm.weight"):
            w = 1.0 + 0.2 * torch.randn(*t["shape"], generator=g)
        elif t["name"].endswith("conv1d.weight"):
            w = torch.randn(*t["shape"], generator=g) * 0.6
        weights[t["name"]] = w.to(torch.bfloat16)
    snap = tmp_path / "snap"
    snap.mkdir()
    save_file(weights, str(snap / "model.safetensors"))
    return snap, weights


def _hf_gated_delta_net(weights, prefix):
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    config_mod = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    cfg = config_mod.Qwen4ExpTextConfig(hidden_size=64, linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=32,
                                        linear_value_head_dim=32, linear_conv_kernel_dim=4, output_gate_type="sigmoid", rms_norm_eps=1e-6,
                                        hidden_act="silu", num_hidden_layers=4,
                                        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    mod = modeling.Qwen4ExpTextGatedDeltaNet(cfg, 0)
    state = {k[len(prefix) + 1:]: v.float() for k, v in weights.items() if k.startswith(prefix + ".")}
    mod.load_state_dict(state)
    return mod.eval()


def test_generated_gated_delta_net_matches_the_transformers_module(monkeypatch, tmp_path):
    text, _p = tiny_qwen4()
    mod = import_port(monkeypatch, text, tmp_path)
    snap, weights = _gdn_checkpoint(tmp_path)
    model = mod.load(str(snap), fake.MeshDevice(1))
    seq = 64
    x_host = (torch.randn(1, seq, 64, generator=torch.Generator().manual_seed(3)) * 0.8).to(torch.bfloat16)
    x = fake.Tensor(x_host.reshape(1, 1, seq, 64).float(), layout="TILE")
    for layer, fn in ((0, mod._A_linear_attn), (1, mod._A_linear_attn), (2, mod._B_linear_attn)):
        prefix = f"model.layers.{layer}.linear_attn"
        ref = _hf_gated_delta_net(weights, prefix)
        with torch.no_grad():
            want = ref(x_host.float()).float()
        got = fn(model, layer, x, list(range(seq))).t.float().reshape(want.shape)
        scale = want.abs().max().item()
        err = (got - want).abs().max().item()
        assert scale > 0.2, f"layer {layer}: the reference output is too small ({scale}) for the comparison to mean anything"
        assert err <= 0.05 * scale, f"layer {layer}: max abs error {err} against scale {scale}"


def test_the_load_roles_put_minus_exp_of_a_log_and_one_tensor_per_convolution_tap_on_the_device(monkeypatch, tmp_path):
    text, _p = tiny_qwen4()
    mod = import_port(monkeypatch, text, tmp_path)
    snap, weights = _gdn_checkpoint(tmp_path)
    model = mod.load(str(snap), fake.MeshDevice(1))
    name = "model.layers.0.linear_attn"
    want = -weights[name + ".A_log"].float().exp()
    assert (model["dev"][name + ".A_log"].t.reshape(-1) - want).abs().max() <= 0.01 * want.abs().max()
    taps = weights[name + ".conv1d.weight"]
    assert name + ".conv1d.weight" not in model["dev"]
    for j in range(taps.shape[-1]):
        assert torch.equal(model["dev"][f"{name}.conv1d.weight.tap{j}"].t.reshape(-1), taps[:, 0, j].float())


# ---- mutations of the emitted text: each must turn the comparison red -----------------------------------------------------------------------

def _run_with(monkeypatch, tmp_path, old, new):
    text, _p = tiny_qwen4()
    assert old in text, old
    monkeypatch.setattr(pt, "render", lambda p: text.replace(old, new))
    test_generated_gated_delta_net_matches_the_transformers_module(monkeypatch, tmp_path)


MUTATIONS = {
    "the decay loses its sign (loaded as exp, not minus exp)": ("(-t.float().exp())", "(t.float().exp())"),
    "the convolution taps run in the wrong order": ("ttnn.slice(xpad, (0, 0, j, 0), (1, 1, j + seq, channels))",
                                                    "ttnn.slice(xpad, (0, 0, kernel - 1 - j, 0), (1, 1, kernel - 1 - j + seq, channels))"),
    "the convolution does not shift (no left padding)": ("torch.zeros(1, 1, kernel - 1, channels, dtype=torch.bfloat16)",
                                                         "torch.ones(1, 1, kernel - 1, channels, dtype=torch.bfloat16)"),
    "beta is not squashed": ("beta = ttnn.sigmoid(b)", "beta = b"),
    "the time-step bias is dropped": ("ttnn.softplus(ttnn.add(a, model[\"dev\"][prefix + \".dt_bias\"]))", "ttnn.softplus(a)"),
    "the output gate uses silu where the construction says sigmoid": ("z = ttnn.sigmoid(z)", "z = ttnn.silu(z)"),
    "the gate is not applied": ("o = ttnn.multiply(o, z)", "o = o"),
}


@pytest.mark.parametrize("why", sorted(MUTATIONS))
def test_the_reference_comparison_fails_when_the_emitted_text_is_broken(monkeypatch, tmp_path, why):
    old, new = MUTATIONS[why]
    with pytest.raises(AssertionError, match="max abs error|too small"):
        _run_with(monkeypatch, tmp_path, old, new)
