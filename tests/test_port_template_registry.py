# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The block-family registry of the port generator (issue 696, slice 1) and its first entry, the hyper-connections of the Qwen4 family.

A family is data plus a test: a class recognizer, an emitter and a reference test that compares the generated block with the transformers
module at tiny sizes. These tests check the registry's shape, the plan and stubs for the tiny Qwen4 rehearsal model, and (through the torch-backed
fake ttnn, no device) the generated prelude, mixer and per-layer connections against the transformers `Qwen4ExpTextGatedResidual`."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from orchard import port_template as pt
import fake_ttnn_compute as fake
from test_port_template import _load, import_port, tiny_qwen4

TESTS = Path(__file__).resolve().parent
HC_TEST = "tests/test_port_template_registry.py::test_generated_hyper_connections_match_the_transformers_module"


# ---- the registry --------------------------------------------------------------------------------------------------------------------------

def test_hyper_connection_is_the_first_registered_family():
    assert list(pt.BLOCK_FAMILIES)[0] == "hyper_connection"
    fam = pt.BLOCK_FAMILIES["hyper_connection"]
    assert callable(fam.recognize) and callable(fam.emit) and fam.reference_test == HC_TEST and fam.describe and fam.model_level


def test_every_family_names_a_reference_test_that_exists():
    """The guard at the layer that fails: a family without a reference comparison is a claim with no check."""
    for name, fam in pt.BLOCK_FAMILIES.items():
        path, _, test = fam.reference_test.partition("::")
        source = (TESTS.parent / path).read_text(encoding="utf-8")
        defined = {n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
        assert test in defined, f"family {name}: {fam.reference_test} is not a test in {path}"


def test_a_family_must_name_a_reference_test():
    with pytest.raises(ValueError, match="reference_test"):
        pt.register_block_family(pt.BlockFamily(name="no_test", recognize=lambda s: None, emit=lambda *a: [], reference_test="", describe="x"))


# ---- the plan: the hyper-connection stubs are gone, everything else stays -----------------------------------------------------------------------

def test_tiny_qwen4_keeps_only_the_blocks_the_registry_does_not_know():
    text, p = tiny_qwen4()
    stubs = {s.function for s in p.stubs}
    assert stubs == {"_layer_A", "_layer_B", "_B_ple", "_layer_C", "_C_self_attn"}          # the gated delta net is the second family (slice 2)
    assert len(pt.stub_lines(text)) == 5
    assert p.placements["model.layers.{i}.attn_hyper_connection.hc_norm.weight"] == "norm_plus_one"
    assert p.placements["model.hyper_connection_mixer.hc_norm.weight"] == "norm_plus_one"
    s = p.summary()
    assert "hyper_connection" in s["generated"]
    layer_stub = next(x for x in s["stubs"] if x["function"] == "_layer_A")
    assert "attn_hyper_connection" not in layer_stub["reason"].split("stubs:")[-1]          # generated parts are listed in the stub's body, not as stubs


def test_the_layer_stub_lists_the_generated_connection_with_its_return_value_and_the_inject_helper():
    text, _ = tiny_qwen4()
    body = text[text.index("def _layer_A("):text.index("def load(")]
    assert "_A_attn_hyper_connection(model, i, x, ids)" in body and "(mixed, hyper_input, injection_weights)" in body
    assert "_hyper_connection_inject(" in body


def test_generated_connection_functions_carry_the_generated_mark_and_no_stub_marker():
    text, _ = tiny_qwen4()
    for name in ("_A_attn_hyper_connection", "_B_mlp_hyper_connection", "_model_hyper_connection_mixer", "_model_prelude", "_hyper_connection"):
        src = ast.get_source_segment(text, next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name))
        assert pt.GENERATED_BLOCK_MARK in src and pt.STUB_MARK not in src


def test_a_model_without_hyper_connections_gets_no_family_code():
    from test_port_template import std_generate
    text, p = std_generate()
    assert "_hyper_connection" not in text and "hyper_connection" not in p.summary()["generated"]


def test_a_gated_residual_with_a_bad_shape_stays_a_stub():
    """A recognizer that cannot prove the shapes returns nothing, so the block is a stub as before (no silent default)."""
    tensors = _load("tiny-qwen4-tensors.json")
    for t in tensors:
        if t["name"].endswith("attn_hyper_connection.input_mix_weight_down.weight"):
            t["shape"] = [t["shape"][0] + 1, t["shape"][1]]
    _text, p = pt.generate(config=_load("tiny-qwen4-config.json"), construction=_load("tiny-qwen4-construction.json"), tensors=tensors,
                           inventory=_load("tiny-qwen4-ops-inventory.json"), excerpts=None)
    assert "_A_attn_hyper_connection" in {s.function for s in p.stubs}


# ---- the reference comparison -------------------------------------------------------------------------------------------------------------------

def _random_checkpoint(tmp_path):
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(7)
    weights = {}
    for t in _load("tiny-qwen4-tensors.json"):
        if t["dtype"] == "BF16":
            weights[t["name"]] = (torch.randn(*t["shape"], generator=g) * 0.1).to(torch.bfloat16)
        else:
            weights[t["name"]] = torch.arange(int(torch.tensor(t["shape"]).prod()), dtype=torch.int64).reshape(t["shape"])
    snap = tmp_path / "snap"
    snap.mkdir()
    save_file(weights, str(snap / "model.safetensors"))
    return snap, weights


def _hf_gated_residual(weights, prefix, use_combine):
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    config_mod = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    cfg = config_mod.Qwen4ExpTextConfig(hidden_size=64, hc_count=4, hc_lowrank=8, rms_norm_eps=1e-6)
    mod = modeling.Qwen4ExpTextGatedResidual(cfg, use_combine=use_combine)
    state = {k[len(prefix) + 1:]: v.float() for k, v in weights.items() if k.startswith(prefix + ".")}
    mod.load_state_dict(state)
    return mod.eval()


def test_generated_hyper_connections_match_the_transformers_module(monkeypatch, tmp_path):
    text, _p = tiny_qwen4()
    mod = import_port(monkeypatch, text, tmp_path)
    snap, weights = _random_checkpoint(tmp_path)
    model = mod.load(str(snap), fake.MeshDevice(1))
    seq, hc, hidden = 32, 4, 64
    x_host = (torch.randn(1, seq, hc * hidden, generator=torch.Generator().manual_seed(3)) * 0.5).to(torch.bfloat16)
    x = fake.Tensor(x_host.reshape(1, 1, seq, hc * hidden), layout="TILE")
    ids = list(range(seq))

    def close(got, want, name):
        got = got.t.float().reshape(want.shape)
        err = (got - want.float()).abs().max().item()
        scale = want.float().abs().max().item()
        assert err <= 0.03 * max(scale, 1.0), f"{name}: max abs error {err} against scale {scale}"

    for layer, prefix, fn in ((0, "model.layers.0.attn_hyper_connection", mod._A_attn_hyper_connection),
                              (1, "model.layers.1.mlp_hyper_connection", mod._B_mlp_hyper_connection)):
        ref = _hf_gated_residual(weights, prefix, True)
        with torch.no_grad():
            want_mixed, want_input, want_inj = ref(x_host.float())
        mixed, hyper_input, inj = fn(model, layer, x, ids)
        close(mixed, want_mixed, prefix + " mixed")
        close(hyper_input, want_input, prefix + " hyper_input")
        close(inj, want_inj, prefix + " injection")
        block_out = fake.Tensor((torch.randn(1, 1, seq, hidden, generator=torch.Generator().manual_seed(5)) * 0.5).to(torch.bfloat16))
        got = mod._hyper_connection_inject(block_out, hyper_input, inj, hc, hidden)
        want = want_input + (block_out.t.float().reshape(1, seq, hidden).unsqueeze(-2) * want_inj.unsqueeze(-1)).flatten(-2)
        close(got, want, prefix + " injected")
    mixer_ref = _hf_gated_residual(weights, "model.hyper_connection_mixer", False)
    with torch.no_grad():
        want = mixer_ref(x_host.float())
    close(mod._model_hyper_connection_mixer(model, x), want, "mixer")
    emb = fake.Tensor((torch.randn(1, 1, seq, hidden, generator=torch.Generator().manual_seed(9))).to(torch.bfloat16))
    close(mod._model_prelude(model, emb, ids), emb.t.float().reshape(1, seq, hidden).repeat(1, 1, hc), "prelude")   # the reference: hidden.repeat(1, 1, hc_count)


def test_the_reference_comparison_fails_when_a_stream_weight_is_dropped(monkeypatch, tmp_path):
    """A mutation of the emitted text (the injection weights stop scaling by 2) must turn the comparison red."""
    text, _p = tiny_qwen4()
    broken = text.replace("ttnn.multiply(ttnn.sigmoid(ttnn.multiply(_linear(model, prefix + \".block_inject_weight\", normed), 1.0 / hc)), 2.0)",
                          "ttnn.sigmoid(ttnn.multiply(_linear(model, prefix + \".block_inject_weight\", normed), 1.0 / hc))")
    assert broken != text
    with pytest.raises(AssertionError, match="injection"):
        monkeypatch.setattr(pt, "render", lambda p: broken)
        test_generated_hyper_connections_match_the_transformers_module(monkeypatch, tmp_path)


def test_the_layer_stub_request_tells_the_coder_how_to_use_the_generated_connections():
    """What the coder reads for `_layer_A`: the connection call, what it returns and the inject helper (its remaining job is the block between)."""
    from orchard import op_cheats, stub_prompts as sp
    text, p = tiny_qwen4()
    stub = next(x for x in p.stubs if x.function == "_layer_A")
    user = sp.stub_request(text, stub, sheet=op_cheats.load_sheet(op_cheats.TTNN_SHEET), excerpts=None)["user"]
    assert "_A_attn_hyper_connection(model, i, x, ids)" in user and "_hyper_connection_inject(block_out, hyper_input, injection_weights, 4, 64)" in user
