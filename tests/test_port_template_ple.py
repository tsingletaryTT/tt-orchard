# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The per-layer embedding block (PLE) of the port generator (issue 696, slice 3): the third entry of the block-family registry.

`Qwen4ExpTextPLELayer` hashes the previous tokens into n-gram ids (integer arithmetic), looks the ids up in a large table, projects the result to
one key per residual stream and one shared value, gates the value by the normalised streams, and adds a dilated depthwise convolution. The
integer hashing and the table lookup run on the host, because the table is large (the real model's is over 100 GB) and the ids are integers; the
projections, the norms, the gate and the convolution run on the device. The reference comparison runs the generated `_B_ple` through the torch-backed
fake ttnn (tests/fake_ttnn_compute.py) against the real transformers module at tiny sizes with random weights. The fake's `sum`, `abs`, `sign`, `sqrt`
and `clamp` are torch calls. The comparison shows that the generated code reads the real tensor names, hashes and looks up the way the module does,
and builds the gate, the group norms and the dilated convolution in the module's order. It does not show that the device ops run on a board."""
from __future__ import annotations

import ast
import json
import re

import pytest
import torch

from orchard import port_template as pt
import fake_ttnn_compute as fake
from test_port_template import _load, import_port, tiny_qwen4

PLE_TEST = "tests/test_port_template_ple.py::test_generated_ple_matches_the_transformers_module"
TABLE = "model.layers.1.ple.ple_embedding.ngram_embedding"


# ---- the registry and the plan ---------------------------------------------------------------------------------------------------------------

def test_ple_is_a_registered_family_with_a_reference_test():
    fam = pt.BLOCK_FAMILIES["ple"]
    assert callable(fam.recognize) and callable(fam.emit) and fam.reference_test == PLE_TEST and callable(fam.helpers) and not fam.model_level


def test_the_ple_stub_is_gone_and_the_others_stay():
    text, p = tiny_qwen4()
    assert {s.function for s in p.stubs} == {"_layer_C", "_C_self_attn"}
    assert "ple" in p.summary()["generated"]
    assert len(pt.stub_lines(text)) == 2


def test_the_table_shards_stay_on_the_host_and_the_convolution_gets_taps():
    _text, p = tiny_qwen4()
    for k in range(4):
        assert p.placements[f"model.layers.{{i}}.ple.ple_embedding.ngram_embedding.shard_{k}.weight"] == "host"
    assert p.placements["model.layers.{i}.ple.conv1d.weight"] == "conv_taps"
    for norm in ("norm_key", "norm_query", "norm_conv"):
        assert p.placements[f"model.layers.{{i}}.ple.{norm}.weight"] == "norm_plus_one"
    assert p.placements["model.layers.{i}.ple.ple_embedding.layer_multipliers"] == "host"


def test_the_generated_function_carries_the_mark_and_no_table_goes_to_the_device():
    text, _p = tiny_qwen4()
    tree = ast.parse(text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_B_ple")
    src = ast.get_source_segment(text, fn)
    assert pt.GENERATED_BLOCK_MARK in src and pt.STUB_MARK not in src
    helper = ast.get_source_segment(text, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_ple_ngram_embeddings"))
    assert "_upload" not in helper and "ttnn." not in helper          # the hashing and the lookup are host code


def test_a_model_without_ple_gets_no_ple_code():
    from test_port_template import std_generate
    text, p = std_generate()
    assert "_ple" not in text and "ple" not in p.summary()["generated"]


def test_the_generated_port_still_passes_the_sheet():
    text, _ = tiny_qwen4()
    assert pt.sheet_problems(text) == []


# ---- the recognizer refuses what it cannot prove: the block stays a stub ----------------------------------------------------------------------

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


def _attrs(con, cls_name):
    return next(c for c in con["classes"] if c["class"] == cls_name)["variants"][0]["attrs"]


def _reshape(name, shape):
    def mutate(tensors):
        _tensor(tensors, name)["shape"] = shape
    return mutate


def _drop(name):
    return lambda tensors: tensors.__setitem__(slice(None), [t for t in tensors if t["name"] != name])


def _norm_variant(con):
    return next(c for c in con["classes"] if c["class"] == "Qwen4ExpTextRMSNorm")["variants"][0]


def _split_norm_conv_eps(con):
    """The convolution's norm gets its own variant with another epsilon: the three PLE norms then disagree."""
    variant = _norm_variant(con)
    cls = next(c for c in con["classes"] if c["class"] == "Qwen4ExpTextRMSNorm")
    cls["variants"].append({"attrs": {"eps": 1e-05, "group_size": 64}, "count": 1, "paths": ["model.layers.{i}.ple.norm_conv"]})
    variant["paths"] = [x for x in variant["paths"] if not x.endswith("norm_conv")]


def _drop_shard_two_and_stretch_the_last(tensors):
    """The table keeps its total rows (4096), so only the numbering gap can refuse it."""
    tensors[:] = [t for t in tensors if t["name"] != f"{TABLE}.shard_2.weight"]
    _tensor(tensors, f"{TABLE}.shard_3.weight")["shape"] = [2048, 16]


def _hidden_48(tensors):
    """Every tensor of the block agrees with hidden 48 (4 streams of 48 = 192), so only the tile guard can refuse it."""
    for t in tensors:
        n = t["name"]
        if not n.startswith("model.layers.1.ple."):
            continue
        if n.endswith(("key_proj.weight",)):
            t["shape"] = [192, 64]
        elif n.endswith("value_proj.weight"):
            t["shape"] = [48, 64]
        elif n.endswith(("norm_key.weight", "norm_query.weight", "norm_conv.weight")):
            t["shape"] = [192]
        elif n.endswith("conv1d.weight"):
            t["shape"] = [192, 1, 4]


REFUSALS = {
    "a key projection of the wrong height": dict(mutate_tensors=_reshape("model.layers.1.ple.key_proj.weight", [192, 64])),
    "a value projection of the wrong height": dict(mutate_tensors=_reshape("model.layers.1.ple.value_proj.weight", [32, 64])),
    "a convolution with the wrong number of channels": dict(mutate_tensors=_reshape("model.layers.1.ple.conv1d.weight", [192, 1, 4])),
    "a convolution that is not depthwise": dict(mutate_tensors=_reshape("model.layers.1.ple.conv1d.weight", [256, 2, 4])),
    "a norm weight of the wrong width": dict(mutate_tensors=_reshape("model.layers.1.ple.norm_key.weight", [128])),
    "no multipliers": dict(mutate_tensors=_drop("model.layers.1.ple.ple_embedding.layer_multipliers")),
    "multipliers of the wrong count": dict(mutate_tensors=_reshape("model.layers.1.ple.ple_embedding.layer_multipliers", [4])),
    "head offsets of the wrong count": dict(mutate_tensors=_reshape("model.layers.1.ple.ple_embedding.ngram_heads_offsets", [3])),
    "a missing table shard": dict(mutate_tensors=_drop("model.layers.1.ple.ple_embedding.ngram_embedding.shard_2.weight")),
    "a table narrower than a head": dict(mutate_tensors=lambda ts: [_tensor(ts, f"{TABLE}.shard_{k}.weight").update(shape=[1024, 8]) for k in range(4)]),
    "a table too short for the head offsets": dict(mutate_tensors=lambda ts: [_tensor(ts, f"{TABLE}.shard_{k}.weight").update(shape=[512, 16]) for k in range(4)]),
    "a convolution whose state length does not match the dilation": dict(mutate_construction=lambda con: _attrs(con, "Qwen4ExpTextPLELayer").update(short_conv_state_len=8)),
    "a hidden size that is not a tile multiple": dict(
        mutate_construction=lambda con: (_attrs(con, "Qwen4ExpTextPLELayer").update(hidden_size=48), _norm_variant(con).update(attrs={"eps": 1e-06, "group_size": 48})),
        mutate_tensors=_hidden_48),
    "a total vocabulary that is not a number": dict(mutate_construction=lambda con: _attrs(con, "Qwen4ExpTextNGramEmbedding").update(total_vocab_size="4062")),
    "a norm with a group that is not the stream width": dict(mutate_construction=lambda con: _norm_variant(con).update(attrs={"eps": 1e-06, "group_size": 32})),
    "norms that disagree on the epsilon": dict(mutate_construction=_split_norm_conv_eps),
    "table shards numbered with a gap": dict(mutate_tensors=_drop_shard_two_and_stretch_the_last),
    "a child module the generator does not know": dict(mutate_construction=lambda con: _attrs(con, "Qwen4ExpTextPLELayer")["children"].update(extra="Linear")),
    "no eos token": dict(mutate_construction=lambda con: _attrs(con, "Qwen4ExpTextNGramEmbedding").pop("eos_token_id")),
    "head counts that do not match the n-gram size": dict(mutate_construction=lambda con: _attrs(con, "Qwen4ExpTextNGramEmbedding").update(ngram_heads=5)),
}


@pytest.mark.parametrize("why", sorted(REFUSALS))
def test_the_recognizer_refuses_what_it_cannot_prove(why):
    _text, p = _generate(**REFUSALS[why])
    assert "_B_ple" in {s.function for s in p.stubs}, why


def test_the_unmutated_construction_is_recognized():
    """The refusals above start from this plan: without it they would pass for any reason."""
    _text, p = _generate()
    assert "_B_ple" not in {s.function for s in p.stubs}


# ---- the reference comparison -----------------------------------------------------------------------------------------------------------------

def _hf_ple(seed):
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    config_mod = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    cfg = _load("tiny-qwen4-config.json")
    for key in ("architectures", "model_type"):
        cfg.pop(key, None)
    config = config_mod.Qwen4ExpTextConfig(**cfg)
    torch.manual_seed(seed)
    ple = modeling.Qwen4ExpTextPLELayer(config, 1, 0)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in ple.named_parameters():
            if name.endswith("norm_key.weight") or name.endswith("norm_query.weight") or name.endswith("norm_conv.weight"):
                p.copy_((0.2 * torch.randn(p.shape, generator=g)).to(torch.bfloat16).float())
            elif name.endswith("conv1d.weight"):
                p.copy_((0.6 * torch.randn(p.shape, generator=g)).to(torch.bfloat16).float())
            else:
                p.copy_((0.4 * torch.randn(p.shape, generator=g)).to(torch.bfloat16).float())
    return ple.eval()


def _ple_checkpoint(tmp_path, ple):
    """The tiny checkpoint's layout: the n-gram table cut into four row shards, the integer buffers as they are, every other tensor from the module."""
    from safetensors.torch import save_file
    weights = {}
    for t in _load("tiny-qwen4-tensors.json"):
        n = t["name"]
        if not n.startswith("model.layers.1.ple."):
            weights[n] = (torch.zeros(t["shape"], dtype=torch.int64) if t["dtype"] != "BF16" else (0.1 * torch.randn(*t["shape"])).to(torch.bfloat16))
    state = {k: v.detach().clone() for k, v in ple.state_dict().items()}
    table = state.pop("ple_embedding.ngram_embedding.weight").to(torch.bfloat16)
    for k, part in enumerate(table.chunk(4, dim=0)):
        weights[f"{TABLE}.shard_{k}.weight"] = part.contiguous()
    for k, v in state.items():
        weights[f"model.layers.1.ple.{k}"] = v if v.dtype == torch.int64 else v.to(torch.bfloat16)
    snap = tmp_path / "snap"
    snap.mkdir()
    save_file(weights, str(snap / "model.safetensors"))
    return snap


def _inputs(seq=64):
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(2, 380, (seq,), generator=g)
    ids[[7, 8, 30]] = 1                                  # eos tokens: the segments of the n-gram context restart there
    x = (torch.randn(1, seq, 256, generator=g) * 0.8).to(torch.bfloat16)
    return ids.tolist(), x


def test_generated_ple_matches_the_transformers_module(monkeypatch, tmp_path):
    text, _p = tiny_qwen4()
    mod = import_port(monkeypatch, text, tmp_path)
    ple = _hf_ple(seed=3)
    snap = _ple_checkpoint(tmp_path, ple)
    model = mod.load(str(snap), fake.MeshDevice(1))
    ids, x_host = _inputs()
    with torch.no_grad():
        want = ple(x_host.float(), torch.tensor([ids]), None).float()
    x = fake.Tensor(x_host.reshape(1, 1, len(ids), 256).float(), layout="TILE")
    got = mod._B_ple(model, 1, x, ids).t.float().reshape(want.shape)
    scale = want.abs().max().item()
    err = (got - want).abs().max().item()
    assert scale > 0.2, f"the reference output is too small ({scale}) for the comparison to mean anything"
    assert err <= 0.05 * scale, f"max abs error {err} against scale {scale}"


def test_the_load_keeps_the_table_and_the_integer_buffers_on_the_host(monkeypatch, tmp_path):
    text, _p = tiny_qwen4()
    mod = import_port(monkeypatch, text, tmp_path)
    ple = _hf_ple(seed=3)
    model = mod.load(str(_ple_checkpoint(tmp_path, ple)), fake.MeshDevice(1))
    for k in range(4):
        assert f"{TABLE}.shard_{k}.weight" in model["host"] and f"{TABLE}.shard_{k}.weight" not in model["dev"]
    assert model["host"]["model.layers.1.ple.ple_embedding.layer_multipliers"].dtype == torch.int64
    taps = ple.conv1d.weight.detach().to(torch.bfloat16).float()
    for j in range(4):
        assert torch.equal(model["dev"][f"model.layers.1.ple.conv1d.weight.tap{j}"].t.reshape(-1), taps[:, 0, j])


# ---- mutations of the emitted text: each must turn the comparison red -----------------------------------------------------------------------

def _run_with(monkeypatch, tmp_path, old, new):
    text, _p = tiny_qwen4()
    assert old in text, old
    monkeypatch.setattr(pt, "render", lambda p: text.replace(old, new))
    test_generated_ple_matches_the_transformers_module(monkeypatch, tmp_path)


MUTATIONS = {
    "the xor of the hash becomes an add": ("torch.bitwise_xor(mixed, shifted[pos] * mult[pos])", "(mixed + shifted[pos] * mult[pos])"),
    "the context is padded with zeros instead of eos": ("torch.full((1, ngram_size - 1), eos, dtype=torch.long)", "torch.zeros((1, ngram_size - 1), dtype=torch.long)"),
    "the head offset is dropped": ("+ off.view(1, 1, -1)", "+ 0"),
    "the gate is not square-rooted": ("mag = ttnn.sqrt(ttnn.clamp(ttnn.abs(s), min=1e-6))", "mag = ttnn.abs(s)"),
    "the gate loses its sign": ("ttnn.multiply(ttnn.sign(s), mag)", "mag"),
    "the gate is not scaled by the width": ("s = ttnn.multiply(s, hidden ** -0.5)", "s = s"),
    "the convolution is not dilated": ("ttnn.slice(xpad, (0, 0, j * dilation, 0), (1, 1, j * dilation + seq, channels))",
                                       "ttnn.slice(xpad, (0, 0, j, 0), (1, 1, j + seq, channels))"),
    "the convolution output is not added to the gated value": ("return ttnn.add(gated, _dilated_causal_conv(", "return ttnn.add(normed, _dilated_causal_conv("),
    "the value is not shared by the streams": ("gated_parts.append(ttnn.multiply(value, gate))", "gated_parts.append(ttnn.multiply(ttnn.slice(value, (0, 0, 0, 0), (1, 1, seq, hidden)), gate if j == 0 else ttnn.multiply(gate, 0.0)))"),
}


@pytest.mark.parametrize("why", sorted(MUTATIONS))
def test_the_reference_comparison_fails_when_the_emitted_text_is_broken(monkeypatch, tmp_path, why):
    old, new = MUTATIONS[why]
    with pytest.raises(AssertionError, match="max abs error|too small"):
        _run_with(monkeypatch, tmp_path, old, new)
