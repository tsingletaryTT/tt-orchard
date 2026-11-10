# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Whole decoder layers of the Qwen4 family written by the port generator (issue 696, slice 3): `_layer_A` (layers 0 and 2: attention connection, gated
delta net, MLP connection, MLP) and `_layer_B` (layer 1: the same behind a PLE block). The reference comparison runs the generated layer through the
torch-backed fake ttnn (tests/fake_ttnn_compute.py) against the real transformers `Qwen4ExpTextDecoderLayer` at tiny sizes with random weights (the layer's
MoE block is swapped for the dense `Qwen4ExpTextMLP`, as the tiny checkpoint's construction records). It shows that the generated composition (the order of
the PLE add, the two connections, the sublayers and the two injections) is the reference's. It does not show that the device ops run on a board."""
from __future__ import annotations

import pytest
import torch

from orchard import port_template as pt
import fake_ttnn_compute as fake
from test_port_template import _load, import_port, tiny_qwen4
from test_port_template_ple import TABLE, _generate

LAYER_TEST = pt.LAYER_COMPOSITION_TEST


def _source(text, name):
    import ast
    return ast.get_source_segment(text, next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name))


# ---- the plan --------------------------------------------------------------------------------------------------------------------------------

def test_the_layer_composition_test_id_names_a_test_of_this_file():
    assert LAYER_TEST == "tests/test_port_template_layer.py::test_generated_layers_match_the_transformers_decoder_layer"
    assert LAYER_TEST.split("::")[1] in globals()


def test_every_layer_is_written_whole_and_only_the_indexer_attention_stays_a_stub():
    text, p = tiny_qwen4()
    assert {s.function for s in p.stubs} == {"_C_self_attn"}
    assert [(l.key, bool(l.composed)) for l in p.layers] == [("A", True), ("B", True), ("C", True)]
    assert [l.composed["sublayer"] for l in p.layers] == ["linear_attn", "linear_attn", "self_attn"] and p.layers[2].composed["stub_sublayer"]
    assert "layer_composition" in p.summary()["generated"] and len(pt.stub_lines(text)) == 1
    a, b = _source(text, "_layer_A"), _source(text, "_layer_B")
    assert pt.GENERATED_BLOCK_MARK in a and "_A_ple" not in a and "_B_ple(model, i, x, ids)" in b
    assert a.index("_A_attn_hyper_connection") < a.index("_A_linear_attn") < a.index("_A_mlp_hyper_connection") < a.index("_mlp(")


def test_a_model_without_hyper_connections_gets_no_composed_layer():
    from test_port_template import std_generate
    _text, p = std_generate()
    assert not any(l.composed for l in p.layers) and "layer_composition" not in p.summary()["generated"]


# ---- the plan refuses what it cannot prove: the layer stays a stub -------------------------------------------------------------------------------

def _layer_attrs(con):
    cls = next(c for c in con["classes"] if c["class"] == "Qwen4ExpTextDecoderLayer")
    return cls["variants"]


def _flag_layer(op):
    def mutate(inventory):
        for o in inventory["ops"]:
            if o.get("op") == op:
                o["modules"] = ["model.layers.0"] + list(o.get("modules", []))
                return
        inventory["ops"].append({"op": op, "status": "candidate", "modules": ["model.layers.0"], "count": 1})
    return mutate


def _generate_with(mutate_construction=None, mutate_inventory=None, mutate_tensors=None):
    inv = _load("tiny-qwen4-ops-inventory.json")
    con = _load("tiny-qwen4-construction.json")
    tensors = _load("tiny-qwen4-tensors.json")
    if mutate_tensors:
        mutate_tensors(tensors)
    if mutate_inventory:
        mutate_inventory(inv)
    if mutate_construction:
        mutate_construction(con)
    return pt.generate(config=_load("tiny-qwen4-config.json"), construction=con, tensors=tensors, inventory=inv, excerpts=None)


def test_the_unmutated_plan_composes_layer_a():
    _text, p = _generate_with()
    assert "_layer_A" not in {s.function for s in p.stubs}


def test_a_layer_whose_attention_is_not_linear_stays_a_stub():
    def mutate(con):
        for v in _layer_attrs(con):
            if "linear_attn" in v["attrs"]["children"]:
                v["attrs"]["layer_type"] = "full_attention"
    _text, p = _generate_with(mutate_construction=mutate)
    assert "_layer_A" in {s.function for s in p.stubs}


def test_a_layer_with_a_stub_part_stays_a_stub(tmp_path):
    _text, p = _generate(mutate_tensors=lambda ts: [t.update(shape=[t["shape"][0] + 32, t["shape"][1]]) for t in ts if t["name"] == "model.layers.0.linear_attn.in_proj_qkv.weight"])
    assert {"_layer_A", "_A_linear_attn"} <= {s.function for s in p.stubs}


def test_a_layer_holding_an_unknown_child_stays_a_stub():
    _text, p = _generate_with(mutate_tensors=lambda ts: ts.append({"name": "model.layers.0.extra.weight", "dtype": "BF16", "shape": [64, 64]}))
    assert {"_layer_A", "_A_extra"} <= {s.function for s in p.stubs}


@pytest.mark.parametrize("op", ["aten.diff.default", "aten.sum.dim_IntList"])
def test_a_layer_with_another_stage_0_flag_than_the_known_one_stays_a_stub(op):
    inv = _load("tiny-qwen4-ops-inventory.json")
    for o in inv["ops"]:
        o["modules"] = [m for m in o.get("modules", []) if m != "model.layers.0"]
    inv["ops"].append({"op": op, "status": "candidate", "modules": ["model.layers.0"], "count": 1})
    _text, p = pt.generate(config=_load("tiny-qwen4-config.json"), construction=_load("tiny-qwen4-construction.json"), tensors=_load("tiny-qwen4-tensors.json"),
                           inventory=inv, excerpts=None)
    assert "_layer_A" in {s.function for s in p.stubs}


# ---- the reference comparison -----------------------------------------------------------------------------------------------------------------

def _config():
    config_mod = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    cfg = _load("tiny-qwen4-config.json")
    for key in ("architectures", "model_type"):
        cfg.pop(key, None)
    return config_mod.Qwen4ExpTextConfig(**cfg)


def _hf_layer(idx, seed):
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    config = _config()
    torch.manual_seed(seed)
    layer = modeling.Qwen4ExpTextDecoderLayer(config, idx)
    layer.mlp = modeling.Qwen4ExpTextMLP(config)          # the tiny checkpoint is dense (the construction records orchard_dense)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if name.endswith("A_log") or name.endswith("dt_bias"):
                v = 0.5 * torch.randn(p.shape, generator=g)
            elif name.endswith("linear_attn.norm.weight"):
                v = 1.0 + 0.2 * torch.randn(p.shape, generator=g)
            elif "norm" in name:
                v = 0.2 * torch.randn(p.shape, generator=g)
            elif name.endswith("conv1d.weight"):
                v = 0.6 * torch.randn(p.shape, generator=g)
            elif name.startswith("mlp."):
                v = 0.15 * torch.randn(p.shape, generator=g)
            else:
                v = 0.3 * torch.randn(p.shape, generator=g)
            p.copy_(v.to(torch.bfloat16).float())
    return layer.eval()


def _checkpoint(tmp_path, layers):
    """The tiny checkpoint's tensors: random for every layer not given, and the given modules' own tensors (the PLE table cut into four row shards)."""
    from safetensors.torch import save_file
    weights = {}
    for t in _load("tiny-qwen4-tensors.json"):
        weights[t["name"]] = (torch.zeros(t["shape"], dtype=torch.int64) if t["dtype"] != "BF16" else (0.1 * torch.randn(*t["shape"])).to(torch.bfloat16))
    for idx, layer in layers.items():
        state = {k: v.detach().clone() for k, v in layer.state_dict().items()}
        table_key = "ple.ple_embedding.ngram_embedding.weight"
        if table_key in state:
            for k, part in enumerate(state.pop(table_key).to(torch.bfloat16).chunk(4, dim=0)):
                weights[f"model.layers.{idx}.ple.ple_embedding.ngram_embedding.shard_{k}.weight"] = part.contiguous()
        for k, v in state.items():
            weights[f"model.layers.{idx}.{k}"] = v if v.dtype == torch.int64 else v.to(torch.bfloat16)
    snap = tmp_path / "snap"
    snap.mkdir()
    save_file(weights, str(snap / "model.safetensors"))
    return snap


def _inputs(seq=64):
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(2, 380, (seq,), generator=g)
    ids[[7, 8, 30]] = 1
    x = (torch.randn(1, seq, 256, generator=g) * 0.8).to(torch.bfloat16)
    return ids.tolist(), x


def test_generated_layers_match_the_transformers_decoder_layer(monkeypatch, tmp_path):
    text, _p = tiny_qwen4()
    mod = import_port(monkeypatch, text, tmp_path)
    layers = {0: _hf_layer(0, seed=3), 1: _hf_layer(1, seed=4)}
    model = mod.load(str(_checkpoint(tmp_path, layers)), fake.MeshDevice(1))
    ids, x_host = _inputs()
    x = fake.Tensor(x_host.reshape(1, 1, len(ids), 256).float(), layout="TILE")
    for idx, fn in ((0, mod._layer_A), (1, mod._layer_B)):
        with torch.no_grad():
            want = layers[idx](x_host.float(), None, ple_input_ids=torch.tensor([ids])).float()
        got = fn(model, idx, x, ids, None, None).t.float().reshape(want.shape)
        scale = want.abs().max().item()
        err = (got - want).abs().max().item()
        assert scale > 0.5, f"layer {idx}: the reference output is too small ({scale}) for the comparison to mean anything"
        assert err <= 0.06 * scale, f"layer {idx}: max abs error {err} against scale {scale}"


# ---- mutations of the emitted text: each must turn the comparison red -----------------------------------------------------------------------

def _run_with(monkeypatch, tmp_path, old, new):
    text, _p = tiny_qwen4()
    assert old in text, old
    monkeypatch.setattr(pt, "render", lambda p: text.replace(old, new, 1))
    test_generated_layers_match_the_transformers_decoder_layer(monkeypatch, tmp_path)


MUTATIONS = {
    "the PLE output is not added": ("    x = ttnn.add(x, _B_ple(model, i, x, ids))\n", "    x = x\n"),
    "the first injection is skipped": ("    x = _hyper_connection_inject(out, hyper_input, weights, 4, 64)\n    mixed, hyper_input, weights = _A_mlp_hyper_connection",
                                       "    x = hyper_input\n    mixed, hyper_input, weights = _A_mlp_hyper_connection"),
    "the MLP reads a rescaled mixed input": ('    out = _mlp(model, p + ".mlp", mixed)\n    return _hyper_connection_inject(out, hyper_input, weights, 4, 64)\n\n\ndef _B_',
                                             '    out = _mlp(model, p + ".mlp", ttnn.multiply(mixed, 0.5))\n    return _hyper_connection_inject(out, hyper_input, weights, 4, 64)\n\n\ndef _B_'),
    "the gated delta net reads a rescaled mixed input": ("    out = _A_linear_attn(model, i, mixed, ids)", "    out = _A_linear_attn(model, i, ttnn.multiply(mixed, 0.5), ids)"),
    "the second connection reads the layer input": ("    mixed, hyper_input, weights = _A_mlp_hyper_connection(model, i, x, ids)", "    mixed, hyper_input, weights = _A_mlp_hyper_connection(model, i, ttnn.multiply(x, 0.5), ids)"),
}


@pytest.mark.parametrize("why", sorted(MUTATIONS))
def test_the_reference_comparison_fails_when_the_emitted_text_is_broken(monkeypatch, tmp_path, why):
    old, new = MUTATIONS[why]
    with pytest.raises(AssertionError, match="max abs error|too small"):
        _run_with(monkeypatch, tmp_path, old, new)


# ---- the composition guards, called on constructed layers ---------------------------------------------------------------------------------------

def _layer(**over):
    hc = {"hc": 4, "hidden": 64, "eps": 1e-06, "inject": True}
    blocks = {"attn_hyper_connection": ("hyper_connection", dict(hc)), "mlp_hyper_connection": ("hyper_connection", dict(hc)), "mlp": ("mlp", {}),
              "linear_attn": ("gated_delta_net", {}), "ple": ("ple", {"hc": 4, "hidden": 64})}
    for k, v in over.items():
        blocks[k] = v
    return pt.Layer("A", [0], {k: "?" for k in blocks}, "Qwen4ExpTextDecoderLayer", standard=False, blocks=blocks)


ATTRS = {"layer_type": "linear_attention"}


def test_a_constructed_layer_with_every_part_is_composed():
    assert pt._compose_layer(_layer(), ATTRS, "aten.flatten.using_ints (candidate)") == {"hc": 4, "hidden": 64, "ple": True, "sublayer": "linear_attn", "stub_sublayer": False}


def test_a_connection_that_does_not_inject_keeps_the_layer_a_stub():
    off = ("hyper_connection", {"hc": 4, "hidden": 64, "eps": 1e-06, "inject": False})
    assert pt._compose_layer(_layer(attn_hyper_connection=off), ATTRS, None) is None
    assert pt._compose_layer(_layer(mlp_hyper_connection=off), ATTRS, None) is None


def test_connections_over_different_streams_keep_the_layer_a_stub():
    other = ("hyper_connection", {"hc": 2, "hidden": 64, "eps": 1e-06, "inject": True})
    assert pt._compose_layer(_layer(mlp_hyper_connection=other), ATTRS, None) is None


def test_a_ple_block_over_other_streams_keeps_the_layer_a_stub():
    assert pt._compose_layer(_layer(ple=("ple", {"hc": 4, "hidden": 32})), ATTRS, None) is None


# ---- the layer around a stub attention: the composition is the reference's whatever the sublayer computes ------------------------------------

def _attention_layer_case(monkeypatch, tmp_path):
    """Layer 3 (indexed attention). Its attention block is a stub the coder writes, so both sides get the same stand-in sublayer: a fixed matrix on the mixed input.
    The comparison then covers the composition (connections, sublayer call, injections, MLP) and nothing the coder will write."""
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    text, _p = tiny_qwen4()
    mod = import_port(monkeypatch, text, tmp_path)
    layer = _hf_layer(3, seed=6)
    g = torch.Generator().manual_seed(9)
    w = (0.2 * torch.randn(64, 64, generator=g)).to(torch.bfloat16).float()

    class Stand(torch.nn.Module):
        def forward(self, hidden_states, *args, **kwargs):
            return hidden_states @ w.t(), None
    layer.self_attn = Stand()
    model = mod.load(str(_checkpoint(tmp_path, {3: layer})), fake.MeshDevice(1))
    mod._C_self_attn = lambda model, i, x, ids: fake.Tensor(x.t @ w.t(), layout=x.layout)
    return mod, model, layer


def test_generated_attention_layer_composition_matches_the_reference(monkeypatch, tmp_path):
    mod, model, layer = _attention_layer_case(monkeypatch, tmp_path)
    ids, x_host = _inputs()
    with torch.no_grad():
        want = layer(x_host.float(), None).float()
    x = fake.Tensor(x_host.reshape(1, 1, len(ids), 256).float(), layout="TILE")
    got = mod._layer_C(model, 3, x, ids, None, None).t.float().reshape(want.shape)
    scale = want.abs().max().item()
    err = (got - want).abs().max().item()
    assert scale > 0.5, f"layer 3: the reference output is too small ({scale}) for the comparison to mean anything"
    assert err <= 0.06 * scale, f"layer 3: max abs error {err} against scale {scale}"


def test_the_attention_layer_comparison_fails_when_the_emitted_text_is_broken(monkeypatch, tmp_path):
    text, _p = tiny_qwen4()
    old = "    x = _hyper_connection_inject(out, hyper_input, weights, 4, 64)\n    mixed, hyper_input, weights = _C_mlp_hyper_connection"
    assert old in text
    monkeypatch.setattr(pt, "render", lambda p: text.replace(old, "    x = hyper_input\n    mixed, hyper_input, weights = _C_mlp_hyper_connection"))
    with pytest.raises(AssertionError, match="max abs error|too small"):
        test_generated_attention_layer_composition_matches_the_reference(monkeypatch, tmp_path)


def _attn_blocks(**over):
    blocks = {k: v for k, v in _layer().blocks.items() if k not in ("linear_attn", "ple")}
    blocks["self_attn"] = ("stub", {"cls": "Qwen4ExpTextAttention", "reason": "r"})
    blocks.update(over)
    return pt.Layer("C", [3], {k: "?" for k in blocks}, "Qwen4ExpTextDecoderLayer", standard=False, blocks=blocks)


def test_a_constructed_attention_layer_is_composed_around_the_stub():
    got = pt._compose_layer(_attn_blocks(), {"layer_type": "indexed_attention"}, None)
    assert got == {"hc": 4, "hidden": 64, "ple": False, "sublayer": "self_attn", "stub_sublayer": True}


def test_an_attention_sublayer_in_a_linear_layer_or_the_reverse_keeps_the_layer_a_stub():
    assert pt._compose_layer(_attn_blocks(), {"layer_type": "linear_attention"}, None) is None
    assert pt._compose_layer(_layer(), {"layer_type": "indexed_attention"}, None) is None


def test_a_generated_standard_attention_is_not_composed_here():
    assert pt._compose_layer(_attn_blocks(self_attn=("attention", {})), {"layer_type": "indexed_attention"}, None) is None
