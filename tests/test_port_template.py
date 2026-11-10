# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 2 port generator (orchard/port_template.py, issue 677): the plan, the generated port run on the CPU through a torch-backed fake
ttnn against a plain torch forward, the stubs for the tiny Qwen4 rehearsal model, the CPU check and the cheat-sheet coverage of every
TTNN name the generator emits. The real ttnn package is never imported here."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import torch

from orchard import op_cheats, port_template as pt
import fake_ttnn_compute as fake

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "port-generator"
SHEET = op_cheats.load_sheet(op_cheats.TTNN_SHEET)


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---- a small standard decoder (Qwen3-like: pre-norm, q and k norms, grouped keys, gated silu MLP) --------------------------------------------

STD_CONFIG = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "hidden_size": 32, "num_attention_heads": 2, "num_key_value_heads": 1,
              "head_dim": 16, "intermediate_size": 48, "vocab_size": 50, "num_hidden_layers": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
              "hidden_act": "silu", "tie_word_embeddings": False, "pad_token_id": 0}


def std_construction(*, norm_class="Qwen3RMSNorm", extra_layer_child=None, model_extra=None, act="SiLUActivation", attention_children=None):
    """The construction `effective_construction.describe` would print for the small decoder (variants list their example paths)."""
    attn_kids = attention_children or {"q_proj": "Linear", "k_proj": "Linear", "v_proj": "Linear", "o_proj": "Linear", "q_norm": norm_class, "k_norm": norm_class}
    layer_kids = {"input_layernorm": norm_class, "self_attn": "Qwen3Attention", "post_attention_layernorm": norm_class, "mlp": "Qwen3MLP"}
    if extra_layer_child:
        layer_kids.update(extra_layer_child)
    model_kids = {"embed_tokens": "Embedding", "layers": "ModuleList", "norm": norm_class, "rotary_emb": "Qwen3RotaryEmbedding"}
    if model_extra:
        model_kids.update(model_extra)
    mlp_kids = {"gate_proj": "Linear", "up_proj": "Linear", "down_proj": "Linear"}
    if act:
        mlp_kids["act_fn"] = act
    return {"model_class": "Qwen3ForCausalLM", "modules_total": 30, "classes_left_out": 0, "modules_cut": False, "classes": [
        {"class": "Qwen3ForCausalLM", "module": "m", "signature": [], "variants_left_out": 0,
         "variants": [{"count": 1, "paths": ["<model>"], "attrs": {"children": {"lm_head": "Linear", "model": "Qwen3Model"}}}]},
        {"class": "Qwen3Model", "module": "m", "signature": [], "variants_left_out": 0,
         "variants": [{"count": 1, "paths": ["model"], "attrs": {"children": model_kids}}]},
        {"class": "Qwen3DecoderLayer", "module": "m", "signature": [], "variants_left_out": 0,
         "variants": [{"count": 2, "paths": ["model.layers.{i}"], "attrs": {"children": layer_kids}}]},
        {"class": "Qwen3Attention", "module": "m", "signature": [], "variants_left_out": 0,
         "variants": [{"count": 2, "paths": ["model.layers.{i}.self_attn"], "attrs": {"children": attn_kids, "head_dim": 16, "scaling": 0.25, "is_causal": True}}]},
        {"class": "Qwen3MLP", "module": "m", "signature": [], "variants_left_out": 0,
         "variants": [{"count": 2, "paths": ["model.layers.{i}.mlp"], "attrs": {"children": mlp_kids, "hidden_size": 32, "intermediate_size": 48}}]},
        {"class": norm_class, "module": "m", "signature": [{"name": "eps", "default": 1e-6, "shown": True}], "variants_left_out": 0,
         "variants": [{"count": 7, "paths": ["model.layers.{i}.input_layernorm", "model.layers.{i}.post_attention_layernorm", "model.norm"], "attrs": {"eps": 1e-6}}]},
        {"class": "Linear", "module": "torch.nn", "signature": [], "variants_left_out": 0,
         "variants": [{"count": 15, "paths": ["model.layers.{i}.mlp.gate_proj", "lm_head"], "attrs": {"in_features": 32, "out_features": 48}}]},
        {"class": "Embedding", "module": "torch.nn", "signature": [], "variants_left_out": 0,
         "variants": [{"count": 1, "paths": ["model.embed_tokens"], "attrs": {"embedding_dim": 32, "num_embeddings": 50, "padding_idx": None}}]},
        {"class": "SiLUActivation", "module": "m", "signature": [], "variants_left_out": 0, "variants": [{"count": 2, "paths": ["model.layers.{i}.mlp.act_fn"], "attrs": {}}]},
        {"class": "ModuleList", "module": "torch.nn", "signature": [], "variants_left_out": 0, "variants": [{"count": 1, "paths": ["model.layers"], "attrs": {}}]},
        {"class": "Qwen3RotaryEmbedding", "module": "m", "signature": [], "variants_left_out": 0, "variants": [{"count": 1, "paths": ["model.rotary_emb"], "attrs": {"rope_type": "default"}}]},
    ]}


STD_EXCERPTS = "## Qwen3RMSNorm\n\n```python\nclass Qwen3RMSNorm(nn.Module):\n    def forward(self, hidden_states):\n        return self.weight * hidden_states\n```\n"
PLUS_ONE_EXCERPTS = STD_EXCERPTS.replace("return self.weight * hidden_states", "output = output * (1.0 + self.weight.float())\n        return output")


def std_weights(seed=0, *, tied=False) -> dict:
    """Random bf16 weights for the small decoder, in the checkpoint's own shapes ([out, in] for every Linear)."""
    g = torch.Generator().manual_seed(seed)
    c = STD_CONFIG
    h, nh, nkv, hd, inter, v = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["intermediate_size"], c["vocab_size"]

    def r(*shape, scale=0.2):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    w = {"model.embed_tokens.weight": r(v, h, scale=1.0), "model.norm.weight": r(h, scale=1.0) + 1}
    if not tied:
        w["lm_head.weight"] = r(v, h, scale=0.5)
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}"
        w.update({f"{p}.input_layernorm.weight": r(h, scale=0.3) + 1, f"{p}.post_attention_layernorm.weight": r(h, scale=0.3) + 1,
                  f"{p}.self_attn.q_proj.weight": r(nh * hd, h), f"{p}.self_attn.k_proj.weight": r(nkv * hd, h),
                  f"{p}.self_attn.v_proj.weight": r(nkv * hd, h), f"{p}.self_attn.o_proj.weight": r(h, nh * hd),
                  f"{p}.self_attn.q_norm.weight": r(hd, scale=0.3) + 1, f"{p}.self_attn.k_norm.weight": r(hd, scale=0.3) + 1,
                  f"{p}.mlp.gate_proj.weight": r(inter, h), f"{p}.mlp.up_proj.weight": r(inter, h), f"{p}.mlp.down_proj.weight": r(h, inter)})
    return w


def headers_of(weights: dict) -> list:
    return [{"name": k, "dtype": "BF16", "shape": list(t.shape)} for k, t in weights.items()]


def write_checkpoint(tmp_path: Path, weights: dict) -> Path:
    from safetensors.torch import save_file
    snap = tmp_path / "snapshot"
    snap.mkdir(exist_ok=True)
    save_file(weights, str(snap / "model.safetensors"))
    (snap / "config.json").write_text(json.dumps(STD_CONFIG))
    return snap


def torch_reference(weights: dict, ids: list, *, plus_one=False) -> torch.Tensor:
    """A plain float32 forward of the small decoder in the HF form, logits of the last position."""
    c = STD_CONFIG
    nh, nkv, hd, eps = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["rms_norm_eps"]
    w = {k: t.float() for k, t in weights.items()}
    bf = lambda t: t.to(torch.bfloat16).float()                                              # the port uploads bf16: the reference rounds the same values
    norm_w = (lambda t: bf(1.0 + t)) if plus_one else (lambda t: t)
    rms = lambda x, g: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm_w(g)
    s = len(ids)
    x = w["model.embed_tokens.weight"][torch.tensor(ids)]                                   # [S, h]
    inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, hd, 2).float() / hd))
    fr = torch.arange(s).float()[:, None] * inv[None]
    emb = torch.cat([fr, fr], -1)
    cos, sin = bf(emb.cos())[None], bf(emb.sin())[None]                                        # [1, S, hd]
    rot = lambda t: torch.cat((-t[..., hd // 2:], t[..., : hd // 2]), -1)
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}"
        h = rms(x, w[f"{p}.input_layernorm.weight"])
        q = (h @ w[f"{p}.self_attn.q_proj.weight"].T).view(s, nh, hd)
        k = (h @ w[f"{p}.self_attn.k_proj.weight"].T).view(s, nkv, hd)
        v = (h @ w[f"{p}.self_attn.v_proj.weight"].T).view(s, nkv, hd).transpose(0, 1)
        q = rms(q, w[f"{p}.self_attn.q_norm.weight"]).transpose(0, 1)                         # [nh, S, hd]
        k = rms(k, w[f"{p}.self_attn.k_norm.weight"]).transpose(0, 1)
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        k, v = k.repeat_interleave(nh // nkv, 0), v.repeat_interleave(nh // nkv, 0)
        sc = (q @ k.transpose(-2, -1)) * 0.25
        sc = sc.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), float("-inf"))
        a = (torch.softmax(sc, -1) @ v).transpose(0, 1).reshape(s, nh * hd)
        x = x + a @ w[f"{p}.self_attn.o_proj.weight"].T
        h = rms(x, w[f"{p}.post_attention_layernorm.weight"])
        m = (torch.nn.functional.silu(h @ w[f"{p}.mlp.gate_proj.weight"].T) * (h @ w[f"{p}.mlp.up_proj.weight"].T)) @ w[f"{p}.mlp.down_proj.weight"].T
        x = x + m
    x = rms(x, w["model.norm.weight"])
    head = w.get("lm_head.weight", w["model.embed_tokens.weight"])
    return (x @ head.T)[-1]


def import_port(monkeypatch, text: str, tmp_path: Path, name="generated_port"):
    """The generated port as a module, with the fake ttnn installed first (the real one is never imported)."""
    fake.install(monkeypatch)
    path = tmp_path / f"{name}.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


def std_generate(**kw):
    weights = kw.pop("weights", None) or std_weights()
    args = {"config": STD_CONFIG, "construction": std_construction(), "tensors": headers_of(weights), "excerpts": STD_EXCERPTS}
    args.update(kw)
    return pt.generate(**args)


# ---- the standard decoder: generated end to end, no stubs ------------------------------------------------------------------------------------

def test_a_standard_decoder_is_generated_whole_with_no_stub():
    text, p = std_generate()
    assert p.stubs == [] and all(l.standard for l in p.layers) and p.final_norm == ("model.norm", 1e-6) and p.attention
    assert pt.stub_lines(text) == []
    assert pt.check_sources({"port_model.py": text})["ok"]


@pytest.mark.parametrize("n_devices", [1, 2])
def test_the_generated_port_matches_a_torch_forward(monkeypatch, tmp_path, n_devices):
    weights = std_weights()
    text, p = std_generate(weights=weights)
    mod = import_port(monkeypatch, text, tmp_path)
    snap = write_checkpoint(tmp_path, weights)
    model = mod.load(str(snap), fake.MeshDevice(n_devices))
    ids = [3, 7, 11, 2, 40]
    got = mod.prefill(model, ids)
    want = torch_reference(weights, ids)
    assert got.shape == want.shape == (STD_CONFIG["vocab_size"],)
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-3), (got - want).abs().max()
    assert int(got.argmax()) == int(want.argmax())
    nxt = mod.decode_step(model, int(got.argmax()), len(ids))
    assert torch.allclose(nxt, torch_reference(weights, ids + [int(got.argmax())]), atol=1e-4, rtol=1e-3)
    mod.close(model)
    assert model["dev"] == {}


def test_every_projection_runs_through_ttnn_linear_and_no_weight_stays_on_the_host(monkeypatch, tmp_path):
    weights = std_weights()
    text, p = std_generate(weights=weights)
    mod = import_port(monkeypatch, text, tmp_path)
    model = mod.load(str(write_checkpoint(tmp_path, weights)), fake.MeshDevice(1))
    assert model["host"] == {}                                   # every tensor of this checkpoint is a float tensor: all on the device
    assert set(model["dev"]) == set(weights)
    for name, t in weights.items():
        if pt._NUMBER.sub("{i}", name) in {k for k, v in p.placements.items() if v == "linear_weight"}:
            assert model["dev"][name].shape == tuple(reversed(t.shape)), name        # transposed once at upload
    fake.COUNTS.clear()
    mod.prefill(model, [1, 2, 3])
    layers = STD_CONFIG["num_hidden_layers"]
    assert fake.COUNTS["linear"] == layers * 7 + 1                # q, k, v, o, gate, up, down per layer and the head
    assert fake.COUNTS["embedding"] == 1 and fake.COUNTS["scaled_dot_product_attention"] == layers
    import ast
    tree = ast.parse(text)
    assert not any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult) for n in ast.walk(tree))      # no projection in torch
    assert not any(isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.matmul" for n in ast.walk(tree))


def test_tied_embeddings_upload_the_table_once_more_as_the_head(monkeypatch, tmp_path):
    weights = std_weights(tied=True)
    config = {**STD_CONFIG, "tie_word_embeddings": True}
    text, p = std_generate(weights=weights, config=config)
    assert p.tied and p.placements["lm_head.weight"] == "linear_weight"
    mod = import_port(monkeypatch, text, tmp_path)
    model = mod.load(str(write_checkpoint(tmp_path, weights)), fake.MeshDevice(1))
    assert "lm_head.weight" in model["dev"]
    got = mod.prefill(model, [5, 6, 7])
    assert torch.allclose(got, torch_reference(weights, [5, 6, 7]), atol=1e-4, rtol=1e-3)


def test_a_norm_class_that_adds_one_to_its_weight_has_it_added_at_load(monkeypatch, tmp_path):
    weights = std_weights()
    text, p = std_generate(weights=weights, excerpts=PLUS_ONE_EXCERPTS)
    assert p.plus_one_norms == {"Qwen3RMSNorm"} and p.placements["model.norm.weight"] == "norm_plus_one"
    mod = import_port(monkeypatch, text, tmp_path)
    model = mod.load(str(write_checkpoint(tmp_path, weights)), fake.MeshDevice(1))
    got = mod.prefill(model, [5, 6, 7, 8])
    assert torch.allclose(got, torch_reference(weights, [5, 6, 7, 8], plus_one=True), atol=1e-4, rtol=1e-3)
    assert not torch.allclose(got, torch_reference(weights, [5, 6, 7, 8]), atol=1e-4, rtol=1e-3)     # the two references differ, so the check means something


def test_without_excerpts_norm_weights_are_used_as_stored_and_the_plan_says_so():
    _, p = std_generate(excerpts=None)
    assert p.plus_one_norms == set() and any("no reference excerpts" in n for n in p.notes)


# ---- blocks that become stubs -----------------------------------------------------------------------------------------------------------------

def test_partial_rotary_makes_the_attention_a_stub_and_the_layer_with_it():
    text, p = std_generate(config={**STD_CONFIG, "partial_rotary_factor": 0.25})
    assert not any(l.standard for l in p.layers)
    names = {s.function for s in p.stubs}
    assert names == {"_layer_A", "_A_self_attn"}
    assert any("partial_rotary_factor" in s.reason for s in p.stubs)
    assert len(pt.stub_lines(text)) == 2 and not pt.check_sources({"port_model.py": text})["ok"]


def test_an_attention_with_an_extra_child_is_a_stub_that_names_the_child():
    kids = {"q_proj": "Linear", "k_proj": "Linear", "v_proj": "Linear", "o_proj": "Linear", "indexer": "MyIndexer"}
    _, p = std_generate(construction=std_construction(attention_children=kids))
    reason = next(s.reason for s in p.stubs if s.function == "_A_self_attn")
    assert "indexer (MyIndexer)" in reason


def test_an_op_labelled_candidate_under_a_block_makes_it_a_stub():
    inventory = {"ops": [{"op": "aten.silu.default", "status": "candidate", "modules": ["model.layers.0.mlp.act_fn"]}]}
    _, p = std_generate(inventory=inventory)
    assert {s.function for s in p.stubs} == {"_layer_A", "_A_mlp"}
    assert "aten.silu.default (candidate)" in next(s.reason for s in p.stubs if s.function == "_A_mlp")
    _, p2 = std_generate(inventory={"ops": [{"op": "aten.silu.default", "status": "mapped", "modules": ["model.layers.0.mlp.act_fn"]}]})
    assert p2.stubs == []


def test_an_op_labelled_host_in_the_text_model_itself_makes_a_prelude_stub():
    inventory = {"ops": [{"op": "aten.cummax.default", "status": "host", "modules": ["model"]}]}
    text, p = std_generate(inventory=inventory)
    assert [s.function for s in p.stubs] == ["_model_prelude"] and "aten.cummax.default (host) in model itself" in p.stubs[0].reason
    assert all(l.standard for l in p.layers)                   # the layers are untouched: the flag is on the model's own forward
    assert "_model_prelude(model, x, padded)" in text


def test_an_op_labelled_host_in_the_layer_itself_makes_the_layer_a_stub_and_keeps_its_parts():
    inventory = {"ops": [{"op": "aten.cummax.default", "status": "host", "modules": ["model.layers.1"]}]}
    _, p = std_generate(inventory=inventory)
    assert [s.function for s in p.stubs] == ["_layer_A"] and "aten.cummax.default (host) in the layer itself" in p.stubs[0].reason
    assert p.layers[0].blocks["mlp"][0] == "mlp" and p.layers[0].blocks["self_attn"][0] == "attention"


def test_a_layer_child_the_generator_does_not_know_makes_the_layer_a_stub_that_lists_the_generated_parts():
    con = std_construction(extra_layer_child={"gate_block": "MyGate"})
    weights = std_weights()
    weights["model.layers.0.gate_block.weight"] = torch.ones(32, 32).to(torch.bfloat16)
    weights["model.layers.1.gate_block.weight"] = torch.ones(32, 32).to(torch.bfloat16)
    text, p = std_generate(construction=con, weights=weights)
    assert {s.function for s in p.stubs} == {"_layer_A", "_A_gate_block"}
    layer = next(s for s in p.stubs if s.function == "_layer_A")
    assert "not the standard pre-norm set" in layer.reason and "gate_block (MyGate)" in layer.reason
    body = text[text.index("def _layer_A("):]
    assert '_mlp(model, p + ".mlp", x)' in body and '_attention(model, p + ".self_attn", x, cos, sin)' in body and "_A_gate_block(model, i, x, ids)" in body
    assert p.summary()["generated"] == ["attention", "embedding", "final_norm", "lm_head", "mlp", "rms_norm"]


def test_an_extra_child_of_the_text_model_makes_prelude_and_epilogue_stubs():
    con = std_construction(model_extra={"mixer": "MyMixer"})
    weights = std_weights()
    weights["model.mixer.weight"] = torch.ones(32).to(torch.bfloat16)
    text, p = std_generate(construction=con, weights=weights)
    assert [s.function for s in p.stubs] == ["_model_prelude", "_model_mixer"]
    assert "x = _model_mixer(model, x)" in text and "x = _model_prelude(model, x, padded)" in text


def test_a_stub_function_raises_a_named_error_when_reached(monkeypatch, tmp_path):
    weights = std_weights()
    text, p = std_generate(weights=weights, config={**STD_CONFIG, "partial_rotary_factor": 0.25})
    mod = import_port(monkeypatch, text, tmp_path)
    model = mod.load(str(write_checkpoint(tmp_path, weights)), fake.MeshDevice(1))
    with pytest.raises(NotImplementedError, match=r"_layer_A: Qwen3DecoderLayer at model\.layers\.\{i\} is not written yet"):
        mod.prefill(model, [1, 2, 3])                   # the layer's own flow is the first stub reached; its parts are stubs of their own
    with pytest.raises(NotImplementedError, match=r"_A_self_attn: Qwen3Attention at model\.layers\.\{i\}\.self_attn is not written yet"):
        mod._A_self_attn(model, 0, None, [1, 2, 3])


# ---- the tiny Qwen4 rehearsal model: real construction, config, tensor headers and inventory ----------------------------------------------------

def tiny_qwen4():
    return pt.generate(config=_load("tiny-qwen4-config.json"), construction=_load("tiny-qwen4-construction.json"),
                       tensors=_load("tiny-qwen4-tensors.json"), inventory=_load("tiny-qwen4-ops-inventory.json"),
                       excerpts="class Qwen4ExpTextRMSNorm(nn.Module):\n    def forward(self, x):\n        output = output * (1.0 + self.weight.float())\n")


def test_tiny_qwen4_generates_the_standard_parts_and_stubs_every_new_block():
    text, p = tiny_qwen4()
    s = p.summary()
    assert s["generated"] == ["embedding", "gated_delta_net", "hyper_connection", "lm_head", "mlp"]
    assert [v["indices"] for v in s["layer_variants"]] == [[0, 2], [1], [3]] and not any(v["standard"] for v in s["layer_variants"])
    stubs = {x["function"]: x for x in s["stubs"]}
    # The hyper-connection blocks (prelude, mixer and the per-layer connections) and the gated delta net are written by registered families
    # (issue 696, slices 1 and 2); what is left is the layers that compose them, the per-layer embedding and the indexer attention.
    assert set(stubs) == {"_layer_A", "_layer_B", "_B_ple", "_layer_C", "_C_self_attn"}
    assert stubs["_B_ple"]["class"] == "Qwen4ExpTextPLELayer"
    assert "indexer (Qwen4ExpTextQSAIndexer)" in stubs["_C_self_attn"]["reason"]
    assert len(pt.stub_lines(text)) == 5
    # Placement: every Linear transposed, the embedding table ROW_MAJOR, the plain head norms with 1 added, integer tables on the host.
    assert p.placements["lm_head.weight"] == "linear_weight" and p.placements["model.embed_tokens.weight"] == "embedding"
    assert p.placements["model.layers.{i}.linear_attn.in_proj_qkv.weight"] == "linear_weight"
    assert p.placements["model.layers.{i}.self_attn.q_norm.weight"] == "norm_plus_one"
    assert p.placements["model.layers.{i}.attn_hyper_connection.hc_norm.weight"] == "norm_plus_one"   # group_size 64: written by the hyper-connection family, with 1 added at load
    assert p.placements["model.layers.{i}.ple.ple_embedding.layer_multipliers"] == "host"
    assert sum(1 for v in p.placements.values() if v == "linear_weight") == 24


def test_tiny_qwen4_port_loads_through_the_fake_and_makes_linear_calls_before_the_first_stub(monkeypatch, tmp_path):
    text, p = tiny_qwen4()
    mod = import_port(monkeypatch, text, tmp_path)
    g = torch.Generator().manual_seed(1)
    weights = {}
    for t in _load("tiny-qwen4-tensors.json"):
        if t["dtype"] == "BF16":
            weights[t["name"]] = (torch.randn(*t["shape"], generator=g) * 0.1).to(torch.bfloat16)
        else:
            weights[t["name"]] = torch.arange(int(torch.tensor(t["shape"]).prod()), dtype=torch.int64).reshape(t["shape"])
    from safetensors.torch import save_file
    snap = tmp_path / "snap"
    snap.mkdir()
    save_file(weights, str(snap / "model.safetensors"))
    model = mod.load(str(snap), fake.MeshDevice(1))
    assert len(model["dev"]) == 104 and len(model["host"]) == 3          # 95 before the gated delta net family: each of its 3 layers holds 4 convolution taps in place of 1 tensor
    assert model["dev"]["model.layers.0.linear_attn.in_proj_qkv.weight"].shape == (64, 320)
    with pytest.raises(NotImplementedError, match="_layer_A"):          # the prelude is generated now; the first layer is a stub
        mod.prefill(model, [1, 2, 3])
    assert fake.COUNTS["embedding"] == 1
    result = pt.check_sources({"port_model.py": text})
    assert result["matmul_calls"] >= 2 and not result["ok"] and "5 # AGENT: stub(s) remain" in result["problems"][0]


# ---- every TTNN name the generator emits is in the verified sheet ------------------------------------------------------------------------------

def test_every_ttnn_name_in_a_generated_port_is_covered_by_the_sheet():
    for text in (std_generate()[0], tiny_qwen4()[0]):
        assert pt.sheet_problems(text, SHEET) == []


def test_sheet_problems_name_an_unknown_call_and_an_unknown_keyword():
    src = "import ttnn\ny = ttnn.linear(x, w, weights=w)\nz = ttnn.no_such_op(x)\nq = ttnn.rms_norm(x, eps=1e-6)\nlayout = ttnn.TILE_LAYOUT\nm = ttnn.DRAM_MEMORY_CONFIG\n"
    problems = pt.sheet_problems(src, SHEET)
    assert problems == ["line 2: ttnn.linear passes 'weights', which the sheet's signature does not list",
                        "line 3: ttnn.no_such_op is not in the verified sheet",
                        "line 4: ttnn.rms_norm passes 'eps', which the sheet's signature does not list"]


def test_names_the_sheet_covers_only_through_a_usage_line_or_a_same_way_note():
    covered = pt.covered_names(SHEET)
    assert covered["ttnn.DRAM_MEMORY_CONFIG"] is None and covered["ttnn.MathFidelity.HiFi4"] is None          # named in usage lines, no entry
    assert covered["ttnn.add"]["name"] == "ttnn.multiply"                                                    # "add, subtract and silu are called the same way"
    assert covered["ttnn.linear"]["name"] == "ttnn.linear"


# ---- the CPU check --------------------------------------------------------------------------------------------------------------------------

HOST_PORT = "import torch\nimport ttnn\ndef load(d, m):\n    return {}\ndef prefill(model, ids):\n    return torch.zeros(10)\n"
DEVICE_PORT = "import ttnn\ndef prefill(model, ids):\n    return ttnn.linear(model['x'], model['w'])\n"


def test_check_sources_refuses_a_port_with_no_reachable_matmul_call():
    r = pt.check_sources({"port_model.py": HOST_PORT})
    assert not r["ok"] and r["matmul_calls"] == 0 and "no file makes a reachable ttnn matmul-family call" in r["problems"][0]
    assert pt.check_sources({"port_model.py": DEVICE_PORT})["ok"]


def test_check_sources_counts_remaining_stubs_and_skips_the_marker_inside_a_string():
    src = DEVICE_PORT + "def block(x):\n    # AGENT: write it\n    raise NotImplementedError\ns = '# AGENT: in a string'\n"
    r = pt.check_sources({"port_model.py": src})
    assert r["stubs"] == [("port_model.py", 5, "# AGENT: write it")] and not r["ok"] and r["problems"] == ["1 # AGENT: stub(s) remain: port_model.py:5"]


def test_check_sources_reports_a_file_that_does_not_parse_and_an_empty_directory():
    r = pt.check_sources({"port_model.py": "def (\n"})
    assert not r["ok"] and r["parse_errors"] and "does not parse" in r["problems"][0]
    assert pt.check_sources({})["problems"] == ["the port directory holds no Python file"]


def test_check_port_reads_regular_files_only_and_refuses_a_missing_directory(tmp_path):
    port = tmp_path / "port"
    port.mkdir()
    (port / "port_model.py").write_text(DEVICE_PORT)
    (port / "helper.py").write_text("x = 1\n# AGENT: left over\n")
    (port / "__pycache__").mkdir()
    (port / "__pycache__" / "port_model.cpython-312.py").write_text("# AGENT: compiled")
    os.symlink(tmp_path / "port" / "port_model.py", port / "link.py")
    r = pt.check_port(port)
    assert r["matmul_calls"] == 1 and r["stubs"] == [("helper.py", 2, "# AGENT: left over")]
    assert not pt.check_port(tmp_path / "no-such-dir")["ok"]


def test_check_port_refuses_too_many_files(tmp_path):
    port = tmp_path / "port"
    port.mkdir()
    for i in range(pt.MAX_PORT_FILES + 1):
        (port / f"f{i}.py").write_text(DEVICE_PORT)
    r = pt.check_port(port)
    assert not r["ok"] and "more than" in r["problems"][0]


# ---- refused inputs ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, [], {"classes": None}, {"classes": [None]}])
def test_a_construction_without_classes_is_refused_or_empty(bad):
    if bad in (None, [], {"classes": None}):
        with pytest.raises(pt.PlanError):
            pt.Construction(bad)
    else:
        assert pt.Construction(bad).resolve("model") == (None, {})


@pytest.mark.parametrize("field,value", [("name", "a..b"), ("name", 5), ("name", "x" * 201), ("dtype", "bf16"), ("dtype", None), ("dtype", []),
                                         ("shape", [-1]), ("shape", [10 ** 20]), ("shape", "1"), ("shape", None), ("shape", [1.5]), ("shape", [True]),
                                         ("shape", [1] * 9)])
def test_a_bad_tensor_record_is_refused_by_the_field_that_is_wrong(field, value):
    """One corrupted record in an otherwise valid header list: the refusal must come from the record check and name the field, so a later
    guard (no layer number, no embedding) is not what makes the test pass."""
    headers = headers_of(std_weights())
    headers[3] = {**headers[3], field: value}
    with pytest.raises(pt.PlanError, match=field):
        pt.plan(config=STD_CONFIG, construction=std_construction(), tensors=headers)
    with pytest.raises(pt.PlanError):
        pt.plan(config=STD_CONFIG, construction=std_construction(), tensors=None)


def test_the_mesh_read_back_keeps_one_copy(monkeypatch, tmp_path):
    """On a mesh every uploaded tensor is replicated; the composer joins one copy per chip along dim 0 and the port keeps the first."""
    weights = std_weights()
    text, _ = std_generate(weights=weights)
    mod = import_port(monkeypatch, text, tmp_path)
    model = mod.load(str(write_checkpoint(tmp_path, weights)), fake.MeshDevice(3))
    t = model["dev"]["model.norm.weight"]
    assert t.replicated and mod._to_host(model, t).shape == (1, 1, 1, 32)


def test_a_checkpoint_without_numbered_layers_or_without_an_embedding_is_refused():
    with pytest.raises(pt.PlanError, match="layer number"):
        pt.plan(config=STD_CONFIG, construction=std_construction(), tensors=[{"name": "lm_head.weight", "dtype": "BF16", "shape": [50, 32]}])
    weights = {k: v for k, v in std_weights().items() if k != "model.embed_tokens.weight"}
    with pytest.raises(pt.PlanError, match="embedding"):
        pt.plan(config=STD_CONFIG, construction=std_construction(), tensors=headers_of(weights))
    with pytest.raises(pt.PlanError, match="layers"):
        pt.plan(config={**STD_CONFIG, "num_hidden_layers": 3}, construction=std_construction(), tensors=headers_of(std_weights()))


def test_a_missing_head_with_untied_embeddings_is_refused():
    with pytest.raises(pt.PlanError, match="lm_head"):
        pt.plan(config={**STD_CONFIG, "tie_word_embeddings": False}, construction=std_construction(), tensors=headers_of(std_weights(tied=True)))


def test_a_non_dict_config_is_refused():
    with pytest.raises(pt.PlanError):
        pt.plan(config=[], construction=std_construction(), tensors=headers_of(std_weights()))


def test_resolve_follows_children_and_picks_the_variant_by_observed_children():
    con = pt.Construction(_load("tiny-qwen4-construction.json"))
    assert con.resolve("lm_head")[0] == "Linear"
    assert con.resolve("model.layers.{i}.mlp.gate_proj")[0] == "Linear"
    assert con.resolve("model.layers.{i}.self_attn.indexer")[0] == "Qwen4ExpTextQSAIndexer"
    cls, attrs = con.resolve("model.layers.{i}", observed={"ple", "linear_attn", "mlp", "attn_hyper_connection", "mlp_hyper_connection"})
    assert cls == "Qwen4ExpTextDecoderLayer" and "ple" in attrs["children"]
    cls, attrs = con.resolve("model.layers.{i}", observed={"self_attn", "mlp", "attn_hyper_connection", "mlp_hyper_connection"})
    assert "self_attn" in attrs["children"] and "ple" not in attrs["children"]
    assert con.resolve("no.such.path") == (None, {})


# ---- the generator is off the supervisor's path ----------------------------------------------------------------------------------------------

def test_the_supervisor_does_not_use_the_generator():
    """The default stage 2 start is unchanged: the supervisor copies the plain skeleton (port_facts.copy_skeleton) and never imports this module."""
    repo = Path(__file__).resolve().parent.parent / "orchard"
    for name in ("supervisor.py", "port_facts.py", "full_port.py"):
        assert "port_template" not in (repo / name).read_text(encoding="utf-8")


def test_the_command_line_generates_from_files_and_checks_a_directory(tmp_path, capsys):
    weights = std_weights()
    snap = write_checkpoint(tmp_path, weights)
    con = tmp_path / "construction.out"
    con.write_text("ORCHARD_EFFECTIVE_CONSTRUCTION " + json.dumps(std_construction()) + "\n")
    out = tmp_path / "port" / "port_model.py"
    out.parent.mkdir()
    assert pt.main(["generate", "--snapshot", str(snap), "--construction", str(con), "--out", str(out)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["stubs"] == [] and out.read_text().startswith("# SPDX")
    assert pt.main(["check", str(out.parent)]) == 0
    out.write_text(HOST_PORT)
    assert pt.main(["check", str(out.parent)]) == 1
    assert pt.main(["generate", "--snapshot", str(tmp_path / "none"), "--construction", str(con), "--out", str(out)]) == 1


# ---- a stub whose marker was deleted (review of PR 685) ------------------------------------------------------------------------------------

STUB_TEXT = "_B_ple: Qwen4ExpTextPLELayer at model.layers.1.ple is not written yet"


def _port_with(body: str) -> str:
    return ("import ttnn\n\n\ndef _lin(x, w):\n    return ttnn.linear(x, w)\n\n\ndef _B_ple(model, i, x, ids):\n" + body + "\n")


def test_deleting_the_marker_comment_does_not_remove_a_stub():
    kept = pt.check_sources({"port_model.py": _port_with(f"    # AGENT: write it\n    raise NotImplementedError({STUB_TEXT!r})")})
    gone = pt.check_sources({"port_model.py": _port_with(f"    raise NotImplementedError({STUB_TEXT!r})")})
    assert not kept["ok"] and not gone["ok"]
    assert len(kept["stubs"]) == 1 and len(gone["stubs"]) == 1                   # counted once, by function, not once per sign
    assert any("stub(s) remain" in p for p in gone["problems"])


def test_a_function_that_raises_another_not_implemented_error_is_not_a_stub():
    ok = pt.check_sources({"port_model.py": _port_with("    raise NotImplementedError('decode_step is not supported for this block')")})
    assert not ok["stubs"]


def test_the_generated_stub_text_is_found_in_a_nested_function_and_a_method_too():
    text = _port_with("    def inner():\n        raise NotImplementedError(" + repr(STUB_TEXT) + ")\n    return inner")
    assert len(pt.check_sources({"port_model.py": text})["stubs"]) == 1


def test_the_stub_marker_in_a_string_is_still_not_a_stub():
    text = _port_with("    return 'x'  # fine\n    s = '# AGENT: not a stub'")
    assert not pt.check_sources({"port_model.py": text})["stubs"]


def test_rms_norm_helper_names_the_prefix_it_has_no_eps_for():
    """Arm E8: the hyper-connection norm (a group_size norm) is not in NORM_EPS, and a stub called _rms_norm with its prefix. A KeyError inside the
    helper pointed at the helper. The helper now raises an error that says the caller must compute that norm itself (system review of
    2026-11-10: a silent fallback to the config eps would turn this into a numerics miss)."""
    import ast
    text, _plan = std_generate()
    ns = {"NORM_EPS": {"model.norm": 1e-5}, "_pattern": pt_pattern}
    calls = {}

    class T:
        @staticmethod
        def rms_norm(x, epsilon, weight):
            calls["eps"] = epsilon
            return x
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_rms_norm")
    ns["ttnn"] = T
    exec(compile(ast.Module([fn], []), "p", "exec"), ns)
    ns["_rms_norm"]({"dev": {"model.norm.weight": 1}}, "model.norm", 7)
    assert calls["eps"] == 1e-5
    with pytest.raises(ValueError, match="not a plain RMS norm.*the function that calls _rms_norm"):
        ns["_rms_norm"]({"dev": {}}, "model.layers.0.attn_hyper_connection.hc_norm", 7)


def test_the_fixed_helpers_are_the_functions_the_generator_always_writes_and_none_is_a_stub():
    import ast
    text, _plan = std_generate()
    names = [n.name for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)]
    assert set(pt.FIXED_HELPERS) <= set(names)
    assert pt.stub_lines(text) == []
    assert "_rms_norm" in pt.FIXED_HELPERS and "_layer_A" not in pt.FIXED_HELPERS and "_mlp" not in pt.FIXED_HELPERS


def pt_pattern(name):
    import re
    return re.sub(r"(?<=\.)\d+(?=\.|$)", "{i}", name)
