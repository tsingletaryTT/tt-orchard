# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Qwen4 sparse MoE block (`Qwen4ExpTextSparseMoeBlock`) written by the port generator (issue 696, slice 6; issue 723).

The reference comparison runs the generated `_A_mlp` through the torch-backed fake ttnn (tests/fake_ttnn_compute.py) against the real transformers
`Qwen4ExpTextSparseMoeBlock` at tiny sizes (8 experts, top 2, hidden 64, expert width 64, a shared expert with its sigmoid gate), in both expert forms the
generator can write: the dense-masked form (every expert on every token, weighted by the routing matrix; used up to `pt.MOE_DENSE_MAX_EXPERTS` experts) and the
gather/scatter form (one launch group per expert that was hit, the gather by `ttnn.embedding`; the form for more experts).

The router runs on the device (`ttnn.topk(stable=True)`, probe 10); the indices and weights come back to the host and the routing matrix, the gather lists and
the scatter matrices are built there and uploaded (the pattern of the QSA block). The comparison shows that the generated composition computes the reference's
values with torch standing in for the device ops. It does not show that the assembled block runs on a board: probes 10, 11 and 13 ran each op form on a board
(PCC 0.9998 or better, tiny and Flash widths) with the CPU router's indices, and the chain from the device router's output to the experts has not run.

Checkpoint layout: the checkpoint stores experts one tensor per expert (`experts.<e>.gate_proj.weight`, `up_proj`, `down_proj`), as the tiny checkpoints and
the real Flash-Next index do. The transformers module fuses them in memory; a checkpoint with fused expert tensors is refused."""
from __future__ import annotations

import copy

import pytest
import torch

from orchard import port_template as pt
import fake_ttnn_compute as fake
from test_port_template import _load, import_port

MOE_TEST = pt.MOE_REFERENCE_TEST
EXCERPTS = "class Qwen4ExpTextRMSNorm(nn.Module):\n    def forward(self, x):\n        output = output * (1.0 + self.weight.float())\n"
SEQ = 32


def _generate(variant="moe", mutate_construction=None, mutate_tensors=None, mutate_config=None):
    con = copy.deepcopy(_load(f"tiny-qwen4-{variant}-construction.json"))
    tensors = copy.deepcopy(_load(f"tiny-qwen4-{variant}-tensors.json"))
    cfg = copy.deepcopy(_load(f"tiny-qwen4-{variant}-config.json"))
    if mutate_construction:
        mutate_construction(con)
    if mutate_tensors:
        mutate_tensors(tensors)
    if mutate_config:
        mutate_config(cfg)
    return pt.generate(config=cfg, construction=con, tensors=tensors, inventory=_load("tiny-qwen4-ops-inventory.json"), excerpts=EXCERPTS)


def _class_attrs(con, cls):
    return [v["attrs"] for c in con["classes"] if c["class"] == cls for v in c["variants"]]


# ---- the plan ---------------------------------------------------------------------------------------------------------------------------------

def test_the_reference_test_id_names_a_test_of_this_file():
    assert MOE_TEST == "tests/test_port_template_moe.py::test_generated_sparse_moe_matches_the_transformers_module"
    assert MOE_TEST.split("::")[1] in globals()


def test_the_all_moe_plan_has_no_stub_and_every_layer_is_composed():
    text, p = _generate()
    assert p.stubs == [] and pt.stub_lines(text) == []
    assert "sparse_moe" in p.summary()["generated"]
    assert all(l.composed and l.composed["mlp"] == "sparse_moe" for l in p.layers)
    assert text.count(pt.GENERATED_BLOCK_MARK + " sparse_moe") >= 6
    assert "_A_mlp(model, i, mixed, ids)" in text and "_mlp(model, p + \".mlp\", mixed)" not in text


def test_the_expert_and_router_tensors_are_placed_as_linear_weights():
    _text, p = _generate()
    for name in ("gate", "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj", "shared_expert_gate",
                 "experts.{i}.gate_proj", "experts.{i}.up_proj", "experts.{i}.down_proj"):
        assert p.placements[f"model.layers.{{i}}.mlp.{name}.weight"] == "linear_weight", name


def test_a_mixed_checkpoint_splits_the_layers_by_the_kind_of_feed_forward():
    """The `mixed` tiny variant has a dense MLP in layer 0 and MoE blocks in layers 1 to 3 (`mlp_only_layers = [0]`). Before the layer variants were split by the
    kind of `mlp`, the plan took them for one variant and wrote the dense MLP for every layer: a port that reads tensors the MoE layers do not have."""
    text, p = _generate("mixed")
    kinds = {tuple(l.indices): l.blocks["mlp"][0] for l in p.layers}
    assert kinds[(0,)] == "mlp" and all(k == "sparse_moe" for idx, k in kinds.items() if idx != (0,))
    assert sorted(i for idx in kinds for i in idx) == [0, 1, 2, 3]
    assert p.stubs == [] and "sparse_moe" in p.summary()["generated"]
    assert {tuple(l.indices): l.children["mlp"] for l in p.layers}[(0,)] == "Qwen4ExpTextMLP"          # the layer class is the variant whose mlp child is the dense class
    assert all(l.children["mlp"] == "Qwen4ExpTextSparseMoeBlock" for l in p.layers if l.indices != [0])
    dense_layer = next(l for l in p.layers if l.indices == [0])
    assert f"_mlp(model, p + \".mlp\", mixed)" in text and f"_{dense_layer.key}_mlp(" not in text.split("def _layer_" + dense_layer.key)[1].split("def _layer_")[0]


def test_a_dense_plan_is_unchanged_by_the_kind_split():
    from test_port_template import tiny_qwen4
    text, p = tiny_qwen4()
    assert [v["indices"] for v in p.summary()["layer_variants"]] == [[0, 2], [1], [3]] and p.stubs == []
    assert "sparse_moe" not in p.summary()["generated"] and "_sparse_moe" not in text


def test_both_expert_forms_use_only_ttnn_calls_the_verified_sheet_covers(monkeypatch):
    text, _p = _generate()
    assert pt.sheet_problems(text) == []
    monkeypatch.setattr(pt, "MOE_DENSE_MAX_EXPERTS", 0)
    text2, _p2 = _generate()
    assert "_moe_experts_gathered(" in text2 and pt.sheet_problems(text2) == [] and pt.stub_lines(text2) == []


def test_the_dense_masked_form_is_written_up_to_the_expert_bound_and_the_gather_form_above_it(monkeypatch):
    text, _p = _generate()
    assert "dense=True" in text
    monkeypatch.setattr(pt, "MOE_DENSE_MAX_EXPERTS", 4)
    text2, p2 = _generate()
    assert "dense=False" in text2 and p2.stubs == []


# ---- the plan refuses what it cannot prove: the block stays a stub ----------------------------------------------------------------------------------

def _gate(con):
    return _class_attrs(con, "Qwen4ExpTextTopKRouter")


def _experts(con):
    return _class_attrs(con, "Qwen4ExpTextExperts")


def _kids(con):
    return _class_attrs(con, "Qwen4ExpTextSparseMoeBlock")


def _set(attrs_of, key, value):
    def go(con):
        for a in attrs_of(con):
            a[key] = value
    return go


def _drop_tensors(prefix_end):
    def go(tensors):
        tensors[:] = [t for t in tensors if not t["name"].endswith(prefix_end)]
    return go


def _fuse_layer_one(tensors):
    out = [t for t in tensors if ".layers.1.mlp.experts." not in t["name"]]
    out.append({"name": "model.layers.1.mlp.experts.gate_up_proj", "dtype": "BF16", "shape": [8, 128, 64]})
    out.append({"name": "model.layers.1.mlp.experts.down_proj", "dtype": "BF16", "shape": [8, 64, 64]})
    tensors[:] = out


def _extra_child(con):
    for a in _kids(con):
        a["children"]["aux"] = "Linear"


def _shared_gate_class(con):
    for a in _kids(con):
        a["children"]["shared_expert_gate"] = "Parameter"


def _shared_gate_tensor_wide(tensors):
    for t in tensors:
        if t["name"].endswith(".mlp.shared_expert_gate.weight"):
            t["shape"] = [2, 64]


def _router_weight_shape(tensors):
    for t in tensors:
        if t["name"] == "model.layers.0.mlp.gate.weight":          # tensors are compared by pattern, through the first layer that has the block
            t["shape"] = [7, 64]


def _one_expert_missing(tensors):
    tensors[:] = [t for t in tensors if t["name"] != "model.layers.2.mlp.experts.5.down_proj.weight"]


def _inter_48_everywhere(con):
    _set(_experts, "intermediate_dim", 48)(con)


def _expert_tensors_48(tensors):
    for t in tensors:
        if ".mlp.experts." in t["name"]:
            t["shape"] = [48, 64] if not t["name"].endswith("down_proj.weight") else [64, 48]


def _expert_act_gelu(con):
    for a in _experts(con):
        a["children"]["act_fn"] = "GELUActivation"


def _shared_act_gelu(con):
    for c in con["classes"]:
        if c["class"] == "Qwen4ExpTextMLP":
            for v in c["variants"]:
                if "act_fn" in v["attrs"].get("children", {}):
                    v["attrs"]["children"]["act_fn"] = "GELUActivation"


def _router_bias(tensors):
    tensors.append({"name": "model.layers.1.mlp.gate.bias", "dtype": "BF16", "shape": [8]})


def _shared_bias(tensors):
    tensors.append({"name": "model.layers.1.mlp.shared_expert.up_proj.bias", "dtype": "BF16", "shape": [64]})


def _gate_proj_shape(tensors):
    for t in tensors:
        if ".mlp.experts." in t["name"] and t["name"].endswith("gate_proj.weight"):
            t["shape"] = [32, 64]


REFUSALS = {
    "the expert width is a multiple of 16 but not of the tile (construction and tensors agree)": ((_inter_48_everywhere), _expert_tensors_48),
    "the experts' activation is not silu": (_expert_act_gelu, None),
    "the shared expert's activation is not silu": (_shared_act_gelu, None),
    "the router has a bias tensor": (None, _router_bias),
    "the shared expert has a bias tensor": (None, _shared_bias),
    "the expert gate projection tensors do not have the expert width": (None, _gate_proj_shape),
    "the router and the experts disagree on the number of experts": (_set(_gate, "num_experts", 9), None),
    "top-k is larger than the number of experts": (_set(_gate, "top_k", 9), None),
    "top-k is not a whole number": (_set(_gate, "top_k", 2.5), None),
    "the router has no norm_topk_prob": (_set(_gate, "norm_topk_prob", None), None),
    "the experts carry a bias": (_set(_experts, "has_bias", True), None),
    "the experts are transposed": (_set(_experts, "is_transposed", True), None),
    "the experts have no gate": (_set(_experts, "has_gate", False), None),
    "the expert width is not a tile multiple": (_set(_experts, "intermediate_dim", 48), None),
    "the block has a child the generator does not know": (_extra_child, None),
    "the shared expert gate is not a Linear": (_shared_gate_class, None),
    "the shared expert gate tensor has two outputs": (None, _shared_gate_tensor_wide),
    "the router weight has the wrong shape": (None, _router_weight_shape),
    "one expert tensor is missing": (None, _one_expert_missing),
    "the checkpoint fuses the experts": (None, _fuse_layer_one),
    "the shared expert has no down projection": (None, _drop_tensors("mlp.shared_expert.down_proj.weight")),
    "the router weight is missing": (None, _drop_tensors("mlp.gate.weight")),
}


def test_the_unmutated_construction_is_recognized():
    _text, p = _generate()
    assert "_A_mlp" not in {s.function for s in p.stubs}


@pytest.mark.parametrize("why", sorted(REFUSALS))
def test_a_block_the_generator_cannot_prove_stays_a_stub(why):
    mc, mt = REFUSALS[why]
    _text, p = _generate(mutate_construction=mc, mutate_tensors=mt)
    stubs = {s.function for s in p.stubs}
    assert stubs & {"_A_mlp", "_B_mlp", "_C_mlp"}, f"{why}: the block was written"
    assert stubs & {"_layer_A", "_layer_B", "_layer_C"}


def test_a_checkpoint_that_fuses_the_experts_is_refused_with_a_reason_that_says_so():
    _text, p = _generate(mutate_tensors=_fuse_layer_one)
    reasons = " ".join(s.reason for s in p.stubs if s.function.endswith("_mlp"))
    assert "per-expert tensors" in reasons and "fused" in reasons


def test_experts_that_do_not_fit_one_chip_stay_a_stub_that_names_the_budget_and_the_multi_chip_issue(monkeypatch):
    monkeypatch.setattr(pt, "MOE_EXPERT_BYTES_MAX", 1000)
    _text, p = _generate()
    reasons = " ".join(s.reason for s in p.stubs if s.function.endswith("_mlp"))
    assert {"_A_mlp", "_B_mlp", "_C_mlp"} <= {s.function for s in p.stubs}
    assert "per-chip budget" in reasons and "multi-chip" in reasons and "723" in reasons


def test_the_budget_counts_every_layer_not_one_layer_variant(monkeypatch):
    """Four layers hold experts: 8 experts x 3 x 64 x 64 x 2 bytes each. A budget of one layer and a bit refuses, a budget of all four layers accepts."""
    per_layer = 8 * 3 * 64 * 64 * 2
    monkeypatch.setattr(pt, "MOE_EXPERT_BYTES_MAX", per_layer * 4 - 1)
    assert _generate()[1].stubs
    monkeypatch.setattr(pt, "MOE_EXPERT_BYTES_MAX", per_layer * 4)
    assert _generate()[1].stubs == []


def _flash_like(experts):
    """The MoE block at Flash-Next widths (hidden 2560, expert and shared width 640, top 10) with `experts` experts in each of the four layers: the construction and the
    tensor shapes of the block only (the plan reads nothing else of it)."""
    hidden, inter = 2560, 640

    def construction(con):
        for a in _gate(con):
            a.update(num_experts=experts, top_k=10, hidden_dim=hidden)
        for a in _experts(con):
            a.update(num_experts=experts, hidden_dim=hidden, intermediate_dim=inter)
        for c in con["classes"]:
            if c["class"] == "Qwen4ExpTextMLP":
                for v in c["variants"]:
                    if v["attrs"].get("intermediate_size") is not None and "shared_expert" in " ".join(v.get("paths", [])):
                        v["attrs"].update(hidden_size=hidden, intermediate_size=inter)

    def tensors(ts):
        keep = [t for t in ts if ".mlp.experts." not in t["name"]]
        for t in keep:
            n = t["name"]
            if n.endswith(".mlp.gate.weight"):
                t["shape"] = [experts, hidden]
            elif n.endswith(".mlp.shared_expert_gate.weight"):
                t["shape"] = [1, hidden]
            elif ".mlp.shared_expert." in n:
                t["shape"] = [hidden, inter] if n.endswith("down_proj.weight") else [inter, hidden]
        for layer in range(4):
            for e in range(experts):
                for proj, shape in (("gate_proj", [inter, hidden]), ("up_proj", [inter, hidden]), ("down_proj", [hidden, inter])):
                    keep.append({"name": f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight", "dtype": "BF16", "shape": shape})
        ts[:] = keep
    return construction, tensors


def test_the_real_512_expert_shape_stays_a_stub_on_the_chip_budget_and_256_experts_are_written():
    """4 layers x 512 experts x 3 x 2560 x 640 x 2 bytes = 18.8 GiB (20.1 GB): over the 16 GiB budget. 256 experts are 9.4 GiB: written, in the gather form (above the dense bound)."""
    mc, mt = _flash_like(512)
    _text, p = _generate(mutate_construction=mc, mutate_tensors=mt)
    reasons = " ".join(s.reason for s in p.stubs if s.function.endswith("_mlp"))
    assert {"_A_mlp", "_B_mlp", "_C_mlp"} <= {s.function for s in p.stubs}
    assert "18.8 GiB" in reasons and "16.0 GiB" in reasons and "multi-chip" in reasons
    mc, mt = _flash_like(256)
    text, p = _generate(mutate_construction=mc, mutate_tensors=mt)
    assert not {s.function for s in p.stubs} & {"_A_mlp", "_B_mlp", "_C_mlp"}
    assert "experts=256, top_k=10, norm_topk=True, dense=False" in text


def test_the_expert_bound_and_the_chip_budget_are_stated_constants():
    assert pt.MOE_DENSE_MAX_EXPERTS == 16
    assert pt.MOE_EXPERT_BYTES_MAX == 16 * 2 ** 30


# ---- the reference comparison -----------------------------------------------------------------------------------------------------------------

def _config():
    config_mod = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    cfg = _load("tiny-qwen4-moe-config.json")
    for key in ("architectures", "model_type"):
        cfg.pop(key, None)
    config = config_mod.Qwen4ExpTextConfig(**cfg)
    config._attn_implementation = "eager"
    return config


def _hf_block(seed):
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    config = _config()
    torch.manual_seed(seed)
    block = modeling.Qwen4ExpTextSparseMoeBlock(config)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in block.named_parameters():
            p.copy_((0.3 * torch.randn(p.shape, generator=g)).to(torch.bfloat16).float())
    return config, block.eval()


def _checkpoint(tmp_path, block, layers=(0, 2)):
    """The tensors of the fixture, zeros or small random numbers, with the block's weights written under each layer in `layers` in the per-expert layout."""
    from safetensors.torch import save_file
    weights = {}
    for t in _load("tiny-qwen4-moe-tensors.json"):
        weights[t["name"]] = (torch.zeros(t["shape"], dtype=torch.int64) if t["dtype"] != "BF16" else (0.1 * torch.randn(*t["shape"])).to(torch.bfloat16))
    state = {k: v.detach().to(torch.bfloat16) for k, v in block.state_dict().items()}
    inter = block.experts.intermediate_dim
    for layer in layers:
        p = f"model.layers.{layer}.mlp"
        weights[f"{p}.gate.weight"] = state["gate.weight"].clone()
        for e in range(block.experts.num_experts):
            gate_up = state["experts.gate_up_proj"][e]
            weights[f"{p}.experts.{e}.gate_proj.weight"] = gate_up[:inter].clone()
            weights[f"{p}.experts.{e}.up_proj.weight"] = gate_up[inter:].clone()
            weights[f"{p}.experts.{e}.down_proj.weight"] = state["experts.down_proj"][e].clone()
        for n in ("gate_proj", "up_proj", "down_proj"):
            weights[f"{p}.shared_expert.{n}.weight"] = state[f"shared_expert.{n}.weight"].clone()
        weights[f"{p}.shared_expert_gate.weight"] = state["shared_expert_gate.weight"].clone()
    snap = tmp_path / "snap"
    snap.mkdir(exist_ok=True)
    save_file(weights, str(snap / "model.safetensors"))
    return snap


def _setup(monkeypatch, tmp_path, text=None, seed=11):
    if text is None:
        text, _p = _generate()
    mod = import_port(monkeypatch, text, tmp_path)
    config, block = _hf_block(seed)
    model = mod.load(str(_checkpoint(tmp_path, block)), fake.MeshDevice(1))
    x_host = (torch.randn(1, SEQ, 64, generator=torch.Generator().manual_seed(5)) * 0.8).to(torch.bfloat16)
    x = fake.Tensor(x_host.reshape(1, 1, SEQ, 64).float(), layout="TILE")
    return mod, model, block, x_host, x


def _compare(monkeypatch, tmp_path, text=None):
    mod, model, block, x_host, x = _setup(monkeypatch, tmp_path, text)
    with torch.no_grad():
        want = block(x_host.float()).float()
    got = mod._A_mlp(model, 0, x, list(range(SEQ))).t.float().reshape(want.shape)
    scale = want.abs().max().item()
    assert scale > 0.2, f"the reference output is too small ({scale}) for the comparison to mean anything"
    err = (got - want).abs().max().item()
    assert err <= 0.05 * scale, f"max abs error {err} against scale {scale}"
    return got, want, scale


def test_generated_sparse_moe_matches_the_transformers_module(monkeypatch, tmp_path):
    _compare(monkeypatch, tmp_path)


def test_the_gather_scatter_form_matches_the_transformers_module_too(monkeypatch, tmp_path):
    monkeypatch.setattr(pt, "MOE_DENSE_MAX_EXPERTS", 0)
    text, _p = _generate()
    assert "dense=False" in text
    _compare(monkeypatch, tmp_path, text)


def test_the_router_selects_different_experts_for_different_tokens(monkeypatch, tmp_path):
    """The comparison would pass for a block that sent every token to the same experts if the reference did the same, so the reference must not."""
    _config_, block = _hf_block(11)
    x = (torch.randn(1, SEQ, 64, generator=torch.Generator().manual_seed(5)) * 0.8).to(torch.bfloat16).float()
    with torch.no_grad():
        _, _, idx = block.gate(x.reshape(-1, 64))
    assert len({tuple(sorted(r.tolist())) for r in idx}) > 4


def test_each_layer_reads_its_own_weights(monkeypatch, tmp_path):
    """`_A_mlp(model, i, ...)` reads `model.layers.<i>.mlp.*`: layers 0 and 2 hold the block's weights (the same output), layer 1 holds the fixture's other numbers (another one)."""
    mod, model, block, x_host, x = _setup(monkeypatch, tmp_path)
    a = mod._A_mlp(model, 0, x, list(range(SEQ))).t.float()
    b = mod._A_mlp(model, 2, x, list(range(SEQ))).t.float()
    other = mod._A_mlp(model, 1, x, list(range(SEQ))).t.float()
    assert torch.allclose(a, b, atol=1e-6)
    assert not torch.allclose(a, other, atol=1e-3)


def test_a_wrong_input_layout_is_named(monkeypatch, tmp_path):
    mod, model, _block, _x_host, _x = _setup(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=r"_sparse_moe needs x \[1, 1, S, hidden\]"):
        mod._A_mlp(model, 0, fake.Tensor(torch.zeros(1, 2, SEQ, 64), layout="TILE"), list(range(SEQ)))


# ---- mutations of the emitted text: each must turn the comparison red -----------------------------------------------------------------------------

MUTATIONS = {
    "the routing weights are ignored": ("scores = _to_host(model, weights).float().reshape(seq, -1)[:, :top_k]", "scores = torch.ones(seq, top_k)"),
    "the selected logits are not normalised": ("weights = ttnn.softmax(values, dim=-1)", "weights = values"),
    "norm_topk is inverted": ("    if norm_topk:", "    if not norm_topk:"),
    "the shared expert gate is not applied": ("ttnn.multiply(_moe_expert_mlp(model, prefix + \".shared_expert\", x), ttnn.sigmoid(_linear(model, prefix + \".shared_expert_gate\", x)))",
                                              "_moe_expert_mlp(model, prefix + \".shared_expert\", x)"),
    "the shared expert is dropped": ("return ttnn.add(routed, shared)", "return routed"),
    "the expert activation is missing": ("ttnn.multiply(ttnn.silu(gate), up)", "ttnn.multiply(gate, up)"),
    "the up projection is missing": ("ttnn.multiply(ttnn.silu(gate), up)", "ttnn.silu(gate)"),
    "top-k takes one expert too few": ("ttnn.topk(logits, top_k, dim=-1", "ttnn.topk(logits, top_k - 1, dim=-1"),
    "the routing column of the wrong expert is used": ("ttnn.slice(routing, (0, 0, 0, e), (1, 1, seq, e + 1))", "ttnn.slice(routing, (0, 0, 0, 0), (1, 1, seq, 1))"),
    "the routing matrix is not scattered by expert": ("torch.zeros(seq, experts).scatter_(1, chosen, scores)", "torch.zeros(seq, experts).scatter_(1, chosen.flip(1), scores)"),
}
GATHER_MUTATIONS = {
    "the scatter matrix adds half a row": ("back[tokens, torch.arange(n)] = 1.0", "back[tokens, torch.arange(n)] = 0.5"),
    "the padding rows carry weight": ("weight[:n] = scores[tokens, slots]", "weight[:] = 1.0"),
    "the gathered rows are the wrong tokens": ("picked = torch.cat([tokens, torch.zeros(rows - n, dtype=tokens.dtype)])", "picked = torch.cat([tokens.flip(0), torch.zeros(rows - n, dtype=tokens.dtype)])"),
}


@pytest.mark.parametrize("why", sorted(MUTATIONS))
def test_the_reference_comparison_fails_when_the_emitted_text_is_broken(monkeypatch, tmp_path, why):
    old, new = MUTATIONS[why]
    text, _p = _generate()
    assert old in text, f"the mutation does not apply: {old!r} is not in the emitted text"
    with pytest.raises(AssertionError):
        _compare(monkeypatch, tmp_path, text.replace(old, new))


@pytest.mark.parametrize("why", sorted(GATHER_MUTATIONS))
def test_the_gather_comparison_fails_when_the_emitted_text_is_broken(monkeypatch, tmp_path, why):
    old, new = GATHER_MUTATIONS[why]
    monkeypatch.setattr(pt, "MOE_DENSE_MAX_EXPERTS", 0)
    text, _p = _generate()
    assert old in text, f"the mutation does not apply: {old!r} is not in the emitted text"
    with pytest.raises(AssertionError):
        _compare(monkeypatch, tmp_path, text.replace(old, new))


# ---- the fake ttnn op this block needs ---------------------------------------------------------------------------------------------------------

def test_the_fake_softmax_normalises_over_the_last_dimension():
    ttnn = fake.make()
    x = fake.Tensor(torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]]), layout="TILE")
    y = ttnn.softmax(x, dim=-1).t
    assert torch.allclose(y.sum(-1), torch.ones(1, 1, 1)) and y[0, 0, 0, 3] > y[0, 0, 0, 0]
