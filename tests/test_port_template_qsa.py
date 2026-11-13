# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Qwen4 indexer attention (`Qwen4ExpTextAttention` with its `Qwen4ExpTextQSAIndexer`) written by the port generator (issue 696, slice 5; issue 711).

The reference comparison runs the generated `_C_self_attn` through the torch-backed fake ttnn (tests/fake_ttnn_compute.py) against the real transformers
`Qwen4ExpTextAttention` at tiny sizes, on a prompt of 64 tokens against the indexer's token budget of 16, so that the block selection decides which tokens a
query sees. Three things are compared:
  - the generated block against the reference with a LOWEST-INDEX tie rule for `topk` (the rule `ttnn.topk(stable=True)` follows on a board, probe 7);
  - the generated block against the unmodified reference (torch.topk): the two differ only where block scores tie, and the test states how far;
  - the generated block with the selection switched off (plain causal attention) against the reference: it has to be far from the reference, so that a port
    that ignores the indexer (issue 638) cannot pass.
The comparison shows that the generated composition computes the reference's values with torch standing in for the device ops. It does not show that the
assembled block runs on a board: probes 6 to 9 ran each op form on a board (PCC 0.9998 or better), and the assembled block has not run."""
from __future__ import annotations

import copy
import contextlib

import pytest
import torch

from orchard import port_template as pt
import fake_ttnn_compute as fake
from test_port_template import _load, import_port, tiny_qwen4

QSA_TEST = pt.QSA_REFERENCE_TEST
SEQ = 64


def _generate(mutate_construction=None, mutate_tensors=None, mutate_config=None):
    con, tensors, cfg = _load("tiny-qwen4-construction.json"), _load("tiny-qwen4-tensors.json"), copy.deepcopy(_load("tiny-qwen4-config.json"))
    if mutate_construction:
        mutate_construction(con)
    if mutate_tensors:
        mutate_tensors(tensors)
    if mutate_config:
        mutate_config(cfg.get("text_config", cfg))
    return pt.generate(config=cfg, construction=con, tensors=tensors, inventory=_load("tiny-qwen4-ops-inventory.json"),
                       excerpts="class Qwen4ExpTextRMSNorm(nn.Module):\n    def forward(self, x):\n        output = output * (1.0 + self.weight.float())\n")


def _class_attrs(con, cls):
    return [v["attrs"] for c in con["classes"] if c["class"] == cls for v in c["variants"]]


# ---- the plan ---------------------------------------------------------------------------------------------------------------------------------

def test_the_reference_test_id_names_a_test_of_this_file():
    assert QSA_TEST == "tests/test_port_template_qsa.py::test_generated_qsa_attention_matches_the_transformers_module_with_the_lowest_index_tie_rule"
    assert QSA_TEST.split("::")[1] in globals()


def test_the_tiny_qwen4_plan_has_no_stub_left():
    text, p = tiny_qwen4()
    assert p.stubs == [] and pt.stub_lines(text) == []
    assert "qsa_attention" in p.summary()["generated"]
    assert [bool(l.composed) and not l.composed["stub_sublayer"] for l in p.layers] == [True, True, True]
    assert "ROTARY = " in text and "def _apply_rotary(" in text            # the block calls the rotary helpers, so the plan writes them
    assert text.count(pt.GENERATED_BLOCK_MARK + " qsa_attention") >= 5


def test_a_model_without_indexer_attention_gets_no_qsa_helpers():
    from test_port_template import std_generate
    text, p = std_generate()
    assert "qsa_attention" not in p.summary()["generated"] and "_qsa_attention" not in text


# ---- the plan refuses what it cannot prove: the block stays a stub ------------------------------------------------------------------------------

def _attn(con):
    return _class_attrs(con, "Qwen4ExpTextAttention")[0]


def _indexer(con):
    return _class_attrs(con, "Qwen4ExpTextQSAIndexer")[0]


def _reshape(name_part, rows=None, cols=None):
    def mutate(tensors):
        for t in tensors:
            if "self_attn." in t["name"] and t["name"].endswith(name_part):
                t["shape"] = [rows if rows is not None else t["shape"][0], cols if cols is not None else t["shape"][-1]] if len(t["shape"]) == 2 else [rows or t["shape"][0]]
    return mutate


def _set(getter, key, value):
    def mutate(con):
        getter(con)[key] = value
    return mutate


def _drop_child(getter, name):
    def mutate(con):
        getter(con)["children"].pop(name)
    return mutate


def _add_child(getter, name):
    def mutate(con):
        getter(con)["children"][name] = "Linear"
    return mutate


def _add_tensor(name):
    def mutate(tensors):
        tensors.append({"name": name, "dtype": "BF16", "shape": [64]})
    return mutate


def _dtype(part, dtype):
    def mutate(tensors):
        for t in tensors:
            if t["name"].endswith(part):
                t["dtype"] = dtype
    return mutate


def _all(*mutators):
    def mutate(x):
        for m in mutators:
            m(x)
    return mutate


def _rows(name_part, rows):
    def mutate(tensors):
        for t in tensors:
            if t["name"].endswith(name_part) and "layers.3." in t["name"]:
                t["shape"] = [rows] + list(t["shape"][1:])
    return mutate


def _cols(name_part, cols):
    def mutate(tensors):
        for t in tensors:
            if t["name"].endswith(name_part) and "layers.3." in t["name"]:
                t["shape"] = [t["shape"][0], cols]
    return mutate


def _width(name_part, n):
    def mutate(tensors):
        for t in tensors:
            if t["name"].endswith(name_part) and "layers.3." in t["name"]:
                t["shape"] = [n]
    return mutate


def _norm_of_another_class(con):
    """q_norm becomes a module of another class (listed under its own path): everything else is as in the recognized construction."""
    for c in con["classes"]:
        if c["class"] == "Qwen4ExpTextRMSNorm":
            for v in c["variants"]:
                if "model.layers.{i}.self_attn.q_norm" in v["paths"]:
                    v["paths"].remove("model.layers.{i}.self_attn.q_norm")
    con["classes"].append({"class": "Qwen4ExpTextNormX", "module": "x", "signature": [], "variants": [
        {"attrs": {"eps": 1e-06, "group_size": None, "children": {}}, "count": 1, "paths": ["model.layers.{i}.self_attn.q_norm"]}], "variants_left_out": 0})


# a head width of 16 with every shape that depends on it changed with it, so that only the tile guard can refuse
_HALF_HEAD = _all(_rows("self_attn.q_proj.weight", 256), _rows("self_attn.k_proj.weight", 32), _rows("self_attn.v_proj.weight", 32), _cols("self_attn.o_proj.weight", 128),
                  _rows("indexer.index_qk_proj.weight", 80), _width("self_attn.q_norm.weight", 16), _width("self_attn.k_norm.weight", 16),
                  _width("indexer.q_layernorm.weight", 16), _width("indexer.k_layernorm.weight", 16))
# an index head width of 64 with its shapes
_WIDE_INDEX = _all(_rows("indexer.index_qk_proj.weight", 480), _width("indexer.q_layernorm.weight", 96), _width("indexer.k_layernorm.weight", 96))      # 96: no group size of the construction divides it

REFUSALS = {
    "the head width is not a multiple of a tile": dict(mutate_construction=_all(_set(_attn, "head_dim", 16), _set(_attn, "scaling", 0.25), _set(_indexer, "index_head_dim", 16)),
                                                       mutate_tensors=_HALF_HEAD, mutate_config=lambda sec: sec.update(head_dim=16)),
    "the query projection has the wrong number of rows": dict(mutate_tensors=_reshape("self_attn.q_proj.weight", rows=544)),
    "the key projection has the wrong number of rows": dict(mutate_tensors=_reshape("self_attn.k_proj.weight", rows=96)),
    "the output projection has the wrong number of columns": dict(mutate_tensors=_reshape("self_attn.o_proj.weight", cols=288)),
    "the groups attribute disagrees with the head counts": dict(mutate_construction=_set(_attn, "num_key_value_groups", 2)),
    "the groups attribute is a float": dict(mutate_construction=_set(_attn, "num_key_value_groups", 4.0)),
    "the scaling is not head_dim ** -0.5": dict(mutate_construction=_set(_attn, "scaling", 0.25)),
    "the attention is not causal": dict(mutate_construction=_set(_attn, "is_causal", False)),
    "the attention has a child the block does not write": dict(mutate_construction=_add_child(_attn, "sinks")),
    "the attention lacks a child the block needs": dict(mutate_construction=_drop_child(_attn, "k_norm")),
    "the indexer has two key heads": dict(mutate_construction=_set(_indexer, "index_kv_heads", 2), mutate_tensors=_rows("indexer.index_qk_proj.weight", 192)),
    "the indexer head width is not a whole number of tiles": dict(mutate_construction=_set(_indexer, "index_head_dim", 48), mutate_tensors=_all(_rows("indexer.index_qk_proj.weight", 240), _width("indexer.q_layernorm.weight", 48), _width("indexer.k_layernorm.weight", 48))),
    "the compression ratio does not divide a tile": dict(mutate_construction=_all(_set(_indexer, "compress_ratio", 5), _set(_indexer, "token_budget", 20))),
    "the block top-k is not the budget over the ratio": dict(mutate_construction=_set(_indexer, "block_topk", 3)),
    "the budget is not a multiple of the ratio": dict(mutate_construction=_set(_indexer, "token_budget", 18)),
    "the indexer has a child the block does not write": dict(mutate_construction=_add_child(_indexer, "extra")),
    "the indexer projection has the wrong number of rows": dict(mutate_tensors=_reshape("indexer.index_qk_proj.weight", rows=128)),
    "a projection has a bias": dict(mutate_tensors=_add_tensor("model.layers.3.self_attn.q_proj.bias")),
    "a norm weight has the wrong width": dict(mutate_tensors=_reshape("indexer.k_layernorm.weight", rows=64)),
    "a weight is not a float tensor": dict(mutate_tensors=_dtype("self_attn.v_proj.weight", "I64")),
    "the rotary head width differs from the attention head width": dict(mutate_config=lambda sec: sec.update(head_dim=64)),
    "the indexer has no heads": dict(mutate_construction=_set(_indexer, "index_n_heads", 0)),
    "the compression ratio is one": dict(mutate_construction=_all(_set(_indexer, "compress_ratio", 1), _set(_indexer, "block_topk", 16))),
    "the key projection rows are not a whole number of heads": dict(mutate_tensors=_reshape("self_attn.k_proj.weight", rows=80)),
    "the key and value projections read another width than the query projection": dict(mutate_tensors=_all(_cols("self_attn.k_proj.weight", 128), _cols("self_attn.v_proj.weight", 128))),
    "the value projection differs from the key projection": dict(mutate_tensors=_reshape("self_attn.v_proj.weight", rows=32)),
    "a norm has no epsilon": dict(mutate_construction=lambda con: [v["attrs"].pop("eps", None) for c in con["classes"] if c["class"] == "Qwen4ExpTextRMSNorm" for v in c["variants"] if v["attrs"].get("group_size") is None]),
    "a norm is not an RMS norm": dict(mutate_construction=_norm_of_another_class),
    "the config gives no usable rotary settings": dict(mutate_config=lambda sec: sec.update(rope_parameters={"rope_type": "yarn", "factor": 2.0, "rope_theta": 10000.0})),
}


@pytest.mark.parametrize("why", sorted(REFUSALS))
def test_the_recognizer_refuses_what_it_cannot_prove(why):
    _text, p = _generate(**REFUSALS[why])
    assert "_C_self_attn" in {s.function for s in p.stubs}, why


def test_the_unmutated_construction_is_recognized():
    """The refusals above start from this plan: without it they would pass for any reason."""
    _text, p = _generate()
    assert "_C_self_attn" not in {s.function for s in p.stubs}


# ---- the reference comparison -----------------------------------------------------------------------------------------------------------------

def _config():
    config_mod = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")
    cfg = _load("tiny-qwen4-config.json")
    for key in ("architectures", "model_type"):
        cfg.pop(key, None)
    config = config_mod.Qwen4ExpTextConfig(**cfg)
    config._attn_implementation = "eager"
    return config


def _hf_attention(seed):
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    config = _config()
    torch.manual_seed(seed)
    attn = modeling.Qwen4ExpTextAttention(config, 3)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in attn.named_parameters():
            v = 0.25 * torch.randn(p.shape, generator=g)
            if "norm" in name or "layernorm" in name:
                v = 0.2 * torch.randn(p.shape, generator=g)
            p.copy_(v.to(torch.bfloat16).float())
    return modeling, config, attn.eval()


def _checkpoint(tmp_path, attn):
    from safetensors.torch import save_file
    weights = {}
    for t in _load("tiny-qwen4-tensors.json"):
        weights[t["name"]] = (torch.zeros(t["shape"], dtype=torch.int64) if t["dtype"] != "BF16" else (0.1 * torch.randn(*t["shape"])).to(torch.bfloat16))
    for k, v in attn.state_dict().items():
        weights[f"model.layers.3.self_attn.{k}"] = v.detach().to(torch.bfloat16)
    snap = tmp_path / "snap"
    snap.mkdir()
    save_file(weights, str(snap / "model.safetensors"))
    return snap


def _lowest_index_topk(self, k, dim=-1, largest=True, sorted=True):
    """torch.topk with the tie rule of ttnn.topk(stable=True): among equal scores the lower index comes first."""
    order = torch.sort(self, dim=dim, descending=largest, stable=True)
    sl = [slice(None)] * self.dim()
    sl[dim] = slice(0, k)
    return torch.return_types.topk((order.values[tuple(sl)], order.indices[tuple(sl)]))


@contextlib.contextmanager
def _topk_rule(rule):
    """Run the reference with the given tie rule for topk. The reference indexer scatters with int32 indices, which torch 2.7 refuses; the index is cast to int64
    here (the same values), so that the reference runs under the interpreter the tests use."""
    original_topk, original_scatter = torch.Tensor.topk, torch.Tensor.scatter
    if rule == "lowest-index":
        torch.Tensor.topk = _lowest_index_topk
    torch.Tensor.scatter = lambda self, dim, index, *args, **kw: original_scatter(self, dim, index.long(), *args, **kw)
    try:
        yield
    finally:
        torch.Tensor.topk, torch.Tensor.scatter = original_topk, original_scatter


def _reference(modeling, config, attn, x_host, rule):
    seq = x_host.shape[1]
    rotary = modeling.Qwen4ExpTextRotaryEmbedding(config)
    cos, sin = rotary(x_host.float(), torch.arange(seq).unsqueeze(0))
    mask = torch.full((seq, seq), torch.finfo(torch.float32).min).triu(1)[None, None]
    with torch.no_grad(), _topk_rule(rule):
        out, _ = attn(x_host.float(), (cos.float(), sin.float()), mask)
    return out.float()


def _setup(monkeypatch, tmp_path, seed=7):
    text, _p = tiny_qwen4()
    mod = import_port(monkeypatch, text, tmp_path)
    modeling, config, attn = _hf_attention(seed)
    model = mod.load(str(_checkpoint(tmp_path, attn)), fake.MeshDevice(1))
    x_host = (torch.randn(1, SEQ, 64, generator=torch.Generator().manual_seed(3)) * 0.8).to(torch.bfloat16)
    x = fake.Tensor(x_host.reshape(1, 1, SEQ, 64).float(), layout="TILE")
    return mod, model, modeling, config, attn, x_host, x


def _tables(modeling, config, x_host):
    seq = x_host.shape[1]
    cos, sin = modeling.Qwen4ExpTextRotaryEmbedding(config)(x_host.float(), torch.arange(seq).unsqueeze(0))
    return cos.float(), sin.float()


def _reference_selection(modeling, config, attn, x_host, rule):
    """(visible [S, S] bool, scores [S, nb] float32 with -1 for blocks that are not complete for the query), from the reference indexer's own layers and rule."""
    seq = x_host.shape[1]
    cos, sin = _tables(modeling, config, x_host)
    causal_float = torch.full((seq, seq), torch.finfo(torch.float32).min).triu(1)[None, None]
    ind = attn.indexer
    ratio, dim, heads = ind.compress_ratio, ind.index_head_dim, ind.index_n_heads
    nb = seq // ratio
    with torch.no_grad(), _topk_rule(rule):
        mask = ind(x_host.float(), (cos, sin), causal_float, None)
        qk = ind.index_qk_proj(x_host.float())
        q, keys = torch.split(qk, [heads * dim, dim], dim=-1)
        q = modeling.apply_rotary_pos_emb(ind.q_layernorm(q.reshape(1, seq, heads, dim)), cos=cos, sin=sin, unsqueeze_dim=2)
        scores = torch.full((seq, nb), -1.0)
        for qi in range(seq):
            done = (qi + 1) // ratio
            if done == 0:
                continue
            pooled = ind.k_layernorm(keys[0, :done * ratio].reshape(done, ratio, dim).float().mean(dim=1))
            starts = torch.arange(done) * ratio
            block_keys = modeling.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos[0].index_select(0, starts), sin=sin[0].index_select(0, starts)).squeeze(1)
            scores[qi, :done] = torch.relu(torch.matmul(q[0, qi].float(), block_keys.float().T).T).sum(dim=-1) / dim ** 0.5
    return (mask[0, 0] == 0) & torch.ones(seq, seq, dtype=torch.bool).tril(), scores


def _blocks(row_mask, ratio):
    return {int(t) // ratio for t in row_mask.nonzero().flatten().tolist()}


def _boundary_rows(got_mask, want_mask, scores, ratio):
    """The rows whose selected blocks differ, each checked: every block that one side has and the other does not scores within bf16 resolution (1.5 percent) of
    the best block that the other side has and this side does not. These are the rows where the device's bf16 scores may pick the other one of two equal blocks."""
    rows = []
    for r in range(got_mask.shape[0]):
        if torch.equal(got_mask[r], want_mask[r]):
            continue
        only_got, only_want = _blocks(got_mask[r] & ~want_mask[r], ratio), _blocks(want_mask[r] & ~got_mask[r], ratio)
        for b in only_got | only_want:
            others = only_want if b in only_got else only_got
            gaps = [abs(float(scores[r, b] - scores[r, o])) / max(float(scores[r, b]), 1e-6) for o in others]
            assert others and 0 < min(gaps) < 0.015, (r, b, scores[r])        # an EXACT tie is not a boundary row: the tie rule decides it, and the block has the reference rule
        rows.append(r)
    return rows


def _selection_mask_of(mod, model, x, cfg):
    gen = mod._qsa_selection_mask(model, "model.layers.3.self_attn.indexer", x, cfg["index_heads"], cfg["index_dim"], cfg["ratio"], cfg["budget"])
    return gen.t.reshape(SEQ, SEQ) > -15000


CFG = {"index_heads": 4, "index_dim": 32, "ratio": 4, "budget": 16}


def _row_errors(got, want):
    return (got - want).abs().amax(dim=(0, 2))


def test_generated_qsa_attention_matches_the_transformers_module_with_the_lowest_index_tie_rule(monkeypatch, tmp_path):
    mod, model, modeling, config, attn, x_host, x = _setup(monkeypatch, tmp_path)
    assert SEQ > config.indexer_budget + config.indexer_compress_ratio - 1, "the prompt must be longer than the budget for the selection to decide anything"
    want = _reference(modeling, config, attn, x_host, "lowest-index")
    got = mod._C_self_attn(model, 3, x, list(range(SEQ))).t.float().reshape(want.shape)
    scale = want.abs().max().item()
    assert scale > 0.3, f"the reference output is too small ({scale}) for the comparison to mean anything"
    want_mask, scores = _reference_selection(modeling, config, attn, x_host, "lowest-index")
    boundary = _boundary_rows(_selection_mask_of(mod, model, x, CFG), want_mask, scores, CFG["ratio"])
    assert len(boundary) <= 3, f"the block selection differs from the reference in {len(boundary)} rows: {boundary}"
    errors = _row_errors(got, want)
    worst = max(float(e) for r, e in enumerate(errors) if r not in boundary)
    assert worst <= 0.06 * scale, f"max abs error {worst} (rows with the same selection as the reference) against scale {scale}"


def test_the_selection_decides_the_output_so_a_port_that_ignores_the_indexer_fails(monkeypatch, tmp_path):
    """Issue 638: a port whose attention is plain causal attention gives a different answer on a prompt longer than the budget. The generated block with its
    selection switched off is far from the reference, on the rows past the budget, by many times the tolerance of the comparison above."""
    mod, model, modeling, config, attn, x_host, x = _setup(monkeypatch, tmp_path)
    want = _reference(modeling, config, attn, x_host, "lowest-index")
    scale = want.abs().max().item()
    causal = fake.Tensor(torch.where(torch.ones(SEQ, SEQ).tril() > 0, 0.0, -30000.0).reshape(1, 1, SEQ, SEQ), layout="TILE")
    monkeypatch.setattr(mod, "_qsa_selection_mask", lambda *a, **k: causal)
    got = mod._C_self_attn(model, 3, x, list(range(SEQ))).t.float().reshape(want.shape)
    assert float(_row_errors(got, want)[config.indexer_budget + config.indexer_compress_ratio:].max()) > 0.12 * scale


def _tied_input(x_host, keep_blocks, ratio):
    """x with every token of the other blocks set to zero: their keys are zero, so their pooled keys and scores are exactly 0 and tie at the boundary."""
    x2 = x_host.clone()
    for t in range(x2.shape[1]):
        if t // ratio not in keep_blocks:
            x2[:, t] = 0
    return x2


def test_the_unmodified_reference_differs_only_where_scores_tie(monkeypatch, tmp_path):
    """The reference runs torch.topk, which does not take the lowest index among tied scores; the generated block follows the lowest-index rule (the device rule,
    probe 7). On an input where all but three blocks score exactly 0, rows with fewer positive blocks than the top-k have tied scores at the boundary. The two
    references pick different blocks on some of those rows, and the generated block matches the lowest-index one. The test states how far the generated block
    is from the unmodified reference."""
    mod, model, modeling, config, attn, x_host, x = _setup(monkeypatch, tmp_path)
    x_host = _tied_input(x_host, {0, 7, 11}, CFG["ratio"])
    x = fake.Tensor(x_host.reshape(1, 1, SEQ, 64).float(), layout="TILE")
    want_low = _reference(modeling, config, attn, x_host, "lowest-index")
    want_torch = _reference(modeling, config, attn, x_host, "torch")
    got = mod._C_self_attn(model, 3, x, list(range(SEQ))).t.float().reshape(want_low.shape)
    scale = want_low.abs().max().item()
    mask_low, scores = _reference_selection(modeling, config, attn, x_host, "lowest-index")
    mask_torch, _ = _reference_selection(modeling, config, attn, x_host, "torch")
    tie_rows = [r for r in range(SEQ) if not torch.equal(mask_low[r], mask_torch[r])]
    assert tie_rows, "the input has to make torch.topk and the lowest-index rule differ, or the test shows nothing about the tie rule"
    for r in tie_rows:     # every row where the two rules differ has tied scores at the boundary
        done = (r + 1) // CFG["ratio"]
        top = torch.sort(scores[r, :done], descending=True, stable=True).values
        k = min(CFG["budget"] // CFG["ratio"], done)
        assert k < done and float(top[k - 1]) == float(top[k]), (r, top)
    boundary = _boundary_rows(_selection_mask_of(mod, model, x, CFG), mask_low, scores, CFG["ratio"])
    errors = _row_errors(got, want_low)
    assert max(float(e) for r, e in enumerate(errors) if r not in boundary) <= 0.06 * scale
    to_torch = _row_errors(got, want_torch)
    assert float(to_torch[tie_rows].max()) > 0.06 * scale         # the generated block follows the lowest-index rule: on the tie rows it is NOT the unmodified reference
    print(f"tie rows {len(tie_rows)} of {SEQ}; max abs distance to the unmodified reference {float(to_torch.max()):.3f} "
          f"(output scale {scale:.3f}); to the lowest-index reference {float(errors.max()):.3f}")


# ---- mutations of the emitted text: each must turn a comparison red ---------------------------------------------------------------------------

def _run_with(monkeypatch, tmp_path, old, new, test):
    text, _p = tiny_qwen4()
    if old not in text:
        pytest.fail(f"the mutation does not apply, the test would pass for the wrong reason: {old[:80]}")
    monkeypatch.setattr(pt, "render", lambda p: text.replace(old, new))
    test(monkeypatch, tmp_path)


MUTATIONS = {
    "the pooled keys are not rotated at the block start": ("pooled = _to_host(model, _qsa_apply_rotary(model, pooled, block_cos, block_sin)).float().reshape(nb, index_dim)",
                                                           "pooled = _to_host(model, pooled).float().reshape(nb, index_dim)"),
    "the block score drops its relu": ("ttnn.sum(ttnn.relu(ttnn.matmul(q, pk)), dim=1, keepdim=True)", "ttnn.sum(ttnn.matmul(q, pk), dim=1, keepdim=True)"),
    "a block that is not complete for the query keeps its score": ("scores = ttnn.add(ttnn.multiply(scores, const[\"valid\"]), const[\"bias\"])",
                                                                   "scores = ttnn.add(scores, const[\"bias\"])"),
    "the selection keeps blocks that are not complete": ("* const[\"valid_host\"].to(torch.float32)", "* 1.0"),
    "the tail is not added to the mask": ("visible = ttnn.add(ttnn.matmul(selected, const[\"expand\"]), const[\"tail\"])",
                                          "visible = ttnn.matmul(selected, const[\"expand\"])"),
    "the index queries are not normalised": ("q = _qsa_apply_rotary(model, _qsa_norm_heads(model, prefix + \".q_layernorm\", q), cos, sin)\n    keys",
                                             "q = _qsa_apply_rotary(model, q, cos, sin)\n    keys"),
    "the output gate is not applied": ("ttnn.multiply(joined, ttnn.sigmoid(gate))", "joined"),
    "the attention queries are not normalised": ("q = _apply_rotary(model, _qsa_norm_heads(model, prefix + \".q_norm\", q), cos, sin)", "q = _apply_rotary(model, q, cos, sin)"),
    "the attention keys are not rotated": ("k = _apply_rotary(model, k, cos, sin)\n    if mode == \"step\":", "if mode == \"step\":"),
    "the attention scale is wrong": ("is_causal=False, scale=hd ** -0.5)", "is_causal=False, scale=1.0)"),
    "the query projection is split into gate and query the wrong way round": ("q = ttnn.permute(ttnn.slice(qg, (0, 0, 0, 0), (1, seq, heads, hd)), (0, 2, 1, 3))\n    gate = ttnn.reshape(ttnn.slice(qg, (0, 0, 0, hd), (1, seq, heads, 2 * hd)), (1, 1, seq, heads * hd))",
                                                                              "q = ttnn.permute(ttnn.slice(qg, (0, 0, 0, hd), (1, seq, heads, 2 * hd)), (0, 2, 1, 3))\n    gate = ttnn.reshape(ttnn.slice(qg, (0, 0, 0, 0), (1, seq, heads, hd)), (1, 1, seq, heads * hd))"),
}


@pytest.mark.parametrize("why", sorted(MUTATIONS))
def test_the_reference_comparison_fails_when_the_emitted_text_is_broken(monkeypatch, tmp_path, why):
    old, new = MUTATIONS[why]
    with pytest.raises(AssertionError):
        _run_with(monkeypatch, tmp_path, old, new, test_generated_qsa_attention_matches_the_transformers_module_with_the_lowest_index_tie_rule)


def test_the_tie_test_fails_when_topk_is_not_stable(monkeypatch, tmp_path):
    """The tie rule is a guard of its own: with `stable=False` the fake (like the device, probe 7) does not take the lowest index among tied scores, and the
    rows where the two rules differ are no longer matched by the block."""
    with pytest.raises(AssertionError):
        _run_with(monkeypatch, tmp_path, "largest=True, sorted=True, stable=True)", "largest=True, sorted=True, stable=False)", test_the_unmodified_reference_differs_only_where_scores_tie)


# ---- the fake ttnn ops this block needs ---------------------------------------------------------------------------------------------------------

def test_the_fake_topk_takes_the_lowest_index_among_equal_scores_only_when_stable():
    ttnn = fake.make()
    scores = fake.Tensor(torch.zeros(1, 1, 3, 8) + torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]), layout="TILE")
    _v, idx = ttnn.topk(scores, 2, dim=-1, largest=True, sorted=True, stable=True)
    assert idx.t.reshape(3, 2).tolist() == [[1, 2]] * 3


# ---- Construction.resolve with the width of the weight vector -------------------------------------------------------------------------------------

def test_resolve_drops_a_group_size_that_cannot_divide_the_width_of_an_unlisted_norm():
    """The construction's `paths` lists are cut: the indexer's `k_layernorm` is in none of them, so resolve falls back to every variant of the class and takes the
    first, the hyper-connection norm with group size 64. A norm of width 32 cannot have group size 64, so the width removes that variant (before: the norm was
    taken for a grouped one, had no NORM_EPS entry, and `_rms_norm` raised for it)."""
    con = pt.Construction(_load("tiny-qwen4-construction.json"))
    path = "model.layers.{i}.self_attn.indexer.k_layernorm"
    assert con.resolve(path)[1]["group_size"] == 64                      # without the width: the first variant, as before
    assert con.resolve(path, width=32)[1]["group_size"] is None
    assert con.resolve(path, width=128)[1]["group_size"] == 64           # a width both variants fit: nothing is removed, the first variant stays
    assert con.resolve("model.layers.{i}.attn_hyper_connection.hc_norm", width=32)[1]["group_size"] == 64      # a listed path is never reconsidered


def test_the_indexer_norm_the_construction_does_not_list_gets_an_epsilon_entry_and_a_norm_placement():
    text, p = tiny_qwen4()
    assert "model.layers.{i}.self_attn.indexer.k_layernorm" in dict(eval(next(l for l in text.split("\n") if l.startswith("NORM_EPS = ")).split(" = ", 1)[1]))
    assert p.placements["model.layers.{i}.self_attn.indexer.k_layernorm.weight"] == "norm_plus_one"
