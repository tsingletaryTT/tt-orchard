# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The generated QSA attention at an index head width that differs from the attention head width (issue 696; Flash-Next has index_head_dim 128 and head_dim 256).

The slice 5 recognizer refused any model whose indexer head width differed from the attention head width, so the real Flash-Next config left `_C_self_attn` a stub.
The reference (`Qwen4ExpTextQSAIndexer`) rotates the first `rotary_dim` columns of its index vectors with the same cos and sin tables as the attention, which are
`rot` columns wide, so an index vector of width D takes the tables padded to D with 1 and 0, and needs D >= rot. These tests build the attention at a pair (head_dim,
index_head_dim), run the generated block through the torch-backed fake ttnn and compare it with the real transformers `Qwen4ExpTextAttention`, on a prompt longer than the
token budget, with the lowest-index tie rule (the device rule, probe 7). The comparison shows the composition at these widths; it does not show that the ops run on a board
at these widths (see the project log entry for the probes that cover them)."""
from __future__ import annotations

import pytest
import torch

from orchard import port_template as pt
import fake_ttnn_compute as fake
from test_port_template import import_port
import test_port_template_qsa as q

HIDDEN, HEADS, KV = 64, 8, 2
PAIRS = [(64, 32), (128, 32), (32, 96)]          # (head_dim, index_head_dim): a 2:1 pair like Flash-Next, a 4:1 pair whose rotary part (32) fills the index vector, and an index vector wider than the head
# index_head_dim 64 is left out: the fixture's hyper-connection norms have group_size 64, and the construction of an unlisted norm of width 64 would take that group size


def _mutators(hd, dim):
    """The fixture's construction, tensors and config with the attention at head_dim `hd` and the indexer at index_head_dim `dim` (heads 8, kv heads 2, hidden 64)."""
    con = q._all(q._set(q._attn, "head_dim", hd), q._set(q._attn, "scaling", hd ** -0.5), q._set(q._indexer, "index_head_dim", dim))
    tensors = q._all(q._rows("self_attn.q_proj.weight", HEADS * 2 * hd), q._rows("self_attn.k_proj.weight", KV * hd), q._rows("self_attn.v_proj.weight", KV * hd),
                     q._cols("self_attn.o_proj.weight", HEADS * hd), q._rows("indexer.index_qk_proj.weight", 5 * dim),
                     q._width("self_attn.q_norm.weight", hd), q._width("self_attn.k_norm.weight", hd),
                     q._width("indexer.q_layernorm.weight", dim), q._width("indexer.k_layernorm.weight", dim))
    return dict(mutate_construction=con, mutate_tensors=tensors, mutate_config=lambda sec: sec.update(head_dim=hd, indexer_head_dim=dim))


def _plan(hd, dim):
    return q._generate(**_mutators(hd, dim))


def _config(hd, dim):
    config = q._config()
    config.head_dim, config.indexer_head_dim = hd, dim
    return config


def _setup(monkeypatch, tmp_path, hd, dim, seed=7):
    text, p = _plan(hd, dim)
    assert "_C_self_attn" not in {s.function for s in p.stubs}, "the recognizer refused: there is nothing to compare"
    mod = import_port(monkeypatch, text, tmp_path)
    modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    config = _config(hd, dim)
    torch.manual_seed(seed)
    attn = modeling.Qwen4ExpTextAttention(config, 3)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, prm in attn.named_parameters():
            v = (0.2 if ("norm" in name or "layernorm" in name) else 0.25) * torch.randn(prm.shape, generator=g)
            prm.copy_(v.to(torch.bfloat16).float())
    attn.eval()
    model = mod.load(str(q._checkpoint(tmp_path, attn)), fake.MeshDevice(1))
    x_host = (torch.randn(1, q.SEQ, HIDDEN, generator=torch.Generator().manual_seed(3)) * 0.8).to(torch.bfloat16)
    x = fake.Tensor(x_host.reshape(1, 1, q.SEQ, HIDDEN).float(), layout="TILE")
    return mod, model, modeling, config, attn, x_host, x


@pytest.mark.parametrize("hd,dim", PAIRS)
def test_the_recognizer_accepts_an_index_head_width_that_differs_from_the_attention_head_width(hd, dim):
    _text, p = _plan(hd, dim)
    assert "_C_self_attn" not in {s.function for s in p.stubs}


@pytest.mark.parametrize("hd,dim", PAIRS)
def test_generated_qsa_attention_matches_the_reference_at_asymmetric_widths(monkeypatch, tmp_path, hd, dim):
    mod, model, modeling, config, attn, x_host, x = _setup(monkeypatch, tmp_path, hd, dim)
    assert q.SEQ > config.indexer_budget + config.indexer_compress_ratio - 1
    want = q._reference(modeling, config, attn, x_host, "lowest-index")
    got = mod._C_self_attn(model, 3, x, list(range(q.SEQ))).t.float().reshape(want.shape)
    scale = want.abs().max().item()
    assert scale > 0.3, f"the reference output is too small ({scale})"
    want_mask, scores = q._reference_selection(modeling, config, attn, x_host, "lowest-index")
    cfg = {"index_heads": 4, "index_dim": dim, "ratio": 4, "budget": 16}
    boundary = q._boundary_rows(q._selection_mask_of(mod, model, x, cfg), want_mask, scores, cfg["ratio"])
    assert len(boundary) <= 3, f"the selection differs from the reference in {len(boundary)} rows: {boundary}"
    errors = q._row_errors(got, want)
    worst = max(float(e) for r, e in enumerate(errors) if r not in boundary)
    assert worst <= 0.06 * scale, f"max abs error {worst} against scale {scale}"


@pytest.mark.parametrize("hd,dim", PAIRS[:2])
def test_a_port_that_ignores_the_indexer_fails_at_asymmetric_widths(monkeypatch, tmp_path, hd, dim):
    mod, model, modeling, config, attn, x_host, x = _setup(monkeypatch, tmp_path, hd, dim)
    want = q._reference(modeling, config, attn, x_host, "lowest-index")
    scale = want.abs().max().item()
    causal = fake.Tensor(torch.where(torch.ones(q.SEQ, q.SEQ).tril() > 0, 0.0, -30000.0).reshape(1, 1, q.SEQ, q.SEQ), layout="TILE")
    monkeypatch.setattr(mod, "_qsa_selection_mask", lambda *a, **k: causal)
    got = mod._C_self_attn(model, 3, x, list(range(q.SEQ))).t.float().reshape(want.shape)
    assert float(q._row_errors(got, want)[config.indexer_budget + config.indexer_compress_ratio:].max()) > 0.12 * scale


# ---- what is still refused ----------------------------------------------------------------------------------------------------------------------

REFUSE = {
    "the rotary part is wider than the index vector": (256, 32),                  # rot = 64 > 32: the index vector has fewer columns than the tables rotate
    "the index head width is not a multiple of a tile": (64, 48),
    "the index head width is under a tile": (64, 16),
}


@pytest.mark.parametrize("why", sorted(REFUSE))
def test_the_recognizer_still_refuses_an_index_width_it_cannot_prove(why):
    hd, dim = REFUSE[why]
    mutators = _mutators(hd, dim)
    _text, p = q._generate(**mutators)
    assert "_C_self_attn" in {s.function for s in p.stubs}, why


# ---- the emitted text keeps the index width apart from the head width ----------------------------------------------------------------------------

def _run_with(monkeypatch, tmp_path, hd, dim, old, new, test):
    text, _p = _plan(hd, dim)
    if old not in text:
        pytest.fail(f"the mutation does not apply, the test would pass for the wrong reason: {old[:80]}")
    monkeypatch.setattr(pt, "render", lambda p: text.replace(old, new))
    test(monkeypatch, tmp_path, hd, dim)


ASYM_MUTATIONS = {
    "the index rotary tables use the attention head width": ("block_cos, block_sin = _qsa_rope_tables(model, torch.arange(nb) * ratio, index_dim)",
                                                              "block_cos, block_sin = _qsa_rope_tables(model, torch.arange(nb) * ratio, ROTARY[\"head_dim\"])"),
    "the pooled keys are not rotated": ("pooled = _to_host(model, _qsa_apply_rotary(model, pooled, block_cos, block_sin)).float().reshape(nb, index_dim)",
                                        "pooled = _to_host(model, pooled).float().reshape(nb, index_dim)"),
}


@pytest.mark.parametrize("why", sorted(ASYM_MUTATIONS))
def test_the_asymmetric_comparison_fails_when_the_emitted_text_is_broken(monkeypatch, tmp_path, why):
    old, new = ASYM_MUTATIONS[why]
    with pytest.raises(Exception):
        _run_with(monkeypatch, tmp_path, 64, 32, old, new, test_generated_qsa_attention_matches_the_reference_at_asymmetric_widths)


def test_the_index_rotary_helper_names_the_layout_it_needs(monkeypatch, tmp_path):
    text, _p = _plan(64, 32)
    mod = import_port(monkeypatch, text, tmp_path)
    model = {"mesh_device": fake.MeshDevice(1), "ck": None, "dev": {}, "host": {}}
    rot = mod.ROTARY["rot"]
    assert rot == 16
    narrow = fake.Tensor(torch.zeros(1, 2, 8, rot // 2), layout="TILE")           # narrower than the rotary part
    table = fake.Tensor(torch.zeros(1, 1, 8, rot // 2), layout="TILE")
    with pytest.raises(ValueError, match=r"width >= 16"):
        mod._qsa_apply_rotary(model, narrow, table, table)
    wrong = fake.Tensor(torch.zeros(1, 2, 8, 32), layout="TILE")
    short_table = fake.Tensor(torch.zeros(1, 1, 4, 32), layout="TILE")             # a table of another length
    with pytest.raises(ValueError, match=r"_qsa_apply_rotary needs x"):
        mod._qsa_apply_rotary(model, wrong, short_table, short_table)
