# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The host decomposition of the Stable Audio 3 diffusion transformer layer (orchard/skills/audio-chip-templates/chip_dit.py).

The chip code in that file repeats the decomposition `host_layer` uses, so these tests check the decomposition against the library's own
`TransformerBlock` (when the library's source is on disk) and against the formulas written out directly (always). Nothing here uses a chip.
"""
import importlib.util
import math
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODULE = Path(__file__).resolve().parents[1] / "orchard" / "skills" / "audio-chip-templates" / "chip_dit.py"
LIB_SRC = Path(os.environ.get("ORCHARD_SA3_SRC", str(Path.home() / "orchard-src" / "stable-audio-3")))


def load_module():
    spec = importlib.util.spec_from_file_location("chip_dit_under_test", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cd = load_module()


def library_block_class():
    """The library's TransformerBlock, or skip when its source is not on disk."""
    if not (LIB_SRC / "stable_audio_3" / "models" / "transformer.py").exists():
        pytest.skip("the stable-audio-3 source is not on disk (set ORCHARD_SA3_SRC)")
    sys.path.insert(0, str(LIB_SRC))
    try:
        from stable_audio_3.models.transformer import TransformerBlock, RotaryEmbedding
    finally:
        sys.path.remove(str(LIB_SRC))
    return TransformerBlock, RotaryEmbedding


DIM, LOCAL = 128, 17


def make_block(TransformerBlock, seed=0, differential=False):
    torch.manual_seed(seed)
    blk = TransformerBlock(DIM, dim_heads=64, cross_attend=True, dim_context=DIM, global_cond_dim=DIM, local_add_cond_dim=LOCAL,
                           zero_init_branch_outputs=False, add_rope=False, norm_type="rms_norm",
                           attn_kwargs={"qk_norm": "rms", "differential": differential}, norm_kwargs={"force_fp32": True}, ff_kwargs={"mult": 4.0})
    blk.eval()
    with torch.no_grad():                    # every gain and bias away from its initial value, so a wrong tensor changes the answer
        for p in blk.parameters():
            p.add_(0.2 * torch.randn_like(p))
        blk.to_scale_shift_gate.mul_(3.0)
    return blk


def params_of(blk):
    return {k: v.detach().float() for k, v in blk.state_dict().items()}


def run_host(blk, x, ctx, gc, local, pmask, mem=cd.MEMORY_TOKENS):
    """host_layer on x [n_real, D] (memory rows included) padded to a tile multiple; returns [n_real, D]."""
    n_real = x.shape[0]
    n = cd.padded_length(n_real)
    cp = cd.padded_length(ctx.shape[0])
    xp = torch.zeros(n, DIM)
    xp[:n_real] = x
    cpad = torch.zeros(cp, DIM)
    cpad[:ctx.shape[0]] = ctx
    t_len = n_real - mem
    lrows = torch.zeros(n, LOCAL)
    lrows[mem:mem + t_len] = local
    row = torch.zeros(n)
    row[mem:mem + t_len] = 1
    vm = cd.value_mask(torch, pmask, n, mem)
    cos, sin = cd.rotary_tables(torch, n_real, n)
    mod = cd.modulation(torch, blk.to_scale_shift_gate.detach(), gc)
    out = cd.host_layer(torch, xp, params_of(blk), mod, cpad, lrows, row, vm, n_real, ctx.shape[0], cos, sin, cd.rotate_matrix(torch))
    return out[:n_real]


def library_forward(RotaryEmbedding, blk, x, ctx, gc, local, pmask, mem=cd.MEMORY_TOKENS):
    n_real = x.shape[0]
    rope = RotaryEmbedding(32)
    freqs = rope.forward_from_seq_len(n_real)
    extended = None
    if pmask is not None:
        extended = torch.cat([torch.ones(1, mem, dtype=torch.bool), pmask.bool().unsqueeze(0)], dim=-1)
    # the library's `local_add_cond` is [B, T, C] and is left-padded to the sequence length inside the block
    with torch.no_grad():
        return blk(x.unsqueeze(0), context=ctx.unsqueeze(0), global_cond=gc.unsqueeze(0), local_add_cond=local.unsqueeze(0),
                   rotary_pos_emb=freqs, padding_mask=extended)[0]


def case(t_len, ctx_len, valid=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    n_real = cd.MEMORY_TOKENS + t_len
    x = torch.randn(n_real, DIM, generator=g)
    ctx = torch.randn(ctx_len, DIM, generator=g)
    gc = 0.5 * torch.randn(6 * DIM, generator=g)
    local = torch.randn(t_len, LOCAL, generator=g)
    pmask = None if valid is None else (torch.arange(t_len) < valid)
    return x, ctx, gc, local, pmask


CASES = [
    pytest.param(32, 257, None, id="tile aligned, no padding mask"),
    pytest.param(30, 257, None, id="tile padding of 2 rows"),
    pytest.param(30, 257, 21, id="padding mask on the last 9 latent rows"),
    pytest.param(50, 33, 50, id="short context, mask all valid"),
]


@pytest.mark.parametrize("t_len, ctx_len, valid", CASES)
def test_layer_matches_the_library_block(t_len, ctx_len, valid):
    TransformerBlock, RotaryEmbedding = library_block_class()
    blk = make_block(TransformerBlock)
    x, ctx, gc, local, pmask = case(t_len, ctx_len, valid)
    want = library_forward(RotaryEmbedding, blk, x, ctx, gc, local, pmask)
    got = run_host(blk, x, ctx, gc, local, pmask)
    assert torch.allclose(got, want, atol=2e-4, rtol=1e-4), (got - want).abs().max()


@pytest.mark.parametrize("t_len, ctx_len, valid", CASES)
def test_differential_layer_matches_the_library_block(t_len, ctx_len, valid):
    """The medium model's layers use differential attention in both the self and the cross section (five and three projections)."""
    TransformerBlock, RotaryEmbedding = library_block_class()
    blk = make_block(TransformerBlock, differential=True)
    x, ctx, gc, local, pmask = case(t_len, ctx_len, valid)
    want = library_forward(RotaryEmbedding, blk, x, ctx, gc, local, pmask)
    got = run_host(blk, x, ctx, gc, local, pmask)
    assert torch.allclose(got, want, atol=2e-4, rtol=1e-4), (got - want).abs().max()


def test_the_differential_decomposition_is_not_the_plain_one():
    """Mutation check: the plain-attention reading of the same tensors (first three projections only) must disagree with the library."""
    TransformerBlock, RotaryEmbedding = library_block_class()
    blk = make_block(TransformerBlock, differential=True)
    x, ctx, gc, local, pmask = case(30, 257, 21)
    want = library_forward(RotaryEmbedding, blk, x, ctx, gc, local, pmask)
    p = params_of(blk)
    p["self_attn.to_qkv.weight"] = p["self_attn.to_qkv.weight"][:3 * DIM]          # drop the difference branch
    p["cross_attn.to_q.weight"] = p["cross_attn.to_q.weight"][:DIM]
    p["cross_attn.to_kv.weight"] = p["cross_attn.to_kv.weight"][:2 * DIM]
    n_real = x.shape[0]
    n = cd.padded_length(n_real)
    cp = cd.padded_length(ctx.shape[0])
    xp = torch.zeros(n, DIM)
    xp[:n_real] = x
    cpad = torch.zeros(cp, DIM)
    cpad[:ctx.shape[0]] = ctx
    lrows = torch.zeros(n, LOCAL)
    lrows[cd.MEMORY_TOKENS:n_real] = local
    row = torch.zeros(n)
    row[cd.MEMORY_TOKENS:n_real] = 1
    cos, sin = cd.rotary_tables(torch, n_real, n)
    mod = cd.modulation(torch, blk.to_scale_shift_gate.detach(), gc)
    plain = cd.host_layer(torch, xp, p, mod, cpad, lrows, row, cd.value_mask(torch, pmask, n, cd.MEMORY_TOKENS), n_real, ctx.shape[0], cos, sin,
                          cd.rotate_matrix(torch))[:n_real]
    assert not torch.allclose(plain, want, atol=1e-3)


def test_the_library_block_is_sensitive_to_every_part_the_decomposition_covers():
    """The comparison above is only a check when each piece changes the output. Zeroing a piece's input must change the library's answer."""
    TransformerBlock, RotaryEmbedding = library_block_class()
    blk = make_block(TransformerBlock)
    x, ctx, gc, local, pmask = case(30, 257, 21)
    base = library_forward(RotaryEmbedding, blk, x, ctx, gc, local, pmask)
    for name, args in {"context": (x, ctx * 0, gc, local, pmask), "local": (x, ctx, gc, local * 0, pmask),
                       "global": (x, ctx, gc * 0, local, pmask), "mask": (x, ctx, gc, local, None)}.items():
        changed = library_forward(RotaryEmbedding, blk, *args)
        assert (changed - base).abs().max() > 1e-3, f"{name} does not affect the output; the case would not detect a break there"


# --- parts that need no library --------------------------------------------------------------------------------------------------------

def test_rotate_matrix_is_rotate_half_on_the_first_32_columns():
    t = torch.randn(5, cd.HEAD_DIM)
    r = cd.rotate_matrix(torch)
    x1, x2 = t[:, :16], t[:, 16:32]
    want = torch.zeros_like(t)
    want[:, :16], want[:, 16:32] = -x2, x1
    assert torch.equal(t @ r, want)


def test_rotary_leaves_the_unrotated_dimensions_alone_and_matches_the_formula():
    n = 7
    t = torch.randn(2, n, cd.HEAD_DIM)
    cos, sin = cd.rotary_tables(torch, n, n)
    out = cd.apply_rotary(torch, t, cos, sin, cd.rotate_matrix(torch))
    assert torch.equal(out[..., 32:], t[..., 32:])
    inv = 1.0 / (10000.0 ** (torch.arange(0, 32, 2).float() / 32))
    f = torch.cat([torch.outer(torch.arange(n).float(), inv)] * 2, dim=-1)
    half = t[..., :32]
    rot = torch.cat([-half[..., 16:], half[..., :16]], dim=-1)
    assert torch.allclose(out[..., :32], half * f.cos() + rot * f.sin(), atol=1e-6)


def test_padded_keys_get_no_weight():
    m = cd.key_mask(torch, 5, 8, 8)
    assert (m[..., :5] == 0).all() and (m[..., 5:] == cd.NEG).all()
    s = torch.randn(1, 1, 8, 8) + m
    w = s.softmax(-1)
    assert float(w[..., 5:].sum()) == 0.0


def test_value_mask_keeps_memory_rows_and_follows_the_padding_mask():
    pm = torch.tensor([True, True, False, True])
    v = cd.value_mask(torch, pm, 96, 64)
    assert v[:64].eq(1).all() and v[64:68].tolist() == [1, 1, 0, 1] and v[68:].eq(0).all()
    assert cd.value_mask(torch, None, 96, 64)[64:].eq(1).all()


def test_modulation_applies_one_plus_scale_and_the_sigmoid_of_one_minus_gate():
    d = 4
    param = torch.arange(6 * d).float() / 10
    gc = torch.ones(6 * d)
    s_s, h_s, g_s, s_f, h_f, g_f = cd.modulation(torch, param, gc)
    z = param + gc
    assert torch.allclose(s_s, 1 + z[:d]) and torch.allclose(h_s, z[d:2 * d])
    assert torch.allclose(g_s, torch.sigmoid(1 - z[2 * d:3 * d]))
    assert torch.allclose(s_f, 1 + z[3 * d:4 * d]) and torch.allclose(h_f, z[4 * d:5 * d]) and torch.allclose(g_f, torch.sigmoid(1 - z[5 * d:]))


def test_rms_matches_torch_rms_norm():
    x, g = torch.randn(3, 64), torch.randn(64)
    assert torch.allclose(cd.rms(torch, x, g, 1e-5), torch.nn.functional.rms_norm(x, (64,), weight=g, eps=1e-5), atol=1e-6)


def test_padded_length_rounds_up_to_tiles():
    assert [cd.padded_length(n) for n in (1, 32, 33, 238, 257)] == [32, 32, 64, 256, 288]


# --- recording of a batched (guided) pass ---------------------------------------------------------------------------------------------

class _Stub(cd.ChipTransformer):
    """The module's call path without a chip: `_one` returns a fixed output and leaves one layer record per call, as the chip version does."""

    def __init__(self, record):
        self.torch, self.ttnn, self.record, self.record_enabled = torch, None, record, True
        self.timings = {"upload_weights_s": 0.0, "forward_s": 0.0, "passes": 0}
        self._pending = {}
        self.calls = 0

    def _one(self, x, ctx, gcond, local, pmask):
        self.calls += 1
        if self.record_enabled:
            self._pending.setdefault("layer_00", []).append(torch.full((1, 3, 4), float(self.calls)))
        return torch.zeros(x.shape[0], 2)


def test_a_batched_first_pass_records_one_array_with_every_batch_element_and_later_passes_record_nothing():
    got = []
    m = _Stub(lambda name, t: got.append((name, t.clone())))
    x = torch.zeros(2, 5, 2)
    ctx = torch.zeros(2, 4, 2)
    out = m(x, context=ctx, global_cond=torch.zeros(2, 2), local_add_cond=None, padding_mask=None)
    assert out.shape == (2, 5, 2) and len(got) == 1 and got[0][0] == "layer_00"
    assert got[0][1].shape == (2, 3, 4) and got[0][1][0, 0, 0] == 1 and got[0][1][1, 0, 0] == 2
    m(x, context=ctx, global_cond=torch.zeros(2, 2))
    assert len(got) == 1, "only the first pass is recorded"


def test_the_module_refuses_inputs_the_text_to_music_path_does_not_use():
    m = _Stub(None)
    for kw in ({"prepend_embeds": torch.zeros(1)}, {"modular_local_cond": {"a": 1}}, {"exit_layer_ix": 3}, {"return_info": True}):
        with pytest.raises(NotImplementedError):
            m(torch.zeros(1, 2, 2), context=torch.zeros(1, 2, 2), global_cond=torch.zeros(1, 2), **kw)
    with pytest.raises(NotImplementedError):
        m(torch.zeros(1, 2, 2), context=None, global_cond=torch.zeros(1, 2))
