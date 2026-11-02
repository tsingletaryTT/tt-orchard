# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 220: the chunked decoder's `host_layer` takes `sine=True` for the sine-gated feed-forward of the medium decoder's last layers. It must apply the
sine gate, as `host_sliding_layer` and `ChipDecoder._layer` do. Nothing here uses a chip."""
import torch  # noqa: F401  (a missing torch is a skip below)
import pytest

pytest.importorskip("torch")

import test_audio_chip_decoder as base  # noqa: E402  the module that loads chip_decoder and the library block helpers

cdec, DIM, N, padded, params_of = base.cdec, base.DIM, base.N, base.padded, base.params_of


def make_sine_block(TransformerBlock, seed=0):
    """The decoder block with the sine-gated feed-forward."""
    torch.manual_seed(seed)
    blk = TransformerBlock(DIM, dim_heads=64, causal=False, zero_init_branch_outputs=False, norm_type="dyt", add_rope=True,
                           attn_kwargs={"qk_norm": "dyt", "qk_norm_eps": 1e-3, "differential": True, "feat_scale": False},
                           ff_kwargs={"mult": 3, "no_bias": False, "sinusoidal": True}, norm_kwargs={"eps": 1e-3})
    blk.eval()
    with torch.no_grad():
        for p in blk.parameters():
            p.add_(0.2 * torch.randn_like(p))
    return blk


def test_sine_gated_layer_matches_the_library_block():
    """With sine=True the host layer matches the library's sine-gated block, and the SiLU result differs from it (so the test sees the gate)."""
    blk = make_sine_block(base.library_block_class())
    x = torch.randn(3, N, DIM)
    cos, sin = cdec.rotary_tables(torch)
    rot = cdec.rotate_matrix(torch)
    with torch.no_grad():
        want = blk(x)
        got = cdec.host_layer(torch, padded(x), params_of(blk), N, cos, sin, rot, sine=True)[:, :N]
        silu = cdec.host_layer(torch, padded(x), params_of(blk), N, cos, sin, rot, sine=False)[:, :N]
    assert torch.allclose(got, want, atol=2e-4, rtol=1e-4), float((got - want).abs().max())
    assert not torch.allclose(silu, want, atol=1e-3)


def _random_params(d, g):
    p = {}
    for k in ("pre_norm", "ff_norm", "self_attn.q_norm", "self_attn.k_norm"):
        w = 64 if "attn" in k else d
        p[f"{k}.alpha"], p[f"{k}.gamma"], p[f"{k}.beta"] = torch.ones(1) * 0.5, torch.ones(w), torch.zeros(w)
    p["self_attn.to_qkv.weight"] = torch.randn(5 * d, d, generator=g) * 0.1
    p["self_attn.to_out.weight"] = torch.randn(d, d, generator=g) * 0.1
    p["ff.ff.0.proj.weight"] = torch.randn(6 * d, d, generator=g) * 0.3
    p["ff.ff.0.proj.bias"] = torch.zeros(6 * d)
    p["ff.ff.2.weight"] = torch.randn(d, 3 * d, generator=g) * 0.1
    p["ff.ff.2.bias"] = torch.zeros(d)
    return p


def test_the_sine_flag_changes_the_layer_output_without_the_library():
    """Needs no library: with random parameters the two gates give different outputs, and the sine one equals the formula by hand."""
    d = 64
    g = torch.Generator().manual_seed(0)
    p = _random_params(d, g)
    x = torch.randn(1, cdec.CHUNK_PAD, d, generator=g)
    cos, sin = cdec.rotary_tables(torch)
    rot = cdec.rotate_matrix(torch)
    tr = {}
    b = cdec.host_layer(torch, x, p, 34, cos, sin, rot, trace=tr, sine=True)
    a = cdec.host_layer(torch, x, p, 34, cos, sin, rot, sine=False)
    assert not torch.allclose(a[:, :34], b[:, :34], atol=1e-3)
    a2, gate = tr["ff1"].chunk(2, dim=-1)
    want = tr["x_attn"] + (a2 * torch.sin(cdec.SINE * gate)) @ p["ff.ff.2.weight"].t() + p["ff.ff.2.bias"]
    assert torch.allclose(b, want, atol=1e-5)
