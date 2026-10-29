# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The host decomposition of the Stable Audio 3 autoencoder decoder layer (orchard/skills/audio-chip-templates/chip_decoder.py).

`host_layer` is the layer in the decomposition the chip code repeats (dynamic-tanh norms, differential attention, padded chunks with a key mask, the
rotary matrix). These tests check it against the library's own `TransformerBlock` configured as the decoder configures it, when the library is on disk,
and check the config guard. Nothing here uses a chip.
"""
import copy
import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODULE = Path(__file__).resolve().parents[1] / "orchard" / "skills" / "audio-chip-templates" / "chip_decoder.py"
LIB_SRC = Path(os.environ.get("ORCHARD_SA3_SRC", str(Path.home() / "orchard-src" / "stable-audio-3")))


def load_module():
    spec = importlib.util.spec_from_file_location("chip_decoder_under_test", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cdec = load_module()


def library_block_class():
    if not (LIB_SRC / "stable_audio_3" / "models" / "transformer.py").exists():
        pytest.skip("the stable-audio-3 source is not on disk (set ORCHARD_SA3_SRC)")
    sys.path.insert(0, str(LIB_SRC))
    try:
        from stable_audio_3.models.transformer import TransformerBlock
    finally:
        sys.path.remove(str(LIB_SRC))
    return TransformerBlock


DIM, N = 128, 34


def make_block(TransformerBlock, seed=0):
    torch.manual_seed(seed)
    blk = TransformerBlock(DIM, dim_heads=64, causal=False, zero_init_branch_outputs=False, norm_type="dyt", add_rope=True,
                           attn_kwargs={"qk_norm": "dyt", "qk_norm_eps": 1e-3, "differential": True, "feat_scale": False},
                           ff_kwargs={"mult": 3, "no_bias": False, "sinusoidal": False}, norm_kwargs={"eps": 1e-3})
    blk.eval()
    with torch.no_grad():                    # every gain, shift and alpha away from its initial value, so a wrong tensor changes the answer
        for p in blk.parameters():
            p.add_(0.2 * torch.randn_like(p))
    return blk


def params_of(blk):
    return {k: v.detach().float() for k, v in blk.state_dict().items()}


def padded(x):
    xp = torch.zeros(x.shape[0], cdec.CHUNK_PAD, DIM)
    xp[:, :N] = x
    return xp


def run_host(blk, x, n_real=N):
    cos, sin = cdec.rotary_tables(torch)
    return cdec.host_layer(torch, padded(x), params_of(blk), n_real, cos, sin, cdec.rotate_matrix(torch))[:, :N]


def test_layer_matches_the_library_block():
    blk = make_block(library_block_class())
    x = torch.randn(3, N, DIM)
    with torch.no_grad():
        want = blk(x)
        got = run_host(blk, x)
    assert torch.allclose(got, want, atol=2e-4, rtol=1e-4), float((got - want).abs().max())


def test_a_wrong_key_mask_disagrees_with_the_library():
    """Mutation check on the mask: with the padded keys visible the result must move, so the match above depends on the mask."""
    blk = make_block(library_block_class())
    x = torch.randn(2, N, DIM)
    with torch.no_grad():
        want = blk(x)
        leaky = run_host(blk, x, n_real=cdec.CHUNK_PAD)
    assert not torch.allclose(leaky, want, atol=1e-3)


def test_padding_rows_do_not_change_real_rows():
    blk = make_block(library_block_class())
    x = torch.randn(2, N, DIM)
    p = params_of(blk)
    cos, sin = cdec.rotary_tables(torch)
    rot = cdec.rotate_matrix(torch)
    a = padded(x)
    b = a.clone()
    b[:, N:] = 100.0 * torch.randn(2, cdec.CHUNK_PAD - N, DIM)
    with torch.no_grad():
        ra = cdec.host_layer(torch, a, p, N, cos, sin, rot)[:, :N]
        rb = cdec.host_layer(torch, b, p, N, cos, sin, rot)[:, :N]
    assert torch.allclose(ra, rb, atol=1e-5)


GOOD = {"model": {"pretransform": {"config": {"decoder": {"type": "taae_v2", "config": {
    "out_channels": 512, "channels": 128, "c_mults": [6], "strides": [16], "latent_dim": 256, "transformer_depths": [6], "sinusoidal_blocks": [0],
    "checkpointing": False, "differential": True, "mapping_style": "none", "dim_heads": 64, "variable_stride": True, "use_flash": True, "dyt": True,
    "chunk_size": 32, "conv_mapping": True, "chunk_midpoint_shift": True, "freeze_backbone": False, "mask_noise": 0.01}}}}}}


def test_the_shipped_config_is_accepted():
    assert cdec.config_problems(GOOD) == []


@pytest.mark.parametrize("key,value", [("differential", False), ("dyt", False), ("dim_heads", 128), ("sinusoidal_blocks", ["x"]), ("sliding_window", [2, 2]),
                                       ("chunk_size", 64), ("transformer_depths", [6, 6]), ("surprise", 1)])
def test_a_changed_setting_is_refused(key, value):
    cfg = copy.deepcopy(GOOD)
    cfg["model"]["pretransform"]["config"]["decoder"]["config"][key] = value
    assert cdec.config_problems(cfg), key


def test_a_missing_decoder_is_refused():
    assert cdec.config_problems({}) and cdec.config_problems({"model": {}})


# --- the sliding-window layout of the medium decoder ---------------------------------------------------------------------------------------

def make_sliding_block(TransformerBlock, seed=0, sine=False):
    torch.manual_seed(seed)
    blk = TransformerBlock(DIM, dim_heads=64, causal=False, zero_init_branch_outputs=False, norm_type="dyt", add_rope=True,
                           attn_kwargs={"qk_norm": "dyt", "qk_norm_eps": 1e-3, "differential": True, "feat_scale": False},
                           ff_kwargs={"mult": 3, "no_bias": False, "sinusoidal": sine}, norm_kwargs={"eps": 1e-3})
    blk.eval()
    with torch.no_grad():
        for p in blk.parameters():
            p.add_(0.2 * torch.randn_like(p))
    return blk


def run_sliding(blk, x, frames, sine=False):
    """x [17 * frames, D] -> the block decomposition's output as [17 * frames, D]."""
    xb = torch.zeros(frames, cdec.SLIDE_ROWS, DIM)
    xb[:, :17] = x.view(frames, 17, DIM)
    cos, sin = cdec.sliding_rotary(torch, frames)
    mask = cdec.sliding_masks(torch, frames)
    out = cdec.host_sliding_layer(torch, xb, params_of(blk), cos, sin, cdec.rotate_matrix(torch), mask, sine=sine)
    return out[:, :17].reshape(frames * 17, DIM)


@pytest.mark.parametrize("frames", [1, 2, 5])
@pytest.mark.parametrize("sine", [False, True])
def test_sliding_layer_matches_the_library_block(frames, sine):
    """The library, without flash attention, masks a band of 17 tokens each way over the whole run; the blocks of 17 with their two neighbours must agree."""
    blk = make_sliding_block(library_block_class(), sine=sine)
    x = torch.randn(17 * frames, DIM)
    with torch.no_grad():
        want = blk(x.unsqueeze(0), self_attention_flash_sliding_window=(17, 17))[0]
        got = run_sliding(blk, x, frames, sine=sine)
    assert torch.allclose(got, want, atol=3e-4, rtol=1e-4), float((got - want).abs().max())


def test_a_wider_window_disagrees_with_the_library():
    """Mutation check: the band mask is doing work. A window of 34 tokens each way (the blocks' own mask kept) must not match a library run with 17."""
    blk = make_sliding_block(library_block_class())
    x = torch.randn(17 * 4, DIM)
    with torch.no_grad():
        wide = blk(x.unsqueeze(0), self_attention_flash_sliding_window=(34, 34))[0]
        got = run_sliding(blk, x, 4)
    assert not torch.allclose(got, wide, atol=1e-3)


def test_the_band_mask_has_one_visible_key_in_every_row_and_no_view_beyond_the_ends():
    m = cdec.sliding_masks(torch, 3)
    assert ((m == 0).sum(-1) >= 1).all()
    assert (m[0, 0, :17, :32] < 0).all()              # no previous block before the first frame
    assert (m[-1, 0, :17, 64:] < 0).all()             # no next block after the last frame
    assert (m[1, 0, 0, :17] == 0).all() and (m[1, 0, 16, 32:49] == 0).all()
    assert (m[1, 0, 0, 17:32] < 0).all()              # padded key rows are never visible


MEDIUM = {"model": {"pretransform": {"config": {"decoder": {"type": "taae_v2", "config": {
    "out_channels": 512, "channels": 256, "c_mults": [6], "strides": [16], "latent_dim": 256, "transformer_depths": [12], "sinusoidal_blocks": [8],
    "use_snake": False, "use_dilated_conv": False, "checkpointing": False, "conformer": False, "layer_scale": False, "differential": True,
    "conv_bias": False, "mapping_style": "none", "dim_heads": 64, "enable_inner_layer_dropout": False, "sliding_window": [1, 1],
    "variable_stride": True, "mask_noise": 0.1}}}}}}


def test_the_medium_decoder_config_is_accepted():
    assert cdec.config_problems(MEDIUM) == []


@pytest.mark.parametrize("key,value", [("sliding_window", [1, 2]), ("conformer", True), ("layer_scale", True), ("chunk_midpoint_shift", True),
                                       ("differential", False), ("strides", [8])])
def test_a_changed_medium_setting_is_refused(key, value):
    cfg = copy.deepcopy(MEDIUM)
    cfg["model"]["pretransform"]["config"]["decoder"]["config"][key] = value
    assert cdec.config_problems(cfg), key
