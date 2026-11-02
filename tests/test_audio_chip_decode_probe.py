# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 220, finding 1: `chip_decode_probe.py` compared the library's layer hooks, shaped [1, 17*T, D] for the medium (sliding-window) decoder, with
`ChipDecoder.probe`, shaped [T, 17, D], and the first-divergence summary raised KeyError('rel_l2') before decode_probe.json was written. Both are now
flattened to [tokens, D]. No chip, no library."""
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

PROBE = Path(__file__).resolve().parents[1] / "orchard" / "skills" / "audio-chip-templates" / "chip_decode_probe.py"


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("chip_decode_probe_under_test", PROBE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _layers(shape_of, n_layers, t, d, scale, seed):
    g = torch.Generator().manual_seed(seed)
    base = [torch.randn(t * 17, d, generator=g) for _ in range(n_layers)]
    return {i: (b + scale * torch.randn(t * 17, d, generator=g)).reshape(shape_of(t, d)) for i, b in enumerate(base)}


def test_tokens_by_features_makes_the_two_layouts_equal():
    x = torch.randn(5 * 17, 8)
    assert torch.equal(probe_fn(x.reshape(1, 5 * 17, 8)), probe_fn(x.reshape(5, 17, 8)))


def probe_fn(t):
    spec = importlib.util.spec_from_file_location("chip_decode_probe_fn", PROBE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.tokens_by_features(t)


def test_first_divergence_summary_accepts_hook_shapes_and_probe_shapes_for_the_medium_decoder(probe):
    t, d, n = 4, 16, 2000
    hooks = lambda t_, d_: (1, t_ * 17, d_)   # what the library's forward hooks see on the medium decoder
    probed = lambda t_, d_: (t_, 17, d_)      # what ChipDecoder.probe returns on the medium decoder
    ref = _layers(hooks, 12, t, d, 0.0, 1)
    bf = {i: v + 0.001 * torch.randn_like(v) for i, v in ref.items()}
    chip = {i: v.reshape(probed(t, d)) + 0.002 * torch.randn(probed(t, d)) for i, v in ref.items()}
    wave = torch.randn(1, 2, n)
    out = probe.first_divergence_summary(np, ref, bf, chip, wave, wave + 0.001, wave + 0.002)
    assert set(out) == {"three_times_the_bfloat16_floor", "rel_l2_0.05"}
    blocks = out["rel_l2_0.05"]["blocks"]
    assert blocks and all("rel_l2" in b for b in blocks)
