# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 220, finding 3: `ChipTransformer._load` kept the bfloat16 qkv, cq and ckv projections for differential attention although `_diff_attention`
reads only the float32 copies. Each layer now loads a projection only in the precision that uses it. A fake ttnn records every upload."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

MODULE = Path(__file__).resolve().parents[1] / "orchard" / "skills" / "audio-chip-templates" / "chip_dit.py"
spec = importlib.util.spec_from_file_location("chip_dit_load_under_test", MODULE)
cd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cd)

DIM = 64


class FakeTtnn:
    """Only what `ChipTransformer.__init__` and `_load` touch. `uploads` lists (shape, dtype) of every tensor uploaded."""
    bfloat16, float32, TILE_LAYOUT, ROW_MAJOR_LAYOUT = "bf16", "f32", "tile", "rm"
    MathFidelity = SimpleNamespace(HiFi4="hifi4")

    def __init__(self):
        self.uploads = []

    def init_device_compute_kernel_config(self, *a, **k):
        return None

    def from_torch(self, t, dtype=None, layout=None, device=None):
        self.uploads.append((tuple(t.shape), dtype))
        return SimpleNamespace(shape=tuple(t.shape), dtype=dtype)


def lin(o, i):
    return SimpleNamespace(weight=torch.zeros(o, i), bias=torch.zeros(o))


def _norm(n=DIM):
    return SimpleNamespace(gamma=torch.ones(n))


def _layer(differential):
    return SimpleNamespace(
        pre_norm=_norm(), cross_attend_norm=_norm(), ff_norm=_norm(),
        self_attn=SimpleNamespace(to_qkv=lin((5 if differential else 3) * DIM, DIM), to_out=lin(DIM, DIM), q_norm=_norm(64), k_norm=_norm(64), num_heads=1,
                                  differential=differential),
        cross_attn=SimpleNamespace(to_q=lin((2 if differential else 1) * DIM, DIM), to_kv=lin((3 if differential else 2) * DIM, DIM), to_out=lin(DIM, DIM),
                                   q_norm=_norm(64), k_norm=_norm(64)),
        to_local_embed=[lin(DIM, 8), None, lin(DIM, DIM)],
        ff=SimpleNamespace(ff=[SimpleNamespace(proj=lin(6 * DIM, DIM)), None, lin(DIM, 3 * DIM)]),
        to_scale_shift_gate=torch.zeros(6 * DIM))


def _module(differential, depth=2):
    return SimpleNamespace(layers=[_layer(differential) for _ in range(depth)], project_in=lin(DIM, 8), project_out=lin(8, DIM),
                           memory_tokens=torch.zeros(cd.MEMORY_TOKENS, DIM), num_memory_tokens=cd.MEMORY_TOKENS, global_cond_embedder=object())


def _build(differential):
    ttnn = FakeTtnn()
    t = cd.ChipTransformer(ttnn, torch, SimpleNamespace(arch=lambda: "arch"), _module(differential))
    return t, ttnn


def test_differential_layers_load_the_projections_only_in_float32():
    t, ttnn = _build(True)
    for d in t.layers:
        assert not {"qkv", "cq", "ckv"} & set(d), "the bfloat16 copies are unused by differential attention"
        assert {"qkv32", "cq32", "ckv32"} <= set(d)
    projections = [(s, dt) for s, dt in ttnn.uploads if s in {(DIM, 5 * DIM), (DIM, 2 * DIM), (DIM, 3 * DIM)}]
    assert projections and all(dt == "f32" for _, dt in projections)


def test_plain_layers_still_load_the_weight_dtype_projections():
    t, _ = _build(False)
    for d in t.layers:
        assert {"qkv", "cq", "ckv"} <= set(d) and not ({"qkv32", "cq32", "ckv32"} & set(d))
