# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 5: ttnn.experimental.deepseek_prefill.update_padded_kv_cache on a 1x1 mesh, scalar call form.

The op writes an input slab into a KV cache at an offset derived from `kv_actual_global`. With a 1x1 mesh and
cluster_axis 0 the whole cache is one shard, so the expected result is simple: rows [kv_actual_global,
kv_actual_global + chunk) of cache slot `slot_idx * num_layers + layer_idx` equal the input and every other row is
unchanged. The probe writes two chunks one after the other into a cache of --cache-rows rows and compares the whole cache
with that expectation.

Run on a chip only inside a gozer lease. `--selftest` checks the CPU expectation and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def expected_cache(cache, slabs, slot_idx, layer_idx, num_layers, starts):
    """CPU expectation: slab i is written at rows [starts[i], starts[i] + rows) of batch slot_idx*num_layers+layer_idx."""
    out = cache.clone()
    batch = slot_idx * num_layers + layer_idx
    for slab, start in zip(slabs, starts):
        out[batch, :, start:start + slab.shape[2], :] = slab[0]
    return out


def selftest() -> dict:
    cache = torch.zeros(4, 1, 128, 32)
    a, b = torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32)
    out = expected_cache(cache, [a, b], slot_idx=1, layer_idx=1, num_layers=2, starts=[0, 32])
    ok = (torch.equal(out[3, :, :32], a[0]) and torch.equal(out[3, :, 32:64], b[0])
          and not out[:3].any() and not out[3, :, 64:].any())
    return {"probe": "p5_update_padded_kv_cache", "mode": "selftest", "ok": bool(ok)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the expectation; no ttnn")
    ap.add_argument("--cache-rows", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--layer-idx", type=int, default=1)
    ap.add_argument("--slot-idx", type=int, default=0)
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest()
        common.emit(res)
        return 0 if res["ok"] else 1
    batch = (args.slot_idx + 1) * args.num_layers
    cache0 = torch.zeros(batch, 1, args.cache_rows, args.dim)
    slabs = [torch.randn(1, 1, args.chunk, args.dim).bfloat16().float() for _ in range(2)]
    starts = [0, args.chunk]
    ref = expected_cache(cache0, slabs, args.slot_idx, args.layer_idx, args.num_layers, starts)
    import ttnn
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    entry = {}
    try:
        try:
            tt_cache = ttnn.from_torch(cache0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                       mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
            for slab, start in zip(slabs, starts):
                tt_in = ttnn.from_torch(slab, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))
                ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                    tt_cache, tt_in, args.slot_idx, args.layer_idx, args.num_layers, start, 0)
            ttnn.synchronize_device(mesh)
            got = ttnn.to_torch(tt_cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            entry = common.error_stats(got, ref)
            entry["written_rows_exact"] = bool(torch.equal(got, ref))
        except Exception as exc:
            entry["failure"] = common.short_exc(exc)
    finally:
        ttnn.close_mesh_device(mesh)
    common.emit({"probe": "p5_update_padded_kv_cache", "shape": vars(args), "dtype": "bf16 TILE",
                 "reference": "CPU expectation (rows written at kv_actual_global, others unchanged)", "result": entry})
    return 0


if __name__ == "__main__":
    sys.exit(main())
