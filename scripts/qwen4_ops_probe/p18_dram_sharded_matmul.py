# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 18: decode matmuls with the weight sharded across the DRAM banks (issue 723).

Probe 17 found that a matmul on a [32, K] activation against a [K, N] weight takes 12 to 37 microseconds and that a batched matmul over 128 expert slots reads
weights at about 32 GB/s, far below what one chip's DRAM delivers. The published Flash-Next ports run every decode linear through tt-metal's DRAM-sharded matmul:
the weight is stored width-sharded over the DRAM banks (a grid of banks x 1, each shard [K, tiles per bank x 32]), the activation is width-sharded in L1 over a
core grid that splits K into whole tiles, and the program config is `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w, per_core_M=1, per_core_N)`.
Each reader streams its bank's slice of the weight, so the read runs near DRAM bandwidth. The public code that shows the recipe is
`models/demos/blackhole/qwen38_flash_next/ttnn/decode_matmul.py` in tt-metal (Apache-2.0); this probe writes the same configuration from the ttnn documentation names.

For each (K, N) of a Flash-Next decode projection and each weight format it runs `ttnn.linear` on 32 rows with the default layout (interleaved DRAM weight) and with the
DRAM-sharded layout, 50 calls back to back, and reports microseconds per call, the weight bytes read per second, and the PCC of the sharded output against the default one.

Run on a chip only inside a gozer lease. `--selftest` checks the layout arithmetic on the CPU and needs no ttnn.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TILE = 32
SHAPES = {"gdn_in_proj": (2560, 4160), "expert_gate_up": (2560, 1280), "expert_down": (640, 2560), "attn_out": (1536, 2560), "lm_head_chunk": (2560, 8192)}
BYTES_PER_ELEMENT = {"bf16": 2.0, "bfp8": 1.0625, "bfp4": 0.5625}


def activation_cores(k: int, cap: int = 64) -> int:
    """The largest core count for the L1 width-sharded activation: it splits K into whole tiles, is at most 8 (one row of cores) or a multiple of 8 up to `cap` (a rectangle
    over 8 columns)."""
    tiles = k // TILE
    for c in range(cap, 0, -1):
        if tiles % c == 0 and (c <= 8 or c % 8 == 0):
            return c
    raise ValueError(f"K={k} has no core count that splits it into whole tiles")


def bank_tiles(n: int, banks: int) -> int:
    """Weight tiles per DRAM bank along N: N over the banks, padded to whole tiles."""
    return math.ceil(n / (TILE * banks))


def in0_block_width(k: int, cores: int) -> int:
    """The largest block width (at most 8 tiles) that divides one core's share of K."""
    per_core = k // (TILE * cores)
    return next(w for w in range(8, 0, -1) if per_core % w == 0)


def selftest() -> dict:
    out, ok = {}, True
    for name, (k, n) in SHAPES.items():
        c = activation_cores(k)
        row = {"cores": c, "in0_block_w": in0_block_width(k, c), "per_core_n": math.ceil(n / (TILE * c)), "bank_tiles_8": bank_tiles(n, 8), "bank_tiles_7": bank_tiles(n, 7)}
        ok &= k % (TILE * c) == 0 and (c <= 8 or c % 8 == 0) and k // (TILE * c) % row["in0_block_w"] == 0 and row["bank_tiles_8"] * 8 * TILE >= n
        out[name] = row
    return {"probe": "p18_dram_sharded_matmul", "mode": "selftest", "ok": bool(ok), "layouts": out}


def run_device(ttnn, device, repeat: int, reps: int, formats, shapes) -> dict:
    banks = device.dram_grid_size().x
    dts = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
    ck = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True)
    g = torch.Generator().manual_seed(3)
    rows = []
    for name in shapes:
        k, n = SHAPES[name]
        x = torch.randn(1, 1, TILE, k, generator=g).bfloat16()
        w = (torch.randn(1, 1, k, n, generator=g) * k ** -0.5).bfloat16()
        cores = activation_cores(k)
        grid = ttnn.CoreGrid(x=cores, y=1) if cores <= 8 else ttnn.CoreGrid(x=8, y=cores // 8)
        act_cfg = ttnn.create_sharded_memory_config((TILE, k // cores), grid, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        w_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                                  ttnn.ShardSpec(bank_grid, (k, bank_tiles(n, banks) * TILE), ttnn.ShardOrientation.ROW_MAJOR))
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=in0_block_width(k, cores), per_core_M=1, per_core_N=math.ceil(n / (TILE * cores)), fused_activation=None)
        x_dram = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_l1 = ttnn.to_memory_config(x_dram, act_cfg)
        for fmt in formats:
            entry = {"shape": name, "k": k, "n": n, "format": fmt, "dram_banks": banks, "activation_cores": cores}
            try:
                w_def = ttnn.from_torch(w, dtype=dts[fmt], layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                w_shd = ttnn.from_torch(w, dtype=dts[fmt], layout=ttnn.TILE_LAYOUT, device=device, memory_config=w_cfg)
                f_def = lambda: ttnn.linear(x_dram, w_def, compute_kernel_config=ck, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                f_shd = lambda: ttnn.linear(x_l1, w_shd, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, program_config=pc, compute_kernel_config=ck)
                outs = {}
                for label, fn in (("default", f_def), ("dram_sharded", f_shd)):
                    fn()
                    ttnn.synchronize_device(device)
                    best = None
                    for _ in range(repeat):
                        t0 = time.perf_counter()
                        for _ in range(reps):
                            r = fn()
                        ttnn.synchronize_device(device)
                        dt = (time.perf_counter() - t0) / reps
                        best = dt if best is None else min(best, dt)
                    outs[label] = ttnn.to_torch(ttnn.to_memory_config(r, ttnn.DRAM_MEMORY_CONFIG)).float().reshape(TILE, n)
                    entry[f"{label}_us"] = round(best * 1e6, 1)
                    entry[f"{label}_gb_per_s"] = round(k * n * BYTES_PER_ELEMENT[fmt] / best / 1e9, 1)
                entry["pcc_sharded_vs_default"] = common.error_stats(outs["dram_sharded"], outs["default"])["pcc"]
            except Exception as exc:  # noqa: BLE001
                entry["error"] = common.short_exc(exc)
            rows.append(entry)
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shapes", nargs="+", default=list(SHAPES), choices=sorted(SHAPES))
    ap.add_argument("--formats", nargs="+", default=["bfp4", "bfp8", "bf16"], choices=sorted(BYTES_PER_ELEMENT))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--reps", type=int, default=50)
    args = ap.parse_args(argv)
    if args.selftest:
        r = selftest()
        common.emit(r)
        return 0 if r["ok"] else 1
    ttnn, device = common.open_device()
    try:
        res = run_device(ttnn, device, args.repeat, args.reps, args.formats, args.shapes)
    finally:
        ttnn.close_device(device)
    common.emit({"probe": "p18_dram_sharded_matmul", **res})
    return 0


if __name__ == "__main__":
    sys.exit(main())
