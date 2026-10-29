# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""(Variant of exp_matmul.py with the tensor-parallel half shapes on 8 cores.) How much DRAM bandwidth does one decode matmul reach? Times a matmul of the model's shapes inside a trace, repeated, with the weights stored
interleaved and DRAM-sharded, in bfloat16 and bfloat8_b. Prints GB/s of weight read per call."""
import math
import sys
import time

import torch
import ttnn

REPS = 20
SHAPES = [(2560, 3328, "tp gate/up half (padded)"), (3328, 2560, "tp down half (padded)"), (1280, 2560, "tp o_proj half"), (2560, 3200, "tp gate/up half unpadded"), (2560, 36736, "tp lm_head half")]
BYTES = {ttnn.bfloat16: 2.0, ttnn.bfloat8_b: 1.0625}


def pick_cores(k, n):
    return (1, 8)
    kt, nt = k // 32, n // 32
    for c in range(64, 0, -1):
        if kt % c == 0 and nt % c == 0:
            for rows in range(1, 9):
                if c % rows == 0 and c // rows <= 8:
                    return rows, c // rows
    return 1, 1


def timed(dev, fn, label, nbytes):
    out = fn()                              # compile
    ttnn.synchronize_device(dev)
    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    for _ in range(REPS):
        out = fn()
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    ttnn.synchronize_device(dev)
    t0 = time.time()
    for _ in range(5):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(dev)
    dt = (time.time() - t0) / (5 * REPS)
    ttnn.release_trace(dev, tid)
    print(f"  {label:34s} {dt * 1e6:8.1f} us  {nbytes / dt / 1e9:7.1f} GB/s", flush=True)
    return out


dev = ttnn.open_device(device_id=0, trace_region_size=64 * 1024 * 1024)
try:
    dram_x = dev.dram_grid_size().x
    print("dram banks", dram_x, flush=True)
    for k, n, name in SHAPES:
        print(f"{name} [{k} x {n}]", flush=True)
        w = torch.randn(k, n).bfloat16()
        x = torch.randn(1, 1, 1, k).bfloat16()
        ref = x.float() @ w.float()
        for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
            nbytes = k * n * BYTES[dt]
            try:
                wi = ttnn.from_torch(w, device=dev, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                xi = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                timed(dev, lambda: ttnn.linear(xi, wi), f"interleaved {dt}", nbytes)
            except Exception as e:  # noqa: BLE001
                print("  interleaved FAILED", str(e).splitlines()[0][:200], flush=True)
            try:
                rows, cols = pick_cores(k, n)
                nc = rows * cols
                padded = math.ceil(n / (32 * dram_x)) * 32 * dram_x
                wg = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_x - 1, 0))})
                wmc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                                        ttnn.ShardSpec(wg, (k, padded // dram_x), ttnn.ShardOrientation.ROW_MAJOR))
                wp = torch.nn.functional.pad(w, (0, padded - n))
                ws = ttnn.from_torch(wp, device=dev, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=wmc)
                in_mc = ttnn.create_sharded_memory_config(shape=(32, k // nc), core_grid=ttnn.CoreGrid(y=rows, x=cols),
                                                          strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                          use_height_and_width_as_shard_shape=True)
                out_mc = ttnn.create_sharded_memory_config(shape=(32, n // nc), core_grid=ttnn.CoreGrid(y=rows, x=cols),
                                                           strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                           use_height_and_width_as_shard_shape=True)
                xs = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                xs = ttnn.to_memory_config(xs, in_mc)
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=max(1, k // 32 // nc), per_core_M=1, per_core_N=math.ceil(n / 32 / nc), fused_activation=None)
                ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                                            fp32_dest_acc_en=False, packer_l1_acc=True)
                out = timed(dev, lambda: ttnn.linear(xs, ws, program_config=pc, memory_config=out_mc, compute_kernel_config=ck),
                            f"dram-sharded {dt} on {rows}x{cols} cores", nbytes)
                o = ttnn.to_torch(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1)[:n].float()
                r = ref.reshape(-1)
                print("    check relL2", round(((o - r).norm() / r.norm()).item(), 4), flush=True)
            except Exception as e:  # noqa: BLE001
                print("  dram-sharded FAILED", str(e).splitlines()[0][:300], flush=True)
finally:
    ttnn.close_device(dev)
