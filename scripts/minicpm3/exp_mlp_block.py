# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""MLP block variants at decode (one row): norm, gate with silu, up, product, down, residual add. Times each variant per layer inside a trace and checks the result
against torch. Variants: current (interleaved weights), DRAM-sharded weights with a sharded chain, and each with bfloat8_b weights."""
import math
import sys
import time

import torch
import ttnn

REPS = 20
D, F = 2560, 6400


def timed(dev, fn, label, ref=None):
    try:
        out = fn()
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
        e = ""
        if ref is not None:
            o = ttnn.to_torch(out).reshape(-1).float()
            e = f" relL2 {((o - ref).norm() / ref.norm()).item():.4f}"
        print(f"  {label:50s} {dt * 1e6:8.1f} us per layer{e}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"  {label:50s} FAILED {str(ex).splitlines()[0][:200]}", flush=True)


dev = ttnn.open_device(device_id=0, trace_region_size=64 * 1024 * 1024)
try:
    dram_x = dev.dram_grid_size().x
    torch.manual_seed(0)
    x = torch.randn(1, 1, 1, D)
    nw = torch.rand(D) + 0.5
    g, u, d = (torch.randn(D, F) * 0.02), (torch.randn(D, F) * 0.02), (torch.randn(F, D) * 0.02)
    xn = nw * x.reshape(-1) * torch.rsqrt(x.reshape(-1).pow(2).mean() + 1e-5)
    ref = x.reshape(-1) + (torch.nn.functional.silu(xn @ g) * (xn @ u)) @ d
    nwt = ttnn.from_torch(nw.reshape(1, 1, -1, 32).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    xt = ttnn.from_torch(x.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    rms_ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                    fp32_dest_acc_en=True, packer_l1_acc=True)
    for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
        wg = ttnn.from_torch(g.bfloat16(), device=dev, dtype=dt, layout=ttnn.TILE_LAYOUT)
        wu = ttnn.from_torch(u.bfloat16(), device=dev, dtype=dt, layout=ttnn.TILE_LAYOUT)
        wd = ttnn.from_torch(d.bfloat16(), device=dev, dtype=dt, layout=ttnn.TILE_LAYOUT)

        def v0():
            h = ttnn.rms_norm(xt, epsilon=1e-5, weight=nwt, compute_kernel_config=rms_ck)
            m = ttnn.multiply(ttnn.linear(h, wg, activation="silu"), ttnn.linear(h, wu))
            return ttnn.add(xt, ttnn.linear(m, wd))

        timed(dev, v0, f"current, interleaved {dt}", ref)

        # ---- DRAM-sharded chain --------------------------------------------------------------------------
        def shard_w(w, k, n):
            padded = math.ceil(n / (32 * dram_x)) * 32 * dram_x
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_x - 1, 0))})
            mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                                   ttnn.ShardSpec(grid, (k, padded // dram_x), ttnn.ShardOrientation.ROW_MAJOR))
            return ttnn.from_torch(torch.nn.functional.pad(w, (0, padded - n)).bfloat16(), device=dev, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=mc)

        def wmc(k, rows, cols):
            return ttnn.create_sharded_memory_config(shape=(32, k // (rows * cols)), core_grid=ttnn.CoreGrid(y=rows, x=cols),
                                                     strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                     use_height_and_width_as_shard_shape=True)
        try:
            sg, su, sd = shard_w(g, D, F), shard_w(u, D, F), shard_w(d, F, D)
            rows, cols = 5, 8
            nc = rows * cols
            in_mc, mid_mc, out_mc = wmc(D, rows, cols), wmc(F, rows, cols), wmc(D, rows, cols)
            pc_gu = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=D // 32 // nc, per_core_M=1, per_core_N=F // 32 // nc,
                                                                               fused_activation=None)
            pc_gu_silu = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=D // 32 // nc, per_core_M=1, per_core_N=F // 32 // nc, fused_activation=ttnn.UnaryOpType.SILU)
            pc_d = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=F // 32 // nc, per_core_M=1, per_core_N=D // 32 // nc,
                                                                              fused_activation=None)
            ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                                        fp32_dest_acc_en=False, packer_l1_acc=True)

            def v2():
                h = ttnn.rms_norm(xt, epsilon=1e-5, weight=nwt, compute_kernel_config=rms_ck)
                hs = ttnn.to_memory_config(h, in_mc)
                gg = ttnn.linear(hs, sg, program_config=pc_gu_silu, memory_config=mid_mc, compute_kernel_config=ck)
                uu = ttnn.linear(hs, su, program_config=pc_gu, memory_config=mid_mc, compute_kernel_config=ck)
                m = ttnn.multiply(gg, uu)
                o = ttnn.linear(m, sd, program_config=pc_d, memory_config=out_mc, compute_kernel_config=ck)
                return ttnn.add(xt, ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG))

            timed(dev, v2, f"dram-sharded chain {dt}", ref)
        except Exception as ex:  # noqa: BLE001
            print("  dram-sharded setup FAILED", str(ex).splitlines()[0][:300], flush=True)
finally:
    ttnn.close_device(dev)
