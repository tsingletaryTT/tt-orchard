# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Sharded rms_norm at decode (a width-sharded residual stream across 40 cores) and fused or thin projections: device time and accuracy."""
import time

import torch
import ttnn

REPS = 20
D = 2560


def timed(dev, name, fn, ref=None):
    try:
        out = fn()
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        for _ in range(REPS):
            fn()
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
            o = ttnn.to_torch(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)).reshape(-1)[: ref.numel()].float()
            e = f" relL2 {((o - ref).norm() / ref.norm()).item():.4f}"
        print(f"  {name:56s} {dt * 1e6:7.1f} us{e}", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"  {name:56s} FAILED {str(ex).splitlines()[0][:200]}", flush=True)


dev = ttnn.open_device(device_id=0, trace_region_size=64 * 1024 * 1024)
try:
    torch.manual_seed(0)
    x = torch.randn(D) * 0.5
    x[[48, 1292, 2166]] = torch.tensor([1000.0, 1050.0, 250.0])          # a massive-activation row, as the first token has
    w = torch.rand(D) + 0.5
    xb = x.bfloat16().float()
    ref = w * xb * torch.rsqrt(xb.pow(2).mean() + 1e-5)
    wt = ttnn.from_torch(w.reshape(1, 1, -1, 32).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ck4 = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
                                                 packer_l1_acc=True)
    xi = ttnn.from_torch(x.reshape(1, 1, 1, D).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    timed(dev, "interleaved rms_norm, fp32 acc", lambda: ttnn.rms_norm(xi, epsilon=1e-5, weight=wt, compute_kernel_config=ck4), ref)
    for rows, cols in ((5, 8), (1, 8), (2, 8), (4, 8), (8, 8), (1, 1), (1, 4)):
        nc = rows * cols
        if (D // 32) % nc:
            continue
        mc = ttnn.create_sharded_memory_config(shape=(32, D // nc), core_grid=ttnn.CoreGrid(y=rows, x=cols), strategy=ttnn.ShardStrategy.WIDTH,
                                               orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        xs = ttnn.to_memory_config(xi, mc)
        bw = D // nc // 32
        sw = next(s for s in (4, 3, 2, 1) if bw % s == 0)
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=[cols, rows], subblock_w=sw, block_h=1, block_w=bw,
                                                         inplace=False)
        timed(dev, f"sharded rms_norm {rows}x{cols} cores, fp32 acc", lambda: ttnn.rms_norm(xs, epsilon=1e-5, weight=wt, program_config=pc,
                                                                                         memory_config=mc, compute_kernel_config=ck4), ref)
    # projections
    h = ttnn.from_torch(torch.randn(1, 1, 1, D).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    for n in (768, 256, 64, 1088, 1024, 1152):
        wq = ttnn.from_torch(torch.randn(D, n).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        timed(dev, f"linear 2560->{n} default", lambda: ttnn.linear(h, wq))
        for g in ((8, 4), (8, 8)):
            try:
                nt = n // 32
                cores = g[0] * g[1]
                pcn = -(-nt // cores)
                pc1 = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=g, in0_block_w=4, out_subblock_h=1,
                                                                        out_subblock_w=1, per_core_M=1, per_core_N=pcn, fuse_batch=True,
                                                                        fused_activation=None, mcast_in0=True)
                timed(dev, f"linear 2560->{n} 1D mcast grid {g}", lambda: ttnn.linear(h, wq, program_config=pc1))
            except Exception as ex:  # noqa: BLE001
                print("   1d config failed", str(ex).splitlines()[0][:100])
finally:
    ttnn.close_device(dev)
