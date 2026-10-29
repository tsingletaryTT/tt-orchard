# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The two per-head matmuls of absorbed multi-latent attention at decode (40 heads, one row each): time them in a trace with program configs and
weight dtypes. Reference: default config took about 100 us per call inside the model."""
import sys
import time

import torch
import ttnn

REPS = 20
H = 40


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return round(((a - b).norm() / b.norm()).item(), 4)


def timed(dev, fn, label, check=None):
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
        extra = f" relL2 {pcc(ttnn.to_torch(out), check)}" if check is not None else ""
        print(f"  {label:60s} {dt * 1e6:8.1f} us{extra}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  {label:60s} FAILED {str(e).splitlines()[0][:160]}", flush=True)


dev = ttnn.open_device(device_id=0, trace_region_size=64 * 1024 * 1024)
try:
    cks = {"lofi": dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False),
           "hifi2": dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False),
           "hifi4": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False),
           "hifi4+fp32": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
           "hifi2+fp32": dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True)}
    for name, K, N in (("wbig 96->288", 96, 288), ("wuv 256->64", 256, 64)):
        print(name, flush=True)
        a = torch.randn(1, H, 1, K)
        w = torch.randn(1, H, K, N)
        ref = a @ w
        at = ttnn.from_torch(a.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        wt = ttnn.from_torch(w.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        kt, nt = K // 32, N // 32
        pc = ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(8, 5), in0_block_w=kt, out_subblock_h=1, out_subblock_w=1,
                                                    per_core_M=1, per_core_N=nt)
        for cn, kw in cks.items():
            ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_approx_mode=False, packer_l1_acc=False, **kw)
            timed(dev, lambda: ttnn.matmul(at, wt, program_config=pc, compute_kernel_config=ck), f"reuse 8x5 {cn}", ref)
            timed(dev, lambda: ttnn.matmul(at, wt, compute_kernel_config=ck), f"default program {cn}", ref)
finally:
    ttnn.close_device(dev)
