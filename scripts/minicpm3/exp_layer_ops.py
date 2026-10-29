# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Device time of each op of one attention layer at decode, alone, repeated 20 times inside a trace (so launch cost from the host is out of it)."""
import time

import torch
import ttnn

REPS = 20
H, KV, ROPE, D = 40, 256, 32, 2560
T = 256


def timed(dev, name, fn):
    try:
        fn()
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
        print(f"  {name:48s} {dt * 1e6:7.1f} us", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  {name:48s} FAILED {str(e).splitlines()[0][:150]}", flush=True)


dev = ttnn.open_device(device_id=0, trace_region_size=128 * 1024 * 1024)
try:
    rn = lambda *s: torch.randn(*s).bfloat16()  # noqa: E731
    up = lambda t, **k: ttnn.from_torch(t, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **k)  # noqa: E731
    nwt = lambda n: ttnn.from_torch(rn(1, 1, n // 32, 32), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)  # noqa: E731
    ck4 = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
                                                 packer_l1_acc=True)
    ck2 = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False)
    x = up(rn(1, 1, 1, D))
    w_n = nwt(D)
    timed(dev, "rms_norm 2560 (hifi4 fp32 acc)", lambda: ttnn.rms_norm(x, epsilon=1e-5, weight=w_n, compute_kernel_config=ck4))
    timed(dev, "rms_norm 2560, weight preloaded", lambda: ttnn.rms_norm(x, epsilon=1e-5, weight=w_n, compute_kernel_config=ck4))
    timed(dev, "rms_norm 2560, default config", lambda: ttnn.rms_norm(x, epsilon=1e-5, weight=w_n))
    wqa, wqb = up(rn(D, 768)), up(rn(768, 3840))
    timed(dev, "linear qa 2560->768", lambda: ttnn.linear(x, wqa))
    qa = up(rn(1, 1, 1, 768))
    w768 = nwt(768)
    timed(dev, "rms_norm 768", lambda: ttnn.rms_norm(qa, epsilon=1e-5, weight=w768, compute_kernel_config=ck4))
    timed(dev, "linear qb 768->3840", lambda: ttnn.linear(qa, wqb))
    qb = up(rn(1, 1, 1, 3840))
    timed(dev, "reshape [1,1,1,3840]->[1,1,40,96]", lambda: ttnn.reshape(qb, (1, 1, H, 96)))
    q = up(rn(1, 1, H, 96))
    cosF = up(rn(1, 1, 1, 96))
    Rf = up(rn(96, 96))
    timed(dev, "multiply [1,1,40,96] * [1,1,1,96]", lambda: ttnn.multiply(q, cosF))
    timed(dev, "matmul [1,1,40,96] x [96,96]", lambda: ttnn.matmul(q, Rf))
    timed(dev, "add [1,1,40,96]", lambda: ttnn.add(q, q))
    timed(dev, "transpose [1,1,40,96] -> [1,40,1,96]", lambda: ttnn.transpose(q, 1, 2))
    qh = up(rn(1, H, 1, 96))
    wbig = up(rn(1, H, 96, 288))
    pc = ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(8, 5), in0_block_w=3, out_subblock_h=1, out_subblock_w=1,
                                                per_core_M=1, per_core_N=9)
    timed(dev, "bmm wbig (explicit config)", lambda: ttnn.matmul(qh, wbig, program_config=pc, compute_kernel_config=ck2))
    ql = up(rn(1, H, 1, 288))
    timed(dev, "transpose [1,40,1,288] -> [1,1,40,288]", lambda: ttnn.transpose(ql, 1, 2))
    wc, wk = up(rn(D, 256)), up(rn(D, 64))
    timed(dev, "linear ckv 2560->256", lambda: ttnn.linear(x, wc))
    c = up(rn(1, 1, 1, 256))
    w256 = nwt(256)
    timed(dev, "rms_norm 256", lambda: ttnn.rms_norm(c, epsilon=1e-5, weight=w256, compute_kernel_config=ck4))
    timed(dev, "linear kpe 2560->64", lambda: ttnn.linear(x, wk))
    k64 = up(rn(1, 1, 1, 64))
    cs = up(rn(1, 1, 1, 64))
    IdI = up(rn(64, 32))
    timed(dev, "multiply [1,1,1,64]", lambda: ttnn.multiply(k64, cs))
    timed(dev, "matmul [1,1,1,64] x [64,32]", lambda: ttnn.matmul(k64, IdI))
    kr = up(rn(1, 1, 1, 32))
    timed(dev, "concat [256]+[32]", lambda: ttnn.concat([c, kr], dim=-1))
    new = up(rn(1, 1, 1, 288))
    mc = ttnn.create_sharded_memory_config(shape=(32, 288), core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.HEIGHT,
                                           orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    timed(dev, "to_memory_config (new -> height sharded)", lambda: ttnn.to_memory_config(new, mc))
    cache = up(torch.zeros(1, 1, T, 288).bfloat16())
    cur = ttnn.from_torch(torch.tensor([9], dtype=torch.int32), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    news = ttnn.to_memory_config(new, mc)
    timed(dev, "paged_update_cache", lambda: ttnn.experimental.paged_update_cache(cache, news, update_idxs_tensor=cur))
    qf = up(rn(1, 1, H, 288))
    spc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=32, k_chunk_size=128, exp_approx_mode=False)
    cur2 = ttnn.from_torch(torch.tensor([200], dtype=torch.int32), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    timed(dev, f"flash MLA T={T} pos 200", lambda: ttnn.transformer.flash_multi_latent_attention_decode(
        qf, cache, head_dim_v=256, cur_pos_tensor=cur2, scale=96 ** -0.5, program_config=spc))
    o = up(rn(1, 1, H, 256))
    timed(dev, "transpose [1,1,40,256] -> [1,40,1,256]", lambda: ttnn.transpose(o, 1, 2))
    oh = up(rn(1, H, 1, 256))
    wuv = up(rn(1, H, 256, 64))
    pc2 = ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(8, 5), in0_block_w=8, out_subblock_h=1, out_subblock_w=1,
                                                 per_core_M=1, per_core_N=2)
    timed(dev, "bmm wuv (explicit config)", lambda: ttnn.matmul(oh, wuv, program_config=pc2, compute_kernel_config=ck2))
    o64 = up(rn(1, H, 1, 64))
    timed(dev, "reshape [1,40,1,64] -> [1,1,1,2560]", lambda: ttnn.reshape(o64, (1, 1, 1, 2560)))
    timed(dev, "add [1,1,1,2560]", lambda: ttnn.add(x, x))
    a6, b6 = up(rn(1, 1, 1, 6400)), up(rn(1, 1, 1, 6400))
    timed(dev, "multiply [1,1,1,6400]", lambda: ttnn.multiply(a6, b6))
finally:
    ttnn.close_device(dev)
