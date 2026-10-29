# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probes for a traced decode step: a cache update that takes its position from a device tensor, a position-indexed rope table, and trace capture of a
small step with host-updated input tensors."""
import time

import torch
import ttnn

T, D = 256, 288


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return round(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), 5), round(((a - b).norm() / b.norm()).item(), 4)


def try_(name, fn):
    t0 = time.time()
    try:
        print(f"[{name}] ok {fn()} ({time.time() - t0:.1f}s)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] FAILED {type(e).__name__}: {str(e).splitlines()[0][:400]}", flush=True)


dev = ttnn.open_device(device_id=0, trace_region_size=64 * 1024 * 1024)
try:
    torch.manual_seed(0)

    def paged_plain():
        cache = ttnn.from_torch(torch.zeros(1, 1, T, D).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        new = torch.randn(1, 1, 1, D)
        nt = ttnn.from_torch(new.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        idx = ttnn.from_torch(torch.tensor([9], dtype=torch.int32), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.paged_update_cache(cache, nt, update_idxs_tensor=idx)
        o = ttnn.to_torch(cache)
        return f"row9 {pcc(o[0, 0, 9], new[0, 0, 0])} row8 absmax {o[0, 0, 8].abs().max().item()}"

    try_("paged_update_cache interleaved input, idx tensor, no page table", paged_plain)

    def paged_sharded():
        cache = ttnn.from_torch(torch.zeros(1, 1, T, D).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        new = torch.randn(1, 1, 1, D)
        nt = ttnn.from_torch(new.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        # height-sharded input [1, 1, 32, 288] on one core as the DeepSeek code does
        pad = torch.zeros(1, 1, 32, D)
        pad[:, :, :1] = new
        mc = ttnn.create_sharded_memory_config(shape=(32, D), core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.HEIGHT,
                                               orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        nt = ttnn.from_torch(pad.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=mc)
        idx = ttnn.from_torch(torch.tensor([9], dtype=torch.int32), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.paged_update_cache(cache, nt, update_idxs_tensor=idx)
        o = ttnn.to_torch(cache)
        return f"row9 {pcc(o[0, 0, 9], new[0, 0, 0])} row8 absmax {o[0, 0, 8].abs().max().item()}"

    try_("paged_update_cache sharded input", paged_sharded)

    def onehot():
        cache = torch.randn(1, 1, T, D)
        ct = ttnn.from_torch(cache.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        new = torch.randn(1, 1, 1, D)
        nt = ttnn.from_torch(new.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        oh = torch.zeros(1, 1, T, 1)
        oh[0, 0, 9, 0] = 1
        oht = ttnn.from_torch(oh.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        keep = ttnn.from_torch((1 - oh).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        res = ttnn.add(ttnn.multiply(ct, keep), ttnn.multiply(oht, nt))
        ttnn.assign(res, ct)
        o = ttnn.to_torch(ct)
        return f"row9 {pcc(o[0, 0, 9], new[0, 0, 0])} row8 {pcc(o[0, 0, 8], cache[0, 0, 8])}"

    try_("one-hot update with broadcast multiply and assign", onehot)

    def emb_table():
        tab = torch.randn(T, 32)
        tt = ttnn.from_torch(tab.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ids = ttnn.from_torch(torch.tensor([[17]], dtype=torch.int32), device=dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        o = ttnn.embedding(ids, tt, layout=ttnn.TILE_LAYOUT)
        o = ttnn.reshape(o, (1, 1, 1, 32))
        return f"{tuple(o.shape)} {pcc(ttnn.to_torch(o), tab[17])}"

    try_("rope table by position via embedding", emb_table)

    def trace_small():
        w = ttnn.from_torch(torch.randn(2560, 2560).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        xin = ttnn.from_torch(torch.randn(1, 1, 1, 2560).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = ttnn.linear(xin, w)      # compile run
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        out = ttnn.linear(xin, w)
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        new = torch.randn(1, 1, 1, 2560).bfloat16()
        host = ttnn.from_torch(new, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, xin)
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=True)
        ref = new.float() @ ttnn.to_torch(w).float()
        r = pcc(ttnn.to_torch(out), ref)
        ttnn.release_trace(dev, tid)
        return r

    try_("trace capture and replay with copy_host_to_device_tensor", trace_small)
finally:
    ttnn.close_device(dev)
