# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Single-op probes for the MiniCPM3 port. Runs under a gozer lease. Each probe compares one ttnn op with torch on random data at the model's shapes."""
import sys
import time

import torch
import ttnn

H, KV, ROPE, T = 40, 256, 32, 128
D = KV + ROPE


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item(), ((a - b).norm() / b.norm()).item()


def try_(name, fn):
    t0 = time.time()
    try:
        r = fn()
        print(f"[{name}] ok {r} ({time.time() - t0:.1f}s)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] FAILED {type(e).__name__}: {str(e)[:600]}", flush=True)


dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)

    def flash_mla():
        q = torch.randn(1, 1, H, D) * 0.5
        k = torch.randn(1, 1, T, D) * 0.5
        pos = 100
        scale = 96 ** -0.5
        qt = ttnn.from_torch(q.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        kt = ttnn.from_torch(k.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        out = ttnn.transformer.flash_multi_latent_attention_decode(qt, kt, head_dim_v=KV, cur_pos=[pos], scale=scale)
        o = ttnn.to_torch(out)
        s = torch.einsum("bshd,bstd->bhst", q.float(), k.float()[:, :, : pos + 1]) if False else (q[0, 0] @ k[0, 0, : pos + 1].T) * scale
        ref = torch.softmax(s, -1) @ k[0, 0, : pos + 1, :KV]
        return f"out {tuple(o.shape)} pcc/relL2 {pcc(o.reshape(-1, KV)[:H], ref)}"

    try_("flash_mla (q[1,1,H,288], k[1,1,T,288], cur_pos list)", flash_mla)

    def update_cache():
        cache = torch.zeros(1, H, T, 96)
        ct = ttnn.from_torch(cache.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        new = torch.randn(1, H, 1, 96)
        nt = ttnn.from_torch(new.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.update_cache(ct, nt, 7)
        o = ttnn.to_torch(ct)
        return f"row7 pcc {pcc(o[0, :, 7], new[0, :, 0])}, row6 zero {o[0, :, 6].abs().max().item()}"

    try_("update_cache [1,H,T,96] <- [1,H,1,96] at 7", update_cache)

    def matmul_tb():
        a = torch.randn(1, H, 1, 96)
        b = torch.randn(1, H, T, 96)
        at = ttnn.from_torch(a.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        bt = ttnn.from_torch(b.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        o = ttnn.matmul(at, bt, transpose_b=True)
        return f"{tuple(o.shape)} {pcc(ttnn.to_torch(o), a @ b.transpose(-1, -2))}"

    try_("matmul transpose_b", matmul_tb)

    def rms():
        x = torch.randn(1, 1, 1, 2560)
        w = torch.rand(2560) + 0.5
        xt = ttnn.from_torch(x.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        wt = ttnn.from_torch(w.reshape(1, 1, 2560 // 32, 32).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        o = ttnn.rms_norm(xt, epsilon=1e-5, weight=wt)
        ref = w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        return pcc(ttnn.to_torch(o), ref)

    try_("rms_norm 2560", rms)

    def rms256():
        x = torch.randn(1, 1, 1, 256)
        w = torch.rand(256) + 0.5
        xt = ttnn.from_torch(x.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        wt = ttnn.from_torch(w.reshape(1, 1, 256 // 32, 32).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        o = ttnn.rms_norm(xt, epsilon=1e-5, weight=wt)
        ref = w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
        return pcc(ttnn.to_torch(o), ref)

    try_("rms_norm 256", rms256)

    def update_cache_kvpe():
        cache = torch.zeros(1, 1, T, D)
        ct = ttnn.from_torch(cache.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        new = torch.randn(1, 1, 1, D)
        nt = ttnn.from_torch(new.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.update_cache(ct, nt, 7)
        o = ttnn.to_torch(ct)
        return f"row7 pcc {pcc(o[0, 0, 7], new[0, 0, 0])}, row6 zero {o[0, 0, 6].abs().max().item()}"

    try_("update_cache [1,1,T,288] <- [1,1,1,288] at 7", update_cache_kvpe)

    def reshape_heads():
        x = torch.randn(1, 1, 1, 3840)
        xt = ttnn.from_torch(x.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        r = ttnn.reshape(xt, (1, 1, 40, 96))
        o = ttnn.to_torch(r)
        a = f"reshape tile {tuple(o.shape)} {pcc(o, x.reshape(1, 1, 40, 96))}"
        t = ttnn.transpose(r, 1, 2)
        o2 = ttnn.to_torch(t)
        return a + f"; transpose {tuple(o2.shape)} {pcc(o2, x.reshape(1, 40, 1, 96))}"

    try_("reshape [1,1,1,3840] -> [1,1,40,96] and transpose", reshape_heads)

    def slice_cat():
        x = torch.randn(1, 1, 40, 96)
        xt = ttnn.from_torch(x.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        a = ttnn.slice(xt, [0, 0, 0, 0], [1, 1, 40, 64])
        b = ttnn.slice(xt, [0, 0, 0, 64], [1, 1, 40, 96])
        c = ttnn.concat([a, b], dim=-1)
        return f"slice/concat {pcc(ttnn.to_torch(c), x)}"

    try_("slice and concat on last dim", slice_cat)

    def bmm():
        a = torch.randn(1, 40, 1, 64)
        w = torch.randn(1, 40, 64, 256)
        at = ttnn.from_torch(a.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        wt = ttnn.from_torch(w.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        o = ttnn.matmul(at, wt)
        return f"{tuple(o.shape)} {pcc(ttnn.to_torch(o), a @ w)}"

    try_("batched matmul 40 heads [1,64]x[64,256]", bmm)
finally:
    ttnn.close_device(dev)
