# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""flash_multi_latent_attention_decode probes at the MiniCPM3 shapes (40 heads, 256 latent + 32 rope, one kv head, batch 1)."""
import time

import torch
import ttnn

H, KV, T = 40, 256, 128
D = KV + 32


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return round(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), 5), round(((a - b).norm() / b.norm()).item(), 4)


dev = ttnn.open_device(device_id=0)
try:
    print("grid", dev.compute_with_storage_grid_size(), flush=True)
    torch.manual_seed(0)
    q = torch.randn(1, 1, H, D) * 0.5
    k = torch.randn(1, 1, T, D) * 0.5
    pos = 100
    scale = 96 ** -0.5
    ref = torch.softmax((q[0, 0] @ k[0, 0, : pos + 1].T) * scale, -1) @ k[0, 0, : pos + 1, :KV]
    qt = ttnn.from_torch(q.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    kt = ttnn.from_torch(k.bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    cur = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def attempt(name, **kw):
        t0 = time.time()
        try:
            out = ttnn.transformer.flash_multi_latent_attention_decode(qt, kt, head_dim_v=KV, scale=scale, **kw)
            o = ttnn.to_torch(out)
            print(f"[{name}] ok {tuple(o.shape)} {pcc(o.reshape(-1, KV)[:H], ref)} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED {str(e).splitlines()[0][:300]}", flush=True)

    for gx, gy in ((8, 8), (8, 4), (4, 4), (8, 7), (10, 6)):
        for qc, kc in ((32, 128), (32, 32)):
            try:
                pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), q_chunk_size=qc, k_chunk_size=kc,
                                            exp_approx_mode=False)
            except Exception as e:  # noqa: BLE001
                print("program config failed", str(e)[:200])
                continue
            attempt(f"grid {gx}x{gy} q{qc} k{kc} cur_pos_tensor", program_config=pc, cur_pos_tensor=cur)
    attempt("grid 8x8 cur_pos list", program_config=ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=32,
                                                                           k_chunk_size=128, exp_approx_mode=False), cur_pos=[pos])
finally:
    ttnn.close_device(dev)
