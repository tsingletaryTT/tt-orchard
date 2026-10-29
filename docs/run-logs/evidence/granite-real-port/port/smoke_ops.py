"""Which ttnn ops work for the Granite step on this chip: M=1 linear, rms_norm with a vector weight, silu*mul, a GQA
attention decode written from matmul/softmax/concat, and a Mamba-2 SSD step composed from stock ops (the baseline)."""
import sys
import time

import torch
import ttnn

sys.path.insert(0, "<RUN_DIR>/port")
import granite_ref as g

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
TILE, DRAM = ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG


def up(t, layout=TILE, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t.to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32), dtype=dtype, layout=layout, device=dev, memory_config=DRAM)


def down(t):
    return ttnn.to_torch(t).float()


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


try:
    # 1. linear at M=1
    x = torch.randn(1, 1536)
    W = torch.randn(6448, 1536) * 0.02
    y = ttnn.linear(up(x.view(1, 1, 1, 1536)), up(W.t().contiguous().view(1, 1, 1536, 6448)))
    print("linear M=1 shape", y.shape, "rel", rel(down(y).view(-1)[:6448], (x.to(torch.bfloat16).float() @ W.to(torch.bfloat16).float().t()).view(-1)))

    # 2. rms_norm with vector weight
    wt = torch.randn(1536).abs() + 0.5
    for name, wt_t in (("tile[1,1,1,1536]", up(wt.view(1, 1, 1, 1536))), ("rm[1,1,48,32]", up(wt.view(1, 1, 48, 32), layout=ttnn.ROW_MAJOR_LAYOUT))):
        try:
            r = ttnn.rms_norm(up(x.view(1, 1, 1, 1536)), weight=wt_t, epsilon=1e-5)
            print("rms_norm", name, "rel", rel(down(r).view(-1)[:1536], g.rmsnorm(x[0], wt, 1e-5)))
        except Exception as e:
            print("rms_norm", name, "ERR", str(e)[:200])

    # 3. silu * mul
    a, b = torch.randn(1, 4096), torch.randn(1, 4096)
    o = ttnn.multiply(ttnn.silu(up(a.view(1, 1, 1, 4096))), up(b.view(1, 1, 1, 4096)))
    print("silu*mul rel", rel(down(o).view(-1)[:4096], (torch.nn.functional.silu(a) * b)[0]))

    # 4. attention decode, T = 37 (not a multiple of 32)
    T, nh, nkv, hd = 37, 12, 4, 128
    q = torch.randn(nh, hd); K = torch.randn(nkv, T, hd); V = torch.randn(nkv, T, hd)
    Kr, Vr = K.repeat_interleave(3, 0), V.repeat_interleave(3, 0)
    ref = (torch.softmax((q[:, None, :] * Kr).sum(-1) * 0.0078125, -1)[..., None] * Vr).sum(1)
    qd = up(q.view(1, nkv, 3, hd))
    Kd = up(K.view(1, nkv, T, hd)); Vd = up(V.view(1, nkv, T, hd))
    s = ttnn.matmul(qd, ttnn.transpose(Kd, -2, -1))                    # [1,4,3,T]
    s = ttnn.multiply(s, 0.0078125)
    p = ttnn.softmax(s, dim=-1)
    ctx = ttnn.matmul(p, Vd)                                           # [1,4,3,hd]
    print("attention shapes", s.shape, ctx.shape, "rel", rel(down(ctx).view(nh, hd), ref))
    # concat growth in row-major
    Kc = up(K[:, :36].reshape(1, nkv, 36, hd), layout=ttnn.ROW_MAJOR_LAYOUT)
    knew = up(K[:, 36:37].reshape(1, nkv, 1, hd), layout=ttnn.ROW_MAJOR_LAYOUT)
    Kc2 = ttnn.concat([Kc, knew], dim=2)
    print("concat row-major", Kc2.shape, "rel", rel(down(Kc2).view(nkv, T, hd), K))

    # 5. SSD step composed from stock ttnn ops (baseline for the kernel)
    H, P, N = 48, 64, 128
    st = torch.randn(H, P, N) * 0.1
    xx, dtr, Bv, Cv = torch.randn(H, P), torch.randn(H), torch.randn(N), torch.randn(N)
    Alog, dtb, Dv = torch.randn(H) * 0.3, torch.randn(H) * 0.3, torch.randn(H)
    st_ref = st.clone()
    y_ref = g.ssd_step(st_ref, xx, dtr, Alog, dtb, Bv, Cv, Dv)
    dt = torch.nn.functional.softplus(dtr + dtb)
    decay = torch.exp(dt * -torch.exp(Alog))
    S = up(st.view(H, P, N))
    t0 = time.time()
    S2 = ttnn.add(ttnn.multiply(S, up(decay.view(H, 1, 1))), ttnn.multiply(up((dt[:, None] * xx).view(H, P, 1)), up(Bv.view(1, 1, N))))
    yy = ttnn.matmul(S2, up(Cv.view(1, N, 1).expand(H, N, 1).contiguous()))          # [H,P,1]
    yy = ttnn.add(yy, up((Dv[:, None] * xx).view(H, P, 1)))
    ttnn.synchronize_device(dev)
    print("composed SSD state rel", rel(down(S2).view(H, P, N), st_ref), "y rel", rel(down(yy).view(H, P), y_ref), "first call s", round(time.time() - t0, 3))
finally:
    ttnn.close_device(dev)
