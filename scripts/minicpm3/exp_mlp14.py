# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where does the 8 percent error on the first token at layer 14 come from? Runs the MLP of one layer on the chip for the reference input row, in variants,
and compares each stage with torch float32 on the same input."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open

sys.path.insert(0, __file__.rsplit("/", 3)[0])
from orchard import audio_reference as ar  # noqa: E402

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 14
ROW = int(sys.argv[2]) if len(sys.argv) > 2 else 0
REF = os.environ["MINICPM3_REF"]          # a recorded float32 reference run directory
W = os.environ["MINICPM3_WEIGHTS"]
run = ar.load_run(REF)
x_attn = torch.tensor(run.array("layer", f"layer_{LAYER:02d}.attn"))[ROW]       # stream after attention (float32)
x_out = torch.tensor(run.array("layer", f"layer_{LAYER:02d}"))[ROW]
f = safe_open(W, "pt")
p = f"model.layers.{LAYER}."
nw = f.get_tensor(p + "post_attention_layernorm.weight").float()
g, u, d = (f.get_tensor(p + f"mlp.{n}_proj.weight").float() for n in ("gate", "up", "down"))
rs = 1.4 / 62 ** 0.5


def rms(x):
    return nw * x * torch.rsqrt(x.pow(2).mean() + 1e-5)


h = rms(x_attn)
gate, up = g @ h, u @ h
m = F.silu(gate) * up
delta = d @ m * rs
print("row", ROW, "norms: x_attn", x_attn.norm().item(), "delta", delta.norm().item(), "ref delta", (x_out - x_attn).norm().item())
print("max |gate| %.1f  max |up| %.1f  max |m| %.1f" % (gate.abs().max(), up.abs().max(), m.abs().max()))


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


dev = ttnn.open_device(device_id=0)
try:
    def up_(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t.reshape(1, 1, 1, -1).bfloat16(), device=dev, dtype=dtype, layout=layout)

    def wt(t):
        return ttnn.from_torch(t.t().contiguous().bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    nwt = ttnn.from_torch(nw.reshape(1, 1, -1, 32).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    gt, ut, dt_ = wt(g), wt(u), wt(d * rs)
    xt = up_(x_attn)
    ht = ttnn.rms_norm(xt, epsilon=1e-5, weight=nwt)
    hh = ttnn.to_torch(ht).reshape(-1).float()
    print("rms_norm vs torch (bf16 input):", rel(hh, h))
    xb = x_attn.bfloat16().float()
    hb = nw * xb * torch.rsqrt(xb.pow(2).mean() + 1e-5)
    print("  torch rms of the bf16-rounded input vs fp32 input:", rel(hb, h), " chip vs torch on the same bf16 input:", rel(hh, hb))
    for name, kw in (("hifi4+fp32acc", dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
                     ("hifi4+fp32acc+approx", dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=True)),
                     ("lofi", dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, math_approx_mode=False))):
        ck = ttnn.init_device_compute_kernel_config(dev.arch(), **kw)
        o = ttnn.to_torch(ttnn.rms_norm(xt, epsilon=1e-5, weight=nwt, compute_kernel_config=ck)).reshape(-1).float()
        print(f"  rms_norm {name}: vs torch on the same input {rel(o, hb):.4f}")
    # a manual norm: square and mean in float32
    x32 = ttnn.typecast(xt, ttnn.float32)
    ms = ttnn.mean(ttnn.multiply(x32, x32), dim=-1, keepdim=True)
    inv = ttnn.rsqrt(ttnn.add(ms, 1e-5))
    o = ttnn.multiply(x32, inv)
    o = ttnn.to_torch(o).reshape(-1).float() * nw
    print(f"  manual fp32 norm: vs torch {rel(o, hb):.4f}")
    gc = ttnn.to_torch(ttnn.linear(ht, gt)).reshape(-1).float()
    uc = ttnn.to_torch(ttnn.linear(ht, ut)).reshape(-1).float()
    print("gate matmul rel err", rel(gc, gate), " up matmul rel err", rel(uc, up))
    gs = ttnn.to_torch(ttnn.linear(ht, gt, activation="silu")).reshape(-1).float()
    print("fused silu(gate) rel err", rel(gs, F.silu(gate)), " worst abs", (gs - F.silu(gate)).abs().max().item())
    gs2 = ttnn.to_torch(ttnn.silu(ttnn.linear(ht, gt))).reshape(-1).float()
    print("separate silu rel err", rel(gs2, F.silu(gate)))
    for name, act in (("fused", ttnn.linear(ht, gt, activation="silu")), ("separate", ttnn.silu(ttnn.linear(ht, gt)))):
        mt = ttnn.multiply(act, ttnn.linear(ht, ut))
        mc = ttnn.to_torch(mt).reshape(-1).float()
        out = ttnn.to_torch(ttnn.linear(mt, dt_)).reshape(-1).float()
        print(f"[{name}] m rel err {rel(mc, m):.4f}; mlp delta rel err {rel(out, delta):.4f}")
    # does the error sit in a few intermediate channels?
    err = (mc - m).abs()
    top = torch.topk(err, 5)
    print("worst m channels", top.indices.tolist(), "err", [round(v, 2) for v in top.values.tolist()], "m there", [round(m[i].item(), 2) for i in top.indices.tolist()])
finally:
    ttnn.close_device(dev)
