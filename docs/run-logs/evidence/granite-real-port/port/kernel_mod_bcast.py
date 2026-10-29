"""granite_tt mode `kernel` with the coder's compact state-update kernel (ssd_state_update_bcast) and a composed readout.

The kernel takes decay and dtx as [3072, 32] (value in column 0) and B as [32, 128] (value in row 0), so the expanded copies are 4 times
smaller than for ssd_state_update. BCAST_KERNEL names the kernel file.
"""
import importlib.util
import os

import torch
import ttnn

KERNEL = os.environ.get("BCAST_KERNEL", "<RUN_DIR>/port/kernels/bcast.py")
H, P, N = 48, 64, 128


def build(dev):
    spec = importlib.util.spec_from_file_location("bcast", KERNEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    state_k = mod.ssd_state_update_bcast
    out_s = ttnn.from_torch(torch.zeros(H * P, N, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def run(S, decay, dtx, B, C):
        s2 = ttnn.reshape(S, (H * P, N))
        dec = ttnn.reshape(ttnn.repeat(decay, (1, P, 32)), (H * P, 32))
        dx = ttnn.reshape(ttnn.repeat(dtx, (1, 1, 32)), (H * P, 32))
        b = ttnn.reshape(ttnn.repeat(B, (1, 32, 1)), (32, N))
        c = ttnn.reshape(ttnn.repeat(C, (H, P, 1)), (H * P, N))
        state_k(s2, dec, dx, b, out_s)
        new_s = ttnn.reshape(ttnn.clone(out_s), (H, P, N))
        y = ttnn.reshape(ttnn.sum(ttnn.multiply(c, out_s), dim=1, keepdim=True), (H, P, 1))
        return y, new_s

    return run
