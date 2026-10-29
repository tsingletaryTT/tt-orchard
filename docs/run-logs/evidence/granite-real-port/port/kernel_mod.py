"""Builds the `kernels` callable for granite_tt mode `kernel` from the two coder-written tt-lang kernels.

KERNEL_DIR holds state.py (ssd_state_update) and readout.py (ssd_readout), copied unchanged from the probe attempts that passed
the simulator. The kernels take the state-shaped inputs at full size, so the per-head decay, dt*x, B and C are expanded first
(ttnn.repeat), which costs extra data movement the composed path does not pay.
"""
import importlib.util
import os

import torch
import ttnn

KERNEL_DIR = os.environ.get("GRANITE_KERNEL_DIR", "<RUN_DIR>/port/kernels")
H, P, N = 48, 64, 128


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(KERNEL_DIR, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)


def build(dev):
    state_k, readout_k = _load("state", "ssd_state_update"), _load("readout", "ssd_readout")
    zeros = lambda: ttnn.from_torch(torch.zeros(H * P, N, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out_s, out_y = zeros(), zeros()

    def run(S, decay, dtx, B, C):
        """S [48,64,128]; decay [48,1,1]; dtx [48,64,1]; B and C [1,1,128]. Returns (y [48,64,1], new S [48,64,128])."""
        s2 = ttnn.reshape(S, (H * P, N))
        dec = ttnn.reshape(ttnn.repeat(decay, (1, P, N)), (H * P, N))
        dx = ttnn.reshape(ttnn.repeat(dtx, (1, 1, N)), (H * P, N))
        b = ttnn.reshape(ttnn.repeat(B, (H, P, 1)), (H * P, N))
        c = ttnn.reshape(ttnn.repeat(C, (H, P, 1)), (H * P, N))
        state_k(s2, dec, dx, b, out_s)
        readout_k(c, out_s, out_y)
        new_s = ttnn.reshape(ttnn.clone(out_s), (H, P, N))
        y = ttnn.reshape(ttnn.slice(out_y, [0, 0], [H * P, 1]), (H, P, 1))
        return y, new_s

    return run
