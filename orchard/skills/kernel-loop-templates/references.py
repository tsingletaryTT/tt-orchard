# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Torch references for the probe's ops. Inputs arrive as float32 tensors, rounded to bf16 first, in the spec's order."""
import torch


def add(x):
    return x["a"] + x["b"]


def silu_mul(x):
    return torch.nn.functional.silu(x["gate"]) * x["up"]


def rmsnorm(x):
    # w is the weight already expanded to the full (rows, cols) shape: a deliberate simplification so the first
    # measurement does not depend on a broadcast. eps is 1e-6.
    h = x["x"]
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * x["w"]


def linear_scan(x):
    # h[t] = a[t] * h[t-1] + b[t], scanned down axis 0, independently per column.
    a, b = x["a"], x["b"]
    h, out = torch.zeros_like(b[0]), torch.empty_like(b)
    for t in range(b.shape[0]):
        h = a[t] * h + b[t]
        out[t] = h
    return out


REFERENCES = {"add": add, "silu_mul": silu_mul, "rmsnorm": rmsnorm, "linear_scan": linear_scan}
