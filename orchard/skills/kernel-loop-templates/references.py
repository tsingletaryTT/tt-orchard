# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Torch references for the probe's ops. Inputs arrive as float32 tensors, rounded to bf16 first, in the spec's order."""
import torch


def add(x):
    return x["a"] + x["b"]


def mul(x):
    return x["a"] * x["b"]


def scale_shift(x):
    return x["x"] * 2.5 + 0.5


def add_row(x):
    # r has 32 rows (one tile row) and is repeated down the rows of x: out[i, j] = x[i, j] + r[i mod 32, j].
    h, r = x["x"], x["r"]
    return h + r.repeat(h.shape[0] // r.shape[0], 1)


def rsqrt(x):
    return torch.rsqrt(x["x"])


def row_sum(x):
    # The sum of each row goes into column 0 of a (rows, 32) output; the other 31 columns are 0.
    h = x["x"]
    out = torch.zeros(h.shape[0], 32, dtype=h.dtype)
    out[:, 0] = h.sum(-1)
    return out


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


def center(x):
    return x["x"] - x["x"].mean(-1, keepdim=True)


def softmax(x):
    return torch.softmax(x["x"], dim=-1)


def tile_scan(x):
    # The recurrence of linear_scan, taken across tile rows (32 rows at a time): out[T] = a[T] * out[T-1] + b[T] for every
    # element of tile row T, with out[-1] = 0. Rows inside one tile do not feed each other.
    a, b = x["a"], x["b"]
    rows, cols = b.shape
    a, b = a.reshape(rows // 32, 32, cols), b.reshape(rows // 32, 32, cols)
    h, out = torch.zeros_like(b[0]), torch.empty_like(b)
    for t in range(b.shape[0]):
        h = a[t] * h + b[t]
        out[t] = h
    return out.reshape(rows, cols)


def ssd_state_update(x):
    # The Mamba-2 state update, state' = state * exp(dt * A) + dt * x * B, with the decay (exp(dt * A)), dt * x and B each
    # already expanded to the full shape of the state: a deliberate simplification, as in rmsnorm's w.
    return x["decay"] * x["s"] + x["dtx"] * x["b"]


def ssd_readout(x):
    # The Mamba-2 readout, y = state . C: the sum of c * s over the last axis, written to every column of its row
    # (the reduction and then the broadcast, as in rmsnorm), with C already expanded to the full shape of the state.
    return (x["c"] * x["s"]).sum(-1, keepdim=True).expand_as(x["s"]).contiguous()


REFERENCES = {"add": add, "mul": mul, "scale_shift": scale_shift, "add_row": add_row, "rsqrt": rsqrt,
              "row_sum": row_sum, "silu_mul": silu_mul, "rmsnorm": rmsnorm, "linear_scan": linear_scan, "center": center,
              "softmax": softmax, "tile_scan": tile_scan, "ssd_state_update": ssd_state_update,
              "ssd_readout": ssd_readout}
