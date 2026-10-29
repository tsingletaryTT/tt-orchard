# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Torch references for the INTERMEDIATE stages of the staged kernel probe (orchard/kernelstaged.py).

Each function takes the inputs as float32 tensors rounded to bf16, keyed by input name, as references.py does, and returns
the output that stage's kernel must write. The staged probe copies this file next to references.py in the attempt
directory and adds STAGE_REFERENCES to references.REFERENCES there, so a stage's spec names its reference like any op.
The final stage of every op uses the op's own reference from references.py.

The reduction stages write their result in the layout of the `row_sum` op: column 0 of a (rows, 32) output, the other 31
columns zero, which is where a reduction over the last axis leaves its result in a tile.
"""
import torch

EPS = 1e-6               # the epsilon of references.rmsnorm


def copy(x):
    return x["x"].clone()


def square(x):
    return x["x"] * x["x"]


def scale_by_inverse_width(x):
    # x * (1 / 256): the factor a mean over the op's 256 columns uses. 256 is a power of two, so the product is exact.
    return x["x"] * (1.0 / x["x"].shape[-1])


def _column0(values):
    out = torch.zeros(values.shape[0], 32, dtype=values.dtype)
    out[:, 0] = values
    return out


def row_sum_of_squares(x):
    return _column0((x["x"] * x["x"]).sum(-1))


def row_mean(x):
    return _column0(x["x"].mean(-1))


def scale_rows_by_inverse_rms(x):
    h = x["x"]
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)


STAGE_REFERENCES = {
    "stage_copy": copy,
    "stage_square": square,
    "stage_scale_by_inverse_width": scale_by_inverse_width,
    "stage_row_sum_of_squares": row_sum_of_squares,
    "stage_row_mean": row_mean,
    "stage_scale_rows_by_inverse_rms": scale_rows_by_inverse_rms,
}
