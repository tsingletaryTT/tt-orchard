# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Torch references for the worked ops of the cheat sheet's recipes (ttlang_recipes.json). None is a probe op.

Used only by scripts/verify_ttlang_recipes.py, which adds them to a scratch copy of references.py so the unedited check can judge a
recipe's skeleton. Nothing copies this file into an attempt directory. Inputs arrive as float32 tensors, rounded to bf16 first, in the
spec's order."""
import torch


def recipe_l2norm(x):
    h = x["x"]
    return h * torch.rsqrt((h * h).sum(-1, keepdim=True) + 1e-6)


def recipe_col_mean(x):
    # The mean of each column goes into row 0 of a (32, columns) output; the other 31 rows are 0.
    h = x["x"]
    out = torch.zeros(32, h.shape[1], dtype=h.dtype)
    out[0, :] = h.mean(0)
    return out


def recipe_gated_bias(x):
    # r has 32 rows (one tile row) and is repeated down the rows of x.
    h, y, r = x["x"], x["y"], x["r"]
    return h * y * 0.5 + r.repeat(h.shape[0] // r.shape[0], 1)


def recipe_running_sum(x):
    # A running sum down the tile rows: out tile-row t is the sum of x tile-rows 0 to t.
    h = x["x"]
    t = h.shape[0] // 32
    return h.reshape(t, 32, h.shape[1]).cumsum(0).reshape(h.shape)


REFERENCES = {"recipe_l2norm": recipe_l2norm, "recipe_col_mean": recipe_col_mean,
              "recipe_gated_bias": recipe_gated_bias, "recipe_running_sum": recipe_running_sum}


# ---- the worked ops of the control section (ttlang_recipes_control.json) -----------------------------------------------------------
# The control section has the same form and length as the recipes and none of their content: four elementwise kernels with no
# reduction, no broadcast, no constant block and no carried value. An A/B arm that shows it measures what more sheet text does by itself.

def recipe_ctl_add(x):
    return x["x"] + x["y"]


def recipe_ctl_sub(x):
    return x["x"] - x["y"]


def recipe_ctl_mul(x):
    return x["x"] * x["y"]


def recipe_ctl_double(x):
    return x["x"] + x["x"]


REFERENCES.update({"recipe_ctl_add": recipe_ctl_add, "recipe_ctl_sub": recipe_ctl_sub, "recipe_ctl_mul": recipe_ctl_mul,
                   "recipe_ctl_double": recipe_ctl_double})
