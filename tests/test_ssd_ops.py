# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Mamba-2 probe ops (`ssd_state_update`, `ssd_readout`): their references against the model's own decode step.

The probe's references expand per-head scalars and the shared B and C to the full state shape, so a kernel can be checked without a
broadcast. These tests hold the two references to the recurrence that Granite-4.0-H (transformers' `mamba2_selective_state_update`)
computes, written out with plain loops, so a reference that drifts from the model is caught here and not on a chip. They run torch
in the ttlang venv and skip, saying so, when that venv is absent.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "kernel-loop-templates"
VENV = Path(os.environ.get("ORCHARD_TTLANG_VENV", Path.home() / "orchard-venvs" / "ttlang"))

needs_torch = pytest.mark.skipif(not (VENV / "bin" / "python").is_file(), reason="the ttlang venv is absent (tt-orchard kernel-env)")

SCRIPT = r'''
import importlib.util, math, sys, torch
spec = importlib.util.spec_from_file_location("references", sys.argv[1] + "/references.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
R = mod.REFERENCES
g = torch.Generator().manual_seed(0)
H, P, N = 3, 4, 8                                        # heads, head dim, state size (small, so the loops are cheap)
state = torch.randn(H, P, N, generator=g)
x = torch.randn(H, P, generator=g)
dt = torch.rand(H, generator=g) + 0.1                    # already softplus'ed
A = -torch.rand(H, generator=g) - 0.1
B, C = torch.randn(N, generator=g), torch.randn(N, generator=g)

# the model's step, with loops: state'[h,p,n] = state[h,p,n] * exp(dt[h]*A[h]) + dt[h]*x[h,p]*B[n]; y[h,p] = sum_n C[n] * state'[h,p,n]
new = torch.empty_like(state); y = torch.empty(H, P)
for h in range(H):
    for p in range(P):
        acc = 0.0
        for n in range(N):
            new[h, p, n] = state[h, p, n] * math.exp(float(dt[h] * A[h])) + float(dt[h] * x[h, p] * B[n])
            acc += float(C[n] * new[h, p, n])
        y[h, p] = acc

# the probe's expanded layout: one row per (head, p), one column per state index
rows = H * P
s2 = state.reshape(rows, N)
decay = torch.exp(dt * A).repeat_interleave(P)[:, None].expand(rows, N)
dtx = (dt[:, None] * x).reshape(rows)[:, None].expand(rows, N)
b2, c2 = B[None, :].expand(rows, N), C[None, :].expand(rows, N)

got_state = R["ssd_state_update"]({"s": s2, "decay": decay, "dtx": dtx, "b": b2})
assert torch.allclose(got_state, new.reshape(rows, N), atol=1e-5), "ssd_state_update differs from the model's state update"
got_y = R["ssd_readout"]({"c": c2, "s": got_state})
assert got_y.shape == (rows, N), got_y.shape
assert torch.allclose(got_y, y.reshape(rows, 1).expand(rows, N), atol=1e-4), "ssd_readout differs from the model's readout"
assert all(torch.equal(got_y[:, 0], got_y[:, j]) for j in range(N)), "the readout must repeat the row sum in every column"
print("SSD REFERENCES OK")
'''


@needs_torch
def test_the_references_match_the_models_decode_step():
    done = subprocess.run([str(VENV / "bin" / "python"), "-c", SCRIPT, str(TEMPLATES)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "SSD REFERENCES OK" in done.stdout, done.stdout + done.stderr


def test_the_specs_give_the_shapes_the_kernels_are_judged_on():
    state = json.loads((TEMPLATES / "ops" / "ssd_state_update.json").read_text())
    readout = json.loads((TEMPLATES / "ops" / "ssd_readout.json").read_text())
    assert list(state["inputs"]) == ["s", "decay", "dtx", "b"] and state["gen"] == {"decay": "sigmoid"}
    assert list(readout["inputs"]) == ["c", "s"]
    for spec in (state, readout):
        assert spec["output"] == [64, 128] and all(v == [64, 128] for v in spec["inputs"].values())
