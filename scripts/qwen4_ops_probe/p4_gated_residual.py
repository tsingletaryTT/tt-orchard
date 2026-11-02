# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 4: the Qwen4Exp gated residual composed from ttnn ops, against the torch module with random weights.

Defaults are Flash-Next: hidden 2560, hc_count 4, hc_lowrank 320. Steps, each a ttnn op:
  grouped RMSNorm   reshape [1,1,T,4*H] -> [1,1,T*4,H], ttnn.rms_norm (no weight), reshape back, multiply by (1 + w)
  mix weights       ttnn.linear down, multiply by 1/hc_count, ttnn.silu, ttnn.linear up, ttnn.sigmoid
  mixed input       multiply the gate by the normed input, reshape to [1,T,4,H], mean over the stream axis; two forms are
                    compared, ttnn.mean(dim=2) and the sum of four slices times 1/4
  injection         ttnn.linear to 4 columns, multiply by 1/hc_count, ttnn.sigmoid, multiply by 2
"workarounds" in the result lists every step that raised or needed the second form.

Run on a chip only inside a gozer lease. `--selftest` checks the CPU reference and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def make_module(hidden, hc_count, lowrank, seed=0):
    """The reference module with random weights rounded to bf16 (so the device sees the same values)."""
    torch.manual_seed(seed)
    mod = common.GatedResidualRef(hidden, hc_count, lowrank)
    with torch.no_grad():
        mod.norm_weight.copy_(torch.randn_like(mod.norm_weight) * 0.1)
        for p in mod.parameters():
            p.copy_(p.bfloat16().float())
    return mod


def selftest() -> dict:
    """The module's mixed output equals an explicit per-stream computation, and its injection stays inside (0, 2)."""
    mod = make_module(8, 4, 6)
    x = torch.randn(1, 5, 32)
    mixed, inj = mod(x)
    normed = mod.hc_norm(x).reshape(1, 5, 4, 8)
    gate = torch.sigmoid(mod.up(torch.nn.functional.silu(mod.down(normed.reshape(1, 5, 32)) / 4))).reshape(1, 5, 4, 8)
    manual = sum(gate[:, :, i] * normed[:, :, i] for i in range(4)) / 4
    out = {"mixed_vs_manual": common.error_stats(mixed, manual)["max_abs"],
           "injection_in_range": bool((inj > 0).all() and (inj < 2).all()), "mixed_shape": list(mixed.shape)}
    out["ok"] = out["mixed_vs_manual"] < 1e-5 and out["injection_in_range"] and out["mixed_shape"] == [1, 5, 8]
    out["probe"], out["mode"] = "p4_gated_residual", "selftest"
    return out


def run_device(ttnn, dev, mod, x, notes):
    """Composed ttnn path. Returns (mixed_mean_form, mixed_slices_form, injection), torch tensors."""
    hidden, hc, seq = mod.hidden, mod.hc_count, x.shape[1]
    n = hidden * hc

    def tile(t, dt=ttnn.bfloat16):
        return ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

    xt = tile(x.reshape(1, 1, seq, n))
    normed = ttnn.reshape(ttnn.rms_norm(ttnn.reshape(xt, (1, 1, seq * hc, hidden)), epsilon=mod.eps), (1, 1, seq, n))
    normed = ttnn.multiply(normed, tile((1.0 + mod.norm_weight.detach()).reshape(1, 1, 1, n)))
    wd, wu, wi = (tile(m.weight.detach().t().reshape(1, 1, *m.weight.t().shape)) for m in (mod.down, mod.up, mod.inject))
    mix = ttnn.silu(ttnn.multiply(ttnn.linear(normed, wd), 1.0 / hc))
    gate = ttnn.sigmoid(ttnn.linear(mix, wu))
    prod = ttnn.reshape(ttnn.multiply(gate, normed), (1, seq, hc, hidden))
    try:
        mean_form = ttnn.to_torch(ttnn.mean(prod, dim=2, keepdim=True)).float().reshape(1, seq, hidden)
    except Exception as exc:
        notes.append("ttnn.mean(dim=2) raised: " + common.short_exc(exc))
        mean_form = None
    parts = [ttnn.slice(prod, [0, 0, i, 0], [1, seq, i + 1, hidden]) for i in range(hc)]
    acc = parts[0]
    for p in parts[1:]:
        acc = ttnn.add(acc, p)
    slices_form = ttnn.to_torch(ttnn.multiply(acc, 1.0 / hc)).float().reshape(1, seq, hidden)
    inj = ttnn.multiply(ttnn.sigmoid(ttnn.multiply(ttnn.linear(normed, wi), 1.0 / hc)), 2.0)
    injection = ttnn.to_torch(inj).float().reshape(1, seq, hc)
    return mean_form, slices_form, injection


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the reference side; no ttnn")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--hc-count", type=int, default=4)
    ap.add_argument("--lowrank", type=int, default=320)
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest()
        common.emit(res)
        return 0 if res["ok"] else 1
    mod = make_module(args.hidden, args.hc_count, args.lowrank)
    x = torch.randn(1, args.seq, args.hidden * args.hc_count).bfloat16().float()
    with torch.no_grad():
        ref_mixed, ref_inj = mod(x)
    ttnn, dev = common.open_device()
    notes, entry = [], {}
    try:
        try:
            (mean_form, slices_form, injection), times = common.timed(lambda: run_device(ttnn, dev, mod, x, notes),
                                                                      ttnn, dev)
            entry.update(wall_s=times, injection=common.error_stats(injection, ref_inj),
                         mixed_slices_form=common.error_stats(slices_form, ref_mixed))
            if mean_form is not None:
                entry["mixed_mean_form"] = common.error_stats(mean_form, ref_mixed)
        except Exception as exc:
            entry["failure"] = common.short_exc(exc)
    finally:
        ttnn.close_device(dev)
    common.emit({"probe": "p4_gated_residual", "shape": {"T": args.seq, "hidden": args.hidden, "hc_count": args.hc_count,
                                                         "lowrank": args.lowrank},
                 "dtype": "bf16 activations and weights", "reference": "torch GatedResidualRef fp32 on bf16-rounded "
                 "weights", "workarounds": sorted(set(notes)), "result": entry})
    return 0


if __name__ == "__main__":
    sys.exit(main())
