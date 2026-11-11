# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 6: the generated `_apply_rotary` (orchard/port_template.py) on a chip: a rotate-half matmul and a multiply that broadcasts a
[1, 1, S, head_dim] table over the heads axis.

The generated helper computes `x * cos + (x @ R) * sin` for x [1, heads, S, head_dim]: `ttnn.linear` of x reshaped to [1, 1, heads * S, head_dim]
with the constant rotate-half matrix R, then `ttnn.multiply` and `ttnn.add` with the tables [1, 1, S, head_dim] broadcast over the heads axis. The CPU
tests show the formula equals the transformers `apply_rotary_pos_emb` for partial rotary. This probe asks whether the two device ops accept those
shapes (arm E14 never reached them with a valid layout). Steps, each compared with the CPU reference and recorded alone:
  linear_rotate     ttnn.linear(x_flat, R)
  multiply_bcast    ttnn.multiply(x, cos) with cos [1, 1, S, hd] against x [1, heads, S, hd]
  generated_full    the exact generated sequence
Workarounds, tried for each step that raised or missed (named in the result's `workarounds`):
  table_concat      the table repeated over heads with ttnn.concat([table] * heads, dim=1), then a same-shape multiply
  table_repeat      ttnn.repeat(table, ttnn.Shape([1, heads, 1, 1]))
  table_host        the table expanded to [1, heads, S, hd] on the host and uploaded
  flat_table_host   x as [1, 1, heads * S, hd] and the table tiled to the same shape on the host: no broadcast at all
Shape sets: --heads and --kv-heads (query heads and key/value heads), --seq, --hd, --rot. Defaults are the tiny checkpoint (8 and 2 heads, 32, 32, 8);
`--preset flash` is the Flash-Next-like set (24 and 2 heads, head_dim 256, rotary 64, 32 tokens).

Run on a chip only inside a gozer lease. `--selftest` checks the CPU reference and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PRESETS = {"tiny": dict(heads=8, kv_heads=2, seq=32, hd=32, rot=8), "flash": dict(heads=24, kv_heads=2, seq=32, hd=256, rot=64)}


def reference(x, seq, hd, rot):
    """(want, cos_pad, sin_pad, R): the transformers-form rotary of x, the padded tables and the rotate-half matrix."""
    cos, sin = common.rope_tables_ref(seq, hd, rot)
    want = common.apply_rotary_ref(x, cos[None, None], sin[None, None])
    pc, ps = common.padded_tables(cos, sin, hd)
    return want, pc, ps, common.rotate_half_matrix(hd, rot)


def selftest() -> dict:
    checks = {}
    for heads in (4, 2):
        for rot in (8, 32):
            x = torch.randn(1, heads, 32, 32)
            want, pc, ps, r = reference(x, 32, 32, rot)
            formula = x * pc + (x.reshape(1, 1, heads * 32, 32) @ r).reshape(1, heads, 32, 32) * ps
            checks[f"formula_heads{heads}_rot{rot}"] = bool((formula - want).abs().max() < 1e-5)
    x = torch.randn(1, 4, 32, 32)
    want, pc, ps, r = reference(x, 32, 32, 8)
    checks["tail_columns_pass_through"] = bool(torch.equal(want[..., 8:], x[..., 8:]))
    checks["rotary_columns_change"] = bool(not torch.allclose(want[..., :8], x[..., :8], atol=1e-3))
    checks["expanded_table_gives_the_same_result"] = bool(((x * pc.expand(1, 4, 32, 32)) - (x * pc)).abs().max() == 0)
    checks["workaround_forms_are_named"] = set(WORKAROUNDS) == {"table_concat", "table_repeat", "table_host", "flat_table_host"}
    return {"probe": "p6_rotary_broadcast", "mode": "selftest", "checks": checks, "ok": all(checks.values())}


WORKAROUNDS = ("table_concat", "table_repeat", "table_host", "flat_table_host")


def run_device(ttnn, dev, heads, seq, hd, rot):
    """The steps for one head count. Returns the dict of step results."""
    x = torch.randn(1, heads, seq, hd).bfloat16().float()
    want, pc, ps, rmat = reference(x, seq, hd, rot)
    tile = lambda t: ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    xt, ct, st, rt = tile(x), tile(pc), tile(ps), tile(rmat)
    out, used = {}, []

    def step(name, fn, ref):
        try:
            got = ttnn.to_torch(fn()).float()
            out[name] = common.error_stats(got, ref)
            out[name]["ok"] = bool(out[name].get("pcc") is not None and out[name]["pcc"] > 0.999)
        except Exception as exc:
            out[name] = {"failure": common.short_exc(exc), "ok": False}
        return out[name]["ok"]

    flat = lambda: ttnn.reshape(xt, (1, 1, heads * seq, hd))
    ref_rot = (x.reshape(1, 1, heads * seq, hd) @ rmat)
    step("linear_rotate", lambda: ttnn.linear(flat(), rt, dtype=ttnn.bfloat16), ref_rot)
    step("multiply_bcast", lambda: ttnn.multiply(xt, ct), x * pc)
    ok = step("generated_full", lambda: _generated(ttnn, xt, ct, st, rt, heads, seq, hd), want)
    if not ok or not out["multiply_bcast"]["ok"]:
        ref_xc = x * pc
        step("w_table_concat", lambda: ttnn.multiply(xt, ttnn.concat([ct] * heads, dim=1)), ref_xc)
        step("w_table_repeat", lambda: ttnn.multiply(xt, ttnn.repeat(ct, ttnn.Shape([1, heads, 1, 1]))), ref_xc)
        step("w_table_host", lambda: ttnn.multiply(xt, tile(pc.expand(1, heads, seq, hd))), ref_xc)
        step("w_flat_table_host", lambda: ttnn.multiply(flat(), tile(pc.expand(1, heads, seq, hd).reshape(1, 1, heads * seq, hd))),
             ref_xc.reshape(1, 1, heads * seq, hd))
        used = [w for w in WORKAROUNDS if out.get("w_" + w, {}).get("ok")]
    return {"steps": out, "workarounds": used}


def _generated(ttnn, xt, ct, st, rt, heads, seq, hd):
    """The exact sequence of the generated helper."""
    flat = ttnn.reshape(xt, (1, 1, heads * seq, hd))
    rotated = ttnn.linear(flat, rt, dtype=ttnn.bfloat16)
    rotated = ttnn.reshape(rotated, (1, heads, seq, hd))
    return ttnn.add(ttnn.multiply(xt, ct), ttnn.multiply(rotated, st))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the reference forms; no ttnn")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    for key in ("heads", "kv-heads", "seq", "hd", "rot"):
        ap.add_argument("--" + key, type=int, default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest()
        common.emit(res)
        return 0 if res["ok"] else 1
    cfg = dict(PRESETS[args.preset])
    for key in cfg:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    opened = common.device_or_report("p6_rotary_broadcast")
    if opened is None:
        return 2
    ttnn, dev = opened
    results = {}
    try:
        for name in ("heads", "kv_heads"):
            results[name] = {"shape": [1, cfg[name], cfg["seq"], cfg["hd"]], **run_device(ttnn, dev, cfg[name], cfg["seq"], cfg["hd"], cfg["rot"])}
    finally:
        ttnn.close_device(dev)
    common.emit({"probe": "p6_rotary_broadcast", "mode": "device", "shape": cfg, "dtype": "bf16 TILE", "ok": True,
                 "reference": "transformers apply_rotary_pos_emb (CPU copy)", "result": results,
                 "workarounds": sorted({w for r in results.values() for w in r["workarounds"]})})
    return 0


if __name__ == "__main__":
    sys.exit(main())
