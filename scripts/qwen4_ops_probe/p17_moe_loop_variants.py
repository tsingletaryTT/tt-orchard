# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 17: five changes to the per-slot expert loop, each alone and all together, on one chip with 128 expert slots (issue 723).

Probe 16 showed that the loop costs about 0.12 ms per hit slot, that trace replay does not help, and that each op of a slot takes 8 to 37 microseconds. This probe
tries the changes that remove ops or make them cheaper. Flags (each is a variant by itself and all are combined in the last one):

  fuse_gate_up   gate and up weights stored side by side as one [hidden, 2*inter] matrix: one matmul and two slices instead of two matmuls
  fold_weight    the routing weight goes into the scatter matrix (entry [token, row] = weight), which drops the per-slot weight multiply
  fuse_silu      silu applied inside the gate-times-up multiply (`input_tensor_a_activations`) instead of its own op
  lofi           matmuls run with LoFi math fidelity and approximate mode (the weights have 3 or 7 mantissa bits, so the extra passes add little)
  bmm_all        (variants bmm_all_slots and bmm_all_slots_hifi) every slot at once: the weights of the 128 slots are stacked as [1, 128, in, out], one
                 embedding gathers 128 x rows rows (rows = the largest slot count padded to 32; slots nobody picked get zero scatter entries) and three batched
                 matmuls run all slots; the shape does not depend on which slots were hit, so the loop can be traced. It reads every expert's weights each layer.
  batch_scatter  one embedding for the ids of all slots, per-slot row slices, one concat of the outputs and one scatter matmul: no per-slot add or scatter

Per variant and token count it reports the device time of the loop (inputs already on the chip; the fastest of `repeat` calls), the time per hit slot and the
PCC of its output against the baseline loop's output on the same inputs. The weights are random bfp4; the routing is random (see probe 16). A variant the installed
ttnn cannot run reports its error and the others continue.

Run on a chip only inside a gozer lease. `--selftest` checks on the CPU that every variant adds up to the baseline loop and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common  # noqa: E402
import p16_moe_loop_cost as p16  # noqa: E402

FLAGS = ("fuse_gate_up", "fold_weight", "fuse_silu", "lofi", "batch_scatter")
HIDDEN, INTER, TOTAL, TOP_K, PER = 2560, 640, 512, 10, 128


def variants() -> list:
    """(name, set of flags): the baseline, each flag alone, then the flags added one at a time in the order of FLAGS."""
    out = [("baseline", frozenset())]
    out += [(f"only_{f}", frozenset([f])) for f in FLAGS]
    out += [("all_" + "+".join(FLAGS[: i + 1]), frozenset(FLAGS[: i + 1])) for i in range(1, len(FLAGS))]
    out += [("best_without_fuse_gate_up", frozenset(["fold_weight", "fuse_silu", "lofi", "batch_scatter"])),
            ("bmm_all_slots", frozenset(["bmm_all", "fold_weight", "fuse_silu", "lofi"])),
            ("bmm_all_slots_hifi", frozenset(["bmm_all", "fold_weight", "fuse_silu"]))]
    return out


# --- CPU forms of the variants (the selftest) ------------------------------------------------------------------------------------------------------

def cpu_variant(x, gate, up, down, items, flags):
    """The loop on the CPU written the way the variant writes it on the chip."""
    silu = torch.nn.functional.silu
    ye_all, back_all, out = [], [], torch.zeros_like(x)
    for j, ids, wt, back in items:
        xe = x[ids.long()]
        if "fuse_gate_up" in flags:
            gu = xe @ torch.cat([gate[j].t(), up[j].t()], dim=1)
            g, u = gu[:, :gate.shape[1]], gu[:, gate.shape[1]:]
        else:
            g, u = xe @ gate[j].t(), xe @ up[j].t()
        ye = (silu(g) * u) @ down[j].t()
        if "fold_weight" in flags:
            back = back * wt[None, :]
        else:
            ye = ye * wt[:, None]
        if "batch_scatter" in flags:
            ye_all.append(ye)
            back_all.append(back)
        else:
            out += back @ ye
    if "batch_scatter" in flags and ye_all:
        out += torch.cat(back_all, dim=1) @ torch.cat(ye_all, dim=0)
    return out


def selftest() -> dict:
    cfg = p16.PRESETS["tiny"]
    per = cfg["total_experts"] // 2
    t = common.moe_inputs(tokens=32, hidden=cfg["hidden"], experts=per, top_k=cfg["top_k"], inter=cfg["inter"], shared=cfg["inter"])
    idx, w = p16.make_routing(32, cfg["total_experts"], cfg["top_k"], 0)
    items = p16.slot_inputs(idx, w, per, 32)
    ref = p16.cpu_slot_loop(t["hidden"], t["gate"], t["up"], t["down"], items)
    worst, per_variant = 0.0, {}
    for name, flags in variants():
        got = cpu_variant(t["hidden"], t["gate"], t["up"], t["down"], items, flags)
        per_variant[name] = float((got - ref).abs().max())
        worst = max(worst, per_variant[name])
    return {"probe": "p17_moe_loop_variants", "mode": "selftest", "ok": bool(worst < 1e-4 and len(items) > 0), "max_abs_by_variant": per_variant}


# --- the chip -------------------------------------------------------------------------------------------------------------------------------------

def run_device(ttnn, device, tokens_list, repeat: int, only=None) -> dict:
    g = torch.Generator().manual_seed(1)
    dt = ttnn.bfloat4_b
    up = lambda a, d=ttnn.bfloat16, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(a, dtype=d, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gate_h = [(torch.randn(HIDDEN, INTER, generator=g) * HIDDEN ** -0.5).bfloat16() for _ in range(PER)]
    up_h = [(torch.randn(HIDDEN, INTER, generator=g) * HIDDEN ** -0.5).bfloat16() for _ in range(PER)]
    down_h = [(torch.randn(INTER, HIDDEN, generator=g) * INTER ** -0.5).bfloat16() for _ in range(PER)]
    w_sep = {"gate": [up(a.reshape(1, 1, HIDDEN, INTER), dt) for a in gate_h], "up": [up(a.reshape(1, 1, HIDDEN, INTER), dt) for a in up_h],
             "down": [up(a.reshape(1, 1, INTER, HIDDEN), dt) for a in down_h]}
    w_fused = [up(torch.cat([a, b], dim=1).reshape(1, 1, HIDDEN, 2 * INTER), dt) for a, b in zip(gate_h, up_h)]
    w_stack = {k: up(torch.stack(v).reshape(1, PER, *v[0].shape), dt) for k, v in (('gate', gate_h), ('up', up_h), ('down', down_h))}
    lofi = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False,
                                                  packer_l1_acc=True)
    results = {"experts_on_chip": PER, "dtype": "bfp4", "by_tokens": []}
    for tokens in tokens_list:
        x = torch.randn(tokens, HIDDEN, generator=g).bfloat16()
        x_dev = up(x)
        idx, w = p16.make_routing(tokens, TOTAL, TOP_K, tokens)
        host_items = p16.slot_inputs(idx, w, PER, tokens)
        row = {"tokens": tokens, "slots_hit": len(host_items), "variants": {}}
        baseline_out = None
        for name, flags in variants():
            if only and name not in only:
                continue
            try:
                loop = build_loop(ttnn, device, flags, host_items, tokens, x_dev, w_sep, w_fused, lofi, up, w_stack, (idx, w))
                loop()                                                       # compile
                ttnn.synchronize_device(device)
                times = []
                for _ in range(repeat):
                    t0 = time.perf_counter()
                    out = loop()
                    ttnn.synchronize_device(device)
                    times.append(time.perf_counter() - t0)
                got = ttnn.to_torch(out).float().reshape(tokens, HIDDEN)
                if name == "baseline":
                    baseline_out = got
                entry = {"ms": round(min(times) * 1e3, 3), "us_per_slot": round(min(times) * 1e6 / max(1, len(host_items)), 1)}
                if baseline_out is not None:
                    entry["pcc_vs_baseline"] = common.error_stats(got, baseline_out)["pcc"]
            except Exception as exc:  # noqa: BLE001
                entry = {"error": common.short_exc(exc)}
            row["variants"][name] = entry
        results["by_tokens"].append(row)
    return results


def build_loop(ttnn, device, flags, host_items, tokens, x_dev, w_sep, w_fused, lofi, up, w_stack, routing):
    """Upload the slot inputs for this variant and return a function that runs the loop on the chip and returns its output [1, 1, tokens, hidden]."""
    kw = {"compute_kernel_config": lofi} if "lofi" in flags else {}
    mm = lambda a, b: ttnn.matmul(a, b, **kw)
    if "bmm_all" in flags:
        return build_bmm_loop(ttnn, flags, tokens, x_dev, mm, up, w_stack, routing)
    ids_up = lambda a: up(a, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    items = []
    for j, ids, wt, back in host_items:
        rows = ids.shape[0]
        wt_dev = None if "fold_weight" in flags else up(wt.reshape(1, 1, rows, 1).bfloat16())
        back_m = back * wt[None, :] if "fold_weight" in flags else back
        items.append((j, ids, wt_dev, up(back_m.reshape(1, 1, tokens, rows).bfloat16())))
    total_rows = sum(int(i[1].shape[0]) for i in host_items)
    ids_all = ids_up(torch.cat([i[1] for i in host_items])) if "batch_scatter" in flags else None
    ids_each = [ids_up(i[1]) for i in host_items] if "batch_scatter" not in flags else None
    back_all = up(torch.cat([(b * w_[None, :] if "fold_weight" in flags else b) for _, _, w_, b in host_items], dim=1).reshape(1, 1, tokens, total_rows).bfloat16()) \
        if "batch_scatter" in flags else None

    def mlp(xe, j, rows):
        if "fuse_gate_up" in flags:
            gu = mm(xe, w_fused[j])
            gt = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, rows, INTER))
            ut = ttnn.slice(gu, (0, 0, 0, INTER), (1, 1, rows, 2 * INTER))
        else:
            gt, ut = mm(xe, w_sep["gate"][j]), mm(xe, w_sep["up"][j])
        h = ttnn.multiply(gt, ut, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]) if "fuse_silu" in flags else ttnn.multiply(ttnn.silu(gt), ut)
        return mm(h, w_sep["down"][j])

    def loop():
        if "batch_scatter" in flags:
            xe_all = ttnn.reshape(ttnn.embedding(ids_all, x_dev, layout=ttnn.TILE_LAYOUT), (1, 1, total_rows, HIDDEN))
            ys = []
            for i, (j, ids, wt_dev, _) in enumerate(items):
                rows = ids.shape[0]
                xe = ttnn.slice(xe_all, (0, 0, i * rows, 0), (1, 1, (i + 1) * rows, HIDDEN))
                ye = mlp(xe, j, rows)
                ys.append(ye if "fold_weight" in flags else ttnn.multiply(ye, wt_dev))
            return mm(back_all, ttnn.concat(ys, dim=2))
        out = None
        for i, (j, ids, wt_dev, back) in enumerate(items):
            rows = ids.shape[0]
            xe = ttnn.reshape(ttnn.embedding(ids_each[i], x_dev, layout=ttnn.TILE_LAYOUT), (1, 1, rows, HIDDEN))
            ye = mlp(xe, j, rows)
            if "fold_weight" not in flags:
                ye = ttnn.multiply(ye, wt_dev)
            part = mm(back, ye)
            out = part if out is None else ttnn.add(out, part)
        return out

    return loop


def build_bmm_loop(ttnn, flags, tokens, x_dev, mm, up, w_stack, routing):
    """Every slot in one batch: ids [PER * rows], scatter matrix [tokens, PER * rows] holding the routing weights, three batched matmuls over the slot dimension."""
    idx, w = routing
    hits = [torch.where(idx == j) for j in range(PER)]
    rows = max(32, -(-max(int(t.numel()) for t, _ in hits) // 32) * 32)
    ids = torch.zeros(PER * rows, dtype=torch.int32)
    back = torch.zeros(tokens, PER * rows)
    for j, (tok, k) in enumerate(hits):
        n = int(tok.numel())
        ids[j * rows: j * rows + n] = tok.to(torch.int32)
        back[tok, j * rows + torch.arange(n)] = w[tok, k]
    ids_dev = up(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    back_dev = up(back.reshape(1, 1, tokens, PER * rows).bfloat16())

    def loop():
        xe = ttnn.reshape(ttnn.embedding(ids_dev, x_dev, layout=ttnn.TILE_LAYOUT), (1, PER, rows, HIDDEN))
        h = ttnn.multiply(mm(xe, w_stack["gate"]), mm(xe, w_stack["up"]), input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        y = ttnn.reshape(mm(h, w_stack["down"]), (1, 1, PER * rows, HIDDEN))
        return mm(back_dev, y)

    return loop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tokens", type=int, nargs="+", default=[8, 32, 256])
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--only", nargs="+", help="variant names to run (default all)")
    args = ap.parse_args(argv)
    if args.selftest:
        r = selftest()
        common.emit(r)
        return 0 if r["ok"] else 1
    ttnn, device = common.open_device()
    try:
        res = run_device(ttnn, device, args.tokens, args.repeat, args.only)
    finally:
        ttnn.close_device(device)
    common.emit({"probe": "p17_moe_loop_variants", "results": res})
    return 0


if __name__ == "__main__":
    sys.exit(main())
