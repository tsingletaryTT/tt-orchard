# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 16: what the per-slot gather/scatter loop costs on one chip that holds 128 experts (issue 723).

On four chips Flash-Next (512 experts per layer, top 10, hidden 2560, expert width 640) puts 128 experts of every layer on each chip. The generated port runs
the experts a token picked as a loop over the slots: per slot the host builds the token ids, the routing weights and a one-hot scatter matrix and uploads them,
then the chip gathers the rows (`ttnn.embedding`), runs the gated MLP, weights the rows and adds them back with a matmul. This probe runs that loop on one chip
with 128 expert slots in bfp4 and reports, per token count, how many slots were hit, and the wall time of three parts:

  host_prep   building the id, weight and scatter tensors for every hit slot (torch on the host)
  upload      moving those tensors to the chip
  device      the gather, the MLP, the weighting and the scatter-add for every hit slot, with the inputs already on the chip
  total       the whole loop as the generated port runs it (prep, upload and device together)

The routing is random: each token picks 10 of the 512 experts of the full layer, and this chip sees the picks that land in its 128 (experts 0 to 127). The weights
are random, so the probe times the loop and does not score quality. The `--selftest` mode checks on the CPU that the slot loop adds up to the reference.

Run on a chip only inside a gozer lease. `--selftest` checks the CPU side (the slot loop adds up to the reference) and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common  # noqa: E402

CHIPS = 4          # the layer is split over this many chips; the probe runs the share of chip 0
PRESETS = {"tiny": dict(hidden=64, inter=64, total_experts=8, top_k=2, tokens=(32,), dtype="bf16"),
           "flash": dict(hidden=2560, inter=640, total_experts=512, top_k=10, tokens=(1, 8, 32, 256), dtype="bfp4")}


def make_routing(tokens: int, total_experts: int, top_k: int, seed: int):
    """Random routing: indices [T, k] (distinct experts per token) and weights [T, k] (softmax of random scores)."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(tokens, total_experts, generator=g)
    idx = scores.topk(top_k, dim=-1).indices
    w = torch.softmax(torch.randn(tokens, top_k, generator=g), dim=-1)
    return idx, w


def slot_inputs(idx: torch.Tensor, w: torch.Tensor, per: int, tokens: int):
    """For chip 0 (experts 0 .. per-1): for each slot that some token picked, (slot, ids [rows], weight [rows], scatter [tokens, rows]); rows padded to 32."""
    out = []
    for j in range(per):
        tok, k = torch.where(idx == j)
        n = int(tok.numel())
        if n == 0:
            continue
        rows = max(32, -(-n // 32) * 32)
        ids = torch.zeros(rows, dtype=torch.int32)
        wt = torch.zeros(rows)
        back = torch.zeros(tokens, rows)
        ids[:n] = tok.to(torch.int32)
        wt[:n] = w[tok, k]
        back[tok, torch.arange(n)] = 1.0
        out.append((j, ids, wt, back))
    return out


def cpu_slot_loop(x, gate, up, down, items):
    """The loop on the CPU: the partial output of chip 0 by gathering, running the MLP, weighting and scattering per hit slot. x [T, H]; gate/up [E, I, H]; down [E, H, I]."""
    out = torch.zeros_like(x)
    for j, ids, wt, back in items:
        xe = x[ids.long()]
        ye = (torch.nn.functional.silu(xe @ gate[j].t()) * (xe @ up[j].t())) @ down[j].t() * wt[:, None]
        out += back @ ye
    return out


def cpu_reference(x, gate, up, down, idx, w, per):
    """The same partial by definition: for every token and every pick that falls on chip 0, add weight * expert(x)."""
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(idx.shape[1]):
            e = int(idx[t, k])
            if e < per:
                y = (torch.nn.functional.silu(x[t] @ gate[e].t()) * (x[t] @ up[e].t())) @ down[e].t()
                out[t] += w[t, k] * y
    return out


def selftest() -> dict:
    cfg = PRESETS["tiny"]
    per = cfg["total_experts"] // 2
    t = common.moe_inputs(tokens=32, hidden=cfg["hidden"], experts=per, top_k=cfg["top_k"], inter=cfg["inter"], shared=cfg["inter"])
    idx, w = make_routing(32, cfg["total_experts"], cfg["top_k"], 0)
    items = slot_inputs(idx, w, per, 32)
    got = cpu_slot_loop(t["hidden"], t["gate"], t["up"], t["down"], items)
    ref = cpu_reference(t["hidden"], t["gate"], t["up"], t["down"], idx, w, per)
    stats = common.error_stats(got, ref)
    return {"probe": "p16_moe_loop_cost", "mode": "selftest", "ok": bool(stats["max_abs"] < 1e-4 and len(items) > 0), "slots_hit": len(items), "stats": stats}


def untraced_times(ttnn, device, loop, dev_items, repeat):
    """Wall seconds of `loop(dev_items)` per call, synchronized after each call (the first call compiles)."""
    import time

    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        loop(dev_items)
        ttnn.synchronize_device(device)
        out.append(round(time.perf_counter() - t0, 5))
    return out


def traced_times(ttnn, device, loop, dev_items, repeat):
    """Capture `loop(dev_items)` once as a trace (after one untraced call so the programs are compiled), replay it `repeat` times and return the wall seconds
    of each replay. The inputs must already be on the chip: a trace replays the same operations on the same buffers."""
    import time

    loop(dev_items)
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    loop(dev_items)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)
    out = []
    try:
        for _ in range(repeat):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            out.append(round(time.perf_counter() - t0, 5))
    finally:
        ttnn.release_trace(device, tid)
    return out


def op_breakdown(ttnn, device, hidden, inter, dt, tokens, reps=50):
    """Wall microseconds per call of each op of one slot (32 rows), each op run `reps` times back to back and synchronized once, so launch and queue time stay in
    the number. The inputs are random; the point is which op takes the time of the 0.12 ms a slot costs."""
    import time

    g = torch.Generator().manual_seed(2)
    up = lambda a, d=ttnn.bfloat16, l=ttnn.TILE_LAYOUT: ttnn.from_torch(a, dtype=d, layout=l, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_all = up(torch.randn(tokens, hidden, generator=g).bfloat16())
    ids = up(torch.randint(0, tokens, (32,), generator=g, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    xe = ttnn.reshape(ttnn.embedding(ids, x_all, layout=ttnn.TILE_LAYOUT), (1, 1, 32, hidden))
    gate = up((torch.randn(1, 1, hidden, inter, generator=g) * hidden ** -0.5).bfloat16(), dt)
    down = up((torch.randn(1, 1, inter, hidden, generator=g) * inter ** -0.5).bfloat16(), dt)
    wt = up(torch.rand(1, 1, 32, 1, generator=g).bfloat16())
    back = up(torch.zeros(1, 1, tokens, 32).bfloat16())
    h = ttnn.matmul(xe, gate)
    hs = ttnn.silu(h)
    ye = ttnn.matmul(hs, down)
    part = ttnn.matmul(back, ye)
    ops = {"embedding": lambda: ttnn.embedding(ids, x_all, layout=ttnn.TILE_LAYOUT),
           "matmul_gate_or_up": lambda: ttnn.matmul(xe, gate), "silu": lambda: ttnn.silu(h), "multiply_gate_up": lambda: ttnn.multiply(h, h),
           "matmul_down": lambda: ttnn.matmul(hs, down), "multiply_weight": lambda: ttnn.multiply(ye, wt),
           "matmul_scatter": lambda: ttnn.matmul(back, ye), "add_partial": lambda: ttnn.add(part, part)}
    out = {}
    for name, fn in ops.items():
        fn()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        ttnn.synchronize_device(device)
        out[name] = round((time.perf_counter() - t0) / reps * 1e6, 1)
    return out


def run_device(ttnn, device, name: str, cfg: dict, repeat: int, trace: bool = False, ops: bool = False) -> dict:
    import time

    per = cfg["total_experts"] // CHIPS if name != "tiny" else cfg["total_experts"] // 2
    hidden, inter = cfg["hidden"], cfg["inter"]
    dt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[cfg["dtype"]]
    g = torch.Generator().manual_seed(1)
    up_t = lambda a: ttnn.from_torch(a.bfloat16(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gate = [up_t(torch.randn(1, 1, hidden, inter, generator=g) * hidden ** -0.5) for _ in range(per)]
    upw = [up_t(torch.randn(1, 1, hidden, inter, generator=g) * hidden ** -0.5) for _ in range(per)]
    down = [up_t(torch.randn(1, 1, inter, hidden, generator=g) * inter ** -0.5) for _ in range(per)]
    res = {"preset": name, "experts_on_chip": per, "dtype": cfg["dtype"], "by_tokens": []}
    for tokens in cfg["tokens"]:
        x = (torch.randn(tokens, hidden, generator=g)).bfloat16()
        x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        idx, w = make_routing(tokens, cfg["total_experts"], cfg["top_k"], tokens)

        def host_prep():
            return slot_inputs(idx, w, per, tokens)

        def upload(items):
            return [(j, ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
                     ttnn.from_torch(wt.reshape(1, 1, -1, 1).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
                     ttnn.from_torch(back.reshape(1, 1, tokens, -1).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)) for j, ids, wt, back in items]

        def device_loop(dev_items):
            out = None
            for j, ids, wt, back in dev_items:
                rows = ids.shape[-1]
                xe = ttnn.reshape(ttnn.embedding(ids, x_dev, layout=ttnn.TILE_LAYOUT), (1, 1, rows, hidden))
                h = ttnn.multiply(ttnn.silu(ttnn.matmul(xe, gate[j])), ttnn.matmul(xe, upw[j]))
                ye = ttnn.multiply(ttnn.matmul(h, down[j]), wt)
                part = ttnn.matmul(back, ye)
                out = part if out is None else ttnn.add(out, part)
            return out

        def timed_s(fn):
            t0 = time.perf_counter()
            r = fn()
            ttnn.synchronize_device(device)
            return r, time.perf_counter() - t0

        rows_out = {"tokens": tokens}
        host_s, up_s, dev_s, tot_s = [], [], [], []
        for _ in range(repeat):                               # the first round includes program compilation
            items, a = timed_s(host_prep)
            dev_items, b = timed_s(lambda: upload(items))
            _, c = timed_s(lambda: device_loop(dev_items))
            _, d = timed_s(lambda: device_loop(upload(host_prep())))
            host_s.append(round(a, 4)); up_s.append(round(b, 4)); dev_s.append(round(c, 4)); tot_s.append(round(d, 4))
        rows_out.update(slots_hit=len(items), rows_per_slot=[int(i[1].shape[0]) for i in items][:8], host_prep_seconds=host_s, upload_seconds=up_s,
                        device_seconds=dev_s, total_seconds=tot_s)
        if trace:
            rows_out["traced"] = {"hit_slots": traced_times(ttnn, device, device_loop, dev_items, repeat)}
            full = upload([(j, torch.zeros(32, dtype=torch.int32), torch.zeros(32), torch.zeros(tokens, 32)) for j in range(per)])
            rows_out["traced"]["all_slots_32_rows"] = {"untraced_seconds": untraced_times(ttnn, device, device_loop, full, repeat),
                                                       "traced_seconds": traced_times(ttnn, device, device_loop, full, repeat)}
        if ops:
            rows_out["op_microseconds"] = op_breakdown(ttnn, device, hidden, inter, dt, tokens)
        res["by_tokens"].append(rows_out)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--presets", nargs="+", default=["flash"], choices=sorted(PRESETS))
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--ops", action="store_true", help="also time each op of one slot (32 rows) on its own")
    ap.add_argument("--trace", action="store_true", help="also time the loop replayed as a trace (hit slots as run, and all slots at 32 rows)")
    args = ap.parse_args(argv)
    if args.selftest:
        r = selftest()
        common.emit(r)
        return 0 if r["ok"] else 1
    if args.trace:
        import ttnn
        device = ttnn.open_device(device_id=0, trace_region_size=256 * 2 ** 20)
    else:
        ttnn, device = common.open_device()
    results = []
    try:
        for name in args.presets:
            try:
                results.append(run_device(ttnn, device, name, PRESETS[name], args.repeat, args.trace, args.ops))
            except Exception as exc:  # noqa: BLE001
                results.append({"preset": name, "error": common.short_exc(exc)})
    finally:
        ttnn.close_device(device)
    common.emit({"probe": "p16_moe_loop_cost", "chips_modelled": CHIPS, "results": results})
    return 0


if __name__ == "__main__":
    sys.exit(main())
