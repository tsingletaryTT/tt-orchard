"""Decode speed of one port mode: 1 warm-up and 5 timed repeats of 64 greedy tokens after a 12-token prompt (batch 1, no trace).

    python bench_decode.py MODE [--kernels module:build] --out FILE
Writes the per-repeat tok/s, their mean, min and max, and the share of the one-die bf16 decode floor.
"""
import argparse
import json
import statistics
import sys
import time

import ttnn

sys.path.insert(0, "<run-dir>/port")
import granite_ref as g
import granite_tt as gt

ap = argparse.ArgumentParser()
ap.add_argument("mode")
ap.add_argument("--kernels")
ap.add_argument("--out", required=True)
ap.add_argument("--floor", type=float, default=175.2, help="one-die bf16 floor in tok/s (decode_floor, p300 die 512 GB/s)")
a = ap.parse_args()
cfg, w = g.load()
dev = ttnn.open_device(device_id=0)
try:
    kern = None
    if a.kernels:
        import importlib
        mod, fn = a.kernels.split(":")
        kern = getattr(importlib.import_module(mod), fn)(dev)
    m = gt.TTModel(dev, cfg, w, ssd_mode=a.mode, kernels=kern)
    prompt = [48, 25, 3639, 374, 220, 1114, 3115, 220, 1419, 5380, 32, 25]
    runs = []
    for rep in range(6):
        m.reset()
        for t in prompt:
            lg = m.step(t)
        tok = int(lg.argmax())
        t0 = time.time()
        for _ in range(64):
            lg = m.step(tok)
            tok = int(lg.argmax())
        dt = time.time() - t0
        if rep > 0:
            runs.append(64 / dt)
    res = {"mode": a.mode, "tok_s_runs": runs, "mean": statistics.mean(runs), "min": min(runs), "max": max(runs),
           "stdev": statistics.stdev(runs), "floor_tok_s_one_die_bf16": a.floor, "share_of_floor": statistics.mean(runs) / a.floor}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))
finally:
    ttnn.close_device(dev)
