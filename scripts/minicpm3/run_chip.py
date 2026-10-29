# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run MiniCPM3-4B on a chip (under a gozer lease) and record the run for the first-divergence probe, or time decode.

    python run_chip.py agree --attn mla --ref REFDIR/float32 --out OUTDIR      # teacher-forced agreement run, recorded
    python run_chip.py bench --attn mla --steps 64 --repeats 3                 # free-running greedy decode timing
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import ttnn  # noqa: E402
from chip_model import ChipMiniCPM3  # noqa: E402
from make_reference import PROMPT, make_recorder, prompt_ids  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["agree", "bench"])
    ap.add_argument("--model-dir", default="/home/ttuser/.cache/orchard-overnight2/minicpm3")
    ap.add_argument("--weights", default="/home/ttuser/.cache/orchard-overnight2/minicpm3.safetensors")
    ap.add_argument("--attn", default="mla", choices=["mla", "host"])
    ap.add_argument("--ref", help="a recorded reference run directory (float32) for the token path")
    ap.add_argument("--out", help="new directory for the recorded chip run")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fidelity", default="default", choices=["default", "hifi4"])
    ap.add_argument("--upd", default="int", choices=["int", "paged", "onehot"], help="cache write; a trace needs onehot or paged")
    ap.add_argument("--trace", action="store_true", help="bench: capture and replay the step as a trace")
    ap.add_argument("--skip", default="", help="timing only: comma list of mlp, lm")
    ap.add_argument("--layers", type=int, default=0, help="debug: truncate the stack (0 = all)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    ids = prompt_ids(tok)[0].tolist()
    n = len(ids)
    dev = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
    try:
        t0 = time.time()
        m = ChipMiniCPM3(dev, a.model_dir, a.weights, attn=a.attn, max_len=256, fidelity=a.fidelity, upd=a.upd)
        m.skip = set(x for x in a.skip.split(",") if x)
        print(f"model loaded on chip in {time.time() - t0:.1f}s", flush=True)
        if a.mode == "agree":
            agree(a, m, ids, n)
        else:
            bench(a, m, ids, n, tok)
    finally:
        ttnn.close_device(dev)


def agree(a, m, ids, n):
    from orchard import audio_reference as ar
    ref = ar.load_run(a.ref)
    toks = [int(v) for v in ref.array("other", "tokens")]
    rec = make_recorder(Path(a.out), "chip", "bfloat16", n, len(toks), f"{a.model_dir}/config.json", Path(a.weights))
    rec.settings["device"] = "blackhole p300 chip 0, " + a.attn + " attention"
    streams = {}
    state = {"phase": "prefill", "k": -1}
    t0 = time.time()

    def tap(name, t):
        if name == "embed":
            return
        streams.setdefault((state["phase"], state["k"], name), []).append(t.numpy())

    logits = None
    for p, t in enumerate(ids):
        logits = m.run(t, p, tap)
    print(f"prefill {n} tokens in {time.time() - t0:.1f}s (first call includes kernel compile)", flush=True)
    for name in sorted({k[2] for k in streams}):
        rec.record("layer", name, np.stack(streams[("prefill", -1, name)]))
    rec.record("other", "prefill_logits", logits.numpy())
    matches = int(logits.argmax().item() == toks[0])
    free_ok = True
    nxt_free = int(logits.argmax().item())
    for k, t in enumerate(toks):
        state["phase"], state["k"] = "dec", k
        lg = m.run(t, n + k, tap if k == 0 else None)
        rec.record("other", f"dec_logits_{k}", lg.numpy())
        if k + 1 < len(toks):
            matches += int(lg.argmax().item() == toks[k + 1])
        if k == 0:
            for name in sorted({kk[2] for kk in streams if kk[0] == "dec"}):
                if "." not in name:
                    rec.record("other", "dec0_" + name, streams[("dec", 0, name)][0])
    rec.record("other", "tokens", np.array(toks, dtype=np.float32))
    rec.finish()
    print(f"teacher-forced next-token matches: {matches}/{len(toks)}", flush=True)


def bench(a, m, ids, n, tok):
    results = []
    step = m.run
    if a.trace:
        t0 = time.time()
        m.enable_trace()
        print(f"trace captured in {time.time() - t0:.1f}s", flush=True)
        step = lambda t, p, tap=None: m.traced_step(t, p)  # noqa: E731
    for r in range(a.repeats + 1):            # repeat 0 is a warm-up and is reported apart
        if not a.trace:
            m.reset()          # a trace is bound to its cache buffers; stale rows past the position are masked by the op
        logits = None
        for p, t in enumerate(ids):
            logits = step(t, p)
        nxt = int(logits.argmax().item())
        out = []
        t0 = time.time()
        for k in range(a.steps):
            out.append(nxt)
            lg = step(nxt, n + k)
            nxt = int(lg.argmax().item())
        dt = time.time() - t0
        results.append({"repeat": r, "warmup": r == 0, "steps": a.steps, "seconds": dt, "tok_s": a.steps / dt})
        print(json.dumps(results[-1]), flush=True)
    print(tok.decode(out))


if __name__ == "__main__":
    main()
