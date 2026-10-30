# SPDX-License-Identifier: Apache-2.0
"""Measurements for the Granite package, on one chip, one process per call (run under a gozer lease):

  evaluate.py bench  --model-dir D --max-len L [--ssd MODE] [--no-trace] --out F     decode speed (batch 1, 1 warm-up + 5 repeats of 64 tokens
                                                                                       after a 12-token prompt) and prefill speed
  evaluate.py agree  --model-dir D --max-len L --ref REFDIR [--long LONG.npz] --out F  agreement with the CPU reference (teacher forced)

REFDIR holds `arrays.npz` and `hf_vs_step_model.json` (fp32 reference logits of 4 prompts, from the CPU reference scripts of the run).
LONG.npz holds `tokens`, `fp32` and `bf16` logits of one 1024-token natural-language sequence from transformers on the CPU.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runtime"))
from granite_chip import GraniteChip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["bench", "agree"])
ap.add_argument("--model-dir", required=True)
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--ssd", default="composed")
ap.add_argument("--kernel-dir", default=str(Path(__file__).resolve().parent.parent / "runtime" / "kernels"))
ap.add_argument("--no-trace", action="store_true")
ap.add_argument("--no-fast-prefill", action="store_true")
ap.add_argument("--skip-read-every-step", action="store_true", help="bench: leave out the slow prefill row that reads the logits after every token")
ap.add_argument("--no-chunked-prefill", action="store_true", help="prefill with one decode step per token (the older path)")
ap.add_argument("--skip-prompts", action="store_true", help="agree: only the long sequence")
ap.add_argument("--ssd-precision", default="fp32")
ap.add_argument("--ref")
ap.add_argument("--long")
ap.add_argument("--out", required=True)
ap.add_argument("--floor", type=float, default=175.2)
ap.add_argument("--prefill-sizes", default="128,512", help="prompt lengths for the prefill speed rows")
a = ap.parse_args()

dev = ttnn.open_device(device_id=0, trace_region_size=400 * 1024 * 1024)
res = {"cmd": a.cmd, "max_len": a.max_len, "ssd": a.ssd, "ssd_precision": a.ssd_precision, "traced": not a.no_trace}
try:
    t0 = time.time()
    m = GraniteChip(dev, a.model_dir, max_len=a.max_len, ssd=a.ssd, kernel_dir=a.kernel_dir, trace=not a.no_trace, ssd_precision=a.ssd_precision, fast_prefill=not a.no_fast_prefill, chunked_prefill=not a.no_chunked_prefill)
    res["load_s"] = round(time.time() - t0, 1)
    res["chunked_prefill"] = bool(m.chunked_prefill)         # what the constructed model uses for prompts of 33 or more tokens
    res["ssd_precision"] = "fp32" if m.f32 else "bf16"       # what the constructed model uses (kernel modes force bf16)
    if a.cmd == "bench":
        prompt = [48, 25, 3639, 374, 220, 1114, 3115, 220, 1419, 5380, 32, 25]
        runs = []
        for rep in range(6):
            m.reset()
            lg = m.prefill(prompt)
            tok = int(lg.argmax())
            t0 = time.time()
            for _ in range(64):
                tok = int(m.step(tok).argmax())
            dt = time.time() - t0
            if rep:
                runs.append(64 / dt)
        res["decode"] = {"tok_s_runs": runs, "mean": statistics.mean(runs), "min": min(runs), "max": max(runs), "share_of_floor": statistics.mean(runs) / a.floor}
        # first-token time after a 12-token prompt (prefill loop plus one logits read), 5 repeats
        ft = []
        for _ in range(5):
            m.reset()
            t0 = time.time()
            m.prefill(prompt)
            ft.append(time.time() - t0)
        res["first_token_s_12_token_prompt"] = {"runs": ft, "median": statistics.median(ft)}
        # prefill speed: a prompt of n tokens, the loop of decode steps with logits read back only at the end, and the same loop with a read at every step
        base = json.load(open(Path(a.model_dir) / "config.json"))
        ids_all = (np.arange(1, 4000) * 7919 % 50000 + 1000).tolist()
        res["prefill"] = []
        for n in [int(x) for x in a.prefill_sizes.split(',') if x and int(x) < a.max_len]:
            ids = ids_all[:n]
            rec = {"n": n}
            rows = (("loop_read_last", lambda: m.prefill(ids)),) + (() if a.skip_read_every_step else (("loop_read_every_step", lambda: [m.step(t) for t in ids]),))
            for label, fn in rows:
                times = []
                for rep in range(3):
                    m.reset()
                    t0 = time.time()
                    fn()
                    times.append(time.time() - t0)
                rec[label] = {"tok_s_runs": [n / t for t in times], "median_tok_s": n / statistics.median(times)}
            m.reset()
            la = m.prefill(ids)
            m.reset()
            lb = [m.step(t) for t in ids][-1]
            rec["prefill_vs_stepwise_last_logits"] = {"max_abs_diff": float((la - lb).abs().max()), "same_argmax": bool(la.argmax() == lb.argmax())}
            res["prefill"].append(rec)
    else:
        ref = np.load(Path(a.ref) / "arrays.npz")
        rep = json.load(open(Path(a.ref) / "hf_vs_step_model.json"))
        res["prompts"] = []
        for pi, p in enumerate([] if a.skip_prompts else rep["prompts"]):
            seq = ref[f"p{pi}.tokens"].tolist()
            n = p["prompt_tokens"]
            hf = torch.from_numpy(ref[f"p{pi}.hf_logits"])
            m.reset()
            mine = []
            for pos, tid in enumerate(seq[:-1]):
                lg = m.step(tid)
                if pos >= n - 1:
                    mine.append(lg)
            mine = torch.stack(mine)
            rel = (mine - hf).norm(dim=-1) / hf.norm(dim=-1)
            m.reset()
            lg = m.prefill(seq[:n])                     # the server's path: prompt by prefill, then greedy decode
            gen = [int(lg.argmax())]
            for _ in range(len(seq) - n - 1):
                gen.append(int(m.step(gen[-1]).argmax()))
            match = next((i for i, (x, y) in enumerate(zip(gen, seq[n:])) if x != y), len(gen))
            res["prompts"].append({"prompt_index": pi, "prompt_tokens": n, "rel_l2_mean": float(rel.mean()), "rel_l2_max": float(rel.max()),
                                   "top1": float((mine.argmax(-1) == hf.argmax(-1)).float().mean()), "free_run_prefix": match, "free_run_len": len(seq) - n})
            print(json.dumps(res["prompts"][-1]), flush=True)
        if a.long:
            z = np.load(a.long)
            toks = z["tokens"].tolist()
            f32, b16 = torch.from_numpy(z["fp32"]), torch.from_numpy(z["bf16"])
            m.reset()
            t0 = time.time()
            mine = torch.stack([m.step(t) for t in toks[:-1]])
            res["long"] = {"tokens": len(toks), "chip_s": time.time() - t0}
            for name, ref_lg in (("chip_vs_fp32", f32), ("bf16_cpu_vs_fp32", None)):
                x = mine if ref_lg is not None else b16
                rel = (x - f32).norm(dim=-1) / f32.norm(dim=-1)
                top1 = (x.argmax(-1) == f32.argmax(-1)).float()
                res["long"][name] = {"rel_l2_mean": float(rel.mean()), "rel_l2_max": float(rel.max()), "top1": float(top1.mean()),
                                     "top1_last_half": float(top1[len(top1) // 2:].mean()), "rel_l2_mean_last_half": float(rel[len(rel) // 2:].mean())}
            if len(toks) - 1 > 1024:                  # the first 1,024 positions of a longer run (causal, so equal to a 1,024-token run)
                first = {}
                for name, x in (("chip_vs_fp32", mine), ("bf16_cpu_vs_fp32", b16)):
                    rel = (x[:1024] - f32[:1024]).norm(dim=-1) / f32[:1024].norm(dim=-1)
                    first[name] = {"rel_l2_mean": float(rel.mean()), "rel_l2_max": float(rel.max()), "top1": float((x[:1024].argmax(-1) == f32[:1024].argmax(-1)).float().mean())}
                res["long"]["first_1024_positions"] = first
            relc = ((mine - f32).norm(dim=-1) / f32.norm(dim=-1)).tolist()
            relb = ((b16 - f32).norm(dim=-1) / f32.norm(dim=-1)).tolist()
            bins = range(0, len(relc), 128)
            res["long"]["rel_l2_by_128_positions"] = {"chip": [sum(relc[i:i + 128]) / len(relc[i:i + 128]) for i in bins],
                                                      "bf16_cpu": [sum(relb[i:i + 128]) / len(relb[i:i + 128]) for i in bins]}
            res["long"]["chip_vs_bf16_cpu_top1"] = float((mine.argmax(-1) == b16.argmax(-1)).float().mean())
            print(json.dumps(res["long"]), flush=True)
finally:
    json.dump(res, open(a.out, "w"), indent=1)
    try:
        m.close()
    except Exception:
        pass
    ttnn.close_device(dev)
