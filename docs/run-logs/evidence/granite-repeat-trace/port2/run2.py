"""One configuration of the static/traced port: agreement on the 4 reference prompts, then decode speed.

    python run2.py NAME --ssd VARIANT [--expand repeat|mul] [--trace] [--pad] [--kernel-dir DIR] [--out DIR] [--no-agree] [--no-bench]
Agreement protocol is run_port.py's (teacher forced over the fp32 reference tokens; free-run greedy prefix); speed protocol is
bench_decode.py's (12-token prompt, 64 greedy tokens, one warm-up and five timed repeats, batch 1). Floor 175.2 tok/s.
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

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE / "port"))
import granite_ref as g  # noqa: E402  (granite_tt2 puts ../port on sys.path)
import granite_tt2 as t2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--ssd", default="composed")
ap.add_argument("--expand", default="mul")
ap.add_argument("--trace", action="store_true")
ap.add_argument("--pad", action="store_true")
ap.add_argument("--kernel-dir", default=str(BASE / "port" / "kernels"))
ap.add_argument("--out", default=str(HERE / "results"))
ap.add_argument("--no-agree", action="store_true")
ap.add_argument("--no-bench", action="store_true")
ap.add_argument("--floor", type=float, default=175.2)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ref = np.load(BASE / "reference" / "arrays.npz")
rep = json.load(open(BASE / "reference" / "hf_vs_step_model.json"))
cfg, w = g.load()
dev = ttnn.open_device(device_id=0, trace_region_size=300_000_000)
res = {"name": a.name, "ssd": a.ssd, "expand": a.expand, "trace": a.trace, "pad": a.pad}
try:
    m = t2.TTModel2(dev, cfg, w, ssd=a.ssd, expand=a.expand, kernel_dir=a.kernel_dir, pad=a.pad)
    if a.trace:
        t0 = time.time()
        m.enable_trace()
        res["trace_capture_s"] = time.time() - t0
    if not a.no_agree:
        res["prompts"] = []
        for pi in range(len(rep["prompts"])):
            seq = ref[f"p{pi}.tokens"].tolist()
            n_prompt = rep["prompts"][pi]["prompt_tokens"]
            hf = torch.from_numpy(ref[f"p{pi}.hf_logits"])
            m.reset()
            mine = []
            for pos, tid in enumerate(seq[:-1]):
                lg = m.step(tid)
                if pos >= n_prompt - 1:
                    mine.append(lg)
            mine = torch.stack(mine)
            rel = (mine - hf).norm(dim=-1) / hf.norm(dim=-1)
            top1 = (mine.argmax(-1) == hf.argmax(-1)).float().mean().item()
            top5 = float(np.mean([len(set(x.topk(5).indices.tolist()) & set(y.topk(5).indices.tolist())) / 5 for x, y in zip(mine, hf)]))
            m.reset()
            for pos in range(n_prompt):
                lg = m.step(seq[pos])
            gen = [int(lg.argmax())]
            for _ in range(len(seq) - n_prompt - 1):
                gen.append(int(m.step(gen[-1]).argmax()))
            match = 0
            for x, y in zip(gen, seq[n_prompt:]):
                if x != y:
                    break
                match += 1
            res["prompts"].append({"prompt_index": pi, "rel_l2_mean": float(rel.mean()), "rel_l2_max": float(rel.max()), "top1": top1,
                                   "top5": top5, "free_run_prefix": match, "free_run_len": len(seq) - n_prompt})
            print(json.dumps(res["prompts"][-1]), flush=True)
    if not a.no_bench:
        prompt = [48, 25, 3639, 374, 220, 1114, 3115, 220, 1419, 5380, 32, 25]
        runs = []
        for rep_i in range(6):
            m.reset()
            for t in prompt:
                lg = m.step(t)
            tok = int(lg.argmax())
            t0 = time.time()
            for _ in range(64):
                lg = m.step(tok)
                tok = int(lg.argmax())
            dt = time.time() - t0
            if rep_i > 0:
                runs.append(64 / dt)
        res.update({"tok_s_runs": runs, "mean": statistics.mean(runs), "min": min(runs), "max": max(runs),
                    "share_of_floor": statistics.mean(runs) / a.floor})
        print(json.dumps({k: res[k] for k in ("name", "mean", "min", "max", "share_of_floor")}), flush=True)
finally:
    json.dump(res, open(out / f"{a.name}.json", "w"), indent=1)
    try:
        if m.tid is not None:
            ttnn.release_trace(dev, m.tid)
    except Exception:
        pass
    ttnn.close_device(dev)
