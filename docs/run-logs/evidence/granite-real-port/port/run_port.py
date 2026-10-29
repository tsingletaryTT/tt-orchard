"""Run the ttnn port and compare it with the CPU reference (reference/arrays.npz, hf_vs_step_model.json).

    python run_port.py MODE [--out DIR] [--prompts N] [--kernels MODULE]

Per prompt: teacher-forced pass over the reference's tokens (logit agreement against transformers' fp32 logits, top-1 and top-5),
free-running greedy decode of the same length (tokens equal to the reference, tokens per second). For prompt 0 the per-layer
hidden states (layer_NN, layer_NN.mixer) are recorded at every prompt position and written to DIR/chip_arrays.npz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

BASE = Path("<RUN_DIR>")
sys.path.insert(0, str(BASE / "port"))
import granite_ref as g
import granite_tt as gt

ap = argparse.ArgumentParser()
ap.add_argument("mode")
ap.add_argument("--out", default=None)
ap.add_argument("--prompts", type=int, default=4)
ap.add_argument("--kernels", default=None, help="module:function that builds the kernel callable (mode kernel)")
ap.add_argument("--layers", type=int, default=None, help="stop the layer comparison report at N (debug)")
args = ap.parse_args()
out = Path(args.out or BASE / "port" / f"run-{args.mode}")
out.mkdir(parents=True, exist_ok=True)

ref = np.load(BASE / "reference" / "arrays.npz")
rep = json.load(open(BASE / "reference" / "hf_vs_step_model.json"))
cfg, w = g.load()
dev = ttnn.open_device(device_id=0)
result = {"mode": args.mode, "prompts": []}
try:
    kern = None
    if args.kernels:
        import importlib
        mod, fn = args.kernels.split(":")
        kern = getattr(importlib.import_module(mod), fn)(dev)
    m = gt.TTModel(dev, cfg, w, ssd_mode=args.mode, kernels=kern)
    for pi in range(min(args.prompts, len(rep["prompts"]))):
        seq = ref[f"p{pi}.tokens"].tolist()
        n_prompt = rep["prompts"][pi]["prompt_tokens"]
        hf_logits = torch.from_numpy(ref[f"p{pi}.hf_logits"])
        # teacher forced
        m.reset()
        rec_all, mine = [], []
        t0 = time.time()
        for pos, tid in enumerate(seq[:-1]):
            rec = {} if (pi == 0 and pos < n_prompt) else None
            lg = m.step(tid, record=rec)
            if rec is not None:
                rec_all.append(rec)
            if pos >= n_prompt - 1:
                mine.append(lg)
        t_tf = time.time() - t0
        mine = torch.stack(mine)
        rel = ((mine - hf_logits).norm(dim=-1) / hf_logits.norm(dim=-1))
        top1 = (mine.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
        top5 = np.mean([len(set(a.topk(5).indices.tolist()) & set(b.topk(5).indices.tolist())) / 5 for a, b in zip(mine, hf_logits)])
        if pi == 0:
            arrs = {k: np.stack([r[k] for r in rec_all]) for k in rec_all[0]}
            np.savez(out / "chip_arrays.npz", **arrs)
        # free running
        m.reset()
        t0 = time.time()
        gen, tok = [], None
        for pos in range(n_prompt):
            lg = m.step(seq[pos])
        t_prefill = time.time() - t0
        t1 = time.time()
        gen = [int(lg.argmax())]
        for _ in range(len(seq) - n_prompt - 1):
            lg = m.step(gen[-1])
            gen.append(int(lg.argmax()))
        t_dec = time.time() - t1
        refgen = seq[n_prompt:]
        match = 0
        for a, b in zip(gen, refgen):
            if a != b:
                break
            match += 1
        e = {"prompt_index": pi, "prompt_tokens": n_prompt, "teacher_forced_rel_l2_mean": float(rel.mean()), "teacher_forced_rel_l2_max": float(rel.max()),
             "top1_agree": top1, "top5_overlap": float(top5), "free_run_match_prefix": match, "free_run_len": len(refgen),
             "decode_tok_s": (len(gen) - 1) / t_dec, "prefill_tok_s": n_prompt / t_prefill, "teacher_forced_s": t_tf}
        result["prompts"].append(e)
        print(json.dumps(e), flush=True)
    json.dump(result, open(out / "result.json", "w"), indent=1)
finally:
    ttnn.close_device(dev)
