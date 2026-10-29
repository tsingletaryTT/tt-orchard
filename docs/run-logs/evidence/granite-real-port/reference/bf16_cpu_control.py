"""Control for the free-running comparison: the same model in bf16 on the CPU (transformers, greedy) against the fp32 tokens.
A chip run that diverges from fp32 where this control also diverges is showing bf16 rounding, not a port fault."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "port"))
import granite_ref as g
from transformers import AutoModelForCausalLM

torch.set_num_threads(6)
ref = np.load(Path(__file__).resolve().parent / "arrays.npz")
rep = json.load(open(Path(__file__).resolve().parent / "hf_vs_step_model.json"))
hf = AutoModelForCausalLM.from_pretrained(str(g.SNAP), dtype=torch.bfloat16).eval()
res = []
for pi, p in enumerate(rep["prompts"]):
    seq, n = ref[f"p{pi}.tokens"].tolist(), p["prompt_tokens"]
    with torch.no_grad():
        out = hf.generate(torch.tensor([seq[:n]]), max_new_tokens=len(seq) - n, do_sample=False, output_scores=True, return_dict_in_generate=True)
    gen = out.sequences[0, n:].tolist()
    match = next((i for i, (a, b) in enumerate(zip(gen, seq[n:])) if a != b), len(seq) - n)
    lg = torch.stack([s[0] for s in out.scores]).float()
    ref_lg = torch.from_numpy(ref[f"p{pi}.hf_logits"])[: lg.shape[0]]
    # teacher-forced logits for the same positions
    with torch.no_grad():
        tf = hf(torch.tensor([seq[:-1]])).logits[0, n - 1:].float()
    rel = ((tf - ref_lg).norm(dim=-1) / ref_lg.norm(dim=-1))
    res.append({"prompt_index": pi, "free_run_match_prefix": match, "free_run_len": len(seq) - n,
                "teacher_forced_rel_l2_mean": float(rel.mean()), "teacher_forced_rel_l2_max": float(rel.max()),
                "top1_agree": float((tf.argmax(-1) == ref_lg.argmax(-1)).float().mean())})
    print(json.dumps(res[-1]), flush=True)
json.dump(res, open(Path(__file__).resolve().parent / "bf16_cpu_control.json", "w"), indent=1)
