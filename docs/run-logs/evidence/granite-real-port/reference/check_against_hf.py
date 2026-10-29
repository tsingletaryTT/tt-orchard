"""Step (a): the CPU reference. Runs transformers' GraniteMoeHybridForCausalLM (fp32, CPU) and the hand-written step model
(port/granite_ref.py) on the same prompts, and writes recorded arrays for the later comparisons.

Outputs (reference/): prompts.json, hf_gen.json (greedy tokens and top-5 per step for each prompt), and arrays.npz for
prompt 0: per-layer hidden states at every prompt position (layer_NN, layer_NN.mixer), final logits, and SSD op inputs/outputs
for two mamba layers at one decode position.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "port"))
import granite_ref as g

torch.set_num_threads(6)
OUT = Path(__file__).resolve().parent
PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the n-th Fibonacci number.\n\ndef fib(n):",
    "State space models differ from transformers because",
    "Q: What is 17 times 23?\nA:",
]
NEW = 24

from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(str(g.SNAP))
hf = AutoModelForCausalLM.from_pretrained(str(g.SNAP), dtype=torch.float32).eval()
cfg, w = g.load()
mine = g.Model(cfg, w)

report = {"snapshot": str(g.SNAP.name), "torch": torch.__version__, "prompts": []}
arrays = {}
for pi, text in enumerate(PROMPTS):
    ids = tok(text, return_tensors="pt").input_ids
    t0 = time.time()
    with torch.no_grad():
        out = hf.generate(ids, max_new_tokens=NEW, do_sample=False, output_scores=True, return_dict_in_generate=True)
    seq = out.sequences[0].tolist()
    hf_logits = torch.stack([s[0] for s in out.scores])                      # [NEW, V]: logits after the prompt and after each new token
    t_hf = time.time() - t0
    # my step model: feed the prompt, then the HF-chosen tokens (teacher forcing), compare logits at each position
    mine.reset()
    rec_all = []
    my_logits = []
    t0 = time.time()
    n_prompt = ids.shape[1]
    for pos, tid in enumerate(seq[:-1]):
        rec = {} if (pi == 0 and pos < n_prompt) else None
        lg = mine.step(tid, record=rec)
        if pos >= n_prompt - 1:
            my_logits.append(lg)
        if rec is not None:
            rec_all.append(rec)
        if pi == 0 and pos == n_prompt - 1:
            pass
    my_logits = torch.stack(my_logits)
    t_my = time.time() - t0
    err = (my_logits - hf_logits).abs()
    rel = (my_logits - hf_logits).norm(dim=-1) / hf_logits.norm(dim=-1)
    agree = (my_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean().item()
    entry = {"prompt": text, "prompt_tokens": n_prompt, "tokens": seq, "text": tok.decode(seq[n_prompt:]),
             "max_abs_logit_err": err.max().item(), "max_rel_l2": rel.max().item(), "argmax_agree": agree,
             "hf_seconds": round(t_hf, 1), "step_model_seconds": round(t_my, 1),
             "top5_ids_step0": hf_logits[0].topk(5).indices.tolist(), "top5_vals_step0": [round(v, 4) for v in hf_logits[0].topk(5).values.tolist()]}
    report["prompts"].append(entry)
    print(json.dumps(entry)[:600], flush=True)
    arrays[f"p{pi}.tokens"] = np.array(seq)
    arrays[f"p{pi}.hf_logits"] = hf_logits.numpy()
    if pi == 0:
        names = [k for k in rec_all[0] if not k.startswith("ssd.")]
        for k in names:
            arrays[f"p0.{k}"] = torch.stack([r[k] for r in rec_all]).numpy()           # [T, hidden]
        # SSD op recordings at the last prompt position, two layers (a first one and a middle one)
        for li in (0, 17):
            r = rec_all[-1][f"ssd.layer_{li:02d}"]
            for part in ("ssd_in", "ssd_out"):
                for k, v in r[part].items():
                    arrays[f"ssd.layer_{li:02d}.{part}.{k}"] = v.numpy()
json.dump(report, open(OUT / "hf_vs_step_model.json", "w"), indent=1)
np.savez(OUT / "arrays.npz", **arrays)
print("done; arrays:", len(arrays))
