"""Per-layer agreement of a chip run with the fp32 CPU reference, with the bf16 CPU run as the rounding floor, and the
first-divergence probe (orchard/first_divergence.py) at a tolerance taken from that floor.

    python layer_compare.py RUN_DIR [RUN_DIR ...]      (each holds chip_arrays.npz from run_port.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "<ORCHARD_CHECKOUT>")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "port"))
import granite_ref as g
from orchard import first_divergence as fd
from transformers import AutoModelForCausalLM

HERE = Path(__file__).resolve().parent
ref = np.load(HERE / "arrays.npz")
n_prompt = json.load(open(HERE / "hf_vs_step_model.json"))["prompts"][0]["prompt_tokens"]
seq = ref["p0.tokens"].tolist()[:n_prompt]
L = 40

# bf16 CPU per-layer outputs (decoder layer hooks), prompt 0
bf_path = HERE / "bf16_cpu_layers.npz"
if not bf_path.exists():
    hf = AutoModelForCausalLM.from_pretrained(str(g.SNAP), dtype=torch.bfloat16).eval()
    outs = {}
    for i, layer in enumerate(hf.model.layers):
        layer.register_forward_hook(lambda m, a, o, i=i: outs.__setitem__(f"layer_{i:02d}", (o[0] if isinstance(o, tuple) else o)[0].float().numpy()))
    with torch.no_grad():
        hf(torch.tensor([seq]))
    np.savez(bf_path, **outs)
bf = np.load(bf_path)


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


ref_layers = {f"layer_{i:02d}": ref[f"p0.layer_{i:02d}"] for i in range(L)}
floor = {k: rel(bf[k], ref_layers[k]) for k in ref_layers}
tol = 1.5 * max(floor.values())
for run in sys.argv[1:]:
    chip = np.load(Path(run) / "chip_arrays.npz")
    rows = []
    for i in range(L):
        k = f"layer_{i:02d}"
        rows.append((k, g.Model.__name__ and ("attn" if i in (5, 15, 25, 35) else "mamba"), rel(chip[k], ref_layers[k]), floor[k], rel(chip[k + ".mixer"], ref[f"p0.{k}.mixer"])))
    print(f"\n{run}: per-layer relative L2 of the residual stream after each layer (prompt 0, {n_prompt} positions)")
    print("layer  kind   chip-vs-fp32  bf16cpu-vs-fp32  ratio  mixer-out chip-vs-fp32")
    for k, kind, c, f, mx in rows:
        print(f"{k}  {kind:5}  {c:.5f}       {f:.5f}        {c / f:4.2f}   {mx:.5f}")
    print(f"max ratio chip/bf16cpu: {max(r[2] / r[3] for r in rows):.2f}; tolerance used for the probe: {tol:.5f} (1.5 x worst bf16-CPU layer)")
    refrun = fd.MemoryRun({("layer", k): v for k, v in ref_layers.items()} | {("layer", k + ".mixer"): ref[f"p0.{k}.mixer"] for k in ref_layers})
    cand = fd.MemoryRun({("layer", k): chip[k] for k in ref_layers} | {("layer", k + ".mixer"): chip[k + ".mixer"] for k in ref_layers})
    rep = fd.find_first_divergence(refrun, cand, tolerance=fd.Tolerance(rel_l2=tol), cross_precision=True)
    print("\n".join(fd.summarise(rep)))
