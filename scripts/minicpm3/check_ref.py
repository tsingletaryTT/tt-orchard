# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run the plain-torch reference on a prompt: print greedy text, and (with --remote) compare with the transformers-native MiniCPM3 class (no remote code is run; the native class reads <model-dir>_w, a copy of config.json plus model.safetensors)."""
import argparse
import sys
import time

import torch
from transformers import AutoTokenizer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ref_model import RefModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--weights", required=True)
ap.add_argument("--native", action="store_true")
ap.add_argument("--dtype", default="float32")
ap.add_argument("--new", type=int, default=24)
a = ap.parse_args()
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(a.model_dir)
msgs = [{"role": "user", "content": "Explain in two sentences why the sky is blue."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
if not torch.is_tensor(ids):
    ids = ids["input_ids"]
print("prompt tokens", ids.shape)
m = RefModel(a.model_dir, a.weights, dtype=getattr(torch, a.dtype))
t0 = time.time()
lg = m.forward(ids)
out = []
nxt = lg[0, -1].argmax().item()
for _ in range(a.new):
    out.append(nxt)
    lg = m.forward(torch.tensor([[nxt]]), start=ids.shape[1] + len(out) - 1)
    nxt = lg[0, -1].argmax().item()
print("ref", out)
print(tok.decode(out), f"({time.time() - t0:.1f}s)")
if a.native:
    from transformers import AutoModelForCausalLM
    rm = AutoModelForCausalLM.from_pretrained(a.model_dir.rstrip("/") + "_w", trust_remote_code=False, torch_dtype=getattr(torch, a.dtype))
    with torch.no_grad():
        g = rm.generate(ids, max_new_tokens=a.new, do_sample=False)
    r = g[0, ids.shape[1]:].tolist()
    print("transformers-native", r)
    print("same tokens:", r[: len(out)] == out)
