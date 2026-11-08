#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Greedy token ids from the transformers CPU reference of Olmo-Hybrid-7B, for bench.py's --reference-ids.

    ~/orchard-venvs/reference/bin/python scripts/olmo_hybrid/reference_ids.py SNAPSHOT_DIR OUT.json

Runs in bf16 on the CPU, one prompt at a time, NEW_TOKENS new tokens each. Takes a few minutes. Reads local files only.
"""
import json
import sys

PROMPTS = ["The capital of France is", "Water boils at a temperature of", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "In 1969, the first humans to walk on the Moon were"]
NEW_TOKENS = 32


def main():
    import torch
    import transformers
    snap, out = sys.argv[1], sys.argv[2]
    tok = transformers.AutoTokenizer.from_pretrained(snap)
    model = transformers.AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    items = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False)
        items.append({"prompt": p, "ids": gen[0, ids.shape[1]:].tolist()})
        print("done", p[:30], flush=True)
    json.dump({"model": "allenai/Olmo-Hybrid-7B", "dtype": "bfloat16", "device": "cpu", "transformers": transformers.__version__, "items": items},
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
