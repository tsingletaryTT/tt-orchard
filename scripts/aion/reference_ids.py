#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Greedy token ids from the transformers CPU reference of Aion-1.0-Instruct-2.3B, for bench.py's --reference-ids.

    ~/orchard-venvs/reference/bin/python scripts/aion/reference_ids.py SNAPSHOT_DIR OUT.json

Runs in bf16 on the CPU, one prompt at a time, NEW_TOKENS new tokens each. Takes a few minutes. The model ships custom code, so it is loaded with trust_remote_code (offline: set HF_HUB_OFFLINE=1). Reads local files only.
"""
import json
import sys

PROMPTS = ["The capital of France is", "Water boils at a temperature of", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
           "In 1969, the first humans to walk on the Moon were"]
# Two chat prompts in the model's turn format (the server's chat route builds the same text). The date is fixed so the text does not change from day to day.
HEAD = "<|system|>You are Aion Instruct, a large language model trained by Microsoft.\nKnowledge cutoff: 2025-12\nCurrent date: 2026-11-22<|end|>"
PROMPTS += [HEAD + "<|user|>Explain in two sentences why the sky is blue.<|end|><|assistant|>",
            HEAD + "<|user|>Write a Python function that returns the n-th Fibonacci number.<|end|><|assistant|>"]
NEW_TOKENS = 128


def main():
    import torch
    import transformers
    snap, out = sys.argv[1], sys.argv[2]
    tok = transformers.AutoTokenizer.from_pretrained(snap, trust_remote_code=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(snap, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    items = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=NEW_TOKENS, do_sample=False, eos_token_id=None)   # no early stop: every prompt gives NEW_TOKENS tokens
        items.append({"prompt": p, "ids": gen[0, ids.shape[1]:].tolist()})
        print("done", p[:30], flush=True)
    json.dump({"model": "microsoft/Aion-1.0-Instruct-2.3B", "dtype": "bfloat16", "device": "cpu", "transformers": transformers.__version__, "items": items},
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
