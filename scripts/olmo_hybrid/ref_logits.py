#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Teacher-forced logits of the transformers CPU reference of Olmo-Hybrid-7B, for the prompts in a reference-ids file.

    ~/orchard-venvs/reference/bin/python scripts/olmo_hybrid/ref_logits.py SNAPSHOT_DIR REFERENCE_IDS.json OUT.json

For each prompt the reference's own greedy continuation (REFERENCE_IDS.json) is fed in, and for every position the record keeps the top five
logits (token id, text, value, float32 from the bf16 model). Position k is the prediction of continuation token k, made after the prompt and
continuation tokens 0 to k-1. Reads local files only.
"""
import json
import sys


def main():
    import torch
    import transformers
    snap, ref_file, out = sys.argv[1:4]
    ref = json.load(open(ref_file))
    tok = transformers.AutoTokenizer.from_pretrained(snap)
    model = transformers.AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    items = []
    for it in ref["items"]:
        p = tok(it["prompt"], return_tensors="pt").input_ids
        cont = torch.tensor([it["ids"][:-1]], dtype=p.dtype)
        full = torch.cat([p, cont], dim=1)
        with torch.no_grad():
            logits = model(full).logits[0, p.shape[1] - 1:].float()
        pos = []
        for k in range(logits.shape[0]):
            v, i = logits[k].topk(5)
            pos.append({"k": k, "top": [{"id": int(a), "text": tok.decode([int(a)]), "logit": round(float(b), 4)} for a, b in zip(i, v)],
                        "gap_top1_top2": round(float(v[0] - v[1]), 4), "reference_choice": it["ids"][k]})
        items.append({"prompt": it["prompt"], "prompt_tokens": int(p.shape[1]), "positions": pos})
        print("done", it["prompt"][:30], flush=True)
    json.dump({"model": "allenai/Olmo-Hybrid-7B", "dtype": "bfloat16", "device": "cpu", "transformers": transformers.__version__, "items": items},
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
