# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Free-running greedy generation on several prompts: CPU float32 reference against the chip, token by token.

    python qual_check.py ref --out ref_text.json                    # CPU float32, plain-torch reference
    python qual_check.py chip --ref ref_text.json --out chip_text.json [chip options]   # under a gozer lease

For each prompt the chip text is compared with the reference: the index of the first differing token (None when all tokens match) and both texts. A greedy run on the
chip is expected to leave the reference at some token (near ties); what matters is that the text stays on topic and fluent, which a reader checks in the output.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROMPTS = [
    "What causes the seasons on Earth? Answer in three sentences.",
    "Write a Python function that returns the n-th Fibonacci number using iteration, with a docstring.",
    "A train leaves a station at 3 pm travelling at 60 km/h. A second train leaves the same station at 4 pm at 90 km/h on the same track. When does the second catch up? Show the steps.",
    "Translate into French: 'The library closes at eight, so please return your books before then.'",
    "Summarize in two sentences: Tidal power uses the rise and fall of the sea to turn turbines. Unlike wind and solar, it is predictable, because tides follow the moon. Its drawbacks are high construction cost and effects on local ecosystems, which is why few plants exist.",
    "Give me a short, vivid description of a market at dawn.",
]


def ids_for(tok, text):
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt")
    return ids if torch.is_tensor(ids) else ids["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "chip"])
    ap.add_argument("--model-dir", default=os.environ.get("MINICPM3_DIR"), required="MINICPM3_DIR" not in os.environ)
    ap.add_argument("--weights", default=os.environ.get("MINICPM3_WEIGHTS"), required="MINICPM3_WEIGHTS" not in os.environ)
    ap.add_argument("--ref", help="chip mode: the JSON the ref mode wrote")
    ap.add_argument("--out", required=True)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--opt", type=int, default=3)
    ap.add_argument("--mlp-dtype", default="bf16")
    ap.add_argument("--mlp-fidelity", default="hifi2")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    results = []
    if a.mode == "ref":
        from ref_model import RefModel
        torch.set_num_threads(8)
        m = RefModel(a.model_dir, a.weights, dtype=torch.float32)
        for text in PROMPTS:
            m.reset()
            ids = ids_for(tok, text)
            lg = m.forward(ids)
            out = []
            nxt = int(lg[0, -1].argmax())
            for k in range(a.new):
                out.append(nxt)
                if nxt in (2, 73440):
                    break
                lg = m.forward(torch.tensor([[nxt]]), ids.shape[1] + k)
                nxt = int(lg[0, -1].argmax())
            results.append({"prompt": text, "tokens": out, "text": tok.decode(out)})
            print(text[:50], "->", tok.decode(out)[:100].replace("\n", " "), flush=True)
    else:
        import ttnn
        from chip_model import ChipMiniCPM3
        ref = json.load(open(a.ref))
        dev = ttnn.open_device(device_id=0, trace_region_size=256 * 1024 * 1024)
        try:
            m = ChipMiniCPM3(dev, a.model_dir, a.weights, attn="mla", max_len=512, upd="paged", opt=a.opt, mlp_dtype=a.mlp_dtype,
                             mlp_fidelity=a.mlp_fidelity)
            m.enable_trace()
            for r in ref:
                ids = ids_for(tok, r["prompt"])[0].tolist()
                t0 = time.time()
                for p, t in enumerate(ids):
                    lg = m.traced_step(t, p)
                out = []
                nxt = int(lg.argmax())
                for k in range(len(r["tokens"])):
                    out.append(nxt)
                    if nxt in (2, 73440):
                        break
                    lg = m.traced_step(nxt, len(ids) + k)
                    nxt = int(lg.argmax())
                first = next((i for i, (x, y) in enumerate(zip(out, r["tokens"])) if x != y), None)
                if first is None and len(out) != len(r["tokens"]):
                    first = min(len(out), len(r["tokens"]))
                results.append({"prompt": r["prompt"], "chip_text": tok.decode(out), "ref_text": r["text"], "tokens": len(out), "first_difference": first,
                                "seconds": round(time.time() - t0, 2)})
                print(f"first difference at {first} of {len(r['tokens'])}: {tok.decode(out)[:100]!r}", flush=True)
        finally:
            ttnn.close_device(dev)
    Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
