#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""How unevenly a mixture-of-experts model uses its experts on real text (issue 723 item 3, issue 739).

Runs a Hugging Face MoE reference on the CPU over some text, takes each layer's router logits, and counts how often each expert is among the
`top_k` picked for a token. The counts decide two things the chip port needs:

  * expert placement (orchard/expert_placement.py `load=`): the hottest experts go to different chips;
  * tiering (issue 739): if a small share of the experts takes most of the picks, those can stay on chip and the rest can live elsewhere.

The report gives, per layer and as a mean over layers, the share of picks that go to the busiest 10, 25 and 50 percent of the experts, the share of
experts that were never picked, and the entropy of the pick distribution against its maximum. A uniform router scores 10, 25 and 50 percent.
Counts are only as representative as the text: the script says how many tokens it saw, and `--json` keeps the per-expert counts so a placement can use them.

Usage (reference venv):
    ~/orchard-venvs/reference/bin/python scripts/expert_skew.py MODEL_DIR --tokens 16384 --json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SHARES = (0.10, 0.25, 0.50)


def skew_from_counts(counts: list[int]) -> dict:
    """Skew figures for one layer from its per-expert pick counts."""
    n = len(counts)
    total = sum(counts)
    if n == 0 or total == 0:
        raise ValueError("need at least one expert and one pick")
    ordered = sorted(counts, reverse=True)
    out = {"experts": n, "picks": total, "never_picked": sum(1 for c in counts if c == 0) / n}
    for s in SHARES:
        k = max(1, math.ceil(n * s - 1e-9))          # the busiest 25% of 10 experts is 3 experts
        out[f"top_{int(s * 100)}pct_share"] = sum(ordered[:k]) / total
    p = [c / total for c in counts if c]
    out["entropy_vs_max"] = -sum(x * math.log(x) for x in p) / math.log(n) if n > 1 else 1.0
    return out


def mean_over_layers(layers: list[dict]) -> dict:
    keys = [k for k in layers[0] if k not in ("experts", "picks")]
    return {k: sum(l[k] for l in layers) / len(layers) for k in keys}


def read_text_chunks(path: Path | None, tokenizer, tokens: int, chunk: int) -> list[list[int]]:
    files = [path] if path else sorted((Path(__file__).resolve().parent.parent / "docs").glob("*.md"))
    ids: list[int] = []
    for f in files:
        ids += tokenizer(f.read_text(errors="replace"), add_special_tokens=False)["input_ids"]
        if len(ids) >= tokens:
            break
    ids = ids[:tokens]
    if len(ids) < chunk:
        raise SystemExit(f"only {len(ids)} tokens of text; pass --text FILE")
    return [ids[i:i + chunk] for i in range(0, len(ids) - chunk + 1, chunk)]


def run(args) -> dict:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.model)
    top_k = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "top_k", None)
    n_exp = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None) or getattr(cfg, "n_routed_experts", None)
    if not top_k or not n_exp:
        raise SystemExit("this config does not name an expert count and a top-k; is it a mixture-of-experts model?")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).eval()
    chunks = read_text_chunks(args.text, tok, args.tokens, args.chunk)
    counts: list[list[int]] | None = None
    seen = 0
    with torch.no_grad():
        for c in chunks:
            out = model(torch.tensor([c]), output_router_logits=True)
            logits = out.router_logits                                  # one [tokens, experts] tensor per MoE layer
            if counts is None:
                counts = [[0] * logits[0].shape[-1] for _ in logits]
            for li, lg in enumerate(logits):
                picked = lg.float().topk(top_k, dim=-1).indices.reshape(-1).tolist()
                for e in picked:
                    counts[li][e] += 1
            seen += len(c)
    layers = [skew_from_counts(c) for c in counts]
    return {"model": str(args.model), "tokens": seen, "experts": len(counts[0]), "top_k": top_k, "moe_layers": len(counts),
            "mean": mean_over_layers(layers), "layers": layers, "counts": counts if args.json else None,
            "note": "tokens counted over the text given; a router's skew on other text can differ"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="local directory or Hugging Face id of a mixture-of-experts reference")
    ap.add_argument("--text", type=Path, help="text file (default: this repo's docs)")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    r = run(args)
    if args.json:
        print(json.dumps(r))
    else:
        m = r["mean"]
        print(f"{r['model']}: {r['moe_layers']} MoE layers, {r['experts']} experts, top-{r['top_k']}, {r['tokens']} tokens")
        print(f"  busiest 10% of experts take {m['top_10pct_share']:.1%} of picks, busiest 25% take {m['top_25pct_share']:.1%}, busiest 50% take {m['top_50pct_share']:.1%}")
        print(f"  never picked: {m['never_picked']:.1%}   entropy vs uniform: {m['entropy_vs_max']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
