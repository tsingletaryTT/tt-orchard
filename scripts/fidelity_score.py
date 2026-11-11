#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fidelity of a candidate model to a reference model, on text the reference wrote itself.

The score comes from Compression-Bench (Infini-AI-Lab, Apache-2.0, arXiv 2611.06858), where it
rates a compressed checkpoint against its BF16 original:

    fidelity = 10 * PPL_reference / PPL_candidate  ~  10 * exp(-KL)

Each block opens with a prompt, then the REFERENCE samples a continuation at temperature 1.
Only the sampled tokens are scored. On text the reference wrote, the reference's own perplexity
is the best any model can do, so the ratio measures how far the candidate's next-token
distribution has moved: 10 means "predicts like the reference", lower means worse.

What this script adds for orchard: it runs on CPU with a Hugging Face reference, and the
candidate is either another checkpoint (`--candidate`) or the reference with its weights
rounded to a lower-precision format (`--simulate`). That lets a dtype choice for a chip port be
scored before any board time. The block-float formats are approximations of the chip formats
(shared exponent per 16 values, rounded mantissa); they are not the chip's kernels.

Two numbers are reported next to the score. `kl_full` is the exact KL(reference || candidate)
over the whole vocabulary at the scored positions. `top1_agree` is how often both pick the same
argmax. Verdicts name the number of scored tokens: with few tokens the score is noisy and the
report says so (`noise_note`).

Usage (reference venv):
    ~/orchard-venvs/reference/bin/python scripts/fidelity_score.py REF --simulate int4 --json
    ~/orchard-venvs/reference/bin/python scripts/fidelity_score.py REF --candidate OTHER_DIR
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SIMULATED = ("none", "fp8", "int8", "int4", "bfp8", "bfp4")
# Below this many scored tokens a fidelity difference of a few percent is inside the noise.
MIN_TOKENS_FOR_CLAIM = 4000


def quantize_tensor(w, kind: str):
    """Round-trip a 2-D weight through a lower-precision format. Returns a tensor of the same dtype and shape."""
    import torch

    if kind == "none":
        return w
    if w.ndim != 2:
        raise ValueError(f"quantize_tensor wants a 2-D weight, got shape {tuple(w.shape)}")
    x = w.float()
    if kind == "fp8":
        return x.to(torch.float8_e4m3fn).float().to(w.dtype)
    if kind == "int8":
        scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
        return ((x / scale).round().clamp(-127, 127) * scale).to(w.dtype)
    if kind == "int4":
        return _group_int(x, 128, 7).to(w.dtype)
    if kind in ("bfp8", "bfp4"):
        return _block_float(x, 16, 7 if kind == "bfp8" else 3).to(w.dtype)
    raise ValueError(f"unknown format {kind!r}; choose from {SIMULATED}")


def _pad_cols(x, group: int):
    import torch

    n = x.shape[1]
    pad = (-n) % group
    if pad:
        x = torch.cat([x, x.new_zeros(x.shape[0], pad)], dim=1)
    return x, n


def _group_int(x, group: int, qmax: int):
    """Symmetric integer rounding with one scale per `group` values along the input dimension."""
    x, n = _pad_cols(x, group)
    g = x.reshape(x.shape[0], -1, group)
    scale = g.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / qmax
    q = (g / scale).round().clamp(-qmax, qmax) * scale
    return q.reshape(x.shape[0], -1)[:, :n]


def _block_float(x, block: int, mant_bits: int):
    """Shared power-of-two exponent per `block` values; each value keeps `mant_bits` mantissa bits plus sign."""
    import torch

    x, n = _pad_cols(x, block)
    g = x.reshape(x.shape[0], -1, block)
    amax = g.abs().amax(dim=2, keepdim=True).clamp_min(1e-30)
    exp = torch.floor(torch.log2(amax))
    step = torch.exp2(exp - (mant_bits - 1))
    lim = (2 ** mant_bits - 1) * step
    q = ((g / step).round() * step).clamp(-lim, lim)
    return q.reshape(x.shape[0], -1)[:, :n]


def quantize_model(model, kind: str) -> int:
    """Round every 2-D weight of every Linear layer (not the embeddings or the output head). Returns the count."""
    import torch

    done = 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
                mod.weight.copy_(quantize_tensor(mod.weight.data, kind))
                done += 1
    return done


def fidelity_from_logprobs(lp_ref, lp_cand, scale: float = 10.0) -> dict:
    """The score from per-token log-probabilities of the sampled tokens (lists of floats, same length)."""
    if len(lp_ref) != len(lp_cand) or not lp_ref:
        raise ValueError("need two non-empty log-probability lists of the same length")
    n = len(lp_ref)
    ppl_ref = math.exp(-sum(lp_ref) / n)
    ppl_cand = math.exp(-sum(lp_cand) / n)
    return {"tokens": n, "ppl_reference": ppl_ref, "ppl_candidate": ppl_cand, "fidelity": scale * ppl_ref / ppl_cand}


def noise_note(tokens: int) -> str | None:
    if tokens < MIN_TOKENS_FOR_CLAIM:
        return (f"only {tokens} scored tokens (< {MIN_TOKENS_FOR_CLAIM}): differences of a few percent "
                "are inside the noise; raise --blocks or --new-tokens before comparing formats")
    return None


def read_prompts(path: Path | None, tokenizer, count: int, prompt_tokens: int) -> list[list[int]]:
    """Prompt token lists: paragraphs of a text file (default: this repo's docs), at least `prompt_tokens` long."""
    files = [path] if path else sorted((Path(__file__).resolve().parent.parent / "docs").glob("*.md"))
    paras: list[str] = []
    for f in files:
        paras += [p for p in f.read_text(errors="replace").split("\n\n") if len(p.split()) > 40]
    out: list[list[int]] = []
    for p in paras:
        ids = tokenizer(p, add_special_tokens=False)["input_ids"]
        if len(ids) >= prompt_tokens:
            out.append(ids[:prompt_tokens])
        if len(out) == count:
            break
    if len(out) < count:
        raise SystemExit(f"found {len(out)} prompts of {prompt_tokens} tokens, wanted {count}; pass --prompts FILE")
    return out


def score_candidate(ref, cand, samples, prompt_lens) -> dict:
    """Score one candidate model on already-sampled sequences (each a [1, n] token tensor)."""
    import torch

    lp_ref: list[float] = []
    lp_cand: list[float] = []
    kl_sum = 0.0
    agree = 0
    with torch.no_grad():
        for full, plen in zip(samples, prompt_lens):
            a = ref(full).logits[0, plen - 1:-1].float().log_softmax(-1)
            b = cand(full).logits[0, plen - 1:-1].float().log_softmax(-1)
            tgt = full[0, plen:]
            lp_ref += a.gather(1, tgt[:, None])[:, 0].tolist()
            lp_cand += b.gather(1, tgt[:, None])[:, 0].tolist()
            kl_sum += (a.exp() * (a - b)).sum(-1).sum().item()
            agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
    res = fidelity_from_logprobs(lp_ref, lp_cand)
    n = res["tokens"]
    res.update(kl_full=kl_sum / n, top1_agree=agree / n, exp_neg_kl_x10=10 * math.exp(-kl_sum / n),
               noise_note=noise_note(n))
    return res


def run(args) -> list[dict]:
    import copy

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.reference)
    ref = AutoModelForCausalLM.from_pretrained(args.reference, torch_dtype=torch.bfloat16).eval()
    prompts = read_prompts(args.prompts, tok, args.blocks, args.prompt_tokens)
    samples = []
    with torch.no_grad():
        for ids in prompts:
            x = torch.tensor([ids])
            seq = ref.generate(x, attention_mask=torch.ones_like(x), do_sample=True, temperature=1.0, top_k=0,
                               top_p=1.0, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                               pad_token_id=tok.eos_token_id or 0)
            samples.append(seq[:, : len(ids) + args.new_tokens])
    plens = [len(p) for p in prompts]
    out = []
    if args.candidate:
        cand = AutoModelForCausalLM.from_pretrained(args.candidate, torch_dtype=torch.bfloat16).eval()
        r = score_candidate(ref, cand, samples, plens)
        r.update(candidate=str(args.candidate), format_is_simulated=False)
        out.append(r)
    else:
        for kind in args.simulate:
            cand = copy.deepcopy(ref)
            n = quantize_model(cand, kind)
            r = score_candidate(ref, cand, samples, plens)
            r.update(candidate=f"{args.reference} with Linear weights as {kind}", format=kind,
                     linear_layers_rounded=n, format_is_simulated=kind != "none")
            out.append(r)
    for r in out:
        r.update(seed=args.seed, blocks=args.blocks)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reference", help="Hugging Face model id or local directory of the reference")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--candidate", help="another checkpoint to score against the reference")
    g.add_argument("--simulate", nargs="+", choices=SIMULATED, default=["none"],
                   help="round the reference's Linear weights to each format and score each (one sample set)")
    ap.add_argument("--blocks", type=int, default=16)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--prompts", type=Path, help="text file whose paragraphs are the prompts (default: this repo's docs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    results = run(args)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(f"{r.get('format', 'candidate'):8} fidelity {r['fidelity']:.3f}  kl_full {r['kl_full']:.4f}  "
                  f"top1 {r['top1_agree']:.3f}  tokens {r['tokens']}")
            if r["noise_note"]:
                print("   note:", r["noise_note"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
