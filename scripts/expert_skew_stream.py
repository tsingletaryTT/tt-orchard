#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Expert-use skew of a Qwen3-Next style mixture-of-experts model that does not fit in host RAM (issue 739 step 2).

scripts/expert_skew.py loads the whole reference model. A model with 512 experts per layer is too large for that on most hosts (Qwen3-Coder-Next
is about 150 GiB in bfloat16). This script runs the same measurement one decoder layer at a time: it reads one layer's tensors from the
checkpoint, runs every text chunk through that layer, counts which experts the router picked, then frees the layer and reads the next one.
Only one layer's weights (about 3.5 GiB for Qwen3-Coder-Next) and the hidden states of the text (tokens x hidden size) are in memory.

The numbers match expert_skew.py: it counts how often each expert is among the `top_k` picked for a token, and reports per layer and as a mean
over layers the share of picks that go to the busiest 10, 25 and 50 percent of the experts, the share of experts never picked, and the entropy
of the pick distribution against its maximum. The JSON has the same keys, so `tt-orchard tier-plan --skew-file` reads either one. A test runs both
scripts on a tiny random checkpoint and requires equal counts.

Chunks are independent sequences (no state carried from one chunk to the next), as in expert_skew.py. Several text sources can share one pass over
the checkpoint: `--text NAME=FILE` may be repeated, and the output has one result per source.

Usage:
    python3 scripts/expert_skew_stream.py MODEL_DIR --tokens 16384 --chunk 2048 --text docs=docs/foo.md --text code=orchard/foo.py --out skew.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from expert_skew import mean_over_layers, skew_from_counts  # noqa: E402  (same figures as the whole-model script)

EXPERT_KEY = "mlp.experts."


def read_ids(files: list[Path], tokenizer, tokens: int) -> list[int]:
    """The first `tokens` token ids of the files' text, read in the order given."""
    ids: list[int] = []
    for f in files:
        ids += tokenizer(f.read_text(errors="replace"), add_special_tokens=False)["input_ids"]
        if len(ids) >= tokens:
            break
    return ids[:tokens]


def to_chunks(ids: list[int], chunk: int) -> list[list[int]]:
    """Whole chunks only: a short tail is dropped, as expert_skew.read_text_chunks does."""
    if len(ids) < chunk:
        raise ValueError(f"only {len(ids)} tokens of text for a chunk of {chunk}")
    return [ids[i:i + chunk] for i in range(0, len(ids) - chunk + 1, chunk)]


class Checkpoint:
    """The tensors of a Hugging Face safetensors checkpoint, read by name from whichever shard holds them."""

    def __init__(self, model_dir: Path):
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            self.weight_map = {}
            single = model_dir / "model.safetensors"
            from safetensors import safe_open
            with safe_open(str(single), framework="pt") as f:
                self.weight_map = {k: single.name for k in f.keys()}
        self.dir = model_dir

    def layer_keys(self, layer: int) -> list[str]:
        prefix = f"model.layers.{layer}."
        return [k for k in self.weight_map if k.startswith(prefix)]

    def read(self, keys: list[str]) -> dict:
        """Read the keys, opening each shard once."""
        from safetensors import safe_open
        by_shard: dict[str, list[str]] = {}
        for k in keys:
            by_shard.setdefault(self.weight_map[k], []).append(k)
        out = {}
        for shard, names in by_shard.items():
            with safe_open(str(self.dir / shard), framework="pt") as f:
                for k in names:
                    out[k] = f.get_tensor(k)
        return out


def load_layer(cfg, ckpt: Checkpoint, layer: int, dtype):
    """One decoder layer with its weights from the checkpoint. Per-expert tensors are fused into the module's 3D expert parameters."""
    import torch
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextDecoderLayer

    with torch.device("meta"):
        mod = Qwen3NextDecoderLayer(cfg, layer)
    mod = mod.to_empty(device="cpu").to(dtype)
    prefix = f"model.layers.{layer}."
    keys = ckpt.layer_keys(layer)
    plain = [k for k in keys if EXPERT_KEY not in k or k.endswith((".gate_up_proj", ".down_proj"))]
    state = {k[len(prefix):]: v.to(dtype) for k, v in ckpt.read(plain).items()}
    per_expert = [k for k in keys if EXPERT_KEY in k and k not in plain]
    if per_expert:
        n = cfg.num_experts
        params = dict(mod.named_parameters())
        gate_up, down = params["mlp.experts.gate_up_proj"], params["mlp.experts.down_proj"]
        for e in range(n):
            names = [f"{prefix}{EXPERT_KEY}{e}.{p}_proj.weight" for p in ("gate", "up", "down")]
            t = ckpt.read(names)
            gate_up.data[e].copy_(torch.cat([t[names[0]], t[names[1]]], dim=0).to(dtype))
            down.data[e].copy_(t[names[2]].to(dtype))
    missing, unexpected = mod.load_state_dict(state, strict=False)
    expected_missing = {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"} if per_expert else set()
    if unexpected or set(missing) - expected_missing:
        raise ValueError(f"layer {layer}: checkpoint does not match the module (missing {sorted(set(missing) - expected_missing)[:3]}, "
                         f"unexpected {sorted(unexpected)[:3]})")
    return mod.eval()


def stream_counts(model_dir: Path, sources: dict[str, list[list[int]]], *, dtype=None, log=None) -> dict[str, list[list[int]]]:
    """Per source, per MoE layer, how many times each expert was among the top_k picked. Runs the model one layer at a time."""
    import torch
    from transformers import AutoConfig
    from transformers.masking_utils import create_causal_mask
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

    dtype = dtype or torch.bfloat16
    cfg = AutoConfig.from_pretrained(model_dir)
    # A config read on its own has no attention implementation set, and the attention layer then falls back to the eager one, which is not causal
    # without an explicit mask. The whole-model path sets sdpa when it builds the model, so this run sets it the same way.
    cfg._attn_implementation = "sdpa"
    ckpt = Checkpoint(model_dir)
    emb = ckpt.read(["model.embed_tokens.weight"])["model.embed_tokens.weight"].to(dtype)
    rotary = Qwen3NextRotaryEmbedding(config=cfg)
    hidden = {name: [emb[torch.tensor(c)].unsqueeze(0) for c in chunks] for name, chunks in sources.items()}
    counts = {name: [] for name in sources}
    with torch.no_grad():
        for layer in range(cfg.num_hidden_layers):
            t0 = time.time()
            mod = load_layer(cfg, ckpt, layer, dtype)
            moe = hasattr(mod.mlp, "gate")
            picked = {}
            if moe:
                hook = mod.mlp.gate.register_forward_hook(lambda m, i, out: picked.__setitem__("idx", out[2]))
            for name, states in hidden.items():
                layer_counts = [0] * cfg.num_experts
                for j, h in enumerate(states):
                    pos = torch.arange(h.shape[1]).unsqueeze(0)
                    mask = create_causal_mask(config=cfg, inputs_embeds=h, attention_mask=None, past_key_values=None, position_ids=pos) \
                        if cfg.layer_types[layer] == "full_attention" else None
                    states[j] = mod(h, position_embeddings=rotary(h, pos), attention_mask=mask, position_ids=pos)
                    if moe:
                        bc = torch.bincount(picked["idx"].reshape(-1), minlength=cfg.num_experts)
                        layer_counts = [a + int(b) for a, b in zip(layer_counts, bc.tolist())]
                if moe:
                    counts[name].append(layer_counts)
            if moe:
                hook.remove()
            del mod
            if log:
                log(f"layer {layer + 1}/{cfg.num_hidden_layers} {time.time() - t0:.1f}s")
    return counts


def result(model_dir: Path, model_name: str, name: str, chunks: list[list[int]], counts: list[list[int]], cfg, text_note: str) -> dict:
    layers = [skew_from_counts(c) for c in counts]
    return {"model": model_name, "snapshot": model_dir.name, "text": name, "text_note": text_note, "tokens": sum(len(c) for c in chunks),
            "experts": cfg.num_experts, "top_k": cfg.num_experts_per_tok, "moe_layers": len(counts), "method": "one decoder layer at a time",
            "mean": mean_over_layers(layers), "layers": layers, "counts": counts,
            "note": "tokens counted over the text given; a router's skew on other text can differ"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", type=Path, help="local directory of a Qwen3-Next style checkpoint (safetensors)")
    ap.add_argument("--model-name", help="the model's name as the result should show it, for example org/name (default: the directory name, which for a "
                                         "Hugging Face cache snapshot is a commit hash)")
    ap.add_argument("--text", action="append", metavar="NAME=FILE[,FILE...]", required=True, help="a text source; repeat for several")
    ap.add_argument("--tokens", type=int, default=16384, help="tokens per source")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True, help="JSON file: {source name: result}")
    args = ap.parse_args(argv)
    import torch
    from transformers import AutoConfig, AutoTokenizer
    torch.set_num_threads(args.threads)
    cfg = AutoConfig.from_pretrained(args.model)
    tok = AutoTokenizer.from_pretrained(args.model)
    chunks_by_source, notes = {}, {}
    for spec in args.text:
        name, _, files = spec.partition("=")
        paths = [Path(f) for f in files.split(",") if f]
        if not name or not paths:
            raise SystemExit(f"--text wants NAME=FILE, got {spec!r}")
        chunks_by_source[name] = to_chunks(read_ids(paths, tok, args.tokens), args.chunk)
        notes[name] = f"{len(paths)} file(s), {args.chunk}-token chunks"
    counts = stream_counts(args.model, chunks_by_source, log=lambda m: print(m, file=sys.stderr, flush=True))
    out = {name: result(args.model, args.model_name or args.model.name, name, chunks_by_source[name], counts[name], cfg, notes[name]) for name in chunks_by_source}
    args.out.write_text(json.dumps(out))
    for name, r in out.items():
        m = r["mean"]
        print(f"{name}: {r['moe_layers']} MoE layers, {r['experts']} experts, top-{r['top_k']}, {r['tokens']} tokens: busiest 10% take "
              f"{m['top_10pct_share']:.1%}, 25% take {m['top_25pct_share']:.1%}, 50% take {m['top_50pct_share']:.1%}; never picked {m['never_picked']:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
