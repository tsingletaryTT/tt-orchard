#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Write a tiny Qwen4-shaped checkpoint with random, seeded weights, and record a CPU reference for it.

PROVISIONAL. No Qwen4 weights or config.json exist (2026-11-02). The shape follows public readings of the model and the Flash-Next
config; see docs/superpowers/specs/2026-11-02-tiny-qwen4-rehearsal-design.md for what each dimension stands for, how the stage 0, 1
and 2 checks see the result, and the runbook for the pilot run that uses it.

Run it with the reference interpreter (torch and transformers 5.19.0, CPU only; ORCHARD_REFERENCE_PYTHON):

    python3 make_checkpoint.py --out <dir outside the repository> [--variant dense|moe|mixed] [--seed 1234] [--layers 4]
                               [--no-tokenizer] [--no-reference] [--weight-scale 1.0] [--n-new 32]
    python3 make_checkpoint.py --reference-only <existing snapshot dir>

It never imports ttnn or ttml, opens no device and uses no network.

What it writes into --out (the snapshot directory itself):
    config.json, generation_config.json   a flat `qwen4_exp_text` config (class Qwen4ExpForCausalLM)
    model.safetensors                     bf16 weights from `torch.manual_seed(seed)`; key names come from the transformers class
    tokenizer.json, tokenizer_config.json, special_tokens_map.json
                                          a byte-level tokenizer sized to `vocab_size` (omitted with --no-tokenizer). Stage 1's template
                                          calls AutoTokenizer.from_pretrained, so it needs one. The id-based prompts do not use it.
    prompts.json                          seeded id lists, lengths 8, 16, 33
    reference/                            the CPU reference: prompt-ids.json and generated-ids.json (the first prompt, in the shape stage 1
                                          writes), reference-ids.json (every prompt, with top-5 ids and logits), load-report.json
    tiny-qwen4-manifest.json              seed, variant, dimensions, versions and the sha256 of every other file

Guards (each has a test in tests/test_tiny_qwen4.py and a mutation that turns it red):
  - the output directory must be outside the repository checkout and empty (checkpoints are never committed, two never mix);
  - the layer count is a positive multiple of 4 (the 3:1 schedule);
  - the reference is recorded only from a clean load (no missing, unexpected or mismatched key), never from a model with fill-in weights;
  - the reference is refused when a prompt's 32 generated tokens have fewer than 4 distinct ids or any logit is not finite;
  - the first generated token must equal the argmax of a separate forward pass over the prompt, and each step must append one token at
    the end; a failure refuses the reference.

Exit codes: 0 done; 2 refused before writing (bad option, output directory); 3 the reference was refused; 4 torch, transformers or
safetensors is not importable.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import platform
import random
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "reference-gate-templates"

VARIANTS = ("dense", "moe", "mixed")
N_NEW = 32                                   # generated tokens per prompt, as stage 1 decodes
TOP_K = 5
MIN_DISTINCT = 4                             # fewer distinct generated ids than this is a degenerate reference
PROMPT_LENGTHS = (8, 16, 33)                 # 33 is not a multiple of the delta-net chunk size
VOCAB_SIZE = 384                             # whole 32-wide tiles on 2 and 4 dies (192 and 96 per die)
SPECIAL_IDS = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
FIRST_BYTE_ID = 4                            # ids 4..259 are the 256 byte tokens; 260..383 are filler tokens

# Dimensions shared by every variant. The full-size reading is in the design note; ratios are kept where a source gives them.
BASE = {
    "model_type": "qwen4_exp_text", "architectures": ["Qwen4ExpForCausalLM"],
    "hidden_size": 64, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 32, "vocab_size": VOCAB_SIZE,
    "linear_num_key_heads": 2, "linear_num_value_heads": 6, "linear_key_head_dim": 32, "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4, "full_attention_interval": 4, "partial_rotary_factor": 0.25, "max_position_embeddings": 512,
    "tie_word_embeddings": False, "output_gate_type": "sigmoid",
    # the sparse indexer of the full-attention layer: the layer does not build without these keys
    "indexer_budget": 16, "indexer_compress_ratio": 4, "indexer_head_dim": 32, "indexer_kv_heads": 1, "indexer_n_heads": 4,
    # hyper-connections: hc_lowrank is the 1/8 ratio of the full-size reading (640 of 5120), written down and never defaulted
    "hc_count": 4, "hc_lowrank": 8,
    # the n-gram embedding with a small table (full-size Flash-Next: base vocabulary 20 million)
    "ngram_size": 3, "ngram_vocab_size_base": 1000, "heads_per_ngram": 2, "split_ngram_parts": 4,
    "ple_layer_ids": [2], "ple_embed_dim": 64, "ple_conv_kernel_size": 4,
    "pad_token_id": SPECIAL_IDS["<pad>"], "eos_token_id": SPECIAL_IDS["<eos>"], "bos_token_id": SPECIAL_IDS["<bos>"],
}
DENSE_FFN = {"intermediate_size": 256, "hidden_act": "silu"}        # a multiple of 128: whole 32-column tiles on 2 and 4 dies
MOE_FFN = {"num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64}


def build_config(variant: str, layers: int = 4) -> dict:
    """The raw config dict for a variant. A dense config carries no expert key and no schedule key at all: transformers fills a
    missing `num_experts` with 512, and the dense shim (qwen4_dense.py) reads the raw config to decide each layer."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose one of {', '.join(VARIANTS)}")
    if not isinstance(layers, int) or isinstance(layers, bool) or layers < 4 or layers % 4:
        raise ValueError(f"layers must be a positive multiple of 4 (the 3:1 schedule); got {layers!r}")
    cfg = copy.deepcopy(BASE)
    cfg["num_hidden_layers"] = layers
    cfg["layer_types"] = (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4)
    if variant == "dense":
        cfg.update(DENSE_FFN)
    elif variant == "moe":
        cfg.update(MOE_FFN)
    else:
        cfg.update(DENSE_FFN, **MOE_FFN, mlp_only_layers=[0])
    return cfg


def make_prompts(seed: int) -> list[list[int]]:
    """Seeded id lists inside the byte-token range, so they decode with the tokenizer and avoid the special ids."""
    rng = random.Random(f"tiny-qwen4-prompts-{seed}")
    return [[rng.randrange(FIRST_BYTE_ID, FIRST_BYTE_ID + 256) for _ in range(n)] for n in PROMPT_LENGTHS]


def check_output_dir(out: Path) -> Path:
    """Exit 2 when `out` is inside the repository checkout or is a non-empty directory. Checkpoints are never committed."""
    out = Path(out).expanduser().resolve()
    try:
        out.relative_to(REPO)
    except ValueError:
        pass
    else:
        sys.exit(refuse(f"{out} is inside the repository checkout ({REPO}); generate into a temp directory"))
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        sys.exit(refuse(f"{out} exists and is not empty; use a new directory so two checkpoints never mix"))
    return out


def refuse(message: str, code: int = 2) -> int:
    print(f"make_checkpoint: {message}", file=sys.stderr, flush=True)
    return code


def say(message: str) -> None:
    print(f"make_checkpoint: {message}", flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def import_stack():
    """(torch, transformers); exit 4 naming the package that is missing."""
    try:
        import torch
        import transformers
        import safetensors  # noqa: F401
    except ImportError as exc:
        sys.exit(refuse(f"{exc.name} is not importable by {sys.executable}; run this with the reference interpreter "
                        "(ORCHARD_REFERENCE_PYTHON: torch, transformers 5.19.0, safetensors)", 4))
    return torch, transformers


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- the weights ----------------------------------------------------------------------------------------------------

def write_weights(torch, transformers, cfg: dict, seed: int, out: Path, weight_scale: float) -> None:
    """Build the model (through the dense shim when a layer is dense) under `manual_seed(seed)`, cast floats to bf16 and save.
    `save_pretrained` writes the keys in the format `from_pretrained` reads back; only its safetensors file is kept."""
    shim = load_module("qwen4_dense", TEMPLATES / "qwen4_dense.py")
    raw = {**cfg, "seed": seed}                      # the n-gram buffers are drawn from the config's seed
    config = shim.prepare_config(transformers, raw) if shim.needs_dense_shim(raw) else transformers.AutoConfig.for_model(**raw)
    torch.manual_seed(seed)
    with (shim.dense_layers() if shim.needs_dense_shim(raw) else contextlib.nullcontext()):
        model = transformers.Qwen4ExpForCausalLM(config)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(weight_scale)
    model = model.to(torch.bfloat16)                 # integer buffers keep their dtype
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp, safe_serialization=True)
        shutil.copy(Path(tmp) / "model.safetensors", out / "model.safetensors")


# ---- the tokenizer --------------------------------------------------------------------------------------------------

def byte_unicode_table() -> dict[int, str]:
    """GPT-2's byte to printable-character table, which the byte-level pre-tokenizer and decoder use."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    table, extra = {}, 0
    for b in range(256):
        if b in keep:
            table[b] = chr(b)
        else:
            table[b] = chr(256 + extra)
            extra += 1
    return table


def write_tokenizer(transformers, out: Path) -> None:
    """256 byte tokens (ids 4..259), four special tokens (0..3) and filler tokens up to VOCAB_SIZE. No merges: every string
    round-trips as bytes. Filler tokens are never produced by encoding; they decode to their own name, so a model's random ids
    decode to non-empty text."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    table = byte_unicode_table()
    vocab = dict(SPECIAL_IDS)
    for b in range(256):
        vocab[table[b]] = FIRST_BYTE_ID + b
    for i in range(VOCAB_SIZE - len(vocab)):
        vocab[f"<ext_{i}>"] = len(vocab)
    assert len(vocab) == VOCAB_SIZE and sorted(vocab.values()) == list(range(VOCAB_SIZE))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
                                                unk_token="<unk>", clean_up_tokenization_spaces=False)
    fast.save_pretrained(str(out))


# ---- the CPU reference ----------------------------------------------------------------------------------------------

class ReferenceRefused(Exception):
    """The reference is not recorded: the load was not clean or the decode is degenerate."""


def layer_blocks(model) -> list[str]:
    return [type(layer.mlp).__name__ for layer in model.model.layers]


def run_reference(snapshot: Path, prompts: list[list[int]], out_dir: Path, *, n_new: int = N_NEW) -> dict:
    """Load the snapshot the way stage 1 does (reference_gate.load_model, which uses the dense shim for a dense layer) and decode
    `n_new` tokens greedily for each prompt, one full forward pass per token with no cache. Writes `out_dir` files; raises
    ReferenceRefused before writing any ids when the load is not clean."""
    torch, transformers = import_stack()
    gate = load_module("reference_gate", TEMPLATES / "reference_gate.py")
    model, load = gate.load_model(torch, transformers, snapshot)
    load["layer_blocks"] = layer_blocks(model)
    problems = {k: load[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if load.get(k)}
    if problems:
        write_json(out_dir / "load-report.json", load)
        raise ReferenceRefused("the weights did not load cleanly, so no reference was recorded (missing keys, unexpected keys or "
                               f"mismatches would leave random fill-in weights): { {k: v[:5] for k, v in problems.items()} }")

    def forward(seq):
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([seq], dtype=torch.long), use_cache=False).logits[0, -1].float()
        if not bool(torch.isfinite(logits).all()):
            raise ReferenceRefused("a logit is non-finite (nan or inf); the weights or the dimensions are broken")
        return logits

    records = []
    for index, prompt_ids in enumerate(prompts):
        say(f"reference: prompt {index} ({len(prompt_ids)} ids), {n_new} greedy tokens")
        first = forward(prompt_ids)
        seq, steps, appended = list(prompt_ids), [], []
        for k in range(n_new):
            logits = forward(seq)
            vals, ids = torch.topk(logits, TOP_K)
            chosen = int(ids[0])
            steps.append({"position": len(seq) - 1, "chosen": chosen, "gap": round(float(vals[0] - vals[1]), 6),
                          "top5": [{"id": int(i), "logit": round(float(v), 5)} for v, i in zip(vals.tolist(), ids.tolist())]})
            appended.append({"chosen": chosen, "before": list(seq), "after": list(seq) + [chosen]})
            seq = seq + [chosen]
        generated = seq[len(prompt_ids):]
        if len(set(generated)) < MIN_DISTINCT:
            raise ReferenceRefused(f"degenerate reference: prompt {index} generated only {len(set(generated))} distinct ids "
                                   f"({generated[:8]}...); a constant output would let any wrong port pass")
        ok, why = gate.appended_at_end(prompt_ids, appended)
        records.append({"prompt_ids": prompt_ids, "generated_ids": generated, "first_token_ok": int(torch.argmax(first)) == generated[0],
                        "appended_ok": ok, "appended_note": why, "steps": steps})
        if not (records[-1]["first_token_ok"] and ok):
            raise ReferenceRefused(f"prompt {index}: the first-token or append check failed ({why})")
    shim_sha = (load.get("dense_qwen4") or {}).get("shim_sha256")
    write_json(out_dir / "load-report.json", load)
    write_json(out_dir / "reference-ids.json", {
        "n_new": n_new, "torch": torch.__version__, "transformers": transformers.__version__, "shim_sha256": shim_sha,
        "note": "greedy, one full forward pass per token, bf16 on the CPU. `gap` is the top-1 minus top-2 logit: a port's first "
                "differing position with a small gap is a near-tie, with a large gap it is a wrong port.",
        "prompts": records})
    # the first prompt, in the shape stage 1 writes (stage 2 reads prompt_ids and generated_ids)
    write_json(out_dir / "prompt-ids.json", {"prompt_ids": prompts[0], "prompt_text": "", "user_prompt": "id-based prompt 0",
                                             "chat_template_used": False, "id_based": True})
    write_json(out_dir / "generated-ids.json", {"generated_ids": records[0]["generated_ids"], "generated_text": "", "complete": True})
    return load


# ---- the manifest and the driver ------------------------------------------------------------------------------------

def write_manifest(out: Path, cfg: dict, variant: str, seed: int, layers: int, weight_scale: float) -> dict:
    import torch
    import transformers
    files = {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != "tiny-qwen4-manifest.json"}
    manifest = {"provisional": "built from public readings of an unreleased model; random weights", "variant": variant, "seed": seed,
                "layers": layers, "weight_scale": weight_scale, "dimensions": cfg,
                "shim_sha256": sha256(TEMPLATES / "qwen4_dense.py"), "torch": torch.__version__,
                "transformers": transformers.__version__, "python": platform.python_version(), "files": files}
    write_json(out / "tiny-qwen4-manifest.json", manifest)
    return manifest


def make(out: Path, variant: str, seed: int, layers: int, *, tokenizer: bool, reference: bool, weight_scale: float,
         n_new: int = N_NEW) -> int:
    cfg = build_config(variant, layers)
    out = check_output_dir(out)
    torch, transformers = import_stack()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {**cfg, "seed": seed})
    write_json(out / "generation_config.json", {"do_sample": False, "eos_token_id": SPECIAL_IDS["<eos>"],
                                                "pad_token_id": SPECIAL_IDS["<pad>"], "bos_token_id": SPECIAL_IDS["<bos>"]})
    say(f"building the {variant} model ({layers} layers, seed {seed})")
    write_weights(torch, transformers, cfg, seed, out, weight_scale)
    if tokenizer:
        write_tokenizer(transformers, out)
    prompts = make_prompts(seed)
    write_json(out / "prompts.json", {"prompts": prompts, "note": "token ids in the byte-token range; no tokenizer is needed"})
    if reference:
        try:
            run_reference(out, prompts, out / "reference", n_new=n_new)
        except ReferenceRefused as exc:
            return refuse(str(exc), 3)
    write_manifest(out, cfg, variant, seed, layers, weight_scale)
    say(f"wrote {out}")
    return 0


def reference_only(snapshot: Path, n_new: int = N_NEW) -> int:
    snapshot = Path(snapshot).resolve()
    prompts_path = snapshot / "prompts.json"
    if not prompts_path.is_file():
        return refuse(f"{prompts_path} is missing", 2)
    try:
        run_reference(snapshot, json.loads(prompts_path.read_text())["prompts"], snapshot / "reference", n_new=n_new)
    except ReferenceRefused as exc:
        return refuse(str(exc), 3)
    say(f"wrote {snapshot / 'reference'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="a new directory outside the repository checkout (the snapshot directory)")
    ap.add_argument("--variant", choices=VARIANTS, default="dense")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--layers", type=int, default=4, help="a positive multiple of 4")
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--no-reference", action="store_true", help="write the checkpoint and prompts only")
    ap.add_argument("--weight-scale", type=float, default=1.0, help="multiply every float weight (0 gives a degenerate model; used by a test)")
    ap.add_argument("--n-new", type=int, default=N_NEW, help="generated tokens per prompt (default 32, what stage 1 decodes; fewer only to save time in tests)")
    ap.add_argument("--reference-only", metavar="SNAPSHOT", help="record the reference for an existing snapshot and write nothing else")
    args = ap.parse_args(argv)
    if args.reference_only:
        return reference_only(Path(args.reference_only), args.n_new)
    if not args.out:
        return refuse("--out is required", 2)
    try:
        return make(Path(args.out), args.variant, args.seed, args.layers, tokenizer=not args.no_tokenizer,
                    reference=not args.no_reference, weight_scale=args.weight_scale, n_new=args.n_new)
    except ValueError as exc:
        return refuse(str(exc), 2)


if __name__ == "__main__":
    sys.exit(main())
