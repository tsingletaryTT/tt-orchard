#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Write a tiny Qwen4-shaped checkpoint with random, seeded weights, and record a CPU reference for it.

PROVISIONAL. No Qwen4 weights or config.json exist (2026-11-02). The shape follows public readings of the model and the Flash-Next
config; see docs/superpowers/specs/2026-11-02-tiny-qwen4-rehearsal-design.md for what each dimension stands for, how the stage 0, 1
and 2 checks see the result, and the runbook for the pilot run that uses it.

Run it with the reference interpreter (torch and transformers 5.19.0, CPU only; ORCHARD_REFERENCE_PYTHON):

    python3 make_checkpoint.py --out <dir outside the repository> [--variant dense|moe|mixed] [--seed 1234] [--layers 4]
                               [--width tiny|27b] [--set KEY=VALUE ...] [--max-gb 12]
                               [--no-tokenizer] [--no-reference] [--weight-scale 1.0] [--n-new 32]
    python3 make_checkpoint.py --reference-only <existing snapshot dir>

Widths. `--width tiny` (the default) is the 64-wide model every earlier run used; its output is byte-identical to what this script wrote
before `--width` existed (except that the manifest gained keys). `--width 27b` keeps four layers and random weights and takes the widths of
the full-size reading in the design note: hidden size 5120, 24 / 4 attention heads of 256, 16 / 48 delta-net heads of 128, `hc_lowrank`
640, a dense feed-forward width of 14336, vocabulary 248320, indexer budget 2048 with heads of 128, 4096 positions (and for `--variant
moe|mixed`: 512 experts, top 10, expert width 640). The n-gram table and the per-layer embedding stay at their tiny sizes: no public reading
gives their widths and the full-size n-gram table is 102 GB. `--set KEY=VALUE` (repeatable) overrides one width after the preset; only the
width keys in WIDTH_KEYS, only keys the variant already has, only positive whole numbers.

Size. Before anything is written the script counts the parameters of the model on torch's meta device (nothing is allocated), prints
`estimated size`, and refuses when the estimate is over `--max-gb` (default 12) or over the free space of the output filesystem minus a
40 GB margin. The `27b` dense checkpoint is 8.2 GB (4.12 billion parameters); `mixed` is 37 GB and `moe` 47 GB, so those two need a raised
`--max-gb` and the disk to match. The weights are written once, into a temporary directory inside `--out` and renamed into place, so the
estimate is the disk the run needs on the output filesystem; they are always one `model.safetensors` file.

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
    tiny-qwen4-manifest.json              seed, variant, width, overrides, dimensions, estimated and actual sizes, versions and the sha256
                                          of every other file

Guards (each has a test in tests/test_tiny_qwen4.py or tests/test_tiny_qwen4_width.py and a mutation that turns it red):
  - the output directory must be outside the repository checkout and empty (checkpoints are never committed, two never mix);
  - the layer count is a positive multiple of 4 (the 3:1 schedule);
  - a scaled config (a preset other than `tiny`, or any `--set`) must tile on 2 and 4 dies: the hidden, feed-forward, expert and
    vocabulary widths are multiples of 128, the head widths multiples of 32, the attention, delta-net and indexer head counts divide
    by 4 (a key-value head count, the attention's and the indexer's, divides by 4 or divides 4), and each count of value heads is a multiple
    of the key heads;
  - an override may name only a width key the variant already has (an expert key on a dense config would make transformers fill in
    512 experts and build a different model without an error);
  - the size estimate must be within `--max-gb` and within the free space minus 40 GB, checked before the directory is created;
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
import math
import os
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
REFERENCE_FILES = ("prompt-ids.json", "generated-ids.json", "reference-ids.json", "load-report.json")
N_NEW = 32                                  # generated tokens per prompt, as stage 1 decodes
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
EXPERT_KEYS = tuple(MOE_FFN)

# ---- the width presets and the size rule ----------------------------------------------------------------------------------
# `tiny` is BASE and the FFN blocks above, unchanged. `27b` replaces the widths below with the full-size reading of the design note
# (hidden 5120, 24 / 4 attention heads of 256, 16 / 48 delta-net heads of 128, hc_lowrank 640, dense FFN about 14336, vocabulary
# 248320, indexer 2048 / 4 / 4 / 128 / 1, MoE 512 / 10 / 640), where a key is present in the variant's config. The layer count stays 4.
# Not scaled, because no public reading gives a width: the per-layer embedding (`ple_embed_dim`, `ple_layer_ids`) and the n-gram table
# (`ngram_vocab_size_base`; the full-size table is 102 GB). `max_position_embeddings` is 4096, a working length (the model has 262144).
WIDTHS = ("tiny", "27b")
WIDTH_PRESETS = {
    "tiny": {},
    "27b": {
        "hidden_size": 5120, "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "vocab_size": 248320,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "hc_lowrank": 640, "indexer_budget": 2048, "indexer_head_dim": 128, "max_position_embeddings": 4096,
        "intermediate_size": 14336,
        "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640, "shared_expert_intermediate_size": 640,
    },
}
# The keys `--set` may change: every number in WIDTH_PRESETS["27b"], and the indexer head counts. Not the layer count (`--layers`),
# the layer schedule, the per-layer embedding or the n-gram table.
WIDTH_KEYS = frozenset(WIDTH_PRESETS["27b"]) | {"indexer_n_heads", "indexer_kv_heads"}
MAX_OVERRIDE = 1 << 20                         # no width or head count above this: counting the parameters of a model with `10**20` columns must not hang
MAX_OVERRIDE_BY_KEY = {"num_experts": 4096}    # one module per expert is built, so the expert count has a lower limit
DIES = (2, 4)                                  # the mesh sizes the widths must tile on
TILE = 32                                      # columns in a tile
DEFAULT_MAX_GB = 12.0
SINGLE_FILE_BYTES = 1 << 50                    # `max_shard_size` for save_pretrained: one safetensors file, whatever the size
SPACE_MARGIN_GB = 40                           # free space that must remain after the checkpoint is written
GB = 10 ** 9
HEADER_BYTES_PER_TENSOR = 128                  # safetensors header: measured 114 per tensor on the tiny model, rounded up
TOKENIZER_BYTES_PER_ID = 32                    # tokenizer.json: measured 23 per id on the tiny model, rounded up
SIDE_FILES_BYTES = 1 << 20                     # config, prompts, manifest and the reference files: well under 1 MiB


class SizeRefused(ValueError):
    """The estimated checkpoint is over --max-gb or does not fit the free space minus the margin."""


def whole_number(value) -> bool:
    return type(value) is int and value > 0


def check_widths(cfg: dict) -> None:
    """ValueError naming every width a 2-die or 4-die mesh cannot tile. The tiny default does not pass these rules (hidden size 64,
    two delta-net key heads) and is not held to them: build_config applies them to a scaled config only."""
    problems = []
    for key in ("hidden_size", "intermediate_size", "vocab_size", "moe_intermediate_size", "shared_expert_intermediate_size"):
        if key in cfg and cfg[key] % (TILE * max(DIES)):
            problems.append(f"{key} {cfg[key]} is not a multiple of {TILE * max(DIES)} (whole {TILE}-column tiles on {max(DIES)} dies)")
    if cfg["vocab_size"] < FIRST_BYTE_ID + 256:
        problems.append(f"vocab_size {cfg['vocab_size']} is too small for the {FIRST_BYTE_ID + 256} special and byte tokens of the tokenizer")
    for key in ("head_dim", "linear_key_head_dim", "linear_value_head_dim", "indexer_head_dim"):
        if key in cfg and cfg[key] % TILE:
            problems.append(f"{key} {cfg[key]} is not a multiple of {TILE} (a whole tile)")
    heads, kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    if heads % max(DIES):
        problems.append(f"num_attention_heads {heads} does not divide by {max(DIES)} dies")
    if heads % kv:
        problems.append(f"num_attention_heads {heads} is not a multiple of num_key_value_heads {kv}")
    for key in ("num_key_value_heads", "indexer_kv_heads"):          # the indexer's KV heads are sharded across the dies like the attention's
        if key in cfg and cfg[key] % max(DIES) and max(DIES) % cfg[key]:
            problems.append(f"{key} {cfg[key]} neither divides by {max(DIES)} dies nor divides {max(DIES)} (replication)")
    for key in ("linear_num_key_heads", "linear_num_value_heads", "indexer_n_heads"):
        if cfg[key] % max(DIES):
            problems.append(f"{key} {cfg[key]} does not divide by {max(DIES)} dies")
    if cfg["linear_num_value_heads"] % cfg["linear_num_key_heads"]:
        problems.append(f"linear_num_value_heads {cfg['linear_num_value_heads']} is not a multiple of linear_num_key_heads "
                        f"{cfg['linear_num_key_heads']}")
    if "num_experts" in cfg and not 1 <= cfg["num_experts_per_tok"] <= cfg["num_experts"]:
        problems.append(f"num_experts_per_tok {cfg['num_experts_per_tok']} must be between 1 and num_experts {cfg['num_experts']}")
    if problems:
        raise ValueError("these widths do not tile on 2 and 4 dies: " + "; ".join(problems))


def check_overrides(cfg: dict, overrides) -> dict:
    """The validated override dict. Each key must be a width key the config already has, each value a positive whole number."""
    if not overrides:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError(f"overrides must be a dict of width keys to numbers; got {type(overrides).__name__}")
    for key, value in overrides.items():
        if key not in WIDTH_KEYS:
            raise ValueError(f"{key!r} is not a width key; --set takes only: {', '.join(sorted(WIDTH_KEYS))}")
        if key not in cfg:
            raise ValueError(f"{key} is not in this variant's config; an expert key on a dense config would make transformers fill in "
                             "512 experts, and a dense width on a MoE-only config would add a dense layer that is not there")
        if not whole_number(value):
            raise ValueError(f"override {key}: need a positive whole number, got {value!r}")
        limit = MAX_OVERRIDE_BY_KEY.get(key, MAX_OVERRIDE)
        if value > limit:
            raise ValueError(f"override {key}: {value} is above the limit of {limit}")
    return dict(overrides)


def parse_overrides(items) -> dict:
    """`KEY=VALUE` strings from `--set` to a dict. A missing `=`, an empty key, a repeated key or a value that is not plain digits is refused."""
    out: dict = {}
    for item in items or []:
        key, sep, value = str(item).partition("=")
        if not sep or not key or not value.isascii() or not value.isdigit():
            raise ValueError(f"--set {item!r}: write KEY=VALUE with a positive whole number, for example hidden_size=2560")
        if key in out:
            raise ValueError(f"--set {key} is given twice")
        out[key] = int(value)
    return out


def build_config(variant: str, layers: int = 4, width: str = "tiny", overrides: dict | None = None) -> dict:
    """The raw config dict for a variant. A dense config carries no expert key and no schedule key at all: transformers fills a
    missing `num_experts` with 512, and the dense shim (qwen4_dense.py) reads the raw config to decide each layer.
    `width` picks a preset (WIDTHS); `overrides` change single width keys after it. A scaled config (a preset other than `tiny`, or any
    override) must tile on 2 and 4 dies (check_widths). The default is returned exactly as before the presets existed."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose one of {', '.join(VARIANTS)}")
    if not isinstance(layers, int) or isinstance(layers, bool) or layers < 4 or layers % 4:
        raise ValueError(f"layers must be a positive multiple of 4 (the 3:1 schedule); got {layers!r}")
    if width not in WIDTHS:
        raise ValueError(f"unknown width {width!r}; choose one of {', '.join(WIDTHS)}")
    cfg = copy.deepcopy(BASE)
    cfg["num_hidden_layers"] = layers
    cfg["layer_types"] = (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4)
    if variant == "dense":
        cfg.update(DENSE_FFN)
    elif variant == "moe":
        cfg.update(MOE_FFN)
    else:
        cfg.update(DENSE_FFN, **MOE_FFN, mlp_only_layers=[0])
    overrides = check_overrides(cfg, overrides)
    cfg.update({k: v for k, v in WIDTH_PRESETS[width].items() if k in cfg})
    cfg.update(overrides)
    if width != "tiny" or overrides:
        check_widths(cfg)
    return cfg


def check_max_gb(max_gb) -> float:
    if isinstance(max_gb, bool) or not isinstance(max_gb, (int, float)) or not math.isfinite(max_gb) or max_gb <= 0:
        raise ValueError(f"--max-gb must be a finite number above 0; got {max_gb!r}")
    return float(max_gb)


def check_size(total_bytes, max_gb, free: int, margin_gb: float = SPACE_MARGIN_GB, where: str = "") -> None:
    """SizeRefused when the estimate is over --max-gb or over the free space minus the margin (a free space under the margin leaves
    nothing). The estimate and the free space must be whole byte counts and the limit a finite number above 0, else ValueError.
    An estimate exactly at either limit is allowed."""
    max_gb = check_max_gb(max_gb)
    if type(total_bytes) is not int or total_bytes < 0:
        raise ValueError(f"the size estimate must be a whole number of bytes; got {total_bytes!r}")
    if type(free) is not int or free < 0:
        raise ValueError(f"the free space must be a whole number of bytes; got {free!r}")
    if total_bytes > max_gb * GB:
        raise SizeRefused(f"estimated size {total_bytes / GB:.1f} GB is over --max-gb {max_gb:g}; raise --max-gb to write it anyway "
                          "(it also has to fit the free space)")
    usable = max(0, free - int(margin_gb * GB))
    if total_bytes > usable:
        raise SizeRefused(f"estimated size {total_bytes / GB:.1f} GB does not fit: {free / GB:.1f} GB of free space{where}, minus the "
                          f"{margin_gb:g} GB margin, leaves {usable / GB:.1f} GB")


def free_bytes(path: Path) -> int:
    """Free bytes on the filesystem that holds `path` (its nearest existing parent when it is not made yet)."""
    probe = Path(path).expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError as exc:
        raise ValueError(f"cannot read the free space of {probe}: {exc}") from exc


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

def build_model(torch, transformers, cfg: dict, seed: int, *, meta: bool = False):
    """The model for `cfg` (through the dense shim when a layer is dense) under `manual_seed(seed)`. With `meta` the parameters have a
    shape and no storage, so a 4-billion-parameter model can be counted without allocating it. The weights written and the weights
    counted come from this one function, so the estimate cannot describe a different model from the file."""
    shim = load_module("qwen4_dense", TEMPLATES / "qwen4_dense.py")
    raw = {**cfg, "seed": seed}                      # the n-gram buffers are drawn from the config's seed
    config = shim.prepare_config(transformers, raw) if shim.needs_dense_shim(raw) else transformers.AutoConfig.for_model(**raw)
    torch.manual_seed(seed)
    with (torch.device("meta") if meta else contextlib.nullcontext()):
        with (shim.dense_layers() if shim.needs_dense_shim(raw) else contextlib.nullcontext()):
            return transformers.Qwen4ExpForCausalLM(config)


def estimate_bytes(torch, transformers, cfg: dict, *, seed: int = 1234, tokenizer: bool = True) -> dict:
    """The size the checkpoint will have, from the parameter count of the meta-device model: bf16 for every float, the buffer's own size
    for an integer buffer, a safetensors header of HEADER_BYTES_PER_TENSOR per saved tensor, the tokenizer at TOKENIZER_BYTES_PER_ID per
    id, and SIDE_FILES_BYTES for the config, prompts, manifest and reference files. Measured against the tiny model: within 0.2%."""
    model = build_model(torch, transformers, cfg, seed, meta=True)
    params = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() * (2 if b.is_floating_point() else b.element_size()) for b in model.buffers())
    weights = 2 * params + buffers + 8 + HEADER_BYTES_PER_TENSOR * len(model.state_dict())
    tok = TOKENIZER_BYTES_PER_ID * cfg["vocab_size"] if tokenizer else 0
    return {"weights": int(weights), "tokenizer": int(tok), "total": int(weights + tok + SIDE_FILES_BYTES), "parameters": int(params)}


def write_weights(torch, transformers, cfg: dict, seed: int, out: Path, weight_scale: float) -> None:
    """Build the model (through the dense shim when a layer is dense) under `manual_seed(seed)`, cast floats to bf16 and save.
    `save_pretrained` writes the keys in the format `from_pretrained` reads back; only its safetensors file is kept.

    The temporary directory is made inside `out`, so the weights are written once on the filesystem that `check_size` measured, and the
    file is renamed into place. A copy from the system temp directory needed the checkpoint's size twice on one filesystem, or filled a
    filesystem nobody had checked (review of PR 434, round 1). `max_shard_size` is set above any checkpoint this script can write, so
    the weights are one `model.safetensors` whatever `--max-gb` allows; the default of 50 GB split a larger one into shards, and the
    rename found no file after the whole checkpoint was written."""
    model = build_model(torch, transformers, cfg, seed)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(weight_scale)
    model = model.to(torch.bfloat16)                 # integer buffers keep their dtype
    with tempfile.TemporaryDirectory(dir=out, prefix=".weights-") as tmp:
        model.save_pretrained(tmp, safe_serialization=True, max_shard_size=SINGLE_FILE_BYTES)
        os.replace(Path(tmp) / "model.safetensors", out / "model.safetensors")


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


def write_tokenizer(transformers, out: Path, vocab_size: int = VOCAB_SIZE) -> None:
    """256 byte tokens (ids 4..259), four special tokens (0..3) and filler tokens up to `vocab_size` (the config's, VOCAB_SIZE by
    default). No merges: every string round-trips as bytes. Filler tokens are never produced by encoding; they decode to their own
    name, so a model's random ids decode to non-empty text."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    table = byte_unicode_table()
    vocab = dict(SPECIAL_IDS)
    for b in range(256):
        vocab[table[b]] = FIRST_BYTE_ID + b
    for i in range(vocab_size - len(vocab)):
        vocab[f"<ext_{i}>"] = len(vocab)
    assert len(vocab) == vocab_size and sorted(vocab.values()) == list(range(vocab_size))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
                                                unk_token="<unk>", clean_up_tokenization_spaces=False)
    fast.save_pretrained(str(out))


# ---- the CPU reference ----------------------------------------------------------------------------------------------

def greedy_token(torch, logits) -> int:
    """The greedy choice, taken the way stage 1 takes it (`torch.argmax`). `torch.topk` may order tied logits differently and bf16
    logits tie often, so topk supplies the diagnostic top-5 only."""
    return int(torch.argmax(logits))


class ReferenceRefused(Exception):
    """The reference is not recorded: the load was not clean or the decode is degenerate."""


def layer_blocks(model) -> list[str]:
    return [type(layer.mlp).__name__ for layer in model.model.layers]


def run_reference(snapshot: Path, prompts: list[list[int]], out_dir: Path, *, n_new: int = N_NEW) -> dict:
    """Load the snapshot the way stage 1 does (reference_gate.load_model, which uses the dense shim for a dense layer) and decode
    `n_new` tokens greedily for each prompt, one full forward pass per token with no cache. Writes `out_dir` files; raises
    ReferenceRefused before writing any ids when the load is not clean. Files from an earlier run are removed first, so a refresh that
    fails leaves nothing that still says `complete`."""
    for name in REFERENCE_FILES:
        (out_dir / name).unlink(missing_ok=True)
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
            chosen = greedy_token(torch, logits)
            steps.append({"position": len(seq) - 1, "chosen": chosen, "gap": round(float(vals[0] - vals[1]), 6),
                          "top5": [{"id": int(i), "logit": round(float(v), 5)} for v, i in zip(vals.tolist(), ids.tolist())]})
            appended.append({"chosen": chosen, "before": list(seq), "after": list(seq) + [chosen]})
            seq = seq + [chosen]
        generated = seq[len(prompt_ids):]
        if len(set(generated)) < MIN_DISTINCT:
            raise ReferenceRefused(f"degenerate reference: prompt {index} generated only {len(set(generated))} distinct ids "
                                   f"({generated[:8]}...); a constant output would let any wrong port pass")
        ok, why = gate.appended_at_end(prompt_ids, appended)
        records.append({"prompt_ids": prompt_ids, "generated_ids": generated, "first_token_ok": greedy_token(torch, first) == generated[0],
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

def write_manifest(out: Path, cfg: dict, variant: str, seed: int, layers: int, weight_scale: float, *, width: str = "tiny",
                   overrides: dict | None = None, estimate: dict | None = None) -> dict:
    """`size` holds the estimate made before the write (null when a snapshot made before the estimate existed is refreshed) and the
    bytes on disk now: the weights file, and every file except this manifest."""
    import torch
    import transformers
    paths = [p for p in sorted(out.rglob("*")) if p.is_file() and p.name != "tiny-qwen4-manifest.json"]
    files = {str(p.relative_to(out)): sha256(p) for p in paths}
    weights = out / "model.safetensors"
    size = {"estimated_weights_bytes": (estimate or {}).get("weights"), "estimated_total_bytes": (estimate or {}).get("total"),
            "actual_weights_bytes": weights.stat().st_size if weights.is_file() else None,
            "actual_total_bytes": sum(p.stat().st_size for p in paths)}
    manifest = {"provisional": "built from public readings of an unreleased model; random weights", "variant": variant, "seed": seed,
                "layers": layers, "weight_scale": weight_scale, "width": width, "overrides": dict(overrides or {}), "size": size,
                "dimensions": cfg,
                "shim_sha256": sha256(TEMPLATES / "qwen4_dense.py"), "torch": torch.__version__,
                "transformers": transformers.__version__, "python": platform.python_version(), "files": files}
    write_json(out / "tiny-qwen4-manifest.json", manifest)
    return manifest


def make(out: Path, variant: str, seed: int, layers: int, *, tokenizer: bool, reference: bool, weight_scale: float,
         n_new: int = N_NEW, width: str = "tiny", overrides: dict | None = None, max_gb: float = DEFAULT_MAX_GB) -> int:
    check_max_gb(max_gb)
    cfg = build_config(variant, layers, width, overrides)
    out = check_output_dir(out)
    torch, transformers = import_stack()
    estimate = estimate_bytes(torch, transformers, cfg, seed=seed, tokenizer=tokenizer)
    say(f"estimated size {estimate['total'] / GB:.1f} GB ({estimate['parameters']:,} parameters; weights {estimate['weights'] / GB:.2f} GB, "
        f"tokenizer {estimate['tokenizer'] / GB:.3f} GB)")
    check_size(estimate["total"], max_gb, free_bytes(out), where=f" under {out}")       # before the directory exists
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {**cfg, "seed": seed})
    write_json(out / "generation_config.json", {"do_sample": False, "eos_token_id": SPECIAL_IDS["<eos>"],
                                                "pad_token_id": SPECIAL_IDS["<pad>"], "bos_token_id": SPECIAL_IDS["<bos>"]})
    say(f"building the {variant} model ({layers} layers, seed {seed}, width {width}{', overrides ' + json.dumps(overrides) if overrides else ''})")
    write_weights(torch, transformers, cfg, seed, out, weight_scale)
    if tokenizer:
        write_tokenizer(transformers, out, cfg["vocab_size"])
    prompts = make_prompts(seed)
    write_json(out / "prompts.json", {"prompts": prompts, "note": "token ids in the byte-token range; no tokenizer is needed"})
    if reference:
        try:
            run_reference(out, prompts, out / "reference", n_new=n_new)
        except ReferenceRefused as exc:
            return refuse(str(exc), 3)
    manifest = write_manifest(out, cfg, variant, seed, layers, weight_scale, width=width, overrides=overrides, estimate=estimate)
    say(f"wrote {out} ({manifest['size']['actual_total_bytes'] / GB:.2f} GB on disk)")
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
    refresh_manifest(snapshot)
    say(f"wrote {snapshot / 'reference'} and refreshed the manifest")
    return 0


def refresh_manifest(snapshot: Path) -> dict:
    """Rewrite the manifest's file inventory to match the snapshot on disk, keeping the variant, seed and layer count it already had
    (or reading them from config.json when the snapshot has no manifest). A reference run adds or replaces files the manifest covers."""
    cfg = json.loads((snapshot / "config.json").read_text())
    seed = cfg.pop("seed", None)
    old_path = snapshot / "tiny-qwen4-manifest.json"
    old = json.loads(old_path.read_text()) if old_path.is_file() else {}
    variant = old.get("variant") or ("mixed" if cfg.get("mlp_only_layers") else "dense" if "num_experts" not in cfg else "moe")
    was = old.get("size") if isinstance(old.get("size"), dict) else {}
    estimate = {"weights": was.get("estimated_weights_bytes"), "total": was.get("estimated_total_bytes")}      # the estimate made at write time
    return write_manifest(snapshot, cfg, variant, old.get("seed", seed), old.get("layers", cfg["num_hidden_layers"]),
                          old.get("weight_scale", 1.0), width=old.get("width", "unknown"), overrides=old.get("overrides"), estimate=estimate)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="a new directory outside the repository checkout (the snapshot directory)")
    ap.add_argument("--variant", choices=VARIANTS, default="dense")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--layers", type=int, default=4, help="a positive multiple of 4")
    ap.add_argument("--width", choices=WIDTHS, default="tiny",
                    help="the width preset: tiny (default, the 64-wide model) or 27b (the full-size widths in four layers)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one width after the preset, for example hidden_size=2560 (repeatable; width keys only; the result must tile on 2 and 4 dies)")
    ap.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB,
                    help=f"refuse when the estimated size is over this many GB (default {DEFAULT_MAX_GB:g}); the free space minus {SPACE_MARGIN_GB} GB is checked too")
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--no-reference", action="store_true", help="write the checkpoint and prompts only")
    ap.add_argument("--weight-scale", type=float, default=1.0, help="multiply every float weight (0 gives a degenerate model; used by a test)")
    ap.add_argument("--n-new", type=int, default=N_NEW, help="generated tokens per prompt (default 32, what stage 1 decodes; fewer only to save time in tests)")
    ap.add_argument("--reference-only", metavar="SNAPSHOT", help="record the reference for an existing snapshot, then refresh its manifest; write nothing else")
    args = ap.parse_args(argv)
    if args.reference_only:
        return reference_only(Path(args.reference_only), args.n_new)
    if not args.out:
        return refuse("--out is required", 2)
    try:
        return make(Path(args.out), args.variant, args.seed, args.layers, tokenizer=not args.no_tokenizer,
                    reference=not args.no_reference, weight_scale=args.weight_scale, n_new=args.n_new, width=args.width,
                    overrides=parse_overrides(args.set), max_gb=args.max_gb)
    except ValueError as exc:
        return refuse(str(exc), 2)


if __name__ == "__main__":
    sys.exit(main())
