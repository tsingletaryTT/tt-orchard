#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Write a Qwen4-shaped "frankenmodel" checkpoint that carries REAL Qwen3.8-27B weights, and prove on CPU what it is equal to.

PROVISIONAL, like scripts/tiny_qwen4/make_checkpoint.py: no Qwen4 weights or config.json exist (2026-11-08). The Qwen4 shape here is the
public reading in docs/superpowers/specs/2026-11-02-tiny-qwen4-rehearsal-design.md: the Qwen3.8-27B skeleton (gated delta net and gated
attention in a 3:1 schedule, hidden 5120) plus hyper-connections (4 streams) and QSA indexers on the attention layers. See
docs/superpowers/specs/2026-11-08-qwen4-franken-design.md for what the result can and cannot show.

What it does. It takes the first K layers (default 8, a multiple of 4) of a Qwen3.8 snapshot: the embedding, those layers, the final
norm and the output head. It writes them as a `qwen4_exp_text` dense checkpoint that transformers' `Qwen4ExpForCausalLM` loads (with the
dense-layer shim of orchard/skills/reference-gate-templates/qwen4_dense.py). The parts Qwen3.8 does not have are set so that the new
model computes the same function as the base:

  * Hyper-connections. Streams start as four copies of the embedding. Each block reads the mean over streams of
    `sigmoid(up(...)) * hc_norm(stream)` and adds `2 * sigmoid(inject(...) / 4) * block_output` to every stream, and the final mixer
    reads the streams the same way. With `input_mix_weight_down`, `input_mix_weight_up` and `block_inject_weight` all zero, the mix
    weight is sigmoid(0) = 0.5 and the injection weight is 2 * sigmoid(0) = 1. The four streams then stay equal, and the block input is
    0.5 * rms_norm(x) * (1 + hc_norm.weight). Setting `hc_norm.weight = 1 + 2 * w` for the base norm weight `w` gives
    (1 + w) * rms_norm(x), the base block input. The base's `input_layernorm` and `post_attention_layernorm` become the two blocks'
    `hc_norm`, and the base's final `norm` becomes the final mixer's `hc_norm`. This is exact in real arithmetic. In bf16 the stored
    weight `1 + 2 * w` is rounded, so the match is exact only for a float32 checkpoint (`--dtype float32`); the proofs measure both.
  * The output gate activation is `silu` (the base calls it swish). `Qwen4ExpText` reads `output_gate_type` for the delta net's gated
    norm, and its default (`sigmoid`) is not the base's.
  * QSA indexers. A short prompt makes the indexer select every block, so the weights do not change the output while the prompt is
    shorter than the indexer budget plus one block. They are seeded random, or (`--indexers DIR`) the real tensors of the
    higa2515 release, mapped as `q_proj` + `k_proj` -> `index_qk_proj`, `q_norm` -> `q_layernorm`, `k_block_norm` -> `k_layernorm`.
    The release applies `k_norm` to the raw keys before pooling and Qwen4Exp does not, so the mapping has the right shapes and
    unverified behaviour past the budget.
  * No per-layer embedding, no n-gram table (`ple_layer_ids` is empty) and no MTP head: transformers' class has none.

`--hc-sigma S` (S > 0) draws the three hyper-connection weights of every block from N(0, S^2) with the seed, which breaks the equality
with the base on purpose: that is the checkpoint a hardware port has to get right.

Run it with the reference interpreter (torch and transformers 5.19.0, CPU only):

    python3 make_franken.py --out ~/orchard-runs/franken/k8 [--layers 8] [--dtype bfloat16|float32] [--hc-sigma 0.0] [--seed 1234]
                            [--base-snapshot DIR] [--indexers DIR] [--set indexer_budget=N ...] [--max-gb 14] [--prove]

`--prove` loads the checkpoint and the same K layers through the real Qwen3.5 class and writes `proofs.json` next to the weights: the
logit difference to the base on three prompts, the equality of two runs, and the indexer budget test. Nothing here imports ttnn or
ttml, opens a device or uses the network. Weights are read with safetensors (the files are the model's own) and everything else
is written with safeio.atomic_write.

Exit codes: 0 done; 2 refused before writing (bad option, snapshot, output directory, size); 3 a proof failed; 4 torch,
transformers or safetensors is not importable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
from orchard import safeio  # noqa: E402

HF_HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
BASE_REPO_DIR = "models--Qwen--Qwen3.8-27B"
GB = 1e9
DEFAULT_LAYERS = 8
DEFAULT_MAX_GB = 14.0
HC_COUNT = 4
INDEXER_RATIO = 4
MAX_CONFIG_BYTES = 1 << 20
PROMPT_LENGTHS = (8, 16, 33)
PROOF_FLOAT32_ABS_TOL = 5e-4      # float32: equal to this absolute logit difference and with every top-1 token equal
PROOF_BF16_REL_TOL = 0.05         # bf16: the stored norm weights are rounded, so equal only to this share of the largest logit (top-1 may flip on near ties)
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")
REQUIRED_DIMENSIONS = ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size", "num_hidden_layers",
                       "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim")
GATE_NAMES = {"swish": "silu", "silu": "silu", "sigmoid": "sigmoid"}
IDENTITY_NOTE = ("hyper-connection weights are zero (mix 0.5, injection 1) and hc_norm.weight = 1 + 2 * the base norm weight, so the "
                 "function equals the base model's; exact in float32, bf16 rounds the stored norm weight")


class FrankenRefused(ValueError):
    """The request cannot be honoured; nothing was written."""


class ProofFailed(Exception):
    """A proof did not hold; the output directory keeps what was written so far."""


def snap_file(snapshot: Path, name: str) -> Path:
    """The real path of `snapshot/name`. A Hugging Face snapshot stores every file as a link into the model's own `blobs/` directory, and
    that is followed. A link that leaves the snapshot and its cache's `blobs/` is refused (safeio.resolved_inside, issue 494)."""
    target = safeio.resolved_inside(snapshot / name, snapshot)
    if target is None:
        # Newer huggingface_hub keeps the bytes in a hub-wide store (<hub>/blobs/<xx>/<hash>) that a model's own blobs/ links to. That store
        # is accepted when the snapshot itself is inside the hub cache; any other target is refused.
        real = safeio.resolved(snapshot / name)
        hub = os.path.realpath(HF_HUB)
        shared = os.path.join(hub, "blobs")
        if real and os.path.realpath(snapshot).startswith(hub + os.sep) and real.startswith(shared + os.sep):
            target = real
    if target is None or not os.path.isfile(target):
        raise FrankenRefused(f"{snapshot / name} is missing or links outside the snapshot and its cache's blobs directory")
    return Path(target)


def say(message: str) -> None:
    print(f"make_franken: {message}", flush=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def helpers():
    """(make_checkpoint, qwen4_dense): the tiny generator's guards and the dense-layer shim, loaded by path."""
    mc = load_module("franken_make_checkpoint", HERE / "make_checkpoint.py")
    qd = load_module("franken_qwen4_dense", REPO / "orchard" / "skills" / "reference-gate-templates" / "qwen4_dense.py")
    return mc, qd


def import_stack():
    try:
        import torch
        import transformers
        import safetensors  # noqa: F401
    except ImportError as exc:
        print(f"make_franken: {exc}", file=sys.stderr)
        raise SystemExit(4)
    from transformers.models.qwen4_exp import modeling_qwen4_exp  # noqa: F401
    return torch, transformers


# ---- the base snapshot ------------------------------------------------------------------------------------------------------------

def default_snapshot(repo_dir: str = BASE_REPO_DIR) -> Path:
    """The one snapshot of the base in the hub cache that holds weight shards (a later `hf download --include '*.json'` makes a second, empty one)."""
    root = HF_HUB / repo_dir / "snapshots"
    snaps = sorted(d for d in root.glob("*") if d.is_dir() and (d / "model.safetensors.index.json").is_file() and any(d.glob("*.safetensors"))) \
        if root.is_dir() else []
    if len(snaps) != 1:
        raise FrankenRefused(f"{repo_dir}: expected exactly one snapshot with weights under {root}, found {len(snaps)}; pass --base-snapshot")
    return snaps[0]


def read_base_text_config(snapshot: Path) -> dict:
    """The text part of the snapshot's config.json, checked against the Qwen3.8-27B skeleton this script knows how to convert."""
    try:
        raw = safeio.read_json(snap_file(snapshot, "config.json"), max_bytes=MAX_CONFIG_BYTES)
    except safeio.SafeReadError as exc:
        raise FrankenRefused(f"{snapshot}/config.json cannot be read: {exc.reason if hasattr(exc, 'reason') else exc}") from None
    text = raw.get("text_config") if isinstance(raw, dict) else None
    text = text if isinstance(text, dict) else raw
    if not isinstance(text, dict) or text.get("model_type") not in ("qwen3_5_text", "qwen3_5"):
        raise FrankenRefused("the base config is not a qwen3_5 text config (the only skeleton this script converts)")
    for key in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size",
                "num_hidden_layers", "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim",
                "linear_value_head_dim", "linear_conv_kernel_dim", "rope_parameters"):
        if key not in text:
            raise FrankenRefused(f"the base config has no `{key}`")
    for key in REQUIRED_DIMENSIONS:
        if not safeio_whole(text[key]):
            raise FrankenRefused(f"the base config's `{key}` is not a positive whole number")
    if not isinstance(text["rope_parameters"], dict):
        raise FrankenRefused("the base config's `rope_parameters` is not a mapping")
    return text


def safeio_whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def check_layers(text: dict, layers) -> list[str]:
    """The first K layer types of the base, refused unless K is a multiple of 4 inside the base and the 3:1 schedule holds."""
    if not safeio_whole(layers) or layers % 4 != 0:
        raise FrankenRefused("--layers must be a positive multiple of 4 (the 3:1 schedule)")
    if layers > text["num_hidden_layers"]:
        raise FrankenRefused(f"--layers {layers} is more than the base's {text['num_hidden_layers']}")
    types = text.get("layer_types") or ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(text["num_hidden_layers"])]
    types = types[:layers]
    expect = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(layers)]
    if types != expect:
        raise FrankenRefused("the base's layer schedule is not 3 linear-attention layers then 1 full-attention layer")
    return types


# ---- the Qwen4 config -------------------------------------------------------------------------------------------------------------

def parse_sets(items) -> dict:
    out = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        if not sep or key not in ("indexer_budget", "indexer_head_dim", "indexer_n_heads", "hc_lowrank"):
            raise FrankenRefused(f"--set {item!r}: only indexer_budget, indexer_head_dim, indexer_n_heads and hc_lowrank are settable")
        try:
            number = int(value)
        except ValueError:
            raise FrankenRefused(f"--set {item!r}: the value must be a whole number") from None
        if number < 1 or number > 1 << 20:
            raise FrankenRefused(f"--set {item!r}: the value must be between 1 and {1 << 20}")
        out[key] = number
    return out


def franken_config(text: dict, layers: int, *, dtype: str, sets: dict | None = None) -> dict:
    """The raw `qwen4_exp_text` config of the franken. Widths come from the base; the Qwen4 parts follow the public reading."""
    sets = dict(sets or {})
    gate = GATE_NAMES.get(text.get("output_gate_type") or "swish")
    if gate is None:
        raise FrankenRefused(f"the base's output_gate_type {text.get('output_gate_type')!r} is not one Qwen4Exp supports")
    cfg = {
        "model_type": "qwen4_exp_text", "architectures": ["Qwen4ExpForCausalLM"], "dtype": dtype,
        "hidden_size": text["hidden_size"], "intermediate_size": text["intermediate_size"], "vocab_size": text["vocab_size"],
        "num_hidden_layers": layers, "num_attention_heads": text["num_attention_heads"], "num_key_value_heads": text["num_key_value_heads"],
        "head_dim": text["head_dim"], "linear_num_key_heads": text["linear_num_key_heads"], "linear_num_value_heads": text["linear_num_value_heads"],
        "linear_key_head_dim": text["linear_key_head_dim"], "linear_value_head_dim": text["linear_value_head_dim"],
        "linear_conv_kernel_dim": text["linear_conv_kernel_dim"], "full_attention_interval": 4,
        "hidden_act": text.get("hidden_act", "silu"), "rms_norm_eps": text.get("rms_norm_eps", 1e-6),
        "max_position_embeddings": text.get("max_position_embeddings", 4096), "rope_parameters": dict(text["rope_parameters"]),
        "attention_bias": bool(text.get("attention_bias", False)), "attention_dropout": 0.0, "tie_word_embeddings": False,
        "output_gate_type": gate, "bos_token_id": text.get("bos_token_id"), "eos_token_id": text.get("eos_token_id"),
        "pad_token_id": text.get("pad_token_id"),
        "hc_count": HC_COUNT, "hc_lowrank": sets.pop("hc_lowrank", max(1, text["hidden_size"] // 8)),
        "indexer_n_heads": sets.pop("indexer_n_heads", 4), "indexer_kv_heads": 1,
        "indexer_head_dim": sets.pop("indexer_head_dim", min(128, text["head_dim"])),
        "indexer_budget": sets.pop("indexer_budget", 2048), "indexer_compress_ratio": INDEXER_RATIO,
        "ple_layer_ids": [],
    }
    if cfg["indexer_budget"] % INDEXER_RATIO:
        raise FrankenRefused(f"indexer_budget must be a multiple of {INDEXER_RATIO}")
    return {k: v for k, v in cfg.items() if v is not None or k in ("pad_token_id",)}


def build_meta_model(torch, transformers, qd, cfg: dict):
    config = qd.prepare_config(transformers, cfg)
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    with torch.device("meta"), qd.dense_layers():
        return Qwen4ExpForCausalLM(config)


# ---- the tensors ------------------------------------------------------------------------------------------------------------------

def base_keys(layers: int) -> list[str]:
    """The base tensor names the franken needs (names inside the snapshot)."""
    keys = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]
    return keys, re.compile(rf"model\.language_model\.layers\.(\d+)\.(.+)")


def read_index(snapshot: Path) -> dict:
    """The base's `weight_map` ({tensor name: shard file name}), read once through one guard: a link outside the cache, invalid JSON, a
    non-object, a `weight_map` that is not a non-empty mapping of strings to strings are all the same refusal (exit 2)."""
    try:
        raw = safeio.read_json(snap_file(snapshot, "model.safetensors.index.json"), max_bytes=1 << 26)
    except safeio.SafeReadError as exc:
        raise FrankenRefused(f"the weight index of {snapshot.name} cannot be read: {getattr(exc, 'reason', exc)}") from None
    table = raw.get("weight_map") if isinstance(raw, dict) else None
    if not isinstance(table, dict) or not table or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in table.items()):
        raise FrankenRefused("the weight index has no `weight_map` of tensor names to shard file names")
    return table


def read_base_tensors(snapshot: Path, layers: int) -> dict:
    """{base name: tensor} for every tensor of the first `layers` layers, the embedding, the final norm and the head."""
    from safetensors import safe_open
    index = read_index(snapshot)
    fixed, layer_re = base_keys(layers)
    want = list(fixed)
    for name in index:
        m = layer_re.fullmatch(name)
        if m and int(m.group(1)) < layers:
            want.append(name)
    missing = [k for k in fixed if k not in index]
    if missing:
        raise FrankenRefused(f"the base index lacks {missing}")
    by_shard: dict[str, list[str]] = {}
    for name in want:
        by_shard.setdefault(index[name], []).append(name)
    out = {}
    for shard, names in sorted(by_shard.items()):
        with safe_open(str(snap_file(snapshot, shard)), framework="pt", device="cpu") as handle:
            for name in names:
                out[name] = handle.get_tensor(name)
    return out


def read_indexers(directory: Path, layers: int, layer_types: list[str]) -> dict:
    """{layer: {q_proj, k_proj, q_norm, k_block_norm}} from the higa2515 release's safetensors file (never a .pt file)."""
    from safetensors import safe_open
    path = snap_file(Path(directory), "qsa-indexers.safetensors")
    out: dict[int, dict] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        for layer in range(layers):
            if layer_types[layer] != "full_attention":
                continue
            stem = f"model.language_model.layers.{layer}.self_attn._qsa_indexer."
            need = {"q_proj": stem + "q_proj.weight", "k_proj": stem + "k_proj.weight", "q_norm": stem + "q_norm",
                    "k_block_norm": stem + "k_block_norm"}
            if not all(v in names for v in need.values()):
                raise FrankenRefused(f"{path} has no indexer tensors for layer {layer}")
            out[layer] = {k: handle.get_tensor(v) for k, v in need.items()}
    return out


def convert_tensors(torch, base: dict, cfg: dict, layer_types: list[str], *, hc_sigma: float, seed: int, indexers: dict | None = None):
    """The franken's state dict, in the dtype of the config. Returns (tensors, provenance) where provenance counts what is real, identity or seeded."""
    dtype = getattr(torch, cfg["dtype"])
    hidden, hc, low = cfg["hidden_size"], cfg["hc_count"], cfg["hc_lowrank"]
    gen = torch.Generator().manual_seed(seed)
    out: dict = {}
    prov = {"real": 0, "hc_identity": 0, "hc_seeded": 0, "hc_norm_derived": 0, "indexer_real": 0, "indexer_seeded": 0}

    def tile(weight):                      # (1 + w) of the base norm becomes 0.5 * (1 + hc_norm.weight) on each of the streams
        return (1.0 + 2.0 * weight.float()).repeat(hc).to(dtype)

    def normal(shape, sigma):
        return (torch.randn(shape, generator=gen) * sigma).to(dtype)

    def hc_block(prefix: str, norm_weight, combine: bool):
        out[prefix + ".hc_norm.weight"] = tile(norm_weight)
        prov["hc_norm_derived"] += 1
        shapes = {"input_mix_weight_down": (low, hc * hidden), "input_mix_weight_up": (hc * hidden, low)}
        if combine:
            shapes["block_inject_weight"] = (hc, hc * hidden)
        for name, shape in shapes.items():
            if hc_sigma > 0:
                out[f"{prefix}.{name}.weight"] = normal(shape, hc_sigma)
                prov["hc_seeded"] += 1
            else:
                out[f"{prefix}.{name}.weight"] = torch.zeros(shape, dtype=dtype)
                prov["hc_identity"] += 1

    layer_re = re.compile(r"model\.language_model\.layers\.(\d+)\.(.+)")
    for name in sorted(base):
        tensor = base[name]
        if name == "model.language_model.embed_tokens.weight":
            out["model.embed_tokens.weight"] = tensor.to(dtype)
            prov["real"] += 1
        elif name == "lm_head.weight":
            out["lm_head.weight"] = tensor.to(dtype)
            prov["real"] += 1
        elif name == "model.language_model.norm.weight":
            hc_block("model.hyper_connection_mixer", tensor, combine=False)
        else:
            m = layer_re.fullmatch(name)
            layer, rest = int(m.group(1)), m.group(2)
            if rest == "input_layernorm.weight":
                hc_block(f"model.layers.{layer}.attn_hyper_connection", tensor, combine=True)
            elif rest == "post_attention_layernorm.weight":
                hc_block(f"model.layers.{layer}.mlp_hyper_connection", tensor, combine=True)
            else:
                out[f"model.layers.{layer}.{rest}"] = tensor.to(dtype)
                prov["real"] += 1
    n_heads, head_dim = cfg["indexer_n_heads"], cfg["indexer_head_dim"]
    for layer, kind in enumerate(layer_types):
        if kind != "full_attention":
            continue
        stem = f"model.layers.{layer}.self_attn.indexer."
        real = (indexers or {}).get(layer)
        if real is not None:
            q, k = real["q_proj"], real["k_proj"]
            if tuple(q.shape) != (n_heads * head_dim, hidden) or tuple(k.shape) != (head_dim, hidden):
                raise FrankenRefused(f"layer {layer}: the release's indexer shapes {tuple(q.shape)}, {tuple(k.shape)} do not fit "
                                     f"indexer_n_heads {n_heads} x indexer_head_dim {head_dim}")
            out[stem + "index_qk_proj.weight"] = torch.cat([q, k], dim=0).to(dtype)
            out[stem + "q_layernorm.weight"] = real["q_norm"].to(dtype)
            out[stem + "k_layernorm.weight"] = real["k_block_norm"].to(dtype)
            prov["indexer_real"] += 3
        else:
            out[stem + "index_qk_proj.weight"] = normal(((n_heads + 1) * head_dim, hidden), 0.02)
            out[stem + "q_layernorm.weight"] = torch.zeros(head_dim, dtype=dtype)
            out[stem + "k_layernorm.weight"] = torch.zeros(head_dim, dtype=dtype)
            prov["indexer_seeded"] += 3
    return out, prov


def check_keys(model_keys, tensor_keys) -> None:
    """The converted tensors and the class's parameters must be the same set: nothing filled in by the loader, nothing left over."""
    a, b = set(model_keys), set(tensor_keys)
    if a != b:
        raise FrankenRefused(f"the converted tensors do not match the class: missing {sorted(a - b)[:6]}, unexpected {sorted(b - a)[:6]}")


def estimate_bytes(torch, base: dict | None, cfg: dict, layer_types: list[str], snapshot: Path | None = None, layers: int = 0) -> int:
    """Disk bytes of model.safetensors, from parameter counts (no tensor is loaded when `base` is None)."""
    per = 4 if cfg["dtype"] == "float32" else 2
    hidden, hc, low = cfg["hidden_size"], cfg["hc_count"], cfg["hc_lowrank"]
    total = 0
    if base is not None:
        total += sum(t.numel() for n, t in base.items() if not n.endswith("layernorm.weight") and n != "model.language_model.norm.weight")
    else:
        from safetensors import safe_open
        index = read_index(snapshot)
        fixed, layer_re = base_keys(layers)
        names = [n for n in index if n in fixed or ((m := layer_re.fullmatch(n)) and int(m.group(1)) < layers)]
        shards: dict[str, list[str]] = {}
        for n in names:
            shards.setdefault(index[n], []).append(n)
        for shard, ns in shards.items():
            with safe_open(str(snap_file(snapshot, shard)), framework="pt", device="cpu") as handle:
                for n in ns:
                    if n.endswith("layernorm.weight") or n == "model.language_model.norm.weight":
                        continue
                    shape = handle.get_slice(n).get_shape()
                    count = 1
                    for d in shape:
                        count *= d
                    total += count
    blocks = 2 * layers + 1
    total += blocks * (hc * hidden + 2 * low * hc * hidden) + 2 * layers * hc * hc * hidden
    full = sum(1 for k in layer_types if k == "full_attention")
    total += full * ((cfg["indexer_n_heads"] + 1) * cfg["indexer_head_dim"] * hidden + 2 * cfg["indexer_head_dim"])
    return total * per


# ---- writing ----------------------------------------------------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    return safeio.sha256_file(path, max_bytes=1 << 44)


def write_json(path: Path, data) -> None:
    safeio.atomic_write(path, json.dumps(data, indent=2, sort_keys=True) + "\n", mode=0o644)


def read_tokenizer(snapshot: Path) -> dict:
    """{file name: bytes} of the tokenizer files the snapshot has, resolved and read before the output directory exists, so a refused link
    leaves nothing behind."""
    files = {}
    for name in TOKENIZER_FILES:
        if (snapshot / name).exists():
            files[name] = safeio.read_bytes(snap_file(snapshot, name), max_bytes=1 << 28)
    return files


def write_tokenizer(out: Path, files: dict) -> list[str]:
    for name, data in files.items():
        safeio.atomic_write(out / name, data, mode=0o644)
    return list(files)


def make_prompts(vocab: int, seed: int) -> list[list[int]]:
    import random
    rng = random.Random(seed)
    low = 1000 if vocab > 2000 else 4
    return [[rng.randrange(low, min(vocab, low + 5000)) for _ in range(n)] for n in PROMPT_LENGTHS]


def inventory(out: Path) -> dict:
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "franken-manifest.json":
            files[str(path.relative_to(out))] = {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
    return files


def write_manifest(out: Path, cfg: dict, args, snapshot: Path, indexer_dir, prov: dict, layer_types: list[str], estimate: int, proofs=None) -> dict:
    manifest = {
        "kind": "qwen4-franken", "provisional": "built from public readings of an unreleased model; see the design note",
        "layers": cfg["num_hidden_layers"], "layer_types": layer_types, "seed": args.seed, "dtype": cfg["dtype"], "hc_sigma": args.hc_sigma,
        "sources": {"base_snapshot": snapshot.name, "base_repo_dir": snapshot.parent.parent.name,
                    "indexers": None if indexer_dir is None else {"repo_dir": Path(indexer_dir).parent.parent.name, "snapshot": Path(indexer_dir).name}},
        "real": "embedding, lm_head, every attention, delta-net and feed-forward tensor of the first K layers (copied from the base, cast to dtype)",
        "identity_initialised": IDENTITY_NOTE if args.hc_sigma == 0 else None,
        "seeded": ("hyper-connection weights drawn from N(0, hc_sigma^2)" if args.hc_sigma > 0 else None),
        "indexers": ("the higa2515 release's q_proj, k_proj, q_norm and k_block_norm (the mapping is unverified past the budget)"
                     if indexer_dir else "seeded random (N(0, 0.02)) projections and zero norm weights"),
        "not_carried": ["visual tower", "mtp head (transformers' Qwen4Exp has none)", "per-layer embedding and n-gram table (ple_layer_ids is empty)"],
        "tensor_counts": prov, "estimated_bytes": estimate, "config": cfg,
        "versions": {"python": platform.python_version(), "torch": args.versions["torch"], "transformers": args.versions["transformers"]},
        "files": inventory(out), "proofs": proofs,
    }
    write_json(out / "franken-manifest.json", manifest)
    return manifest


# ---- the proofs -------------------------------------------------------------------------------------------------------------------

def load_franken(torch, transformers, qd, out: Path, budget: int | None = None):
    """The franken as written, in the dtype it was written in. transformers turns `config.dtype` into a torch dtype, so the dtype is
    taken from the raw config file. `budget` replaces `indexer_budget` for the indexer proofs."""
    raw = safeio.read_json(out / "config.json", max_bytes=MAX_CONFIG_BYTES)
    if budget is not None:
        raw = dict(raw, indexer_budget=budget)
    dtype = getattr(torch, raw["dtype"])
    config = qd.prepare_config(transformers, raw)
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
    with qd.dense_layers():
        model = Qwen4ExpForCausalLM.from_pretrained(str(out), config=config, dtype=dtype)
    return model.eval()


def execution_dtypes(torch, franken, base) -> dict:
    """The dtype each model really runs in, for the proof record."""
    return {"franken": str(next(franken.parameters()).dtype), "base": str(next(base.parameters()).dtype)}


def load_base_model(torch, transformers, text: dict, base: dict, cfg: dict):
    """The same K layers through transformers' real Qwen3.5 class."""
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    keep = {k: v for k, v in text.items() if k not in ("model_type", "layer_types", "num_hidden_layers", "dtype")}
    config = Qwen3_5TextConfig(**{**keep, "num_hidden_layers": cfg["num_hidden_layers"]})
    dtype = getattr(torch, cfg["dtype"])
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)
    state = {}
    for name, tensor in base.items():
        state[name.replace("model.language_model.", "model.")] = tensor.to(dtype)
    model.load_state_dict(state, strict=True, assign=True)
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    model.model.rotary_emb = m.Qwen3_5TextRotaryEmbedding(config=config)      # its buffer is not in the state dict, so it is still on the meta device
    for name, buf in model.named_buffers():
        if buf.is_meta:
            raise ProofFailed(f"the base model still has a meta buffer: {name}")
    return model.to(dtype).eval()


def logits_of(torch, model, prompts):
    outs = []
    with torch.no_grad():
        for ids in prompts:
            outs.append(model(torch.tensor([ids]), use_cache=False).logits[0].float())
    return outs


def compare(torch, a, b, prompts) -> dict:
    rows = []
    for ids, x, y in zip(prompts, a, b):
        rows.append({"prompt_tokens": len(ids), "max_abs_diff": float((x - y).abs().max()), "max_abs_logit": float(x.abs().max()),
                     "top1_equal": bool((x.argmax(-1) == y.argmax(-1)).all()), "last_top1_equal": bool(x[-1].argmax() == y[-1].argmax())})
    return {"prompts": rows, "max_abs_diff": max(r["max_abs_diff"] for r in rows), "top1_equal_all": all(r["top1_equal"] for r in rows)}


def within(result: dict, dtype: str) -> bool:
    """Whether a comparison with the base is inside the rule for `dtype` (see PROOF_FLOAT32_ABS_TOL and PROOF_BF16_REL_TOL)."""
    biggest = max(r["max_abs_logit"] for r in result["prompts"])
    if dtype == "float32":
        return result["max_abs_diff"] <= PROOF_FLOAT32_ABS_TOL and result["top1_equal_all"]
    return result["max_abs_diff"] <= PROOF_BF16_REL_TOL * biggest


def all_selected_budget(longest: int) -> int:
    """A budget at which the indexer keeps every block of a prompt of `longest` tokens (the blocks that fit plus the partial tail)."""
    blocks = -(-longest // INDEXER_RATIO) + 1
    return (blocks + 1) * INDEXER_RATIO


def run_proofs(torch, transformers, qd, out: Path, text: dict, base: dict, cfg: dict, hc_sigma: float) -> dict:
    """Four proofs. The identity and perturbation claims are judged on the model with every block selected, so that a small configured
    budget (sparse attention) cannot stand in for the claim. The wide/narrow indexer comparison always runs."""
    prompts = make_prompts(cfg["vocab_size"], 7)
    longest = max(len(p) for p in prompts)
    dtype = cfg["dtype"]
    franken = load_franken(torch, transformers, qd, out)
    first = logits_of(torch, franken, prompts)
    second = logits_of(torch, franken, prompts)
    base_model = load_base_model(torch, transformers, text, base, cfg)
    proofs = {"prompt_lengths": [len(p) for p in prompts], "dtype": dtype, "hc_sigma": hc_sigma,
              "execution_dtypes": execution_dtypes(torch, franken, base_model),
              "deterministic": all(bool(torch.equal(x, y)) for x, y in zip(first, second))}
    reference = logits_of(torch, base_model, prompts)
    del base_model
    configured = compare(torch, first, reference, prompts)
    biggest = max(r["max_abs_logit"] for r in configured["prompts"])
    configured["tolerance"] = PROOF_FLOAT32_ABS_TOL if dtype == "float32" else PROOF_BF16_REL_TOL * biggest
    configured["tolerance_rule"] = ("absolute, and every top-1 equal" if dtype == "float32"
                                    else f"{PROOF_BF16_REL_TOL} of the largest logit; top-1 may flip on near ties")
    proofs["against_base"] = configured                       # the franken as configured (its own indexer budget)
    every = all_selected_budget(longest)
    if cfg["indexer_budget"] >= every:
        all_logits, all_budget = first, cfg["indexer_budget"]
    else:
        del franken
        sparse = load_franken(torch, transformers, qd, out, budget=every)
        all_logits, all_budget = logits_of(torch, sparse, prompts), every
        del sparse
    selected = compare(torch, all_logits, reference, prompts)
    selected["tolerance"], selected["tolerance_rule"] = configured["tolerance"], configured["tolerance_rule"]
    proofs["against_base_all_selected"] = selected
    equal = within(selected, dtype)
    if hc_sigma == 0:
        proofs["identity_holds"] = equal
    else:
        proofs["perturbed_differs"] = not equal
    narrow = INDEXER_RATIO * 2
    narrow_model = load_franken(torch, transformers, qd, out, budget=narrow)
    narrow_logits = logits_of(torch, narrow_model, prompts)
    del narrow_model
    versus = compare(torch, all_logits, narrow_logits, prompts)
    proofs["indexer"] = {"configured_budget": cfg["indexer_budget"], "all_selected_budget": all_budget, "longest_prompt": longest,
                         "narrow_budget": narrow, "narrow_vs_all_selected_max_abs_diff": versus["max_abs_diff"],
                         "narrow_differs": versus["max_abs_diff"] > PROOF_FLOAT32_ABS_TOL, "all_selected_equals_base": equal}
    return proofs


def proofs_pass(proofs: dict) -> bool:
    """True only when every proof was run and held. A proof that is missing is a failure."""
    indexer = proofs.get("indexer") or {}
    if proofs.get("deterministic") is not True or indexer.get("narrow_differs") is not True:
        return False
    if "identity_holds" in proofs:
        return proofs["identity_holds"] is True and indexer.get("all_selected_equals_base") is True
    if "perturbed_differs" in proofs:
        return proofs["perturbed_differs"] is True
    return False


# ---- command line -----------------------------------------------------------------------------------------------------------------

def make(args) -> int:
    mc, qd = helpers()
    torch, transformers = import_stack()
    args.versions = {"torch": torch.__version__, "transformers": transformers.__version__}
    snapshot = Path(args.base_snapshot) if args.base_snapshot else default_snapshot()
    if not snapshot.is_dir():
        raise FrankenRefused(f"{snapshot} is not a directory")
    text = read_base_text_config(snapshot)
    layer_types = check_layers(text, args.layers)
    if args.dtype not in ("bfloat16", "float32"):
        raise FrankenRefused("--dtype is bfloat16 or float32")
    if not (0.0 <= args.hc_sigma <= 1.0):          # also refuses nan, which compares false with everything
        raise FrankenRefused("--hc-sigma must be a number between 0 and 1")
    cfg = franken_config(text, args.layers, dtype=args.dtype, sets=parse_sets(args.set))
    out = mc.check_output_dir(Path(args.out))
    indexer_dir = Path(args.indexers) if args.indexers else None
    estimate = estimate_bytes(torch, None, cfg, layer_types, snapshot, args.layers)
    say(f"estimated size {estimate / GB:.1f} GB for {args.layers} layers in {args.dtype}")
    mc.check_size(estimate, args.max_gb, mc.free_bytes(out), where=f" under {out}")
    tokenizer_files = read_tokenizer(snapshot)
    meta = build_meta_model(torch, transformers, qd, cfg)
    say("reading the base tensors")
    base = read_base_tensors(snapshot, args.layers)
    indexers = read_indexers(indexer_dir, args.layers, layer_types) if indexer_dir else None
    tensors, prov = convert_tensors(torch, base, cfg, layer_types, hc_sigma=args.hc_sigma, seed=args.seed, indexers=indexers)
    check_keys(meta.state_dict().keys(), tensors.keys())
    for key, value in meta.state_dict().items():
        if tuple(value.shape) != tuple(tensors[key].shape):
            raise FrankenRefused(f"{key}: the class wants {tuple(value.shape)}, the converted tensor is {tuple(tensors[key].shape)}")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", {**cfg, "seed": args.seed})
    write_json(out / "generation_config.json", {"do_sample": False, "eos_token_id": cfg.get("eos_token_id"), "bos_token_id": cfg.get("bos_token_id")})
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "model.safetensors"), metadata={"format": "pt"})
    del tensors
    copied = write_tokenizer(out, tokenizer_files)
    say(f"wrote {out} (tokenizer files: {', '.join(copied) or 'none'})")
    proofs = None
    if args.prove:
        say("running the proofs on the CPU")
        proofs = run_proofs(torch, transformers, qd, out, text, base, cfg, args.hc_sigma)
        write_json(out / "proofs.json", proofs)
    write_manifest(out, cfg, args, snapshot, indexer_dir, prov, layer_types, estimate, proofs)
    if proofs is not None:
        ok = proofs_pass(proofs)
        if not ok:
            print(json.dumps(proofs, indent=2), file=sys.stderr)
            return 3
        say("the proofs hold")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Write a Qwen4-shaped frankenmodel checkpoint with real Qwen3.8-27B weights.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--hc-sigma", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--base-snapshot")
    ap.add_argument("--indexers")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--max-gb", type=float, default=DEFAULT_MAX_GB)
    ap.add_argument("--prove", action="store_true")
    args = ap.parse_args(argv)
    try:
        return make(args)
    except SystemExit as exc:                        # the tiny generator's output-directory guard exits with 2 after printing why
        return exc.code if isinstance(exc.code, int) else 2
    except FrankenRefused as exc:
        print(f"make_franken: refused: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:                      # the tiny generator's own refusals (output directory, size)
        print(f"make_franken: refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
