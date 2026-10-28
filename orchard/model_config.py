# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Read a Hugging Face `config.json` and normalise the numbers that the decode floor (orchard/decode_floor.py) and the hazard scan
(orchard/hazards.py) use.

Model families name the same things differently (`num_experts`, `n_routed_experts`, `num_local_experts`). This module is the single
place that knows the names. Everything else works on `ModelDims`.

`read_model_config(path)` reads a file another program wrote, so it goes through `orchard/safeio.py` and raises only
`ModelConfigError` (docs/safe-reading.md). `dims(config)` takes a config that is already in memory and refuses a number that is
missing, a fraction, zero or negative, with a message that names the key. It does not guess: a config it cannot read raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from orchard import safeio

MAX_CONFIG_BYTES = 1 << 20          # a config.json is a few kilobytes; a megabyte is far beyond any real one
# Model types whose full-attention query projection is always twice as wide (an output gate) with no config key saying so.
# Found by summing the Qwen3-Coder-Next shard headers: q_proj holds 2 x heads x head_dim rows and its config has no attn_output_gate.
GATED_BY_TYPE = ("qwen3_next",)


class ModelConfigError(ValueError):
    """A model config could not be read or holds a value the harness cannot use. The message names the key and never holds file content."""


def read_model_config(path) -> dict:
    """The JSON object in the config file at `path`.

    Raises only ModelConfigError: a missing or unreadable file, a link, a directory or a FIFO, a file over MAX_CONFIG_BYTES, text
    that is not strict JSON (NaN, a repeated key, deep nesting) and a top-level value that is not an object are all refused with
    one sentence and no file content."""
    try:
        return safeio.read_json(path, max_bytes=MAX_CONFIG_BYTES, expect=dict)
    except safeio.SafeReadError as exc:
        raise ModelConfigError(f"model config not usable: {exc.reason}") from exc


def text_config(config) -> dict:
    """The language model's settings as one flat dict.

    A multimodal config keeps the language model under `text_config` and a few shared keys (`tie_word_embeddings`) at the top level.
    The text config wins on a clash; a key only the top level holds is kept."""
    if not isinstance(config, dict):
        raise ModelConfigError("model config must be a JSON object")
    inner = config.get("text_config")
    if isinstance(inner, dict):
        merged = {k: v for k, v in config.items() if k != "text_config"}
        merged.update(inner)
        return merged
    return dict(config)


@dataclass(frozen=True)
class ModelDims:
    """The numbers about a model that the floor and the scan use. A zero means the model has none of that thing."""
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied_embeddings: bool
    layer_types: tuple                      # one name per layer: full_attention, sliding_attention, linear_attention, ...
    moe_layers: tuple                       # one bool per layer: does the layer's MLP hold routed experts
    routed_experts: int = 0
    top_k: int = 0
    moe_intermediate: int = 0               # width of one routed expert
    shared_intermediate: int = 0            # total width of the shared expert or experts (0 when none)
    attn_output_gate: bool = False
    linear: dict = field(default_factory=dict)   # linear_attention shapes: key_heads, value_heads, key_dim, value_dim, conv_kernel
    mla: dict = field(default_factory=dict)      # multi-latent attention shapes, empty when the model has none
    keys: frozenset = frozenset()           # every key of the flat config, for detecting mixers that have no layer-type name


def _num(cfg: dict, key: str, *, default=None, lo: int = 1) -> int:
    """A whole number at `key` (at least `lo`). A missing key returns `default` when one is given and is an error otherwise."""
    if key not in cfg or cfg[key] is None:
        if default is not None:
            return default
        raise ModelConfigError(f"model config has no {key}")
    try:
        return safeio.whole_number(cfg[key], lo=lo, what=key)
    except safeio.BadValue as exc:
        raise ModelConfigError(f"model config {exc.reason}") from exc


def _first(cfg: dict, names: tuple[str, ...]):
    """The first of `names` that the config holds (a family's spelling of one setting), or None."""
    for n in names:
        if n in cfg and cfg[n] is not None:
            return n
    return None


def dims(config) -> ModelDims:
    """Normalise a config (a flat dict, or a multimodal one with a `text_config`) to ModelDims. Raises ModelConfigError."""
    cfg = text_config(config)
    hidden = _num(cfg, "hidden_size")
    layers = _num(cfg, "num_hidden_layers")
    heads = _num(cfg, "num_attention_heads")
    vocab = _num(cfg, "vocab_size")
    kv_heads = _num(cfg, "num_key_value_heads", default=heads)
    head_dim = _num(cfg, "head_dim", default=max(1, hidden // heads))

    expert_key = _first(cfg, ("num_experts", "n_routed_experts", "num_local_experts"))
    experts = _num(cfg, expert_key) if expert_key else 0
    top_key = _first(cfg, ("num_experts_per_tok", "num_experts_per_token", "top_k_experts"))
    top_k = _num(cfg, top_key) if (experts and top_key) else 0
    if experts and not top_k:
        raise ModelConfigError("model config has routed experts but no num_experts_per_tok")
    if top_k > experts:
        raise ModelConfigError("model config num_experts_per_tok is above the number of experts")

    moe_width = _num(cfg, "moe_intermediate_size", default=0, lo=1) if "moe_intermediate_size" in cfg else 0
    if "intermediate_size" in cfg:
        intermediate = _num(cfg, "intermediate_size")
    elif experts and moe_width:
        intermediate = moe_width
    else:
        raise ModelConfigError("model config has no intermediate_size")
    if experts and not moe_width:
        moe_width = intermediate                     # a family with no separate expert width (Mixtral) uses the MLP width

    shared = 0
    if "shared_expert_intermediate_size" in cfg and cfg["shared_expert_intermediate_size"] is not None:
        shared = _num(cfg, "shared_expert_intermediate_size", lo=0)
    elif "n_shared_experts" in cfg and cfg["n_shared_experts"] is not None and experts:
        shared = _num(cfg, "n_shared_experts", lo=0) * moe_width      # a count of experts, each the expert width

    layer_types = _layer_types(cfg, layers)
    moe_layers = _moe_layers(cfg, layers) if experts else (False,) * layers

    linear: dict = {}
    if any(t == "linear_attention" for t in layer_types):
        lk, lv = _num(cfg, "linear_num_key_heads"), _num(cfg, "linear_num_value_heads")
        lkd, lvd = _num(cfg, "linear_key_head_dim"), _num(cfg, "linear_value_head_dim")
        linear = {"key_heads": lk, "value_heads": lv, "key_dim": lk * lkd, "value_dim": lv * lvd,
                  "conv_kernel": _num(cfg, "linear_conv_kernel_dim", default=4)}

    mla: dict = {}
    if "kv_lora_rank" in cfg and cfg["kv_lora_rank"] is not None:
        mla = {"kv_lora_rank": _num(cfg, "kv_lora_rank"),
               "q_lora_rank": _num(cfg, "q_lora_rank", default=0, lo=0) if cfg.get("q_lora_rank") is not None else 0,
               "qk_nope_head_dim": _num(cfg, "qk_nope_head_dim"), "qk_rope_head_dim": _num(cfg, "qk_rope_head_dim"),
               "v_head_dim": _num(cfg, "v_head_dim")}

    return ModelDims(
        hidden=hidden, intermediate=intermediate, layers=layers, heads=heads, kv_heads=kv_heads, head_dim=head_dim, vocab=vocab,
        tied_embeddings=cfg.get("tie_word_embeddings") is True, layer_types=layer_types, moe_layers=moe_layers,
        routed_experts=experts, top_k=top_k, moe_intermediate=moe_width if experts else 0, shared_intermediate=shared if experts else 0,
        attn_output_gate=cfg.get("attn_output_gate") is True or cfg.get("model_type") in GATED_BY_TYPE, linear=linear, mla=mla, keys=frozenset(str(k) for k in cfg))


def _layer_types(cfg: dict, layers: int) -> tuple:
    """One mixer name per layer: the config's `layer_types` list, else `full_attention_interval` (every Nth layer is full attention
    and the rest are linear attention, the Qwen3-Next layout), else full attention everywhere."""
    given = cfg.get("layer_types")
    if given is not None:
        if not isinstance(given, list) or len(given) != layers or not all(isinstance(t, str) and t for t in given):
            raise ModelConfigError(f"model config layer_types must be a list of {layers} names")
        return tuple(given)
    if cfg.get("full_attention_interval") is not None:
        n = _num(cfg, "full_attention_interval")
        return tuple("full_attention" if (i + 1) % n == 0 else "linear_attention" for i in range(layers))
    return ("full_attention",) * layers


def _moe_layers(cfg: dict, layers: int) -> tuple:
    """Which layers hold routed experts. A layer is dense when it is below `first_k_dense_replace`, listed in `mlp_only_layers`,
    or not on the `decoder_sparse_step` stride."""
    first_dense = _num(cfg, "first_k_dense_replace", default=0, lo=0) if cfg.get("first_k_dense_replace") is not None else 0
    step = _num(cfg, "decoder_sparse_step", default=1)
    only = cfg.get("mlp_only_layers") or []
    if not isinstance(only, list) or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in only):
        raise ModelConfigError("model config mlp_only_layers must be a list of layer indexes")
    return tuple(i >= first_dense and i not in only and (i + 1) % step == 0 for i in range(layers))
