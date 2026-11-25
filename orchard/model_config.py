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
GATED_BY_TYPE = ("qwen3_next", "inkling2_loop_text")
# inkling2_loop_text (Inkling-2-Mini): the safetensors header of thinkingmachines/Inkling-2-Mini lists `attn.gate_attn.weight` as [3072, 2304] in
# every layer, the same size as `wq_du.weight`, and its config has no key for it. Counting the query twice as wide gives the same parameter total.
# Granite MoE families: every layer has a `shared_mlp` of `shared_intermediate_size`; `intermediate_size` is the routed expert width
# (or unused, when the expert count is 0).
SHARED_MLP_TYPES = ("granitemoehybrid", "granitemoeshared")
# Model types whose MLP has two projections (up and down) and no gate. Every other type is counted with three (gate, up, down).
# The list is by model_type because the config has no key for it; a two-projection family that is missing here is overcounted by a third.
TWO_PROJECTION_MLP_TYPES = ("gpt_neox", "gpt2", "gptj", "gpt_bigcode", "opt", "bloom", "falcon", "bert", "roberta", "phi", "mpt", "codegen")
# The most layers a config may claim. The largest real models have a few hundred; the bound keeps a hostile number from allocating
# the per-layer tuples below.
MAX_LAYERS = 10_000
# The most passes a `loop_config` may claim over its block. The real model runs its block 3 times.
MAX_LOOP_PASSES = 64


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
    mamba: dict = field(default_factory=dict)    # Mamba-2 mixer shapes (heads, head_dim, state, groups, conv_kernel), empty when the model has none
    mla: dict = field(default_factory=dict)      # multi-latent attention shapes, empty when the model has none
    expert_hidden: int = 0                  # the width the routed experts work in when it is not the hidden size (Hy4-flash latent MoE: 1,536 against 3,072), else 0
    swa_heads: int = 0                      # query heads of the sliding-window layers when the config gives them apart from `num_attention_heads` (Hy4-flash: 64 against 32), else 0
    mlp_projections: int = 3                # matrices per MLP or expert: 3 for a gated MLP, 2 for the families in TWO_PROJECTION_MLP_TYPES
    keys: frozenset = frozenset()           # every key of the flat config whose value is not null, for detecting mixers that have no layer-type name
    loop_passes: tuple = ()                 # how many times each layer runs for one token (all 1 without a `loop_config`); () only on a hand-built ModelDims
    loop_variable: bool = False             # the config says the pass count varies per token (`loop_config.is_varloop`): loop_passes holds the maximum


def _num(cfg: dict, key: str, *, default=None, lo: int = 1, hi: int | None = None) -> int:
    """A whole number at `key` (from `lo` to `hi`). A missing key returns `default` when one is given and is an error otherwise."""
    if key not in cfg or cfg[key] is None:
        if default is not None:
            return default
        raise ModelConfigError(f"model config has no {key}")
    try:
        return safeio.whole_number(cfg[key], lo=lo, hi=hi, what=key)
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
    layers = _num(cfg, "num_hidden_layers", hi=MAX_LAYERS)
    heads = _num(cfg, "num_attention_heads")
    vocab = _num(cfg, "vocab_size")
    kv_heads = _kv_heads(cfg, heads)
    mla = _mla(cfg)
    # multi-latent attention heads are as wide as the query/key (nope + rope) parts. A `head_dim` key does not change that: Hy4-flash's config
    # carries head_dim 64, which is its rope part (qk_head_dim is 256). Other models take the key, or split the hidden size.
    head_dim = (mla["qk_nope_head_dim"] + mla["qk_rope_head_dim"]) if mla else _num(cfg, "head_dim", default=max(1, hidden // heads))
    swa_heads = _swa_heads(cfg, mla)
    expert_hidden = _num(cfg, "routed_expert_hidden_size", default=0, lo=1) if cfg.get("routed_expert_hidden_size") is not None else 0

    expert_key = _first(cfg, ("num_experts", "n_routed_experts", "num_local_experts"))
    experts = _num(cfg, expert_key, lo=0) if expert_key else 0      # a count of 0 (Granite-4.0-H dense sizes) says there are none
    top_key = _first(cfg, ("num_experts_per_tok", "num_experts_per_token", "top_k_experts"))
    top_k = _num(cfg, top_key, lo=0) if (experts and top_key) else 0
    if experts and not top_k:
        raise ModelConfigError("model config has routed experts but no num_experts_per_tok")
    if top_k > experts:
        raise ModelConfigError("model config num_experts_per_tok is above the number of experts")

    moe_width = _num(cfg, "moe_intermediate_size", default=0, lo=1) if "moe_intermediate_size" in cfg else 0
    if cfg.get("model_type") in SHARED_MLP_TYPES and not experts and cfg.get("shared_intermediate_size") is not None:
        intermediate = _num(cfg, "shared_intermediate_size")        # the dense Granite hybrid's only MLP is its shared_mlp
    elif "intermediate_size" in cfg:
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
    elif cfg.get("model_type") in SHARED_MLP_TYPES and cfg.get("shared_intermediate_size") is not None:
        shared = _num(cfg, "shared_intermediate_size", lo=0)        # Granite MoE: a shared MLP of this total width beside the experts
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

    mamba: dict = {}
    if any(t == "mamba" for t in layer_types):
        mamba = {"heads": _num(cfg, "mamba_n_heads"), "head_dim": _num(cfg, "mamba_d_head"), "state": _num(cfg, "mamba_d_state"),
                 "groups": _num(cfg, "mamba_n_groups", default=1), "conv_kernel": _num(cfg, "mamba_d_conv", default=4)}

    loop_passes, loop_variable = _loop_passes(cfg, layers)

    return ModelDims(
        hidden=hidden, intermediate=intermediate, layers=layers, heads=heads, kv_heads=kv_heads, head_dim=head_dim, vocab=vocab,
        tied_embeddings=cfg.get("tie_word_embeddings") is True, layer_types=layer_types, moe_layers=moe_layers,
        routed_experts=experts, top_k=top_k, moe_intermediate=moe_width if experts else 0, shared_intermediate=shared if experts else 0,
        attn_output_gate=cfg.get("attn_output_gate") is True or cfg.get("model_type") in GATED_BY_TYPE, linear=linear, mamba=mamba, mla=mla, swa_heads=swa_heads, expert_hidden=expert_hidden if expert_hidden != hidden else 0, mlp_projections=2 if cfg.get("model_type") in TWO_PROJECTION_MLP_TYPES else 3, keys=frozenset(str(k) for k, v in cfg.items() if v is not None),
        loop_passes=loop_passes, loop_variable=loop_variable)


def _loop_key(loop: dict, key: str, *, lo: int, hi: int) -> int:
    if loop.get(key) is None:
        raise ModelConfigError(f"model config loop_config has no {key}")
    try:
        return safeio.whole_number(loop[key], lo=lo, hi=hi, what=f"loop_config {key}")
    except safeio.BadValue as exc:
        raise ModelConfigError(f"model config {exc.reason}") from exc


def _loop_passes(cfg: dict, layers: int) -> tuple[tuple, bool]:
    """(how many times each layer runs for one token, whether that count varies). A config with `loop_config` runs layers
    `core_start` up to but not including `core_end` `num_passes` times each (Inkling-2-Mini: 6, 18, 3: 12 layers three times, 48
    applications for 24 layers). The weights are stored once. A config without the key, or with it set to null, runs every layer once.
    `is_varloop` true means the pass count varies per token; the count here is then the largest."""
    loop = cfg.get("loop_config")
    if loop is None:
        return (1,) * layers, False
    if not isinstance(loop, dict):
        raise ModelConfigError("model config loop_config must be an object")
    start = _loop_key(loop, "core_start", lo=0, hi=layers - 1)
    end = _loop_key(loop, "core_end", lo=1, hi=layers)
    passes = _loop_key(loop, "num_passes", lo=1, hi=MAX_LOOP_PASSES)
    if start >= end:
        raise ModelConfigError("model config loop_config core_start must be below core_end")
    varloop = loop.get("is_varloop")
    if varloop is not None and not isinstance(varloop, bool):
        raise ModelConfigError("model config loop_config is_varloop must be true or false")
    return tuple(passes if start <= i < end else 1 for i in range(layers)), bool(varloop)


KV_HEAD_KEYS = ("num_key_value_heads", "num_kv_heads", "n_head_kv", "multi_query_group_num")   # a family's spelling of the KV-head count


def _kv_heads(cfg: dict, heads: int) -> int:
    """The number of key-value heads: the first of KV_HEAD_KEYS the config holds, else 1 when `multi_query` is true (Falcon's legacy
    layout and GPT-BigCode), else one per query head."""
    key = _first(cfg, KV_HEAD_KEYS)
    if key:
        return _num(cfg, key)
    return 1 if cfg.get("multi_query") is True else heads


# Model types whose code sets the value head width from the hidden size and the head count, so their config has no `v_head_dim`
# (MiniCPM3: `self.v_head_dim = config.hidden_size // config.num_attention_heads` in modeling_minicpm.py; its kv_b_proj is
# kv_lora_rank by heads * (qk_nope_head_dim + v_head_dim), which confirms 64 for openbmb/MiniCPM3-4B).
V_HEAD_FROM_HIDDEN_TYPES = ("minicpm3",)


def _mla(cfg: dict) -> dict:
    """Multi-latent attention shapes, or an empty dict for a model without `kv_lora_rank`."""
    if cfg.get("kv_lora_rank") is None:
        return {}
    if cfg.get("v_head_dim") is None and cfg.get("model_type") in V_HEAD_FROM_HIDDEN_TYPES:
        cfg = dict(cfg, v_head_dim=_num(cfg, "hidden_size") // _num(cfg, "num_attention_heads"))
    return {"kv_lora_rank": _num(cfg, "kv_lora_rank"), "q_lora_rank": _num(cfg, "q_lora_rank", default=0, lo=0),
            "qk_nope_head_dim": _num(cfg, "qk_nope_head_dim"), "qk_rope_head_dim": _num(cfg, "qk_rope_head_dim"),
            "v_head_dim": _num(cfg, "v_head_dim")}


def _swa_heads(cfg: dict, mla: dict) -> int:
    """The query heads of the sliding-window layers of an MLA model that gives them apart from `num_attention_heads` (`swa_num_attention_heads`),
    or 0. The sliding-window layers must then use the same latent shapes as the others (`swa_qk_nope_head_dim`, `swa_qk_rope_head_dim` and
    `swa_v_head_dim` equal the plain ones when present): the counting reads one set of shapes, so a model whose sliding layers differ is refused
    instead of being counted with the wrong numbers."""
    if not mla or cfg.get("swa_num_attention_heads") is None:
        return 0
    for swa_key, plain_key in (("swa_qk_nope_head_dim", "qk_nope_head_dim"), ("swa_qk_rope_head_dim", "qk_rope_head_dim"), ("swa_v_head_dim", "v_head_dim")):
        if cfg.get(swa_key) is not None and _num(cfg, swa_key) != mla[plain_key]:
            raise ModelConfigError(f"model config {swa_key} differs from {plain_key}: the counting reads one set of latent shapes for the sliding-window layers too")
    return _num(cfg, "swa_num_attention_heads")


def engram_table(cfg: dict) -> dict | None:
    """The Engram memory of a config (Hy4-flash: hashed 2-gram and 3-gram embedding tables kept in host memory), or None when the config has no
    `engram_*` table description or the description cannot be read. `cfg` is the language model's settings (`text_config`).

    Rows are the sum of every head's vocabulary over every layer and n-gram order in `engram_layer_head_vocab_sizes`; one row holds
    `engram_n_embed_per_ngram / engram_n_head_per_ngram` values. With `engram_fp8_embedding` a value takes one byte and each row carries a bf16 scale
    (two bytes); without it a value takes two bytes. These are the config's own sizes, not a measurement of a loaded model."""
    tables = cfg.get("engram_layer_head_vocab_sizes")
    if not isinstance(tables, dict) or not tables:
        return None
    try:
        embed, heads = _num(cfg, "engram_n_embed_per_ngram"), _num(cfg, "engram_n_head_per_ngram")
        if embed % heads:
            return None
        rows, orders, per_order = 0, set(), set()
        for orders_of_layer in tables.values():
            if not isinstance(orders_of_layer, list) or not orders_of_layer:
                return None
            orders.add(len(orders_of_layer))
            for head_vocabs in orders_of_layer:
                if not isinstance(head_vocabs, list) or not head_vocabs:
                    return None
                per_order.add(len(head_vocabs))
                for v in head_vocabs:
                    rows += safeio.whole_number(v, lo=1, hi=1 << 40, what="an engram head vocabulary")
    except (ModelConfigError, safeio.BadValue):
        return None
    values = embed // heads
    params = rows * values
    fp8 = cfg.get("engram_fp8_embedding") is True
    return {"layers": len(tables), "orders": orders.pop() if len(orders) == 1 else None, "heads_per_order": per_order.pop() if len(per_order) == 1 else None,
            "values_per_row": values, "rows": rows, "params": params, "fp8": fp8, "bytes": params + 2 * rows if fp8 else params * 2}


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
    return _window_layers(cfg, layers)


def _window_layers(cfg: dict, layers: int) -> tuple:
    """Layer types for a config with no `layer_types` and no `full_attention_interval`: full attention, except where a sliding window
    applies. A positive `sliding_window` is read by family: Mistral and Mixtral apply it to every layer; the Qwen2 and Qwen3 families
    apply it from layer `max_window_layers` on when `use_sliding_window` is true. A positive window on any other family is refused,
    because ignoring it would hide every sliding layer from the hazard scan."""
    window = cfg.get("sliding_window")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return ("full_attention",) * layers
    family = str(cfg.get("model_type", ""))
    if family in ("mistral", "mixtral"):
        return ("sliding_attention",) * layers
    if family.startswith(("qwen2", "qwen3")):
        if cfg.get("use_sliding_window") is not True:
            return ("full_attention",) * layers
        first = _num(cfg, "max_window_layers", default=0, lo=0)
        return tuple("sliding_attention" if i >= first else "full_attention" for i in range(layers))
    raise ModelConfigError(f"model config has a sliding_window but no layer_types, and the layout of model_type {family!r} is not known")


def _moe_layers(cfg: dict, layers: int) -> tuple:
    """Which layers hold routed experts. A layer is dense when it is below `first_k_dense_replace`, listed in `mlp_only_layers`,
    or not on the `decoder_sparse_step` stride."""
    first_dense = _num(cfg, "first_k_dense_replace", default=0, lo=0) if cfg.get("first_k_dense_replace") is not None else 0
    step = _num(cfg, "decoder_sparse_step", default=1)
    listed = cfg.get("mlp_only_layers") or []
    if not isinstance(listed, list) or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in listed):
        raise ModelConfigError("model config mlp_only_layers must be a list of layer indexes")
    only = set(listed)                       # one pass over the list; a membership test per layer on a list is quadratic
    return tuple(i >= first_dense and i not in only and (i + 1) % step == 0 for i in range(layers))
