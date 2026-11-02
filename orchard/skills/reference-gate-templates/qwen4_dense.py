# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Dense-layer shim for `qwen4_exp` / `qwen4_exp_text` configs. PROVISIONAL: no Qwen4 weights or config.json exist yet.

transformers 5.19.0 builds a mixture-of-experts block in every decoder layer of `qwen4_exp` and its config refuses
`num_experts <= 0`. A dense Qwen4 checkpoint (publicly read as a 27B model with a plain gated MLP) therefore has no upstream
reference. This file lets stage 1 (reference_gate.py) and the op inventory (ops_inventory.py) build one without editing the
installed transformers.

The layer rule comes from Intel's public vLLM patch (intel/llm-scaler build/v0.31.0): layer `i` is MoE when
`i not in mlp_only_layers`, `num_experts > 0` and `(i + 1) % decoder_sparse_step == 0`. Every other layer is dense. A
config key that is missing counts as: `mlp_only_layers` empty, `decoder_sparse_step` 1, `num_experts` 0 (a dense checkpoint is
expected to omit it or write 0; the transformers default of 512 is NOT used). All of that is an assumption until the real
config.json is read.

How it works, in order:
 1. `prepare_config` reads the raw config dict, works out which layers are dense, and (only when the model has no routed
    experts at all) writes 1-wide placeholder expert settings so transformers' config validation accepts it. It records the
    schedule on the text config as `orchard_dense_layers`.
 2. `dense_layers()` is a context manager. Inside it, `Qwen4ExpTextDecoderLayer.__init__` runs unchanged and then, for a layer
    listed in `orchard_dense_layers`, replaces `self.mlp` with the model's own `Qwen4ExpTextMLP` (width `intermediate_size`,
    activation `hidden_act`). A config with no `orchard_dense_layers` is built exactly as before. It works with
    `from_pretrained` as well as with `cls(config)`, because the swap happens while the layer is constructed.
 3. Nothing outside the context is changed, and the class is restored on exit.

Everything that does not need torch (the layer rule, `needs_dense_shim`) is plain Python so the tests can run it anywhere.
"""
from __future__ import annotations

import contextlib
import copy

QWEN4_TYPES = ("qwen4_exp", "qwen4_exp_text")
PROVISIONAL = ("provisional: built from public readings of an unreleased model (no weights or config.json exist); "
               "the layer schedule, intermediate_size and the MTP shape are assumptions")


def text_part(raw: dict) -> dict:
    """The dict that holds the language model's keys: `text_config` of a multimodal config, else the config itself."""
    inner = raw.get("text_config")
    return inner if isinstance(inner, dict) else raw


def layer_is_moe(text: dict, idx: int) -> bool:
    """Intel's rule. `text` is the text part of a raw config."""
    experts = text.get("num_experts")
    experts = 0 if experts is None else experts
    only = set(text.get("mlp_only_layers") or [])
    step = text.get("decoder_sparse_step") or 1
    return idx not in only and experts > 0 and (idx + 1) % step == 0


def dense_layer_list(text: dict) -> list[int]:
    """The indexes of the dense layers, in order."""
    layers = text.get("num_hidden_layers")
    if not isinstance(layers, int) or isinstance(layers, bool) or layers < 1:
        raise ValueError("config has no usable num_hidden_layers")
    return [i for i in range(layers) if not layer_is_moe(text, i)]


def is_qwen4(raw: dict) -> bool:
    return raw.get("model_type") in QWEN4_TYPES


def needs_dense_shim(raw: dict) -> bool:
    """True for a Qwen4-family config in which at least one layer is dense. A Flash-Next-style config (every layer MoE) is False."""
    return is_qwen4(raw) and bool(dense_layer_list(text_part(raw)))


def _has_experts(text: dict) -> bool:
    n = text.get("num_experts")
    return isinstance(n, int) and not isinstance(n, bool) and n > 0


def prepare_config(transformers, raw: dict):
    """A transformers config object for `raw`, ready to be built inside `dense_layers()`.

    A config with no routed experts gets placeholder expert settings (1 expert, top 1, width 1) because transformers refuses
    `num_experts <= 0`; they are never used, since every layer is dense. A config with some MoE layers keeps its own numbers.
    `orchard_dense_layers` (the schedule) and `orchard_dense_note` are set on the text config."""
    raw = copy.deepcopy(raw)
    text = text_part(raw)
    dense = dense_layer_list(text)
    if not _has_experts(text):
        text.update(num_experts=1, num_experts_per_tok=1, moe_intermediate_size=1, shared_expert_intermediate_size=1)
    for key in ("mlp_only_layers", "decoder_sparse_step"):
        text.pop(key, None)                  # not keys of the transformers config; the schedule travels as orchard_dense_layers
    config = transformers.AutoConfig.for_model(**raw)
    target = config.get_text_config() if hasattr(config, "get_text_config") else config
    target.orchard_dense_layers = dense
    target.orchard_dense_note = PROVISIONAL
    return config


@contextlib.contextmanager
def dense_layers():
    """Make `Qwen4ExpTextDecoderLayer` build a dense MLP for each layer in its config's `orchard_dense_layers`."""
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m
    layer_cls = m.Qwen4ExpTextDecoderLayer
    original = layer_cls.__init__

    def init(self, config, layer_idx):
        original(self, config, layer_idx)
        if layer_idx in (getattr(config, "orchard_dense_layers", None) or ()):
            self.mlp = m.Qwen4ExpTextMLP(config, intermediate_size=config.intermediate_size)
            self.orchard_dense = True

    layer_cls.__init__ = init
    try:
        yield
    finally:
        layer_cls.__init__ = original
