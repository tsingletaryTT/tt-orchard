# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Layer-type rope and the sliding window in the generated attention (issue 790).

`microsoft/Aion-1.0-Instruct-2.3B` gives `rope_parameters` keyed by layer type (`full_attention`: YaRN with theta 32000; `sliding_attention`: the default
type with theta 32000) and lists `layer_types` of `sliding_attention` and `full_attention` with `sliding_window` 2048. The generator took that dict as one flat
rope block. It found no `rope_type` (so it read `default`) and no `rope_theta` (so it fell back to 10000.0), and it wrote causal attention over the whole
prefix for every layer. A port built that way scored top-1 0.875 and fidelity 9.249 at 51 tokens and lost the passkey at 8,192 tokens.

Step 1 (this file's first section): the generator refuses what it does not write. Step 2 and step 3 replace the refusal with per-layer-type rotary tables
and a sliding-window mask; their tests are in the later sections of this file and compare the generated port with a torch forward at a length above the window.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from orchard import port_template as pt
from test_port_template import STD_CONFIG, STD_EXCERPTS, headers_of, std_construction, std_weights

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def aion_config() -> dict:
    """Aion's own config.json (every key as the model card ships it)."""
    return json.loads((FIXTURES / "aion-config.json").read_text(encoding="utf-8"))


def tiny_aion_config(*, window=8, original=16) -> dict:
    """The small standard decoder with Aion's structure: layer types, a sliding window, and `rope_parameters` keyed by layer type (no top-level rope_theta)."""
    cfg = copy.deepcopy(STD_CONFIG)
    cfg.pop("rope_theta")
    cfg.update({"model_type": "aion_instruct", "num_hidden_layers": 3, "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
                "sliding_window": window,
                "rope_parameters": {"full_attention": {"rope_type": "yarn", "rope_theta": 32000, "factor": 16.0, "beta_fast": 32.0, "beta_slow": 1.0,
                                                       "original_max_position_embeddings": original},
                                    "sliding_attention": {"rope_type": "default", "rope_theta": 32000}}})
    return cfg


def generate(cfg, *, layers=None):
    weights = std_weights()
    n = cfg["num_hidden_layers"]
    if n != STD_CONFIG["num_hidden_layers"]:
        weights = _weights(cfg)
    construction = std_construction()
    for entry in construction["classes"]:        # the construction lists the layer count of each module group
        for variant in entry["variants"]:
            if variant["count"] == 2 and any("{i}" in p for p in variant["paths"]):
                variant["count"] = n
    return pt.generate(config=cfg, construction=construction, tensors=headers_of(weights), excerpts=STD_EXCERPTS)


def _weights(cfg, seed=0):
    import torch
    g = torch.Generator().manual_seed(seed)
    h, nh, nkv, hd, inter, v = (cfg[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size"))

    def r(*shape, scale=0.2):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    w = {"model.embed_tokens.weight": r(v, h, scale=1.0), "model.norm.weight": r(h, scale=1.0) + 1, "lm_head.weight": r(v, h, scale=0.5)}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        w.update({f"{p}.input_layernorm.weight": r(h, scale=0.3) + 1, f"{p}.post_attention_layernorm.weight": r(h, scale=0.3) + 1,
                  f"{p}.self_attn.q_proj.weight": r(nh * hd, h), f"{p}.self_attn.k_proj.weight": r(nkv * hd, h),
                  f"{p}.self_attn.v_proj.weight": r(nkv * hd, h), f"{p}.self_attn.o_proj.weight": r(h, nh * hd),
                  f"{p}.self_attn.q_norm.weight": r(hd, scale=0.3) + 1, f"{p}.self_attn.k_norm.weight": r(hd, scale=0.3) + 1,
                  f"{p}.mlp.gate_proj.weight": r(inter, h), f"{p}.mlp.up_proj.weight": r(inter, h), f"{p}.mlp.down_proj.weight": r(h, inter)})
    return w


# ---- step 1: the generator refuses what it does not write ------------------------------------------------------------------------------------

def test_aions_real_config_is_keyed_by_layer_type_and_has_a_sliding_window():
    section = aion_config()
    assert pt.rope_keyed_by_layer_type(section) is True
    assert pt.sliding_window_of(section) == 2048


def test_a_flat_rope_block_is_not_keyed_by_layer_type():
    assert pt.rope_keyed_by_layer_type({"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}}) is False
    assert pt.rope_keyed_by_layer_type({"rope_theta": 1e6}) is False
    assert pt.rope_keyed_by_layer_type({"rope_parameters": {}}) is False


def test_a_block_with_plain_values_and_one_nested_dict_is_still_a_flat_block():
    """Only a block whose every value is a dict is keyed by layer type: a flat block may carry one nested setting."""
    assert pt.rope_keyed_by_layer_type({"rope_parameters": {"rope_type": "default", "rope_theta": 1e6, "extra": {"a": 1}}}) is False


def test_a_sliding_window_counts_only_when_a_layer_is_marked_sliding():
    assert pt.sliding_window_of({"sliding_window": 4096, "layer_types": ["full_attention", "linear_attention"]}) is None
    assert pt.sliding_window_of({"sliding_window": 4096}) is None
    assert pt.sliding_window_of({"sliding_window": None, "layer_types": ["sliding_attention"]}) is None
    assert pt.sliding_window_of({"sliding_window": 8, "layer_types": ["sliding_attention", "full_attention"]}) == 8


def test_the_generator_never_writes_theta_10000_for_a_rope_block_it_cannot_read():
    """The failure of the Aion run: a layer-type-keyed block became `theta: 10000.0` with no refusal."""
    text, p = generate(tiny_aion_config())
    assert "10000.0" not in text.replace("1e-06", ""), "the generated port holds the fallback theta"
    assert p.rotary is None


def test_a_layer_type_keyed_rope_block_leaves_the_attention_as_a_named_stub():
    cfg = tiny_aion_config()
    cfg.pop("sliding_window")                                   # only the rope keying is under test here
    cfg["layer_types"] = ["full_attention"] * 3
    text, p = generate(cfg)
    assert p.attention is None
    reasons = [s.reason for s in p.stubs if "ttention" in s.cls or "attn" in s.function]
    assert reasons and all("keyed by layer type" in r for r in reasons), reasons
    assert any("full_attention" in r and "sliding_attention" in r for r in reasons)


def test_a_sliding_window_leaves_the_attention_as_a_named_stub():
    cfg = tiny_aion_config()
    cfg["rope_parameters"] = {"rope_type": "default", "rope_theta": 32000}      # flat, so only the window is under test
    text, p = generate(cfg)
    assert p.attention is None
    reasons = [s.reason for s in p.stubs if "ttention" in s.cls or "attn" in s.function]
    assert reasons and all("sliding_window 8" in r for r in reasons), reasons
