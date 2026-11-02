# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Safetensors headers built from a model config, for the residency tests (no weights, no network).

A real checkpoint of 360 GB cannot live in a test. The estimator reads only the header of each shard (an 8-byte little-endian length and
that many bytes of JSON), so a shard file that holds the header alone is enough. `write_snapshot` writes such files into a directory the
way a Hugging Face snapshot lays them out (model-0000N-of-0000M.safetensors plus an index), with the data offsets a real file would
carry.

`flash_next_tensors(config)` follows the tensor names and shapes in section 1 of docs/run-logs/2026-11-01-flash-next-graft-study.md (read
from the real checkpoint's 131 headers): fused routed experts, 128 n-gram shards in the PLE module of layer 1, a gated delta net or
QSA mixer per layer, a router and one shared expert per MoE block, a vision tower and one MTP layer with its own MoE. The small
tensors (gated-residual modules, indexer) are given plausible shapes: their exact sizes move the totals by less than a percent.
`dense_tensors(config)` builds a hybrid dense model of the shape of config qwen3.8-27b.json.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

ITEM = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "U8": 1, "I8": 1, "F8_E4M3": 1}


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def header_bytes(tensors: list[tuple[str, str, list[int]]]) -> bytes:
    """The header of one shard for `tensors` [(name, dtype, shape)], with consecutive data offsets, as the file format lays it out."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, dtype, shape in tensors:
        size = numel(shape) * ITEM[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    body = json.dumps(header).encode()
    return struct.pack("<Q", len(body)) + body


def write_snapshot(directory, tensors, *, shards: int = 3, config: dict | None = None) -> Path:
    """Write `tensors` into `shards` header-only files and an index in `directory`. Returns the directory."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    groups = [tensors[i::shards] for i in range(shards)]
    weight_map = {}
    for i, group in enumerate(groups, 1):
        name = f"model-{i:05d}-of-{shards:05d}.safetensors"
        (d / name).write_bytes(header_bytes(group))
        for t in group:
            weight_map[t[0]] = name
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    if config is not None:
        (d / "config.json").write_text(json.dumps(config))
    return d


def _gdn(p, hidden, kh, vh, hd):
    qkv = (kh * 2 + vh) * hd
    return [(f"{p}.linear_attn.in_proj_qkv.weight", "BF16", [qkv, hidden]),
            (f"{p}.linear_attn.in_proj_z.weight", "BF16", [vh * hd, hidden]),
            (f"{p}.linear_attn.in_proj_a.weight", "BF16", [vh, hidden]),
            (f"{p}.linear_attn.in_proj_b.weight", "BF16", [vh, hidden]),
            (f"{p}.linear_attn.conv1d.weight", "BF16", [qkv, 1, 4]),
            (f"{p}.linear_attn.A_log", "F32", [vh]), (f"{p}.linear_attn.dt_bias", "BF16", [vh]),
            (f"{p}.linear_attn.norm.weight", "F32", [hd]),
            (f"{p}.linear_attn.out_proj.weight", "BF16", [hidden, vh * hd])]


def _attn(p, hidden, heads, kv, hd):
    return [(f"{p}.self_attn.q_proj.weight", "BF16", [heads * hd * 2, hidden]),
            (f"{p}.self_attn.k_proj.weight", "BF16", [kv * hd, hidden]),
            (f"{p}.self_attn.v_proj.weight", "BF16", [kv * hd, hidden]),
            (f"{p}.self_attn.o_proj.weight", "BF16", [hidden, heads * hd]),
            (f"{p}.self_attn.q_norm.weight", "BF16", [hd]), (f"{p}.self_attn.k_norm.weight", "BF16", [hd])]


def _norms(p, hidden):
    return [(f"{p}.input_layernorm.weight", "BF16", [hidden]), (f"{p}.post_attention_layernorm.weight", "BF16", [hidden])]


def flash_next_tensors(config: dict) -> list[tuple[str, str, list[int]]]:
    """Tensor (name, dtype, shape) of a Flash-Next-shaped checkpoint from the text_config of tests/fixtures/model-configs/qwen3.8-flash-next.json."""
    c = config["text_config"]
    h, e, ff, sh = c["hidden_size"], c["num_experts"], c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
    pre = "model.language_model"
    out = [(f"{pre}.embed_tokens.weight", "BF16", [c["vocab_size"], h]), ("lm_head.weight", "BF16", [c["vocab_size"], h]),
           (f"{pre}.norm.weight", "BF16", [h])]

    def moe(p):
        return [(f"{p}.mlp.gate.weight", "BF16", [e, h]),
                (f"{p}.mlp.experts.gate_up_proj", "BF16", [e, 2 * ff, h]), (f"{p}.mlp.experts.down_proj", "BF16", [e, h, ff]),
                (f"{p}.mlp.shared_expert.gate_proj.weight", "BF16", [sh, h]), (f"{p}.mlp.shared_expert.up_proj.weight", "BF16", [sh, h]),
                (f"{p}.mlp.shared_expert.down_proj.weight", "BF16", [h, sh]), (f"{p}.mlp.shared_expert_gate.weight", "BF16", [1, h])]

    for i, kind in enumerate(c["layer_types"]):
        p = f"{pre}.layers.{i}"
        mixer = (_gdn(p, h, c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"]) if kind == "linear_attention"
                 else _attn(p, h, c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"])
                 + [(f"{p}.self_attn.indexer.index_qk_proj.weight", "BF16", [640, h])])
        hc = [(f"{p}.hc_{k}.weight", "BF16", [c["hc_count"], c["hc_lowrank"]]) for k in ("attn", "mlp", "out")]
        out += _norms(p, h) + mixer + moe(p) + hc
    shards = c["split_ngram_parts"]
    rows = c["ngram_vocab_size_base"] * 16 // shards + 12            # 2,500,012 rows per shard, as the study states
    for s in range(shards):
        out.append((f"{pre}.layers.1.ple.ngram_embeddings.{s}.weight", "BF16", [rows, 160]))
    out += [(f"{pre}.layers.1.ple.ngram_index.{k}", "I64", [8]) for k in range(3)]
    out += [(f"{pre}.layers.1.ple.proj.weight", "BF16", [h, h])]
    mp = "mtp.layers.0"                                                    # one full-attention layer with its own MoE and gated residual
    out += _norms(mp, h) + _attn(mp, h, c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]) + moe(mp)
    out += [("mtp.fc.weight", "BF16", [h, 2 * h])]
    for k in range(27):
        v = f"model.visual.blocks.{k}"
        out += [(f"{v}.attn.qkv.weight", "BF16", [3 * 1152, 1152]), (f"{v}.attn.proj.weight", "BF16", [1152, 1152]),
                (f"{v}.mlp.linear_fc1.weight", "BF16", [4304, 1152]), (f"{v}.mlp.linear_fc2.weight", "BF16", [1152, 4304]),
                (f"{v}.norm1.weight", "BF16", [1152])]
    return out


def dense_tensors(config: dict) -> list[tuple[str, str, list[int]]]:
    """A hybrid dense model (gated delta net layers with a full-attention layer every `full_attention_interval`) with a plain gated MLP."""
    c = config.get("text_config", config)
    h, ff = c["hidden_size"], c["intermediate_size"]
    pre = "model.language_model"
    out = [(f"{pre}.embed_tokens.weight", "BF16", [c["vocab_size"], h]), ("lm_head.weight", "BF16", [c["vocab_size"], h]),
           (f"{pre}.norm.weight", "BF16", [h])]
    for i in range(c["num_hidden_layers"]):
        p = f"{pre}.layers.{i}"
        full = (i + 1) % c["full_attention_interval"] == 0
        mixer = (_attn(p, h, c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]) if full
                 else _gdn(p, h, c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"]))
        out += _norms(p, h) + mixer + [(f"{p}.mlp.gate_proj.weight", "BF16", [ff, h]), (f"{p}.mlp.up_proj.weight", "BF16", [ff, h]),
                                       (f"{p}.mlp.down_proj.weight", "BF16", [h, ff])]
    return out
