# 2026-11-10: the stage 2 port generator, a sample for the tiny Qwen4 rehearsal model

Issue 677, Part 1. `orchard/port_template.py` writes a first `port_model.py` for stage 2 of a full port from the checkpoint's config and
tensor headers, the effective construction (`effective_construction.py`'s JSON), stage 0's ops inventory, the reference excerpts and the
verified TTNN cheat sheet. The standard blocks are written through cheat sheet calls; each block the generator does not write is a
function whose body is a `# AGENT:` comment and a NotImplementedError. This page is the generator's output for the tiny Qwen4 rehearsal
checkpoint (`local/tiny-qwen4-dense`, 4 layers, hidden 64, vocab 384), produced on the CPU with

    python3 -m orchard.port_template generate --snapshot <snapshot dir> --construction <effective_construction.py output> \
        --inventory stages/0/ops-inventory.json --excerpts stages/2/reference_excerpts.md --out port_model.py

The construction JSON was produced by running the stage 2 template `effective_construction.py` under the reference interpreter (meta
device, no weights, no chip). The generator is not on the supervisor's path; the supervisor still copies the plain skeleton.

## What was generated and what is a stub

Generated (complete, through TTNN calls from the sheet): `load` (every float tensor on the device: 24 Linear weight patterns transposed
once and TILE bf16, the embedding table ROW_MAJOR bf16, the three plain head norms as [1, 1, 1, 32] with 1 added because
`Qwen4ExpTextRMSNorm.forward` multiplies by `(1.0 + self.weight)`, every other float tensor TILE bf16, the three int64 n-gram tables on the
host), `_embedding`, `_linear`, `_rms_norm`, `_mlp` (gate, up, silu, multiply, down), `lm_head`, `prefill`, `decode_step` (re-runs the
whole sequence, the idiom of the port that passed) and `close`.

Stubs (15), each named by its function, the reference class and the reason:

| function | reference class | reason |
|---|---|---|
| `_model_prelude` | Qwen4ExpTextModel | the text model has a child the flow does not know (`hyper_connection_mixer`); the reference widens the embedding to `hc_count` copies before the first layer |
| `_model_hyper_connection_mixer` | Qwen4ExpTextGatedResidual | a block at the end of the text model in place of a final norm |
| `_layer_A` (layers 0, 2) | Qwen4ExpTextDecoderLayer | children are not the standard pre-norm set; stubs: attn_hyper_connection, linear_attn, mlp_hyper_connection |
| `_A_attn_hyper_connection` | Qwen4ExpTextGatedResidual | stage 0 labelled `aten.flatten.using_ints` (candidate) under its `hc_norm` |
| `_A_linear_attn` | Qwen4ExpTextGatedDeltaNet | stage 0 labelled `aten.clone.default` (candidate) under it |
| `_A_mlp_hyper_connection` | Qwen4ExpTextGatedResidual | as `_A_attn_hyper_connection` |
| `_layer_B` (layer 1) | Qwen4ExpTextDecoderLayer | as `_layer_A`, plus `ple` |
| `_B_attn_hyper_connection`, `_B_linear_attn`, `_B_mlp_hyper_connection` | as above | as above |
| `_B_ple` | Qwen4ExpTextPLELayer | stage 0 labelled `aten.sum.dim_IntList` (candidate) under it |
| `_layer_C` (layer 3) | Qwen4ExpTextDecoderLayer | as `_layer_A`, with `self_attn` in place of `linear_attn` |
| `_C_attn_hyper_connection`, `_C_mlp_hyper_connection` | Qwen4ExpTextGatedResidual | as above |
| `_C_self_attn` | Qwen4ExpTextAttention | attention with a child beyond q, k, v, o and their norms: `indexer` (Qwen4ExpTextQSAIndexer) |

The `_mlp` helper is generated and every layer stub names it as a part (`_mlp(model, p + ".mlp", x)`), so the coder composes the layer
from generated helpers and its own stubs. The grouped norms (`hc_norm`, group_size 64) and the gated norm (`linear_attn.norm`) are
uploaded as plain TILE tensors and belong to their stubs.

## What was checked on the CPU

- `tests/test_port_template.py`: the generated port is imported with a torch-backed fake `ttnn` (`tests/fake_ttnn_compute.py`, the
  sheet's argument names, layouts checked) and random weights in the checkpoint's shapes; `load` places 95 tensors on the device and 3
  on the host, `prefill` runs the embedding and raises at `_model_prelude`, the first stub. For a standard decoder (a Qwen3-like fixture)
  the same path produces logits that match a plain torch forward to 1e-3 on one and on two devices.
- Every `ttnn.<name>` in the generated text is an entry of the verified sheet, is named in an entry's usage line, or is one of the ops the
  sheet's multiply entry says are called the same way (`ttnn.add`); every keyword passed is in the entry's signature
  (`port_template.sheet_problems` returns an empty list for this file).
- The CPU check (`port_template.check_port`) reports 2 reachable matmul-family calls (`ttnn.linear` in `_linear`) and 15 stubs.

## What was not checked

- Nothing here ran on a chip. The tt-metal source tree the sheet was verified against is not on this machine's path, so the calls were
  checked against the sheet's entries (names and keyword arguments). The bindings themselves were not read.
- `ttnn.rms_norm` on a `[1, 1, 1, 32]` TILE weight against a `[1, heads, S, 32]` input, `ttnn.embedding` with a ROW_MAJOR bf16 table and
  `layout=ttnn.TILE_LAYOUT`, and `ttnn.ConcatMeshToTensor(..., dim=0)` for a replicated read-back follow the sheet's notes and the idioms
  file; the first board run is where they are confirmed.
- The ordinary attention path (`_attention`: `ttnn.concat` of q, k, v, `nlp_create_qkv_heads`, `rotary_embedding_hf`,
  `scaled_dot_product_attention`, `nlp_concat_heads`) is generated only for a plain attention block (q_proj of `heads * head_dim` rows,
  whole-head rotary, default rope). The tiny Qwen4's attention has a gate fused into q_proj, partial rotary and an indexer, so it is a stub
  here and that path has run only against the fake.

## The generated port

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# generated-by: orchard/port_template.py
"""Generated first port of Qwen4ExpForCausalLM for stage 2 of a full port (orchard/port_template.py).

The standard blocks are written through TTNN calls from the verified cheat sheet (stages/2/cheat_sheet.md): weight upload in
load, the embedding, RMSNorm, the gated MLP, ordinary attention and lm_head. Each function whose body starts with `# AGENT:` is a
block the generator does not write: fill it in from the reference class named in stages/2/reference_excerpts.md and remove the
marker. The CPU check refuses a port that keeps a marker.

Stubs:
- _model_prelude: Qwen4ExpTextModel at model: Qwen4ExpTextModel has children the generated flow does not know: hyper_connection_mixer (Qwen4ExpTextGatedResidual); stage 0 labelled aten.diff.default (host) in model itself
- _layer_A: Qwen4ExpTextDecoderLayer at model.layers.{i}: stage 0 labelled aten.flatten.using_ints (candidate) in the layer itself; stubs: attn_hyper_connection (Qwen4ExpTextGatedResidual), linear_attn (Qwen4ExpTextGatedDeltaNet), mlp_hyper_connection (Qwen4ExpTextGatedResidual)
- _A_attn_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.attn_hyper_connection: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.attn_hyper_connection.hc_norm
- _A_linear_attn: Qwen4ExpTextGatedDeltaNet at model.layers.{i}.linear_attn: stage 0 labelled aten.clone.default (candidate) under model.layers.{i}.linear_attn
- _A_mlp_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.mlp_hyper_connection: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.mlp_hyper_connection.hc_norm
- _layer_B: Qwen4ExpTextDecoderLayer at model.layers.{i}: stage 0 labelled aten.flatten.using_ints (candidate) in the layer itself; stubs: attn_hyper_connection (Qwen4ExpTextGatedResidual), linear_attn (Qwen4ExpTextGatedDeltaNet), mlp_hyper_connection (Qwen4ExpTextGatedResidual), ple (Qwen4ExpTextPLELayer)
- _B_attn_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.attn_hyper_connection: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.attn_hyper_connection.hc_norm
- _B_linear_attn: Qwen4ExpTextGatedDeltaNet at model.layers.{i}.linear_attn: stage 0 labelled aten.clone.default (candidate) under model.layers.{i}.linear_attn
- _B_mlp_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.mlp_hyper_connection: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.mlp_hyper_connection.hc_norm
- _B_ple: Qwen4ExpTextPLELayer at model.layers.{i}.ple: stage 0 labelled aten.sum.dim_IntList (candidate) under model.layers.{i}.ple
- _layer_C: Qwen4ExpTextDecoderLayer at model.layers.{i}: stage 0 labelled aten.flatten.using_ints (candidate) in the layer itself; stubs: attn_hyper_connection (Qwen4ExpTextGatedResidual), mlp_hyper_connection (Qwen4ExpTextGatedResidual), self_attn (Qwen4ExpTextAttention)
- _C_attn_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.attn_hyper_connection: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.attn_hyper_connection.hc_norm
- _C_mlp_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.mlp_hyper_connection: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.mlp_hyper_connection.hc_norm
- _C_self_attn: Qwen4ExpTextAttention at model.layers.{i}.self_attn: Qwen4ExpTextAttention: attention with children beyond q, k, v, o and their norms: indexer (Qwen4ExpTextQSAIndexer)
- _model_hyper_connection_mixer: Qwen4ExpTextGatedResidual at model.hyper_connection_mixer: a Qwen4ExpTextGatedResidual block at the end of the text model that the generator does not write

Notes:
- the inventory lists at most a few modules per op, so a block it did not flag may still run an op labelled candidate or host
"""
from __future__ import annotations

import glob
import os
import re

import torch
import ttnn
from safetensors import safe_open

CONFIG = {'model_type': 'qwen4_exp_text', 'hidden_size': 64, 'num_attention_heads': 8, 'num_key_value_heads': 2, 'head_dim': 32, 'vocab_size': 384, 'linear_num_key_heads': 2, 'linear_num_value_heads': 6, 'linear_key_head_dim': 32, 'linear_value_head_dim': 32, 'linear_conv_kernel_dim': 4, 'full_attention_interval': 4, 'partial_rotary_factor': 0.25, 'max_position_embeddings': 512, 'tie_word_embeddings': False, 'output_gate_type': 'sigmoid', 'indexer_budget': 16, 'indexer_compress_ratio': 4, 'indexer_head_dim': 32, 'indexer_kv_heads': 1, 'indexer_n_heads': 4, 'hc_count': 4, 'hc_lowrank': 8, 'ngram_size': 3, 'ngram_vocab_size_base': 1000, 'heads_per_ngram': 2, 'split_ngram_parts': 4, 'ple_embed_dim': 64, 'ple_conv_kernel_size': 4, 'pad_token_id': 0, 'eos_token_id': 1, 'bos_token_id': 2, 'num_hidden_layers': 4, 'intermediate_size': 256, 'hidden_act': 'silu', 'seed': 1234}
N_LAYERS = 4
HIDDEN = 64
VOCAB = 384
PAD_ID = 0
PAD_TO = 32                       # the sequence is padded to a tile multiple; the logits are read at the last real position
LAYER_VARIANT = {0: 'A', 1: 'B', 2: 'A', 3: 'C'}
# Every tensor pattern of the checkpoint and how load() places it: linear_weight (transposed once, TILE bf16), linear_bias, embedding
# (ROW_MAJOR bf16 table), norm and norm_plus_one ([1, 1, 1, dim] TILE bf16), tile (any other float tensor), host (integer tensors stay in torch).
PLACEMENT = {
    'lm_head.weight': 'linear_weight',
    'model.embed_tokens.weight': 'embedding',
    'model.hyper_connection_mixer.hc_norm.weight': 'tile',
    'model.hyper_connection_mixer.input_mix_weight_down.weight': 'linear_weight',
    'model.hyper_connection_mixer.input_mix_weight_up.weight': 'linear_weight',
    'model.layers.{i}.attn_hyper_connection.block_inject_weight.weight': 'linear_weight',
    'model.layers.{i}.attn_hyper_connection.hc_norm.weight': 'tile',
    'model.layers.{i}.attn_hyper_connection.input_mix_weight_down.weight': 'linear_weight',
    'model.layers.{i}.attn_hyper_connection.input_mix_weight_up.weight': 'linear_weight',
    'model.layers.{i}.linear_attn.A_log': 'tile',
    'model.layers.{i}.linear_attn.conv1d.weight': 'tile',
    'model.layers.{i}.linear_attn.dt_bias': 'tile',
    'model.layers.{i}.linear_attn.in_proj_a.weight': 'linear_weight',
    'model.layers.{i}.linear_attn.in_proj_b.weight': 'linear_weight',
    'model.layers.{i}.linear_attn.in_proj_qkv.weight': 'linear_weight',
    'model.layers.{i}.linear_attn.in_proj_z.weight': 'linear_weight',
    'model.layers.{i}.linear_attn.norm.weight': 'tile',
    'model.layers.{i}.linear_attn.out_proj.weight': 'linear_weight',
    'model.layers.{i}.mlp.down_proj.weight': 'linear_weight',
    'model.layers.{i}.mlp.gate_proj.weight': 'linear_weight',
    'model.layers.{i}.mlp.up_proj.weight': 'linear_weight',
    'model.layers.{i}.mlp_hyper_connection.block_inject_weight.weight': 'linear_weight',
    'model.layers.{i}.mlp_hyper_connection.hc_norm.weight': 'tile',
    'model.layers.{i}.mlp_hyper_connection.input_mix_weight_down.weight': 'linear_weight',
    'model.layers.{i}.mlp_hyper_connection.input_mix_weight_up.weight': 'linear_weight',
    'model.layers.{i}.ple.conv1d.weight': 'tile',
    'model.layers.{i}.ple.key_proj.weight': 'linear_weight',
    'model.layers.{i}.ple.norm_conv.weight': 'tile',
    'model.layers.{i}.ple.norm_key.weight': 'tile',
    'model.layers.{i}.ple.norm_query.weight': 'tile',
    'model.layers.{i}.ple.ple_embedding.layer_multipliers': 'host',
    'model.layers.{i}.ple.ple_embedding.ngram_embedding.shard_0.weight': 'tile',
    'model.layers.{i}.ple.ple_embedding.ngram_embedding.shard_1.weight': 'tile',
    'model.layers.{i}.ple.ple_embedding.ngram_embedding.shard_2.weight': 'tile',
    'model.layers.{i}.ple.ple_embedding.ngram_embedding.shard_3.weight': 'tile',
    'model.layers.{i}.ple.ple_embedding.ngram_heads_offsets': 'host',
    'model.layers.{i}.ple.ple_embedding.ngram_heads_vocab_sizes': 'host',
    'model.layers.{i}.ple.value_proj.weight': 'linear_weight',
    'model.layers.{i}.self_attn.indexer.index_qk_proj.weight': 'linear_weight',
    'model.layers.{i}.self_attn.indexer.k_layernorm.weight': 'tile',
    'model.layers.{i}.self_attn.indexer.q_layernorm.weight': 'norm_plus_one',
    'model.layers.{i}.self_attn.k_norm.weight': 'norm_plus_one',
    'model.layers.{i}.self_attn.k_proj.weight': 'linear_weight',
    'model.layers.{i}.self_attn.o_proj.weight': 'linear_weight',
    'model.layers.{i}.self_attn.q_norm.weight': 'norm_plus_one',
    'model.layers.{i}.self_attn.q_proj.weight': 'linear_weight',
    'model.layers.{i}.self_attn.v_proj.weight': 'linear_weight',
}
NORM_EPS = {'model.layers.{i}.self_attn.indexer.q_layernorm': 1e-06, 'model.layers.{i}.self_attn.k_norm': 1e-06, 'model.layers.{i}.self_attn.q_norm': 1e-06}
_NUMBER = re.compile(r"(?<=\.)\d+(?=\.|$)")


def read_tensors(model_dir):
    """Every tensor of the checkpoint as {name: torch tensor}, from one file or from shards."""
    out = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                out[name] = f.get_tensor(name)
    return out


def _pattern(name):
    return _NUMBER.sub("{i}", name)


def _n_devices(mesh_device):
    """How many chips the device the check opened holds (1 for ttnn.open_device, N for a mesh)."""
    try:
        return int(mesh_device.get_num_devices())
    except Exception:          # noqa: BLE001 - a device object without the method is one chip
        return 1


def _upload(model, t, layout):
    """A host tensor onto the device as bf16 in `layout`; replicated on every chip of a mesh."""
    if model["n_devices"] > 1:
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=model["mesh_device"], memory_config=ttnn.DRAM_MEMORY_CONFIG,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(model["mesh_device"]))
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=model["mesh_device"], memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _upload_ids(model, ids):
    """Token ids as a uint32 ROW_MAJOR [1, S] tensor on the device."""
    ids_t = torch.tensor([ids], dtype=torch.int32)
    if model["n_devices"] > 1:
        return ttnn.from_torch(ids_t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=model["mesh_device"],
                               mesh_mapper=ttnn.ReplicateTensorToMesh(model["mesh_device"]))
    return ttnn.from_torch(ids_t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=model["mesh_device"])


def _to_host(model, t):
    """A device tensor back to torch. On a mesh the composer joins one copy per chip along dim 0; the first copy is kept."""
    if model["n_devices"] > 1:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(model["mesh_device"], dim=0))[:1]
    return ttnn.to_torch(t)


def _linear(model, prefix, x):
    """x [1, 1, S, in] @ W (+ bias) for the Linear module at `prefix`; W was uploaded as [in, out] in load."""
    w = model["dev"][prefix + ".weight"]
    b = model["dev"].get(prefix + ".bias")
    if b is not None:
        return ttnn.linear(x, w, bias=b, dtype=ttnn.bfloat16, compute_kernel_config=model["ck"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.linear(x, w, dtype=ttnn.bfloat16, compute_kernel_config=model["ck"], memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rms_norm(model, prefix, x):
    """RMSNorm over the last dim with the weight at `prefix` (uploaded as [1, 1, 1, dim]; a class that adds 1 to its weight had it added at load)."""
    return ttnn.rms_norm(x, epsilon=NORM_EPS[_pattern(prefix)], weight=model["dev"][prefix + ".weight"])


def _embedding(model, prefix, ids):
    """The embedding table at `prefix` looked up for `ids` (a list of ints), as [1, 1, S, HIDDEN] TILE bf16."""
    ids_dev = _upload_ids(model, ids)
    x = ttnn.embedding(ids_dev, model["dev"][prefix + ".weight"], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(x, (1, 1, len(ids), HIDDEN))


def _mlp(model, prefix, x):
    """down(silu(gate(x)) * up(x)) for the gated MLP at `prefix`."""
    gate = _linear(model, prefix + ".gate_proj", x)
    up = _linear(model, prefix + ".up_proj", x)
    h = ttnn.multiply(ttnn.silu(gate), up)
    y = _linear(model, prefix + ".down_proj", h)
    for t in (gate, up, h):
        ttnn.deallocate(t)
    return y


def _model_prelude(model, x, ids):
    """Qwen4ExpTextModel at model. Not generated: Qwen4ExpTextModel has children the generated flow does not know: hyper_connection_mixer (Qwen4ExpTextGatedResidual); stage 0 labelled aten.diff.default (host) in model itself."""
    # AGENT: write Qwen4ExpTextModel (model) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # The reference's forward may change the embedding before the first layer; x is [1, 1, S, HIDDEN] TILE bf16 from _embedding.
    raise NotImplementedError('_model_prelude: Qwen4ExpTextModel at model is not written yet')


def _model_hyper_connection_mixer(model, x):
    """Qwen4ExpTextGatedResidual at model.hyper_connection_mixer. Not generated: a Qwen4ExpTextGatedResidual block at the end of the text model that the generator does not write."""
    # AGENT: write Qwen4ExpTextGatedResidual (model.hyper_connection_mixer) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    raise NotImplementedError('_model_hyper_connection_mixer: Qwen4ExpTextGatedResidual at model.hyper_connection_mixer is not written yet')


def _A_attn_hyper_connection(model, i, x, ids):
    """Qwen4ExpTextGatedResidual at model.layers.{i}.attn_hyper_connection. Not generated: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.attn_hyper_connection.hc_norm."""
    # AGENT: write Qwen4ExpTextGatedResidual (model.layers.{i}.attn_hyper_connection) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # The Linear children of this block are uploaded by load() and can be called as _linear(model, f"model.layers.{i}.attn_hyper_connection.<name>", x).
    raise NotImplementedError('_A_attn_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.attn_hyper_connection is not written yet')


def _A_linear_attn(model, i, x, ids):
    """Qwen4ExpTextGatedDeltaNet at model.layers.{i}.linear_attn. Not generated: stage 0 labelled aten.clone.default (candidate) under model.layers.{i}.linear_attn."""
    # AGENT: write Qwen4ExpTextGatedDeltaNet (model.layers.{i}.linear_attn) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # The Linear children of this block are uploaded by load() and can be called as _linear(model, f"model.layers.{i}.linear_attn.<name>", x).
    raise NotImplementedError('_A_linear_attn: Qwen4ExpTextGatedDeltaNet at model.layers.{i}.linear_attn is not written yet')


def _A_mlp_hyper_connection(model, i, x, ids):
    """Qwen4ExpTextGatedResidual at model.layers.{i}.mlp_hyper_connection. Not generated: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.mlp_hyper_connection.hc_norm."""
    # AGENT: write Qwen4ExpTextGatedResidual (model.layers.{i}.mlp_hyper_connection) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # The Linear children of this block are uploaded by load() and can be called as _linear(model, f"model.layers.{i}.mlp_hyper_connection.<name>", x).
    raise NotImplementedError('_A_mlp_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.mlp_hyper_connection is not written yet')


def _layer_A(model, i, x, ids, cos, sin):
    """Qwen4ExpTextDecoderLayer at model.layers.{i}. Not generated: stage 0 labelled aten.flatten.using_ints (candidate) in the layer itself; stubs: attn_hyper_connection (Qwen4ExpTextGatedResidual), linear_attn (Qwen4ExpTextGatedDeltaNet), mlp_hyper_connection (Qwen4ExpTextGatedResidual)."""
    # AGENT: write Qwen4ExpTextDecoderLayer (model.layers.{i}) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # Layers [0, 2]. Parts, each a generated helper or a stub above; compose them as Qwen4ExpTextDecoderLayer.forward does:
    #   attn_hyper_connection (Qwen4ExpTextGatedResidual): _A_attn_hyper_connection(model, i, x, ids)
    #   linear_attn (Qwen4ExpTextGatedDeltaNet): _A_linear_attn(model, i, x, ids)
    #   mlp (Qwen4ExpTextMLP): _mlp(model, p + ".mlp", x)
    #   mlp_hyper_connection (Qwen4ExpTextGatedResidual): _A_mlp_hyper_connection(model, i, x, ids)
    # p = f"model.layers.{i}"
    raise NotImplementedError('_layer_A: Qwen4ExpTextDecoderLayer at model.layers.{i} is not written yet')


def _B_attn_hyper_connection(model, i, x, ids):
    """Qwen4ExpTextGatedResidual at model.layers.{i}.attn_hyper_connection. Not generated: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.attn_hyper_connection.hc_norm."""
    # AGENT: write Qwen4ExpTextGatedResidual (model.layers.{i}.attn_hyper_connection) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # The Linear children of this block are uploaded by load() and can be called as _linear(model, f"model.layers.{i}.attn_hyper_connection.<name>", x).
    raise NotImplementedError('_B_attn_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.attn_hyper_connection is not written yet')


def _B_linear_attn(model, i, x, ids):
    """Qwen4ExpTextGatedDeltaNet at model.layers.{i}.linear_attn. Not generated: stage 0 labelled aten.clone.default (candidate) under model.layers.{i}.linear_attn."""
    # AGENT: write Qwen4ExpTextGatedDeltaNet (model.layers.{i}.linear_attn) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # The Linear children of this block are uploaded by load() and can be called as _linear(model, f"model.layers.{i}.linear_attn.<name>", x).
    raise NotImplementedError('_B_linear_attn: Qwen4ExpTextGatedDeltaNet at model.layers.{i}.linear_attn is not written yet')


def _B_mlp_hyper_connection(model, i, x, ids):
    """Qwen4ExpTextGatedResidual at model.layers.{i}.mlp_hyper_connection. Not generated: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.mlp_hyper_connection.hc_norm."""
    # AGENT: write Qwen4ExpTextGatedResidual (model.layers.{i}.mlp_hyper_connection) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # The Linear children of this block are uploaded by load() and can be called as _linear(model, f"model.layers.{i}.mlp_hyper_connection.<name>", x).
    raise NotImplementedError('_B_mlp_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.mlp_hyper_connection is not written yet')


def _B_ple(model, i, x, ids):
    """Qwen4ExpTextPLELayer at model.layers.{i}.ple. Not generated: stage 0 labelled aten.sum.dim_IntList (candidate) under model.layers.{i}.ple."""
    # AGENT: write Qwen4ExpTextPLELayer (model.layers.{i}.ple) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # The Linear children of this block are uploaded by load() and can be called as _linear(model, f"model.layers.{i}.ple.<name>", x).
    raise NotImplementedError('_B_ple: Qwen4ExpTextPLELayer at model.layers.{i}.ple is not written yet')


def _layer_B(model, i, x, ids, cos, sin):
    """Qwen4ExpTextDecoderLayer at model.layers.{i}. Not generated: stage 0 labelled aten.flatten.using_ints (candidate) in the layer itself; stubs: attn_hyper_connection (Qwen4ExpTextGatedResidual), linear_attn (Qwen4ExpTextGatedDeltaNet), mlp_hyper_connection (Qwen4ExpTextGatedResidual), ple (Qwen4ExpTextPLELayer)."""
    # AGENT: write Qwen4ExpTextDecoderLayer (model.layers.{i}) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # Layers [1]. Parts, each a generated helper or a stub above; compose them as Qwen4ExpTextDecoderLayer.forward does:
    #   attn_hyper_connection (Qwen4ExpTextGatedResidual): _B_attn_hyper_connection(model, i, x, ids)
    #   linear_attn (Qwen4ExpTextGatedDeltaNet): _B_linear_attn(model, i, x, ids)
    #   mlp (Qwen4ExpTextMLP): _mlp(model, p + ".mlp", x)
    #   mlp_hyper_connection (Qwen4ExpTextGatedResidual): _B_mlp_hyper_connection(model, i, x, ids)
    #   ple (Qwen4ExpTextPLELayer): _B_ple(model, i, x, ids)
    # p = f"model.layers.{i}"
    raise NotImplementedError('_layer_B: Qwen4ExpTextDecoderLayer at model.layers.{i} is not written yet')


def _C_attn_hyper_connection(model, i, x, ids):
    """Qwen4ExpTextGatedResidual at model.layers.{i}.attn_hyper_connection. Not generated: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.attn_hyper_connection.hc_norm."""
    # AGENT: write Qwen4ExpTextGatedResidual (model.layers.{i}.attn_hyper_connection) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # The Linear children of this block are uploaded by load() and can be called as _linear(model, f"model.layers.{i}.attn_hyper_connection.<name>", x).
    raise NotImplementedError('_C_attn_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.attn_hyper_connection is not written yet')


def _C_mlp_hyper_connection(model, i, x, ids):
    """Qwen4ExpTextGatedResidual at model.layers.{i}.mlp_hyper_connection. Not generated: stage 0 labelled aten.flatten.using_ints (candidate) under model.layers.{i}.mlp_hyper_connection.hc_norm."""
    # AGENT: write Qwen4ExpTextGatedResidual (model.layers.{i}.mlp_hyper_connection) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # The Linear children of this block are uploaded by load() and can be called as _linear(model, f"model.layers.{i}.mlp_hyper_connection.<name>", x).
    raise NotImplementedError('_C_mlp_hyper_connection: Qwen4ExpTextGatedResidual at model.layers.{i}.mlp_hyper_connection is not written yet')


def _C_self_attn(model, i, x, ids):
    """Qwen4ExpTextAttention at model.layers.{i}.self_attn. Not generated: Qwen4ExpTextAttention: attention with children beyond q, k, v, o and their norms: indexer (Qwen4ExpTextQSAIndexer)."""
    # AGENT: write Qwen4ExpTextAttention (model.layers.{i}.self_attn) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # The Linear children of this block are uploaded by load() and can be called as _linear(model, f"model.layers.{i}.self_attn.<name>", x).
    raise NotImplementedError('_C_self_attn: Qwen4ExpTextAttention at model.layers.{i}.self_attn is not written yet')


def _layer_C(model, i, x, ids, cos, sin):
    """Qwen4ExpTextDecoderLayer at model.layers.{i}. Not generated: stage 0 labelled aten.flatten.using_ints (candidate) in the layer itself; stubs: attn_hyper_connection (Qwen4ExpTextGatedResidual), mlp_hyper_connection (Qwen4ExpTextGatedResidual), self_attn (Qwen4ExpTextAttention)."""
    # AGENT: write Qwen4ExpTextDecoderLayer (model.layers.{i}) here from its forward in stages/2/reference_excerpts.md, then remove this marker.
    # Layers [3]. Parts, each a generated helper or a stub above; compose them as Qwen4ExpTextDecoderLayer.forward does:
    #   attn_hyper_connection (Qwen4ExpTextGatedResidual): _C_attn_hyper_connection(model, i, x, ids)
    #   mlp (Qwen4ExpTextMLP): _mlp(model, p + ".mlp", x)
    #   mlp_hyper_connection (Qwen4ExpTextGatedResidual): _C_mlp_hyper_connection(model, i, x, ids)
    #   self_attn (Qwen4ExpTextAttention): _C_self_attn(model, i, x, ids)
    # p = f"model.layers.{i}"
    raise NotImplementedError('_layer_C: Qwen4ExpTextDecoderLayer at model.layers.{i} is not written yet')


def load(model_dir, mesh_device):
    """Read the checkpoint and place every tensor on the device as PLACEMENT says; return the model object the other functions take."""
    model = {"mesh_device": mesh_device, "n_devices": _n_devices(mesh_device), "dev": {}, "host": {}, "ids": []}
    model["ck"] = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    dev, host = model["dev"], model["host"]
    tensors = read_tensors(model_dir)          # a local: the model object keeps no host copy of an uploaded weight
    for name, t in tensors.items():
        role = PLACEMENT.get(_pattern(name), "tile" if t.is_floating_point() else "host")
        if role == "linear_weight":
            dev[name] = _upload(model, t.to(torch.bfloat16).t().contiguous(), ttnn.TILE_LAYOUT)
        elif role == "linear_bias" or role == "norm":
            dev[name] = _upload(model, t.to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.TILE_LAYOUT)
        elif role == "norm_plus_one":
            dev[name] = _upload(model, (t.float() + 1.0).to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.TILE_LAYOUT)
        elif role == "embedding":
            dev[name] = _upload(model, t.to(torch.bfloat16).contiguous(), ttnn.ROW_MAJOR_LAYOUT)
        elif role == "tile":
            dev[name] = _upload(model, t.to(torch.bfloat16).reshape(1, 1, 1, -1) if t.dim() == 1 else t.to(torch.bfloat16).contiguous(), ttnn.TILE_LAYOUT)
        else:
            host[name] = t
    if "lm_head.weight" not in dev:         # tied embeddings: the head is the embedding table, uploaded once more as a Linear weight
        dev["lm_head.weight"] = _upload(model, tensors["model.embed_tokens.weight"].to(torch.bfloat16).t().contiguous(), ttnn.TILE_LAYOUT)
    return model


def _forward(model, ids):
    """Logits for the token after the last of `ids`, as a float32 torch vector of VOCAB values."""
    n = len(ids)
    padded = list(ids) + [PAD_ID] * (-n % PAD_TO)
    x = _embedding(model, "model.embed_tokens", padded)
    cos = sin = None
    x = _model_prelude(model, x, padded)
    for i in range(N_LAYERS):
        x = LAYERS[LAYER_VARIANT[i]](model, i, x, padded, cos, sin)
    x = _model_hyper_connection_mixer(model, x)
    logits = _linear(model, "lm_head", x)
    out = _to_host(model, logits).float()
    return out.reshape(-1, out.shape[-1])[n - 1, :VOCAB]


LAYERS = {'A': _layer_A, 'B': _layer_B, 'C': _layer_C}


def prefill(model, ids):
    """Start the request afresh: run the prompt and return the logits for the next token."""
    model["ids"] = [int(t) for t in ids]
    return _forward(model, model["ids"])


def decode_step(model, token_id, position):
    """Append `token_id` at `position` and run the whole sequence again (no cache: the same device calls as prefill)."""
    model["ids"] = model["ids"][:position] + [int(token_id)]
    return _forward(model, model["ids"])


def close(model):
    """Free every device tensor load took. The check closes the device itself."""
    for t in list(model["dev"].values()):
        try:
            ttnn.deallocate(t)
        except Exception:      # noqa: BLE001 - a failed free must not stop the run
            pass
    model["dev"].clear()
    model["host"].clear()
```
