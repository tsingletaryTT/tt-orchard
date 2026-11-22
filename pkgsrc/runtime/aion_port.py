# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
# generated-by: orchard/port_template.py
"""Generated first port of AionInstructForCausalLM for stage 2 of a full port (orchard/port_template.py).

The standard blocks are written through TTNN calls from the verified cheat sheet (stages/2/cheat_sheet.md): weight upload in
load, the embedding, RMSNorm, the gated MLP, ordinary attention and lm_head. Each function whose body starts with `# AGENT:` is a
block the generator does not write: fill it in from the reference class named in stages/2/reference_excerpts.md and remove the
marker. The CPU check refuses a port that keeps a marker.

Stubs:
- none

Notes:
- the inventory lists at most a few modules per op, so a block it did not flag may still run an op labelled candidate or host
- no rotary helpers (_rope_tables, _apply_rotary) are written: rope_parameters is keyed by layer type ('full_attention', 'sliding_attention'); one rotary table for every layer would be wrong; an attention block has to build its own rotary tables
"""
from __future__ import annotations

import glob
import os
import re

import torch
import ttnn
from safetensors import safe_open

CONFIG = {'model_type': 'aion_instruct', 'vocab_size': 200029, 'hidden_size': 2048, 'intermediate_size': 8192, 'num_hidden_layers': 28, 'num_attention_heads': 24, 'num_key_value_heads': 8, 'head_dim': 128, 'hidden_act': 'silu', 'max_position_embeddings': 32768, 'initializer_range': 0.02, 'rms_norm_eps': 1e-06, 'tie_word_embeddings': True, 'attention_bias': False, 'attention_dropout': 0.0, 'pad_token_id': 199999, 'bos_token_id': 199999, 'eos_token_id': 200020, 'sliding_window': 2048, 'use_cache': True, 'dtype': 'bfloat16', 'transformers_version': '5.9.0'}
N_LAYERS = 28
HIDDEN = 2048
VOCAB = 200029
PAD_ID = 199999
PAD_TO = 32                       # the sequence is padded to a tile multiple; the logits are read at the last real position
TOKEN_CHOICE = "host-row"           # prefill and decode_step return the whole vocabulary row to the host; the caller picks the token there, so no sampler runs on a die
LAYER_VARIANT = {0: 'A', 1: 'A', 2: 'A', 3: 'A', 4: 'A', 5: 'A', 6: 'A', 7: 'A', 8: 'A', 9: 'A', 10: 'A', 11: 'A', 12: 'A', 13: 'A', 14: 'A', 15: 'A', 16: 'A', 17: 'A', 18: 'A', 19: 'A', 20: 'A', 21: 'A', 22: 'A', 23: 'A', 24: 'A', 25: 'A', 26: 'A', 27: 'A'}
# Every tensor pattern of the checkpoint and how load() places it: linear_weight (transposed once, TILE bf16), linear_bias, embedding
# (ROW_MAJOR bf16 table), norm and norm_plus_one ([1, 1, 1, dim] TILE bf16), tile (any other float tensor), host (integer tensors stay in torch).
PLACEMENT = {
    'lm_head.weight': 'linear_weight',
    'model.embed_tokens.weight': 'embedding',
    'model.layers.{i}.input_layernorm.weight': 'norm',
    'model.layers.{i}.mlp.down_proj.weight': 'linear_weight',
    'model.layers.{i}.mlp.gate_proj.weight': 'linear_weight',
    'model.layers.{i}.mlp.up_proj.weight': 'linear_weight',
    'model.layers.{i}.post_attention_layernorm.weight': 'norm',
    'model.layers.{i}.self_attn.k_norm.weight': 'norm',
    'model.layers.{i}.self_attn.k_proj.weight': 'linear_weight',
    'model.layers.{i}.self_attn.o_proj.weight': 'linear_weight',
    'model.layers.{i}.self_attn.q_norm.weight': 'norm',
    'model.layers.{i}.self_attn.q_proj.weight': 'linear_weight',
    'model.layers.{i}.self_attn.v_proj.weight': 'linear_weight',
    'model.norm.weight': 'norm',
}
NORM_EPS = {'model.layers.{i}.input_layernorm': 1e-06, 'model.layers.{i}.post_attention_layernorm': 1e-06, 'model.layers.{i}.self_attn.k_norm': 1e-06, 'model.layers.{i}.self_attn.q_norm': 1e-06, 'model.norm': 1e-06}
_NUMBER = re.compile(r"(?<=\.)\d+(?=\.|$)")


class _Tensors:
    """The checkpoint as a mapping {name: torch tensor} that reads a tensor from disk only when it is asked for (`tensors[name]`), from one file or from shards.
    `shape(name)`, `numel(name)` and `is_float(name)` come from the file headers and read no data; `keys()` and `len()` read only the headers too."""

    _FLOATS = frozenset({'F64', 'F32', 'F16', 'BF16', 'F8_E4M3', 'F8_E5M2'})

    def __init__(self, model_dir):
        self._where = {}
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(path, framework="pt") as f:
                for name in f.keys():
                    self._where[name] = path

    def keys(self):
        return self._where.keys()

    def __iter__(self):
        return iter(self._where)

    def __len__(self):
        return len(self._where)

    def __contains__(self, name):
        return name in self._where

    def __getitem__(self, name):
        with safe_open(self._where[name], framework="pt") as f:
            return f.get_tensor(name)

    def _slice(self, name):
        with safe_open(self._where[name], framework="pt") as f:
            s = f.get_slice(name)
            return list(s.get_shape()), s.get_dtype()

    def shape(self, name):
        return self._slice(name)[0]

    def numel(self, name):
        n = 1
        for d in self.shape(name):
            n *= d
        return n

    def is_float(self, name):
        return self._slice(name)[1] in self._FLOATS


def read_tensors(model_dir):
    """The checkpoint as a read-on-demand mapping {name: torch tensor} (see _Tensors): a checkpoint far larger than host memory can be loaded one tensor at a time."""
    return _Tensors(model_dir)


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


def _cache_of(model, name):
    """(mode, state, n) of the decode cache for the block `name`. mode is None when the model runs without a cache; "prefill" while a prompt is run and its state kept; "step" while one token is decoded from that state. state is the block's own dict. n is the number of real tokens of the prompt (prefill) or the position of the token being decoded (step)."""
    cache = model.get("cache")
    if cache is None:
        return None, None, 0
    return cache["mode"], cache["state"].setdefault(name, {}), cache["n"]


def _linear(model, prefix, x):
    """x [1, 1, S, in] @ W (+ bias) for the Linear module at `prefix`; W was uploaded as [in, out] in load."""
    w = model["dev"][prefix + ".weight"]
    b = model["dev"].get(prefix + ".bias")
    if b is not None:
        return ttnn.linear(x, w, bias=b, dtype=ttnn.bfloat16, compute_kernel_config=model["ck"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.linear(x, w, dtype=ttnn.bfloat16, compute_kernel_config=model["ck"], memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _rms_norm(model, prefix, x):
    """RMSNorm over the last dim with the weight at `prefix` (uploaded as [1, 1, 1, dim]; a class that adds 1 to its weight had it added at load)."""
    eps = NORM_EPS.get(_pattern(prefix))
    if eps is None:
        raise ValueError(f"{prefix} is not a plain RMS norm (NORM_EPS has no entry for it): the function that calls _rms_norm must compute this norm itself")
    return ttnn.rms_norm(x, epsilon=eps, weight=model["dev"][prefix + ".weight"])


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


ATTENTION = {'heads': 24, 'kv_heads': 8, 'head_dim': 128, 'layer_types': ['sliding_attention', 'sliding_attention', 'full_attention', 'full_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'full_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention'], 'rope': {'full_attention': {'kind': 'yarn', 'theta': 32000.0, 'factor': 16.0, 'original_max': 2048, 'beta_fast': 32.0, 'beta_slow': 1.0}, 'sliding_attention': {'kind': 'default', 'theta': 32000.0}}, 'scale': 0.08838834764831845, 'q_norm': True, 'k_norm': True, 'norm_eps': 1e-06, 'window': 2048}


def _rope_inv_freq(spec, hd):
    """Inverse frequencies [hd // 2] and the attention scaling of one rope block (a value of ATTENTION["rope"]), computed the way transformers computes
    them. The default type is 1 / theta ** (2i / hd). Yarn mixes the interpolated and the original frequencies over a ramp between the beta_fast and beta_slow
    rotation counts, and scales cos and sin by 0.1 * ln(factor) + 1 unless the block gives attention_factor. The ramp bounds are always truncated: the
    reference reads `truncate` from the top-level rope_parameters, which holds no such key when the config is keyed by layer type."""
    # GENERATED (not a stub): a failure inside this function is not sent to the coder as a repair target
    import math
    base = spec["theta"]
    pos_freqs = base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd)
    if spec["kind"] == "default":
        return 1.0 / pos_freqs, 1.0
    factor, original = spec["factor"], spec["original_max"]
    scaling = spec.get("attention_factor")
    if scaling is None:
        mscale, mscale_all = spec.get("mscale"), spec.get("mscale_all_dim")

        def get_mscale(scale, m=1.0):
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
        scaling = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all)) if mscale and mscale_all else get_mscale(factor)

    def correction_dim(rotations):
        return (hd * math.log(original / (rotations * 2 * math.pi))) / (2 * math.log(base))
    low = max(math.floor(correction_dim(spec.get("beta_fast") or 32)), 0)
    high = min(math.ceil(correction_dim(spec.get("beta_slow") or 1)), hd - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(hd // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    extrapolation = 1 - ramp
    return (1.0 / (factor * pos_freqs)) * (1 - extrapolation) + (1.0 / pos_freqs) * extrapolation, scaling


def _rope_tables(model, n):
    """cos and sin for positions 0..n-1 of every layer type, as two dicts {layer type: [1, 1, n, head_dim] TILE bf16 on the device} in the HF layout (freqs
    repeated twice). `_attention` takes the pair of its own layer type from them."""
    # GENERATED (not a stub): a failure inside this function is not sent to the coder as a repair target
    hd = ATTENTION["head_dim"]
    cos, sin = {}, {}
    for layer_type in sorted(set(ATTENTION["layer_types"])):
        inv_freq, scaling = _rope_inv_freq(ATTENTION["rope"][layer_type], hd)
        freqs = torch.arange(n, dtype=torch.float32)[:, None] * inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos[layer_type] = _upload(model, (emb.cos() * scaling).reshape(1, 1, n, hd).to(torch.bfloat16), ttnn.TILE_LAYOUT)
        sin[layer_type] = _upload(model, (emb.sin() * scaling).reshape(1, 1, n, hd).to(torch.bfloat16), ttnn.TILE_LAYOUT)
    return cos, sin



def _window_mask(model, n):
    """The additive attention mask [1, 1, n, n] (bf16, TILE) of the sliding layers: 0 where a query may attend (itself and the ATTENTION["window"] - 1
    positions before it, the reference's `kv_idx > q_idx - sliding_window`), -30000 elsewhere. It is built once per padded length and kept in
    model["window_mask"]; a new length frees the old mask, and close frees the last one."""
    # GENERATED (not a stub): a failure inside this function is not sent to the coder as a repair target
    held = model.get("window_mask")
    if held is not None and held[0] == n:
        return held[1]
    if held is not None:
        ttnn.deallocate(held[1])
    rows = torch.arange(n)[:, None]
    cols = torch.arange(n)[None, :]
    allowed = (cols <= rows) & (cols > rows - ATTENTION["window"])
    mask = _upload(model, torch.where(allowed, 0.0, -30000.0).reshape(1, 1, n, n).to(torch.bfloat16), ttnn.TILE_LAYOUT)
    model["window_mask"] = (n, mask)
    return mask



def _attention(model, prefix, x, cos, sin):
    """Ordinary causal attention for the block at `prefix`: q, k, v projections, the head split, per-head RMSNorm on q and k when the
    reference has one, rotary in the HF layout, SDPA with grouped keys, the head join and the output projection."""
    layer_type = ATTENTION["layer_types"][int(prefix.rsplit(".", 2)[-2])]          # the prefix is "<layers path>.<i>.<attention name>"
    cos, sin = cos[layer_type], sin[layer_type]
    q = _linear(model, prefix + ".q_proj", x)
    k = _linear(model, prefix + ".k_proj", x)
    v = _linear(model, prefix + ".v_proj", x)
    xqkv = ttnn.concat([q, k, v], dim=-1)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(xqkv, num_heads=ATTENTION["heads"], num_kv_heads=ATTENTION["kv_heads"],
                                                     transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q = ttnn.rms_norm(q, epsilon=ATTENTION["norm_eps"], weight=model["dev"][prefix + ".q_norm.weight"])
    k = ttnn.rms_norm(k, epsilon=ATTENTION["norm_eps"], weight=model["dev"][prefix + ".k_norm.weight"])
    q = ttnn.experimental.rotary_embedding_hf(q, cos, sin, is_decode_mode=False)
    k = ttnn.experimental.rotary_embedding_hf(k, cos, sin, is_decode_mode=False)
    seq = q.shape[2]
    if layer_type == "sliding_attention" and seq > ATTENTION["window"]:          # inside the window a sliding layer is plain causal attention
        o = ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=_window_mask(model, seq), is_causal=False, scale=ATTENTION["scale"])
    else:
        o = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=ATTENTION["scale"])
    o = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return _linear(model, prefix + ".o_proj", o)


def _layer_A(model, i, x, ids, cos, sin):
    """Layers [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]: AionInstructDecoderLayer, pre-norm attention and MLP with residual adds."""
    p = f"model.layers.{i}"
    h = _rms_norm(model, p + ".input_layernorm", x)
    x = ttnn.add(x, _attention(model, p + ".self_attn", h, cos, sin))
    h = _rms_norm(model, p + ".post_attention_layernorm", x)
    return ttnn.add(x, _mlp(model, p + ".mlp", h))


def load(model_dir, mesh_device):
    """Read the checkpoint and place every tensor on the device as PLACEMENT says; return the model object the other functions take."""
    model = {"mesh_device": mesh_device, "n_devices": _n_devices(mesh_device), "dev": {}, "host": {}, "ids": []}
    model["ck"] = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    dev, host = model["dev"], model["host"]
    tensors = read_tensors(model_dir)          # a local: the model object keeps no host copy of an uploaded weight
    for name in list(tensors.keys()):
        role = PLACEMENT.get(_pattern(name))
        if role == "expert_shard":
            continue          # uploaded sharded over the chips by _moe_upload_parallel after the loop, read from disk there
        t = tensors[name]          # one tensor at a time: it is released when the next one is read
        role = role or ("tile" if t.is_floating_point() else "host")
        if role == "linear_weight":
            dev[name] = _upload(model, t.to(torch.bfloat16).t().contiguous(), ttnn.TILE_LAYOUT)
        elif role == "linear_bias" or role == "norm":
            dev[name] = _upload(model, t.to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.TILE_LAYOUT)
        elif role == "norm_plus_one":
            dev[name] = _upload(model, (t.float() + 1.0).to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.TILE_LAYOUT)
        elif role == "embedding":
            dev[name] = _upload(model, t.to(torch.bfloat16).contiguous(), ttnn.ROW_MAJOR_LAYOUT)
        elif role == "neg_exp":
            dev[name] = _upload(model, (-t.float().exp()).to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.TILE_LAYOUT)
        elif role == "conv_taps":
            for j in range(t.shape[-1]):
                dev[f"{name}.tap{j}"] = _upload(model, t[:, 0, j].to(torch.bfloat16).reshape(1, 1, 1, -1), ttnn.TILE_LAYOUT)
        elif role == "tile":
            dev[name] = _upload(model, t.to(torch.bfloat16).reshape(1, 1, 1, -1) if t.dim() == 1 else t.to(torch.bfloat16).contiguous(), ttnn.TILE_LAYOUT)
        else:
            host[name] = t
    if "expert_shard" in PLACEMENT.values():
        _moe_upload_parallel(model, tensors)
    if "lm_head.weight" not in dev:         # tied embeddings: the head is the embedding table, uploaded once more as a Linear weight
        dev["lm_head.weight"] = _upload(model, tensors["model.embed_tokens.weight"].to(torch.bfloat16).t().contiguous(), ttnn.TILE_LAYOUT)
    return model


def _forward(model, ids):
    """Logits for the token after the last of `ids`, as a float32 torch vector of VOCAB values."""
    n = len(ids)
    padded = list(ids) + [PAD_ID] * (-n % PAD_TO)
    x = _embedding(model, "model.embed_tokens", padded)
    cos, sin = _rope_tables(model, len(padded))
    for i in range(N_LAYERS):
        x = LAYERS[LAYER_VARIANT[i]](model, i, x, padded, cos, sin)
    x = _rms_norm(model, "model.norm", x)
    logits = _linear(model, "lm_head", x)
    out = _to_host(model, logits).float()
    return out.reshape(-1, out.shape[-1])[n - 1, :VOCAB]


LAYERS = {'A': _layer_A}


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
    held = model.pop("window_mask", None)
    if held is not None:
        try:
            ttnn.deallocate(held[1])
        except Exception:      # noqa: BLE001 - a failed free must not stop the run
            pass
