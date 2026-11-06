# Replaced by the supervisor with a corrected version from an outside helper model (telegraph call f12fd054d372, model anthropic/claude-sonnet-5-5).
# It is advice from another model and may be wrong. Check it against stages/2/port_facts.json before you rely on it.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Port module for local/tiny-qwen4-dense (Qwen4ExpForCausalLM)."""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Any

import torch  # noqa: F401
import torch.nn.functional as F
import ttnn  # noqa: F401
from safetensors import safe_open

EPS = 1e-6


def read_tensors(model_dir: str) -> dict[str, torch.Tensor]:
    """Every tensor of the checkpoint as {name: torch tensor}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                out[name] = f.get_tensor(name)
    return out


def load(model_dir: str, mesh_device: Any) -> dict[str, Any]:
    """Load the weights from the snapshot directory onto the open device."""
    raw = read_tensors(model_dir)
    tensors = {}
    for k, v in raw.items():
        tensors[k] = v.float() if v.is_floating_point() else v.long()
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    return {
        "config": config,
        "tensors": tensors,
        "mesh_device": mesh_device,
        "history": [],
    }


def _rms(x, w=None, plus_one=True, group=None):
    xf = x.float()
    if group is not None and xf.shape[-1] != group:
        shp = xf.shape
        xg = xf.reshape(*shp[:-1], shp[-1] // group, group)
        xg = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + EPS)
        out = xg.reshape(shp)
    else:
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    if w is not None:
        out = out * ((1.0 + w) if plus_one else w)
    return out


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def _rope(x, cos, sin):
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    xr = xr * cos + _rotate_half(xr) * sin
    return torch.cat([xr, xp], dim=-1)


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _cos_sin(seq_len, rot_dim, theta):
    inv = 1.0 / (theta ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
    freqs = torch.arange(seq_len).float()[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None], emb.sin()[None]


def _hyper(T, prefix, h, hc, hid, with_inject=True):
    normed = _rms(h, T[prefix + "hc_norm.weight"], group=hid)
    down = F.linear(normed, T[prefix + "input_mix_weight_down.weight"])
    mix = F.silu(down / hc)
    mix = torch.sigmoid(F.linear(mix, T[prefix + "input_mix_weight_up.weight"]))
    mix = mix.unflatten(-1, (hc, hid))
    mixed = (mix * normed.unflatten(-1, (hc, hid))).mean(dim=-2)
    if not with_inject:
        return mixed
    inj = 2 * torch.sigmoid(F.linear(normed, T[prefix + "block_inject_weight.weight"]) / hc)
    return mixed, h, inj


def _combine(x, hyper_input, inj):
    injection = x.unsqueeze(-2) * inj.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)


def _mlp(T, i, x):
    p = f"model.layers.{i}.mlp."
    g = F.linear(x, T[p + "gate_proj.weight"])
    u = F.linear(x, T[p + "up_proj.weight"])
    return F.linear(F.silu(g) * u, T[p + "down_proj.weight"])


def _linear_attn(T, i, x, cfg):
    p = f"model.layers.{i}.linear_attn."
    B, S, _ = x.shape
    nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    key_dim, val_dim = nk * dk, nv * dv

    mixed = F.linear(x, T[p + "in_proj_qkv.weight"]).transpose(1, 2)
    w = T[p + "conv1d.weight"]
    bias = T.get(p + "conv1d.bias")
    mixed = F.conv1d(mixed, w, bias, padding=w.shape[-1] - 1, groups=w.shape[0])[:, :, :S]
    mixed = F.silu(mixed).transpose(1, 2)

    q, k, v = torch.split(mixed, [key_dim, key_dim, val_dim], dim=-1)
    q = q.reshape(B, S, -1, dk)
    k = k.reshape(B, S, -1, dk)
    v = v.reshape(B, S, -1, dv)

    z = F.linear(x, T[p + "in_proj_z.weight"]).reshape(B, S, -1, dv)
    b = F.linear(x, T[p + "in_proj_b.weight"])
    a = F.linear(x, T[p + "in_proj_a.weight"])

    beta = b.sigmoid()
    g = -T[p + "A_log"].float().exp() * F.softplus(a.float() + T[p + "dt_bias"])

    rep = nv // nk
    if rep > 1:
        q = q.repeat_interleave(rep, dim=2)
        k = k.repeat_interleave(rep, dim=2)

    q = _l2norm(q) * (dk ** -0.5)
    k = _l2norm(k)

    H = q.shape[2]
    state = torch.zeros(B, H, dk, dv)
    outs = []
    for t in range(S):
        state = state * g[:, t].exp()[..., None, None]
        kv_mem = (state * k[:, t, :, :, None]).sum(dim=-2)
        delta = (v[:, t] - kv_mem) * beta[:, t, :, None]
        state = state + k[:, t, :, :, None] * delta[:, :, None, :]
        outs.append((state * q[:, t, :, :, None]).sum(dim=-2))
    core = torch.stack(outs, dim=1)

    core = core.reshape(-1, dv)
    zf = z.reshape(-1, dv)
    core = _rms(core, T[p + "norm.weight"], plus_one=False)
    core = core * F.silu(zf.float())
    core = core.reshape(B, S, -1)
    return F.linear(core, T[p + "out_proj.weight"])


def _indexer_mask(T, p, x, cos, sin, cfg):
    """Boolean (S, S) mask of tokens selected by the QSA indexer (batch 1)."""
    B, S, _ = x.shape
    nh = cfg["indexer_n_heads"]
    d = cfg["indexer_head_dim"]
    ratio = cfg["indexer_compress_ratio"]
    budget = cfg["indexer_budget"]
    topk_blocks = max(budget // ratio, 1)

    qk = F.linear(x, T[p + "indexer.index_qk_proj.weight"])
    q = qk[..., : nh * d].reshape(B, S, nh, d)
    keys = qk[..., nh * d:].reshape(B, S, d)
    q = _rms(q, T[p + "indexer.q_layernorm.weight"])
    q = _rope(q, cos[:, :, None, :], sin[:, :, None, :])
    kln = T[p + "indexer.k_layernorm.weight"]

    mask = torch.zeros(S, S, dtype=torch.bool)
    for t in range(S):
        vis = torch.arange(t + 1)
        n = (t + 1) // ratio
        sel = []
        if n > 0:
            blocks = vis[: n * ratio].view(n, ratio)
            pooled = keys[0][blocks.flatten()].view(n, ratio, d).mean(dim=1)
            pooled = _rms(pooled, kln)
            starts = blocks[:, 0]
            bk = _rope(pooled, cos[0][starts], sin[0][starts])
            scores = torch.matmul(q[0, t], bk.transpose(0, 1)).transpose(0, 1)
            scores = torch.relu(scores).sum(dim=-1) / math.sqrt(d)
            top = scores.topk(min(topk_blocks, n), dim=0).indices
            sel.append(blocks.index_select(0, top).flatten())
        sel.append(vis[n * ratio:])
        idx = torch.cat(sel)
        mask[t, idx] = True
    return mask


def _full_attn(T, i, x, cfg):
    p = f"model.layers.{i}.self_attn."
    B, S, _ = x.shape
    hd = cfg["head_dim"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    rot = int(hd * cfg.get("partial_rotary_factor", 1.0))
    theta = cfg.get("rope_theta")
    if theta is None:
        theta = (cfg.get("rope_parameters") or {}).get("rope_theta", 10000.0)
    cos, sin = _cos_sin(S, rot, float(theta))

    sel_mask = _indexer_mask(T, p, x, cos, sin, cfg)

    qg = F.linear(x, T[p + "q_proj.weight"]).view(B, S, -1, hd * 2)
    q, gate = torch.chunk(qg, 2, dim=-1)
    gate = gate.reshape(B, S, -1)

    q = _rms(q, T[p + "q_norm.weight"]).transpose(1, 2)

    k = _rms(F.linear(x, T[p + "k_proj.weight"]).view(B, S, -1, hd), T[p + "k_norm.weight"]).transpose(1, 2)
    v = F.linear(x, T[p + "v_proj.weight"]).view(B, S, -1, hd).transpose(1, 2)

    c = cos.unsqueeze(1)
    s = sin.unsqueeze(1)
    q = _rope(q, c, s)
    k = _rope(k, c, s)

    rep = nh // nkv
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    w = torch.matmul(q, k.transpose(2, 3)) * (hd ** -0.5)

    causal = torch.tril(torch.ones(S, S, dtype=torch.bool))
    allowed = causal & sel_mask
    add = torch.zeros(S, S, dtype=torch.float32).masked_fill(~allowed, float("-inf"))
    w = w + add[None, None]

    w = torch.softmax(w, dim=-1, dtype=torch.float32)

    out = torch.matmul(w, v).transpose(1, 2).reshape(B, S, -1)

    out = out * torch.sigmoid(gate)
    return F.linear(out, T[p + "o_proj.weight"])


def _forward(model, ids):
    cfg = model["config"]
    T = model["tensors"]
    hc, hid = cfg["hc_count"], cfg["hidden_size"]
    input_ids = torch.tensor([ids], dtype=torch.long)

    h = T["model.embed_tokens.weight"][input_ids]
    h = h.repeat(1, 1, hc)

    ple_ids = cfg.get("ple_layer_ids") or []

    for i in range(cfg["num_hidden_layers"]):
        if (i + 1) in ple_ids:
            h = h + _ple(T, i, h, input_ids, cfg)

        x, hin, inj = _hyper(T, f"model.layers.{i}.attn_hyper_connection.", h, hc, hid)

        lt = cfg["layer_types"][i]
        if lt == "linear_attention":
            x = _linear_attn(T, i, x, cfg)
        else:
            x = _full_attn(T, i, x, cfg)
        h = _combine(x, hin, inj)

        x, hin, inj = _hyper(T, f"model.layers.{i}.mlp_hyper_connection.", h, hc, hid)
        x = _mlp(T, i, x)
        h = _combine(x, hin, inj)

    h = _hyper(T, "model.hyper_connection_mixer.", h, hc, hid, with_inject=False)
    logits = F.linear(h, T["lm_head.weight"])
    return logits[:, -1, :]


def _ple(T, i, h, ids, cfg):
    """Per-Layer Embedding - compute PLE contribution for this layer."""
    p = f"model.layers.{i}.ple."
    hc, hid = cfg["hc_count"], cfg["hidden_size"]
    
    # Get n-gram embeddings from the embedding table
    # The embedding is computed based on input_ids with proper n-gram logic
    emb = _ngram_embedding(T, i, ids, cfg)
    
    # Project embeddings to key and value
    key = _rms(F.linear(emb, T[p + "key_proj.weight"]), T[p + "norm_key.weight"], group=hid).unflatten(-1, (hc, hid))
    value = F.linear(emb, T[p + "value_proj.weight"])
    
    # Compute query from current hidden state
    query = _rms(h, T[p + "norm_query.weight"], group=hid).unflatten(-1, (hc, hid))
    
    # Compute gating signal: (key * query).sum() / sqrt(hid)
    gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(hid)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    
    # Apply gating to value
    gv = torch.sigmoid(gate) * value.unsqueeze(-2)
    gv = gv.flatten(-2)
    
    # Apply short convolution
    gv_normed = _rms(gv, T[p + "norm_conv.weight"], group=hid)
    S = gv.shape[1]
    w = T[p + "conv1d.weight"]
    c = F.conv1d(gv_normed.transpose(1, 2), w, None, padding=w.shape[-1] - 1, groups=w.shape[0])[:, :, :S]
    c = c.transpose(1, 2)
    
    # Residual connection: gv + c
    return gv + c


def _ngram_embedding(T, i, ids, cfg):
    """Compute n-gram embeddings for PLE - matches Qwen4ExpTextNGramEmbedding.forward."""
    p = f"model.layers.{i}.ple.ple_embedding."
    
    # Configuration
    ngram_size = cfg["ngram_size"]
    hpn = cfg["heads_per_ngram"]
    eos = cfg["eos_token_id"]
    eos = eos[0] if isinstance(eos, list) else eos
    
    # Get tensors
    mult = T[p + "layer_multipliers"].long()
    offsets = T[p + "ngram_heads_offsets"].long()
    vocabs = T[p + "ngram_heads_vocab_sizes"].long()
    
    # Build embedding table from shards
    table = torch.cat(
        [T[p + f"ngram_embedding.shard_{k}.weight"] for k in range(cfg["split_ngram_parts"])], dim=0
    )
    
    # Get history with EOS padding at the beginning
    B, S = ids.shape
    ctx = ngram_size - 1
    hist = torch.cat([ids.new_full((B, ctx), eos), ids], dim=-1)
    
    # Shift right for each n-gram level
    shifted = []
    for s in range(ngram_size):
        if s == 0:
            shifted.append(hist)
        else:
            # Shift right by s positions, filling left with EOS
            shifted.append(torch.cat([hist.new_full((B, s), eos), hist[:, : hist.shape[-1] - s]], dim=-1))
    
    # Compute n-gram IDs using XOR with multipliers
    blocks = []
    for ng in range(2, ngram_size + 1):
        a = (ng - 2) * hpn
        b = a + hpn
        
        # Compute mixed IDs using XOR with multipliers
        mixed = shifted[0] * mult[0]
        for pos in range(1, ng):
            mixed = torch.bitwise_xor(mixed, shifted[pos] * mult[pos])
        
        # Apply remainder and add offsets
        nid = torch.remainder(mixed.unsqueeze(-1), vocabs[a:b].view(1, 1, -1))
        blocks.append(nid + offsets[a:b].view(1, 1, -1))
    
    # Get final n-gram IDs for current sequence positions
    nid = torch.cat(blocks, dim=-1)[:, -S:]
    
    # Clamp and lookup embeddings
    nid = nid.clamp(0, table.shape[0] - 1)
    return table[nid].flatten(-2)


def prefill(model: dict[str, Any], ids: list[int]) -> torch.Tensor:
    model["history"] = list(ids)
    with torch.no_grad():
        return _forward(model, model["history"])


def decode_step(model: dict[str, Any], token_id: int, position: int) -> torch.Tensor:
    hist = list(model.get("history", []))[:position]
    if len(hist) < position:
        hist = hist + [0] * (position - len(hist))
    hist.append(int(token_id))
    model["history"] = hist
    with torch.no_grad():
        return _forward(model, hist)


def close(model: dict[str, Any]) -> None:
    model["tensors"] = None
    model["config"] = None
    model["mesh_device"] = None
    model["history"] = None
