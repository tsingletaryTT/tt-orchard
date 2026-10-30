# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Plain-torch CPU reference for MiniCPM3-4B (multi-latent attention, dense gated-SiLU MLP).

Written from the model's own `modeling_minicpm.py` (openbmb/MiniCPM3-4B, Apache-2.0), because that file targets transformers 4.41 and the shared
environment has 5.x. `check_against_remote.py` runs the two on the same prompt to show that this file computes what the model's code computes.

Every op the chip port replaces is a small function so a probe can read the stream after it. `forward(..., tap=fn)` calls `fn(name, tensor)` with
the residual stream after each named step: `embed`, `layer_NN.attn`, `layer_NN.mlp` (the block output, also called `layer_NN`), `final_norm`,
`logits`. The attention internals (`layer_NN.q`, `.k`, `.v`, `.attn_out`) are tapped too, for the first-divergence probe.
"""
from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F
from safetensors import safe_open


def rms(x, w, eps):
    xf = x.float()
    return (w.float() * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))).to(x.dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_tables(cfg, n):
    """cos and sin tables [n, rope_dim] in float32, as MiniCPMLongRoPE builds them."""
    dim = cfg["qk_rope_head_dim"]
    rs = cfg["rope_scaling"]
    base = cfg.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    orig = rs["original_max_position_embeddings"]
    ext = torch.tensor(rs["long_factor"] if n > orig else rs["short_factor"], dtype=torch.float32)
    scale = cfg["max_position_embeddings"] / orig
    scaling = math.sqrt(1 + math.log(scale) / math.log(orig))
    t = torch.arange(n, dtype=torch.float32)
    freqs = torch.outer(t, 1.0 / ext) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos() * scaling, emb.sin() * scaling


class RefModel:
    def __init__(self, model_dir, weights_file, dtype=torch.float32, max_len=2048):
        self.cfg = json.load(open(f"{model_dir}/config.json"))
        self.dtype = dtype
        c = self.cfg
        self.nl = c["num_hidden_layers"]
        self.H = c["num_attention_heads"]
        # the value head width is hidden size over head count (the model's code, and the native transformers class, derive it that way: the config has no v_head_dim)
        self.dn, self.dr, self.dv = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["hidden_size"] // c["num_attention_heads"]
        self.eps = c["rms_norm_eps"]
        self.w = {}
        with safe_open(weights_file, "pt") as f:
            for k in f.keys():
                self.w[k] = f.get_tensor(k).to(dtype)
        self.cos, self.sin = rope_tables(c, max_len)
        self.cache = None

    def reset(self):
        self.cache = [None] * self.nl

    def attn(self, i, x, pos, tap):
        c, w, H = self.cfg, self.w, self.H
        p = f"model.layers.{i}.self_attn."
        B, S, _ = x.shape
        q = F.linear(rms(F.linear(x, w[p + "q_a_proj.weight"]), w[p + "q_a_layernorm.weight"], self.eps), w[p + "q_b_proj.weight"])
        q = q.view(B, S, H, self.dn + self.dr).transpose(1, 2)
        q_nope, q_pe = q[..., : self.dn], q[..., self.dn:]
        ckv = F.linear(x, w[p + "kv_a_proj_with_mqa.weight"])
        ckv, k_pe = ckv[..., : c["kv_lora_rank"]], ckv[..., c["kv_lora_rank"]:]
        k_pe = k_pe.view(B, S, 1, self.dr).transpose(1, 2)
        kv = F.linear(rms(ckv, w[p + "kv_a_layernorm.weight"], self.eps), w[p + "kv_b_proj.weight"])
        kv = kv.view(B, S, H, self.dn + self.dv).transpose(1, 2)
        k_nope, v = kv[..., : self.dn], kv[..., self.dn:]
        cos = self.cos[pos].to(x.dtype if x.dtype != torch.bfloat16 else torch.bfloat16)[None, None]
        sin = self.sin[pos].to(cos.dtype)[None, None]
        qf, kf = q_pe.float(), k_pe.float()
        q_pe = (qf * cos.float() + rotate_half(qf) * sin.float()).to(x.dtype)
        k_pe = (kf * cos.float() + rotate_half(kf) * sin.float()).to(x.dtype)
        qs = torch.cat([q_nope, q_pe], dim=-1)
        ks = torch.cat([k_nope, k_pe.expand(-1, H, -1, -1)], dim=-1)
        if self.cache[i] is not None:
            ks = torch.cat([self.cache[i][0], ks], dim=2)
            v = torch.cat([self.cache[i][1], v], dim=2)
        self.cache[i] = (ks, v)
        T = ks.shape[2]
        s = torch.matmul(qs, ks.transpose(2, 3)) * ((self.dn + self.dr) ** -0.5)
        mask = torch.full((S, T), float("-inf")).triu(T - S + 1)
        s = s + mask.to(s.dtype)
        a = torch.softmax(s.float(), dim=-1).to(x.dtype)
        o = torch.matmul(a, v)
        tap(f"layer_{i:02d}.q", qs.transpose(1, 2).reshape(B, S, -1))
        tap(f"layer_{i:02d}.attn_core", o.transpose(1, 2).reshape(B, S, -1))
        o = o.transpose(1, 2).reshape(B, S, H * self.dv)
        return F.linear(o, w[p + "o_proj.weight"])

    def mlp(self, i, x):
        w = self.w
        p = f"model.layers.{i}.mlp."
        return F.linear(F.silu(F.linear(x, w[p + "gate_proj.weight"])) * F.linear(x, w[p + "up_proj.weight"]), w[p + "down_proj.weight"])

    @torch.no_grad()
    def forward(self, ids, start=0, tap=lambda n, t: None):
        """ids [B, S] int; returns logits [B, S, vocab] float32. Positions start at `start`; the cache holds the earlier ones."""
        if self.cache is None:
            self.reset()
        c = self.cfg
        if start + ids.shape[1] > self.cos.shape[0]:
            raise ValueError(f"the rope table holds {self.cos.shape[0]} positions and this call needs {start + ids.shape[1]}; build the model with a larger max_len "
                             "(the prompt length plus the decode length)")
        pos = torch.arange(start, start + ids.shape[1])
        x = F.embedding(ids, self.w["model.embed_tokens.weight"]) * c["scale_emb"]
        tap("embed", x)
        rs = c["scale_depth"] / math.sqrt(self.nl)
        for i in range(self.nl):
            p = f"model.layers.{i}."
            h = rms(x, self.w[p + "input_layernorm.weight"], self.eps)
            x = x + self.attn(i, h, pos, tap) * rs
            tap(f"layer_{i:02d}.attn", x)
            h = rms(x, self.w[p + "post_attention_layernorm.weight"], self.eps)
            x = x + self.mlp(i, h) * rs
            tap(f"layer_{i:02d}", x)
        x = rms(x, self.w["model.norm.weight"], self.eps)
        tap("final_norm", x)
        logits = F.linear(x / (c["hidden_size"] / c["dim_model_base"]), self.w["model.embed_tokens.weight"]).float()
        tap("logits", logits)
        return logits
