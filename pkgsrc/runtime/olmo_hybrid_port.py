# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Port module for stage 3 of the full port check: OlmoHybrid (gated-delta-net + full attention, NoPE) on multi-chip mesh."""
from __future__ import annotations

import glob
import json
import math
import os

import torch
import torch.nn.functional as F
import ttnn
from safetensors import safe_open


def read_tensors(model_dir):
    """Every tensor of the checkpoint as {name: torch tensor}, from one file or from shards."""
    out = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for name in f.keys():
                out[name] = f.get_tensor(name)
    return out


BIG_DTYPE = ttnn.bfloat8_b
SMALL_DTYPE = ttnn.bfloat16


def _rms(x, w, eps):
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _rotate(x, cos, sin):
    # x [S, h, d]; cos/sin [S, rd]
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    x1, x2 = xr[..., : rd // 2], xr[..., rd // 2:]
    rot = torch.cat((-x2, x1), dim=-1)
    xr = xr * cos[:, None, :] + rot * sin[:, None, :]
    return torch.cat([xr, xp], dim=-1)


def _weight_getter(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            mapping = json.load(f)["weight_map"]
        files = sorted(set(mapping.values()))
    else:
        mapping = None
        files = ["model.safetensors"]
    handles = {fn: safe_open(os.path.join(model_dir, fn), framework="pt") for fn in files}

    def get(name):
        fn = mapping[name] if mapping is not None else "model.safetensors"
        return handles[fn].get_tensor(name)

    return get


def _rope_setup(model_dir, cfg, head_dim):
    """Return (inv_freq float32 [rd/2], attention_scaling) or (None, 1.0) for NoPE.

    OlmoHybrid runs NoPE when rope_parameters.rope_theta is None (the reference builds no rotary_emb then)."""
    rp = cfg.get("rope_parameters")
    if isinstance(rp, dict) and isinstance(rp.get("full_attention"), dict):
        rp = rp["full_attention"]
    if rp is None and cfg.get("rope_scaling"):
        rp = dict(cfg["rope_scaling"])
        rp.setdefault("rope_theta", cfg.get("rope_theta"))
    if rp is None and cfg.get("rope_theta") is not None:
        rp = {"rope_type": "default", "rope_theta": cfg["rope_theta"]}
    if not isinstance(rp, dict):
        print("rope: no rope parameters in config, running NoPE")
        return None, 1.0
    rtype = rp.get("rope_type", rp.get("type", "default")) or "default"
    theta = rp.get("rope_theta")
    if theta is None:
        theta = cfg.get("rope_theta")
    if theta is None or str(rtype).lower() in ("none", "nope"):
        print("rope: rope_theta is None, running NoPE")
        return None, 1.0
    base = float(theta)
    partial = float(rp.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0)) or 1.0)
    dim = int(head_dim * partial)
    max_pos = cfg.get("max_position_embeddings", 4096)

    if rtype != "default":
        try:  # exact reference code when the environment has it
            from transformers import AutoConfig
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            hc = AutoConfig.from_pretrained(model_dir)
            inv_t, sc = ROPE_INIT_FUNCTIONS[rtype](hc, "cpu")
            if inv_t.numel() == dim // 2:
                print(f"rope: {rtype} via transformers, theta {base}, scaling {float(sc):.6f}")
                return inv_t.float().cpu(), float(sc)
        except Exception as e:  # noqa: BLE001
            print(f"rope: transformers init not usable ({type(e).__name__}); using built-in {rtype}")

    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    scaling = 1.0
    if rtype in ("default", "dynamic"):
        pass
    elif rtype == "linear":
        inv = inv / float(rp["factor"])
    elif rtype == "llama3":
        factor = float(rp["factor"])
        lff = float(rp["low_freq_factor"])
        hff = float(rp["high_freq_factor"])
        old = float(rp["original_max_position_embeddings"])
        low_wl, high_wl = old / lff, old / hff
        wavelen = 2 * math.pi / inv
        inv_l = torch.where(wavelen > low_wl, inv / factor, inv)
        smooth = (old / wavelen - lff) / (hff - lff)
        smoothed = (1 - smooth) * inv_l / factor + smooth * inv_l
        medium = ~(wavelen < high_wl) & ~(wavelen > low_wl)
        inv = torch.where(medium, smoothed, inv_l)
    elif rtype == "yarn":
        if rp.get("original_max_position_embeddings"):
            orig = float(rp["original_max_position_embeddings"])
            factor = float(max_pos) / orig
        else:
            orig = float(max_pos)
            factor = float(rp["factor"])
        af = rp.get("attention_factor")
        mscale, mscale_all = rp.get("mscale"), rp.get("mscale_all_dim")

        def get_mscale(scale, m=1.0):
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

        if af is None:
            if mscale and mscale_all:
                af = get_mscale(factor, mscale) / get_mscale(factor, mscale_all)
            else:
                af = get_mscale(factor)
        beta_fast = float(rp.get("beta_fast") or 32)
        beta_slow = float(rp.get("beta_slow") or 1)
        truncate = rp.get("truncate", True)

        def corr_dim(nrot):
            return (dim * math.log(orig / (nrot * 2 * math.pi))) / (2 * math.log(base))

        low, high = corr_dim(beta_fast), corr_dim(beta_slow)
        if truncate:
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
        inv_extra = 1.0 / pos_freqs
        inv_inter = 1.0 / (factor * pos_freqs)
        extra_f = 1 - ramp
        inv = inv_inter * (1 - extra_f) + inv_extra * extra_f
        scaling = float(af)
    else:
        raise NotImplementedError(f"rope_type {rtype} is not supported by this port")
    print(f"rope: {rtype} built-in, theta {base}, dim {dim}, scaling {scaling:.6f}")
    return inv.float(), scaling


class OlmoHybridPort:
    def __init__(self, model_dir, mesh_device):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        cfg = cfg.get("text_config", cfg) if "layer_types" not in cfg else cfg
        self.cfg = cfg
        self.H = cfg["hidden_size"]
        self.L = cfg["num_hidden_layers"]
        self.types = cfg["layer_types"]
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg.get("num_key_value_heads") or self.nh
        self.hd = cfg.get("head_dim") or self.H // self.nh
        self.lk = cfg["linear_num_key_heads"]
        self.lv = cfg["linear_num_value_heads"]
        self.dk = cfg["linear_key_head_dim"]
        self.dv = cfg["linear_value_head_dim"]
        self.key_dim = self.lk * self.dk
        self.value_dim = self.lv * self.dv
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.K = int(cfg.get("linear_conv_kernel_dim", 4))
        self.neg = bool(cfg.get("linear_allow_neg_eigval", False))
        self.gated_eps = 1e-5  # effective_construction: OlmoHybridRMSNormGated eps resolved 1e-05

        self.dev = mesh_device
        # Detect number of devices from mesh
        self.ndev = 1
        try:
            if hasattr(mesh_device, 'get_num_devices'):
                self.ndev = int(mesh_device.get_num_devices())
        except Exception:  # noqa: BLE001
            pass
        
        # Create mesh mapper - replicate by default for simplicity
        # For a proper multi-chip port, some weights would be sharded instead
        self.rep = None
        if hasattr(mesh_device, "get_num_devices") and self.ndev > 1:
            try:
                self.rep = ttnn.ReplicateTensorToMesh(mesh_device)
            except Exception:  # noqa: BLE001
                self.rep = None
        self.ckc = None
        try:
            self.ckc = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        except Exception:  # noqa: BLE001
            try:
                self.ckc = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                    fp32_dest_acc_en=True, packer_l1_acc=True)
            except Exception:  # noqa: BLE001
                self.ckc = None
        self._dev_tensors = []

        self.inv_freq, self.rope_scaling = _rope_setup(model_dir, cfg, self.hd)

        get = _weight_getter(model_dir)
        self.embed = get("model.embed_tokens.weight")  # host bf16 lookup table
        self.norm_w = get("model.norm.weight").float()
        self.lm_head = self._w(get("lm_head.weight"), SMALL_DTYPE)

        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            P = {}
            if self.types[i] == "linear_attention":
                a = p + "linear_attn."
                P["in_ln"] = get(p + "input_layernorm.weight").float()
                P["post_ln"] = get(p + "post_attention_layernorm.weight").float()
                P["wqkvg"] = self._w(torch.cat([get(a + "q_proj.weight"), get(a + "k_proj.weight"),
                                                get(a + "v_proj.weight"), get(a + "g_proj.weight")], 0), BIG_DTYPE)
                P["wab"] = self._w(torch.cat([get(a + "a_proj.weight"), get(a + "b_proj.weight")], 0), SMALL_DTYPE)
                P["wo"] = self._w(get(a + "o_proj.weight"), BIG_DTYPE)
                P["conv"] = torch.cat([get(a + "q_conv1d.weight"), get(a + "k_conv1d.weight"),
                                       get(a + "v_conv1d.weight")], 0).float().reshape(self.conv_dim, self.K)
                P["A"] = get(a + "A_log").float()
                P["dt"] = get(a + "dt_bias").float()
                P["onorm"] = get(a + "o_norm.weight").float()
            elif self.types[i] == "full_attention":
                a = p + "self_attn."
                P["wqkv"] = self._w(torch.cat([get(a + "q_proj.weight"), get(a + "k_proj.weight"),
                                               get(a + "v_proj.weight")], 0), BIG_DTYPE)
                P["wo"] = self._w(get(a + "o_proj.weight"), BIG_DTYPE)
                P["qn"] = get(a + "q_norm.weight").float()
                P["kn"] = get(a + "k_norm.weight").float()
                P["post_ln"] = get(p + "post_attention_layernorm.weight").float()
                P["post_ff_ln"] = get(p + "post_feedforward_layernorm.weight").float()
            else:
                raise ValueError(f"layer {i}: unknown layer type {self.types[i]}")
            P["wg"] = self._w(get(p + "mlp.gate_proj.weight"), BIG_DTYPE)
            P["wu"] = self._w(get(p + "mlp.up_proj.weight"), BIG_DTYPE)
            P["wd"] = self._w(get(p + "mlp.down_proj.weight"), BIG_DTYPE)
            self.layers.append(P)
            print(f"loaded layer {i} ({self.types[i]})", flush=True)
        self._reset()

    # ---------------------------------------------------------------- device plumbing
    def _w(self, w, dtype):
        """Place a weight on the device. Replicated by default for simplicity."""
        t = w.t().contiguous()
        t = t.reshape(1, 1, t.shape[0], t.shape[1])
        kw = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.rep is not None:
            kw["mesh_mapper"] = self.rep
        tt = ttnn.from_torch(t, **kw)
        self._dev_tensors.append(tt)
        return tt

    def _up(self, x):
        """Move input to device."""
        t = x.to(torch.bfloat16).reshape(1, 1, x.shape[0], x.shape[1])
        kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev,
                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.rep is not None:
            kw["mesh_mapper"] = self.rep
        return ttnn.from_torch(t, **kw)

    def _down(self, t):
        """Move output from device to host."""
        x = None
        if self.ndev > 1:
            try:
                x = ttnn.to_torch(ttnn.get_device_tensors(t)[0])
            except Exception:  # noqa: BLE001
                x = None
        if x is None:
            try:
                x = ttnn.to_torch(t)
            except Exception:  # noqa: BLE001
                x = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.dev, dim=0))[:1]
        x = x.float()
        return x.reshape(-1, x.shape[-1])

    def _mm(self, xt, w):
        """Matrix multiply on device."""
        if self.ckc is not None:
            return ttnn.linear(xt, w, compute_kernel_config=self.ckc)
        return ttnn.linear(xt, w)

    def _proj(self, x, w):
        """Project: move to device, matmul, move back."""
        xt = self._up(x)
        yt = self._mm(xt, w)
        y = self._down(yt)
        ttnn.deallocate(yt)
        ttnn.deallocate(xt)
        return y

    def _mlp(self, x, P):
        """MLP block: up-projection, silu, gate, down-projection."""
        xt = self._up(x)
        g = self._mm(xt, P["wg"])
        u = self._mm(xt, P["wu"])
        ttnn.deallocate(xt)
        gs = ttnn.silu(g)
        ttnn.deallocate(g)
        hm = ttnn.multiply(gs, u)
        ttnn.deallocate(gs)
        ttnn.deallocate(u)
        d = self._mm(hm, P["wd"])
        ttnn.deallocate(hm)
        y = self._down(d)
        ttnn.deallocate(d)
        return y

    # ---------------------------------------------------------------- state
    def _reset(self):
        self.conv_state = {}
        self.rec = {}
        self.kc = {}
        self.vc = {}
        for i, t in enumerate(self.types):
            if t == "linear_attention":
                self.conv_state[i] = torch.zeros(self.K - 1, self.conv_dim)
                self.rec[i] = torch.zeros(self.lv, self.dk, self.dv)
            else:
                self.kc[i] = None
                self.vc[i] = None

    # ---------------------------------------------------------------- layers
    def _delta(self, st, q, k, v, g, beta):
        # q,k [S,h,dk]; v [S,h,dv]; g, beta [S,h]; st [h,dk,dv]  (torch_recurrent_gated_delta_rule)
        q = _l2norm(q) * (1.0 / math.sqrt(q.shape[-1]))
        k = _l2norm(k)
        S = q.shape[0]
        out = torch.empty(S, v.shape[1], v.shape[2])
        for t in range(S):
            st = st * g[t].exp()[:, None, None]
            kv = torch.einsum("hkv,hk->hv", st, k[t])
            delta = (v[t] - kv) * beta[t][:, None]
            st = st + k[t][:, :, None] * delta[:, None, :]
            out[t] = torch.einsum("hkv,hk->hv", st, q[t])
        return out, st

    def _linear_layer(self, i, h, P):
        S = h.shape[0]
        x = _rms(h, P["in_ln"], self.eps)
        xt = self._up(x)
        yt = self._mm(xt, P["wqkvg"])
        abt = self._mm(xt, P["wab"])
        ttnn.deallocate(xt)
        y = self._down(yt)
        ab = self._down(abt)
        ttnn.deallocate(yt)
        ttnn.deallocate(abt)
        mixed = y[:, : self.conv_dim]
        gate = y[:, self.conv_dim: self.conv_dim + self.value_dim]
        a = ab[:, : self.lv]
        b = ab[:, self.lv: 2 * self.lv]

        xx = torch.cat([self.conv_state[i], mixed], 0)  # [K-1+S, C]
        w = P["conv"]
        conv = torch.zeros_like(mixed)
        for j in range(self.K):
            conv = conv + xx[j: j + S] * w[:, j]
        self.conv_state[i] = xx[-(self.K - 1):].clone()
        conv = F.silu(conv)

        q = conv[:, : self.key_dim].reshape(S, self.lk, self.dk)
        k = conv[:, self.key_dim: 2 * self.key_dim].reshape(S, self.lk, self.dk)
        v = conv[:, 2 * self.key_dim:].reshape(S, self.lv, self.dv)
        if self.lv > self.lk:
            r = self.lv // self.lk
            q = q.repeat_interleave(r, dim=1)
            k = k.repeat_interleave(r, dim=1)
        beta = torch.sigmoid(b)
        if self.neg:
            beta = beta * 2.0
        g = -P["A"].exp() * F.softplus(a + P["dt"])
        core, self.rec[i] = self._delta(self.rec[i], q, k, v, g, beta)

        core = core.reshape(S * self.lv, self.dv)
        gate = gate.reshape(S * self.lv, self.dv)
        var = core.pow(2).mean(-1, keepdim=True)
        core = core * torch.rsqrt(var + self.gated_eps) * P["onorm"] * F.silu(gate)
        core = core.reshape(S, self.value_dim)
        h = h + self._proj(core, P["wo"])
        h = h + self._mlp(_rms(h, P["post_ln"], self.eps), P)
        return h

    def _attn_layer(self, i, h, P, pos0):
        S = h.shape[0]
        qd, kd = self.nh * self.hd, self.nkv * self.hd
        qkv = self._proj(h, P["wqkv"])  # no input norm in this layer type (post-norm layer)
        q = _rms(qkv[:, :qd], P["qn"], self.eps).reshape(S, self.nh, self.hd)
        k = _rms(qkv[:, qd: qd + kd], P["kn"], self.eps).reshape(S, self.nkv, self.hd)
        v = qkv[:, qd + kd: qd + 2 * kd].reshape(S, self.nkv, self.hd)
        if self.inv_freq is not None:  # NoPE when None
            pos = torch.arange(pos0, pos0 + S, dtype=torch.float32)
            freqs = pos[:, None] * self.inv_freq[None, :]
            emb = torch.cat([freqs, freqs], -1)
            cos = emb.cos() * self.rope_scaling
            sin = emb.sin() * self.rope_scaling
            q = _rotate(q, cos, sin)
            k = _rotate(k, cos, sin)
        self.kc[i] = k if self.kc[i] is None else torch.cat([self.kc[i], k], 0)
        self.vc[i] = v if self.vc[i] is None else torch.cat([self.vc[i], v], 0)
        Kc, Vc = self.kc[i], self.vc[i]
        if self.nkv < self.nh:
            r = self.nh // self.nkv
            Kc = Kc.repeat_interleave(r, dim=1)
            Vc = Vc.repeat_interleave(r, dim=1)
        T = Kc.shape[0]
        scores = torch.einsum("shd,thd->hst", q, Kc) * (self.hd ** -0.5)
        qpos = torch.arange(T - S, T)
        kpos = torch.arange(T)
        mask = kpos[None, :] > qpos[:, None]
        scores = scores.masked_fill(mask[None], float("-inf"))
        p = torch.softmax(scores, dim=-1)
        out = torch.einsum("hst,thd->shd", p, Vc).reshape(S, qd)
        o = self._proj(out, P["wo"])
        h = h + _rms(o, P["post_ln"], self.eps)
        m = self._mlp(h, P)
        h = h + _rms(m, P["post_ff_ln"], self.eps)
        return h

    def forward(self, ids, pos0, collect=None):
        h = self.embed[torch.tensor(ids, dtype=torch.long)].float()
        for i, P in enumerate(self.layers):
            if self.types[i] == "linear_attention":
                h = self._linear_layer(i, h, P)
            else:
                h = self._attn_layer(i, h, P, pos0)
            if collect is not None:
                collect.append(h.clone())
        x = _rms(h[-1:], self.norm_w, self.eps)
        logits = self._proj(x, self.lm_head)
        return logits[0]

    def close(self):
        for t in self._dev_tensors:
            try:
                ttnn.deallocate(t)
            except Exception:  # noqa: BLE001
                pass
        self._dev_tensors = []
        self.layers = []


def layer_outputs(model, ids):
    model._reset()
    outs = []
    model.forward([int(t) for t in ids], 0, collect=outs)
    return outs


def load(model_dir, mesh_device):
    """Load the model onto the mesh device."""
    return OlmoHybridPort(model_dir, mesh_device)


def prefill(model, ids):
    """Run the prompt and return logits for the next token."""
    model._reset()
    return model.forward([int(t) for t in ids], 0)


def decode_step(model, token_id, position):
    """Append token_id at position and return logits for the next token."""
    return model.forward([int(token_id)], int(position))


def close(model):
    """Release resources."""
    model.close()
