# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Stable Audio 3 diffusion transformer on one Tenstorrent chip (runtime design, step 1).

Design: docs/superpowers/specs/2026-10-28-stable-audio-runtime-design.md, sections 2 and 3.

What runs where. The library's own `generate` keeps running on the host: the T5Gemma encoder, the duration embedder, the noise draws, the
ping-pong sampler and the autoencoder decoder. This file replaces one module, the library's `ContinuousTransformer` (the stack of 20 layers
with its input and output projections and the 64 memory tokens), with `ChipTransformer`, which runs the stack with ttnn operations on one chip.
Three small pieces of that module stay on the host in float32 because they are tiny and sit before the stack: the timestep and global
embedders (they live in the library's `DiffusionTransformer`, outside `ContinuousTransformer`) and `global_cond_embedder` (1024 to 6144), whose
output is turned into the per-layer adaLN vectors on the host once per pass and uploaded.

The layer, in the library's order (`TransformerBlock.forward`, adaLN branch, `rms_norm` blocks, cross attention, local conditioning):

    scale_s, shift_s, gate_s, scale_f, shift_f, gate_f = chunk6(to_scale_shift_gate + global_cond)
    x = x + sigmoid(1 - gate_s) * self_attn(rms_norm(x) * (1 + scale_s) + shift_s)       # q and k RMS-normed per head, partial rotary
    x = x + cross_attn(rms_norm(x), context)
    x = x + pad_left(to_local_embed(local_add_cond))                                      # zero on the memory-token rows
    x = x + sigmoid(1 - gate_f) * ff(rms_norm(x) * (1 + scale_f) + shift_f)               # gated GLU, SiLU, biases

How the ttnn version differs from the library's arithmetic (each is a decision the numeric check measures; none is a claim of equality):
  * weights and activations are bfloat16, matrix multiplies accumulate in float32 (HiFi4);
  * the partial rotary (32 of the 64 head dimensions) is done as `q * cos + (q @ R) * sin`, where R is a constant 64 by 64 matrix with one
    signed one per column. The library does it in float32; here q is bfloat16;
  * the sequence is padded to a whole number of 32-row tiles; the padded key rows get a large negative additive mask in the softmax, so they
    receive no weight, as if they were absent;
  * the library's padding mask (`mask_padding_attention`) zeroes the value rows of padded latent positions before attention and still lets
    those positions take softmax weight. The same is done here (a value mask), because dropping those keys instead would change the result;
  * the attention scale is 1/sqrt(64), as the library's `scaled_dot_product_attention` uses by default.

Two layers of the module are plain torch so that a test can run the same decomposition without a chip: `host_layer` below is the layer written with
torch operations in exactly the decomposition the chip code uses (the rotary matrix, the masks, the vector forms). A test compares it with the
library's own `TransformerBlock` when the library is installed, and with a straightforward reading of the formulas above when it is not.
"""
from __future__ import annotations

import math
import time

NUM_HEADS = 16
HEAD_DIM = 64
ROT_DIM = 32                      # RotaryEmbedding(max(dim_heads // 2, 32)): 16 inverse frequencies, 32 rotated dimensions of 64
ROPE_BASE = 10000.0
QK_EPS = 1e-6                     # attn_kwargs qk_norm_eps default
NORM_EPS = 1e-5                   # RMSNorm default
TILE = 32
NEG = -1.0e9                      # additive mask value for keys that do not exist (bfloat16 holds it; softmax gives exactly zero)
MEMORY_TOKENS = 64


# The configuration this file implements. `config_problems` refuses anything else, by name, before a chip is opened: a model whose
# `model_config.json` differs in a key listed here (or carries a key not listed) would run through code that does not do what the config asks.
SUPPORTED_DIFFUSION = {"type": "dit", "diffusion_objective": ("rf_denoiser", "rectified_flow")}
SUPPORTED_NETWORK = {
    "io_channels": None, "embed_dim": None, "depth": None, "num_heads": None, "cond_token_dim": None, "global_cond_dim": None,
    "local_add_cond_dim": None,                      # None: any value; the shapes are read from the weights
    "global_cond_type": "adaLN", "timestep_features_type": "expo", "timestep_features_logsnr": False,
    "attn_kwargs": {"qk_norm": "rms", "differential": False}, "norm_type": "rms_norm", "norm_kwargs": {"force_fp32": True},
    "ff_kwargs": {"mult": 4.0}, "num_memory_tokens": MEMORY_TOKENS,
}


def config_problems(model_config) -> list[str]:
    """Reasons the chip path cannot run a model with this `model_config.json` (a parsed dict); empty when it can. Unknown keys count as reasons,
    because a key this file has never seen changes what the library computes in a way this file does not reproduce."""
    problems: list[str] = []
    try:
        diffusion = model_config["model"]["diffusion"]
        net = diffusion["config"]
    except (KeyError, TypeError):
        return ["model_config.json has no model.diffusion.config"]
    if not isinstance(diffusion, dict) or not isinstance(net, dict):
        return ["model.diffusion and its config must be objects"]
    for key, want in SUPPORTED_DIFFUSION.items():
        got = diffusion.get(key)
        ok = got in want if isinstance(want, tuple) else got == want
        if not ok:
            problems.append(f"model.diffusion.{key} is {got!r}; this path implements {want!r}")
    for key in net:
        if key not in SUPPORTED_NETWORK:
            problems.append(f"model.diffusion.config.{key} is not a setting this path implements")
    for key, want in SUPPORTED_NETWORK.items():
        if want is None:
            if key not in net:
                problems.append(f"model.diffusion.config.{key} is missing")
        elif net.get(key) != want:
            problems.append(f"model.diffusion.config.{key} is {net.get(key)!r}; this path implements {want!r}")
    embed, heads = net.get("embed_dim"), net.get("num_heads")
    if isinstance(embed, int) and isinstance(heads, int) and heads > 0 and embed // heads != HEAD_DIM:
        problems.append(f"the head dimension is {embed // heads}; this path implements {HEAD_DIM}")
    if isinstance(heads, int) and heads != NUM_HEADS:
        problems.append(f"num_heads is {heads}; this path implements {NUM_HEADS}")
    return problems


def padded_length(n: int, tile: int = TILE) -> int:
    """`n` rounded up to a whole number of tiles."""
    return -(-n // tile) * tile


def rotary_tables(torch, n: int, n_pad: int):
    """cos and sin tables, shape [n_pad, HEAD_DIM]. Positions are 0..n_pad-1 over the whole sequence (memory tokens included), the library's
    `forward_from_seq_len(x.shape[1])`. The first ROT_DIM columns carry the rotation; the rest have cos 1 and sin 0, so those dimensions pass
    through unchanged (the library's `t_unrotated`)."""
    inv = 1.0 / (ROPE_BASE ** (torch.arange(0, ROT_DIM, 2).float() / ROT_DIM))            # 16 values
    freqs = torch.einsum("i,j->ij", torch.arange(n_pad).float(), inv)                       # [n_pad, 16]
    freqs = torch.cat((freqs, freqs), dim=-1)                                               # [n_pad, 32]
    cos = torch.ones(n_pad, HEAD_DIM)
    sin = torch.zeros(n_pad, HEAD_DIM)
    cos[:, :ROT_DIM] = freqs.cos()
    sin[:, :ROT_DIM] = freqs.sin()
    return cos, sin


def rotate_matrix(torch):
    """R such that `t @ R` is the library's `rotate_half` on the first ROT_DIM columns of t and zero elsewhere. rotate_half(x) for a vector split
    in halves (x1, x2) is (-x2, x1), so output column j (j < 16) is -x[j + 16] and output column j (16 <= j < 32) is x[j - 16]."""
    half = ROT_DIM // 2
    r = torch.zeros(HEAD_DIM, HEAD_DIM)
    for j in range(half):
        r[j + half, j] = -1.0
        r[j, j + half] = 1.0
    return r


def apply_rotary(torch, t, cos, sin, rot):
    """`t` is [..., n, HEAD_DIM]; cos, sin are [n, HEAD_DIM]; rot is the matrix of `rotate_matrix`."""
    return t * cos + (t @ rot) * sin


def rms(torch, x, gamma, eps):
    """torch.nn.functional.rms_norm over the last dimension with a gain, in float32."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * gamma.float()).to(x.dtype)


def key_mask(torch, n_keys: int, n_keys_pad: int, n_queries_pad: int):
    """Additive attention mask [1, 1, n_queries_pad, n_keys_pad]: 0 on real keys, NEG on tile-padding keys."""
    m = torch.zeros(1, 1, n_queries_pad, n_keys_pad)
    m[..., n_keys:] = NEG
    return m


def value_mask(torch, padding_mask, n_pad: int, memory: int = MEMORY_TOKENS):
    """The library's V-zeroing mask as a [n_pad] vector of 0 and 1. Memory tokens are always valid; `padding_mask` ([T] bool, True = valid) covers the
    latent positions; the tile padding after them is 0 (those rows are masked as keys anyway)."""
    v = torch.zeros(n_pad)
    v[:memory] = 1.0
    t = padding_mask.shape[0] if padding_mask is not None else n_pad - memory
    t = min(t, n_pad - memory)
    v[memory:memory + t] = 1.0 if padding_mask is None else padding_mask[:t].float()
    return v


def modulation(torch, gate_param, global_cond):
    """The six adaLN vectors of one layer, prepared as the chip uses them: (1 + scale_s, shift_s, sigmoid(1 - gate_s), 1 + scale_f, shift_f,
    sigmoid(1 - gate_f)), each [dim] for one batch element. `gate_param` is the layer's `to_scale_shift_gate` ([6 * dim]); `global_cond` is the
    output of `global_cond_embedder` for this element ([6 * dim])."""
    s_s, h_s, g_s, s_f, h_f, g_f = (gate_param.float() + global_cond.float()).chunk(6, dim=-1)
    return 1 + s_s, h_s, torch.sigmoid(1 - g_s), 1 + s_f, h_f, torch.sigmoid(1 - g_f)


def host_layer(torch, x, p, mod, context, local_rows, row_mask, vmask, n_real, ctx_real, cos, sin, rot, trace=None):
    """One layer with torch operations in the chip's decomposition. Shapes (batch 1, no batch dimension):
    x [N, D] (N = padded sequence), context [Cp, D], local_rows [N, 257] or None (zero rows on the memory tokens and the tile padding), row_mask [N] (1 on the latent rows, 0 elsewhere), vmask [N],
    n_real the number of real rows (memory + latent), ctx_real the number of real context rows. `p` holds the layer's tensors by library name.
    Returns [N, D]. When `trace` is a dict, the stream after each section is stored in it ('attn', 'cross', 'local', 'ff')."""
    f = torch.nn.functional
    n = x.shape[0]
    sc_s, sh_s, g_s, sc_f, sh_f, g_f = mod

    def heads(t):                                     # [N, D] -> [H, N, 64]
        return t.view(t.shape[0], t.shape[1] // HEAD_DIM, HEAD_DIM).transpose(0, 1)

    def attend(q, k, v, kmask):                       # [H, Nq, 64], [H, Nk, 64]
        s = (q @ k.transpose(-1, -2)) / math.sqrt(HEAD_DIM) + kmask[0]
        return (s.softmax(-1) @ v).transpose(0, 1).reshape(q.shape[1], -1)

    h = rms(torch, x, p["pre_norm.gamma"], NORM_EPS) * sc_s + sh_s
    q, k, v = (h @ p["self_attn.to_qkv.weight"].t()).chunk(3, dim=-1)
    q, k, v = heads(q), heads(k), heads(v)
    q, k = rms(torch, q, p["self_attn.q_norm.gamma"], QK_EPS), rms(torch, k, p["self_attn.k_norm.gamma"], QK_EPS)
    q, k = apply_rotary(torch, q, cos, sin, rot), apply_rotary(torch, k, cos, sin, rot)
    v = v * vmask.view(1, n, 1)
    a = attend(q, k, v, key_mask(torch, n_real, n, n))
    x = x + (a @ p["self_attn.to_out.weight"].t()) * g_s
    if trace is not None:
        trace["attn"] = x

    h = rms(torch, x, p["cross_attend_norm.gamma"], NORM_EPS)
    q = heads(h @ p["cross_attn.to_q.weight"].t())
    k, v = (context @ p["cross_attn.to_kv.weight"].t()).chunk(2, dim=-1)
    k, v = heads(k), heads(v)
    q, k = rms(torch, q, p["cross_attn.q_norm.gamma"], QK_EPS), rms(torch, k, p["cross_attn.k_norm.gamma"], QK_EPS)
    a = attend(q, k, v, key_mask(torch, ctx_real, context.shape[0], n))
    x = x + a @ p["cross_attn.to_out.weight"].t()
    if trace is not None:
        trace["cross"] = x

    if local_rows is not None:
        le = f.silu(local_rows @ p["to_local_embed.0.weight"].t() + p["to_local_embed.0.bias"])
        le = le @ p["to_local_embed.2.weight"].t() + p["to_local_embed.2.bias"]
        x = x + le * row_mask.view(n, 1)
    if trace is not None:
        trace["local"] = x

    h = rms(torch, x, p["ff_norm.gamma"], NORM_EPS) * sc_f + sh_f
    z = h @ p["ff.ff.0.proj.weight"].t() + p["ff.ff.0.proj.bias"]
    a, gate = z.chunk(2, dim=-1)
    x = x + ((a * f.silu(gate)) @ p["ff.ff.2.weight"].t() + p["ff.ff.2.bias"]) * g_f
    if trace is not None:
        trace["ff"] = x
    return x


# ----------------------------------------------------------------------------------------------------------------------------------------
# The chip side. Everything below needs ttnn and a leased chip; nothing above does.
# ----------------------------------------------------------------------------------------------------------------------------------------

class ChipTransformer:
    """Drop-in for the library's `ContinuousTransformer` in `DiffusionTransformer.transformer`. Build it from the original module (its weights are
    copied to the chip and the host copies of the stack are not used again), then assign it with `install`."""

    def __init__(self, ttnn, torch, device, original, *, record=None, dtype=None, act_dtype=None):
        self.ttnn, self.torch, self.device = ttnn, torch, device
        self.record = record                      # callable(name, torch tensor) or None; layer outputs of the first pass are passed to it
        self.record_enabled = record is not None
        self.dtype = dtype if dtype is not None else ttnn.bfloat16        # weights and the attention tensors
        self.act = act_dtype if act_dtype is not None else ttnn.float32   # the residual stream and the outputs that are added to it
        self.timings = {"upload_weights_s": 0.0, "forward_s": 0.0, "passes": 0}
        self.kernel = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        t0 = time.time()
        self._load(original)
        self.timings["upload_weights_s"] = round(time.time() - t0, 3)
        self._consts = {}
        self.probe = None                          # a dict: when set, `_layer` stores the stream after each section (a diagnostic; costs one read back each)

    # -- weights -------------------------------------------------------------------------------------------------------------------------
    def _w(self, w):
        """A [out, in] torch weight as a [in, out] tile tensor on the chip."""
        return self.ttnn.from_torch(w.detach().float().t().contiguous(), dtype=self.dtype, layout=self.ttnn.TILE_LAYOUT, device=self.device)

    def _b(self, b):
        return self.ttnn.from_torch(b.detach().float().reshape(1, 1, 1, -1), dtype=self.dtype, layout=self.ttnn.TILE_LAYOUT, device=self.device)

    def _g(self, g):
        """An RMS gain as the layout `ttnn.rms_norm` takes: row-major [1, 1, dim / 32, 32]."""
        g = g.detach().float().reshape(1, 1, -1, TILE)
        return self.ttnn.from_torch(g, dtype=self.dtype, layout=self.ttnn.ROW_MAJOR_LAYOUT, device=self.device)

    def _load(self, m):
        torch, ttnn = self.torch, self.ttnn
        self.depth = len(m.layers)
        self.dim = m.project_in.weight.shape[0]
        self.num_memory = m.num_memory_tokens
        assert self.num_memory == MEMORY_TOKENS, "the memory-token count is a fixed part of this model's layout"
        self.w_in = self._w(m.project_in.weight)
        self.w_out = self._w(m.project_out.weight)
        self.memory = m.memory_tokens.detach().float().clone()                  # host copy; the memory-row table is built per length
        self.gcond = m.global_cond_embedder                                      # stays on the host (tiny)
        self.gate_params = [l.to_scale_shift_gate.detach().float().clone() for l in m.layers]
        self.layers = []
        for l in m.layers:
            d = {}
            d["pre_g"], d["cross_g"], d["ff_g"] = self._g(l.pre_norm.gamma), self._g(l.cross_attend_norm.gamma), self._g(l.ff_norm.gamma)
            d["qkv"], d["out"] = self._w(l.self_attn.to_qkv.weight), self._w(l.self_attn.to_out.weight)
            d["sq_g"], d["sk_g"] = self._g(l.self_attn.q_norm.gamma), self._g(l.self_attn.k_norm.gamma)
            d["cq"], d["ckv"], d["cout"] = self._w(l.cross_attn.to_q.weight), self._w(l.cross_attn.to_kv.weight), self._w(l.cross_attn.to_out.weight)
            d["cq_g"], d["ck_g"] = self._g(l.cross_attn.q_norm.gamma), self._g(l.cross_attn.k_norm.gamma)
            d["loc1"], d["loc1_b"] = self._w(l.to_local_embed[0].weight), self._b(l.to_local_embed[0].bias)
            d["loc2"], d["loc2_b"] = self._w(l.to_local_embed[2].weight), self._b(l.to_local_embed[2].bias)
            d["ff1"], d["ff1_b"] = self._w(l.ff.ff[0].proj.weight), self._b(l.ff.ff[0].proj.bias)
            d["ff2"], d["ff2_b"] = self._w(l.ff.ff[2].weight), self._b(l.ff.ff[2].bias)
            self.layers.append(d)
        self.local_dim = m.layers[0].to_local_embed[0].weight.shape[1]

    # -- per-length constants --------------------------------------------------------------------------------------------------------------
    def _up(self, t, *, layout=None, act=False, dtype=None):
        """Upload a torch tensor; `act=True` uses the residual-stream dtype, otherwise the weight dtype (or `dtype`, when given)."""
        ttnn = self.ttnn
        return ttnn.from_torch(t.float(), dtype=dtype or (self.act if act else self.dtype), layout=layout or ttnn.TILE_LAYOUT, device=self.device)

    def _length_consts(self, t_len: int, ctx_len: int, padding_mask):
        """Constants that depend only on the latent length, the context length and the padding mask: rotary tables, masks. Cached by those."""
        torch, ttnn = self.torch, self.ttnn
        pm_key = None if padding_mask is None else bytes(padding_mask.to(torch.uint8).flatten().tolist())
        key = (t_len, ctx_len, pm_key)
        if key in self._consts:
            return self._consts[key]
        n_real = self.num_memory + t_len
        n = padded_length(n_real)
        cp = padded_length(ctx_len)
        cos, sin = rotary_tables(torch, n_real, n)
        cos4 = cos.view(1, 1, n, HEAD_DIM).expand(1, NUM_HEADS, n, HEAD_DIM).contiguous()
        sin4 = sin.view(1, 1, n, HEAD_DIM).expand(1, NUM_HEADS, n, HEAD_DIM).contiguous()
        vm = value_mask(torch, None if padding_mask is None else padding_mask.reshape(-1), n, self.num_memory)
        vm4 = vm.view(1, 1, n, 1).expand(1, NUM_HEADS, n, HEAD_DIM).contiguous()
        row = torch.zeros(n)
        row[self.num_memory:self.num_memory + t_len] = 1.0
        mem = torch.zeros(n, self.dim)
        mem[:self.num_memory] = self.memory
        c = {
            "n_real": n_real, "n": n, "cp": cp,
            "cos": self._up(cos4), "sin": self._up(sin4), "vmask": self._up(vm4),
            "rowmask": self._up(row.view(1, 1, n, 1).expand(1, 1, n, self.dim).contiguous(), act=True),
            "mem": self._up(mem.view(1, 1, n, self.dim), act=True),
            "rot": self._up(rotate_matrix(torch).view(1, 1, HEAD_DIM, HEAD_DIM)),
            "self_mask": self._up(key_mask(torch, n_real, n, n), dtype=ttnn.bfloat16),
            "cross_mask": self._up(key_mask(torch, ctx_len, cp, n), dtype=ttnn.bfloat16),
        }
        self._consts[key] = c
        return c

    # -- ops -----------------------------------------------------------------------------------------------------------------------------
    def _lin(self, x, w, b=None, *, out=None):
        """x @ w + b with the operands in the weight dtype and the result in `out` (default the weight dtype; the residual-stream dtype for the
        outputs that are added to the residual, so a large cancelling sum is not rounded to bfloat16 before it is added)."""
        ttnn = self.ttnn
        if x.dtype != self.dtype:
            x = ttnn.typecast(x, self.dtype)
        return ttnn.linear(x, w, bias=b, dtype=out or self.dtype, compute_kernel_config=self.kernel)

    def _norm(self, x, g, eps):
        return self.ttnn.rms_norm(x, epsilon=eps, weight=g, compute_kernel_config=self.kernel)

    def _heads(self, x, n):
        """[1, 1, n, D] -> [1, H, n, 64]."""
        ttnn = self.ttnn
        x = ttnn.reshape(x, (1, n, NUM_HEADS, HEAD_DIM))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _merge(self, x, n):
        """[1, H, n, 64] -> [1, 1, n, D]."""
        ttnn = self.ttnn
        x = ttnn.permute(x, (0, 2, 1, 3))
        return ttnn.reshape(x, (1, 1, n, NUM_HEADS * HEAD_DIM))

    def _rope(self, t, c):
        ttnn = self.ttnn
        rot = ttnn.matmul(t, c["rot"], compute_kernel_config=self.kernel)
        return ttnn.add(ttnn.multiply(t, c["cos"]), ttnn.multiply(rot, c["sin"]))

    def _sdpa(self, q, k, v, mask):
        """Scaled dot product attention. The kernel takes bfloat16, so wider inputs are cast down and the result back."""
        ttnn = self.ttnn
        wide = q.dtype != ttnn.bfloat16
        if wide:
            q, k, v = (ttnn.typecast(t, ttnn.bfloat16) for t in (q, k, v))
        out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=False, scale=1.0 / math.sqrt(HEAD_DIM), compute_kernel_config=self.kernel)
        return ttnn.typecast(out, self.dtype) if wide else out

    def _layer(self, i, x, ctx, c, mods, local):
        ttnn = self.ttnn
        d, n = self.layers[i], c["n"]
        s_sc, s_sh, s_g, f_sc, f_sh, f_g = mods
        # self attention
        h = self._norm(x, d["pre_g"], NORM_EPS)
        h = ttnn.add(ttnn.multiply(h, s_sc), s_sh)
        qkv = self._lin(h, d["qkv"])
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(ttnn.reshape(qkv, (1, n, 3 * self.dim)), num_heads=NUM_HEADS,
                                                                         transpose_key=False)
        q, k = self._norm(q, d["sq_g"], QK_EPS), self._norm(k, d["sk_g"], QK_EPS)
        q, k = self._rope(q, c), self._rope(k, c)
        v = ttnn.multiply(v, c["vmask"])
        a = self._sdpa(q, k, v, c["self_mask"])
        a = self._lin(self._merge(a, n), d["out"], out=self.act)
        x = ttnn.add(x, ttnn.multiply(a, s_g))
        self._probe(i, "attn", x)
        # cross attention
        h = self._norm(x, d["cross_g"], NORM_EPS)
        q = self._heads(self._lin(h, d["cq"]), n)
        kv = self._lin(ctx, d["ckv"])
        k = self._heads(ttnn.slice(kv, (0, 0, 0, 0), (1, 1, c["cp"], self.dim)), c["cp"])
        v = self._heads(ttnn.slice(kv, (0, 0, 0, self.dim), (1, 1, c["cp"], 2 * self.dim)), c["cp"])
        q, k = self._norm(q, d["cq_g"], QK_EPS), self._norm(k, d["ck_g"], QK_EPS)
        a = self._sdpa(q, k, v, c["cross_mask"])
        x = ttnn.add(x, self._lin(self._merge(a, n), d["cout"], out=self.act))
        self._probe(i, "cross", x)
        # local conditioning (zero on the memory rows)
        if local is not None:
            le = ttnn.silu(self._lin(local, d["loc1"], d["loc1_b"]))
            le = self._lin(le, d["loc2"], d["loc2_b"], out=self.act)
            x = ttnn.add(x, ttnn.multiply(le, c["rowmask"]))
        self._probe(i, "local", x)
        # feed forward
        h = self._norm(x, d["ff_g"], NORM_EPS)
        h = ttnn.add(ttnn.multiply(h, f_sc), f_sh)
        z = self._lin(h, d["ff1"], d["ff1_b"])
        half = z.shape[-1] // 2
        a = ttnn.slice(z, (0, 0, 0, 0), (1, 1, n, half))
        g = ttnn.slice(z, (0, 0, 0, half), (1, 1, n, 2 * half))
        z = ttnn.multiply(a, ttnn.silu(g))
        z = self._lin(z, d["ff2"], d["ff2_b"], out=self.act)
        x = ttnn.add(x, ttnn.multiply(z, f_g))
        self._probe(i, "ff", x)
        return x

    def _probe(self, layer, name, x):
        if self.probe is not None and layer in self.probe_layers:
            self.probe[(layer, name)] = self.ttnn.to_torch(x).reshape(x.shape[-2], -1).float()

    probe_layers = (0,)

    # -- the module interface ---------------------------------------------------------------------------------------------------------------
    def __call__(self, x, context=None, prepend_embeds=None, global_cond=None, local_add_cond=None, modular_local_cond=None, return_info=False,
                 use_checkpointing=True, exit_layer_ix=None, padding_mask=None, **kwargs):
        torch, ttnn = self.torch, self.ttnn
        if prepend_embeds is not None or modular_local_cond is not None or exit_layer_ix is not None or return_info:
            raise NotImplementedError("this model's text-to-music path uses no prepend embeddings, modular local conditioning, early exit or info")
        if context is None or global_cond is None:
            raise NotImplementedError("the model needs a cross-attention context and a global condition")
        t0 = time.time()
        out = []
        with torch.no_grad():
            for b in range(x.shape[0]):
                out.append(self._one(x[b], context[b], global_cond[b], None if local_add_cond is None else local_add_cond[b],
                                     None if padding_mask is None else padding_mask[b]))
        self.timings["forward_s"] += time.time() - t0
        self.timings["passes"] += 1
        self.record_enabled = False                                        # the first pass only is recorded, as the reference does
        return torch.stack(out).to(x.dtype)

    def _one(self, x, ctx, gcond, local, pmask):
        torch, ttnn = self.torch, self.ttnn
        t_len, ctx_len = x.shape[0], ctx.shape[0]
        c = self._length_consts(t_len, ctx_len, pmask)
        n = c["n"]
        # adaLN vectors for every layer, once per pass, on the host in float32
        gc = self.gcond(gcond.float().unsqueeze(0))[0]
        mods = []
        for i in range(self.depth):
            vecs = modulation(torch, self.gate_params[i], gc)
            mods.append(tuple(self._up(v.view(1, 1, 1, -1), act=True) for v in vecs))
        xin = torch.zeros(n, x.shape[-1])
        xin[self.num_memory:self.num_memory + t_len] = x.float()
        h = ttnn.add(self._lin(self._up(xin.view(1, 1, n, -1)), self.w_in, out=self.act), c["mem"])
        cpad = torch.zeros(c["cp"], ctx.shape[-1])
        cpad[:ctx_len] = ctx.float()
        ctx_t = self._up(cpad.view(1, 1, c["cp"], -1))
        local_t = None
        if local is not None:
            lpad = torch.zeros(n, self.local_dim)
            lpad[self.num_memory:self.num_memory + t_len] = local.float()
            local_t = self._up(lpad.view(1, 1, n, -1))
        self._probe(0, "in", h)
        for i in range(self.depth):
            h = self._layer(i, h, ctx_t, c, mods[i], local_t)
            if self.record_enabled:
                self.record(f"layer_{i:02d}", ttnn.to_torch(h).reshape(n, -1)[:c["n_real"]].float().unsqueeze(0))
        y = self._lin(h, self.w_out, out=self.act)
        y = ttnn.to_torch(y).reshape(n, -1).float()
        return y[self.num_memory:self.num_memory + t_len]


def install(model_dit, chip):
    """Replace `DiffusionTransformer.transformer` (the library's `ContinuousTransformer`) with `chip`. Returns the original module."""
    original = model_dit.transformer
    del model_dit._modules["transformer"]                 # an nn.Module refuses a plain object in a child slot
    object.__setattr__(model_dit, "transformer", chip)
    return original
