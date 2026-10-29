# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Stable Audio 3 autoencoder decoder's transformer on one Tenstorrent chip (runtime design, step 2; issue 161).

Design: docs/superpowers/specs/2026-10-28-stable-audio-runtime-design.md, section 3 (step 2). Run log: docs/run-logs/2026-10-29-overnight-stable-audio-decoder.md.

What this replaces. The library's `AudioAutoencoder.decoder` is a `SAMEDecoder`: a Linear from the 256-wide latent to 768, then one
`TransformerResamplingBlock` (6 transformer layers, then a Conv1d from 768 to 512 with kernel 3). `ChipDecoder` is a drop-in for that module. The Linear, the
token layout, the learned "new tokens" and their noise, the shift between the two halves and the final convolution stay on the host in float32 (they are small
and the noise draw has to come from the same torch generator as the reference). The 6 layers run on the chip. These two things had no ttnn code before:

  * dynamic-tanh normalisation, `gamma * tanh(alpha * x) + beta` with a scalar `alpha` (the norm of every layer and of the per-head q and k);
  * differential attention: two softmax attentions with separate q and k and shared v, `out = attn(q, k, v) - attn(q_diff, k_diff, v)`, then `to_out`.

Layer, in the library's order (`TransformerBlock.forward`, the branch without global conditioning, `dyt` norms, no cross attention, no conformer):

    x = x + to_out(softmax(q k^T / 8) v - softmax(q' k'^T / 8) v)     q, k, q', k' = rotary(dyt_head(heads(to_qkv(dyt(x)))))
    x = x + ff(dyt(x))                                                 ff = (a * silu(gate)) @ W2 + b2, with (a, gate) = chunk2(x @ W1 + b1)

Sequences are short: a chunk is 34 tokens (2 latent frames of 17 tokens: one input token and 16 new tokens). The chunks of a clip form the batch dimension.
Each chunk is padded from 34 to 64 rows (two tiles); the padded keys get a large negative additive mask. The rotary positions are 0 to 33 inside each chunk
(`forward_from_seq_len(34)` per chunk). The second half of the layers sees chunks shifted by 17 tokens, with the first and last 17 tokens repeated at the ends.

The torch functions here (`host_layer`) are the same layer in the chip's decomposition, so a test can check the decomposition against the library without a chip.
"""
from __future__ import annotations

import math
import time

HEAD_DIM = 64
ROT_DIM = 32
ROPE_BASE = 10000.0
TILE = 32
NEG = -1.0e9
CHUNK_PAD = 64                      # 34 tokens padded to two tiles


def config_problems(model_config) -> list[str]:
    """Reasons this file cannot run the decoder described by a model's `model_config.json` (a parsed dict); empty when it can. Anything not
    listed as supported is a reason, because an unknown key changes what the library computes in a way this file does not reproduce."""
    try:
        dec = model_config["model"]["pretransform"]["config"]["decoder"]
    except (KeyError, TypeError):
        return ["model_config.json has no model.pretransform.config.decoder"]
    if not isinstance(dec, dict) or not isinstance(dec.get("config"), dict):
        return ["the decoder entry must be an object with a config object"]
    cfg = dec["config"]
    problems = []
    if dec.get("type") != "taae_v2":
        problems.append(f"decoder type is {dec.get('type')!r}; this path implements 'taae_v2'")
    if dec.get("soft_clip"):
        problems.append("the decoder uses soft_clip, which this path does not implement")
    want = {"differential": True, "dyt": True, "dim_heads": HEAD_DIM, "variable_stride": True, "chunk_midpoint_shift": True, "conv_mapping": True,
            "mapping_style": "none"}
    for key, value in want.items():
        if cfg.get(key) != value:
            problems.append(f"decoder.config.{key} is {cfg.get(key)!r}; this path implements {value!r}")
    for key in ("c_mults", "strides", "transformer_depths", "sinusoidal_blocks"):
        if not (isinstance(cfg.get(key), list) and len(cfg[key]) == 1):
            problems.append(f"decoder.config.{key} must be a one-entry list (one resampling block); got {cfg.get(key)!r}")
    if not problems:
        if cfg["sinusoidal_blocks"] != [0]:
            problems.append("sinusoidal feed-forward blocks are not implemented")
        if cfg["strides"][0] * 2 != cfg.get("chunk_size"):
            problems.append("a chunk is two latent frames in this path (chunk_size == 2 * stride)")
        if cfg["transformer_depths"][0] % 2:
            problems.append("an even number of layers is needed for the shifted half")
    known = {"out_channels", "channels", "c_mults", "strides", "latent_dim", "transformer_depths", "sinusoidal_blocks", "checkpointing", "differential",
             "mapping_style", "dim_heads", "variable_stride", "use_flash", "dyt", "chunk_size", "conv_mapping", "chunk_midpoint_shift", "freeze_backbone",
             "mask_noise"}
    for key in cfg:
        if key not in known:
            problems.append(f"decoder.config.{key} is not a setting this path implements")
    return problems


def rotary_tables(torch, n_pad: int = CHUNK_PAD):
    """cos and sin tables [n_pad, HEAD_DIM]; positions 0..n_pad-1; the first ROT_DIM columns rotate, the rest pass through."""
    inv = 1.0 / (ROPE_BASE ** (torch.arange(0, ROT_DIM, 2).float() / ROT_DIM))
    freqs = torch.einsum("i,j->ij", torch.arange(n_pad).float(), inv)
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos, sin = torch.ones(n_pad, HEAD_DIM), torch.zeros(n_pad, HEAD_DIM)
    cos[:, :ROT_DIM], sin[:, :ROT_DIM] = freqs.cos(), freqs.sin()
    return cos, sin


def rotate_matrix(torch):
    """R with `t @ R` equal to the library's `rotate_half` on the first ROT_DIM columns and zero elsewhere."""
    half = ROT_DIM // 2
    r = torch.zeros(HEAD_DIM, HEAD_DIM)
    for j in range(half):
        r[j + half, j] = -1.0
        r[j, j + half] = 1.0
    return r


def key_mask(torch, n_real: int, n_pad: int = CHUNK_PAD):
    m = torch.zeros(1, 1, n_pad, n_pad)
    m[..., n_real:] = NEG
    return m


def dyt(torch, x, alpha, gamma, beta):
    """The library's DynamicTanh: gamma * tanh(alpha * x) + beta."""
    return gamma * torch.tanh(alpha * x) + beta


def host_layer(torch, x, p, n_real: int, cos, sin, rot, trace=None):
    """One layer in the chip's decomposition. x is [B, 64, D] (chunks padded to 64 rows); p holds the layer's tensors by library name.
    Returns [B, 64, D]; rows from n_real on are padding and carry no meaning. When `trace` is a dict, the intermediates are stored in it."""
    f = torch.nn.functional
    b, n, d = x.shape
    h = d // HEAD_DIM
    mask = key_mask(torch, n_real, n)

    def heads(t):                                    # [B, N, D] -> [B, H, N, 64]
        return t.view(b, n, h, HEAD_DIM).transpose(1, 2)

    def attend(q, k, v):
        s = (q @ k.transpose(-1, -2)) / math.sqrt(HEAD_DIM) + mask
        return s.softmax(-1) @ v

    def qk_norm(t, which):
        return dyt(torch, t, p[f"self_attn.{which}_norm.alpha"], p[f"self_attn.{which}_norm.gamma"], p[f"self_attn.{which}_norm.beta"])

    def rope(t):
        return t * cos + (t @ rot) * sin

    def note(name, t):
        if trace is not None:
            trace[name] = t

    y = dyt(torch, x, p["pre_norm.alpha"], p["pre_norm.gamma"], p["pre_norm.beta"])
    note("dyt_pre", y)
    q, k, v, qd, kd = (y @ p["self_attn.to_qkv.weight"].t()).chunk(5, dim=-1)
    note("qkv_q", q)
    q, k, v, qd, kd = heads(q), heads(k), heads(v), heads(qd), heads(kd)
    q, k, qd, kd = qk_norm(q, "q"), qk_norm(k, "k"), qk_norm(qd, "q"), qk_norm(kd, "k")
    note("q_dyt", q)
    q, k, qd, kd = rope(q), rope(k), rope(qd), rope(kd)
    note("q_rope", q)
    a1, a2_ = attend(q, k, v), attend(qd, kd, v)
    note("sdpa_a", a1)
    note("sdpa_d", a2_)
    note("attn", a1 - a2_)
    a = (a1 - a2_).transpose(1, 2).reshape(b, n, d)
    x = x + a @ p["self_attn.to_out.weight"].t()
    note("x_attn", x)
    y = dyt(torch, x, p["ff_norm.alpha"], p["ff_norm.gamma"], p["ff_norm.beta"])
    note("dyt_ff", y)
    z = y @ p["ff.ff.0.proj.weight"].t() + p["ff.ff.0.proj.bias"]
    note("ff1", z)
    a2, gate = z.chunk(2, dim=-1)
    out = x + (a2 * f.silu(gate)) @ p["ff.ff.2.weight"].t() + p["ff.ff.2.bias"]
    note("x_out", out)
    return out


# ----------------------------------------------------------------------------------------------------------------------------------------
# The chip side. Everything below needs ttnn and a leased chip.
# ----------------------------------------------------------------------------------------------------------------------------------------

class ChipDecoder:
    """Drop-in for the library's `SAMEDecoder` in `AudioAutoencoder.decoder`. Built from the original module; install with `install`."""

    def __init__(self, ttnn, torch, device, original, *, dtype=None, act_dtype=None, record=None, precise_attention=True):
        self.ttnn, self.torch, self.device = ttnn, torch, device
        # Differential attention subtracts two attention outputs that are nearly equal (each has norm near 290 where the difference has norm near 80), so
        # a bfloat16 relative error of 2 to 4 percent in each output becomes an error of tens of percent in the difference (measured, see the run log).
        # `precise_attention` keeps the query, key and value path in float32 and computes the attention by hand (the fused kernel takes bfloat16 only).
        self.precise = precise_attention
        self.dtype = dtype if dtype is not None else ttnn.bfloat16
        self.act = act_dtype if act_dtype is not None else ttnn.float32
        self.record = record                    # callable(name, tensor [B, T, D]) for layer outputs of every call, or None
        self.timings = {"upload_weights_s": 0.0, "forward_s": 0.0, "layers_s": 0.0, "calls": 0}
        self.kernel = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        t0 = time.time()
        self._load(original)
        self.timings["upload_weights_s"] = round(time.time() - t0, 3)
        self.trace = None                       # a dict: when set, the sections of every layer overwrite its entries (a diagnostic)
        self.probe = None                       # a dict: when set, the stream after each layer is stored in it (one read back each)

    def _w(self, w):
        return self.ttnn.from_torch(w.detach().float().t().contiguous(), dtype=self.dtype, layout=self.ttnn.TILE_LAYOUT, device=self.device)

    def _w32(self, w):
        return self.ttnn.from_torch(w.detach().float().t().contiguous(), dtype=self.ttnn.float32, layout=self.ttnn.TILE_LAYOUT, device=self.device)

    def _row(self, v, dtype=None):
        """A vector as a [1, 1, 1, N] tile tensor (broadcasts over rows)."""
        return self.ttnn.from_torch(v.detach().float().reshape(1, 1, 1, -1), dtype=dtype or self.dtype, layout=self.ttnn.TILE_LAYOUT, device=self.device)

    def _dyt_params(self, m):
        return {"alpha": float(m.alpha.detach().float().reshape(-1)[0]), "gamma": self._row(m.gamma), "beta": self._row(m.beta),
                "gamma32": self._row(m.gamma, self.ttnn.float32), "beta32": self._row(m.beta, self.ttnn.float32)}

    def _load(self, decoder):
        torch, ttnn = self.torch, self.ttnn
        self.lin_in = decoder.layers[1]                                         # Linear latent -> width, host, float32
        blocks = [l for l in decoder.layers if hasattr(l, "transformers")]
        assert len(blocks) == 1, "one resampling block is implemented"
        self.block = blocks[0]
        self.depth = len(self.block.transformers)
        self.layers = []
        for l in self.block.transformers:
            d = {"pre": self._dyt_params(l.pre_norm), "ff_n": self._dyt_params(l.ff_norm),
                 "q_n": self._dyt_params(l.self_attn.q_norm), "k_n": self._dyt_params(l.self_attn.k_norm),
                 "qkv": self._w(l.self_attn.to_qkv.weight), "qkv32": self._w32(l.self_attn.to_qkv.weight), "out": self._w(l.self_attn.to_out.weight),
                 "ff1": self._w(l.ff.ff[0].proj.weight), "ff1_b": self._row(l.ff.ff[0].proj.bias),
                 "ff2": self._w(l.ff.ff[2].weight), "ff2_b": self._row(l.ff.ff[2].bias)}
            self.layers.append(d)
        self.dim = self.block.in_channels
        self.stride = self.block.stride
        self.sub = self.stride + 1                                              # tokens per latent frame
        self.chunk_tokens = self.block.chunk_size + self.block.chunk_size // self.stride      # 34
        assert self.chunk_tokens == 2 * self.sub, "a chunk is two latent frames"
        cos, sin = rotary_tables(torch)
        self.consts = {
            "cos": self._up(cos.view(1, 1, CHUNK_PAD, HEAD_DIM)), "sin": self._up(sin.view(1, 1, CHUNK_PAD, HEAD_DIM)),
            "rot": self._up(rotate_matrix(torch).view(1, 1, HEAD_DIM, HEAD_DIM)),
            "mask": self._up(key_mask(torch, self.chunk_tokens), dtype=ttnn.bfloat16),
            "cos32": self._up(cos.view(1, 1, CHUNK_PAD, HEAD_DIM), dtype=ttnn.float32), "sin32": self._up(sin.view(1, 1, CHUNK_PAD, HEAD_DIM), dtype=ttnn.float32),
            "rot32": self._up(rotate_matrix(torch).view(1, 1, HEAD_DIM, HEAD_DIM), dtype=ttnn.float32),
            "mask32": self._up(key_mask(torch, self.chunk_tokens), dtype=ttnn.float32),
        }

    def _up(self, t, *, act=False, dtype=None):
        ttnn = self.ttnn
        return ttnn.from_torch(t.float(), dtype=dtype or (self.act if act else self.dtype), layout=ttnn.TILE_LAYOUT, device=self.device)

    # -- ops -------------------------------------------------------------------------------------------------------------------------
    def _lin(self, x, w, b=None, *, out=None):
        ttnn = self.ttnn
        if x.dtype != self.dtype:
            x = ttnn.typecast(x, self.dtype)
        return ttnn.linear(x, w, bias=b, dtype=out or self.dtype, compute_kernel_config=self.kernel)

    def _dyt(self, x, p, wide=False):
        """gamma * tanh(alpha * x) + beta, result in the weight dtype (it feeds a matrix multiply or a per-head product), or float32 when `wide`."""
        ttnn = self.ttnn
        if wide and x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        y = ttnn.tanh(ttnn.multiply(x, p["alpha"]))
        y = ttnn.add(ttnn.multiply(y, p["gamma32" if wide else "gamma"]), p["beta32" if wide else "beta"])
        want = ttnn.float32 if wide else self.dtype
        return ttnn.typecast(y, want) if y.dtype != want else y

    def _heads(self, x, b):
        """[B, 1, 64, 768] -> [B, 12, 64, 64]."""
        ttnn = self.ttnn
        x = ttnn.reshape(x, (b, CHUNK_PAD, self.dim // HEAD_DIM, HEAD_DIM))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _rope(self, t, wide=False):
        ttnn = self.ttnn
        sfx = "32" if wide else ""
        rot = ttnn.matmul(t, self.consts["rot" + sfx], compute_kernel_config=self.kernel)
        return ttnn.add(ttnn.multiply(t, self.consts["cos" + sfx]), ttnn.multiply(rot, self.consts["sin" + sfx]))

    def _probs(self, q, k):
        """softmax(q k^T / 8 + mask) in float32, [B, H, 64, 64]."""
        ttnn = self.ttnn
        s = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.kernel)
        s = ttnn.add(ttnn.multiply(s, 1.0 / math.sqrt(HEAD_DIM)), self.consts["mask32"])
        return ttnn.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=self.kernel)

    def _sdpa(self, q, k, v):
        ttnn = self.ttnn
        out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=self.consts["mask"], is_causal=False, scale=1.0 / math.sqrt(HEAD_DIM), compute_kernel_config=self.kernel)
        return out

    def _layer(self, i, x, b):
        ttnn = self.ttnn
        d, dim = self.layers[i], self.dim
        wide = self.precise
        y = self._dyt(x, d["pre"], wide=wide)
        self._note("dyt_pre", y, b)
        qkv = ttnn.linear(y, d["qkv32"], dtype=ttnn.float32, compute_kernel_config=self.kernel) if wide else self._lin(y, d["qkv"])   # [B, 1, 64, 5 * 768]
        parts = [ttnn.slice(qkv, (0, 0, 0, j * dim), (b, 1, CHUNK_PAD, (j + 1) * dim)) for j in range(5)]
        self._note("qkv_q", parts[0], b)
        q, k, v, qd, kd = (self._heads(t, b) for t in parts)
        q, qd = self._dyt(q, d["q_n"], wide=wide), self._dyt(qd, d["q_n"], wide=wide)
        k, kd = self._dyt(k, d["k_n"], wide=wide), self._dyt(kd, d["k_n"], wide=wide)
        self._note("q_dyt", q, b, heads=True)
        q, qd = self._rope(q, wide), self._rope(qd, wide)
        k, kd = self._rope(k, wide), self._rope(kd, wide)
        self._note("q_rope", q, b, heads=True)
        if wide:
            # (softmax(s1) - softmax(s2)) v: the difference of the probabilities is formed in float32 before the product with v
            a = ttnn.matmul(ttnn.subtract(self._probs(q, k), self._probs(qd, kd)), v, compute_kernel_config=self.kernel)
        else:
            a1, a2_ = self._sdpa(q, k, v), self._sdpa(qd, kd, v)
            self._note("sdpa_a", a1, b, heads=True)
            self._note("sdpa_d", a2_, b, heads=True)
            a = ttnn.subtract(a1, a2_)
        self._note("attn", a, b, heads=True)
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), (b, 1, CHUNK_PAD, dim))
        x = ttnn.add(x, self._lin(a, d["out"], out=self.act))
        self._note("x_attn", x, b)
        y = self._dyt(x, d["ff_n"])
        self._note("dyt_ff", y, b)
        z = self._lin(y, d["ff1"], d["ff1_b"])
        self._note("ff1", z, b)
        half = z.shape[-1] // 2
        a2 = ttnn.slice(z, (0, 0, 0, 0), (b, 1, CHUNK_PAD, half))
        g = ttnn.slice(z, (0, 0, 0, half), (b, 1, CHUNK_PAD, 2 * half))
        z = self._lin(ttnn.multiply(a2, ttnn.silu(g)), d["ff2"], d["ff2_b"], out=self.act)
        x = ttnn.add(x, z)
        if self.probe is not None:
            self.probe[i] = self._down(x, b)
        self._note("x_out", x, b)
        return x

    def _note(self, name, t, b, heads=False):
        """Store an intermediate of the layer in `self.trace` (a dict; None disables). Costs a read back; a diagnostic."""
        if self.trace is None:
            return
        v = self.ttnn.to_torch(t).float()
        if heads:                                                   # [B, H, 64, 64]
            v = v[:, :, :self.chunk_tokens]
        else:                                                       # [B, 1, 64, W]
            v = v.reshape(b, CHUNK_PAD, -1)[:, :self.chunk_tokens]
        self.trace[name] = v

    def _down(self, x, b):
        """Read a [B, 1, 64, D] stream back as float32 [B, 34, D] (the padding rows dropped)."""
        return self.ttnn.to_torch(x).float().reshape(b, CHUNK_PAD, self.dim)[:, :self.chunk_tokens]

    def _upload_chunks(self, chunks):
        """[B, 34, D] float32 host -> [B, 1, 64, D] on the chip in the stream dtype."""
        torch = self.torch
        b = chunks.shape[0]
        pad = torch.zeros(b, CHUNK_PAD, self.dim)
        pad[:, :self.chunk_tokens] = chunks
        return self._up(pad.view(b, 1, CHUNK_PAD, self.dim), act=True)

    # -- the module interface --------------------------------------------------------------------------------------------------------
    def __call__(self, x, override_stride=None, **kwargs):
        """x [B, latent, T] -> [B, 512, 16 * T'] as `SAMEDecoder.forward` does. Same host code for the token layout, the noise and the final convolution."""
        torch, ttnn = self.torch, self.ttnn
        if override_stride is not None:
            raise NotImplementedError("the stride is fixed in this path")
        t0 = time.time()
        blk = self.block
        with torch.no_grad():
            # host: Transpose, Linear, Transpose, then the block's own preamble (library lines, same torch generator)
            h = self.lin_in(x.transpose(-1, -2)).transpose(-1, -2)
            batch = h.shape[0]
            h = h.transpose(-1, -2)                                             # [B, T, D]
            pad = (-h.shape[1]) % (blk.chunk_size // self.stride)
            if pad:
                h = torch.cat([h, torch.zeros(batch, pad, h.shape[2], dtype=h.dtype)], dim=1)
            n = h.shape[1]
            h = h.reshape(batch * n, 1, self.dim)
            new_tokens = blk.new_tokens.expand([h.shape[0], self.stride, -1])
            if blk.mask_noise > 0:
                new_tokens = new_tokens + torch.randn_like(new_tokens) * blk.mask_noise
            h = torch.cat([h, new_tokens], dim=-2).reshape(batch, n * self.sub, self.dim)
            nc = h.shape[1] // self.chunk_tokens
            chunks = h.reshape(batch * nc, self.chunk_tokens, self.dim)
            t1 = time.time()
            s = self._upload_chunks(chunks)
            split, shift = self.depth // 2, self.chunk_tokens // 2
            for i in range(split):
                s = self._layer(i, s, batch * nc)
            first = self._down(s, batch * nc).reshape(batch, nc * self.chunk_tokens, self.dim)
            if self.record is not None:
                self.record("half1", first)
            # host: the shifted chunks (first and last half chunk repeated at the ends)
            shifted = torch.cat([first[:, :shift], first, first[:, -shift:]], dim=1)
            nc2 = shifted.shape[1] // self.chunk_tokens
            s = self._upload_chunks(shifted.reshape(batch * nc2, self.chunk_tokens, self.dim))
            for i in range(split, self.depth):
                s = self._layer(i, s, batch * nc2)
            out = self._down(s, batch * nc2).reshape(batch, nc2 * self.chunk_tokens, self.dim)[:, shift:-shift]
            if self.record is not None:
                self.record("final", out)
            t2 = time.time()
            out = out.reshape(batch * n, self.sub, self.dim)[:, -self.stride:, :]
            out = out.reshape(batch, n * self.stride, self.dim).transpose(1, 2)  # [B, D, n * stride]
            out = blk.mapping(out)
        self.timings["forward_s"] += time.time() - t0
        self.timings["layers_s"] += t2 - t1
        self.timings["calls"] += 1
        return out


def install(autoencoder, chip):
    """Replace `AudioAutoencoder.decoder` with `chip`. Returns the original module."""
    original = autoencoder.decoder
    del autoencoder._modules["decoder"]
    object.__setattr__(autoencoder, "decoder", chip)
    return original
