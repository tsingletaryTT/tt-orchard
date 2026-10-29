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
SLIDE_ROWS = 32                     # 17 tokens (one latent frame) padded to one tile pair


def config_problems(model_config) -> list[str]:
    """Reasons this file cannot run the decoder described by a model's `model_config.json` (a parsed dict); empty when it can. Anything not
    listed as supported is a reason, because an unknown key changes what the library computes in a way this file does not reproduce.

    Two block layouts are implemented. Chunked (the small models): `chunk_midpoint_shift` true, `chunk_size` twice the stride, no sliding window,
    an even number of layers. Sliding window (the medium model): `sliding_window` [1, 1] (one latent frame to each side), no chunking, any number of
    layers. In both, `sinusoidal_blocks` is a one-entry list: the last that many layers use the sine gate."""
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
    sliding = cfg.get("sliding_window") is not None
    want = {"differential": True, "dim_heads": HEAD_DIM, "variable_stride": True, "mapping_style": "none"}
    for key, value in want.items():
        if cfg.get(key) != value:
            problems.append(f"decoder.config.{key} is {cfg.get(key)!r}; this path implements {value!r}")
    if cfg.get("dyt", True) is not True:                       # the block's own default is dynamic-tanh normalisation
        problems.append(f"decoder.config.dyt is {cfg.get('dyt')!r}; this path implements true")
    for key in ("conformer", "layer_scale"):
        if cfg.get(key, False) is not False:
            problems.append(f"decoder.config.{key} is {cfg.get(key)!r}; this path implements false")
    if sliding:
        if cfg["sliding_window"] != [1, 1]:
            problems.append(f"decoder.config.sliding_window is {cfg['sliding_window']!r}; this path implements [1, 1]")
        if cfg.get("chunk_midpoint_shift", False):
            problems.append("a sliding window with chunk_midpoint_shift is not implemented")
    else:
        for key, value in {"chunk_midpoint_shift": True, "conv_mapping": True}.items():
            if cfg.get(key) != value:
                problems.append(f"decoder.config.{key} is {cfg.get(key)!r}; this path implements {value!r}")
    for key in ("c_mults", "strides", "transformer_depths", "sinusoidal_blocks"):
        if not (isinstance(cfg.get(key), list) and len(cfg[key]) == 1):
            problems.append(f"decoder.config.{key} must be a one-entry list (one resampling block); got {cfg.get(key)!r}")
    if not problems:
        if cfg["strides"][0] != 16:
            problems.append("a stride other than 16 is not implemented (the sliding blocks and the chunk of two frames assume 17 tokens a frame)")
        if not sliding and cfg["strides"][0] * 2 != cfg.get("chunk_size"):
            problems.append("a chunk is two latent frames in this path (chunk_size == 2 * stride)")
        if not sliding and cfg["transformer_depths"][0] % 2:
            problems.append("an even number of layers is needed for the shifted half")
        if not isinstance(cfg["sinusoidal_blocks"][0], int) or cfg["sinusoidal_blocks"][0] < 0:
            problems.append("sinusoidal_blocks must hold a whole number, 0 or more")
    known = {"out_channels", "channels", "c_mults", "strides", "latent_dim", "transformer_depths", "sinusoidal_blocks", "checkpointing", "differential",
             "mapping_style", "dim_heads", "variable_stride", "use_flash", "dyt", "chunk_size", "conv_mapping", "chunk_midpoint_shift", "freeze_backbone",
             "mask_noise", "sliding_window", "conformer", "layer_scale", "use_snake", "use_dilated_conv", "conv_bias", "enable_inner_layer_dropout"}
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


SINE = 3.14159265359                 # the library's `Sin` activation: sin(SINE * x)


def sliding_masks(torch, frames: int, rows: int = 32, real: int = 17):
    """Additive attention mask [frames, 1, rows, 3 * rows] for the sliding window of the medium decoder, in the block layout: a block is one latent frame
    (17 tokens padded to `rows`), and the keys of block f are the blocks f - 1, f and f + 1 (each padded to `rows`). Query o of block f may see the key
    at block-relative position r of block f + j (j in -1, 0, 1) when its token distance, 17 * j + r - o, is within 17 either way, the key row is real (r < 17)
    and the block exists. A padded query row sees its own block's first key, so no row of the softmax is empty."""
    q = torch.arange(rows).view(rows, 1)
    kk = torch.arange(3 * rows).view(1, 3 * rows)
    blk, r = kk // rows - 1, kk % rows
    dist = real * blk + r - q
    ok = (r < real) & (dist >= -real) & (dist <= real) & (q < real)
    ok = ok | ((q >= real) & (kk == rows))
    m = torch.where(ok, torch.zeros(rows, 3 * rows), torch.full((rows, 3 * rows), NEG)).view(1, 1, rows, 3 * rows).repeat(frames, 1, 1, 1)
    if frames > 0:
        left = (blk < 0).view(1, 1, 1, 3 * rows).expand(1, 1, rows, 3 * rows)
        right = (blk > 0).view(1, 1, 1, 3 * rows).expand(1, 1, rows, 3 * rows)
        m[0] = torch.where(left[0] & (q < real), torch.full_like(m[0], NEG), m[0])
        m[-1] = torch.where(right[0] & (q < real), torch.full_like(m[-1], NEG), m[-1])
    return m


def sliding_rotary(torch, frames: int, rows: int = 32, real: int = 17):
    """cos and sin [frames, 1, rows, HEAD_DIM]: the rotary position of row o of block f is 17 f + o (the whole sequence is one run of 17 * frames tokens)."""
    pos = (torch.arange(frames).view(frames, 1) * real + torch.arange(rows).view(1, rows)).float()                # [frames, rows]
    inv = 1.0 / (ROPE_BASE ** (torch.arange(0, ROT_DIM, 2).float() / ROT_DIM))
    freqs = pos.unsqueeze(-1) * inv                                                                               # [frames, rows, 16]
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos, sin = torch.ones(frames, rows, HEAD_DIM), torch.zeros(frames, rows, HEAD_DIM)
    cos[..., :ROT_DIM], sin[..., :ROT_DIM] = freqs.cos(), freqs.sin()
    return cos.unsqueeze(1), sin.unsqueeze(1)


def host_sliding_layer(torch, x, p, cos, sin, rot, mask, sine=False):
    """One layer of the sliding-window decoder in the chip's block decomposition. x is [T, 32, D] (T blocks of 17 real rows); returns [T, 32, D]."""
    f = torch.nn.functional
    t, rows, d = x.shape
    h = d // HEAD_DIM

    def heads(z):
        return z.view(t, rows, h, HEAD_DIM).transpose(1, 2)                       # [T, H, 32, 64]

    def extend(z):                                                               # keys and values of the neighbours: [T, H, 96, 64]
        zero = torch.zeros_like(z[:1])
        return torch.cat([torch.cat([zero, z[:-1]], dim=0), z, torch.cat([z[1:], zero], dim=0)], dim=2)

    def attend(q, k, v):
        return ((q @ extend(k).transpose(-1, -2)) / math.sqrt(HEAD_DIM) + mask).softmax(-1)

    def qk(z, which):
        return dyt(torch, z, p[f"self_attn.{which}_norm.alpha"], p[f"self_attn.{which}_norm.gamma"], p[f"self_attn.{which}_norm.beta"])

    def rope(z):
        return z * cos + (z @ rot) * sin

    y = dyt(torch, x, p["pre_norm.alpha"], p["pre_norm.gamma"], p["pre_norm.beta"])
    q, k, v, qd, kd = (heads(z) for z in (y @ p["self_attn.to_qkv.weight"].t()).chunk(5, dim=-1))
    q, qd, k, kd = rope(qk(q, "q")), rope(qk(qd, "q")), rope(qk(k, "k")), rope(qk(kd, "k"))
    a = (attend(q, k, v) - attend(qd, kd, v)) @ extend(v)
    x = x + a.transpose(1, 2).reshape(t, rows, d) @ p["self_attn.to_out.weight"].t()
    y = dyt(torch, x, p["ff_norm.alpha"], p["ff_norm.gamma"], p["ff_norm.beta"])
    z = y @ p["ff.ff.0.proj.weight"].t() + p["ff.ff.0.proj.bias"]
    a2, gate = z.chunk(2, dim=-1)
    act = torch.sin(SINE * gate) if sine else f.silu(gate)
    return x + (a2 * act) @ p["ff.ff.2.weight"].t() + p["ff.ff.2.bias"]


def host_layer(torch, x, p, n_real: int, cos, sin, rot, trace=None, sine=False):
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
    """Drop-in for the library's `SAMEDecoder` in `AudioAutoencoder.decoder`. Built from the original module; install with `install`.

    Two layouts, chosen from the block's own settings. Chunked (small models): the stream is [chunks, 1, 64, D], two latent frames (34 tokens) to a chunk,
    with a shifted second half. Sliding window (medium): the stream is [frames, 1, 32, D], one latent frame (17 tokens) to a block, and each block
    attends to itself and its two neighbours with a band mask."""

    def __init__(self, ttnn, torch, device, original, *, dtype=None, act_dtype=None, record=None):
        self.ttnn, self.torch, self.device = ttnn, torch, device
        # Differential attention subtracts two attention outputs that are nearly equal (each has norm near 290 where the difference has norm near 80), so
        # a bfloat16 relative error of 2 to 4 percent in each output becomes an error of tens of percent in the difference (measured, see the run log).
        # The query, key and value path stays in float32 and the attention is computed with matrix multiplies and a float32 softmax (the fused
        # attention kernel takes bfloat16 only).
        self.dtype = dtype if dtype is not None else ttnn.bfloat16
        self.act = act_dtype if act_dtype is not None else ttnn.float32
        self.record = record                    # callable(name, tensor [B, T, D]) for layer outputs of every call, or None
        self.timings = {"upload_weights_s": 0.0, "forward_s": 0.0, "layers_s": 0.0, "calls": 0}
        self.kernel = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        self.trace = None                       # a dict: when set, the sections of every layer overwrite its entries (a diagnostic)
        self.probe = None                       # a dict: when set, the stream after each layer is stored in it (one read back each)
        self._sliding_cache = {}
        t0 = time.time()
        self._load(original)
        self.timings["upload_weights_s"] = round(time.time() - t0, 3)

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
        self.sliding = self.block.sliding_window_seq is not None
        self.layers = []
        for l in self.block.transformers:
            d = {"pre": self._dyt_params(l.pre_norm), "ff_n": self._dyt_params(l.ff_norm),
                 "q_n": self._dyt_params(l.self_attn.q_norm), "k_n": self._dyt_params(l.self_attn.k_norm),
                 "qkv32": self._w32(l.self_attn.to_qkv.weight), "out": self._w(l.self_attn.to_out.weight),
                 "ff1": self._w(l.ff.ff[0].proj.weight), "ff1_b": self._row(l.ff.ff[0].proj.bias),
                 "ff2": self._w(l.ff.ff[2].weight), "ff2_b": self._row(l.ff.ff[2].bias),
                 "sine": type(l.ff.ff[0].act).__name__ == "Sin"}
            self.layers.append(d)
        self.dim = self.block.in_channels
        self.heads = self.dim // HEAD_DIM
        self.stride = self.block.stride
        self.sub = self.stride + 1                                              # tokens per latent frame (17)
        if self.sliding:
            self.rows, self.real = SLIDE_ROWS, self.sub
        else:
            self.chunk_tokens = self.block.chunk_size + self.block.chunk_size // self.stride      # 34
            assert self.chunk_tokens == 2 * self.sub, "a chunk is two latent frames"
            self.rows, self.real = CHUNK_PAD, self.chunk_tokens
            cos, sin = rotary_tables(torch)
            self.consts = {"cos32": self._up(cos.view(1, 1, CHUNK_PAD, HEAD_DIM), dtype=ttnn.float32),
                           "sin32": self._up(sin.view(1, 1, CHUNK_PAD, HEAD_DIM), dtype=ttnn.float32),
                           "rot32": self._up(rotate_matrix(torch).view(1, 1, HEAD_DIM, HEAD_DIM), dtype=ttnn.float32),
                           "mask32": self._up(key_mask(torch, self.chunk_tokens), dtype=ttnn.float32)}
        self.rot32 = self._up(rotate_matrix(torch).view(1, 1, HEAD_DIM, HEAD_DIM), dtype=ttnn.float32)

    def _up(self, t, *, act=False, dtype=None):
        ttnn = self.ttnn
        return ttnn.from_torch(t.float(), dtype=dtype or (self.act if act else self.dtype), layout=ttnn.TILE_LAYOUT, device=self.device)

    def _sliding_consts(self, frames):
        """Rotary tables and the band mask for a run of `frames` latent frames, uploaded once per length."""
        if frames not in self._sliding_cache:
            cos, sin = sliding_rotary(self.torch, frames)
            f32 = self.ttnn.float32
            self._sliding_cache[frames] = {"cos32": self._up(cos, dtype=f32), "sin32": self._up(sin, dtype=f32), "rot32": self.rot32,
                                           "mask32": self._up(sliding_masks(self.torch, frames), dtype=f32)}
        return self._sliding_cache[frames]

    # -- ops -------------------------------------------------------------------------------------------------------------------------
    def _lin(self, x, w, b=None, *, out=None):
        ttnn = self.ttnn
        if x.dtype != self.dtype:
            x = ttnn.typecast(x, self.dtype)
        return ttnn.linear(x, w, bias=b, dtype=out or self.dtype, compute_kernel_config=self.kernel)

    def _dyt(self, x, p, wide=True):
        """gamma * tanh(alpha * x) + beta, in float32 when `wide`, otherwise in the weight dtype (it feeds a matrix multiply)."""
        ttnn = self.ttnn
        if wide and x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        y = ttnn.tanh(ttnn.multiply(x, p["alpha"]))
        y = ttnn.add(ttnn.multiply(y, p["gamma32" if wide else "gamma"]), p["beta32" if wide else "beta"])
        want = ttnn.float32 if wide else self.dtype
        return ttnn.typecast(y, want) if y.dtype != want else y

    def _heads(self, x, b):
        """[B, 1, rows, D] -> [B, H, rows, 64]."""
        ttnn = self.ttnn
        x = ttnn.reshape(x, (b, self.rows, self.heads, HEAD_DIM))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _rope(self, t, c):
        ttnn = self.ttnn
        rot = ttnn.matmul(t, c["rot32"], compute_kernel_config=self.kernel)
        return ttnn.add(ttnn.multiply(t, c["cos32"]), ttnn.multiply(rot, c["sin32"]))

    def _extend(self, z, b):
        """Keys or values of the neighbouring blocks: [T, H, rows, 64] -> [T, H, 3 * rows, 64] (previous block, this block, next block; zeros beyond the ends)."""
        ttnn = self.ttnn
        zero = ttnn.zeros((1, self.heads, self.rows, HEAD_DIM), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device)
        prev = ttnn.concat([zero, ttnn.slice(z, (0, 0, 0, 0), (b - 1, self.heads, self.rows, HEAD_DIM))], dim=0) if b > 1 else zero
        nxt = ttnn.concat([ttnn.slice(z, (1, 0, 0, 0), (b, self.heads, self.rows, HEAD_DIM)), zero], dim=0) if b > 1 else zero
        return ttnn.concat([prev, z, nxt], dim=2)

    def _probs(self, q, k, mask):
        """softmax(q k^T / 8 + mask) in float32."""
        ttnn = self.ttnn
        s = ttnn.matmul(q, ttnn.transpose(k, -2, -1), compute_kernel_config=self.kernel)
        s = ttnn.add(ttnn.multiply(s, 1.0 / math.sqrt(HEAD_DIM)), mask)
        return ttnn.softmax(s, dim=-1, numeric_stable=True, compute_kernel_config=self.kernel)

    def _layer(self, i, x, b, c):
        ttnn = self.ttnn
        d, dim, rows = self.layers[i], self.dim, self.rows
        y = self._dyt(x, d["pre"])
        self._note("dyt_pre", y, b)
        qkv = ttnn.linear(y, d["qkv32"], dtype=ttnn.float32, compute_kernel_config=self.kernel)       # [B, 1, rows, 5 * dim]
        parts = [ttnn.slice(qkv, (0, 0, 0, j * dim), (b, 1, rows, (j + 1) * dim)) for j in range(5)]
        self._note("qkv_q", parts[0], b)
        q, k, v, qd, kd = (self._heads(t, b) for t in parts)
        q, qd = self._dyt(q, d["q_n"]), self._dyt(qd, d["q_n"])
        k, kd = self._dyt(k, d["k_n"]), self._dyt(kd, d["k_n"])
        self._note("q_dyt", q, b, heads=True)
        q, qd = self._rope(q, c), self._rope(qd, c)
        k, kd = self._rope(k, c), self._rope(kd, c)
        self._note("q_rope", q, b, heads=True)
        if self.sliding:
            k, kd, v = self._extend(k, b), self._extend(kd, b), self._extend(v, b)
        # (softmax(s1) - softmax(s2)) v: the difference of the probabilities is formed in float32 before the product with v
        a = ttnn.matmul(ttnn.subtract(self._probs(q, k, c["mask32"]), self._probs(qd, kd, c["mask32"])), v, compute_kernel_config=self.kernel)
        self._note("attn", a, b, heads=True)
        a = ttnn.reshape(ttnn.permute(a, (0, 2, 1, 3)), (b, 1, rows, dim))
        x = ttnn.add(x, self._lin(a, d["out"], out=self.act))
        self._note("x_attn", x, b)
        y = self._dyt(x, d["ff_n"], wide=d["sine"])
        self._note("dyt_ff", y, b)
        if d["sine"]:
            # the sine gate takes arguments up to tens, so the feed-forward input projection returns float32 and the sine runs in float32
            z = ttnn.linear(ttnn.typecast(y, self.dtype), d["ff1"], bias=d["ff1_b"], dtype=ttnn.float32, compute_kernel_config=self.kernel)
        else:
            z = self._lin(y, d["ff1"], d["ff1_b"])
        self._note("ff1", z, b)
        half = z.shape[-1] // 2
        a2 = ttnn.slice(z, (0, 0, 0, 0), (b, 1, rows, half))
        g = ttnn.slice(z, (0, 0, 0, half), (b, 1, rows, 2 * half))
        gate = ttnn.sin(ttnn.multiply(g, SINE)) if d["sine"] else ttnn.silu(g)
        z = self._lin(ttnn.multiply(a2, gate), d["ff2"], d["ff2_b"], out=self.act)
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
        if heads:                                                   # [B, H, rows, W]
            v = v[:, :, :self.real]
        else:                                                       # [B, 1, rows, W]
            v = v.reshape(b, self.rows, -1)[:, :self.real]
        self.trace[name] = v

    def _down(self, x, b):
        """Read a [B, 1, rows, D] stream back as float32 [B, real, D] (the padding rows dropped)."""
        return self.ttnn.to_torch(x).float().reshape(b, self.rows, self.dim)[:, :self.real]

    def _upload_chunks(self, chunks):
        """[B, real, D] float32 host -> [B, 1, rows, D] on the chip in the stream dtype."""
        torch = self.torch
        b = chunks.shape[0]
        pad = torch.zeros(b, self.rows, self.dim)
        pad[:, :self.real] = chunks
        return self._up(pad.view(b, 1, self.rows, self.dim), act=True)

    # -- the module interface --------------------------------------------------------------------------------------------------------
    def __call__(self, x, override_stride=None, **kwargs):
        """x [B, latent, T] -> [B, out, 16 * T'] as `SAMEDecoder.forward` does. Same host code for the token layout, the noise and the final convolution."""
        torch = self.torch
        if override_stride is not None:
            raise NotImplementedError("the stride is fixed in this path")
        t0 = time.time()
        blk = self.block
        with torch.no_grad():
            # host: Transpose, Linear, Transpose, then the block's own preamble (library lines, same torch generator)
            h = self.lin_in(x.transpose(-1, -2)).transpose(-1, -2)
            batch = h.shape[0]
            h = h.transpose(-1, -2)                                             # [B, T, D]
            if self.sliding:
                if batch != 1:
                    raise NotImplementedError("the sliding-window path decodes one clip at a time")
            else:
                pad = (-h.shape[1]) % (blk.chunk_size // self.stride)
                if pad:
                    h = torch.cat([h, torch.zeros(batch, pad, h.shape[2], dtype=h.dtype)], dim=1)
            n = h.shape[1]
            h = h.reshape(batch * n, 1, self.dim)
            new_tokens = blk.new_tokens.expand([h.shape[0], self.stride, -1])
            if blk.mask_noise > 0:
                new_tokens = new_tokens + torch.randn_like(new_tokens) * blk.mask_noise
            h = torch.cat([h, new_tokens], dim=-2)                              # [B * n, 17, D]
            t1 = time.time()
            if self.sliding:
                c = self._sliding_consts(n)
                s = self._upload_chunks(h)                                      # one block per latent frame
                for i in range(self.depth):
                    s = self._layer(i, s, n, c)
                out = self._down(s, n)                                          # [n, 17, D]
                t2 = time.time()
            else:
                h = h.reshape(batch, n * self.sub, self.dim)
                nc = h.shape[1] // self.chunk_tokens
                s = self._upload_chunks(h.reshape(batch * nc, self.chunk_tokens, self.dim))
                split, shift = self.depth // 2, self.chunk_tokens // 2
                for i in range(split):
                    s = self._layer(i, s, batch * nc, self.consts)
                first = self._down(s, batch * nc).reshape(batch, nc * self.chunk_tokens, self.dim)
                if self.record is not None:
                    self.record("half1", first)
                # host: the shifted chunks (first and last half chunk repeated at the ends)
                shifted = torch.cat([first[:, :shift], first, first[:, -shift:]], dim=1)
                nc2 = shifted.shape[1] // self.chunk_tokens
                s = self._upload_chunks(shifted.reshape(batch * nc2, self.chunk_tokens, self.dim))
                for i in range(split, self.depth):
                    s = self._layer(i, s, batch * nc2, self.consts)
                out = self._down(s, batch * nc2).reshape(batch, nc2 * self.chunk_tokens, self.dim)[:, shift:-shift]
                t2 = time.time()
            if self.record is not None:
                self.record("final", out)
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
