# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""MiniCPM3-4B decode on one Blackhole chip with ttnn (multi-latent attention, dense gated-SiLU MLP).

One token per step. Prompt tokens go through the same step one at a time (prefill by decode: simple, and slow on the first token only).

Scalings are folded into weights so the step has no extra ops: `scale_emb` into the embedding table, `scale_depth / sqrt(layers)` into `o_proj` and
`down_proj`, and `1 / (hidden_size / dim_model_base)` into the final norm weight.

Attention modes (`attn=`):
  host    the multi-latent attention core runs on the host in torch (the stage-b proof that the rest of the port works); q, the compressed
          key-value latent and the rope part are read back from the chip each layer and the attention output is sent back to it.
  mla     the whole attention on the chip in the absorbed form: the per-head up-projection of keys is folded into the query (`q_nope @ W_UK`),
          the cache holds only the 256-wide latent plus the 32-wide rope key (288 values per token, one head), `flash_multi_latent_attention_decode`
          attends over the latent and the per-head up-projection of values (`W_UV`) is applied to its output.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import ttnn
from safetensors import safe_open

TILE = 32


def _rot_matrix():
    """R with x @ R == rotate_half(x) for the 32-wide rope part."""
    R = torch.zeros(32, 32)
    for j in range(16):
        R[j + 16, j] = -1.0
        R[j, j + 16] = 1.0
    return R


class ChipMiniCPM3:
    def __init__(self, dev, model_dir, weights_file, *, attn="mla", max_len=256, wdtype=None, fidelity="default", upd="int", log=print):
        import json
        self.dev = dev
        self.attn = attn
        self.skip = set()        # timing experiments only: parts of the step to leave out ("mlp", "lm")
        self.upd = upd           # cache write: "int" (update_cache at a Python position), "paged" or "onehot" (position from a device tensor)
        self.cfg = c = json.load(open(f"{model_dir}/config.json"))
        self.L, self.H = c["num_hidden_layers"], c["num_attention_heads"]
        self.hid, self.kvr, self.dn, self.dr = c["hidden_size"], c["kv_lora_rank"], c["qk_nope_head_dim"], c["qk_rope_head_dim"]
        self.dv = 64
        self.eps = c["rms_norm_eps"]
        self.max_len = max_len
        self.wdtype = wdtype or ttnn.bfloat16
        self.log = log
        # `fidelity="hifi4"`: highest math fidelity and 32-bit accumulation in every matmul (the default config accumulates in 16 bits)
        self.rms_ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                              fp32_dest_acc_en=True, packer_l1_acc=True)
        self.ckc = None
        if fidelity == "hifi4":
            self.ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                              fp32_dest_acc_en=True, packer_l1_acc=True)
        # One kv head and batch 1 would spread the op over all 110 cores of a Blackhole chip, and the op refuses more than 64 cores per head
        # (TT_FATAL: tree reduction max 6 rounds). An 8 by 8 grid is accepted and matches torch (PCC 0.99995 at the model's shapes).
        self.sdpa_pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=32, k_chunk_size=128,
                                              exp_approx_mode=False)
        self.rs = c["scale_depth"] / math.sqrt(self.L)
        from ref_model import rope_tables
        self.cos, self.sin = rope_tables(c, max_len)
        self.f = safe_open(weights_file, "pt")
        self._load(log)
        self.reset()

    # ---- weights -----------------------------------------------------------------------------------
    def _w(self, name, scale=1.0, transpose=True, dtype=None):
        t = self.f.get_tensor(name).float() * scale
        if transpose:
            t = t.t()
        return ttnn.from_torch(t.contiguous().bfloat16(), device=self.dev, dtype=dtype or self.wdtype, layout=ttnn.TILE_LAYOUT,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _norm_w(self, name, scale=1.0):
        t = self.f.get_tensor(name).float() * scale
        return ttnn.from_torch(t.reshape(1, 1, -1, 32).bfloat16(), device=self.dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _load(self, log):
        c, dev = self.cfg, self.dev
        emb = self.f.get_tensor("model.embed_tokens.weight").float()
        # embedding table (scaled), row major as ttnn.embedding wants; the lm head is the unscaled table transposed
        self.emb = ttnn.from_torch((emb * c["scale_emb"]).bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.lm_head = ttnn.from_torch(emb.t().contiguous().bfloat16(), device=dev, dtype=self.wdtype, layout=ttnn.TILE_LAYOUT)
        self.vocab = emb.shape[0]
        self.final_norm = self._norm_w("model.norm.weight", scale=1.0 / (c["hidden_size"] / c["dim_model_base"]))
        self.R = ttnn.from_torch(_rot_matrix().bfloat16(), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.layers = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            a = p + "self_attn."
            d = {
                "in_norm": self._norm_w(p + "input_layernorm.weight"),
                "post_norm": self._norm_w(p + "post_attention_layernorm.weight"),
                "qa": self._w(a + "q_a_proj.weight"),
                "qa_norm": self._norm_w(a + "q_a_layernorm.weight"),
                "qb": self._w(a + "q_b_proj.weight"),
                "kva": self._w(a + "kv_a_proj_with_mqa.weight"),
                "kv_norm": self._norm_w(a + "kv_a_layernorm.weight"),
                "o": self._w(a + "o_proj.weight", scale=self.rs),
                "gate": self._w(p + "mlp.gate_proj.weight"),
                "up": self._w(p + "mlp.up_proj.weight"),
                "down": self._w(p + "mlp.down_proj.weight", scale=self.rs),
            }
            kvb = self.f.get_tensor(a + "kv_b_proj.weight").float().view(self.H, self.dn + self.dv, self.kvr)
            if self.attn == "mla":
                # W_UK_h [dn, kvr] used as q_nope_h @ W_UK_h; W_UV_h [dv, kvr] used as o_lat @ W_UV_h^T
                d["wuk"] = ttnn.from_torch(kvb[:, : self.dn].contiguous().bfloat16().unsqueeze(0), device=dev, dtype=self.wdtype,
                                           layout=ttnn.TILE_LAYOUT)
                d["wuv"] = ttnn.from_torch(kvb[:, self.dn:].transpose(1, 2).contiguous().bfloat16().unsqueeze(0), device=dev,
                                           dtype=self.wdtype, layout=ttnn.TILE_LAYOUT)
            else:
                d["kvb_host"] = kvb.bfloat16()
            self.layers.append(d)
            if i % 10 == 0:
                log(f"loaded layer {i}")

    # ---- state ---------------------------------------------------------------------------------------
    def reset(self):
        dev = self.dev
        if self.attn == "mla":
            z = torch.zeros(1, 1, self.max_len, self.kvr + self.dr).bfloat16()
            self.kvpe = [ttnn.from_torch(z, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for _ in range(self.L)]
        else:
            self.hk = [None] * self.L

    # ---- helpers -------------------------------------------------------------------------------------
    def _up(self, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, device=self.dev, dtype=dtype, layout=layout)

    def _rms(self, x, w):
        # The default rms_norm config accumulates the sum of squares in 16 bits. On the first token of a prompt (a stream with a few channels near 200)
        # it was 3.7 percent off; with 32-bit accumulation it is 0.08 percent (scripts/minicpm3/exp_mlp14.py).
        return ttnn.rms_norm(x, epsilon=self.eps, weight=w, compute_kernel_config=self.rms_ckc)

    def _lin(self, x, w, **kw):
        return ttnn.linear(x, w, compute_kernel_config=self.ckc, **kw) if self.ckc is not None else ttnn.linear(x, w, **kw)

    def _rope(self, x, cos, sin):
        """x [.., 32]; cos, sin [1,1,1,32] on the chip."""
        return ttnn.add(ttnn.multiply(x, cos), ttnn.multiply(ttnn.matmul(x, self.R), sin))

    def _host(self, t):
        return ttnn.to_torch(t).float()

    # ---- one step ------------------------------------------------------------------------------------
    def _host_inputs(self, tok, pos):
        """Host tensors for the per-step inputs: token id, position as uint32 (rope table row) and int32 (cache and attention position), and the
        one-hot rows for the cache write."""
        t = {
            "tok": ttnn.from_torch(torch.tensor([[tok]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            "pos_u": ttnn.from_torch(torch.tensor([[pos]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            "cur": ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        }
        if self.upd == "onehot":
            oh = torch.zeros(1, 1, self.max_len, 1)
            oh[0, 0, pos, 0] = 1.0
            t["oh"] = ttnn.from_torch(oh.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            t["keep"] = ttnn.from_torch((1.0 - oh).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return t

    def _tables(self):
        """Device rope tables (row per position) so the step needs no host-built cos and sin."""
        if getattr(self, "cos_tab", None) is None:
            mk = lambda t: ttnn.from_torch(t.bfloat16(), device=self.dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)  # noqa: E731
            self.cos_tab, self.sin_tab = mk(self.cos), mk(self.sin)

    def step(self, tok: int, pos: int, tap=None):
        """Run one token at position `pos` without a trace; returns logits [vocab] (float32 torch). `tap(name, torch_tensor)` receives the stream
        after each layer section when given (a read back per call; for the agreement probes only)."""
        self._pos = pos
        self._tables()
        h = self._host_inputs(tok, pos)
        dev_in = {k: ttnn.to_device(v, self.dev) for k, v in h.items()}
        logits = self._forward(dev_in, tap)
        return self._host(logits)[0, 0, 0, : self.vocab]

    def _forward(self, inp, tap=None):
        """The step on device tensors only (so it can be captured as a trace); `inp` holds the device inputs of `_host_inputs`."""
        ids = inp["tok"]
        x = ttnn.embedding(ids, self.emb, layout=ttnn.TILE_LAYOUT)          # [1,1,hid]
        x = ttnn.reshape(x, (1, 1, 1, self.hid))
        cos = ttnn.reshape(ttnn.embedding(inp["pos_u"], self.cos_tab, layout=ttnn.TILE_LAYOUT), (1, 1, 1, 32))
        sin = ttnn.reshape(ttnn.embedding(inp["pos_u"], self.sin_tab, layout=ttnn.TILE_LAYOUT), (1, 1, 1, 32))
        cur = inp["cur"]
        if tap:
            tap("embed", self._host(x)[0, 0, 0])
        for i, d in enumerate(self.layers):
            h = self._rms(x, d["in_norm"])
            qa = self._rms(self._lin(h, d["qa"]), d["qa_norm"])
            qb = self._lin(qa, d["qb"])                               # [1,1,1,H*96]
            kva = self._lin(h, d["kva"])                              # [1,1,1,288]
            ckv = ttnn.slice(kva, [0, 0, 0, 0], [1, 1, 1, self.kvr])
            kpe = ttnn.slice(kva, [0, 0, 0, self.kvr], [1, 1, 1, self.kvr + self.dr])
            ckv = self._rms(ckv, d["kv_norm"])
            if self.attn == "host":
                ao = self._host_core(i, d, qb, ckv, kpe, self._pos)
            else:
                ao = self._mla_core(i, d, qb, ckv, kpe, cos, sin, cur, inp)
            x = ttnn.add(x, self._lin(ao, d["o"]))
            if tap:
                tap(f"layer_{i:02d}.attn", self._host(x)[0, 0, 0])
            if "mlp" in self.skip:
                continue
            h = self._rms(x, d["post_norm"])
            m = ttnn.multiply(self._lin(h, d["gate"], activation="silu"), self._lin(h, d["up"]))
            x = ttnn.add(x, self._lin(m, d["down"]))
            if tap:
                tap(f"layer_{i:02d}", self._host(x)[0, 0, 0])
        x = self._rms(x, self.final_norm)
        if "lm" in self.skip:
            return x
        return self._lin(x, self.lm_head)

    # ---- trace ----------------------------------------------------------------------------------------
    def enable_trace(self):
        """Compile once, capture the step as a trace on persistent input tensors. Later calls to `traced_step` copy the new token and position into
        those tensors and replay. Needs the device opened with a trace region."""
        self._tables()
        h = self._host_inputs(0, 0)
        self.t_in = {k: ttnn.to_device(v, self.dev) for k, v in h.items()}
        self._pos = 0
        self.reset()
        self._forward(self.t_in)                                       # compile run
        ttnn.synchronize_device(self.dev)
        self.reset()
        self.trace_id = ttnn.begin_trace_capture(self.dev, cq_id=0)
        self.t_out = self._forward(self.t_in)
        ttnn.end_trace_capture(self.dev, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.dev)
        # no reset here: a trace is bound to the cache buffers it was captured with. Stale rows beyond the current position are never read.

    def traced_step(self, tok, pos):
        h = self._host_inputs(tok, pos)
        for k, v in h.items():
            ttnn.copy_host_to_device_tensor(v, self.t_in[k])
        ttnn.execute_trace(self.dev, self.trace_id, cq_id=0, blocking=True)
        return self._host(self.t_out)[0, 0, 0, : self.vocab]

    # ---- attention cores -----------------------------------------------------------------------------
    def _mla_core(self, i, d, qb, ckv, kpe, cos, sin, cur, inp):
        H = self.H
        q = ttnn.reshape(qb, (1, 1, H, self.dn + self.dr))               # heads in the row dimension
        q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, 1, H, self.dn])
        q_pe = ttnn.slice(q, [0, 0, 0, self.dn], [1, 1, H, self.dn + self.dr])
        q_pe = self._rope(q_pe, cos, sin)
        kpe = self._rope(kpe, cos, sin)
        # absorb the key up-projection into the query: [1,H,1,64] x [1,H,64,256] -> [1,H,1,256]
        qn = ttnn.transpose(q_nope, 1, 2)
        q_lat = ttnn.matmul(qn, d["wuk"])
        q_lat = ttnn.transpose(q_lat, 1, 2)                              # [1,1,H,256]
        q_full = ttnn.concat([q_lat, q_pe], dim=-1)                      # [1,1,H,288]
        new = ttnn.concat([ckv, kpe], dim=-1)                            # [1,1,1,288]
        self._write_cache(i, new, cur, inp)
        o = ttnn.transformer.flash_multi_latent_attention_decode(
            q_full, self.kvpe[i], head_dim_v=self.kvr, cur_pos_tensor=cur, scale=(self.dn + self.dr) ** -0.5,
            program_config=self.sdpa_pc)                                  # [1,1,H,256]
        o = ttnn.transpose(o, 1, 2)                                      # [1,H,1,256]
        o = ttnn.matmul(o, d["wuv"])                                     # [1,H,1,64]
        o = ttnn.reshape(o, (1, 1, 1, H * self.dv))
        return o

    def _write_cache(self, i, new, cur, inp):
        if self.upd == "int":
            ttnn.update_cache(self.kvpe[i], new, self._pos)          # position baked into the program: not for traces
        elif self.upd == "paged":
            ttnn.experimental.paged_update_cache(self.kvpe[i], new, update_idxs_tensor=cur)
        else:                                                        # one-hot row write: cache = cache * keep + onehot * new
            res = ttnn.add(ttnn.multiply(self.kvpe[i], inp["keep"]), ttnn.multiply(inp["oh"], new))
            ttnn.assign(res, self.kvpe[i])

    def _host_core(self, i, d, qb, ckv, kpe, pos):
        """Stage-b fallback: the attention core in torch on the host."""
        from ref_model import rotate_half
        H = self.H
        q = self._host(qb).reshape(1, H, 1, self.dn + self.dr)
        ckv_h = self._host(ckv).reshape(1, self.kvr)
        kpe_h = self._host(kpe).reshape(1, 1, 1, self.dr)
        cos = self.cos[pos].reshape(1, 1, 1, 32)
        sin = self.sin[pos].reshape(1, 1, 1, 32)
        q_nope, q_pe = q[..., : self.dn], q[..., self.dn:]
        q_pe = q_pe * cos + rotate_half(q_pe) * sin
        kpe_h = kpe_h * cos + rotate_half(kpe_h) * sin
        kvb = d["kvb_host"].float()                                      # [H, 128, 256]
        kv = torch.einsum("hdk,tk->htd", kvb, ckv_h.reshape(1, -1))      # [H,1,128]
        k = torch.cat([kv[..., : self.dn], kpe_h.reshape(1, 1, self.dr).expand(H, 1, self.dr)], -1)[None]   # [1,H,1,96]
        v = kv[..., self.dn:][None]
        if self.hk[i] is not None:
            k = torch.cat([self.hk[i][0], k], 2)
            v = torch.cat([self.hk[i][1], v], 2)
        self.hk[i] = (k, v)
        s = (torch.cat([q_nope, q_pe], -1) @ k.transpose(2, 3)) * ((self.dn + self.dr) ** -0.5)
        o = torch.softmax(s, -1) @ v                                      # [1,H,1,64]
        o = o.reshape(1, 1, 1, H * self.dv).bfloat16()
        return self._up(o)

    _pos = 0

    def run(self, tok, pos, tap=None):
        return self.step(tok, pos, tap)
