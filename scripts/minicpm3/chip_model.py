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
    def __init__(self, dev, model_dir, weights_file, *, attn="mla", max_len=256, wdtype=None, fidelity="default", upd="int", n_layers=0, opt=1, shard=False, mlp_dtype="bf16", mlp_fidelity="hifi2", log=print):
        import json
        self.dev = dev
        self.attn = attn
        self.shard = shard       # DRAM-sharded weights and a sharded chain for the MLP and o_proj (measured 283 against 350 us per MLP layer)
        self.fuse = opt == 3     # opt 3: opt 2 plus a sharded residual stream with sharded norms and one fused projection for the query and key inputs
        if self.fuse:
            opt, shard = 2, True
        self.opt = opt           # 1: first composition; 2: fewer ops (rope folded into the absorbed matmul, split key projection)
        self.shard = shard or self.fuse
        self.mlp_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[mlp_dtype]
        self.mlp_fidelity = mlp_fidelity
        self.skip = set()        # timing experiments only: parts of the step to leave out ("mlp", "lm")
        self.upd = upd           # cache write: "int" (update_cache at a Python position), "paged" or "onehot" (position from a device tensor)
        self.cfg = c = json.load(open(f"{model_dir}/config.json"))
        self.L, self.H = n_layers or c["num_hidden_layers"], c["num_attention_heads"]
        self.hid, self.kvr, self.dn, self.dr = c["hidden_size"], c["kv_lora_rank"], c["qk_nope_head_dim"], c["qk_rope_head_dim"]
        self.dv = 64
        self.eps = c["rms_norm_eps"]
        self.max_len = -(-max_len // 128) * 128      # the flash MLA op needs a cache length that is a multiple of its key chunk size (128)
        self.wdtype = wdtype or ttnn.bfloat16
        self.log = log
        # `fidelity="hifi4"`: highest math fidelity and 32-bit accumulation in every matmul (the default config accumulates in 16 bits)
        self.rms_ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                              fp32_dest_acc_en=True, packer_l1_acc=True)
        # Per-head matmuls (40 batches of one row) with the default program config take 85 to 127 us each; one batch per core with an explicit
        # config takes 10 to 17 us (scripts/minicpm3/exp_bmm.py). HiFi2 keeps the default's accuracy.
        self.bmm_ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                                              fp32_dest_acc_en=False, packer_l1_acc=False)
        self.ckc = None
        if fidelity == "hifi4":
            self.ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                              fp32_dest_acc_en=True, packer_l1_acc=True)
        # One kv head and batch 1 would spread the op over all 110 cores of a Blackhole chip, and the op refuses more than 64 cores per head
        # (TT_FATAL: tree reduction max 6 rounds). An 8 by 8 grid is accepted and matches torch (PCC 0.99995 at the model's shapes).
        self.sdpa_pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=32, k_chunk_size=128,
                                              exp_approx_mode=False)
        self.qa_n = c["q_lora_rank"]
        self.rs = c["scale_depth"] / math.sqrt(c["num_hidden_layers"])
        from ref_model import rope_tables
        self.cos, self.sin = rope_tables(c, max_len)
        self.f = safe_open(weights_file, "pt")
        if self.shard:
            nc = 40                                    # 5 by 8 cores: divides the tile counts of 2560, 6400 (and 3840 is not used)
            wmc = lambda k: ttnn.create_sharded_memory_config(  # noqa: E731
                shape=(32, k // nc), core_grid=ttnn.CoreGrid(y=5, x=8), strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
            self.mc_d, self.mc_f = wmc(self.hid), wmc(c["intermediate_size"])
            mk = lambda kt, nt, act=None: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(  # noqa: E731
                in0_block_w=kt // nc, per_core_M=1, per_core_N=nt // nc, fused_activation=act)
            hk, fk = self.hid // 32, c["intermediate_size"] // 32
            self.pc_gate = mk(hk, fk, ttnn.UnaryOpType.SILU)
            self.pc_up = mk(hk, fk)
            self.pc_down = mk(fk, hk)
            self.pc_o = mk(hk, hk)
            self.sh_ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                                                 fp32_dest_acc_en=False, packer_l1_acc=True)
            fid = {"hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi, "hifi4": ttnn.MathFidelity.HiFi4}[self.mlp_fidelity]
            self.mlp_ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=fid, math_approx_mode=True, fp32_dest_acc_en=False,
                                                                  packer_l1_acc=True)
        if self.fuse:
            nc = 40
            bw = self.hid // nc // 32
            self.pc_norm = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=[8, 5], subblock_w=min(4, bw), block_h=1,
                                                                       block_w=bw, inplace=False)
            n_tiles = (self.qa_n + self.kvr + 2 * self.dr) // 32
            self.pc_qkv = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4), in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1,
                per_core_N=-(-n_tiles // 32), fuse_batch=True, fused_activation=None, mcast_in0=True)
        self._load(log)
        self.reset()

    # ---- weights -----------------------------------------------------------------------------------
    def _w(self, name, scale=1.0, transpose=True, dtype=None):
        t = self.f.get_tensor(name).float() * scale
        if transpose:
            t = t.t()
        return ttnn.from_torch(t.contiguous().bfloat16(), device=self.dev, dtype=dtype or self.wdtype, layout=ttnn.TILE_LAYOUT,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _sw(self, name, scale=1.0, dtype=None):
        """A weight [out, in] as [in, out] bfloat16 sharded across the DRAM banks (width sharded), as the DRAM-sharded matmul wants."""
        t = self.f.get_tensor(name).float() * scale
        t = t.t().contiguous()
        k, n = t.shape
        banks = self.dev.dram_grid_size().x
        padded = math.ceil(n / (32 * banks)) * 32 * banks
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM,
                               ttnn.ShardSpec(grid, (k, padded // banks), ttnn.ShardOrientation.ROW_MAJOR))
        return ttnn.from_torch(torch.nn.functional.pad(t, (0, padded - n)).bfloat16(), device=self.dev, dtype=dtype or self.wdtype,
                               layout=ttnn.TILE_LAYOUT, memory_config=mc)

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
            }
            if self.shard:
                d["o"] = self._sw(a + "o_proj.weight", scale=self.rs)
                d["gate"] = self._sw(p + "mlp.gate_proj.weight", dtype=self.mlp_dtype)
                d["up"] = self._sw(p + "mlp.up_proj.weight", dtype=self.mlp_dtype)
                d["down"] = self._sw(p + "mlp.down_proj.weight", scale=self.rs, dtype=self.mlp_dtype)
            else:
                d["o"] = self._w(a + "o_proj.weight", scale=self.rs)
                d["gate"] = self._w(p + "mlp.gate_proj.weight")
                d["up"] = self._w(p + "mlp.up_proj.weight")
                d["down"] = self._w(p + "mlp.down_proj.weight", scale=self.rs)
            if self.opt == 2:
                kva_w = self.f.get_tensor(a + "kv_a_proj_with_mqa.weight").float()           # [288, 2560]
                R = _rot_matrix()
                d["ckv_w"] = ttnn.from_torch(kva_w[: self.kvr].t().contiguous().bfloat16(), device=dev, dtype=self.wdtype, layout=ttnn.TILE_LAYOUT)
                wk = kva_w[self.kvr:].t()                                                      # [2560, 32]
                d["kpe_w"] = ttnn.from_torch(torch.cat([wk, wk @ R], dim=1).contiguous().bfloat16(), device=dev, dtype=self.wdtype,
                                             layout=ttnn.TILE_LAYOUT)                            # [2560, 64]: kpe and rotate_half(kpe)
                kvb2 = self.f.get_tensor(a + "kv_b_proj.weight").float().view(self.H, self.dn + self.dv, self.kvr)
                wbig = torch.zeros(self.H, self.dn + self.dr, self.kvr + self.dr)
                wbig[:, : self.dn, : self.kvr] = kvb2[:, : self.dn]
                wbig[:, self.dn:, self.kvr:] = torch.eye(self.dr)
                d["wbig"] = ttnn.from_torch(wbig.bfloat16().unsqueeze(0), device=dev, dtype=self.wdtype, layout=ttnn.TILE_LAYOUT)
            kvb = self.f.get_tensor(a + "kv_b_proj.weight").float().view(self.H, self.dn + self.dv, self.kvr)
            if self.attn == "mla" and self.opt == 2:
                d["wuv"] = ttnn.from_torch(kvb[:, self.dn:].transpose(1, 2).contiguous().bfloat16().unsqueeze(0), device=dev,
                                           dtype=self.wdtype, layout=ttnn.TILE_LAYOUT)
            elif self.attn == "mla":
                # W_UK_h [dn, kvr] used as q_nope_h @ W_UK_h; W_UV_h [dv, kvr] used as o_lat @ W_UV_h^T
                d["wuk"] = ttnn.from_torch(kvb[:, : self.dn].contiguous().bfloat16().unsqueeze(0), device=dev, dtype=self.wdtype,
                                           layout=ttnn.TILE_LAYOUT)
                d["wuv"] = ttnn.from_torch(kvb[:, self.dn:].transpose(1, 2).contiguous().bfloat16().unsqueeze(0), device=dev,
                                           dtype=self.wdtype, layout=ttnn.TILE_LAYOUT)
            else:
                d["kvb_host"] = kvb.bfloat16()
            if self.fuse:
                kva_w = self.f.get_tensor(a + "kv_a_proj_with_mqa.weight").float()
                qa_w = self.f.get_tensor(a + "q_a_proj.weight").float()
                R = _rot_matrix()
                wk = kva_w[self.kvr:].t()
                fused = torch.cat([qa_w.t(), kva_w[: self.kvr].t(), wk, wk @ R], dim=1)[:, : self.qa_n + self.kvr + 2 * self.dr]
                d["qkv"] = ttnn.from_torch(fused.contiguous().bfloat16(), device=dev, dtype=self.wdtype, layout=ttnn.TILE_LAYOUT)
                for k in ("qa", "kva", "ckv_w", "kpe_w"):
                    ttnn.deallocate(d.pop(k))
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

    def _bmm(self, a, w, k, n):
        """a [1,H,1,k] times w [1,H,k,n], one head per core."""
        pc = ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(8, 5), in0_block_w=k // 32, out_subblock_h=1,
                                                    out_subblock_w=1, per_core_M=1, per_core_N=n // 32)
        return ttnn.matmul(a, w, program_config=pc, compute_kernel_config=self.bmm_ckc)

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
            if self.opt == 2:
                n = self.cos.shape[0]
                self.cosF = mk(torch.cat([torch.ones(n, self.dn), self.cos], dim=1))             # [T, 96]: 1 on the no-position part
                self.sinF = mk(torch.cat([torch.zeros(n, self.dn), self.sin], dim=1))            # [T, 96]: 0 on the no-position part
                self.cs64 = mk(torch.cat([self.cos, self.sin], dim=1))                           # [T, 64]
                Rf = torch.zeros(self.dn + self.dr, self.dn + self.dr)
                Rf[self.dn:, self.dn:] = _rot_matrix()
                self.Rf = ttnn.from_torch(Rf.bfloat16(), device=self.dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                self.IdI = ttnn.from_torch(torch.cat([torch.eye(32), torch.eye(32)], dim=0).bfloat16(), device=self.dev, dtype=ttnn.bfloat16,
                                           layout=ttnn.TILE_LAYOUT)
                self.new_mc = ttnn.create_sharded_memory_config(shape=(32, self.kvr + self.dr), core_grid=ttnn.CoreGrid(y=1, x=1),
                                                                strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                                use_height_and_width_as_shard_shape=True)

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
        if self.fuse:
            return self._forward3(inp, tap)
        ids = inp["tok"]
        x = ttnn.embedding(ids, self.emb, layout=ttnn.TILE_LAYOUT)          # [1,1,hid]
        x = ttnn.reshape(x, (1, 1, 1, self.hid))
        cos = ttnn.reshape(ttnn.embedding(inp["pos_u"], self.cos_tab, layout=ttnn.TILE_LAYOUT), (1, 1, 1, 32))
        sin = ttnn.reshape(ttnn.embedding(inp["pos_u"], self.sin_tab, layout=ttnn.TILE_LAYOUT), (1, 1, 1, 32))
        cur = inp["cur"]
        if self.opt == 2:
            emb = lambda tab, w: ttnn.reshape(ttnn.embedding(inp["pos_u"], tab, layout=ttnn.TILE_LAYOUT), (1, 1, 1, w))  # noqa: E731
            self._cosF, self._sinF, self._cs64 = emb(self.cosF, 96), emb(self.sinF, 96), emb(self.cs64, 64)
        if tap:
            tap("embed", self._host(x)[0, 0, 0])
        for i, d in enumerate(self.layers):
            h = self._rms(x, d["in_norm"])
            if self.opt == 2:
                ao = self._attn_v2(i, d, h, cur, inp)
            else:
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
            if self.shard:
                o = ttnn.linear(ttnn.to_memory_config(ao, self.mc_d), d["o"], program_config=self.pc_o, memory_config=self.mc_d,
                                compute_kernel_config=self.sh_ckc)
                x = ttnn.add(x, ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG))
            else:
                x = ttnn.add(x, self._lin(ao, d["o"]))
            if tap:
                tap(f"layer_{i:02d}.attn", self._host(x)[0, 0, 0])
            if "mlp" in self.skip:
                continue
            h = self._rms(x, d["post_norm"])
            if self.shard:
                hs = ttnn.to_memory_config(h, self.mc_d)
                gg = ttnn.linear(hs, d["gate"], program_config=self.pc_gate, memory_config=self.mc_f, compute_kernel_config=self.mlp_ckc)
                uu = ttnn.linear(hs, d["up"], program_config=self.pc_up, memory_config=self.mc_f, compute_kernel_config=self.mlp_ckc)
                dn = ttnn.linear(ttnn.multiply(gg, uu), d["down"], program_config=self.pc_down, memory_config=self.mc_d,
                                 compute_kernel_config=self.mlp_ckc)
                x = ttnn.add(x, ttnn.to_memory_config(dn, ttnn.DRAM_MEMORY_CONFIG))
            else:
                m = ttnn.multiply(self._lin(h, d["gate"], activation="silu"), self._lin(h, d["up"]))
                x = ttnn.add(x, self._lin(m, d["down"]))
            if tap:
                tap(f"layer_{i:02d}", self._host(x)[0, 0, 0])
        x = self._rms(x, self.final_norm)
        if "lm" in self.skip:
            return x
        # row-major output: a tile layout read back is 32 rows (4.7 MB, about 8 ms); one row is 147 KB
        return ttnn.to_layout(self._lin(x, self.lm_head, dtype=ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)   # float32 logits: bf16 has 0.125 steps at 16 to 32

    def _srms(self, x, w):
        """rms_norm on the width-sharded residual stream (6.5 microseconds against 42 interleaved, same accuracy)."""
        return ttnn.rms_norm(x, epsilon=self.eps, weight=w, program_config=self.pc_norm, memory_config=self.mc_d, compute_kernel_config=self.rms_ckc)

    def _forward3(self, inp, tap=None):
        """Opt 3: the residual stream stays width-sharded in L1 over 40 cores; norms, the output projection and the MLP run sharded."""
        cur = inp["cur"]
        self._cosF, self._sinF, self._cs64 = (
            ttnn.reshape(ttnn.embedding(inp["pos_u"], tab, layout=ttnn.TILE_LAYOUT), (1, 1, 1, w))
            for tab, w in ((self.cosF, 96), (self.sinF, 96), (self.cs64, 64)))
        x = ttnn.reshape(ttnn.embedding(inp["tok"], self.emb, layout=ttnn.TILE_LAYOUT), (1, 1, 1, self.hid))
        x = ttnn.to_memory_config(x, self.mc_d)
        if tap:
            tap("embed", self._host(x)[0, 0, 0])
        qa_n, kvr, dr = self.qa_n, self.kvr, self.dr
        for i, d in enumerate(self.layers):
            hs = self._srms(x, d["in_norm"])
            qkv = ttnn.linear(ttnn.to_memory_config(hs, ttnn.DRAM_MEMORY_CONFIG), d["qkv"], program_config=self.pc_qkv)
            qa = ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, 1, qa_n])
            ckv = ttnn.slice(qkv, [0, 0, 0, qa_n], [1, 1, 1, qa_n + kvr])
            k64 = ttnn.slice(qkv, [0, 0, 0, qa_n + kvr], [1, 1, 1, qa_n + kvr + 2 * dr])
            ao = self._attn_core(i, d, self._rms(qa, d["qa_norm"]), self._rms(ckv, d["kv_norm"]), k64, cur, inp)
            o = ttnn.linear(ttnn.to_memory_config(ao, self.mc_d), d["o"], program_config=self.pc_o, memory_config=self.mc_d,
                            compute_kernel_config=self.sh_ckc)
            x = ttnn.add(x, o)
            if tap:
                tap(f"layer_{i:02d}.attn", self._host(x)[0, 0, 0])
            if "mlp" in self.skip:
                continue
            hs = self._srms(x, d["post_norm"])
            gg = ttnn.linear(hs, d["gate"], program_config=self.pc_gate, memory_config=self.mc_f, compute_kernel_config=self.mlp_ckc)
            uu = ttnn.linear(hs, d["up"], program_config=self.pc_up, memory_config=self.mc_f, compute_kernel_config=self.mlp_ckc)
            x = ttnn.add(x, ttnn.linear(ttnn.multiply(gg, uu), d["down"], program_config=self.pc_down, memory_config=self.mc_d,
                                        compute_kernel_config=self.mlp_ckc))
            if tap:
                tap(f"layer_{i:02d}", self._host(x)[0, 0, 0])
        x = self._srms(x, self.final_norm)
        if "lm" in self.skip:
            return x
        return ttnn.to_layout(self._lin(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), self.lm_head, dtype=ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)

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
        q_lat = self._bmm(qn, d["wuk"], self.dn, self.kvr)
        q_lat = ttnn.transpose(q_lat, 1, 2)                              # [1,1,H,256]
        q_full = ttnn.concat([q_lat, q_pe], dim=-1)                      # [1,1,H,288]
        new = ttnn.concat([ckv, kpe], dim=-1)                            # [1,1,1,288]
        self._write_cache(i, new, cur, inp)
        o = ttnn.transformer.flash_multi_latent_attention_decode(
            q_full, self.kvpe[i], head_dim_v=self.kvr, cur_pos_tensor=cur, scale=(self.dn + self.dr) ** -0.5,
            program_config=self.sdpa_pc)                                  # [1,1,H,256]
        o = ttnn.transpose(o, 1, 2)                                      # [1,H,1,256]
        o = self._bmm(o, d["wuv"], self.kvr, self.dv)                    # [1,H,1,64]
        o = ttnn.reshape(o, (1, 1, 1, H * self.dv))
        return o

    def _attn_v2(self, i, d, h, cur, inp):
        """The attention with fewer ops: rotary on the whole 96-wide query (the no-position part has cos 1 and sin 0), the key up-projection and the
        pass-through of the rotated part in one batched matmul with a 288-wide result, and the key's rotate-half folded into its projection."""
        qa = self._rms(self._lin(h, d["qa"]), d["qa_norm"])
        ckv = self._rms(self._lin(h, d["ckv_w"]), d["kv_norm"])           # [1,1,1,256]
        k64 = self._lin(h, d["kpe_w"])                                    # [1,1,1,64]: kpe, rotate_half(kpe)
        return self._attn_core(i, d, qa, ckv, k64, cur, inp)

    def _attn_core(self, i, d, qa, ckv, k64, cur, inp):
        H = self.H
        qb = self._lin(qa, d["qb"])                                       # [1,1,1,H*96]
        q = ttnn.reshape(qb, (1, 1, H, self.dn + self.dr))
        qr = q if "qrope" in self.skip else ttnn.add(ttnn.multiply(q, self._cosF), ttnn.multiply(ttnn.matmul(q, self.Rf), self._sinF))
        qh = ttnn.transpose(qr, 1, 2)                                     # [1,H,1,96]
        if "bmm" in self.skip:
            ql = ttnn.concat([qh, qh, qh], dim=-1)
        else:
            ql = self._bmm(qh, d["wbig"], self.dn + self.dr, self.kvr + self.dr)   # [1,H,1,288]: latent query and rotated rope query
        q_full = ttnn.transpose(ql, 1, 2)                                 # [1,1,H,288]
        if "kpath" in self.skip:
            new = ttnn.slice(q_full, [0, 0, 0, 0], [1, 1, 1, self.kvr + self.dr])
        else:
            krot = ttnn.matmul(ttnn.multiply(k64, self._cs64), self.IdI)  # kpe * cos + rotate_half(kpe) * sin
            new = ttnn.concat([ckv, krot], dim=-1)                        # [1,1,1,288]
        if "cache" not in self.skip:
            if self.upd == "paged":
                ttnn.experimental.paged_update_cache(self.kvpe[i], ttnn.to_memory_config(new, self.new_mc), update_idxs_tensor=cur)
            else:
                self._write_cache(i, new, cur, inp)
        if "flash" in self.skip:
            o = ttnn.slice(q_full, [0, 0, 0, 0], [1, 1, H, self.kvr])
        else:
            o = ttnn.transformer.flash_multi_latent_attention_decode(
                q_full, self.kvpe[i], head_dim_v=self.kvr, cur_pos_tensor=cur, scale=(self.dn + self.dr) ** -0.5, program_config=self.sdpa_pc)
        if "bmm" in self.skip:
            o = ttnn.slice(ttnn.transpose(o, 1, 2), [0, 0, 0, 0], [1, H, 1, self.dv])
        else:
            o = self._bmm(ttnn.transpose(o, 1, 2), d["wuv"], self.kvr, self.dv)   # [1,H,1,64]
        return ttnn.reshape(o, (1, 1, 1, H * self.dv))

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
