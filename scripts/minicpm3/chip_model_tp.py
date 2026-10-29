# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""MiniCPM3-4B decode on the two chips of one Blackhole board (tensor parallel over a 1 by 2 mesh).

The same absorbed multi-latent attention as `chip_model.py` (opt 3), split over two chips:
  - residual stream and the small query/key input projection (`qkv`, 2560 to 1088) are replicated; the latent KV cache is replicated (it has one head);
  - query heads are split, 20 per chip (`qb`, the per-head matmuls, the flash MLA op); `o_proj` is row-parallel, so its output is a partial sum;
  - MLP: gate and up are column-parallel, down is row-parallel (partial sum). The intermediate width per chip is padded from 3,200 to 3,328 with zeros so each DRAM bank
    holds a whole number of tiles; the matmuls run DRAM-sharded on 8 cores (one per bank);
  - two all-reduces per layer (after `o_proj` and after down); the LM head is column-parallel over the vocabulary.
Scalings are folded into the weights as in `chip_model.py`.
"""
from __future__ import annotations

import math

import torch
import ttnn
from safetensors import safe_open

from chip_model import _rot_matrix

NB = 8                      # DRAM banks per chip, one worker core per bank in the sharded matmuls
FPAD = 3328                 # per-chip intermediate width after padding (3,200 rounded up to 8 banks of whole tiles)


class ChipMiniCPM3TP:
    def __init__(self, mesh, model_dir, weights_file, *, max_len=256, n_layers=0, mlp_dtype="bf16", mlp_fidelity="hifi2", links=2, log=print):
        import json
        from ref_model import rope_tables
        self.mesh, self.nd, self.links, self.log = mesh, 2, links, log
        self.cfg = c = json.load(open(f"{model_dir}/config.json"))
        self.L = n_layers or c["num_hidden_layers"]
        self.H = c["num_attention_heads"]
        self.Hl = self.H // 2
        self.hid, self.kvr, self.dn, self.dr, self.dv = c["hidden_size"], c["kv_lora_rank"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["hidden_size"] // c["num_attention_heads"]
        self.qa_n = c["q_lora_rank"]
        self.eps = c["rms_norm_eps"]
        self.max_len = -(-max_len // 128) * 128      # the flash MLA op needs a cache length that is a multiple of its key chunk size (128)
        self.rs = c["scale_depth"] / math.sqrt(c["num_hidden_layers"])
        self.f_dim = c["intermediate_size"]
        self.fl = self.f_dim // 2
        self.mlp_dtype = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}[mlp_dtype]
        self.cos, self.sin = rope_tables(c, max_len)
        self.f = safe_open(weights_file, "pt")
        arch = mesh.arch() if hasattr(mesh, "arch") else mesh.get_device(0).arch()
        hifi4 = dict(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        self.rms_ckc = ttnn.init_device_compute_kernel_config(arch, **hifi4)
        self.bmm_ckc = ttnn.init_device_compute_kernel_config(arch, math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False,
                                                              packer_l1_acc=False)
        self.sh_ckc = ttnn.init_device_compute_kernel_config(arch, math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False,
                                                             packer_l1_acc=True)
        fid = {"hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}[mlp_fidelity]
        self.mlp_ckc = ttnn.init_device_compute_kernel_config(arch, math_fidelity=fid, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True)
        self.sdpa_pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=32, k_chunk_size=128, exp_approx_mode=False)
        # sharded layouts on 8 cores (1 by 8)
        wmc = lambda k: ttnn.create_sharded_memory_config(  # noqa: E731
            shape=(32, k // NB), core_grid=ttnn.CoreGrid(y=1, x=NB), strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True)
        self.mc_d, self.mc_f, self.mc_o = wmc(self.hid), wmc(FPAD), wmc(self.hid // 2)
        mk = lambda kt, nt, act=None: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(  # noqa: E731
            in0_block_w=kt // NB, per_core_M=1, per_core_N=nt // NB, fused_activation=act)
        hk, fk = self.hid // 32, FPAD // 32
        self.pc_gate, self.pc_up = mk(hk, fk, ttnn.UnaryOpType.SILU), mk(hk, fk)
        self.pc_down = mk(fk, hk)
        self.pc_o = mk(hk // 2, hk)
        bw = self.hid // NB // 32
        self.pc_norm = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=[NB, 1], subblock_w=next(s for s in (4, 3, 2, 1) if bw % s == 0),
                                                                   block_h=1, block_w=bw, inplace=False)
        n_tiles = (self.qa_n + self.kvr + 2 * self.dr) // 32
        self.pc_qkv = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4), in0_block_w=4, out_subblock_h=1, out_subblock_w=1, per_core_M=1, per_core_N=-(-n_tiles // 32),
            fuse_batch=True, fused_activation=None, mcast_in0=True)
        self.new_mc = ttnn.create_sharded_memory_config(shape=(32, self.kvr + self.dr), core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.HEIGHT,
                                                        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        self.rep = ttnn.ReplicateTensorToMesh(mesh)
        self.skip = set()
        self._load(log)
        self.reset()

    # ---- weights -----------------------------------------------------------------------------------
    def _rep(self, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(t.contiguous().bfloat16() if dtype != ttnn.uint32 else t, device=self.mesh, mesh_mapper=self.rep, dtype=dtype, layout=layout,
                               memory_config=mc)

    def _shard(self, t, dim, dtype=ttnn.bfloat16, mc=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(t.contiguous().bfloat16(), device=self.mesh, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=dim), dtype=dtype,
                               layout=ttnn.TILE_LAYOUT, memory_config=mc)

    def _dram_shard_mc(self, k, n):
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(NB - 1, 0))})
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, ttnn.ShardSpec(grid, (k, n // NB), ttnn.ShardOrientation.ROW_MAJOR))

    def _norm_w(self, name, scale=1.0):
        t = self.f.get_tensor(name).float() * scale
        return self._rep(t.reshape(1, 1, -1, 32), layout=ttnn.ROW_MAJOR_LAYOUT)

    def _pad_halves(self, w, dim):
        """w [.., f_dim, ..] along `dim`: split in two halves, pad each to FPAD with zeros, and join them again (so sharding on `dim` gives each chip a padded half)."""
        a, b = w.chunk(2, dim=dim)
        pad_shape = list(a.shape)
        pad_shape[dim] = FPAD - self.fl
        z = torch.zeros(pad_shape)
        return torch.cat([a, z, b, z], dim=dim)

    def _load(self, log):
        c = self.cfg
        emb = self.f.get_tensor("model.embed_tokens.weight").float()
        self.vocab = emb.shape[0]
        vpad = -(-self.vocab // 64) * 64                                    # 73,472: two halves of whole tiles
        self.emb = self._rep(emb * c["scale_emb"], layout=ttnn.ROW_MAJOR_LAYOUT)
        head = torch.zeros(self.hid, vpad)
        head[:, : self.vocab] = emb.t()
        self.lm_head = self._shard(head, dim=1)
        self.final_norm = self._norm_w("model.norm.weight", scale=1.0 / (c["hidden_size"] / c["dim_model_base"]))
        n = self.cos.shape[0]
        mk = lambda t: self._rep(t, layout=ttnn.ROW_MAJOR_LAYOUT)  # noqa: E731
        self.cos_tab, self.sin_tab = mk(self.cos), mk(self.sin)
        self.cosF = mk(torch.cat([torch.ones(n, self.dn), self.cos], dim=1))
        self.sinF = mk(torch.cat([torch.zeros(n, self.dn), self.sin], dim=1))
        self.cs64 = mk(torch.cat([self.cos, self.sin], dim=1))
        Rf = torch.zeros(self.dn + self.dr, self.dn + self.dr)
        Rf[self.dn:, self.dn:] = _rot_matrix()
        self.Rf = self._rep(Rf)
        self.IdI = self._rep(torch.cat([torch.eye(32), torch.eye(32)], dim=0))
        self.layers = []
        R = _rot_matrix()
        for i in range(self.L):
            p = f"model.layers.{i}."
            a = p + "self_attn."
            g = lambda nme: self.f.get_tensor(nme).float()  # noqa: E731
            kva_w, qa_w, qb_w, kvb = g(a + "kv_a_proj_with_mqa.weight"), g(a + "q_a_proj.weight"), g(a + "q_b_proj.weight"), g(a + "kv_b_proj.weight")
            wk = kva_w[self.kvr:].t()
            fused = torch.cat([qa_w.t(), kva_w[: self.kvr].t(), wk, wk @ R], dim=1)
            kvb = kvb.view(self.H, self.dn + self.dv, self.kvr)
            wbig = torch.zeros(self.H, self.dn + self.dr, self.kvr + self.dr)
            wbig[:, : self.dn, : self.kvr] = kvb[:, : self.dn]
            wbig[:, self.dn:, self.kvr:] = torch.eye(self.dr)
            o_t = g(a + "o_proj.weight").t() * self.rs                     # [in, out]: rows are head outputs, split by head
            gate, up = g(p + "mlp.gate_proj.weight").t(), g(p + "mlp.up_proj.weight").t()       # [hid, f]
            down = g(p + "mlp.down_proj.weight").t() * self.rs                                   # [f, hid]
            d = {
                "in_norm": self._norm_w(p + "input_layernorm.weight"), "post_norm": self._norm_w(p + "post_attention_layernorm.weight"),
                "qa_norm": self._norm_w(a + "q_a_layernorm.weight"), "kv_norm": self._norm_w(a + "kv_a_layernorm.weight"),
                "qkv": self._rep(fused),
                "qb": self._shard(qb_w.t(), dim=1),
                "wbig": self._shard(wbig.unsqueeze(0), dim=1),
                "wuv": self._shard(kvb[:, self.dn:].transpose(1, 2).contiguous().unsqueeze(0), dim=1),
                "o": self._shard(o_t, dim=0, mc=self._dram_shard_mc(self.hid // 2, self.hid)),
                "gate": self._shard(self._pad_halves(gate, 1), dim=1, dtype=self.mlp_dtype, mc=self._dram_shard_mc(self.hid, FPAD)),
                "up": self._shard(self._pad_halves(up, 1), dim=1, dtype=self.mlp_dtype, mc=self._dram_shard_mc(self.hid, FPAD)),
                "down": self._shard(self._pad_halves(down, 0), dim=0, dtype=self.mlp_dtype, mc=self._dram_shard_mc(FPAD, self.hid)),
            }
            self.layers.append(d)
            if i % 10 == 0:
                log(f"loaded layer {i}")

    def reset(self):
        z = torch.zeros(1, 1, self.max_len, self.kvr + self.dr)
        self.kvpe = [self._rep(z) for _ in range(self.L)]

    # ---- helpers -------------------------------------------------------------------------------------
    def _host(self, t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).float()[0:1]

    def _bmm(self, a, w, k, n):
        pc = ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(8, 5), in0_block_w=k // 32, out_subblock_h=1, out_subblock_w=1,
                                                    per_core_M=1, per_core_N=n // 32)
        return ttnn.matmul(a, w, program_config=pc, compute_kernel_config=self.bmm_ckc)

    def _srms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=w, program_config=self.pc_norm, memory_config=self.mc_d, compute_kernel_config=self.rms_ckc)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=w, compute_kernel_config=self.rms_ckc)

    def _allreduce_add(self, x, partial):
        """x (sharded) plus the sum over both chips of `partial` (sharded, one partial sum per chip)."""
        r = ttnn.all_reduce(ttnn.to_memory_config(partial, ttnn.DRAM_MEMORY_CONFIG), cluster_axis=1, num_links=self.links, topology=ttnn.Topology.Linear)
        return ttnn.add(x, ttnn.to_memory_config(r, self.mc_d))

    # ---- one step ------------------------------------------------------------------------------------
    def _host_inputs(self, tok, pos):
        mm = self.rep
        return {
            "tok": ttnn.from_torch(torch.tensor([[tok]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mm),
            "pos_u": ttnn.from_torch(torch.tensor([[pos]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mm),
            "cur": ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mm),
        }

    def step(self, tok, pos, tap=None):
        dev_in = {k: ttnn.to_device(v, self.mesh) for k, v in self._host_inputs(tok, pos).items()}
        return self._out(self._forward(dev_in, tap))

    run = step

    def _out(self, logits):
        full = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=3)).float()
        return full[0, 0, 0, : self.vocab]

    def _forward(self, inp, tap=None):
        cur = inp["cur"]
        emb = lambda tab, w: ttnn.reshape(ttnn.embedding(inp["pos_u"], tab, layout=ttnn.TILE_LAYOUT), (1, 1, 1, w))  # noqa: E731
        cosF, sinF, cs64 = emb(self.cosF, 96), emb(self.sinF, 96), emb(self.cs64, 64)
        x = ttnn.reshape(ttnn.embedding(inp["tok"], self.emb, layout=ttnn.TILE_LAYOUT), (1, 1, 1, self.hid))
        x = ttnn.to_memory_config(x, self.mc_d)
        if tap:
            tap("embed", self._host(x)[0, 0, 0])
        qa_n, kvr, dr, Hl = self.qa_n, self.kvr, self.dr, self.Hl
        for i, d in enumerate(self.layers):
            hs = self._srms(x, d["in_norm"])
            qkv = ttnn.linear(ttnn.to_memory_config(hs, ttnn.DRAM_MEMORY_CONFIG), d["qkv"], program_config=self.pc_qkv)
            qa = self._rms(ttnn.slice(qkv, [0, 0, 0, 0], [1, 1, 1, qa_n]), d["qa_norm"])
            ckv = self._rms(ttnn.slice(qkv, [0, 0, 0, qa_n], [1, 1, 1, qa_n + kvr]), d["kv_norm"])
            k64 = ttnn.slice(qkv, [0, 0, 0, qa_n + kvr], [1, 1, 1, qa_n + kvr + 2 * dr])
            # ---- attention on this chip's 20 heads
            qb = ttnn.linear(qa, d["qb"])                                           # [1,1,1,Hl*96]
            q = ttnn.reshape(qb, (1, 1, Hl, self.dn + self.dr))
            qr = ttnn.add(ttnn.multiply(q, cosF), ttnn.multiply(ttnn.matmul(q, self.Rf), sinF))
            ql = self._bmm(ttnn.transpose(qr, 1, 2), d["wbig"], self.dn + self.dr, self.kvr + self.dr)
            q_full = ttnn.transpose(ql, 1, 2)                                       # [1,1,Hl,288]
            krot = ttnn.matmul(ttnn.multiply(k64, cs64), self.IdI)
            new = ttnn.concat([ckv, krot], dim=-1)
            ttnn.experimental.paged_update_cache(self.kvpe[i], ttnn.to_memory_config(new, self.new_mc), update_idxs_tensor=cur)
            o = ttnn.transformer.flash_multi_latent_attention_decode(q_full, self.kvpe[i], head_dim_v=self.kvr, cur_pos_tensor=cur,
                                                                     scale=(self.dn + self.dr) ** -0.5, program_config=self.sdpa_pc)
            o = self._bmm(ttnn.transpose(o, 1, 2), d["wuv"], self.kvr, self.dv)     # [1,Hl,1,64]
            ao = ttnn.reshape(o, (1, 1, 1, Hl * self.dv))
            part = ttnn.linear(ttnn.to_memory_config(ao, self.mc_o), d["o"], program_config=self.pc_o, memory_config=self.mc_d, compute_kernel_config=self.sh_ckc)
            x = self._allreduce_add(x, part)
            if tap:
                tap(f"layer_{i:02d}.attn", self._host(x)[0, 0, 0])
            hs = self._srms(x, d["post_norm"])
            gg = ttnn.linear(hs, d["gate"], program_config=self.pc_gate, memory_config=self.mc_f, compute_kernel_config=self.mlp_ckc)
            uu = ttnn.linear(hs, d["up"], program_config=self.pc_up, memory_config=self.mc_f, compute_kernel_config=self.mlp_ckc)
            part = ttnn.linear(ttnn.multiply(gg, uu), d["down"], program_config=self.pc_down, memory_config=self.mc_d, compute_kernel_config=self.mlp_ckc)
            x = self._allreduce_add(x, part)
            if tap:
                tap(f"layer_{i:02d}", self._host(x)[0, 0, 0])
        x = self._srms(x, self.final_norm)
        return ttnn.to_layout(ttnn.linear(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), self.lm_head, dtype=ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)

    # ---- trace ----------------------------------------------------------------------------------------
    def enable_trace(self):
        self.t_in = {k: ttnn.to_device(v, self.mesh) for k, v in self._host_inputs(0, 0).items()}
        self.reset()
        self._forward(self.t_in)
        ttnn.synchronize_device(self.mesh)
        self.reset()
        self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.t_out = self._forward(self.t_in)
        ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh)

    def traced_step(self, tok, pos):
        for k, v in self._host_inputs(tok, pos).items():
            ttnn.copy_host_to_device_tensor(v, self.t_in[k])
        ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=True)
        return self._out(self.t_out)
