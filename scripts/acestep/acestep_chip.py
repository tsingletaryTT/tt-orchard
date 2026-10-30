# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Pieces of an ACE-Step 1.5 turbo request that the community bundle runs on the CPU and this module runs on the chip (issue 254, 2026-10-30).

The bundle's code directory is pinned by hash and is not edited. Each piece here is installed on a loaded `LocalWorker` at run time, replacing one method of the
bundle's `AceStepDiT` (or the handler's text encoder) with a version that computes the same function with ttnn ops. `acestep_fast.STEPS` names them, so each can be
switched on and off for a measurement.

  kv_on_chip    `AceStepDiT.prepare` builds the cross-attention keys and values of the 24 layers on the host in float32 (a 2048 by 2048 projection of the encoder states, then
                per layer a 2048 by 2048 projection and a per-head RMS norm of the keys), then uploads them. Here the encoder states are uploaded once and the projections
                and the norm run on the chip in bfloat16 with the same weights. The result has the same shape, layout and dtype on the device as the uploaded one.

  text_on_chip  The Qwen3-Embedding-0.6B text encoder (28 layers, run once per request on the CPU in float32) runs on the chip in bfloat16: `ChipEncoderStack`, built from the
                handler's own weights. The token embedding lookup stays on the CPU.

  lyric_on_chip The lyric encoder (8 layers, width 2048, bidirectional attention, a sliding window on alternate layers) runs on the chip through the same stack. Its input
                projection (1024 to 2048) stays on the CPU. It is the largest CPU piece of a request with long lyrics (0.29 s for a 230-token lyric).

Every piece needs `ttnn`, so nothing here is imported until `install_*` runs. The unit tests use stand-ins; the chip checks are the per-step bars and the end-to-end agreement
(`docs/run-logs/2026-10-30-acestep-chip-path.md`).
"""
from __future__ import annotations


def install_kv_on_chip(worker):
    """Replaces `worker.model.dit.prepare` with a version that computes the cross-attention K/V on the chip. Call once after `LocalWorker.start()`, before any request."""
    import torch
    import ttnn
    from tt_acestep import acestep_dit as dd
    from tt_acestep import acestep_host as host

    dit = worker.model.dit
    c, ref = dit.cfg, dit.ref
    MEM = dd.MEM
    # Weights go on the device now, before any song's tensors, for the allocation-order reason given in AceStepDiT.__init__.
    w_cond = dd._dev(dit.dev, ref.w_cond.to(torch.bfloat16))
    b_cond = dd._row(dit.dev, ref.b_cond)
    kv_dim = c.kv_heads * c.head_dim
    wk, wv, knorm = [], [], []
    for w in ref.weights:
        wk.append(dd._dev(dit.dev, w.ckv[:, :kv_dim].to(torch.bfloat16).contiguous()))
        wv.append(dd._dev(dit.dev, w.ckv[:, kv_dim:].to(torch.bfloat16).contiguous()))
        knorm.append(dd._row(dit.dev, w.ck_norm))

    def cross_kv(enc: torch.Tensor, geo):
        pad = torch.zeros(geo.enc_pad, enc.shape[-1])
        pad[: geo.enc_len] = enc.float()
        e0 = dd._dev(dit.dev, pad.to(torch.bfloat16).reshape(1, 1, geo.enc_pad, -1))
        e = ttnn.linear(e0, w_cond, bias=b_cond, compute_kernel_config=dit.ck_mm, memory_config=MEM, dtype=ttnn.bfloat16)
        ttnn.deallocate(e0)
        out = []
        for i in range(len(ref.weights)):
            k = dit._mm(e, wk[i], geo.enc_pad, c.hidden, kv_dim)
            v = dit._mm(e, wv[i], geo.enc_pad, c.hidden, kv_dim)
            kh = dit._heads(ttnn.reshape(k, [1, 1, geo.enc_pad, kv_dim]) if len(k.shape) != 4 else k, geo.enc_pad, c.kv_heads)
            ttnn.deallocate(k)
            kn = dit._rms(kh, knorm[i])
            ttnn.deallocate(kh)
            vh = dit._heads(ttnn.reshape(v, [1, 1, geo.enc_pad, kv_dim]) if len(v.shape) != 4 else v, geo.enc_pad, c.kv_heads)
            ttnn.deallocate(v)
            out.append((kn, vh))
        ttnn.deallocate(e)
        return out

    def prepare(encoder_hidden_states: torch.Tensor, frames: int):
        geo = host.Geometry(frames=frames, enc_len=encoder_hidden_states.shape[0])
        cos, sin = host.rope_tables(geo.seq_pad, c)
        full, sliding = host.self_masks(geo, c.sliding_window)
        kv = cross_kv(encoder_hidden_states, geo)
        f = lambda m: dd._dev(dit.dev, m.to(torch.bfloat16)[None, None])
        return dd.Song(geo, f(cos), f(sin), f(full), f(sliding), f(host.cross_mask(geo)), kv)

    dit.prepare = prepare
    dit.cross_kv_on_chip = cross_kv         # for the checks
    return dit


def pad_to(n: int, bucket: int) -> int:
    return (n + bucket - 1) // bucket * bucket


class ChipEncoderStack:
    """A Qwen3-style transformer stack (RMS norm, grouped-query attention with per-head q/k norm and rotary embedding, SwiGLU) on the chip, prefill only, batch 1.

    Both ACE-Step encoders have this shape: the text encoder (Qwen3-Embedding-0.6B: 28 layers, width 1024, causal) and the lyric encoder (8 layers, width 2048,
    bidirectional, sliding window on some layers). `get(name)` returns a host tensor of the HF state dict, `prefix` names the layer list ("layers"), and `layer_types`
    says per layer whether the attention is "full_attention" or "sliding_attention". The sequence is padded with zeros to a multiple of `bucket` so one compiled shape serves
    many lengths; the padded keys are masked out and the padded rows are dropped, so they change nothing. Weights are uploaded once, at construction.

    `forward(x)` takes [S, hidden] float32 and returns [S, hidden] float32 (the stack's last hidden state after the final norm when `final_norm` is given).
    """

    def __init__(self, dit, get, *, layers, hidden, heads, kv_heads, head_dim, intermediate, eps, rope_theta, causal, layer_types=None, sliding_window=None,
                 final_norm=None, bucket=128, fidelity="HiFi2"):
        import torch
        import ttnn
        from tt_acestep import acestep_dit as dd
        from tt_acestep import acestep_host as host

        self.dit, self.dd, self.host, self.ttnn, self.torch = dit, dd, host, ttnn, torch
        self.hidden, self.heads, self.kv_heads, self.head_dim = hidden, heads, kv_heads, head_dim
        self.intermediate, self.eps, self.rope_theta, self.causal, self.bucket = intermediate, eps, rope_theta, causal, bucket
        self.sliding_window = sliding_window
        self.layer_types = layer_types or ["full_attention"] * layers
        self.layers = []
        perm = host.interleave_pairs_permutation(head_dim)
        for i in range(layers):
            g = lambda k, i=i: get(f"layers.{i}.{k}").float()
            wq = host.permute_heads_rows(g("self_attn.q_proj.weight"), heads, head_dim, perm)
            wk = host.permute_heads_rows(g("self_attn.k_proj.weight"), kv_heads, head_dim, perm)
            wqkv = torch.cat([host.linear_to_mm(wq), host.linear_to_mm(wk), host.linear_to_mm(g("self_attn.v_proj.weight"))], dim=1)
            self.layers.append({
                "wqkv": dd._dev(dit.dev, wqkv.to(torch.bfloat16)),
                "q_norm": dd._row(dit.dev, g("self_attn.q_norm.weight")[perm]), "k_norm": dd._row(dit.dev, g("self_attn.k_norm.weight")[perm]),
                "wo": dd._dev(dit.dev, host.linear_to_mm(g("self_attn.o_proj.weight")).to(torch.bfloat16)),
                "in_norm": dd._row(dit.dev, g("input_layernorm.weight")), "post_norm": dd._row(dit.dev, g("post_attention_layernorm.weight")),
                "w_gateup": dd._dev(dit.dev, host.swiglu_interleave(g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")).to(torch.bfloat16)),
                "w_down": dd._dev(dit.dev, host.linear_to_mm(g("mlp.down_proj.weight")).to(torch.bfloat16)),
            })
        self.final_norm = dd._row(dit.dev, final_norm.float()) if final_norm is not None else None
        # matmul precision of the encoder stacks: the DiT's HiFi2 is tuned for speed; the encoders are small, so they take the higher setting unless asked otherwise
        self.ck_mm = ttnn.init_device_compute_kernel_config(dit.dev.arch(), math_fidelity=getattr(ttnn.MathFidelity, fidelity), math_approx_mode=False,
                                                            fp32_dest_acc_en=True, packer_l1_acc=False)
        self.trans_mat = dd._dev(dit.dev, host.rot_transformation_mat())
        self._tables = {}

    def _tables_for(self, s_pad, valid):
        """cos/sin and the additive masks for one padded length and valid length (cached: a server sees a few of them)."""
        key = (s_pad, valid)
        if key not in self._tables:
            torch, dd, host = self.torch, self.dd, self.host
            cfg = type("C", (), {"head_dim": self.head_dim, "rope_theta": self.rope_theta})
            cos, sin = host.rope_tables(s_pad, cfg)
            i = torch.arange(s_pad)[:, None]
            j = torch.arange(s_pad)[None, :]
            ok = (j < valid).expand(s_pad, s_pad)
            if self.causal:
                ok = ok & (j <= i)
            full = torch.where(ok, 0.0, host.NEG)
            sliding = torch.where(ok & ((i - j).abs() <= (self.sliding_window or 0)), 0.0, host.NEG) if self.sliding_window else full
            f = lambda m: dd._dev(self.dit.dev, m.to(torch.bfloat16)[None, None])
            if len(self._tables) >= 8:
                for old in list(self._tables.values())[0]:
                    self.ttnn.deallocate(old)
                self._tables.pop(next(iter(self._tables)))
            self._tables[key] = (f(cos), f(sin), f(full), f(sliding))
        return self._tables[key]

    def forward(self, x):
        torch, ttnn, dit = self.torch, self.ttnn, self.dit
        s = x.shape[0]
        s_pad = pad_to(s, self.bucket)
        cos, sin, full, sliding = self._tables_for(s_pad, s)
        xp = torch.zeros(s_pad, self.hidden)
        xp[:s] = x.float()
        h = self.dd._dev(dit.dev, xp.to(torch.bfloat16).reshape(1, 1, s_pad, self.hidden))
        qd, kvd = self.heads * self.head_dim, self.kv_heads * self.head_dim
        for i, L in enumerate(self.layers):
            n = dit._rms(h, L["in_norm"])
            qkv = self._mm(n, L["wqkv"], s_pad, self.hidden, qd + 2 * kvd)
            ttnn.deallocate(n)
            if len(qkv.shape) != 4:
                qkv = ttnn.reshape(qkv, [1, 1, s_pad, qkv.shape[-1]])
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=self.heads, num_kv_heads=self.kv_heads, transpose_k_heads=False, memory_config=self.dd.MEM)
            ttnn.deallocate(qkv)
            roped = []
            for t, w in ((q, L["q_norm"]), (k, L["k_norm"])):
                nt = dit._rms(t, w)
                ttnn.deallocate(t)
                roped.append(ttnn.experimental.rotary_embedding_llama(nt, cos, sin, self.trans_mat, is_decode_mode=False, compute_kernel_config=dit.ck_rope))
                ttnn.deallocate(nt)
            mask = sliding if self.layer_types[i] == "sliding_attention" else full
            chunk = next(c for c in dit.sdpa_chunks if s_pad % c == 0)
            cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=dit.grid, q_chunk_size=chunk, k_chunk_size=chunk, exp_approx_mode=False)
            a = ttnn.transformer.scaled_dot_product_attention(roped[0], roped[1], v, attn_mask=mask, is_causal=False, scale=self.head_dim ** -0.5,
                                                              program_config=cfg, compute_kernel_config=dit.ck_sdpa, memory_config=self.dd.MEM)
            for t in (*roped, v):
                ttnn.deallocate(t)
            a2 = ttnn.transformer.concatenate_heads(a, memory_config=self.dd.MEM)
            ttnn.deallocate(a)
            o = self._mm(a2, L["wo"], s_pad, qd, self.hidden)
            ttnn.deallocate(a2)
            h = self._add(h, o)
            n = dit._rms(h, L["post_norm"])
            m = self._mm(n, L["w_gateup"], s_pad, self.hidden, 2 * self.intermediate, fuse_swiglu=True)
            ttnn.deallocate(n)
            d = self._mm(m, L["w_down"], s_pad, self.intermediate, self.hidden)
            ttnn.deallocate(m)
            h = self._add(h, d)
        if self.final_norm is not None:
            n = dit._rms(h, self.final_norm)
            ttnn.deallocate(h)
            h = n
        out = ttnn.to_torch(h)[0, 0, :s].float()
        ttnn.deallocate(h)
        return out

    def _mm(self, x, w, M, K, N, fuse_swiglu=False):
        """`AceStepDiT._mm` with this stack's compute kernel config."""
        dit = self.dit
        key = (M, K, N)
        if key not in dit._mm_cfg_cache:
            dit._mm_cfg_cache[key] = self.dd.get_matmul_config(M, K, N, dit.grid)
        return self.ttnn.experimental.minimal_matmul(x, w, config=dit._mm_cfg_cache[key], compute_kernel_config=self.ck_mm, dtype=self.ttnn.bfloat16,
                                                     memory_config=self.dd.MEM, fuse_swiglu=fuse_swiglu)

    def _add(self, x, branch):
        out = self.ttnn.add(x, branch, memory_config=self.dd.MEM)
        self.ttnn.deallocate(branch)
        self.ttnn.deallocate(x)
        return out


def install_text_on_chip(worker, handler):
    """Replaces `handler.infer_text_embeddings` with a version that runs the Qwen3 text encoder on the chip (batch 1; other batches use the original CPU path).
    The weights are read from the handler's loaded text encoder, so the checkpoint is the one the CPU run uses. Call after `LocalWorker.start()` and the handler's init."""
    import torch

    dit = worker.model.dit
    te = handler.text_encoder
    cfg = te.config
    sd = {k: v for k, v in te.state_dict().items()}
    stack = ChipEncoderStack(
        dit, lambda name: sd[name], layers=cfg.num_hidden_layers, hidden=cfg.hidden_size, heads=cfg.num_attention_heads, kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim, intermediate=cfg.intermediate_size, eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta, causal=True, final_norm=sd["norm.weight"], fidelity="HiFi4")      # measured: relative L2 0.0187 at HiFi4, 0.0228 at HiFi2 (CPU bfloat16 floor 0.0182)
    original = handler.infer_text_embeddings

    def infer_text_embeddings(text_token_idss):
        if text_token_idss.dim() != 2 or text_token_idss.shape[0] != 1:
            return original(text_token_idss)
        with torch.inference_mode():
            emb = te.embed_tokens(text_token_idss)[0].float()
        out = stack.forward(emb)
        return out[None].to(emb.dtype)

    stack.forward(torch.zeros(64, cfg.hidden_size))       # compiles the kernels of the smallest bucket now, not on the first request
    handler.infer_text_embeddings = infer_text_embeddings
    handler.chip_text_stack = stack
    return stack


def install_lyric_on_chip(worker, handler):
    """Replaces `handler.model.encoder.lyric_encoder.forward` with a version that runs its 8 layers on the chip. Used only for batch 1 with every token valid (the
    lyric mask all ones, which is what a single request makes) and no explicit position ids (the stack builds zero-based rotary tables); anything else goes to the
    original CPU path."""
    import torch
    from transformers.modeling_outputs import BaseModelOutput

    dit = worker.model.dit
    le = handler.model.encoder.lyric_encoder
    cfg = handler.model.config
    sd = {k: v for k, v in le.state_dict().items()}
    n = cfg.num_lyric_encoder_hidden_layers
    stack = ChipEncoderStack(
        dit, lambda name: sd[name], layers=n, hidden=cfg.hidden_size, heads=cfg.num_attention_heads, kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        intermediate=cfg.intermediate_size, eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta, causal=False, layer_types=list(cfg.layer_types[:n]),
        sliding_window=cfg.sliding_window if cfg.use_sliding_window else None, final_norm=sd["norm.weight"],
        fidelity="HiFi2")      # measured on a 207-token lyric: relative L2 0.0125 at HiFi2, 0.0316 at HiFi4 (CPU bfloat16 0.186)
    original = le.forward

    def forward(input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, *a, **kw):
        ok = (inputs_embeds is not None and attention_mask is not None and not a and input_ids is None and position_ids is None and inputs_embeds.dim() == 3 and inputs_embeds.shape[0] == 1
              and bool(attention_mask.bool().all()) and not kw.get("output_hidden_states") and not kw.get("output_attentions"))
        if not ok:
            return original(input_ids, attention_mask, position_ids, inputs_embeds, *a, **kw)
        with torch.inference_mode():
            x = le.embed_tokens(inputs_embeds)[0].float()
        out = stack.forward(x)
        return BaseModelOutput(last_hidden_state=out[None].to(inputs_embeds.dtype))

    stack.forward(torch.zeros(64, cfg.hidden_size))
    le.forward = forward
    handler.chip_lyric_stack = stack
    return stack
