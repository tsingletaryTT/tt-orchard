# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""ibm-granite/granite-4.0-h-1b (a Mamba-2 and attention hybrid) on one Blackhole chip with ttnn, one token per step.

The decode step is a fixed-shape program that is captured once as a ttnn trace and replayed for every token. For that to work:

  * the conv state, the Mamba-2 state and the key/value caches are persistent device buffers, updated in place with `ttnn.copy`;
  * the key/value cache has a fixed length (`max_len`). The new key and value are written at the current position with
    `cache * keep + onehot (x) new`, and the attention scores get an additive mask for the positions that are not written yet;
  * each step the host writes four small tensors (the embedding, the one-hot position, its complement, the mask) into persistent
    device buffers and replays the trace.

Prefill is a loop of these decode steps (one token per step). `prefill_chunked` is a chunked-scan prefill (composed mode only, not used by
the server). Its algebra is checked on the CPU (tests/test_granite_scan.py). With trace capture off it agrees with the step path at 33 to 1,024
tokens (logits cosine above 0.99997, same top-1 token, same next four greedy tokens) and is 3 to 30 times faster than the eager step loop
(128 tokens 88 against 28 tok/s, 1,024 tokens 552 against 17 to 24 tok/s; evidence/granite-chunked-bench). With trace capture on, the single
timed run (512 tokens, after a finished warm-up) did not finish and left a chip unreadable. The chunked path allocates and frees device
buffers while a decode trace is alive, and ttnn warns that this is unsafe. The path therefore stays off by default and the server does
not call it.

`ssd` picks how the Mamba-2 state update and readout are computed:
  composed (default)  stock ttnn ops. The readout is one batched matmul with C broadcast over the heads.
  state_full, state_full_ro, bcast, bcast_ro   optional: the tt-lang kernels written by a local coder model (kernels/), see the card.
                      They need `ttl` (tt-lang) importable and `kernel_dir` pointing at the kernels. Measured speeds are in the card.

`ssd_precision` (default fp32) keeps dt, the decay and the Mamba-2 state in float32 on the chip. With bf16 the state update rounds the decay
(a value close to 1) and the state to 8 bits of mantissa at every step, and the error grows with the sequence length (see the card).
The kernel modes run in bf16 only.

Everything the model needs from the weights directory is `config.json` and `model.safetensors`.
"""
import importlib.util
import json
import time
from pathlib import Path

import torch
import ttnn

TILE = ttnn.TILE_LAYOUT
DRAM = ttnn.DRAM_MEMORY_CONFIG
NEG = -1.0e9
SSD_MODES = ("composed", "state_full", "state_full_ro", "bcast", "bcast_ro")


def _load_kernel(path, fn):
    spec = importlib.util.spec_from_file_location("k_" + fn, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)


class GraniteChip:
    def __init__(self, dev, model_dir, max_len=2048, ssd="composed", kernel_dir=None, trace=True, ssd_precision="fp32", fast_prefill=True):
        from safetensors import safe_open  # imported here so the module can be read without the package
        if ssd not in SSD_MODES:
            raise ValueError(f"ssd must be one of {SSD_MODES}")
        if max_len % 32:
            raise ValueError("max_len must be a multiple of 32 (the tile size)")
        if ssd_precision not in ("fp32", "bf16"):
            raise ValueError("ssd_precision must be fp32 or bf16")
        ssd_precision = self.effective_precision(ssd, ssd_precision)
        self.dev, self.L, self.ssd, self.use_trace, self.tid, self.out = dev, max_len, ssd, trace, None, None
        self.fast_prefill, self.tid_nohead = fast_prefill, None
        self.f32 = ssd_precision == "fp32"                               # dt, decay and the state in float32
        model_dir = Path(model_dir)
        cfg = json.load(open(model_dir / "config.json"))
        self.cfg = cfg
        self.NL, self.types = cfg["num_hidden_layers"], cfg["layer_types"]
        self.H, self.P, self.N = cfg["mamba_n_heads"], cfg["mamba_d_head"], cfg["mamba_d_state"]
        self.inter, self.convdim = self.H * self.P, self.H * self.P + 2 * self.N
        self.eps = cfg["rms_norm_eps"]
        self.nh, self.nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hid = cfg["hidden_size"]
        self.hd = self.hid // self.nh
        self.vocab = cfg["vocab_size"]
        self.ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                        fp32_dest_acc_en=True, packer_l1_acc=True)
        w = {}
        with safe_open(str(model_dir / "model.safetensors"), "pt") as f:
            for k in f.keys():
                w[k] = f.get_tensor(k).to(torch.bfloat16)
        self.embed = w["model.embed_tokens.weight"]                     # kept on the host: the embedding lookup is a row read
        self._load(w)
        del w
        self.kernels = {}
        if ssd != "composed":
            kd = Path(kernel_dir)
            if ssd.startswith("state_full"):
                self.kernels["state"] = _load_kernel(kd / "state.py", "ssd_state_update")
            else:
                self.kernels["state"] = _load_kernel(kd / "bcast.py", "ssd_state_update_bcast")
            if ssd.endswith("_ro"):
                self.kernels["readout"] = _load_kernel(kd / "readout.py", "ssd_readout")
            self.out_s = self.up(torch.zeros(self.H * self.P, self.N))
            self.out_y = self.up(torch.zeros(self.H * self.P, self.N))
            self.ones_full = self.up(torch.ones(self.H, self.P, self.N))
            self.ones_col = self.up(torch.ones(self.H, self.P, 1))
        self._buffers()
        self.pos = 0
        if trace:
            self.enable_trace()

    @staticmethod
    def effective_precision(ssd, requested):
        """The kernel modes run with a bf16 state whatever was requested; the composed path honours the request."""
        return requested if ssd == "composed" else "bf16"

    # ---- weights ------------------------------------------------------------------------------------------------------------
    def up(self, t, layout=TILE):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=self.dev, memory_config=DRAM)

    def up32(self, t):
        return ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=TILE, device=self.dev, memory_config=DRAM)

    def host32(self, t):
        return ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=TILE)

    def host(self, t):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=TILE)

    def _lin(self, w, name):
        t = w[name]
        return self.up(t.t().contiguous().view(1, 1, t.shape[1], t.shape[0]))

    def _vec(self, w, name, shape):
        return self.up(w[name].view(*shape))

    def _load(self, w):
        self.embed_t = self._lin(w, "model.embed_tokens.weight")        # the tied lm_head, [hidden, vocab]
        self.norm = self._vec(w, "model.norm.weight", (1, 1, 1, self.hid))
        self.Ls = []
        for i in range(self.NL):
            p = f"model.layers.{i}."
            L = {"ln1": self._vec(w, p + "input_layernorm.weight", (1, 1, 1, self.hid)),
                 "ln2": self._vec(w, p + "post_attention_layernorm.weight", (1, 1, 1, self.hid)),
                 "mlp_in": self._lin(w, p + "shared_mlp.input_linear.weight"), "mlp_out": self._lin(w, p + "shared_mlp.output_linear.weight")}
            if self.types[i] == "mamba":
                m = p + "mamba."
                L["in_proj"], L["out_proj"] = self._lin(w, m + "in_proj.weight"), self._lin(w, m + "out_proj.weight")
                L["gnorm"] = self._vec(w, m + "norm.weight", (1, 1, 1, self.inter))
                cw = w[m + "conv1d.weight"][:, 0, :]                    # [convdim, 4]
                L["cw"] = [self.up(cw[:, j].contiguous().view(1, 1, 1, self.convdim)) for j in range(4)]
                L["cb"] = self._vec(w, m + "conv1d.bias", (1, 1, 1, self.convdim))
                A = (-torch.exp(w[m + "A_log"].float())).view(self.H, 1, 1)
                L["A"] = self.up32(A) if self.f32 else self.up(A)
                Ar = A.view(1, 1, 1, self.H)                            # the same values as a row, for the chunked scan
                L["Ar"] = self.up32(Ar) if self.f32 else self.up(Ar)
                dtb, D = w[m + "dt_bias"].float().view(1, 1, 1, self.H), w[m + "D"].float().view(self.H, 1, 1)
                L["dtb"], L["D"] = (self.up32(dtb), self.up32(D)) if self.f32 else (self.up(dtb), self.up(D))
                Dr = D.view(self.H, 1).expand(self.H, self.P).reshape(1, 1, 1, self.inter)      # D repeated over the head dimension
                L["Dr"] = self.up32(Dr) if self.f32 else self.up(Dr)
            else:
                a = p + "self_attn."
                L["q"], L["k"], L["v"], L["o"] = (self._lin(w, a + n + "_proj.weight") for n in "qkvo")
            self.Ls.append(L)

    def _buffers(self):
        """Persistent device buffers: per-token inputs, and the state that the step updates in place."""
        L = self.L
        self.x_in = self.up(torch.zeros(1, 1, 1, self.hid))
        self.sel = self.up(torch.zeros(1, self.nkv, L, 1))
        self.keep = self.up(torch.ones(1, self.nkv, L, 1))
        self.mask = self.up(torch.zeros(1, self.nkv, self.nh // self.nkv, L))
        self.cs, self.ss, self.kc, self.vc = {}, {}, {}, {}
        for i, t in enumerate(self.types):
            if t == "mamba":
                self.cs[i] = [self.up(torch.zeros(1, 1, 1, self.convdim)) for _ in range(3)]
                self.ss[i] = (self.up32 if self.f32 else self.up)(torch.zeros(self.H * self.P, self.N))
            else:
                self.kc[i] = self.up(torch.zeros(1, self.nkv, L, self.hd))
                self.vc[i] = self.up(torch.zeros(1, self.nkv, L, self.hd))

    def reset(self):
        """Start a new sequence. The recurrent state is zeroed. The key/value caches are not: positions that are not written yet are
        masked, and a stale value that is masked contributes exactly zero."""
        self.pos = 0
        z = lambda *s: self.host(torch.zeros(*s))  # noqa: E731
        for i in self.cs:
            for b in self.cs[i]:
                ttnn.copy_host_to_device_tensor(z(1, 1, 1, self.convdim), b)
            zs = torch.zeros(self.H * self.P, self.N)
            ttnn.copy_host_to_device_tensor(self.host32(zs) if self.f32 else self.host(zs), self.ss[i])

    # ---- stepping -----------------------------------------------------------------------------------------------------------
    def _set_inputs(self, token):
        pos, L = self.pos, self.L
        if pos >= L:
            raise ValueError(f"position {pos} is past the context limit {L}")
        emb = (self.embed[token].float() * self.cfg["embedding_multiplier"]).view(1, 1, 1, self.hid)
        ttnn.copy_host_to_device_tensor(self.host(emb), self.x_in)
        sel = torch.zeros(1, self.nkv, L, 1)
        sel[:, :, pos] = 1
        ttnn.copy_host_to_device_tensor(self.host(sel), self.sel)
        ttnn.copy_host_to_device_tensor(self.host(1 - sel), self.keep)
        m = torch.zeros(1, self.nkv, self.nh // self.nkv, L)
        m[..., pos + 1:] = NEG
        ttnn.copy_host_to_device_tensor(self.host(m), self.mask)

    def _run(self, read, head=True):
        out = None
        if self.tid is not None:
            tid = self.tid if head or self.tid_nohead is None else self.tid_nohead
            ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=read)
            out = self.out
        else:
            out = self.program(head)
        self.pos += 1
        if not read:
            return None
        return ttnn.to_torch(out).float().reshape(-1)[: self.vocab] / self.cfg["logits_scaling"]

    def step(self, token):
        """One token in, the logits of the next token out (a float tensor of the vocabulary size)."""
        self._set_inputs(token)
        return self._run(True)

    def prefill(self, ids):
        """A loop of decode steps. The logits are read back only after the last prompt token."""
        for t in ids[:-1]:
            self._set_inputs(t)
            self._run(False, head=False)                    # a second trace that stops after the last layer: no final norm, no lm_head
        return self.step(ids[-1])

    # ---- chunked prefill (candidate) ------------------------------------------------------------------------------------------
    CHUNKS = (128, 64, 32)

    def prefill_chunked(self, ids, chunk=128):
        """Prefill with blocks of tokens instead of one token per step. Needs the composed mode.

        Blocks are multiples of 32 (the tile size), at most `chunk` tokens, largest first. The last token and the tokens that do not fill a
        block of 32 go through the decode step, so the logits come from the same path as before. The recurrent state, the conv state
        and the key/value caches are the buffers the decode trace uses, so decode continues from them unchanged.
        """
        if self.ssd != "composed":
            raise ValueError("prefill_chunked needs ssd='composed'")
        sizes = [c for c in self.CHUNKS if c <= chunk] or [32]
        body, i = ids[:-1], 0
        while len(body) - i >= sizes[-1]:
            T = next(c for c in sizes if c <= len(body) - i)
            if self.pos + T > self.L:
                raise ValueError(f"position {self.pos + T} is past the context limit {self.L}")
            self._run_chunk(body[i:i + T])
            i += T
        for t in body[i:]:
            self._set_inputs(t)
            self._run(False, head=False)
        return self.step(ids[-1])

    def _rm(self, t):
        return ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)

    def _tl(self, t):
        return ttnn.to_layout(t, TILE)

    def _tri(self, T):
        """[1,1,T,T] lower-triangular ones (cumulative sum by matmul) and the additive mask for the upper triangle, in float32."""
        if not hasattr(self, "_tri_cache"):
            self._tri_cache = {}
        if T not in self._tri_cache:
            lo = torch.tril(torch.ones(T, T))
            self._tri_cache[T] = (self.up32(lo.view(1, 1, T, T)), self.up32(((1 - lo) * NEG).view(1, 1, T, T)))
        return self._tri_cache[T]

    def _run_chunk(self, toks):
        T, pos0, L, r = len(toks), self.pos, self.L, self.nh // self.nkv
        emb = (self.embed[torch.tensor(toks)].float() * self.cfg["embedding_multiplier"]).view(1, 1, T, self.hid)
        x = self.up(emb)
        sel = torch.zeros(1, self.nkv, L, T)
        sel[:, :, pos0 + torch.arange(T), torch.arange(T)] = 1
        keep = (1 - sel.sum(-1, keepdim=True))
        mask = torch.zeros(1, 1, T * r, L)
        for t in range(T):
            mask[0, 0, t * r:(t + 1) * r, pos0 + t + 1:] = NEG
        att = {"sel": self.up(sel), "keep": self.up(keep), "mask": self.up(mask)}
        rm = self.cfg["residual_multiplier"]
        for i in range(self.NL):
            Ly = self.Ls[i]
            h = ttnn.rms_norm(x, weight=Ly["ln1"], epsilon=self.eps, compute_kernel_config=self.ck)
            m = self._mamba_chunk(i, h, T) if self.types[i] == "mamba" else self._attention_chunk(i, h, T, att)
            x = ttnn.add(x, ttnn.multiply(m, rm))
            h = ttnn.rms_norm(x, weight=Ly["ln2"], epsilon=self.eps, compute_kernel_config=self.ck)
            a = self._linear(h, Ly["mlp_in"])
            gate = ttnn.slice(a, [0, 0, 0, 0], [1, 1, T, 4096])
            up = ttnn.slice(a, [0, 0, 0, 4096], [1, 1, T, 8192])
            x = ttnn.add(x, ttnn.multiply(self._linear(ttnn.multiply(ttnn.silu(gate), up), Ly["mlp_out"]), rm))
        ttnn.synchronize_device(self.dev)
        self.pos += T

    def _attention_chunk(self, i, h, T, att):
        """Causal attention of T new tokens over the cache. The new keys and values are written at positions pos .. pos+T-1 with
        cache * keep + sel (x) new, as the decode step does for one."""
        Ly, r, nkv, hd = self.Ls[i], self.nh // self.nkv, self.nkv, self.hd
        q = self._rm(self._linear(h, Ly["q"]))
        k = self._rm(self._linear(h, Ly["k"]))
        v = self._rm(self._linear(h, Ly["v"]))
        q = self._tl(ttnn.reshape(ttnn.permute(ttnn.reshape(q, (1, T, nkv, r * hd)), (0, 2, 1, 3)), (1, nkv, T * r, hd)))
        k = self._tl(ttnn.permute(ttnn.reshape(k, (1, T, nkv, hd)), (0, 2, 1, 3)))
        v = self._tl(ttnn.permute(ttnn.reshape(v, (1, T, nkv, hd)), (0, 2, 1, 3)))
        Kn = ttnn.add(ttnn.multiply(self.kc[i], att["keep"]), ttnn.matmul(att["sel"], k, compute_kernel_config=self.ck))
        Vn = ttnn.add(ttnn.multiply(self.vc[i], att["keep"]), ttnn.matmul(att["sel"], v, compute_kernel_config=self.ck))
        ttnn.copy(Kn, self.kc[i])
        ttnn.copy(Vn, self.vc[i])
        s = ttnn.multiply(ttnn.matmul(q, ttnn.transpose(Kn, -2, -1), compute_kernel_config=self.ck), self.cfg["attention_multiplier"])
        p = ttnn.softmax(ttnn.add(s, att["mask"]), dim=-1)
        o = self._rm(ttnn.matmul(p, Vn, compute_kernel_config=self.ck))
        o = ttnn.reshape(ttnn.permute(ttnn.reshape(o, (1, nkv, T, r * hd)), (0, 2, 1, 3)), (1, 1, T, self.hid))
        return self._linear(self._tl(o), Ly["o"])

    def _mamba_chunk(self, i, h, T):
        """The chunked scan of granite_scan.ssd_chunk with ttnn ops (see that file for the algebra)."""
        Ly, H, P, N = self.Ls[i], self.H, self.P, self.N
        proj = self._linear(h, Ly["in_proj"])
        gate = ttnn.slice(proj, [0, 0, 0, 0], [1, 1, T, self.inter])
        xbc = ttnn.slice(proj, [0, 0, 0, self.inter], [1, 1, T, self.inter + self.convdim])
        dtr = ttnn.slice(proj, [0, 0, 0, self.inter + self.convdim], [1, 1, T, self.inter + self.convdim + H])
        # causal conv over [three saved rows, T new rows]
        pad = ttnn.concat([self._rm(b) for b in self.cs[i]] + [self._rm(xbc)], dim=2)          # [1,1,T+3,convdim]
        taps = [self._tl(ttnn.slice(pad, [0, 0, j, 0], [1, 1, j + T, self.convdim])) for j in range(4)]
        c = ttnn.multiply(taps[0], Ly["cw"][0])
        for j in (1, 2, 3):
            c = ttnn.add(c, ttnn.multiply(taps[j], Ly["cw"][j]))
        c = ttnn.silu(ttnn.add(c, Ly["cb"]))
        for j, b in enumerate(self.cs[i]):                                                      # the last three inputs become the conv state
            ttnn.copy(self._tl(ttnn.slice(pad, [0, 0, T + j, 0], [1, 1, T + j + 1, self.convdim])), b)
        f = (lambda t: ttnn.typecast(t, ttnn.float32)) if self.f32 else (lambda t: t)  # noqa: E731
        x = f(ttnn.slice(c, [0, 0, 0, 0], [1, 1, T, self.inter]))                               # [1,1,T,H*P]
        B = f(ttnn.slice(c, [0, 0, 0, self.inter], [1, 1, T, self.inter + N]))                  # [1,1,T,N]
        C = f(ttnn.slice(c, [0, 0, 0, self.inter + N], [1, 1, T, self.convdim]))
        dt = ttnn.softplus(ttnn.add(f(dtr), Ly["dtb"]))                                         # [1,1,T,H]
        tri, negm = self._tri(T)
        if not self.f32:
            tri, negm = ttnn.typecast(tri, ttnn.bfloat16), ttnn.typecast(negm, ttnn.bfloat16)
        cs = ttnn.matmul(tri, ttnn.multiply(dt, Ly["Ar"]), compute_kernel_config=self.ck)       # inclusive cumulative sum of dt*A: [1,1,T,H]
        csh = ttnn.transpose(cs, -2, -1)                                                        # [1,1,H,T]
        diff = ttnn.subtract(ttnn.reshape(csh, (1, H, T, 1)), ttnn.reshape(csh, (1, H, 1, T)))  # cs_i - cs_j
        Lm = ttnn.exp(ttnn.add(diff, negm))                                                     # [1,H,T,T], zero above the diagonal
        M = ttnn.multiply(Lm, ttnn.matmul(C, ttnn.transpose(B, -2, -1), compute_kernel_config=self.ck))
        xh = ttnn.permute(ttnn.reshape(self._rm(x), (1, T, H, P)), (0, 2, 1, 3))                # [1,H,T,P]
        xh = self._tl(xh)
        dth = ttnn.reshape(ttnn.transpose(dt, -2, -1), (1, H, T, 1))
        dtx = ttnn.multiply(xh, dth)
        S = self.ss[i]
        y_in = ttnn.matmul(M, dtx, compute_kernel_config=self.ck)                               # [1,H,T,P]
        y_st = ttnn.matmul(C, ttnn.transpose(S, -2, -1), compute_kernel_config=self.ck)         # [1,1,T,H*P]: C . S0 for every head
        y_st = self._tl(ttnn.permute(ttnn.reshape(self._rm(y_st), (1, T, H, P)), (0, 2, 1, 3)))
        y = ttnn.add(y_in, ttnn.multiply(y_st, ttnn.reshape(ttnn.exp(csh), (1, H, T, 1))))
        y = ttnn.reshape(self._rm(ttnn.permute(self._rm(y), (0, 2, 1, 3))), (1, 1, T, self.inter))
        y = ttnn.add(self._tl(y), ttnn.multiply(Ly["Dr"], x))
        # state after the block
        csT = ttnn.slice(csh, [0, 0, 0, T - 1], [1, 1, H, T])                                   # [1,1,H,1]
        wgt = ttnn.multiply(ttnn.exp(ttnn.subtract(ttnn.reshape(csT, (1, H, 1, 1)), ttnn.reshape(csh, (1, H, T, 1)))), dtx)
        wT = self._tl(ttnn.reshape(self._rm(ttnn.permute(self._rm(wgt), (0, 1, 3, 2))), (1, 1, H * P, T)))
        add = ttnn.matmul(wT, B, compute_kernel_config=self.ck)                                 # [1,1,H*P,N]
        Sd = ttnn.multiply(ttnn.reshape(S, (1, H, P, N)), ttnn.reshape(ttnn.exp(csT), (1, H, 1, 1)))
        ttnn.copy(ttnn.reshape(ttnn.add(ttnn.reshape(Sd, (1, 1, H * P, N)), add), (H * P, N)), S)
        if self.f32:
            y = ttnn.typecast(y, ttnn.bfloat16)
        z = ttnn.multiply(y, ttnn.silu(gate))
        z = ttnn.rms_norm(z, weight=Ly["gnorm"], epsilon=self.eps, compute_kernel_config=self.ck)
        return self._linear(z, Ly["out_proj"])

    def enable_trace(self):
        """Run the program once eagerly (this compiles every op), then capture it."""
        self.tid = None
        self._set_inputs(0)
        self.program()
        ttnn.synchronize_device(self.dev)
        tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        self.out = self.program()
        ttnn.end_trace_capture(self.dev, tid, cq_id=0)
        ttnn.synchronize_device(self.dev)
        self.tid = tid
        if self.fast_prefill:
            self.program(head=False)                        # compile the shorter program (every op in it already exists, so this is quick)
            ttnn.synchronize_device(self.dev)
            self.tid_nohead = ttnn.begin_trace_capture(self.dev, cq_id=0)
            self.program(head=False)
            ttnn.end_trace_capture(self.dev, self.tid_nohead, cq_id=0)
            ttnn.synchronize_device(self.dev)
        self.reset()

    def close(self):
        for name in ("tid", "tid_nohead"):
            if getattr(self, name) is not None:
                ttnn.release_trace(self.dev, getattr(self, name))
                setattr(self, name, None)

    # ---- the step program ---------------------------------------------------------------------------------------------------
    def program(self, head=True):
        rm = self.cfg["residual_multiplier"]
        x = self.x_in
        for i in range(self.NL):
            L = self.Ls[i]
            h = ttnn.rms_norm(x, weight=L["ln1"], epsilon=self.eps, compute_kernel_config=self.ck)
            m = self._mamba(i, h) if self.types[i] == "mamba" else self._attention(i, h)
            x = ttnn.add(x, ttnn.multiply(m, rm))
            h = ttnn.rms_norm(x, weight=L["ln2"], epsilon=self.eps, compute_kernel_config=self.ck)
            a = self._linear(h, L["mlp_in"])
            gate = ttnn.slice(a, [0, 0, 0, 0], [1, 1, 1, 4096])
            up = ttnn.slice(a, [0, 0, 0, 4096], [1, 1, 1, 8192])
            x = ttnn.add(x, ttnn.multiply(self._linear(ttnn.multiply(ttnn.silu(gate), up), L["mlp_out"]), rm))
        if not head:
            return x
        x = ttnn.rms_norm(x, weight=self.norm, epsilon=self.eps, compute_kernel_config=self.ck)
        return self._linear(x, self.embed_t)

    def _linear(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=self.ck, memory_config=DRAM)

    def _attention(self, i, h):
        L = self.Ls[i]
        q = ttnn.reshape(self._linear(h, L["q"]), (1, self.nkv, self.nh // self.nkv, self.hd))
        k = ttnn.reshape(self._linear(h, L["k"]), (1, self.nkv, 1, self.hd))
        v = ttnn.reshape(self._linear(h, L["v"]), (1, self.nkv, 1, self.hd))
        Kn = ttnn.add(ttnn.multiply(self.kc[i], self.keep), ttnn.matmul(self.sel, k, compute_kernel_config=self.ck))
        Vn = ttnn.add(ttnn.multiply(self.vc[i], self.keep), ttnn.matmul(self.sel, v, compute_kernel_config=self.ck))
        ttnn.copy(Kn, self.kc[i])
        ttnn.copy(Vn, self.vc[i])
        s = ttnn.multiply(ttnn.matmul(q, ttnn.transpose(Kn, -2, -1), compute_kernel_config=self.ck), self.cfg["attention_multiplier"])
        p = ttnn.softmax(ttnn.add(s, self.mask), dim=-1)
        o = ttnn.matmul(p, Vn, compute_kernel_config=self.ck)
        return self._linear(ttnn.reshape(o, (1, 1, 1, self.hid)), L["o"])

    def _expand_full(self, t):
        """t broadcast to [rows, 128] by a multiply with a resident tensor of ones (no ttnn.repeat)."""
        return ttnn.reshape(ttnn.multiply(t, self.ones_full), (self.H * self.P, self.N))

    def _mamba(self, i, h):
        L, H, P, N = self.Ls[i], self.H, self.P, self.N
        proj = self._linear(h, L["in_proj"])
        gate = ttnn.slice(proj, [0, 0, 0, 0], [1, 1, 1, self.inter])
        xbc = ttnn.slice(proj, [0, 0, 0, self.inter], [1, 1, 1, self.inter + self.convdim])
        dtr = ttnn.slice(proj, [0, 0, 0, self.inter + self.convdim], [1, 1, 1, self.inter + self.convdim + H])
        s0, s1, s2 = self.cs[i]
        c = ttnn.add(ttnn.multiply(s0, L["cw"][0]), ttnn.multiply(s1, L["cw"][1]))
        c = ttnn.add(c, ttnn.multiply(s2, L["cw"][2]))
        c = ttnn.add(c, ttnn.multiply(xbc, L["cw"][3]))
        c = ttnn.silu(ttnn.add(c, L["cb"]))
        ttnn.copy(s1, s0)
        ttnn.copy(s2, s1)
        ttnn.copy(xbc, s2)
        x = ttnn.reshape(ttnn.slice(c, [0, 0, 0, 0], [1, 1, 1, self.inter]), (H, P, 1))
        B = ttnn.slice(c, [0, 0, 0, self.inter], [1, 1, 1, self.inter + N])
        C = ttnn.slice(c, [0, 0, 0, self.inter + N], [1, 1, 1, self.convdim])
        if self.f32:                                                    # dt, the decay and the state in float32: the decay is close to 1,
            f = lambda t: ttnn.typecast(t, ttnn.float32)  # noqa: E731  # which bf16 cannot hold to the precision a long sequence needs
            x, dtr = f(x), f(dtr)
        dt = ttnn.reshape(ttnn.softplus(ttnn.add(dtr, L["dtb"])), (H, 1, 1))
        decay = ttnn.exp(ttnn.multiply(dt, L["A"]))                     # [48,1,1]
        dtx = ttnn.multiply(dt, x)                                      # [48,64,1]
        S = self.ss[i]                                                  # [3072,128], persistent
        if self.ssd == "composed":
            Bv, Cv = ttnn.reshape(B, (1, 1, N)), ttnn.reshape(C, (1, N, 1))
            if self.f32:
                Bv, Cv = f(Bv), f(Cv)
            Sn = ttnn.add(ttnn.multiply(ttnn.reshape(S, (H, P, N)), decay), ttnn.multiply(dtx, Bv))
            ttnn.copy(ttnn.reshape(Sn, (H * P, N)), S)
            y = ttnn.matmul(Sn, Cv, compute_kernel_config=self.ck)
            y = ttnn.add(y, ttnn.multiply(L["D"], x))
            if self.f32:
                y = ttnn.typecast(y, ttnn.bfloat16)
        else:
            if self.ssd.startswith("state_full"):
                dec, dx, b = (self._expand_full(t) for t in (decay, dtx, ttnn.reshape(B, (1, 1, N))))
            else:                                                       # compact inputs for the bcast kernel
                dec = ttnn.reshape(ttnn.multiply(decay, self.ones_col), (H * P, 1))
                dx = ttnn.reshape(dtx, (H * P, 1))
                b = ttnn.reshape(B, (1, N))
            self.kernels["state"](S, dec, dx, b, self.out_s)
            ttnn.copy(self.out_s, S)
            if self.ssd.endswith("_ro"):
                self.kernels["readout"](self._expand_full(ttnn.reshape(C, (1, 1, N))), self.out_s, self.out_y)
                y = ttnn.reshape(ttnn.slice(self.out_y, [0, 0], [H * P, 1]), (H, P, 1))
            else:
                y = ttnn.matmul(ttnn.reshape(self.out_s, (H, P, N)), ttnn.reshape(C, (1, N, 1)), compute_kernel_config=self.ck)
            y = ttnn.add(y, ttnn.multiply(L["D"], x))
        z = ttnn.multiply(ttnn.reshape(y, (1, 1, 1, self.inter)), ttnn.silu(gate))
        z = ttnn.rms_norm(z, weight=L["gnorm"], epsilon=self.eps, compute_kernel_config=self.ck)
        return self._linear(z, L["out_proj"])
