"""Granite-4.0-H-1B on one chip, static-shape step that can be traced (ttnn trace capture and replay).

Differences from granite_tt.py (the earlier, dynamic port):
  * every tensor the step reads or writes across steps is a persistent device buffer (conv state, SSM state, K/V cache);
    the step updates them in place with ttnn.copy, so the same program can be captured once and replayed;
  * the K/V cache has a fixed length MAXLEN; the new key and value are written at the current position with a one-hot
    multiply-add, and the attention scores get an additive mask for the positions not yet written;
  * per-token host inputs (embedding, one-hot position, mask) are written into persistent device tensors before each step;
  * the Mamba-2 SSD update has variants (see `ssd`), so the effect of the expanded input copies can be measured.

ssd variants
  composed         stock ttnn ops; readout is a batched matmul with C broadcast over heads (no ttnn.repeat anywhere)
  state_full       coder state-update kernel (ssd_state_update), all inputs expanded to [3072,128]; readout composed
  state_full_ro    the same, readout by the coder readout kernel (ssd_readout), which needs C expanded to [3072,128]
  bcast            coder compact state-update kernel (ssd_state_update_bcast): decay, dtx as [3072,1] (value in column 0 of the
                   tile), B as [1,128] (row 0 of the tile); readout composed
  bcast_ro         compact state update + readout kernel
expand: how the expanded inputs of the *_full variants are made: "repeat" (ttnn.repeat, as the earlier port) or "mul"
(multiply by a persistent tensor of ones, a broadcast multiply).
Glue (not coder code): everything in this file. The kernel files are loaded unchanged.
"""
import importlib.util
import sys
import time

import torch
import ttnn

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "port"))
import granite_ref as g
import granite_tt as gt

TILE = ttnn.TILE_LAYOUT
DRAM = ttnn.DRAM_MEMORY_CONFIG
MAXLEN = 128
NEG = -1.0e9


def load_kernel(path, fn):
    spec = importlib.util.spec_from_file_location("k_" + fn, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fn)


class TTModel2(gt.TTModel):
    def __init__(self, dev, cfg, w, ssd="composed", expand="mul", kernel_dir=None, pad=False):
        self.ssd, self.expand, self.pad, self.tid, self.out = ssd, expand, pad, None, None
        self.kdir = kernel_dir
        super().__init__(dev, cfg, w, ssd_mode="composed", kernels=None)   # loads weights and the conv/SSD constants
        self._extras()

    # ---- host-to-device helpers -------------------------------------------------------------------------------------------
    def host(self, t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=dtype, layout=TILE)

    def write(self, dev_t, t):
        ttnn.copy_host_to_device_tensor(self.host(t), dev_t)

    def _extras(self):
        H, P, N, L = self.H, self.P, self.N, MAXLEN
        self.ones_full = self.up(torch.ones(H, P, N))
        self.ones_col = self.up(torch.ones(H, P, 1))
        if self.ssd != "composed":
            if self.ssd.startswith("state_full") or self.ssd.startswith("bcast"):
                f = "ssd_state_update" if self.ssd.startswith("state_full") else "ssd_state_update_bcast"
                fn = "state.py" if self.ssd.startswith("state_full") else "bcast.py"
                self.state_k = load_kernel(f"{self.kdir}/{fn}", f)
            if self.ssd.endswith("_ro"):
                self.ro_k = load_kernel(f"{self.kdir}/readout.py", "ssd_readout")
            self.out_s = self.up(torch.zeros(H * P, N))
            self.out_y = self.up(torch.zeros(H * P, N))
        # persistent per-token inputs
        self.x_in = self.up(torch.zeros(1, 1, 1, self.hid))
        self.sel = self.up(torch.zeros(1, self.nkv, L, 1))
        self.keep = self.up(torch.ones(1, self.nkv, L, 1))
        self.mask = self.up(torch.zeros(1, self.nkv, self.nh // self.nkv, L))
        # persistent state
        self.cs, self.ss, self.kc, self.vc = {}, {}, {}, {}
        for i, t in enumerate(self.types):
            if t == "mamba":
                self.cs[i] = [self.up(torch.zeros(1, 1, 1, self.convdim)) for _ in range(3)]
                self.ss[i] = self.up(torch.zeros(H * P, N))
            else:
                self.kc[i] = self.up(torch.zeros(1, self.nkv, L, self.hd))
                self.vc[i] = self.up(torch.zeros(1, self.nkv, L, self.hd))
        self.reset()

    def reset(self):
        self.pos = 0
        if not hasattr(self, "cs"):
            return
        z = lambda *s: self.host(torch.zeros(*s))
        for i in self.cs:
            for b in self.cs[i]:
                ttnn.copy_host_to_device_tensor(z(1, 1, 1, self.convdim), b)
            ttnn.copy_host_to_device_tensor(z(self.H * self.P, self.N), self.ss[i])
        for i in self.kc:
            ttnn.copy_host_to_device_tensor(z(1, self.nkv, MAXLEN, self.hd), self.kc[i])
            ttnn.copy_host_to_device_tensor(z(1, self.nkv, MAXLEN, self.hd), self.vc[i])

    # ---- one step -----------------------------------------------------------------------------------------------------------
    def set_inputs(self, token):
        c, pos, L = self.cfg, self.pos, MAXLEN
        assert pos < L
        emb = (self.w["model.embed_tokens.weight"][token] * c["embedding_multiplier"]).view(1, 1, 1, self.hid)
        self.write(self.x_in, emb)
        sel = torch.zeros(1, self.nkv, L, 1)
        sel[:, :, pos] = 1
        self.write(self.sel, sel)
        self.write(self.keep, 1 - sel)
        m = torch.zeros(1, self.nkv, self.nh // self.nkv, L)
        m[..., pos + 1:] = NEG
        self.write(self.mask, m)

    def step(self, token, record=None):
        self.set_inputs(token)
        if self.tid is not None:
            ttnn.execute_trace(self.dev, self.tid, cq_id=0, blocking=True)
            out = self.out
        else:
            out = self.program()
        self.pos += 1
        return self.to_host(out, self.cfg["vocab_size"]) / self.cfg["logits_scaling"]

    def enable_trace(self):
        """Run the program once eagerly (compiles every op), then capture it. State is reset by the caller."""
        self.set_inputs(0)
        self.program()
        ttnn.synchronize_device(self.dev)
        self.tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        self.out = self.program()
        ttnn.end_trace_capture(self.dev, self.tid, cq_id=0)
        ttnn.synchronize_device(self.dev)
        self.reset()

    def program(self):
        w, c, rm = self.w, self.cfg, self.cfg["residual_multiplier"]
        x = self.x_in
        for i in range(self.L):
            L = self.L_[i]
            h = ttnn.rms_norm(x, weight=L["ln1"], epsilon=self.eps, compute_kernel_config=self.ck)
            m = self.mamba2(i, h) if self.types[i] == "mamba" else self.attention2(i, h)
            x = ttnn.add(x, ttnn.multiply(m, rm))
            h = ttnn.rms_norm(x, weight=L["ln2"], epsilon=self.eps, compute_kernel_config=self.ck)
            a = self.linear(h, L["mlp_in"])
            gt_ = ttnn.slice(a, [0, 0, 0, 0], [1, 1, 1, 4096])
            up_ = ttnn.slice(a, [0, 0, 0, 4096], [1, 1, 1, 8192])
            mlp = self.linear(ttnn.multiply(ttnn.silu(gt_), up_), L["mlp_out"])
            x = ttnn.add(x, ttnn.multiply(mlp, rm))
        x = ttnn.rms_norm(x, weight=self.d["norm"], epsilon=self.eps, compute_kernel_config=self.ck)
        return self.linear(x, self.d["embed_t"])

    # ---- attention with a fixed-length cache --------------------------------------------------------------------------------
    def attention2(self, i, h):
        L = self.L_[i]
        q = ttnn.reshape(self.linear(h, L["q"]), (1, self.nkv, self.nh // self.nkv, self.hd))
        k = ttnn.reshape(self.linear(h, L["k"]), (1, self.nkv, 1, self.hd))
        v = ttnn.reshape(self.linear(h, L["v"]), (1, self.nkv, 1, self.hd))
        # write k, v at the current position: cache = cache * keep + sel (x) k   (sel is one-hot over the length axis)
        Kn = ttnn.add(ttnn.multiply(self.kc[i], self.keep), ttnn.matmul(self.sel, k, compute_kernel_config=self.ck))
        Vn = ttnn.add(ttnn.multiply(self.vc[i], self.keep), ttnn.matmul(self.sel, v, compute_kernel_config=self.ck))
        ttnn.copy(Kn, self.kc[i])
        ttnn.copy(Vn, self.vc[i])
        s = ttnn.multiply(ttnn.matmul(q, ttnn.transpose(Kn, -2, -1), compute_kernel_config=self.ck), self.cfg["attention_multiplier"])
        p = ttnn.softmax(ttnn.add(s, self.mask), dim=-1)
        o = ttnn.matmul(p, Vn, compute_kernel_config=self.ck)
        return self.linear(ttnn.reshape(o, (1, 1, 1, self.hid)), L["o"])

    # ---- Mamba-2 mixer with persistent state -------------------------------------------------------------------------------
    def expand_full(self, t, shape):
        """t broadcast to [48,64,128] (shape is the repeat factor for the ttnn.repeat form)."""
        if self.expand == "repeat":
            return ttnn.reshape(ttnn.repeat(t, shape), (self.H * self.P, self.N))
        return ttnn.reshape(ttnn.multiply(t, self.ones_full), (self.H * self.P, self.N))

    def mamba2(self, i, h):
        L, H, P, N = self.L_[i], self.H, self.P, self.N
        proj = self.linear(h, L["in_proj"])
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
        dt = ttnn.reshape(ttnn.softplus(ttnn.add(dtr, L["dtb"])), (H, 1, 1))
        decay = ttnn.exp(ttnn.multiply(dt, L["A"]))                   # [48,1,1]
        dtx = ttnn.multiply(dt, x)                                    # [48,64,1]
        S = self.ss[i]                                                # [3072,128] persistent
        if self.ssd == "composed":
            S3 = ttnn.reshape(S, (H, P, N))
            Sn = ttnn.add(ttnn.multiply(S3, decay), ttnn.multiply(dtx, ttnn.reshape(B, (1, 1, N))))
            ttnn.copy(ttnn.reshape(Sn, (H * P, N)), S)
            y = ttnn.matmul(Sn, ttnn.reshape(C, (1, N, 1)), compute_kernel_config=self.ck)   # [48,64,1]
        else:
            if self.ssd.startswith("state_full"):
                dec = self.expand_full(decay, (1, P, N))
                dx = self.expand_full(dtx, (1, 1, N))
                b = self.expand_full(ttnn.reshape(B, (1, 1, N)), (H, P, 1))
            else:                                                     # bcast: compact inputs
                dec = ttnn.reshape(ttnn.multiply(decay, self.ones_col), (H * P, 1))
                dx = ttnn.reshape(dtx, (H * P, 1))
                b = ttnn.reshape(B, (1, N))
                if self.pad:
                    dec, dx = ttnn.pad(dec, [(0, 0), (0, 31)], 0), ttnn.pad(dx, [(0, 0), (0, 31)], 0)
                    b = ttnn.pad(b, [(0, 31), (0, 0)], 0)
            self.state_k(S, dec, dx, b, self.out_s)
            ttnn.copy(self.out_s, S)
            if self.ssd.endswith("_ro"):
                cf = self.expand_full(ttnn.reshape(C, (1, 1, N)), (H, P, 1))
                self.ro_k(cf, self.out_s, self.out_y)
                y = ttnn.reshape(ttnn.slice(self.out_y, [0, 0], [H * P, 1]), (H, P, 1))
            else:
                y = ttnn.matmul(ttnn.reshape(self.out_s, (H, P, N)), ttnn.reshape(C, (1, N, 1)), compute_kernel_config=self.ck)
        y = ttnn.add(y, ttnn.multiply(L["D"], x))
        z = ttnn.multiply(ttnn.reshape(y, (1, 1, 1, self.inter)), ttnn.silu(gate))
        z = ttnn.rms_norm(z, weight=L["gnorm"], epsilon=self.eps, compute_kernel_config=self.ck)
        return self.linear(z, L["out_proj"])
