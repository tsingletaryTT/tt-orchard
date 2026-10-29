"""Granite-4.0-H-1B on one Blackhole chip with ttnn, one token per step (prefill is a loop of steps).

ssd_mode picks where the Mamba-2 short convolution and state update run:
  host      conv + SSD state update + readout in torch on the host (the missing op as a fallback)  [step b]
  composed  the same, written from stock ttnn ops on the chip (matmul, multiply, add, softplus, exp)   [baseline]
  kernel    the state update and readout are tt-lang kernels (filled in by step d)
Everything else (norms, projections, the 4 attention layers, the MLP, lm_head) is ttnn in every mode.
"""
import math
import sys
import time

import torch
import torch.nn.functional as F
import ttnn

sys.path.insert(0, "<RUN_DIR>/port")
import granite_ref as g

TILE, RM, DRAM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG
HIFI = None


class TTModel:
    def __init__(self, dev, cfg, w, ssd_mode="host", kernels=None):
        self.dev, self.cfg, self.w, self.mode, self.kernels = dev, cfg, w, ssd_mode, kernels
        self.L, self.types = cfg["num_hidden_layers"], cfg["layer_types"]
        self.H, self.P, self.N = cfg["mamba_n_heads"], cfg["mamba_d_head"], cfg["mamba_d_state"]
        self.inter, self.convdim = self.H * self.P, self.H * self.P + 2 * self.N
        self.eps = cfg["rms_norm_eps"]
        self.nh, self.nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd = cfg["hidden_size"] // self.nh
        self.hid = cfg["hidden_size"]
        self.ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                        fp32_dest_acc_en=True, packer_l1_acc=True)
        self._load()
        self.reset()

    # ---- weights ----------------------------------------------------------------------------------------------------
    def up(self, t, layout=TILE, dtype=ttnn.bfloat16):
        return ttnn.from_torch(t.to(torch.bfloat16), dtype=dtype, layout=layout, device=self.dev, memory_config=DRAM)

    def lin_w(self, name):
        wt = self.w[name]                                          # [out, in]
        return self.up(wt.t().contiguous().view(1, 1, wt.shape[1], wt.shape[0]))

    def vec(self, name, shape):
        return self.up(self.w[name].view(*shape))

    def _load(self):
        w, d = self.w, {}
        t0 = time.time()
        d["embed_t"] = self.lin_w("model.embed_tokens.weight")      # [1536, vocab]: the tied lm_head
        d["norm"] = self.vec("model.norm.weight", (1, 1, 1, self.hid))
        self.L_ = []
        for i in range(self.L):
            p = f"model.layers.{i}."
            L = {"ln1": self.vec(p + "input_layernorm.weight", (1, 1, 1, self.hid)),
                 "ln2": self.vec(p + "post_attention_layernorm.weight", (1, 1, 1, self.hid)),
                 "mlp_in": self.lin_w(p + "shared_mlp.input_linear.weight"), "mlp_out": self.lin_w(p + "shared_mlp.output_linear.weight")}
            if self.types[i] == "mamba":
                m = p + "mamba."
                L["in_proj"], L["out_proj"] = self.lin_w(m + "in_proj.weight"), self.lin_w(m + "out_proj.weight")
                L["gnorm"] = self.vec(m + "norm.weight", (1, 1, 1, self.inter))
                if self.mode != "host":                              # conv and SSD constants live on the chip
                    cw = w[m + "conv1d.weight"][:, 0, :]             # [3328, 4]
                    L["cw"] = [self.up(cw[:, j].contiguous().view(1, 1, 1, self.convdim)) for j in range(4)]
                    L["cb"] = self.vec(m + "conv1d.bias", (1, 1, 1, self.convdim))
                    A = -torch.exp(w[m + "A_log"].float())
                    L["A"] = self.up(A.view(self.H, 1, 1))
                    L["dtb"] = self.up(w[m + "dt_bias"].view(1, 1, 1, self.H))
                    L["D"] = self.up(w[m + "D"].view(self.H, 1, 1))
            else:
                a = p + "self_attn."
                L["q"], L["k"], L["v"], L["o"] = (self.lin_w(a + n + "_proj.weight") for n in "qkvo")
            self.L_.append(L)
        self.d = d
        print(f"weights on chip in {time.time() - t0:.1f}s", flush=True)

    # ---- state ------------------------------------------------------------------------------------------------------
    def reset(self):
        self.pos = 0
        self.h_conv, self.h_ssm = {}, {}                             # host mode
        self.c_conv, self.c_ssm = {}, {}                             # composed / kernel mode (device)
        self.kc, self.vc = {}, {}
        for i, t in enumerate(self.types):
            if t != "mamba":
                self.kc[i] = self.vc[i] = None
                continue
            if self.mode == "host":
                self.h_conv[i] = torch.zeros(self.convdim, 3)
                self.h_ssm[i] = torch.zeros(self.H, self.P, self.N)
            else:
                z = self.up(torch.zeros(1, 1, 1, self.convdim))
                self.c_conv[i] = [z, z, z]
                self.c_ssm[i] = self.up(torch.zeros(self.H, self.P, self.N), dtype=ttnn.bfloat16)

    # ---- pieces -----------------------------------------------------------------------------------------------------
    def linear(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=self.ck, memory_config=DRAM)

    def to_host(self, t, n=None):
        v = ttnn.to_torch(t).float().reshape(-1)
        return v if n is None else v[:n]

    def mamba(self, i, h):
        L, T = self.L_[i], ttnn
        proj = self.linear(h, L["in_proj"])                          # [1,1,1,6448]
        gate = ttnn.slice(proj, [0, 0, 0, 0], [1, 1, 1, self.inter])
        xbc = ttnn.slice(proj, [0, 0, 0, self.inter], [1, 1, 1, self.inter + self.convdim])
        dtr = ttnn.slice(proj, [0, 0, 0, self.inter + self.convdim], [1, 1, 1, self.inter + self.convdim + self.H])
        if self.mode == "host":
            m = f"model.layers.{i}.mamba."
            xbc_h, dt_h = self.to_host(xbc), self.to_host(dtr)
            xbc_o, self.h_conv[i] = g.conv_step(self.h_conv[i], xbc_h, self.w[m + "conv1d.weight"], self.w[m + "conv1d.bias"])
            x, B, C = xbc_o[: self.inter].view(self.H, self.P), xbc_o[self.inter: self.inter + self.N], xbc_o[self.inter + self.N:]
            y = g.ssd_step(self.h_ssm[i], x, dt_h, self.w[m + "A_log"], self.w[m + "dt_bias"], B, C, self.w[m + "D"])
            yd = self.up(y.reshape(1, 1, 1, self.inter))
        else:
            yd = self.ssd_device(i, L, xbc, dtr)
        # gated RMSNorm: norm(y * silu(gate)) * weight
        z = ttnn.multiply(yd, ttnn.silu(gate))
        z = ttnn.rms_norm(z, weight=L["gnorm"], epsilon=self.eps, compute_kernel_config=self.ck)
        return self.linear(z, L["out_proj"])

    def ssd_device(self, i, L, xbc, dtr):
        """Conv step and SSD step on the chip, composed from stock ttnn ops (mode composed) or with the kernels (mode kernel)."""
        s0, s1, s2 = self.c_conv[i]
        c = ttnn.add(ttnn.multiply(s0, L["cw"][0]), ttnn.multiply(s1, L["cw"][1]))
        c = ttnn.add(c, ttnn.multiply(s2, L["cw"][2]))
        c = ttnn.add(c, ttnn.multiply(xbc, L["cw"][3]))
        c = ttnn.silu(ttnn.add(c, L["cb"]))
        self.c_conv[i] = [s1, s2, xbc]
        x = ttnn.reshape(ttnn.slice(c, [0, 0, 0, 0], [1, 1, 1, self.inter]), (self.H, self.P, 1))
        B = ttnn.reshape(ttnn.slice(c, [0, 0, 0, self.inter], [1, 1, 1, self.inter + self.N]), (1, 1, self.N))
        C = ttnn.slice(c, [0, 0, 0, self.inter + self.N], [1, 1, 1, self.convdim])
        C = ttnn.reshape(C, (1, 1, self.N)) if self.mode == "kernel" else ttnn.reshape(C, (1, self.N, 1))
        dt = ttnn.softplus(ttnn.add(dtr, L["dtb"]))                  # [1,1,1,48]
        dt = ttnn.reshape(dt, (self.H, 1, 1))
        decay = ttnn.exp(ttnn.multiply(dt, L["A"]))                  # [48,1,1]
        dtx = ttnn.multiply(dt, x)                                   # [48,64,1]
        if self.mode == "kernel":
            y, self.c_ssm[i] = self.kernels(self.c_ssm[i], decay, dtx, B, C)
        else:
            S = ttnn.add(ttnn.multiply(self.c_ssm[i], decay), ttnn.multiply(dtx, B))          # [48,64,128]
            self.c_ssm[i] = S
            y = ttnn.matmul(S, ttnn.repeat(C, (self.H, 1, 1)), compute_kernel_config=self.ck)  # [48,64,1]
        y = ttnn.add(y, ttnn.multiply(L["D"], x))
        return ttnn.reshape(y, (1, 1, 1, self.inter))

    def attention(self, i, h):
        L = self.L_[i]
        q = ttnn.reshape(self.linear(h, L["q"]), (1, self.nkv, self.nh // self.nkv, self.hd))
        k = ttnn.reshape(self.linear(h, L["k"]), (1, self.nkv, 1, self.hd))
        v = ttnn.reshape(self.linear(h, L["v"]), (1, self.nkv, 1, self.hd))
        k, v = ttnn.to_layout(k, RM), ttnn.to_layout(v, RM)
        self.kc[i] = k if self.kc[i] is None else ttnn.concat([self.kc[i], k], dim=2)
        self.vc[i] = v if self.vc[i] is None else ttnn.concat([self.vc[i], v], dim=2)
        K, V = ttnn.to_layout(self.kc[i], TILE), ttnn.to_layout(self.vc[i], TILE)
        s = ttnn.multiply(ttnn.matmul(q, ttnn.transpose(K, -2, -1), compute_kernel_config=self.ck), self.cfg["attention_multiplier"])
        p = ttnn.softmax(s, dim=-1)
        o = ttnn.matmul(p, V, compute_kernel_config=self.ck)         # [1,4,3,128]
        return self.linear(ttnn.reshape(o, (1, 1, 1, self.hid)), L["o"])

    def step(self, token, record=None):
        w, c, rm = self.w, self.cfg, self.cfg["residual_multiplier"]
        emb = (w["model.embed_tokens.weight"][token] * c["embedding_multiplier"]).view(1, 1, 1, self.hid)
        x = self.up(emb)
        for i in range(self.L):
            L = self.L_[i]
            h = ttnn.rms_norm(x, weight=L["ln1"], epsilon=self.eps, compute_kernel_config=self.ck)
            m = self.mamba(i, h) if self.types[i] == "mamba" else self.attention(i, h)
            x = ttnn.add(x, ttnn.multiply(m, rm))
            if record is not None:
                record[f"layer_{i:02d}.mixer"] = self.to_host(x, self.hid)
            h = ttnn.rms_norm(x, weight=L["ln2"], epsilon=self.eps, compute_kernel_config=self.ck)
            a = self.linear(h, L["mlp_in"])
            gt = ttnn.slice(a, [0, 0, 0, 0], [1, 1, 1, 4096])
            up_ = ttnn.slice(a, [0, 0, 0, 4096], [1, 1, 1, 8192])
            mlp = self.linear(ttnn.multiply(ttnn.silu(gt), up_), L["mlp_out"])
            x = ttnn.add(x, ttnn.multiply(mlp, rm))
            if record is not None:
                record[f"layer_{i:02d}"] = self.to_host(x, self.hid)
        x = ttnn.rms_norm(x, weight=self.d["norm"], epsilon=self.eps, compute_kernel_config=self.ck)
        logits = self.linear(x, self.d["embed_t"])
        self.pos += 1
        return self.to_host(logits, c["vocab_size"]) / c["logits_scaling"]
