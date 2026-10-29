"""Granite-4.0-H-1B in plain torch, one token at a time (a recurrent decode step; prefill is a loop of steps).

This is the model's own math written out so that (1) every op the chip port needs has a torch twin with a name,
(2) the op-level references for the kernel probe come from real activations, and (3) the chip port can be compared
layer by layer. It is validated against transformers' GraniteMoeHybridForCausalLM by `check_against_hf.py`.

Shapes (granite-4.0-h-1b): hidden 1536, 40 layers (36 mamba, 4 attention at layers 5, 15, 25, 35), mamba: 48 heads of
64, state 128, one group, conv kernel 4, conv channels 3328 (= 3072 x + 128 B + 128 C). Attention: 12 heads, 4 KV
heads, head dim 128, no position embedding, softmax scale 0.0078125. muP: embedding x12, residual x0.22, logits /6.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

SNAP = Path("<cache>/huggingface/hub/models--ibm-granite--granite-4.0-h-1b/snapshots/d18cca4c121edb87d022116d281ce212c9136f57")


def load(snap=SNAP, dtype=torch.float32):
    from safetensors import safe_open
    cfg = json.load(open(Path(snap) / "config.json"))
    w = {}
    with safe_open(str(Path(snap) / "model.safetensors"), "pt") as f:
        for k in f.keys():
            w[k] = f.get_tensor(k).to(dtype)
    return cfg, w


def rmsnorm(x, weight, eps):
    x32 = x.float()
    return weight * (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps))


def gated_rmsnorm(y, gate, weight, eps):
    """norm(y * silu(gate)) * weight: the Mamba-2 output norm (one group, so one mean over all 3072 channels)."""
    h = y.float() * F.silu(gate.float())
    return weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))


def conv_step(conv_state, xbc, w, b):
    """Depthwise causal conv, kernel 4, one token. conv_state [C, 3] holds the previous three inputs. Returns (silu(out), new state)."""
    window = torch.cat([conv_state, xbc[:, None]], dim=1)          # [C, 4]
    out = (window * w[:, 0, :]).sum(-1) + b
    return F.silu(out), window[:, 1:]


def ssd_step(state, x, dt_raw, A_log, dt_bias, B, C, D):
    """One Mamba-2 state update and readout. state [H, P, N] (updated in place), x [H, P], dt_raw [H], B and C [N] (one group).
    state' = state * exp(dt * A) + (dt * x)[..., None] * B ;  y = (state' * C).sum(-1) + D * x.   Returns y [H, P]."""
    dt = F.softplus(dt_raw.float() + dt_bias.float())              # [H]
    A = -torch.exp(A_log.float())                                  # [H]
    decay = torch.exp(dt * A)                                      # [H]
    state.mul_(decay[:, None, None]).add_((dt[:, None] * x)[..., None] * B[None, None, :])
    return (state * C[None, None, :]).sum(-1) + D[:, None] * x


class Model:
    def __init__(self, cfg, w):
        self.cfg, self.w = cfg, w
        self.L = cfg["num_hidden_layers"]
        self.types = cfg["layer_types"]
        self.H, self.P, self.N = cfg["mamba_n_heads"], cfg["mamba_d_head"], cfg["mamba_d_state"]
        self.inter = self.H * self.P
        self.convdim = self.inter + 2 * self.N
        self.eps = cfg["rms_norm_eps"]
        self.reset()

    def reset(self):
        self.conv = {}
        self.ssm = {}
        self.k = {}
        self.v = {}
        for i, t in enumerate(self.types):
            if t == "mamba":
                self.conv[i] = torch.zeros(self.convdim, self.cfg["mamba_d_conv"] - 1)
                self.ssm[i] = torch.zeros(self.H, self.P, self.N)
            else:
                self.k[i], self.v[i] = [], []

    def mixer_mamba(self, i, h, rec=None):
        w, p = self.w, f"model.layers.{i}.mamba."
        proj = w[p + "in_proj.weight"] @ h                          # [6448]
        gate, xbc, dt = proj[: self.inter], proj[self.inter: self.inter + self.convdim], proj[self.inter + self.convdim:]
        xbc, self.conv[i] = conv_step(self.conv[i], xbc, w[p + "conv1d.weight"], w[p + "conv1d.bias"])
        x, B, C = xbc[: self.inter], xbc[self.inter: self.inter + self.N], xbc[self.inter + self.N:]
        x = x.view(self.H, self.P)
        if rec is not None:                                          # record the SSD op's inputs (before the update) and outputs
            rec["ssd_in"] = dict(state=self.ssm[i].clone(), x=x.clone(), dt_raw=dt.clone(), A_log=w[p + "A_log"].clone(),
                                 dt_bias=w[p + "dt_bias"].clone(), B=B.clone(), C=C.clone(), D=w[p + "D"].clone())
        y = ssd_step(self.ssm[i], x, dt, w[p + "A_log"], w[p + "dt_bias"], B, C, w[p + "D"])
        if rec is not None:
            rec["ssd_out"] = dict(state=self.ssm[i].clone(), y=y.clone())
        y = gated_rmsnorm(y.reshape(-1), gate, w[p + "norm.weight"], self.eps)
        return w[p + "out_proj.weight"] @ y

    def mixer_attn(self, i, h):
        w, p, c = self.w, f"model.layers.{i}.self_attn.", self.cfg
        nh, nkv = c["num_attention_heads"], c["num_key_value_heads"]
        hd = c["hidden_size"] // nh
        q = (w[p + "q_proj.weight"] @ h).view(nh, hd)
        self.k[i].append((w[p + "k_proj.weight"] @ h).view(nkv, hd))
        self.v[i].append((w[p + "v_proj.weight"] @ h).view(nkv, hd))
        K, V = torch.stack(self.k[i], 1), torch.stack(self.v[i], 1)    # [nkv, T, hd]
        K, V = K.repeat_interleave(nh // nkv, 0), V.repeat_interleave(nh // nkv, 0)
        att = torch.softmax((q[:, None, :] * K).sum(-1) * c["attention_multiplier"], dim=-1)   # [nh, T]
        o = (att[..., None] * V).sum(1).reshape(-1)
        return w[p + "o_proj.weight"] @ o

    def mlp(self, i, h):
        w, p = self.w, f"model.layers.{i}.shared_mlp."
        a = w[p + "input_linear.weight"] @ h
        g, u = a.chunk(2)
        return w[p + "output_linear.weight"] @ (F.silu(g) * u)

    def step(self, token, record=None):
        """One token in, logits out. `record`, when a dict, receives layer_NN, layer_NN.mixer, layer_NN.mlp and the SSD op arrays."""
        w, c = self.w, self.cfg
        x = w["model.embed_tokens.weight"][token] * c["embedding_multiplier"]
        rm = c["residual_multiplier"]
        for i in range(self.L):
            rec = {} if record is not None else None
            h = rmsnorm(x, w[f"model.layers.{i}.input_layernorm.weight"], self.eps)
            m = self.mixer_mamba(i, h, rec) if self.types[i] == "mamba" else self.mixer_attn(i, h)
            x = x + m * rm
            if record is not None:
                record[f"layer_{i:02d}.mixer"] = x.clone()
            h = rmsnorm(x, w[f"model.layers.{i}.post_attention_layernorm.weight"], self.eps)
            x = x + self.mlp(i, h) * rm
            if record is not None:
                record[f"layer_{i:02d}"] = x.clone()
                if rec and "ssd_in" in rec:
                    record[f"ssd.layer_{i:02d}"] = rec
        x = rmsnorm(x, w["model.norm.weight"], self.eps)
        logits = (w["model.embed_tokens.weight"] @ x) / c["logits_scaling"]
        if record is not None:
            record["final_norm"] = x.clone()
        return logits
