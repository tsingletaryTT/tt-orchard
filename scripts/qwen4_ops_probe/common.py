# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Shared code for the Qwen4-style ops probes (docs/run-logs/2026-11-02-qwen4-ops-probe.md).

Two halves:
  * the CPU reference side: torch copies of the Qwen4Exp reference code (transformers 5.19.0,
    `modeling_qwen4_exp.py`): the chunked and the recurrent gated delta rule, the grouped RMSNorm, the gated residual and
    the dilated depthwise short convolution. It imports only torch, so each probe's `--selftest` runs without ttnn.
  * the device side (`open_device`, `error_stats`, `timed`): ttnn is imported only inside `open_device`, because
    importing ttnn opens a device. Run a probe on a chip only inside a gozer lease.

Every probe prints one line `PROBE_RESULT {json}` as the last thing it does.
"""
from __future__ import annotations

import json
import math
import time

import torch
import torch.nn.functional as F

MARK = "PROBE_RESULT "


def emit(result: dict) -> None:
    """Print the one JSON result line a probe ends with."""
    print(MARK + json.dumps(_strict(result), sort_keys=True, allow_nan=False), flush=True)


def _strict(obj):
    """Replace every non-finite float with None, so the line is standard JSON (no NaN or Infinity tokens). The
    `nonfinite` count in `error_stats` records that an op returned bad values."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict(v) for v in obj]
    return obj


# --------------------------------------------------------------------------------------------------------------------
# CPU reference: gated delta rule (copied from Qwen4Exp, which matches Qwen3.5's torch fallbacks)
# --------------------------------------------------------------------------------------------------------------------

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                                 output_final_state=False, use_qk_l2norm_in_kernel=False):
    """Copy of the reference `torch_chunk_gated_delta_rule`. query/key: [B, T, HV, K] (already expanded to the value
    head count), value: [B, T, HV, V], g and beta: [B, T, HV]. Returns (o [B, T, HV, V], state [B, HV, K, V] or None)."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
    decay = g
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32).contiguous()
                                      for x in (query, key, value, beta, decay)]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query * (query.shape[-1] ** -0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    total = sequence_length + pad_size
    num_chunks = total // chunk_size
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                  for x in (query, key, k_beta, v_beta)]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    strictly_upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
    pairwise = pairwise.masked_fill(strictly_upper, float("-inf")).exp()
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise
    intra = (query @ key.transpose(-1, -2)) * pairwise
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut_system, decayed_k_beta, upper=False, unitriangular=True)
    state = (torch.zeros(recurrent_state_shape, dtype=new_values.dtype) if initial_state is None
             else initial_state.to(new_values))
    out = torch.zeros_like(new_values)
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ state
        out[:, :, i] = query[:, :, i] @ state + intra[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    state = state if output_final_state else None
    out = out.reshape(padded_output_shape)[:, :, :sequence_length]
    return out.transpose(1, 2).to(initial_dtype).contiguous(), state


def torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None, output_final_state=False,
                                     use_qk_l2norm_in_kernel=False):
    """Copy of the reference `torch_recurrent_gated_delta_rule`; same shapes as the chunked one."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32).contiguous()
                                      for x in (query, key, value, beta, g)]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query = query / (query.shape[-1] ** 0.5)
    state = (torch.zeros((batch_size, num_v_heads, k_head_dim, v_head_dim), dtype=value.dtype)
             if initial_state is None else initial_state.to(value))
    out = torch.zeros_like(value)
    for i in range(sequence_length):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        state = state * decay[:, :, i].exp()[..., None, None]
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, i].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)
    state = state if output_final_state else None
    return out.transpose(1, 2).contiguous().to(initial_dtype), state


def make_gdn_inputs(batch, seq, heads_k, heads_v, dim_k, dim_v, seed=0):
    """Random raw q/k/v (key heads, not expanded), a log-space decay g <= 0 and a beta in (0, 1)."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, seq, heads_k, dim_k, generator=gen)
    k = torch.randn(batch, seq, heads_k, dim_k, generator=gen)
    v = torch.randn(batch, seq, heads_v, dim_v, generator=gen)
    g = -(0.02 + 0.5 * torch.rand(batch, seq, heads_v, generator=gen))
    beta = torch.sigmoid(torch.randn(batch, seq, heads_v, generator=gen))
    return q, k, v, g, beta


def expand_heads(x: torch.Tensor, heads_v: int) -> torch.Tensor:
    """Key heads to value heads the way the model does it: repeat_interleave on the head axis."""
    return x.repeat_interleave(heads_v // x.shape[2], dim=2)


# --------------------------------------------------------------------------------------------------------------------
# CPU reference: gated residual and dilated depthwise short convolution
# --------------------------------------------------------------------------------------------------------------------

class GatedResidualRef(torch.nn.Module):
    """Copy of `Qwen4ExpTextGatedResidual` with its `Qwen4ExpTextRMSNorm(group_size=hidden)`."""

    def __init__(self, hidden=2560, hc_count=4, lowrank=320, eps=1e-6):
        super().__init__()
        self.hc_count, self.hidden, self.eps = hc_count, hidden, eps
        n = hc_count * hidden
        self.norm_weight = torch.nn.Parameter(torch.zeros(n))
        self.down = torch.nn.Linear(n, lowrank, bias=False)
        self.up = torch.nn.Linear(lowrank, n, bias=False)
        self.inject = torch.nn.Linear(n, hc_count, bias=False)

    def hc_norm(self, x):
        x32 = x.float().reshape(*x.shape[:-1], -1, self.hidden)
        out = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).flatten(-2)
        return (out * (1.0 + self.norm_weight.float())).type_as(x)

    def forward(self, x):
        normed = self.hc_norm(x)
        mix = F.silu(self.down(normed) / self.hc_count)
        mix = torch.sigmoid(self.up(mix)).unflatten(-1, (self.hc_count, self.hidden))
        mixed = (mix * normed.unflatten(-1, (self.hc_count, self.hidden))).mean(dim=-2)
        injection = 2 * torch.sigmoid(self.inject(normed) / self.hc_count)
        return mixed, injection


def short_conv_ref(x: torch.Tensor, weight: torch.Tensor, kernel=4, dilation=3) -> torch.Tensor:
    """`_short_conv` of the PLE layer without a cache: x [B, T, C], weight [C, 1, kernel]. Left-pads by
    (kernel - 1) * dilation, runs a depthwise dilated conv1d and applies silu. Returns [B, T, C]."""
    state_len = (kernel - 1) * dilation
    h = F.pad(x.transpose(1, 2), (state_len, 0))
    return F.silu(F.conv1d(h, weight, groups=x.shape[-1], dilation=dilation)).transpose(1, 2)


def short_conv_shifted(x: torch.Tensor, weight: torch.Tensor, kernel=4, dilation=3) -> torch.Tensor:
    """The same convolution as `kernel` shifted multiply-adds (the shape of the ttnn fallback), on the CPU:
    out[t] = sum_j w[:, 0, j] * xpad[t + j * dilation], xpad being x left-padded by (kernel - 1) * dilation."""
    state_len = (kernel - 1) * dilation
    seq = x.shape[1]
    xpad = F.pad(x, (0, 0, state_len, 0))
    acc = torch.zeros_like(x)
    for j in range(kernel):
        acc = acc + xpad[:, j * dilation: j * dilation + seq, :] * weight[:, 0, j]
    return F.silu(acc)


# --------------------------------------------------------------------------------------------------------------------
# Comparison and device helpers
# --------------------------------------------------------------------------------------------------------------------

def error_stats(got: torch.Tensor, ref: torch.Tensor) -> dict:
    """Error of `got` against `ref` (both cast to fp32). max_rel is max_abs divided by max |ref|. max_rel_elem is the
    largest |got - ref| / max(|ref|, 0.01 * rms(ref)), which does not blow up where ref is near zero. pcc is the
    Pearson correlation of the flattened tensors. Non-finite values in `got` are counted."""
    got, ref = got.detach().float().reshape(-1), ref.detach().float().reshape(-1)
    if got.shape != ref.shape:
        return {"error": f"shape mismatch got {tuple(got.shape)} ref {tuple(ref.shape)}"}
    nonfinite = int((~torch.isfinite(got)).sum())
    diff = (got - ref).abs()
    ref_abs_max = float(ref.abs().max())
    floor = 0.01 * float(ref.pow(2).mean().sqrt())
    pcc = float(torch.corrcoef(torch.stack([got, ref]))[0, 1]) if nonfinite == 0 and ref.std() > 0 else None
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "max_rel": float(diff.max()) / ref_abs_max if ref_abs_max else None,
            "max_rel_elem": float((diff / ref.abs().clamp(min=floor)).max()) if floor else None,
            "pcc": pcc, "ref_abs_max": ref_abs_max, "nonfinite": nonfinite}


def open_device(l1_small_size: int = 0):
    """Import ttnn and open device 0 of the lease. Returns (ttnn, device). Call only inside `gozer run`.
    `l1_small_size` reserves the L1_SMALL region at open; ttnn.conv1d needs it (the default 0 fails with an
    out-of-memory error on a 16 B L1_SMALL buffer)."""
    import ttnn
    if l1_small_size:
        return ttnn, ttnn.open_device(device_id=0, l1_small_size=l1_small_size)
    return ttnn, ttnn.open_device(device_id=0)


def timed(fn, ttnn, device, repeat=2):
    """Run fn() `repeat` times, synchronizing after each. Returns (last result, [wall seconds per call]). The first
    call includes program compilation."""
    times, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        times.append(round(time.perf_counter() - t0, 4))
    return out, times


def short_exc(exc: BaseException, limit=600) -> str:
    """A one-line description of an exception for a result line."""
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[:limit]}"
