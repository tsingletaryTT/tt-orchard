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


# --------------------------------------------------------------------------------------------------------------------
# CPU reference: rotary and the QSA indexer (copied from Qwen4Exp, transformers 5.19.0), for probes 6 to 9
# --------------------------------------------------------------------------------------------------------------------

def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Qwen4ExpTextRMSNorm without a group: x * rsqrt(mean(x^2) + eps) * (1 + weight), computed in fp32."""
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + weight.float())).to(x.dtype)


def rotate_half_ref(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The reference `apply_rotary_pos_emb` for one tensor: the first cos.shape[-1] columns rotate (rotate-half), the rest pass
    through. cos and sin must already broadcast against x[..., :rot]."""
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    return torch.cat([x_rot * cos + rotate_half_ref(x_rot) * sin, x_pass], dim=-1)


def rope_tables_ref(n: int, head_dim: int, rot: int, theta: float = 10000.0):
    """(cos, sin), each [n, rot]: the text-only reference rotary embedding (the three position streams are equal), frequencies repeated twice."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    freqs = torch.arange(n, dtype=torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def padded_tables(cos: torch.Tensor, sin: torch.Tensor, head_dim: int):
    """The tables the generated port uses: [1, 1, n, head_dim], cos = 1 and sin = 0 past the rotary columns, so that the whole row passes through
    `x * cos + (x @ R) * sin`."""
    n, rot = cos.shape
    one, zero = torch.ones(n, head_dim - rot), torch.zeros(n, head_dim - rot)
    return (torch.cat([cos, one], dim=-1).reshape(1, 1, n, head_dim), torch.cat([sin, zero], dim=-1).reshape(1, 1, n, head_dim))


def rotate_half_matrix(head_dim: int, rot: int) -> torch.Tensor:
    """R [head_dim, head_dim] with x @ R = rotate_half on the first `rot` columns and 0 elsewhere (the generated `_apply_rotary` matrix)."""
    r, half = torch.zeros(head_dim, head_dim), rot // 2
    for i in range(half):
        r[i, i + half] = 1.0
        r[i + half, i] = -1.0
    return r


def stable_topk_ref(scores: torch.Tensor, k: int):
    """Top-k along the last dimension with exact ties broken by the lowest index (what `ttnn.topk(stable=True)` documents)."""
    order = torch.sort(scores, dim=-1, descending=True, stable=True)
    return order.values[..., :k], order.indices[..., :k]


def qsa_selection_ref(q, raw_keys, cos, sin, *, budget, ratio, k_weight, eps=1e-6, topk=None):
    """The reference `Qwen4ExpTextQSAIndexer` selection for one sequence, causal, batch 1, as plain tensors.
    q [S, H, D]: the index queries after their RMSNorm and rotary; raw_keys [S, D]; cos, sin [S, rot] (full positions);
    k_weight [D]. `topk` replaces `scores.topk(...)` (default: torch.topk, as the reference). Returns a dict: `mask` [S, S] bool (token j selected
    for query i, the tail included), `blocks` (per query: the selected block indices, as a list), `scores` (per query: the block scores, fp32),
    `pooled` [S, nb_max, D] and `n_blocks` (per query)."""
    seq, heads, dim = q.shape
    block_topk = budget // ratio
    mask = torch.zeros(seq, seq, dtype=torch.bool)
    blocks, scores_all, n_blocks = [], [], []
    for qi in range(seq):
        visible = torch.arange(qi + 1)
        nb = visible.shape[0] // ratio
        n_blocks.append(nb)
        if nb > 0:
            block_tokens = visible[: nb * ratio].view(nb, ratio)
            groups = raw_keys.index_select(0, block_tokens.flatten()).view(nb, ratio, dim)
            pooled = rms_norm_ref(groups.float().mean(dim=1).to(raw_keys.dtype), k_weight, eps)
            starts = block_tokens[:, 0]
            bks = apply_rotary_ref(pooled.unsqueeze(1), cos.index_select(0, starts).unsqueeze(1), sin.index_select(0, starts).unsqueeze(1)).squeeze(1)
            scores = torch.matmul(q[qi].float(), bks.float().transpose(-1, -2)).transpose(-1, -2)
            scores = torch.relu(scores).sum(dim=-1) / math.sqrt(dim)
            k = min(block_topk, nb)
            idx = (topk or (lambda s, kk: s.topk(kk, dim=0).indices))(scores, k)
            chosen = block_tokens.index_select(0, idx).flatten()
            blocks.append([int(b) for b in idx])
            scores_all.append(scores)
        else:
            chosen = torch.tensor([], dtype=torch.long)
            blocks.append([])
            scores_all.append(torch.zeros(0))
        tail = visible[nb * ratio:]
        mask[qi, torch.cat([chosen, tail])] = True
    return {"mask": mask, "blocks": blocks, "scores": scores_all, "n_blocks": n_blocks}


def selection_inputs(seq=32, heads=4, dim=32, rot=8, seed=0, zero_bias=False):
    """Random bf16-rounded inputs for `qsa_selection_ref`: (q [S,H,D], raw_keys [S,D], cos, sin [S,rot], k_weight [D]).
    With `zero_bias` the queries are made non-negative and the keys non-positive, so every query-key product is at most 0 before the pooling
    and the normalisation. The block scores that come out are not all 0 (the pooled key is normalised and rotated), so this input only
    raises the share of tied scores; `tied_zero_scores` below builds exact ties for the topk probe."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).bfloat16().float()
    q, keys, w = r(seq, heads, dim), r(seq, dim), r(dim) * 0.1
    if zero_bias:
        q = q.abs()
        keys = -keys.abs()                  # every q.k is <= 0: relu gives exactly 0, every block ties
    cos, sin = rope_tables_ref(seq, dim, rot)
    return q, keys, cos, sin, w


def tied_zero_scores(rows: int, n_blocks: int, zero_share: float = 0.6, seed: int = 0) -> torch.Tensor:
    """Block scores [rows, n_blocks] as the reference computes them after relu and the sum over index heads: non-negative, with a share of
    entries exactly 0 (the tie case; random weights give many all-negative blocks). Values are rounded to bf16."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(rows, n_blocks, generator=g) * 4.0
    scores[torch.rand(rows, n_blocks, generator=g) < zero_share] = 0.0
    return scores.bfloat16().float()


def device_or_report(probe: str, l1_small_size: int = 0):
    """(ttnn, device) from `open_device`, or None after printing a result line that says ttnn is not importable. A probe's device mode returns 2 then."""
    try:
        return open_device(l1_small_size)
    except ImportError as exc:
        emit({"probe": probe, "mode": "device", "ok": False, "failure": f"ttnn is not importable ({short_exc(exc)}); run inside a gozer lease with the ttnn environment"})
        return None


# --------------------------------------------------------------------------------------------------------------------
# CPU reference: the Qwen4Exp sparse MoE block (transformers 5.19.0 `Qwen4ExpTextSparseMoeBlock`), for probes 10 to 14
# --------------------------------------------------------------------------------------------------------------------

def moe_inputs(tokens=32, hidden=64, experts=8, top_k=2, inter=64, shared=64, seed=0, tied_pairs=0):
    """Random bf16-rounded tensors for one MoE block, in the checkpoint's layout (one gate/up/down matrix per expert):
    hidden [T, H]; router [E, H]; gate and up [E, I, H]; down [E, H, I]; the shared expert s_gate and s_up [S, H], s_down [H, S]; s_gate_w [1, H].
    The router rows are scaled so that the logits have a standard deviation of about 2 (a softmax that is neither flat nor one-hot). `tied_pairs` makes
    that many pairs of experts (0 and 1, 2 and 3, ...) share one router row, so that their logits tie exactly for every token: the case where the
    tie rule of the top-k matters."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, scale=1.0: (torch.randn(*s, generator=g) * scale).bfloat16().float()
    router = r(experts, hidden, scale=2.0 / math.sqrt(hidden))
    for p in range(tied_pairs):
        router[2 * p + 1] = router[2 * p]
    return {"hidden": r(tokens, hidden), "router": router,
            "gate": r(experts, inter, hidden, scale=hidden ** -0.5), "up": r(experts, inter, hidden, scale=hidden ** -0.5),
            "down": r(experts, hidden, inter, scale=inter ** -0.5),
            "s_gate": r(shared, hidden, scale=hidden ** -0.5), "s_up": r(shared, hidden, scale=hidden ** -0.5),
            "s_down": r(hidden, shared, scale=shared ** -0.5), "s_gate_w": r(1, hidden, scale=hidden ** -0.5),
            "top_k": top_k}


def moe_router_ref(hidden, router, top_k, norm_topk_prob=True, stable=False):
    """`Qwen4ExpTextTopKRouter.forward`: logits = hidden @ router^T; probs = softmax(logits) in fp32; top-k of the probs; when `norm_topk_prob` (the
    config default) the selected probs are divided by their sum. Returns (logits [T, E], weights [T, k], indices [T, k]). `stable=True` breaks exact
    ties by the lowest index (what `ttnn.topk(stable=True)` does); the reference uses torch.topk, whose tie order is not the lowest index."""
    logits = hidden.float() @ router.float().t()
    probs = torch.softmax(logits, dim=-1)
    if stable:
        values, indices = stable_topk_ref(probs, top_k)
    else:
        values, indices = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        values = values / values.sum(dim=-1, keepdim=True)
    return logits, values, indices


def moe_experts_ref(hidden, indices, weights, gate, up, down):
    """`Qwen4ExpTextExperts.forward`: for each expert that some token selected, the gated MLP on those tokens
    (silu(x @ gate^T) * (x @ up^T)) @ down^T, multiplied by the token's routing weight and added into the token's row. gate/up [E, I, H], down [E, H, I].
    (The module keeps gate and up fused as one [E, 2I, H] tensor, gate first; the checkpoint stores them per expert.)"""
    out = torch.zeros(hidden.shape[0], down.shape[1])
    for e in range(gate.shape[0]):
        tok, pos = torch.where(indices == e)
        if tok.numel() == 0:
            continue
        x = hidden[tok].float()
        y = (F.silu(x @ gate[e].float().t()) * (x @ up[e].float().t())) @ down[e].float().t()
        out.index_add_(0, tok, y * weights[tok, pos, None])
    return out


def moe_shared_ref(hidden, s_gate, s_up, s_down, s_gate_w):
    """The shared expert: sigmoid(hidden @ s_gate_w^T) * ((silu(hidden @ s_gate^T) * (hidden @ s_up^T)) @ s_down^T). Returns [T, H]."""
    x = hidden.float()
    mlp = (F.silu(x @ s_gate.float().t()) * (x @ s_up.float().t())) @ s_down.float().t()
    return torch.sigmoid(x @ s_gate_w.float().t()) * mlp


def moe_block_ref(t: dict, norm_topk_prob=True):
    """`Qwen4ExpTextSparseMoeBlock.forward` on the tensors of `moe_inputs`: routed experts plus the gated shared expert. Returns [T, H]."""
    _, weights, indices = moe_router_ref(t["hidden"], t["router"], t["top_k"], norm_topk_prob)
    routed = moe_experts_ref(t["hidden"], indices, weights, t["gate"], t["up"], t["down"])
    return routed + moe_shared_ref(t["hidden"], t["s_gate"], t["s_up"], t["s_down"], t["s_gate_w"])


def routing_matrix(indices, weights, experts):
    """The dense routing weights [T, E]: the selected expert's weight at its column, 0 elsewhere (what the dense form multiplies each expert's output by)."""
    out = torch.zeros(indices.shape[0], experts)
    out.scatter_(1, indices, weights.to(out.dtype))
    return out
