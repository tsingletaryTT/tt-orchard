# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The chunked Mamba-2 scan (state space duality form) for a block of T prompt tokens, in plain torch.

This is the algorithm that `GraniteChip.prefill_chunked` runs with ttnn ops, written so it can be checked on the CPU against the
one-token recurrence that the decode step uses:

    S_t = exp(dt_t A) S_{t-1} + (dt_t x_t) (x) B_t         y_t = S_t C_t + D x_t           (per head; one B and C group)

For a block of T tokens, with a_t = dt_t A and cs_t = a_1 + ... + a_t (inclusive, per head):

    y_t = sum_{j<=t} exp(cs_t - cs_j) (C_t . B_j) dt_j x_j     the tokens of this block
        + exp(cs_t) (S_0 C_t)                                  the state the block starts from
        + D x_t
    S_T = exp(cs_T) S_0 + sum_j exp(cs_T - cs_j) dt_j x_j (x) B_j

The decay is always written as exp of a difference that is <= 0 (the upper triangle is masked before the exp). Writing it as
exp(cs_t) * exp(-cs_j) overflows for a strong decay.
"""
import torch


def ssd_chunk(x, dt, B, C, A, D, S0, *, H, P, N):
    """x [T, H*P], dt [T, H] (after softplus), B and C [T, N], A and D [H], S0 [H*P, N].
    Returns (y [T, H*P], S_T [H*P, N]). The state is in the same layout as the decode step's persistent buffer."""
    T = x.shape[0]
    a = dt * A                                                       # [T, H]
    cs = torch.cumsum(a, dim=0)                                      # [T, H]
    csh = cs.t()                                                     # [H, T]
    diff = csh.unsqueeze(2) - csh.unsqueeze(1)                       # [H, T, T]: cs_i - cs_j
    tri = torch.tril(torch.ones(T, T, dtype=torch.bool))
    Lm = torch.exp(torch.where(tri, diff, torch.full_like(diff, -1e30)))     # zero above the diagonal, no overflow
    CB = C @ B.t()                                                   # [T, T], shared by the heads
    xh = x.view(T, H, P).permute(1, 0, 2)                            # [H, T, P]
    dtx = dt.t().unsqueeze(2) * xh                                   # [H, T, P]
    y_in = (Lm * CB) @ dtx                                           # [H, T, P]
    S0h = S0.view(H, P, N)
    y_st = torch.exp(cs).t().unsqueeze(2) * (C @ S0h.reshape(H * P, N).t()).view(T, H, P).permute(1, 0, 2)
    y = (y_in + y_st).permute(1, 0, 2).reshape(T, H * P) + D.repeat_interleave(P) * x
    w = torch.exp(csh[:, -1:] - csh).unsqueeze(2) * dtx              # [H, T, P]: exp(cs_T - cs_j) dt_j x_j
    S = torch.exp(csh[:, -1]).view(H, 1, 1) * S0h + w.transpose(1, 2) @ B       # [H, P, N]
    return y, S.reshape(H * P, N)


def causal_conv_chunk(xbc, state, w, b):
    """Depth-wise causal convolution of width 4 with a SiLU, over T tokens at once.
    xbc [T, C], state [3, C] (the three inputs before this block, oldest first), w [4, C] (oldest tap first), b [C].
    Returns (silu(conv + b) [T, C], the new state [3, C])."""
    T = xbc.shape[0]
    pad = torch.cat([state, xbc], dim=0)                             # [T+3, C]
    c = sum(w[j] * pad[j:j + T] for j in range(4)) + b
    return torch.nn.functional.silu(c), pad[T:T + 3]
