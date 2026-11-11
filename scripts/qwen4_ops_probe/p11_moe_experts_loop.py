# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 11: the routed experts of the Qwen4 sparse MoE block (`Qwen4ExpTextExperts`) from plain ttnn ops, with no sparse matmul (issue 723).

The reference runs, for each expert that some token selected, the gated MLP on those tokens, scales each row by its routing weight and adds it into the
token's row. Flash-Next has 512 experts per layer, hidden 2560, expert width 640, k = 10. This probe is the baseline that needs no special op. The routing
(indices and weights) comes from the CPU router, so only the expert compute is under test. Forms:
  dense_masked              every expert runs on every token and its output is multiplied by the dense routing weight [T, E] (0 for an unrouted token), then
                            summed. The same numbers as the reference by linearity; no data-dependent shape. Cost: E experts per token where the reference runs k. (The nomic MoE port in
                            tt-metal does this; it weights before the down projection, which is the same sum.)
  dense_masked_fused_gate_up the same with gate and up stored as one [H, 2I] matrix (as the transformers module keeps them): one ttnn.linear, then two slices
  gather_scatter_host       per expert that was hit: the tokens routed to it are gathered on the host (padded to a multiple of 32 rows, the padding rows have
                            routing weight 0), the expert's MLP runs on that many rows, and the result is added back with a one-hot [T, n] matrix and
                            ttnn.matmul (the scatter has no data-dependent shape on the device)
  gather_scatter_device     the same with the gather done by ttnn.embedding of the row indices into the hidden states (the form of the diffusion_gemma ragged
                            MoE in tt-metal)
Each is compared with `common.moe_experts_ref`. What scales: the dense forms cost E expert MLPs per token; the gather forms cost the experts that were hit, with
one launch group per expert (512 per layer at Flash-Next, most of them hit by a batch of tokens). The `flash` preset runs 16 of the 512 experts at the real widths
(the weights of 512 experts do not fit one chip in bf16: see probe 14).

Run on a chip only inside a gozer lease. `--selftest` checks the CPU reference and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PRESETS = {"tiny": dict(tokens=32, hidden=64, experts=8, top_k=2, inter=64, shared=64),
           "flash": dict(tokens=32, hidden=2560, experts=16, top_k=4, inter=640, shared=640)}

OP_COVERAGE = {"experts.gate_up_linear": "p11 dense_masked, dense_masked_fused_gate_up, gather_scatter_*", "experts.silu_multiply": "p11 all forms",
               "experts.down_linear": "p11 all forms", "experts.routing_weight_multiply": "p11 all forms (a [T, 1] column against [T, H])",
               "experts.index_add_scatter": "p11 gather_scatter_* (one-hot matmul)", "experts.gather_rows": "p11 gather_scatter_host (host), gather_scatter_device (ttnn.embedding)",
               "experts.sum_over_experts": "p11 all forms (ttnn.add accumulation)"}


def pad_rows(tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
    """The row indices of one expert's tokens padded to a multiple of 32 (at least 32) with token 0, and the number of real rows. Padding rows get weight 0."""
    n = int(tokens.numel())
    total = max(32, -(-n // 32) * 32)
    return torch.cat([tokens, torch.zeros(total - n, dtype=tokens.dtype)]), n


def scatter_matrix(tokens: torch.Tensor, rows: int, total_tokens: int) -> torch.Tensor:
    """S [T, rows] with S[tokens[i], i] = 1 for the real rows: S @ y_e adds the expert's rows back into the token rows."""
    s = torch.zeros(total_tokens, rows)
    s[tokens, torch.arange(tokens.numel())] = 1.0
    return s


def expert_mlp_ref(x, gate, up, down):
    return (F.silu(x @ gate.t()) * (x @ up.t())) @ down.t()


def selftest() -> dict:
    checks = {}
    t = common.moe_inputs(**PRESETS["tiny"])
    _, w, idx = common.moe_router_ref(t["hidden"], t["router"], t["top_k"], stable=True)
    want = common.moe_experts_ref(t["hidden"], idx, w, t["gate"], t["up"], t["down"])
    R = common.routing_matrix(idx, w, t["gate"].shape[0])
    dense = sum(R[:, e:e + 1] * expert_mlp_ref(t["hidden"], t["gate"][e], t["up"][e], t["down"][e]) for e in range(t["gate"].shape[0]))
    checks["dense_masked_equals_the_reference_loop"] = bool(torch.allclose(dense, want, atol=1e-5))
    fused = [torch.cat([t["gate"][e], t["up"][e]], dim=0) for e in range(t["gate"].shape[0])]
    inter = t["gate"].shape[1]
    fd = sum(R[:, e:e + 1] * ((F.silu((t["hidden"] @ fused[e].t())[:, :inter]) * (t["hidden"] @ fused[e].t())[:, inter:]) @ t["down"][e].t())
             for e in range(len(fused)))
    checks["fused_gate_up_equals_separate_gate_and_up_with_the_gate_first"] = bool(torch.allclose(fd, want, atol=1e-5))
    before = sum(((F.silu(t["hidden"] @ t["gate"][e].t()) * (t["hidden"] @ t["up"][e].t())) * R[:, e:e + 1]) @ t["down"][e].t() for e in range(len(fused)))
    checks["weighting_before_the_down_projection_is_the_same_sum"] = bool(torch.allclose(before, want, atol=1e-5))
    out = torch.zeros_like(want)
    hit = 0
    per_token = torch.zeros(t["hidden"].shape[0], dtype=torch.long)
    for e in range(t["gate"].shape[0]):
        tok, pos = torch.where(idx == e)
        if tok.numel() == 0:
            continue
        hit += 1
        per_token.index_add_(0, tok, torch.ones_like(tok))
        rows, n = pad_rows(tok)
        xe = t["hidden"][rows]
        we = torch.zeros(rows.numel())
        we[:n] = w[tok, pos]
        ye = expert_mlp_ref(xe, t["gate"][e], t["up"][e], t["down"][e]) * we[:, None]
        out += scatter_matrix(tok, rows.numel(), t["hidden"].shape[0]) @ ye
    checks["gather_with_padding_then_one_hot_scatter_equals_index_add"] = bool(torch.allclose(out, want, atol=1e-5))
    checks["every_token_is_gathered_k_times"] = bool((per_token == t["top_k"]).all())
    checks["padding_rows_are_a_multiple_of_32"] = all(pad_rows(torch.arange(n))[0].numel() % 32 == 0 for n in (1, 5, 32, 33, 64))
    checks["scatter_matrix_rows_with_a_token_sum_to_one"] = bool(scatter_matrix(torch.tensor([3, 5]), 32, 8)[:, :2].sum(0).eq(1).all())
    checks["experts_were_hit"] = hit > 1
    checks["flash_preset_runs_a_bounded_number_of_experts"] = PRESETS["flash"]["experts"] <= 16
    return {"probe": "p11_moe_experts_loop", "mode": "selftest", "checks": checks, "experts_hit_of_8": hit, "ok": all(checks.values())}


# ---- device -------------------------------------------------------------------------------------------------------------------------------

def _up(ttnn, dev, t, dtype=None, layout=None):
    return ttnn.from_torch(t, dtype=dtype or ttnn.bfloat16, layout=layout or ttnn.TILE_LAYOUT, device=dev)


def upload_experts(ttnn, dev, t, fused=False):
    """Per expert [1, 1, K, N] matrices (the weights transposed for ttnn.linear): gate, up (or the fused gate-up) and down."""
    E = t["gate"].shape[0]
    mk = lambda m: _up(ttnn, dev, m.t().contiguous().reshape(1, 1, m.shape[1], m.shape[0]))
    if fused:
        return [(mk(torch.cat([t["gate"][e], t["up"][e]], dim=0)), None, mk(t["down"][e])) for e in range(E)]
    return [(mk(t["gate"][e]), mk(t["up"][e]), mk(t["down"][e])) for e in range(E)]


def expert_mlp(ttnn, x, wg, wu, wd, inter=None):
    """The gated MLP of one expert on x [1, 1, n, H]; `wu is None` means wg holds gate and up side by side [H, 2I]."""
    if wu is None:
        gu = ttnn.linear(x, wg)
        n = gu.shape[2]
        g = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, n, inter))
        u = ttnn.slice(gu, (0, 0, 0, inter), (1, 1, n, 2 * inter))
    else:
        g, u = ttnn.linear(x, wg), ttnn.linear(x, wu)
    return ttnn.linear(ttnn.multiply(ttnn.silu(g), u), wd)


def dense_masked(ttnn, dev, t, idx, w, fused=False):
    T, E = t["hidden"].shape[0], t["gate"].shape[0]
    inter = t["gate"].shape[1]
    x = _up(ttnn, dev, t["hidden"].reshape(1, 1, T, -1))
    R = _up(ttnn, dev, common.routing_matrix(idx, w, E).reshape(1, 1, T, E))
    acc = None
    for e, (wg, wu, wd) in enumerate(upload_experts(ttnn, dev, t, fused)):
        y = ttnn.multiply(expert_mlp(ttnn, x, wg, wu, wd, inter), ttnn.slice(R, (0, 0, 0, e), (1, 1, T, e + 1)))
        acc = y if acc is None else ttnn.add(acc, y)
    return ttnn.to_torch(acc).float().reshape(T, -1)


def gather_scatter(ttnn, dev, t, idx, w, device_gather=False):
    T, E = t["hidden"].shape[0], t["gate"].shape[0]
    x = _up(ttnn, dev, t["hidden"].reshape(1, 1, T, -1))
    flat = ttnn.reshape(x, (T, t["hidden"].shape[1])) if device_gather else None
    experts = upload_experts(ttnn, dev, t)
    acc, hit = None, 0
    for e in range(E):
        tok, pos = torch.where(idx == e)
        if tok.numel() == 0:
            continue
        hit += 1
        rows, n = pad_rows(tok)
        if device_gather:
            ids = ttnn.from_torch(rows.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            xe = ttnn.reshape(ttnn.embedding(ids, flat, layout=ttnn.TILE_LAYOUT), (1, 1, rows.numel(), t["hidden"].shape[1]))
        else:
            xe = _up(ttnn, dev, t["hidden"][rows].reshape(1, 1, rows.numel(), -1))
        we = torch.zeros(rows.numel())
        we[:n] = w[tok, pos]
        wg, wu, wd = experts[e]
        ye = ttnn.multiply(expert_mlp(ttnn, xe, wg, wu, wd), _up(ttnn, dev, we.reshape(1, 1, -1, 1)))
        back = ttnn.matmul(_up(ttnn, dev, scatter_matrix(tok, rows.numel(), T).reshape(1, 1, T, -1)), ye)
        acc = back if acc is None else ttnn.add(acc, back)
    return ttnn.to_torch(acc).float().reshape(T, -1), hit


def run_device(ttnn, dev, t):
    _, w, idx = common.moe_router_ref(t["hidden"], t["router"], t["top_k"], stable=True)
    want = common.moe_experts_ref(t["hidden"], idx, w, t["gate"], t["up"], t["down"])
    out = {}
    forms = {"dense_masked": lambda: (dense_masked(ttnn, dev, t, idx, w), {}),
             "dense_masked_fused_gate_up": lambda: (dense_masked(ttnn, dev, t, idx, w, fused=True), {}),
             "gather_scatter_host": lambda: (lambda r: (r[0], {"experts_hit": r[1]}))(gather_scatter(ttnn, dev, t, idx, w)),
             "gather_scatter_device": lambda: (lambda r: (r[0], {"experts_hit": r[1]}))(gather_scatter(ttnn, dev, t, idx, w, device_gather=True))}
    for name, fn in forms.items():
        try:
            got, extra = fn()
            stats = common.error_stats(got, want)
            out[name] = {**stats, **extra, "ok": bool((stats.get("pcc") or 0) > 0.99)}
        except Exception as exc:
            out[name] = {"ok": False, "failure": common.short_exc(exc)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the reference forms; no ttnn")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    ap.add_argument("--tokens", type=int, default=None)
    ap.add_argument("--experts", type=int, default=None, help="experts to run (the flash preset uses 16 of 512)")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest()
        common.emit(res)
        return 0 if res["ok"] else 1
    cfg = dict(PRESETS[args.preset])
    for key in ("tokens", "experts", "top_k"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    opened = common.device_or_report("p11_moe_experts_loop")
    if opened is None:
        return 2
    ttnn, dev = opened
    t = common.moe_inputs(**cfg)
    try:
        forms = run_device(ttnn, dev, t)
    finally:
        ttnn.close_device(dev)
    good = [n for n, r in forms.items() if r.get("ok")]
    common.emit({"probe": "p11_moe_experts_loop", "mode": "device", "ok": True, "shape": cfg, "dtype": "bf16 TILE",
                 "reference": "Qwen4ExpTextExperts (CPU copy) with the CPU router's stable-topk routing", "result": forms,
                 "workarounds": [n for n in good if n != "dense_masked"] if "dense_masked" not in good else [], "forms_that_pass": good})
    return 0


if __name__ == "__main__":
    sys.exit(main())
