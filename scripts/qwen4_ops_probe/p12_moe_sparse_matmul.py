# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 12: the routed experts of the Qwen4 sparse MoE block through `ttnn.sparse_matmul` (issue 723).

Sources read (tt-metal, read-only): the binding documentation in ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp (modes, sparsity layout, `nnz`, `indices`),
models/experimental/diffusion_gemma/tt/sparse_moe.py (ragged prefill: gather with ttnn.embedding, three sparse matmuls per group, program config per output width),
models/autoports/aleph_alpha_kolibri_1_bf16/tt/optimized_decoder.py (`_moe`, `_grouped`/`_indexed_moe`: 384 experts, top 6; `_sparse_config`).

What the op does (from the binding documentation; the CPU `emulate_sparse_matmul` below writes it out and the selftest checks it against a dense einsum):
  input a and b are tiled rank-4 tensors, `sparsity` is a bfloat16 ROW_MAJOR rank-4 tensor on the device; only `MatmulMultiCoreReuseMultiCast1DProgramConfig` is supported.
  a dense, b sparse     a [A, B, M, K], b [1, E, K, N], sparsity [A, B, 1, E]  ->  [A, B, 1, E, M, N]       (skipped positions are zero)
  a sparse, b dense     a [A, E, M, K], b [1, E, K, N], sparsity [1, 1, A, E]  ->  [A, E, M, N]
  both sparse           a [1, E, M, K], b [1, E, K, N], sparsity [1, 1, 1, E]  ->  [1, E, M, N]
  `indices` (a UINT16 ROW_MAJOR tensor of the active group ids, `is_input_b_sparse` True): the group axis is compact: slot i holds group indices[i].
  `nnz`, when given, MUST equal count_nonzero(sparsity) or the kernel hangs (deadlock) on the device. This probe never passes a guessed value: it passes None
  (the op counts) or `exact_nnz(sparsity)`, and the selftest checks that.

Forms (the CPU router's stable-topk routing is the input, so only the expert compute is under test):
  grouped      tokens in groups of 32 rows (one tile row). gate and up: a [1, G, 32, H] against b [1, E, H, I] with sparsity [1, G, 1, E] (1 where some token of the
               group selected the expert); the activation of silu(gate) * up is [G, E, 32, I]; down: a sparse, b dense with sparsity [1, 1, G, E];
               then multiplied by the routing weights [G, E, 32, 1] and summed over E. The program configs are tried from `candidate_configs` (the rules of the two
               ports above) until one runs and matches; each config that raises is listed with its error text.
  grouped_nnz  the same with `nnz` = exact_nnz(sparsity) for each call (the `nnz` form the ports use for fixed-size decode groups)
  indexed      one token (the decode path of the kolibri port): x [1, 1, 1, H], `indices` = the k selected expert ids, gate and up as [1, 1, 1, k, 1, I] reshaped to
               [1, k, 1, I], down with `is_input_a_sparse` True and the same indices, times the weights [1, k, 1, 1], summed over k.

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
           "flash": dict(tokens=64, hidden=2560, experts=16, top_k=4, inter=640, shared=640)}
GRIDS = ((8, 4), (8, 2), (4, 4), (8, 1), (4, 2), (2, 2), (1, 1))      # (x, y) core grids, tried in this order, each capped by the device grid
MAX_ATTEMPTS = 8                                                      # configs tried per matmul shape

OP_COVERAGE = {"experts.gate_up_linear": "p12 grouped (sparse_matmul, a dense b sparse), indexed", "experts.down_linear": "p12 grouped (a sparse b dense), indexed (both sparse form)",
               "experts.silu_multiply": "p12 (ttnn.silu, ttnn.multiply on the [G, E, 32, I] activation)", "experts.routing_weight_multiply": "p12 grouped, indexed",
               "experts.sum_over_experts": "p12 grouped (ttnn.sum over the expert axis), indexed (over the k slots)", "experts.skip_unrouted": "p12 sparsity tensors"}


def emulate_sparse_matmul(a, b, sparsity=None, *, a_sparse=False, b_sparse=True, indices=None):
    """The documented semantics of `ttnn.sparse_matmul` on the CPU (fp32). a: [A, B, M, K] (b sparse only), [A, E, M, K] (a sparse only), [1, E, M, K] (both) or
    [1, n, M, K] with `indices` and a sparse; b: [1, E, K, N]; sparsity as in the module docstring. Returns the output the table gives, zero where skipped."""
    E, N = b.shape[1], b.shape[3]
    if indices is not None:
        if not b_sparse:
            raise ValueError("indices needs is_input_b_sparse")
        n = int(indices.numel())
        if not a_sparse:
            A, B_, M, _ = a.shape
            out = torch.zeros(A, B_, 1, n, M, N)
            for i in range(n):
                out[:, :, 0, i] = a @ b[0, int(indices[i])]
            return out
        out = torch.zeros(1, n, a.shape[2], N)
        for i in range(n):
            out[0, i] = a[0, i] @ b[0, int(indices[i])]
        return out
    if not a_sparse and b_sparse:
        A, B_, M, _ = a.shape
        out = torch.zeros(A, B_, 1, E, M, N)
        for ia in range(A):
            for ib in range(B_):
                for e in range(E):
                    if sparsity[ia, ib, 0, e] != 0:
                        out[ia, ib, 0, e] = a[ia, ib] @ b[0, e]
        return out
    if a_sparse and not b_sparse:
        A = a.shape[0]
        out = torch.zeros(A, E, a.shape[2], N)
        for ia in range(A):
            for e in range(E):
                if sparsity[0, 0, ia, e] != 0:
                    out[ia, e] = a[ia, e] @ b[0, e]
        return out
    if a_sparse and b_sparse:
        out = torch.zeros(1, E, a.shape[2], N)
        for e in range(E):
            if sparsity[0, 0, 0, e] != 0:
                out[0, e] = a[0, e] @ b[0, e]
        return out
    raise ValueError("neither input sparse is invalid")


def exact_nnz(sparsity: torch.Tensor) -> int:
    """The `nnz` the op requires when it is given: the number of non-zero entries of the sparsity tensor, exactly."""
    return int(torch.count_nonzero(sparsity))


def group_sparsity(indices: torch.Tensor, experts: int, group_rows: int = 32) -> torch.Tensor:
    """[G, E] float: 1 where some token of the group (consecutive `group_rows` tokens) selected the expert. indices [T, k] with T a multiple of group_rows."""
    T = indices.shape[0]
    g = torch.zeros(T // group_rows, experts)
    for r in range(T):
        g[r // group_rows, indices[r]] = 1.0
    return g


def grouped_reference(t, idx, w, group_rows=32):
    """The grouped pipeline on the CPU through `emulate_sparse_matmul`: returns [T, H]."""
    T, H = t["hidden"].shape
    E, I = t["gate"].shape[0], t["gate"].shape[1]
    G = T // group_rows
    sp = group_sparsity(idx, E, group_rows)
    x = t["hidden"].reshape(1, G, group_rows, H)
    bg, bu = t["gate"].transpose(1, 2).reshape(1, E, H, I), t["up"].transpose(1, 2).reshape(1, E, H, I)
    bd = t["down"].transpose(1, 2).reshape(1, E, I, H)
    sp_b = sp.reshape(1, G, 1, E)
    gate = emulate_sparse_matmul(x, bg, sp_b, a_sparse=False, b_sparse=True).reshape(G, E, group_rows, I)
    up = emulate_sparse_matmul(x, bu, sp_b, a_sparse=False, b_sparse=True).reshape(G, E, group_rows, I)
    mid = F.silu(gate) * up
    down = emulate_sparse_matmul(mid, bd, sp.reshape(1, 1, G, E), a_sparse=True, b_sparse=False)           # [G, E, 32, H]
    R = common.routing_matrix(idx, w, E).reshape(G, group_rows, E).permute(0, 2, 1).unsqueeze(-1)           # [G, E, 32, 1]
    return (down * R).sum(dim=1).reshape(T, H)


def indexed_reference(t, idx, w, token):
    """The indexed (decode) pipeline for one token through `emulate_sparse_matmul`: returns [H]."""
    H = t["hidden"].shape[1]
    E, I = t["gate"].shape[0], t["gate"].shape[1]
    ids = idx[token]
    x = t["hidden"][token].reshape(1, 1, 1, H)
    bg, bu = t["gate"].transpose(1, 2).reshape(1, E, H, I), t["up"].transpose(1, 2).reshape(1, E, H, I)
    bd = t["down"].transpose(1, 2).reshape(1, E, I, H)
    gate = emulate_sparse_matmul(x, bg, a_sparse=False, b_sparse=True, indices=ids).reshape(1, len(ids), 1, I)
    up = emulate_sparse_matmul(x, bu, a_sparse=False, b_sparse=True, indices=ids).reshape(1, len(ids), 1, I)
    down = emulate_sparse_matmul(F.silu(gate) * up, bd, a_sparse=True, b_sparse=True, indices=ids)           # [1, k, 1, H]
    return (down * w[token].reshape(1, -1, 1, 1)).sum(dim=1).reshape(H)


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def config_valid(cfg: dict, k: int, n: int) -> bool:
    """The rules of the two ports' program configs: in0_block_w divides the K tiles; the grid's cores times per_core_N cover the N tiles; the subblock
    divides the block and holds at most 4 tiles (the DST limit used in both ports); out_block_h equals per_core_M."""
    kt, nt = k // 32, -(-n // 32)
    cores = cfg["grid"][0] * cfg["grid"][1]
    return (kt % cfg["in0_block_w"] == 0 and cores * cfg["per_core_N"] >= nt and cfg["out_block_w"] % cfg["out_subblock_w"] == 0
            and cfg["out_subblock_h"] * cfg["out_subblock_w"] <= 4 and cfg["out_block_h"] == cfg["per_core_M"] and cfg["per_core_N"] % cfg["out_subblock_w"] == 0)


def candidate_configs(m_tiles: int, k: int, n: int, grid_max=(8, 8)) -> list[dict]:
    """Program configs to try for a [M, K] x [K, N] sparse matmul, most likely first (the diffusion_gemma shape: subblock width 1, block width per_core_N, then the
    kolibri shape with the widest subblock). Each core grid is capped by `grid_max`; every returned config passes `config_valid`."""
    kt, nt = k // 32, -(-n // 32)
    out, seen = [], set()
    for gx, gy in GRIDS:
        gx, gy = min(gx, grid_max[0]), min(gy, grid_max[1])
        cores = gx * gy
        per_core_n = -(-nt // cores)
        for in0 in sorted([d for d in _divisors(kt) if d <= 8], reverse=True)[:2]:
            for sw in (1, max([d for d in _divisors(per_core_n) if d <= 4])):
                cfg = dict(grid=(gx, gy), in0_block_w=in0, out_subblock_h=1, out_subblock_w=sw, out_block_h=m_tiles, out_block_w=per_core_n,
                           per_core_M=m_tiles, per_core_N=per_core_n)
                key = tuple(sorted(cfg.items()))
                if key not in seen and config_valid(cfg, k, n):
                    seen.add(key)
                    out.append(cfg)
    return out


def selftest() -> dict:
    checks = {}
    t = common.moe_inputs(**PRESETS["tiny"])
    _, w, idx = common.moe_router_ref(t["hidden"], t["router"], t["top_k"], stable=True)
    want = common.moe_experts_ref(t["hidden"], idx, w, t["gate"], t["up"], t["down"])
    got = grouped_reference(t, idx, w)
    checks["the_grouped_pipeline_through_the_emulated_op_equals_the_reference"] = bool(torch.allclose(got, want, atol=1e-5))
    checks["the_indexed_pipeline_equals_the_reference_for_every_token"] = all(
        torch.allclose(indexed_reference(t, idx, w, tok), want[tok], atol=1e-5) for tok in range(t["hidden"].shape[0]))
    # unsorted indices: slot i holds group indices[i], so the order does not change the sum
    shuffled = idx.clone()
    shuffled[0] = idx[0].flip(0)
    w2 = w.clone()
    w2[0] = w[0].flip(0)
    checks["slot_order_does_not_change_the_indexed_sum"] = bool(torch.allclose(indexed_reference(t, shuffled, w2, 0), want[0], atol=1e-5))
    # skipped positions are zero, and the sparsity tensor marks exactly the (group, expert) pairs that some token selected
    sp = group_sparsity(idx, t["gate"].shape[0])
    a = t["hidden"].reshape(1, 1, 32, -1)
    bg = t["gate"].transpose(1, 2).reshape(1, t["gate"].shape[0], 64, 64)
    out = emulate_sparse_matmul(a, bg, sp.reshape(1, 1, 1, -1), a_sparse=False, b_sparse=True)
    skipped = (sp.reshape(-1) == 0).nonzero().flatten()
    checks["skipped_positions_are_zero_filled"] = bool(skipped.numel() == 0 or out[0, 0, 0, skipped].abs().max() == 0)
    checks["the_sparsity_marks_exactly_the_selected_pairs"] = bool(sp.sum() == torch.unique(idx).numel())
    hand = torch.tensor([[0, 1]] * 32 + [[2, 3]] * 32)            # two groups of 32 tokens that select different experts
    want_sp = torch.zeros(2, 8)
    want_sp[0, [0, 1]] = 1.0
    want_sp[1, [2, 3]] = 1.0
    checks["the_sparsity_of_a_hand_built_routing_marks_each_group_separately"] = bool(torch.equal(group_sparsity(hand, 8), want_sp))
    # the nnz rule: the exact count, never a guess
    checks["exact_nnz_equals_the_count_of_non_zero_entries"] = exact_nnz(sp) == int((sp != 0).sum())
    checks["exact_nnz_changes_with_the_sparsity"] = exact_nnz(sp) != exact_nnz(torch.cat([sp, torch.ones(1, sp.shape[1])]))
    # every candidate config is valid for the shapes of the tiny and the flash presets, and there is at least one per shape
    shapes = [(1, 64, 64), (1, 2560, 640), (1, 640, 2560), (2, 2560, 640), (2, 640, 2560)]
    checks["candidate_configs_exist_and_are_valid_for_every_preset_shape"] = all(
        candidate_configs(m, k, n) and all(config_valid(c, k, n) for c in candidate_configs(m, k, n)) for m, k, n in shapes)
    bad = dict(candidate_configs(1, 2560, 640)[0], in0_block_w=7)            # 7 does not divide 80 K tiles
    checks["a_config_whose_block_width_does_not_divide_the_k_tiles_is_invalid"] = not config_valid(bad, 2560, 640)
    checks["configs_respect_a_smaller_device_grid"] = all(c["grid"][0] <= 4 and c["grid"][1] <= 4 for c in candidate_configs(1, 2560, 640, grid_max=(4, 4)))
    try:
        emulate_sparse_matmul(a, bg, a_sparse=False, b_sparse=False)
        checks["neither_input_sparse_is_refused"] = False
    except ValueError:
        checks["neither_input_sparse_is_refused"] = True
    return {"probe": "p12_moe_sparse_matmul", "mode": "selftest", "checks": checks, "candidates_flash_gate": len(candidate_configs(2, 2560, 640)),
            "ok": all(checks.values())}


# ---- device -------------------------------------------------------------------------------------------------------------------------------

def _up(ttnn, dev, t, dtype=None, layout=None):
    return ttnn.from_torch(t, dtype=dtype or ttnn.bfloat16, layout=layout or ttnn.TILE_LAYOUT, device=dev)


def program_config(ttnn, c):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*c["grid"]), in0_block_w=c["in0_block_w"], out_subblock_h=c["out_subblock_h"], out_subblock_w=c["out_subblock_w"],
        out_block_h=c["out_block_h"], out_block_w=c["out_block_w"], per_core_M=c["per_core_M"], per_core_N=c["per_core_N"], fuse_batch=False, fused_activation=None,
        mcast_in0=True)


def compute_config(ttnn, dev):
    try:
        return ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False,
                                                      packer_l1_acc=False)
    except Exception:
        return None


def grid_of(dev):
    try:
        g = dev.compute_with_storage_grid_size()
        return (int(g.x), int(g.y))
    except Exception:
        return (8, 8)


def try_configs(ttnn, run, m_tiles, k, n, grid, tried):
    """Run `run(config_dict)` for up to MAX_ATTEMPTS candidates; returns (result, config) of the first that returns, and appends one line per failure to `tried`."""
    for c in candidate_configs(m_tiles, k, n, grid)[:MAX_ATTEMPTS]:
        try:
            return run(c), c
        except Exception as exc:
            tried.append({"k": k, "n": n, "config": c, "failure": common.short_exc(exc, 300)})
    return None, None


def run_grouped(ttnn, dev, t, idx, w, use_nnz, tried):
    T, H = t["hidden"].shape
    E, I = t["gate"].shape[0], t["gate"].shape[1]
    G = T // 32
    sp = group_sparsity(idx, E)
    grid, kc = grid_of(dev), compute_config(ttnn, dev)
    x = _up(ttnn, dev, t["hidden"].reshape(1, G, 32, H))
    bg = _up(ttnn, dev, t["gate"].transpose(1, 2).contiguous().reshape(1, E, H, I))
    bu = _up(ttnn, dev, t["up"].transpose(1, 2).contiguous().reshape(1, E, H, I))
    bd = _up(ttnn, dev, t["down"].transpose(1, 2).contiguous().reshape(1, E, I, H))
    sp_gate = _up(ttnn, dev, sp.reshape(1, G, 1, E), layout=ttnn.ROW_MAJOR_LAYOUT)
    sp_down = _up(ttnn, dev, sp.reshape(1, 1, G, E), layout=ttnn.ROW_MAJOR_LAYOUT)
    nnz_g, nnz_d = (exact_nnz(sp), exact_nnz(sp)) if use_nnz else (None, None)

    def gate_up(c, b):
        return ttnn.sparse_matmul(x, b, sparsity=sp_gate, nnz=nnz_g, is_input_a_sparse=False, is_input_b_sparse=True, program_config=program_config(ttnn, c),
                                  compute_kernel_config=kc, dtype=ttnn.bfloat16)
    g, cfg_gate = try_configs(ttnn, lambda c: gate_up(c, bg), 1, H, I, grid, tried)
    if g is None:
        raise RuntimeError("no program config ran the gate matmul: " + (tried[-1]["failure"] if tried else "no candidate"))
    u = gate_up(cfg_gate, bu)
    mid = ttnn.multiply(ttnn.silu(ttnn.reshape(g, (G, E, 32, I))), ttnn.reshape(u, (G, E, 32, I)))

    def down(c):
        return ttnn.sparse_matmul(mid, bd, sparsity=sp_down, nnz=nnz_d, is_input_a_sparse=True, is_input_b_sparse=False, program_config=program_config(ttnn, c),
                                  compute_kernel_config=kc, dtype=ttnn.bfloat16)
    d, cfg_down = try_configs(ttnn, down, 1, I, H, grid, tried)
    if d is None:
        raise RuntimeError("no program config ran the down matmul: " + (tried[-1]["failure"] if tried else "no candidate"))
    R = _up(ttnn, dev, common.routing_matrix(idx, w, E).reshape(G, 32, E).permute(0, 2, 1).unsqueeze(-1).contiguous())
    out = ttnn.sum(ttnn.multiply(d, R), dim=1)
    return ttnn.to_torch(out).float().reshape(T, H), {"gate_up_config": cfg_gate, "down_config": cfg_down}


def run_indexed(ttnn, dev, t, idx, w, tokens, tried):
    H = t["hidden"].shape[1]
    E, I = t["gate"].shape[0], t["gate"].shape[1]
    grid, kc = grid_of(dev), compute_config(ttnn, dev)
    bg = _up(ttnn, dev, t["gate"].transpose(1, 2).contiguous().reshape(1, E, H, I))
    bu = _up(ttnn, dev, t["up"].transpose(1, 2).contiguous().reshape(1, E, H, I))
    bd = _up(ttnn, dev, t["down"].transpose(1, 2).contiguous().reshape(1, E, I, H))
    k = idx.shape[1]
    got, cfgs = [], None
    for tok in tokens:
        x = _up(ttnn, dev, t["hidden"][tok].reshape(1, 1, 1, H))
        ids = ttnn.from_torch(idx[tok].to(torch.int32).reshape(1, 1, 1, k), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        sparsity = _up(ttnn, dev, torch.zeros(1, 1, 1, E), layout=ttnn.ROW_MAJOR_LAYOUT)

        def sm(a, b, c, a_sparse):
            return ttnn.sparse_matmul(a, b, sparsity=sparsity, indices=ids, is_input_a_sparse=a_sparse, is_input_b_sparse=True,
                                      program_config=program_config(ttnn, c), compute_kernel_config=kc, dtype=ttnn.bfloat16)
        g, c1 = try_configs(ttnn, lambda c: sm(x, bg, c, False), 1, H, I, grid, tried)
        if g is None:
            raise RuntimeError("no program config ran the indexed gate matmul: " + (tried[-1]["failure"] if tried else "no candidate"))
        u = sm(x, bu, c1, False)
        mid = ttnn.multiply(ttnn.silu(ttnn.reshape(g, (1, k, 1, I))), ttnn.reshape(u, (1, k, 1, I)))
        d, c2 = try_configs(ttnn, lambda c: sm(mid, bd, c, True), 1, I, H, grid, tried)
        if d is None:
            raise RuntimeError("no program config ran the indexed down matmul: " + (tried[-1]["failure"] if tried else "no candidate"))
        wt = _up(ttnn, dev, w[tok].reshape(1, k, 1, 1))
        got.append(ttnn.to_torch(ttnn.sum(ttnn.multiply(ttnn.reshape(d, (1, k, 1, H)), wt), dim=1)).float().reshape(H))
        cfgs = {"gate_up_config": c1, "down_config": c2}
    return torch.stack(got), cfgs


def run_device(ttnn, dev, t):
    _, w, idx = common.moe_router_ref(t["hidden"], t["router"], t["top_k"], stable=True)
    want = common.moe_experts_ref(t["hidden"], idx, w, t["gate"], t["up"], t["down"])
    out, tried = {}, []
    for name, nnz in (("grouped", False), ("grouped_nnz", True)):
        try:
            got, cfgs = run_grouped(ttnn, dev, t, idx, w, nnz, tried)
            stats = common.error_stats(got, want)
            out[name] = {**stats, **cfgs, "ok": bool((stats.get("pcc") or 0) > 0.99)}
        except Exception as exc:
            out[name] = {"ok": False, "failure": common.short_exc(exc)}
    tokens = list(range(min(4, t["hidden"].shape[0])))
    try:
        got, cfgs = run_indexed(ttnn, dev, t, idx, w, tokens, tried)
        stats = common.error_stats(got, want[tokens])
        out["indexed"] = {**stats, **cfgs, "tokens": tokens, "ok": bool((stats.get("pcc") or 0) > 0.99)}
    except Exception as exc:
        out["indexed"] = {"ok": False, "failure": common.short_exc(exc)}
    return out, tried


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the reference forms; no ttnn")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    ap.add_argument("--tokens", type=int, default=None, help="a multiple of 32")
    ap.add_argument("--experts", type=int, default=None)
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
    if cfg["tokens"] % 32:
        print("--tokens must be a multiple of 32", file=sys.stderr)
        return 2
    opened = common.device_or_report("p12_moe_sparse_matmul")
    if opened is None:
        return 2
    ttnn, dev = opened
    t = common.moe_inputs(**cfg)
    try:
        forms, tried = run_device(ttnn, dev, t)
    finally:
        ttnn.close_device(dev)
    good = [n for n, r in forms.items() if r.get("ok")]
    common.emit({"probe": "p12_moe_sparse_matmul", "mode": "device", "ok": True, "shape": cfg, "dtype": "bf16 TILE, sparsity bf16 ROW_MAJOR",
                 "reference": "Qwen4ExpTextExperts (CPU copy) with the CPU router's stable-topk routing", "result": forms, "configs_that_raised": tried[:12],
                 "workarounds": [], "forms_that_pass": good})
    return 0


if __name__ == "__main__":
    sys.exit(main())
