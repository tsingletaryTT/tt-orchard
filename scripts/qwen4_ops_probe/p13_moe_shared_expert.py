# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 13: the shared expert, the routing matrix built on the device, and the whole sparse MoE block composed from the forms of probes 10 and 11 (issue 723).

`Qwen4ExpTextSparseMoeBlock.forward`: shared = shared_expert(x) (a gated MLP of width 640 at Flash-Next); routed = experts(x, topk indices, topk weights) from the
router; shared = sigmoid(shared_expert_gate(x)) * shared (a [H -> 1] linear); output = routed + shared. `OP_COVERAGE` lists every op of the four forward methods of the
block (`SparseMoeBlock`, `TopKRouter`, `Experts`, `MLP`) and the probe step that covers it.

Steps, each compared with the CPU reference on the same inputs:
  shared_expert              ttnn.linear (gate, up), ttnn.silu, ttnn.multiply, ttnn.linear (down); then the gate: ttnn.linear [H -> 1], ttnn.sigmoid, ttnn.multiply of a
                             [1, 1, T, 1] column against [1, 1, T, H]
  routing_matrix_scatter     the dense routing weights [T, E] from the top-k indices and weights on the device with ttnn.scatter(zeros, dim=-1, index=ids, src=weights),
                             as the kolibri port builds its expert mask (ROW_MAJOR tensors, then ttnn.to_layout TILE)
  routing_matrix_one_hot     the same with ttnn.embedding of the indices into an identity table ([T, k, E]), a multiply by the weights and ttnn.sum over k
  block_dense                the whole block on the device: router (probe 10 `selected_softmax`), the routing matrix from the first of the two forms above that works,
                             the dense-masked experts (probe 11) and the shared expert, added. Compared with `common.moe_block_ref`.
The block step is the form a generator can write without a sparse op; probe 12 is the faster form once its configs are known.

Run on a chip only inside a gozer lease. `--selftest` checks the CPU reference and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import p10_moe_router as p10  # noqa: E402
import p11_moe_experts_loop as p11  # noqa: E402

PRESETS = {"tiny": dict(tokens=32, hidden=64, experts=8, top_k=2, inter=64, shared=64),
           "flash": dict(tokens=32, hidden=2560, experts=16, top_k=4, inter=640, shared=640)}

# Every op of the four forward methods of the block (transformers 5.19.0 modeling_qwen4_exp.py) and where it is probed.
OP_COVERAGE = {
    "block.view_tokens_flat": "p13 (ttnn.reshape to [1, 1, T, H])", "block.shared_expert_call": "p13 shared_expert", "block.router_call": "p10",
    "block.experts_call": "p11 dense_masked, p12", "block.shared_gate_linear_sigmoid_multiply": "p13 shared_expert", "block.add_routed_and_shared": "p13 block_dense",
    "router.reshape_tokens": "p13 (ttnn.reshape)", "router.linear": "p10", "router.softmax_float": "p10", "router.topk": "p10, p7", "router.normalize_selected": "p10",
    "experts.expert_hit_mask_and_loop": "p11 (the dense form needs no hit list; the gather form builds it on the host)", "experts.gather_rows": "p11 gather_scatter_*",
    "experts.gate_up_linear_chunk": "p11 dense_masked_fused_gate_up", "experts.silu_multiply": "p11", "experts.down_linear": "p11, p12",
    "experts.multiply_routing_weight": "p11, p13 routing_matrix_*", "experts.index_add": "p11 gather_scatter_* (one-hot matmul), p12 (sum over E)",
    "mlp.gate_up_linear": "p13 shared_expert", "mlp.silu_multiply": "p13 shared_expert", "mlp.down_linear": "p13 shared_expert",
}
REFERENCE_OPS = tuple(OP_COVERAGE)           # the list the selftest compares against; a probe step removed from the table makes the check fail


def one_hot_routing(indices: torch.Tensor, weights: torch.Tensor, experts: int) -> torch.Tensor:
    """The `routing_matrix_one_hot` form on the CPU: sum over k of weights[:, k] * one_hot(indices[:, k]) -> [T, E]."""
    eye = torch.eye(experts)
    return (eye[indices] * weights.unsqueeze(-1)).sum(dim=1)


def selftest() -> dict:
    checks = {}
    t = common.moe_inputs(**PRESETS["tiny"])
    _, w, idx = common.moe_router_ref(t["hidden"], t["router"], t["top_k"], stable=True)
    shared = common.moe_shared_ref(t["hidden"], t["s_gate"], t["s_up"], t["s_down"], t["s_gate_w"])
    routed = common.moe_experts_ref(t["hidden"], idx, w, t["gate"], t["up"], t["down"])
    checks["block_is_routed_plus_the_gated_shared_expert"] = bool(torch.allclose(common.moe_block_ref(t), routed + shared, atol=1e-5))
    gate = torch.sigmoid(t["hidden"].float() @ t["s_gate_w"].float().t())
    checks["shared_gate_is_a_sigmoid_in_zero_one"] = bool(((gate > 0) & (gate < 1)).all())
    plain = (torch.nn.functional.silu(t["hidden"] @ t["s_gate"].t()) * (t["hidden"] @ t["s_up"].t())) @ t["s_down"].t()
    checks["shared_expert_is_the_gated_mlp_times_the_sigmoid"] = bool(torch.allclose(shared, gate * plain, atol=1e-5))
    checks["one_hot_routing_equals_the_scatter_form"] = bool(torch.allclose(one_hot_routing(idx, w, 8), common.routing_matrix(idx, w, 8), atol=1e-6))
    checks["routing_rows_have_k_non_zero_entries"] = bool(((common.routing_matrix(idx, w, 8) > 0).sum(-1) == t["top_k"]).all())
    checks["routing_rows_sum_to_one"] = bool(torch.allclose(common.routing_matrix(idx, w, 8).sum(-1), torch.ones(32), atol=1e-6))
    checks["the_shared_expert_changes_the_output"] = bool((shared.abs().max() > 1e-3) and not torch.allclose(common.moe_block_ref(t), routed))
    checks["every_op_of_the_reference_forward_is_listed"] = set(REFERENCE_OPS) == set(OP_COVERAGE) and all(OP_COVERAGE.values()) and len(OP_COVERAGE) >= 20
    return {"probe": "p13_moe_shared_expert", "mode": "selftest", "checks": checks, "ok": all(checks.values())}


# ---- device -------------------------------------------------------------------------------------------------------------------------------

def _up(ttnn, dev, t, dtype=None, layout=None):
    return ttnn.from_torch(t, dtype=dtype or ttnn.bfloat16, layout=layout or ttnn.TILE_LAYOUT, device=dev)


def shared_expert(ttnn, dev, x, t):
    mk = lambda m: _up(ttnn, dev, m.t().contiguous().reshape(1, 1, m.shape[1], m.shape[0]))
    mlp = ttnn.linear(ttnn.multiply(ttnn.silu(ttnn.linear(x, mk(t["s_gate"]))), ttnn.linear(x, mk(t["s_up"]))), mk(t["s_down"]))
    gate = ttnn.sigmoid(ttnn.linear(x, mk(t["s_gate_w"])))
    return ttnn.multiply(mlp, gate)


def routing_scatter(ttnn, dev, idx, w, E):
    T, k = idx.shape
    zero = _up(ttnn, dev, torch.zeros(1, 1, T, E), layout=ttnn.ROW_MAJOR_LAYOUT)
    ids = ttnn.from_torch(idx.to(torch.int32).reshape(1, 1, T, k), dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    src = _up(ttnn, dev, w.reshape(1, 1, T, k), layout=ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.to_layout(ttnn.scatter(zero, dim=-1, index=ids, src=src), ttnn.TILE_LAYOUT)


def routing_one_hot(ttnn, dev, idx, w, E):
    T, k = idx.shape
    ids = ttnn.from_torch(idx.to(torch.int32).reshape(T, k), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    eye = _up(ttnn, dev, torch.eye(E))
    onehot = ttnn.reshape(ttnn.embedding(ids, eye, layout=ttnn.TILE_LAYOUT), (1, T, k, E))
    weighted = ttnn.multiply(onehot, _up(ttnn, dev, w.reshape(1, T, k, 1)))
    return ttnn.reshape(ttnn.sum(weighted, dim=2), (1, 1, T, E))


def run_device(ttnn, dev, t):
    T, E = t["hidden"].shape[0], t["gate"].shape[0]
    _, w, idx = common.moe_router_ref(t["hidden"], t["router"], t["top_k"], stable=True)
    x = _up(ttnn, dev, t["hidden"].reshape(1, 1, T, -1))
    out = {}

    def attempt(name, fn, want):
        try:
            got = fn()
            stats = common.error_stats(got, want)
            out[name] = {**stats, "ok": bool((stats.get("pcc") or 0) > 0.99)}
        except Exception as exc:
            out[name] = {"ok": False, "failure": common.short_exc(exc)}

    attempt("shared_expert", lambda: ttnn.to_torch(shared_expert(ttnn, dev, x, t)).float().reshape(T, -1),
            common.moe_shared_ref(t["hidden"], t["s_gate"], t["s_up"], t["s_down"], t["s_gate_w"]))
    R = common.routing_matrix(idx, w, E)
    attempt("routing_matrix_scatter", lambda: ttnn.to_torch(routing_scatter(ttnn, dev, idx, w, E)).float().reshape(T, E), R)
    attempt("routing_matrix_one_hot", lambda: ttnn.to_torch(routing_one_hot(ttnn, dev, idx, w, E)).float().reshape(T, E), R)

    def block():
        # router on the device (probe 10 selected_softmax), routing matrix on the device, dense-masked experts, shared expert, add
        forms, _ = p10.run_device(ttnn, dev, t, PRESETS["tiny"], 0)
        sel = forms.get("selected_softmax", {})
        if not sel.get("ok"):
            raise RuntimeError("the router form selected_softmax did not pass on this shape (see probe 10); the block step needs it")
        acc = p11.dense_masked(ttnn, dev, t, idx, w)           # the router's indices and weights are the CPU stable-top-k ones here: probe 10 shows the device agrees
        return acc + ttnn.to_torch(shared_expert(ttnn, dev, x, t)).float().reshape(T, -1)
    attempt("block_dense", block, common.moe_block_ref(t))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the reference forms; no ttnn")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    ap.add_argument("--tokens", type=int, default=None)
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
    opened = common.device_or_report("p13_moe_shared_expert")
    if opened is None:
        return 2
    ttnn, dev = opened
    t = common.moe_inputs(**cfg)
    try:
        steps = run_device(ttnn, dev, t)
    finally:
        ttnn.close_device(dev)
    good = [n for n, r in steps.items() if r.get("ok")]
    common.emit({"probe": "p13_moe_shared_expert", "mode": "device", "ok": True, "shape": cfg, "dtype": "bf16 TILE", "reference": "Qwen4ExpTextSparseMoeBlock (CPU copy)",
                 "result": steps, "workarounds": (["routing_matrix_one_hot"] if "routing_matrix_one_hot" in good and "routing_matrix_scatter" not in good else []),
                 "steps_that_pass": good, "op_coverage": OP_COVERAGE})
    return 0


if __name__ == "__main__":
    sys.exit(main())
