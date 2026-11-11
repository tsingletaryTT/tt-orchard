# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 10: the router of the Qwen4 sparse MoE block (`Qwen4ExpTextTopKRouter`) from ttnn ops (issue 723).

The reference: logits = hidden @ router^T; probs = softmax(logits) in fp32; (values, indices) = topk(probs, k); with `norm_topk_prob` (the config
default) the selected probs are divided by their sum. Flash-Next: 512 experts, k = 10, hidden 2560. The tiny checkpoint: 8 experts, k = 2, hidden 64.

Three device forms are compared with the CPU reference:
  reference_form             ttnn.linear, ttnn.softmax over all experts, ttnn.topk(stable=True) on the probs, then the selected probs times 1 / their sum
  selected_softmax           ttnn.linear, ttnn.topk(stable=True) on the logits, ttnn.softmax over the k selected logits. The same numbers: softmax is
                             monotone, so the top-k of the probs and of the logits are the same entries, and probs / sum(selected probs) is the softmax of the
                             selected logits. It skips the softmax over 512 columns.
  selected_softmax_manual    the same with exp(v - v0) / sum(exp(v - v0)) written out (v0 is the first, largest, selected logit), for the case that
                             ttnn.softmax mishandles the tile padding when E or k is below the tile width.
`ttnn.topk` is tried on the unpadded tensor first and, when it raises, on the tensor padded to a multiple of 32 columns with a value below every real
entry (`topk_padded` in `workarounds`, as in the kolibri port's `padded_topk`).

Recorded per form: rows whose selected expert set equals the fp32 CPU reference (bf16 changes some selections where two scores are closer than the bf16
step), rows whose set equals the CPU top-k of the DEVICE logits (this isolates the topk op from the linear's rounding: it should be all rows), the error
of the weights on the rows whose sets agree, and, with tied experts (experts 2p and 2p + 1 share one router row), the rows where exactly one of a tied pair
was selected and the lowest index won (the `stable=True` rule that probe 7 measured on the QSA block scores).

Run on a chip only inside a gozer lease. `--selftest` checks the CPU reference and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PRESETS = {"tiny": dict(tokens=32, hidden=64, experts=8, top_k=2, inter=64, shared=64),
           "flash": dict(tokens=32, hidden=2560, experts=512, top_k=10, inter=640, shared=640)}

OP_COVERAGE = {"router.linear": "p10 all forms", "router.softmax_float": "p10 reference_form, selected_softmax", "router.topk": "p10 (and p7 for the tie rule)",
               "router.normalize_selected": "p10 reference_form (sum and reciprocal), selected_softmax (softmax over k)"}


def selected_softmax_ref(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """softmax over the k selected logits [T, k] (the `selected_softmax` forms): equal to the reference's normalised probs."""
    return torch.softmax(logits.gather(1, indices), dim=-1)


def manual_selected_softmax_ref(selected: torch.Tensor) -> torch.Tensor:
    """exp(v - v0) / sum(exp(v - v0)) with v0 the first (largest) selected logit: `selected_softmax_manual` on the CPU."""
    e = torch.exp(selected - selected[:, :1])
    return e / e.sum(dim=-1, keepdim=True)


def tie_report(indices: torch.Tensor, pairs: int) -> dict:
    """With experts 2p and 2p + 1 tied for every token: the rows where exactly one of the pair is selected, and how many of those took the lower index."""
    rows = won = 0
    for p in range(pairs):
        lo, hi = 2 * p, 2 * p + 1
        has_lo, has_hi = (indices == lo).any(dim=-1), (indices == hi).any(dim=-1)
        single = has_lo ^ has_hi
        rows += int(single.sum())
        won += int((single & has_lo).sum())
    return {"rows_with_one_of_a_tied_pair": rows, "rows_where_the_lower_index_won": won}


def selftest() -> dict:
    checks = {}
    t = common.moe_inputs(**{**PRESETS["tiny"]})
    logits, weights, indices = common.moe_router_ref(t["hidden"], t["router"], t["top_k"])
    checks["selected_weights_are_normalised_to_one"] = bool(torch.allclose(weights.sum(-1), torch.ones(weights.shape[0]), atol=1e-6))
    _, w_raw, _ = common.moe_router_ref(t["hidden"], t["router"], t["top_k"], norm_topk_prob=False)
    checks["without_norm_topk_prob_the_weights_sum_below_one"] = bool((w_raw.sum(-1) < 0.9999).all())
    checks["softmax_over_the_selected_logits_equals_the_normalised_probs"] = bool(torch.allclose(selected_softmax_ref(logits, indices), weights, atol=1e-6))
    sel = logits.gather(1, indices)
    checks["manual_exp_form_equals_softmax"] = bool(torch.allclose(manual_selected_softmax_ref(sel), weights, atol=1e-6))
    checks["topk_of_probs_and_of_logits_are_the_same_set"] = bool(torch.equal(indices.sort(-1).values, torch.topk(logits, t["top_k"], dim=-1).indices.sort(-1).values))
    checks["indices_are_distinct_per_row"] = bool(all(len(set(r.tolist())) == t["top_k"] for r in indices))
    # ties: experts 2p and 2p+1 share a router row, so their logits are equal for every token; the stable reference takes the lower index
    tt = common.moe_inputs(**{**PRESETS["tiny"], "tied_pairs": 2}, seed=1)
    _, _, ti = common.moe_router_ref(tt["hidden"], tt["router"], tt["top_k"], stable=True)
    rep = tie_report(ti, 2)
    checks["tied_pairs_exist_in_the_selection"] = rep["rows_with_one_of_a_tied_pair"] > 0
    checks["the_stable_reference_always_takes_the_lower_index_of_a_tied_pair"] = rep["rows_with_one_of_a_tied_pair"] == rep["rows_where_the_lower_index_won"]
    # bf16 rounding of the logits can change a selection: count the rows (informational; it checks nothing on the device)
    big = common.moe_inputs(tokens=256, hidden=64, experts=64, top_k=4, inter=64, shared=64, seed=2)
    l32, _, i32 = common.moe_router_ref(big["hidden"], big["router"], 4, stable=True)
    ib = common.stable_topk_ref(l32.bfloat16().float(), 4)[1]
    changed = int((i32.sort(-1).values != ib.sort(-1).values).any(dim=-1).sum())
    flash = PRESETS["flash"]
    checks["flash_shape_selects_ten_distinct_of_512"] = bool(
        common.moe_router_ref(torch.randn(4, 16), torch.randn(flash["experts"], 16), flash["top_k"])[2].shape == (4, 10))
    return {"probe": "p10_moe_router", "mode": "selftest", "checks": checks, "tie_report_tiny": rep, "rows_changed_by_bf16_logits_of_256": changed,
            "ok": all(checks.values())}


# ---- device -------------------------------------------------------------------------------------------------------------------------------

def _up(ttnn, dev, t, dtype=None):
    return ttnn.from_torch(t, dtype=dtype or ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def _topk(ttnn, dev, x, k, fill, notes):
    """ttnn.topk(stable=True) on x [1, 1, T, E]; when it raises, again on x padded to a multiple of 32 columns with `fill`."""
    try:
        return ttnn.topk(x, k, dim=-1, largest=True, sorted=True, stable=True)
    except Exception as exc:
        notes.append("topk on the unpadded tensor raised: " + common.short_exc(exc, 200))
    e = x.shape[-1]
    pad = (-e) % 32 or 32
    x2 = ttnn.pad(x, ((0, 0), (0, 0), (0, 0), (0, pad)), fill)
    notes.append("topk_padded")
    return ttnn.topk(x2, k, dim=-1, largest=True, sorted=True, stable=True)


def run_device(ttnn, dev, t, cfg, pairs):
    T, E, k = t["hidden"].shape[0], t["router"].shape[0], t["top_k"]
    x = _up(ttnn, dev, t["hidden"].reshape(1, 1, T, -1))
    w = _up(ttnn, dev, t["router"].t().contiguous().reshape(1, 1, -1, E))
    logits = ttnn.linear(x, w)
    l_dev = ttnn.to_torch(logits).float().reshape(T, E)
    _, w_ref, i_ref = common.moe_router_ref(t["hidden"], t["router"], k, stable=True)
    i_cpu_of_dev = common.stable_topk_ref(l_dev, k)[1]
    out, notes = {}, []

    def finish(name, values, idx, extra=None):
        i_dev = ttnn.to_torch(idx).reshape(T, -1)[:, :k].long()
        v_dev = ttnn.to_torch(values).float().reshape(T, -1)[:, :k]
        eq_ref = (i_dev.sort(-1).values == i_ref.sort(-1).values).all(-1)
        eq_cpu = (i_dev.sort(-1).values == i_cpu_of_dev.sort(-1).values).all(-1)
        order_d, order_r = i_dev.argsort(-1), i_ref.argsort(-1)
        err = None
        if bool(eq_ref.any()):
            err = common.error_stats(v_dev.gather(1, order_d)[eq_ref], w_ref.gather(1, order_r)[eq_ref])
        out[name] = {"rows": T, "rows_set_equals_fp32_reference": int(eq_ref.sum()), "rows_set_equals_cpu_topk_of_device_logits": int(eq_cpu.sum()),
                     "weights": err, "ties": tie_report(i_dev, pairs) if pairs else None, **(extra or {})}
        out[name]["ok"] = bool(err and (err.get("pcc") or 0) > 0.99 and int(eq_cpu.sum()) == T)

    def attempt(name, fn):
        try:
            fn()
        except Exception as exc:
            out[name] = {"ok": False, "failure": common.short_exc(exc)}

    def reference_form():
        probs = ttnn.softmax(logits, dim=-1)
        values, idx = _topk(ttnn, dev, probs, k, -1.0, notes)
        total = ttnn.sum(values, dim=-1, keepdim=True)
        finish("reference_form", ttnn.multiply(values, ttnn.reciprocal(total)), idx)

    def selected_softmax():
        values, idx = _topk(ttnn, dev, logits, k, -1.0e4, notes)
        finish("selected_softmax", ttnn.softmax(values, dim=-1), idx)

    def selected_softmax_manual():
        values, idx = _topk(ttnn, dev, logits, k, -1.0e4, notes)
        v0 = ttnn.slice(values, (0, 0, 0, 0), (1, 1, T, 1))
        e = ttnn.exp(ttnn.subtract(values, v0))
        finish("selected_softmax_manual", ttnn.multiply(e, ttnn.reciprocal(ttnn.sum(e, dim=-1, keepdim=True))), idx)

    attempt("reference_form", reference_form)
    attempt("selected_softmax", selected_softmax)
    attempt("selected_softmax_manual", selected_softmax_manual)
    return out, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the reference forms; no ttnn")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    ap.add_argument("--tokens", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--experts", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--tied-pairs", type=int, default=2, help="pairs of experts (2p, 2p+1) that share one router row")
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest()
        common.emit(res)
        return 0 if res["ok"] else 1
    cfg = dict(PRESETS[args.preset])
    for key in ("tokens", "hidden", "experts", "top_k"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    opened = common.device_or_report("p10_moe_router")
    if opened is None:
        return 2
    ttnn, dev = opened
    t = common.moe_inputs(**cfg, tied_pairs=args.tied_pairs)
    try:
        forms, notes = run_device(ttnn, dev, t, cfg, args.tied_pairs)
    finally:
        ttnn.close_device(dev)
    good = [n for n, r in forms.items() if r.get("ok")]
    common.emit({"probe": "p10_moe_router", "mode": "device", "ok": True, "shape": cfg, "dtype": "bf16 TILE",
                 "reference": "Qwen4ExpTextTopKRouter (CPU copy), lowest-index tie rule", "result": forms, "notes": notes,
                 "workarounds": sorted(set(notes) & {"topk_padded"} | ({"manual_softmax"} if "selected_softmax_manual" in good and "selected_softmax" not in good else set())),
                 "forms_that_pass": good})
    return 0


if __name__ == "__main__":
    sys.exit(main())
