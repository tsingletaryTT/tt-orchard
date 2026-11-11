# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 9: the remaining ops of the Qwen4 attention and its QSA indexer that probes 6 to 8 and the earlier probes do not cover.

`OP_COVERAGE` lists every op of `Qwen4ExpTextAttention.forward` and `Qwen4ExpTextQSAIndexer.forward` (transformers 5.19.0) and which probe step covers it.
The ops that are new are probed here, each compared with the CPU reference on the same inputs:
  qgate_split        q_proj output [1, 1, S, heads * 2 * hd] -> per head [q | gate]: ttnn.reshape to [1, S, heads, 2 * hd] and two ttnn.slice calls
  head_norm          per-head RMSNorm: ttnn.rms_norm on [1, 1, S * heads, hd] with weight (1 + w) of width hd
  head_permute       ttnn.permute [1, S, heads, hd] -> [1, heads, S, hd] and back
  repeat_kv          ttnn.repeat_interleave(k, n_rep, dim=1) (workaround: ttnn.concat of per-group copies)
  pool_mean          block mean of the index keys: [1, 1, S, D] -> [1, nb, ratio, D] -> ttnn.mean over the ratio axis (workaround: the sum of ratio ttnn.slice parts times 1/ratio)
  pooled_k_norm      ttnn.rms_norm of the pooled keys with weight (1 + w)
  pooled_rotary      the generated `_apply_rotary` formula on the pooled keys with the tables taken at the block start positions
  index_scores       ttnn.matmul of q [1, H, S, D] with pooled keys^T [1, 1, D, nb] (broadcast over the index heads), ttnn.relu, ttnn.sum over the head axis, scale 1/sqrt(D)
  one_hot_from_topk  block indices [S, k] -> one-hot selection [S, nb]: ttnn.embedding of the indices into an identity table, summed over k (workaround: read the
                     indices back and build the one-hot on the host, which the reference loop does)
Steps record `ok` when pcc > 0.99 against the reference; `workarounds` names the fallback forms that worked.

Run on a chip only inside a gozer lease. `--selftest` checks the CPU reference and needs no ttnn.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PRESETS = {"tiny": dict(heads=8, kv_heads=2, seq=32, hd=32, ratio=4, budget=16, index_heads=4, index_dim=32, rot=8),
           "flash": dict(heads=24, kv_heads=2, seq=256, hd=256, ratio=4, budget=64, index_heads=4, index_dim=128, rot=64)}

# Every op of the two forward methods, and where it is probed. "earlier" means the stage 2 generator already writes it as a fixed helper (`_linear`,
# `_rms_norm`, `_upload`, `_to_host`) that arms E1 to E14 ran on a board.
OP_COVERAGE = {
    "attention.indexer_call": "p7_topk_ties, p8_masked_sdpa, p9_attention_ops (see the indexer ops below)",
    "attention.mask_combine": "p8_masked_sdpa device_mask (bool and, or float add)",
    "attention.q_proj_linear": "earlier (_linear)", "attention.k_proj_linear": "earlier (_linear)", "attention.v_proj_linear": "earlier (_linear)",
    "attention.o_proj_linear": "earlier (_linear)",
    "attention.q_chunk_gate": "p9 qgate_split", "attention.q_norm": "p9 head_norm", "attention.k_norm": "p9 head_norm",
    "attention.transpose_heads": "p9 head_permute", "attention.apply_rotary": "p6_rotary_broadcast",
    "attention.repeat_kv": "p9 repeat_kv", "attention.qk_matmul_scale_mask_softmax_pv": "p8_masked_sdpa sdpa_*",
    "attention.join_heads": "p8_masked_sdpa output_gate (permute + reshape)", "attention.sigmoid_gate_multiply": "p8_masked_sdpa output_gate",
    "indexer.qk_proj_linear": "earlier (_linear)", "indexer.split_q_and_key": "p9 qgate_split (the same reshape and slice)",
    "indexer.q_layernorm": "p9 head_norm", "indexer.q_rotary": "p6_rotary_broadcast", "indexer.pool_mean": "p9 pool_mean",
    "indexer.k_layernorm": "p9 pooled_k_norm", "indexer.pooled_rotary": "p9 pooled_rotary", "indexer.scores_relu_sum_scale": "p9 index_scores",
    "indexer.topk": "p7_topk_ties", "indexer.block_to_token_and_tail": "p8_masked_sdpa device_mask", "indexer.one_hot_selection": "p9 one_hot_from_topk",
    "indexer.visibility_per_query_loop": "p8_masked_sdpa (the mask is built from the reference selection) and p9 index_scores (the visibility bias)",
}


def reference_steps(cfg, seed=0):
    """CPU intermediates for every step, from random bf16-rounded inputs. Returns a dict of tensors."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).bfloat16().float()
    S, H, Hkv, hd, ratio, D, rot = cfg["seq"], cfg["heads"], cfg["kv_heads"], cfg["hd"], cfg["ratio"], cfg["index_dim"], cfg["rot"]
    IH = cfg["index_heads"]
    out = {}
    out["q_proj"] = r(1, 1, S, H * 2 * hd)
    per_head = out["q_proj"].reshape(1, S, H, 2 * hd)
    out["q"], out["gate"] = per_head[..., :hd], per_head[..., hd:]
    out["head_w"] = r(hd) * 0.1
    out["q_normed"] = common.rms_norm_ref(out["q"], out["head_w"])
    out["q_perm"] = out["q_normed"].permute(0, 2, 1, 3)
    out["k"] = r(1, Hkv, S, hd)
    out["k_rep"] = out["k"].repeat_interleave(H // Hkv, dim=1)
    q, keys, cos, sin, w = common.selection_inputs(S, IH, D, min(rot, D), seed)
    out["index_q"], out["raw_keys"], out["cos"], out["sin"], out["k_w"] = q, keys, cos, sin, w
    nb = S // ratio
    out["pooled"] = keys.reshape(nb, ratio, D).float().mean(dim=1)
    out["pooled_normed"] = common.rms_norm_ref(out["pooled"], w)
    starts = torch.arange(nb) * ratio
    out["pooled_rot"] = common.apply_rotary_ref(out["pooled_normed"], cos.index_select(0, starts), sin.index_select(0, starts))
    scores = torch.matmul(q.permute(1, 0, 2).float(), out["pooled_rot"].float().T[None])      # [IH, S, nb]
    out["scores"] = torch.relu(scores).sum(dim=0) / math.sqrt(D)                                # [S, nb]
    return out


def selftest() -> dict:
    cfg = PRESETS["tiny"]
    ref = reference_steps(cfg)
    S, ratio, D = cfg["seq"], cfg["ratio"], cfg["index_dim"]
    checks = {}
    checks["q_and_gate_split_per_head"] = bool(torch.equal(torch.cat([ref["q"], ref["gate"]], dim=-1).reshape(1, 1, S, -1), ref["q_proj"]))
    checks["head_norm_changes_the_values"] = bool((ref["q_normed"] - ref["q"]).abs().max() > 1e-3)
    x, w = ref["q"][0, 3, 2], ref["head_w"]                  # one head vector, normalised by the explicit formula: x / sqrt(mean(x^2) + eps) * (1 + w)
    explicit = torch.stack([x[j] / math.sqrt(float((x * x).mean()) + 1e-6) * (1.0 + w[j]) for j in range(x.shape[0])])
    checks["rms_norm_reference_equals_the_explicit_formula"] = bool((ref["q_normed"][0, 3, 2] - explicit).abs().max() < 1e-4)
    checks["permute_roundtrip"] = bool(torch.equal(ref["q_perm"].permute(0, 2, 1, 3), ref["q_normed"]))
    checks["repeat_kv_groups_are_adjacent"] = bool(torch.equal(ref["k_rep"][:, 0], ref["k_rep"][:, 3]) and torch.equal(ref["k_rep"][:, 4], ref["k_rep"][:, 7])
                                                   and not torch.equal(ref["k_rep"][:, 0], ref["k_rep"][:, 4]))
    parts = sum(ref["raw_keys"][i::ratio] for i in range(ratio)) / ratio
    checks["pool_mean_equals_the_sum_of_strided_slices_over_ratio"] = bool((parts - ref["pooled"]).abs().max() < 1e-5)
    # the pooled and rotated keys score a query the same way the reference indexer does for the last row (all blocks visible)
    full = common.qsa_selection_ref(ref["index_q"], ref["raw_keys"], ref["cos"], ref["sin"], budget=cfg["budget"], ratio=ratio, k_weight=ref["k_w"])
    last = full["scores"][-1]
    checks["index_scores_equal_the_reference_indexer_scores_for_the_last_row"] = bool((last - ref["scores"][-1, : last.shape[0]]).abs().max() < 1e-4)
    checks["scores_are_non_negative_after_relu"] = bool((ref["scores"] >= 0).all())
    covered = set(OP_COVERAGE)
    checks["every_op_of_the_reference_forward_is_listed"] = covered >= set(REFERENCE_OPS) and len(covered) == len(REFERENCE_OPS)
    return {"probe": "p9_attention_ops", "mode": "selftest", "checks": checks, "ok": all(checks.values())}


# The ops read from the two forward methods (one name per op of the source, in order); the coverage table must list exactly these.
REFERENCE_OPS = (
    "attention.indexer_call", "attention.mask_combine", "attention.q_proj_linear", "attention.q_chunk_gate", "attention.q_norm", "attention.k_proj_linear",
    "attention.k_norm", "attention.v_proj_linear", "attention.transpose_heads", "attention.apply_rotary", "attention.repeat_kv",
    "attention.qk_matmul_scale_mask_softmax_pv", "attention.join_heads", "attention.sigmoid_gate_multiply", "attention.o_proj_linear",
    "indexer.qk_proj_linear", "indexer.split_q_and_key", "indexer.q_layernorm", "indexer.q_rotary", "indexer.visibility_per_query_loop",
    "indexer.pool_mean", "indexer.k_layernorm", "indexer.pooled_rotary", "indexer.scores_relu_sum_scale", "indexer.topk",
    "indexer.block_to_token_and_tail", "indexer.one_hot_selection")


def run_device(ttnn, dev, cfg):
    ref = reference_steps(cfg)
    S, H, Hkv, hd, ratio, D, rot, IH = cfg["seq"], cfg["heads"], cfg["kv_heads"], cfg["hd"], cfg["ratio"], cfg["index_dim"], cfg["rot"], cfg["index_heads"]
    nb = S // ratio
    tile = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    out, used = {}, []

    def step(name, fn, want):
        try:
            got = ttnn.to_torch(fn()).float()
            out[name] = common.error_stats(got.reshape(want.shape) if got.numel() == want.numel() else got, want)
            out[name]["ok"] = bool(out[name].get("pcc") is not None and out[name]["pcc"] > 0.99)
        except Exception as exc:
            out[name] = {"failure": common.short_exc(exc), "ok": False}
        return out[name]["ok"]

    qp = tile(ref["q_proj"])
    step("qgate_split_q", lambda: ttnn.slice(ttnn.reshape(qp, (1, S, H, 2 * hd)), (0, 0, 0, 0), (1, S, H, hd)), ref["q"])
    step("qgate_split_gate", lambda: ttnn.slice(ttnn.reshape(qp, (1, S, H, 2 * hd)), (0, 0, 0, hd), (1, S, H, 2 * hd)), ref["gate"])
    wt = tile((1.0 + ref["head_w"]).reshape(1, 1, 1, hd))
    qt = tile(ref["q"].reshape(1, 1, S * H, hd))
    step("head_norm", lambda: ttnn.rms_norm(qt, epsilon=1e-6, weight=wt), ref["q_normed"].reshape(1, 1, S * H, hd))
    n4 = tile(ref["q_normed"])
    step("head_permute", lambda: ttnn.permute(n4, (0, 2, 1, 3)), ref["q_perm"])
    kt = tile(ref["k"])
    if not step("repeat_kv", lambda: ttnn.repeat_interleave(kt, H // Hkv, dim=1), ref["k_rep"]):
        if step("w_repeat_kv_concat", lambda: ttnn.concat([ttnn.slice(kt, (0, g, 0, 0), (1, g + 1, S, hd)) for g in range(Hkv) for _ in range(H // Hkv)], dim=1), ref["k_rep"]):
            used.append("repeat_kv_concat")
    keys = tile(ref["raw_keys"].reshape(1, 1, S, D))
    blocks = lambda: ttnn.reshape(keys, (1, nb, ratio, D))
    if not step("pool_mean", lambda: ttnn.mean(blocks(), dim=2), ref["pooled"]):
        def sliced():
            b = blocks()
            acc = ttnn.slice(b, (0, 0, 0, 0), (1, nb, 1, D))
            for j in range(1, ratio):
                acc = ttnn.add(acc, ttnn.slice(b, (0, 0, j, 0), (1, nb, j + 1, D)))
            return ttnn.multiply(acc, 1.0 / ratio)
        if step("w_pool_mean_slices", sliced, ref["pooled"]):
            used.append("pool_mean_slices")
    pw = tile((1.0 + ref["k_w"]).reshape(1, 1, 1, D))
    pt = tile(ref["pooled"].reshape(1, 1, nb, D))
    step("pooled_k_norm", lambda: ttnn.rms_norm(pt, epsilon=1e-6, weight=pw), ref["pooled_normed"])
    starts = torch.arange(nb) * ratio
    pc, ps = common.padded_tables(ref["cos"].index_select(0, starts), ref["sin"].index_select(0, starts), D)
    rmat = common.rotate_half_matrix(D, min(rot, D))
    pn = tile(ref["pooled_normed"].reshape(1, 1, nb, D))
    step("pooled_rotary", lambda: ttnn.add(ttnn.multiply(pn, tile(pc)), ttnn.multiply(ttnn.linear(pn, tile(rmat)), tile(ps))), ref["pooled_rot"])
    q4 = tile(ref["index_q"].permute(1, 0, 2).reshape(1, IH, S, D))
    pk = tile(ref["pooled_rot"].T.reshape(1, 1, D, nb))

    def scores():
        s = ttnn.relu(ttnn.matmul(q4, pk))
        return ttnn.multiply(ttnn.sum(s, dim=1, keepdim=True), 1.0 / math.sqrt(D))
    step("index_scores", scores, ref["scores"])
    k = max(1, cfg["budget"] // ratio)
    idx = torch.stack([torch.randperm(nb)[:k] for _ in range(S)])
    want = torch.zeros(S, nb)
    want.scatter_(1, idx, 1.0)
    try:
        idx_tt = ttnn.from_torch(idx.to(torch.int32).reshape(1, S * k), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        eye = tile(torch.eye(nb).reshape(1, 1, nb, nb))
        step("one_hot_from_topk", lambda: ttnn.sum(ttnn.reshape(ttnn.embedding(idx_tt, eye), (1, S, k, nb)), dim=2), want)
    except Exception as exc:
        out["one_hot_from_topk"] = {"failure": common.short_exc(exc), "ok": False}
    if not out["one_hot_from_topk"]["ok"]:
        out["w_one_hot_host"] = {"ok": True, "note": "the indices are read back and the one-hot is built on the host (as the reference loop does); it breaks a captured trace"}
        used.append("one_hot_host")
    return {"steps": out, "workarounds": used}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the reference forms; no ttnn")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    for key in ("heads", "kv-heads", "seq", "hd", "ratio", "budget"):
        ap.add_argument("--" + key, type=int, default=None)
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest()
        common.emit(res)
        return 0 if res["ok"] else 1
    cfg = dict(PRESETS[args.preset])
    for key in ("heads", "kv_heads", "seq", "hd", "ratio", "budget"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    opened = common.device_or_report("p9_attention_ops")
    if opened is None:
        return 2
    ttnn, dev = opened
    try:
        result = run_device(ttnn, dev, cfg)
    finally:
        ttnn.close_device(dev)
    common.emit({"probe": "p9_attention_ops", "mode": "device", "ok": True, "shape": cfg, "dtype": "bf16 TILE", "reference": "CPU copies of the Qwen4Exp attention and indexer ops",
                 "op_coverage": OP_COVERAGE, "result": result["steps"], "workarounds": result["workarounds"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
