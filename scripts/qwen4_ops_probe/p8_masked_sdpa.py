# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 8: attention over a per-query block-selection mask on a chip: `ttnn.transformer.scaled_dot_product_attention` with `attn_mask`, grouped
key/value heads and the sigmoid output gate of Qwen4ExpTextAttention.

The reference attention adds the indexer's selection mask to the causal mask (the selected blocks' tokens plus the tail tokens stay visible; every other
token gets the lowest float), runs softmax(q k^T * scale + mask) v with the key/value heads repeated for the query heads, joins the heads and multiplies by
sigmoid(gate). The tt-metal binding documents `attn_mask` as [b x nqh x s x s] (batch and head dims may be 1), K and V as [b x nkv x s x dh], and says to pass
is_causal=False with a mask. Steps, each compared with the CPU reference:
  sdpa_kv_expanded   K and V repeated to the query head count on the host, mask [1, 1, S, S] bf16 additive, is_causal=False
  sdpa_gqa_native    K and V with the key/value head count (the op's own grouped-query path)
  sdpa_mask_per_head the mask expanded to [1, heads, S, S]
  device_mask        the additive mask built on the device from the block selection: one-hot selection [S, nb] x expansion matrix [nb, S] (ttnn.matmul),
                     plus the host-built tail mask, then (m - 1) * 30000 with ttnn.subtract and ttnn.multiply; compared with the host mask pattern
  output_gate        ttnn.permute, ttnn.reshape to [1, 1, S, heads * dh] and ttnn.multiply with ttnn.sigmoid(gate)
`workarounds` names the forms that worked when the first one in its group failed. The selection comes from the CPU reference indexer
(`common.qsa_selection_ref`), so the mask has the real structure (blocks of --ratio tokens, top --budget / ratio blocks, tail).

Run on a chip only inside a gozer lease. `--selftest` checks the CPU reference and needs no ttnn.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PRESETS = {"tiny": dict(heads=8, kv_heads=2, seq=32, hd=32, ratio=4, budget=16, index_heads=4, index_dim=32, rot=8),
           "flash": dict(heads=24, kv_heads=2, seq=256, hd=256, ratio=4, budget=64, index_heads=4, index_dim=128, rot=64),
           # Flash-Next widths with the real token budget and a prompt past it (2048 / 4 = 512 blocks of 4; the masks are S x S): the shapes the generated block meets at the real budget
           "flash-long": dict(heads=24, kv_heads=2, seq=2560, hd=256, ratio=4, budget=2048, index_heads=4, index_dim=128, rot=64)}
NEG = -30000.0                     # the additive mask value on the device (bf16 holds it; exp underflows to 0)


def expansion_matrix(nb: int, ratio: int, seq: int) -> torch.Tensor:
    """E [nb, seq]: row b is 1 on tokens b * ratio .. (b + 1) * ratio - 1. A one-hot block selection times E gives the selected tokens."""
    e = torch.zeros(nb, seq)
    for b in range(nb):
        e[b, b * ratio:(b + 1) * ratio] = 1.0
    return e


def selection_parts(seq, ratio, budget, index_heads, index_dim, rot, seed=0):
    """(mask [S,S] bool, one-hot block selection [S, nb_max], tail mask [S,S] 0/1) from the reference indexer."""
    q, keys, cos, sin, w = common.selection_inputs(seq, index_heads, index_dim, min(rot, index_dim), seed)
    out = common.qsa_selection_ref(q, keys, cos, sin, budget=budget, ratio=ratio, k_weight=w)
    nb_max = max(1, max(out["n_blocks"]))
    sel = torch.zeros(seq, nb_max)
    tail = torch.zeros(seq, seq)
    for i, blocks in enumerate(out["blocks"]):
        for b in blocks:
            sel[i, b] = 1.0
        tail[i, (out["n_blocks"][i] * ratio):i + 1] = 1.0
    return out["mask"], sel, tail


def reference_attention(q, k, v, mask, gate, scale, n_rep):
    """q [1, Hq, S, D], k and v [1, Hkv, S, D], mask [S, S] bool, gate [1, 1, S, Hq * D]. Returns (out [1, 1, S, Hq * D], attn_heads [1, Hq, S, D])."""
    kk, vv = k.repeat_interleave(n_rep, dim=1), v.repeat_interleave(n_rep, dim=1)
    scores = torch.matmul(q, kk.transpose(2, 3)) * scale
    scores = scores.masked_fill(~mask[None, None], float("-inf"))
    heads = torch.matmul(torch.softmax(scores.float(), dim=-1), vv.float())
    joined = heads.transpose(1, 2).reshape(1, 1, q.shape[2], -1)
    return joined * torch.sigmoid(gate), heads


def make_inputs(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).bfloat16().float()
    S, hd = cfg["seq"], cfg["hd"]
    return r(1, cfg["heads"], S, hd), r(1, cfg["kv_heads"], S, hd), r(1, cfg["kv_heads"], S, hd), r(1, 1, S, cfg["heads"] * hd)


def selftest() -> dict:
    cfg = PRESETS["tiny"]
    S, ratio = cfg["seq"], cfg["ratio"]
    mask, sel, tail = selection_parts(S, ratio, cfg["budget"], cfg["index_heads"], cfg["index_dim"], cfg["rot"])
    checks = {}
    e = expansion_matrix(sel.shape[1], ratio, S)
    rebuilt = ((sel @ e + tail) > 0)
    checks["one_hot_selection_times_expansion_plus_tail_is_the_reference_mask"] = bool(torch.equal(rebuilt, mask))
    checks["blocks_and_tail_do_not_overlap"] = bool(((sel @ e) * tail).sum() == 0 and ((sel @ e) <= 1).all())
    checks["the_mask_stays_inside_the_causal_triangle"] = bool(torch.equal(mask, mask & torch.ones(S, S, dtype=torch.bool).tril()))
    checks["every_query_sees_at_least_one_token"] = bool(mask.any(dim=1).all())
    checks["the_mask_is_sparse_past_the_budget"] = bool(mask[-1].sum() < S)
    q, k, v, gate = make_inputs(cfg)
    n_rep = cfg["heads"] // cfg["kv_heads"]
    scale = cfg["hd"] ** -0.5
    out, heads = reference_attention(q, k, v, mask, gate, scale, n_rep)
    full = torch.ones(S, S, dtype=torch.bool).tril()
    out_full, heads_full = reference_attention(q, k, v, full, gate, scale, n_rep)
    ref_sdpa = F.scaled_dot_product_attention(q, k.repeat_interleave(n_rep, 1), v.repeat_interleave(n_rep, 1), is_causal=True, scale=scale)
    checks["reference_matches_torch_sdpa_for_a_causal_mask"] = bool((heads_full - ref_sdpa).abs().max() < 1e-4)
    add = torch.where(mask, 0.0, NEG)[None, None]
    ref_add = F.scaled_dot_product_attention(q, k.repeat_interleave(n_rep, 1), v.repeat_interleave(n_rep, 1), attn_mask=add, scale=scale)
    checks["additive_minus_30000_mask_equals_the_bool_mask"] = bool((heads - ref_add).abs().max() < 1e-4)
    v2 = v.clone()
    unseen_last = ~mask[-1]
    v2[..., unseen_last, :] += 100.0
    _o, h2 = reference_attention(q, k, v2, mask, gate, scale, n_rep)
    checks["tokens_outside_the_selection_do_not_change_the_last_row"] = bool((h2[:, :, -1] - heads[:, :, -1]).abs().max() < 1e-4)
    checks["the_selection_changes_the_output_against_full_attention"] = bool((heads[:, :, -1] - heads_full[:, :, -1]).abs().max() > 1e-3)
    checks["output_gate_scales_by_a_sigmoid"] = bool(((out / torch.sigmoid(gate)) - heads.transpose(1, 2).reshape(1, 1, S, -1)).abs().max() < 1e-4)
    # not a check: a query whose own block is not selected does not see itself (the reference mask is not guaranteed to contain the diagonal)
    return {"probe": "p8_masked_sdpa", "mode": "selftest", "checks": checks, "queries_that_do_not_see_themselves": int((~mask.diagonal()).sum()),
            "ok": all(checks.values())}


def run_device(ttnn, dev, cfg):
    S, hd, ratio = cfg["seq"], cfg["hd"], cfg["ratio"]
    n_rep = cfg["heads"] // cfg["kv_heads"]
    scale = hd ** -0.5
    mask, sel, tail = selection_parts(S, ratio, cfg["budget"], cfg["index_heads"], cfg["index_dim"], cfg["rot"])
    q, k, v, gate = make_inputs(cfg)
    ref_out, ref_heads = reference_attention(q, k, v, mask, gate, scale, n_rep)
    tile = lambda t, dt=ttnn.bfloat16: ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
    add = torch.where(mask, 0.0, NEG)[None, None]
    qt, kt, vt = tile(q), tile(k), tile(v)
    kx, vx = tile(k.repeat_interleave(n_rep, 1)), tile(v.repeat_interleave(n_rep, 1))
    out, used = {}, []

    def step(name, fn, ref, pattern=None):
        try:
            got = ttnn.to_torch(fn()).float()
            if pattern is not None:
                out[name] = {"pattern_equal": bool(torch.equal(pattern(got), ref)), "mismatches": int((pattern(got) != ref).sum())}
                out[name]["ok"] = out[name]["pattern_equal"]
            else:
                out[name] = common.error_stats(got, ref)
                out[name]["ok"] = bool(out[name].get("pcc") is not None and out[name]["pcc"] > 0.99)
        except Exception as exc:
            out[name] = {"failure": common.short_exc(exc), "ok": False}
        return out[name]["ok"]

    sdpa = lambda a, b, c, m: ttnn.transformer.scaled_dot_product_attention(a, b, c, attn_mask=m, is_causal=False, scale=scale)
    step("sdpa_kv_expanded", lambda: sdpa(qt, kx, vx, tile(add)), ref_heads)
    step("sdpa_gqa_native", lambda: sdpa(qt, kt, vt, tile(add)), ref_heads)
    step("sdpa_mask_per_head", lambda: sdpa(qt, kx, vx, tile(add.expand(1, cfg["heads"], S, S))), ref_heads)
    e, sel_t, tail_t = tile(expansion_matrix(sel.shape[1], ratio, S)), tile(sel), tile(tail)

    def built():
        m = ttnn.add(ttnn.matmul(sel_t, e), tail_t)
        return ttnn.multiply(ttnn.subtract(m, 1.0), 30000.0)
    step("device_mask", built, mask, pattern=lambda g: g.reshape(S, S) > NEG / 2)
    heads_t = tile(ref_heads)
    step("output_gate", lambda: ttnn.multiply(ttnn.reshape(ttnn.permute(heads_t, (0, 2, 1, 3)), (1, 1, S, cfg["heads"] * hd)), ttnn.sigmoid(tile(gate))), ref_out)
    if not out["sdpa_kv_expanded"]["ok"]:
        used = [n for n in ("sdpa_gqa_native", "sdpa_mask_per_head") if out[n]["ok"]]      # the forms that worked where the first form did not
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
    opened = common.device_or_report("p8_masked_sdpa")
    if opened is None:
        return 2
    ttnn, dev = opened
    try:
        result = run_device(ttnn, dev, cfg)
    finally:
        ttnn.close_device(dev)
    common.emit({"probe": "p8_masked_sdpa", "mode": "device", "ok": True, "shape": cfg, "dtype": "bf16 TILE", "reference": "CPU masked attention with the reference indexer's selection",
                 "result": result["steps"], "workarounds": result["workarounds"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
