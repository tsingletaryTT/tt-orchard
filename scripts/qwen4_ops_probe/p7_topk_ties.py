# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Probe 7: `ttnn.topk` on the QSA block scores, where many scores tie at exactly 0.

The reference indexer scores each complete key block with `relu(q . block_key).sum(heads) / sqrt(dim)` and takes `scores.topk(min(budget / ratio, n_blocks))`.
With random weights many blocks have only negative products, so relu gives exact 0 and the scores tie; torch.topk then returns an implementation-defined
choice among the tied blocks, and attention over different blocks gives a different output. This probe asks what the device op returns in that case.

Per row i the valid blocks are the first (i + 1) // ratio entries; the rest of the padded row is -1 (below every relu score). For each k in --ks the probe runs
  ttnn.topk(scores, k, dim=-1, largest=True, sorted=True, stable=False) and stable=True
and compares both with torch.topk and with the lowest-index-first reference (`common.stable_topk_ref`, what `stable=True` documents) on the first
min(k, valid) entries of each row. Recorded per run: rows whose index lists are equal, rows whose index sets are equal, rows whose values are equal,
whether a second call returns the same indices (determinism), and the first k for which the op fails (a size limit). The CPU side records which tie rule
torch.topk follows on this machine and how often bf16 rounding changes a selection (scores that differ by less than the bf16 step).

`workarounds` names `stable_true` when the stable=True selection equals torch.topk's on every row. The CPU side shows that torch.topk does not follow the
lowest-index rule on tied scores (`torch_tie_rule` in the selftest result), so a device selection that follows it can differ from the reference on exactly
the rows with ties, and no device form is expected to reproduce torch's order. What the model needs is then a statement about tied rows: the chip and the
reference agree on every row without a tie at the boundary, and on tied rows the chip's choice is deterministic (`deterministic`) and follows one rule.

Run on a chip only inside a gozer lease. `--selftest` checks the CPU reference and needs no ttnn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

PRESETS = {"tiny": dict(seq=32, ratio=4, budget=16), "real": dict(seq=2560, ratio=4, budget=2048)}


def valid_counts(seq: int, ratio: int):
    return [(i + 1) // ratio for i in range(seq)]


def padded_scores(seq: int, ratio: int, zero_share: float, seed: int = 0):
    """(scores [seq, nb_pad] with -1 past the valid blocks of each row, valid counts). nb_pad is the block count of the last row, rounded up to 32."""
    nb_max = max(valid_counts(seq, ratio))
    nb_pad = max(32, -(-nb_max // 32) * 32)
    raw = common.tied_zero_scores(seq, nb_pad, zero_share, seed)
    n = torch.tensor(valid_counts(seq, ratio))
    raw[torch.arange(nb_pad)[None, :] >= n[:, None]] = -1.0
    return raw, n


def tie_rule_of_torch(scores: torch.Tensor, k: int, n: torch.Tensor):
    """Fraction of rows with ties at the boundary whose torch.topk indices are the lowest-index ones among the ties (1.0: torch follows the lowest-index rule)."""
    follows, rows = 0, 0
    for i in range(scores.shape[0]):
        ke = min(k, int(n[i]))
        if ke == 0:
            continue
        row = scores[i, : int(n[i])]
        t = row.topk(ke).indices.sort().values
        s = common.stable_topk_ref(row, ke)[1].sort().values
        if not torch.equal(t, s):
            rows += 1
        else:
            follows += 1
    return {"rows_equal_to_lowest_index_rule": follows, "rows_different": rows}


def selftest() -> dict:
    checks = {}
    scores, n = padded_scores(32, 4, zero_share=0.6)
    k = 4
    tied_rows = 0
    lowest_first = True
    for i in range(32):
        ke = min(k, int(n[i]))
        if ke == 0:
            continue
        row = scores[i, : int(n[i])]
        vals, idx = common.stable_topk_ref(row, ke)
        boundary = vals[-1]
        tied = (row == boundary).nonzero().flatten()
        tied_rows += int(len(tied) > 1)
        chosen_ties = [int(j) for j in idx if row[j] == boundary]
        lowest_first &= chosen_ties == [int(j) for j in tied[: len(chosen_ties)]]
    checks["rows_have_ties_at_the_boundary"] = tied_rows > 0
    checks["stable_reference_takes_the_lowest_indices_among_ties"] = bool(lowest_first)
    # rows without ties: torch.topk and the stable reference agree as sets
    clean = torch.rand(8, 16)
    checks["no_ties_means_the_two_agree"] = bool(torch.equal(clean.topk(4).indices.sort().values, common.stable_topk_ref(clean, 4)[1].sort().values))
    # all-tied row: the stable reference is exactly the first k indices
    checks["all_tied_row_gives_the_first_k"] = common.stable_topk_ref(torch.zeros(16), 4)[1].tolist() == [0, 1, 2, 3]
    # bf16 rounding can change a selection: two fp32 scores that differ by less than the bf16 step collapse into one value
    close = torch.tensor([1.0, 1.0 + 1e-4, 0.5, 0.25])
    checks["bf16_rounding_can_change_the_selection"] = (common.stable_topk_ref(close, 1)[1].tolist() != common.stable_topk_ref(close.bfloat16().float(), 1)[1].tolist())
    # padding at -1 is never selected while a valid block is left
    row = scores[20]
    checks["padding_is_never_chosen_before_a_valid_block"] = bool((common.stable_topk_ref(row, int(n[20]))[1] < int(n[20])).all())
    # the index bias form (workaround) reproduces the lowest-index rule: bias strictly below the bf16 step of the scores would vanish, so the bias is applied in fp32 and compared in fp32
    biased = scores[28, :8].clone() - torch.arange(8) * 1e-3
    checks["index_bias_form_matches_the_lowest_index_rule"] = bool(torch.equal(biased.topk(4).indices.sort().values, common.stable_topk_ref(scores[28, :8], 4)[1].sort().values))
    rule = tie_rule_of_torch(scores, k, n)
    return {"probe": "p7_topk_ties", "mode": "selftest", "checks": checks, "torch_tie_rule": rule, "ok": all(checks.values())}


def run_device(ttnn, dev, scores, n, ks):
    tt = ttnn.from_torch(scores.reshape(1, 1, *scores.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    runs = {}
    for k in ks:
        for stable in (False, True):
            key = f"k{k}_stable{int(stable)}"
            try:
                v1, i1 = ttnn.topk(tt, k, dim=-1, largest=True, sorted=True, stable=stable)
                v2, i2 = ttnn.topk(tt, k, dim=-1, largest=True, sorted=True, stable=stable)
                idx = ttnn.to_torch(i1).reshape(scores.shape[0], -1).long()
                val = ttnn.to_torch(v1).reshape(scores.shape[0], -1).float()
                idx2 = ttnn.to_torch(i2).reshape(scores.shape[0], -1).long()
            except Exception as exc:
                runs[key] = {"failure": common.short_exc(exc)}
                continue
            eq_list = eq_set = eq_val = eq_torch = rows = 0
            for i in range(scores.shape[0]):
                ke = min(k, int(n[i]))
                if ke == 0:
                    continue
                rows += 1
                row = scores[i, : int(n[i])]
                sv, si = common.stable_topk_ref(row, ke)
                tv, ti = row.topk(ke)
                got = idx[i, :ke]
                eq_list += int(torch.equal(got, si))
                eq_set += int(torch.equal(got.sort().values, si.sort().values))
                eq_torch += int(torch.equal(got.sort().values, ti.sort().values))
                eq_val += int(torch.allclose(val[i, :ke], sv, atol=1e-2))
            runs[key] = {"rows": rows, "index_list_equals_lowest_index_rule": eq_list, "index_set_equals_lowest_index_rule": eq_set,
                         "index_set_equals_torch_topk": eq_torch, "values_equal": eq_val, "deterministic": bool(torch.equal(idx, idx2))}
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true", help="CPU-only check of the reference forms; no ttnn")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    ap.add_argument("--seq", type=int, default=None, help="query rows (tokens)")
    ap.add_argument("--ratio", type=int, default=None, help="tokens per block")
    ap.add_argument("--budget", type=int, default=None, help="token budget; k = budget / ratio")
    ap.add_argument("--ks", type=int, nargs="*", default=None, help="k values to try (default: budget/ratio, and for the real preset also 32 64 128 256)")
    ap.add_argument("--zero-share", type=float, default=0.6, help="share of block scores that are exactly 0")
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest()
        common.emit(res)
        return 0 if res["ok"] else 1
    cfg = dict(PRESETS[args.preset])
    for key in cfg:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    k_full = cfg["budget"] // cfg["ratio"]
    ks = args.ks or sorted({k_full} | ({32, 64, 128, 256} if args.preset == "real" else set()))
    opened = common.device_or_report("p7_topk_ties")
    if opened is None:
        return 2
    ttnn, dev = opened
    scores, n = padded_scores(cfg["seq"], cfg["ratio"], args.zero_share)
    try:
        runs = run_device(ttnn, dev, scores, n, ks)
    finally:
        ttnn.close_device(dev)
    rows_with_ties = int(sum(1 for i in range(cfg["seq"]) if int(n[i]) and len(scores[i, : int(n[i])].unique()) < int(n[i])))
    stable_ok = [k for k in ks if runs.get(f"k{k}_stable1", {}).get("rows") and
                 runs[f"k{k}_stable1"]["index_set_equals_torch_topk"] == runs[f"k{k}_stable1"]["rows"]]
    common.emit({"probe": "p7_topk_ties", "mode": "device", "ok": True, "shape": {**cfg, "ks": ks, "zero_share": args.zero_share},
                 "dtype": "bf16 TILE", "reference": "torch.topk and the lowest-index-first rule (CPU)", "cpu_tie_rule_of_torch": tie_rule_of_torch(scores, k_full, n),
                 "rows_with_ties": rows_with_ties, "result": runs,
                 "workarounds": (["stable_true"] if stable_ok else []), "ks_where_stable_true_equals_torch_topk": stable_ok})
    return 0


if __name__ == "__main__":
    sys.exit(main())
