# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The bar for a text model's chip run, set from CPU-only floors (issue 214, items 2 and 3; issue 206).

`orchard/noise_floor.py` refuses to set bars for a chip run from two CPU runs: the floor holds no chip arithmetic, and the good chip run fails bars derived that
way. `orchard/first_divergence.py` with its default margin of 3 reports block 0 as the first to leave on every chip run of MiniCPM3-4B. This module states a rule
that works with what a text run has, and says what the rule cannot show.

The runs. `reference` is the float32 CPU run. `floor` is the same code in bfloat16 on the CPU (the chip's own dtype, with CPU accumulation order); a second
bfloat16 run from independent code may be given as a cross-check. `chip` is the chip run recorded with the same array names (orchard/text_reference.py).

The rule, array by array (every `layer_NN`, every `layer_NN.attn` both runs hold, `prefill_logits`, every `dec_logits_K`, `final_norm`):

    floor_i     = relative L2 error of the bfloat16 CPU run against the float32 reference at array i (the larger of the two floor runs when two are given)
    tolerance_i = CHIP_MARGIN * max(floor_i, MIN_FLOOR)
    array i leaves when the chip run's relative L2 error against the reference is above tolerance_i

The run passes the numeric check when no array leaves, and the teacher-forced next-token agreement is at least AGREE_MIN_FRACTION, and the perplexity of
the chip on a fixed text window is within PPL_REL_TOL of the float32 reference's.

How CHIP_MARGIN was set. It is a choice, taken from one measurement: on MiniCPM3-4B the chip's error at block outputs was 2.8 to 6 times the difference between two
bfloat16 CPU implementations (issue 214), and 1.0 to 2.7 times the bfloat16 CPU error against float32 on Granite-4.0-H-1B (issue 206). 8 sits above both. It is
provisional. A third model whose chip run exceeds it moves the number, and the run says so in its limits.

What the rule cannot show:
  * The floor holds no chip arithmetic. The bar is a multiple of CPU bfloat16 error and not a measured chip noise level, so a pass says the chip's error is within 8
    times what bfloat16 alone does on a CPU, and nothing about whether the chip's own error has a different cause.
  * A defect smaller than the tolerance at an array passes. At layer 30 of MiniCPM3-4B the tolerance is a few percent of relative L2 error.
  * The rule does not tell a defect from chip numerics when an array leaves. The first array to leave locates where to look, and the stage reports it.
  * Teacher-forced agreement and perplexity are measured on one prompt and one text window. They say nothing about long contexts or other prompts.
"""
from __future__ import annotations

import math

import numpy as np

from orchard import audio_reference as ar
from orchard import text_reference as tr

CHIP_MARGIN = 8.0                 # choice: see the module text
MIN_FLOOR = 1e-6                  # choice: a floor of zero would give a bar nobody can meet (the same lift as first_divergence)
AGREE_MIN_FRACTION = 0.9          # choice: teacher-forced next-token matches over the decode steps
PPL_REL_TOL = 0.02                # choice: the chip's perplexity within 2 percent of the float32 reference's; MiniCPM3-4B measured 0.6 percent below

RULE = (f"tolerance at each array = {CHIP_MARGIN:g} times the bfloat16 CPU run's relative L2 error against the float32 reference at that array "
        f"(at least {MIN_FLOOR:g}); teacher-forced agreement at least {AGREE_MIN_FRACTION:.0%}; perplexity within {PPL_REL_TOL:.0%} of the reference's")
CANNOT_SHOW = (
    "The floor is two CPU runs and holds no chip arithmetic, so the bar is a multiple of CPU bfloat16 error. It is not a measured chip noise level.",
    f"A chip error below {CHIP_MARGIN:g} times the CPU bfloat16 error at an array passes, even when a defect caused it.",
    "When an array leaves its tolerance the rule locates the first one; it does not say whether the cause is a defect or chip numerics.",
    "Agreement and perplexity are measured on one prompt and one text window; nothing here covers long contexts or other prompts.",
)


def _entries(run, kind):
    return {n for (k, n) in run.entries if k == kind}


def _rel(ref_arr, other_arr):
    m = ar.metrics(np.asarray(ref_arr), np.asarray(other_arr))
    return m["rel_l2"] if m.get("comparable") else None


def floor_summary(ref, floors) -> dict:
    """Per block, the relative L2 error of each floor run against the reference at `layer_NN`, and the worst and median over blocks for each floor run.
    Used by stage 1 and recomputed by its gate."""
    out = {}
    for label, run in floors.items():
        per = {}
        for name in sorted(n for n in _entries(ref, "layer") if "." not in n):
            if ("layer", name) in run.entries:
                per[name] = _rel(ref.array("layer", name), run.array("layer", name))
        vals = [v for v in per.values() if v is not None]
        out[label] = {"blocks": per, "worst": max(vals) if vals else None, "median": float(np.median(vals)) if vals else None}
    return out


def _floor_at(ref, floors, kind, name):
    vals = []
    for run in floors.values():
        if (kind, name) in run.entries:
            v = _rel(ref.array(kind, name), run.array(kind, name))
            if v is not None:
                vals.append(v)
    return max(vals) if vals else None


def _entry(ref, floors, chip, kind, name) -> dict:
    floor = _floor_at(ref, floors, kind, name)
    got = _rel(ref.array(kind, name), chip.array(kind, name)) if (kind, name) in chip.entries else None
    if floor is None:
        return {"kind": kind, "name": name, "rel_l2": got, "floor": None, "tolerance": None, "ratio_to_floor": None, "leaves": True,
                "why": "no floor run holds this array, so no bar can be set"}
    tol = CHIP_MARGIN * max(floor, MIN_FLOOR)
    leaves = got is None or not math.isfinite(got) or got > tol
    return {"kind": kind, "name": name, "rel_l2": got, "floor": floor, "tolerance": tol, "ratio_to_floor": (got / floor) if (got is not None and floor > 0) else None,
            "leaves": bool(leaves), "why": None if not leaves else ("the chip run has no such array or it is not comparable" if got is None else
                                                                   f"relative L2 {got:.5g} is above {tol:.5g}")}


def _argmax(a):
    return int(np.asarray(a).reshape(-1).argmax())


def agreement(ref, chip) -> dict:
    """Teacher-forced next-token agreement: the chip's argmax at the last prompt position against the reference path's first token, then its argmax at each
    decode step K against the path's token K+1. Every chip step was fed the reference's own token, so one flip does not change what follows."""
    toks = [int(v) for v in ref.array("other", tr.TOKENS)]
    preds, wants, margins = [], [], []
    for k in range(len(toks)):
        name = tr.PREFILL_LOGITS if k == 0 else tr.dec_logits(k - 1)
        if ("other", name) not in chip.entries:
            return {"matches": 0, "of": len(toks), "fraction": 0.0, "flips": [], "problem": f"the chip run has no other/{name}"}
        preds.append(_argmax(chip.array("other", name)))
        wants.append(toks[k])
        r = np.asarray(ref.array("other", name)).reshape(-1)
        top2 = np.partition(r, -2)[-2:] if r.size > 1 else r
        margins.append(float(top2.max() - top2.min()) if r.size > 1 else 0.0)
    flips = [{"step": k, "reference_token": wants[k], "chip_token": preds[k], "reference_margin": margins[k]} for k in range(len(toks)) if preds[k] != wants[k]]
    n = len(toks)
    return {"matches": n - len(flips), "of": n, "fraction": (n - len(flips)) / n if n else 0.0, "flips": flips, "problem": None}


def chip_report(ref, floors: dict, chip) -> dict:
    """The numeric check of a chip run against the reference and the CPU floor runs (see the module text). `floors` maps a label to a loaded run.
    Returns the per-array entries, the first block and first operation to leave, the largest ratio to the floor, the agreement and the verdict."""
    if not floors:
        raise ar.ReferenceError("no floor run was given; the rule needs the bfloat16 CPU run")
    blocks = sorted((n for n in _entries(ref, "layer") if "." not in n), key=lambda n: int(n.split("_")[1]))
    ops = sorted(n for n in _entries(ref, "layer") if "." in n)
    block_entries = [_entry(ref, floors, chip, "layer", n) for n in blocks]
    op_entries = [_entry(ref, floors, chip, "layer", n) for n in ops if ("layer", n) in chip.entries]
    # final_norm is optional: it is compared when both runs hold it. The logits arrays are required of the chip run.
    other_names = [n for n in sorted(_entries(ref, "other")) if n == tr.PREFILL_LOGITS or n.startswith("dec_logits_")
                   or (n == "final_norm" and ("other", n) in chip.entries)]
    other_entries = [_entry(ref, floors, chip, "other", n) for n in other_names]
    first_block = next((e for e in block_entries if e["leaves"]), None)
    first_op = None
    if first_block is not None:
        first_op = next((e for e in op_entries if e["name"].startswith(first_block["name"] + ".") and e["leaves"]), None)
    ratios = [e for e in block_entries if e["ratio_to_floor"] is not None]
    worst = max(ratios, key=lambda e: e["ratio_to_floor"], default=None)
    agree = agreement(ref, chip)
    leaving_other = [e["name"] for e in other_entries if e["leaves"]]
    reasons = []
    if first_block is not None:
        reasons.append(f"block {first_block['name']} leaves its tolerance: {first_block['why']}")
    if leaving_other:
        reasons.append("these arrays leave their tolerance: " + ", ".join(leaving_other[:6]))
    if agree["problem"]:
        reasons.append(agree["problem"])
    elif agree["fraction"] < AGREE_MIN_FRACTION:
        reasons.append(f"teacher-forced agreement {agree['matches']} of {agree['of']} is below {AGREE_MIN_FRACTION:.0%}")
    return {"rule": RULE, "margin": CHIP_MARGIN, "blocks": block_entries, "ops": op_entries, "other": other_entries,
            "first_block_leaving": first_block["name"] if first_block else None, "first_op_leaving": first_op["name"] if first_op else None,
            "worst_ratio_to_floor": None if worst is None else {"name": worst["name"], "ratio": worst["ratio_to_floor"]},
            "agreement": agree, "ok": not reasons, "reasons": reasons, "cannot_show": list(CANNOT_SHOW)}


def perplexity_report(ref: dict, chip: dict) -> dict:
    """The chip's perplexity against the float32 reference's on the same token window. Both inputs are the JSON the runtime scripts write: `tokens`,
    `perplexity`, `mean_nll`, `logprobs`."""
    reasons = []
    for label, d in (("reference", ref), ("chip", chip)):
        p = d.get("perplexity") if isinstance(d, dict) else None
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p) or p <= 0:
            reasons.append(f"the {label} perplexity is not a positive finite number")
    if not reasons and ref.get("tokens") != chip.get("tokens"):
        reasons.append(f"the two perplexity runs used windows of {ref.get('tokens')} and {chip.get('tokens')} tokens")
    rel = None
    mean_abs = None
    if not reasons:
        rel = (chip["perplexity"] - ref["perplexity"]) / ref["perplexity"]
        if abs(rel) > PPL_REL_TOL:
            reasons.append(f"the chip's perplexity {chip['perplexity']:.4f} is {rel:+.1%} from the reference's {ref['perplexity']:.4f}; the bar is {PPL_REL_TOL:.0%}")
        a, b = ref.get("logprobs"), chip.get("logprobs")
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and a:
            mean_abs = float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
    return {"ok": not reasons, "reasons": reasons, "reference_ppl": ref.get("perplexity"), "chip_ppl": chip.get("perplexity"), "rel_diff": rel,
            "tokens": ref.get("tokens"), "mean_abs_logprob_difference": mean_abs, "tolerance": PPL_REL_TOL}
