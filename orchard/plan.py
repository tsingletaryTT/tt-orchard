# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Will this model fit on the chips I have, and at what precision? Answered from a checkpoint's safetensors headers alone.

    tt-orchard plan SNAPSHOT_DIR [--chips 4] [--json]

It reads only the headers of the `*.safetensors` files (tensor names and shapes, never a weight), so it can run before any chip is leased. It splits
the parameters into the routed experts of a mixture-of-experts model (tensors with `.experts.` in the name) and everything else, prices each part in
bf16, bfp8 and bfp4, and says for the chip count asked what each combination needs per chip, in two placements:

  replicated  every chip holds every weight (what the generated ports do today);
  sharded     the routed experts are split over the chips and everything else is replicated (not written yet: issue 723).

Bytes per element: bf16 2, bfp8 1.0625 and bfp4 0.5625 (one shared exponent byte per 16 values), which are the ratios of the probe 14 figures for
Flash-Next. A chip's usable DRAM is 93.4% of its 32 GiB (29.9 of 32, probe 14). Weights may take WEIGHT_SHARE of that; the rest is left for the KV
cache and activations. That share is an assumption and a flag changes it. The plan does not measure speed, and it does not measure quality: the notes
quote the one measured fidelity pair (scripts/fidelity_score.py) and say to score a candidate before choosing bfp4.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from orchard import defaults, safeio

BYTES_PER_ELEMENT = {"bf16": 2.0, "bfp8": 1.0625, "bfp4": 0.5625}
USABLE_FRACTION = 0.934            # probe 14: 29.9 GB usable of a 32 GB chip
WEIGHT_SHARE = 0.85                # choice: the share of usable DRAM the weights may take; the KV cache and activations need the rest
MAX_HEADER_BYTES = 64 << 20
MAX_FILES = 4096
# (name, format of everything else, format of the routed experts), best quality first
COMBINATIONS = (("all bf16", "bf16", "bf16"), ("all bfp8", "bfp8", "bfp8"), ("bfp8 + bfp4 experts", "bfp8", "bfp4"), ("all bfp4", "bfp4", "bfp4"))
QUALITY_NOTES = {
    "bfp8": "bfp8 weights scored 9.991 of 10 on 4000 sampled tokens of one 7B port (Olmo-Hybrid-7B): close to the reference.",
    "bfp4": "bfp4 weights scored 9.821 on the same port: a visible loss. Score a candidate with scripts/fidelity_score.py before choosing it.",
}
GIB = float(1 << 30)
# The rule for a format below bf16 (issue 731). bfp8 is the default: its measured score on the one pair we have is 9.991. bfp4 is recommended only
# when a score measured on the model's own port (scripts/fidelity_score.py, or the stage 2 fidelity block) is at or above FIDELITY_BAR on at least
# MIN_SCORED_TOKENS tokens. The bar is provisional: it sits between the two measured points (bfp8 9.991, bfp4 9.821 on one 7B port) and a loss of
# 1 percent of perplexity or less is what it allows. More calibration points should move it. A score below the bar removes the row.
FIDELITY_BAR = 9.9
MIN_SCORED_TOKENS = 4000


def is_routed_expert(name: str) -> bool:
    return ".experts." in name          # a shared expert is named `shared_expert` or `shared_experts`, so it does not match


def inventory(snapshot: Path) -> dict:
    """Element counts split into routed experts and the rest, from the headers of the snapshot's safetensors files."""
    files = sorted(p for p in snapshot.iterdir() if p.name.endswith(".safetensors"))[:MAX_FILES]
    experts = other = tensors = 0
    unreadable = []
    for f in files:
        target = safeio.resolved_inside(f, snapshot)
        if target is None:
            unreadable.append(f"{f.name}: a link that leaves the snapshot")
            continue
        try:
            header = safeio.safetensors_header(target, max_header_bytes=MAX_HEADER_BYTES)
        except safeio.SafeReadError as exc:
            unreadable.append(f"{f.name}: {exc.reason}")
            continue
        for name, e in header.entries.items():
            n = 1
            for d in e.shape:
                n *= d
            tensors += 1
            if is_routed_expert(name):
                experts += n
            else:
                other += n
    return {"files": len(files), "tensors": tensors, "routed_expert_params": experts, "other_params": other, "unreadable": unreadable}


def budget_gib(chip_gib: float, share: float) -> float:
    return chip_gib * USABLE_FRACTION * share


def format_gate(fmt: str, scores: dict | None) -> str:
    """"ok", "unscored", "too-few-tokens" or "below-bar" for one weight format, from the measured scores {format: (score, tokens)}."""
    if fmt in ("bf16", "bfp8"):
        return "ok"
    got = (scores or {}).get(fmt)
    if got is None:
        return "unscored"
    score, tokens = got
    if tokens < MIN_SCORED_TOKENS:
        return "too-few-tokens"
    return "ok" if score >= FIDELITY_BAR else "below-bar"


def _worst(*gates: str) -> str:
    for g in ("below-bar", "too-few-tokens", "unscored"):
        if g in gates:
            return g
    return "ok"


def rows(inv: dict, chips: int, chip_gib: float, share: float, scores: dict | None = None) -> list[dict]:
    budget = budget_gib(chip_gib, share)
    out = []
    for name, other_fmt, expert_fmt in COMBINATIONS:
        if not inv["routed_expert_params"] and other_fmt != expert_fmt:
            continue                                  # a dense model has no experts to price differently
        o = inv["other_params"] * BYTES_PER_ELEMENT[other_fmt] / GIB
        e = inv["routed_expert_params"] * BYTES_PER_ELEMENT[expert_fmt] / GIB
        sharded = lambda n: o + e / n
        out.append({"name": name, "other_format": other_fmt, "expert_format": expert_fmt, "total_gib": round(o + e, 2),
                    "format_gate": _worst(format_gate(other_fmt, scores), format_gate(expert_fmt, scores)),
                    "replicated_per_chip_gib": round(o + e, 2), "fits_replicated": o + e <= budget,
                    "sharded_per_chip_gib": round(sharded(chips), 2), "fits_sharded": sharded(chips) <= budget,
                    "min_chips_sharded": next((n for n in range(1, 65) if sharded(n) <= budget), None)})
    return out


def recommend(table: list[dict]) -> dict | None:
    """The best-quality row that fits and whose formats pass the rule (`format_gate`), replicated first; a row that needs sharded experts says so.
    When no row passes, the best fitting row that is only unscored is returned with its gate, so the plan can say what to measure; a row below the
    bar is never recommended."""
    sharded_note = "the routed experts split over the chips, which the generator does not write yet (issue 723)"
    for wanted in (("ok",), ("unscored", "too-few-tokens")):
        for placement, key in (("replicated", "fits_replicated"), ("sharded", "fits_sharded")):
            for r in table:
                if r[key] and r["format_gate"] in wanted:
                    return {"row": r["name"], "placement": placement, "gate": r["format_gate"],
                            "needs": sharded_note if placement == "sharded" else None}
    return None


def plan(snapshot: Path, chips: int, chip_gib: float = defaults.CHIP_DRAM_GIB, share: float = WEIGHT_SHARE, scores: dict | None = None) -> dict:
    inv = inventory(snapshot)
    table = rows(inv, chips, chip_gib, share, scores)
    rec = recommend(table)
    notes = [QUALITY_NOTES[f] for f in ("bfp8", "bfp4") if rec and f in (next(r for r in table if r["name"] == rec["row"])["other_format"],
                                                                         next(r for r in table if r["name"] == rec["row"])["expert_format"])]
    if rec is None and table:
        below = [r["name"] for r in table if r["format_gate"] == "below-bar" and (r["fits_replicated"] or r["fits_sharded"])]
        notes.append("nothing that passes the fidelity rule fits at this chip count" + (f" (rows that fit but score below {FIDELITY_BAR}: {', '.join(below)})" if below
                     else " even with the experts split") + "; add chips, choose a smaller model or a better-scoring format")
    if rec and rec["gate"] != "ok":
        notes.append(f"this row uses bfp4 with no usable score ({rec['gate']}): score the port with scripts/fidelity_score.py on at least {MIN_SCORED_TOKENS} tokens "
                     f"and pass the result with --fidelity-bfp4 SCORE --fidelity-tokens N; the bar is {FIDELITY_BAR}")
    return {"snapshot": snapshot.name, **inv, "chips": chips, "chip_gib": chip_gib, "weight_budget_gib_per_chip": round(budget_gib(chip_gib, share), 2),
            "weight_share": share, "rows": table, "recommendation": rec, "notes": notes}


def render(res: dict) -> str:
    lines = [f"{res['snapshot']}: {res['tensors']} tensors in {res['files']} files, {res['routed_expert_params'] / 1e9:.2f} B routed-expert and "
             f"{res['other_params'] / 1e9:.2f} B other parameters",
             f"{res['chips']} chips, {res['weight_budget_gib_per_chip']} GiB per chip for weights ({res['weight_share']:.0%} of usable DRAM)", ""]
    lines.append(f"{'combination':22} {'total GiB':>10} {'replicated':>11} {'experts split':>14}   {'min chips if split':<19}fidelity rule")
    for r in res["rows"]:
        lines.append(f"{r['name']:22} {r['total_gib']:>10} {'fits' if r['fits_replicated'] else 'no':>11} "
                     f"{'fits' if r['fits_sharded'] else 'no':>14}   {str(r['min_chips_sharded'] if r['min_chips_sharded'] else 'more than 64'):<19}{r['format_gate']}")
    rec = res["recommendation"]
    lines.append("")
    lines.append(f"best fit: {rec['row']}, {rec['placement']}" + (f"; needs {rec['needs']}" if rec["needs"] else "") if rec else "best fit: none at this chip count")
    lines += [f"note: {n}" for n in res["notes"]] + [f"unreadable: {u}" for u in res["unreadable"]]
    return "\n".join(lines)


def parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tt-orchard plan", add_help=add_help,
                                description="say what precision and placement a checkpoint needs on N chips, from its safetensors headers alone")
    p.add_argument("snapshot", help="a directory of *.safetensors files (a Hugging Face snapshot)")
    p.add_argument("--chips", type=int, default=4, help="chips to plan for (default 4)")
    p.add_argument("--chip-gib", type=float, default=defaults.CHIP_DRAM_GIB, help="DRAM per chip in GiB (default %(default)s)")
    p.add_argument("--weight-share", type=float, default=WEIGHT_SHARE, help="share of usable DRAM the weights may take (default %(default)s)")
    p.add_argument("--fidelity-bfp4", type=float, help=f"a fidelity score measured for this model's port with bfp4 weights (the bar is {FIDELITY_BAR})")
    p.add_argument("--fidelity-tokens", type=int, default=0, help=f"scored tokens behind --fidelity-bfp4 (at least {MIN_SCORED_TOKENS} to count)")
    p.add_argument("--json", action="store_true", help="print JSON")
    return p


def main(argv=None, out=None) -> int:
    args = parser().parse_args(argv)
    out = out or sys.stdout
    snap = Path(args.snapshot)
    if not snap.is_dir():
        print(f"{args.snapshot} is not a directory", file=sys.stderr)
        return 2
    if not 1 <= args.chips <= 64 or not 0 < args.weight_share <= 1 or not 0 < args.chip_gib <= 4096:
        print("--chips must be 1 to 64, --weight-share in (0, 1] and --chip-gib positive", file=sys.stderr)
        return 2
    scores = None
    if args.fidelity_bfp4 is not None:
        if not safeio.is_finite_number(args.fidelity_bfp4) or not 0 <= args.fidelity_bfp4 <= 10.5 or args.fidelity_tokens < 0:
            print("--fidelity-bfp4 must be a score from 0 to 10 and --fidelity-tokens a count", file=sys.stderr)
            return 2
        scores = {"bfp4": (args.fidelity_bfp4, args.fidelity_tokens)}
    res = plan(snap, args.chips, args.chip_gib, args.weight_share, scores)
    if not res["tensors"]:
        print("no readable safetensors tensors in the snapshot" + "".join(f"; {u}" for u in res["unreadable"]), file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2) if args.json else render(res), file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
