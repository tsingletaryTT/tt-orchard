#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Teacher-forced agreement of the Aion port with the CPU reference, from the two logits records (ref_logits.py and chip_logits.py).

    python3 scripts/aion/quality.py --ref-logits REF.json --chip-logits CHIP.json --out QUALITY.json

Position k of a prompt is the prediction of continuation token k, made after the prompt and continuation tokens 0 to k-1 (the reference's own
continuation is fed to both sides). Two agreement figures per prompt and in total:
  top1_vs_reference_text  the port's top token equals the reference's continuation token (the stage 6 definition of top-1 agreement);
  top1_vs_reference_top1  the port's top token equals the reference's own top token at that position.
A position where the port's top token differs from the reference's continuation token is a miss. `near_tie_misses` counts the misses whose reference gap
between its top two logits is at most NEAR_TIE_LOGITS (0.25: neighbouring bf16 values near 20 are 0.125 apart); `misses_ref_choice_in_port_top2` counts
misses where the reference's token is the port's second choice. Nothing here covers text the reference did not write.
"""
import argparse
import json

NEAR_TIE_LOGITS = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-logits", required=True)
    ap.add_argument("--chip-logits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = json.load(open(a.ref_logits))["items"]
    chip = json.load(open(a.chip_logits))["items"]
    rows, tot = [], {"positions": 0, "top1_vs_reference_text": 0, "top1_vs_reference_top1": 0, "near_tie_misses": 0, "misses": 0, "misses_ref_choice_in_port_top2": 0}
    for r, c in zip(ref, chip):
        if r["prompt"] != c["prompt"] or len(r["positions"]) != len(c["positions"]):
            raise SystemExit(f"the two records do not describe the same prompt or length: {r['prompt'][:30]!r}")
        n = len(r["positions"])
        text = sum(1 for p, q in zip(r["positions"], c["positions"]) if q["top"][0]["id"] == p["reference_choice"])
        top1 = sum(1 for p, q in zip(r["positions"], c["positions"]) if q["top"][0]["id"] == p["top"][0]["id"])
        miss = [(p, q) for p, q in zip(r["positions"], c["positions"]) if q["top"][0]["id"] != p["reference_choice"]]
        near = sum(1 for p, _ in miss if p["gap_top1_top2"] <= NEAR_TIE_LOGITS)
        in2 = sum(1 for p, q in miss if len(q["top"]) > 1 and q["top"][1]["id"] == p["reference_choice"])
        rows.append({"prompt": " ".join(r["prompt"].split())[:60], "positions": n, "top1_vs_reference_text": round(text / n, 4), "top1_vs_reference_top1": round(top1 / n, 4),
                     "misses": len(miss), "near_tie_misses": near, "misses_ref_choice_in_port_top2": in2})
        for k, v in (("positions", n), ("top1_vs_reference_text", text), ("top1_vs_reference_top1", top1), ("near_tie_misses", near), ("misses", len(miss)),
                     ("misses_ref_choice_in_port_top2", in2)):
            tot[k] += v
    n = tot["positions"]
    out = {"near_tie_logits": NEAR_TIE_LOGITS, "prompts": rows,
           "total": {**tot, "top1_vs_reference_text_fraction": round(tot["top1_vs_reference_text"] / n, 4), "top1_vs_reference_top1_fraction": round(tot["top1_vs_reference_top1"] / n, 4)}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["total"], indent=1))


if __name__ == "__main__":
    main()
