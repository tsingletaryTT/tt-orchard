#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Explain where the served greedy tokens leave the CPU reference, from three records.

    python3 scripts/olmo_hybrid/analyze_divergence.py --bench BENCH_WARM.json --ref-logits REF.json --chip-logits CHIP.json --out DIAG.json

For every prompt the bench recorded `leading_match` (how many served tokens equal the reference's before the first difference). For a prompt
that differs, position k = leading_match is the first difference. The reference's teacher-forced logits there (top five, gap between the
top two) and the port's logits at the same position, fed the same prefix, are compared. NEAR_TIE_LOGITS is the size of a gap that two
neighbouring bf16 values of a logit near 20 can produce (the spacing at 16 to 32 is 0.125), so a reference gap at or below it is called a near tie.
"""
import argparse
import json

NEAR_TIE_LOGITS = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--ref-logits", required=True)
    ap.add_argument("--chip-logits", required=True)
    ap.add_argument("--repeat-log", help="repeat.log: 'served run N [ids]' lines for the first differing prompt, from the same server")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bench = json.load(open(a.bench))["reference_match"]
    ref = {i["prompt"][:40]: i for i in json.load(open(a.ref_logits))["items"]}
    chip = {i["prompt"][:40]: i for i in json.load(open(a.chip_logits))["items"]}
    rows = []
    for b in bench:
        key = b["prompt"]
        row = {"prompt": key, "compared": b["compared"], "leading_match": b["leading_match"], "matching_positions": b["matching_positions"],
               "matched_fully": b["leading_match"] >= b["compared"]}
        if not row["matched_fully"]:
            k = b["leading_match"]
            r, c = ref[key]["positions"][k], chip[key]["positions"][k]
            r1, r2 = r["top"][0], r["top"][1]
            row["first_difference_at_token_index"] = k
            row["reference_top2"] = [{"id": r1["id"], "text": r1["text"], "logit": r1["logit"]}, {"id": r2["id"], "text": r2["text"], "logit": r2["logit"]}]
            row["reference_gap"] = r["gap_top1_top2"]
            row["chip_top2"] = [{"id": t["id"], "text": t["text"], "logit": t["logit"]} for t in c["top"][:2]]
            row["chip_gap"] = c["gap_top1_top2"]
            row["chip_agrees_with_reference_top1_in_teacher_forcing"] = c["top"][0]["id"] == r1["id"]
            row["near_tie"] = r["gap_top1_top2"] <= NEAR_TIE_LOGITS
            # the chip's own values for the reference's two best tokens, when they are in its top five
            ch = {t["id"]: t["logit"] for t in c["top"]}
            row["reference_generate_choice"] = {"id": r["reference_choice"], "text": [t["text"] for t in r["top"] if t["id"] == r["reference_choice"]][0]
                                                if any(t["id"] == r["reference_choice"] for t in r["top"]) else None}
            row["reference_generate_choice_differs_from_its_full_forward_top1"] = r["reference_choice"] != r1["id"]
            if a.repeat_log:
                runs = []
                for line in open(a.repeat_log):
                    if line.startswith("served run"):
                        runs.append(json.loads(line[line.index("["):]))
                row["served_repeats"] = {"runs": len(runs), "token_at_index": [x[k] for x in runs if len(x) > k], "all_identical": len({tuple(x) for x in runs}) == 1}
            row["chip_logit_of_reference_top1"] = ch.get(r1["id"])
            row["chip_logit_of_reference_top2"] = ch.get(r2["id"])
        rows.append(row)
    json.dump({"near_tie_logits": NEAR_TIE_LOGITS, "prompts": rows}, open(a.out, "w"), indent=1)
    print(json.dumps({"near_tie_logits": NEAR_TIE_LOGITS, "prompts": rows}, indent=1))


if __name__ == "__main__":
    main()
