"""Compare device output with the CPU reference. Per prompt: how many leading tokens agree, whether the first difference is a near tie in the
reference (the device token inside the reference's top 5, with the log-probability gap), and the share of positions with the same token.
Refuses inputs that cannot be compared: different numbers of prompts, different prompts, or different token counts for a prompt.
Usage: compare.py REF_JSON DEVICE_JSON"""
import json, sys


def compare_rows(ref, dev):
    if len(ref) != len(dev):
        raise ValueError(f"the reference has {len(ref)} prompts and the device run has {len(dev)}")
    rows = []
    for i, (r, d) in enumerate(zip(ref, dev)):
        if r["prompt"] != d["prompt"]:
            raise ValueError(f"prompt {i} differs between the reference and the device run")
        a, b = r["tokens"], d["tokens"]
        if not a or len(a) != len(b):
            raise ValueError(f"prompt {i}: the reference has {len(a)} tokens and the device run has {len(b)}")
        k = next((j for j in range(len(a)) if a[j] != b[j]), None)
        row = {"prompt": r["prompt"][:40], "tokens": len(a), "leading_agree": len(a) if k is None else k,
               "same_position_share": round(sum(x == y for x, y in zip(a, b)) / len(a), 3)}
        if k is not None:
            t5 = dict(zip(r["top5"][k]["ids"], r["top5"][k]["logprobs"]))
            row["first_diff"] = {"index": k, "ref_token": a[k], "device_token": b[k], "device_token_in_ref_top5": b[k] in t5,
                                 "ref_gap_nats": round(t5[a[k]] - t5[b[k]], 3) if b[k] in t5 else None}
        rows.append(row)
    return rows


if __name__ == "__main__":
    try:
        print(json.dumps(compare_rows(json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))["completions"]), indent=1))
    except ValueError as exc:
        print(f"compare.py: {exc}", file=sys.stderr)
        sys.exit(2)
