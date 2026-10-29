"""Compare device output with the CPU reference. Per prompt: how many leading tokens agree, whether the first difference is a near tie in the
reference (the device token inside the reference's top 5, with the log-probability gap), and the share of positions with the same token.
Usage: compare.py REF_JSON DEVICE_JSON"""
import json, sys
ref = json.load(open(sys.argv[1])); dev = json.load(open(sys.argv[2]))["completions"]
rows = []
for r, d in zip(ref, dev):
    assert r["prompt"] == d["prompt"]
    a, b = r["tokens"], d["tokens"]
    k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
    row = {"prompt": r["prompt"][:40], "tokens": len(a), "leading_agree": len(a) if k is None else k,
           "same_position_share": round(sum(x == y for x, y in zip(a, b)) / len(a), 3)}
    if k is not None:
        t5 = dict(zip(r["top5"][k]["ids"], r["top5"][k]["logprobs"]))
        row["first_diff"] = {"index": k, "ref_token": a[k], "device_token": b[k], "device_token_in_ref_top5": b[k] in t5,
                             "ref_gap_nats": round(t5[a[k]] - t5[b[k]], 3) if b[k] in t5 else None}
    rows.append(row)
print(json.dumps(rows, indent=1))
