#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The long-context part of bench.py only (prefill of long prompts and the decode rate with a long context), through a running server.

    python3 scripts/aion/bench_long.py --port 20180 --out RECORD.json

Same code and settings as bench.py (PREFILL_TARGETS, DECODE_AT, the counting prompt). The first run of bench.py used targets that made one prompt longer
than the server's 8192-token limit (a 400) and a plain filler prompt that ended after one token, so those rows were taken again with this script.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench as b  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = {"schema": "aion-bench-long/1", "load_before": os.getloadavg(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pre = []
    for target in b.PREFILL_TARGETS:
        text = b.FILLER * max(1, target // 18)
        secs, ntok = [], None
        for _ in range(b.REPEATS):
            code, out = b.post(a.port, "/v1/completions", {"prompt": text, "max_tokens": 1})
            if code != 200:
                pre.append({"target": target, "error": f"{code} {json.dumps(out)[:200]}"})
                break
            ntok = out["usage"]["prompt_tokens"]
            secs.append(out["timings"]["prefill_s"])
        else:
            pre.append({"prompt_tokens": ntok, "seconds": b.spread(secs), "tok_s_median": round(ntok / b.statistics.median(secs), 1)})
        print(pre[-1], flush=True)
    rec["prefill"] = pre
    at = []
    for target in b.DECODE_AT:
        text = b.FILLER * max(1, target // 18) + b.COUNTING
        rates, ntok, toks = [], None, []
        for _ in range(b.DECODE_AT_REPEATS):
            code, out = b.post(a.port, "/v1/completions", {"prompt": text, "max_tokens": b.DECODE_AT_TOKENS})
            if code != 200:
                at.append({"target": target, "error": f"{code} {json.dumps(out)[:200]}"})
                break
            ntok = out["usage"]["prompt_tokens"]
            toks.append(out["usage"]["completion_tokens"])
            rates.append(out["timings"].get("decode_tok_s"))
        else:
            at.append({"prompt_tokens": ntok, "new_tokens": b.DECODE_AT_TOKENS, "completion_tokens": toks, "tok_s": b.spread(rates)})
        print(at[-1], flush=True)
    rec["decode_at_length"] = at
    rec["load_after"] = os.getloadavg()
    json.dump(rec, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
