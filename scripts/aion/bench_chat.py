#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Decode rate through the chat route of a running server: five prompts that ask for a paragraph, REPEATS runs each, 64 new tokens, greedy.

    python3 scripts/aion/bench_chat.py --port 20180 --out RECORD.json

The chat route puts a system header of about 40 tokens in front of every prompt, so the context is longer than the raw prompts of bench.py at the
same answer length. One discarded warm-up request first. Runs that end before 64 tokens are counted in `tokens` and kept.
"""
import argparse
import json
import os
import statistics
import time
import urllib.request

PROMPTS = ["Write a short paragraph about the history of the bicycle.", "Explain how a refrigerator keeps food cold, in a paragraph.",
           "Describe a rainy day in a small town in one paragraph.", "Give a paragraph of advice for someone learning to play the guitar.",
           "Summarize why the Roman Empire fell, in a paragraph."]
REPEATS = 3


def chat(port, content):
    body = json.dumps({"messages": [{"role": "user", "content": content}], "max_tokens": 64}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = {"schema": "aion-bench-chat/1", "load_before": os.getloadavg(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    chat(a.port, PROMPTS[0])
    runs = []
    for p in PROMPTS:
        for _ in range(REPEATS):
            r = chat(a.port, p)
            runs.append({"prompt_tokens": r["usage"]["prompt_tokens"], "tokens": r["usage"]["completion_tokens"], "tok_s": r["timings"].get("decode_tok_s")})
    rates = [x["tok_s"] for x in runs if x["tok_s"]]
    rec["runs"] = runs
    rec["tok_s"] = {"n": len(rates), "min": round(min(rates), 3), "median": round(statistics.median(rates), 3), "max": round(max(rates), 3)}
    rec["prompt_tokens"] = {"min": min(x["prompt_tokens"] for x in runs), "max": max(x["prompt_tokens"] for x in runs)}
    rec["load_after"] = os.getloadavg()
    json.dump(rec, open(a.out, "w"), indent=2)
    print(json.dumps({k: rec[k] for k in ("tok_s", "prompt_tokens")}))


if __name__ == "__main__":
    main()
