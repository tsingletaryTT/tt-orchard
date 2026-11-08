#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Measure a running olmo-hybrid-7b-blackhole server over HTTP and write one JSON record.

    python3 scripts/olmo_hybrid/bench.py --port 20120 --out RECORD.json [--cache-state warm|cold] [--reference-ids REF.json]

What it measures (all wall time through the server, greedy decoding, one request at a time):
- decode tokens per second at batch size 1: DECODE_PROMPTS different prompts, DECODE_TOKENS new tokens each, REPEATS runs of each, after one
  discarded warm-up request. The server reports decode_tok_s per request (tokens after the first, over the decode seconds).
- prefill seconds for prompts of about 128, 512, 2048 and 8192 tokens, REPEATS runs each, one new token requested.
- concurrency: two requests at once. The server runs one at a time, so one of them must get 503; the record says what happened.
- with --reference-ids: the leading match between the server's greedy token ids and a reference's for each prompt in the reference file.
The host load average is read before and after, and the number of runs is in the record. Nothing here is a claim about other prompts.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

DECODE_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "In 1969, the first humans to walk on the Moon were",
    "The three primary colors are red, blue, and",
]
DECODE_TOKENS = 64
REPEATS = 3
FILLER = "The grass is green. The sky is blue. The sun is warm. Here we go again. "
PREFILL_TARGETS = [128, 512, 2048, 8192]


def post(port, path, body, timeout=3600):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def spread(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "min": round(min(xs), 3), "median": round(statistics.median(xs), 3), "max": round(max(xs), 3)} if xs else {"n": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-state", choices=["warm", "cold"], required=True)
    ap.add_argument("--reference-ids")
    a = ap.parse_args()
    rec = {"schema": "olmo-hybrid-7b-bench/1", "cache_state": a.cache_state, "repeats": REPEATS, "load_before": os.getloadavg(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    t0 = time.time()
    post(a.port, "/v1/completions", {"prompt": DECODE_PROMPTS[0], "max_tokens": 8})          # discarded warm-up; its time shows kernel compilation
    rec["first_request_s"] = round(time.time() - t0, 2)
    dec = []
    for p in DECODE_PROMPTS:
        for _ in range(REPEATS):
            code, out = post(a.port, "/v1/completions", {"prompt": p, "max_tokens": DECODE_TOKENS})
            if code != 200:
                sys.exit(f"decode request failed: {code} {out}")
            dec.append({"tok_s": out["timings"].get("decode_tok_s"), "tokens": out["usage"]["completion_tokens"], "prefill_s": out["timings"]["prefill_s"]})
    def save():
        json.dump(rec, open(a.out, "w"), indent=2)
    rec["decode"] = {"prompts": len(DECODE_PROMPTS), "new_tokens_each": DECODE_TOKENS, "runs": len(dec), "tok_s": spread([d["tok_s"] for d in dec]),
                     "tokens_per_run": spread([d["tokens"] for d in dec])}
    save()
    pre = []
    for target in PREFILL_TARGETS:
        text = FILLER * max(1, target // 18)
        secs, ntok = [], None
        for _ in range(REPEATS):
            code, out = post(a.port, "/v1/completions", {"prompt": text, "max_tokens": 1})
            if code != 200:
                pre.append({"target": target, "error": f"{code} {json.dumps(out)[:200]}"})
                break
            ntok = out["usage"]["prompt_tokens"]
            secs.append(out["timings"]["prefill_s"])
        else:
            pre.append({"prompt_tokens": ntok, "seconds": spread(secs), "tok_s_median": round(ntok / statistics.median(secs), 1)})
    rec["prefill"] = pre
    save()
    res = []
    long_prompt = FILLER * 40
    def one():
        res.append(post(a.port, "/v1/completions", {"prompt": long_prompt, "max_tokens": 24})[0])
    ts = [threading.Thread(target=one) for _ in range(2)]
    for t in ts:
        t.start()
        time.sleep(0.2)
    for t in ts:
        t.join()
    rec["concurrency"] = {"two_at_once_status_codes": sorted(res), "batching": "not supported: a second request gets 503 while one runs"}
    rec["load_after"] = os.getloadavg()
    save()
    if a.reference_ids:
        ref = json.load(open(a.reference_ids))
        rows = []
        for item in ref["items"]:
            code, out = post(a.port, "/v1/completions", {"prompt": item["prompt"], "max_tokens": len(item["ids"]), "return_token_ids": True})
            ids = out["choices"][0].get("token_ids", []) if code == 200 else []
            lead = next((i for i, (x, y) in enumerate(zip(ids, item["ids"])) if x != y), min(len(ids), len(item["ids"])))
            rows.append({"prompt": item["prompt"][:40], "compared": len(item["ids"]), "leading_match": lead,
                         "matching_positions": sum(1 for x, y in zip(ids, item["ids"]) if x == y)})
        rec["reference_match"] = rows
    rec["load_after"] = os.getloadavg()
    json.dump(rec, open(a.out, "w"), indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
