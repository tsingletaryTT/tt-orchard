#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks through a running aion-1.0-instruct-2.3b-blackhole server: a passkey at three lengths and five short chat questions.

    python3 scripts/aion/served_checks.py --port 20180 --out RECORD.json

The passkey prompt hides `The passkey is NNNNN.` in the middle of repeated filler and asks for it through the chat route (greedy). The lengths are the
prompt token counts the server reports. The five questions each name a fact the answer must contain (a case-insensitive substring check, so it cannot
tell a good answer from a lucky one). One request at a time. Nothing here measures speed; bench.py does.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

FILLER = "The grass is green. The sky is blue. The sun is warm. Here we go again. "
PASSKEYS = [("about 256", 256, "48217"), ("about 2048", 2048, "73905"), ("about 7800", 6800, "20653")]
QUESTIONS = [("What is the capital of France? Answer in one word.", "paris"), ("What is 17 plus 25? Reply with just the number.", "42"),
             ("Which planet is known as the Red Planet? Answer in one word.", "mars"), ("What is the chemical symbol for water? Give the formula.", "h2o"),
             ("Name the author of the play Hamlet. Answer with the name only.", "shakespeare")]


def chat(port, content, max_tokens):
    body = json.dumps({"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            out = json.loads(r.read())
        return r.status, out, time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = {"schema": "aion-served-checks/1", "load_before": os.getloadavg(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "passkey": [], "chat": []}
    for label, target, key in PASSKEYS:
        n = max(2, target // 18)
        text = FILLER * (n // 2) + f"The passkey is {key}. Remember it. " + FILLER * (n // 2)
        code, out, secs = chat(a.port, text + "\nWhat is the passkey? Answer with the number only.", 24)
        ans = out["choices"][0]["message"]["content"] if code == 200 else json.dumps(out)[:200]
        rec["passkey"].append({"length": label, "prompt_tokens": out.get("usage", {}).get("prompt_tokens"), "status": code, "answer": ans, "found": key in ans, "seconds": round(secs, 1)})
        print(rec["passkey"][-1], flush=True)
    for q, fact in QUESTIONS:
        code, out, secs = chat(a.port, q, 48)
        ans = out["choices"][0]["message"]["content"] if code == 200 else json.dumps(out)[:200]
        rec["chat"].append({"question": q, "expected": fact, "status": code, "answer": ans, "found": fact in ans.lower(), "seconds": round(secs, 1)})
        print(rec["chat"][-1], flush=True)
    rec["load_after"] = os.getloadavg()
    json.dump(rec, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
