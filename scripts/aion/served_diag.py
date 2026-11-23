# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Why did the served greedy run of the first chat prompt leave the reference at token 6 when the teacher-forced port agrees there? Fresh-server runs."""
import json, sys, urllib.request
port = int(sys.argv[1])
ref = json.load(open(sys.argv[2]))["items"]
def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())
out = {}
def run(tag, item, n=40):
    r = post({"prompt": item["prompt"], "max_tokens": n, "return_token_ids": True})
    ids = r["choices"][0].get("token_ids", [])
    lead = next((i for i, (x, y) in enumerate(zip(ids, item["ids"])) if x != y), min(len(ids), len(item["ids"])))
    out[tag] = {"served_ids": ids, "reference_ids": item["ids"][:n], "leading_match": lead, "finish": r["choices"][0].get("finish_reason"), "prompt_tokens": r["usage"]["prompt_tokens"]}
    print(tag, "lead", lead, "of", len(ids), "finish", out[tag]["finish"], "prompt_tokens", out[tag]["prompt_tokens"], flush=True)
for name, i in (("chat1", 4), ("chat2", 5), ("water", 1)):
    run(name + "-fresh-1", ref[i]); run(name + "-fresh-2", ref[i])
post({"prompt": "The grass is green. " * 100, "max_tokens": 2})      # a longer prompt in between
for name, i in (("chat1", 4), ("water", 1)):
    run(name + "-after-long", ref[i])
json.dump(out, open(sys.argv[3], "w"), indent=1)
