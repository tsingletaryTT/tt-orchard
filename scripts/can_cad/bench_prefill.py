#!/usr/bin/env python3
"""Prefill time without prefix-cache hits: for each target length, three prompts that differ from their first word on (so no block is shared),
max_tokens 1. Prints and saves the server's prompt token count and the time to the first token."""
import json, sys, time, urllib.request
port = int(sys.argv[1]); out = sys.argv[2]
model = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))["data"][0]["id"]
words = "aluminium steel bracket flange bolt washer plate tube slot hole bore chamfer fillet thread boss rib gusset".split()
res = []
for target in (2048, 7000):
    for k in range(3):
        w = words[(k * 5 + target // 1000) % len(words)]
        text = " ".join(f"{words[(i * (k + 3) + k) % len(words)]}{i % 97}" for i in range(int(target * 0.30)))
        body = {"model": model, "messages": [{"role": "user", "content": f"Part {w}-{k}-{target}. " + text}], "max_tokens": 1, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
        t0 = time.time(); first = None; usage = None
        with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
            for raw in r:
                l = raw.decode().strip()
                if l.startswith("data:") and l[5:].strip() != "[DONE]":
                    j = json.loads(l[5:])
                    if j.get("usage"): usage = j["usage"]
                    if j.get("choices") and (j["choices"][0]["delta"].get("content") or j["choices"][0]["delta"].get("reasoning_content") or j["choices"][0]["delta"].get("reasoning")) and first is None:
                        first = time.time()
        row = {"target": target, "prompt_tokens": usage["prompt_tokens"], "ttft_s": round((first or time.time()) - t0, 3)}
        res.append(row); print(row, flush=True)
json.dump(res, open(out, "w"), indent=1)
