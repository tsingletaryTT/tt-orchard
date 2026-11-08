#!/usr/bin/env python3
"""Benchmark a served Berkelium CadQuery model through the OpenAI chat API (standard library only).

Measures, per request, time to first streamed token and decode speed ((tokens - 1) / time between first and last token, tokens from the
server's usage block); decode rate at batch size one over several CAD prompts and repeats; prefill (time to first token for a request of
about N prompt tokens, max_tokens 1); host load before and after; and saves every reply so the generated CadQuery can be checked.
Usage: bench.py --port 8061 --out RESULT.json [--repeats 3] [--prefill 2048,6144]
"""
import argparse, json, os, statistics, time, urllib.request

SYSTEM = ('You are a CAD compiler. Turn the request into one line of JSON {"summary","units":"mm","assumptions":[],"open_questions":[]} '
          'followed by one ```python block: a CadQuery script in millimetres that imports only cadquery and math and ends with `result = ...`. '
          'Unstated sizes go in assumptions; missing key sizes go in open_questions.')
PROMPTS = [
    "Create a cylinder with a height of 179 mm and a diameter of 100 mm, centered on the ZX plane.",
    "A rectangular plate 120 mm by 80 mm and 10 mm thick with four 8 mm holes, one near each corner, 12 mm in from the edges.",
    "An L-bracket: two flanges 60 mm wide, each 40 mm long and 5 mm thick, joined at a right angle, with one 6 mm hole through each flange.",
    "A round flange with 100 mm outer diameter, 60 mm bore, 12 mm thick, and six 6 mm bolt holes on an 80 mm circle.",
    "A hex bolt with an M10 shaft 40 mm long and a hexagonal head 17 mm across flats and 7 mm tall.",
    "A washer with 20 mm outer diameter, 10.5 mm hole and 2 mm thickness.",
    "A hollow tube 200 mm long with 30 mm outer diameter and 3 mm wall, with a 10 mm slot cut through one side 20 mm from one end.",
    "Create a 3D solid by extruding a closed 2D profile 115 mm from the XY plane. The profile is a rectangle 200 mm by 90 mm.",
]

def post(port, body, stream):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)

def models(port):
    return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=30))["data"][0]["id"]

def chat(port, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = last = None; text = []; usage = None
    with post(port, body, True) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices", []):
                d = c.get("delta", {})
                piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                if piece:
                    now = time.time()
                    first = first or now; last = now; text.append(piece)
    n = (usage or {}).get("completion_tokens", 0)
    out = {"ttft_s": round(first - t0, 3) if first else None, "completion_tokens": n, "prompt_tokens": (usage or {}).get("prompt_tokens"),
           "total_s": round(time.time() - t0, 3), "text": "".join(text)}
    if first and last and n > 1 and last > first:
        out["decode_tok_s"] = round((n - 1) / (last - first), 2)
    return out

def filler(model, port, target):
    """A prompt of about `target` tokens made of repeated plain sentences (the server's own count is recorded)."""
    s = "The bracket is made of aluminium and is held by four bolts. "
    n = max(1, target // 12)
    return [{"role": "user", "content": s * n + "\nSummarise the sentence above in one word."}]

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8061)
ap.add_argument("--out", required=True)
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--prefill", default="2048,6144")
ap.add_argument("--cache-state", default="unstated")
a = ap.parse_args()
model = models(a.port)
res = {"model": model, "cache_state": a.cache_state, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "load_before": open("/proc/loadavg").read().split()[:3]}
chat(a.port, model, [{"role": "user", "content": "hi"}], 8)       # warm-up, not counted
res["decode"] = []
for rep in range(a.repeats):
    for i, p in enumerate(PROMPTS):
        r = chat(a.port, model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}], a.max_tokens)
        r.update(prompt_index=i, repeat=rep); res["decode"].append(r)
        print(i, rep, r.get("decode_tok_s"), r["ttft_s"], r["completion_tokens"], flush=True)
rates = [r["decode_tok_s"] for r in res["decode"] if "decode_tok_s" in r]
res["decode_summary"] = {"runs": len(rates), "median": statistics.median(rates), "min": min(rates), "max": max(rates)} if rates else None
res["prefill"] = []
for target in [int(x) for x in a.prefill.split(",") if x]:
    for rep in range(2):
        r = chat(a.port, model, filler(model, a.port, target), 1)
        r.update(target_tokens=target, repeat=rep); res["prefill"].append(r)
        print("prefill", target, r["prompt_tokens"], r["ttft_s"], flush=True)
res["load_after"] = open("/proc/loadavg").read().split()[:3]
json.dump(res, open(a.out, "w"), indent=1)
