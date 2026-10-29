"""Decode benchmark client for a local vLLM server (docs/benchmark-rules.md): 1 discarded warm-up, N measured runs, min/median/max,
host load average recorded with each run. Decode tok/s = (completion tokens - 1) / (last chunk time - first chunk time)."""
import json, os, statistics, sys, time, urllib.request
port, runs, max_tokens = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
prompt = open(sys.argv[4]).read().strip()
model = sys.argv[5]
def post(body, stream=False):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)
def one():
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True, "ignore_eos": True}
    t0 = time.perf_counter(); first = last = None; n = 0
    with post(body) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            ev = json.loads(line[5:]); now = time.perf_counter()
            if ev["choices"] and ev["choices"][0].get("text") is not None:
                n += 1; first = first or now; last = now
    return {"tokens": n, "ttft_s": round(first - t0, 3), "decode_tok_s": round((n - 1) / (last - first), 2), "load1": os.getloadavg()[0]}
one()   # warm-up, discarded
res = [one() for _ in range(runs)]
v = [r["decode_tok_s"] for r in res]
print("BENCH " + json.dumps({"runs": res, "min": min(v), "median": statistics.median(v), "max": max(v)}))
if len(sys.argv) > 6:    # greedy check: tokens and top-5 logprobs for the fixed prompt
    body = {"model": model, "prompt": prompt, "max_tokens": int(sys.argv[6]), "temperature": 0, "logprobs": 5, "ignore_eos": True, "return_tokens_as_token_ids": True}
    with post(body) as r: out = json.load(r)
    ch = out["choices"][0]
    print("GREEDY " + json.dumps({"tokens": ch["logprobs"]["tokens"], "top": ch["logprobs"]["top_logprobs"], "text": ch["text"]}))
