#!/usr/bin/env python3
"""Benchmark a served MiniCPM3 thin bundle over HTTP (standard library only).

    bench.py --url http://127.0.0.1:PORT --tokenizer-dir DIR --out result.json [--prefill 512,2048] [--decode-runs 3]

Decode: five different prompts, one discarded warm-up run per prompt and 3 measured runs of 64 output tokens each (non-streaming; the
server reports decode tokens per second in `timings`). Prefill: a prompt of about N tokens (counted with the model's tokenizer), one
output token, 1 warm-up and 3 measured runs; the time is the whole request (the bundle feeds the prompt one token per decode step).
The one-minute load average is recorded at the start and end of every run. Nothing is read from the environment but the arguments."""
import argparse, json, os, statistics, sys, time, urllib.parse, urllib.request

PROMPTS = {
    "capital": "What is the capital of France? Answer in one word.",
    "story": "Write a short story of about 60 words about a lighthouse keeper on a stormy night.",
    "code": "Write a Python function that returns the n-th Fibonacci number, with a one-line docstring.",
    "translate": "Translate into French: The weather is very nice today, so we will walk along the river after lunch.",
    "summary": "Summarise in two sentences: Multi-latent attention compresses the keys and values of every token into one small latent vector, "
               "which is stored in the cache and expanded per head when attention is computed. This makes the cache much smaller than in "
               "standard multi-head attention, at the cost of extra matrix multiplications.",
}
FILLER = ("The harbour town kept a ledger of every ship that passed the lamp, and each entry named the cargo, the weather and the hour. "
          "A clerk copied the entries into a second book each morning so that one copy would survive a fire. ")


def is_loopback(url):
    """True only for plain http to 127.0.0.1, localhost or ::1, judged by the parsed host name (a prefix test accepts http://127.0.0.1.example)."""
    try:
        parts = urllib.parse.urlsplit(url)
        return parts.scheme == "http" and parts.hostname in ("127.0.0.1", "localhost", "::1")
    except ValueError:
        return False


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def load1():
    return os.getloadavg()[0]


def spread(xs):
    return {"min": min(xs), "median": statistics.median(xs), "max": max(xs), "n": len(xs)}


def prompt_of_tokens(tok, n):
    text = FILLER
    while len(tok(text)["input_ids"]) < n:
        text += FILLER
    ids = tok(text)["input_ids"][:n]
    return tok.decode(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefill", default="")
    ap.add_argument("--decode-runs", type=int, default=3)
    ap.add_argument("--skip-decode", action="store_true")
    a = ap.parse_args()
    if not is_loopback(a.url):
        ap.error("--url must be plain http to 127.0.0.1, localhost or ::1: the client measures a server on this host")
    res = {"url_is_loopback": True, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "load_start": load1(), "decode": {}, "prefill": {}}
    if not a.skip_decode:
        for name, text in PROMPTS.items():
            body = {"messages": [{"role": "user", "content": text}], "max_tokens": 64}
            post(a.url, body, 600)                                   # warm-up, discarded
            runs = []
            for _ in range(a.decode_runs):
                l0 = load1()
                out, wall = post(a.url, body, 600)
                t = out.get("timings", {})
                runs.append({"decode_tok_s": t.get("decode_tokens_per_s") or t.get("decode_tok_s"), "ttft_s": t.get("first_token_s") or t.get("ttft_s"),
                             "completion_tokens": out["usage"]["completion_tokens"], "prompt_tokens": out["usage"]["prompt_tokens"],
                             "wall_s": round(wall, 3), "load1_start": l0, "load1_end": load1(), "timings_raw": t})
            res["decode"][name] = runs
            print(name, [r["decode_tok_s"] for r in runs], flush=True)
    if a.prefill:
        sys.path.insert(0, a.tokenizer_dir)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer_dir)
        for n in [int(x) for x in a.prefill.split(",")]:
            text = prompt_of_tokens(tok, n)
            body = {"messages": [{"role": "user", "content": text}], "max_tokens": 1}
            post(a.url, body, 3600)                                  # warm-up, discarded
            runs = []
            for _ in range(3):
                l0 = load1()
                out, wall = post(a.url, body, 3600)
                runs.append({"wall_s": round(wall, 3), "prompt_tokens": out["usage"]["prompt_tokens"], "load1_start": l0, "load1_end": load1()})
            res["prefill"][str(n)] = runs
            print("prefill", n, [r["wall_s"] for r in runs], flush=True)
    res["load_end"] = load1()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
