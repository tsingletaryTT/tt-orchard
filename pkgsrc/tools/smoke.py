# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Smoke test for a running server (standard library only):

    python tools/smoke.py --port 20110 [--out smoke.json]

Sends chat requests and checks the answers: a short question (the answer must mention Paris), a request cut by max_tokens (finish_reason
`length`), a request that ends on the end-of-text token (finish_reason `stop`), a stop string, a streamed request (first-token time is the
time to the first content chunk), a long prompt, and two requests the server must refuse (400). Writes the measurements as JSON.
Exit status 0 when every check passed."""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post(base, body, stream=False, timeout=600):
    rq = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(rq, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), time.time() - t0
    if not stream:
        return r.status, json.load(r), time.time() - t0
    first, text, fin = None, "", None
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: {"):
            ch = json.loads(line[6:])["choices"][0]
            if ch["delta"].get("content") and first is None:
                first = time.time() - t0
            text += ch["delta"].get("content", "")
            fin = ch["finish_reason"] or fin
    return 200, {"text": text, "finish_reason": fin, "first_content_s": first}, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=20110)
    ap.add_argument("--out")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    res, checks = {}, []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)[:200]})
        print(("PASS " if ok else "FAIL ") + name, str(detail)[:120], flush=True)

    health = json.load(urllib.request.urlopen(base + "/health"))
    res["health"] = health
    user = lambda c: [{"role": "user", "content": c}]  # noqa: E731
    code, r, dt = post(base, {"messages": user("What is the capital of France? Answer in one sentence."), "max_tokens": 48})
    res["short"] = {"text": r["choices"][0]["message"]["content"], "finish_reason": r["choices"][0]["finish_reason"], "usage": r["usage"], "timings": r["timings"], "wall_s": dt}
    check("short answer mentions Paris", "Paris" in res["short"]["text"], res["short"]["text"])
    check("short answer ended on the end-of-text token", res["short"]["finish_reason"] == "stop", res["short"]["finish_reason"])
    code, r, dt = post(base, {"messages": user("Describe the sea in a long paragraph."), "max_tokens": 64})
    res["decode_run"] = {"usage": r["usage"], "timings": r["timings"], "wall_s": dt}
    check("a 64-token completion runs to max_tokens", r["usage"]["completion_tokens"] == 64 and r["choices"][0]["finish_reason"] == "length", r["usage"])
    code, r, _ = post(base, {"messages": user("Write a long story about a dragon."), "max_tokens": 5})
    check("max_tokens=5 gives finish_reason length and 5 tokens", r["choices"][0]["finish_reason"] == "length" and r["usage"]["completion_tokens"] == 5, r["usage"])
    code, r, _ = post(base, {"messages": user("Count from 1 to 10 separated by commas."), "max_tokens": 64, "stop": ["4"]})
    t = r["choices"][0]["message"]["content"]
    check("stop string cuts the text before it", "4" not in t and "3" in t and r["choices"][0]["finish_reason"] == "stop", t)
    code, r, dt = post(base, {"messages": [{"role": "system", "content": "You answer in French."}, {"role": "user", "content": "Say hello."}], "max_tokens": 24, "stream": True}, stream=True)
    res["stream"] = {**r, "wall_s": dt}
    check("stream returns text and a finish reason", bool(r["text"]) and r["finish_reason"] in ("stop", "length"), r)
    long_prompt = " ".join(f"Item {i} is number {i * 7 % 13}." for i in range(180)) + " How many items were listed? Answer with a number."
    code, r, dt = post(base, {"messages": user(long_prompt), "max_tokens": 16})
    res["long_prompt"] = {"prompt_tokens": r["usage"]["prompt_tokens"], "timings": r["timings"], "wall_s": dt}
    check("a prompt of about a thousand tokens is accepted", r["usage"]["prompt_tokens"] > 900, r["usage"])
    code, r, _ = post(base, {"messages": user("x " * (health["context"] + 50)), "max_tokens": 8})
    check("a prompt longer than the context is refused with 400", code == 400, code)
    code, r, _ = post(base, {"messages": []})
    check("an empty messages list is refused with 400", code == 400, code)
    res["checks"] = checks
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(0 if all(c["ok"] for c in checks) else 1)


if __name__ == "__main__":
    main()
