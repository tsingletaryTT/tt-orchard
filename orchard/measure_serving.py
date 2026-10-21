# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Time a served model on a fixed prompt set, so a performance claim has a number and a context.

`python3 -m orchard.measure_serving --label pi --run-sh .../run.sh --model-dir DIR --tt-cache DIR --hf-home DIR --out FILE`

It starts the bundle's own run.sh the way the weights-swap test does (own process group, the tensor cache, HF_HOME and the weights directory in
the environment, a lease already held by the caller), waits for health, sends one warm-up request, then each prompt `--reps` times as a
streamed chat completion with greedy decoding, and stops the server. Per run it records time to first token and the decode rate (tokens after
the first, over the time after the first). The prompt set is fixed in this file and hashed, so two runs of two models can be compared only
when their hashes match. The DFlash2 drafter's acceptance rate is read from the server's /metrics when it exports spec-decode counters; when
it does not, the file says so. Nothing here runs without hardware held by the caller (`gozer run` around the command).
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWAP_TEMPLATE = HERE / "skills" / "weights-swap-templates" / "serve_and_compare.py"
MIN_TOKENS_ASKED = 100                  # the operator's floor: at least this many generated tokens per prompt

CODE_SAMPLE = '''def retry(fn, attempts=5, base=0.5):
    delay = base
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i == attempts:
                raise
            time.sleep(delay)
            delay *= 2
'''

PROMPTS = [
    {"id": "pi-coding", "temperature": 0, "max_tokens": 384,
     "messages": [{"role": "system", "content": "You are Pi, a coding agent working inside a repository. Read the code you are given, change it "
                                                  "as little as the task needs, and explain the change in two sentences."},
                  {"role": "user", "content": "This helper in src/utils/retry.py never re-raises the last error and sleeps after the final "
                                              "attempt. Fix it and add a pytest test for the fixed behaviour.\n\n```python\n" + CODE_SAMPLE + "```"}]},
    {"id": "tool-call", "temperature": 0, "max_tokens": 256, "tool_choice": "auto",
     "tools": [{"type": "function", "function": {"name": "run_tests", "description": "Run the project's unit tests and return the failures.",
                                                  "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
               {"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository.",
                                                  "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}],
     "messages": [{"role": "system", "content": "You are Pi, a coding agent. Use the tools to look at the repository before you answer."},
                  {"role": "user", "content": "Run the tests under tests/utils and tell me which ones fail."}]},
    {"id": "tool-call-write", "temperature": 0, "max_tokens": 384, "tool_choice": "auto",
     "tools": [{"type": "function", "function": {"name": "write_file", "description": "Write a whole file in the repository.",
                                                  "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                                                                 "required": ["path", "content"]}}}],
     "messages": [{"role": "system", "content": "You are Pi, a coding agent. Use the tools to change the repository; do not paste code into your reply."},
                  {"role": "user", "content": "Create tests/test_retry.py with four pytest tests for the retry helper in src/utils/retry.py: it "
                                              "succeeds on the first call, succeeds after two failures, re-raises the last error, and sleeps with "
                                              "doubling delays. Write the whole file with the write_file tool."}]},
    {"id": "chat", "temperature": 0, "max_tokens": 256,
     "messages": [{"role": "user", "content": "Write the text I send my landlord about the broken boiler."}]},
]


def warmup_prompt() -> dict:
    """One short request, sent once and not recorded, so the first measured request is not the one that pays for any lazy setup."""
    return {"id": "warmup", "temperature": 0, "max_tokens": 16, "messages": PROMPTS[-1]["messages"]}


class StreamError(RuntimeError):
    """The server sent an error object inside the stream (it answers 200 and then reports the refusal)."""


class NoTokens(RuntimeError):
    """The stream ended without a content, reasoning or tool-call chunk."""


def prompt_set_hash(prompts) -> str:
    return hashlib.sha256(json.dumps(prompts, sort_keys=True).encode("utf-8")).hexdigest()


def request_body(model: str, prompt: dict) -> dict:
    body = {"model": model, "messages": prompt["messages"], "stream": True, "stream_options": {"include_usage": True},
            "temperature": prompt["temperature"], "max_tokens": prompt["max_tokens"]}
    for key in ("tools", "tool_choice"):
        if key in prompt:
            body[key] = prompt[key]
    return body


def read_stream(lines, *, clock=time.monotonic) -> dict:
    """Walk server-sent-event lines (`data: {...}`). t_first is the clock at the first chunk with content, reasoning or a tool call;
    t_last is the clock at the last chunk before [DONE] (the usage chunk, when there is one)."""
    t_first = t_last = None
    text, calls = [], {}
    usage = {}
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        now = clock()
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        if isinstance(chunk.get("error"), dict):
            raise StreamError(str(chunk["error"].get("message") or chunk["error"]))
        t_last = now
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            d = choice.get("delta") or {}
            got = False
            for key in ("content", "reasoning_content", "reasoning"):
                if d.get(key):
                    text.append(d[key])
                    got = True
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": None, "arguments": ""})
                fn = tc.get("function") or {}
                slot["name"] = fn.get("name") or slot["name"]
                slot["arguments"] += fn.get("arguments") or ""
                got = True
            if got and t_first is None:
                t_first = now
    if t_first is None:
        raise NoTokens("the stream held no content, reasoning or tool-call chunk")
    return {"t_first": t_first, "t_last": t_last, "text": "".join(text), "tool_calls": [calls[i] for i in sorted(calls)],
            "completion_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens")}


def timing(*, sent: float, t_first: float, t_last: float, completion_tokens) -> dict:
    rate = None
    if isinstance(completion_tokens, int) and completion_tokens > 1 and t_last > t_first:
        rate = (completion_tokens - 1) / (t_last - t_first)
    return {"ttft_s": t_first - sent, "decode_tok_s": rate, "completion_tokens": completion_tokens}


def _stat(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(runs: list[dict]) -> dict:
    return {"runs": len(runs), "ttft_s": _stat([r["ttft_s"] for r in runs]), "decode_tok_s": _stat([r["decode_tok_s"] for r in runs]),
            "completion_tokens": _stat([r["completion_tokens"] for r in runs])}


SPEC_LINE = re.compile(r"^(vllm:spec_decode_[A-Za-z_]*(?:accepted|draft)_tokens_total)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.MULTILINE)


def spec_decode_counters(text: str) -> dict:
    out: dict = {}
    for name, value in SPEC_LINE.findall(text or ""):
        out[name] = out.get(name, 0.0) + float(value)
    return out


def acceptance(*, before: dict, after: dict):
    """accepted / drafted tokens between two scrapes, or None when the counters are missing or nothing was drafted."""
    def pick(d, word):
        return next((v for k, v in d.items() if word in k), None)
    acc0, acc1, dr0, dr1 = pick(before, "accepted"), pick(after, "accepted"), pick(before, "draft"), pick(after, "draft")
    if None in (acc0, acc1, dr0, dr1) or dr1 - dr0 <= 0:
        return None
    return (acc1 - acc0) / (dr1 - dr0)


# ---- talking to a server ------------------------------------------------------------------------------------------------------------

def post_stream(port: int, body: dict, timeout: float = 900.0):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"the server answered {resp.status}: {resp.read()[:300]!r}")
    for raw in resp:
        yield raw.decode("utf-8", errors="replace").strip()
    conn.close()


def get_text(port: int, path: str, timeout: float = 30.0) -> str:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.read().decode("utf-8", errors="replace") if resp.status == 200 else ""
    except OSError:
        return ""
    finally:
        conn.close()


def run_prompt(port: int, model: str, prompt: dict, *, clock=time.monotonic) -> dict:
    sent = clock()
    ev = read_stream(post_stream(port, request_body(model, prompt)), clock=clock)
    t = timing(sent=sent, t_first=ev["t_first"], t_last=ev["t_last"], completion_tokens=ev["completion_tokens"])
    return {**t, "prompt_tokens": ev["prompt_tokens"], "tool_calls": ev["tool_calls"],
            "text_head": ev["text"][:200], "text_sha256": hashlib.sha256(ev["text"].encode("utf-8")).hexdigest()}


def load_swap_helpers():
    spec = importlib.util.spec_from_file_location("serve_and_compare_for_measure", str(SWAP_TEMPLATE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure(port: int, model: str, prompts: list[dict], reps: int) -> dict:
    out = {}
    for p in prompts:
        runs = [run_prompt(port, model, p) for _ in range(reps)]
        out[p["id"]] = {"summary": summarize(runs), "runs": runs,
                        "meets_100_tokens": all((r["completion_tokens"] or 0) >= MIN_TOKENS_ASKED for r in runs)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--model-id", required=True, help="the id the tensor cache marker must name (the weights served)")
    ap.add_argument("--run-sh", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tt-cache", required=True)
    ap.add_argument("--hf-home", required=True)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--health-timeout", type=float, default=2400.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sc = load_swap_helpers()
    cache, out_path = Path(a.tt_cache), Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sc.guard_cache(cache, a.model_id)
    log_path = out_path.with_suffix(".server.log")
    env = dict(os.environ, TT_CACHE_PATH=str(cache), TT_CACHE_HOME=str(cache), HF_HOME=a.hf_home, HF_HUB_OFFLINE="1",
               MODEL_WEIGHTS_DIR=a.model_dir, HF_MODEL=a.model_dir)
    log = open(log_path, "wb")
    proc = subprocess.Popen(["bash", a.run_sh, "--port", str(a.port)], env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    try:
        ready_s = sc.wait_healthy(proc, a.port, a.health_timeout, log_path)
        models = json.loads(get_text(a.port, "/v1/models") or "{}")
        served = (models.get("data") or [{}])[0].get("id") or a.model_dir
        metrics0 = spec_decode_counters(get_text(a.port, "/metrics"))
        run_prompt(a.port, served, warmup_prompt())                                       # warm-up: not recorded
        results = measure(a.port, served, PROMPTS, a.reps)
        metrics1 = spec_decode_counters(get_text(a.port, "/metrics"))
    finally:
        sc.stop_server(proc)
        log.close()
    record = {"label": a.label, "model_id": a.model_id, "served_name": served, "server_ready_s": round(ready_s, 1),
              "wall_s": round(time.time() - t0, 1), "prompt_set_sha256": prompt_set_hash(PROMPTS), "reps": a.reps,
              "greedy": True, "prompts": results,
              "drafter_acceptance": acceptance(before=metrics0, after=metrics1),
              "drafter_acceptance_note": ("computed from the server's /metrics spec-decode counters" if metrics1 else
                                          "not measured: this server exports no spec-decode counters on /metrics and prints none in its log"),
              "spec_decode_counters": metrics1, "run_sh": str(a.run_sh), "server_log": str(log_path),
              "unit_notes": {"ttft_s": "seconds from sending the request to the first content, reasoning or tool-call chunk",
                             "decode_tok_s": "(completion tokens - 1) / seconds from the first chunk to the last, per run"}}
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    for pid, r in results.items():
        s = r["summary"]
        print(f"  {pid}: ttft {s['ttft_s']['median']:.2f} s, decode "
              f"{(s['decode_tok_s'] or {}).get('median', float('nan')):.1f} tok/s, tokens {s['completion_tokens']['median']:.0f}"
              + ("" if r["meets_100_tokens"] else "  (under 100 tokens)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
