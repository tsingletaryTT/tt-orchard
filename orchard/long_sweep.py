# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A speed sweep over input length and concurrency, and a long-prompt output comparison against the base model (issue 18).

The tool talks to a model server that is already running (the stage 2 or stage 4 server, or a shipped package's). It opens no device, so it
does not need a lease of its own; the board that serves the model is the caller's. Run it like this on a leased board with the server up:

    python3 -m orchard.long_sweep run --port <server port> --model <served model name> --out sweep.json \\
        [--api chat|completions] [--lengths 1024,8192,32768,100000,200000] [--concurrency 1,4] [--decode-tokens 64]

and the same command against the base model's server to get the control. `python3 -m orchard.long_sweep compare new.json base.json` then reports,
for every cell, how much of the new model's greedy output matches the base's (the share of the common character prefix). It refuses cells whose
prompts differ (each cell records the sha256 of its first prompt).

How a cell is measured. Each of the `concurrency` requests carries its own prompt: a first line with a nonce, the deterministic filler of
`agreement_core.long_prompt_text(input_tokens)`, and an instruction to write a long numbered list, so that the reply keeps going for the
requested decode length. The nonce is `--nonce` plus the cell and request number, so two runs with the same `--nonce` send the same prompts (a
paired comparison) and the requests inside one run differ (a prefix cache cannot answer one from another). Change `--nonce` between runs
against the same server. The prompts are built before the requests start and the requests leave together (a barrier), so a cell with
concurrency 4 measures four requests in flight. The request is streamed. The time to the first text is prefill (plus one token). The
time from the first text to the last text is decode; the wait for the end-of-stream marker is not counted. `prefill_tok_s` is the prompt
tokens the server reports (`usage.prompt_tokens`) summed over the requests and divided by the slowest first-text time; `decode_tok_s` is
`(completion_tokens - 1)` over the decode time, per sequence. A server that reports no usage gives no number (the count of streamed chunks
is not a token count), and the cell says so. A cell fails, with no number, when a request fails, when the stream carries an error event
or ends without `[DONE]`, or when a reply is shorter than half the requested decode length.

`--api chat` (the default) uses /v1/chat/completions, which the shipped package servers provide. `--api completions` uses /v1/completions for
servers that have it. The record names the requested model and says the command did not verify which weights the server has loaded.

Limits. A CPU reference at 100K tokens is too slow, so cells above `agreement_core.CPU_REFERENCE_MAX_TOKENS` are compared with the base served
on the chip, by output text, and the comparison says so. Greedy output of two different models is expected to differ; the comparison is evidence
for a person, with the base's own run as the control, and sets no pass bar. The sweep needs hours of board time at 200K tokens. Nothing here has run
on a board: the tests use a fake server.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import threading
import time
import urllib.error
import urllib.request

from orchard import agreement_core as core
from orchard import safeio

REQUEST_TIMEOUT_S = 3600.0        # choice: a 200K-token prefill on one board can take many minutes
MAX_OUTPUT_HEAD = 200             # characters of each cell's output kept for a person to read
MIN_REPLY_SHARE = 0.5             # choice: a reply shorter than this share of the requested decode length makes the cell incomplete
DEFAULT_NONCE = "orchard-sweep"
LABEL = "measured"
LOOPBACK = ("127.0.0.1", "localhost", "::1")
TASK = ("\nAfter reading the log, write a numbered list of 40 observations about the entries, one full sentence each. "
        "Do not stop early.")


def _prompt(n_tokens: int, nonce: str) -> str:
    return f"Request {nonce}.\n" + core.long_prompt_text(n_tokens) + TASK


def _netloc(host: str, port: int) -> str:
    """host:port for a URL; an IPv6 literal needs brackets."""
    return f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"


def http_send(host: str, port: int, model: str, prompt: str, max_tokens: int, api: str = "chat", timeout: float = REQUEST_TIMEOUT_S) -> dict:
    """One streamed request. Returns {t_first, t_last, t_end, prompt_tokens, completion_tokens, text} with times in seconds from the start of
    the request; the token counts are None when the server reports no `usage`. Raises OSError, ValueError or http.client.HTTPException on a
    failure, on an error event in the stream, and on a stream that ends without `[DONE]`; the caller records it."""
    if api == "chat":
        path, body = "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}]}
    elif api == "completions":
        path, body = "/v1/completions", {"model": model, "prompt": prompt}
    else:
        raise ValueError(f"api must be 'chat' or 'completions', got {api!r}")
    body.update({"max_tokens": max_tokens, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}})
    req = urllib.request.Request(f"http://{_netloc(host, port)}{path}", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    start = time.monotonic()
    t_first = t_last = None
    text, usage, done = [], {}, False
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            obj = json.loads(payload)
            if not isinstance(obj, dict):
                raise ValueError("a stream event is not an object")
            if obj.get("error") is not None:
                raise ValueError(f"the server sent an error event: {str(obj['error'])[:200]}")
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                piece = (choice.get("delta") or {}).get("content") if api == "chat" else choice.get("text")
                if isinstance(piece, str) and piece:
                    text.append(piece)
                    t_last = time.monotonic() - start
                    if t_first is None:
                        t_first = t_last
    end = time.monotonic() - start
    if not done:
        raise ValueError("the stream ended without the [DONE] marker")
    if t_first is None:
        raise ValueError("the server streamed no text")

    def count(key):
        v = usage.get(key)
        return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
    return {"t_first": t_first, "t_last": t_last, "t_end": end, "text": "".join(text),
            "prompt_tokens": count("prompt_tokens"), "completion_tokens": count("completion_tokens")}


def measure_cell(send, cell: dict, decode_tokens: int, nonce_base: str) -> dict:
    """One cell: `concurrency` streamed requests of about `input_tokens` tokens that leave together. `send(prompt, max_tokens)` returns the dict of
    `http_send`. The result carries the cell's plan fields plus the measured numbers, or an `error`."""
    n, conc = cell["input_tokens"], cell["concurrency"]
    prompts = [_prompt(n, f"{nonce_base}-{n}-{conc}-{i}") for i in range(conc)]      # built before any request leaves
    results, errors = [None] * conc, []
    barrier = threading.Barrier(conc)

    def work(i):
        try:
            barrier.wait(timeout=60)
            results[i] = send(prompts[i], decode_tokens)
        except threading.BrokenBarrierError:
            errors.append("the requests could not be started together")
        except (OSError, ValueError, http.client.HTTPException, KeyError, TypeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}"[:300])

    threads = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out = {**cell, "prefill_tok_s": None, "decode_tok_s": None, "prompt_tokens": None, "completion_tokens": None, "output_sha256": None,
           "output_head": None, "prompt_sha256": hashlib.sha256(prompts[0].encode("utf-8")).hexdigest(), "error": None}
    if errors or any(r is None for r in results):
        out["error"] = "; ".join(errors)[:600] or "a request returned nothing"
        return out
    pt = [r["prompt_tokens"] for r in results]
    toks = [r["completion_tokens"] for r in results]
    problems = []
    if any(x is None for x in pt) or any(x is None for x in toks):
        problems.append("the server reported no token usage, so no rate is given (a streamed chunk is not a token)")
    else:
        out["prompt_tokens"], out["completion_tokens"] = min(pt), min(toks)
        short = [c for c in toks if c < max(8, int(decode_tokens * MIN_REPLY_SHARE))]
        if short:
            problems.append(f"a reply had {min(short)} tokens; at least {max(8, int(decode_tokens * MIN_REPLY_SHARE))} are needed for a decode speed")
        else:
            slowest_first = max(r["t_first"] for r in results)
            if slowest_first > 0:
                out["prefill_tok_s"] = round(sum(pt) / slowest_first, 3)      # aggregate, to the slowest first text
            decode_s = [r["t_last"] - r["t_first"] for r in results]
            if all(d > 0 for d in decode_s):
                out["decode_tok_s"] = round(sum((c - 1) / d for c, d in zip(toks, decode_s)) / conc, 3)       # per sequence
            else:
                problems.append("a request produced no measurable decode time")
    out["error"] = "; ".join(problems) or None
    out["output_sha256"] = hashlib.sha256(results[0]["text"].encode("utf-8")).hexdigest()
    out["output_head"] = results[0]["text"][:MAX_OUTPUT_HEAD]
    return out


def run_sweep(send, plan: list[dict], decode_tokens: int = 64, *, nonce_base: str = DEFAULT_NONCE, model: str | None = None,
              api: str | None = None, progress=None) -> dict:
    """Measure every cell of `plan` (agreement_core.sweep_plan) and return the record. A failing cell does not stop the sweep."""
    cells = []
    for cell in plan:
        result = measure_cell(send, cell, decode_tokens, nonce_base)
        cells.append(result)
        if progress:
            progress(result)
    return {"label": LABEL, "model": {"requested": model, "verified": False,
                                      "note": "the name sent in each request; this command does not check which weights the server loaded"},
            "api": api, "nonce": nonce_base, "decode_tokens": decode_tokens, "plan": plan, "summary": core.sweep_summary(cells), "cells": cells,
            "limits": ("per-sequence decode speed from a streamed request; prefill speed to the slowest first text of the concurrent requests; "
                       "cells above %d input tokens have no CPU reference and are checked against the base on the chip"
                       % core.CPU_REFERENCE_MAX_TOKENS)}


def compare_sweeps(new: dict, base: dict) -> dict:
    """For each cell present in both records, the share of the longer output that the two outputs share as a common character prefix
    (1.0 means identical heads of output). A cell whose two prompts differ (another sha256) is listed as `different_prompts` with no share,
    and so is a cell without output in either record. This is a reading aid and sets no bar."""
    def index(rec):
        out = {}
        for c in rec.get("cells", []) if isinstance(rec, dict) and isinstance(rec.get("cells"), list) else []:
            if isinstance(c, dict):
                out[(c.get("input_tokens"), c.get("concurrency"))] = c
        return out
    a, b = index(new), index(base)
    rows = []
    for key in sorted(k for k in a if k in b and all(isinstance(x, int) and not isinstance(x, bool) for x in k)):
        ha, hb = a[key].get("output_head"), b[key].get("output_head")
        same_prompt = isinstance(a[key].get("prompt_sha256"), str) and a[key].get("prompt_sha256") == b[key].get("prompt_sha256")
        share = None
        if same_prompt and isinstance(ha, str) and isinstance(hb, str) and (ha or hb):
            common = 0
            for x, y in zip(ha, hb):
                if x != y:
                    break
                common += 1
            share = round(common / max(len(ha), len(hb)), 4)
        rows.append({"input_tokens": key[0], "concurrency": key[1], "common_prefix_share": share, "different_prompts": not same_prompt,
                     "identical_output": (same_prompt and a[key].get("output_sha256") is not None
                                          and a[key].get("output_sha256") == b[key].get("output_sha256"))})
    return {"cells": rows, "note": "output heads of two models; a reading aid, no pass bar"}


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m orchard.long_sweep", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure a running server")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--host", default="127.0.0.1", help="the server's address; only a loopback address is accepted")
    r.add_argument("--model", required=True)
    r.add_argument("--api", choices=("chat", "completions"), default="chat")
    r.add_argument("--nonce", default=DEFAULT_NONCE, help="the same value in two runs sends the same prompts; change it between runs on one server")
    r.add_argument("--out", required=True)
    r.add_argument("--lengths", default=",".join(str(n) for n in core.DEFAULT_LENGTHS))
    r.add_argument("--concurrency", default=",".join(str(n) for n in core.DEFAULT_CONCURRENCY))
    r.add_argument("--decode-tokens", type=int, default=64)
    c = sub.add_parser("compare", help="compare two sweep records' outputs")
    c.add_argument("new")
    c.add_argument("base")
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        try:
            recs = [safeio.read_json(a, max_bytes=64 << 20) for a in (args.new, args.base)]
        except safeio.SafeReadError as exc:
            print(f"long_sweep: cannot read a sweep record: {exc.reason}", file=sys.stderr)
            return 2
        print(json.dumps(compare_sweeps(*recs), indent=2))
        return 0
    if args.host not in LOOPBACK:
        print("long_sweep: only a server on this machine's loopback address is accepted", file=sys.stderr)
        return 2
    try:
        plan = core.sweep_plan(_ints(args.lengths), _ints(args.concurrency))
    except ValueError as exc:
        print(f"long_sweep: {exc}", file=sys.stderr)
        return 2
    if args.decode_tokens < 8:
        print("long_sweep: --decode-tokens must be at least 8 for a decode speed to mean anything", file=sys.stderr)
        return 2
    record = run_sweep(lambda prompt, k: http_send(args.host, args.port, args.model, prompt, k, args.api), plan, args.decode_tokens,
                       nonce_base=args.nonce, model=args.model, api=args.api,
                       progress=lambda cell: print(json.dumps({k: cell[k] for k in ("input_tokens", "concurrency", "prefill_tok_s", "decode_tok_s", "error")}),
                                                   flush=True))
    safeio.atomic_write(args.out, json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
