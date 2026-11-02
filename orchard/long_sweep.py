# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A speed sweep over input length and concurrency, and a long-prompt output comparison against the base model (issue 18).

The tool talks to a model server that is already running (the stage 2 or stage 4 server, or a shipped package's). It opens no device, so it
does not need a lease of its own; the board that serves the model is the caller's. Run it like this on a leased board with the server up:

    python3 -m orchard.long_sweep run --port <server port> --model <served model name> --out sweep.json \\
        [--lengths 1024,8192,32768,100000,200000] [--concurrency 1,4] [--decode-tokens 64]

and the same command against the base model's server to get the control. `python3 -m orchard.long_sweep compare new.json base.json` then reports,
for every cell, how much of the new model's greedy output matches the base's (the share of the common character prefix).

How a cell is measured. Each of the `concurrency` requests carries its own prompt: a unique first line (a nonce) and the deterministic filler
of `agreement_core.long_prompt_text(input_tokens)`. The unique first line keeps the server's prefix cache from answering a request from an
earlier one. The request is streamed. The time to the first token is prefill (plus one token); the time from the first token to the last is
decode. `prefill_tok_s` is the prompt tokens the server reports (`usage.prompt_tokens`, else the length asked for) over the time to first
token; `decode_tok_s` is `(completion_tokens - 1)` over the decode time, per sequence. A cell whose requests fail keeps the error and no
number. Input lengths are asked for in tokens through an English-word estimate, and the record carries the count the server reports.

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
LABEL = "measured"


def _prompt(n_tokens: int, nonce: str) -> str:
    return f"Request {nonce}.\n" + core.long_prompt_text(n_tokens)


def http_send(host: str, port: int, model: str, prompt: str, max_tokens: int, timeout: float = REQUEST_TIMEOUT_S) -> dict:
    """One streamed completion. Returns {t_first, t_end, prompt_tokens, completion_tokens, text} with times in seconds from the start of the
    request. Raises OSError, ValueError or http.client.HTTPException on a failure; the caller records it."""
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode("utf-8")
    req = urllib.request.Request(f"http://{host}:{port}/v1/completions", data=body, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    t_first, chunks, text, usage = None, 0, [], {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            for choice in obj.get("choices") or []:
                piece = choice.get("text") or ""
                if piece:
                    chunks += 1
                    text.append(piece)
                    if t_first is None:
                        t_first = time.monotonic() - start
    end = time.monotonic() - start
    if t_first is None:
        raise ValueError("the server streamed no text")
    comp = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    return {"t_first": t_first, "t_end": end, "text": "".join(text),
            "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) else None,
            "completion_tokens": comp if isinstance(comp, int) and not isinstance(comp, bool) else chunks}


def measure_cell(send, cell: dict, decode_tokens: int, nonce_base: str) -> dict:
    """One cell: `concurrency` simultaneous streamed requests of about `input_tokens` tokens. `send(prompt, max_tokens)` returns the dict of
    `http_send`. The result carries the cell's plan fields plus the measured numbers, or an `error`."""
    n, conc = cell["input_tokens"], cell["concurrency"]
    results, errors = [None] * conc, []

    def work(i):
        try:
            results[i] = send(_prompt(n, f"{nonce_base}-{n}-{conc}-{i}"), decode_tokens)
        except (OSError, ValueError, http.client.HTTPException, KeyError, TypeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}"[:300])

    threads = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out = {**cell, "prefill_tok_s": None, "decode_tok_s": None, "prompt_tokens": None, "completion_tokens": None, "output_sha256": None,
           "output_head": None, "error": None}
    if errors or any(r is None for r in results):
        out["error"] = "; ".join(errors)[:600] or "a request returned nothing"
        return out
    pt = [r["prompt_tokens"] if r["prompt_tokens"] else n for r in results]
    firsts = [r["t_first"] for r in results]
    decode_s = [r["t_end"] - r["t_first"] for r in results]
    toks = [r["completion_tokens"] for r in results]
    out["prompt_tokens"], out["completion_tokens"] = min(pt), min(toks)
    if all(f > 0 for f in firsts):
        out["prefill_tok_s"] = round(sum(pt) / max(firsts), 3)           # aggregate over the concurrent requests, to the slowest first token
    rates = [(c - 1) / d for c, d in zip(toks, decode_s) if c > 1 and d > 0]
    if len(rates) == conc:
        out["decode_tok_s"] = round(sum(rates) / len(rates), 3)           # per sequence
    elif out["error"] is None:
        out["error"] = "a request produced no measurable decode time"
    out["output_sha256"] = hashlib.sha256(results[0]["text"].encode("utf-8")).hexdigest()
    out["output_head"] = results[0]["text"][:MAX_OUTPUT_HEAD]
    return out


def run_sweep(send, plan: list[dict], decode_tokens: int = 64, *, nonce_base: str | None = None, progress=None) -> dict:
    """Measure every cell of `plan` (agreement_core.sweep_plan) and return the record. A failing cell does not stop the sweep."""
    nonce_base = nonce_base or hashlib.sha256(str(time.time_ns()).encode()).hexdigest()[:8]
    cells = []
    for cell in plan:
        result = measure_cell(send, cell, decode_tokens, nonce_base)
        cells.append(result)
        if progress:
            progress(result)
    return {"label": LABEL, "decode_tokens": decode_tokens, "plan": plan, "summary": core.sweep_summary(cells), "cells": cells,
            "limits": ("per-sequence decode speed from a streamed request; prefill speed to the slowest first token of the concurrent requests; "
                       "cells above %d input tokens have no CPU reference and are checked against the base on the chip"
                       % core.CPU_REFERENCE_MAX_TOKENS)}


def compare_sweeps(new: dict, base: dict) -> dict:
    """For each cell present in both records, the share of the longer output that the two outputs share as a common character prefix
    (1.0 means identical heads of output). A cell without output in either record is listed with no share. This is a reading aid and sets no bar."""
    def index(rec):
        out = {}
        for c in rec.get("cells", []) if isinstance(rec, dict) and isinstance(rec.get("cells"), list) else []:
            if isinstance(c, dict):
                out[(c.get("input_tokens"), c.get("concurrency"))] = c
        return out
    a, b = index(new), index(base)
    rows = []
    for key in sorted(k for k in a if k in b and all(isinstance(x, int) for x in k)):
        ha, hb = a[key].get("output_head"), b[key].get("output_head")
        share = None
        if isinstance(ha, str) and isinstance(hb, str) and (ha or hb):
            common = 0
            for x, y in zip(ha, hb):
                if x != y:
                    break
                common += 1
            share = round(common / max(len(ha), len(hb)), 4)
        rows.append({"input_tokens": key[0], "concurrency": key[1], "common_prefix_share": share,
                     "identical_output": (a[key].get("output_sha256") is not None and a[key].get("output_sha256") == b[key].get("output_sha256"))})
    return {"cells": rows, "note": "output heads of two models; a reading aid, no pass bar"}


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m orchard.long_sweep", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="measure a running server")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--host", default="127.0.0.1", help="the server's address; only the loopback address is accepted")
    r.add_argument("--model", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--lengths", default=",".join(str(n) for n in core.DEFAULT_LENGTHS))
    r.add_argument("--concurrency", default=",".join(str(n) for n in core.DEFAULT_CONCURRENCY))
    r.add_argument("--decode-tokens", type=int, default=64)
    c = sub.add_parser("compare", help="compare two sweep records' outputs")
    c.add_argument("new")
    c.add_argument("base")
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        recs = [safeio.read_json(a, max_bytes=64 << 20) for a in (args.new, args.base)]
        print(json.dumps(compare_sweeps(*recs), indent=2))
        return 0
    if args.host not in ("127.0.0.1", "localhost", "::1"):
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
    record = run_sweep(lambda prompt, k: http_send(args.host, args.port, args.model, prompt, k), plan, args.decode_tokens,
                       progress=lambda cell: print(json.dumps({k: cell[k] for k in ("input_tokens", "concurrency", "prefill_tok_s", "decode_tok_s", "error")}),
                                                   flush=True))
    safeio.atomic_write(args.out, json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
