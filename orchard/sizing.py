"""Measure a CPU-served model so the tier choice rests on numbers (spec sections 5.1 and 12).

Talks to a local ollama over HTTP. It measures load time, prefill speed, decode speed and the
resident size ollama reports. It does not pull or start anything. The caller runs the model server
and decides which models to download.

The ceiling is arithmetic: each generated token reads every active weight once, so tokens per second
cannot exceed memory bandwidth divided by the gigabytes read per token. A measured speed far below
the ceiling points at compute or contention. The theoretical bandwidth is the DIMM rate times the bus
width times the channels, and the real figure is lower.

Failures a person can fix (server not running, HTTP error, bad JSON, timeout, unreadable prompt file,
locked ledger) end the run with one line on stderr and exit code 1. They do not print a traceback.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked


class SizingError(Exception):
    """A measurement could not be completed. The message says why, in one line."""


def theoretical_gb_s(mt_s: float, channels: int = 2, bus_bytes: int = 8) -> float:
    return mt_s * bus_bytes * channels / 1000


def ceiling_tok_s(bandwidth_gb_s: float, gb_per_token: float) -> float:
    return bandwidth_gb_s / gb_per_token


def _request(host: str, path: str, body: dict | None, timeout: float) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(host + path, data=data,
                                 headers={"Content-Type": "application/json"})
    url = host + path
    # Each failure becomes a SizingError with a plain message, so main() can print one line.
    # HTTPError is a subclass of URLError, so it must be listed first.
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            reply = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise SizingError(f"{url} answered HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        # A timeout while connecting arrives here too, wrapped in a URLError.
        raise SizingError(f"cannot reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        # A timeout while reading the reply is raised bare, not wrapped.
        raise SizingError(f"{url} did not answer within {timeout} seconds") from exc
    except OSError as exc:
        raise SizingError(f"connection to {url} failed: {exc}") from exc
    except ValueError as exc:  # json.JSONDecodeError and bad UTF-8 are both ValueError
        raise SizingError(f"{url} did not return valid JSON") from exc
    if not isinstance(reply, dict):
        raise SizingError(f"{url} did not return a JSON object")
    return reply


def _rate(count, duration_ns):
    """Tokens per second, or None when ollama did not report the pair."""
    if not count or not duration_ns:
        return None
    return count / (duration_ns / 1e9)


def measure(host: str, model: str, prompt: str, num_predict: int = 128,
            timeout: float = 900) -> dict:
    reply = _request(host, "/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        # temperature 0 and a fixed seed make the output reproducible between runs.
        "options": {"num_predict": num_predict, "temperature": 0, "seed": 1},
    }, timeout)
    # ollama can answer HTTP 200 with an error text (for example a model it cannot load).
    if "error" in reply:
        raise SizingError(f"ollama reported an error for {model}: {reply['error']}")
    running = _request(host, "/api/ps", None, timeout).get("models", [])
    resident = next((m.get("size") for m in running
                     if model in (m.get("name"), m.get("model"))), None)
    load_ns = reply.get("load_duration")
    return {
        "model": model,
        # None means ollama did not say. Reporting 0.0 would look like an instant load.
        "load_s": None if load_ns is None else load_ns / 1e9,
        "prompt_tokens": reply.get("prompt_eval_count"),
        # ollama leaves the prompt fields out when it reused a cached prompt.
        "prefill_tok_s": _rate(reply.get("prompt_eval_count"), reply.get("prompt_eval_duration")),
        "decode_tokens": reply.get("eval_count"),
        "decode_tok_s": _rate(reply.get("eval_count"), reply.get("eval_duration")),
        "resident_bytes": resident,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Measure a model served by a local ollama.")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--model", action="append", required=True, help="repeat for several models")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--num-predict", type=int, default=128)
    p.add_argument("--timeout", type=float, default=900,
                   help="seconds to wait for each request; a cold CPU load can be slow")
    p.add_argument("--mt-s", type=float, default=3600,
                   help="configured DIMM rate; the host reports 3600 (rated 5600)")
    p.add_argument("--gb-per-token", type=float,
                   help="gigabytes read per token (about the active weights); enables the ceiling")
    p.add_argument("--note", default="", help="conditions, for example 'coder serving'")
    p.add_argument("--ledger", help="append each result to this ledger as a measured entry")
    args = p.parse_args(argv)

    ledger = None
    try:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read()
        bandwidth = theoretical_gb_s(args.mt_s)
        ledger = Ledger(args.ledger) if args.ledger else None
        for model in args.model:
            result = measure(args.host, model, prompt, args.num_predict, args.timeout)
            result.update(note=args.note, theoretical_gb_s=bandwidth)
            if args.gb_per_token:
                result["ceiling_tok_s"] = ceiling_tok_s(bandwidth, args.gb_per_token)
            print(json.dumps(result, indent=2))
            if ledger:
                ledger.append("measurement", None, label="measured", **result)
    except (SizingError, OSError, UnicodeDecodeError, LedgerLocked, LedgerCorrupt) as exc:
        print(f"sizing: {exc}", file=sys.stderr)
        return 1
    finally:
        # Release the ledger lock whether the run finished or failed.
        if ledger:
            ledger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
