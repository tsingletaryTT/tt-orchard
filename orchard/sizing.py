"""Measure a CPU-served model so the tier choice rests on numbers (spec sections 5.1 and 12).

Talks to a local ollama over HTTP. It measures load time, prefill speed, decode speed and the
resident size ollama reports. It does not pull or start anything. The caller runs the model server
and decides which models to download.

The ceiling is arithmetic: each generated token reads every active weight once, so tokens per second
cannot exceed memory bandwidth divided by the gigabytes read per token. A measured speed far below
the ceiling points at compute or contention. The theoretical bandwidth is the DIMM rate times the bus
width times the channels, and the real figure is lower.

What a result means, so a number is not read as something it is not:
  * load_s is only meaningful for a cold first run (the model was not already resident). load_cold
    says whether the run looked cold. A warm run reports a tiny load time that says nothing about
    how long a real cold start takes.
  * prefill_tok_s needs a prompt ollama really processed. When ollama reused its prompt cache, the
    token count is a small leftover and the rate is meaningless, so prefill_cached is True and
    prefill_tok_s is None.
  * decode_tok_s from a run that stopped early is a short sample. decode_complete says whether
    generation reached num_predict (or ended on the length limit).
  * prompt_chars and prompt_sha256 identify the prompt. Two measurements are comparable only when
    these match.
  * resident_bytes is None when ollama's /api/ps does not list the model. A warning on stderr
    names what it does list.
  * mem_available_bytes is the host's MemAvailable. It is read after /api/generate and /api/ps, so
    it is the free memory with the measured model resident, and with any other model still
    resident, for example the large model (which is what spec section 12 asks for). A keep_alive
    of 0, or unloading a model, changes the number. In a multi-model run, an earlier model that is
    still loaded lowers it for the later ones.

Failures a person can fix (server not running, HTTP error, bad JSON or a truncated reply, timeout,
unreadable prompt file, locked ledger) end the run with one line on stderr and exit code 1. They do not print a traceback.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
import urllib.error
import urllib.request

from orchard.ledger import Ledger, LedgerCorrupt, LedgerLocked


# A load under this many seconds means the model was already in memory, so load_s is not a cold
# start. A real cold load from disk takes many seconds; a warm hit takes milliseconds.
COLD_LOAD_THRESHOLD_S = 0.5

# Rough characters per token for English text and code. Only used to notice a cached prompt:
# if ollama counts fewer than half the tokens this estimate predicts, it did not process the prompt.
CHARS_PER_TOKEN_ESTIMATE = 4

# Where the host reports free memory. A module constant so tests can point it at a temp file.
MEMINFO_PATH = "/proc/meminfo"


class SizingError(Exception):
    """A measurement could not be completed. The message says why, in one line."""


def theoretical_gb_s(mt_s: float, channels: int = 2, bus_bytes: int = 8) -> float:
    """Peak memory bandwidth in GB/s: the DIMM rate (MT/s) times bytes per transfer times channels."""
    return mt_s * bus_bytes * channels / 1000


def ceiling_tok_s(bandwidth_gb_s: float, gb_per_token: float) -> float:
    """Upper bound on decode tokens per second: each token reads every active weight once."""
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
    except http.client.HTTPException as exc:
        # IncompleteRead (body shorter than Content-Length) and BadStatusLine are not OSError.
        raise SizingError(
            f"{url} sent a malformed or truncated reply ({type(exc).__name__})") from exc
    except ValueError as exc:  # json.JSONDecodeError and bad UTF-8 are both ValueError
        raise SizingError(f"{url} did not return valid JSON") from exc
    if not isinstance(reply, dict):
        raise SizingError(f"{url} did not return a JSON object")
    return reply


def _rate(count, duration_ns):
    """Tokens per second, or None when ollama did not report the pair."""
    # A count of 0 with a duration is a real rate of 0.0 (nothing generated in a measured time).
    if count is None or not duration_ns:
        return None
    return count / (duration_ns / 1e9)


def mem_available_bytes() -> int | None:
    """The host's MemAvailable in bytes, or None when it cannot be read.

    measure() calls this after /api/generate and /api/ps, so the model it measured is still
    resident (and so is any other model ollama keeps loaded). A keep_alive of 0 or an unload
    changes the number."""
    try:
        with open(MEMINFO_PATH, encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # the file reports kB
    except (OSError, ValueError, IndexError):
        return None
    return None


def _warn(message: str) -> None:
    print(f"sizing: warning: {message}", file=sys.stderr)


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
    # ollama lists "qwen3:latest" even when asked for "qwen3", so accept either spelling.
    wanted = {model, model[:-len(":latest")] if model.endswith(":latest") else model + ":latest"}
    resident = next((m.get("size") for m in running
                     if wanted & {m.get("name"), m.get("model")}), None)
    if not any(wanted & {m.get("name"), m.get("model")} for m in running):
        listed = ", ".join(m.get("name") or m.get("model") or "?" for m in running) or "nothing"
        _warn(f"model {model} is not in ollama's /api/ps list (lists: {listed}); "
              "resident_bytes recorded as null")

    load_ns = reply.get("load_duration")
    load_s = None if load_ns is None else load_ns / 1e9  # None: ollama did not say
    prompt_eval_count = reply.get("prompt_eval_count")
    # A prompt ollama really processed has a token count near chars/4. Far fewer tokens, or no
    # count at all, means it reused its prompt cache, and the rate would be a leftover divided by
    # a leftover.
    prefill_cached = (prompt_eval_count is None
                      or prompt_eval_count < len(prompt) / CHARS_PER_TOKEN_ESTIMATE / 2)
    decode_tokens = reply.get("eval_count")
    done_reason = reply.get("done_reason")
    decode_complete = done_reason == "length" or (
        decode_tokens is not None and decode_tokens >= num_predict)
    if not decode_complete:
        _warn(f"decode rate came from a short run ({decode_tokens} of {num_predict} tokens, "
              f"done_reason {done_reason}); the prompt may need to be longer")
    return {
        "model": model,
        "load_s": load_s,
        "load_cold": None if load_s is None else load_s >= COLD_LOAD_THRESHOLD_S,
        # Identify the prompt: two measurements are comparable only when these match.
        "prompt_chars": len(prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_tokens": prompt_eval_count,
        "prefill_cached": prefill_cached,
        "prefill_tok_s": None if prefill_cached else
                         _rate(prompt_eval_count, reply.get("prompt_eval_duration")),
        "decode_tokens": decode_tokens,
        "decode_tok_s": _rate(decode_tokens, reply.get("eval_duration")),
        "done_reason": done_reason,
        "decode_complete": decode_complete,
        "resident_bytes": resident,
        "mem_available_bytes": mem_available_bytes(),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Measure a model served by a local ollama. Note: load_s is only meaningful "
                    "for a cold first run (model not already resident); load_cold says whether "
                    "this run looked cold.")
    p.add_argument("--host", default="http://127.0.0.1:11434")
    p.add_argument("--model", action="append", required=True, help="repeat for several models")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--num-predict", type=int, default=128)
    p.add_argument("--timeout", type=float, default=900,
                   help="seconds to wait; applies to each request, so a multi-model run can take "
                        "longer in total. A cold CPU load can be slow")
    p.add_argument("--mt-s", type=float, default=3600,
                   help="configured DIMM rate; the host reports 3600 (rated 5600)")
    p.add_argument("--gb-per-token", type=float,
                   help="gigabytes read per token (about the active weights); enables the ceiling")
    p.add_argument("--note", default="", help="conditions, for example 'coder serving'")
    p.add_argument("--ledger", help="append each result to this ledger as a measured entry")
    args = p.parse_args(argv)

    host = args.host.rstrip("/")   # a trailing slash would give `//api/generate`
    ledger = None
    try:
        with open(args.prompt_file, encoding="utf-8") as f:
            prompt = f.read()
        if not prompt.strip():
            # Zero prompt tokens would give a meaningless prefill rate, so stop before measuring.
            raise SizingError(f"prompt file {args.prompt_file} is empty or only whitespace")
        bandwidth = theoretical_gb_s(args.mt_s)
        ledger = Ledger(args.ledger) if args.ledger else None
        for model in args.model:
            result = measure(host, model, prompt, args.num_predict, args.timeout)
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
