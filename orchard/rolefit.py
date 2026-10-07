# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The role-fit test: can a candidate model do the agent role in this loop?

A model takes a role in tiers.toml only after it passes this test. The success rates and capability scores
in docs/analysis were never measured, so they decide nothing. The test replays what the loop actually sent
and checks what comes back.

    python3 -m orchard.rolefit --endpoint http://127.0.0.1:20010/v1 --model Qwen/Qwen3-Coder-Next \\
        --logs /path/to/orchard-runs --out result.json

What it does, in order:

1. Canary. "What is 7 times 6?" must come back with 42 (the question the coder gets on its first start).
2. Format. It picks up to 50 recorded turns from the run logs (each log line holds the full prompt the loop
   sent and the reply it got), spread evenly across the logs, and sends each prompt, with the loop's own
   tool definitions, to the candidate at temperature 0. A reply passes when it is a well-formed tool call
   (a known tool, JSON arguments with the required keys) or non-empty text, and was not cut off at
   max_tokens. The bar is 95 percent.
3. Failures. Every recorded turn that failed in a live run (cut off at max_tokens, or empty with no tool
   call) is replayed several times at a higher temperature, and any repeat of a failure fails the part.
   The live failures came from a model that circled in its reasoning (docs/run-logs).
4. Throughput. End-to-end prefill and decode speed at 8K and 32K prompt tokens. It has no bar: the result
   file records it, because the loop's step budgets depend on it.

The endpoint must be on this machine, as with the tier config. Nothing here talks to a remote service.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from orchard import defaults
from orchard.agent import TOOL_SCHEMAS
from orchard.canary import post_json

FORMAT_BAR = 0.95
FAILURE_REPLAYS = 5
FAILURE_TEMPERATURE = 0.7       # choice: the loop decodes greedily, so a repeat at temperature 0 would only
                                # say "the same". A higher temperature asks whether the failure is a habit.
MAX_PROMPT_CHARS = 60_000       # choice: replays of very long prompts take minutes each
MAX_FAILURE_TURNS = 20
REQUEST_TIMEOUT_S = 900.0
CANARY_PROMPT = "What is 7 times 6? Reply with only the number."
TOOL_NAMES = {t["function"]["name"]: t["function"]["parameters"].get("required", []) for t in TOOL_SCHEMAS}
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


@dataclass
class Turn:
    source: str
    turn: int
    messages: list
    recorded: dict
    finish_reason: str | None
    prompt_chars: int = 0
    recorded_failure: str | None = field(default=None)


def _recorded_failure(reply: dict, finish) -> str | None:
    if finish == "length":
        return "truncated"
    if not (reply.get("tool_calls") or (reply.get("content") or "").strip()):
        return "empty"
    return None


def _read_log(path: Path, root: Path) -> list[Turn]:
    turns = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return turns
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or not isinstance(row.get("sent"), list) \
                or not isinstance(row.get("received"), dict) or not isinstance(row.get("turn"), int):
            continue
        chars = sum(len(str(m.get("content") or "")) for m in row["sent"] if isinstance(m, dict))
        try:
            source = str(path.relative_to(root))
        except ValueError:
            source = str(path)
        turns.append(Turn(source, row["turn"], row["sent"], row["received"], row.get("finish_reason"), chars,
                          _recorded_failure(row["received"], row.get("finish_reason"))))
    return turns


def _all_turns(dirs) -> list[Turn]:
    out = []
    for d in dirs:
        root = Path(d)
        if root.is_dir():
            for p in sorted(root.rglob("*.jsonl")):
                if "log" in p.parts:
                    out += _read_log(p, root)
    return out


def collect_turns(dirs, *, limit: int = 50, max_prompt_chars: int = MAX_PROMPT_CHARS) -> list[Turn]:
    """Turns that worked in a live run, spread evenly across the logs, the same selection every time."""
    good = [t for t in _all_turns(dirs) if t.recorded_failure is None and t.prompt_chars <= max_prompt_chars]
    if len(good) <= limit:
        return good
    return [good[int(i * len(good) / limit)] for i in range(limit)]


def collect_failures(dirs, *, limit: int = MAX_FAILURE_TURNS, max_prompt_chars: int = MAX_PROMPT_CHARS) -> list[Turn]:
    """Recorded turns that failed live: cut off at max_tokens, or empty with no tool call."""
    bad = [t for t in _all_turns(dirs) if t.recorded_failure and t.prompt_chars <= max_prompt_chars]
    return bad[:limit]


def classify(message: dict, finish) -> str:
    """ok, truncated, empty, unknown_tool or bad_arguments."""
    if finish == "length":
        return "truncated"
    calls = message.get("tool_calls") or []
    if calls:
        for c in calls:
            if (c.get("function") or {}).get("name") not in TOOL_NAMES:
                return "unknown_tool"
        for c in calls:
            fn = c["function"]
            try:
                args = json.loads(fn.get("arguments") or "")
            except ValueError:
                return "bad_arguments"
            if not isinstance(args, dict) or any(k not in args for k in TOOL_NAMES[fn["name"]]):
                return "bad_arguments"
        return "ok"
    return "ok" if (message.get("content") or "").strip() else "empty"


def _chat(endpoint: str, body: dict, http) -> dict:
    return http(endpoint.rstrip("/") + "/chat/completions", body, REQUEST_TIMEOUT_S)


def _one(endpoint, model, turn: Turn, temperature, http, clock, kwargs) -> dict:
    body = {"model": model, "messages": turn.messages, "tools": TOOL_SCHEMAS, "temperature": temperature,
            "max_tokens": defaults.AGENT_MAX_TOKENS, "stream": False}
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    row = {"source": turn.source, "turn": turn.turn}
    start = clock()
    try:
        data = _chat(endpoint, body, http)
        choice = data["choices"][0]
        row.update(verdict=classify(choice.get("message") or {}, choice.get("finish_reason")),
                   finish_reason=choice.get("finish_reason"),
                   completion_tokens=(data.get("usage") or {}).get("completion_tokens"), detail="")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        row.update(verdict="error", finish_reason=None, completion_tokens=None, detail=f"{type(exc).__name__}: {exc}")
    row["seconds"] = round(max(clock() - start, 0.0), 3)
    return row


def replay(endpoint, model, turns, *, temperature: float = 0.0, http=post_json, clock=time.time,
           chat_template_kwargs=None) -> list[dict]:
    return [_one(endpoint, model, t, temperature, http, clock, chat_template_kwargs) for t in turns]


def replay_failures(endpoint, model, fails, *, repeats: int = FAILURE_REPLAYS,
                    temperature: float = FAILURE_TEMPERATURE, http=post_json, clock=time.time,
                    chat_template_kwargs=None) -> list[dict]:
    out = []
    for t in fails:
        verdicts = [_one(endpoint, model, t, temperature, http, clock, chat_template_kwargs)["verdict"]
                    for _ in range(repeats)]
        out.append({"source": t.source, "turn": t.turn, "recorded_failure": t.recorded_failure,
                    "repeats": repeats, "failed_again": sum(v != "ok" for v in verdicts), "verdicts": verdicts})
    return out


def judge_format(results: list[dict], bar: float = FORMAT_BAR) -> dict:
    n = len(results)
    ok = sum(1 for r in results if r["verdict"] == "ok")
    by = {}
    for r in results:
        by[r["verdict"]] = by.get(r["verdict"], 0) + 1
    return {"passed": n > 0 and ok / n >= bar, "n": n, "ok": ok, "fraction": round(ok / n, 4) if n else None,
            "bar": bar, "by_verdict": by}


def judge_failures(replayed: list[dict]) -> dict:
    note = ("every recorded failure turn was replayed and none repeated" if replayed
            else "no recorded failure turns were found in the logs, so nothing was replayed")
    return {"passed": all(r["failed_again"] == 0 for r in replayed), "replayed": replayed, "note": note}


def verdict(parts: dict) -> dict:
    failed = [name for name in ("canary", "format", "failures") if not parts[name]["passed"]]
    return {"passed": not failed, "failed": failed}


def throughput(endpoint, model, *, sizes=(8192, 32768), out_tokens: int = 256, http=post_json,
               clock=time.time, chat_template_kwargs=None) -> dict:
    """Prefill and decode speed at each prompt size: a request that asks for one token times the prefill, and
    a request that asks for `out_tokens` times prefill plus decode, so the difference is the decode time."""
    decode, prefill, errors = {}, {}, {}
    for size in sizes:
        key = str(size)
        text = "The quick brown fox jumps over the lazy dog. " * (size * 4 // 45 + 1)

        def timed(max_tokens):
            body = {"model": model, "messages": [{"role": "user", "content": text + "\nSay hello."}],
                    "temperature": 0, "max_tokens": max_tokens, "stream": False}
            if chat_template_kwargs:
                body["chat_template_kwargs"] = chat_template_kwargs
            t0 = clock()
            data = _chat(endpoint, body, http)
            return clock() - t0, data.get("usage") or {}
        try:
            t_pre, u_pre = timed(1)
            t_full, u_full = timed(out_tokens)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            decode[key], prefill[key], errors[key] = None, None, f"{type(exc).__name__}: {exc}"
            continue
        n = u_full.get("completion_tokens") or 0
        dt = t_full - t_pre
        decode[key] = round((n - 1) / dt, 2) if n > 1 and dt > 0 else None
        pt = u_pre.get("prompt_tokens") or 0
        prefill[key] = round(pt / t_pre, 1) if pt and t_pre > 0 else None
    return {"decode_tok_s": decode, "prefill_tok_s": prefill, "out_tokens": out_tokens, "errors": errors,
            "note": "end to end through the server, one request at a time, temperature 0"}


def _canary(endpoint, model, http, kwargs) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": CANARY_PROMPT}], "temperature": 0,
            "max_tokens": 64, "stream": False}
    if kwargs:
        body["chat_template_kwargs"] = kwargs
    try:
        text = (_chat(endpoint, body, http)["choices"][0]["message"].get("content") or "")
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        return {"passed": False, "answer": None, "detail": f"{type(exc).__name__}: {exc}"}
    return {"passed": "42" in text, "answer": text.strip()[:200]}


def main(argv=None, *, http=post_json) -> int:
    p = argparse.ArgumentParser(prog="python3 -m orchard.rolefit", description=__doc__.split("\n\n")[0])
    p.add_argument("--endpoint", required=True, help="a local OpenAI-style endpoint, with /v1")
    p.add_argument("--model", required=True)
    p.add_argument("--logs", action="append", required=True, help="a directory of run logs (repeatable)")
    p.add_argument("--out", required=True, help="where to write the result JSON")
    p.add_argument("--turns", type=int, default=50)
    p.add_argument("--repeats", type=int, default=FAILURE_REPLAYS)
    p.add_argument("--sizes", default="8192,32768", help="prompt sizes in tokens for the speed measurement")
    p.add_argument("--thinking-off", action="store_true",
                   help="send enable_thinking=false, as the loop does for the Qwen3.8 coder")
    args = p.parse_args(argv)
    if urlparse(args.endpoint).hostname not in LOCAL_HOSTS:
        print("refused: the endpoint must be local (127.0.0.1 or localhost); this test sends recorded "
              "prompts and talks to nothing remote", file=sys.stderr)
        return 2
    kwargs = {"enable_thinking": False} if args.thinking_off else None
    turns = collect_turns(args.logs, limit=args.turns)
    fails = collect_failures(args.logs)
    parts = {"canary": _canary(args.endpoint, args.model, http, kwargs)}
    parts["format"] = {**judge_format(replay(args.endpoint, args.model, turns, http=http,
                                             chat_template_kwargs=kwargs)),
                       "sources": sorted({t.source for t in turns})}
    parts["failures"] = judge_failures(replay_failures(args.endpoint, args.model, fails, repeats=args.repeats,
                                                       http=http, chat_template_kwargs=kwargs))
    parts["throughput"] = throughput(args.endpoint, args.model, sizes=tuple(int(s) for s in args.sizes.split(",")),
                                     http=http, chat_template_kwargs=kwargs)
    result = {"model": args.model, "endpoint": args.endpoint,
              "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **parts, "verdict": verdict(parts)}
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    v = result["verdict"]
    print(f"role fit for {args.model}: {'PASS' if v['passed'] else 'FAIL ' + ', '.join(v['failed'])}; format "
          f"{parts['format']['ok']}/{parts['format']['n']}; failure turns replayed "
          f"{len(parts['failures']['replayed'])}; decode tok/s {parts['throughput']['decode_tok_s']}")
    return 0 if v["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
