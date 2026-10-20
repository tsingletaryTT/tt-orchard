# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Score a served model's tool calling on the Berkeley Function Calling Leaderboard `simple` and `multiple` sets.

lm-evaluation-harness has no tool-call task, and a package that is offered for coding agents needs one. Each item is one question, a list
of function definitions and one expected call. The scorer sends the question with the functions as OpenAI-style `tools`, reads the tool
call the server returns, and grades it against the leaderboard's ground truth. Nothing is executed.

Grading follows the leaderboard's rules as far as they apply to one expected call: the function name must match, every parameter the model
sends must be one the function declares and the truth lists, every required parameter must be present, and each value must equal one of the
values the truth accepts for it. A truth value of `""` means the parameter may be left out. Numbers compare by value (an integer equals the
same float, a bool is never a number). Strings compare without case, spaces and the punctuation `, . / - _ * ^`, and with `'` read as `"`.
Lists compare element by element, and objects by key with the accepted values listed per key.

`python3 -m orchard.toolcall_bench --url URL --model NAME --questions Q.jsonl --answers A.jsonl --out FILE` runs a set against a server.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

TYPE_WORDS = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}
STRING_DROP = re.compile(r"[ ,./\-_*^]")


# ---- loading ----------------------------------------------------------------------------------------------------------------------

def _rows(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_items(questions_path, answers_path) -> list[dict]:
    """Pair each question with its answer by position and check that the ids agree. The leaderboard's files are JSON lines."""
    questions, answers = _rows(questions_path), _rows(answers_path)
    if len(questions) != len(answers):
        raise ValueError(f"{len(questions)} questions but {len(answers)} answers")
    out = []
    for q, a in zip(questions, answers):
        if q["id"] != a["id"]:
            raise ValueError(f"the question id {q['id']} does not match the answer id {a['id']}")
        if len(q["question"]) != 1:
            raise ValueError(f"{q['id']}: {len(q['question'])} turns; this scorer handles one turn per item (not multi-turn)")
        truth = a["ground_truth"]
        if len(truth) != 1:
            raise ValueError(f"{q['id']}: {len(truth)} expected calls; this scorer handles one expected call per item (simple and multiple, not parallel)")
        out.append({"id": q["id"], "messages": q["question"][0], "functions": q["function"], "truth": truth[0]})
    return out


def order_key(seed: int, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def choose(items: list[dict], *, limit=None, seed: int = 0) -> list[dict]:
    """A fixed subset: items ordered by a hash of the seed and the id, so the choice does not depend on the order of the file."""
    ordered = sorted(items, key=lambda i: order_key(seed, i["id"]))
    return ordered if limit is None else ordered[:limit]


# ---- the request ------------------------------------------------------------------------------------------------------------------

def _schema(node):
    """Copy a JSON-schema node with the leaderboard's type words replaced. `properties` holds names, so its keys are never read as types."""
    if not isinstance(node, dict):
        return copy.deepcopy(node)
    out = {}
    for key, value in node.items():
        if key == "type" and isinstance(value, str):
            out[key] = TYPE_WORDS.get(value, value)
        elif key == "properties" and isinstance(value, dict):
            out[key] = {name: _schema(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = _schema(value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def openai_tools(functions: list[dict]):
    """OpenAI `tools` for the leaderboard's function definitions, and a map from the name sent back to the original name.
    The API allows letters, digits, `_` and `-` in a name, and the leaderboard's names have dots (`math.factorial`)."""
    tools, names = [], {}
    for fn in functions:
        sent = re.sub(r"[^A-Za-z0-9_-]", "_", fn["name"])
        if sent in names:
            raise ValueError(f"two functions are both called {sent} after the dots are replaced")
        names[sent] = fn["name"]
        tools.append({"type": "function", "function": {"name": sent, "description": fn.get("description", ""),
                                                       "parameters": _schema(fn.get("parameters", {"type": "dict", "properties": {}}))}})
    return tools, names


def request_body(model: str, item: dict, *, max_tokens: int, thinking=None) -> dict:
    tools, _ = openai_tools(item["functions"])
    body = {"model": model, "messages": item["messages"], "tools": tools, "tool_choice": "auto", "temperature": 0,
            "max_tokens": max_tokens, "stream": False}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    return body


# ---- reading the reply ------------------------------------------------------------------------------------------------------------

def parse_calls(message: dict, names: dict) -> list[dict]:
    """The calls in a reply, each as {name, arguments}. Arguments the server sent as text that is not a JSON object are None."""
    calls = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw = fn.get("arguments")
        args = None
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            args = parsed if isinstance(parsed, dict) else None
        calls.append({"name": names.get(fn.get("name"), fn.get("name")), "arguments": args})
    return calls


# ---- grading ----------------------------------------------------------------------------------------------------------------------

def _norm(text: str) -> str:
    return STRING_DROP.sub("", text).lower().replace("'", '"')


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _match(value, accepted) -> bool:
    if _is_number(value) or _is_number(accepted):
        return _is_number(value) and _is_number(accepted) and value == accepted
    if isinstance(value, str) or isinstance(accepted, str):
        return isinstance(value, str) and isinstance(accepted, str) and _norm(value) == _norm(accepted)
    if isinstance(value, list) or isinstance(accepted, list):
        return (isinstance(value, list) and isinstance(accepted, list) and len(value) == len(accepted)
                and all(_match(v, a) for v, a in zip(value, accepted)))
    if isinstance(value, dict) or isinstance(accepted, dict):
        if not (isinstance(value, dict) and isinstance(accepted, dict)) or not set(value) <= set(accepted):
            return False
        if any(k not in value and "" not in accepted[k] for k in accepted):
            return False
        return all(_allowed(value[k], accepted[k]) for k in value)
    return value == accepted


def _allowed(value, accepted: list) -> bool:
    return any(_match(value, a) for a in accepted)


def grade(item: dict, calls: list[dict]) -> dict:
    """{"passed": bool, "reason": str}. The reason names the first rule that failed, with the parameter where there is one."""
    if not calls:
        return {"passed": False, "reason": "no_call"}
    if len(calls) > 1:
        return {"passed": False, "reason": f"too_many_calls: {len(calls)}"}
    (want, params), = item["truth"].items()
    call = calls[0]
    if call["name"] != want:
        return {"passed": False, "reason": f"wrong_name: got {call['name']}, expected {want}"}
    args = call["arguments"]
    if args is None:
        return {"passed": False, "reason": "bad_arguments"}
    spec = next(f for f in item["functions"] if f["name"] == want).get("parameters", {})
    declared, required = spec.get("properties", {}), spec.get("required", [])
    for key in args:
        if key not in declared or key not in params:
            return {"passed": False, "reason": f"extra_param: {key}"}
    for key in list(required) + [k for k in params if k not in required]:
        if key not in args and (key in required or "" not in params[key]):
            return {"passed": False, "reason": f"missing_required: {key}"}
    for key, value in args.items():
        if not _allowed(value, params[key]):
            return {"passed": False, "reason": f"bad_value: {key}"}
    return {"passed": True, "reason": "ok"}


def stderr(*, passed: int, n: int):
    """The binomial standard error of passed / n."""
    if n <= 0:
        return None
    p = passed / n
    return math.sqrt(p * (1 - p) / n)


# ---- running against a server -----------------------------------------------------------------------------------------------------

def _post(url: str, body: dict, timeout: float) -> dict:
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    try:
        conn.request("POST", parts.path or "/", json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {raw[:200]!r}")
        return json.loads(raw)
    finally:
        conn.close()


def run(url: str, model: str, items: list[dict], *, concurrency: int = 1, max_tokens: int = 512, thinking=None, seed: int = 0,
        dataset=None, max_consecutive_errors: int = 3, timeout: float = 900.0) -> dict:
    """Send each item, grade the reply, and return the score with per-item records in item order. After `max_consecutive_errors` failed
    requests in a row no further item is sent and `stopped` says why; items already in flight finish."""
    lock = threading.Lock()
    state = {"streak": 0, "stopped": None}

    def one(item):
        with lock:
            if state["stopped"]:
                return None
        _, names = openai_tools(item["functions"])
        sent = time.monotonic()
        try:
            reply = _post(url, request_body(model, item, max_tokens=max_tokens, thinking=thinking), timeout)
            choice = (reply.get("choices") or [{}])[0]
            calls = parse_calls(choice.get("message") or {}, names)
            graded = grade(item, calls)
            usage = reply.get("usage") or {}
            rec = {"id": item["id"], **graded, "calls": calls, "finish_reason": choice.get("finish_reason"),
                   "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                   "latency_s": round(time.monotonic() - sent, 3)}
            failed = False
        except Exception as exc:                                                        # any failure of the request is an item failure
            rec = {"id": item["id"], "passed": False, "reason": f"error: {exc}", "error": True, "calls": [], "finish_reason": None,
                   "prompt_tokens": None, "completion_tokens": None, "latency_s": round(time.monotonic() - sent, 3)}
            failed = True
        with lock:
            state["streak"] = state["streak"] + 1 if failed else 0
            if state["streak"] >= max_consecutive_errors and not state["stopped"]:
                state["stopped"] = f"{max_consecutive_errors} errors in a row; the last was: {rec['reason'][:160]}"
        return rec

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        records = [r for r in pool.map(one, items) if r is not None]
    n = len(records)
    passed = sum(1 for r in records if r["passed"])
    tokens = [r["completion_tokens"] for r in records if isinstance(r.get("completion_tokens"), int)]
    return {"n": n, "passed": passed, "failed": n - passed, "score": (passed / n) if n else None, "stderr": stderr(passed=passed, n=n),
            "errors": sum(1 for r in records if r.get("error")), "cut_off": sum(1 for r in records if r.get("finish_reason") == "length"),
            "completion_tokens_total": sum(tokens), "stopped": state["stopped"],
            "item_ids_sha256": hashlib.sha256("\n".join(r["id"] for r in records).encode("utf-8")).hexdigest(),
            "settings": {"max_tokens": max_tokens, "thinking": thinking, "temperature": 0, "concurrency": concurrency, "seed": seed, "dataset": dataset},
            "items": records}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True, help="the chat completions URL, for example http://127.0.0.1:8100/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--thinking", choices=("on", "off", "default"), default="default")
    ap.add_argument("--dataset", help="the dataset id and revision, recorded in the result")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    chosen = choose(load_items(a.questions, a.answers), limit=a.limit, seed=a.seed)
    thinking = {"on": True, "off": False, "default": None}[a.thinking]
    out = run(a.url, a.model, chosen, concurrency=a.concurrency, max_tokens=a.max_tokens, thinking=thinking, seed=a.seed, dataset=a.dataset)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    se = out["stderr"]
    print(f"{out['passed']}/{out['n']} passed" + (f" (score {out['score']:.3f}, stderr {se:.3f})" if se is not None else "")
          + (f"; stopped: {out['stopped']}" if out["stopped"] else ""))
    return 0 if not out["stopped"] else 3


if __name__ == "__main__":
    sys.exit(main())
