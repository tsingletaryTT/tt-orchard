# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Read-only probes of a live OpenAI-shaped server that send more than one request, used by `tt-orchard can boot` (issues 360, 369, 371, 372).

`orchard/served_verify.py` holds the single-request checks and the judges. This module holds the probes that need a burst of requests, a long
prompt or two budgets. They follow the same rules as served_verify:

  * every probe returns `{"name", "state", "detail", ...}` with `state` one of pass, fail or not_checked, and a missing capability
    (no route, no usage field, no metrics, an undeclared limit) is `not_checked`, never a pass;
  * requests are greedy, bounded in count and in tokens, and send nothing the server could act on beyond generating text;
  * the time of a probe is bounded (`timeout` per request, and a burst waits at most `timeout` in all).

The code probe parses the reply with `ast` and never runs it: the endpoint can be remote and the reply is untrusted. Its result says the check is static.
"""
from __future__ import annotations

import ast
import builtins
import http.client
import json
import symtable
import threading
import time
import uuid

from orchard import served_verify as sv, speedwatch

MAX_CODE_CHARS = 20000            # choice: the reply to a request for one short function is a few hundred characters; more is not parsed
MAX_BURST = 8                     # choice: at most this many requests in one batching burst, whatever the can declares
BURST_TOKENS = 32
BATCHED_RATIO = 1.5               # choice: aggregate tokens per second at least 1.5 times the single-stream rate is batching
SERIAL_RATIO = 1.2                # choice: below 1.2 the requests ran one after another; between the two the probe does not decide
LONG_PROMPT_TOKENS = 2000
PREFIX_TOKENS = 1500              # choice: long enough that a cache hit is visible in the time, short enough to run in seconds on a chip
CODE_PROMPT = ("Write a Python function named is_prime that takes an integer n and returns True when n is prime and False otherwise. "
               "Reply with only the code.")
SECRET = "tangerine"
_FILLER = "The quick brown fox jumps over the lazy dog. "


def _result(name: str, state: str, detail: str, **more) -> dict:
    return {"name": name, "state": state, "detail": detail, **more}


def _chat(base: str, body: dict, timeout: float):
    """(status, parsed reply or None, raw bytes). Raises OSError, ValueError or HTTPException for a transport failure."""
    status, raw = sv._post(base.rstrip("/") + "/v1/chat/completions", body, timeout)
    return status, sv._json(raw, "reply"), raw


# ---- generated code, parsed and never run (verify V6, issue 369) ---------------------------------------------------------------

def _code_of(text: str) -> str:
    """The reply's code: the first fenced block when there is one (an unclosed fence runs to the end), else the whole reply."""
    fence = text.find("```")
    if fence < 0:
        return text
    body = text[fence + 3:]
    newline = body.find("\n")
    body = body[newline + 1:] if newline >= 0 else ""
    end = body.find("```")
    return body if end < 0 else body[:end]


def _loops_forever(node: ast.While) -> bool:
    """A `while` whose test is a true constant and whose body has no way out: no `return`, no `raise`, and no `break` that belongs to this loop
    (a `break` inside an inner loop leaves that loop only)."""
    if not (isinstance(node.test, ast.Constant) and node.test.value):
        return False

    def exits(stmts, inner_loop: bool) -> bool:
        for st in stmts:
            if isinstance(st, (ast.Return, ast.Raise)) or (isinstance(st, ast.Break) and not inner_loop):
                return True
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            loop = inner_loop or isinstance(st, (ast.For, ast.AsyncFor, ast.While))
            for field in ("body", "orelse", "finalbody"):
                if exits(getattr(st, field, None) or [], loop):
                    return True
            for h in getattr(st, "handlers", None) or []:
                if exits(h.body, loop):
                    return True
            for c in getattr(st, "cases", None) or []:
                if exits(c.body, loop):
                    return True
        return False
    return not exits(node.body, False)


def _undefined_globals(source: str, module: ast.Module) -> list[str]:
    """Names a function reads that are neither local, nor a builtin, nor defined at module level (found with `symtable`, which handles
    comprehension and nested scopes). Sorted."""
    defined = set(dir(builtins))
    for st in module.body:
        for n in ast.walk(st) if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else [st]:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(n.name)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                defined.add(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                defined.update((a.asname or a.name).split(".")[0] for a in n.names)
    missing: set[str] = set()

    def walk(table):
        for sym in table.get_symbols():
            if table.get_type() != "module" and sym.is_referenced() and sym.is_global() and sym.get_name() not in defined:
                missing.add(sym.get_name())
        for child in table.get_children():
            walk(child)
    walk(symtable.symtable(source, "<reply>", "exec"))
    return sorted(missing)


def check_generated_code(text, *, function: str = "is_prime") -> dict:
    """Parse `text` (a chat reply that should hold the function `function`) and report what is wrong with it without running it (issue 369):
    a syntax error, a missing function, no return value, an undefined name, a `while True` with no exit. The reply is read with `ast` and
    `symtable` only. The result carries `static: True`."""
    name = "code_parse"
    static = "static check: the reply was parsed and never run"
    if not isinstance(text, str) or not text.strip():
        return _result(name, "fail", "the reply holds no text to parse (" + static + ")", static=True)
    source = _code_of(text[:MAX_CODE_CHARS])
    try:
        module = ast.parse(source)
        undefined = _undefined_globals(source, module)
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError) as exc:
        where = f" at line {exc.lineno}" if isinstance(exc, SyntaxError) and exc.lineno else ""
        return _result(name, "fail", f"the reply does not parse{where}: {type(exc).__name__} ({static})", static=True)
    funcs = [n for n in ast.walk(module) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function]
    if not funcs:
        return _result(name, "fail", f"the reply parses and defines no function named {function} ({static})", static=True)
    fn = funcs[0]
    own = [n for n in ast.walk(fn) if not (n is not fn and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))]
    if not any(isinstance(n, ast.Return) and n.value is not None for n in own):
        return _result(name, "fail", f"{function} has no `return` with a value ({static})", static=True)
    if undefined:
        return _result(name, "fail", f"the reply reads names that are not defined anywhere: {', '.join(undefined[:5])} ({static})", static=True)
    if any(isinstance(n, ast.While) and _loops_forever(n) for n in ast.walk(fn)):
        return _result(name, "fail", f"{function} has a `while True` loop with no way out ({static})", static=True)
    return _result(name, "pass", f"the reply parses, defines {function}, returns a value, reads no undefined name and has no endless loop ({static})", static=True)


def probe_code(base: str, model: str, extra: dict | None = None, *, timeout: float = sv.PROBE_TIMEOUT_S) -> dict:
    """Ask for a short function at temperature 0 and check it with `check_generated_code`. `extra` is the can's `request_extra` (a thinking model
    needs thinking turned off, or a small token budget ends with empty content and finish reason length)."""
    name = "code_parse"
    body = {"model": model, "messages": [{"role": "user", "content": CODE_PROMPT}], "temperature": 0, "max_tokens": 256, **(extra or {})}
    try:
        status, doc, _ = _chat(base, body, timeout)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return _result(name, "fail", f"no answer from /v1/chat/completions: {type(exc).__name__}", static=True)
    if status in sv.ABSENT_ROUTE:
        return _result(name, "not_checked", f"the server has no chat route (it answered {status})")
    if status != 200:
        return _result(name, "fail", f"POST /v1/chat/completions answered {status}", static=True)
    choices = doc.get("choices") if doc else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        why = ("the token budget was spent before an answer (turn thinking off through request_extra)" if choice.get("finish_reason") == "length"
               else f"finish reason {choice.get('finish_reason')!r}")
        return _result(name, "fail", f"the reply content is empty: {why}", static=True)
    return check_generated_code(content)


# ---- batching (verify V11, issue 371) -----------------------------------------------------------------------------------------

def judge_batching(*, single_rate, tokens, seconds, n, max_running) -> dict:
    """Aggregate tokens per second of `n` concurrent requests against the single-stream rate. `tokens` is the sum of the reported
    completion tokens, `seconds` the wall time of the burst and `max_running` the largest `vllm:num_requests_running` seen during it (None when
    the server has no metrics). A ratio of at least BATCHED_RATIO is batching and below SERIAL_RATIO is serialised; between the two the probe does
    not decide (`not_checked`). When the server states how many requests it ran at once, `max_running` of 1 across a burst of more is serialised
    whatever the ratio says, and 2 or more is batching. Latency spread is not used."""
    name = "batching"
    if not isinstance(single_rate, (int, float)) or isinstance(single_rate, bool) or not single_rate > 0:
        return _result(name, "not_checked", "no single-stream decode rate was measured, so the burst has nothing to be compared with")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0 or not isinstance(seconds, (int, float)) or not seconds > 0:
        return _result(name, "not_checked", "the burst did not report how many tokens it generated, or took no time")
    aggregate = tokens / seconds
    ratio = aggregate / single_rate
    facts = {"requests": n, "aggregate_tok_s": round(aggregate, 2), "single_tok_s": round(single_rate, 2), "ratio": round(ratio, 2), "max_running": max_running}
    where = f"{n} concurrent requests gave {aggregate:.1f} tok/s against {single_rate:.1f} tok/s for one (ratio {ratio:.2f})"
    if isinstance(max_running, int) and max_running >= 2:
        return _result(name, "pass", f"{where}; the engine ran {max_running} requests at once", **facts)
    if isinstance(max_running, int) and max_running <= 1:
        return _result(name, "fail", f"{where}; the engine never ran more than {max_running} request at once, so the requests were served one after another", **facts)
    if ratio >= BATCHED_RATIO:
        return _result(name, "pass", f"{where}: batching (no engine metrics to confirm)", **facts)
    if ratio < SERIAL_RATIO:
        return _result(name, "fail", f"{where}: the requests were served one after another (no engine metrics to confirm)", **facts)
    return _result(name, "not_checked", f"{where}: between the serialised and the batched ratio, so no verdict", **facts)


def probe_batching(base: str, model: str, extra: dict | None, *, single_rate, max_num_seqs, timeout: float = sv.PROBE_TIMEOUT_S) -> dict:
    """Send min(max_num_seqs, MAX_BURST) identical requests at once, each for BURST_TOKENS tokens with `ignore_eos`, and judge the aggregate rate
    (`judge_batching`). Runs only when the can declares `max_num_seqs` above 1. Waits at most `timeout` for the whole burst; a request that
    does not finish in time is a fail."""
    name = "batching"
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int) or max_num_seqs <= 1:
        return _result(name, "not_checked", "the can does not declare max_num_seqs above 1, so concurrent batching is not expected")
    n = min(max_num_seqs, MAX_BURST)
    body = {"model": model, "messages": [{"role": "user", "content": "Count upward from one, one number per line."}], "temperature": 0,
            "max_tokens": BURST_TOKENS, "ignore_eos": True, **(extra or {})}
    results: list = [None] * n
    running = []
    stop = threading.Event()

    def send(i):
        try:
            status, doc, _ = _chat(base, body, timeout)
            usage = doc.get("usage") if doc else None
            results[i] = (status, usage.get("completion_tokens") if isinstance(usage, dict) else None)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            results[i] = (None, type(exc).__name__)

    def watch():
        while not stop.is_set():
            v = speedwatch.running_requests(base)
            if v is not None:
                running.append(v)
            stop.wait(0.2)
    threads = [threading.Thread(target=send, args=(i,), daemon=True) for i in range(n)]
    watcher = threading.Thread(target=watch, daemon=True)
    started = time.monotonic()
    watcher.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join(max(0.0, timeout - (time.monotonic() - started)))
    seconds = time.monotonic() - started
    stop.set()
    watcher.join(2.0)
    if any(r is None for r in results):
        return _result(name, "fail", f"the burst of {n} requests did not finish within {timeout:g} s")
    bad = [r for r in results if r[0] != 200]
    if bad:
        return _result(name, "fail", f"{len(bad)} of {n} concurrent requests failed ({bad[0][0] if bad[0][0] is not None else bad[0][1]})")
    counts = [r[1] for r in results]
    if not all(isinstance(c, int) and not isinstance(c, bool) and c > 0 for c in counts):
        return _result(name, "not_checked", "the replies did not report usage.completion_tokens")
    return judge_batching(single_rate=single_rate, tokens=sum(counts), seconds=seconds, n=n, max_running=max(running) if running else None)


# ---- a long prompt (verify V12, issue 371) -------------------------------------------------------------------------------------

def _prompt_of(reps: int, tag: str = "") -> str:
    return f"{tag}The secret word is {SECRET}. Remember it.\n" + _FILLER * reps + "\nWhat is the secret word? Answer with the word only."


def _tokens_of(doc) -> int | None:
    usage = doc.get("usage") if isinstance(doc, dict) else None
    n = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else None


def _sized_prompt(base, model, extra, target, timeout, tag="") -> tuple[str, int] | str:
    """A prompt whose measured token count (usage.prompt_tokens, never a word count) is near `target`: one small request gives the tokens per
    repetition, and the prompt is scaled from it. Returns (prompt, measured tokens of the calibration scaled) or a reason string."""
    probe_reps = 40
    status, doc, _ = _chat(base, {"model": model, "messages": [{"role": "user", "content": _prompt_of(probe_reps, tag)}], "temperature": 0,
                                  "max_tokens": 1, **(extra or {})}, timeout)
    n = _tokens_of(doc) if status == 200 else None
    if n is None:
        return f"the calibration request answered {status} and reported no usage.prompt_tokens"
    per_rep = max(n / probe_reps, 1.0)
    return _prompt_of(max(1, int(target / per_rep)), tag), int(target)


def probe_long_prefill(base: str, model: str, extra: dict | None = None, *, max_model_len=None, target_tokens: int = LONG_PROMPT_TOKENS,
                       timeout: float = sv.PROBE_TIMEOUT_S) -> dict:
    """A prompt of about `target_tokens` (measured from `usage`; a word count is never used) with a secret word at its start. The probe reports the
    prompt tokens the server counted, the seconds and whether the answer was reached (a reply with text) and whether it names the secret word.
    A reply that arrives passes; the recall is reported and does not decide. A declared `max_model_len` too small for the prompt is `not_checked`."""
    name = "long_prefill"
    if isinstance(max_model_len, int) and not isinstance(max_model_len, bool) and max_model_len < target_tokens + 128:
        return _result(name, "not_checked", f"the declared max_model_len {max_model_len} is too small for a prompt of {target_tokens} tokens")
    try:
        sized = _sized_prompt(base, model, extra, target_tokens, timeout)
        if isinstance(sized, str):
            return _result(name, "not_checked", sized)
        t0 = time.monotonic()
        status, doc, _ = _chat(base, {"model": model, "messages": [{"role": "user", "content": sized[0]}], "temperature": 0, "max_tokens": 32,
                                      **(extra or {})}, timeout)
        seconds = round(time.monotonic() - t0, 3)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return _result(name, "fail", f"the long prompt got no answer within {timeout:g} s per request: {type(exc).__name__}")
    if status != 200:
        return _result(name, "fail", f"the long prompt answered {status}", seconds=seconds)
    choices = doc.get("choices") if doc else None
    message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) and isinstance(message.get("content"), str) else ""
    prompt_tokens = _tokens_of(doc)
    facts = {"prompt_tokens": prompt_tokens, "seconds": seconds, "answered": bool(content.strip()), "recalled": SECRET in content.lower()}
    if prompt_tokens is None:
        return _result(name, "not_checked", "the reply reported no usage.prompt_tokens, so the prompt's length is not known", **facts)
    if not content.strip():
        return _result(name, "fail", f"a prompt of {prompt_tokens} tokens was accepted and the reply is empty after {seconds:g} s", **facts)
    return _result(name, "pass", f"a prompt of {prompt_tokens} tokens was answered in {seconds:g} s ({'named' if facts['recalled'] else 'did not name'} the secret word)", **facts)


# ---- a reasoning model (verify V13 to V17, issue 372) --------------------------------------------------------------------------

REASONING_QUESTION = "What is 17 times 24? Think it through, then give the number."
BUDGETS = (48, 1536)               # choice: a budget too small for thinking and one large enough for a short question


def _thinking_on(extra: dict | None) -> dict:
    """`extra` without any switch that turns thinking off."""
    out = {k: v for k, v in (extra or {}).items() if k != "chat_template_kwargs"}
    kwargs = {k: v for k, v in ((extra or {}).get("chat_template_kwargs") or {}).items() if k != "enable_thinking"} if isinstance((extra or {}).get("chat_template_kwargs"), dict) else {}
    if kwargs:
        out["chat_template_kwargs"] = kwargs
    return out


def probe_reasoning(base: str, model: str, extra: dict | None = None, *, log_text=None, timeout: float = sv.PROBE_TIMEOUT_S) -> list[dict]:
    """For a can that declares a reasoning model. One short question at temperature 0 with thinking on, at two budgets, and once with
    `enable_thinking` false. Results: `reasoning_field` (populated, leaking, dropped or none, from `classify_reasoning`), `reasoning_parser` (the
    engine argument the log prints), `thinking_switch` (turning thinking off must cut the completion tokens) and `budget_sweep` (the answer
    arrives at the large budget; the small budget's result is recorded). Each request is a single chat call; the sweep is two."""
    on = _thinking_on(extra)
    off = {**on, "chat_template_kwargs": {**(on.get("chat_template_kwargs") or {}), "enable_thinking": False}}
    asks = {}
    try:
        for key, tokens, more in (("small", BUDGETS[0], on), ("large", BUDGETS[1], on), ("off", BUDGETS[1], off)):
            status, doc, _ = _chat(base, {"model": model, "messages": [{"role": "user", "content": REASONING_QUESTION}], "temperature": 0,
                                          "max_tokens": tokens, **more}, timeout)
            asks[key] = (status, doc)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return [_result("reasoning_field", "fail", f"no answer from /v1/chat/completions: {type(exc).__name__}")]
    if any(a[0] in sv.ABSENT_ROUTE for a in asks.values()):
        return [_result("reasoning_field", "not_checked", "the server has no chat route")]
    if any(a[0] != 200 or a[1] is None for a in asks.values()):
        return [_result("reasoning_field", "fail", "a reasoning probe request did not get a 200 with a JSON reply")]
    out = []
    large = sv.classify_reasoning(asks["large"][1])
    split = large.get("split")
    kind = large["kind"]
    out.append(_result("reasoning_field", "pass" if kind == "populated" else "fail" if kind in ("leaking", "dropped", "none", "unusable") else "not_checked",
                       f"{kind}: {large['detail']}", kind=kind, split=split))
    parser = sv.reasoning_parser_in_log(log_text)
    if not isinstance(log_text, str):
        out.append(_result("reasoning_parser", "not_checked", "the server log could not be read"))
    elif parser is None and "reasoning" not in log_text.lower():
        out.append(_result("reasoning_parser", "not_checked", "the server log prints no engine arguments with a reasoning parser"))
    elif parser is None:
        out.append(_result("reasoning_parser", "fail", "the engine arguments in the server log name no reasoning parser"))
    else:
        out.append(_result("reasoning_parser", "pass", f"the engine runs the reasoning parser {parser!r}", parser=parser))
    on_split, off_split = sv.reasoning_split(asks["large"][1]), sv.reasoning_split(asks["off"][1])
    if on_split is None or off_split is None:
        out.append(_result("thinking_switch", "not_checked", "a reply reported no usage.completion_tokens"))
    elif off_split["completion_tokens"] < on_split["completion_tokens"]:
        out.append(_result("thinking_switch", "pass", f"thinking off generated {off_split['completion_tokens']} tokens against {on_split['completion_tokens']} with it on",
                           on_tokens=on_split["completion_tokens"], off_tokens=off_split["completion_tokens"]))
    else:
        out.append(_result("thinking_switch", "fail", f"thinking off generated {off_split['completion_tokens']} tokens and thinking on {on_split['completion_tokens']}: the switch did not cut the tokens",
                           on_tokens=on_split["completion_tokens"], off_tokens=off_split["completion_tokens"]))
    small_doc, large_doc = asks["small"][1], asks["large"][1]

    def answered(doc):
        c = doc.get("choices")
        m = c[0].get("message") if isinstance(c, list) and c and isinstance(c[0], dict) else None
        return isinstance(m, dict) and isinstance(m.get("content"), str) and bool(m["content"].strip()), (c[0].get("finish_reason") if isinstance(c, list) and c and isinstance(c[0], dict) else None)
    (small_ok, small_finish), (large_ok, large_finish) = answered(small_doc), answered(large_doc)
    sweep = {"small_budget": BUDGETS[0], "small_answered": small_ok, "small_finish": small_finish,
             "large_budget": BUDGETS[1], "large_answered": large_ok, "large_finish": large_finish}
    if large_ok:
        out.append(_result("budget_sweep", "pass", f"the answer arrives at {BUDGETS[1]} tokens; at {BUDGETS[0]} it {'also arrives' if small_ok else 'does not'}", **sweep))
    else:
        out.append(_result("budget_sweep", "fail", f"no answer even at {BUDGETS[1]} tokens (finish reason {large_finish!r})", **sweep))
    return out


# ---- a repeated prefix (issues 360, 361) ---------------------------------------------------------------------------------------

def _timed_prompt(base, model, extra, prompt, timeout) -> dict | None:
    t0 = time.monotonic()
    status, doc, _ = _chat(base, {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1, **(extra or {})}, timeout)
    seconds = time.monotonic() - t0
    if status != 200 or doc is None:
        return None
    usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    cached = details.get("cached_tokens")
    return {"seconds": round(seconds, 4), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": cached if isinstance(cached, int) and not isinstance(cached, bool) else None}


def probe_prefix_reuse(base: str, model: str, extra: dict | None = None, *, declared: bool = False, tokens: int = PREFIX_TOKENS,
                       timeout: float = sv.PROBE_TIMEOUT_S) -> dict:
    """Three one-token requests with a long prompt: A (cold), a different prompt B of the same size (the control, so warm kernels are not
    mistaken for a cache), and A again. `served_verify.check_prefix_reuse` judges them. Every prompt carries its own random tag at its start only
    where it must differ, so A and its repeat are byte for byte equal."""
    name = "prefix_reuse"
    try:
        sized = _sized_prompt(base, model, extra, tokens, timeout)
        if isinstance(sized, str):
            return _result(name, "not_checked", sized)
        reps = max(1, sized[0].count(_FILLER))
        a = _prompt_of(reps, f"Document {uuid.uuid4()}.\n")
        b = _prompt_of(reps, f"Document {uuid.uuid4()}.\n")
        first, control, repeat = (_timed_prompt(base, model, extra, p, timeout) for p in (a, b, a))
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return _result(name, "fail", f"no answer while timing a repeated prefix: {type(exc).__name__}")
    if first is None or control is None or repeat is None:
        return _result(name, "fail", "a prefix request did not get a 200 with a JSON reply")
    return sv.check_prefix_reuse(first, repeat, control, declared=declared)
