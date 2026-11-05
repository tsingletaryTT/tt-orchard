# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Read-only probes of a live OpenAI-shaped server that send more than one request, used by `tt-orchard can boot` (issues 360, 369, 371, 372).

`orchard/served_verify.py` holds the single-request checks and the judges. This module holds the probes that need a burst of requests, a long
prompt or two budgets. They follow the same rules as served_verify:

  * every probe returns `{"name", "state", "detail", ...}` with `state` one of pass, fail or not_checked, and a missing capability
    (no route, no usage field, no metrics, an undeclared limit) is `not_checked`, never a pass;
  * requests are greedy, bounded in count and in tokens, and send nothing the server could act on beyond generating text. One probe,
    `probe_prefix_shared_suffix`, also posts `/reset_prefix_cache` (it empties the server's prefix cache so a cold reference exists); it does so
    only when the caller passes `allow_reset=True`, which the boot check does because it started that server itself;
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
import urllib.request
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
CODE_NUMBER = "4417"
SUFFIX_TOKENS = 16                # choice: a one-word or one-number answer; short, so the comparison is of a few tokens and a server that is merely not bit-exact does not fail it
REPLY_CHARS = 80                  # choice: a reply kept in a result is cut to this many characters
QUESTION_WORD = "What is the secret word? Answer with the word only."
QUESTION_CODE = "What is the code number? Answer with the number only."
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


def _own_nodes(fn):
    """The nodes of a function's own lexical body: nested functions, lambdas and classes are not entered (their `return` and loops are theirs)."""
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


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
    own = list(_own_nodes(fn))
    if not any(isinstance(n, ast.Return) and n.value is not None for n in own):
        return _result(name, "fail", f"{function} has no `return` with a value ({static})", static=True)
    if undefined:
        return _result(name, "fail", f"the reply reads names that are not defined anywhere: {', '.join(undefined[:5])} ({static})", static=True)
    if any(isinstance(n, ast.While) and _loops_forever(n) for n in own):
        return _result(name, "fail", f"{function} has a `while True` loop with no way out ({static})", static=True)
    return _result(name, "pass", f"the reply parses, defines {function}, returns a value, reads no undefined name and has no endless loop ({static})", static=True)


def probe_code(base: str, model: str, extra: dict | None = None, *, timeout: float = sv.PROBE_TIMEOUT_S) -> dict:
    """Ask for a short function at temperature 0 and check it with `check_generated_code`. `extra` is the can's `request_extra` (a thinking model
    needs thinking turned off, or a small token budget ends with empty content and finish reason length)."""
    name = "code_parse"
    body = {**(extra or {}), "model": model, "messages": [{"role": "user", "content": CODE_PROMPT}], "temperature": 0, "max_tokens": 256}
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

def _metrics_text(url: str, timeout: float) -> str:
    """GET `url` and return the text; a body over served_verify.MAX_METRICS_BYTES raises OSError (so the gauge is unavailable)."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        raw = resp.read(sv.MAX_METRICS_BYTES + 1)
    if len(raw) > sv.MAX_METRICS_BYTES:
        raise OSError("the metrics page is larger than the bound")
    return raw.decode("utf-8", errors="replace")


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
    seen = "the sampled running count never rose above 1" if isinstance(max_running, int) else "no engine metrics"      # sampling every 0.2 s can miss a short burst
    if ratio >= BATCHED_RATIO:
        return _result(name, "pass", f"{where}: batching ({seen})", **facts)
    if ratio < SERIAL_RATIO:
        return _result(name, "fail", f"{where}: the requests were served one after another ({seen})", **facts)
    return _result(name, "not_checked", f"{where}: between the serialised and the batched ratio, so no verdict", **facts)


def probe_batching(base: str, model: str, extra: dict | None, *, single_rate, max_num_seqs, timeout: float = sv.PROBE_TIMEOUT_S) -> dict:
    """Send min(max_num_seqs, MAX_BURST) identical requests at once, each for BURST_TOKENS tokens with `ignore_eos`, and judge the aggregate rate
    (`judge_batching`). Runs only when the can declares `max_num_seqs` above 1. Waits at most `timeout` for the whole burst; a request that
    does not finish in time is a fail."""
    name = "batching"
    if isinstance(max_num_seqs, bool) or not isinstance(max_num_seqs, int) or max_num_seqs <= 1:
        return _result(name, "not_checked", "the can does not declare max_num_seqs above 1, so concurrent batching is not expected")
    n = min(max_num_seqs, MAX_BURST)
    body = {**(extra or {}), "model": model, "messages": [{"role": "user", "content": "Count upward from one, one number per line."}],
            "temperature": 0, "max_tokens": BURST_TOKENS, "ignore_eos": True}
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
            v = speedwatch.running_requests(base, get=_metrics_text)
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
    status, doc, _ = _chat(base, {**(extra or {}), "model": model, "messages": [{"role": "user", "content": _prompt_of(probe_reps, tag)}],
                                  "temperature": 0, "max_tokens": 1}, timeout)
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
        status, doc, _ = _chat(base, {**(extra or {}), "model": model, "messages": [{"role": "user", "content": sized[0]}], "temperature": 0,
                                      "max_tokens": 32}, timeout)
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
            status, doc, _ = _chat(base, {**more, "model": model, "messages": [{"role": "user", "content": REASONING_QUESTION}], "temperature": 0,
                                          "max_tokens": tokens}, timeout)
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
    elif parser is None and not sv.reasoning_parser_printed(log_text):
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
        finish = c[0].get("finish_reason") if isinstance(c, list) and c and isinstance(c[0], dict) else None
        return isinstance(m, dict) and isinstance(m.get("content"), str) and bool(m["content"].strip()), finish
    (small_ok, small_finish), (large_ok, large_finish) = answered(small_doc), answered(large_doc)
    sweep = {"small_budget": BUDGETS[0], "small_answered": small_ok, "small_finish": small_finish,
             "large_budget": BUDGETS[1], "large_answered": large_ok, "large_finish": large_finish}
    if large_ok and large_finish == "stop":
        out.append(_result("budget_sweep", "pass", f"the answer arrives at {BUDGETS[1]} tokens; at {BUDGETS[0]} it {'also arrives' if small_ok else 'does not'}", **sweep))
    else:
        out.append(_result("budget_sweep", "fail", f"no answer even at {BUDGETS[1]} tokens (finish reason {large_finish!r})", **sweep))
    return out


# ---- a repeated prefix (issues 360, 361) ---------------------------------------------------------------------------------------

def _timed_prompt(base, model, extra, prompt, timeout) -> dict | None:
    t0 = time.monotonic()
    status, doc, _ = _chat(base, {**(extra or {}), "model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1}, timeout)
    seconds = time.monotonic() - t0
    if status != 200 or doc is None:
        return None
    usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    cached = details.get("cached_tokens")
    return {"seconds": round(seconds, 4), "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": cached if isinstance(cached, int) and not isinstance(cached, bool) else None}


def probe_prefix_reuse(base: str, model: str, extra: dict | None = None, *, declared: bool = False, tokens: int = PREFIX_TOKENS,
                       max_model_len=None, timeout: float = sv.PROBE_TIMEOUT_S) -> dict:
    """Three one-token requests with a long prompt: A (cold), a different prompt B of the same size (the control, so warm kernels are not
    mistaken for a cache), and A again. `served_verify.check_prefix_reuse` judges them. Every prompt carries its own random tag at its start only
    where it must differ, so A and its repeat are byte for byte equal."""
    name = "prefix_reuse"
    if isinstance(max_model_len, int) and not isinstance(max_model_len, bool) and max_model_len < tokens + 128:
        return _result(name, "not_checked", f"the declared max_model_len {max_model_len} is too small for a prefix of {tokens} tokens")
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


# ---- a shared prefix with a new suffix, warm against cold (issue 83; tenstorrent/skills PR 42) -----------------------------------

def _shared_prefix(reps: int, tag: str) -> str:
    return f"{tag}The secret word is {SECRET}. The code number is {CODE_NUMBER}. Remember both.\n" + _FILLER * reps


def _clip(text) -> str:
    text = text if isinstance(text, str) else ""
    return text if len(text) <= REPLY_CHARS else text[:REPLY_CHARS - 3] + "..."


def _count(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _reset_prefix_cache(base: str, timeout: float) -> bool:
    """POST /reset_prefix_cache. True for a 200. A server that was not started with VLLM_SERVER_DEV_MODE=1 has no such route (a 404)."""
    try:
        status, _ = sv._post(base.rstrip("/") + "/reset_prefix_cache", {}, timeout)
    except (OSError, ValueError, http.client.HTTPException):
        return False
    return status == 200


def _asked(base, model, extra, prompt, timeout) -> tuple[str, int | None] | None:
    """(the reply text, usage.prompt_tokens_details.cached_tokens or None) of one greedy request, or None when the server did not answer 200 with a reply."""
    status, doc, _ = _chat(base, {**(extra or {}), "model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
                                  "max_tokens": SUFFIX_TOKENS}, timeout)
    choices = doc.get("choices") if isinstance(doc, dict) else None
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if status != 200 or not isinstance(content, str):
        return None
    usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else {}
    details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    return content, _count(details.get("cached_tokens"))


def judge_prefix_shared_suffix(other, warm, cold, *, cached_tokens, declared: bool) -> dict:
    """`other` is the reply to question 1 on the prefix; `warm` and `cold` are replies to question 2 on the same prefix, `warm` served after
    the prefix was cached by the first request and `cold` after a cache reset. The control: `other` stands in for a contaminated `warm` (an
    answer cached for a different question) and must be a different reply from `cold`; when it is not, the probe cannot tell a stale answer
    from a right one and does not decide. `cached_tokens` is what the warm request's usage reported (None when it reports none)."""
    name = "prefix_shared_suffix"
    if not declared:
        return _result(name, "not_checked", "the can does not declare prefix caching", advisory=True)
    if not all(isinstance(t, str) and t.strip() for t in (other, warm, cold)):
        return _result(name, "not_checked", "a reply was empty, so the replies cannot be compared", advisory=True)
    hits = _count(cached_tokens)
    hit = None if hits is None else hits > 0
    control = {"stand_in": "the reply to the other question", "missed": other.strip() != cold.strip()}
    facts = {"other_reply": _clip(other), "warm_reply": _clip(warm), "cold_reply": _clip(cold), "cached_tokens": hits, "hit_observed": hit,
             "control": control, "advisory": True}
    seen = ("a cache hit was observed (usage.prompt_tokens_details.cached_tokens " + str(hits) + ")" if hit else
            "a cache hit was not observed (the server reports no cached_tokens)")
    facts["label"] = ("Compares the text of two greedy replies of at most " + str(SUFFIX_TOKENS) + " tokens to one prompt: one served from a prefix "
                      "another request had cached, one after /reset_prefix_cache. It does not compare logits, does not send concurrent requests, "
                      "and says nothing about other prompts. " + seen[0].upper() + seen[1:] + ".")
    if not control["missed"]:
        return _result(name, "not_checked", "the control did not miss: the reply to the other question equals the cold reply, so the probe cannot "
                       "tell a stale answer from a right one", **facts)
    if hit is False:
        return _result(name, "not_checked", "no cache hit was observed: the warm request reported 0 cached tokens, so it did not test a hit", **facts)
    if warm.strip() != cold.strip():
        return _result(name, "fail", f"the warm reply differs from the cold reply to the same prompt: {_clip(warm)!r} against {_clip(cold)!r}", **facts)
    return _result(name, "pass", "the warm reply to a prompt on a cached prefix equals the cold reply to it; the control (the reply to the other "
                   "question) differs, so the probe can tell a stale answer", **facts)


def probe_prefix_shared_suffix(base: str, model: str, extra: dict | None = None, *, declared: bool = False, allow_reset: bool = False,
                               tokens: int = PREFIX_TOKENS, max_model_len=None, timeout: float = sv.PROBE_TIMEOUT_S) -> dict:
    """Reset the cache; request A (a fresh prefix and question 1) fills it; request B (the same prefix and question 2) hits the prefix with a
    new suffix; reset again; request B once more is the cold reference. `judge_prefix_shared_suffix` compares them. Advisory: the result carries
    `advisory: True`, and the boot check lists it without letting it fail the boot. Nothing is sent unless the can declares prefix caching, the caller
    allows the reset and the server has a reset route."""
    name = "prefix_shared_suffix"
    if not declared:
        return _result(name, "not_checked", "the can does not declare prefix caching", advisory=True)
    if not allow_reset:
        return _result(name, "not_checked", "the probe needs /reset_prefix_cache for a cold reference and the caller did not allow a reset", advisory=True)
    if isinstance(max_model_len, int) and not isinstance(max_model_len, bool) and max_model_len < tokens + 128:
        return _result(name, "not_checked", f"the declared max_model_len {max_model_len} is too small for a prefix of {tokens} tokens", advisory=True)
    try:
        if not _reset_prefix_cache(base, timeout):
            return _result(name, "not_checked", "the server has no /reset_prefix_cache, so there is no cold reference (start vLLM with "
                           "VLLM_SERVER_DEV_MODE=1 to enable it)", advisory=True)
        sized = _sized_prompt(base, model, extra, tokens, timeout)
        if isinstance(sized, str):
            return _result(name, "not_checked", sized, advisory=True)
        prefix = _shared_prefix(max(1, sized[0].count(_FILLER)), f"Document {uuid.uuid4()}.\n")
        first = _asked(base, model, extra, prefix + "\n\n" + QUESTION_WORD, timeout)
        warm = _asked(base, model, extra, prefix + "\n\n" + QUESTION_CODE, timeout)
        if not _reset_prefix_cache(base, timeout):
            return _result(name, "not_checked", "the second /reset_prefix_cache failed, so there is no cold reference", advisory=True)
        cold = _asked(base, model, extra, prefix + "\n\n" + QUESTION_CODE, timeout)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        return _result(name, "not_checked", f"no answer while comparing a warm and a cold reply: {type(exc).__name__}", advisory=True)
    if first is None or warm is None or cold is None:
        return _result(name, "not_checked", "a request did not get a 200 with a reply, so the replies cannot be compared", advisory=True)
    return judge_prefix_shared_suffix(first[0], warm[0], cold[0], cached_tokens=warm[1], declared=declared)
