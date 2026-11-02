# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Read-only probes of a live OpenAI-shaped server, used by `tt-orchard can boot` after /health answers.

These checks come from the study in docs/run-logs/2026-11-02-serve-model-verify-mapping.md, which maps the verification table of the
tt-serve-model plugin (tenstorrent/skills, PR 29, skill `verify`) onto what this repository checks. The probes are written from that
study's description of each check. They send ordinary requests at temperature 0, change nothing on the server and open no device.

Every probe returns a dict `{"name", "state", "detail", ...}` where `state` is one of

  pass         the server did what the check asks;
  fail         the server answered and the answer is wrong, or did not answer;
  not_checked  the server does not offer the thing asked about (no /metrics, no /v1/completions), or an input the check needs is missing.

`not_checked` is never a pass. `record` turns a result into a boot-check entry: pass and fail become a named check, and a
`not_checked` goes in `evidence["not_checked"]` with its reason, so a reader can see what was not looked at.

What is checked here:
  * `check_listing`: the model name and context length `/v1/models` reports against the values the can declares (the context length
    is `max_model_len` of the spec; the name is compared only when a name is declared). A server that answered /health and lists
    no model fails.
  * `check_reply`: a chat reply names the model that was asked for, and is not empty. Empty content together with finish reason
    `length` is reported as a spent token budget (a thinking model that used all its tokens before it answered).
  * `probe_completions`: two raw completions at temperature 0 (a product, and a counting sequence) that show a server that answers
    but computes wrongly, or repeats tokens.
  * `scrape` and `check_counter`: the server's own `vllm:generation_tokens_total` counter must rise by at least the tokens one
    request reports generating. This shows the engine saw the work.
  * `check_mesh`: the device count the engine logs when it opens its mesh must equal the devices the can declares.

Not checked here, and named in the study: throughput and batching (V10, V11), long prefill (V12), reasoning-model parser checks
(V13 to V18), the quiet-warning log filter (V22) and the upstream lookups (V23 to V25).
"""
from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request

from orchard import safeio

MAX_BODY_BYTES = 4 << 20             # choice: a listing, a reply or a completion from a server under test is small; more is not read
MAX_METRICS_BYTES = 4 << 20          # choice: a vLLM /metrics page is tens of kilobytes
MAX_NAME_CHARS = 512
MAX_CONTEXT = 1 << 40                # choice: no served context is larger than this; a larger value is a malformed answer
PROBE_TIMEOUT_S = 60.0
ABSENT_ROUTE = (404, 405, 501)       # statuses that mean "this server has no such route"
COUNTER = "vllm:generation_tokens_total"
_COUNTER_LINE = re.compile(r"^" + re.escape(COUNTER) + r"(?:\{[^}\n]*\})?[ \t]+(\S+)[ \t]*$", re.MULTILINE)
_MESH_LINE = re.compile(r"(?:^|\s)multidevice with (\d{1,5}) devices and grid \(", re.MULTILINE)
_SEQUENCE = re.compile(r"\A\s*9[\s,]+10[\s,]+11(?!\d)")
_FIRST_INT = re.compile(r"\A\s*(-?\d+)(?!\d)")


def _result(name: str, state: str, detail: str, **more) -> dict:
    return {"name": name, "state": state, "detail": detail, **more}


def _json(body, what: str):
    """The parsed JSON object in `body`, or None."""
    try:
        return safeio.parse_json_bytes(body if body is not None else b"", max_bytes=MAX_BODY_BYTES, expect=dict)
    except (safeio.SafeReadError, TypeError, ValueError):
        return None


def _name(value) -> str | None:
    try:
        return safeio.bounded_str(value, max_len=MAX_NAME_CHARS, min_len=1, what="name")
    except safeio.BadValue:
        return None


def _context(value) -> int | None:
    """A reported context length when it is a whole number from 1 to MAX_CONTEXT; None for anything else (a string, a float, a bool, 0)."""
    try:
        return safeio.whole_number(value, lo=1, hi=MAX_CONTEXT, what="max_model_len")
    except safeio.BadValue:
        return None


# ---- /v1/models: served name and context length (verify V2) -------------------------------------------------------------------

def check_listing(status, body, *, declared_name=None, declared_context=None, native_context=None) -> dict:
    """The model the server lists, against what the can declares. `declared_context` is the `max_model_len` of the spec (None when it
    declares none, and then the context length is reported and not compared). `native_context` (the model's own
    `max_position_embeddings`) only labels the served value as below, equal to or above native; it never fails the check."""
    name = "served_model"
    if status != 200:
        return _result(name, "fail", f"GET /v1/models answered {status}, not 200")
    doc = _json(body, "listing")
    data = doc.get("data") if doc is not None else None
    if doc is None or not isinstance(data, list):
        return _result(name, "fail", "GET /v1/models did not answer a JSON object with a `data` list")
    entries = [e for e in data if isinstance(e, dict) and _name(e.get("id")) is not None]
    if not entries:
        return _result(name, "fail", "the server answered /health and lists no model on /v1/models (the list is empty or its entries have no usable id)")
    chosen = entries[0]
    if declared_name is not None:
        match = [e for e in entries if e["id"] == declared_name]
        if not match:
            return _result(name, "fail", f"the server lists {[e['id'] for e in entries][:5]}; the can declares the name {declared_name!r}")
        chosen = match[0]
    served = {"id": chosen["id"], "max_model_len": _context(chosen.get("max_model_len"))}
    notes = []
    if declared_context is not None:
        if "max_model_len" not in chosen or chosen.get("max_model_len") is None:
            return _result(name, "not_checked", f"the server does not report max_model_len, so the declared context {declared_context} was not compared",
                           served=served)
        if served["max_model_len"] is None:
            return _result(name, "fail", f"the server reports an unusable max_model_len ({type(chosen['max_model_len']).__name__}); "
                                         f"the can declares {declared_context}", served=served)
        if served["max_model_len"] != declared_context:
            return _result(name, "fail", f"the server serves a context of {served['max_model_len']} tokens; the can declares {declared_context}", served=served)
        notes.append(f"context {declared_context} matches the declared value")
    else:
        notes.append("the context was not compared (the can declares none)")
    if native_context is not None and served["max_model_len"] is not None and _context(native_context) is not None:
        served_len, native = served["max_model_len"], _context(native_context)
        notes.append("context is " + ("below native" if served_len < native else "equal to native" if served_len == native else "above native (rope scaling)"))
    return _result(name, "pass", f"{served['id']!r} is listed; " + "; ".join(notes), served=served)


# ---- a chat reply (verify V17, and a model-name echo) -------------------------------------------------------------------------

def check_reply(status, body, *, asked_model=None) -> dict:
    """The chat reply names the model asked for (when it names one at all) and holds text. Empty content with finish reason `length`
    means the whole token budget went to thinking."""
    name = "reply_shape"
    if status != 200:
        return _result(name, "fail", f"POST /v1/chat/completions answered {status}, not 200")
    doc = _json(body, "reply")
    choices = doc.get("choices") if doc is not None else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        return _result(name, "fail", "the reply has no choices[0].message object")
    echoed = doc.get("model")
    if asked_model is not None and isinstance(echoed, str) and echoed != asked_model:
        return _result(name, "fail", f"the reply names the model {echoed[:MAX_NAME_CHARS]!r}; the request asked for {asked_model!r}")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        if choice.get("finish_reason") == "length":
            return _result(name, "fail", "the content is empty and the finish reason is length: the token budget was spent before an answer "
                                         "(a thinking model needs a larger budget or thinking turned off)")
        return _result(name, "fail", f"the content is empty (finish reason {choice.get('finish_reason')!r})")
    return _result(name, "pass", "the reply holds text" + (" and names the requested model" if isinstance(echoed, str) else ""))


# ---- raw completions at temperature 0 (verify V4, V5) -------------------------------------------------------------------------

def _post(url: str, body: dict, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(MAX_BODY_BYTES + 1)[:MAX_BODY_BYTES]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(MAX_BODY_BYTES + 1)[:MAX_BODY_BYTES]


def _completion_text(body) -> str | None:
    doc = _json(body, "completion")
    choices = doc.get("choices") if doc is not None else None
    first = choices[0] if isinstance(choices, list) and choices else None
    text = first.get("text") if isinstance(first, dict) else None
    return text if isinstance(text, str) else None


def probe_completions(base: str, model: str, *, timeout: float = PROBE_TIMEOUT_S) -> list[dict]:
    """The arithmetic and the sequential-state probes, one request each, at temperature 0 with at most 16 new tokens. A server with no
    /v1/completions route (a chat-only server) is `not_checked`. A server that cannot be reached is a fail."""
    probes = (("arithmetic", "17*24=", lambda t: _FIRST_INT.match(t) is not None and int(_FIRST_INT.match(t).group(1)) == 408,
               "the product 17*24 is 408"),
              ("sequence", "1 2 3 4 5 6 7 8", lambda t: _SEQUENCE.match(t) is not None, "counting on from 8 gives 9, 10, 11 with no repeat or skip"))
    out = []
    for name, prompt, good, meaning in probes:
        try:
            status, body = _post(base.rstrip("/") + "/v1/completions", {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 10}, timeout)
        except (OSError, ValueError) as exc:        # a refused connection, a timeout, a cut-off body; http.client errors are OSError subclasses here
            out.append(_result(name, "fail", f"no answer from /v1/completions: {type(exc).__name__}"))
            continue
        if status in ABSENT_ROUTE:
            out.append(_result(name, "not_checked", f"the server has no /v1/completions route (it answered {status})"))
        elif status != 200:
            out.append(_result(name, "fail", f"POST /v1/completions answered {status}"))
        else:
            text = _completion_text(body)
            if text is None:
                out.append(_result(name, "fail", "the completion has no choices[0].text string"))
            elif good(text):
                out.append(_result(name, "pass", meaning, text=text[:40]))
            else:
                out.append(_result(name, "fail", f"expected: {meaning}; the server answered {text[:40]!r}", text=text[:40]))
    return out


# ---- the server's own counters (verify V8) -------------------------------------------------------------------------------------

def scrape(base: str, *, timeout: float = 5.0) -> str | None:
    """The text of GET /metrics, or None when the server has none, answers an error, is unreachable, or sends more than MAX_METRICS_BYTES."""
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=timeout) as r:
            if r.status != 200:
                return None
            raw = r.read(MAX_METRICS_BYTES + 1)
    except (OSError, ValueError):
        return None
    if len(raw) > MAX_METRICS_BYTES:
        return None
    return raw.decode("utf-8", errors="replace")


def parse_counter(text) -> float | None:
    """The sum of `vllm:generation_tokens_total` over every engine label, or None when the counter is absent or no value is a finite
    number of zero or more."""
    if not isinstance(text, str):
        return None
    values = []
    for raw in _COUNTER_LINE.findall(text):
        try:
            v = float(raw)
        except ValueError:
            continue
        if math.isfinite(v) and v >= 0:
            values.append(v)
    return sum(values) if values else None


def check_counter(before, after, *, completion_tokens) -> dict:
    """The token counter read before and after one request must have risen by at least the tokens that request reports generating.
    Other traffic may raise it by more. Missing text, a missing counter or a missing token count is `not_checked`."""
    name = "engine_counter"
    if isinstance(completion_tokens, bool) or not isinstance(completion_tokens, int) or completion_tokens <= 0:
        return _result(name, "not_checked", "the reply did not report how many tokens it generated (usage.completion_tokens)")
    b, a = parse_counter(before), parse_counter(after)
    if b is None or a is None:
        return _result(name, "not_checked", f"the server's /metrics does not report {COUNTER} (before: {'yes' if b is not None else 'no'}, "
                                           f"after: {'yes' if a is not None else 'no'})")
    delta = a - b
    if delta <= 0:
        return _result(name, "fail", f"{COUNTER} did not rise across a request that generated {completion_tokens} tokens: the engine's own "
                                     "counter did not see the work", delta=delta)
    if delta + 1e-6 < completion_tokens:
        return _result(name, "fail", f"{COUNTER} rose by {delta:g}; the request reports {completion_tokens} generated tokens", delta=delta)
    return _result(name, "pass", f"{COUNTER} rose by {delta:g} across a request that generated {completion_tokens} tokens", delta=delta)


# ---- the mesh the engine opened (verify V7) -------------------------------------------------------------------------------------

def check_mesh(log_text, *, expected_devices) -> dict:
    """The last `multidevice with N devices and grid (...)` line of the server's log against the devices the can declares."""
    name = "mesh_devices"
    if isinstance(expected_devices, bool) or not isinstance(expected_devices, int) or expected_devices <= 0:
        return _result(name, "not_checked", "no device count is declared, so the mesh the engine opened was not compared")
    if not isinstance(log_text, str):
        return _result(name, "not_checked", "the server log could not be read")
    seen = _MESH_LINE.findall(log_text)
    if not seen:
        return _result(name, "not_checked", "the server log has no `multidevice with N devices and grid` line (not every server logs its mesh)")
    n = int(seen[-1])
    if n != expected_devices:
        return _result(name, "fail", f"the engine opened a mesh of {n} devices; the can declares {expected_devices}", devices=n)
    return _result(name, "pass", f"the engine opened a mesh of {n} devices, as declared", devices=n)


# ---- into the boot check's evidence ---------------------------------------------------------------------------------------------

def record(result: dict, check, evidence: dict) -> None:
    """`pass` and `fail` become a named check (`check(name, ok, detail)` is the boot check's own recorder). `not_checked` is listed in
    `evidence["not_checked"]` with its reason and does not change `passed`."""
    if result["state"] == "not_checked":
        evidence.setdefault("not_checked", []).append({"name": result["name"], "why": result["detail"]})
    else:
        check(result["name"], result["state"] == "pass", result["detail"])
