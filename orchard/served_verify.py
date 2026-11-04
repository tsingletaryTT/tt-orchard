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

Added for issues 360 to 373 and 385 (the live multi-request probes are in orchard/served_probes.py):
  * `served_names` and `request_model`: a `--served-model-name` alias set only through `extra_args` is the name the request carries (385).
  * `judge_completion`: the judgement of one raw completion, so a recorded body (stage 2 and 4, issue 368) is judged by the same rules.
  * `check_engine_log`: layers loaded and KV-cache capacity from the server log, against config.json and the can's limits (370).
  * `log_warnings`: the quiet-warning filter over the server log (373). It never fails a boot.
  * `reasoning_split`, `distribution`, `classify_reasoning`, `check_prefix_reuse`: judges for reasoning replies and prefix reuse (360, 361, 372).

Not checked here, and named in the study: the upstream lookups (V23 to V25).
"""
from __future__ import annotations

import http.client
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

def served_names(extra_args) -> tuple[str, ...]:
    """The names `--served-model-name` gives in a spec's serving arguments (issue 385): `--served-model-name a b`, `--served_model_name=a`.
    The option takes one or more names, each the word after it up to the next word that starts with a dash. Dashes and underscores in the
    option name are the same option, and the last use wins (argparse). A name that is empty or longer than MAX_NAME_CHARS is dropped.
    No such option gives an empty tuple."""
    found: tuple[str, ...] = ()
    words = [w for w in (extra_args or ()) if isinstance(w, str)]
    i = 0
    while i < len(words):
        head, eq, tail = words[i].partition("=")
        if head.startswith("--") and head[2:].replace("-", "_") == "served_model_name":
            names = [tail] if eq else []
            i += 1
            while i < len(words) and not words[i].startswith("-"):
                names.append(words[i])
                i += 1
            found = tuple(n for n in names if _name(n) is not None)
            continue
        i += 1
    return found


def request_model(body, declared_names=()) -> str | None:
    """The model name a request will carry: the first declared name that `/v1/models` lists, else (when none is declared) the first listed
    id. None when the listing is unusable or lists none of the declared names; `check_listing` says why."""
    doc = _json(body, "listing")
    data = doc.get("data") if doc is not None else None
    ids = [e["id"] for e in data if isinstance(e, dict) and _name(e.get("id")) is not None] if isinstance(data, list) else []
    if declared_names:
        return next((n for n in declared_names if n in ids), None)
    return ids[0] if ids else None


def check_listing(status, body, *, declared_name=None, declared_names=(), declared_context=None, native_context=None) -> dict:
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
    declared = tuple(n for n in ((declared_name,) + tuple(declared_names or ())) if isinstance(n, str) and n)
    if declared:
        listed = {e["id"]: e for e in entries}
        hit = next((n for n in declared if n in listed), None)
        if hit is None:
            shown = repr(declared[0]) if len(declared) == 1 else repr(list(declared))
            return _result(name, "fail", f"the server lists {[e['id'] for e in entries][:5]}; the can declares the name {shown}")
        chosen = listed[hit]
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
    out = []
    for name, prompt in COMPLETION_PROBES:
        try:
            status, body = _post(base.rstrip("/") + "/v1/completions", {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 10}, timeout)
        except (OSError, ValueError, http.client.HTTPException) as exc:        # a refused connection, a timeout, a cut-off body (IncompleteRead is not an OSError)
            out.append(_result(name, "fail", f"no answer from /v1/completions: {type(exc).__name__}"))
            continue
        out.append(judge_completion(name, status, body))
    return out


COMPLETION_PROBES = (("arithmetic", "17*24="), ("sequence", "1 2 3 4 5 6 7 8"))     # (name, prompt); the stage 2 template sends the same two (a test holds them equal)
_GOOD = {"arithmetic": (lambda t: _FIRST_INT.match(t) is not None and int(_FIRST_INT.match(t).group(1)) == 408, "the product 17*24 is 408"),
         "sequence": (lambda t: _SEQUENCE.match(t) is not None, "counting on from 8 gives 9, 10, 11 with no repeat or skip")}


def judge_completion(name: str, status, body) -> dict:
    """The verdict on one raw completion answered with `status` and `body` for the probe `name` of COMPLETION_PROBES. The live probe and the
    stage 2 and 4 recorded probe both call this, so one rule judges both."""
    good, meaning = _GOOD[name]
    if status in ABSENT_ROUTE:
        return _result(name, "not_checked", f"the server has no /v1/completions route (it answered {status})")
    if status != 200:
        return _result(name, "fail", f"POST /v1/completions answered {status}")
    text = _completion_text(body)
    if text is None:
        return _result(name, "fail", "the completion has no choices[0].text string")
    if good(text):
        return _result(name, "pass", meaning, text=text[:40])
    return _result(name, "fail", f"expected: {meaning}; the server answered {text[:40]!r}", text=text[:40])


# ---- the server's own counters (verify V8) -------------------------------------------------------------------------------------

def scrape(base: str, *, timeout: float = 5.0) -> str | None:
    """The text of GET /metrics, or None when the server has none, answers an error, is unreachable, or sends more than MAX_METRICS_BYTES."""
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=timeout) as r:
            if r.status != 200:
                return None
            raw = r.read(MAX_METRICS_BYTES + 1)
            if r.length:              # the server promised more bytes than arrived: a cut-off page could hold half a number
                return None
    except (OSError, ValueError, http.client.HTTPException):
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


# ---- layers loaded and the KV cache, from the server log (verify V9, issue 370) --------------------------------------------------
# The formats below were read from a real server log on 2026-11-03: vLLM 0.26.0 (V1 engine) serving a Qwen3.8-27B-class package with the
# tt-metal `qwen36` model code on two Blackhole chips. The model prints `Loading 64 transformer layers (indices=[0, 1, ...])` before it builds
# its stack and a progress bar `Loading layers: 100%|##########| 64/64` as it finishes; the engine prints `Overriding num_gpu_blocks=4100 with
# num_gpu_blocks_override=4100`, then `GPU KV cache size: 262,400 tokens` and `Maximum concurrency for 262,144 tokens per request: 1.00x`. Each
# appeared once in every such log on this box (23 logs). Other releases and models print other shapes: the older engine prints `# GPU blocks: N`,
# and the guesses `built N decoder layers` and `layers loaded: N` stay in the list. A log with none of them is `not_checked`. The tests pin
# the fixtures by release (tests/test_served_verify_engine_real.py, tests/test_served_verify_logs.py).
#
# Two facts the real lines show. The layer line is printed before the layers load, so it states an intent: the progress bar is what says they
# finished. And on a TT server the engine's block count is set by the plugin from the size the model asked for (`Getting max_tokens_all_users=
# 262144 for number of blocks in KV cache`), so the `GPU KV cache size` that follows is that request rounded up to whole blocks. It is not a
# measurement of device memory, and a check of it against `max_model_len` cannot fail on such a server. The result says which it is.
_LAYERS_LINES = (re.compile(r"\b(?P<verb>loaded|loading|built|created|initiali[sz]ed)\s+(?P<n>\d{1,4})\s+(?:decoder\s+|transformer\s+)?layers\b", re.IGNORECASE),
                 re.compile(r"\blayers\s+loaded\s*[:=]\s*(?P<n>\d{1,4})\b", re.IGNORECASE))
# The tqdm bar of a tt-metal model's layer loop. Every width is bounded, so a line of repeated bar text costs a fixed amount per repeat.
_LAYERS_BAR = re.compile(r"Loading layers:\s{0,8}\d{1,3}%\|[^|\r\n]{0,64}\|\s{0,8}(\d{1,4})/(\d{1,4})\b")
_KV_TOKENS = re.compile(r"\bGPU KV cache size:\s*([\d,]{1,20})\s*tokens\b", re.IGNORECASE)
_KV_CONCURRENCY = re.compile(r"\bMaximum concurrency for\s*([\d,]{1,20})\s*tokens per request:\s*(\d{1,9}(?:\.\d{1,6})?)x\b", re.IGNORECASE)
_KV_BLOCKS = re.compile(r"#\s*GPU blocks:\s*(\d{1,12})\b", re.IGNORECASE)
_KV_OVERRIDE = re.compile(r"\bOverriding num_gpu_blocks=\d{1,12}\s+with\s+num_gpu_blocks_override=(\d{1,12})\b")
MAX_ENGINE_LOG_CHARS = 64 * 1024 * 1024      # choice: the bound the boot check puts on its read of the server log (orchard/can.py MAX_LOG_BYTES)
MAX_LAYER_COUNTS_LISTED = 5                  # choice: a log that states more different layer counts than this is listed by its first few


def layers_in_config(config) -> int | None:
    """`num_hidden_layers` of a parsed config.json (top level, or under `text_config` for a multimodal wrapper), or None."""
    if not isinstance(config, dict):
        return None
    for holder in (config, config.get("text_config")):
        if isinstance(holder, dict):
            try:
                return safeio.whole_number(holder.get("num_hidden_layers"), lo=1, hi=100000, what="num_hidden_layers")
            except safeio.BadValue:
                continue
    return None


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def _whole(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _announcements(log_text: str) -> list[tuple[int, int, bool]]:
    """(position, layer count, intent) of every line that states a layer count, in log order. `intent` is true for `Loading N ... layers`
    (the present continuous that a tt-metal model prints before its layer loop), false for the past-tense and `layers loaded: N` forms that
    are printed after the layers finish."""
    found = []
    for rx in _LAYERS_LINES:
        for m in rx.finditer(log_text):
            verb = m.groupdict().get("verb")
            found.append((m.start(), int(m.group("n")), verb is not None and verb.lower() == "loading"))
    return sorted(found)


def _bars_of_last_announcement(announcements: list[tuple[int, int, bool]], log_text: str) -> list[tuple[int, int]]:
    """The layer progress bars (done, total) that belong to the last announcement. A bar belongs to the component whose announcement it goes
    with, and a log can hold the bars of several components (a drafter, then the target). For an `intent` line the layer loop runs after the
    line, so only bars after it count; a bar before it is an earlier component's. For a past-tense line the loop ran before the line, so the
    bars between the previous announcement and this one count, with one exception: when the previous announcement is a `Loading N layers` line
    for a different count, the bars after it that count N are that component's loop and are left out (review of PR 443, round 3: a completion
    statement for 64 layers must not take the finished 32-layer bar of the component announced just before it). A past-tense line for the same
    count completes that `Loading` line and takes its bar. No announcement, no bars."""
    if not announcements:
        return []
    pos, count, intent = announcements[-1]
    bars = [(m.start(), int(m.group(1)), int(m.group(2))) for m in _LAYERS_BAR.finditer(log_text)]
    if intent:
        return [(done, total) for at, done, total in bars if at > pos]
    prev = announcements[-2] if len(announcements) > 1 else None
    lo = prev[0] if prev is not None else -1
    inside = [(done, total) for at, done, total in bars if lo < at < pos]
    if prev is not None and prev[2] and prev[1] != count:
        inside = [(done, total) for done, total in inside if total != prev[1]]
    return inside


def _layers_result(seen: list[int], bars: list[tuple[int, int]], expected: int | None) -> dict:
    """`layers_loaded` from the layer counts the log states (in log order), the last layer progress bar, and `num_hidden_layers`. The last
    stated count decides, as for the other log lines. When several different counts are stated they are listed, and a count that equals
    the config's but is not the last is named: the last line may belong to another component (a drafter, say)."""
    if expected is None:
        return _result("layers_loaded", "not_checked", "no num_hidden_layers is known (the bundle holds no config.json that states it)")
    if not seen:
        return _result("layers_loaded", "not_checked", "the server log has no line that states how many layers were loaded")
    last = seen[-1]
    distinct = sorted(set(seen))
    facts = {"layers": last, "layer_counts_seen": distinct[:MAX_LAYER_COUNTS_LISTED], "layers_completed": bars[-1][0] if bars else None}
    if last != expected:
        hint = f" (another line states {expected}, the config's count; the last line may belong to another component)" if expected in seen[:-1] else ""
        return _result("layers_loaded", "fail", f"the engine loaded {last} layers; config.json says num_hidden_layers is {expected}{hint}", **facts)
    if bars and bars[-1][0] != bars[-1][1]:
        return _result("layers_loaded", "fail", f"the engine announced {last} layers but its layer progress bar ended at {bars[-1][0]}/{bars[-1][1]}", **facts)
    if bars and bars[-1][1] != last:
        return _result("layers_loaded", "fail", f"the engine announced {last} layers but its layer progress bar counts {bars[-1][1]}", **facts)
    note = f"progress bar {bars[-1][0]}/{bars[-1][1]}" if bars else "no progress bar in the log, so the count is the engine's announcement"
    return _result("layers_loaded", "pass", f"the engine loaded {last} layers, as config.json says ({note})", **facts)


def check_engine_log(log_text, *, config=None, max_model_len=None, max_num_seqs=None, block_size=None) -> list[dict]:
    """Two results from the server log: `layers_loaded` (the layers the engine says it loaded, against `num_hidden_layers` of `config`, and the
    layer progress bar where the model prints one) and `kv_cache` (the cache capacity in tokens and the concurrency the engine states, against the
    can's `max_model_len`). The last matching line wins. At most `MAX_ENGINE_LOG_CHARS` characters are read.

    A cache smaller than one full-length sequence, or a stated concurrency below 1, is a fail. A cache that holds fewer full-length sequences than
    `max_num_seqs` is not a fail: the result says so (`full_length_sequences`, `fewer_than_max_num_seqs`), because that many requests that all
    approach `max_model_len` cannot be in flight together. `capacity_basis` says whether the capacity follows a block count the plugin set
    (`plugin override`: a size the plugin chose, which says nothing about device memory) or is the engine's own. A log without the line, or no `config` to compare
    with, is `not_checked`."""
    out = []
    expected = layers_in_config(config)
    if not isinstance(log_text, str):
        why = "the server log could not be read"
        return [_result("layers_loaded", "not_checked", why), _result("kv_cache", "not_checked", why)]
    if len(log_text) > MAX_ENGINE_LOG_CHARS:
        log_text = log_text[:MAX_ENGINE_LOG_CHARS]
    announced = _announcements(log_text)                                      # log order, with positions
    out.append(_layers_result([count for _, count, _ in announced], _bars_of_last_announcement(announced, log_text), expected))
    # Each capacity report is a line with its position. The selected report is the last one, and the override and the concurrency line that
    # describe it are the ones next to it in the log: an override printed before the report since the previous report, and the concurrency
    # line printed after it. One that belongs to another report is not read as this one's (review of PR 443, round 3).
    tokens = [(m.start(), _int(m.group(1))) for m in _KV_TOKENS.finditer(log_text)]
    blocks = [(m.start(), int(m.group(1))) for m in _KV_BLOCKS.finditer(log_text)]
    all_conc = [(m.start(), _int(m.group(1)), float(m.group(2))) for m in _KV_CONCURRENCY.finditer(log_text)]
    all_overrides = [(m.start(), int(m.group(1))) for m in _KV_OVERRIDE.finditer(log_text)]
    at, capacity = None, None
    if tokens:
        at, capacity = tokens[-1]
    elif blocks and _whole(block_size) and block_size > 0:
        at, capacity = blocks[-1][0], blocks[-1][1] * block_size
    if at is None:
        conc = [(per_request, factor) for _, per_request, factor in all_conc[-1:]]
        overrides = []
    else:
        earlier = [p for p, _ in tokens + blocks if p < at]
        previous = max(earlier) if earlier else -1
        overrides = [n for p, n in all_overrides if previous < p < at]
        conc = [(per_request, factor) for p, per_request, factor in all_conc if p > at][-1:]
    if capacity is None and not conc:
        out.append(_result("kv_cache", "not_checked", "the server log states no KV-cache size (no `GPU KV cache size` line, and no block count with a declared block size)"))
        return out
    basis = None if capacity is None else ("plugin override" if overrides else "engine")
    parts = []
    if capacity is not None:
        parts.append(f"capacity {capacity} tokens")
        if overrides:
            parts.append(f"the plugin set the block count to {overrides[-1]}, so this figure is the size it asked for and not a measured allocation")
    if conc:
        parts.append(f"concurrency {conc[-1][1]:g}x at {conc[-1][0]} tokens per request")
    if max_num_seqs is not None:
        parts.append(f"the can's max_num_seqs is {max_num_seqs}")
    # A sequence takes whole blocks, so with a usable block size a request of `max_model_len` tokens needs ceil(max_model_len / block) of them and the
    # cache holds floor(capacity / block) of them; without one, the unrounded division stays (issue 370, review of PR 443).
    block = block_size if _whole(block_size) and block_size > 0 else None
    sequences = blocks_per_sequence = None
    if capacity is not None and _whole(max_model_len) and max_model_len > 0:
        if block is not None:
            blocks_per_sequence = -(-max_model_len // block)
            sequences = (capacity // block) // blocks_per_sequence
            if max_model_len % block:
                parts.append(f"a sequence of {max_model_len} tokens takes {blocks_per_sequence} whole blocks of {block} tokens")
        else:
            sequences = capacity // max_model_len
    fewer = sequences < max_num_seqs if sequences is not None and _whole(max_num_seqs) else None
    if fewer:
        parts.append(f"only {sequences} full-length sequence{'' if sequences == 1 else 's'} {'fits' if sequences == 1 else 'fit'}, so requests that all approach "
                     f"max_model_len wait or are preempted")
    facts = {"capacity_tokens": capacity, "capacity_basis": basis, "override_blocks": overrides[-1] if overrides else None,
             "concurrency": conc[-1][1] if conc else None, "tokens_per_request": conc[-1][0] if conc else None,
             "full_length_sequences": sequences, "fewer_than_max_num_seqs": fewer, "blocks_per_sequence": blocks_per_sequence}
    if capacity is not None and _whole(max_model_len) and capacity < max_model_len:
        out.append(_result("kv_cache", "fail", f"the KV cache holds {capacity} tokens, less than one sequence of the declared max_model_len {max_model_len}", **facts))
    elif sequences == 0 and blocks_per_sequence is not None:
        out.append(_result("kv_cache", "fail", f"the KV cache holds {capacity // block} whole blocks of {block} tokens, fewer than the {blocks_per_sequence} whole blocks "
                           f"one sequence of the declared max_model_len {max_model_len} needs", **facts))
    elif conc and conc[-1][1] < 1:
        out.append(_result("kv_cache", "fail", f"the engine states a concurrency of {conc[-1][1]:g}x for {conc[-1][0]} tokens per request: one full-length sequence does not fit", **facts))
    else:
        out.append(_result("kv_cache", "pass", "; ".join(parts), **facts))
    return out


# ---- quiet warnings in the server log (verify V22, issue 373) ------------------------------------------------------------------

MAX_WARNING_LINES = 20             # choice: a person reads these; more than this is a log that needs reading in full
MAX_WARNING_CHARS = 300
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# a line that every healthy run prints: the periodic engine statistics, and the import messages of backends this machine does not use
_BACKEND = r"\b(?:cuda|libcuda|rocm|hip|xpu|nvml|pynvml|triton|flash_?attn|flashinfer|nccl|cudnn)\b"
# a backend word with an import or start-up message (the backend is absent from this machine), and the engine statistics line. A warning that
# names a backend and reports a fallback or a mismatch does not match the second form and stays.
_NOISE = re.compile(r"Avg (?:prompt|generation) throughput|Engine \d+:.*(?:Running|Waiting):|\bPrefix cache hit rate\b|"
                    r"(?=.*" + _BACKEND + r")(?=.*(?:failed to import|importerror|no module named|cannot open shared object|not installed|is installed but|"
                    r"not available|no (?:cuda|rocm|xpu|gpu)|driver\(s\) found|libcuda))", re.IGNORECASE)
_WARNING = re.compile(r"\b(?:warn(?:ing)?s?|fall(?:s|ing)?[ -]?back|fell back|unrecogni[sz]ed|unknown|unsupported|not supported|not recogni[sz]ed|"
                      r"ignor(?:e|es|ed|ing)|mismatch(?:es|ed)?|deprecat\w*|defaulting|overrid(?:e|es|ing|den)|disabled|disabling|"
                      r"not (?:found|available)|no such|cannot|could not|failed)\b", re.IGNORECASE)
_SHAPE = re.compile(r"\d[\d,.:%-]*")


def log_warnings(log_text) -> dict:
    """The lines of a server log that may explain a slow or wrong run (issue 373): fallbacks, an unrecognised device or model name, ignored
    options, mismatches, deprecations. It drops the lines every healthy run prints (see _NOISE), keeps lines that match a broad keyword
    list, collapses lines that differ only in numbers and times, keeps at most MAX_WARNING_LINES, and cuts each at MAX_WARNING_CHARS.
    A hit means read the line. It does not mean the run is wrong, and the boot check never fails on it. Returns
    `{"lines": [{"text", "count"}], "hits": N, "dropped_as_noise": N, "truncated": bool}`; a log that cannot be read gives `lines: None`."""
    if not isinstance(log_text, str):
        return {"lines": None, "hits": 0, "dropped_as_noise": 0, "truncated": False}
    seen: dict[str, dict] = {}
    hits = noise = 0
    truncated = False
    for raw in log_text.splitlines():
        line = _ANSI.sub("", raw).strip()
        if not line or not _WARNING.search(line):
            continue
        if _NOISE.search(line):
            noise += 1
            continue
        hits += 1
        key = _SHAPE.sub("#", line)[:MAX_WARNING_CHARS]
        if key in seen:
            seen[key]["count"] += 1
        elif len(seen) < MAX_WARNING_LINES:
            seen[key] = {"text": line[:MAX_WARNING_CHARS], "count": 1}
        else:
            truncated = True
    return {"lines": list(seen.values()), "hits": hits, "dropped_as_noise": noise, "truncated": truncated}


# ---- reasoning replies and prefix reuse (verify V13 to V17, issues 360, 361, 372) ----------------------------------------------------

def distribution(values) -> dict | None:
    """`{"n", "min", "median", "max"}` of the whole numbers in `values` (booleans and other types are skipped), or None when there are none."""
    xs = sorted(v for v in (values or ()) if isinstance(v, int) and not isinstance(v, bool) and v >= 0)
    if not xs:
        return None
    mid = xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2
    return {"n": len(xs), "min": xs[0], "median": mid, "max": xs[-1]}


def _message_texts(message) -> tuple[str, str]:
    """(reasoning text, answer text) of a chat message. The reasoning field is `reasoning_content` (vLLM before 0.11) or `reasoning`."""
    if not isinstance(message, dict):
        return "", ""
    reasoning = next((message[k] for k in ("reasoning_content", "reasoning") if isinstance(message.get(k), str) and message[k].strip()), "")
    content = message.get("content") if isinstance(message.get("content"), str) else ""
    return reasoning, content


def reasoning_split(body) -> dict | None:
    """Reasoning tokens and answer tokens of one chat reply (issue 361), from a reply body (bytes, or an already parsed dict). Returns
    `{"completion_tokens", "reasoning_tokens", "answer_tokens", "basis", "finish_reason"}` or None when the reply has no usable
    `usage.completion_tokens`. `basis` says where the split comes from: `reported` (usage.completion_tokens_details.reasoning_tokens), `estimated
    by characters` (the share of reasoning characters in the reply, which is an estimate) or `no reasoning text`."""
    doc = body if isinstance(body, dict) else _json(body, "reply")
    if doc is None:
        return None
    usage = doc.get("usage")
    total = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        return None
    choices = doc.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    reasoning, content = _message_texts(choice.get("message"))
    details = usage.get("completion_tokens_details")
    reported = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if isinstance(reported, int) and not isinstance(reported, bool) and 0 <= reported <= total:
        r, basis = reported, "reported"
    elif reasoning:
        r, basis = round(total * len(reasoning) / (len(reasoning) + len(content))), "estimated by characters"
    else:
        r, basis = 0, "no reasoning text"
    return {"completion_tokens": total, "reasoning_tokens": r, "answer_tokens": total - r, "basis": basis, "finish_reason": choice.get("finish_reason")}


_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
DROPPED_RATIO = 3          # choice: a reply whose token count is three times its visible answer and more, with no reasoning field, generated thinking that the server dropped
DROPPED_FLOOR = 16         # choice: and at least this many tokens beyond the visible answer


def classify_reasoning(body) -> dict:
    """How one chat reply of a reasoning model carries its thinking (issue 372): `populated` (a reasoning field holds text and the content is
    clean), `leaking` (a think tag in the content), `dropped` (no reasoning field, and the token count far above the visible answer: thinking
    was generated and the parser threw it away), `none` (no thinking text and no sign of dropped thinking) or `unusable` (not a chat reply)."""
    doc = body if isinstance(body, dict) else _json(body, "reply")
    choices = doc.get("choices") if doc is not None else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    if choice is None or not isinstance(choice.get("message"), dict):
        return {"kind": "unusable", "detail": "the reply has no choices[0].message object"}
    reasoning, content = _message_texts(choice["message"])
    split = reasoning_split(doc)
    if _THINK_TAG.search(content) or _THINK_TAG.search(reasoning):
        return {"kind": "leaking", "detail": "a think tag is in the reply text: the server's reasoning parser is not separating the thinking", "split": split}
    if reasoning and content.strip():
        return {"kind": "populated", "detail": "the reasoning field holds text and the content is the answer", "split": split}
    if reasoning:
        return {"kind": "populated", "detail": "the reasoning field holds text and the content is empty (the budget ended during thinking)", "split": split}
    if split is not None:
        visible = len(content) // 3 + 1                      # three characters to the token is an over-count of tokens, so the dropped test errs toward `none`
        if split["completion_tokens"] >= DROPPED_RATIO * visible + DROPPED_FLOOR:
            return {"kind": "dropped", "detail": f"{split['completion_tokens']} tokens were generated and the visible answer is about {visible}; the reasoning field is empty",
                    "split": split}
    return {"kind": "none", "detail": "no thinking text and no sign of dropped thinking", "split": split}


_PARSER_LINE = re.compile(r"reasoning[_-]parser\W{0,6}([A-Za-z0-9_.-]{1,64})", re.IGNORECASE)


def reasoning_parser_printed(log_text) -> bool:
    """True when the log holds an engine-argument form of the parser setting (`reasoning_parser=...`, `--reasoning-parser x`), with a value or `None`.
    The word `reasoning` alone (a model name) is not one."""
    return isinstance(log_text, str) and _PARSER_LINE.search(log_text) is not None


def reasoning_parser_in_log(log_text) -> str | None:
    """The name after `reasoning_parser` (or `--reasoning-parser`) in the server log's printed engine arguments; the last one wins. A value of
    `None` or `null` means no parser. None when the log prints none."""
    if not isinstance(log_text, str):
        return None
    found = _PARSER_LINE.findall(log_text)
    if not found:
        return None
    return None if found[-1].lower() in ("none", "null") else found[-1]


def check_prefix_reuse(first: dict, repeat: dict, control: dict, *, declared: bool) -> dict:
    """Is a repeated long prefix served faster (issues 360, 361)? Each argument is `{"seconds", "prompt_tokens", "cached_tokens"}` for the same-size
    requests: `first` (a long prompt, cold), `repeat` (the same prompt again) and `control` (a different prompt of the same size, so the
    server's warm kernels are not mistaken for a cache). The server's own count decides when it reports one (`usage.prompt_tokens_details.cached_tokens`
    of at least half the prompt); otherwise the repeat must take at most 60 percent of the control's time, and the verdict says it was judged by
    timing. `declared` says the can claims prefix caching: a repeat that is not faster is then a fail, and otherwise it is recorded as `not_checked`
    with the numbers (a can that does not claim prefix caching is not failed for lacking it)."""
    name = "prefix_reuse"

    def num(d, k):
        v = d.get(k) if isinstance(d, dict) else None
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 else None
    cached, prompt, t_rep, t_ctl = num(repeat, "cached_tokens"), num(repeat, "prompt_tokens"), num(repeat, "seconds"), num(control, "seconds")
    facts = {"first_s": num(first, "seconds"), "repeat_s": t_rep, "control_s": t_ctl, "prompt_tokens": prompt, "cached_tokens": cached}
    if cached is not None and prompt:
        reused, basis = cached >= prompt / 2, "the server's cached-token count"
    elif t_rep is not None and t_ctl:
        reused, basis = t_rep <= 0.6 * t_ctl, "timing (the server reports no cached-token count)"
    else:
        return _result(name, "not_checked", "the requests reported neither a cached-token count nor a usable time", **facts)
    if reused:
        return _result(name, "pass", f"a repeated prefix was served from the cache, judged by {basis}", basis=basis, **facts)
    detail = f"a repeated prefix was not served faster, judged by {basis}"
    return _result(name, "fail" if declared else "not_checked", detail + ("" if declared else " (the can does not declare prefix caching)"), basis=basis, **facts)


# ---- into the boot check's evidence ---------------------------------------------------------------------------------------------

# The structured facts of the two engine-log results that the boot evidence keeps beside the verdict (issue 370, review of PR 443, round 3).
# A fixed list per result name, so the evidence holds only values this module computed (whole numbers, a float, a short text, a short list of
# whole numbers) and its size stays bounded; any other result adds nothing.
ENGINE_FACT_KEYS = {
    "layers_loaded": ("layers", "layer_counts_seen", "layers_completed"),
    "kv_cache": ("capacity_tokens", "capacity_basis", "override_blocks", "concurrency", "tokens_per_request", "full_length_sequences",
                 "fewer_than_max_num_seqs", "blocks_per_sequence"),
}


def record(result: dict, check, evidence: dict) -> None:
    """`pass` and `fail` become a named check (`check(name, ok, detail)` is the boot check's own recorder). `not_checked` is listed in
    `evidence["not_checked"]` with its reason and does not change `passed`. The structured facts of `layers_loaded` and `kv_cache` are kept
    in `evidence["engine_facts"][name]`, so a consumer of the saved boot result reads the numbers and not only the sentence."""
    if result["state"] == "not_checked":
        evidence.setdefault("not_checked", []).append({"name": result["name"], "why": result["detail"]})
    else:
        check(result["name"], result["state"] == "pass", result["detail"])
    keys = ENGINE_FACT_KEYS.get(result["name"])
    facts = {k: result[k] for k in keys if k in result} if keys else {}
    if facts:
        evidence.setdefault("engine_facts", {})[result["name"]] = facts
