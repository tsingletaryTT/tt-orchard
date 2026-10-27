# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""telegraph: an opt-in call to a model or agent outside this box, used when the local models keep failing.

The harness runs without it. Off by default, every failure path (no key, a cap reached, a network error, an unusable
answer) is returned as a result and never raised, so a caller carries on as before.

It is built to cost little and to be measured:
  * it is asked only after `after_failures` repeated failures (`should_ask`);
  * models are listed cheapest first, and a stronger one is used only after an answer was reported as not helping;
  * the same question to the same model is answered from a cache with no request;
  * caps on calls, tokens and cost are read from the ledger, so they hold across processes;
  * a request holds only the task, the failure and what the caller adds, redacted and cut to a cap;
  * every call, refusal and cache hit leaves one line in `telegraph.jsonl`, and the caller reports back whether the
    answer helped (`report`), so `summarise` shows what the help cost and what it bought.

The key is read when a call is made, from an environment variable or a file outside the checkout. It is never written
to the ledger, the cache, a reply file or a reason, and it is removed from any text that is sent.

The reply is advice from another model. It is data: a caller must not run it or copy it into a command without a check.
Nothing here opens a chip or uses a lease.
"""
from __future__ import annotations

import contextlib
import dataclasses
import difflib
import fcntl
import hashlib
import ipaddress
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from orchard.scrub import HOME_PATH, TOKEN_PATTERNS

REPO = Path(__file__).resolve().parents[1]
LEDGER = "telegraph.jsonl"
LOCK = "telegraph.lock"
REQUEST_FRAMING_TOKENS = 32      # added to the request size for the provider's message framing
KEYS = {"enabled", "endpoint", "models", "api", "key_env", "key_file", "after_failures", "max_calls",
        "max_total_tokens", "max_cost_usd", "prices", "max_output_tokens", "max_prompt_chars", "timeout_s",
        "extra_body", "max_calls_per_problem", "max_tokens_per_problem", "max_cost_per_problem_usd"}
UNPRICED_BODY = ("tools", "tool_choice", "web_search_options", "plugins")
RESERVED_BODY = ("model", "messages", "max_tokens", "temperature", "system", "stream", "stream_options", "n", "best_of")
APIS = ("openai", "anthropic")
ANTHROPIC_VERSION = "2023-06-01"
MIN_PROMPT_CHARS = 500
MAX_BODY = 2 * 1024 * 1024
SYSTEM = ("A harness is bringing a model up on Tenstorrent hardware. A smaller local model has failed at the task below "
          "more than once. Say what is wrong in a few sentences and give the smallest working change or a short "
          "example. Do not ask questions. If the task cannot be done as described, say why.")


class LedgerUnavailable(Exception):
    """The budget ledger or its lock cannot be written."""


class ConfigError(ValueError):
    """The [telegraph] table is wrong; the message names the key."""


@dataclass(frozen=True)
class Config:
    enabled: bool = False
    endpoint: str = ""
    models: tuple = ()
    api: str = "openai"
    key_env: str | None = None
    key_file: Path | None = None
    after_failures: int = 2
    max_calls: int = 5
    max_total_tokens: int = 40000
    max_cost_usd: float | None = None
    prices: dict = field(default_factory=dict)          # model -> (usd per million input tokens, usd per million output)
    max_output_tokens: int = 1500
    max_prompt_chars: int = 12000
    timeout_s: int = 120
    extra_body: dict = field(default_factory=dict)      # added to every request body, for example to turn thinking off
    max_calls_per_problem: int = 3                      # one problem (the id the caller passes) may not use more than this
    max_tokens_per_problem: int | None = None
    max_cost_per_problem_usd: float | None = None


def _int(table: dict, key: str, default: int, lowest: int) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < lowest:
        raise ConfigError(f"telegraph.{key} must be a whole number of at least {lowest}")
    return value


def _http_ok(host: str) -> bool:
    """Plain http is for this machine and the private network: a name with no dot, or a loopback or private address."""
    if "." not in host and ":" not in host:
        # a machine name starts with a letter; digits and "0x" forms are legacy spellings of public IPv4 addresses
        return re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", host) is not None
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private


def _plain_json(value, where: str, depth: int = 0) -> None:
    """Raise ConfigError unless `value` is plain JSON data: strings, finite numbers, booleans, null, lists, and tables with string
    keys. TOML dates and times, other objects, and a table nested too deeply are refused by name."""
    if depth > 20:
        raise ConfigError(f"{where} is nested too deeply")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ConfigError(f"{where} holds a number that is not finite")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _plain_json(item, f"{where}[{i}]", depth + 1)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ConfigError(f"{where} has a key that is not a string")
            _plain_json(v, f"{where}.{k}", depth + 1)
        return
    raise ConfigError(f"{where} must be plain JSON data (strings, numbers, booleans, null, lists and tables), "
                      f"not {type(value).__name__}")


def _refuse_prompt_caching(value, where: str) -> None:
    """Prompt caching bills cached tokens at other rates and reports them apart from the input count, so telegraph cannot price
    it. A `cache_control` key anywhere in extra_body is refused by name."""
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "cache_control":
                raise ConfigError(f"{where} holds cache_control: prompt caching is not supported, because telegraph cannot "
                                  "price cached tokens")
            _refuse_prompt_caching(v, f"{where}.{k}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _refuse_prompt_caching(item, f"{where}[{i}]")


def load(table: dict | None, *, base: Path) -> Config:
    """The `[telegraph]` table as a Config. None, an empty table or `enabled = false` is off and is not checked further.
    A relative key_file is relative to `base`, the config file's directory."""
    if table is None:
        return Config()
    if not isinstance(table, dict):
        raise ConfigError("telegraph must be a table")
    enabled = table.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("telegraph.enabled must be true or false")
    if not enabled:
        return Config()
    for key in table:
        if key not in KEYS:
            near = difflib.get_close_matches(key, sorted(KEYS), n=1)
            raise ConfigError(f"telegraph: unknown key {key!r}" + (f"; did you mean {near[0]!r}?" if near else ""))
    endpoint = table.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        raise ConfigError("telegraph.endpoint is required (the base URL, for example https://host/v1)")
    parts = urllib.parse.urlsplit(endpoint)
    if parts.username or parts.password:
        raise ConfigError("telegraph.endpoint has credentials in the endpoint; use key_env or key_file")
    if parts.scheme == "https":
        pass
    elif parts.scheme == "http" and _http_ok(parts.hostname or ""):
        pass
    else:
        raise ConfigError("telegraph.endpoint must be https (http is allowed only on this machine or the private network)")
    models = table.get("models")
    if not isinstance(models, list):
        raise ConfigError("telegraph.models must be a list of model names, cheapest first")
    if not models or not all(isinstance(m, str) and m for m in models):
        raise ConfigError("telegraph.models needs at least one model name")
    if len(set(models)) != len(models):
        raise ConfigError("telegraph.models names a model twice")
    api = table.get("api", "openai")
    if api not in APIS:
        raise ConfigError(f"telegraph.api must be one of {', '.join(APIS)}")
    key_env, key_file = table.get("key_env"), table.get("key_file")
    for name, value in (("key_env", key_env), ("key_file", key_file)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f"telegraph.{name} must be a non-empty string")
    if key_env and key_file:
        raise ConfigError("telegraph: set only one of key_env and key_file")
    if key_file:
        key_file = Path(key_file)
        key_file = (key_file if key_file.is_absolute() else Path(base) / key_file).resolve()
        if key_file == REPO or REPO in key_file.parents:
            raise ConfigError("telegraph.key_file is inside the checkout; keep the key outside any repository")
    prices = {}
    raw_prices = table.get("prices", {})
    if not isinstance(raw_prices, dict):
        raise ConfigError("telegraph.prices must be a table of model names to [input, output] dollars per million tokens")
    for model, pair in raw_prices.items():
        if (not isinstance(pair, list) or len(pair) != 2 or
                not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0 for x in pair)):
            raise ConfigError(f"telegraph.prices.{model} must be [input, output] dollars per million tokens")
        prices[model] = (float(pair[0]), float(pair[1]))
    cap = table.get("max_cost_usd")
    if cap is not None:
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap <= 0:
            raise ConfigError("telegraph.max_cost_usd must be a finite number above zero")
        if not prices:
            raise ConfigError("telegraph.max_cost_usd needs telegraph.prices for the models it may call")
        cap = float(cap)
    per_cost = table.get("max_cost_per_problem_usd")
    if per_cost is not None:
        if (isinstance(per_cost, bool) or not isinstance(per_cost, (int, float)) or not math.isfinite(per_cost)
                or per_cost <= 0):
            raise ConfigError("telegraph.max_cost_per_problem_usd must be a finite number above zero")
        if not prices:
            raise ConfigError("telegraph.max_cost_per_problem_usd needs telegraph.prices for the models it may call")
        per_cost = float(per_cost)
    per_tokens = _int(table, "max_tokens_per_problem", 0, 1) if "max_tokens_per_problem" in table else None
    extra = table.get("extra_body", {})
    if not isinstance(extra, dict):
        raise ConfigError("telegraph.extra_body must be a table")
    _plain_json(extra, "telegraph.extra_body")
    _refuse_prompt_caching(extra, "telegraph.extra_body")
    for name in UNPRICED_BODY:
        if name in extra:
            raise ConfigError(f"telegraph.extra_body cannot set {name}: tools and web search can add charges per use that "
                              "telegraph does not account for")
    for name in RESERVED_BODY:
        if name in extra:
            raise ConfigError(f"telegraph.extra_body cannot set {name}; telegraph sets it")
    return Config(
        enabled=True, endpoint=endpoint.rstrip("/"), models=tuple(models), api=api, key_env=key_env or None,
        key_file=key_file or None, after_failures=_int(table, "after_failures", 2, 1),
        max_calls=_int(table, "max_calls", 5, 0), max_total_tokens=_int(table, "max_total_tokens", 40000, 0),
        max_cost_usd=cap, prices=prices, max_output_tokens=_int(table, "max_output_tokens", 1500, 1),
        max_prompt_chars=_int(table, "max_prompt_chars", 12000, MIN_PROMPT_CHARS),
        timeout_s=_int(table, "timeout_s", 120, 1), extra_body=dict(extra),
        max_calls_per_problem=_int(table, "max_calls_per_problem", 3, 1), max_tokens_per_problem=per_tokens,
        max_cost_per_problem_usd=per_cost)


# ----------------------------------------------------------------------------------------------------- redaction

_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}")
_SK = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")


_PEM = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE KEY-----|\Z)", re.DOTALL)


def _usage_pair(u, name_in: str, name_out: str) -> tuple | None:
    """(input, output) token counts, or None when either is missing, is not a whole number (a bool, a float, a string) or is
    negative: a count the provider got wrong cannot be added to a budget, so the call counts as unknown usage."""
    if not isinstance(u, dict):
        return None
    a, b = u.get(name_in), u.get(name_out)
    for v in (a, b):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            return None
    return a, b


def redact(text: str, *, hostname: str = "", home: str = "", secrets: tuple = ()) -> str:
    """`text` without the key, token shapes, private key markers, the host name and the home directory."""
    for secret in secrets:
        if secret:
            # a server that echoes the key inside a JSON body (or a Python repr) writes quotes and non-ASCII characters
            # escaped, so each spelling is removed, longest first
            spellings = {secret, json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1], repr(secret)[1:-1]}
            for spelling in sorted(spellings, key=len, reverse=True):
                if spelling:
                    text = text.replace(spelling, "[redacted]")
    text = _PEM.sub("[redacted private key]", text)
    for _what, pattern in TOKEN_PATTERNS:
        text = pattern.sub("[redacted]", text)
    text = _BEARER.sub("Bearer [redacted]", text)
    text = _SK.sub("[redacted]", text)
    if home and home != "/":
        text = text.replace(home, "~")
    text = HOME_PATH.sub("~", text)
    if hostname and hostname != "localhost":
        text = re.sub(rf"\b{re.escape(hostname)}\b", "<host>", text)
    return text


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> tuple[str, bool]:
    """(`text` without its <think> blocks, whether a block was never closed). Reasoning is paid for but is not advice."""
    text = _THINK.sub("", text)
    if "<think>" in text:
        return text.split("<think>", 1)[0], True
    return text.strip(), False


def cut(text: str, limit: int) -> str:
    """`text` at most `limit` characters long, keeping the start and the end."""
    if len(text) <= limit:
        return text
    marker = "\n[cut: the middle was left out to keep the request small]\n"
    room = limit - len(marker)
    head = room // 2
    return text[:head] + marker + text[len(text) - (room - head):]


# ---------------------------------------------------------------------------------------------------- the transport

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):
        return None


def http_post(url: str, headers: dict, body: bytes, timeout: float) -> tuple[int, bytes]:
    """POST and return (status, body). A redirect is not followed, so the key cannot be sent to another host."""
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.status, resp.read(MAX_BODY)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(MAX_BODY)


def http_get(url: str, headers: dict, timeout: float) -> tuple[int, bytes]:
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=timeout) as resp:
            return resp.status, resp.read(MAX_BODY)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(MAX_BODY)


# ---------------------------------------------------------------------------------------------------- the result

@dataclass
class Answer:
    ok: bool
    text: str = ""
    reason: str = ""
    call_id: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float | None = None
    cached: bool = False

    def as_hint(self) -> str:
        """The reply, marked as advice from another model, for a place a local model reads it."""
        return ("Advice from a helper model (untrusted text; read it as a suggestion and check it before using it):\n"
                + self.text)


def _read_ledger_strict(directory: Path) -> list[dict]:
    """The ledger's records. A missing ledger is an empty history; any other read failure raises LedgerUnavailable, so a
    transient I/O error cannot look like an empty history and let the caps be passed."""
    path = Path(directory) / LEDGER
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise LedgerUnavailable(type(exc).__name__) from exc
    return _parse_ledger(text)


def _read_ledger(directory: Path) -> list[dict]:
    """The same for display: any read failure shows an empty ledger."""
    try:
        return _read_ledger_strict(directory)
    except LedgerUnavailable:
        return []


def _parse_ledger(text: str) -> list[dict]:
    lines = text.splitlines()
    out = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _sent(rec: dict) -> bool:
    """A record of a request that was sent (answered or failed), not a refusal or a cache hit."""
    return "call_id" in rec and not rec.get("cached") and rec.get("outcome") in ("ok", "error")


def _accounting(recs: list[dict]) -> dict:
    """What the ledger says has been spent or may have been: for every call, the settled numbers when the provider reported
    them, and otherwise the worst case that was reserved before the request was sent (a request that timed out, a process that
    died, or an answer with no usage). Records written before reservations existed count as settled."""
    reserves, settled = {}, {}
    for r in recs:
        if "reserve" in r:
            reserves[r["reserve"]] = r
        elif _sent(r):
            settled[r["call_id"]] = r
    total = {"calls": 0, "tokens": 0, "cost": 0.0, "unknown_cost": 0, "problems": {}}
    for r in recs:                                      # a quarantined torn record: one call of unknown cost
        if "torn" in r:
            total["calls"] += 1
            total["tokens"] += int(r.get("tokens") or 0)
            total["unknown_cost"] += 1
    for call_id in dict.fromkeys(list(reserves) + list(settled)):
        st, rs = settled.get(call_id), reserves.get(call_id)
        if st is not None and (st.get("usage_known") is not False or rs is None):
            tokens = int(st.get("tokens_in") or 0) + int(st.get("tokens_out") or 0)
            source = st.get("cost_usd")
        else:
            tokens = int(rs.get("tokens") or 0)
            source = rs.get("cost_usd")
        known = isinstance(source, (int, float)) and not isinstance(source, bool)
        cost = source if known else 0.0                    # an unknown cost is counted apart; it is never taken as zero
        name = (st or rs).get("problem")
        here = total["problems"].setdefault(name, {"calls": 0, "tokens": 0, "cost": 0.0, "unknown_cost": 0})
        for t in (total, here):
            t["calls"] += 1
            t["tokens"] += tokens
            t["cost"] += cost
            t["unknown_cost"] += 0 if known else 1
    return total


def summarise(directory) -> dict:
    """What the help cost and bought: counts, tokens, cost (known calls only), and per model and problem."""
    recs = _read_ledger(Path(directory))
    calls = [r for r in recs if _sent(r)]
    verdict = {}
    for r in recs:
        if "report" in r:
            verdict[r["report"]] = bool(r.get("helped"))
    ok = [r for r in calls if r["outcome"] == "ok"]
    by_model, by_problem = {}, {}
    for r in calls:
        by_model.setdefault(r.get("model"), {"calls": 0})["calls"] += 1
        p = by_problem.setdefault(r.get("problem"), {"calls": 0, "helped": 0})
        p["calls"] += 1
        p["helped"] += 1 if verdict.get(r["call_id"]) else 0
    helped = sum(1 for r in ok if verdict.get(r["call_id"]) is True)
    not_helped = sum(1 for r in ok if verdict.get(r["call_id"]) is False)
    reserved = {r["reserve"] for r in recs if "reserve" in r}
    return {"calls": len(calls), "ok": len(ok), "helped": helped, "not_helped": not_helped,
            "unsettled": len(reserved - {r["call_id"] for r in calls}),
            "unknown_usage_calls": sum(1 for r in calls if r.get("usage_known") is False),
            "unreported": sum(1 for r in ok if r["call_id"] not in verdict),
            "cache_hits": sum(1 for r in recs if "call_id" in r and r.get("cached")),
            "refused": sum(1 for r in recs if "call_id" in r and r.get("outcome") in ("budget", "no-key")),
            "tokens_in": sum(int(r.get("tokens_in") or 0) for r in calls),
            "tokens_out": sum(int(r.get("tokens_out") or 0) for r in calls),
            "cost_usd": sum(r["cost_usd"] for r in calls if isinstance(r.get("cost_usd"), (int, float))),
            "cost_unknown_calls": sum(1 for r in calls if r.get("cost_usd") is None),
            "latency_s": sum(float(r.get("latency_s") or 0) for r in calls),
            "by_model": by_model, "by_problem": by_problem}


class Telegraph:
    """One configured helper. `directory` holds the ledger, the cache and the reply files (mode 0600)."""

    def __init__(self, config: Config, directory, *, transport: Callable | None = None, environ: dict | None = None,
                 clock: Callable[[], float] = time.time, hostname: str | None = None, home: str | None = None):
        self.cfg = config
        self.dir = Path(directory)
        self.transport = transport or http_post
        self.environ = os.environ if environ is None else environ
        self.clock = clock
        self.hostname = socket.gethostname() if hostname is None else hostname
        self.home = str(Path.home()) if home is None else home

    # ------------------------------------------------------------------------------------------- policy

    def should_ask(self, consecutive_failures: int) -> bool:
        """True when it is enabled and the caller has failed at least `after_failures` times in a row."""
        return self.cfg.enabled and consecutive_failures >= self.cfg.after_failures

    # ------------------------------------------------------------------------------------------- the ask

    def ask(self, problem: str, task: str, failure: str) -> Answer:
        """Ask for help with `problem` (a short id the caller chooses, used for the ladder and the summary).
        Always returns an Answer; `ok` False carries a reason and the harness goes on without help."""
        if not self.cfg.enabled:
            return Answer(False, reason="telegraph is not enabled")
        key, key_problem = self._key()
        secrets = self._secrets(key)
        clean = lambda t: redact(str(t), hostname=self.hostname, home=self.home, secrets=secrets)   # noqa: E731
        problem = clean(problem)[:80]
        prompt = cut(f"Problem: {problem}\n\nTask:\n{clean(task)}\n\nWhat failed:\n{clean(failure)}",
                     self.cfg.max_prompt_chars - len(SYSTEM))
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        try:
            recs = _read_ledger_strict(self.dir)
        except LedgerUnavailable as exc:
            return Answer(False, reason=f"the budget ledger cannot be written or read ({exc}), so no outside call was made")
        model = self._choose_model(problem, recs)
        url, _headers, body = self._request(model, prompt, key)
        identity = json.dumps({"url": url, "body": body}, sort_keys=True)        # no credential in the identity
        cache = self.dir / "cache" / (hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".json")
        hit = self._cached(cache)
        base = {"call_id": uuid.uuid4().hex[:12], "problem": problem, "model": model,
                "prompt_chars": len(prompt), "prompt_sha256": sha}
        if hit is not None:
            recorded = self._append({**base, "ts": self.clock(), "outcome": "ok", "cached": True,
                                     "origin": hit.get("call_id"), "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
                                     "cost_usd": 0.0, "reply_file": hit["reply_file"]})
            return Answer(True, hit["text"], call_id=base["call_id"], model=model, cached=True, cost_usd=0.0,
                          reason="" if recorded else ("the answer came from the cache but the ledger could not record "
                                                      "this use, so it cannot be reported"))
        if key_problem:
            return self._refuse(base, "no-key", key_problem)
        worst_in = len(json.dumps(body).encode("utf-8")) + REQUEST_FRAMING_TOKENS     # a token is at least one byte
        price = self.cfg.prices.get(model)
        worst = {"tokens": worst_in + self.cfg.max_output_tokens,
                 "cost": (worst_in * price[0] + self.cfg.max_output_tokens * price[1]) / 1e6 if price else None}
        try:
            with self._ledger_lock():
                self._quarantine_torn_tail()
                acct = _accounting(_read_ledger_strict(self.dir))           # fresh: another process may have spent since
                why = self._over_budget(model, worst, acct, problem)
                if why:
                    return self._refuse(base, "budget", why, held=True)
                if not self._append({"reserve": base["call_id"], "problem": problem, "model": model,
                                     "tokens": worst["tokens"], "cost_usd": worst["cost"]}, held=True):
                    raise LedgerUnavailable("the reservation could not be written")
        except LedgerUnavailable as exc:
            return Answer(False, reason=f"the budget ledger cannot be written or read ({exc}), so no outside call was made",
                          call_id=base["call_id"], model=model)
        return self._send(base, model, prompt, key, secrets, cache)

    def report(self, call_id: str, helped: bool, note: str = "") -> None:
        """Say whether the answer to `call_id` helped. A helped answer keeps the cheap model for the next question; one
        that did not help moves the next question to the next model."""
        recs = {r["call_id"]: r for r in _read_ledger(self.dir) if "call_id" in r}
        rec = recs.get(call_id)
        if rec is not None and rec.get("cached"):
            call_id = rec.get("origin") or ""                    # a cache hit is reported against the call that made the answer
            rec = recs.get(call_id)
        if rec is None:
            raise KeyError(call_id)
        key, _ = self._key()
        if not self._append({"report": call_id, "helped": bool(helped), "ts": self.clock(),
                             "note": redact(note, hostname=self.hostname, home=self.home,
                                            secrets=self._secrets(key))[:300]}):
            raise LedgerUnavailable("the verdict could not be written")

    # ------------------------------------------------------------------------------------------- parts

    def _secrets(self, key: str | None) -> tuple:
        """What redaction must remove: the key in use, and a key that was refused for its characters (it is a credential too,
        and a prompt or an answer could still contain it)."""
        return tuple(s for s in (key, getattr(self, "_refused_key", None)) if s)

    def _key(self) -> tuple[str | None, str | None]:
        """(key, problem). A key with a newline, a control character or a character outside Latin-1 is refused here, before
        any use: it cannot be sent in a header, and an HTTP library would show it in an escaped form that redaction misses."""
        if self.cfg.key_env:
            value = str(self.environ.get(self.cfg.key_env, "")).strip()
            if not value:
                return None, f"the key variable {self.cfg.key_env} is not set"
        elif self.cfg.key_file:
            try:
                value = self.cfg.key_file.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                return None, "cannot read the key file"
            if not value:
                return None, "the key file is empty"
        else:
            return None, None
        self._refused_key = None
        if any(ord(c) < 0x20 or ord(c) == 0x7F or ord(c) > 0xFF for c in value):
            self._refused_key = value
            return None, ("the key has a character that cannot be sent in an HTTP header (a newline, a control character or a "
                          "character outside Latin-1); check the key source")
        return value, None

    def _choose_model(self, problem: str, recs: list[dict]) -> str:
        """The model for the next question on `problem`: each answered call counts once, by its latest verdict. A verdict of
        not helped moves one step up the ladder and a verdict of helped goes back to the first model."""
        verdict = {}
        for r in recs:
            if "report" in r:
                verdict[r["report"]] = bool(r.get("helped"))
        steps = 0
        for r in recs:
            if _sent(r) and r.get("problem") == str(problem)[:80] and r["call_id"] in verdict:
                steps = 0 if verdict[r["call_id"]] else steps + 1
        return self.cfg.models[min(steps, len(self.cfg.models) - 1)]

    def _over_budget(self, model: str, worst: dict, acct: dict, problem: str) -> str | None:
        """Why this call may not be sent, or None. `worst` is what the call could cost at most (its input bounded by the
        request's size in bytes, its output by max_output_tokens); `acct` is `_accounting` of the ledger."""
        cfg = self.cfg
        here = acct["problems"].get(problem, {"calls": 0, "tokens": 0, "cost": 0.0, "unknown_cost": 0})
        name = f"problem {problem}"
        if here["calls"] >= cfg.max_calls_per_problem:
            return f"the call cap for {name} is reached ({here['calls']} of {cfg.max_calls_per_problem})"
        if cfg.max_tokens_per_problem is not None and here["tokens"] >= cfg.max_tokens_per_problem:
            return f"the token cap for {name} is reached ({here['tokens']} of {cfg.max_tokens_per_problem})"
        if cfg.max_cost_per_problem_usd is not None:
            if here["unknown_cost"]:
                return (f"{here['unknown_cost']} earlier calls for {name} have no known cost (they were made without a price), "
                        "so the cost cap cannot be checked; price them or use a new ledger directory")
            if worst["cost"] is None:
                return f"the cost cap for {name} needs a price, and there is no price for {model}"
            if here["cost"] + worst["cost"] > cfg.max_cost_per_problem_usd:
                return (f"the cost cap for {name} would be passed (spent {here['cost']:.6f}, this call could cost up "
                        f"to {worst['cost']:.6f}, cap {cfg.max_cost_per_problem_usd})")
        if acct["calls"] >= cfg.max_calls:
            return f"the call cap is reached ({acct['calls']} of {cfg.max_calls})"
        if acct["tokens"] >= cfg.max_total_tokens:
            return f"the token cap is reached ({acct['tokens']} of {cfg.max_total_tokens})"
        if cfg.max_cost_usd is not None:
            if acct["unknown_cost"]:
                return (f"{acct['unknown_cost']} earlier calls have no known cost (they were made without a price), so the "
                        "cost cap cannot be checked; price them or use a new ledger directory")
            if worst["cost"] is None:
                return f"the cost cap needs a price, and there is no price for {model}"
            if acct["cost"] + worst["cost"] > cfg.max_cost_usd:
                return (f"the cost cap would be passed (spent {acct['cost']:.6f}, this call could cost up to "
                        f"{worst['cost']:.6f}, cap {cfg.max_cost_usd})")
        return None

    def _clean_tree(self, value, key: str | None):
        """`value` with every string redacted (the key, token shapes, home paths, the host name), through dicts and lists."""
        if isinstance(value, str):
            return redact(value, hostname=self.hostname, home=self.home, secrets=self._secrets(key))
        if isinstance(value, dict):
            return {(self._clean_tree(k, key) if isinstance(k, str) else k): self._clean_tree(v, key)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self._clean_tree(v, key) for v in value]
        return value

    def _request(self, model: str, prompt: str, key: str | None) -> tuple[str, dict, dict]:
        cfg = self.cfg
        extra = self._clean_tree(cfg.extra_body, key)
        headers = {"Content-Type": "application/json"}
        if cfg.api == "anthropic":
            base = cfg.endpoint[:-3] if cfg.endpoint.endswith("/v1") else cfg.endpoint
            if key:
                headers["x-api-key"] = key
            headers["anthropic-version"] = ANTHROPIC_VERSION
            body = {**extra, "model": model, "max_tokens": cfg.max_output_tokens, "temperature": 0,
                    "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]}
            return base + "/v1/messages", headers, body
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {**extra, "model": model, "max_tokens": cfg.max_output_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
        return cfg.endpoint + "/chat/completions", headers, body

    def _send(self, base: dict, model: str, prompt: str, key, secrets, cache: Path) -> Answer:
        url, headers, body = self._request(model, prompt, key)
        t0 = self.clock()
        text, usage, error = "", None, ""
        usage_known = False                      # a timeout or a server error may still have been processed and billed
        try:
            status, raw = self.transport(url, headers, json.dumps(body).encode(), self.cfg.timeout_s)
            if status != 200:
                error = f"HTTP {status}: " + raw.decode("utf-8", "replace")
                usage_known = 400 <= status < 500 and status != 408    # refused before any processing
            else:
                text, usage, error = self._parse(raw)
                usage_known = usage is not None
        except Exception as exc:                                    # noqa: BLE001 - nothing may reach the harness
            error = f"{type(exc).__name__}: {exc}"
        t1 = self.clock()
        clean = lambda t: redact(str(t), hostname=self.hostname, home=self.home, secrets=secrets)   # noqa: E731
        rec = {**base, "ts": t0, "cached": False, "latency_s": round(t1 - t0, 3)}
        if error:
            reason = clean(error)[:200]
            tin, tout = usage if usage else (0, 0)
            price = self.cfg.prices.get(model)
            if usage:
                cost = round(tin * price[0] / 1e6 + tout * price[1] / 1e6, 9) if price else None
            else:
                cost = 0.0 if usage_known else None          # refused before processing: free. Otherwise the cost is unknown
            self._append({**rec, "outcome": "error", "reason": reason, "tokens_in": tin, "tokens_out": tout,
                          "cost_usd": cost, "usage_known": usage_known or usage is not None})
            return Answer(False, reason=reason, call_id=base["call_id"], model=model, latency_s=t1 - t0,
                          tokens_in=tin, tokens_out=tout, cost_usd=cost)
        text = clean(text)
        estimated = usage is None
        tin = usage[0] if usage else (len(prompt) + len(SYSTEM)) // 3
        tout = usage[1] if usage else len(text) // 3
        price = self.cfg.prices.get(model)
        cost = None if estimated else (round(tin * price[0] / 1e6 + tout * price[1] / 1e6, 9) if price else None)
        reply_file = f"replies/{base['call_id']}.txt"
        self._write(self.dir / reply_file, text)
        recorded = self._append({**rec, "outcome": "ok", "tokens_in": tin, "tokens_out": tout, "cost_usd": cost,
                                 "usage_known": not estimated, "reply_file": reply_file,
                                 **({"usage_estimated": True} if estimated else {})})
        if recorded:                                    # the cache is published only once the call can be reported
            self._write(cache, json.dumps({"text": text, "reply_file": reply_file, "call_id": base["call_id"]}))
        return Answer(True, text, call_id=base["call_id"], model=model, tokens_in=tin, tokens_out=tout,
                      latency_s=t1 - t0, cost_usd=cost,
                      reason="" if recorded else ("the answer arrived but the ledger could not record this call, so it is "
                                                  "not cached and cannot be reported"))

    def _parse(self, raw: bytes) -> tuple[str, tuple | None, str]:
        """(text, usage, error). The usage is returned with an error too, because a failed answer was still paid for."""
        usage = None
        try:
            obj = json.loads(raw)
            if self.cfg.api == "anthropic":
                text = "".join(b.get("text", "") for b in obj.get("content", []) if isinstance(b, dict))
                u = obj.get("usage") or {}
                usage = _usage_pair(u, "input_tokens", "output_tokens")
                if any(isinstance(u.get(k), (int, float)) and u.get(k) > 0
                       for k in ("cache_creation_input_tokens", "cache_read_input_tokens")):
                    usage = None               # cached input is reported apart and billed at other rates: unknown usage
                finish = obj.get("stop_reason")
            else:
                choice = obj["choices"][0]
                text = choice["message"].get("content")
                u = obj.get("usage") or {}
                usage = _usage_pair(u, "prompt_tokens", "completion_tokens")
                finish = choice.get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            return "", None, f"the answer could not be read ({type(exc).__name__})"
        if not isinstance(text, str) or not text.strip():
            return "", usage, f"the answer was empty (finish_reason {finish})"
        text, unclosed = strip_thinking(text)
        if unclosed:
            return "", usage, ("the answer was cut off inside its reasoning (finish_reason "
                               f"{finish}); raise max_output_tokens or turn thinking off with extra_body, for example "
                               "extra_body = { chat_template_kwargs = { enable_thinking = false } }")
        if not text.strip():
            return "", usage, "the answer was empty (only reasoning, no answer)"
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return "", usage, "the answer contains characters that cannot be stored (a lone surrogate)"
        return text, usage, ""

    def _refuse(self, base: dict, outcome: str, reason: str, held: bool = False) -> Answer:
        self._append({**base, "ts": self.clock(), "outcome": outcome, "reason": reason, "cached": False,
                      "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}, held=held)
        return Answer(False, reason=reason, call_id=base["call_id"], model=base["model"])

    def _cached(self, path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        ok = isinstance(data, dict) and isinstance(data.get("text"), str) and data.get("reply_file")
        return data if ok and "<think>" not in data["text"] else None

    # ------------------------------------------------------------------------------------------- files (private)

    @contextlib.contextmanager
    def _ledger_lock(self):
        """An exclusive lock on the ledger directory, held while the budget is checked and a reservation is written, so two
        processes cannot both spend the same allowance. Raises LedgerUnavailable when the directory or the lock cannot be used
        (a lock that cannot be taken included); the descriptor is always closed."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self.dir / LOCK, os.O_WRONLY | os.O_CREAT, 0o600)
        except OSError as exc:
            raise LedgerUnavailable(type(exc).__name__) from exc
        try:
            try:
                os.fchmod(fd, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)
            except OSError as exc:
                raise LedgerUnavailable(type(exc).__name__) from exc
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)          # closing the descriptor releases the lock if this fails
            except OSError:
                pass
            os.close(fd)

    def _quarantine_torn_tail(self) -> None:
        """Called with the lock held. A last line with no line break is a write that was cut short. It moves to a private
        sidecar file, the ledger is cut back to its last whole line, and a note record counts the fragment as one call of
        unknown cost until a person reconciles it. Raises LedgerUnavailable when this cannot be done."""
        path = self.dir / LEDGER
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            data = b""
        except OSError as exc:
            raise LedgerUnavailable(type(exc).__name__) from exc
        if data and not data.endswith(b"\n"):
            cut = data.rfind(b"\n") + 1
            name = f"{LEDGER}.torn-{uuid.uuid4().hex[:8]}"
            fragment = data[cut:]
            try:
                fd = os.open(self.dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    if os.write(fd, fragment) != len(fragment):
                        raise OSError("short write of the torn tail")
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.truncate(path, cut)                   # only after the whole fragment is safe in the sidecar
            except OSError as exc:
                try:
                    (self.dir / name).unlink()           # the ledger is intact, so the partial sidecar must not be counted
                except OSError:
                    pass
                raise LedgerUnavailable(type(exc).__name__) from exc
        # Every sidecar gets exactly one note. A crash between the cut and the note leaves a sidecar with no note, and the
        # next call writes it here, so recovery can be replayed.
        noted = {r.get("torn") for r in _read_ledger_strict(self.dir) if "torn" in r}
        for sidecar in sorted(self.dir.glob(LEDGER + ".torn-*")):
            if sidecar.name not in noted:
                if not self._append({"torn": sidecar.name, "tokens": self.cfg.max_output_tokens}, held=True):
                    raise LedgerUnavailable("the note for a torn record could not be written")

    def _append(self, rec: dict, *, held: bool = False) -> bool:
        """Append one record as one write, under the ledger lock (`held` says the caller already holds it, since the lock
        is not re-entrant). Recovery of a torn tail also runs under this lock, so no record can land between its read and its
        cut. A last line that was cut short gets a line break first, so it cannot swallow this record. True when written."""
        if not held:
            try:
                with self._ledger_lock():
                    return self._append(rec, held=True)
            except LedgerUnavailable:
                return False
        try:
            self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self.dir / LEDGER, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.fchmod(fd, 0o600)
                size = os.fstat(fd).st_size
                torn = size > 0 and os.pread(fd, 1, size - 1) != b"\n"
                data = (b"\n" if torn else b"") + (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8")
                if os.write(fd, data) != len(data):
                    return False                     # a short write leaves a record nobody can read
                os.fsync(fd)                         # the record must survive a crash before a request depends on it
            finally:
                os.close(fd)
            return True
        except OSError:
            return False

    def _write(self, path: Path, text: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.fchmod(fd, 0o600)
                os.write(fd, text.encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            pass
