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

import dataclasses
import difflib
import hashlib
import ipaddress
import json
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
KEYS = {"enabled", "endpoint", "models", "api", "key_env", "key_file", "after_failures", "max_calls",
        "max_total_tokens", "max_cost_usd", "prices", "max_output_tokens", "max_prompt_chars", "timeout_s",
        "extra_body"}
RESERVED_BODY = ("model", "messages", "max_tokens", "temperature", "system", "stream")
APIS = ("openai", "anthropic")
ANTHROPIC_VERSION = "2023-06-01"
MIN_PROMPT_CHARS = 500
MAX_BODY = 2 * 1024 * 1024
SYSTEM = ("A harness is bringing a model up on Tenstorrent hardware. A smaller local model has failed at the task below "
          "more than once. Say what is wrong in a few sentences and give the smallest working change or a short "
          "example. Do not ask questions. If the task cannot be done as described, say why.")


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


def _int(table: dict, key: str, default: int, lowest: int) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < lowest:
        raise ConfigError(f"telegraph.{key} must be a whole number of at least {lowest}")
    return value


def _http_ok(host: str) -> bool:
    """Plain http is for this machine and the private network: a name with no dot, or a loopback or private address."""
    if host in ("localhost",) or "." not in host and ":" not in host:
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private


def load(table: dict | None, *, base: Path) -> Config:
    """The `[telegraph]` table as a Config. None, an empty table or `enabled = false` is off and is not checked further.
    A relative key_file is relative to `base`, the config file's directory."""
    if not table or not table.get("enabled", False):
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
    if key_env and key_file:
        raise ConfigError("telegraph: set only one of key_env and key_file")
    if key_file:
        key_file = Path(key_file)
        key_file = (key_file if key_file.is_absolute() else Path(base) / key_file).resolve()
        if key_file == REPO or REPO in key_file.parents:
            raise ConfigError("telegraph.key_file is inside the checkout; keep the key outside any repository")
    prices = {}
    for model, pair in (table.get("prices") or {}).items():
        if (not isinstance(pair, list) or len(pair) != 2 or
                not all(isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0 for x in pair)):
            raise ConfigError(f"telegraph.prices.{model} must be [input, output] dollars per million tokens")
        prices[model] = (float(pair[0]), float(pair[1]))
    cap = table.get("max_cost_usd")
    if cap is not None:
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap <= 0:
            raise ConfigError("telegraph.max_cost_usd must be a number above zero")
        if not prices:
            raise ConfigError("telegraph.max_cost_usd needs telegraph.prices for the models it may call")
        cap = float(cap)
    extra = table.get("extra_body", {})
    if not isinstance(extra, dict):
        raise ConfigError("telegraph.extra_body must be a table")
    for name in RESERVED_BODY:
        if name in extra:
            raise ConfigError(f"telegraph.extra_body cannot set {name}; telegraph sets it")
    return Config(
        enabled=True, endpoint=endpoint.rstrip("/"), models=tuple(models), api=api, key_env=key_env or None,
        key_file=key_file or None, after_failures=_int(table, "after_failures", 2, 1),
        max_calls=_int(table, "max_calls", 5, 0), max_total_tokens=_int(table, "max_total_tokens", 40000, 0),
        max_cost_usd=cap, prices=prices, max_output_tokens=_int(table, "max_output_tokens", 1500, 1),
        max_prompt_chars=_int(table, "max_prompt_chars", 12000, MIN_PROMPT_CHARS),
        timeout_s=_int(table, "timeout_s", 120, 1), extra_body=dict(extra))


# ----------------------------------------------------------------------------------------------------- redaction

_BEARER = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}")
_SK = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")


def redact(text: str, *, hostname: str = "", home: str = "", secrets: tuple = ()) -> str:
    """`text` without the key, token shapes, private key markers, the host name and the home directory."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
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


def _read_ledger(directory: Path) -> list[dict]:
    path = Path(directory) / LEDGER
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
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
    return {"calls": len(calls), "ok": len(ok), "helped": helped, "not_helped": not_helped,
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
        secrets = (key,) if key else ()
        clean = lambda t: redact(str(t), hostname=self.hostname, home=self.home, secrets=secrets)   # noqa: E731
        prompt = cut(f"Problem: {clean(problem)}\n\nTask:\n{clean(task)}\n\nWhat failed:\n{clean(failure)}",
                     self.cfg.max_prompt_chars - len(SYSTEM))
        sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        recs = _read_ledger(self.dir)
        model = self._choose_model(problem, recs)
        cache = self.dir / "cache" / (hashlib.sha256(f"{model}\0{sha}".encode()).hexdigest() + ".json")
        hit = self._cached(cache)
        base = {"call_id": uuid.uuid4().hex[:12], "problem": str(problem)[:80], "model": model,
                "prompt_chars": len(prompt), "prompt_sha256": sha}
        if hit is not None:
            self._append({**base, "ts": self.clock(), "outcome": "ok", "cached": True, "tokens_in": 0, "tokens_out": 0,
                          "latency_s": 0.0, "cost_usd": 0.0, "reply_file": hit["reply_file"]})
            return Answer(True, hit["text"], call_id=base["call_id"], model=model, cached=True, cost_usd=0.0)
        if key_problem:
            return self._refuse(base, "no-key", key_problem)
        why = self._over_budget(model, prompt, summarise(self.dir))
        if why:
            return self._refuse(base, "budget", why)
        return self._send(base, model, prompt, key, secrets, cache)

    def report(self, call_id: str, helped: bool, note: str = "") -> None:
        """Say whether the answer to `call_id` helped. A helped answer keeps the cheap model for the next question; one
        that did not help moves the next question to the next model."""
        if not any(r.get("call_id") == call_id and not r.get("cached") for r in _read_ledger(self.dir)):
            raise KeyError(call_id)
        self._append({"report": call_id, "helped": bool(helped), "ts": self.clock(),
                      "note": redact(note, hostname=self.hostname, home=self.home)[:300]})

    # ------------------------------------------------------------------------------------------- parts

    def _key(self) -> tuple[str | None, str | None]:
        if self.cfg.key_env:
            value = str(self.environ.get(self.cfg.key_env, "")).strip()
            return (value, None) if value else (None, f"the key variable {self.cfg.key_env} is not set")
        if self.cfg.key_file:
            try:
                value = self.cfg.key_file.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                return None, "cannot read the key file"
            return (value, None) if value else (None, "the key file is empty")
        return None, None

    def _choose_model(self, problem: str, recs: list[dict]) -> str:
        calls = {r["call_id"]: r for r in recs if "call_id" in r}
        steps = 0
        for r in recs:
            if "report" not in r or calls.get(r["report"], {}).get("problem") != str(problem)[:80]:
                continue
            steps = 0 if r.get("helped") else steps + 1
        return self.cfg.models[min(steps, len(self.cfg.models) - 1)]

    def _over_budget(self, model: str, prompt: str, used: dict) -> str | None:
        cfg = self.cfg
        if used["calls"] >= cfg.max_calls:
            return f"the call cap is reached ({used['calls']} of {cfg.max_calls})"
        if used["tokens_in"] + used["tokens_out"] >= cfg.max_total_tokens:
            return f"the token cap is reached ({used['tokens_in'] + used['tokens_out']} of {cfg.max_total_tokens})"
        if cfg.max_cost_usd is not None:
            price = cfg.prices.get(model)
            if price is None:
                return f"the cost cap needs a price, and there is no price for {model}"
            worst = (len(prompt) + len(SYSTEM)) / 3 * price[0] / 1e6 + cfg.max_output_tokens * price[1] / 1e6
            if used["cost_usd"] + worst > cfg.max_cost_usd:
                return (f"the cost cap would be passed (spent {used['cost_usd']:.6f}, this call could cost up to "
                        f"{worst:.6f}, cap {cfg.max_cost_usd})")
        return None

    def _request(self, model: str, prompt: str, key: str | None) -> tuple[str, dict, dict]:
        cfg = self.cfg
        headers = {"Content-Type": "application/json"}
        if cfg.api == "anthropic":
            base = cfg.endpoint[:-3] if cfg.endpoint.endswith("/v1") else cfg.endpoint
            if key:
                headers["x-api-key"] = key
            headers["anthropic-version"] = ANTHROPIC_VERSION
            body = {**cfg.extra_body, "model": model, "max_tokens": cfg.max_output_tokens, "temperature": 0,
                    "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]}
            return base + "/v1/messages", headers, body
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {**cfg.extra_body, "model": model, "max_tokens": cfg.max_output_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
        return cfg.endpoint + "/chat/completions", headers, body

    def _send(self, base: dict, model: str, prompt: str, key, secrets, cache: Path) -> Answer:
        url, headers, body = self._request(model, prompt, key)
        t0 = self.clock()
        text, usage, error = "", None, ""
        try:
            status, raw = self.transport(url, headers, json.dumps(body).encode(), self.cfg.timeout_s)
            if status != 200:
                error = f"HTTP {status}: " + raw[:160].decode("utf-8", "replace")
            else:
                text, usage, error = self._parse(raw)
        except Exception as exc:                                    # noqa: BLE001 - nothing may reach the harness
            error = f"{type(exc).__name__}: {exc}"
        t1 = self.clock()
        clean = lambda t: redact(str(t), hostname=self.hostname, home=self.home, secrets=secrets)   # noqa: E731
        rec = {**base, "ts": t0, "cached": False, "latency_s": round(t1 - t0, 3)}
        if error:
            reason = clean(error)[:200]
            tin, tout = usage if usage else (0, 0)
            price = self.cfg.prices.get(model)
            cost = round(tin * price[0] / 1e6 + tout * price[1] / 1e6, 9) if price and usage else None
            self._append({**rec, "outcome": "error", "reason": reason, "tokens_in": tin, "tokens_out": tout,
                          "cost_usd": cost})
            return Answer(False, reason=reason, call_id=base["call_id"], model=model, latency_s=t1 - t0)
        text = clean(text)
        estimated = usage is None
        tin = usage[0] if usage else (len(prompt) + len(SYSTEM)) // 3
        tout = usage[1] if usage else len(text) // 3
        price = self.cfg.prices.get(model)
        cost = round(tin * price[0] / 1e6 + tout * price[1] / 1e6, 9) if price else None
        reply_file = f"replies/{base['call_id']}.txt"
        self._write(self.dir / reply_file, text)
        self._write(cache, json.dumps({"text": text, "reply_file": reply_file}))
        self._append({**rec, "outcome": "ok", "tokens_in": tin, "tokens_out": tout, "cost_usd": cost,
                      "reply_file": reply_file, **({"usage_estimated": True} if estimated else {})})
        return Answer(True, text, call_id=base["call_id"], model=model, tokens_in=tin, tokens_out=tout,
                      latency_s=t1 - t0, cost_usd=cost)

    def _parse(self, raw: bytes) -> tuple[str, tuple | None, str]:
        """(text, usage, error). The usage is returned with an error too, because a failed answer was still paid for."""
        usage = None
        try:
            obj = json.loads(raw)
            if self.cfg.api == "anthropic":
                text = "".join(b.get("text", "") for b in obj.get("content", []) if isinstance(b, dict))
                u = obj.get("usage") or {}
                usage = (int(u["input_tokens"]), int(u["output_tokens"])) if "input_tokens" in u else None
                finish = obj.get("stop_reason")
            else:
                choice = obj["choices"][0]
                text = choice["message"].get("content")
                u = obj.get("usage") or {}
                usage = (int(u["prompt_tokens"]), int(u["completion_tokens"])) if "prompt_tokens" in u else None
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
        return text, usage, ""

    def _refuse(self, base: dict, outcome: str, reason: str) -> Answer:
        self._append({**base, "ts": self.clock(), "outcome": outcome, "reason": reason, "cached": False,
                      "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0})
        return Answer(False, reason=reason, call_id=base["call_id"], model=base["model"])

    def _cached(self, path: Path) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        ok = isinstance(data, dict) and isinstance(data.get("text"), str) and data.get("reply_file")
        return data if ok and "<think>" not in data["text"] else None

    # ------------------------------------------------------------------------------------------- files (private)

    def _append(self, rec: dict) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self.dir / LEDGER, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, (json.dumps(rec, sort_keys=True) + "\n").encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            pass

    def _write(self, path: Path, text: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, text.encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            pass
