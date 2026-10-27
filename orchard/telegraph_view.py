# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the web page shows about telegraph, the opt-in helper model (orchard/telegraph.py, README 5.9g).

Read only. It reads the configuration the supervisor already loaded and the ledger `telegraph.jsonl`, and it returns plain
facts for `GET /api/telegraph`. It sends no request to the helper, spends nothing, and never reads a reply file, the cache
or a prompt. The key is never returned: only whether one is present and whether it comes from a variable or a file. The key
file is not opened (its size is checked), and neither its path nor the variable's name is returned.

The meter is a single line in the top bar that is always shown. Its text is built here, so a test can check what a person reads.
States: `off` (telegraph disabled), `ok`, `near` (any cap at 80 percent or more), `capped` (a cap is used up, or the latest
call was refused for a budget reason) and `problem` (the key is missing, or the ledger cannot be read).
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path

from orchard import telegraph as tg

RECENT_MAX = 50                      # rows in the table of recent calls
PROBLEMS_MAX = 50                    # rows in the per-problem table
LEDGER_MAX_BYTES = 16 * 1024 * 1024  # a larger ledger is not read
NEAR_NUM, NEAR_DEN = 4, 5            # near a cap at 80 percent: used * 5 >= limit * 4
REASON_MAX = 200
TEXT_MAX = 80                        # longest problem or model text shown in a row

TONE = {"off": "dim", "ok": "good", "near": "warn", "capped": "bad", "problem": "warn"}


def usd(value) -> str:
    """Dollars with enough digits that a small cost never shows as zero."""
    value = float(value or 0)
    if value == 0:
        return "$0.00"
    return f"${value:.4f}" if value < 0.01 else f"${value:.2f}"


def _num(value, default=0):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def default_dir(home) -> Path:
    return Path(home) / ".cache" / "tt-orchard" / "telegraph"


def _clean(text, redact, limit=REASON_MAX) -> str:
    text = tg.redact(str(text if text is not None else ""))
    return redact(text)[:limit]


# ---- the ledger, read once per change ----------------------------------------------------------------------------------

_cache: dict = {}                     # directory -> ((mtime_ns, size), facts)


def _ledger_facts(directory: Path) -> dict:
    """Everything the page needs from the ledger, as plain data. Raises nothing: a ledger that cannot be used gives `error`."""
    path = Path(directory) / tg.LEDGER
    try:
        st = path.stat()
        key = (st.st_mtime_ns, st.st_size)
    except OSError:
        return {"error": None, "records": [], "summary": _empty_summary(), "acct": _empty_acct()}
    cached = _cache.get(str(directory))
    if cached and cached[0] == key:
        return cached[1]
    if st.st_size > LEDGER_MAX_BYTES:
        facts = {"error": "the ledger is too large to show (over 16 MB)", "records": [], "summary": _empty_summary(),
                 "acct": _empty_acct()}
    else:
        try:
            records = tg.read_ledger(directory)
            summary = tg.summarise(directory)
            acct = tg.accounting(records)
            torn = sum(1 for r in records if "torn" in r)
            facts = {"error": None, "records": records, "summary": {**summary, "torn": torn}, "acct": acct}
        except Exception:                                           # noqa: BLE001 - a bad record must not break the page
            facts = {"error": "the ledger has a record this page cannot read", "records": [],
                     "summary": _empty_summary(), "acct": _empty_acct()}
    if len(_cache) > 16:
        _cache.clear()
    _cache[str(directory)] = (key, facts)
    return facts


def _empty_acct() -> dict:
    return {"calls": 0, "tokens": 0, "cost": 0.0, "unknown_cost": 0, "problems": {}}


def _empty_summary() -> dict:
    return {"torn": 0, "unsettled": 0, "unknown_usage_calls": 0, "calls": 0, "ok": 0, "helped": 0, "not_helped": 0, "unreported": 0, "cache_hits": 0, "refused": 0,
            "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "cost_unknown_calls": 0, "latency_s": 0.0,
            "by_model": {}, "by_problem": {}}


def _verdicts(records: list[dict]) -> dict:
    out = {}
    for r in records:
        if "report" in r:
            out[r["report"]] = bool(r.get("helped"))
    return out


def _problem_rows(records: list[dict], acct: dict, verdict: dict, cfg: tg.Config, redact) -> list[dict]:
    """One row for each problem, from the same accounting the caps use (a reservation or a torn record counts as a call)."""
    helped: dict = {}
    for r in records:
        if tg._sent(r) and r.get("outcome") == "ok":
            v = verdict.get(r.get("call_id"))
            h = helped.setdefault(r.get("problem"), [0, 0])
            h[0] += 1 if v is True else 0
            h[1] += 1 if v is False else 0
    rows = []
    for name, p in acct["problems"].items():
        row = {"name": str(name), "calls": p["calls"], "tokens": p["tokens"], "cost_usd": p["cost"],
               "cost_unknown_calls": p["unknown_cost"], "helped": helped.get(name, [0, 0])[0],
               "not_helped": helped.get(name, [0, 0])[1]}
        limits = (("calls", cfg.max_calls_per_problem), ("tokens", cfg.max_tokens_per_problem),
                  ("cost_usd", cfg.max_cost_per_problem_usd))
        percent, near, full = 0, False, False
        for key, limit in limits:
            if limit is None:
                continue
            used = row[key]
            if limit <= 0 or used >= limit:
                full = True
            if used * NEAR_DEN >= limit * NEAR_NUM:
                near = True
            percent = max(percent, 100 if limit <= 0 else int(used * 100 / limit))
        if cfg.max_cost_per_problem_usd is not None and row["cost_unknown_calls"]:
            full = near = True                    # the producer refuses this problem's calls while a cost is unknown
        row["percent"], row["near"], row["at_cap"] = percent, near, full
        rows.append(row)
    rows.sort(key=lambda r: (-r["calls"], r["name"]))
    rows = rows[:PROBLEMS_MAX]
    for row in rows:
        row["name"] = _clean(row["name"], redact, TEXT_MAX)
    return rows


def _caps(cfg: tg.Config, acct: dict) -> list[dict]:
    """The global caps with what the producer counts against them (acct): settled calls, reservations that were never settled
    (at their worst case) and torn records. While a cost is unknown the producer cannot check the cost cap and refuses new
    calls, so that cap is full (`blocked`) however little is known to be spent."""
    spec = [("calls", "calls", acct["calls"], cfg.max_calls, "calls"),
            ("tokens", "tokens", acct["tokens"], cfg.max_total_tokens, "tokens")]
    if cfg.max_cost_usd is not None:
        spec.append(("cost", "cost", acct["cost"], cfg.max_cost_usd, "usd"))
    caps = []
    for cid, label, used, limit, unit in spec:
        blocked = cid == "cost" and acct["unknown_cost"] > 0
        full = limit <= 0 or used >= limit or blocked
        near = used * NEAR_DEN >= limit * NEAR_NUM
        percent = 100 if limit <= 0 or blocked else int(used * 100 / limit)
        caps.append({"id": cid, "label": label, "used": used, "limit": limit, "unit": unit, "percent": percent,
                     "blocked": blocked, "tone": "bad" if full else "warn" if near else "good"})
    return caps


def _cap_reason(cap: dict, acct: dict) -> str:
    if cap["blocked"]:
        n = acct["unknown_cost"]
        return (f"{n} earlier {'call has' if n == 1 else 'calls have'} no known cost, so the cost cap cannot be checked "
                "and new calls are refused")
    if cap["id"] == "calls":
        return f"the call cap is reached ({cap['used']} of {cap['limit']})"
    if cap["id"] == "tokens":
        return f"the token cap is reached ({cap['used']} of {cap['limit']})"
    return f"the cost cap is reached ({usd(cap['used'])} of {usd(cap['limit'])})"


def _cap_phrase(cap: dict) -> str:
    if cap["unit"] == "usd":
        return f"cost {usd(cap['used'])} of {usd(cap['limit'])}"
    return f"{cap['label']} {cap['used']} of {cap['limit']}"


def _key_state(cfg: tg.Config, environ) -> dict:
    """Present, missing or not needed. The key file is not opened and nothing about where the key lives is returned beyond
    "variable" or "file"."""
    if cfg.key_env:
        present = bool(str(environ.get(cfg.key_env, "")).strip())
        return {"state": "present" if present else "missing", "source": "variable",
                "reason": None if present else "the key variable is not set"}
    if cfg.key_file:
        try:
            # Only to see whether there is anything but white space in it; the bytes are dropped at once.
            with open(cfg.key_file, "rb") as fh:
                present = bool(fh.read(4096).strip())
        except OSError:
            present = False
        return {"state": "present" if present else "missing", "source": "file",
                "reason": None if present else "the key file is missing or empty"}
    return {"state": "not-needed", "source": "none", "reason": None}


def _host(endpoint: str, redact) -> str:
    try:
        host = urllib.parse.urlsplit(endpoint).hostname or ""
    except ValueError:
        host = ""
    return redact(host)


def _recent(records: list[dict], verdict: dict, redact) -> list[dict]:
    """Calls, refusals and cache hits, newest first, and every reservation that was never settled (shown as `unsettled`).
    `reserve` and `torn` lines are not calls of their own and never appear as rows."""
    settled = {r.get("call_id") for r in records if "call_id" in r and not r.get("cached")}
    rows = []
    for r in records:
        if "reserve" in r:
            if r["reserve"] in settled:
                continue
            cost = r.get("cost_usd")
            rows.append({
                "ts": None, "call_id": _clean(r["reserve"], redact, 32), "problem": _clean(r.get("problem"), redact, TEXT_MAX),
                "model": _clean(r.get("model"), redact, TEXT_MAX), "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
                "cost_usd": float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
                "outcome": "unsettled", "held": True, "usage_known": None, "helped": None,
                "reason": "No result was recorded: the call is in flight or the process stopped. It counts at its worst case."})
            continue
        if "call_id" not in r:
            continue
        cached = bool(r.get("cached"))
        outcome = "cache hit" if cached else str(r.get("outcome"))
        cost = r.get("cost_usd")
        unknown_usage = r.get("usage_known") is False and not cached
        reason = _clean(r.get("reason"), redact) if r.get("reason") else None
        if unknown_usage and not reason:
            reason = "The provider reported no usage; the caps count this call at its reserved worst case."
        rows.append({
            "ts": _num(r.get("ts"), None), "call_id": _clean(r.get("call_id"), redact, 32),
            "problem": _clean(r.get("problem"), redact, TEXT_MAX), "model": _clean(r.get("model"), redact, TEXT_MAX),
            "tokens_in": int(_num(r.get("tokens_in"))), "tokens_out": int(_num(r.get("tokens_out"))),
            "latency_s": float(_num(r.get("latency_s"))), "held": False,
            "cost_usd": float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
            "outcome": _clean(outcome, redact, 20), "usage_known": False if unknown_usage else None, "reason": reason,
            "helped": verdict.get(r.get("call_id")) if r.get("outcome") == "ok" and not cached else None,
        })
    return rows[::-1][:RECENT_MAX]


def _latest_refusal(records: list[dict]) -> dict | None:
    """The newest call record, when it is a refusal for a budget reason."""
    for r in reversed(records):
        if "call_id" not in r:
            continue
        return r if r.get("outcome") == "budget" and not r.get("cached") else None
    return None


def _meter(state: str, spend: str, calls: int, cap_text: str, reason: str | None, key_state: dict, error: str | None) -> dict:
    if state == "off":
        text = "Helper: off"
    elif state == "problem":
        word = "key missing" if key_state["state"] == "missing" and not error else "ledger unreadable"
        text = f"Helper: {word} · {spend} · {calls} {'call' if calls == 1 else 'calls'}"
    else:
        text = f"Helper: {spend} · {calls} {'call' if calls == 1 else 'calls'}"
        if state == "capped":
            text += f" · CAPPED: {(reason or 'a cap is used up')[:140]}"
        elif state == "near":
            text += f" · near cap: {cap_text}"
        elif cap_text:
            text += f" · {cap_text}"
    title = {"off": "Telegraph is off. Click for how to turn it on.",
             "ok": "What the outside helper has cost. Click for the calls.",
             "near": "A cap is almost used up. Click for the calls and caps.",
             "capped": f"A cap stopped the helper: {reason}. Click for the calls and caps.",
             "problem": (error or key_state.get("reason") or "Check the setup") + ". Click for details."}[state]
    return {"text": text, "title": title}


def _spend_text(cost: float, unknown: int) -> str:
    """Known spend, with the number of calls whose cost is not known. An unknown cost is never shown as zero."""
    if unknown and not cost:
        return "unknown cost"
    return f"{usd(cost)} + {unknown} unknown" if unknown else usd(cost)


def build(cfg: tg.Config, directory, *, environ, redact=lambda t: t, hostname: str = "", now: float = 0.0,
          ledger_label: str = "") -> dict:
    """The telegraph page's facts. `directory` holds `telegraph.jsonl`; `ledger_label` is how to name it on screen."""
    facts = _ledger_facts(Path(directory))
    summary, records, error, acct = facts["summary"], facts["records"], facts["error"], facts["acct"]
    verdict = _verdicts(records)
    caps = _caps(cfg, acct) if cfg.enabled else []
    key = _key_state(cfg, environ) if cfg.enabled else {"state": "n/a", "source": "none", "reason": None}
    problems = _problem_rows(records, acct, verdict, cfg, redact) if cfg.enabled or records else []
    refusal = _latest_refusal(records)
    full = [c for c in caps if c["tone"] == "bad"]
    if not cfg.enabled:
        state, reason = "off", None
    elif refusal is not None or full:
        state = "capped"
        reason = _clean(refusal.get("reason"), redact, 160) if refusal is not None else _cap_reason(full[0], acct)
    elif key["state"] == "missing" or error:
        state, reason = "problem", None
    elif any(c["tone"] == "warn" for c in caps) or any(p["near"] for p in problems):
        state, reason = "near", None
    else:
        state, reason = "ok", None
    closest = max(caps, key=lambda c: (c["percent"], -caps.index(c)), default=None)
    if state == "near" and not any(c["tone"] == "warn" for c in caps):
        hot = max(problems, key=lambda p: p["percent"])
        cap_text = f"problem {hot['name']} at {hot['percent']}% of its cap"
    else:
        cap_text = _cap_phrase(closest) if closest else ""
    meter = _meter(state, _spend_text(acct["cost"], acct["unknown_cost"]), acct["calls"], cap_text, reason, key, error)
    by_model = sorted(({"model": _clean(m, redact, TEXT_MAX), "calls": v["calls"],
                        "cost_usd": sum(r["cost_usd"] for r in records if tg._sent(r) and r.get("model") == m
                                        and isinstance(r.get("cost_usd"), (int, float)) and not isinstance(r.get("cost_usd"), bool)),
                        "cost_unknown_calls": sum(1 for r in records if tg._sent(r) and r.get("model") == m
                                                  and not isinstance(r.get("cost_usd"), (int, float))),
                        "tokens": sum(int(_num(r.get("tokens_in"))) + int(_num(r.get("tokens_out")))
                                      for r in records if tg._sent(r) and r.get("model") == m)}
                       for m, v in summary["by_model"].items()), key=lambda r: (-r["calls"], r["model"]))
    config = {
        "endpoint_host": _host(cfg.endpoint, redact) if cfg.enabled else None, "api": cfg.api if cfg.enabled else None,
        "models": [{"name": _clean(m, redact, TEXT_MAX),
                    "input_per_mtok": cfg.prices[m][0] if m in cfg.prices else None,
                    "output_per_mtok": cfg.prices[m][1] if m in cfg.prices else None} for m in cfg.models],
        "after_failures": cfg.after_failures, "max_output_tokens": cfg.max_output_tokens,
        "max_prompt_chars": cfg.max_prompt_chars, "timeout_s": cfg.timeout_s,
        "per_problem": {"calls": cfg.max_calls_per_problem, "tokens": cfg.max_tokens_per_problem,
                        "cost_usd": cfg.max_cost_per_problem_usd},
        "ledger_dir": ledger_label,
    }
    return {
        "state": state, "tone": TONE[state], "enabled": cfg.enabled, "meter": meter, "config": config, "key": key,
        "caps": caps, "problems": problems, "by_model": by_model,
        "summary": {k: v for k, v in summary.items() if k not in ("by_model", "by_problem")},
        "spend": {"cost_usd": acct["cost"], "unknown_cost": acct["unknown_cost"], "calls": acct["calls"],
                  "tokens": acct["tokens"], "unsettled": summary["unsettled"], "torn": summary["torn"],
                  "unknown_usage_calls": summary["unknown_usage_calls"]},
        "capped": {"reason": reason} if state == "capped" else None,
        "recent": _recent(records, verdict, redact), "error": error, "now": now,
    }


def run_spend(run_dir: Path) -> dict | None:
    """One run's own spend, from `<run>/telegraph.jsonl` when the run has one (it must be a regular file; a link is ignored)."""
    path = Path(run_dir) / tg.LEDGER
    try:
        if path.is_symlink() or not path.is_file():
            return None
    except OSError:
        return None
    facts = _ledger_facts(Path(run_dir))
    if facts["error"]:
        return None
    s, a = facts["summary"], facts["acct"]
    return {"calls": a["calls"], "cost_usd": a["cost"], "unknown_cost": a["unknown_cost"], "tokens": a["tokens"],
            "unsettled": s["unsettled"], "cache_hits": s["cache_hits"], "refused": s["refused"]}
