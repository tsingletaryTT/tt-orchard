# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the web page shows about telegraph, the opt-in helper model (orchard/telegraph.py, README 5.9g).

Read only. It reads the configuration the supervisor already loaded and the ledger in the project database (orchard/db.py,
orchard/telegraph_store.py), and it returns plain facts for `GET /api/telegraph`. The ledger is read in ONE read transaction, so the
summary, the accounting and the rows describe the same moment. It never creates, recovers or migrates a file. It sends no request to the helper, spends nothing, and never reads a reply file, the cache
or a prompt. The key is never returned: only whether one is present and whether it comes from a variable or a file. The key
file is not opened (its size is checked), and neither its path nor the variable's name is returned.

The meter is a single line in the top bar that is always shown. Its text is built here, so a test can check what a person reads.
States: `off` (telegraph disabled), `ok`, `near` (any cap at 80 percent or more), `capped` (a cap is used up, or the latest
call was refused for a budget reason) and `problem` (the key is missing, or the ledger cannot be read).
"""
from __future__ import annotations

import math
import urllib.parse
from pathlib import Path

from orchard import telegraph as tg
from orchard import telegraph_store as ts
from orchard.db import DB_NAME, LedgerRefused, LedgerUnavailable

RECENT_MAX = 50                      # rows in the table of recent calls
PROBLEMS_MAX = 50                    # rows in the per-problem table
LEDGER_MAX_BYTES = 16 * 1024 * 1024  # a larger ledger is not read
NEAR_NUM, NEAR_DEN = 4, 5            # near a cap at 80 percent: used * 5 >= limit * 4
REASON_MAX = 200
TEXT_MAX = 80                        # longest problem or model text shown in a row

TONE = {"off": "dim", "ok": "good", "near": "warn", "capped": "bad", "problem": "warn"}


def usd(value) -> str:
    """Dollars. Zero is "$0.00", a positive amount under a hundredth of a cent is "<$0.0001" (four decimals would round it
    to $0.0000 and hide the spend), and the rest has four decimals under a cent and two above."""
    value = float(value or 0)
    if value == 0:
        return "$0.00"
    if value < 0.0001:
        return "<$0.0001"
    return f"${value:.4f}" if value < 0.01 else f"${value:.2f}"


def _num(value, default=0):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def default_dir(home) -> Path:
    return Path(home) / ".cache" / "tt-orchard" / "telegraph"


def _clean(text, redact, limit=REASON_MAX) -> str:
    text = tg.redact(str(text if text is not None else ""))
    return redact(text)[:limit]


# ---- the ledger, read once per change ----------------------------------------------------------------------------------

_cache: dict = {}                     # directory -> (database change key, facts)

_REFUSED_TEXT = {ts.NotMigrated: "Run `tt-orchard db migrate telegraph --dry-run`, then `tt-orchard db migrate telegraph`: the ledger is still a jsonl file",
                 ts.LegacyLedgerPresent: "an old writer recreated telegraph.jsonl after the migration; stop it and reconcile",
                 ts.MigrationIncomplete: "Run `tt-orchard db migrate telegraph` again: a migration of the ledger did not finish"}


def _ledger_facts(directory: Path) -> dict:
    """Everything the page needs from the ledger, as plain data, from one read transaction. Raises nothing: a ledger that cannot be
    used gives `error`. A directory with no database and no old ledger is an empty history. `blocked` is the reason reservations
    are refused (the ledger is unconfirmed after a restore), or None."""
    directory = Path(directory)
    found = ts.fingerprint(directory)
    key = found[0] if found else None                       # None: no database, or one that cannot be read
    cached = _cache.get(str(directory))
    if key is not None and cached and cached[0] == key:
        return cached[1]
    if found is not None and found[1] > LEDGER_MAX_BYTES:
        facts = {"error": "the ledger is too large to show (over 16 MB)", "records": [], "summary": _empty_summary(),
                 "acct": _empty_acct(), "blocked": None}
    else:
        try:
            snap = ts.read_snapshot(directory)                       # one transaction: the summary, accounting and rows agree
            records = snap.records
            _check_records(records)
            summary = tg.summarise_records(records)
            acct = tg.accounting(records)
            torn = sum(1 for r in records if "torn" in r)
            _check_derived(summary, acct)
            _dry_run(records, acct)
            facts = {"error": None, "records": records, "summary": {**summary, "torn": torn}, "acct": acct,
                     "blocked": snap.blocked}
        except LedgerRefused as exc:                                 # a state a person must fix: the message says which
            text = next((t for cls, t in _REFUSED_TEXT.items() if isinstance(exc, cls)), "the ledger cannot be read")
            return _unreadable(text)
        except LedgerUnavailable:
            return _unreadable("the ledger cannot be read")        # not cached: the next poll tries again
        except Exception:                                           # noqa: BLE001 - a bad record must not break the page
            facts = {"error": "the ledger has a record this page cannot read", "records": [],
                     "summary": _empty_summary(), "acct": _empty_acct(), "blocked": None}
    if key is not None:
        if len(_cache) > 16:
            _cache.clear()
        _cache[str(directory)] = (key, facts)
    return facts


ID_KEYS = ("call_id", "origin", "problem", "model", "reserve", "report")


def _check_records(records: list[dict]) -> None:
    """A record whose id, problem or model is a list or a table is damaged: it cannot be a name or a key. Raising here puts the
    whole ledger in the `problem` state with an unknown spend, inside the guarded read, so no later step meets it."""
    for r in records:
        for key in ID_KEYS:
            if isinstance(r.get(key), (list, dict)):
                raise ValueError(f"record field {key} is not a name")
        _finite(r)
        for key in NUMBER_KEYS:
            _non_negative(key, r.get(key))


NUMBER_KEYS = ("cost_usd", "tokens_in", "tokens_out", "tokens", "latency_s")


def _non_negative(key: str, value) -> None:
    """An amount below zero would lower the spend, and a string such as "NaN" reads as a number in `float()`; both make a ledger a
    damaged ledger."""
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return                                              # not a number at all: the accounting refuses it
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"record field {key} is not a finite amount of zero or more")


def _check_derived(*parts) -> None:
    """Every number the page derives (sums, counts, per-model and per-problem totals) must be finite and not negative. Two finite
    costs can add up to infinity, so this checks what is derived and not only what was read. It runs before anything is cached."""
    def walk(v, depth=0):
        if depth > 20:
            raise ValueError("derived value is nested too deeply")
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            if not math.isfinite(v) or v < 0:
                raise ValueError("a derived amount is not finite or is below zero")
        elif isinstance(v, dict):
            for x in v.values():
                walk(x, depth + 1)
        elif isinstance(v, list):
            for x in v:
                walk(x, depth + 1)
    for part in parts:
        walk(part)


def _percent(used, limit) -> int:
    """`used` as a whole percent of `limit`, never raising and never above PERCENT_MAX (a huge ratio would overflow to infinity)."""
    if limit <= 0:
        return 100
    ratio = used * 100 / limit
    return PERCENT_MAX if not math.isfinite(ratio) or ratio > PERCENT_MAX else int(ratio)


PERCENT_MAX = 10 ** 9


def _finite(value, depth: int = 0) -> None:
    """Raise unless every number in `value` is finite. Python's JSON reader accepts NaN and Infinity, and either one would turn
    the spend, the caps and the answer itself into values a browser cannot read, so such a ledger is a damaged ledger."""
    if depth > 20:
        raise ValueError("record is nested too deeply")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("record holds a number that is not finite")
    elif isinstance(value, dict):
        for v in value.values():
            _finite(v, depth + 1)
    elif isinstance(value, list):
        for v in value:
            _finite(v, depth + 1)


def _dry_run(records: list[dict], acct: dict) -> None:
    """Run every derivation the page makes on these records once, with nothing redacted, so a record that breaks one of them
    is found here (and cached as an error) and not while an answer is being built."""
    ident = lambda t: t                                          # noqa: E731
    _recent(records, _verdicts(records), ident)
    _model_rows(records, ident)
    _problem_rows(records, acct, _verdicts(records), tg.Config(), ident)
    _latest_refusal(records)


def _unreadable(message: str) -> dict:
    return {"error": message, "records": [], "summary": _empty_summary(), "acct": _empty_acct(), "blocked": None}


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
            percent = max(percent, _percent(used, limit))
        if cfg.max_cost_per_problem_usd is not None and row["cost_unknown_calls"]:
            full = near = True                    # the producer refuses this problem's calls while a cost is unknown
            percent = 100                         # so it counts as fully used, like the global cost cap does
        row["percent"], row["near"], row["at_cap"] = percent, near, full
        rows.append(row)
    # Every problem is returned: the state and the closest cap are worked out over all of them, and only the table is cut.
    # Problems at their cap come first, then the ones at 80 percent or more, then the busiest.
    rows.sort(key=lambda r: (not r["at_cap"], not r["near"], -r["calls"], r["name"]))
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
        percent = 100 if blocked else _percent(used, limit)
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


TS_MAX = 4e10                         # year 3200: a larger number is not a time stamp a browser can show


def _time(value):
    """A time stamp in seconds, or None when it is not a number in a range a date can show (the page would print Invalid Date)."""
    v = _num(value, None)
    return v if v is not None and math.isfinite(v) and 0 <= v <= TS_MAX else None


def _reservation_row(r: dict, redact) -> dict:
    cost = r.get("cost_usd")
    return {
        "ts": None, "call_id": _clean(r["reserve"], redact, 32), "problem": _clean(r.get("problem"), redact, TEXT_MAX),
        "model": _clean(r.get("model"), redact, TEXT_MAX), "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
        "cost_usd": float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
        "outcome": "unsettled", "held": True, "usage_known": None, "helped": None, "cost_basis": "worst case",
        "reason": "No result was recorded: the call is in flight or the process stopped. It counts at its worst case."}


def _call_row(r: dict, reserves: dict, verdict: dict, redact) -> dict:
    cached = bool(r.get("cached"))
    outcome = "cache hit" if cached else str(r.get("outcome"))
    cost = r.get("cost_usd")
    unknown_usage = r.get("usage_known") is False and not cached
    basis, reserved = "recorded", reserves.get(r.get("call_id"))
    if unknown_usage:
        # The caps count this call at its reservation (a bound), not at the settlement's estimate; with no reservation
        # (a ledger from before reservations) they count the recorded estimate.
        basis = "worst case" if reserved is not None else "estimate"
        cost = reserved.get("cost_usd") if reserved is not None else cost
    reason = _clean(r.get("reason"), redact) if r.get("reason") else None
    if unknown_usage and not reason:
        reason = ("The provider reported no usage. The caps count this call at its reserved worst case, not at the "
                  "estimate in the recorded numbers." if basis == "worst case" else
                  "The provider reported no usage. The numbers are an estimate, and the caps count them as recorded.")
    return {
        "ts": _time(r.get("ts")), "call_id": _clean(r.get("call_id"), redact, 32),
        "problem": _clean(r.get("problem"), redact, TEXT_MAX), "model": _clean(r.get("model"), redact, TEXT_MAX),
        "tokens_in": int(_num(r.get("tokens_in"))), "tokens_out": int(_num(r.get("tokens_out"))),
        "latency_s": float(_num(r.get("latency_s"))), "held": False,
        "cost_usd": float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
        "outcome": _clean(outcome, redact, 20), "usage_known": False if unknown_usage else None, "reason": reason,
        "cost_basis": basis,
        "helped": verdict.get(r.get("call_id")) if r.get("outcome") == "ok" and not cached else None,
    }


def _recent(records: list[dict], verdict: dict, redact) -> list[dict]:
    """Calls, refusals and cache hits, newest first, and every reservation that was never settled (shown as `unsettled`).
    `reserve` and `torn` lines are not calls of their own and never appear as rows. Only the newest RECENT_MAX rows are built:
    the walk starts at the end of the ledger and stops there, so a long ledger costs no more than a short one."""
    settled = {r.get("call_id") for r in records if "call_id" in r and not r.get("cached")}
    reserves = {r["reserve"]: r for r in records if "reserve" in r}
    rows = []
    for r in reversed(records):
        if len(rows) >= RECENT_MAX:
            break
        if "reserve" in r:
            if r["reserve"] not in settled:
                rows.append(_reservation_row(r, redact))
        elif "call_id" in r:
            rows.append(_call_row(r, reserves, verdict, redact))
    return rows


def _model_rows(records: list[dict], redact) -> list[dict]:
    """Totals for each model, by the accounting the caps use: the same `telegraph.accounting`, run with each record's model in
    the place of its problem. A reservation counts at its reserved bound until a result settles it, a reply with no usage
    counts at its reservation, and a torn record is one call of unknown cost under its own row."""
    moved = []
    for r in records:
        if "reserve" in r or tg._sent(r):
            moved.append({**r, "problem": r.get("model")})
    acct = tg.accounting(moved)
    rows = [{"model": _clean(m, redact, TEXT_MAX), "calls": p["calls"], "tokens": p["tokens"], "cost_usd": p["cost"],
             "cost_unknown_calls": p["unknown_cost"]} for m, p in acct["problems"].items()]
    torn = [r for r in records if "torn" in r]
    if torn:
        rows.append({"model": "(torn records)", "calls": len(torn), "tokens": sum(int(_num(r.get("tokens"))) for r in torn),
                     "cost_usd": 0.0, "cost_unknown_calls": len(torn)})
    rows.sort(key=lambda r: (-r["calls"], r["model"]))
    return rows


def _latest_refusal(records: list[dict]) -> dict | None:
    """The newest call record, when it is a refusal for a budget reason."""
    for r in reversed(records):
        if "call_id" not in r:
            continue
        return r if r.get("outcome") == "budget" and not r.get("cached") else None
    return None


def _meter(state: str, spend: str, calls: int, cap_text: str, reason: str | None, key_state: dict, error: str | None,
           migration: dict | None = None) -> dict:
    if state == "off":
        text = "Helper: off"
    elif state == "problem" and error and migration and migration.get("kind") in ("unmigrated", "half-finished"):
        text = "Helper: ledger needs moving · spend unknown"       # the commands, with every option, are in the title and the page
    elif state == "problem" and error:
        text = "Helper: ledger unreadable · spend unknown"        # nothing was read, so no amount and no count is shown
    elif state == "problem":
        text = f"Helper: key missing · {spend} · {calls} {'call' if calls == 1 else 'calls'}"
    else:
        text = f"Helper: {spend} · {calls} {'call' if calls == 1 else 'calls'}"
        if state == "capped":
            text += f" · CAPPED: {(reason or 'a cap is used up')[:140]}"
        elif state == "near":
            text += f" · near cap: {cap_text}"
        elif cap_text:
            text += f" · {cap_text}"
    more = {"off": "Telegraph is off. Click for how to turn it on.",
            "ok": "What the outside helper has cost. Click for the calls.",
            "near": "A cap is almost used up. Click for the calls and caps.",
            "capped": f"A cap stopped the helper: {reason}. Click for the calls and caps.",
            "problem": (error or key_state.get("reason") or "Check the setup") + ". Click for details."}[state]
    title = f"{text}. {more}"                                    # the title carries everything the (wrapped or cut) text shows
    return {"text": text, "title": title}


def _spend_text(cost: float, unknown: int) -> str:
    """Known spend, with the number of calls whose cost is not known. An unknown cost is never shown as zero."""
    if unknown and not cost:
        return "unknown cost"
    return f"{usd(cost)} + {unknown} unknown" if unknown else usd(cost)


def build(cfg: tg.Config, directory, *, environ, redact=lambda t: t, hostname: str = "", now: float = 0.0,
          ledger_label: str = "", problem: str | None = None, migration: dict | None = None) -> dict:
    """The telegraph page's facts. `directory` holds `telegraph.jsonl`; `ledger_label` is how to name it on screen."""
    facts = _unreadable("the ledger directory cannot be chosen") if problem else _ledger_facts(Path(directory))
    summary, records, error, acct = facts["summary"], facts["records"], facts["error"], facts["acct"]
    blocked = facts.get("blocked")
    verdict = _verdicts(records)
    caps = _caps(cfg, acct) if cfg.enabled and not error else []     # an unread ledger shows no usage bars at 0 percent
    key = _key_state(cfg, environ) if cfg.enabled else {"state": "n/a", "source": "none", "reason": None}
    problems = _problem_rows(records, acct, verdict, cfg, redact) if cfg.enabled or records else []
    refusal = _latest_refusal(records)
    full = [c for c in caps if c["tone"] == "bad"]
    if not cfg.enabled:
        state, reason = "off", None
    elif blocked:
        state, reason = "capped", _clean(blocked, redact, 160)          # reservations are refused until a person confirms the history
    elif refusal is not None or full:
        state = "capped"
        reason = _clean(refusal.get("reason"), redact, 160) if refusal is not None else _cap_reason(full[0], acct)
    elif key["state"] == "missing" or error:
        state, reason = "problem", None
    elif any(c["tone"] == "warn" for c in caps) or any(p["near"] for p in problems):
        state, reason = "near", None
    else:
        state, reason = "ok", None
    # The meter names the cap that is closest to its limit, whether it is a global cap or one problem's, even below the amber
    # threshold. A global cap wins a tie, and the first of equal global caps is the one named.
    closest = max(caps, key=lambda c: (c["percent"], -caps.index(c)), default=None)
    hot = max(problems, key=lambda p: (p["percent"], p["calls"]), default=None)
    if hot is not None and (closest is None or hot["percent"] > closest["percent"]):
        cap_text = f"problem {_clean(hot['name'], redact, TEXT_MAX)} at {hot['percent']}% of its cap"
    else:
        cap_text = _cap_phrase(closest) if closest else ""
    migration = migration if not problem else None
    if migration and error:
        error = migration["summary"]         # the store's own message carries bare commands; the summary has the ones with every option
    meter = _meter(state, _spend_text(acct["cost"], acct["unknown_cost"]), acct["calls"], cap_text, reason, key, error, migration)
    if migration and state == "problem" and error:
        meter["title"] = f"{meter['text']}. {migration['summary']} Click for details."
    by_model = _model_rows(records, redact)
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
        "caps": caps, "problems": [{**p, "name": _clean(p["name"], redact, TEXT_MAX)} for p in problems[:PROBLEMS_MAX]],
        "problems_hidden": max(0, len(problems) - PROBLEMS_MAX), "by_model": by_model,
        "summary": {k: v for k, v in summary.items() if k not in ("by_model", "by_problem")},
        "spend": {"cost_usd": acct["cost"], "unknown_cost": acct["unknown_cost"], "calls": acct["calls"],
                  "tokens": acct["tokens"], "unsettled": summary["unsettled"], "torn": summary["torn"],
                  "unknown_usage_calls": summary["unknown_usage_calls"]},
        "capped": {"reason": reason} if state == "capped" else None,
        "recent": _recent(records, verdict, redact), "error": error, "now": now,
        "migration": migration,           # what to do about an old, half-moved or lost ledger (orchard/telegraph_notice.py), or None
    }


def run_spend(run_dir: Path) -> dict | None:
    """One run's own spend, from the ledger in the run directory (`<run>/project.db`, or an old `<run>/telegraph.jsonl`, which is
    reported as not migrated). None means the run has no usable ledger path (no file, or a link, which is not followed). A ledger
    that exists but cannot be read, is damaged or is too large gives {"error": ...}, so the page can say the run's spend is unknown
    instead of showing nothing."""
    found = False
    for name in (DB_NAME, tg.LEDGER):
        path = Path(run_dir) / name
        try:
            if path.is_symlink():
                return None
            path.lstat()
            found = True
        except FileNotFoundError:
            continue
        except OSError:
            return {"error": "the run's ledger cannot be read"}
    if not found:
        return None
    facts = _ledger_facts(Path(run_dir))
    if facts["error"]:
        return {"error": facts["error"].replace("the ledger", "the run's ledger", 1)}
    s, a = facts["summary"], facts["acct"]
    return {"calls": a["calls"], "cost_usd": a["cost"], "unknown_cost": a["unknown_cost"], "tokens": a["tokens"],
            "unsettled": s["unsettled"], "cache_hits": s["cache_hits"], "refused": s["refused"]}
