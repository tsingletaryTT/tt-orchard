# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the web page shows about `tt-orchard updates` (issue 231).

Read only, like orchard/cans_view.py. It reads the cache file that `tt-orchard updates` wrote (orchard/updates.py `cache_path`) and
returns the rows as plain facts for `GET /api/updates`. It makes no network call, starts no process, writes no file, opens no chip and
returns no path. A missing or unreadable cache is a state of its own ("no check has been recorded"), never an error.

Every string from the cache file goes through `updates.clean` (size, type, control characters) and then `redact`. A row field of the
wrong type becomes "-" or an empty list. The number of rows, of moves-with lines and of proposal lines is bounded.
"""
from __future__ import annotations

import time
from pathlib import Path

from orchard import skillsfresh, updates

MAX_ROWS = 40
MAX_LINES = 8
RUN_COMMAND = "tt-orchard updates --refresh"          # shown as text: the page never runs it
STATUS_TONE = {updates.CURRENT: "good", updates.BEHIND: "warn", updates.CANDIDATE: "warn", updates.UNKNOWN: "dim"}


def _lines(value, redact) -> list[str]:
    if not isinstance(value, list):
        return []
    return [redact(updates.clean(v, 200)) for v in value[:MAX_LINES] if isinstance(v, str)]


def _text(value, redact, n=updates.TEXT_MAX) -> str:
    return redact(updates.clean(value, n)) or "-"


def _date(value):
    return value if isinstance(value, str) else None


def build(cache_file: Path | None = None, *, redact=lambda t: t, clock=time.time) -> dict:
    """The answer of `GET /api/updates`: {read_only, state, checked_age_s, run_command, note, rows}. `state` is "ok", "none" (no check
    recorded) or "stale" (older than the cache lifetime that `tt-orchard updates` uses; the rows are still shown)."""
    out = {"read_only": True, "state": "none", "checked_age_s": None, "run_command": RUN_COMMAND, "rows": [],
           "note": "No check has been recorded. Run the command shown, then reload this page."}
    cache = skillsfresh.Cache(cache_file or updates.cache_path(), clock=clock, write=False)
    ent = cache.get("report")
    rows = ent.get("rows") if ent else None
    if not isinstance(rows, list):
        return out
    age = cache.age_s(ent)
    now = clock()
    out["checked_age_s"] = round(age) if age is not None else None
    fresh = age is not None and 0 <= age <= updates.CACHE_HOURS * 3600
    out["state"] = "ok" if fresh else "stale"
    out["note"] = ("" if fresh else
                   "The last check is older than the cache lifetime of 6 hours, or its time is unreadable. Run the command shown for a new one.")
    for r in rows[:MAX_ROWS]:
        if not isinstance(r, dict):
            continue
        status = r.get("status") if r.get("status") in STATUS_TONE else updates.UNKNOWN
        why = _text(r.get("why"), redact, 200) if r.get("why") else ""
        out["rows"].append({
            "item": _text(r.get("item"), redact), "installed": _text(r.get("installed"), redact),
            "installed_age": updates.age_text(_date(r.get("installed_date")), now),
            "upstream": _text(r.get("upstream"), redact),
            "upstream_age": updates.age_text(_date(r.get("upstream_date")), now),
            "status": status, "tone": STATUS_TONE[status],
            "label": f"unknown ({why})" if status == updates.UNKNOWN and why else status,
            "why": why, "moves_with": _lines(r.get("moves_with"), redact),
            "ready": _text(r.get("ready"), redact, 200), "proposal": _lines(r.get("proposal"), redact),
        })
    return out
