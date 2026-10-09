# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the web UI shows about one run beyond `status`: what is happening now, the results, a timeline,
and how it compares with other runs of the same model.

Everything here is read from the ledger and the run directory; nothing is written. Times are epoch
seconds (ledger timestamps are whole seconds, UTC).
"""
from __future__ import annotations

import json
from pathlib import Path

from orchard.narrate import TIER_ACTOR
from orchard.stages import STAGES, ledger_ts

STAGE_NAMES = {s.number: s.name for s in STAGES}
# The order in which open activities win when several are open at once (a test runs inside an agent step's
# stage, a copy comes before its test, and so on).
PRIORITY = ("test", "copy", "queue", "coder", "park", "agent")
FREE_TEXT_CHARS = 200


def _host(where) -> str | None:
    """'lab node4' -> 'node4'; anything else (this box) -> None."""
    return where[4:] if isinstance(where, str) and where.startswith("lab ") else None


def _spans(entries: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Closed spans (for the timeline) and the spans still open at the end of the ledger, by kind."""
    done, open_ = [], {}

    def start(kind, e, **kw):
        open_[kind] = {"kind": kind, "start": ledger_ts(e), "stage": e["stage"], **kw}

    def end(kind, e, **kw):
        s = open_.pop(kind, None)
        if s is not None:
            done.append({**s, "end": ledger_ts(e), **kw})

    for e in entries:
        ev, d = e["event"], e["data"]
        what = d.get("decision") if ev == "decision" else None
        if what in ("coder starting", "relaunch the coder under this supervisor's lease",
                    "finishing an unfinished coder boot"):
            start("coder", e, label="coder boot")
        elif what in ("coder started", "coder died; restarting it once"):
            end("coder", e)
        elif what == "copying files to the lab":
            start("copy", e, label=f"copy to {_host(d.get('where')) or 'the lab'}", host=_host(d.get("where")))
        elif what == "files copied to the lab":
            end("copy", e, repos=d.get("repos") or [], seconds=d.get("seconds"))
        elif what == "queued for lease":
            start("queue", e, label="waiting for chips", host=_host(d.get("where")))
        elif what in ("lease granted", "test lease taken", "lease taken"):
            end("queue", e)
        elif what == "hardware test started":
            chips = d.get("chips") or []
            start("test", e, label=f"{d.get('config') or len(chips)}-chip test", chips=chips,
                  deadline_s=d.get("deadline_s"), host=_host(d.get("where")), config=d.get("config"))
        elif ev == "evidence" and d.get("what") == "hardware test":
            end("test", e, ok=d.get("returncode") == 0 and not d.get("timed_out"), not_run=d.get("not_run"))
        elif what == "agent step":
            start("agent", e, label=f"{d.get('phase', 'step')}", actor=TIER_ACTOR.get(d.get("tier"), "grafter"),
                  phase=d.get("phase"), escalated=bool(d.get("escalated")))
        elif ev == "evidence" and d.get("what") == "transcript":
            end("agent", e, ok=d.get("status") == "done")
        elif ev == "park":
            if "park" not in open_:
                start("park", e, label="coder parked")
        elif ev == "restore" and d.get("step") in ("canary", "resumed"):
            end("park", e)
        elif ev == "stage_end":
            for kind in ("agent", "test", "copy", "queue"):
                end(kind, e)
        elif what in ("blocked", "abort", "ready for operator review", "pause"):
            for kind in list(open_):
                if kind != "park":
                    end(kind, e)
    return done, open_


def activity(entries: list[dict], state: str, now: float) -> dict:
    """What is happening now, in plain words, with when it began and (for a test) its deadline."""
    _, open_ = _spans(entries)
    kind = next((k for k in PRIORITY if k in open_), None) if state == "running" else None
    if kind is None:
        text = {"ready-for-operator-review": "Finished: ready for your review",
                "blocked": "Stopped: blocked", "paused": "Paused", "aborted": "Aborted",
                "stopped-or-crashed": "The supervisor stopped without finishing",
                "not-started": "Not started", "running": "Between steps"}.get(state, state)
        return {"kind": state if state != "running" else "idle", "text": text, "since": None,
                "elapsed_s": None, "deadline_s": None, "host": None, "actor": None}
    s = open_[kind]
    host = s.get("host")
    where = f"on {host}" if host else "on this box"
    st = s.get("stage")
    stage = f"stage {st} ({STAGE_NAMES.get(st, '')})" if isinstance(st, int) else "the run"
    text = {
        "test": f"Hardware test {where}: {s['label']} for {stage}",
        "copy": f"Copying the run's files to {host or 'the lab'} for {stage}",
        "queue": f"Waiting for free chips {where}",
        "coder": "Starting the coder (a cold boot takes about 6 to 10 minutes)",
        "park": "The coder is parked while a test uses its boards",
        "agent": f"The {s.get('actor', 'grafter')} is working on {stage}: the {s.get('phase', 'step')} step"
                 + (" (escalated)" if s.get("escalated") else ""),
    }[kind]
    return {"kind": kind, "text": text, "since": s["start"], "elapsed_s": max(0.0, now - s["start"]),
            "deadline_s": s.get("deadline_s"), "host": host, "actor": s.get("actor"), "stage": st}


NEXT = {
    "running": "Nothing to do: the run is going. The line above says what it is doing.",
    "paused": "Paused. Resume it when you are ready; the reason is below.",
    "blocked": "Blocked. Read what was tried and how to unblock below, fix the cause, then Retry.",
    "ready-for-operator-review": "Ready for review: open the bundle and read RESULTS.md. Nothing has been published.",
    "aborted": "Aborted. Start a new bring-up to try again.",
    "stopped-or-crashed": "The supervisor stopped without finishing. Retry resumes it from the ledger.",
    "not-started": "Not started yet.",
    "unreadable": "The ledger cannot be read; see `tt-orchard status --run-dir` for the error.",
}


def next_step(state: str) -> str:
    return NEXT.get(state, "")


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def results(run_dir, entries: list[dict]) -> dict:
    """The run's measured results: stage 0's finding, stage 2's swap check, stage 4 per chip count."""
    run_dir = Path(run_dir)
    where: dict[tuple, str | None] = {}
    for e in entries:
        d = e["data"]
        if e["event"] == "decision" and d.get("decision") == "hardware test started":
            where[(e["stage"], d.get("config"))] = _host(d.get("where"))
    out: dict = {}
    delta = _read(run_dir / "stages" / "0" / "delta.json")
    if isinstance(delta, dict):
        out["triage"] = {k: delta.get(k) for k in ("class", "path", "nearest_model", "model")}
    s2 = _read(run_dir / "stages" / "2" / "evidence" / "swap-check.json")
    if isinstance(s2, dict) and isinstance(s2.get("result_draft"), dict):
        r = s2["result_draft"]
        test = _read(run_dir / "stages" / "2" / "test-result.json") or {}
        out["swap"] = {"top1_agreement": r.get("top1_agreement"), "n_tokens": r.get("n_tokens"),
                       "server_ready_s": r.get("server_ready_s"), "coherent": r.get("coherent"),
                       "free_run_text": (r.get("free_run_text") or "")[:FREE_TEXT_CHARS],
                       "test_seconds": test.get("seconds"), "host": where.get((2, None))}
    s4 = _read(run_dir / "stages" / "4" / "result.json")
    plan = _read(run_dir / "stages" / "4" / "tests" / "plan.json") or {}
    counts = sorted({c.get("chips") for c in (s4 or {}).get("configs", []) if isinstance(c, dict)}
                    | {t.get("chips") for t in plan.get("tests", []) if isinstance(t, dict)} - {None})
    rows = []
    by_chips = {c.get("chips"): c for c in (s4 or {}).get("configs", []) if isinstance(c, dict)}
    for n in counts:
        c = by_chips.get(n, {})
        rec = _read(run_dir / "stages" / "4" / "tests" / str(n) / "test-result.json") or {}
        rows.append({"chips": n, "pass": c.get("pass"), "top1_agreement": c.get("top1_agreement"),
                     "server_ready_s": c.get("server_ready_s"), "package": c.get("package"),
                     "reason": rec.get("not_run") or c.get("reason"), "test_seconds": rec.get("seconds"),
                     "host": where.get((4, n))})
    if rows:
        out["configs"] = rows
    if (run_dir / "stages" / "8" / "bundle" / "RESULTS.md").is_file():
        out["bundle"] = "stages/8/bundle/RESULTS.md"
    return out


LANES = ("stage", "agent", "coder", "copy", "queue", "test", "park")


def timeline(entries: list[dict], now: float) -> dict:
    """Spans on one time axis: each stage attempt, agent step, coder boot, copy, lease wait, test and park."""
    if not entries:
        return {"start": None, "end": None, "spans": []}
    spans, open_ = _spans(entries)
    started: dict[int, float] = {}
    for e in entries:
        n = e["stage"]
        if e["event"] == "stage_start" and isinstance(n, int) and not e["data"].get("skip"):
            started[n] = ledger_ts(e)
        elif e["event"] == "stage_end" and isinstance(n, int) and n in started:
            spans.append({"kind": "stage", "start": started.pop(n), "end": ledger_ts(e), "stage": n,
                          "label": f"{n} {STAGE_NAMES.get(n, '')}", "ok": e["data"].get("result") == "pass",
                          "result": e["data"].get("result")})
    end_of_run = ledger_ts(entries[-1])
    for n, t0 in started.items():
        spans.append({"kind": "stage", "start": t0, "end": now, "stage": n, "open": True,
                      "label": f"{n} {STAGE_NAMES.get(n, '')}"})
    for s in open_.values():
        spans.append({**s, "end": now, "open": True})
    spans.sort(key=lambda s: (LANES.index(s["kind"]) if s["kind"] in LANES else 9, s["start"]))
    return {"start": ledger_ts(entries[0]), "end": max(end_of_run, max((s["end"] for s in spans), default=end_of_run)),
            "lanes": list(LANES), "spans": spans}


def summary(run_dir, entries: list[dict]) -> dict:
    """The few numbers that compare across runs of one model."""
    r = results(run_dir, entries)
    start = next((e for e in entries if e["event"] == "run_start"), None)
    return {"started": ledger_ts(entries[0]) if entries else None,
            "wall_s": (ledger_ts(entries[-1]) - ledger_ts(entries[0])) if entries else None,
            "mode": ("lab" if (start or {}).get("data", {}).get("lab") else "local") if start else None,
            "swap_ready_s": (r.get("swap") or {}).get("server_ready_s"),
            "swap_top1": (r.get("swap") or {}).get("top1_agreement"),
            "configs": {str(c["chips"]): {"pass": c["pass"], "ready_s": c["server_ready_s"]}
                        for c in r.get("configs", [])}}
