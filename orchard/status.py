"""Read-only status of a run: `python3 -m orchard.supervisor status --run-dir DIR [--json]`.

This exists so a small local model, acting as the run's operator, can see the whole state of a
run in about 30 lines instead of reading a ledger. It is also useful to a person.

It changes nothing. It never opens the ledger writer, so it takes no ledger lock and works while
the supervisor is running. It reads `ledger.jsonl` with `orchard.ledger.read_entries`, which
verifies the hash chain. A write in progress (bytes after the last newline) is ignored and is
not repaired. The run directory is only read: no sidecar, lock or pid file is created.

What it reports, and where each fact comes from:

- state: one of running, paused, ready-for-operator-review, aborted, stopped-or-crashed,
  not-started. "ready-for-operator-review" and "aborted" come from the ledger alone. "paused" and
  "running" additionally need a live supervisor. A run that is neither finished nor aborted and
  has no live supervisor is "stopped-or-crashed", whatever the ledger says about a pause.
- supervisor liveness: the supervisor writes `<run dir>/supervisor.pid` when it starts. Runs that
  started before that file existed have none, so the fallback is /proc/locks: the supervisor holds
  an flock on `ledger.jsonl.lock` for as long as it lives, and /proc/locks names the holder. Reading
  /proc/locks takes no lock. With neither signal the supervisor is reported as not alive and the
  source is "none". A pid from the file counts as alive only if the process exists, is not a
  zombie, and its command line mentions the supervisor (a reused pid is not a supervisor).
- the hint (`next:`): HINT_RULES, a table of (name, test, text). The first matching row wins.

Exit codes: 0 when a status was produced (including a paused or crashed run), 2 for a bad run
directory or a corrupt ledger.

JSON (`--json`) is one object with these keys:

    ledger           {"ok": bool, "entries": int, "error": str or null}
    model            model id from run_start, or null
    run_dir          absolute path
    run_name         last component of the run directory
    state            one of the six states above
    supervisor       {"pid": int or null, "alive": bool, "source": "supervisor.pid" | "ledger lock"
                      | "none"}
    stage            {"current": int or null, "name": str or null, "attempt": int}
    stages           list of {"stage", "name", "status", "wall_s", "attempts"}; status is pass, fail,
                      skipped, running or escalated; only stages that appear in the ledger are listed
    counts           {"retries", "escalations", "nudges", "pauses", "operator_commands"};
                      operator_commands is the number of control.done-* files in the run directory
    pause            null, or {"reason", "kind", "detector", "stage"}; kind is watchdog, operator,
                      stage_failed, blocked, budget, full_port or other
    last_events      the last 5 entries as {"seq", "ts", "event", "stage", "summary"}
    control_pending  the word in the `control` file, or null
    disk             {"run_dir_free_gb", "home_free_gb"} (decimal GB, as the supervisor counts)
    leases           one line per board from `gozer status`, or [] when gozer is missing or fails
    hint             the `next:` text
    now              the time the status was taken, seconds since the epoch
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from orchard import orchard_view, ui
from orchard.defaults import TEST_DISK_GB
from orchard.ledger import LedgerCorrupt, read_entries
from orchard.stages import STAGES, ledger_ts, run_progress

EXIT_OK, EXIT_BAD = 0, 2
LAST_EVENTS = 5
SUMMARY_CHARS = 100         # one event summary, so the whole block stays short
QUIET_STUCK_S = 5 * 3600.0  # choice: longer than the longest stage budget (4 h) plus margin
DISK_STOP_GB = TEST_DISK_GB  # the same 40 GB the supervisor wants before a hardware test
STATES = ("running", "paused", "ready-for-operator-review", "aborted", "stopped-or-crashed",
          "not-started")


# ---- outside signals (each one is injectable in `collect`, so tests need no real process) ------

def _pid_alive(pid: int) -> bool:
    """True if the process exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                  # it exists; it belongs to someone else
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True                  # no /proc: the kill probe above is all we have


def _supervisor_alive(pid: int) -> bool:
    """A live process whose command line mentions the supervisor. Without /proc the command line
    cannot be read, and a live pid is accepted."""
    if not _pid_alive(pid):
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return True
    return "orchard" in cmdline or "supervisor" in cmdline


def _read_pid_file(run_dir) -> int | None:
    try:
        return int((Path(run_dir) / "supervisor.pid").read_text().strip())
    except (OSError, ValueError):
        return None


def _lock_holder(run_dir) -> int | None:
    """The pid that holds the flock on ledger.jsonl.lock, from /proc/locks, or None. Reading
    /proc/locks takes no lock and does not touch the lock file (which is only stat-ed)."""
    try:
        st = (Path(run_dir) / "ledger.jsonl.lock").stat()
        lines = Path("/proc/locks").read_text().splitlines()
    except OSError:
        return None
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}"
    for line in lines:
        parts = line.split()
        # "1: FLOCK  ADVISORY  WRITE 12345 08:02:1234567 0 EOF" (a "->" marks a waiter: skip it)
        if len(parts) >= 6 and parts[1] == "FLOCK" and parts[0] != "->":
            major_minor_inode = parts[5].split(":")
            if len(major_minor_inode) == 3 and ":".join(major_minor_inode[:2]).lower() == dev.lower() \
                    and int(major_minor_inode[2]) == st.st_ino:
                return int(parts[4])
    return None


def _disk_free_gb(path) -> float:
    return round(shutil.disk_usage(path).free / 1e9, 1)


def _gozer_status() -> str:
    """`gozer status` output, or "" when gozer is missing, slow or fails. Read-only."""
    exe = shutil.which("gozer")
    if not exe:
        return ""
    try:
        done = subprocess.run([exe, "status"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def lease_lines(text: str) -> list[str]:
    """One line per board from `gozer status` output:
    `board <id>: <n> chips <state> [<owner>]`. Lines that do not look like a board or a chip
    are ignored, so a changed output format gives fewer lines and no error."""
    boards: list[tuple[str, list[list[str]]]] = []
    for line in text.splitlines():
        parts = line.split()
        if parts[:1] == ["board"] and len(parts) >= 2:
            boards.append((parts[1], []))
        elif parts[:1] == ["chip"] and len(parts) >= 4 and boards:
            boards[-1][1].append(parts[3:])
    out = []
    for board, chips in boards:
        states = sorted({c[0] for c in chips})
        owners = sorted({" ".join(c[1:]) for c in chips if len(c) > 1})
        tail = f" ({'; '.join(owners)})" if owners else ""
        out.append(f"board {board}: {len(chips)} chips {'/'.join(states) or 'unknown'}{tail}")
    return out


# ---- reading the ledger ------------------------------------------------------------------------

def classify_pause(data: dict) -> tuple[str, str | None]:
    """(kind, detector) of a pause decision. The detector is named only for a watchdog pause."""
    reason = str(data.get("reason") or "paused")
    findings = data.get("findings")
    if data.get("watchdog") and isinstance(findings, list) and findings:
        return "watchdog", str(findings[0].get("detector")) if isinstance(findings[0], dict) else None
    if reason == "operator":
        return "operator", None
    if reason.startswith("blocked:"):
        return "blocked", None
    if "failed after escalation" in reason or "is not escalated" in reason:
        return "stage_failed", None
    if "full port" in reason:
        return "full_port", None
    if "cap" in reason and ("since" in reason or "lasted" in reason):
        return "budget", None
    return "other", None


def _one_line(text, limit=SUMMARY_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summarize(entry: dict) -> str:
    """A one-line summary of a ledger entry. Unknown shapes fall back to the event's data keys."""
    ev, d = entry["event"], entry["data"]
    if ev == "decision":
        what = d.get("decision", "?")
        extra = d.get("reason") or d.get("by") or ""
        return _one_line(f"{what}: {extra}" if extra else what)
    if ev == "notice":
        findings = d.get("findings")
        if isinstance(findings, list) and findings and isinstance(findings[0], dict):
            return _one_line("watchdog " + str(findings[0].get("detector")) + ": "
                             + str(findings[0].get("summary", "")))
        return _one_line(d.get("what") or d.get("reason") or "notice")
    if ev == "stage_start":
        return f"stage {entry['stage']} started"
    if ev == "stage_end":
        reasons = d.get("reasons")
        tail = f": {reasons[0]}" if isinstance(reasons, list) and reasons else ""
        return _one_line(f"stage {entry['stage']} {d.get('result')}{tail}")
    if ev == "retry":
        return _one_line("nudge: " + str(d.get("message", "")))
    if ev == "escalate":
        return _one_line(f"stage {entry['stage']} escalated")
    if ev in ("park", "restore"):
        return _one_line(f"{ev} {d.get('step', '')}".strip())
    if ev == "evidence":
        return _one_line(f"evidence {d.get('path', '')}")
    if ev == "run_start":
        return _one_line(f"run started for {d.get('model')}")
    return _one_line(f"{ev} " + ",".join(sorted(d)))


def stage_rows(entries: list[dict], now: float) -> list[dict]:
    """One row per stage that appears in the ledger: status of its latest event, wall time summed
    over its attempts (a running attempt counts up to `now`), and the attempt count."""
    names = {s.number: s.name for s in STAGES}
    rows: dict[int, dict] = {}
    started: dict[int, float] = {}
    for e in entries:
        n = e["stage"]
        if not isinstance(n, int) or e["event"] not in ("stage_start", "stage_end"):
            continue
        row = rows.setdefault(n, {"stage": n, "name": names.get(n), "status": None, "wall_s": 0,
                                  "attempts": 0})
        t = ledger_ts(e)
        if e["event"] == "stage_start":
            started[n] = t
            row["attempts"] += 1
            row["status"] = "running"
        else:
            row["wall_s"] += int(t - started.pop(n)) if n in started else 0
            result = e["data"].get("result")
            row["status"] = "escalated" if result == "escalate" else result
    for n, t in started.items():
        rows[n]["wall_s"] += max(0, int(now - t))
    return [rows[n] for n in sorted(rows)]


def _last_pause(entries: list[dict]) -> dict | None:
    for e in reversed(entries):
        if e["event"] == "decision" and e["data"].get("decision") == "pause":
            kind, detector = classify_pause(e["data"])
            return {"reason": str(e["data"].get("reason") or "paused"), "kind": kind,
                    "detector": detector, "stage": e["stage"]}
    return None


# ---- the hint table ----------------------------------------------------------------------------
#
# Each row is (name, test, text). `test` takes the hint inputs (state, pause, disk, control_pending,
# pauses_at_stage, quiet_s) and the first row whose test is true supplies the text, formatted with
# the same inputs. Order matters: terminal states come first, then the disk rule, then the rest.
# To change what the operator is told, change a row; the printer does not know any hint text.

def _paused(f, *kinds):
    return f["state"] == "paused" and f["pause"] is not None and (not kinds or f["pause"]["kind"] in kinds)


HINT_RULES = (
    ("aborted", lambda f: f["state"] == "aborted",
     "terminal, a human decides. Do nothing."),
    ("ready", lambda f: f["state"] == "ready-for-operator-review",
     "run the post-run checks in the operator runbook; never publish."),
    ("not started", lambda f: f["state"] == "not-started",
     "start the run with the run script a human gave you; do not edit its flags."),
    ("disk low", lambda f: f["disk"]["run_dir_free_gb"] < DISK_STOP_GB,
     "disk is below 40 GB; stop and ask a human."),
    ("crashed", lambda f: f["state"] == "stopped-or-crashed",
     "restart with the run script, then send control resume if paused; do not abort."),
    ("control waiting", lambda f: f["state"] == "paused" and bool(f["control_pending"]),
     "a control word is waiting; wait 30 seconds and run status again."),
    ("same stage twice", lambda f: _paused(f) and f["pauses_at_stage"] >= 2,
     "the same stage has paused {pauses_at_stage} times; stop and ask a human."),
    ("needs a human", lambda f: _paused(f, "full_port", "blocked", "budget", "operator"),
     "this pause needs a decision; stop and ask a human."),
    ("resume once", lambda f: f["state"] == "paused",
     "resume once; if it pauses again at the same stage, stop and ask a human."),
    ("quiet", lambda f: f["quiet_s"] >= QUIET_STUCK_S,
     "no ledger event for over 5 hours; the supervisor may be stuck; stop and ask a human."),
    ("running", lambda f: True,
     "running; nothing to do. Wait: sleep 300, then run status again."),
)


def hint_for(inputs: dict) -> str:
    for _name, test, text in HINT_RULES:
        if test(inputs):
            return text.format_map(inputs)
    return "stop and ask a human."          # unreachable while the last row accepts everything


# ---- putting it together -----------------------------------------------------------------------

def collect(run_dir, *, now: float | None = None, pid_alive=_supervisor_alive, lock_holder=_lock_holder,
            disk_free_gb=_disk_free_gb, gozer_status=_gozer_status, home=None) -> dict:
    """All the facts as one dict (the JSON object). Raises LedgerCorrupt for a bad chain.
    Every outside signal is a parameter, so a test supplies its own."""
    run_dir = Path(run_dir)
    now = time.time() if now is None else now
    ledger_path = run_dir / "ledger.jsonl"
    entries = read_entries(ledger_path)             # may raise LedgerCorrupt; takes no lock
    progress = run_progress(entries)

    # Is a supervisor alive? The pid file first, then the holder of the ledger lock.
    pid = _read_pid_file(run_dir)
    sup = {"pid": pid, "alive": bool(pid and pid_alive(pid)), "source": "supervisor.pid" if pid else "none"}
    if not sup["alive"]:
        holder = lock_holder(run_dir)
        if holder:
            sup = {"pid": holder, "alive": True, "source": "ledger lock"}
        elif pid is None:
            sup = {"pid": None, "alive": False, "source": "none"}

    if not progress.started:
        state = "not-started"
    elif progress.aborted:
        state = "aborted"
    elif progress.finished:
        state = "ready-for-operator-review"
    elif not sup["alive"]:
        state = "stopped-or-crashed"
    else:
        state = "paused" if progress.paused is not None else "running"

    pause = _last_pause(entries) if progress.paused is not None else None
    pauses_at_stage = 0
    if pause is not None and pause["stage"] is not None:
        pauses_at_stage = sum(1 for e in entries if e["event"] == "decision" and e["stage"] == pause["stage"]
                              and e["data"].get("decision") == "pause")

    stages = stage_rows(entries, now)
    current = progress.open_stage if progress.open_stage is not None else progress.next_stage
    attempt = sum(1 for e in entries if e["event"] == "stage_start" and e["stage"] == current)
    names = {s.number: s.name for s in STAGES}

    control_word = None
    try:
        control_word = (run_dir / "control").read_text().strip() or None
    except OSError:
        pass

    home = Path(home) if home is not None else Path.home()
    disk = {"run_dir_free_gb": disk_free_gb(run_dir), "home_free_gb": disk_free_gb(home)}
    count = lambda pred: sum(1 for e in entries if pred(e))   # noqa: E731
    counts = {
        "retries": count(lambda e: e["event"] == "retry"),
        "escalations": count(lambda e: e["event"] == "escalate"),
        "nudges": count(lambda e: e["event"] == "retry" and e["data"].get("rung") == "nudge"),
        "pauses": count(lambda e: e["event"] == "decision" and e["data"].get("decision") == "pause"),
        "operator_commands": len(list(run_dir.glob("control.done-*"))),
    }
    quiet_s = now - ledger_ts(entries[-1]) if entries else 0.0
    inputs = {"state": state, "pause": pause, "disk": disk, "control_pending": control_word,
              "pauses_at_stage": pauses_at_stage, "quiet_s": quiet_s}
    return {
        "ledger": {"ok": True, "entries": len(entries), "error": None},
        "model": progress.run_start.get("model") if progress.run_start else None,
        "run_dir": str(run_dir.resolve()),
        "run_name": run_dir.resolve().name,
        "state": state,
        "supervisor": sup,
        "stage": {"current": current, "name": names.get(current), "attempt": attempt},
        "stages": stages,
        "counts": counts,
        "pause": pause,
        "last_events": [{"seq": e["seq"], "ts": e["ts"], "event": e["event"], "stage": e["stage"],
                         "summary": summarize(e)} for e in entries[-LAST_EVENTS:]],
        "control_pending": control_word,
        "disk": disk,
        "leases": lease_lines(gozer_status()),
        "hint": hint_for(inputs),
        "now": now,
    }


def _hms(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m{rem % 60:02d}s"


def render(f: dict) -> str:
    """The plain-text block. Every line is clipped by the summaries, so the block stays short."""
    sup = f["supervisor"]
    if sup["pid"] is None:
        who = "no supervisor pid recorded and no ledger lock holder"
    else:
        who = f"supervisor pid {sup['pid']} {'alive' if sup['alive'] else 'not alive'} (from {sup['source']})"
    stage = f["stage"]
    lines = [
        f"ledger: ok ({f['ledger']['entries']} entries, chain verified)",
        f"model: {f['model']}",
        f"run: {f['run_name']}",
        f"state: {f['state']}",
        f"decided by: {who}",
        "stage: " + (f"{stage['current']} {stage['name']}, attempt {stage['attempt']}"
                     if stage["current"] is not None else "none left"),
    ]
    c = f["counts"]
    lines.append(f"counts: retries {c['retries']}, escalations {c['escalations']}, nudges {c['nudges']}, "
                 f"pauses {c['pauses']}, operator commands {c['operator_commands']}")
    if f["pause"]:
        p = f["pause"]
        lines.append(f"paused at stage {p['stage']} by {p['detector'] or p['kind']}:")
        lines.append("  " + _one_line(p["reason"], 200))
    if f["control_pending"]:
        lines.append(f"control: {f['control_pending']} (waiting to be read by the supervisor)")
    d = f["disk"]
    lines.append(f"disk free: run dir {d['run_dir_free_gb']} GB, home {d['home_free_gb']} GB")
    lines += [f"lease: {s}" for s in f["leases"]]
    lines.append("stages:")
    lines += [f"  {r['stage']} {r['name']}: {r['status']}, {_hms(r['wall_s'])}"
              + (f", {r['attempts']} attempts" if r["attempts"] > 1 else "") for r in f["stages"]]
    lines.append(f"last {len(f['last_events'])} events:")
    lines += [f"  {e['seq']} {e['ts'][11:19]} {e['event']}: {e['summary']}" for e in f["last_events"]]
    lines.append("next: " + f["hint"])
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m orchard.supervisor status",
                                description="read-only status of a run (takes no lock, changes nothing)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--json", action="store_true", help="print the facts as one JSON object")
    p.add_argument("--style", choices=ui.STYLES, default="auto",
                   help="auto: colour and emoji on a capable terminal, plain text otherwise (default); "
                        "pretty: always decorate; plain: never decorate. --json is never styled")
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"refused: {run_dir} is not a directory", file=sys.stderr)
        return EXIT_BAD
    try:
        facts = collect(run_dir)
    except LedgerCorrupt as exc:
        if args.json:
            print(json.dumps({"ledger": {"ok": False, "entries": None, "error": str(exc)}}, indent=2))
        else:
            print(f"ledger: CORRUPT ({exc}). Do not touch the run. Stop and ask a human.")
        return EXIT_BAD
    if args.json:
        print(json.dumps(facts, indent=2, sort_keys=True))
        return EXIT_OK
    style = ui.detect(sys.stdout, os.environ, args.style)
    print(orchard_view.render_pretty(facts, style) if style.pretty else render(facts))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
