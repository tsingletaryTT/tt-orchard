# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Say what a run is doing while it does it.

A run used to print almost nothing: the ledger and the agent logs held everything, and a person who
wanted to know why a stage had stopped had to read both. The narrator follows them and prints one
plain line for each thing that starts, ends or is tried, with the role that did it:

    05:20:31  orchardist   stage 2 (graft): starting
    05:20:40  grafter      reads stages/2/evidence/swap-check.json
    05:21:02  sheepdog     watchdog repeated_tool_call: the same tool call (write_file) 3 times in a row

It reads the ledger without a lock (`ledger.read_entries`) and the agent logs as plain files, and it
writes nothing. A bug in it must not stop a run, so the background thread gives up on its first error
and says so once.

`what_was_tried` and `how_to_unblock` are for the moment a run stops. They answer the two questions a
blocked run left open: what was attempted, and what a person can do about it.
"""
from __future__ import annotations

import json
import textwrap
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from orchard import lexicon
from orchard.ledger import LedgerCorrupt, read_entries
from orchard.ui import Style

ACTOR_ICON = {"orchardist": "🧑‍🌾", "grafter": "✂️", "head grower": "🌳", "seasonal hand": "🧤",
              "sheepdog": "🐕", "almanac": "📖"}
ACTOR_ROLE = {"orchardist": "accent", "grafter": "good", "head grower": "title", "seasonal hand": "dim",
              "sheepdog": "warn", "almanac": "dim"}
TIER_ACTOR = {"large": "head grower", "small": "grafter", "cpu": "seasonal hand"}
WIDTH = 118                       # a printed line wraps here, with a hanging indent
RESULT_CHARS = 110
TEXT_CHARS = 160
HEARTBEAT_S = 60.0
POLL_S = 1.0


class Line(NamedTuple):
    actor: str
    text: str
    bad: bool = False             # a failure or a stop: painted in the warning colour


def _clip(text, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def duration(seconds: float) -> str:
    s = int(round(seconds))
    if s >= 3600:
        h, m = divmod(s // 60, 60)
        return f"{h}h{m:02d}m" if m else f"{h}h"
    if s >= 60:
        m, r = divmod(s, 60)
        return f"{m}m{r:02d}s" if r else f"{m}m"
    return f"{s}s"


def _stage(n) -> str:
    name = lexicon.STAGES[n].name if isinstance(n, int) and n in lexicon.STAGES else None
    return f"stage {n} ({name})" if name else f"stage {n}"


def _reasons(reasons, keep: int = 2) -> str:
    if not isinstance(reasons, list) or not reasons:
        return ""
    shown = "; ".join(_clip(r, 200) for r in reasons[:keep])
    more = len(reasons) - keep
    return shown + (f" (+{more} more)" if more > 0 else "")


# ---- ledger entries ---------------------------------------------------------------------------

def _decision(e: dict, d: dict) -> list[Line]:
    what, st = d.get("decision", "?"), e["stage"]
    o = "orchardist"
    if what == "coder starting":
        s = d.get("server") or {}
        return [Line(o, f"starting the coder {s.get('target', '?')} on port {s.get('port', '?')}; a cold "
                        "boot can take 10 minutes or more")]
    if what == "coder started":
        return [Line(o, "the coder is ready (it answered the canary question)")]
    if what == "coder adopted":
        return [Line(o, "using the coder an earlier run kept up (no boot); checking its canary answer")]
    if what == "kept coder not used":
        return [Line(o, f"not using the kept coder: {d.get('reason', '?')}")]
    if what == "coder left up":
        return [Line(o, "left the coder serving for the next run (`tt-orchard coder stop` stops it)")]
    if what == "lease granted":
        return [Line(o, f"lease {d.get('lease_id', '?')} granted on {', '.join(d.get('chips') or []) or '?'} "
                        f"(waited {d.get('waited_s', 0)}s)")]
    if what == "agent step":
        who = TIER_ACTOR.get(d.get("tier"), "grafter")
        skill = Path(str(d.get("skill", ""))).stem
        return [Line(who, f"{d.get('model', '?')} starts the {d.get('phase', '?')} step"
                          + (f" ({skill})" if skill else "") + (" [escalated]" if d.get("escalated") else ""))]
    if what == "hardware test started":
        box = f" on {d['where'][4:]}" if str(d.get("where", "")).startswith("lab ") else ""
        return [Line(o, f"hardware test started{box} on {', '.join(d.get('chips') or []) or '?'}: "
                        f"{d.get('command', '?')} (deadline {duration(d.get('deadline_s') or 0)})")]
    if what == "copying files to the lab":
        return [Line(o, f"copying the run's files to {str(d.get('where', 'lab'))[4:] or 'the lab'}")]
    if what == "files copied to the lab":
        repos = [r.removeprefix("models--").replace("--", "/") for r in d.get("repos") or []]
        return [Line(o, f"copied to {str(d.get('where', 'lab'))[4:] or 'the lab'} in {d.get('seconds', '?')} s"
                        + (f" (with {', '.join(repos)})" if repos else ""))]
    if what == "gate feedback":
        return [Line(o, "the stage gate rejected the result; giving the agent one more go with these reasons: "
                        + _reasons(d.get("reasons"), 3), True)]
    if what.startswith("no gate feedback"):
        return [Line(o, f"not asking the agent to retry: {d.get('problem', what)}", True)]
    if what == "pause":
        return [Line(o, f"paused: {d.get('reason', '?')}", True)]
    if what == "blocked":
        return [Line(o, f"BLOCKED ({d.get('code', '?')}): {d.get('reason', '?')}", True)]
    if what == "resume":
        return [Line(o, f"resumed ({d.get('by', 'operator')})")]
    if what == "hardware phase":
        if d.get("action") == "lab":
            return [Line(o, f"hardware phase on the lab {str(d.get('where', ''))[4:]}".rstrip())]
        return [Line(o, f"hardware phase: {d.get('action', '?')}")]
    if what in ("test lease taken", "test lease released"):
        lease = d.get("test_lease") or {}
        box = f" on {d['where'][4:]}" if str(d.get("where", "")).startswith("lab ") else ""
        return [Line(o, f"{what}{box}" + (f": {', '.join(lease.get('chips') or [])}" if lease else ""))]
    if what == "ready for operator review":
        return [Line(o, f"ready for operator review: the bundle is at {d.get('bundle', '?')}")]
    if what == "tier substituted":
        return [Line(o, _clip(d.get("note", what), 220))]
    extra = d.get("reason") or d.get("note") or d.get("by") or ""
    return [Line(o, f"{what}" + (f": {_clip(extra, 200)}" if extra else ""))]


def _evidence(e: dict, d: dict) -> list[Line]:
    what = d.get("what")
    o = "orchardist"
    if what == "hardware test":
        rc = d.get("returncode")
        if d.get("timed_out"):
            return [Line(o, "hardware test failed: it timed out and was stopped", True)]
        if rc == 0:
            return [Line(o, "hardware test passed (exit 0)")]
        return [Line(o, f"hardware test failed (exit {rc})", True)]
    if what == "hardware test list":
        return [Line(o, "will test these chip counts: " + ", ".join(str(c) for c in d.get("configs") or []))]
    if what == "transcript":
        return [Line(o, f"agent step finished: {d.get('status', '?')} after {d.get('turns', '?')} turns")]
    return []


def lines_for(e: dict) -> list[Line]:
    """The lines to print for one ledger entry. Routine bookkeeping (evidence files) prints nothing."""
    ev, d, st = e["event"], e["data"], e["stage"]
    if ev == "run_start":
        c = d.get("coder") or {}
        return [Line("orchardist", f"run started for {d.get('model', '?')}; coder {c.get('target', '?')}")]
    if ev == "stage_start":
        if d.get("skip"):
            return [Line("orchardist", f"{_stage(st)}: skipped")]
        tail = (" [escalated attempt]" if d.get("escalated") else "") + (" [resumed]" if d.get("resumed") else "")
        return [Line("orchardist", f"{_stage(st)}: starting{tail}")]
    if ev == "stage_end":
        result = d.get("result", "?")
        why = d.get("reasons") or ([d["reason"]] if d.get("reason") else [])
        return [Line("orchardist", f"{_stage(st)}: {result}" + (f": {_reasons(why)}" if why else ""),
                     result not in ("pass", "skipped"))]
    if ev == "decision":
        return _decision(e, d)
    if ev == "evidence":
        return _evidence(e, d)
    if ev == "escalate":
        f = (d.get("findings") or [{}])[0]
        why = f"watchdog {f.get('detector')}: {f.get('summary')}" if f else "the step did not finish"
        return [Line("sheepdog" if d.get("watchdog") else "orchardist",
                     f"{_stage(st)} handed to the next tier. {why}", True)]
    if ev == "retry":
        return [Line("sheepdog", f"nudged the agent: {_clip(d.get('message', ''), 200)}")]
    if ev == "notice":
        f = (d.get("findings") or [None])[0]
        text = (f"watchdog {f.get('detector')}: {f.get('summary')}" if isinstance(f, dict)
                else str(d.get("what") or d.get("reason") or "notice"))
        return [Line("sheepdog" if d.get("watchdog") else "orchardist", _clip(text, 220), True)]
    if ev in ("park", "restore"):
        return [Line("orchardist", f"{ev}: {d.get('step', '?')}")]
    if ev == "measurement":
        return [Line("orchardist", f"measured {d.get('name')} = {d.get('value')} {d.get('unit', '')}".rstrip())]
    return [Line("orchardist", f"{ev}: {','.join(sorted(d))}")]


# ---- the agent's turns ------------------------------------------------------------------------

class TurnNarrator:
    """Lines for the records of an agent log. A tool call is shown when it is made, and its result when
    the next turn arrives, because the log holds a result only in the following request."""

    def __init__(self):
        self._pending: dict[str, tuple[str, str]] = {}

    def lines_for(self, rec: dict, *, actor: str) -> list[Line]:
        out: list[Line] = []
        for m in rec.get("sent") or []:
            if m.get("role") == "tool" and m.get("tool_call_id") in self._pending:
                name, _ = self._pending.pop(m["tool_call_id"])
                out.append(self._result(actor, name, str(m.get("content", ""))))
        got = rec.get("received") or {}
        calls = got.get("tool_calls") or []
        content = (got.get("content") or "").strip()
        if content:
            out.append(Line(actor, "says: " + _clip(content, TEXT_CHARS)))
        for c in calls:
            fn = c.get("function") or {}
            name, args = fn.get("name", "?"), _args(fn.get("arguments"))
            self._pending[c.get("id", "")] = (name, "")
            out.append(Line(actor, _call_text(name, args)))
        if not calls and not content:
            if rec.get("finish_reason") == "length":
                out.append(Line(actor, "reply cut off at the token limit, with no tool call", True))
            else:
                out.append(Line(actor, "empty reply (no text, no tool call)", True))
        return out

    @staticmethod
    def _result(actor: str, name: str, content: str) -> Line:
        if name == "read_file":
            return Line(actor, f"  got {len(content)} characters")
        first, _, rest = content.partition("\n")
        if first.startswith("exit "):
            body = next((l for l in rest.splitlines() if l.strip()), "")
            ok = first.strip() == "exit 0"
            text = ("  " + ("exit 0" if ok else "FAILED " + first.strip())) + (f": {_clip(body, RESULT_CHARS)}" if body else "")
            return Line(actor, text, not ok)
        return Line(actor, "  " + _clip(first, RESULT_CHARS))


def _args(raw) -> dict:
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}


def _call_text(name: str, args: dict) -> str:
    if name == "shell":
        return "runs: " + _clip(args.get("command", ""), 200)
    if name == "read_file":
        return f"reads {args.get('path', '?')}"
    if name == "write_file":
        return f"writes {args.get('path', '?')} ({len(str(args.get('content', '')))} characters)"
    return f"calls {name}"


# ---- the narrator -----------------------------------------------------------------------------

DONE_DECISIONS = ("ready for operator review", "blocked")


class Narrator:
    def __init__(self, run_dir, style: Style, out, *, replay=False, clock=time.time,
                 heartbeat_s: float = HEARTBEAT_S, tz=None):
        self.run = Path(run_dir)
        self.style, self.out = style, out
        self.replay = replay                       # False, True (everything) or N (the last N entries)
        self.clock, self.heartbeat_s, self.tz = clock, heartbeat_s, tz
        self._first = True
        self._seq = 0
        self._offsets: dict[Path, int] = {}
        self._turns = TurnNarrator()
        self._stage_actor: dict = {}
        self._activity: tuple[str, float, int | None, bool] | None = None   # text, since, stage, hardware test
        self._last_print = clock()
        self.finished = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._broken = False

    # -- output --
    def _stamp(self, ts: str | None) -> str:
        if ts:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                return dt.astimezone(self.tz).strftime("%H:%M:%S")
            except ValueError:
                pass
        return datetime.now(self.tz).strftime("%H:%M:%S")

    def _emit(self, line: Line, ts: str | None = None) -> None:
        prefix = f"{self._stamp(ts)}  {self.style.icon(ACTOR_ICON.get(line.actor, ''))}"
        label = f"{line.actor:<13}"
        head = prefix + self.style.paint(label, "bad" if line.bad else ACTOR_ROLE.get(line.actor, "dim")) + " "
        plain_width = len(self._stamp(ts)) + 2 + (3 if self.style.emoji else 0) + len(label) + 1
        body = textwrap.wrap(line.text, max(40, WIDTH - plain_width), break_on_hyphens=False,
                             break_long_words=True) or [""]
        self.out.write(head + body[0] + "\n")
        for more in body[1:]:
            self.out.write(" " * plain_width + more + "\n")
        self.out.flush()
        self._last_print = self.clock()

    # -- reading --
    def _ledger(self) -> list[dict]:
        try:
            return read_entries(self.run / "ledger.jsonl")
        except (OSError, LedgerCorrupt, ValueError):
            return []

    def _track(self, e: dict) -> None:
        ev, d, st = e["event"], e["data"], e["stage"]
        now = self.clock()
        what = d.get("decision") if ev == "decision" else d.get("what") if ev == "evidence" else None
        if ev == "decision" and what == "agent step":
            self._stage_actor[st] = TIER_ACTOR.get(d.get("tier"), "grafter")
            who = self._stage_actor[st]
            self._activity = (f"waiting for the {who}'s next reply", now, st, False)
        elif ev == "decision" and what == "coder starting":
            s = d.get("server") or {}
            self._activity = (f"waiting for the coder {s.get('target', '?')} to boot", now, None, False)
        elif ev == "decision" and what == "hardware test started":
            self._activity = (f"hardware test running: {_clip(d.get('command', ''), 80)}", now, st, True)
        elif (ev == "evidence" and what in ("hardware test", "transcript")) or ev == "stage_end" \
                or (ev == "decision" and what in ("coder started", "pause", "blocked", "hardware released",
                                                  "ready for operator review")):
            self._activity = None
        if ev == "decision" and what in DONE_DECISIONS:
            self.finished = True
        elif ev == "decision" and what == "resume" or ev == "stage_start":
            self.finished = False

    def _logs(self) -> None:
        for f in sorted((self.run / "stages").glob("*/log/*.jsonl")):
            if f not in self._offsets:
                self._offsets[f] = f.stat().st_size if self._first_logs else 0
            try:
                with open(f, "rb") as fh:
                    fh.seek(self._offsets[f])
                    data = fh.read()
            except OSError:
                continue
            end = data.rfind(b"\n")
            if end < 0:
                continue
            self._offsets[f] += end + 1
            stage = _stage_of(f)
            actor = self._stage_actor.get(stage, "grafter")
            for raw in data[: end + 1].splitlines():
                try:
                    rec = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    for line in self._turns.lines_for(rec, actor=actor):
                        self._emit(line)

    _first_logs = False

    def poll(self) -> None:
        with self._lock:
            entries = self._ledger()
            if self._first:
                # What is already in the ledger is history, shown only as far back as `replay` asks.
                n = len(entries)
                shown = n if self.replay is True else (0 if not self.replay else min(n, int(self.replay)))
                self._seq = entries[n - shown - 1]["seq"] if n - shown > 0 else 0
                for e in entries:
                    if e["seq"] <= self._seq:
                        self._track(e)
                # Everything already in the agent logs is history. A file that appears later is new.
                self._first_logs = True
                self._first = False
                self._logs()
                self._first_logs = False
            else:
                self._logs()
            for e in entries:
                if e["seq"] > self._seq:
                    self._track(e)
                    for line in lines_for(e):
                        self._emit(line, e.get("ts"))
                    if e["event"] == "evidence" and e["data"].get("what") == "hardware test" \
                            and e["data"].get("returncode") != 0 and e["stage"] is not None:
                        # The reason is in the test's output, which the ledger only points to.
                        for text in _output_tail(self.run, e["stage"], 3):
                            self._emit(Line("orchardist", "  last output: " + _clip(text, 150), True), e.get("ts"))
                    self._seq = e["seq"]
            self._heartbeat()

    def _heartbeat(self) -> None:
        if self._activity is None or self.clock() - self._last_print < self.heartbeat_s:
            return
        text, since, stage, hardware = self._activity
        msg = f"still {text} ({duration(self.clock() - since)})"
        if hardware and stage is not None:
            tail = _last_output_line(self.run, stage)
            if tail:
                msg += f"; last output: {_clip(tail, 100)}"
        self._emit(Line("orchardist", msg))

    # -- the background thread --
    def start(self) -> None:
        def loop():
            try:
                while not self._stop.wait(POLL_S):
                    self.poll()
            except Exception as exc:               # a narrator bug must not stop a run
                self._broken = True
                self.out.write(f"(live narration stopped: {type(exc).__name__}: {exc}; the run goes on)\n")
        self._thread = threading.Thread(target=loop, name="orchard-narrator", daemon=True)
        try:
            self.poll()
        except Exception as exc:
            self._broken = True
            self.out.write(f"(live narration stopped: {type(exc).__name__}: {exc}; the run goes on)\n")
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=5)
        if not self._broken:
            self.poll()


def _stage_of(log_file: Path):
    name = log_file.parent.parent.name            # "2" or "1.partial-1"
    head = name.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _last_output_line(run: Path, stage: int) -> str | None:
    found = sorted((p for p in (run / "stages").glob(f"{stage}*/**/hw-test-output.txt")),
                   key=lambda p: p.stat().st_mtime)
    if not found:
        return None
    lines = [l for l in found[-1].read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    return lines[-1] if lines else None


# ---- when a run stops -------------------------------------------------------------------------

def what_was_tried(run_dir, *, stage, limit: int = 8) -> list[str]:
    """What was attempted in `stage`, newest last: the watchdog's findings and the hardware test's result
    from the ledger, then the agent's last `limit` actions from its logs."""
    run = Path(run_dir)
    try:
        entries = read_entries(run / "ledger.jsonl")
    except (OSError, LedgerCorrupt, ValueError):
        entries = []
    start = max((i for i, e in enumerate(entries) if e["event"] == "stage_start" and e["stage"] == stage), default=0)
    out: list[str] = []
    events, failed_test = [], False
    for e in entries[start:]:
        if e["stage"] != stage:
            continue
        if e["event"] in ("escalate", "retry", "notice") or (
                e["event"] == "decision" and e["data"].get("decision") in ("gate feedback", "pause")
                or str(e["data"].get("decision", "")).startswith("no gate feedback")):
            events += [l.text for l in lines_for(e)]
        if e["event"] == "evidence" and e["data"].get("what") == "hardware test":
            events += [l.text for l in lines_for(e)]
            failed_test = e["data"].get("returncode") != 0
    if events:
        out += ["The supervisor and watchdog recorded:"] + [f"  - {t}" for t in events]
    if failed_test:
        tail = _output_tail(run, stage)
        if tail:
            out += ["The hardware test's last output:"] + [f"  | {_clip(l, 160)}" for l in tail]
    actions = _agent_actions(run, stage)[-limit:]
    if actions:
        out += [f"The agent's last {len(actions)} actions:"] + [f"  - {a}" for a in actions]
    return out or [f"Nothing was recorded for stage {stage}: no agent turns and no hardware test."]


def _output_tail(run: Path, stage, n: int = 4) -> list[str]:
    found = sorted((run / "stages").glob(f"{stage}*/**/hw-test-output.txt"), key=lambda p: p.stat().st_mtime)
    if not found:
        return []
    return [l for l in found[-1].read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()][-n:]


def _agent_actions(run: Path, stage) -> list[str]:
    files = sorted((run / "stages").glob(f"{stage}*/log/*.jsonl"), key=lambda p: (p.stat().st_mtime, p.name))
    acts: list[str] = []
    for f in files:
        try:
            raw = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in raw:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            for c in ((rec.get("received") or {}).get("tool_calls") or []) if isinstance(rec, dict) else []:
                fn = c.get("function") or {}
                acts.append(_call_text(fn.get("name", "?"), _args(fn.get("arguments"))))
    return acts


UNBLOCK = {
    "stage-failed": [
        "The stage's agent could not finish it, even after the next tier took over. Read the lists above, then the "
        "agent logs in stages/<N>/log and any stages/<N>/evidence/hw-test-output.txt.",
        "If a template or a gate rejected work that was right, fix the harness. Then run `tt-orchard bringup "
        "<MODEL>` again: a retry boots the coder again and starts the stage in a fresh directory.",
        "If the model needs code that does not exist yet, see needs-new-model-code."],
    "agent-stuck": [
        "The watchdog stopped an agent that was repeating itself or writing nothing. The last actions are listed above.",
        "An agent that reads logs in a loop usually has a failed hardware test behind it. Read "
        "stages/<N>/evidence/hw-test-output.txt yourself and fix that cause.",
        "Then run `tt-orchard bringup <MODEL>` again. The same cause will stop it again."],
    "retry-budget-spent": [
        "A cap was reached (escalations, cold coder boots or wall clock). `tt-orchard status <MODEL>` shows the counts.",
        "Find the stage that keeps failing in the list above and fix its cause first; a retry with the same "
        "cause spends the budget again. Then run `tt-orchard bringup <MODEL>`."],
    "disk-full": [
        "Run `df -h` on the run directory and on the Hugging Face cache. Stage 4 needs about 110 GB free on the run "
        "directory's filesystem.",
        "Failed attempts leave tensor caches (about 31 GB each) in stages/<N>.partial-*. Delete those from "
        "abandoned attempts, then run `tt-orchard bringup <MODEL>` again."],
    "hardware-unhealthy": [
        "Run `gozer status`. tt-orchard never clears a lease or resets a chip that it does not hold.",
        "A STALE lease whose owner is gone is cleared with `gozer reconcile`. A chip held by another agent "
        "needs that agent to finish. Do not run `tt-smi -r` by hand while others use the box.",
        "Then run `tt-orchard bringup <MODEL>` again."],
    "coder-unusable": [
        "The coder model did not serve or failed its canary question. Read stages/*/evidence/coder-canary-*.txt "
        "and the coder container's log (`docker logs` on the tt-model container).",
        "`python3 -m orchard.rolefit` checks a model for the agent role. Fix the coder, then run "
        "`tt-orchard bringup <MODEL>` again."],
    "needs-new-model-code": [
        "The model differs from the nearest supported model in a way that needs new model code (a full port). An "
        "unattended run cannot write that.",
        "Read stages/0/delta.json (`path_reasons`) for what differs. Choosing another nearest model or porting by "
        "hand is a person's decision; `tt-orchard bringup <MODEL>` blocks again until stage 0 finds weights-only."],
    "blocked": [
        "The supervisor needed something it could not get. The reason is quoted above.",
        "`tt-orchard status <MODEL>` shows the full state. Fix what the reason names, then run `tt-orchard "
        "bringup <MODEL>` again."],
    "unclassified": [
        "The run stopped for a reason this tool does not have a name for. The reason is quoted above.",
        "Read the ledger's last entries with `tt-orchard status <MODEL>`, fix what they show, then run "
        "`tt-orchard bringup <MODEL>` again."],
}


def how_to_unblock(code: str, model: str | None = None) -> list[str]:
    return [a.replace("<MODEL>", model or "MODEL") for a in UNBLOCK.get(code, UNBLOCK["unclassified"])]
