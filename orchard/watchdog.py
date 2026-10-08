# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Loop detection and the response ladder (spec section 7).

This module owns:
- `Event`, the normalised record every detector reads. The model proxy (plan 4) and the transcript
  reader (orchard/transcripts.py) both produce it.
- The detectors. Each is a small class whose `feed(event)` returns a `Finding` or None. Each one
  re-arms after it fires, so a loop that continues after a nudge fires again and the ladder climbs.
- The response ladder (added below the detectors).

Default thresholds come from replaying the recorded qwen-code loop and 30 quiet chats
(orchard/defaults.py).
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from orchard.adapters import AdapterError
from orchard.defaults import (IDENTICAL_N, LEASE_IDLE_S, LEASE_POLL_S, NO_EVIDENCE_S, REPEAT_TOOL_N,
                              RUNG_CAPS, THINKING_CAP, TURN_REPEAT_N, TURN_SHAPE_N,
                              WRITELESS_TURNS)

KINDS = frozenset({"response", "tool_call", "tool_result", "evidence", "ledger"})


@dataclass(frozen=True)
class Event:
    ts: float                         # seconds since the epoch
    agent: str
    kind: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    text_hash: str | None = None      # sha256 of the visible response text; None when it is empty
    duration_s: float | None = None
    tool: str | None = None
    args_hash: str | None = None
    output_hash: str | None = None
    had_tool_call: bool = False       # the response asked for at least one tool call
    name: str | None = None           # ledger event name, or evidence path
    stage: int | None = None
    # On a tool_result: True when the call wrote a file, False when a file-writing call was
    # refused or failed, None when the source does not say (transcripts, or a tool that does not
    # write files). AgentStep sets it for its write_file tool.
    wrote: bool | None = None
    # The agent step's model turn that produced this event: its response, the tool calls and
    # results that follow, and any evidence file they wrote. It counts up through a step and its
    # continuation. None when the source does not number turns (transcripts).
    turn: int | None = None
    # On a tool_call: the call's shape (command_shape of a shell command), so TurnRepeat can match
    # commands that differ only in a trailing argument. None for other tools and for transcripts.
    shape: str | None = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"event kind must be one of {sorted(KINDS)}, got {self.kind!r}")


@dataclass(frozen=True)
class Finding:
    detector: str
    agent: str
    ts: float
    summary: str
    evidence: dict = field(default_factory=dict, hash=False)
    pause: bool = False               # a budget cap: go straight to pause (spec section 10)
    notice_only: bool = False         # a supervisor matter (an idle lease): a notice, no rung
    nudge: str | None = None          # the text the nudge rung sends; None uses the repeat text

    def record(self) -> dict:
        return {"detector": self.detector, "agent": self.agent, "ts": self.ts,
                "summary": self.summary, "evidence": self.evidence, "pause": self.pause,
                "notice_only": self.notice_only}


class IdenticalResponses:
    """N responses in a row with the same text, or the same token counts and no tool call.

    The recorded loop was five calls with identical input and output token counts and an empty
    visible text, so the counts carry the signal. An empty text never matches by text: empty
    retries are common in quiet chats.
    """
    name = "identical_responses"

    def __init__(self, n: int = IDENTICAL_N):
        if n < 2:
            raise ValueError("n must be at least 2")
        self.n = n
        self._last: dict[str, Event] = {}
        self._count: dict[str, int] = {}

    @staticmethod
    def _same(a: Event, b: Event) -> bool:
        if a.text_hash is not None and a.text_hash == b.text_hash:
            return True
        return (not a.had_tool_call and not b.had_tool_call
                and a.input_tokens is not None and a.output_tokens is not None
                and (a.input_tokens, a.output_tokens) == (b.input_tokens, b.output_tokens))

    def feed(self, ev: Event) -> Finding | None:
        if ev.kind != "response":
            return None
        prev = self._last.get(ev.agent)
        count = self._count.get(ev.agent, 0) + 1 if prev is not None and self._same(prev, ev) else 1
        self._last[ev.agent] = ev
        if count >= self.n:
            self._count[ev.agent] = 0          # re-arm: the next n repeats fire again
            return Finding(self.name, ev.agent, ev.ts,
                           f"{self.n} identical responses in a row ({ev.input_tokens} input, "
                           f"{ev.output_tokens} output tokens)",
                           {"count": self.n, "input_tokens": ev.input_tokens,
                            "output_tokens": ev.output_tokens, "text_hash": ev.text_hash})
        self._count[ev.agent] = count
        return None


class ThinkingWithoutAction:
    name = "thinking_without_action"

    def __init__(self, cap: int = THINKING_CAP):
        self.cap = cap

    def feed(self, ev: Event) -> Finding | None:
        if ev.kind == "response" and not ev.had_tool_call and (ev.thinking_tokens or 0) > self.cap:
            return Finding(self.name, ev.agent, ev.ts,
                           f"{ev.thinking_tokens} thinking tokens with no tool call (cap {self.cap})",
                           {"thinking_tokens": ev.thinking_tokens, "cap": self.cap})
        return None


# Ledger events that mean the work moved forward. Ladder entries (retry, escalate, notice,
# decision), polls and anything else never count, so the watchdog's own writes cannot silence it.
# Plan 4 records hardware test results and evidence files as `evidence` ledger entries. The agent
# loop (orchard/agent.py) feeds each new evidence file to the watchdog directly as an `evidence`
# event, and no agent runs while a hardware test runs, so no ledger event name is added here.
PROGRESS_LEDGER_EVENTS = frozenset({"measurement", "stage_end"})


class NoNewEvidence:
    """An agent's model responses keep coming for window_s with no progress.

    Progress is an evidence file, or a ledger event named in PROGRESS_LEDGER_EVENTS. Each agent
    has its own clock. A progress event from "supervisor" (a stage_end, a measurement the
    supervisor wrote) moves every agent's clock; one from an agent moves only that agent's.
    """
    name = "no_new_evidence"

    def __init__(self, window_s: float = NO_EVIDENCE_S):
        self.window_s = window_s
        self._since: dict[str, float] = {}

    @staticmethod
    def _is_progress(ev: Event) -> bool:
        return ev.kind == "evidence" or (ev.kind == "ledger" and ev.name in PROGRESS_LEDGER_EVENTS)

    def feed(self, ev: Event) -> Finding | None:
        if self._is_progress(ev):
            if ev.agent == "supervisor":
                for agent in self._since:
                    self._since[agent] = ev.ts
            else:
                self._since[ev.agent] = ev.ts
            return None
        if ev.kind != "response":
            return None
        since = self._since.setdefault(ev.agent, ev.ts)
        if ev.ts - since >= self.window_s:
            self._since[ev.agent] = ev.ts
            return Finding(self.name, ev.agent, ev.ts,
                           f"{int(ev.ts - since)} s of model responses with no new evidence, "
                           "measurement, stage end or test result",
                           {"since": since, "window_s": self.window_s})
        return None


class RepeatedToolCall:
    """The same tool call (tool and arguments), or the same tool output, N times in a row."""
    name = "repeated_tool_call"

    def __init__(self, n: int = REPEAT_TOOL_N):
        if n < 2:
            raise ValueError("n must be at least 2")
        self.n = n
        self._last: dict[tuple, tuple] = {}
        self._count: dict[tuple, int] = {}

    def feed(self, ev: Event) -> Finding | None:
        if ev.kind == "tool_call":
            key, what = (ev.tool, ev.args_hash), "call"
        elif ev.kind == "tool_result":
            key, what = (ev.tool, ev.output_hash), "output"
        else:
            return None
        slot = (ev.agent, ev.kind)
        count = self._count.get(slot, 0) + 1 if self._last.get(slot) == key else 1
        self._last[slot] = key
        if count >= self.n:
            self._count[slot] = 0
            return Finding(self.name, ev.agent, ev.ts,
                           f"the same tool {what} ({ev.tool}) {self.n} times in a row",
                           {"what": what, "tool": ev.tool, "count": self.n})
        self._count[slot] = count
        return None


# Redirect suffixes (`> out`, `2>&1`, `2>/dev/null`, `< in`) and command separators, for
# command_shape.
_REDIRECT = re.compile(r"\s*(?:\d?>>?|&>)\s*\S+|\s*<\s*\S+")
_SEPARATOR = re.compile(r"\|\||&&|;|\|")
_NUMERIC = re.compile(r"-?[\d.,:]+[a-zA-Z]?")
_EXTENSION = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,7}$")
SHAPE_CHARS = 40


def _path_like(token: str) -> bool:
    return "/" in token or token.startswith(("~", ".")) or bool(_EXTENSION.search(token))


def command_shape(command: str) -> str:
    """What a shell command is doing, without its details.

    Only the part before the first pipe or separator counts, and redirect suffixes are dropped.
    The shape is the first word plus the first path-like argument (one with a slash, a leading
    `~` or `.`, or a file extension). A command with no path-like argument is cut to its first
    SHAPE_CHARS characters after trailing numeric arguments are removed and whitespace is
    normalised. So `grep -rn x /src | head -50` and `grep -rn y /src | head -80` share the shape
    `grep /src`, and `tail -n 200` and `tail -n 400` share `tail -n`.
    """
    if not isinstance(command, str) or not command.strip():
        return ""
    head = _REDIRECT.sub("", _SEPARATOR.split(command, 1)[0])
    try:
        tokens = shlex.split(head)
    except ValueError:                    # unbalanced quotes: fall back to whitespace
        tokens = head.split()
    if not tokens:
        return ""
    for t in tokens[1:]:
        if not t.startswith("-") and _path_like(t):
            return f"{tokens[0]} {t}"
    while len(tokens) > 1 and _NUMERIC.fullmatch(tokens[-1]):
        tokens.pop()
    return " ".join(tokens)[:SHAPE_CHARS]


class TurnRepeat:
    """The same set of tool calls in model turns in a row, matched two ways.

    Identical track: a turn's signature is the sorted tuple of (tool, args_hash) of its calls, so a
    turn that runs the same commands in another order still matches. It fires after `n` such
    turns in a row. RepeatedToolCall misses this loop: two commands that alternate never repeat
    back to back.

    Shape track: a turn's signature is the set of its calls' shapes (Event.shape, set by AgentStep
    from command_shape; a call with no shape uses its tool and args_hash). It fires after
    `shape_n` such turns in a row. It catches commands that differ only in a trailing argument,
    which the second-model run repeated in turns 53 to 60 of stage 0.

    A turn with no tool calls breaks both streaks. When either track fires, both re-arm, so an
    identical loop is reported once by the identical track and the shape track does not report
    it again one turn later. Both findings use this detector's name, so they share the ladder.

    Turns are told apart by Event.turn. An event with no turn number (a transcript) starts a new
    turn at each response, which comes before that turn's calls. A turn is complete when the next
    one starts, so a finding comes at the response of the turn after the last repeat.
    """
    name = "turn_repeat"

    def __init__(self, n: int = TURN_REPEAT_N, shape_n: int = TURN_SHAPE_N):
        if n < 2 or shape_n < 2:
            raise ValueError("n and shape_n must be at least 2")
        self.n, self.shape_n = n, shape_n
        self._turn: dict[str, object] = {}      # agent -> the current turn's key
        self._calls: dict[str, list] = {}       # agent -> the current turn's (tool, args_hash, shape)
        self._last: dict[str, tuple] = {}       # agent -> the last complete turn's signature
        self._count: dict[str, int] = {}        # agent -> turns in a row with that signature
        self._shape_last: dict[str, tuple] = {}  # agent -> the last complete turn's shape set
        self._shape_count: dict[str, int] = {}  # agent -> turns in a row with that shape set
        self._seq: dict[str, int] = {}          # agent -> responses seen, for unnumbered turns

    def _rearm(self, agent: str) -> None:
        self._count[agent] = 0
        self._shape_count[agent] = 0

    def _close(self, agent: str, ts: float) -> Finding | None:
        calls = self._calls.pop(agent, [])
        sig = tuple(sorted((str(t), str(a)) for t, a, _ in calls))
        if not sig:
            self._last.pop(agent, None)
            self._shape_last.pop(agent, None)
            self._rearm(agent)
            return None
        shapes = tuple(sorted({s if s else f"{t}:{a}" for t, a, s in calls}))
        count = self._count.get(agent, 0) + 1 if self._last.get(agent) == sig else 1
        shape_count = self._shape_count.get(agent, 0) + 1 if self._shape_last.get(agent) == shapes else 1
        self._last[agent], self._shape_last[agent] = sig, shapes
        tools = sorted({t for t, _ in sig})
        if count >= self.n:
            self._rearm(agent)                  # re-arm: the next repeats fire again
            return Finding(self.name, agent, ts,
                           f"the same {len(sig)} tool call(s) ({', '.join(tools)}) in {self.n} turns "
                           "in a row", {"turns": self.n, "calls": len(sig), "tools": tools,
                                        "match": "identical"})
        if shape_count >= self.shape_n:
            self._rearm(agent)
            return Finding(self.name, agent, ts,
                           f"near-duplicate tool calls with the same shape ({'; '.join(shapes)}) in "
                           f"{self.shape_n} turns in a row",
                           {"turns": self.shape_n, "calls": len(sig), "tools": tools, "match": "shape",
                            "shapes": list(shapes)})
        self._count[agent], self._shape_count[agent] = count, shape_count
        return None

    def feed(self, ev: Event) -> Finding | None:
        if ev.kind not in ("response", "tool_call"):
            return None
        if ev.turn is not None:
            key = ("turn", ev.turn)
        elif ev.kind == "response":
            self._seq[ev.agent] = self._seq.get(ev.agent, 0) + 1
            key = ("seq", self._seq[ev.agent])
        else:
            key = self._turn.get(ev.agent)
        found = None
        if key != self._turn.get(ev.agent):
            if ev.agent in self._turn:
                found = self._close(ev.agent, ev.ts)
            self._turn[ev.agent] = key
        if ev.kind == "tool_call":
            self._calls.setdefault(ev.agent, []).append((ev.tool, ev.args_hash, ev.shape))
        return found


# Tool names that write a file, for sources that do not say whether a write succeeded (Event.wrote
# is None). write_file is orchard's own tool; edit and replace are qwen-code's.
WRITE_TOOLS = frozenset({"write_file", "edit", "replace"})


class NoFileWritten:
    """A step that makes `turns` model turns in a row without writing any file.

    A write is a tool_result with wrote=True, a tool_result from a tool in WRITE_TOOLS whose
    source does not say (wrote=None), or a new evidence file. A refused write_file (wrote=False)
    is not a write. The detector fires once and then stays quiet until the agent writes a file,
    which starts a new count. The finding is nudge level and carries its own nudge text.
    """
    name = "no_file_written"

    def __init__(self, turns: int = WRITELESS_TURNS):
        if turns < 1:
            raise ValueError("turns must be at least 1")
        self.turns = turns
        self._count: dict[str, int] = {}
        self._fired: set[str] = set()

    @staticmethod
    def _is_write(ev: Event) -> bool:
        if ev.kind == "evidence":
            return True
        if ev.kind != "tool_result":
            return False
        return ev.wrote is True or (ev.wrote is None and ev.tool in WRITE_TOOLS)

    def feed(self, ev: Event) -> Finding | None:
        if self._is_write(ev):
            self._count[ev.agent] = 0
            self._fired.discard(ev.agent)
            return None
        if ev.kind != "response" or ev.agent in self._fired:
            return None
        count = self._count.get(ev.agent, 0) + 1
        self._count[ev.agent] = count
        if count < self.turns:
            return None
        self._fired.add(ev.agent)
        return Finding(self.name, ev.agent, ev.ts,
                       f"{count} model turns in a row and no file written",
                       {"turns": count},
                       nudge=(f"{count} model turns have passed and no file was written. "
                              "Stop investigating: write the files the skill names now, or say "
                              "what blocks you from writing them."))


class LeaseIdle:
    """A lease held while its chips have no device open, for longer than idle_s.

    `status` is the adapter's status call; `lease_chips` returns the BDFs of the lease to watch;
    `exempt` returns True while idle chips are expected (during a park, between the stop and the
    reset). The status is read at most once per poll_s, on the clock of the events fed in.
    """
    name = "lease_idle"

    def __init__(self, status, lease_chips, *, agent: str, idle_s: float = LEASE_IDLE_S,
                 poll_s: float = LEASE_POLL_S, exempt=lambda: False):
        self.status, self.lease_chips, self.agent = status, lease_chips, agent
        self.idle_s, self.poll_s, self.exempt = idle_s, poll_s, exempt
        self._last_poll: float | None = None
        self._idle_since: float | None = None

    def feed(self, ev: Event) -> Finding | None:
        if self._last_poll is not None and ev.ts - self._last_poll < self.poll_s:
            return None
        self._last_poll = ev.ts
        chips = set(self.lease_chips())
        if not chips or self.exempt():
            self._idle_since = None
            return None
        try:
            states = {c.bdf: c.state for c in self.status() if c.bdf in chips}
        except AdapterError:
            return None                       # unknown is not idle
        if not states or any(s != "CLAIMED" for s in states.values()):
            self._idle_since = None
            return None
        if self._idle_since is None:
            self._idle_since = ev.ts
            return None
        if ev.ts - self._idle_since >= self.idle_s:
            since, self._idle_since = self._idle_since, ev.ts
            # An idle lease is the supervisor's decision (handoff.release_for_idle_phase), so
            # it is reported in a notice and never climbs the repeat ladder.
            return Finding(self.name, self.agent, ev.ts,
                           f"lease chips {sorted(chips)} held with no device open for "
                           f"{int(ev.ts - since)} s; consider release_for_idle_phase",
                           {"chips": sorted(chips), "since": since}, notice_only=True)
        return None


class StageOverBudget:
    """A stage past its wall-clock budget. The finding asks for a pause (spec section 10)."""
    name = "stage_over_budget"

    def __init__(self, budgets: dict[int, float], *, agent: str = "supervisor"):
        self.budgets, self.agent = dict(budgets), agent
        self._start: tuple[int | None, float] | None = None
        self._fired = False

    def feed(self, ev: Event) -> Finding | None:
        if ev.kind == "ledger" and ev.name == "stage_start":
            self._start, self._fired = (ev.stage, ev.ts), False
        elif ev.kind == "ledger" and ev.name == "stage_end":
            self._start = None
        if self._start is None or self._fired:
            return None
        stage, t0 = self._start
        budget = self.budgets.get(stage)
        if budget is not None and ev.ts - t0 > budget:
            self._fired = True
            return Finding(self.name, self.agent, ev.ts,
                           f"stage {stage} has run {int(ev.ts - t0)} s, past its budget of "
                           f"{int(budget)} s", {"stage": stage, "budget_s": budget}, pause=True)
        return None


def transcript_detectors() -> list:
    """The detectors a session transcript can feed. The other three need ledger, evidence or lease
    events, which a transcript does not carry."""
    return [IdenticalResponses(), ThinkingWithoutAction(), RepeatedToolCall()]


def replay(events, detectors) -> list[Finding]:
    """Feed every event to every detector, in order, and collect the findings."""
    found = []
    for ev in events:
        for d in detectors:
            f = d.feed(ev)
            if f is not None:
                found.append(f)
    return found


RUNGS = ("nudge", "escalate", "pause")


class Actuator(Protocol):
    """What the ladder can do to a launched agent. Plan 4's proxy and stage machine provide it."""

    def nudge(self, agent: str, message: str) -> None: ...

    def escalate(self, agent: str, stage: int | None) -> None: ...

    def pause(self, reason: str, evidence: dict) -> None: ...


def nudge_message(findings: list[Finding]) -> str:
    """The text of a nudge. A finding with its own nudge text sends that text. The others share
    the repeat text, which names each of them."""
    repeats = [f for f in findings if f.nudge is None]
    lines = []
    if repeats:
        lines.append("The supervisor stopped this step because it is repeating itself:")
        lines += [f"- {f.summary}" for f in repeats]
        lines.append("The model server decodes greedily, so the same request returns the same "
                     "answer. Do something different: run a tool to get new information, write "
                     "down what you have found so far, or say what is blocking you.")
    lines += [f.nudge for f in findings if f.nudge is not None]
    return "\n".join(lines)


class Ladder:
    """The response ladder (spec section 7): nudge, then escalate, then pause, each capped.

    Rungs are counted per agent and stage, and the counts are read back from the ledger, so a
    restarted supervisor does not repeat a rung. A resume starts the counts again. An agent the supervisor did not launch gets one
    `notice` per finding with the evidence, and no action. A finding with pause=True (a budget
    cap) goes straight to the pause rung, for any agent, because pausing the run is the
    supervisor's own action.
    """

    def __init__(self, actuator: Actuator, ledger, launched, *, caps: dict | None = None):
        self.actuator, self.ledger = actuator, ledger
        self.launched = frozenset(launched)
        self.caps = dict(RUNG_CAPS if caps is None else caps)
        missing = set(RUNGS) - set(self.caps)
        if missing:
            raise ValueError(f"caps must name every rung; missing {sorted(missing)}")
        self._taken: Counter = Counter()
        for e in ledger.read():
            d = e["data"]
            if e["event"] == "decision" and d.get("decision") == "resume":
                # A resume (the operator's, or an unattended retry) is a fresh start for the ladder;
                # only a restart without one must not repeat a rung.
                self._taken.clear()
            elif d.get("watchdog") and d.get("rung") in RUNGS:
                self._taken[(d.get("agent"), e["stage"], d["rung"])] += 1

    def _left(self, agent: str, stage, rung: str) -> bool:
        return self._taken[(agent, stage, rung)] < self.caps[rung]

    def _take(self, event: str, agent: str, stage, rung: str, records: list, **extra) -> None:
        # The entry is written before the action. A crash between the two counts the rung as
        # used, so a restart cannot take it twice.
        self.ledger.append(event, stage, watchdog=True, rung=rung, agent=agent, findings=records,
                           **extra)
        self._taken[(agent, stage, rung)] += 1

    def _pause(self, agent: str, stage, records: list) -> str:
        reason = "watchdog: " + "; ".join(r["summary"] for r in records)
        self._take("decision", agent, stage, "pause", records, decision="pause", reason=reason)
        self.actuator.pause(reason, {"agent": agent, "findings": records})
        return "pause"

    def respond(self, agent: str, findings: list[Finding], stage) -> str:
        quiet = [f for f in findings if f.notice_only]
        if quiet:
            # An idle lease and similar supervisor matters: one notice, never a nudge.
            self.ledger.append("notice", stage, watchdog=True, agent=agent,
                               findings=[f.record() for f in quiet])
        findings = [f for f in findings if not f.notice_only]
        if not findings:
            return "notice"
        records = [f.record() for f in findings]
        if any(f.pause for f in findings):
            return self._pause(agent, stage, records) if self._left(agent, stage, "pause") else "none"
        if agent not in self.launched:
            self.ledger.append("notice", stage, watchdog=True, report_only=True, agent=agent,
                               findings=records)
            return "notice"
        if self._left(agent, stage, "nudge"):
            message = nudge_message(findings)
            self._take("retry", agent, stage, "nudge", records, message=message)
            self.actuator.nudge(agent, message)
            return "nudge"
        if self._left(agent, stage, "escalate"):
            self._take("escalate", agent, stage, "escalate", records)
            self.actuator.escalate(agent, stage)
            return "escalate"
        if self._left(agent, stage, "pause"):
            return self._pause(agent, stage, records)
        return "none"                 # every rung used; the run is already paused


class RetryGuard:
    """Spec section 3: never retry the same call with the same inputs more than once.

    Nothing in plan 3 calls it. Plan 4's proxy asks before it forwards a request, and plan 4 must
    test that wiring. The counts live in memory: they are lost on a restart and never forgotten
    while the process lives. An identical request is allowed twice
    (the call and one retry). A nudged request has a different prompt, so it has a new key.
    """
    MAX_SENDS = 2

    def __init__(self):
        self._sends: Counter = Counter()

    @staticmethod
    def key(agent: str, request: dict) -> str:
        blob = json.dumps([agent, request], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def allow(self, agent: str, request: dict) -> bool:
        k = self.key(agent, request)
        if self._sends[k] >= self.MAX_SENDS:
            return False
        self._sends[k] += 1
        return True


class Watchdog:
    """Feeds each event to every detector and answers the findings once per agent."""

    def __init__(self, detectors, ladder: Ladder):
        self.detectors, self.ladder = list(detectors), ladder
        self.stage: int | None = None

    def feed(self, ev: Event) -> list[Finding]:
        if ev.kind == "ledger" and ev.name == "stage_start":
            self.stage = ev.stage
        findings = replay([ev], self.detectors)
        by_agent: dict[str, list[Finding]] = {}
        for f in findings:
            by_agent.setdefault(f.agent, []).append(f)
        for agent, fs in by_agent.items():
            self.ladder.respond(agent, fs, self.stage)
        return findings
