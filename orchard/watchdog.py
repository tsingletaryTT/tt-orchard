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
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

from orchard.adapters import AdapterError
from orchard.defaults import (IDENTICAL_N, LEASE_IDLE_S, LEASE_POLL_S, NO_EVIDENCE_S, REPEAT_TOOL_N,
                              RUNG_CAPS, THINKING_CAP)

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
PROGRESS_LEDGER_EVENTS = frozenset({"measurement", "stage_end", "test_result"})


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
