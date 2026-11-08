# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 512 to 515, found in the reviews of PRs 496 and 497.

512  An operator `pause` read while a watchdog stop was active was consumed and lost: `stop_reason()` returned the watchdog's status, and the
     rescue then cleared the stop, so no pause remained and another hardware check started.
513  The stage budget was checked before the helper's blocking reply and not after it, so a budget that ran out during the wait did not stop
     the rescue from starting a hardware check past the deadline.
514  A helper reply is tied to one attempt: a reply of an attempt that ended is settled as not helped, whatever a later attempt's check found.
     (The attempt boundary was added in review round 7 of PR 496; these tests pin the case in the issue's words.)
515  The best-port snapshot is durable before its ledger entry: every directory it creates and the rename of each file are synced first.
"""
import dataclasses
import os
import stat

import pytest

from orchard import repair_help as rh
from orchard import supervisor as sv
from orchard.supervisor import Control, RunActuator
from test_supervisor_telegraph_in_repair import (PREFILL_BROKEN, HONEST, BROKEN, decs, looping_coder, rig, start, wire_helper)  # noqa: F401
from test_supervisor_telegraph_in_repair import pytestmark  # noqa: F401


# ---- 512 --------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("watchdog_stop", ["pause", "escalate"])
def test_an_operator_pause_read_during_a_watchdog_stop_is_kept_and_wins(tmp_path, watchdog_stop):
    a = RunActuator(Control(tmp_path))
    if watchdog_stop == "pause":
        a.pause("ladder", {"findings": [{"detector": "loop"}]})
    else:
        a.escalate("stage-agent", 2)
    Control(tmp_path).write("pause")
    assert a.stop_reason() == "operator-pause"
    assert a.stop_reason() == "operator-pause"                  # still there on the next read: the command is not a one-time answer


def test_an_operator_abort_still_beats_an_operator_pause(tmp_path):
    a = RunActuator(Control(tmp_path))
    Control(tmp_path).write("abort")
    assert a.stop_reason() == "abort"
    Control(tmp_path).write("pause")
    assert a.stop_reason() == "abort"


def test_a_rescued_repair_step_does_not_clear_an_operator_pause_read_during_the_watchdog_stop(rig, monkeypatch, tmp_path):
    """End to end: the repair step's own stop poll reads the operator's pause while the watchdog's pause is active. The rescue must stop and
    start no further hardware check."""
    wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN, HONEST])
    start(rig, BROKEN, [PREFILL_BROKEN + "# a\n"])
    looping_coder(rig)
    real_step = sv.Supervisor._step

    def step(self, spec, phase, *a, **k):
        out = real_step(self, spec, phase, *a, **k)
        if phase == "repair" and "round 2 of" in (k.get("extra") or "") and out[0].status in ("pause", "escalate"):
            Control(self.run_dir).write("pause")
            self.actuator.stop_reason()                         # the step's last poll: it consumes the file while the watchdog's stop is active
        return out
    monkeypatch.setattr(sv.Supervisor, "_step", step)
    reruns = []
    real_hw = sv.Supervisor._hardware_phase
    monkeypatch.setattr(sv.Supervisor, "_hardware_phase", lambda self, *a, **k: (reruns.append(1), real_hw(self, *a, **k))[1])
    rig.run()
    assert decs(rig, "helper rescue stopped"), "the rescue did not notice the operator's pause"
    assert len(reruns) == 2, "a hardware check started after the operator's pause (the first test and round 1's check are the two allowed)"


# ---- 513 --------------------------------------------------------------------------------------------------------------------------------

def test_a_stage_budget_that_runs_out_while_the_helper_answers_stops_the_rescue(rig, monkeypatch, tmp_path):
    """The budget guard ran before the blocking helper call. The clock is moved past the stage's budget while the reply is awaited, and the
    rescue must end in a budget pause with no further hardware check."""
    calls = {"asks": 0}

    def answer(body):
        calls["asks"] += 1
        if calls["asks"] == 2:                                  # the rescue ask of round 2
            clock["now"] += 10 ** 7
        from test_supervisor_telegraph_in_repair import reply
        return reply(PREFILL_BROKEN if calls["asks"] == 1 else HONEST)(body)
    clock = {"now": 0.0}
    wire_helper(monkeypatch, tmp_path, [answer])
    start(rig, BROKEN, [PREFILL_BROKEN + "# a\n"])
    looping_coder(rig)
    real_clock = rig.clock if hasattr(rig, "clock") else None
    base = sv.Supervisor.__init__

    def init(self, *a, **k):
        base(self, *a, **k)
        wall = self.clock
        self.clock = lambda: wall() + clock["now"]
    monkeypatch.setattr(sv.Supervisor, "__init__", init)
    reruns = []
    real_hw = sv.Supervisor._hardware_phase
    monkeypatch.setattr(sv.Supervisor, "_hardware_phase", lambda self, *a, **k: (reruns.append(1), real_hw(self, *a, **k))[1])
    rig.run()
    assert calls["asks"] == 2
    assert decs(rig, "helper rescue stopped"), "the budget that ran out during the wait was not noticed"
    assert len(reruns) == 2, "a hardware check started past the stage budget"


# ---- 514 --------------------------------------------------------------------------------------------------------------------------------

def test_a_placed_reply_of_an_attempt_that_ended_after_a_watchdog_escalation_is_settled_as_not_helped(tmp_path, monkeypatch):
    """The issue's case: the stop came after the escalation and before the reply was reported; recovery started a fresh (escalated) attempt, and
    that attempt's check passed. The old reply must not be credited."""
    from orchard.ledger import Ledger
    from test_repair_help_round4 import _Tele, _sup
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("stage_start", 2)
    led.append("decision", 2, decision=rh.ASKED, round=2, failure_op="prefill")
    led.append("decision", 2, decision=rh.ANSWERED, ok=True, call_id="old", cached=False)
    led.append("decision", 2, decision=rh.PLACED, call_id="old")
    led.append("escalate", 2, rung="escalate")
    led.append("stage_end", 2, result="escalate", reasons=["the repair step ended: escalate"])
    led.append("decision", 2, decision="moved the partial stage directory aside", path="stages/2.partial-1")
    led.append("stage_start", 2, escalated=True, resumed=False)
    led.append("decision", 2, decision=rh.port_repair.TESTED)
    sup = _sup(tmp_path, led, monkeypatch)
    sup._repair_verdicts(2, tmp_path)
    assert sup.telegraph.reports == [("old", False)]
    verdicts = [e["data"] for e in led.read() if e["data"].get("decision") == rh.PROGRESS]
    assert verdicts and verdicts[-1]["helped"] is False and "new attempt began" in verdicts[-1]["outcome"]


def test_a_reply_asked_in_the_new_attempt_is_still_judged_by_the_new_attempts_check(tmp_path, monkeypatch):
    from orchard.ledger import Ledger
    from test_repair_help_round4 import _sup
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("stage_start", 2)
    led.append("decision", 2, decision=rh.ASKED, round=1, failure_op="load")
    led.append("decision", 2, decision=rh.ANSWERED, ok=True, call_id="old", cached=False)
    led.append("decision", 2, decision=rh.PLACED, call_id="old")
    led.append("stage_start", 2, escalated=True, resumed=False)
    led.append("decision", 2, decision=rh.ASKED, round=1, failure_op="load")
    led.append("decision", 2, decision=rh.ANSWERED, ok=True, call_id="new", cached=False)
    led.append("decision", 2, decision=rh.PLACED, call_id="new")
    led.append("decision", 2, decision=rh.port_repair.TESTED)
    sup = _sup(tmp_path, led, monkeypatch)
    sup._repair_verdicts(2, tmp_path)
    assert dict(sup.telegraph.reports) == {"old": False, "new": True}


# ---- 515 --------------------------------------------------------------------------------------------------------------------------------

def test_the_best_port_snapshot_is_fully_synced_before_save_best_returns(tmp_path, monkeypatch):
    """Every directory the snapshot creates (stages/2.best, the digest directory, a nested directory) and each file's rename are synced before
    save_best returns, so the ledger entry written after it cannot be durable while its files or directories are not."""
    stage = tmp_path / "stages" / "2"
    (stage / "port" / "sub").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text("x = 1\n")
    (stage / "port" / "sub" / "helper.py").write_text("y = 2\n")
    synced, real = [], os.fsync

    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            synced.append(os.readlink(f"/proc/self/fd/{fd}"))
        return real(fd)
    monkeypatch.setattr(os, "fsync", fsync)
    rec = rh.save_best(stage)
    assert rec is not None
    snap = str(tmp_path / "stages" / "2.best" / rec["snapshot"])
    must = {str(tmp_path / "stages"), str(tmp_path / "stages" / "2.best"), snap, snap + "/sub"}
    assert must <= set(synced), sorted(must - set(synced))
