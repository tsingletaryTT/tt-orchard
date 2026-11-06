# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The outside helper in the repair rounds: the findings of the second Copilot round of PR 496. Each test names the finding."""
import dataclasses
import hashlib
import os

import pytest

from orchard import repair_help as rh
from orchard import supervisor as sv
from orchard.supervisor import RunActuator, Control
from test_repair_help_review import _answered_then_tested, _port_run
from test_supervisor_telegraph_in_repair import (PREFILL_BROKEN, HONEST, BROKEN, decs, looping_coder, reply, rig, start, stage_ends, wire_helper)  # noqa: F401
from orchard.supervisor import EXIT_ABORTED
from run_fakes import where
from test_supervisor_telegraph_in_repair import pytestmark  # noqa: F401  (the same fixtures: stub tools, unbuilt stages 5 and 6)


# ---- the placement is refused when `port` names another directory afterwards -------------------------------------------------------

def test_a_port_directory_swapped_during_the_placement_is_refused_and_not_reported_as_placed(tmp_path, monkeypatch):
    """Finding: the write went into the moved-away directory and `place` still returned success."""
    stage = _port_run(tmp_path)
    sent = hashlib.sha256(b"x = 1\n").hexdigest()
    real = rh.safeio.write_beneath_if

    def swapping(fd, parts, data, **kw):
        out = real(fd, parts, data, **kw)
        os.rename(stage / "port", stage / "port.old")
        (stage / "port").mkdir()
        (stage / "port" / "port_model.py").write_text("coder_file = 2\n")
        return out
    monkeypatch.setattr(rh.safeio, "write_beneath_if", swapping)
    placed, why, _ = rh.place(stage, "y = 2\n", sent, coder_failed=False, call_id="c1", model="m")
    assert placed is False and "replaced" in why
    assert (stage / "port" / "port_model.py").read_text() == "coder_file = 2\n"


def test_an_unswapped_placement_is_still_placed(tmp_path):
    stage = _port_run(tmp_path)
    placed, why, _ = rh.place(stage, "y = 2\n", hashlib.sha256(b"x = 1\n").hexdigest(), coder_failed=False, call_id="c1", model="m")
    assert placed is True, why


# ---- the deadline flag stays up until clear() -----------------------------------------------------------------------------------------

def test_a_ladder_pause_after_a_budget_pause_does_not_lower_the_deadline_flag(tmp_path):
    """Finding: the agent step feeds events after a budget pause, and a later coder-ladder pause reset the flag, so a helper rescue could follow."""
    a = RunActuator(Control(tmp_path))
    a.pause("budget", {"findings": [{"detector": "stage_over_budget"}]})
    assert a.deadline_pause is True
    a.pause("ladder", {"findings": [{"detector": "loop"}]})
    assert a.deadline_pause is True
    a.clear()
    assert a.deadline_pause is False


# ---- a verdict is tied to the newest test ---------------------------------------------------------------------------------------

def test_an_earlier_tests_marker_restored_over_a_later_rerun_judges_nothing(tmp_path):
    """Finding: the first test's marker and output, put back after a later rerun failed, matched the first test's own evidence entry."""
    s, stage, _ = _answered_then_tested(tmp_path, tested=True)
    assert s._repair_outcome(2, stage) == (False, "prefill", True, None)          # the control: this is the newest test's record
    s.ledger.append("evidence", 2, what="hardware test", config=[], cache=None, returncode=2, timed_out=False,
                    path="stages/2/test-result.json", sha256="0" * 64)            # a later rerun wrote other bytes at the same path
    assert s._repair_outcome(2, stage) == (False, None, False, None)


# ---- --no-repair-help survives a resume -------------------------------------------------------------------------------------------

def test_a_resume_keeps_the_recorded_repair_help_opt_out(tmp_path):
    from test_supervisor import Rig
    from test_tool_dedupe_option import built, start_entry
    off = Rig(tmp_path)
    off.args.no_repair_help = True
    off.args.unattended = False
    off.run()
    assert start_entry(off)["repair_help"] is False
    off.args.no_repair_help = False
    assert built(off).repair_help is False
    off.close()


def test_a_default_run_records_nothing_about_repair_help_and_a_resume_keeps_it_on(tmp_path):
    from test_supervisor import Rig
    from test_tool_dedupe_option import built, start_entry
    on = Rig(tmp_path)
    on.args.unattended = False
    on.run()
    assert "repair_help" not in start_entry(on)
    assert built(on).repair_help is True
    on.close()


# ---- end to end: an operator command that arrives while the helper's reply is awaited, and a deadline pause that ends `error` --------

def test_an_operator_abort_written_while_the_helper_answers_starts_no_further_hardware_check(rig, monkeypatch, tmp_path):
    """Finding: a successful rescue cleared the stop and went on to the hardware check without reading the control file again."""
    state = {"asks": 0}

    def answer(body):
        state["asks"] += 1
        if state["asks"] == 2:                              # the rescue ask of round 2
            Control(rig.run_dir).write("abort")
        return reply(PREFILL_BROKEN if state["asks"] == 1 else HONEST)(body)
    wire_helper(monkeypatch, tmp_path, [answer])
    start(rig, BROKEN, [PREFILL_BROKEN + "# a\n"])
    looping_coder(rig)
    reruns = []
    real_hw = sv.Supervisor._hardware_phase
    monkeypatch.setattr(sv.Supervisor, "_hardware_phase", lambda self, *a, **k: (reruns.append(1), real_hw(self, *a, **k))[1])
    rig.run()
    assert state["asks"] == 2 and decs(rig, "helper rescue stopped")
    assert len(reruns) == 2, "a hardware check started after the operator's abort (the first test and round 1's check are the two allowed)"


def test_a_deadline_pause_that_ends_the_step_with_error_is_not_rescued(rig, monkeypatch, tmp_path):
    """Finding: the deadline flag was read only when the status was `pause`; a step that raised it and then returned `error` was rescued."""
    wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN, HONEST])
    start(rig, BROKEN, [PREFILL_BROKEN + "# a\n"])
    looping_coder(rig)
    real_step = sv.Supervisor._step

    def step(self, spec, phase, *a, **k):
        out = real_step(self, spec, phase, *a, **k)
        if phase == "repair" and "round 2 of" in (k.get("extra") or ""):
            self.actuator.deadline_pause = True
            return dataclasses.replace(out[0], status="error"), *out[1:]
        return out
    monkeypatch.setattr(sv.Supervisor, "_step", step)
    rig.run()
    asked = decs(rig, rh.ASKED)
    assert not [a for a in asked if a["coder_failed"]], [a["why"] for a in asked]


# ---- review of PR 496, round 5 ---------------------------------------------------------------------------------------------------

def _interrupted_ledger(tmp_path, *, placed):
    """A ledger that stops after the answer (and after the placement when `placed`), then a passing check run by the coder's own edit."""
    from orchard.ledger import Ledger
    led = Ledger(tmp_path / "ledger.jsonl")
    led.append("stage_start", 2)
    led.append("decision", 2, decision=rh.ASKED, round=1, failure_op="load")
    led.append("decision", 2, decision=rh.ANSWERED, ok=True, call_id="aa", cached=False)
    if placed:
        led.append("decision", 2, decision=rh.PLACED, call_id="aa")
    led.append("decision", 2, decision=rh.port_repair.TESTED)
    return led


class _Tele:
    def __init__(self):
        self.reports = []

    def report(self, call_id, helped, note=""):
        self.reports.append((call_id, helped))


def _sup(tmp_path, led, monkeypatch):
    sup = sv.Supervisor.__new__(sv.Supervisor)
    sup.ledger, sup.telegraph, sup.run_dir = led, _Tele(), tmp_path
    monkeypatch.setattr(sv.Supervisor, "_repair_outcome", lambda self, n, d: (True, None, True, None))      # the coder's own edit passed the check
    return sup


def test_an_answer_the_supervisor_never_recorded_as_placed_is_not_credited_with_a_later_pass(tmp_path, monkeypatch):
    """Finding: a stop between the answer and the placement left the answer pending, and a later passing check graded it as helped."""
    led = _interrupted_ledger(tmp_path, placed=False)
    sup = _sup(tmp_path, led, monkeypatch)
    sup._repair_verdicts(2, tmp_path)
    assert sup.telegraph.reports == [("aa", False)]
    prog = [e["data"] for e in led.read() if e["data"].get("decision") == rh.PROGRESS]
    assert prog and prog[-1]["helped"] is False and "never became the port" in prog[-1]["outcome"]


def test_a_placed_answer_is_still_credited_when_the_check_it_led_to_passed(tmp_path, monkeypatch):
    led = _interrupted_ledger(tmp_path, placed=True)       # the control: the same ledger with the placement recorded
    sup = _sup(tmp_path, led, monkeypatch)
    sup._repair_verdicts(2, tmp_path)
    assert sup.telegraph.reports == [("aa", True)]


# ---- review of PR 496, round 6 ---------------------------------------------------------------------------------------------------

def test_a_watchdog_escalation_that_ends_the_step_with_error_is_consumed_after_a_successful_rescue(rig, monkeypatch, tmp_path):
    """Finding: the stop was consumed only when the step's status was pause or escalate. A step that raised an escalation and then ended `error`
    (a capped reply) left the stop in the actuator, so the next agent step exited at once."""
    wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN, HONEST])
    start(rig, BROKEN, [PREFILL_BROKEN + "# a\n"])
    looping_coder(rig)
    real_step = sv.Supervisor._step

    def step(self, spec, phase, *a, **k):
        out = real_step(self, spec, phase, *a, **k)
        if phase == "repair" and "round 2 of" in (k.get("extra") or ""):
            self.actuator._stop = "escalate"
            return dataclasses.replace(out[0], status="error"), *out[1:]
        return out
    monkeypatch.setattr(sv.Supervisor, "_step", step)
    seen = []
    real_hw = sv.Supervisor._hardware_phase
    monkeypatch.setattr(sv.Supervisor, "_hardware_phase", lambda self, *a, **k: (seen.append(self.actuator._stop), real_hw(self, *a, **k))[1])
    rig.run()
    asked = [a for a in decs(rig, rh.ASKED) if a["coder_failed"]]
    assert asked, "the rescue must have happened for this test to say anything"
    assert seen[-1] is None, seen
