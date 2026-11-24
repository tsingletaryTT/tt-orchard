# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 616 (found in the Copilot review of PR 614).

An earlier stop poll can read the operator's `pause` and consume the control file; `stop_reason()` then keeps it as `_stop = "operator-pause"`.
If the stage's wall-clock budget runs out during the helper wait, the watchdog calls `RunActuator.pause()`. That call used to overwrite
`operator-pause` with `pause`, and the next poll found no file to read again. The rescue then recorded `stage-budget` without the
operator's decision. An operator pause, like an abort, is not lowered by a watchdog pause.
"""
import pytest

from orchard import supervisor as sv
from orchard.supervisor import Control, RunActuator
from test_supervisor_telegraph_in_repair import (PREFILL_BROKEN, HONEST, BROKEN, decs, looping_coder, rig, start, wire_helper)  # noqa: F401
from test_supervisor_telegraph_in_repair import pytestmark  # noqa: F401

BUDGET_FINDING = {"findings": [{"detector": "stage_over_budget"}]}


def test_a_budget_pause_does_not_overwrite_an_operator_pause_that_was_already_read(tmp_path):
    a = RunActuator(Control(tmp_path))
    Control(tmp_path).write("pause")
    assert a.stop_reason() == "operator-pause"              # the earlier poll: it consumes the file
    a.pause("stage budget", BUDGET_FINDING)                  # the budget runs out during the helper wait
    assert a.stop_reason() == "operator-pause"              # no file is left to read; the stop must still be the operator's
    assert a.deadline_pause is True                          # the budget flag still rises, as the rescue's budget branch reads it


def test_a_watchdog_pause_does_not_overwrite_an_operator_abort_that_was_already_read(tmp_path):
    a = RunActuator(Control(tmp_path))
    Control(tmp_path).write("abort")
    assert a.stop_reason() == "abort"
    a.pause("stage budget", BUDGET_FINDING)
    assert a.stop_reason() == "abort"


@pytest.mark.parametrize("first", ["escalate", "pause"])
def test_a_watchdog_pause_still_replaces_a_watchdog_escalation_or_pause(tmp_path, first):
    """The guard is for the operator's commands only: the watchdog's own stops keep the order they had."""
    a = RunActuator(Control(tmp_path))
    if first == "escalate":
        a.escalate("stage-agent", 2)
    else:
        a.pause("ladder", {"findings": []})
    a.pause("ladder", {"findings": []})
    assert a.stop_reason() == "pause"


def test_an_operator_pause_read_before_the_budget_ran_out_is_what_the_rescue_records(rig, monkeypatch, tmp_path):
    """End to end: a poll consumes the operator's pause, then the budget runs out while the helper's reply is awaited. The rescue stops with the
    operator's pause as its command, not `stage-budget`."""
    calls = {"asks": 0}
    clock = {"now": 0.0}
    seen = {}

    def answer(body):
        calls["asks"] += 1
        if calls["asks"] == 2:                                  # the rescue ask of round 2
            Control(rig.run_dir).write("pause")
            seen["sup"].actuator.stop_reason()                  # an earlier poll consumes the file ...
            clock["now"] += 10 ** 7                             # ... and the budget runs out in the same wait
        from test_supervisor_telegraph_in_repair import reply
        return reply(PREFILL_BROKEN if calls["asks"] == 1 else HONEST)(body)
    wire_helper(monkeypatch, tmp_path, [answer])
    start(rig, BROKEN, [PREFILL_BROKEN + "# a\n"])
    looping_coder(rig)
    base = sv.Supervisor.__init__

    def init(self, *a, **k):
        base(self, *a, **k)
        seen["sup"] = self
        wall = self.clock
        self.clock = lambda: wall() + clock["now"]
    monkeypatch.setattr(sv.Supervisor, "__init__", init)
    rig.run()
    stopped = decs(rig, "helper rescue stopped")
    assert stopped and stopped[-1]["command"] == "operator-pause", stopped
