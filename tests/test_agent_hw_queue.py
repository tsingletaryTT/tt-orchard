# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A busy board is waited for on ONE queue ticket (`gozer acquire`'s ticket and `gozer wait`), never by asking again in a
loop (PR 116, Copilot's review: the old loop gave up its place in the queue on every try). The probe runs the harness
under the lease that `gozer wait` grants and releases it itself. Interrupted waits and runs record the time spent.

Fakes only: no gozer, no chip."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from orchard import hwtool, kernelprobe as kp
from test_agent_hw import (FACTS, GOOD_HW_SET, KERNELS, QUEUED, World, attempt_dir, fake_kenv, good_hw,  # noqa: F401
                           hw_writer, make_check, queued, queued_runner, skills_current, ttlang_skill, CLOCK)


# ---- the policy object ----------------------------------------------------------------------------------------------


class QWorld(World):
    """A World whose busy answer keeps its ticket and whose wait_for is scripted: `waits` holds callables or answers."""

    def __init__(self, answers=None, waits=None):
        super().__init__(answers)
        self.waits = list(waits or [])
        self.wait_calls = []

    def wait_for(self, k, q, wait_s, timeout_s, on_grant):
        self.wait_calls.append({"k": k, "ticket": q.get("ticket"), "wait_s": wait_s, "timeout_s": timeout_s})
        item = self.waits.pop(0)
        return item(self, on_grant) if callable(item) else item

    def check(self, **policy):
        return hwtool.HardwareCheck(hwtool.HwPolicy(**policy), prepare=self.prepare, run=self.run, wait_for=self.wait_for,
                                    clock=self.clock, sleep=self.sleep, stamp=lambda: "2026-10-27T00:00:00Z")


def kept_ticket(ticket="T1"):
    return (dict(QUEUED, queued={"ticket": ticket, "cancel_error": None, "kept": True}), "queued", "")


def timed_out_wait(cancel_error=None, ticket="T1", took=120.0):
    def wait(world, on_grant):
        world.now += took
        return (dict(QUEUED, elapsed_s=0.0, queued={"ticket": ticket, "cancel_error": cancel_error}), "queued", "")
    return wait


def granted_pass(waited=21.0, elapsed=47.0):
    def wait(world, on_grant):
        world.now += waited
        on_grant()
        world.now += elapsed
        return (dict(good_hw(), elapsed_s=elapsed, waited_s=waited), None, "")
    return wait


def test_a_busy_board_waits_on_the_one_ticket_and_never_runs_gozer_again():
    w = QWorld([kept_ticket()], [timed_out_wait()])
    c = w.check()
    text = c.call({})
    assert len(w.runs) == 1 and len(w.wait_calls) == 1 and w.sleeps == []          # one try, one wait, no loop
    assert w.wait_calls[0]["ticket"] == "T1"
    assert "BUSY" in text and "simulator" in text and "were cancelled" in text
    assert "waited on 1 queue ticket" in text


def test_the_wait_gets_what_is_left_of_wait_s_after_the_first_try():
    w = QWorld([kept_ticket()], [timed_out_wait()])
    w.check(wait_s=120.0).call({})
    assert w.wait_calls[0]["wait_s"] == 119.5                                      # the busy try itself took 0.5 s


def test_the_wait_also_gets_the_runs_timeout_with_the_cleanup_reserved():
    w = QWorld([kept_ticket()], [timed_out_wait()])
    w.check(lease_s=300.0, call_timeout_s=240.0, cleanup_s=120.0).call({})
    assert w.wait_calls[0]["timeout_s"] == 180.0


def test_a_zero_wait_still_hands_the_ticket_to_wait_for_so_that_it_is_cancelled():
    w = QWorld([kept_ticket()], [timed_out_wait(took=0.0)])
    w.check(wait_s=0.0).call({})
    assert w.wait_calls[0]["wait_s"] == 0.0


def test_a_board_granted_during_the_wait_runs_the_kernel_and_records_the_wait_and_the_lease_time():
    w = QWorld([kept_ticket()], [granted_pass(waited=21.0, elapsed=47.0)])
    c = w.check()
    text = c.call({})
    assert text.splitlines()[0].startswith("hardware_check 1 of 6: PASS")
    rec = c.summary()["calls"][0]
    assert rec["waited_s"] == 21.0 and rec["lease_s"] == 47.0 and rec["queued_polls"] == 1 and rec["cancelled"] is None
    assert rec["outcome"] == "pass" and c.summary()["lease_s_used"] == 47.0
    assert "waited 21.0 s for a lease" in text


def test_a_wait_that_ran_out_is_a_busy_call_that_uses_no_lease_time_and_counts():
    w = QWorld([kept_ticket()], [timed_out_wait()])
    c = w.check()
    c.call({})
    rec = c.summary()["calls"][0]
    assert rec["outcome"] == "busy" and rec["cancelled"] is True and rec["lease_s"] == 0.0 and rec["counted"] is True
    assert rec["queued_polls"] == 1 and rec["waited_s"] >= 119.0 and c.summary()["lease_s_used"] == 0.0
    assert c.summary()["calls_counted"] == 1


def test_a_ticket_that_could_not_be_cancelled_is_named_in_the_answer():
    w = QWorld([kept_ticket("T9")], [timed_out_wait(cancel_error="gozer cancel exited 1", ticket="T9")])
    c = w.check()
    text = c.call({})
    assert "could not be cancelled" in text and "T9" in text and "gozer cancel exited 1" in text
    assert c.summary()["calls"][0]["cancelled"] is False


def test_a_busy_answer_without_a_ticket_is_not_waited_on():
    nothing = dict(QUEUED, queued={"ticket": None, "cancel_error": "no ticket in gozer's output"})
    w = QWorld([(nothing, "queued", "")], [])
    text = w.check().call({})
    assert w.wait_calls == [] and "BUSY" in text and "no ticket" in text


def test_without_a_wait_function_a_busy_answer_is_final():
    w = World([queued()])
    c = w.check()
    assert "BUSY" in c.call({}) and len(w.runs) == 1


def test_a_hang_after_the_grant_is_charged_and_turns_the_tool_off():
    def hang(world, on_grant):
        world.now += 10.0
        on_grant()
        world.now += 66.0
        return ({"passed": False, "category": "hang", "first_line": "stalled: x", "pcc_min": None, "sets": [],
                 "violations": [], "elapsed_s": 66.0, "waited_s": 10.0}, "hang", "")
    w = QWorld([kept_ticket()], [hang])
    c = w.check()
    text = c.call({})
    assert "HANG" in text and c.summary()["lease_s_used"] == 66.0 and c.summary()["calls"][0]["waited_s"] == 10.0
    assert "hung" in c.call({})


# ---- interrupted waits and runs record the time ---------------------------------------------------------------------


def test_an_interrupt_while_waiting_records_the_wait_and_no_board_time():
    def interrupted(world, on_grant):
        world.now += 30.0
        raise KeyboardInterrupt
    w = QWorld([kept_ticket()], [interrupted])
    c = w.check()
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["outcome"] == "interrupted" and rec["waited_s"] == 30.5 and rec["total_s"] == 30.5
    assert rec["lease_s"] == 0.0 and not rec.get("lease_upper_bound") and rec["queued_polls"] == 1
    assert c.summary()["lease_s_used"] == 0.0


def test_an_interrupt_after_the_grant_records_the_wait_and_the_board_time_as_an_upper_bound():
    def interrupted(world, on_grant):
        world.now += 20.0
        on_grant()
        world.now += 5.0
        raise KeyboardInterrupt
    w = QWorld([kept_ticket()], [interrupted])
    c = w.check()
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["waited_s"] == 20.5 and rec["lease_s"] == 5.0 and rec["lease_upper_bound"] is True
    assert c.summary()["lease_s_used"] == 5.0


def test_an_interrupt_in_the_first_try_records_no_wait():
    w = QWorld()

    def boom(k, t):
        w.now += 7.0
        raise KeyboardInterrupt
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=boom, wait_for=w.wait_for, clock=w.clock,
                             sleep=w.sleep)
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["waited_s"] == 0.0 and rec["lease_s"] == 7.0 and rec["lease_upper_bound"] is True


def test_the_report_row_of_a_waited_call_says_one_ticket_and_whether_it_was_cancelled():
    from test_agent_hw import _report_md, _row_attempt, hw_summary
    busy = {"outcome": "busy", "passed": False, "lease_s": 0.0, "waited_s": 120.0, "queued_polls": 1, "cancelled": True}
    granted = {"outcome": "pass", "waited_s": 21.0, "queued_polls": 1, "cancelled": None}
    md = _report_md(__import__("pathlib").Path(__import__("tempfile").mkdtemp()),
                    [_row_attempt(agent_hw=hw_summary(busy, granted))])
    assert "1 ticket, cancelled" in md and "1 ticket, granted" in md


def test_the_report_row_shows_a_failed_release():
    from test_agent_hw import _report_md, _row_attempt, hw_summary
    rec = {"outcome": "pass", "waited_s": 21.0, "queued_polls": 1, "cancelled": None,
           "release_error": "chips [0, 1] still open by [123]"}
    md = _report_md(Path(__import__("tempfile").mkdtemp()), [_row_attempt(agent_hw=hw_summary(rec))])
    assert "release failed: chips [0, 1] still open by [123]" in md
