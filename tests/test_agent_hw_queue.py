# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The policy object's side of waiting for a lease and of leases that are not seen released (orchard/hwtool.py): the wait is
part of one run, interrupts record the wait and the board time apart, and the time of a lease whose release was not seen is
replaced by the duration gozer records. Fakes only: no gozer, no chip."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from orchard import hwtool
from test_agent_hw import QUEUED, World, good_hw, queued, skills_current, ttlang_skill  # noqa: F401


def busy_answer(ticket="T1", cancel_error=None):
    return (dict(QUEUED, queued={"ticket": ticket, "cancel_error": cancel_error}), "queued", "")


def granted_script(waited=21.0, elapsed=47.0, **extra):
    def script(world, on_grant, ctx):
        world.now += waited
        ctx["lease_id"] = "L1"
        on_grant()
        world.now += elapsed
        return (dict(good_hw(), elapsed_s=elapsed, waited_s=waited, lease_id="L1", **extra), None, "")
    return script


def test_the_run_is_given_the_whole_wait_to_use():
    w = World([busy_answer()])
    w.check(wait_s=120.0).call({})
    assert w.run_args == [{"wait_s": 120.0}] and len(w.runs) == 1


def test_a_negative_wait_is_passed_as_zero():
    w = World([busy_answer()])
    w.check(wait_s=-5.0).call({})
    assert w.run_args == [{"wait_s": 0.0}]


def test_a_busy_answer_is_a_counted_call_that_uses_no_lease_time_and_says_one_ticket():
    w = World([busy_answer()])
    c = w.check()
    text = c.call({})
    rec = c.summary()["calls"][0]
    assert "BUSY" in text and "simulator" in text and "were cancelled" in text and "waited on 1 queue ticket" in text
    assert rec["outcome"] == "busy" and rec["cancelled"] is True and rec["lease_s"] == 0.0 and rec["counted"] is True
    assert rec["queued_polls"] == 1 and c.summary()["lease_s_used"] == 0.0 and c.summary()["calls_counted"] == 1


def test_a_ticket_that_could_not_be_cancelled_is_named_in_the_answer():
    w = World([busy_answer("T9", "gozer cancel exited 1")])
    c = w.check()
    text = c.call({})
    assert "could not be cancelled" in text and "T9" in text and "gozer cancel exited 1" in text
    assert c.summary()["calls"][0]["cancelled"] is False


def test_a_busy_answer_without_a_ticket_says_so():
    nothing = dict(QUEUED, queued={"ticket": None, "cancel_error": "no ticket"})
    text = World([(nothing, "queued", "")]).check().call({})
    assert "BUSY" in text and "no ticket" in text


def test_a_grant_during_the_wait_records_the_wait_and_the_lease_time_apart():
    w = World([granted_script(waited=21.0, elapsed=47.0)])
    c = w.check()
    text = c.call({})
    rec = c.summary()["calls"][0]
    assert text.splitlines()[0].startswith("hardware_check 1 of 6: PASS") and "waited 21.0 s for a lease" in text
    assert rec["waited_s"] == 21.0 and rec["lease_s"] == 47.0 and rec["queued_polls"] == 0 and rec["cancelled"] is None
    assert rec["lease_id"] == "L1" and rec["lease_incomplete"] is False and c.summary()["lease_s_used"] == 47.0


def test_a_hang_after_the_grant_is_charged_and_turns_the_tool_off():
    def hang(world, on_grant, ctx):
        world.now += 10.0
        ctx["lease_id"] = "L1"
        on_grant()
        world.now += 66.0
        return ({"passed": False, "category": "hang", "first_line": "stalled: x", "pcc_min": None, "sets": [],
                 "violations": [], "elapsed_s": 66.0, "waited_s": 10.0, "lease_id": "L1"}, "hang", "")
    w = World([hang])
    c = w.check()
    text = c.call({})
    assert "HANG" in text and c.summary()["lease_s_used"] == 66.0 and c.summary()["calls"][0]["waited_s"] == 10.0
    assert "hung" in c.call({})


# ---- interrupted waits and runs record the time ---------------------------------------------------------------------


def test_an_interrupt_while_waiting_records_the_wait_and_no_board_time():
    def interrupted(world, on_grant, ctx):
        world.now += 30.0
        raise KeyboardInterrupt
    w = World([interrupted])
    c = w.check()
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["outcome"] == "interrupted" and rec["waited_s"] == 30.0 and rec["total_s"] == 30.0
    assert rec["lease_s"] == 0.0 and not rec.get("lease_upper_bound") and rec["lease_incomplete"] is False
    assert c.summary()["lease_s_used"] == 0.0


def test_an_interrupt_after_the_grant_records_the_wait_and_the_board_time_as_an_upper_bound():
    def interrupted(world, on_grant, ctx):
        world.now += 20.0
        ctx["lease_id"] = "L1"
        on_grant()
        world.now += 5.0
        raise KeyboardInterrupt
    w = World([interrupted])
    c = w.check()
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["waited_s"] == 20.0 and rec["lease_s"] == 5.0 and rec["lease_upper_bound"] is True
    assert rec["lease_id"] == "L1" and rec["lease_incomplete"] is True and c.summary()["lease_s_used"] == 5.0


def test_an_interrupt_before_any_wait_records_no_wait_and_the_run_time():
    def interrupted(world, on_grant, ctx):
        on_grant()
        world.now += 7.0
        raise KeyboardInterrupt
    w = World([interrupted])
    c = w.check()
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["waited_s"] == 0.0 and rec["lease_s"] == 7.0 and rec["lease_upper_bound"] is True


def test_the_wait_is_not_counted_as_board_time_when_the_grant_came_late():
    def interrupted(world, on_grant, ctx):
        world.now += 100.0
        on_grant()
        world.now += 2.0
        raise KeyboardInterrupt
    w = World([interrupted])
    c = w.check()
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["waited_s"] == 100.0 and rec["lease_s"] == 2.0


# ---- the time of a lease whose release was not seen -----------------------------------------------------------------------


def make(w, durations, **policy):
    seen = []

    def settle(lease_id):
        seen.append(lease_id)
        return durations.get(lease_id)
    c = hwtool.HardwareCheck(hwtool.HwPolicy(**policy), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep,
                             settle=settle)
    c.settled = seen
    return c


def incomplete_pass(elapsed=60.0, lease="L1"):
    return (dict(good_hw(), elapsed_s=elapsed, lease_id=lease, lease_incomplete=True,
                 release_error="gozer release L1 is still running"), None, "")


def test_a_call_whose_release_was_not_confirmed_is_settled_by_the_duration_gozer_records_later():
    w = World([incomplete_pass(60.0)])
    durations = {}
    c = make(w, durations, lease_s=1000.0)
    c.call({})
    assert c.summary()["lease_s_used"] == 60.0 and c.summary()["calls"][0]["lease_incomplete"] is True
    durations["L1"] = 500.0                                   # the reset ran long; gozer wrote the release at last
    c.call({})                                                # the next call settles it first
    rec = c.summary()["calls"][0]
    assert rec["lease_s"] == 500.0 and rec["lease_incomplete"] is False and rec["lease_reconciled"] is True
    assert c.summary()["lease_s_used"] == 500.0 + 47.0


def test_an_overlong_release_can_exhaust_the_budget_before_the_next_call_is_admitted():
    w = World([incomplete_pass(60.0)])
    durations = {"L1": 500.0}
    c = make(w, durations, lease_s=600.0)
    c.call({})                                                # 60 s charged, then settled to 500 s at the end of the call
    assert c.summary()["lease_s_used"] == 500.0
    assert c.call({}).startswith("hardware_check refused") and len(w.runs) == 1


def test_a_lease_that_gozer_has_no_duration_for_keeps_the_estimate_and_stays_incomplete():
    w = World([incomplete_pass(60.0)])
    c = make(w, {})
    c.call({})
    c.call({})
    rec = c.summary()["calls"][0]
    assert rec["lease_s"] == 60.0 and rec["lease_incomplete"] is True and c.summary()["lease_s_used"] == 60.0 + 47.0


def test_a_duration_shorter_than_the_estimate_never_lowers_the_charge():
    w = World([incomplete_pass(60.0)])
    c = make(w, {"L1": 30.0})
    c.call({})
    assert c.summary()["lease_s_used"] == 60.0 and c.summary()["calls"][0]["lease_s"] == 30.0


def test_an_interrupted_call_is_settled_when_its_call_ends():
    def interrupted(world, on_grant, ctx):
        ctx["lease_id"] = "L1"
        on_grant()
        world.now += 5.0
        raise KeyboardInterrupt
    w = World([interrupted])
    c = make(w, {"L1": 44.0})
    with pytest.raises(KeyboardInterrupt):
        c.call({})
    rec = c.summary()["calls"][0]
    assert rec["lease_s"] == 44.0 and rec["lease_incomplete"] is False and c.summary()["lease_s_used"] == 44.0
    assert "lease_upper_bound" not in rec


def test_a_settle_that_raises_changes_nothing():
    w = World([incomplete_pass(60.0)])

    def boom(lease_id):
        raise RuntimeError("history unreadable")
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep, settle=boom)
    c.call({})
    assert c.summary()["calls"][0]["lease_s"] == 60.0 and c.summary()["calls"][0]["lease_incomplete"] is True


def test_a_settled_duration_that_is_not_a_number_is_ignored():
    w = World([incomplete_pass(60.0)])
    c = make(w, {"L1": "soon"})
    c.call({})
    assert c.summary()["calls"][0]["lease_incomplete"] is True


def test_the_settled_record_is_written_to_disk():
    w = World([incomplete_pass(60.0)])
    saved = []
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep,
                             settle=lambda lease: 500.0, on_record=saved.append)
    c.call({})
    assert saved[-1]["calls"][0]["lease_s"] == 500.0


# ---- the report -----------------------------------------------------------------------------------------------------


def test_the_report_row_of_a_waited_call_says_one_ticket_and_whether_it_was_cancelled():
    from test_agent_hw import _report_md, _row_attempt, hw_summary
    busy = {"outcome": "busy", "passed": False, "lease_s": 0.0, "waited_s": 120.0, "queued_polls": 1, "cancelled": True}
    granted = {"outcome": "pass", "waited_s": 21.0, "queued_polls": 1, "cancelled": None}
    md = _report_md(Path(tempfile.mkdtemp()), [_row_attempt(agent_hw=hw_summary(busy, granted))])
    assert "1 ticket, cancelled" in md and "1 ticket, granted" in md


def test_the_report_row_shows_a_failed_release():
    from test_agent_hw import _report_md, _row_attempt, hw_summary
    rec = {"outcome": "pass", "waited_s": 21.0, "queued_polls": 1, "cancelled": None,
           "release_error": "chips [0, 1] still open by [123]"}
    md = _report_md(Path(tempfile.mkdtemp()), [_row_attempt(agent_hw=hw_summary(rec))])
    assert "release failed: chips [0, 1] still open by [123]" in md


def test_a_duration_that_gozer_writes_between_calls_counts_before_the_next_call_is_admitted():
    w = World([incomplete_pass(60.0)])
    durations = {}
    c = make(w, durations, lease_s=600.0)
    c.call({})
    assert c.summary()["lease_s_used"] == 60.0                # nothing known yet: the estimate stands
    durations["L1"] = 500.0                                   # gozer finished the long reset between the calls
    assert c.call({}).startswith("hardware_check refused") and len(w.runs) == 1
    assert c.summary()["lease_s_used"] == 500.0
