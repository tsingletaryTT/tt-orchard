# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Round 8 (PR 116): interrupts that land inside the cleanup, current ownership read from `gozer status` rather than from a windowed
history, a final read that is fresh, known tickets and leases cleaned up even when a read fails, and the wait deadline checked before
every claim. Fakes only."""
from __future__ import annotations

import json
import signal
import subprocess

import pytest

from orchard import hwlease
from test_hwlease import PID, QUEUED, WHO, FakeCmd, granted_event, leases, res
from test_hwlease_reap import ReapProc


def held_chip(bdf="0000:01:00.0", pid=PID, who=WHO):
    return {"bdf": bdf, "pid": pid, "who": who, "state": "CLAIMED"}


# ---- status: the current ownership, not a window of history ---------------------------------------------------------------------


def test_mine_reads_status_and_a_status_that_cannot_be_read_is_not_readable():
    cmd = FakeCmd()
    cmd.status_fails = True
    found = leases(cmd).mine()
    assert found.readable is False
    assert any(c["argv"][1] == "status" for c in cmd.calls)


def test_a_held_chip_of_ours_with_no_lease_in_history_triggers_one_wider_history_read():
    cmd = FakeCmd()
    cmd.status_chips = [held_chip()]
    cmd.history = []
    leases(cmd).mine()
    sizes = [c["argv"][c["argv"].index("-n") + 1] for c in cmd.calls if c["argv"][1] == "history"]
    assert sizes == [str(hwlease.HISTORY_EVENTS), str(hwlease.HISTORY_WIDE_EVENTS)]


def test_the_wider_read_finds_the_lease_the_default_window_missed():
    cmd = FakeCmd()
    cmd.status_chips = [held_chip()]
    original = cmd.run

    def run(argv, timeout, env=None, kill_on_timeout=True):
        if argv[1] == "history" and argv[argv.index("-n") + 1] == str(hwlease.HISTORY_EVENTS):
            cmd.calls.append({"argv": list(argv), "timeout": timeout, "kill_on_timeout": kill_on_timeout})
            return res(0, json.dumps({"history": []}))                  # the window shows nothing
        return original(argv, timeout, env=env, kill_on_timeout=kill_on_timeout)
    cmd.run = run
    cmd.history = [granted_event("L1")]
    found = leases(cmd).mine()
    assert [x["lease_id"] for x in found.leases] == ["L1"] and found.unidentified == []


def test_held_chips_with_no_lease_id_even_in_the_wider_read_are_unidentified_and_fail_closed():
    cmd = FakeCmd()
    cmd.status_chips = [held_chip("0000:01:00.0"), held_chip("0000:02:00.0")]
    cmd.history = []
    led = leases(cmd)
    found = led.mine()
    assert found.unidentified == ["0000:01:00.0", "0000:02:00.0"] and found.leases == []
    text = led.unresolved()
    assert "no lease id" in text and "0000:01:00.0" in text and "gozer status" in text


def test_chips_held_by_another_pid_or_who_are_not_ours():
    cmd = FakeCmd()
    cmd.status_chips = [held_chip(pid=PID + 1), held_chip(who="claude:other"), {"bdf": "x", "pid": None, "who": None, "state": "FREE"}]
    found = leases(cmd).mine()
    assert found.unidentified == [] and leases(cmd).unresolved() is None


def test_a_lease_whose_chips_status_shows_is_not_unidentified():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.status_chips = [held_chip("0000:01:00.0"), held_chip("0000:02:00.0")]
    found = leases(cmd).mine()
    assert found.unidentified == [] and [x["lease_id"] for x in found.leases] == ["L1"]


def test_unresolved_with_a_failing_status_names_what_history_shows_and_says_it_cannot_confirm():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.status_fails = True
    text = leases(cmd).unresolved()
    assert "lease L1" in text and "cannot confirm" in text and "status" in text


# ---- known tickets and leases are cleaned up even when a read fails -------------------------------------------------------------


def test_a_known_ticket_is_cancelled_even_when_the_queue_cannot_be_read():
    cmd = FakeCmd()
    cmd.queue_fails = True
    result = leases(cmd).reclaim(known_tickets=["T1"])
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "cancel"] == ["T1"]
    assert "cannot confirm" in result.left


def test_a_known_lease_is_released_even_when_history_cannot_be_read():
    cmd = FakeCmd()
    cmd.history_fails = True
    result = leases(cmd).reclaim(known_leases=["L1"])
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "release"] == ["L1"]
    assert "cannot confirm" in result.left


def test_a_ticket_both_known_and_shown_is_cancelled_once_and_a_lease_both_known_and_shown_is_released_once():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    cmd.history = [granted_event("L1")]
    leases(cmd).reclaim(known_tickets=["T1"], known_leases=["L1"])
    subs = [(c["argv"][1], c["argv"][2]) for c in cmd.calls if c["argv"][1] in ("cancel", "release")]
    assert subs == [("cancel", "T1"), ("release", "L1")]


def test_a_known_lease_that_is_in_flight_is_waited_for_not_released_again():
    cmd = FakeCmd()
    led = leases(cmd)
    led.in_flight["L1"] = None
    led.reclaim(known_leases=["L1"])
    assert [c for c in cmd.calls if c["argv"][1] == "release"] == []


# ---- the final read is fresh ----------------------------------------------------------------------------------------------------


def test_the_final_read_of_a_reclaim_is_made_after_the_last_release_not_taken_from_the_first_read():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    order = []
    original = cmd.run

    def run(argv, timeout, env=None, kill_on_timeout=True):
        order.append(argv[1])
        if argv[1] == "release":
            cmd.history.append({"event": "released", "lease_id": "L1", "duration_s": 40.0})
        return original(argv, timeout, env=env, kill_on_timeout=kill_on_timeout)
    cmd.run = run
    result = leases(cmd).reclaim()
    assert result.left is None
    last_release = max(i for i, s in enumerate(order) if s == "release")
    assert {"history", "queue", "status"} <= set(order[last_release + 1:])                  # all three re-read after it


def test_a_reclaim_whose_release_did_not_take_still_reports_the_lease():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.say("release", res(15, "", "device still open"))
    assert "lease L1" in leases(cmd).reclaim().left


# ---- a second interrupt inside the cleanup -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("step", ["read", "cancel", "release", "final-read"])
def test_a_second_interrupt_at_each_reclaim_step_is_recorded_and_the_other_steps_still_run(step):
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.history[0]["chips"] = ["0000:01:00.0"]
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    cmd.status_chips = [held_chip("0000:01:00.0")]
    original = cmd.run
    trigger = {"read": ("history", 1), "cancel": ("cancel", 1), "release": ("release", 1), "final-read": ("queue", 2)}[step]
    counts, hit, attempted = {}, [], []                              # attempted: every command started, the interrupted one too

    def run(argv, timeout, env=None, kill_on_timeout=True):
        sub = argv[1]
        counts[sub] = counts.get(sub, 0) + 1
        attempted.append(sub)
        if sub == trigger[0] and counts[sub] == trigger[1] and not hit:
            hit.append(sub)
            raise KeyboardInterrupt
        return original(argv, timeout, env=env, kill_on_timeout=kill_on_timeout)
    cmd.run = run
    led = leases(cmd)
    result = led.reclaim()                                          # a second interrupt does not escape the cleanup
    assert hit and isinstance(result.secondary, KeyboardInterrupt)
    subs = attempted
    assert "cancel" in subs and "release" in subs
    assert subs[-1] in ("history", "queue", "status")                  # and a final read was still made after the interrupted step


def test_a_second_interrupt_in_the_wait_for_a_release_is_recorded_and_the_lease_is_left_unresolved():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    led = leases(cmd)
    led.in_flight["L1"] = None
    state = {"n": 0}

    def sleep(s):
        state["n"] += 1
        raise KeyboardInterrupt
    led.sleep = sleep
    result = led.reclaim()
    assert isinstance(result.secondary, KeyboardInterrupt) and "lease L1" in result.left


def test_the_first_exception_is_never_replaced_by_the_cleanup_result():
    """reclaim never raises: the caller re-raises its own first exception and attaches the secondary one."""
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.say("release", KeyboardInterrupt())
    result = leases(cmd).reclaim()
    assert isinstance(result.secondary, KeyboardInterrupt)


# ---- run_reaping: an interrupt injected at each step of the stop ----------------------------------------------------------------------


class SteppedProc(ReapProc):
    """A child whose stop sequence is interrupted at one chosen step: "term" (the SIGTERM send), "wait1", "kill" (the SIGKILL send),
    "wait2" (the last wait). The child ignores SIGTERM, so every step is reached."""

    def __init__(self, interrupt_at):
        super().__init__(mode="interrupt", dies_on="KILL")
        self.interrupt_at = interrupt_at
        self.wait_n = 0

    def wait(self, timeout=None):
        self.wait_n += 1
        step = "wait1" if self.wait_n == 1 else "wait2"
        if self.interrupt_at == step and step not in self.fired:
            self.fired.append(step)
            raise KeyboardInterrupt
        return super().wait(timeout)

    fired: list = []


@pytest.mark.parametrize("at", ["term", "wait1", "kill", "wait2"])
def test_an_interrupt_at_each_step_of_the_stop_still_kills_and_waits_for_the_child(at):
    proc = SteppedProc(at)
    proc.fired = []
    sent = []

    def killpg(pgid, sig):
        sent.append(sig)
        if at == "term" and sig == signal.SIGTERM and "term" not in proc.fired:
            proc.fired.append("term")
            raise KeyboardInterrupt
        if at == "kill" and sig == signal.SIGKILL and "kill" not in proc.fired:
            proc.fired.append("kill")
            proc.deliver(sig)                                       # the signal went out; the interrupt landed right after
            raise KeyboardInterrupt
        proc.deliver(sig)
    with pytest.raises(KeyboardInterrupt) as info:
        hwlease.run_reaping(["gozer", "acquire"], 5.0, popen=lambda argv, **k: proc, killpg=killpg)
    assert signal.SIGKILL in sent and proc.returncode is not None            # killed and reaped whatever step was interrupted
    assert proc.wait_n >= 2 or at in ("wait1", "wait2")
    assert getattr(info.value, "secondary_interrupt", None) is not None      # the second interrupt is recorded on the first


def test_an_interrupt_during_the_timeout_kill_still_stops_and_reaps_the_child():
    proc = ReapProc(mode="hang", dies_on="KILL")
    sent = []

    def killpg(pgid, sig):
        sent.append(sig)
        proc.deliver(sig)
        if len(sent) == 1:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        hwlease.run_reaping(["gozer", "acquire"], 5.0, popen=lambda argv, **k: proc, killpg=killpg)
    assert proc.returncode is not None


# ---- the wait deadline is checked before every claim ----------------------------------------------------------------------------------


def test_a_late_wake_up_does_not_start_another_claim_after_the_deadline():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    led = leases(cmd)
    original = led.sleep

    def late_sleep(s):
        original(s + 100.0)                                         # the scheduler wakes the process far too late
    led.sleep = late_sleep
    led.wait("T1", "why", 4.0)
    claims = [c for c in cmd.calls if c["argv"][1] == "acquire"]
    assert len(claims) == 1                                          # only the immediate first claim


def test_a_zero_wait_still_makes_exactly_the_immediate_claim():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    leases(cmd).wait("T1", "why", 0.0)
    assert len([c for c in cmd.calls if c["argv"][1] == "acquire"]) == 1


def test_a_failed_claim_after_the_deadline_is_not_followed_by_another():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    cmd.say("acquire", res(1, "", "boom"))
    led = leases(cmd)
    original = led.sleep
    led.sleep = lambda s: original(s + 100.0)
    led.wait("T1", "why", 4.0)
    assert len([c for c in cmd.calls if c["argv"][1] == "acquire"]) == 1
