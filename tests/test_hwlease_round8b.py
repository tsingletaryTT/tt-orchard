# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 116, round 8, second file: the cases the first mutation table of the round left alive. Each test here was written for one
surviving mutation (the test name says what it pins). Fakes only."""
from __future__ import annotations

import signal

import pytest

from orchard import hwlease, hwtool
from test_agent_hw import CLOCK, GOOD_HW_SET, fake_kenv, skills_current, ttlang_skill  # noqa: F401
from test_agent_hw_paths import PID, WHO, Rig, SimGozer, busy_rig, res
from test_hwlease import FakeCmd, granted_event, leases
from test_hwlease_round8 import SteppedProc, held_chip
from test_hwlease_reap import ReapProc


# ---- run_reaping --------------------------------------------------------------------------------------------------------------------


def test_the_last_wait_is_tried_a_second_time_when_an_interrupt_lands_in_it():
    proc = SteppedProc("wait2")
    proc.fired = []

    def killpg(pgid, sig):
        proc.deliver(sig)
    with pytest.raises(KeyboardInterrupt):
        hwlease.run_reaping(["gozer", "acquire"], 5.0, popen=lambda argv, **k: proc, killpg=killpg)
    assert proc.wait_n == 3                       # the first wait, the interrupted last wait, and its second try


def test_an_interrupt_while_the_timed_out_child_is_killed_still_stops_it_with_sigterm_and_sigkill():
    proc = ReapProc(mode="hang", dies_on="KILL")
    sent = []

    def killpg(pgid, sig):
        sent.append(sig)
        if len(sent) == 1:
            raise KeyboardInterrupt                # the first signal never arrives: only the guarded stop can still end the child
        proc.deliver(sig)
    with pytest.raises(KeyboardInterrupt) as info:
        hwlease.run_reaping(["gozer", "acquire"], 5.0, popen=lambda argv, **k: proc, killpg=killpg)
    assert proc.returncode is not None and signal.SIGTERM in sent and sent.count(signal.SIGKILL) >= 1
    assert not hasattr(info.value, "secondary_interrupt") or info.value.secondary_interrupt is None


# ---- reclaim: known ids, retries, a fresh final read --------------------------------------------------------------------------------


def test_a_known_ticket_and_lease_that_every_read_shows_gone_are_not_cancelled_or_released_again():
    cmd = FakeCmd()
    result = leases(cmd).reclaim(known_tickets=["T9"], known_leases=["L9"])
    assert [s for s in cmd.subs() if s in ("cancel", "release")] == [] and result.left is None


def test_a_cancel_that_gozer_refuses_is_sent_a_second_time():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    cmd.say("cancel", res(1, "", "boom"), res(0, "cancelled"))
    leases(cmd).reclaim()
    assert cmd.subs().count("cancel") == 2


def test_a_cancel_that_works_is_sent_once():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    leases(cmd).reclaim()
    assert cmd.subs().count("cancel") == 1


def test_an_interrupted_final_read_is_repeated_and_decides():
    cmd = FakeCmd()
    original = cmd.run
    state = {"queue": 0}

    def run(argv, timeout, env=None, kill_on_timeout=True):
        if argv[1] == "queue":
            state["queue"] += 1
            if state["queue"] == 2:                       # the first read is the first call; the final read is the second
                raise KeyboardInterrupt
        return original(argv, timeout, env=env, kill_on_timeout=kill_on_timeout)
    cmd.run = run
    result = leases(cmd).reclaim()
    assert result.left is None and isinstance(result.secondary, KeyboardInterrupt) and state["queue"] == 3


def test_the_final_read_is_a_new_read_that_shows_a_grant_made_after_the_first_one():
    """Nothing was shown by the first read. A grant that lands before the final read must still be reported."""
    cmd = FakeCmd()
    original = cmd.run
    state = {"status": 0}

    def run(argv, timeout, env=None, kill_on_timeout=True):
        got = original(argv, timeout, env=env, kill_on_timeout=kill_on_timeout)
        if argv[1] == "status":
            state["status"] += 1
            if state["status"] == 1:                      # right after the first read, gozer grants a lease to this pid
                cmd.history.append(granted_event("L2"))
                cmd.status_chips = [held_chip("0000:01:00.0"), held_chip("0000:02:00.0")]
        return got
    cmd.run = run
    result = leases(cmd).reclaim()
    assert result.left is not None and "lease L2" in result.left


# ---- the probe's wiring ---------------------------------------------------------------------------------------------------------------


def interrupting_sleep(sim, *, queue_fails):
    def sleep(s):
        sim_state["n"] += 1
        sim.queue_fails = queue_fails
        raise KeyboardInterrupt
    sim_state = {"n": 0}
    return sleep


def test_an_interrupt_in_the_wait_cancels_the_ticket_the_probe_holds_even_when_the_queue_cannot_be_read(tmp_path, fake_kenv):
    sim = SimGozer(busy_calls=10**6)
    rig = Rig(tmp_path, fake_kenv, sim=sim, sleep=interrupting_sleep(sim, queue_fails=True))
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert sim.cancelled() == ["T1"] and sim.truth_tickets() == []


def test_a_second_interrupt_inside_the_cleanup_is_attached_to_the_exception_that_goes_on(tmp_path, fake_kenv):
    sim = SimGozer(busy_calls=10**6)
    rig = Rig(tmp_path, fake_kenv, sim=sim, sleep=interrupting_sleep(sim, queue_fails=False))
    original = sim.run
    fired = []

    def run(argv, timeout, env=None, kill_on_timeout=True):
        if argv[1] == "cancel" and not fired:
            fired.append(1)
            raise KeyboardInterrupt                       # a second Ctrl-C lands in the first cancel
        return original(argv, timeout, env=env, kill_on_timeout=kill_on_timeout)
    rig.leases.run = rig.leases.run_acquire = rig.leases.run_release = run
    with pytest.raises(KeyboardInterrupt) as info:
        rig.call()
    second = getattr(info.value, "secondary_interrupt", None)
    assert isinstance(second, KeyboardInterrupt) and second is not info.value
    assert sim.truth_tickets() == []                      # and the cancel was still made, on its second try


def test_an_interrupted_release_with_unreadable_history_is_waited_for_by_its_known_lease_id(tmp_path, fake_kenv):
    sim = SimGozer()
    sim.release_mode = "interrupt_running"
    sim.finish_after = None
    rig = Rig(tmp_path, fake_kenv, sim=sim)

    def harness(argv, kw):
        sim.history_fails = True                          # the history breaks while the harness runs
        return rig._watched_ok(argv, kw)
    rig.harness = harness
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert CLOCK["now"] >= hwlease.RELEASE_TIMEOUT_S      # the cleanup waited for the release it knows is running
