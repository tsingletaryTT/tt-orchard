# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Recovery in orchard/hwlease.py (PR 116, fifth review round): partial reads, the lease metadata that a reclaim returns, a release that
is still running when an interrupt lands, and claims and recovery reads that honour the wait deadline. Fakes only."""
from __future__ import annotations

import json

import pytest

from orchard import hwlease
from test_hwlease import GRANT, PID, QUEUED, WHO, FakeCmd, granted_event, leases, res


# ---- 1. one failing read must not discard what the other returned ---------------------------------------------------------


def test_a_queue_that_cannot_be_read_still_gives_the_leases_history_shows():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.queue_fails = True
    found = leases(cmd).mine()
    assert [x["lease_id"] for x in found.leases] == ["L1"] and found.tickets == [] and found.readable is False


def test_a_history_that_cannot_be_read_still_gives_the_tickets_the_queue_shows():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    cmd.history_fails = True
    found = leases(cmd).mine()
    assert found.tickets == ["T1"] and found.leases == [] and found.readable is False


def test_unresolved_names_what_it_can_see_and_says_what_it_cannot_confirm():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.queue_fails = True
    text = leases(cmd).unresolved()
    assert "lease L1" in text and "cannot confirm" in text and "queue" in text


def test_unresolved_with_a_failing_history_names_the_tickets_and_says_history_is_unreadable():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    cmd.history_fails = True
    text = leases(cmd).unresolved()
    assert "ticket T1" in text and "cannot confirm" in text and "history" in text


def test_a_reclaim_with_a_failing_queue_still_releases_the_lease_history_identified():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.queue_fails = True
    result = leases(cmd).reclaim()
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "release"] == ["L1"]
    assert "cannot confirm" in result.left                  # the queue is still unreadable, so the check still refuses new work


def test_a_reclaim_with_a_failing_history_still_cancels_the_tickets_the_queue_identified():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    cmd.history_fails = True
    result = leases(cmd).reclaim()
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "cancel"] == ["T1"]
    assert "cannot confirm" in result.left


def test_the_wait_still_finds_a_lease_when_only_the_queue_read_fails():
    cmd = FakeCmd()
    cmd.history = [granted_event("L7")]
    cmd.queue_fails = True
    cmd.say("acquire", res(1, "", "boom"))
    grant, _ = leases(cmd).wait("T1", "why", 60.0)
    assert grant["lease_id"] == "L7"


# ---- 2. a reclaim says what it released, before it releases it --------------------------------------------------------------


def test_a_reclaim_returns_the_leases_it_released_with_their_grant_time():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1", ts="2026-10-27T10:00:00Z")]
    result = leases(cmd).reclaim()
    assert [x["lease_id"] for x in result.leases] == ["L1"] and result.leases[0]["granted_at"] == "2026-10-27T10:00:00Z"


def test_the_found_callback_runs_before_the_release_so_the_chip_time_can_be_noted_first():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    order = []
    original = cmd.run

    def run(argv, timeout, env=None, kill_on_timeout=True):
        if argv[1] == "release":
            order.append("release")
        return original(argv, timeout, env=env, kill_on_timeout=kill_on_timeout)
    cmd.run = run
    led = leases(cmd)
    led.reclaim(on_found=lambda lease: order.append(f"found {lease['lease_id']}"))
    assert order == ["found L1", "release"]


def test_a_found_callback_that_raises_does_not_stop_the_release():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]

    def boom(lease):
        raise RuntimeError("callback failed")
    leases(cmd).reclaim(on_found=boom)
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "release"] == ["L1"]


# ---- 3. a release that is still running is waited for, not repeated ---------------------------------------------------------


def test_an_interrupted_release_is_in_flight_and_a_reclaim_waits_for_it_instead_of_releasing_again():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.say("release", KeyboardInterrupt())
    led = leases(cmd)
    with pytest.raises(KeyboardInterrupt):
        led.release("L1")
    assert "L1" in led.in_flight
    sleeps = led.time.sleeps

    def finish_after_two_polls(seconds):                     # gozer's release finishes while the reclaim waits
        sleeps.append(seconds)
        led.time.now += seconds
        if len(sleeps) == 2:
            cmd.history.append({"event": "released", "lease_id": "L1", "duration_s": 55.0})
    led.sleep = finish_after_two_polls
    result = led.reclaim()
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "release"] == ["L1"]      # the one interrupted release only
    assert result.left is None and "L1" not in led.in_flight


def test_a_release_that_never_finishes_is_waited_for_up_to_its_timeout_and_then_left_unresolved():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.say("release", KeyboardInterrupt())
    led = leases(cmd)
    with pytest.raises(KeyboardInterrupt):
        led.release("L1")
    result = led.reclaim()
    assert sum(led.time.sleeps) >= hwlease.RELEASE_TIMEOUT_S - hwlease.CLAIM_POLL_S
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "release"] == ["L1"]      # still only the first release
    assert "lease L1" in result.left


def test_a_release_that_overran_is_in_flight_too_and_is_not_repeated_by_a_reclaim():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.say("release", res(None, timed_out=True, left_running=True, pid=999))
    led = leases(cmd)
    assert led.release("L1").held is True and "L1" in led.in_flight
    led.reclaim()
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "release"] == ["L1"]


def test_a_release_that_completed_is_no_longer_in_flight():
    cmd = FakeCmd()
    led = leases(cmd)
    led.release("L1")
    assert "L1" not in led.in_flight


def test_a_failed_release_is_not_in_flight_and_a_reclaim_may_try_again():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.say("release", res(15, "", "device still open"), res(0, json.dumps({"message": "ok"})))
    led = leases(cmd)
    led.release("L1")
    assert "L1" not in led.in_flight
    led.reclaim()
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "release"] == ["L1", "L1"]


# ---- 4. claims and recovery honour the remaining wait ------------------------------------------------------------------------


def claim_timeouts(cmd):
    return [c["timeout"] for c in cmd.calls if c["argv"][1] == "acquire"]


def test_a_claim_is_given_no_more_than_the_wait_that_is_left_but_not_less_than_the_floor():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    led = leases(cmd)
    led.wait("T1", "why", 1.0)
    assert claim_timeouts(cmd)[0] == hwlease.CLAIM_MIN_TIMEOUT_S


def test_a_long_wait_gets_the_full_claim_timeout_at_first_and_less_as_the_deadline_nears():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    led = leases(cmd)
    led.wait("T1", "why", 70.0)
    timeouts = claim_timeouts(cmd)
    assert timeouts[0] == hwlease.CLAIM_TIMEOUT_S and timeouts[-1] == hwlease.CLAIM_MIN_TIMEOUT_S
    assert timeouts == sorted(timeouts, reverse=True)


def test_no_claim_starts_after_the_deadline():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    led = leases(cmd)
    led.wait("T1", "why", 4.0)
    assert len(claim_timeouts(cmd)) == 3                     # at 0, 2 and 4 s; none at 6 s


def test_a_zero_wait_still_makes_the_immediate_check():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    leases(cmd).wait("T1", "why", 0.0)
    assert len(claim_timeouts(cmd)) == 1


def test_the_recovery_reads_after_a_failed_claim_use_the_short_recovery_timeout_not_the_claim_timeout():
    cmd = FakeCmd()
    cmd.say("acquire", res(1, "", "boom"))
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    leases(cmd).wait("T1", "why", 0.0)
    reads = [c["timeout"] for c in cmd.calls if c["argv"][1] in ("history", "queue")]
    assert reads and all(t <= hwlease.RECOVERY_TIMEOUT_S for t in reads)


def test_a_claim_that_times_out_with_the_deadline_nearly_spent_still_recovers_a_lease_it_granted():
    cmd = FakeCmd()
    cmd.say("acquire", res(None, timed_out=True))
    cmd.history = [granted_event("L9")]
    grant, _ = leases(cmd).wait("T1", "why", 0.5)
    assert grant["lease_id"] == "L9"


def test_the_constants_are_ordered_so_a_claim_floor_fits_inside_the_claim_timeout():
    assert 0 < hwlease.CLAIM_MIN_TIMEOUT_S <= hwlease.CLAIM_TIMEOUT_S and hwlease.RECOVERY_TIMEOUT_S <= hwlease.CLAIM_TIMEOUT_S


# ---- details the mutation table asked for ---------------------------------------------------------------------------------


def test_a_release_that_could_not_start_is_not_in_flight_so_a_reclaim_may_release_again():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.say("release", OSError("no gozer"), res(0, json.dumps({"message": "ok"})))
    led = leases(cmd)
    assert led.release("L1").held is True and "L1" not in led.in_flight
    led.reclaim()
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "release"] == ["L1", "L1"]


def test_an_overrunning_release_keeps_the_pid_of_the_command_that_was_left_running():
    cmd = FakeCmd()
    cmd.say("release", res(None, timed_out=True, left_running=True, pid=999))
    led = leases(cmd)
    led.release("L1")
    assert led.in_flight["L1"] == 999


def test_a_reaped_lease_counts_as_finished_for_the_wait():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1"), {"event": "reaped", "lease_id": "L1"}]
    led = leases(cmd)
    led.in_flight["L1"] = None
    assert led._await_release("L1") is True and "L1" not in led.in_flight and led.time.sleeps == []


def test_another_leases_release_does_not_end_the_wait_for_this_one():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1"), {"event": "released", "lease_id": "L2", "duration_s": 5.0}]
    led = leases(cmd)
    led.in_flight["L1"] = None
    assert led._await_release("L1") is False and "L1" in led.in_flight


def test_the_reads_of_a_reclaim_use_the_recovery_timeout():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    leases(cmd).reclaim()
    reads = [c["timeout"] for c in cmd.calls if c["argv"][1] in ("history", "queue")][:2]
    assert reads == [hwlease.RECOVERY_TIMEOUT_S, hwlease.RECOVERY_TIMEOUT_S]
