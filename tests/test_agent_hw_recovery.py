# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 116, fifth review round, through the probe's wiring and the simulated gozer: partial reads, a lost acquire reply whose lease must
be reclaimed and charged, an interrupted release that is still running, a short wait with recovery, and the cleanup reserve pinned
where it is checked. No real gozer, no chip."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import hwlease, hwtool, kernelprobe as kp
from test_agent_hw import CLOCK, GOOD_HW_SET, attempt_dir, fake_kenv, skills_current, ttlang_skill  # noqa: F401
from test_agent_hw_paths import PID, WHO, Rig, SimGozer, busy_rig, res


# ---- 1. one failing read must not hide what the other returned ------------------------------------------------------------


def test_a_refused_release_with_a_failing_queue_still_names_the_lease_and_blocks_the_next_call(rig_factory):
    rig = rig_factory()
    rig.sim.release_mode = "refused"
    rig.call()
    rig.sim.queue_fails = True
    second = rig.call()
    assert second.startswith("hardware_check refused") and "lease L1" in second and "cannot confirm" in second
    assert len(rig.sim.requests()) == 1


def test_a_failing_history_with_a_readable_queue_still_names_the_ticket(tmp_path, fake_kenv):
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=10**6, policy=hwtool.HwPolicy(wait_s=0.0))
    rig.sim.cancel_mode = "fail"
    rig.call()
    rig.sim.history_fails = True
    second = rig.call()
    assert second.startswith("hardware_check refused") and "ticket T1" in second and "cannot confirm" in second


def test_an_interrupt_with_a_failing_queue_still_releases_the_lease_history_shows(rig_factory):
    rig = rig_factory()

    def harness(argv, kw):
        rig.sim.queue_fails = True                       # the queue breaks while the harness runs
        raise KeyboardInterrupt
    rig.harness = harness
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert rig.sim.released() == ["L1"]


def test_an_interrupt_with_a_failing_history_still_cancels_the_ticket_the_queue_shows(tmp_path, fake_kenv):
    state = {}

    def interrupt(s):
        state["rig"].sim.history_fails = True            # history breaks while the call waits
        raise KeyboardInterrupt
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=10**6, sleep=interrupt)
    state["rig"] = rig
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert rig.sim.cancelled() == ["T1"] and rig.sim.queue == []


@pytest.fixture
def rig_factory(tmp_path, fake_kenv):
    return lambda **kw: Rig(tmp_path, fake_kenv, **kw)


# ---- 2. a lost acquire reply: the lease is found, released and its time charged ---------------------------------------------


def lost_reply(rig, when="request"):
    original = rig.sim._acquire

    def lost(argv):
        out = original(argv)
        if (when == "request") == ("--ticket" not in argv) and out.returncode == 0:
            raise OSError("the reply was lost after gozer granted the lease")
        return out
    rig.sim._acquire = lost


def test_a_lease_granted_to_a_request_whose_reply_was_lost_is_released_and_its_chip_time_is_charged(rig_factory):
    rig = rig_factory()
    lost_reply(rig, "request")
    text = rig.call()
    assert "gozer could not give a lease" in text and rig.harness_calls == [] and rig.sim.released() == ["L1"]
    rec = rig.check.summary()["calls"][0]
    assert rec["lease_id"] == "L1" and rec["lease_incomplete"] is False and rec["lease_reconciled"] is True
    assert rec["lease_s"] >= 42.0 and rig.check.summary()["lease_s_used"] >= 42.0         # gozer's duration, reset included


def test_the_charged_time_of_a_lost_reply_counts_against_the_budget_before_the_next_call(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv, policy=hwtool.HwPolicy(lease_s=300.0))
    lost_reply(rig, "request")
    rig.call()
    rig.sim._acquire = SimGozer._acquire.__get__(rig.sim)                                    # replies arrive again
    assert rig.check.summary()["lease_s_used"] >= 42.0
    assert rig.call().startswith("hardware_check refused")                                  # 300 - 42 - 210 reserve < 60 s needed


def test_a_claim_that_gozer_granted_and_that_was_interrupted_before_its_id_was_noted_is_charged_too(tmp_path, fake_kenv):
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=1)
    original = rig.sim._acquire

    def claim_then_interrupt(argv):
        if "--ticket" in argv:
            rig.sim.busy_calls = 0
            original(argv)
            raise KeyboardInterrupt
        return original(argv)
    rig.sim._acquire = claim_then_interrupt
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    rec = rig.check.summary()["calls"][0]
    assert rig.sim.released() == ["L1"] and rec["lease_id"] == "L1" and rec["outcome"] == "interrupted"
    assert rec["lease_s"] >= 42.0 and rec["lease_incomplete"] is False                       # settled from gozer's duration
    assert rig.check.summary()["lease_s_used"] >= 42.0


def test_the_recovered_grant_is_noted_before_it_is_released(tmp_path, fake_kenv):
    """The hardware policy learns of the grant first, so an interrupt during the release still has a clock to read."""
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=1)
    original = rig.sim._acquire
    order = []

    def claim_then_interrupt(argv):
        if "--ticket" in argv:
            rig.sim.busy_calls = 0
            original(argv)
            raise KeyboardInterrupt
        return original(argv)
    rig.sim._acquire = claim_then_interrupt
    release = rig.sim._release

    def watching(argv):
        order.append(("release", rig.check.summary()["calls"]))
        return release(argv)
    rig.sim._release = watching
    real_on_grant_order = []
    original_run = rig.check.run

    def run(k, timeout_s, wait_s, on_grant, ctx):
        def noting():
            real_on_grant_order.append("on_grant")
            on_grant()
        return original_run(k, timeout_s, wait_s, noting, ctx)
    rig.check.run = run
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert real_on_grant_order == ["on_grant"] and order and rig.sim.released() == ["L1"]


# ---- 3. a release that is still running when an interrupt lands --------------------------------------------------------------


def test_an_interrupt_during_a_running_release_is_waited_for_and_then_charged_from_gozers_duration(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv)
    rig.sim.release_mode = "interrupt_running"
    rig.sim.finish_after = 30.0
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert rig.sim.released() == ["L1"] and rig.leases.unresolved() is None
    assert rig.check.summary()["calls"][0]["lease_s"] >= 42.0


# ---- 4. a short wait is not stretched by the recovery ----------------------------------------------------------------------


def test_a_short_wait_gives_its_claims_the_floor_and_its_recovery_reads_the_recovery_timeout(tmp_path, fake_kenv):
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=10**6, policy=hwtool.HwPolicy(wait_s=1.0))
    rig.sim.acquire_fail_at = {2: res(1, "", "boom")}
    rig.call()
    claims = [c for c in rig.sim.calls if c["argv"][1] == "acquire" and "--ticket" in c["argv"]]
    assert claims and all(c["timeout"] == hwlease.CLAIM_MIN_TIMEOUT_S for c in claims)
    reads = [c["timeout"] for c in rig.sim.calls if c["argv"][1] in ("history", "queue")][2:]            # after the health check
    assert reads and all(t <= hwlease.RECOVERY_TIMEOUT_S for t in reads[:4])


def test_no_claim_is_started_after_the_wait_is_over(tmp_path, fake_kenv):
    rig = busy_rig(tmp_path, fake_kenv, busy_calls=10**6, policy=hwtool.HwPolicy(wait_s=4.0))
    rig.call()
    assert len([c for c in rig.sim.calls if c["argv"][1] == "acquire" and "--ticket" in c["argv"]]) == 3


# ---- 5. the cleanup reserve, where it is checked ------------------------------------------------------------------------------


def test_the_harness_deadline_and_both_cleanup_phases_fit_inside_the_lease_cap(tmp_path, fake_kenv):
    """The reserve is checked in HardwareCheck._call: the harness backstop passed to run_watched is the lesser of the call timeout and
    the lease left minus the reserve; the stop (term grace) and the separate release (its timeout) are what the reserve is for."""
    rig = Rig(tmp_path, fake_kenv)
    seen = []

    def harness(argv, kw):
        seen.append(kw)
        Path(argv[argv.index("--out") + 1]).write_text(
            json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 1.0}))
        return kp.Watched(0, None, False, 3.0)
    rig.harness = harness
    p = rig.policy
    rig.call()
    kw = seen[0]
    assert kw["backstop_s"] == min(p.call_timeout_s, p.lease_s - p.cleanup_s)
    assert kw["backstop_s"] + kw["term_grace_s"] + hwlease.RELEASE_TIMEOUT_S <= p.lease_s


def test_after_chip_time_is_spent_the_deadline_shrinks_so_the_worst_case_still_fits(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv)
    seen = []

    def harness(argv, kw):
        seen.append(kw["backstop_s"])
        CLOCK["now"] += 200.0                                                          # a long call
        Path(argv[argv.index("--out") + 1]).write_text(
            json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 1.0}))
        return kp.Watched(0, None, False, 200.0)
    rig.harness = harness
    rig.call()
    rig.call()
    p, used = rig.policy, rig.check.summary()["lease_s_used"]
    assert len(seen) == 2 and seen[1] < seen[0]
    assert seen[1] + kp.GOZER_TERM_GRACE_S + hwlease.RELEASE_TIMEOUT_S <= p.lease_s - (used - 200.0) + 1e-6


def test_a_policy_with_a_reserve_smaller_than_the_two_phases_is_still_what_the_default_is_pinned_against():
    p = hwtool.HwPolicy()
    assert p.cleanup_s >= kp.GOZER_TERM_GRACE_S + hwlease.RELEASE_TIMEOUT_S


def test_the_estimated_time_of_a_lost_reply_is_the_time_since_the_request_until_gozers_duration_is_known(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv)
    rig.sim.release_mode = "overrun"                                       # gozer never writes a duration for this release
    original = rig.sim._acquire

    def slow_lost(argv):
        out = original(argv)
        CLOCK["now"] += 10.0
        raise OSError("the reply was lost after gozer granted the lease")
    rig.sim._acquire = slow_lost
    rig.call()
    rec = rig.check.summary()["calls"][0]
    assert rec["lease_s"] == 10.0 and rec["lease_incomplete"] is True and rec["lease_id"] == "L1"
    assert rig.check.summary()["lease_s_used"] == 10.0


def test_what_a_lost_reply_reclaim_could_not_give_back_is_told_to_the_agent_and_recorded(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv)
    rig.sim.release_mode = "refused"
    lost_reply(rig, "request")
    text = rig.call()
    assert "something was left unresolved" in text and "lease L1" in text
    assert rig.check.summary()["calls"][0]["release_error"].startswith("something was left unresolved")


def test_a_recovered_grant_starts_the_chip_clock_once_and_a_normal_grant_is_not_noted_twice(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv)
    calls = []
    original_run = rig.check.run

    def run(k, timeout_s, wait_s, on_grant, ctx):
        def counting():
            calls.append("on_grant")
            on_grant()
        return original_run(k, timeout_s, wait_s, counting, ctx)
    rig.check.run = run
    rig.sim.release_mode = "refused"                                      # the lease is still shown when the reclaim looks

    def harness(argv, kw):
        raise KeyboardInterrupt
    rig.harness = harness
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert calls == ["on_grant"]                                           # the reclaim finds the lease again: no second start
