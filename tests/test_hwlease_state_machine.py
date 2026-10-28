# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The cleanup of the agent hardware tool as a state machine (design note, "Round 8"): every state runs against every event on a fake
gozer that keeps ground truth, and the invariant is checked on that truth, not on what the probe read:

    after a call's cleanup, either gozer shows nothing of ours, or the probe is UNRESOLVED and every new call is refused.

States: N (the request is running), T (a ticket is held), L (a lease is held), R (the release command is running), F (a release is in
flight), U (unresolved). Events: grant (a reply lost after gozer granted), interrupt, second interrupt (during the cleanup), timeout,
read failure, release done. No real gozer, no chip."""
from __future__ import annotations

import json

import pytest

from orchard import hwlease, hwtool, kernelprobe as kp
from test_agent_hw import CLOCK, GOOD_HW_SET, fake_kenv, skills_current, ttlang_skill  # noqa: F401
from test_agent_hw_paths import CHIPS, PID, WHO, Rig, SimGozer, busy_rig, res


class Once:
    """A one-shot injection: the first use raises KeyboardInterrupt, later uses do nothing (the world calms down for the next call)."""

    def __init__(self):
        self.used = False

    def hit(self):
        if not self.used:
            self.used = True
            raise KeyboardInterrupt


def run_call(rig):
    """Run one call; return ("returned", text) or ("raised", the exception)."""
    try:
        return "returned", rig.call()
    except BaseException as exc:                      # KeyboardInterrupt is the point of most scenarios
        return "raised", exc


def check_invariant(rig):
    """The invariant, on the fake's ground truth. A new call must be refused unless gozer really holds nothing of ours."""
    clean = rig.sim.truth_clean()
    requests_before = len(rig.sim.requests())
    outcome, text = run_call(rig)
    assert outcome == "returned", f"the next call raised {text!r}"
    if not clean:
        assert text.startswith("hardware_check refused"), f"gozer still holds {rig.sim.truth_held()} {rig.sim.truth_tickets()} but a call ran: {text[:80]}"
        assert len(rig.sim.requests()) == requests_before                      # and nothing was requested
    return clean


def settle_world(rig):
    """Let gozer finish what it is doing and let every read work again, then look: a clean world must allow a call."""
    for lease, at in list(rig.sim.pending_finish.items()):
        rig.sim.finish_release(lease, seconds=60.0)
    rig.sim.pending_finish.clear()
    for lease in rig.sim.truth_held():                  # gozer finishes every release it was running, and expires every ticket
        rig.sim.finish_release(lease, seconds=60.0)
    rig.sim.queue.clear()
    rig.sim.history_fails = rig.sim.queue_fails = rig.sim.status_fails = False
    rig.sim.history_window = None
    rig.sim.release_mode = "ok"


# ---- scenario builders: each sets a state up, injects the event, and returns the rig --------------------------------------------


def interrupt_in_sleep(rig):
    once = Once()
    original = rig.leases.sleep

    def sleep(s):
        once.hit()
        original(s)
    rig.leases.sleep = sleep


def second_interrupt_once(sim, name):
    """Make the first call of a gozer command raise KeyboardInterrupt: the second interrupt, landing during the cleanup."""
    original = getattr(sim, name)
    state = {"done": False}

    def wrapped(argv):
        if not state["done"]:
            state["done"] = True
            raise KeyboardInterrupt
        return original(argv)
    setattr(sim, name, wrapped)


def grant_then_lose_reply(sim, when):
    original = sim._acquire

    def lost(argv):
        out = original(argv)
        if when == "request" and "--ticket" not in argv and out.returncode == 0:
            raise OSError("the reply was lost")
        if when == "claim" and "--ticket" in argv and out.returncode == 0:
            raise OSError("the reply was lost")
        return out
    sim._acquire = lost


def N_grant(tmp, kenv):
    rig = Rig(tmp, kenv)
    grant_then_lose_reply(rig.sim, "request")
    return rig


def N_interrupt_after_grant(tmp, kenv):
    rig = Rig(tmp, kenv)
    original = rig.sim._acquire
    once = Once()

    def granted_then_interrupt(argv):
        out = original(argv)
        once.hit()
        return out
    rig.sim._acquire = granted_then_interrupt
    return rig


def N_interrupt_before_grant(tmp, kenv):
    rig = Rig(tmp, kenv)
    rig.sim.acquire_fail_at = {1: KeyboardInterrupt()}               # the table is keyed by call number: later calls are normal
    return rig


def N_second_interrupt(tmp, kenv):
    rig = N_interrupt_after_grant(tmp, kenv)
    second_interrupt_once(rig.sim, "_release")
    rig.sim.finish_after = None                                   # the interrupted release never finishes
    return rig


def N_timeout(tmp, kenv):
    rig = Rig(tmp, kenv)
    original = rig.sim._acquire

    def granted_then_timeout(argv):
        original(argv)
        return res(None, timed_out=True)
    rig.sim._acquire = granted_then_timeout
    return rig


def N_read_failure_unidentified(tmp, kenv):
    rig = N_grant(tmp, kenv)
    rig.sim.history_fails = True                                  # the lease is held but its id cannot be read
    return rig


def N_read_failure_all(tmp, kenv):
    rig = N_grant(tmp, kenv)
    rig.sim.history_fails = rig.sim.queue_fails = rig.sim.status_fails = True
    return rig


def T_grant(tmp, kenv):
    rig = busy_rig(tmp, kenv, busy_calls=1)
    grant_then_lose_reply(rig.sim, "claim")
    return rig


def T_interrupt(tmp, kenv):
    rig = busy_rig(tmp, kenv, busy_calls=10**6)
    interrupt_in_sleep(rig)
    return rig


def T_second_interrupt(tmp, kenv):
    rig = T_interrupt(tmp, kenv)
    second_interrupt_once(rig.sim, "_cancel")
    return rig


def T_timeout(tmp, kenv):
    return busy_rig(tmp, kenv, busy_calls=10**6, policy=hwtool.HwPolicy(wait_s=4.0))


def T_read_failure_queue(tmp, kenv):
    rig = T_interrupt(tmp, kenv)
    original = rig.sim._acquire

    def queue_breaks_after_request(argv):
        out = original(argv)
        rig.sim.queue_fails = True
        return out
    rig.sim._acquire = queue_breaks_after_request
    return rig


def T_cancel_fails(tmp, kenv):
    rig = busy_rig(tmp, kenv, busy_calls=10**6, policy=hwtool.HwPolicy(wait_s=0.0))
    rig.sim.cancel_mode = "fail"
    return rig


def L_interrupt(tmp, kenv):
    rig = Rig(tmp, kenv)
    once = Once()

    def harness(argv, kw):
        once.hit()
        return Rig._watched_ok(rig, argv, kw)
    rig.harness = harness
    return rig


def L_second_interrupt(tmp, kenv):
    rig = L_interrupt(tmp, kenv)
    rig.sim.release_mode = "interrupt_running"
    rig.sim.finish_after = None
    return rig


def L_timeout(tmp, kenv):
    rig = Rig(tmp, kenv)
    once = Once()

    def harness(argv, kw):
        if not once.used:
            once.used = True
            return kp.Watched(-15, "stalled: no output and no CPU use for 60 s", True, 66.0)
        return Rig._watched_ok(rig, argv, kw)
    rig.harness = harness
    return rig


def L_read_failure(tmp, kenv):
    rig = Rig(tmp, kenv)
    once = Once()

    def harness(argv, kw):
        if not once.used:
            rig.sim.history_fails = True                          # history breaks while the harness runs
            once.hit()
        return Rig._watched_ok(rig, argv, kw)
    rig.harness = harness
    return rig


def L_release_done(tmp, kenv):
    return Rig(tmp, kenv)


def L_history_window(tmp, kenv):
    """The grant falls outside the 200 events history shows by default: status shows the chips held, the wider read finds the id."""
    rig = Rig(tmp, kenv)
    once = Once()

    def harness(argv, kw):
        if not once.used:
            rig.sim.history.extend({"event": "queued", "ticket": f"X{i}"} for i in range(300))
            once.hit()
        return Rig._watched_ok(rig, argv, kw)
    rig.harness = harness
    return rig


def L_history_window_hard_cap(tmp, kenv):
    """No read can show the grant at all: held chips with no lease id must fail closed."""
    rig = L_history_window(tmp, kenv)
    rig.sim.history_window = 100
    rig.sim.release_mode = "refused"                              # the first release (known id) is refused too
    return rig


def R_interrupt(tmp, kenv):
    rig = Rig(tmp, kenv)
    rig.sim.release_mode = "interrupt_running"
    rig.sim.finish_after = 20.0
    return rig


def R_second_interrupt(tmp, kenv):
    rig = R_interrupt(tmp, kenv)
    state = {"n": 0}
    original = rig.leases.sleep

    def sleep(s):
        state["n"] += 1
        if state["n"] == 1:
            raise KeyboardInterrupt                               # lands while the reclaim waits for the release
        original(s)
    rig.leases.sleep = sleep
    return rig


def R_overrun(tmp, kenv):
    rig = Rig(tmp, kenv)
    rig.sim.release_mode = "overrun"
    return rig


def R_read_failure_while_waiting(tmp, kenv):
    rig = R_interrupt(tmp, kenv)
    original = rig.sim._release

    def release(argv):
        try:
            return original(argv)
        finally:
            rig.sim.history_fails = True
    rig.sim._release = release
    return rig


def R_release_done(tmp, kenv):
    rig = R_interrupt(tmp, kenv)
    rig.sim.finish_after = 5.0
    return rig


def F_start(tmp, kenv):
    """A release in flight from an earlier call that never finishes: the next call must be refused."""
    rig = Rig(tmp, kenv)
    rig.sim.release_mode = "interrupt_running"
    rig.sim.finish_after = None
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    rig.sim.release_mode = "ok"
    return rig


def U_refused_start(tmp, kenv):
    rig = Rig(tmp, kenv)
    rig.sim.release_mode = "refused"
    rig.call()
    rig.sim.release_mode = "ok"
    return rig


SCENARIOS = [
    ("N-grant", N_grant), ("N-interrupt-after-grant", N_interrupt_after_grant), ("N-interrupt-before-grant", N_interrupt_before_grant),
    ("N-second-interrupt", N_second_interrupt), ("N-timeout", N_timeout), ("N-read-failure-unidentified", N_read_failure_unidentified),
    ("N-read-failure-all", N_read_failure_all),
    ("T-grant", T_grant), ("T-interrupt", T_interrupt), ("T-second-interrupt", T_second_interrupt), ("T-timeout", T_timeout),
    ("T-read-failure-queue", T_read_failure_queue), ("T-cancel-fails", T_cancel_fails),
    ("L-interrupt", L_interrupt), ("L-second-interrupt", L_second_interrupt), ("L-timeout", L_timeout),
    ("L-read-failure", L_read_failure), ("L-release-done", L_release_done), ("L-history-window", L_history_window),
    ("L-history-window-hard-cap", L_history_window_hard_cap),
    ("R-interrupt", R_interrupt), ("R-second-interrupt", R_second_interrupt), ("R-overrun", R_overrun),
    ("R-read-failure-while-waiting", R_read_failure_while_waiting), ("R-release-done", R_release_done),
]


TOOL_OFF = {"L-timeout"}


@pytest.mark.parametrize("name,build", SCENARIOS, ids=[s[0] for s in SCENARIOS])
def test_every_state_and_event_ends_clean_or_unresolved_and_refusing(name, build, tmp_path, fake_kenv):
    rig = build(tmp_path, fake_kenv)
    outcome, result = run_call(rig)
    if outcome == "raised":
        assert isinstance(result, KeyboardInterrupt), f"{name}: an unexpected exception {result!r}"
    clean = check_invariant(rig)
    if not clean:
        # Unresolved: it stays so until gozer shows nothing, and then the next call may run.
        settle_world(rig)
        assert rig.sim.truth_clean()
    settle_world(rig)
    assert rig.leases.unresolved() is None, f"{name}: gozer shows nothing of ours but the probe still says {rig.leases.unresolved()}"
    outcome, text = run_call(rig)
    assert outcome == "returned"
    if name in TOOL_OFF:
        assert text.startswith("hardware_check refused")        # a hang turns the tool off for the rest of the attempt, by policy
    else:
        assert not text.startswith("hardware_check refused"), f"{name}: the world is clean again but a call is refused: {text[:120]}"


@pytest.mark.parametrize("name,build", [("F-in-flight-start", F_start), ("U-refused-start", U_refused_start)],
                         ids=["F-in-flight-start", "U-refused-start"])
def test_an_unresolved_start_refuses_every_event_until_gozer_shows_nothing(name, build, tmp_path, fake_kenv):
    rig = build(tmp_path, fake_kenv)
    assert rig.call().startswith("hardware_check refused")                        # the state itself
    rig.sim.history_fails = True                                                  # read failure: still refused, and says so
    assert "cannot confirm" in rig.call()
    rig.sim.history_fails = False
    rig.sim.finish_release(rig.sim.truth_held()[0], seconds=60.0)                 # release done
    assert rig.call().splitlines()[0].startswith("hardware_check 2 of 6: PASS") or "PASS" in rig.call()
