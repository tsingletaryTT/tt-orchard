# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The lease and ticket bookkeeping behind the agent's `hardware_check` tool (orchard/hwlease.py).

Every path between taking a queue ticket and the final release has a test here or in test_agent_hw_queue.py; the table
of paths is in docs/superpowers/specs/2026-10-27-agent-hardware-check-design.md. Fakes only: no gozer, no chip."""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from orchard import hwlease
from orchard.commands import CommandResult


class FakeCmd:
    """A fake command runner for gozer: scripted answers per subcommand, every call recorded."""

    def __init__(self):
        self.calls = []
        self.answers = {}            # subcommand -> list of CommandResult or Exception (popped in order; the last repeats)
        self.history = []
        self.queue = []
        self.history_fails = False
        self.queue_fails = False

    def say(self, sub, *answers):
        self.answers[sub] = list(answers)

    def run(self, argv, timeout, env=None, kill_on_timeout=True):
        argv = list(argv)
        self.calls.append({"argv": argv, "timeout": timeout, "kill_on_timeout": kill_on_timeout})
        sub = argv[1]
        if sub == "history":
            if self.history_fails:
                return CommandResult(tuple(argv), 1, "", "boom")
            return CommandResult(tuple(argv), 0, json.dumps({"history": self.history}))
        if sub == "queue":
            if self.queue_fails:
                return CommandResult(tuple(argv), 1, "", "boom")
            return CommandResult(tuple(argv), 0, json.dumps({"queue": self.queue}))
        items = self.answers.get(sub)
        if not items:
            return CommandResult(tuple(argv), 0, json.dumps({"message": "ok"}))
        item = items.pop(0) if len(items) > 1 else items[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def subs(self):
        return [c["argv"][1] for c in self.calls]


def res(code=0, out="", err="", **kw):
    return CommandResult(("gozer",), code, out, err, **kw)


def ledger(cmd, **kw):
    return hwlease.LeaseLedger(run=cmd.run, **kw)


GRANT_JSON = json.dumps({"granted": True, "lease_id": "L1", "chips": ["0000:01:00.0", "0000:02:00.0"],
                         "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}})


# ---- release: every way `gozer release` can end ------------------------------------------------------------------


def test_a_clean_release_is_ok_and_leaves_nothing_pending():
    cmd = FakeCmd()
    led = ledger(cmd)
    out = led.release("L1")
    assert out == hwlease.ReleaseResult(True, False, "") and led.pending_leases == {}
    argv = cmd.calls[0]["argv"]
    assert argv == ["gozer", "release", "L1", "--json"]


def test_the_release_is_not_killed_when_it_overruns_because_its_reset_must_not_be_cut_short():
    cmd = FakeCmd()
    cmd.say("release", res(None, timed_out=True, left_running=True, pid=4242))
    led = ledger(cmd)
    out = led.release("L1")
    assert cmd.calls[0]["kill_on_timeout"] is False and cmd.calls[0]["timeout"] == hwlease.RELEASE_TIMEOUT_S
    assert out.ok is False and out.held is True and "still running" in out.text
    assert led.pending_leases["L1"]["pid"] == 4242


@pytest.mark.parametrize("message", ["released; NOT marked clean", "released but not resetting: no longer locked"])
def test_a_release_that_exits_0_but_did_not_reset_is_a_failure_that_is_not_held(message):
    cmd = FakeCmd()
    cmd.say("release", res(0, json.dumps({"message": message})))
    led = ledger(cmd)
    out = led.release("L1")
    assert out.ok is False and out.held is False and message in out.text and led.pending_leases == {}


def test_a_release_that_finds_no_lease_is_fine_because_it_is_already_gone():
    cmd = FakeCmd()
    cmd.say("release", res(13, "", "no such lease"))
    out = ledger(cmd).release("L1")
    assert out.ok is True and out.held is False


@pytest.mark.parametrize("code", [15, 1, 12])
def test_a_refused_or_failed_release_leaves_the_lease_pending_and_held(code):
    cmd = FakeCmd()
    cmd.say("release", res(code, "", "device still open"))
    led = ledger(cmd)
    out = led.release("L1")
    assert out.ok is False and out.held is True and "device still open" in out.text
    assert "L1" in led.pending_leases


def test_a_release_that_could_not_start_leaves_the_lease_pending():
    cmd = FakeCmd()
    cmd.say("release", OSError("no gozer"))
    led = ledger(cmd)
    out = led.release("L1")
    assert out.held is True and "no gozer" in out.text and "L1" in led.pending_leases


def test_the_lease_is_registered_before_the_release_runs_so_an_interrupt_cannot_lose_it():
    cmd = FakeCmd()
    cmd.say("release", KeyboardInterrupt())
    led = ledger(cmd)
    with pytest.raises(KeyboardInterrupt):
        led.release("L1")
    assert "L1" in led.pending_leases


def test_a_lease_id_that_could_pass_as_an_option_is_never_handed_to_gozer():
    cmd = FakeCmd()
    out = ledger(cmd).release("--force")
    assert cmd.calls == [] and out.held is False and "lease id" in out.text


# ---- cancel ---------------------------------------------------------------------------------------------------------


def test_a_cancelled_ticket_is_no_error_and_nothing_pending():
    cmd = FakeCmd()
    led = ledger(cmd)
    led.add_ticket("T1")
    assert led.cancel("T1") is None and led.pending_tickets == {}
    assert cmd.calls[0]["argv"] == ["gozer", "cancel", "T1"]


def test_a_ticket_that_is_already_gone_is_no_error():
    cmd = FakeCmd()
    cmd.say("cancel", res(13, "no such ticket: T1"))
    led = ledger(cmd)
    led.add_ticket("T1")
    assert led.cancel("T1") is None and led.pending_tickets == {}


def test_a_cancel_that_fails_returns_the_text_and_keeps_the_ticket_pending():
    cmd = FakeCmd()
    cmd.say("cancel", res(1, "", "boom"))
    led = ledger(cmd)                                        # not registered before: a failure registers it
    assert "boom" in led.cancel("T1") and "T1" in led.pending_tickets


def test_a_cancel_that_cannot_run_keeps_the_ticket_pending():
    cmd = FakeCmd()
    cmd.say("cancel", OSError("no gozer"))
    led = ledger(cmd)
    assert "no gozer" in led.cancel("T1") and "T1" in led.pending_tickets


def test_a_ticket_that_could_pass_as_an_option_is_never_handed_to_gozer():
    cmd = FakeCmd()
    assert "not a ticket" in ledger(cmd).cancel("--force") and cmd.calls == []


# ---- unresolved: what an earlier call left behind -----------------------------------------------------------------


def test_nothing_pending_is_resolved_and_asks_gozer_nothing():
    cmd = FakeCmd()
    assert ledger(cmd).unresolved() is None and cmd.calls == []


def test_a_pending_lease_that_history_shows_released_is_resolved_and_dropped():
    cmd = FakeCmd()
    cmd.history = [{"event": "granted", "lease_id": "L1"}, {"event": "released", "lease_id": "L1"}]
    led = ledger(cmd)
    led.add_lease("L1", why="x")
    assert led.unresolved() is None and led.pending_leases == {}


def test_a_pending_lease_that_history_shows_reaped_is_resolved_too():
    cmd = FakeCmd()
    cmd.history = [{"event": "reaped", "lease_id": "L1"}]
    led = ledger(cmd)
    led.add_lease("L1", why="x")
    assert led.unresolved() is None


def test_a_pending_lease_with_no_release_event_is_unresolved_and_named():
    cmd = FakeCmd()
    cmd.history = [{"event": "granted", "lease_id": "L1"}]
    led = ledger(cmd)
    led.add_lease("L1", why="x")
    text = led.unresolved()
    assert "L1" in text and "gozer status" in text and "L1" in led.pending_leases


def test_a_release_that_is_still_running_is_said_to_be_running():
    cmd = FakeCmd()
    led = ledger(cmd, alive=lambda pid: pid == 4242)
    led.add_lease("L1", why="x", pid=4242)
    assert "still running" in led.unresolved()


def test_a_release_whose_process_is_gone_with_no_release_event_is_unresolved():
    cmd = FakeCmd()
    led = ledger(cmd, alive=lambda pid: False)
    led.add_lease("L1", why="x", pid=4242)
    text = led.unresolved()
    assert "L1" in text and "still running" not in text


def test_history_that_cannot_be_read_leaves_the_lease_unresolved_and_says_so():
    cmd = FakeCmd()
    cmd.history_fails = True
    led = ledger(cmd)
    led.add_lease("L1", why="x")
    assert "cannot confirm" in led.unresolved()


def test_a_pending_ticket_that_is_no_longer_queued_is_resolved():
    cmd = FakeCmd()
    led = ledger(cmd)
    led.add_ticket("T1")
    assert led.unresolved() is None and led.pending_tickets == {}


def test_a_pending_ticket_that_is_still_queued_is_unresolved_and_named():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1"}]
    led = ledger(cmd)
    led.add_ticket("T1")
    assert "T1" in led.unresolved() and "T1" in led.pending_tickets


def test_a_queue_that_cannot_be_read_leaves_the_ticket_unresolved():
    cmd = FakeCmd()
    cmd.queue_fails = True
    led = ledger(cmd)
    led.add_ticket("T1")
    assert "cannot confirm" in led.unresolved()


# ---- orphan grants --------------------------------------------------------------------------------------------------


def granted(lease, reason, chips=("0000:01:00.0", "0000:02:00.0")):
    return {"event": "granted", "lease_id": lease, "reason": reason, "chips": list(chips), "who": "claude:x"}


def test_a_grant_whose_reason_carries_the_token_and_was_never_released_is_found():
    cmd = FakeCmd()
    cmd.history = [granted("L9", "kernel-probe add attempt 1 agent check 1 #abcd1234")]
    grant = ledger(cmd).orphan_grant("abcd1234")
    assert grant == {"lease_id": "L9", "chips": ["0000:01:00.0", "0000:02:00.0"],
                     "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}}


def test_a_grant_that_was_released_or_reaped_is_not_an_orphan():
    for ending in ("released", "reaped"):
        cmd = FakeCmd()
        cmd.history = [granted("L9", "x #abcd1234"), {"event": ending, "lease_id": "L9"}]
        assert ledger(cmd).orphan_grant("abcd1234") is None


def test_a_grant_with_another_token_is_not_ours():
    cmd = FakeCmd()
    cmd.history = [granted("L9", "x #other999")]
    assert ledger(cmd).orphan_grant("abcd1234") is None


def test_a_token_that_is_too_short_matches_nothing():
    cmd = FakeCmd()
    cmd.history = [granted("L9", "x #a")]
    assert ledger(cmd).orphan_grant("a") is None


def test_history_that_cannot_be_read_finds_no_orphan():
    cmd = FakeCmd()
    cmd.history_fails = True
    assert ledger(cmd).orphan_grant("abcd1234") is None


# ---- the waiter ---------------------------------------------------------------------------------------------------


class FakeProc:
    def __init__(self, out="", code=0, hang_first=False, interrupt=False, pid=777):
        self.out, self.returncode, self.hang_first, self.interrupt = out, code, hang_first, interrupt
        self.killed, self.timeouts, self.pid = False, [], pid

    def communicate(self, timeout=None):
        self.timeouts.append(timeout)
        if self.interrupt and not self.killed:
            raise KeyboardInterrupt
        if self.hang_first and not self.killed:
            raise subprocess.TimeoutExpired("gozer wait", timeout)
        return self.out, ""

    def kill(self):
        self.killed = True


def waiter(cmd, proc, **kw):
    seen = {}

    def popen(argv, **k):
        seen["argv"] = list(argv)
        if isinstance(proc, BaseException):
            raise proc
        return proc
    return hwlease.LeaseLedger(run=cmd.run, popen=popen, **kw), seen


def test_the_waiter_asks_for_the_ticket_in_seconds_as_json():
    """Bare digits mean minutes to gozer: a 12 s wait must be written `12s`."""
    cmd = FakeCmd()
    led, seen = waiter(cmd, FakeProc(GRANT_JSON))
    grant = led.wait_for_grant("T1", 12.0, "tok12345")
    assert seen["argv"] == ["gozer", "wait", "--json", "--timeout", "12s", "T1"]
    assert grant["lease_id"] == "L1" and grant["env"]["TT_VISIBLE_DEVICES"] == "0000:01:00.0,0000:02:00.0"
    assert "L1" in led.pending_leases                       # registered the moment it is granted
    assert "T1" not in led.pending_tickets


def test_the_waiter_rounds_a_fractional_wait_up_and_never_asks_for_zero():
    led, seen = waiter(FakeCmd(), FakeProc(GRANT_JSON))
    led.wait_for_grant("T1", 0.4, "tok12345")
    assert seen["argv"][4] == "1s"
    led, seen = waiter(FakeCmd(), FakeProc(GRANT_JSON))
    led.wait_for_grant("T1", 12.2, "tok12345")
    assert seen["argv"][4] == "13s"


def test_the_waiter_is_stopped_by_the_probe_only_after_gozers_own_timeout_plus_a_slack():
    proc = FakeProc(GRANT_JSON)
    led, _ = waiter(FakeCmd(), proc)
    led.wait_for_grant("T1", 20.0, "tok12345")
    assert proc.timeouts == [20.0 + hwlease.WAIT_SLACK_S]


def test_a_wait_that_gozer_ended_by_its_own_timeout_is_not_a_grant_and_nothing_is_killed():
    proc = FakeProc(json.dumps({"granted": False, "ticket": "T1", "position": 1}), code=14)
    led, _ = waiter(FakeCmd(), proc)
    assert led.wait_for_grant("T1", 10.0, "tok12345") is None and proc.killed is False


@pytest.mark.parametrize("out,code", [("not json", 0), (json.dumps({"granted": False}), 0), (json.dumps({"granted": True}), 0),
                                      (GRANT_JSON, 1), (json.dumps({"granted": True, "lease_id": "bad id"}), 0),
                                      ("", 0)])
def test_an_answer_that_is_not_a_complete_grant_is_not_a_grant(out, code):
    led, _ = waiter(FakeCmd(), FakeProc(out, code=code))
    assert led.wait_for_grant("T1", 10.0, "tok12345") is None and led.pending_leases == {}


def test_a_grant_without_a_device_list_is_still_returned_and_registered_so_that_it_gets_released():
    led, _ = waiter(FakeCmd(), FakeProc(json.dumps({"granted": True, "lease_id": "L1"})))
    grant = led.wait_for_grant("T1", 10.0, "tok12345")
    assert grant == {"lease_id": "L1", "chips": [], "env": {}} and "L1" in led.pending_leases


def test_a_grant_that_arrives_while_the_waiter_is_being_killed_at_the_deadline_is_kept():
    """communicate() after the kill can return the grant JSON that gozer printed just before it was killed."""
    proc = FakeProc(GRANT_JSON, hang_first=True)
    led, _ = waiter(FakeCmd(), proc)
    grant = led.wait_for_grant("T1", 10.0, "tok12345")
    assert proc.killed is True and grant["lease_id"] == "L1" and "L1" in led.pending_leases


def test_a_grant_made_before_its_output_was_printed_is_found_in_history_after_the_kill():
    cmd = FakeCmd()
    cmd.history = [granted("L7", "agent check #tok12345")]
    proc = FakeProc("", hang_first=True)
    led, _ = waiter(cmd, proc)
    grant = led.wait_for_grant("T1", 10.0, "tok12345")
    assert grant["lease_id"] == "L7" and "L7" in led.pending_leases


def test_a_killed_waiter_with_no_grant_anywhere_gives_none():
    cmd = FakeCmd()
    led, _ = waiter(cmd, FakeProc("", hang_first=True))
    assert led.wait_for_grant("T1", 10.0, "tok12345") is None and led.pending_leases == {}


def test_an_interrupt_during_the_wait_kills_the_waiter_and_cancels_the_ticket():
    cmd = FakeCmd()
    proc = FakeProc("", interrupt=True)
    led, _ = waiter(cmd, proc)
    led.add_ticket("T1")
    with pytest.raises(KeyboardInterrupt):
        led.wait_for_grant("T1", 10.0, "tok12345")
    assert proc.killed is True and "cancel" in cmd.subs() and "release" not in cmd.subs()


def test_an_interrupt_during_the_wait_releases_a_grant_that_arrived_meanwhile():
    cmd = FakeCmd()
    cmd.history = [granted("L7", "agent check #tok12345")]
    proc = FakeProc("", interrupt=True)
    led, _ = waiter(cmd, proc)
    led.add_ticket("T1")
    with pytest.raises(KeyboardInterrupt):
        led.wait_for_grant("T1", 10.0, "tok12345")
    release = [c["argv"] for c in cmd.calls if c["argv"][1] == "release"]
    assert release == [["gozer", "release", "L7", "--json"]]


def test_an_interrupt_with_a_grant_in_the_drained_output_releases_it():
    cmd = FakeCmd()

    class Late(FakeProc):
        def communicate(self, timeout=None):
            if not self.killed:
                raise KeyboardInterrupt
            return GRANT_JSON, ""
    led, _ = waiter(cmd, Late())
    with pytest.raises(KeyboardInterrupt):
        led.wait_for_grant("T1", 10.0, "tok12345")
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "release"] == ["L1"]


def test_a_waiter_that_cannot_start_gives_none_and_leaves_the_ticket_to_the_caller():
    led, _ = waiter(FakeCmd(), OSError("no gozer"))
    assert led.wait_for_grant("T1", 10.0, "tok12345") is None


def test_a_ticket_that_could_pass_as_an_option_is_never_waited_on():
    led, seen = waiter(FakeCmd(), FakeProc(GRANT_JSON))
    assert led.wait_for_grant("--force", 10.0, "tok12345") is None and "argv" not in seen


def test_the_waiter_is_also_killed_when_the_drain_itself_is_interrupted():
    cmd = FakeCmd()

    class Stubborn(FakeProc):
        def communicate(self, timeout=None):
            if self.killed:
                raise KeyboardInterrupt
            raise subprocess.TimeoutExpired("gozer wait", timeout)
    led, _ = waiter(cmd, Stubborn())
    with pytest.raises(KeyboardInterrupt):
        led.wait_for_grant("T1", 10.0, "tok12345")


def test_the_release_budget_is_inside_the_policys_cleanup_reserve():
    from orchard import hwtool, kernelprobe as kp
    assert hwtool.HwPolicy().cleanup_s >= kp.GOZER_TERM_GRACE_S + hwlease.RELEASE_TIMEOUT_S
    assert os.getpid() > 0


def test_a_zero_wait_still_asks_for_one_second_because_gozer_reads_zero_as_no_time():
    led, seen = waiter(FakeCmd(), FakeProc(GRANT_JSON))
    led.wait_for_grant("T1", 0.0, "tok12345")
    assert seen["argv"][4] == "1s"


def test_an_interrupt_between_the_parse_and_the_registration_still_releases_the_grant():
    cmd = FakeCmd()
    led, _ = waiter(cmd, FakeProc(GRANT_JSON))
    real = led.add_lease
    calls = []

    def add_lease(lease_id, **kw):
        calls.append(lease_id)
        if len(calls) == 1:
            raise KeyboardInterrupt                           # before the lease was recorded
        return real(lease_id, **kw)
    led.add_lease = add_lease
    with pytest.raises(KeyboardInterrupt):
        led.wait_for_grant("T1", 10.0, "tok12345")
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "release"] == ["L1"]
