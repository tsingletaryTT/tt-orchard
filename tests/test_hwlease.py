# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The gozer side of the agent's `hardware_check` tool (orchard/hwlease.py): the probe asks for the lease with
`gozer acquire --owner-pid`, claims its own ticket, releases, and reads back from gozer what it still holds.

The paths from the ticket to the final release and their tests are in the design note
(docs/superpowers/specs/2026-10-27-agent-hardware-check-design.md). Fakes only: no gozer, no chip."""
from __future__ import annotations

import json

import pytest

from orchard import hwlease
from orchard.commands import CommandResult

PID, WHO = 4321, "claude:test"


class FakeCmd:
    """A fake command runner for gozer: scripted answers per subcommand (the last repeats), history and queue as data."""

    def __init__(self):
        self.calls = []
        self.answers = {}
        self.history = []
        self.queue = []
        self.history_fails = False
        self.queue_fails = False
        self.status_fails = False
        self.status_chips = []          # [{"bdf", "pid", "who", "state"}]: what `gozer status --json` shows as held

    def say(self, sub, *answers):
        self.answers[sub] = list(answers)

    def run(self, argv, timeout, env=None, kill_on_timeout=True):
        argv = list(argv)
        self.calls.append({"argv": argv, "timeout": timeout, "kill_on_timeout": kill_on_timeout})
        sub = argv[1]
        if sub == "history":
            return res(1, "", "boom") if self.history_fails else res(0, json.dumps({"history": self.history}))
        if sub == "queue":
            return res(1, "", "boom") if self.queue_fails else res(0, json.dumps({"queue": self.queue}))
        if sub == "status":
            return res(1, "", "boom") if self.status_fails else res(0, json.dumps({"chips": self.status_chips}))
        items = self.answers.get(sub)
        if not items:
            return res(0, json.dumps({"message": "ok"}))
        item = items.pop(0) if len(items) > 1 else items[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def subs(self):
        return [c["argv"][1] for c in self.calls]


def res(code=0, out="", err="", **kw):
    return CommandResult(("gozer",), code, out, err, **kw)


class Clock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def leases(cmd, **kw):
    c = Clock()
    led = hwlease.GozerLeases(who=WHO, pid=PID, run=cmd.run, clock=c.clock, sleep=c.sleep, **kw)
    led.time = c
    return led


GRANT = json.dumps({"granted": True, "lease_id": "L1", "chips": ["0000:01:00.0", "0000:02:00.0"], "owner_pid": PID,
                    "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}})
QUEUED = json.dumps({"queued": True, "granted": False, "ticket": "T1", "position": 1, "ahead": []})


def granted_event(lease, pid=PID, who=WHO, ts="2026-10-27T10:00:00Z"):
    return {"event": "granted", "lease_id": lease, "pid": pid, "who": who, "ts": ts, "chips": ["0000:01:00.0", "0000:02:00.0"]}


# ---- asking ---------------------------------------------------------------------------------------------------------


def test_the_request_is_an_acquire_owned_by_the_probes_pid_for_one_chip_with_json():
    cmd = FakeCmd()
    cmd.say("acquire", res(0, GRANT))
    out = leases(cmd).request("why #1")
    argv = cmd.calls[0]["argv"]
    assert argv == ["gozer", "acquire", "--json", "--chips", "1", "--exact", "1", "--who", WHO, "--reason", "why #1",
                    "--owner-pid", str(PID)]
    assert out.grant == {"lease_id": "L1", "chips": ["0000:01:00.0", "0000:02:00.0"],
                         "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}}
    assert out.ticket is None and out.error is None


def test_a_busy_board_answers_with_the_ticket():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    out = leases(cmd).request("why")
    assert out.grant is None and out.ticket == "T1" and out.error is None


@pytest.mark.parametrize("answer", [res(12, json.dumps({"granted": False, "error": "no"})), res(1, "", "boom"),
                                    res(0, "not json"), res(0, json.dumps({"granted": False})),
                                    res(0, json.dumps({"granted": True})), res(0, json.dumps({"granted": True, "lease_id": "bad id"})),
                                    res(0, json.dumps({"granted": False, "lease_id": "L1"})), res(1, GRANT),
                                    res(0, QUEUED),
                                    res(10, "not json"), res(10, json.dumps({"queued": True, "ticket": "bad id"}))])
def test_anything_that_is_not_a_complete_grant_or_a_ticket_is_an_error(answer):
    cmd = FakeCmd()
    cmd.say("acquire", answer)
    out = leases(cmd).request("why")
    assert out.grant is None and out.ticket is None and out.error


def test_a_gozer_that_cannot_start_is_an_error():
    cmd = FakeCmd()
    cmd.say("acquire", OSError("no gozer"))
    assert "no gozer" in leases(cmd).request("why").error


def test_a_grant_without_a_device_list_takes_it_from_the_chips():
    cmd = FakeCmd()
    cmd.say("acquire", res(0, json.dumps({"granted": True, "lease_id": "L1", "chips": ["0000:01:00.0"]})))
    assert leases(cmd).request("why").grant["env"] == {"TT_VISIBLE_DEVICES": "0000:01:00.0"}


def test_a_grant_with_neither_devices_nor_chips_has_an_empty_environment():
    cmd = FakeCmd()
    cmd.say("acquire", res(0, json.dumps({"granted": True, "lease_id": "L1"})))
    assert leases(cmd).request("why").grant == {"lease_id": "L1", "chips": [], "env": {}}


def test_the_claim_is_the_same_acquire_with_the_ticket_and_the_same_pid():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    out = leases(cmd).claim("T1", "why")
    argv = cmd.calls[0]["argv"]
    assert argv[-2:] == ["--ticket", "T1"] and "--owner-pid" in argv and argv[argv.index("--owner-pid") + 1] == str(PID)
    assert out.ticket == "T1"


def test_a_ticket_that_could_pass_as_an_option_is_never_claimed_cancelled_or_released():
    cmd = FakeCmd()
    led = leases(cmd)
    assert led.claim("--force", "w").error and "not a ticket" in led.cancel("--force")
    assert led.release("--force").held is False and cmd.calls == []


# ---- the wait on the one ticket ------------------------------------------------------------------------------------------


def test_the_wait_claims_the_same_ticket_every_two_seconds_and_never_asks_for_a_new_one():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED), res(10, QUEUED), res(0, GRANT))
    led = leases(cmd)
    grant, error = led.wait("T1", "why", 60.0)
    assert grant["lease_id"] == "L1" and error is None and led.time.sleeps == [2.0, 2.0]
    for c in cmd.calls:
        assert c["argv"][1] == "acquire" and c["argv"][-2:] == ["--ticket", "T1"]
    assert "cancel" not in cmd.subs()


def test_a_wait_that_runs_out_cancels_the_ticket_once_and_gives_none():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    led = leases(cmd)
    assert led.wait("T1", "why", 6.0) == (None, None)
    assert cmd.subs().count("cancel") == 1 and sum(led.time.sleeps) == pytest.approx(6.0)


def test_a_zero_wait_claims_once_and_cancels():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    led = leases(cmd)
    assert led.wait("T1", "why", 0.0) == (None, None) and cmd.subs() == ["acquire", "cancel", "history", "queue", "status"]


def test_the_last_sleep_never_overruns_the_deadline():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    led = leases(cmd)
    led.wait("T1", "why", 3.0)
    assert led.time.sleeps == [2.0, 1.0]


def test_a_claim_that_cannot_be_answered_looks_in_gozers_record_before_giving_up():
    """The claim may have been granted just before its output was lost: the lease is found by the probe's pid."""
    cmd = FakeCmd()
    cmd.say("acquire", OSError("killed"))
    cmd.history = [granted_event("L7")]
    grant, _ = leases(cmd).wait("T1", "why", 60.0)
    assert grant["lease_id"] == "L7" and grant["env"] == {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}
    assert "cancel" not in cmd.subs()                        # not queued any more: nothing to cancel


def test_a_claim_that_says_the_ticket_is_gone_with_no_lease_anywhere_gives_none():
    cmd = FakeCmd()
    cmd.say("acquire", res(12, json.dumps({"granted": False, "queued": False})))
    assert leases(cmd).wait("T1", "why", 60.0) == (None, None)


def test_a_grant_that_lands_while_the_ticket_is_being_cancelled_is_found():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    cmd.say("cancel", res(13, "no such ticket: T1"))
    cmd.history = [granted_event("L7")]
    assert leases(cmd).wait("T1", "why", 0.0)[0]["lease_id"] == "L7"


def test_a_lease_of_another_pid_or_who_is_not_found_by_the_wait():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    cmd.history = [granted_event("L7", pid=PID + 1), granted_event("L8", who="claude:other")]
    assert leases(cmd).wait("T1", "why", 0.0) == (None, None)


def test_a_cancel_that_fails_is_returned_with_the_missing_grant():
    cmd = FakeCmd()
    cmd.say("acquire", res(10, QUEUED))
    cmd.say("cancel", res(1, "", "boom"))
    grant, error = leases(cmd).wait("T1", "why", 0.0)
    assert grant is None and "boom" in error


# ---- giving back ----------------------------------------------------------------------------------------------------


def test_a_clean_release_is_ok():
    cmd = FakeCmd()
    out = leases(cmd).release("L1")
    assert out == hwlease.ReleaseResult(True, False, "") and cmd.calls[0]["argv"] == ["gozer", "release", "L1", "--json"]


def test_the_release_is_not_killed_when_it_overruns():
    cmd = FakeCmd()
    cmd.say("release", res(None, timed_out=True, left_running=True, pid=4242))
    out = leases(cmd).release("L1")
    assert cmd.calls[0]["kill_on_timeout"] is False and cmd.calls[0]["timeout"] == hwlease.RELEASE_TIMEOUT_S
    assert out.ok is False and out.held is True and "still running" in out.text


@pytest.mark.parametrize("message", ["released; NOT marked clean", "released but not resetting: no longer locked"])
def test_a_release_that_exits_0_but_did_not_reset_is_a_failure_that_is_not_held(message):
    cmd = FakeCmd()
    cmd.say("release", res(0, json.dumps({"message": message})))
    out = leases(cmd).release("L1")
    assert out.ok is False and out.held is False and message in out.text


def test_a_release_that_finds_no_lease_is_fine():
    cmd = FakeCmd()
    cmd.say("release", res(13, "", "no such lease"))
    out = leases(cmd).release("L1")
    assert out.ok is True and out.held is False


@pytest.mark.parametrize("code", [15, 1, 12])
def test_a_refused_or_failed_release_is_held(code):
    cmd = FakeCmd()
    cmd.say("release", res(code, "", "device still open"))
    out = leases(cmd).release("L1")
    assert out.ok is False and out.held is True and "device still open" in out.text


def test_a_release_that_could_not_start_is_held():
    cmd = FakeCmd()
    cmd.say("release", OSError("no gozer"))
    out = leases(cmd).release("L1")
    assert out.held is True and "no gozer" in out.text


@pytest.mark.parametrize("code,ok", [(0, True), (13, True)])
def test_a_cancel_that_finds_the_ticket_gone_is_no_error(code, ok):
    cmd = FakeCmd()
    cmd.say("cancel", res(code, "x"))
    assert leases(cmd).cancel("T1") is None


def test_a_cancel_that_fails_returns_the_text():
    cmd = FakeCmd()
    cmd.say("cancel", res(1, "", "boom"))
    assert "boom" in leases(cmd).cancel("T1")
    cmd.say("cancel", OSError("no gozer"))
    assert "no gozer" in leases(cmd).cancel("T1")


# ---- what gozer shows -----------------------------------------------------------------------------------------------


def test_nothing_granted_and_nothing_queued_is_resolved():
    assert leases(FakeCmd()).unresolved() is None


def test_a_lease_granted_to_this_pid_and_not_released_is_unresolved_and_named():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    text = leases(cmd).unresolved()
    assert "L1" in text and "gozer status" in text


@pytest.mark.parametrize("ending", ["released", "reaped"])
def test_a_released_or_reaped_lease_is_resolved(ending):
    cmd = FakeCmd()
    cmd.history = [granted_event("L1"), {"event": ending, "lease_id": "L1"}]
    assert leases(cmd).unresolved() is None


def test_leases_of_other_pids_or_whos_are_not_ours():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1", pid=PID + 1), granted_event("L2", who="claude:other")]
    assert leases(cmd).unresolved() is None


def test_a_ticket_queued_by_this_pid_is_unresolved_and_named():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}, {"ticket": "T2", "pid": PID + 1, "who": WHO}]
    text = leases(cmd).unresolved()
    assert "T1" in text and "T2" not in text


def test_history_or_a_queue_that_cannot_be_read_is_unresolved_and_says_so():
    for attr in ("history_fails", "queue_fails"):
        cmd = FakeCmd()
        setattr(cmd, attr, True)
        assert "cannot confirm" in leases(cmd).unresolved()


def test_the_duration_of_a_lease_is_what_gozer_says_it_was_held():
    cmd = FakeCmd()
    cmd.history = [{"event": "released", "lease_id": "L1", "duration_s": 88.0}, {"event": "released", "lease_id": "L2", "duration_s": 5.0}]
    led = leases(cmd)
    assert led.duration("L1") == 88.0 and led.duration("L3") is None


@pytest.mark.parametrize("value", ["soon", True, -1, None])
def test_a_duration_that_is_not_a_plausible_number_is_none(value):
    cmd = FakeCmd()
    cmd.history = [{"event": "released", "lease_id": "L1", "duration_s": value}]
    assert leases(cmd).duration("L1") is None


def test_the_duration_is_none_when_history_cannot_be_read():
    cmd = FakeCmd()
    cmd.history_fails = True
    assert leases(cmd).duration("L1") is None


# ---- reclaim: the exception path ------------------------------------------------------------------------------------------


def test_reclaim_cancels_the_tickets_and_releases_the_leases_gozer_shows_for_this_pid():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    leases(cmd).reclaim()
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "cancel"] == ["T1"]
    assert [c["argv"][2] for c in cmd.calls if c["argv"][1] == "release"] == ["L1"]


def test_reclaim_says_what_it_could_not_give_back_and_never_raises():
    cmd = FakeCmd()
    cmd.history = [granted_event("L1")]
    cmd.say("release", OSError("no gozer"))
    assert "L1" in leases(cmd).reclaim().left


def test_reclaim_with_nothing_held_is_quiet():
    assert leases(FakeCmd()).reclaim().left is None


def test_a_queue_entry_of_this_pid_but_another_who_is_not_ours():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": "claude:other"}]
    assert leases(cmd).unresolved() is None


def test_reclaim_never_raises_even_when_gozer_cannot_be_read_at_all():
    led = leases(FakeCmd())

    def boom():
        raise RuntimeError("history exploded")
    led.mine = boom
    assert "cannot confirm" in led.reclaim().left


# ---- a claim that fails for a moment must not end the wait or leak the ticket (seen on the real gozer: 2026-10-27) ----------


def test_a_claim_that_errs_while_the_ticket_is_still_queued_is_tried_again():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    cmd.say("acquire", res(1, "", "gozer is busy for a moment"), res(10, QUEUED), res(0, GRANT))
    led = leases(cmd)
    grant, error = led.wait("T1", "why", 60.0)
    assert grant["lease_id"] == "L1" and error is None and led.time.sleeps == [2.0, 2.0]
    assert "cancel" not in cmd.subs()


def test_claims_that_keep_erring_with_the_ticket_queued_end_at_the_deadline_with_the_ticket_cancelled():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T1", "pid": PID, "who": WHO}]
    cmd.say("acquire", res(1, "", "boom"))
    led = leases(cmd)
    assert led.wait("T1", "why", 6.0) == (None, None)
    assert cmd.subs().count("cancel") == 1 and sum(led.time.sleeps) == pytest.approx(6.0)


def test_claims_with_an_unreadable_queue_are_not_taken_to_mean_the_ticket_is_gone():
    cmd = FakeCmd()
    cmd.queue_fails = True
    cmd.say("acquire", res(1, "", "boom"), res(0, GRANT))
    grant, _ = leases(cmd).wait("T1", "why", 60.0)
    assert grant["lease_id"] == "L1"


def test_a_ticket_that_is_gone_from_the_queue_with_no_lease_ends_the_wait_at_once_and_is_cancelled():
    cmd = FakeCmd()
    cmd.say("acquire", res(1, "", "boom"))
    led = leases(cmd)
    assert led.wait("T1", "why", 60.0) == (None, None) and led.time.sleeps == []
    assert cmd.subs().count("cancel") == 1


def test_a_wait_that_ends_for_any_reason_other_than_a_grant_cancels_the_ticket():
    for answer in (res(12, json.dumps({"granted": False, "queued": False})), OSError("killed"), res(0, "not json")):
        cmd = FakeCmd()
        cmd.say("acquire", answer)
        leases(cmd).wait("T1", "why", 60.0)
        assert "cancel" in cmd.subs()


def test_another_ticket_in_the_queue_does_not_keep_this_wait_going():
    cmd = FakeCmd()
    cmd.queue = [{"ticket": "T9", "pid": PID + 1, "who": "claude:other"}]
    cmd.say("acquire", res(1, "", "boom"))
    led = leases(cmd)
    assert led.wait("T1", "why", 60.0) == (None, None) and led.time.sleeps == []
