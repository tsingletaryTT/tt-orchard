# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Every path between taking a queue ticket and the final release, through the probe's own wiring (kernelprobe.make_hardware_check
with hwlease.LeaseLedger). The table of paths, with the test of each, is in
docs/superpowers/specs/2026-10-27-agent-hardware-check-design.md. Fakes only: no gozer, no chip."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from orchard import hwlease, hwtool, kernelprobe as kp
from test_agent_hw import CLOCK, FACTS, GOOD_HW_SET, KERNELS, attempt_dir, fake_kenv, skills_current, ttlang_skill  # noqa: F401
from test_hwlease import FakeCmd, FakeProc, GRANT_JSON, granted, res


class Rig:
    """One attempt with a busy board: `gozer run` queues ticket T1, the waiter and every other gozer command are fakes."""

    def __init__(self, tmp_path, fake_kenv, *, proc=None, policy=None):
        self.adir = attempt_dir(tmp_path, fake_kenv)
        self.cmd = FakeCmd()
        self.events = []
        self.argvs = []
        self.proc = proc if proc is not None else FakeProc(GRANT_JSON)
        self.harness = None                       # an optional callable(argv, kw) replacing the harness run
        self.ledger = hwlease.LeaseLedger(run=self._run, popen=self._popen, alive=lambda pid: False)
        CLOCK["now"] = 0.0
        self.check = kp.make_hardware_check(self.adir, op="add", n=1, shared_python="/shared/python",
                                            policy=policy or hwtool.HwPolicy(), run_watched=self._watched,
                                            ledger=self.ledger, clock=lambda: CLOCK["now"],
                                            sleep=lambda s: CLOCK.__setitem__("now", CLOCK["now"] + s))

    def _run(self, argv, timeout, env=None, kill_on_timeout=True):
        self.events.append(argv[1])
        return self.cmd.run(argv, timeout, env=env, kill_on_timeout=kill_on_timeout)

    def _popen(self, argv, **kw):
        self.events.append("wait")
        self.argvs.append(list(argv))
        if isinstance(self.proc, BaseException):
            raise self.proc
        return self.proc

    def _watched(self, argv, **kw):
        if argv[0] == "gozer":
            self.events.append("gozer-run")
            with open(kw["log_path"], "a") as f:
                f.write("queued: ticket T1\n")
            return kp.Watched(10, None, False, 0.4)
        self.events.append("harness")
        if self.harness is not None:
            return self.harness(argv, kw)
        Path(argv[argv.index("--out") + 1]).write_text(
            json.dumps({"phase": "hw", "sets": [GOOD_HW_SET], "error": None, "device_s": 1.0}))
        return kp.Watched(0, None, False, 3.0)

    def call(self):
        return self.check.call({})

    def released(self):
        return [c["argv"][2] for c in self.cmd.calls if c["argv"][1] == "release"]

    def cancelled(self):
        return [c["argv"][2] for c in self.cmd.calls if c["argv"][1] == "cancel"]


@pytest.fixture
def rig(tmp_path, fake_kenv):
    return Rig(tmp_path, fake_kenv)


# P1 ---------------------------------------------------------------------------------------------------------------


def test_p1_a_grant_runs_the_harness_under_the_lease_and_releases_it(rig):
    text = rig.call()
    assert rig.events == ["gozer-run", "wait", "harness", "release"], rig.events
    assert text.splitlines()[0].startswith("hardware_check 1 of 6: PASS") and rig.released() == ["L1"]
    assert rig.ledger.unresolved() is None and rig.cancelled() == []
    assert rig.argvs[0][3:5] == ["--timeout", "120s"] or "--timeout" in rig.argvs[0]


# P2 ---------------------------------------------------------------------------------------------------------------


def test_p2_gozers_own_wait_timeout_cancels_the_ticket_once_and_runs_nothing(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv, proc=FakeProc(json.dumps({"granted": False, "position": 1}), code=14))
    text = rig.call()
    assert rig.events == ["gozer-run", "wait", "cancel"] and rig.released() == [] and "BUSY" in text
    assert rig.ledger.pending_tickets == {}


# P3 ---------------------------------------------------------------------------------------------------------------


def test_p3_a_ticket_that_cannot_be_cancelled_is_named_and_blocks_the_next_call(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv, proc=FakeProc("", code=14))
    rig.cmd.say("cancel", res(1, "", "boom"))
    rig.cmd.queue = [{"ticket": "T1"}]
    text = rig.call()
    assert "could not be cancelled" in text and "T1" in text and "boom" in text
    runs_before = rig.events.count("gozer-run")
    second = rig.call()
    assert second.startswith("hardware_check refused") and "T1" in second and "still queued" in second
    assert rig.events.count("gozer-run") == runs_before                      # nothing was started
    assert rig.check.summary()["unresolved"].startswith("ticket T1")


def test_p3b_the_next_call_may_start_once_the_ticket_has_left_the_queue(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv, proc=FakeProc("", code=14))
    rig.cmd.say("cancel", res(1, "", "boom"))
    rig.cmd.queue = [{"ticket": "T1"}]
    rig.call()
    rig.cmd.queue = []                                                       # gozer expired it
    rig.proc = FakeProc(GRANT_JSON)
    assert rig.call().splitlines()[0].startswith("hardware_check 2 of 6: PASS")


# P4 ---------------------------------------------------------------------------------------------------------------


def test_p4_a_grant_in_the_output_drained_after_the_deadline_kill_is_used_and_released(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv, proc=FakeProc(GRANT_JSON, hang_first=True))
    text = rig.call()
    assert rig.proc.killed is True and "PASS" in text and rig.released() == ["L1"] and rig.cancelled() == []


# P5 ---------------------------------------------------------------------------------------------------------------


def test_p5_a_grant_found_only_in_gozers_history_is_used_and_released(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv, proc=FakeProc("", hang_first=True))
    token_reason = []

    def watched(argv, **kw):                                                  # read the token the probe put in the reason
        token_reason.append(argv[argv.index("--reason") + 1] if "--reason" in argv else "")
        return Rig._watched(rig, argv, **kw)
    rig.check = kp.make_hardware_check(rig.adir, op="add", n=1, shared_python="/shared/python", policy=hwtool.HwPolicy(),
                                       run_watched=watched, ledger=rig.ledger, clock=lambda: CLOCK["now"],
                                       sleep=lambda s: None)
    original = rig.cmd.run

    def history_with_token(argv, timeout, env=None, kill_on_timeout=True):
        if argv[1] == "history" and token_reason:
            rig.cmd.history = [granted("L7", token_reason[0])]
        return original(argv, timeout, env=env, kill_on_timeout=kill_on_timeout)
    rig.cmd.run = history_with_token
    text = rig.call()
    assert "PASS" in text and rig.released() == ["L7"] and "#" in token_reason[0]


# P6 ---------------------------------------------------------------------------------------------------------------


def test_p6_an_interrupt_during_the_wait_cancels_the_ticket_and_reraises(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv, proc=FakeProc("", interrupt=True))
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert rig.cancelled() == ["T1"] and rig.released() == [] and rig.check.summary()["calls"][0]["outcome"] == "interrupted"


def test_p6b_an_interrupt_during_the_wait_releases_a_grant_that_arrived_in_the_meantime(tmp_path, fake_kenv):
    class Late(FakeProc):
        def communicate(self, timeout=None):
            if not self.killed:
                raise KeyboardInterrupt
            return GRANT_JSON, ""
    rig = Rig(tmp_path, fake_kenv, proc=Late())
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert rig.released() == ["L1"] and "harness" not in rig.events


# P7 ---------------------------------------------------------------------------------------------------------------


def test_p7_a_waiter_that_cannot_start_cancels_the_ticket(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv, proc=OSError("no gozer"))
    text = rig.call()
    assert "BUSY" in text and rig.cancelled() == ["T1"] and "harness" not in rig.events


# P8 ---------------------------------------------------------------------------------------------------------------


def test_p8_a_grant_without_a_device_list_is_released_and_never_run(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv, proc=FakeProc(json.dumps({"granted": True, "lease_id": "L1"})))
    text = rig.call()
    assert "harness" not in rig.events and rig.released() == ["L1"] and "no device list" in text


# P9 ---------------------------------------------------------------------------------------------------------------


def test_p9_an_exception_in_the_grant_callback_still_releases_the_lease(rig, monkeypatch):
    original = rig.check.wait_for

    def failing(k, queued, wait_s, timeout_s, on_grant):
        def boom():
            raise RuntimeError("callback failed")
        return original(k, queued, wait_s, timeout_s, boom)
    rig.check.wait_for = failing
    with pytest.raises(RuntimeError):
        rig.call()
    assert rig.released() == ["L1"] and rig.ledger.unresolved() is None


def test_p9b_an_interrupt_between_the_grant_and_the_run_still_releases_the_lease(rig):
    original = rig.check.wait_for

    def failing(k, queued, wait_s, timeout_s, on_grant):
        def boom():
            raise KeyboardInterrupt
        return original(k, queued, wait_s, timeout_s, boom)
    rig.check.wait_for = failing
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert rig.released() == ["L1"] and "harness" not in rig.events


# P10 --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["sweep", "gate"])
def test_p10_an_exception_in_the_checks_after_the_grant_releases_the_lease(rig, monkeypatch, where):
    calls = []
    if where == "sweep":
        def sweep(run_dir, **kw):
            calls.append(1)
            if len(calls) == 4:                    # prepare, run, after the gozer run, then under the lease
                raise RuntimeError("sweep failed")
            return []
        monkeypatch.setattr(kp, "sweep_strays", sweep)
    else:
        real = kp._hw_gate

        def gate(kernel, recorded):
            calls.append(1)
            if len(calls) == 3:                    # prepare, run, then the gate under the lease
                raise RuntimeError("gate failed")
            return real(kernel, recorded)
        monkeypatch.setattr(kp, "_hw_gate", gate)
    with pytest.raises(RuntimeError):
        rig.call()
    assert rig.released() == ["L1"] and "harness" not in rig.events


# P11 --------------------------------------------------------------------------------------------------------------


def test_p11_an_interrupt_in_the_harness_run_releases_the_lease(rig):
    def harness(argv, kw):
        raise KeyboardInterrupt
    rig.harness = harness
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert rig.released() == ["L1"]
    assert rig.check.summary()["calls"][0]["outcome"] == "interrupted"


def test_p11b_an_error_in_the_harness_run_releases_the_lease_and_propagates(rig):
    def harness(argv, kw):
        raise OSError("fork failed")
    rig.harness = harness
    with pytest.raises(OSError):
        rig.call()
    assert rig.released() == ["L1"]


# P12 --------------------------------------------------------------------------------------------------------------


def test_p12_a_hang_releases_the_lease_and_turns_the_tool_off(rig):
    rig.harness = lambda argv, kw: kp.Watched(-15, "stalled: no output and no CPU use for 60 s", True, 66.0)
    text = rig.call()
    assert "HANG" in text and rig.released() == ["L1"] and rig.call().startswith("hardware_check refused")


# P13 --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["lease released; NOT marked clean", "released but not resetting: no longer locked"])
def test_p13_a_release_that_did_not_reset_the_chips_is_told_to_the_agent_and_recorded(rig, message):
    rig.cmd.say("release", res(0, json.dumps({"message": message})))
    text = rig.call()
    assert "PASS" in text and "could not release the lease" in text and message in text
    assert rig.check.summary()["calls"][0]["release_error"].startswith("the lease was released but its chip reset")
    assert rig.ledger.unresolved() is None                     # released, so nothing blocks the next call


# P14 --------------------------------------------------------------------------------------------------------------


def test_p14_a_refused_release_keeps_the_lease_pending_and_blocks_the_next_call(rig):
    rig.cmd.say("release", res(15, "", "device still open"))
    rig.cmd.history = [granted("L1", "x")]
    text = rig.call()
    assert "device still open" in text and "gozer status" in text
    second = rig.call()
    assert second.startswith("hardware_check refused") and "L1" in second and "may still be held" in second
    assert rig.events.count("gozer-run") == 1


def test_p14b_the_next_call_may_start_once_history_shows_the_lease_released(rig):
    rig.cmd.say("release", res(15, "", "device still open"))
    rig.call()
    rig.cmd.history = [{"event": "released", "lease_id": "L1"}]
    rig.proc = FakeProc(GRANT_JSON)
    rig.cmd.say("release", res(0, json.dumps({"message": "ok"})))
    assert rig.call().splitlines()[0].startswith("hardware_check 2 of 6: PASS")


# P15 --------------------------------------------------------------------------------------------------------------


def test_p15_a_release_that_overruns_is_left_running_and_blocks_until_it_is_done(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv)
    alive = {"yes": True}
    rig.ledger.alive = lambda pid: alive["yes"]
    rig.cmd.say("release", res(None, timed_out=True, left_running=True, pid=4242))
    text = rig.call()
    assert "still running" in text
    release = [c for c in rig.cmd.calls if c["argv"][1] == "release"][0]
    assert release["kill_on_timeout"] is False                             # never killed: the reset must finish
    second = rig.call()
    assert second.startswith("hardware_check refused") and "still running (pid 4242)" in second
    alive["yes"] = False
    rig.cmd.history = [{"event": "released", "lease_id": "L1"}]
    rig.proc = FakeProc(GRANT_JSON)
    rig.cmd.say("release", res(0, json.dumps({"message": "ok"})))
    assert rig.call().splitlines()[0].startswith("hardware_check 2 of 6: PASS")


# P16 --------------------------------------------------------------------------------------------------------------


def test_p16_an_interrupt_during_the_release_leaves_the_lease_registered_and_blocks_the_next_call(rig):
    rig.cmd.say("release", KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert "L1" in rig.ledger.pending_leases
    rig.cmd.say("release", res(0, json.dumps({"message": "ok"})))
    assert rig.call().startswith("hardware_check refused")


# P17 --------------------------------------------------------------------------------------------------------------


def test_p17_gozer_history_that_cannot_be_read_blocks_the_next_call_and_says_so(rig):
    rig.cmd.say("release", res(15, "", "device still open"))
    rig.call()
    rig.cmd.history_fails = True
    assert "cannot confirm" in rig.call()


# P18 --------------------------------------------------------------------------------------------------------------


def test_p18_the_attempt_record_says_what_is_unresolved(rig):
    rig.cmd.say("release", res(15, "", "device still open"))
    rig.call()
    assert "L1" in rig.check.summary()["unresolved"]
    saved = json.loads(kp.hwcheck_calls_path(rig.adir.parent.parent, "add", 1).read_text())
    assert "L1" in saved["unresolved"]


def test_p19_the_report_says_what_is_unresolved(tmp_path):
    from test_agent_hw import _report_md, _row_attempt, hw_summary
    summary = dict(hw_summary({}), unresolved="lease L1 may still be held; look with `gozer status`")
    md = _report_md(tmp_path, [_row_attempt(agent_hw=summary)])
    assert "UNRESOLVED" in md and "lease L1 may still be held" in md


# the policy's cleanup reserve covers both phases ----------------------------------------------------------------


def test_the_cleanup_reserve_covers_the_harness_stop_and_the_separate_release(rig):
    p = hwtool.HwPolicy()
    assert p.cleanup_s >= kp.GOZER_TERM_GRACE_S + hwlease.RELEASE_TIMEOUT_S
    # a call's worst case, deadline plus both cleanup phases, never exceeds the lease cap
    assert (p.lease_s - p.cleanup_s) + kp.GOZER_TERM_GRACE_S + hwlease.RELEASE_TIMEOUT_S <= p.lease_s


def test_p20_a_ticket_kept_by_the_first_try_is_registered_before_the_wait_starts(rig):
    def interrupted(k, queued, wait_s, timeout_s, on_grant):
        raise KeyboardInterrupt
    rig.check.wait_for = interrupted
    rig.cmd.queue = [{"ticket": "T1"}]
    with pytest.raises(KeyboardInterrupt):
        rig.call()
    assert "T1" in rig.ledger.pending_tickets
    assert rig.call().startswith("hardware_check refused") and "ticket T1 is still queued" in rig.call()


def test_p21_a_zero_wait_cancels_without_starting_a_waiter(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv, policy=hwtool.HwPolicy(wait_s=0.0))
    text = rig.call()
    assert "wait" not in rig.events and rig.cancelled() == ["T1"] and "BUSY" in text


def test_p22_the_harness_runs_with_the_chips_the_lease_names(tmp_path, fake_kenv):
    rig = Rig(tmp_path, fake_kenv)
    seen = []
    original = rig._watched

    def watched(argv, **kw):
        if argv[0] != "gozer":
            seen.append(kw["env"].get("TT_VISIBLE_DEVICES"))
        return original(argv, **kw)
    rig.check = kp.make_hardware_check(rig.adir, op="add", n=1, shared_python="/shared/python", policy=hwtool.HwPolicy(),
                                       run_watched=watched, ledger=rig.ledger, clock=lambda: CLOCK["now"],
                                       sleep=lambda s: None)
    rig.call()
    assert seen == ["0000:01:00.0,0000:02:00.0"]


def test_p23_a_failing_lease_check_blocks_the_next_call_and_says_so():
    from test_agent_hw import World, good_hw

    class Boom(Exception):
        pass

    def health():
        raise Boom("history unreadable")
    w = World()
    c = hwtool.HardwareCheck(hwtool.HwPolicy(), prepare=w.prepare, run=w.run, clock=w.clock, sleep=w.sleep, health=health)
    text = c.call({})
    assert text.startswith("hardware_check refused") and "the lease check failed" in text and w.runs == []
