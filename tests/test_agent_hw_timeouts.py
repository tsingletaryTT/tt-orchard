# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 116, round 10, finding 1: the wait for a busy board and the run after the grant are two phases, each with its own allowance.

The wrapper starts the call-timeout clock at the grant and stops the harness at that deadline, so a long wait cannot shorten the run
and a long run cannot be cut short by the wait. The probe's outer timeout is the sum of every phase: the acquire, the wait (and its
margin), the call timeout and the release grace, so the outer timeout never fires before the wrapper's own deadlines. Fakes only."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hwfakes import PID, FakeGozer
from orchard import hwtool, kernelprobe as kp
from test_agent_hw import CLOCK, WHO, attempt_dir, fake_gozer, make_check  # noqa: F401
from test_agent_hw import fake_kenv, skills_current, ttlang_skill  # noqa: F401
from test_kernelprobe import GOOD_HW_SET, hw_writer  # noqa: F401


class Mono:
    """A fake monotonic clock that only moves when something waits."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class HungCmd:
    """A harness that never ends by itself: each bounded wait uses up its whole timeout, and an unbounded wait is a bug in the wrapper."""

    def __init__(self, mono):
        self.mono, self.terminated, self.killed = mono, False, False
        self.began = None

    def spawn(self, argv, env=None):
        self.began = self.mono.t
        return self

    def wait(self, timeout=None):
        if self.terminated:
            return -15
        assert timeout is not None, "the wrapper waited for a hung harness with no deadline"
        self.mono.t += timeout
        raise subprocess.TimeoutExpired("harness", timeout)

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def spend_on_wait(gz, mono, seconds):
    """`gozer wait` takes `seconds` of the fake clock before it answers."""
    original = gz.sh

    def sh(argv, timeout):
        if argv[1] == "wait":
            mono.t += seconds
        return original(argv, timeout)
    gz.sh = sh


def run(tmp_path, gz, cmd, mono, *, wait_s=120.0, call_timeout_s=30.0):
    status = hwtool.run_wrapper(who=WHO, reason="why", wait_s=wait_s, result_path=tmp_path / "lease.json", command=["harness"], sh=gz.sh,
                                spawn=cmd.spawn, now=lambda: mono.t, mono=mono, out=lambda line: None, pid=PID,
                                call_timeout_s=call_timeout_s)
    return status, json.loads((tmp_path / "lease.json").read_text())


@pytest.mark.parametrize("wait_spent", [100.0, 0.0], ids=["after a long wait", "after no wait"])
def test_a_hung_harness_is_stopped_at_the_call_timeout_counted_from_the_grant(tmp_path, wait_spent):
    mono, gz = Mono(), FakeGozer(busy=1 if wait_spent else 0)
    cmd = HungCmd(mono)
    if wait_spent:
        spend_on_wait(gz, mono, wait_spent)
    status, rec = run(tmp_path, gz, cmd, mono)
    assert status == hwtool.WRAP_TIMEOUT and rec["outcome"] == "timeout"
    assert mono.t - cmd.began == 30.0                      # the whole call timeout after the grant, whatever the wait took
    assert cmd.terminated and gz.released() == ["L1"] and gz.clean() and rec["release_ok"] is True


def test_a_long_wait_does_not_shorten_the_run(tmp_path):
    """The wait took 100 s of a 120 s allowance and the run still gets 30 s of its own."""
    mono, gz = Mono(), FakeGozer(busy=1)
    spend_on_wait(gz, mono, 100.0)
    cmd = HungCmd(mono)
    run(tmp_path, gz, cmd, mono, wait_s=120.0, call_timeout_s=30.0)
    assert mono.t == 100.0 + 30.0 and cmd.began == 100.0


def test_a_wait_that_ends_without_a_grant_never_starts_a_run_or_a_run_clock(tmp_path):
    mono, gz = Mono(), FakeGozer(busy=1, wait="timeout")
    spend_on_wait(gz, mono, 120.0)
    cmd = HungCmd(mono)
    status, rec = run(tmp_path, gz, cmd, mono)
    assert status == hwtool.WRAP_BUSY and cmd.began is None and rec["outcome"] == "busy" and mono.t == 120.0
    assert gz.cancelled() == [rec["ticket"]]


def test_a_harness_that_ends_before_the_deadline_is_not_a_timeout(tmp_path):
    class Quick(HungCmd):
        def wait(self, timeout=None):
            self.mono.t += 5.0
            return 0
    mono, gz = Mono(), FakeGozer()
    status, rec = run(tmp_path, gz, Quick(mono), mono)
    assert status == 0 and rec["outcome"] == "ran"


def test_the_deadline_is_checked_before_every_wait_so_a_late_wake_up_still_stops_the_harness(tmp_path):
    class Late(HungCmd):
        def wait(self, timeout=None):
            if self.terminated:
                return -15
            self.mono.t += timeout + 50.0                  # the scheduler woke the wrapper far too late
            raise subprocess.TimeoutExpired("harness", timeout)
    mono, gz = Mono(), FakeGozer()
    cmd = Late(mono)
    status, rec = run(tmp_path, gz, cmd, mono)
    assert status == hwtool.WRAP_TIMEOUT and cmd.terminated


def test_the_outer_timeout_is_the_sum_of_every_phase_so_it_never_fires_before_the_wrappers_own_deadlines():
    outer = hwtool.outer_backstop_s(120.0, 240.0)
    assert outer == hwtool.ACQUIRE_TIMEOUT_S + 120.0 + hwtool.WAIT_MARGIN_S + 240.0 + hwtool.RELEASE_GRACE_S
    assert outer > 120.0 + 240.0 + hwtool.RELEASE_GRACE_S
    assert hwtool.outer_backstop_s(0.0, 10.0) == hwtool.ACQUIRE_TIMEOUT_S + hwtool.WAIT_MARGIN_S + 10.0 + hwtool.RELEASE_GRACE_S


def test_the_probe_gives_the_watcher_the_outer_timeout_and_the_wrapper_the_call_timeout(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)
    seen = []

    def run_watched(argv, **kw):
        seen.append((list(argv), kw))
        return kp.Watched(0, None, False, 1.0)
    gz = fake_gozer()
    kp.make_hardware_check(adir, op="add", n=1, shared_python="/shared/python", policy=hwtool.HwPolicy(wait_s=33.0, call_timeout_s=111.0),
                           run_watched=run_watched, status=gz.sh).call({})
    argv, kw = seen[0]
    assert argv[argv.index("--call-timeout-s") + 1] == "111.0" and argv[argv.index("--wait-s") + 1] == "33.0"
    assert kw["backstop_s"] == hwtool.outer_backstop_s(33.0, 111.0)


def test_a_call_timeout_the_wrapper_enforced_is_a_hang_that_turns_the_tool_off_and_charges_its_own_stamps(tmp_path, fake_kenv):
    adir = attempt_dir(tmp_path, fake_kenv)

    def run_watched(argv, **kw):
        result = Path(argv[argv.index("--result") + 1])
        result.write_text(json.dumps({"pid": PID, "started_at": 100.0, "granted_at": 160.0, "ended_at": 250.0, "lease_id": "L1",
                                      "outcome": "timeout", "phase": "done", "release_ok": True}))
        CLOCK["now"] += 150.0
        return kp.Watched(hwtool.WRAP_TIMEOUT, None, False, 150.0)
    gz = fake_gozer()
    CLOCK["now"] = 0.0
    check = kp.make_hardware_check(adir, op="add", n=1, shared_python="/shared/python", policy=hwtool.HwPolicy(call_timeout_s=60.0),
                                   run_watched=run_watched, status=gz.sh, clock=lambda: CLOCK["now"], sleep=lambda s: None)
    text = check.call({})
    rec = check.summary()["calls"][0]
    assert "HANG" in text and "call timeout" in text and rec["outcome"] == "hang" and rec["timed_out"] is True
    assert rec["lease_s"] == 90.0 and rec["waited_s"] == 60.0 and check.hung is True
    assert "hung" in check.call({})


def test_the_command_line_carries_the_call_timeout_to_the_wrapper(tmp_path):
    mono, gz = Mono(), FakeGozer()
    cmd = HungCmd(mono)
    status = hwtool.wrapper_cli(["run", "--who", WHO, "--reason", "r", "--wait-s", "5", "--result", str(tmp_path / "r.json"),
                                 "--call-timeout-s", "30.0", "--", "harness"], install_signals=False, sh=gz.sh, spawn=cmd.spawn,
                                now=lambda: mono.t, mono=mono, out=lambda line: None, pid=PID)
    assert status == hwtool.WRAP_TIMEOUT and mono.t - cmd.began == 30.0
    assert hwtool.wrapper_cli(["run", "--who", WHO, "--reason", "r", "--wait-s", "5", "--result", str(tmp_path / "r.json"),
                               "--call-timeout-s", "soon", "--", "harness"], install_signals=False) == 2
