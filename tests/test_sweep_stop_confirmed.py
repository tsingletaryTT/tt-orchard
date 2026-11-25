# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 476: a SIGSTOP sent is not a process stopped.

`kernelprobe.sweep_strays` freezes every helper it knows (SIGSTOP) and scans again, so that a helper that keeps forking cannot start a child
after the last scan. A helper can fork between the signal and the moment it stops, and that child is not in the scan that follows. The sweep
therefore waits, with a bound, until each helper it signalled shows state T (or is gone) before it scans, and when the bound passes it adds
a problem to the sweep status so that a caller that guards hardware treats the sweep as incomplete.

The helpers here are real processes. The kill function is replaced by one that delays, drops or rewrites the SIGSTOP, so each case is a real
/proc state, not a made-up one.
"""
from __future__ import annotations

import os
import signal
import threading
import time

import pytest

from orchard import kernelprobe as kp
from test_kernelprobe import _child, _stop, _wait_for_env


def _state(pid):
    st = kp._observe("/proc", pid)
    return None if st is None or st is kp._UNOBSERVABLE else st[0]


def _stubborn(run_dir):
    """A helper that ignores SIGTERM, so it is still alive when the freeze starts."""
    import subprocess, sys
    env = dict(os.environ, ORCHARD_RUN_DIR=str(run_dir.resolve()))
    p = subprocess.Popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"],
                         env=env, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert p.stdout.readline().strip() == b"ready"
    return p


def _attempt(tmp_path):
    adir = tmp_path / "attempt-1"
    adir.mkdir()
    return adir


def test_the_scan_after_a_freeze_runs_only_once_the_helper_shows_state_stopped(tmp_path, monkeypatch):
    """The stand-in delivers SIGSTOP 0.3 s late. Every scan that comes after a SIGSTOP was sent must see the helper in state T."""
    adir = _attempt(tmp_path)
    helper = _stubborn(adir)
    seen = []                                     # (stop sent yet, state of the helper at the moment of the scan)
    sent_stop = []
    real_scan = kp.scan_strays

    def kill(pid, sig):
        if sig == signal.SIGSTOP:
            sent_stop.append(pid)
            threading.Timer(0.3, os.kill, (pid, sig)).start()      # the signal takes effect late
        else:
            os.kill(pid, sig)

    def scan(run_dir, **kw):
        seen.append((bool(sent_stop), _state(helper.pid)))
        return real_scan(run_dir, **kw)

    monkeypatch.setattr(kp, "scan_strays", scan)
    try:
        _wait_for_env(helper.pid, str(adir.resolve()))
        status = {}
        kp.sweep_strays(adir, kill=kill, killpg=lambda pgid, sig: kill(pgid, sig), grace_s=0.0, stop_wait_s=3.0, status=status)
        after_stop = [state for sent, state in seen if sent]
        assert after_stop, "the freeze never scanned after a SIGSTOP"
        assert all(state in ("T", "t", "Z", None) for state in after_stop), after_stop
        assert status["complete"] is True, status
    finally:
        _stop(helper)


def test_a_helper_that_does_not_stop_within_the_bound_makes_the_sweep_incomplete(tmp_path):
    adir = _attempt(tmp_path)
    helper = _stubborn(adir)

    def kill(pid, sig):
        if sig != signal.SIGSTOP:                  # the stop never takes effect
            os.kill(pid, sig)

    try:
        _wait_for_env(helper.pid, str(adir.resolve()))
        status = {}
        records = kp.sweep_strays(adir, kill=kill, killpg=lambda pgid, sig: kill(pgid, sig), grace_s=0.0, stop_wait_s=0.3, status=status)
        assert status["complete"] is False
        mine = [p for p in status["problems"] if str(helper.pid) in p and "stop" in p]
        assert mine, status["problems"]
        assert records[0]["stopped"] is True          # the SIGKILL still went out and removed it
    finally:
        _stop(helper)


def test_a_helper_that_stops_in_time_leaves_no_stop_problem(tmp_path):
    adir = _attempt(tmp_path)
    helper = _child(adir)
    try:
        _wait_for_env(helper.pid, str(adir.resolve()))
        status = {}
        kp.sweep_strays(adir, grace_s=0.0, stop_wait_s=3.0, status=status)
        assert status == {"problems": [], "complete": True}, status
    finally:
        _stop(helper)


def test_a_helper_that_is_gone_after_the_signal_needs_no_stop(tmp_path):
    """The stand-in turns the SIGSTOP into a SIGKILL. A process that is gone (or a zombie) is as good as stopped."""
    adir = _attempt(tmp_path)
    helper = _child(adir)

    def kill(pid, sig):
        os.kill(pid, signal.SIGKILL if sig == signal.SIGSTOP else sig)

    try:
        _wait_for_env(helper.pid, str(adir.resolve()))
        status = {}
        kp.sweep_strays(adir, kill=kill, killpg=lambda pgid, sig: kill(pgid, sig), grace_s=0.0, stop_wait_s=3.0, status=status)
        assert status["complete"] is True, status
    finally:
        _stop(helper)


def test_the_wait_is_bounded_by_the_injected_clock(tmp_path):
    """A fake clock that only the sleep function advances: the wait may poll at most bound / poll interval times (plus one), then gives up."""
    adir = _attempt(tmp_path)
    helper = _stubborn(adir)
    now = [1000.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    def kill(pid, sig):
        if sig != signal.SIGSTOP:
            os.kill(pid, sig)

    try:
        _wait_for_env(helper.pid, str(adir.resolve()))
        status = {}
        kp.sweep_strays(adir, kill=kill, killpg=lambda pgid, sig: kill(pgid, sig), grace_s=0.0, stop_wait_s=0.5,
                        clock=lambda: now[0], sleep=sleep, status=status)
        assert status["complete"] is False
        per_wait = 0.5 / kp.SWEEP_POLL_S + 2
        rounds = kp.SWEEP_FREEZE_ROUNDS + 1
        assert len(sleeps) <= per_wait * rounds + 40, len(sleeps)          # a bound, not a spin; the +40 covers the other waits of the sweep
        assert sum(sleeps) <= 0.5 * rounds + 5
    finally:
        _stop(helper)


def test_the_default_bound_is_a_module_constant_a_caller_can_read():
    assert isinstance(kp.SWEEP_STOP_WAIT_S, float) and 0 < kp.SWEEP_STOP_WAIT_S <= 5


@pytest.mark.parametrize("state, running", [("S", True), ("R", True), ("D", True), ("T", False), ("t", False), ("Z", False), ("X", False)])
def test_which_proc_states_count_as_stopped(tmp_path, state, running):
    d = tmp_path / "4242"
    d.mkdir()
    (d / "stat").write_text(f"4242 (helper) {state} " + " ".join(["1"] * 18) + " 500 0 0\n")
    assert kp._still_running(str(tmp_path), 4242) is running


def test_a_process_with_no_proc_entry_is_not_running(tmp_path):
    assert kp._still_running(str(tmp_path), 4242) is False


def test_a_proc_entry_that_cannot_be_read_may_still_run(tmp_path):
    """A stat entry that cannot be read says nothing about the process: it is not confirmed stopped."""
    d = tmp_path / "4242"
    (d / "stat").mkdir(parents=True)               # opening a directory as a file fails with an OSError that is not "no such file"
    assert kp._still_running(str(tmp_path), 4242) is True


def test_the_default_bound_gives_a_slow_stop_time_to_take_effect(tmp_path):
    """No `stop_wait_s` given: a SIGSTOP that lands 0.3 s late is inside the default bound, so the sweep is complete."""
    adir = _attempt(tmp_path)
    helper = _stubborn(adir)

    def kill(pid, sig):
        if sig == signal.SIGSTOP:
            threading.Timer(0.3, os.kill, (pid, sig)).start()
        else:
            os.kill(pid, sig)

    try:
        status = {}
        kp.sweep_strays(adir, kill=kill, killpg=lambda pgid, sig: kill(pgid, sig), grace_s=0.0, status=status)
        assert status == {"problems": [], "complete": True}, status
    finally:
        _stop(helper)


def test_the_process_that_the_last_round_found_is_confirmed_stopped_too(tmp_path, monkeypatch):
    """With one freeze round, the second helper is found by that round's scan and is stopped by the extra SIGSTOP after the loop. That stop is
    checked as well: a second helper that does not stop is named in a problem."""
    adir = _attempt(tmp_path)
    first, second = _stubborn(adir), _stubborn(adir)
    calls = []

    def scan(run_dir, **kw):
        calls.append(1)
        return [first.pid] if len(calls) == 1 else [first.pid, second.pid]

    def kill(pid, sig):
        if sig == signal.SIGSTOP and pid == second.pid:
            return                                  # only the second helper ignores the stop
        os.kill(pid, sig)

    monkeypatch.setattr(kp, "scan_strays", scan)
    monkeypatch.setattr(kp, "SWEEP_FREEZE_ROUNDS", 1)
    try:
        status = {}
        kp.sweep_strays(adir, kill=kill, killpg=lambda pgid, sig: kill(pgid, sig), grace_s=0.0, stop_wait_s=0.3, status=status)
        assert any(str(second.pid) in p and "SIGSTOP" in p for p in status["problems"]), status["problems"]
        assert not any(str(first.pid) in p and "SIGSTOP" in p for p in status["problems"]), status["problems"]
    finally:
        _stop(first, second)
