# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 116, sixth review round: an interrupt during `gozer acquire` or a claim must stop and reap that subprocess, so no acquire is still
running (or a zombie) when the probe moves on, and whatever it granted is found by the reclaim. The runner is hwlease.run_reaping, which
starts the acquisition commands; release and reset commands keep their leave-running behaviour (hwlease.release). Fakes only."""
from __future__ import annotations

import json
import signal
import subprocess

import pytest

from orchard import hwlease
from orchard.commands import CommandResult


class ReapProc:
    """A fake child. `mode`: "ok" (communicate returns), "hang" (communicate times out), "interrupt" (the first communicate is
    interrupted). `dies_on`: the first signal that ends it: "TERM", "KILL" or "TERM_AFTER_DELAY" (ends on SIGTERM but only after a
    wait that is too short for the grace, so wait(timeout=grace) times out once)."""

    def __init__(self, mode="ok", dies_on="TERM", out="{}", code=0, pid=5000):
        self.mode, self.dies_on, self.out, self.code, self.pid = mode, dies_on, out, code, pid
        self.returncode = None
        self.signals, self.waits, self.communicates, self.killed = [], [], [], False
        self._delay_left = 1 if dies_on == "TERM_AFTER_DELAY" else 0
        self._interrupted = False

    def deliver(self, sig):
        self.signals.append(sig)
        if sig == signal.SIGKILL:
            self.killed, self.returncode = True, -9
        elif sig == signal.SIGTERM and self.dies_on in ("TERM", "TERM_AFTER_DELAY"):
            self.terminated = True

    terminated = False

    def communicate(self, timeout=None):
        self.communicates.append(timeout)
        if self.mode == "interrupt" and not self._interrupted:
            self._interrupted = True
            raise KeyboardInterrupt
        if self.mode == "hang" and not (self.killed or self.terminated):
            raise subprocess.TimeoutExpired("gozer acquire", timeout)
        if self.returncode is None:
            self.returncode = self.code
        return self.out, ""

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.returncode is not None:
            return self.returncode
        if self.killed:
            self.returncode = -9
            return self.returncode
        if self.terminated:
            if self._delay_left > 0:
                self._delay_left -= 1
                raise subprocess.TimeoutExpired("gozer acquire", timeout)
            self.returncode = -15
            return self.returncode
        raise subprocess.TimeoutExpired("gozer acquire", timeout)         # ignores SIGTERM: only SIGKILL ends it

    def poll(self):
        return self.returncode


def run(proc, **kw):
    sent = []

    def killpg(pgid, sig):
        sent.append((pgid, sig))
        proc.deliver(sig)
    popen_args = {}

    def popen(argv, **k):
        popen_args["argv"], popen_args["kw"] = list(argv), k
        return proc
    result = hwlease.run_reaping(["gozer", "acquire"], 30.0, popen=popen, killpg=killpg, **kw)
    return result, sent, popen_args


def test_a_command_that_finishes_is_returned_and_reaped_without_a_signal():
    proc = ReapProc(out="done", code=0)
    result, sent, popen_args = run(proc)
    assert result.returncode == 0 and result.stdout == "done" and sent == [] and result.timed_out is False
    assert popen_args["kw"]["start_new_session"] is True and proc.communicates == [30.0]


def test_a_command_that_cannot_start_is_a_127_result():
    def popen(argv, **k):
        raise FileNotFoundError("no gozer")
    result = hwlease.run_reaping(["gozer"], 5.0, popen=popen)
    assert result.returncode == 127 and "no gozer" in result.stderr


def test_a_timeout_kills_the_group_and_reaps_it_and_says_it_timed_out():
    proc = ReapProc(mode="hang", dies_on="KILL")
    result, sent, _ = run(proc)
    assert result.timed_out is True and result.returncode is None
    assert [s for _, s in sent][-1] == signal.SIGKILL and proc.returncode is not None        # waited: no zombie


def test_an_interrupt_stops_the_acquire_with_sigterm_and_waits_for_it_before_the_interrupt_goes_on():
    proc = ReapProc(mode="interrupt", dies_on="TERM")
    with pytest.raises(KeyboardInterrupt):
        run(proc)
    assert proc.signals == [signal.SIGTERM] and proc.returncode == -15 and proc.waits        # terminated, then waited on


def test_an_acquire_that_ignores_sigterm_is_killed_after_the_grace_and_still_reaped():
    proc = ReapProc(mode="interrupt", dies_on="KILL")
    with pytest.raises(KeyboardInterrupt):
        run(proc)
    assert proc.signals == [signal.SIGTERM, signal.SIGKILL] and proc.returncode == -9
    assert proc.waits[0] == hwlease.ACQUIRE_TERM_GRACE_S and len(proc.waits) == 2            # the grace, then the final reap


def test_an_acquire_that_exits_only_after_a_delay_gets_the_whole_grace_and_is_not_killed_early():
    proc = ReapProc(mode="interrupt", dies_on="TERM_AFTER_DELAY")
    with pytest.raises(KeyboardInterrupt):
        run(proc)
    assert signal.SIGKILL in proc.signals                                                    # the first wait (the grace) timed out
    assert proc.returncode is not None


def test_the_group_that_is_signalled_is_the_childs_own_never_ours_or_pid_one():
    proc = ReapProc(mode="interrupt", dies_on="TERM", pid=5000)
    with pytest.raises(KeyboardInterrupt):
        run(proc)
    sent_to = {pgid for pgid, _ in run_sent(proc)}
    assert sent_to == {5000}


def run_sent(proc):
    sent = []

    def killpg(pgid, sig):
        sent.append((pgid, sig))
        proc.deliver(sig)
    proc2 = ReapProc(mode="interrupt", dies_on="TERM", pid=proc.pid)
    with pytest.raises(KeyboardInterrupt):
        hwlease.run_reaping(["gozer"], 5.0, popen=lambda argv, **k: proc2, killpg=killpg)
    return sent


@pytest.mark.parametrize("pid", [0, 1, -3])
def test_a_pid_that_could_be_everyone_is_never_signalled(pid):
    proc = ReapProc(mode="interrupt", dies_on="KILL", pid=pid)
    sent = []
    with pytest.raises(KeyboardInterrupt):
        hwlease.run_reaping(["gozer"], 5.0, popen=lambda argv, **k: proc, killpg=lambda pgid, sig: sent.append((pgid, sig)))
    assert sent == []


def test_a_group_that_is_already_gone_is_not_an_error():
    proc = ReapProc(mode="interrupt", dies_on="TERM")

    def killpg(pgid, sig):
        raise ProcessLookupError
    with pytest.raises(KeyboardInterrupt):
        hwlease.run_reaping(["gozer"], 5.0, popen=lambda argv, **k: proc, killpg=killpg)


def test_the_reaper_never_hides_the_interrupt_even_when_the_reap_itself_fails():
    class Stubborn(ReapProc):
        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("gozer", timeout)
    proc = Stubborn(mode="interrupt", dies_on="KILL")
    with pytest.raises(KeyboardInterrupt):
        hwlease.run_reaping(["gozer"], 5.0, popen=lambda argv, **k: proc, killpg=lambda p, s: proc.deliver(s))


# ---- the acquisition commands use the reaping runner; the others do not ----------------------------------------------------


def test_every_command_but_the_release_uses_the_reaping_runner(monkeypatch):
    plain, reaping = [], []

    def run_plain(argv, timeout, **kw):
        plain.append(list(argv))
        return CommandResult(tuple(argv), 0, json.dumps({"history": [], "queue": [], "message": "ok"}), "")

    def run_reap(argv, timeout, **kw):
        reaping.append(list(argv))
        return CommandResult(tuple(argv), 10, json.dumps({"queued": True, "granted": False, "ticket": "T1"}), "")
    def run_read(argv, timeout, **kw):
        reaping.append(list(argv))
        return CommandResult(tuple(argv), 0, json.dumps({"history": [], "queue": [], "message": "ok"}), "")

    def reap_dispatch(argv, timeout, **kw):
        return (run_reap if argv[1] == "acquire" else run_read)(argv, timeout, **kw)
    monkeypatch.setattr(hwlease, "_run_gozer", run_plain)
    monkeypatch.setattr(hwlease, "run_reaping", reap_dispatch)
    led = hwlease.GozerLeases(who="claude:test", pid=4321)
    led.request("why")
    led.claim("T1", "why")
    led.release("L1")
    led.cancel("T1")
    led.mine()
    assert [a[1] for a in reaping] == ["acquire", "acquire", "cancel", "history", "queue"]                   # acquisitions and cancels are reaped
    assert [a[1] for a in plain] == ["release"]                                         # only the release may be left running


def test_an_injected_runner_is_used_for_every_command_so_tests_never_start_a_process():
    seen = []

    def run(argv, timeout, **kw):
        seen.append(argv[1])
        return CommandResult(tuple(argv), 0, json.dumps({"history": [], "queue": []}), "")
    led = hwlease.GozerLeases(who="claude:test", pid=4321, run=run)
    led.request("why")
    led.release("L1")
    assert seen == ["acquire", "release"]


def test_an_interrupt_in_the_acquire_is_followed_by_a_reclaim_that_finds_what_it_granted():
    """The closure calls reclaim after the interrupt: the child was stopped first, so history shows a grant or nothing, never later."""
    events = []

    def run(argv, timeout, **kw):
        events.append(argv[1])
        if argv[1] == "history":
            return CommandResult(tuple(argv), 0, json.dumps({"history": [
                {"event": "granted", "lease_id": "L1", "pid": 4321, "who": "claude:test", "chips": ["a"], "ts": "t"}]}), "")
        if argv[1] == "queue":
            return CommandResult(tuple(argv), 0, json.dumps({"queue": []}), "")
        if argv[1] == "acquire":
            raise KeyboardInterrupt
        return CommandResult(tuple(argv), 0, json.dumps({"message": "ok"}), "")
    led = hwlease.GozerLeases(who="claude:test", pid=4321, run=run)
    with pytest.raises(KeyboardInterrupt):
        led.request("why")
    result = led.reclaim()
    assert [x["lease_id"] for x in result.leases] == ["L1"] and events.count("release") == 1


def test_our_own_process_group_is_never_signalled():
    import os
    proc = ReapProc(mode="interrupt", dies_on="KILL", pid=os.getpgrp())
    sent = []
    with pytest.raises(KeyboardInterrupt):
        hwlease.run_reaping(["gozer"], 5.0, popen=lambda argv, **k: proc, killpg=lambda pgid, sig: sent.append((pgid, sig)))
    assert sent == []
