# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stall watcher: it kills on signs that nothing is happening, and a deadline is only its backstop."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from orchard.stallwatch import StallWatcher, run_watched, tree_cpu_ticks


def test_no_output_and_no_cpu_for_the_quiet_window_is_a_stall():
    w = StallWatcher(quiet_s=10, min_cpu_ticks=2)
    assert w.observe(0.0, 100, 50) is None
    assert w.observe(9.0, 100, 50) is None
    assert "stalled" in w.observe(10.5, 100, 51)          # one tick of CPU is below min_cpu_ticks


def test_output_growth_counts_as_progress():
    w = StallWatcher(quiet_s=10)
    w.observe(0.0, 100, 50)
    assert w.observe(9.0, 101, 50) is None                # a byte of output resets the window
    assert w.observe(18.0, 101, 50) is None
    assert "stalled" in w.observe(19.5, 101, 50)


def test_cpu_use_counts_as_progress_and_slow_cpu_adds_up():
    w = StallWatcher(quiet_s=10, min_cpu_ticks=2)
    w.observe(0.0, 100, 50)
    assert w.observe(9.0, 100, 52) is None                # two ticks since the mark: progress
    assert w.observe(18.0, 100, 52) is None               # the mark moved at 9 s, so 9 quiet seconds is still inside the window
    w2 = StallWatcher(quiet_s=10, min_cpu_ticks=3)
    w2.observe(0.0, 100, 50)
    w2.observe(4.0, 100, 51)
    w2.observe(8.0, 100, 52)
    assert w2.observe(9.0, 100, 53) is None               # one tick a sample still reaches 3 over the window
    assert w2.observe(15.0, 100, 53) is None              # that moved the mark to 9 s; without the CPU rule it stalls here


def test_a_repeating_line_is_reported():
    w = StallWatcher(repeat_lines=5)
    assert w.feed_lines(["compiling"] * 4) is None
    assert "repeating" in w.feed_lines(["compiling"])


def test_a_changed_line_resets_the_repeat_count():
    w = StallWatcher(repeat_lines=5)
    w.feed_lines(["a"] * 4)
    assert w.feed_lines(["b"]) is None
    assert w.feed_lines(["a"] * 4) is None


def test_tree_cpu_ticks_sums_a_process_and_its_descendants(tmp_path):
    def proc(pid, ppid, ut, st, comm="x"):
        d = tmp_path / str(pid)
        d.mkdir()
        # fields after the comm: state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime
        (d / "stat").write_text(f"{pid} ({comm}) S {ppid} 1 1 0 0 0 0 0 0 0 {ut} {st} 0 0 20 0 1 0 0 0 0\n")
    proc(100, 1, 10, 5)
    proc(101, 100, 3, 2, comm="a (b) c")                  # a command name with spaces and parentheses
    proc(102, 101, 1, 1)
    proc(200, 1, 999, 999)                                # not a descendant
    assert tree_cpu_ticks(100, proc_root=str(tmp_path)) == 10 + 5 + 3 + 2 + 1 + 1


def test_a_silent_idle_child_is_killed_as_stalled(tmp_path):
    w = StallWatcher(quiet_s=1.0, min_cpu_ticks=2)
    got = run_watched(["sleep", "60"], cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert got.killed and "stalled" in got.reason and got.elapsed_s < 10


def test_a_busy_silent_child_is_not_a_stall_and_the_backstop_ends_it(tmp_path):
    w = StallWatcher(quiet_s=1.0, min_cpu_ticks=2)
    spin = [sys.executable, "-c", "while True: pass"]
    got = run_watched(spin, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=3, poll_s=0.2)
    assert got.killed and got.reason.startswith("backstop")


def test_a_child_that_keeps_printing_is_left_alone_until_it_exits(tmp_path):
    w = StallWatcher(quiet_s=1.0, repeat_lines=1000)
    cmd = [sys.executable, "-c", "import time\nfor i in range(8):\n print(i, flush=True); time.sleep(0.3)"]
    got = run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert not got.killed and got.returncode == 0


def test_a_child_that_repeats_one_line_is_killed(tmp_path):
    w = StallWatcher(quiet_s=30, repeat_lines=10)
    cmd = [sys.executable, "-c", "import time\nwhile True:\n print('same', flush=True); time.sleep(0.05)"]
    got = run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert got.killed and "repeating" in got.reason


def test_the_whole_process_group_is_killed_so_a_grandchild_does_not_survive(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    w = StallWatcher(quiet_s=1.0)
    cmd = ["sh", "-c", f"sleep 120 & echo $! > {pidfile}; wait"]
    got = run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert got.killed
    gone = False
    pid = int(pidfile.read_text())
    for _ in range(20):                      # the kernel needs a moment to reap; the loop checks the pid itself
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            gone = True
            break
        time.sleep(0.1)
    assert gone


# ---------------------------------------------------------------------------
# Review fixes: stale log, dropped CPU ticks, cleanup on error, partial lines,
# a bounded wait after the kill, and a child that ends before the kill.
# ---------------------------------------------------------------------------
import subprocess as _sp

import orchard.stallwatch as stallwatch


class _ExitedProc:
    """A fake child that has already ended by itself, but whose wait() still times out once."""
    pid = 0
    returncode = 0

    def wait(self, timeout=None):
        raise _sp.TimeoutExpired("fake", timeout)

    def poll(self):
        return 0


class _StuckProc:
    """A fake child that never ends and ignores the kill. A wait with no timeout would block forever."""
    pid = 0
    returncode = None

    def wait(self, timeout=None):
        assert timeout is not None, "an unbounded wait on a child that will not end"
        raise _sp.TimeoutExpired("fake", timeout)

    def poll(self):
        return None


def _alive_with(pid, marker):
    """True only when `pid` is alive, is no zombie, and its command line still holds `marker` (this test's unique tmp
    path), so a pid the kernel has reused for another process is never killed."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
            stat = f.read()
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2:].split()[0] != "Z" and marker.encode() in cmdline


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_an_old_log_is_not_read_as_this_childs_output(tmp_path):
    log = tmp_path / "log"
    log.write_bytes(b"same\n" * 50)                       # left by an earlier attempt, appended to
    w = StallWatcher(quiet_s=30, repeat_lines=5)
    cmd = [sys.executable, "-c", "import time\nfor i in range(6):\n print('line', i, flush=True); time.sleep(0.3)"]
    got = run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=log, watcher=w, backstop_s=30, poll_s=0.2)
    assert not got.killed and got.returncode == 0


def test_a_drop_in_tree_cpu_ticks_is_rebased_and_later_cpu_use_counts():
    w = StallWatcher(quiet_s=10, min_cpu_ticks=2)
    w.observe(0.0, 100, 500)
    assert w.observe(5.0, 100, 100) is None               # a busy descendant exited: its ticks left the total
    assert w.observe(11.0, 100, 103) is None              # 3 ticks since the rebase is progress


def test_a_drop_in_tree_cpu_ticks_does_not_reset_the_quiet_clock():
    w = StallWatcher(quiet_s=10, min_cpu_ticks=2)
    w.observe(0.0, 100, 500)
    assert w.observe(5.0, 100, 100) is None
    assert "stalled" in w.observe(10.5, 100, 100)         # no output and no new CPU since 0 s


def test_the_child_is_killed_when_the_watch_loop_raises(tmp_path):
    class Boom(StallWatcher):
        def observe(self, *a):
            raise RuntimeError("boom")
    made = []

    def popen(*a, **k):
        made.append(subprocess.Popen(*a, **k))
        return made[-1]
    try:
        with pytest.raises(RuntimeError):
            run_watched(["sleep", "60"], cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                        watcher=Boom(), backstop_s=30, poll_s=0.2, popen=popen)
        assert not _alive(made[0].pid)
    finally:
        for p in made:
            p.kill()                         # skips a child run_watched has already reaped, so no freed pid is hit
            p.wait(timeout=5)


def test_a_line_split_across_two_reads_is_counted_once(tmp_path):
    w = StallWatcher(quiet_s=30, repeat_lines=3)
    code = ("import sys, time\n"
            "sys.stdout.write('same\\nsame\\nsa'); sys.stdout.flush(); time.sleep(0.8)\n"
            "sys.stdout.write('me\\n'); sys.stdout.flush(); time.sleep(5)")
    got = run_watched([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=w, backstop_s=30, poll_s=0.2)
    assert got.killed and "repeating" in got.reason and got.elapsed_s < 4


def test_the_wait_after_the_kill_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(stallwatch, "_kill_group", lambda proc, **kw: None)
    monkeypatch.setattr(stallwatch, "WAIT_AFTER_KILL_S", 0.2)
    got = run_watched(["x"], cwd=tmp_path, env={}, log_path=tmp_path / "log", watcher=StallWatcher(),
                      backstop_s=0, poll_s=0.01, popen=lambda *a, **k: _StuckProc())
    assert got.killed and got.returncode is None and got.reason.startswith("backstop")


def test_a_child_that_ended_before_the_kill_is_not_reported_as_killed(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(stallwatch, "_kill_group", lambda proc, **kw: killed.append(proc))
    got = run_watched(["x"], cwd=tmp_path, env={}, log_path=tmp_path / "log", watcher=StallWatcher(),
                      backstop_s=0, poll_s=0.01, popen=lambda *a, **k: _ExitedProc())
    assert not got.killed and got.reason is None and got.returncode == 0 and not killed


def test_the_repeat_count_is_exact():
    w = StallWatcher(repeat_lines=3)
    assert w.feed_lines(["a", "b", "b"]) is None          # b has run twice
    assert "repeating" in w.feed_lines(["b"])             # the third b fires


# ---------------------------------------------------------------------------
# Task 3b: SIGTERM the group first so a wrapper (gozer run) can release its lease, SIGKILL after a grace.
# ---------------------------------------------------------------------------
import signal as _signal


class _ReadyWatcher(StallWatcher):
    """Reports a stall as soon as the child has printed 'ready', which a child prints after its signal setup."""

    def __init__(self, reason="stalled: test"):
        super().__init__(quiet_s=1000)
        self._reason = reason

    def feed_lines(self, lines):
        return self._reason if any(line.strip() == "ready" for line in lines) else None


_STARTED: list = []


def _tracked_popen(*a, **k):
    proc = subprocess.Popen(*a, **k)
    _STARTED.append(proc)
    return proc


@pytest.fixture(autouse=True)
def _kill_every_started_child_by_pid():
    """Whatever a test started is killed by pid at teardown, so a regression cannot leave a 120 s sleeper behind."""
    _STARTED.clear()
    yield
    for proc in _STARTED:
        proc.kill()                          # skips a child that has been reaped, so a freed pid is never signalled
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _watch(tmp_path, cmd, grace, watcher=None, **kw):
    return run_watched(cmd, cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log", popen=_tracked_popen,
                       watcher=watcher or _ReadyWatcher(), backstop_s=60, poll_s=0.1, term_grace_s=grace, **kw)


def _dead_or_zombie(pid):
    """A zombie has been killed and only waits for its parent to reap it, which an init that does not reap never does."""
    if not _alive(pid):
        return True
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return True
    return text[text.rfind(")") + 2:].split()[0] == "Z"


def _wait_gone(pid):
    for _ in range(40):          # the kernel needs a moment to reap a killed orphan; the loop checks the pid itself
        if _dead_or_zombie(pid):
            return True
        time.sleep(0.05)
    return False


def _py(code):
    return [sys.executable, "-u", "-c", code]


def test_a_child_that_exits_on_sigterm_is_reported_killed_with_return_code_minus_15(tmp_path):
    got = _watch(tmp_path, ["sh", "-c", "echo ready; exec sleep 120"], 1.0)
    assert got.killed and got.reason == "stalled: test" and got.returncode == -_signal.SIGTERM


def test_a_child_that_traps_sigterm_gets_to_run_its_cleanup(tmp_path):
    marker = tmp_path / "released"
    code = ("import signal, sys, time\n"
            f"def done(*a):\n open({str(marker)!r}, 'w').write('x'); sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, done)\nprint('ready')\ntime.sleep(120)")
    got = _watch(tmp_path, _py(code), 1.0)
    assert got.killed and got.returncode == 0 and marker.exists()


def test_a_leader_that_ignores_sigterm_is_killed_after_the_grace(tmp_path):
    code = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nprint('ready')\ntime.sleep(120)"
    got = _watch(tmp_path, _py(code), 0.5)
    assert got.killed and got.returncode == -_signal.SIGKILL
    assert 0.4 <= got.elapsed_s < 4          # waited for the grace, but not for the default five seconds and more


def test_a_grandchild_that_ignores_sigterm_is_killed_after_the_grace_even_when_the_leader_exits(tmp_path):
    pidfile = tmp_path / "grandchild.pid"
    grandchild = ("import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                  f"open({str(pidfile)!r}, 'w').write(str(__import__('os').getpid()))\ntime.sleep(120)")
    leader = ("import subprocess, sys, time, os\n"
              f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
              f"while not os.path.exists({str(pidfile)!r}) or not open({str(pidfile)!r}).read(): time.sleep(0.02)\n"
              "print('ready')\ntime.sleep(120)")
    pid = None
    try:
        got = _watch(tmp_path, _py(leader), 0.5)
        pid = int(pidfile.read_text())
        assert got.killed and got.returncode == -_signal.SIGTERM        # the leader went on SIGTERM
        assert got.elapsed_s >= 0.4                                     # the group was given its grace
        assert _wait_gone(pid)
    finally:
        if pid and _alive_with(pid, str(pidfile)):
            os.kill(pid, 9)


def test_a_repeating_child_is_stopped_with_sigterm_and_keeps_the_repeating_reason(tmp_path):
    code = "import time\nwhile True:\n print('same', flush=True); time.sleep(0.02)"
    got = run_watched(_py(code), cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=StallWatcher(quiet_s=1000, repeat_lines=5), backstop_s=60, poll_s=0.1, term_grace_s=1.0, popen=_tracked_popen)
    assert got.killed and got.reason.startswith("repeating") and got.returncode == -_signal.SIGTERM


def test_the_backstop_stops_a_child_with_sigterm_and_keeps_the_backstop_reason(tmp_path):
    got = run_watched(["sleep", "120"], cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      watcher=StallWatcher(quiet_s=1000), backstop_s=0.3, poll_s=0.1, term_grace_s=1.0, popen=_tracked_popen)
    assert got.killed and got.reason.startswith("backstop") and got.returncode == -_signal.SIGTERM


def test_the_cleanup_after_an_error_also_lets_the_child_run_its_sigterm_cleanup(tmp_path):
    marker = tmp_path / "released"
    code = ("import signal, sys, time\n"
            f"def done(*a):\n open({str(marker)!r}, 'w').write('x'); sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, done)\nprint('ready')\ntime.sleep(120)")

    class Boom(StallWatcher):
        def feed_lines(self, lines):
            if any(line.strip() == "ready" for line in lines):
                raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_watched(_py(code), cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                    watcher=Boom(), backstop_s=60, poll_s=0.1, term_grace_s=1.0, popen=_tracked_popen)
    assert marker.exists()


def test_run_watched_hands_term_grace_s_to_the_kill(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(stallwatch, "_kill_group", lambda proc, **kw: seen.append(kw))
    monkeypatch.setattr(stallwatch, "WAIT_AFTER_KILL_S", 0.1)
    run_watched(["x"], cwd=tmp_path, env={}, log_path=tmp_path / "log", watcher=StallWatcher(),
                backstop_s=0, poll_s=0.01, term_grace_s=0.7, popen=lambda *a, **k: _StuckProc())
    assert seen == [{"term_grace_s": 0.7}]


def test_killing_a_child_that_already_exited_is_quiet(tmp_path):
    proc = subprocess.Popen(["true"], start_new_session=True)
    proc.wait()
    stallwatch._kill_group(proc, term_grace_s=0.3)


# ---------------------------------------------------------------------------
# Safety guard: os.killpg(0, sig) or a group that is not the child's own would signal the test runner itself.
# ---------------------------------------------------------------------------
_REAL_KILLPG = os.killpg
_KILLPG_CALLS: list = []


@pytest.fixture(autouse=True)
def _never_signal_a_foreign_group(monkeypatch):
    """Every os.killpg call in this file goes through here. A call for pgid <= 1 or for pytest's own group fails the
    test before any signal is sent. Allowed calls are recorded and passed on."""
    _KILLPG_CALLS.clear()

    def guarded(pgid, sig):
        _KILLPG_CALLS.append((pgid, sig))
        if pgid is None or pgid <= 1 or pgid == os.getpgrp():
            raise AssertionError(f"os.killpg({pgid!r}, {sig}) would signal a group that is not the child's own")
        return _REAL_KILLPG(pgid, sig)
    monkeypatch.setattr(os, "killpg", guarded)


class _PidProc:
    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        return None


@pytest.mark.parametrize("pid", [None, 0, 1, -5])
def test_kill_group_sends_no_signal_for_a_pid_that_cannot_be_a_child_group(pid):
    stallwatch._kill_group(_PidProc(pid), term_grace_s=0.3)
    assert _KILLPG_CALLS == []


def test_kill_group_sends_no_signal_to_the_callers_own_group():
    stallwatch._kill_group(_PidProc(os.getpgrp()), term_grace_s=0.3)
    assert _KILLPG_CALLS == []


def test_kill_group_sends_no_signal_for_a_live_process_that_does_not_lead_its_own_group():
    p = subprocess.Popen(["sleep", "120"])            # no new session: its group is ours
    try:
        assert os.getpgid(p.pid) != p.pid
        stallwatch._kill_group(_PidProc(p.pid), term_grace_s=0.3)
        assert _KILLPG_CALLS == []
        assert p.poll() is None and _alive(p.pid)
    finally:
        p.kill()
        p.wait()


def test_group_gone_never_probes_a_foreign_group():
    for pgid in (None, 0, 1, os.getpgrp()):
        assert stallwatch._group_gone(pgid) is True
        assert _KILLPG_CALLS == []
    assert _KILLPG_CALLS == []


# ---------------------------------------------------------------------------
# Fully faked group: the guards say yes, os.killpg is a script, so no real signal can be sent.
# ---------------------------------------------------------------------------
class _FakeGroup:
    """Scripts os.killpg for pid 4242. `alive` is how many signal-0 probes still say the group exists."""

    def __init__(self, monkeypatch, *, alive=10**9, term=None, kill=None):
        self.calls, self.alive, self.term, self.kill = [], alive, term, kill
        monkeypatch.setattr(stallwatch, "_own_group", lambda pgid: True)
        monkeypatch.setattr(stallwatch, "_signalable", lambda pgid: True)
        monkeypatch.setattr(os, "killpg", self._killpg)

    def _killpg(self, pgid, sig):
        assert pgid == 4242, pgid
        self.calls.append(sig)
        if sig == 0:
            if self.alive <= 0:
                raise ProcessLookupError()
            self.alive -= 1
        elif sig == _signal.SIGTERM and self.term:
            raise self.term()
        elif sig == _signal.SIGKILL and self.kill:
            raise self.kill()

    def sent(self, sig):
        return self.calls.count(sig)


class _Fake4242:
    pid = 4242

    def __init__(self, poll=None):
        self._poll = poll

    def poll(self):
        if self._poll:
            return self._poll()
        return None


@pytest.mark.parametrize("exc", [ProcessLookupError, PermissionError])
def test_a_sigterm_that_finds_no_group_is_swallowed_and_nothing_else_is_sent(monkeypatch, exc):
    g = _FakeGroup(monkeypatch, term=exc)
    t = time.monotonic()
    stallwatch._kill_group(_Fake4242(), term_grace_s=5.0)
    assert g.calls == [_signal.SIGTERM] and time.monotonic() - t < 1.0


@pytest.mark.parametrize("exc", [ProcessLookupError, PermissionError])
def test_a_sigkill_that_finds_no_group_is_swallowed(monkeypatch, exc):
    g = _FakeGroup(monkeypatch, kill=exc)
    stallwatch._kill_group(_Fake4242(), term_grace_s=0.1)
    assert g.sent(_signal.SIGTERM) == 1 and g.sent(_signal.SIGKILL) == 1


def test_a_group_that_vanishes_during_the_grace_ends_the_wait_without_a_sigkill(monkeypatch):
    g = _FakeGroup(monkeypatch, alive=2)
    t = time.monotonic()
    stallwatch._kill_group(_Fake4242(), term_grace_s=5.0)
    assert g.sent(_signal.SIGKILL) == 0 and time.monotonic() - t < 1.0


def test_the_group_is_probed_again_just_before_the_sigkill(monkeypatch):
    g = _FakeGroup(monkeypatch, alive=1)          # alive for the first probe, gone when the grace has run out
    stallwatch._kill_group(_Fake4242(), term_grace_s=0.0)
    assert g.sent(_signal.SIGKILL) == 0 and g.sent(0) == 2


def test_a_second_interrupt_during_the_grace_still_sigkills_once_and_propagates(monkeypatch):
    g = _FakeGroup(monkeypatch)

    def interrupted(_s):
        raise KeyboardInterrupt()
    monkeypatch.setattr(stallwatch.time, "sleep", interrupted)
    with pytest.raises(KeyboardInterrupt):
        stallwatch._kill_group(_Fake4242(), term_grace_s=5.0)
    assert g.sent(_signal.SIGTERM) == 1 and g.sent(_signal.SIGKILL) == 1


def test_an_interrupt_in_the_leader_check_during_the_grace_still_sigkills_once_and_propagates(monkeypatch):
    g = _FakeGroup(monkeypatch)

    def interrupted():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        stallwatch._kill_group(_Fake4242(poll=interrupted), term_grace_s=5.0)
    assert g.sent(_signal.SIGTERM) == 1 and g.sent(_signal.SIGKILL) == 1


# I7: output with no newline must not grow the watcher's memory without bound.

# Every child here writes a bounded amount and has a file-size limit of 16 MiB, so a regression cannot fill the disk.
FSIZE_LIMIT = "import resource; resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))\n"
NO_NEWLINE_5MIB = (FSIZE_LIMIT + "import sys, time\n"
                   "chunk = 'x' * 65536\n"
                   "for _ in range(80):\n"
                   "    sys.stdout.write(chunk); sys.stdout.flush(); time.sleep(0.01)\n"
                   "time.sleep(0.5)\n")


def test_five_mib_without_a_newline_keeps_the_pending_buffer_bounded(tmp_path):
    from orchard import stallwatch
    got = run_watched([sys.executable, "-c", NO_NEWLINE_5MIB], cwd=tmp_path, env=dict(os.environ),
                      log_path=tmp_path / "log", popen=_tracked_popen, poll_s=0.05, backstop_s=60,
                      watcher=StallWatcher(quiet_s=1000, repeat_lines=10 ** 9))
    assert got.returncode == 0 and not got.killed
    assert (tmp_path / "log").stat().st_size == 80 * 65536
    assert 0 < got.max_pending_bytes <= stallwatch.PENDING_MAX_BYTES == 64 * 1024


def test_a_child_that_prints_4_mib_of_one_character_without_a_newline_is_stopped_as_repeating(tmp_path):
    """The cut pieces of a long line are fed to the repeat detector as lines, so identical output with no newline is
    caught by the repeat rule."""
    code = (FSIZE_LIMIT + "import sys, time\n"
            "for _ in range(1024):\n"                       # 4 MiB in all, then it waits
            "    sys.stdout.write('x' * 4096); sys.stdout.flush()\n"
            "time.sleep(30)\n")
    got = run_watched([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      popen=_tracked_popen, poll_s=0.05, backstop_s=10,
                      watcher=StallWatcher(quiet_s=1000, repeat_lines=3), term_grace_s=1.0)
    assert got.killed and got.reason.startswith("repeating:"), got.reason
    assert (tmp_path / "log").stat().st_size <= 4 << 20


# I8: the watcher reports the longest quiet stretch and the longest repeat it saw, so the limits can be set from a
# measurement.

def test_the_watcher_keeps_the_longest_quiet_stretch_and_the_longest_repeat():
    w = StallWatcher(quiet_s=100, min_cpu_ticks=2, repeat_lines=50)
    w.observe(0.0, 10, 5)
    w.observe(7.5, 10, 5)                                  # 7.5 s quiet so far
    w.observe(9.0, 11, 5)                                  # output: the stretch ended at 9 s
    w.observe(12.0, 11, 5)
    assert w.max_quiet_s == pytest.approx(9.0)
    w.feed_lines(["a", "a", "a", "b", "a", "a"])
    assert w.max_repeat == 3


def test_run_watched_returns_the_watchers_longest_quiet_stretch_and_repeat(tmp_path):
    code = "import sys, time\nfor _ in range(5):\n    print('same', flush=True)\ntime.sleep(1.5)\n"
    got = run_watched([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ), log_path=tmp_path / "log",
                      popen=_tracked_popen, poll_s=0.1, backstop_s=30, watcher=StallWatcher(quiet_s=100))
    assert got.returncode == 0 and got.max_repeat == 5
    assert 0.5 <= got.max_quiet_s <= 5.0, got.max_quiet_s
