# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 390: a command that spawn_checked kills (timeout or interrupt) leaves no helper behind, even one that started its own session.

The command's process group is killed with SIGKILL, which reaches everything that stayed in the group. A helper started with
`start_new_session=True` (or a double fork with `setsid`) is in another group, so the group kill missed it and it kept running, possibly with a
chip open after the lease went back. Every spawn now carries a tag in its environment, and the kill is followed by a sweep for the processes that
carry it. No chip is used: the helpers here only sleep."""
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from orchard import agent
from orchard.agent import spawn_checked

HELPER_SESSION = textwrap.dedent('''
    import os, subprocess, sys, time
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    open(sys.argv[1], "w").write(str(helper.pid))
    time.sleep(120)
''')

HELPER_DOUBLE_FORK = textwrap.dedent('''
    import os, sys, time
    if os.fork() == 0:
        os.setsid()
        if os.fork() == 0:
            open(sys.argv[1], "w").write(str(os.getpid()))
            time.sleep(120)
        os._exit(0)
    time.sleep(120)
''')

HELPER_IGNORES_TERM = textwrap.dedent('''
    import signal, subprocess, sys, time
    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"
    helper = subprocess.Popen([sys.executable, "-c", code], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    open(sys.argv[1], "w").write(str(helper.pid))
    time.sleep(120)
''')

HELPER_LEFT_BY_A_CLEAN_EXIT = textwrap.dedent('''
    import subprocess, sys
    helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    open(sys.argv[1], "w").write(str(helper.pid))
''')

HELPER_CLEAN = textwrap.dedent('''
    import sys
    open(sys.argv[1], "w").write("done")
''')


def alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def wait_for(path: Path, timeout: float = 20.0) -> str:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            text = path.read_text()
            if text:
                return text
        except OSError:
            pass
        time.sleep(0.05)
    raise AssertionError(f"{path} was not written")


@pytest.fixture
def rig(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    pids = []
    yield run, tmp_path, pids
    for pid in pids:                              # whatever a test left running, so a failure does not leak a sleeper
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def settle(pid, seconds=5.0):
    end = time.monotonic() + seconds
    while alive(pid) and time.monotonic() < end:
        time.sleep(0.05)
    return alive(pid)


@pytest.mark.parametrize("source", [HELPER_SESSION, HELPER_DOUBLE_FORK], ids=["start_new_session", "double_fork_setsid"])
def test_a_helper_in_its_own_session_is_dead_after_the_command_times_out(rig, source):
    run, tmp, pids = rig
    script = run / "port_check.py"
    script.write_text(source)
    pidfile = run / "helper.pid"
    # a helper's pid is read while the command runs, so the timeout is long enough for the script to write it and short enough to end the test quickly
    env = {"PATH": os.environ["PATH"]}
    with open(tmp / "out.txt", "wb") as out:
        code, timed_out = spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, env, 4.0, out)
    helper = int(pidfile.read_text())
    pids.append(helper)
    assert timed_out is True and code is None
    assert not settle(helper), f"the helper {helper} outlived the killed command"


def test_a_helper_that_ignores_sigterm_is_killed_too(rig):
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_IGNORES_TERM)
    pidfile = run / "helper.pid"
    with open(tmp / "out.txt", "wb") as out:
        code, timed_out = spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 4.0, out)
    helper = int(pidfile.read_text())
    pids.append(helper)
    assert timed_out is True
    assert not settle(helper)


def test_a_helper_is_dead_after_an_interrupt_of_the_wait_and_the_interrupt_still_propagates(rig, monkeypatch):
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    real_wait = agent.subprocess.Popen.wait

    calls = []

    def interrupted(self, timeout=None):
        if calls:                                   # _kill_session waits for the killed command through the same method
            return real_wait(self, timeout)
        calls.append(1)
        wait_for(pidfile)                           # the helper exists before the supervisor is interrupted
        raise KeyboardInterrupt
    monkeypatch.setattr(agent.subprocess.Popen, "wait", interrupted)
    with open(tmp / "out.txt", "wb") as out, pytest.raises(KeyboardInterrupt):
        spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 60.0, out)
    monkeypatch.setattr(agent.subprocess.Popen, "wait", real_wait)
    helper = int(pidfile.read_text())
    pids.append(helper)
    assert not settle(helper)


def test_the_sweep_names_what_it_stopped_in_the_commands_output(rig):
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    with open(tmp / "out.txt", "wb") as out:
        spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 4.0, out)
    helper = int(pidfile.read_text())
    pids.append(helper)
    text = (tmp / "out.txt").read_text()
    assert f"pid {helper}" in text and "outlived the command" in text


def test_a_command_that_exits_normally_is_not_swept_and_leaves_no_tag_in_the_output(rig):
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_CLEAN)
    pidfile = run / "helper.pid"
    with open(tmp / "out.txt", "wb") as out:
        code, timed_out = spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 30.0, out)
    assert (code, timed_out) == (0, False)
    assert (tmp / "out.txt").read_text() == ""


def test_a_helper_left_by_a_command_that_exited_normally_is_not_stopped_here(rig):
    """Only a kill is swept. A server that a graft trial's helper leaves behind is stopped, and recorded with its identity, by the supervisor's own sweep."""
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_LEFT_BY_A_CLEAN_EXIT)
    pidfile = run / "helper.pid"
    with open(tmp / "out.txt", "wb") as out:
        code, timed_out = spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 30.0, out)
    helper = int(pidfile.read_text())
    pids.append(helper)
    assert (code, timed_out) == (0, False)
    assert alive(helper) and (tmp / "out.txt").read_text() == ""


def test_a_process_of_another_run_is_never_swept(rig):
    """A sleeper that carries no tag (it was started by something else) stays alive while a command beside it is killed."""
    import subprocess
    run, tmp, pids = rig
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    pids.append(bystander.pid)
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    with open(tmp / "out.txt", "wb") as out:
        spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 4.0, out)
    pids.append(int(pidfile.read_text()))
    assert alive(bystander.pid)
    bystander.kill()
    bystander.wait()


def test_two_spawns_get_different_tags(monkeypatch):
    seen = []
    real = agent.subprocess.Popen

    class Spy(real):
        def __init__(self, argv, **kw):
            seen.append(kw["env"].get(agent.SPAWN_TAG_VAR))
            super().__init__(argv, **kw)
    monkeypatch.setattr(agent.subprocess, "Popen", Spy)
    for _ in range(2):
        with open(os.devnull, "wb") as out:
            spawn_checked("true", Path.cwd(), {"PATH": os.environ["PATH"]}, 30.0, out)
    assert len(seen) == 2 and all(seen) and seen[0] != seen[1]


def test_the_callers_environment_is_not_changed_by_the_tag(rig):
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_CLEAN)
    env = {"PATH": os.environ["PATH"]}
    with open(tmp / "out.txt", "wb") as out:
        spawn_checked(f"{sys.executable} port_check.py {run / 'helper.pid'}", run, env, 30.0, out)
    assert agent.SPAWN_TAG_VAR not in env


def test_a_sweep_that_fails_does_not_replace_the_result_or_the_interrupt(rig, monkeypatch):
    from orchard import kernelprobe

    def broken(*a, **k):
        raise RuntimeError("the sweep failed")
    monkeypatch.setattr(kernelprobe, "sweep_strays", broken)
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    with open(tmp / "out.txt", "wb") as out:
        code, timed_out = spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 4.0, out)
    pids.append(int(pidfile.read_text()))
    assert (code, timed_out) == (None, True)


# ---- review of PR 431, round 1 --------------------------------------------------------------------------------------------------------------------
# The supervisor's SIGINT and SIGTERM handlers raise a BaseException. One that arrives while the cleanup runs must not skip the group kill or the
# sweep, and the interrupt that was already on its way is the one that propagates.


class FirstInterrupt(BaseException):
    pass


class SecondInterrupt(BaseException):
    pass


def kill_wait_interrupted(monkeypatch, pidfile, *, first=True):
    """The wait of spawn_checked raises FirstInterrupt (the supervisor's handler ran); _kill_session's own wait then raises SecondInterrupt (a
    second signal), once. Later waits run for real."""
    real_wait = agent.subprocess.Popen.wait
    calls = []

    def wait(self, timeout=None):
        calls.append(1)
        WAITED.append(self)
        if len(calls) == 1 and first:
            wait_for(pidfile)                      # the helper exists before the supervisor is interrupted
            raise FirstInterrupt
        if len(calls) == (2 if first else 1) and timeout is None:
            raise SecondInterrupt
        return real_wait(self, timeout)
    monkeypatch.setattr(agent.subprocess.Popen, "wait", wait)
    return real_wait


WAITED: list = []          # the Popen objects whose wait kill_wait_interrupted saw


def test_a_second_interrupt_during_the_group_kill_wait_does_not_skip_the_sweep_or_replace_the_first(rig, monkeypatch):
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    WAITED.clear()
    real_wait = kill_wait_interrupted(monkeypatch, pidfile)
    with open(tmp / "out.txt", "wb") as out, pytest.raises(FirstInterrupt):
        spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 60.0, out)
    monkeypatch.setattr(agent.subprocess.Popen, "wait", real_wait)
    helper = int(pidfile.read_text())
    pids.append(helper)
    assert not settle(helper), "the interrupt of the group kill's wait skipped the sweep"
    assert WAITED and WAITED[0].returncode is not None, "the interrupted wait was not run again, so the killed command was never reaped"


@pytest.fixture
def stop_signal_raises(monkeypatch):
    """SIGTERM raises SecondInterrupt in this process, as the supervisor's handler raises. The old handler comes back after the test."""
    def handler(signum, frame):
        raise SecondInterrupt
    old = signal.signal(signal.SIGTERM, handler)
    yield
    signal.signal(signal.SIGTERM, old)


def signal_during_the_sweep(monkeypatch):
    """The first scan of the sweep runs after a SIGTERM to this process (one arrives while the sweep works)."""
    from orchard import kernelprobe
    real = kernelprobe.sweep_strays
    sent = []

    def sweep(*a, **k):
        if not sent:
            sent.append(1)
            os.kill(os.getpid(), signal.SIGTERM)
        return real(*a, **k)
    monkeypatch.setattr(kernelprobe, "sweep_strays", sweep)


def test_a_stop_signal_during_the_sweep_is_deferred_until_the_helpers_are_dead_and_the_first_interrupt_propagates(rig, monkeypatch, stop_signal_raises):
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    real_wait = agent.subprocess.Popen.wait
    calls = []

    def wait(self, timeout=None):
        if not calls:
            calls.append(1)
            wait_for(pidfile)
            raise FirstInterrupt
        return real_wait(self, timeout)
    monkeypatch.setattr(agent.subprocess.Popen, "wait", wait)
    signal_during_the_sweep(monkeypatch)
    with open(tmp / "out.txt", "wb") as out, pytest.raises(FirstInterrupt):
        spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 60.0, out)
    monkeypatch.setattr(agent.subprocess.Popen, "wait", real_wait)
    helper = int(pidfile.read_text())
    pids.append(helper)
    assert not settle(helper), "a stop signal during the sweep skipped the kill"


def test_a_stop_signal_during_the_sweep_after_a_timeout_is_raised_once_the_helpers_are_dead(rig, monkeypatch, stop_signal_raises):
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    signal_during_the_sweep(monkeypatch)
    with open(tmp / "out.txt", "wb") as out, pytest.raises(SecondInterrupt):
        spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 4.0, out)
    helper = int(pidfile.read_text())
    pids.append(helper)
    assert not settle(helper), "a stop signal during the sweep skipped the kill"


def test_a_stop_signal_at_every_call_of_the_sweep_cannot_use_up_the_retries(rig, monkeypatch, stop_signal_raises):
    """Retrying alone does not cover a signal that keeps coming: with the signals held back the sweep runs once, uninterrupted."""
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    from orchard import kernelprobe
    real = kernelprobe.sweep_strays

    def sweep(*a, **k):
        os.kill(os.getpid(), signal.SIGTERM)            # every call is interrupted by a stop signal
        return real(*a, **k)
    monkeypatch.setattr(kernelprobe, "sweep_strays", sweep)
    with open(tmp / "out.txt", "wb") as out, pytest.raises(SecondInterrupt):
        spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 4.0, out)
    helper = int(pidfile.read_text())
    pids.append(helper)
    assert not settle(helper), "a stop signal at every call used up the retries and the sweep never ran"


def test_the_stop_signal_handlers_are_put_back_after_the_cleanup(rig, stop_signal_raises):
    run, tmp, pids = rig
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    (run / "port_check.py").write_text(HELPER_CLEAN)
    with open(tmp / "out.txt", "wb") as out:
        spawn_checked(f"{sys.executable} port_check.py {run / 'x'}", run, {"PATH": os.environ["PATH"]}, 4.0, out)
    assert {s: signal.getsignal(s) for s in before} == before                  # an ordinary command runs no cleanup
    (run / "port_check.py").write_text(HELPER_SESSION)
    with open(tmp / "out.txt", "wb") as out:
        spawn_checked(f"{sys.executable} port_check.py {run / 'h.pid'}", run, {"PATH": os.environ["PATH"]}, 3.0, out)
    pids.append(int((run / "h.pid").read_text()))
    assert {s: signal.getsignal(s) for s in before} == before


def test_a_stop_signal_that_arrives_in_another_thread_is_deferred_too(rig, monkeypatch, stop_signal_raises):
    """A signal reaches any thread that does not block it, and Python runs the handler in the main thread: a mask of the main thread would not
    have held it back. Here a second thread sends it during the sweep."""
    import threading
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    from orchard import kernelprobe
    real = kernelprobe.sweep_strays

    def sweep(*a, **k):
        t = threading.Thread(target=lambda: os.kill(os.getpid(), signal.SIGTERM))
        t.start()
        t.join()
        time.sleep(0.2)                        # the handler gets its chance to run before the real sweep
        return real(*a, **k)
    monkeypatch.setattr(kernelprobe, "sweep_strays", sweep)
    with open(tmp / "out.txt", "wb") as out, pytest.raises(SecondInterrupt):
        spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 4.0, out)
    helper = int(pidfile.read_text())
    pids.append(helper)
    assert not settle(helper)


# A helper that keeps forking while it is being stopped must not leave a child the sweep never scanned.

FORKER = textwrap.dedent('''
    import os, signal, sys, time
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    log = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    end = time.time() + 20
    while time.time() < end:
        pid = os.fork()
        if pid == 0:
            os.setsid()
            time.sleep(120)
            os._exit(0)
        os.write(log, f"{pid}\\n".encode())
        time.sleep(0.01)
''')


def test_a_helper_that_keeps_forking_leaves_no_child_the_sweep_did_not_scan(tmp_path):
    from orchard import kernelprobe
    tag = str(tmp_path / ".spawn-forker")
    log = tmp_path / "children.txt"
    forker = subprocess.Popen([sys.executable, "-c", FORKER, str(log)], env={**os.environ, agent.SPAWN_TAG_VAR: tag}, start_new_session=True,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    children = []
    try:
        wait_for(log)
        time.sleep(0.2)

        def slow(real):
            def send(target, sig):
                if sig == signal.SIGKILL:
                    time.sleep(0.25)          # a slow delivery: the window in which an unfrozen forker starts more children
                return real(target, sig)
            return send
        kernelprobe.sweep_strays(tag, var=agent.SPAWN_TAG_VAR, grace_s=0.3, kill=slow(os.kill), killpg=slow(os.killpg))
        children = [int(x) for x in log.read_text().split()]
        assert len(children) >= 3
        assert not [p for p in children + [forker.pid] if alive(p)], "a child started during the sweep was never scanned"
    finally:
        for p in children + [forker.pid]:
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
        forker.wait()


# What the sweep stopped is handed to the code that called spawn_checked, with the identity each process had before it was signalled, so the
# structured audit paths (the ledger, the attempt's `swept` list) hold it. A later scan by the caller cannot find a process this sweep removed.


def test_the_helpers_the_sweep_stopped_are_collected_with_their_identity(rig):
    run, tmp, pids = rig
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    with agent.collect_survivors() as records, open(tmp / "out.txt", "wb") as out:
        spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 4.0, out)
    helper = int(pidfile.read_text())
    pids.append(helper)
    assert [r["pid"] for r in records] == [helper]
    rec = records[0]
    assert set(rec) >= {"pid", "cmdline", "stopped", "boot_id", "start"} and rec["stopped"] is True
    assert isinstance(rec["start"], int) and rec["start"] > 0 and rec["boot_id"]


def test_a_helper_that_is_still_running_is_collected_as_not_stopped(rig, monkeypatch):
    run, tmp, pids = rig
    from orchard import kernelprobe
    seen = {}

    def fake(tag, **kw):
        seen.update(kw)
        return [{"pid": 4242, "cmdline": "srv", "stopped": False, "boot_id": "b", "start": 7}]
    monkeypatch.setattr(kernelprobe, "sweep_strays", fake)
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    with agent.collect_survivors() as records, open(tmp / "out.txt", "wb") as out:
        spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 3.0, out)     # times out, so the sweep runs
    pids.append(int(pidfile.read_text()))                  # the fake sweep stops nothing; the rig does
    assert seen.get("identity") is True and records == [{"pid": 4242, "cmdline": "srv", "stopped": False, "boot_id": "b", "start": 7}]
    assert "still running" in (tmp / "out.txt").read_text()


def test_the_sweep_asks_for_no_identity_when_no_collector_is_active(rig, monkeypatch):
    run, tmp, pids = rig
    from orchard import kernelprobe
    seen = {}
    monkeypatch.setattr(kernelprobe, "sweep_strays", lambda tag, **kw: seen.update(kw) or [])
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    with open(tmp / "out.txt", "wb") as out:
        spawn_checked(f"{sys.executable} port_check.py {pidfile}", run, {"PATH": os.environ["PATH"]}, 3.0, out)
    pids.append(int(pidfile.read_text()))
    assert "var" in seen and seen["identity"] is False      # the fake ran (the sweep was reached) and no identity was asked for


def test_the_agent_shell_keeps_the_helpers_it_stopped_for_the_attempt_record(rig):
    run, tmp, pids = rig
    stage = run / "stages" / "0"
    stage.mkdir(parents=True)
    (run / "port_check.py").write_text(HELPER_SESSION)
    pidfile = run / "helper.pid"
    tools = agent.Tools(run, stage, {"PATH": os.environ["PATH"]}, timeout=4.0)
    reply = tools.shell(f"{sys.executable} port_check.py {pidfile}")
    helper = int(pidfile.read_text())
    pids.append(helper)
    assert reply.startswith("killed after")
    assert [r["pid"] for r in tools.swept] == [helper] and tools.swept[0]["stopped"] is True
