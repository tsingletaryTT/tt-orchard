# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 390: a command that spawn_checked kills (timeout or interrupt) leaves no helper behind, even one that started its own session.

The command's process group is killed with SIGKILL, which reaches everything that stayed in the group. A helper started with
`start_new_session=True` (or a double fork with `setsid`) is in another group, so the group kill missed it and it kept running, possibly with a
chip open after the lease went back. Every spawn now carries a tag in its environment, and the kill is followed by a sweep for the processes that
carry it. No chip is used: the helpers here only sleep."""
import os
import signal
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
