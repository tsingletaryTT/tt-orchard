# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/bounded_run.py: run a helper with a deadline and a cap on each output stream, in a session of its own.

PR 422 review, round 3: the gate collected the reference-interpreter helper's output after it had finished, so a helper that wrote without end
filled the supervisor's memory first, and a wrapper that started a child left the child running after the deadline. The tests here use real
child processes (a shell and `sleep`), because the claims are about processes."""
import os
import sys
import textwrap
import time

import pytest

from orchard import bounded_run as br


def py(tmp_path, body, name="child.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return [sys.executable, "-I", str(path)]


def gone(pid, wait_s=8.0):
    """True when no process has this pid any more (a zombie of ours counts as there)."""
    end = time.time() + wait_s
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_a_helper_that_answers_returns_its_output_and_exit_code(tmp_path):
    res = br.run(py(tmp_path, "import sys; sys.stdout.write(sys.stdin.read().upper()); sys.stderr.write('e')"), "abc",
                 timeout_s=20, out_max=1000, err_max=1000)
    assert (res.problem, res.returncode, res.stdout, res.stderr) == (None, 0, "ABC", "e")


def test_stdout_over_the_cap_stops_the_helper_and_holds_only_the_cap(tmp_path):
    cmd = py(tmp_path, "import sys\nwhile True:\n    sys.stdout.write('x' * 65536); sys.stdout.flush()\n")
    t0 = time.time()
    res = br.run(cmd, "", timeout_s=60, out_max=100_000, err_max=1000)
    assert res.problem == "stdout" and time.time() - t0 < 30
    assert len(res.stdout) <= 100_000 + br.CHUNK


def test_stderr_over_the_cap_also_stops_the_helper(tmp_path):
    cmd = py(tmp_path, "import sys\nwhile True:\n    sys.stderr.write('e' * 65536); sys.stderr.flush()\n")
    res = br.run(cmd, "", timeout_s=60, out_max=1000, err_max=50_000)
    assert res.problem == "stderr"
    assert len(res.stderr) <= 50_000 + br.CHUNK


def test_a_helper_that_never_finishes_is_stopped_at_the_deadline(tmp_path):
    res = br.run(py(tmp_path, "import time; time.sleep(300)"), "", timeout_s=1.5, out_max=1000, err_max=1000)
    assert res.problem == "timeout"


def test_the_whole_process_group_is_stopped_when_a_wrapper_started_a_child(tmp_path):
    """A wrapper that launches Python leaves the Python process behind when only the wrapper is killed."""
    pid_file = tmp_path / "child.pid"
    cmd = ["bash", "-c", f"sleep 300 & echo $! > {pid_file}; wait"]
    res = br.run(cmd, "", timeout_s=1.5, out_max=1000, err_max=1000)
    assert res.problem == "timeout"
    assert gone(int(pid_file.read_text()))


def test_the_group_is_stopped_when_the_output_cap_is_what_ended_the_run(tmp_path):
    pid_file = tmp_path / "child.pid"
    cmd = ["bash", "-c", f"sleep 300 & echo $! > {pid_file}; while true; do head -c 65536 /dev/zero | tr '\\0' x; done"]
    res = br.run(cmd, "", timeout_s=60, out_max=100_000, err_max=1000)
    assert res.problem == "stdout"
    assert gone(int(pid_file.read_text()))


def test_the_group_is_stopped_when_the_caller_is_interrupted(tmp_path, monkeypatch):
    """Cancellation (an exception out of the wait) kills the group and reaps the child before it propagates."""
    pid_file = tmp_path / "child.pid"
    cmd = ["bash", "-c", f"sleep 300 & echo $! > {pid_file}; wait"]
    real = br._wait_for_end

    def boom(*a, **k):
        for _ in range(50):
            if pid_file.exists() and pid_file.read_text().strip():
                break
            time.sleep(0.1)
        raise KeyboardInterrupt
    monkeypatch.setattr(br, "_wait_for_end", boom)
    with pytest.raises(KeyboardInterrupt):
        br.run(cmd, "", timeout_s=60, out_max=1000, err_max=1000)
    monkeypatch.setattr(br, "_wait_for_end", real)
    assert gone(int(pid_file.read_text()))


def test_a_program_that_cannot_start_is_a_problem_and_not_an_exception(tmp_path):
    res = br.run([str(tmp_path / "no-such-python")], "", timeout_s=5, out_max=10, err_max=10)
    assert res.problem == "start" and res.detail


def test_a_large_request_does_not_deadlock_a_helper_that_answers_first(tmp_path):
    """The request goes in from a thread of its own: a helper that writes before it reads cannot block the caller."""
    cmd = py(tmp_path, "import sys; sys.stdout.write('done'); sys.stdout.flush(); sys.stdin.read()")
    res = br.run(cmd, "r" * (4 << 20), timeout_s=30, out_max=1000, err_max=1000)
    assert res.problem is None and res.stdout == "done"
