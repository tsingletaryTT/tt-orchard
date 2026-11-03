# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/bounded_run.py: run a helper with a deadline and a cap on each output stream, in a session of its own.

PR 422 review, round 3: the gate collected the reference-interpreter helper's output after it had finished, so a helper that wrote without end
filled the supervisor's memory first, and a wrapper that started a child left the child running after the deadline. The tests use real child
processes (a shell, `sleep`, a Python loop that writes without end), because the claims are about processes.

Every scenario runs inside a second child, `limited()`, that has 2 GiB of address space and a 60 s deadline and is a session of its own, so
the code under test and the hostile writer are both out of the pytest process. A mutation that removes a cap or a deadline then turns the test
red within seconds (the limited child dies of MemoryError, or the deadline kills its group) and cannot take the machine down (the memory note
hostile-input-tests-need-a-limited-child; on 2026-11-03 an unlimited version of these tests did, when the stdout cap was removed)."""
import json
import os
import resource
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from orchard import bounded_run as br

REPO = str(Path(br.__file__).resolve().parents[1])
LIMIT_BYTES = 2 << 30
DEADLINE_S = 60

PRELUDE = f"""
import json, sys
sys.path.insert(0, {REPO!r})
from orchard import bounded_run as br
def show(res):
    print(json.dumps({{"problem": res.problem, "returncode": res.returncode, "stdout_len": len(res.stdout), "stderr_len": len(res.stderr),
                       "stdout": res.stdout[:200], "stderr": res.stderr[:200], "detail": res.detail}}), flush=True)
"""


def limited(code: str, deadline_s: float = DEADLINE_S) -> dict:
    """Run `code` (Python source; it prints one JSON line with show(res), or its own) in a child of this test that has LIMIT_BYTES of address
    space, is a session of its own, and is killed as a group at the deadline. A child that dies (MemoryError, a signal) or runs past the deadline
    fails the test with what it printed."""
    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (LIMIT_BYTES, LIMIT_BYTES))
    proc = subprocess.Popen([sys.executable, "-I", "-c", PRELUDE + textwrap.dedent(code)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, preexec_fn=limit, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=deadline_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        pytest.fail(f"the scenario ran past {deadline_s:.0f} s: a cap or a deadline in the code under test is missing")
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)         # whatever the scenario left behind
        except ProcessLookupError:
            pass
    if proc.returncode != 0 or not out.strip():
        pytest.fail(f"the scenario died (exit {proc.returncode}); stderr ends: {err.strip()[-400:]}")
    return json.loads(out.strip().splitlines()[-1])


def gone(pid, wait_s=8.0):
    """True when no process has this pid any more."""
    end = time.time() + wait_s
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_a_helper_that_answers_returns_its_output_and_exit_code():
    got = limited("""
        res = br.run([sys.executable, "-I", "-c", "import sys; sys.stdout.write(sys.stdin.read().upper()); sys.stderr.write('e')"], "abc",
                     timeout_s=20, out_max=1000, err_max=1000)
        show(res)
    """)
    assert (got["problem"], got["returncode"], got["stdout"], got["stderr"]) == (None, 0, "ABC", "e")


def test_stdout_over_the_cap_stops_the_helper_and_holds_only_the_cap():
    got = limited(f"""
        import time
        writer = "import sys\\nwhile True:\\n    sys.stdout.write('x' * 65536); sys.stdout.flush()\\n"
        t0 = time.time()
        res = br.run([sys.executable, "-I", "-c", writer], "", timeout_s=40, out_max=100000, err_max=1000)
        out = {{"elapsed": time.time() - t0}}
        show(res)
        print(json.dumps(out | {{"problem": res.problem, "stdout_len": len(res.stdout)}}), flush=True)
    """)
    assert got["problem"] == "stdout" and got["elapsed"] < 30
    assert got["stdout_len"] <= 100_000 + br.CHUNK


def test_stderr_over_the_cap_also_stops_the_helper():
    got = limited("""
        writer = "import sys\\nwhile True:\\n    sys.stderr.write('e' * 65536); sys.stderr.flush()\\n"
        res = br.run([sys.executable, "-I", "-c", writer], "", timeout_s=40, out_max=1000, err_max=50000)
        print(json.dumps({"problem": res.problem, "stderr_len": len(res.stderr)}), flush=True)
    """)
    assert got["problem"] == "stderr"
    assert got["stderr_len"] <= 50_000 + br.CHUNK


def test_a_helper_that_never_finishes_is_stopped_at_the_deadline():
    got = limited("""
        res = br.run([sys.executable, "-I", "-c", "import time; time.sleep(300)"], "", timeout_s=1.5, out_max=1000, err_max=1000)
        show(res)
    """)
    assert got["problem"] == "timeout"


def test_the_whole_process_group_is_stopped_when_a_wrapper_started_a_child(tmp_path):
    """A wrapper that launches Python leaves the Python process behind when only the wrapper is killed."""
    pid_file = tmp_path / "child.pid"
    got = limited(f"""
        res = br.run(["bash", "-c", "sleep 300 & echo $! > {pid_file}; wait"], "", timeout_s=1.5, out_max=1000, err_max=1000)
        show(res)
    """)
    assert got["problem"] == "timeout"
    assert gone(int(pid_file.read_text()))


def test_the_group_is_stopped_when_the_output_cap_is_what_ended_the_run(tmp_path):
    pid_file = tmp_path / "child.pid"
    got = limited(f"""
        script = "sleep 300 & echo $! > {pid_file}; while true; do head -c 65536 /dev/zero | tr '\\\\0' x; done"
        res = br.run(["bash", "-c", script], "", timeout_s=40, out_max=100000, err_max=1000)
        show(res)
    """)
    assert got["problem"] == "stdout"
    assert gone(int(pid_file.read_text()))


def test_the_group_is_stopped_when_the_caller_is_interrupted(tmp_path):
    """Cancellation (an exception out of the wait) kills the group and reaps the child before it propagates."""
    pid_file = tmp_path / "child.pid"
    got = limited(f"""
        import time
        def boom(*a, **k):
            for _ in range(100):
                try:
                    if open({str(pid_file)!r}).read().strip():
                        break
                except OSError:
                    pass
                time.sleep(0.1)
            raise KeyboardInterrupt
        br._wait_for_end = boom
        try:
            br.run(["bash", "-c", "sleep 300 & echo $! > {pid_file}; wait"], "", timeout_s=40, out_max=1000, err_max=1000)
            print(json.dumps({{"raised": False}}), flush=True)
        except KeyboardInterrupt:
            print(json.dumps({{"raised": True}}), flush=True)
    """)
    assert got["raised"] is True
    assert gone(int(pid_file.read_text()))


def test_a_program_that_cannot_start_is_a_problem_and_not_an_exception(tmp_path):
    got = limited(f"""
        res = br.run([{str(tmp_path / "no-such-python")!r}], "", timeout_s=5, out_max=10, err_max=10)
        show(res)
    """)
    assert got["problem"] == "start" and got["detail"]


def test_a_helper_that_never_reads_its_request_cannot_stop_the_deadline_from_being_checked():
    """A request bigger than a pipe, to a helper that never reads it: a write from the calling thread would block for good and the deadline
    would never be looked at. The request goes in from a thread of its own."""
    got = limited("""
        res = br.run([sys.executable, "-I", "-c", "import time; time.sleep(300)"], "r" * (4 << 20), timeout_s=1.5, out_max=1000, err_max=1000)
        show(res)
    """)
    assert got["problem"] == "timeout"


def test_a_helper_that_ignores_the_first_signal_is_killed_after_the_grace_period(tmp_path):
    """SIGTERM first, SIGKILL after the grace period: a wrapper that traps SIGTERM and its child are both gone when run() returns."""
    pid_file = tmp_path / "pids"
    got = limited(f"""
        br.GRACE_S = 1.0
        script = "trap '' TERM; sleep 300 & echo $$ $! > {pid_file}; while true; do sleep 1; done"
        res = br.run(["bash", "-c", script], "", timeout_s=1.0, out_max=1000, err_max=1000)
        show(res)
    """)
    assert got["problem"] == "timeout"
    wrapper, child = (int(x) for x in pid_file.read_text().split())
    assert gone(wrapper) and gone(child)


def test_the_limited_child_is_what_stops_a_scenario_that_has_lost_its_cap():
    """The guard of the guard: a scenario that allocates without end dies at the address-space limit instead of growing, and the test says so."""
    with pytest.raises(pytest.fail.Exception, match="died"):
        limited("""
            held = []
            while True:
                held.append(bytearray(64 << 20))
        """, deadline_s=30)
