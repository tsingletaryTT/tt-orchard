"""run_command: exit codes, output, timeouts, and leaving a reset running."""
import os
import signal
import sys

import pytest

from orchard.commands import run_command

PY = sys.executable


def test_captures_exit_code_and_both_streams():
    r = run_command([PY, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"], 10)
    assert (r.returncode, r.stdout, r.stderr, r.timed_out) == (3, "out\n", "err\n", False)


def test_missing_program_is_exit_127():
    r = run_command(["/nonexistent/orchard-no-such-binary"], 5)
    assert r.returncode == 127 and "not found" in r.stderr


def test_timeout_kills_the_command():
    r = run_command([PY, "-c", "import time; time.sleep(30)"], 0.3)
    assert r.timed_out and r.returncode is None and not r.left_running
    with pytest.raises(ProcessLookupError):
        os.kill(r.pid, 0)


def test_timeout_can_leave_a_reset_running():
    r = run_command([PY, "-c", "import time; time.sleep(30)"], 0.3, kill_on_timeout=False)
    try:
        assert r.timed_out and r.left_running and r.returncode is None
        os.kill(r.pid, 0)          # still alive: a reset must never be cut short
    finally:
        os.killpg(r.pid, signal.SIGKILL)


def test_env_is_passed_through():
    r = run_command([PY, "-c", "import os; print(os.environ['ORCHARD_T'])"], 10,
                    env={**os.environ, "ORCHARD_T": "x"})
    assert r.stdout == "x\n"


def test_record_truncates_each_stream():
    r = run_command([PY, "-c", "print('a' * 5000)"], 10)
    rec = r.record()
    assert rec["returncode"] == 0 and len(rec["stdout"]) == 2000 and rec["argv"][0] == PY
