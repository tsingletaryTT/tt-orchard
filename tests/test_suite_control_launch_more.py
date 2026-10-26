# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""More guards for the launcher and the run flow (found by the mutation table of the review fixes of PR 80)."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_suite_control import CHECKOUT, _prep, prep  # noqa: E402,F401
from test_suite_control_launch import BDFS, GRANT, Gozer, control_dir, fakes, run_cmd  # noqa: E402,F401

from orchard import suite_control as sc  # noqa: E402


@pytest.mark.parametrize("index", [True, "2", None, 2.0, -1.5])
def test_a_device_index_that_is_not_a_whole_number_is_refused(index):
    status = {"chips": [{"bdf": BDFS[0], "dev_index": index}]}
    with pytest.raises(sc.ControlError, match="device index"):
        sc.device_ids(BDFS[0], status)


def test_a_container_check_that_cannot_list_after_removal_is_reported(tmp_path):
    calls = []

    def docker(argv):
        calls.append(argv[0])
        if argv[0] == "ps":
            return (0, "abc\n") if calls.count("ps") == 1 else (1, "daemon gone")
        return (0, "")
    problems = sc.cleanup(tmp_path, "lbl", sweep=lambda *a, **k: [], docker=docker, uses_docker=True)
    assert calls == ["ps", "rm", "ps"] and problems and "could not confirm" in problems[0]


def test_the_serve_script_gets_a_sigterm_of_its_own_before_it_is_killed(tmp_path, fakes):
    marker = tmp_path / "marker"
    body = (f"import signal, sys, time\n"
            f"signal.signal(signal.SIGTERM, lambda *a: (open({str(marker)!r}, 'w').write('term'), sys.exit(0)))\n"
            f"open({str(tmp_path / 'ready')!r}, 'w').write('1')\n"
            f"time.sleep(120)\n")
    d = control_dir(tmp_path, serve_body=body)
    env = dict(os.environ, PATH=f"{fakes['bin']}:{os.environ['PATH']}", TT_VISIBLE_DEVICES=f"{BDFS[0]},{BDFS[1]}", PYTHONPATH=str(CHECKOUT))
    proc = subprocess.Popen([sys.executable, "-m", "orchard.suite_control", "launch", "--control-dir", str(d)], env=env, cwd=str(CHECKOUT),
                            start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        os.kill(proc.pid, signal.SIGTERM)
        started = time.monotonic()
        proc.wait(timeout=60)
        assert marker.exists() and time.monotonic() - started < sc.STOP_GRACE_S
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)


def test_the_acquire_request_carries_the_bundles_chip_count_and_the_expected_duration(prep):
    (prep["bundle"] / "tt_kernel_manifest.json").write_text(json.dumps({"device_count": 4}))
    g = Gozer(acquire=GRANT)
    run_cmd(prep, g, "--expect", "75m")
    a = g.calls[0]
    assert a[a.index("--chips") + 1] == "4" and a[a.index("--expect") + 1] == "75m"
