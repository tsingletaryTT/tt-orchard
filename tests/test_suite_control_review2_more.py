# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Guards found by the mutation table of the second review fixes of PR 80."""
import fcntl
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_suite_control_review2 import child_script, terminate_self_when  # noqa: E402

from orchard import suite_control as sc  # noqa: E402


def test_the_launchers_own_exit_code_after_sigterm_is_reported_as_it_is(tmp_path):
    ready = tmp_path / "ready"
    argv = child_script(tmp_path, f"import signal, sys, time\nsignal.signal(signal.SIGTERM, lambda *a: sys.exit(4))\n"
                                  f"open({str(ready)!r}, 'w').write('1')\ntime.sleep(60)\n")
    t = terminate_self_when(ready)
    try:
        with pytest.raises(sc.Terminated) as info:
            sc._spawn(argv, dict(os.environ), str(tmp_path))
    finally:
        t.join()
    assert info.value.code == 4


def test_a_descriptor_of_another_file_is_not_taken_for_the_lock(tmp_path):
    other = os.open(tmp_path / "other", os.O_CREAT | os.O_RDWR)
    holder = os.open(tmp_path / sc.LOCK_NAME, os.O_CREAT | os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(sc.ControlError, match="already in use"):
            with sc.stage_lock(tmp_path, inherited_fd=other):
                pass
    finally:
        os.close(other)
        os.close(holder)


def test_a_closed_inherited_descriptor_falls_back_to_a_new_lock(tmp_path):
    with sc.stage_lock(tmp_path, inherited_fd=987) as fd:
        assert fd != 987


def test_the_launcher_gets_the_lock_descriptor_and_can_use_it(tmp_path):
    with sc.stage_lock(tmp_path) as fd:
        out = tmp_path / "out"
        argv = child_script(tmp_path, f"import os\nfd = int(os.environ[{sc.LOCK_FD_VAR!r}])\nopen({str(out)!r}, 'w').write(str(os.fstat(fd).st_size))\n")
        assert sc._spawn(argv, dict(os.environ, **{sc.LOCK_FD_VAR: str(fd)}), str(tmp_path)) == 0
        assert out.read_text() == "0"


def test_the_serve_script_does_not_get_the_lock_variable():
    env = sc.child_env({sc.LOCK_FD_VAR: "9", "TT_VISIBLE_DEVICES": "0000:01:00.0"}, "/tmp/x",
                       {"chips": [{"bdf": "0000:01:00.0", "dev_index": 0}]})
    assert sc.LOCK_FD_VAR not in env
