# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The suite's guard on the skills freshness check (tests/conftest.py, `no_skills_network`).

Review M5: the guard must fail a test that reaches skillsfresh's real git or gh runner, and must keep the result cache
inside the test's own directory. This runs a small inner test file with a copy of conftest.py in a child pytest, under
a timeout, and reads its outcome."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

INNER = '''
from pathlib import Path
from orchard import skillsfresh as sf


def test_reaches_the_real_runner(tmp_path):
    sf.check_skills_dirs([tmp_path], cache=sf.Cache(tmp_path / "c.json"))      # runs git through _raw_run


def test_keeps_away_from_the_runner(tmp_path):
    assert sf.Cache().path.is_relative_to(tmp_path.parent.parent)                # the cache is under pytest's tmp
'''


def test_the_guard_fails_a_test_that_reaches_the_real_runner_and_keeps_the_cache_in_tmp(tmp_path):
    shutil.copy(REPO / "tests" / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_inner.py").write_text(INNER)
    env = dict(os.environ, PYTHONPATH=str(REPO))
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rA", str(tmp_path)],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    out = done.stdout + done.stderr
    assert done.returncode != 0, out
    assert "ERROR" in out and "test_reaches_the_real_runner" in out and "real git or gh" in out, out
    assert "PASSED test_inner.py::test_keeps_away_from_the_runner" in out, out
