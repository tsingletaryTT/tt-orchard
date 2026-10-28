# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/telegraph_screenshots.py needs an output directory (issue 131).

With no directory the script used to run the whole browser session and write no picture, with no error. It now stops at once with a
usage message and a non-zero exit, before it starts a browser. The test runs the script as a child, so it holds whether or not
playwright is installed: the argument check comes before the playwright import.
"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "telegraph_screenshots.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=60)


def test_only_the_url_exits_non_zero_and_names_the_usage():
    r = run("http://127.0.0.1:1/")
    assert r.returncode != 0
    assert "usage" in r.stderr.lower() and "output" in r.stderr.lower(), r.stderr


def test_no_arguments_exits_non_zero_and_names_the_usage():
    r = run()
    assert r.returncode != 0 and "usage" in r.stderr.lower(), r.stderr


def test_no_directory_is_created_or_written_when_the_check_fails(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "http://127.0.0.1:1/"], capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert r.returncode != 0 and list(tmp_path.iterdir()) == []
