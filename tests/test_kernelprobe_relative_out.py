# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A relative --out directory must reach the check as absolute paths (found in the issue 175 ladder pilot)."""
from __future__ import annotations

import io
import json
from pathlib import Path

from fake_model import FakeModel
from orchard import kernelprobe as kp
# The fixtures of test_kernelprobe.py: the stand-in tt-lang skill and freshness answers (both autouse) and the fake environment.
from test_kernelprobe import (consistency_ok, env_ready, fake_kenv, probe_args, skills_current, ttlang_skill,  # noqa: F401
                              writes_good_kernel_once)


def test_a_relative_out_directory_reaches_the_check_as_absolute_paths(tmp_path, fake_kenv, env_ready, monkeypatch):
    """The check runs with its own directory as the working directory, so a relative path on its command line pointed
    into that directory and the check wrote no result: every attempt ended `harness_error`. The default `--out
    kernel-probe` is relative too."""
    monkeypatch.chdir(tmp_path)
    runs = []

    def run_watched(argv, **kw):
        runs.append((list(argv), kw))
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({"passed": True, "category": None, "first_line": "",
                                                                   "sets": []}))
        return kp.Watched(0, None, False, 0.1)
    with FakeModel(writes_good_kernel_once()) as server:
        kp.main(probe_args(tmp_path, out="rel-out", endpoint=server.endpoint), env=fake_kenv, out=io.StringIO(),
                probe=lambda e, m: True, run_watched_fn=run_watched)
    checks = [r for r in runs if "--spec" in r[0]]
    assert len(checks) == 1, "the authoritative check should have run once"
    argv, kw = checks[0]
    for flag in ("--attempt-dir", "--out", "--spec", "--sim"):
        assert Path(argv[argv.index(flag) + 1]).is_absolute(), (flag, argv)
    assert Path(kw["log_path"]).is_absolute() and Path(kw["cwd"]).is_absolute()
    assert (tmp_path / "rel-out" / "kernel-probe-report.json").is_file()      # the output still lands where --out said
