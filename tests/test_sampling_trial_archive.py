# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 196: a restart that skips the sampling trial moves the trial's outputs into sampling-trial/stale/. A kill between the two moves must
not let the next restart delete the only copy of the trial result."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from orchard.supervisor import Supervisor


def make_trial(stage_dir: Path) -> Path:
    trial = stage_dir / "sampling-trial"
    (trial / "variant" / "evidence").mkdir(parents=True)
    (trial / "variant" / "evidence" / "swap-check.json").write_text('{"v": 1}')
    (trial / "trial-result.json").write_text('{"result": "the only copy"}')
    return trial


def all_files(trial: Path) -> dict[str, str]:
    return {str(p.relative_to(trial)): p.read_text() for p in trial.rglob("*") if p.is_file()}


def test_a_kill_between_the_two_moves_keeps_the_first_moved_file_through_two_restarts(tmp_path, monkeypatch):
    trial = make_trial(tmp_path)
    real = os.replace
    calls = {"n": 0}

    def replace(a, b):
        calls["n"] += 1
        if calls["n"] == 2:                                   # killed after trial-result.json moved, before variant/ moves
            raise KeyboardInterrupt
        return real(a, b)
    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(KeyboardInterrupt):
        Supervisor._set_stale_trial_outputs_aside(tmp_path)
    monkeypatch.setattr(os, "replace", real)
    assert (trial / "stale" / "trial-result.json").read_text() == '{"result": "the only copy"}'
    Supervisor._set_stale_trial_outputs_aside(tmp_path)        # restart 1
    Supervisor._set_stale_trial_outputs_aside(tmp_path)        # restart 2 changes nothing more
    files = all_files(trial)
    assert '{"result": "the only copy"}' in files.values()
    assert '{"v": 1}' in files.values()
    assert not (trial / "trial-result.json").exists() and not (trial / "variant").exists()
