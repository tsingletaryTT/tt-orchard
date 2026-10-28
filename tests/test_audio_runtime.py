# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/audio_runtime.py: which models the harness's own audio runtime serves, where its programs live, and the ledger record of the choice."""
import json
import os
import stat
from pathlib import Path

import pytest

from orchard import audio_runtime as rt
from test_audio_chip_config import REAL_CONFIG


def snapshot(tmp_path, config=REAL_CONFIG, raw=None):
    d = tmp_path / "snap"
    d.mkdir()
    (d / "model_config.json").write_text(raw if raw is not None else json.dumps(config))
    return d


def test_a_supported_snapshot_has_no_problems(tmp_path):
    assert rt.runtime_problems(snapshot(tmp_path)) == []


def test_an_unsupported_configuration_is_named(tmp_path):
    cfg = json.loads(json.dumps(REAL_CONFIG))
    cfg["model"]["diffusion"]["config"]["conformer"] = True
    problems = rt.runtime_problems(snapshot(tmp_path, cfg))
    assert problems and "conformer" in problems[0]


@pytest.mark.parametrize("raw", ["", "not json", "[]", "3", '{"a": 1, "a": 2}', '{"x": NaN}'])
def test_a_config_that_cannot_be_read_is_a_reason_and_not_a_crash(tmp_path, raw):
    problems = rt.runtime_problems(snapshot(tmp_path, raw=raw))
    assert problems and ("cannot be read" in problems[0] or "model.diffusion" in problems[0])


def test_a_missing_snapshot_is_a_reason(tmp_path):
    assert "cannot be read" in rt.runtime_problems(tmp_path / "nowhere")[0]


def test_an_oversize_config_is_refused_and_not_parsed(tmp_path):
    big = json.dumps({"pad": "x" * (rt.CONFIG_MAX_BYTES + 10)})
    assert "cannot be read" in rt.runtime_problems(snapshot(tmp_path, raw=big))[0]


def test_default_paths_sit_under_the_operator_home_and_variables_replace_them(tmp_path):
    p = rt.audio_paths(tmp_path, {})
    assert p.reference_python == str(tmp_path / "orchard-venvs" / "stable-audio" / "bin" / "python")
    assert p.chip_python == str(tmp_path / ".tenstorrent-venv" / "bin" / "python")
    assert p.src_dir == str(tmp_path / "orchard-src" / "stable-audio-3")
    q = rt.audio_paths(tmp_path, {"ORCHARD_AUDIO_PYTHON": "/a/py", "ORCHARD_TTNN_PYTHON": "/b/py", "ORCHARD_AUDIO_SRC": "/c/src"})
    assert (q.reference_python, q.chip_python, q.src_dir) == ("/a/py", "/b/py", "/c/src")


def make_env(tmp_path, *, ref=True, chip=True, src=True):
    p = rt.audio_paths(tmp_path, {})
    for want, exe in ((ref, p.reference_python), (chip, p.chip_python)):
        if want:
            Path(exe).parent.mkdir(parents=True, exist_ok=True)
            Path(exe).write_text("#!/bin/sh\n")
            os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR)
    if src:
        (Path(p.src_dir) / "stable_audio_3").mkdir(parents=True)
        (Path(p.src_dir) / "stable_audio_3" / "model.py").write_text("")
    return p


def test_a_complete_environment_has_no_problems(tmp_path):
    assert rt.environment_problems(make_env(tmp_path)) == []


@pytest.mark.parametrize("kw, word", [({"ref": False}, "reference interpreter"), ({"chip": False}, "Tenstorrent interpreter"),
                                      ({"src": False}, "library checkout")])
def test_each_missing_part_is_named_with_the_fix(tmp_path, kw, word):
    problems = rt.environment_problems(make_env(tmp_path, **kw))
    assert len(problems) == 1 and word in problems[0]
    if "Tenstorrent" not in word:
        assert "audio-reference-env" in problems[0]


def test_a_file_that_is_not_executable_does_not_count(tmp_path):
    p = make_env(tmp_path)
    os.chmod(p.reference_python, 0o644)
    assert any("reference interpreter" in x for x in rt.environment_problems(p))


def test_script_hashes_cover_every_program_and_follow_the_file_contents():
    h = rt.script_hashes()
    assert set(h) == set(rt.SCRIPTS) and all(len(v) == 64 for v in h.values())


def entry(decision, **kw):
    return {"event": "decision", "data": {"decision": decision, **kw}}


def test_the_last_selection_in_the_ledger_is_the_run_s_choice():
    entries = [entry("other"), entry(rt.AUDIO_RUNTIME_SELECTED, runtime="a"), entry(rt.AUDIO_RUNTIME_SELECTED, runtime="b")]
    assert rt.selected(entries)["runtime"] == "b" and rt.selected([entry("x")]) is None and rt.selected([]) is None


def test_the_paths_come_back_from_the_record_and_a_bad_record_gives_none():
    p = rt.AudioPaths("/r", "/c", "/s")
    assert rt.selection_paths([entry(rt.AUDIO_RUNTIME_SELECTED, paths=p.record())]) == p
    assert rt.selection_paths([entry(rt.AUDIO_RUNTIME_SELECTED, paths="x")]) is None
    assert rt.selection_paths([entry(rt.AUDIO_RUNTIME_SELECTED, paths={"reference_python": "/r"})]) is None
    assert rt.selection_paths([entry(rt.AUDIO_RUNTIME_SELECTED)]) is None
    assert rt.selection_paths([]) is None
