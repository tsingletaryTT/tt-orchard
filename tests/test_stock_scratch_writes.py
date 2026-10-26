# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""write_file into <run>/tt-metal-scratch (issue 61, operator decision 3). The scratch directory is a copy of the host tt-metal
tree's models/ that a stock-runtime run may change. It is the only place outside the stage directory an agent may write, it exists
only for stages 2 to 4 of that path, and it never reaches the host tree."""
import json
import os
from pathlib import Path

import pytest

from orchard import stages
from orchard.agent import Tools


def make(tmp_path, scratch=True):
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    stage.mkdir(parents=True)
    sc = run / "tt-metal-scratch"
    sc.mkdir()
    return Tools(run, stage, {"PATH": os.environ["PATH"]}, scratch=sc if scratch else None), run, stage, sc


def write(t, path, content="x"):
    return t.call("write_file", json.dumps({"path": path, "content": content}))


def test_a_run_relative_scratch_path_writes_into_the_scratch_directory(tmp_path):
    t, run, stage, sc = make(tmp_path)
    out = write(t, "tt-metal-scratch/models/tt_transformers/model_params/X/config.json", "{}")
    assert out.startswith("wrote") and "tt-metal-scratch/models/tt_transformers/model_params/X/config.json" in out
    assert (sc / "models/tt_transformers/model_params/X/config.json").read_text() == "{}"
    assert not (stage / "tt-metal-scratch").exists()


def test_an_absolute_path_inside_the_scratch_directory_works(tmp_path):
    t, run, stage, sc = make(tmp_path)
    assert write(t, str(sc / "a.txt"), "abs").startswith("wrote")
    assert (sc / "a.txt").read_text() == "abs"


def test_without_a_scratch_directory_the_path_is_only_a_stage_subdirectory(tmp_path):
    t, run, stage, sc = make(tmp_path, scratch=False)
    write(t, "tt-metal-scratch/a.txt")
    assert not (sc / "a.txt").exists()                       # nothing reaches the scratch directory
    assert (stage / "tt-metal-scratch" / "a.txt").is_file()   # it is a stage-relative path like any other


@pytest.mark.parametrize("path", ["tt-metal-scratch/../../x.txt", "tt-metal-scratch/..", "../tt-metal-scratch/x.txt",
                                  "stages/1/x.txt", "/etc/x.txt", "/tmp/x.txt"])
def test_escapes_are_refused(tmp_path, path):
    t, run, stage, sc = make(tmp_path)
    out = write(t, path)
    assert out.startswith("refused:"), out
    assert not (tmp_path / "x.txt").exists() and not (run / "x.txt").exists() and not list(sc.iterdir())


@pytest.mark.parametrize("path", ["tt-metal-scratch", "tt-metal-scratch/", "tt-metal-scratchy/x.txt"])
def test_the_directory_itself_and_look_alikes_never_write_into_the_scratch_directory(tmp_path, path):
    t, run, stage, sc = make(tmp_path)
    write(t, path)
    assert not list(sc.iterdir())


def test_a_link_in_the_scratch_directory_that_leaves_it_writes_nothing_outside(tmp_path):
    t, run, stage, sc = make(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (sc / "link").symlink_to(outside)
    write(t, "tt-metal-scratch/link/x.txt")
    assert not list(outside.iterdir())


def test_the_ledger_name_is_still_protected_in_the_scratch_directory(tmp_path):
    t, run, stage, sc = make(tmp_path)
    assert write(t, "tt-metal-scratch/ledger.jsonl").startswith("refused:")


def test_the_stage_directory_still_works_as_before(tmp_path):
    t, run, stage, sc = make(tmp_path)
    assert write(t, "result.json", "a").startswith("wrote") and (stage / "result.json").read_text() == "a"


def test_the_supervisor_offers_the_scratch_directory_only_to_stages_2_to_4_of_the_stock_path():
    assert stages.scratch_stage(2, "stock-runtime") and stages.scratch_stage(3, "stock-runtime") and stages.scratch_stage(4, "stock-runtime")
    for n in (0, 1, 5, 6, 7, 8):
        assert not stages.scratch_stage(n, "stock-runtime")
    for path in ("weights-only", "full-port", None):
        assert not stages.scratch_stage(2, path)
