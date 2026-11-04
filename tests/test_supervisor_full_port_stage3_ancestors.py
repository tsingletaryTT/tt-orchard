# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 3 seed refuses a link standing at ANY name of its paths, not only the last one (issue 326, review of PR 455, round 4).

Round 3 pinned `stages/2/port` and `stages/3/port` by their last name. A coder that replaced `stages/2` (or `stages/3`, or `stages`) with a
link to a directory outside the run made the seed copy files from there and record success, or create directories, write files and remove
a half-written port outside the run. The copy is now anchored beneath one descriptor of the run directory. Each test builds the link in the
tree before the seed runs and checks that nothing outside the run was read, written or removed."""
import os

from test_supervisor_full_port_stage3_race import INSIDE_TEXT, OUTSIDE_TEXT, outside_dir, seed, seeder


def snapshot(root):
    """Every path under `root` with its type and, for a file, its bytes: a change anywhere outside the run shows as a difference."""
    out = {}
    for base, dirs, files in os.walk(root):
        for name in dirs + files:
            p = os.path.join(base, name)
            out[os.path.relpath(p, root)] = None if os.path.isdir(p) and not os.path.islink(p) else (os.readlink(p) if os.path.islink(p) else open(p, "rb").read())
    return out


def test_a_link_at_stage_2_makes_the_seed_refuse_and_copy_nothing_from_outside(tmp_path):
    sup, stage, port = seeder(tmp_path)
    (port / "port_model.py").write_text(INSIDE_TEXT)
    other = tmp_path / "outside"
    (other / "port").mkdir(parents=True)
    (other / "port" / "x.py").write_text(OUTSIDE_TEXT)
    real = port.parent
    real.rename(tmp_path / "moved-stage-2")
    os.symlink(other, real)                                   # stages/2 is now a link to the outside directory
    before = snapshot(other)
    assert seed(sup, stage) == ["stage 2 port not copied"]
    assert not (stage / "port").exists() and sup._port_seed == {"copied": 0, "kept": False}
    assert snapshot(other) == before


def test_a_link_at_stage_3_sends_no_write_and_no_removal_outside_the_run(tmp_path):
    sup, stage, port = seeder(tmp_path)
    (port / "port_model.py").write_text(INSIDE_TEXT)
    other = outside_dir(tmp_path, names=())
    (other / "port").mkdir()
    (other / "port" / "keep.txt").write_text("must survive")
    stage.rmdir()
    os.symlink(other, stage)                                  # stages/3 is now a link to the outside directory
    before = snapshot(other)
    assert seed(sup, stage) == ["stage 2 port not copied"]
    assert snapshot(other) == before, "nothing outside the run is created, written or removed"
    assert sup._port_seed == {"copied": 0, "kept": False}


def test_a_link_at_stage_3_whose_port_directory_holds_python_is_not_taken_for_a_kept_port(tmp_path):
    sup, stage, port = seeder(tmp_path)
    (port / "port_model.py").write_text(INSIDE_TEXT)
    other = outside_dir(tmp_path, names=())
    (other / "port").mkdir()
    (other / "port" / "theirs.py").write_text(OUTSIDE_TEXT)
    stage.rmdir()
    os.symlink(other, stage)
    seed(sup, stage)
    assert sup._port_seed["kept"] is False, "a port read through a link is not the coder's port"


def test_a_link_at_stages_is_refused_on_both_sides(tmp_path):
    sup, stage, port = seeder(tmp_path)
    (port / "port_model.py").write_text(INSIDE_TEXT)
    run = sup.run_dir
    other = tmp_path / "outside"
    other.mkdir()
    (run / "stages").rename(other / "stages")                 # the real tree moves outside the run, and `stages` links to it
    os.symlink(other / "stages", run / "stages")
    before = snapshot(other)
    assert seed(sup, stage) == ["stage 2 port not copied"]
    assert snapshot(other) == before


def test_a_link_planted_at_the_port_directory_is_still_replaced_by_the_copy(tmp_path):
    sup, stage, port = seeder(tmp_path)
    (port / "port_model.py").write_text(INSIDE_TEXT)
    other = outside_dir(tmp_path)
    os.symlink(other, stage / "port")
    before = snapshot(other)
    assert seed(sup, stage) == ["stage 2 port copied"]
    assert (stage / "port" / "port_model.py").read_text() == INSIDE_TEXT and not (stage / "port").is_symlink()
    assert snapshot(other) == before


def test_the_ordinary_copy_still_works_through_the_anchored_paths(tmp_path):
    sup, stage, port = seeder(tmp_path)
    (port / "sub").mkdir()
    (port / "sub" / "a.py").write_text(INSIDE_TEXT)
    assert seed(sup, stage) == ["stage 2 port copied"]
    assert (stage / "port" / "sub" / "a.py").read_text() == INSIDE_TEXT
