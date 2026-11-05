# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 3 seed cannot be steered outside stage 2's port by a swapped link (issue 326, review of PR 455, round 3).

The seed lists stage 2's port directory, then reads each file and writes it into `stages/3/port/`. The coder's shell can still change the
directory tree between those steps. The first version listed and read by path, so a directory replaced by a link after it was classified, or
after its entries were listed, made the scan list or read files from OUTSIDE the port, and the seed still reported success. A link planted
inside the destination made the writes land outside too.

Every swap here happens at a fixed point, with no sleeps and no threads: a wrapper around the entries `os.scandir` gives back runs the swap
right after the call that classified the entry, which is the moment the old code had already decided what the entry was.
"""
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import supervisor
from orchard.supervisor import Supervisor

OUTSIDE_TEXT = "OUTSIDE = 'this file is not in the port'\n"
INSIDE_TEXT = "INSIDE = 'this file is in the port'\n"


class Ledger:
    def __init__(self):
        self.rows = []

    def append(self, event, stage, **data):
        self.rows.append((event, stage, data))


def seeder(tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "3"
    stage.mkdir(parents=True)
    port = run / "stages" / "2" / "port"
    port.mkdir(parents=True)
    return SimpleNamespace(run_dir=run, ledger=Ledger()), stage, port


def seed(sup, stage):
    Supervisor._seed_stage_3_port(sup, SimpleNamespace(number=3), stage)
    return [d.get("decision") for _, _, d in sup.ledger.rows]


def outside_dir(tmp_path, names=("x.py",)):
    out = tmp_path / "outside"
    out.mkdir()
    for n in names:
        (out / n).write_text(OUTSIDE_TEXT)
    return out


class Entry:
    """A directory entry that runs `fire()` once, right after the first classification call that answers `when`."""

    def __init__(self, real, fire, when):
        self._real, self._fire, self._when = real, fire, when
        self._stat = None

    @property
    def name(self):
        return self._real.name

    @property
    def path(self):
        return self._real.path

    def inode(self):
        return self._real.inode()

    def _answered(self, call, result):
        if self._fire is not None and call == self._when and result:
            # What the scan knew about the entry when it classified it: the swap comes after that, so a later `stat` still says the old file.
            self._stat = self._real.stat(follow_symlinks=False)
            fire, self._fire = self._fire, None
            fire()
        return result

    def is_symlink(self):
        return self._answered("is_symlink", self._real.is_symlink())

    def is_dir(self, follow_symlinks=True):
        return self._answered("is_dir", self._real.is_dir(follow_symlinks=follow_symlinks))

    def is_file(self, follow_symlinks=True):
        return self._answered("is_file", self._real.is_file(follow_symlinks=follow_symlinks))

    def stat(self, follow_symlinks=True):
        return self._stat if self._stat is not None else self._real.stat(follow_symlinks=follow_symlinks)


def swap_on(monkeypatch, entry_name, when, fire):
    """Wrap `os.scandir` so the entry called `entry_name` runs `fire` after the `when` call that said yes (`is_dir` or `is_file`)."""
    real = os.scandir

    class Listing:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        def close(self):
            self._inner.close()

        def __iter__(self):
            for e in self._inner:
                yield Entry(e, fire if e.name == entry_name else None, when)

    monkeypatch.setattr(supervisor.os, "scandir", lambda *a, **k: Listing(real(*a, **k)))


def replace_dir_with_link(directory: Path, target: Path):
    """What the coder's shell can do at any moment: move the real directory away and put a link to `target` in its place."""
    def do():
        os.rename(directory, str(directory) + ".orig")
        os.symlink(target, directory)
    return do


def test_a_directory_swapped_for_a_link_after_it_was_classified_is_not_listed(tmp_path, monkeypatch):
    sup, stage, port = seeder(tmp_path)
    out = outside_dir(tmp_path)
    (port / "port_model.py").write_text("import ttnn\n")
    (port / "sub").mkdir()
    (port / "sub" / "x.py").write_text(INSIDE_TEXT)
    swap_on(monkeypatch, "sub", "is_dir", replace_dir_with_link(port / "sub", out))
    assert seed(sup, stage) == ["stage 2 port not copied"], "the seed listed a directory that was no longer the one it classified"
    assert not (stage / "port").exists()
    assert (out / "x.py").read_text() == OUTSIDE_TEXT


def test_a_directory_swapped_for_a_link_after_its_files_were_listed_is_not_read_through(tmp_path, monkeypatch):
    sup, stage, port = seeder(tmp_path)
    out = outside_dir(tmp_path)
    (port / "sub").mkdir()
    (port / "sub" / "x.py").write_text(INSIDE_TEXT)
    swap_on(monkeypatch, "x.py", "is_file", replace_dir_with_link(port / "sub", out))
    decisions = seed(sup, stage)
    assert decisions == ["stage 2 port not copied"], f"the seed read through a link and said: {decisions}"
    assert not (stage / "port").exists(), "a refused seed leaves no half a port"
    for row in sup.ledger.rows:
        assert OUTSIDE_TEXT.strip() not in str(row)


def test_a_file_replaced_by_another_file_after_the_scan_is_refused(tmp_path, monkeypatch):
    sup, stage, port = seeder(tmp_path)
    (port / "port_model.py").write_text(INSIDE_TEXT)

    def replace_file():
        new = port / "port_model.py.new"
        new.write_text("REPLACED = True\n")
        os.replace(new, port / "port_model.py")
    swap_on(monkeypatch, "port_model.py", "is_file", replace_file)
    assert seed(sup, stage) == ["stage 2 port not copied"], "a file that is not the one the scan saw was copied"
    assert "changed" in sup.ledger.rows[0][2]["problem"], sup.ledger.rows[0][2]["problem"]


def test_a_link_planted_inside_the_destination_is_not_written_through(tmp_path):
    sup, stage, port = seeder(tmp_path)
    (port / "sub").mkdir()
    (port / "sub" / "x.py").write_text(INSIDE_TEXT)
    out = outside_dir(tmp_path, names=())
    (stage / "port").mkdir()
    (stage / "port" / "sub").symlink_to(out)          # the coder (or anything else) planted this before the seed ran
    seed(sup, stage)
    assert list(out.iterdir()) == [], f"the seed wrote through a planted link into {list(out.iterdir())}"
    assert sup.ledger.rows[-1][2].get("decision") != "stage 2 port copied"


def test_a_link_planted_where_the_destination_directory_goes_is_replaced_by_a_real_directory(tmp_path):
    sup, stage, port = seeder(tmp_path)
    (port / "port_model.py").write_text(INSIDE_TEXT)
    out = outside_dir(tmp_path, names=())
    (stage / "port").symlink_to(out)                    # stages/3/port is a link to somewhere else
    assert seed(sup, stage) == ["stage 2 port copied"]
    assert list(out.iterdir()) == [], "the seed wrote into the directory the link pointed at"
    assert not (stage / "port").is_symlink() and (stage / "port" / "port_model.py").read_text() == INSIDE_TEXT


def test_a_port_directory_that_is_itself_a_link_is_refused(tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "3"
    stage.mkdir(parents=True)
    out = outside_dir(tmp_path)
    (run / "stages" / "2").mkdir(parents=True)
    (run / "stages" / "2" / "port").symlink_to(out)
    sup = SimpleNamespace(run_dir=run, ledger=Ledger())
    assert seed(sup, stage) == ["stage 2 port not copied"], "a link standing where the port directory should be was followed"
    assert not (stage / "port").exists()


def test_a_normal_port_is_copied_and_the_ledger_hash_is_the_hash_of_the_bytes_copied(tmp_path):
    sup, stage, port = seeder(tmp_path)
    (port / "port_model.py").write_text(INSIDE_TEXT)
    (port / "sub").mkdir()
    (port / "sub" / "x.py").write_text("x = 2\n")
    assert seed(sup, stage) == ["stage 2 port copied"]
    files = {f["path"]: f["sha256"] for f in sup.ledger.rows[0][2]["files"]}
    assert set(files) == {"stages/3/port/port_model.py", "stages/3/port/sub/x.py"}
    for rel, digest in files.items():
        copied = stage.parent.parent / rel
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == digest, rel
    assert (stage / "port" / "sub" / "x.py").read_text() == "x = 2\n"
