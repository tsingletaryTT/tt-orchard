# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 3 seed reads stage 2's port directory lazily and stops as soon as a bound is passed (issue 326, review of PR 455, round 2).

The seed copies the port that passed stage 2 into `stages/3/port/`. The first version listed the whole tree with `sorted(src.rglob("*"))` before the
200-file guard ran, so a port directory with millions of entries was read to the end, held in memory, and refused only afterwards. These tests count
the entries the scan pulls from the operating system (a fake `os.scandir` that serves a huge directory one entry at a time), so a guard that runs after
the listing is seen as the failure it is, and one real hostile directory runs in a child with a 2 GiB address-space limit and a hard timeout.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import supervisor
from orchard.supervisor import Supervisor


class Ledger:
    def __init__(self):
        self.rows = []

    def append(self, event, stage, **data):
        self.rows.append((event, stage, data))


def seeder(tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "3"
    stage.mkdir(parents=True)
    (run / "stages" / "2" / "port").mkdir(parents=True)
    return SimpleNamespace(run_dir=run, ledger=Ledger()), stage, run / "stages" / "2" / "port"


def seed(sup, stage):
    Supervisor._seed_stage_3_port(sup, SimpleNamespace(number=3), stage)
    return [d.get("decision") for _, _, d in sup.ledger.rows]


class FakeEntry:
    """What `os.scandir` yields, for a name that is not on the disk."""

    def __init__(self, parent, name, kind):
        self.name, self.path, self._kind = name, f"{parent}/{name}", kind

    def is_symlink(self):
        return self._kind == "link"

    def is_dir(self, follow_symlinks=True):
        return self._kind == "dir"

    def is_file(self, follow_symlinks=True):
        return self._kind == "file"

    def stat(self, follow_symlinks=True):
        return SimpleNamespace(st_dev=1, st_ino=hash(self.name) & 0xFFFFFF)


class Counter:
    pulled = 0


def fake_directory(monkeypatch, root: Path, count: int, kind: str = "file"):
    """Make the listing of the directory `root` serve `count` entries one at a time, and count how many were asked for. The seed lists a directory
    through a descriptor it holds, so the fake recognises `root` by its identity (device and inode); other directories are real."""
    counter, real, root_text = Counter(), os.scandir, os.fspath(root)
    root_stat = os.stat(root)

    class Scan:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def close(self):
            pass

        def __iter__(self):
            for i in range(count):
                counter.pulled += 1
                yield FakeEntry(root_text, f"f{i:07d}.py" if kind == "file" else f"d{i:07d}", kind)

    def scandir(path="."):
        if isinstance(path, int):
            st = os.fstat(path)
            if (st.st_dev, st.st_ino) == (root_stat.st_dev, root_stat.st_ino):
                return Scan()
        return real(path)

    monkeypatch.setattr(supervisor.os, "scandir", scandir)
    return counter


@pytest.fixture
def fake_copy(monkeypatch):
    """The first files of a faked directory are 'copied' without a disk behind them."""
    monkeypatch.setattr(supervisor.safeio, "read_beneath", lambda root_fd, parts, *, max_bytes=None, expect_identity=None: b"x = 1\n")


@pytest.mark.timeout(60)
def test_a_directory_of_300000_files_is_refused_after_about_201_entries(tmp_path, monkeypatch, fake_copy):
    sup, stage, src = seeder(tmp_path)
    counter = fake_directory(monkeypatch, src, 300_000)
    assert seed(sup, stage) == ["stage 2 port not copied"]
    assert counter.pulled <= 205, f"the scan read {counter.pulled} entries before the file guard ran"
    problem = sup.ledger.rows[0][2]["problem"]
    assert "more than 200 files" in problem, problem
    assert not (stage / "port").exists(), "a refused seed leaves no half a port"


@pytest.mark.timeout(60)
def test_a_directory_of_300000_subdirectories_is_refused_at_the_entry_bound(tmp_path, monkeypatch, fake_copy):
    sup, stage, src = seeder(tmp_path)
    counter = fake_directory(monkeypatch, src, 300_000, kind="dir")
    assert seed(sup, stage) == ["stage 2 port not copied"]
    assert counter.pulled <= 1002, f"the scan read {counter.pulled} entries"
    problem = sup.ledger.rows[0][2]["problem"]
    assert f"more than {supervisor.STAGE_3_SEED_ENTRIES} entries" in problem, problem


def test_the_refusal_names_how_many_entries_were_read(tmp_path, monkeypatch, fake_copy):
    sup, stage, src = seeder(tmp_path)
    fake_directory(monkeypatch, src, 300_000)
    seed(sup, stage)
    assert "after reading 20" in sup.ledger.rows[0][2]["problem"], "the counts belong in the reason (201 or 202 entries)"


@pytest.mark.timeout(60)
def test_a_deep_tree_is_refused_at_the_depth_bound_without_reading_the_rest(tmp_path):
    sup, stage, src = seeder(tmp_path)
    here = src
    for _ in range(supervisor.STAGE_3_SEED_DEPTH + 30):
        here = here / "d"
        here.mkdir()
    (here / "deep.py").write_text("x = 1\n")
    (src / "port_model.py").write_text("import ttnn\n")
    assert seed(sup, stage) == ["stage 2 port not copied"]
    assert f"nested deeper than {supervisor.STAGE_3_SEED_DEPTH} levels" in sup.ledger.rows[0][2]["problem"]
    assert not (stage / "port").exists()


@pytest.mark.timeout(30)
def test_a_link_to_a_directory_is_refused_and_a_link_loop_is_not_followed(tmp_path):
    sup, stage, src = seeder(tmp_path)
    (src / "port_model.py").write_text("import ttnn\n")
    (src / "loop").symlink_to(src)
    assert seed(sup, stage) == ["stage 2 port not copied"]
    assert sup.ledger.rows[0][2]["problem"] == "loop is not a regular file"
    sup2, stage2, src2 = seeder(tmp_path / "b")
    (src2 / "port_model.py").write_text("import ttnn\n")
    (tmp_path / "elsewhere").mkdir()
    (src2 / "sub").symlink_to(tmp_path / "elsewhere")
    assert seed(sup2, stage2) == ["stage 2 port not copied"]
    assert sup2.ledger.rows[0][2]["problem"] == "sub is not a regular file", "a link to a directory used to be skipped without a word"


def test_the_files_are_listed_in_the_ledger_in_path_order(tmp_path, monkeypatch):
    sup, stage, src = seeder(tmp_path)
    for name in ("b.py", "a.py", "c.py"):
        (src / name).write_text("x = 1\n")
    (src / "z").mkdir()
    (src / "z" / "m.py").write_text("x = 1\n")
    real = os.scandir

    class Reversed:
        """The operating system's order is its own: here it is the reverse of the names, so a record that is not sorted afterwards shows it."""

        def __init__(self, path):
            self.entries = sorted(real(path), key=lambda e: e.name, reverse=True)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            return iter(self.entries)

    monkeypatch.setattr(supervisor.os, "scandir", lambda p=".": Reversed(p) if isinstance(p, int) else real(p))
    assert seed(sup, stage) == ["stage 2 port copied"]
    paths = [f["path"] for f in sup.ledger.rows[0][2]["files"]]
    assert len(paths) == 4 and paths == sorted(paths), paths


def test_a_refusal_text_is_cut_to_300_characters_whatever_the_names_are(tmp_path):
    sup, stage, src = seeder(tmp_path)
    here = src
    for _ in range(5):
        here = here / ("d" * 200)
        here.mkdir()
    (here / ("l" * 200)).symlink_to(src / "nowhere")
    assert seed(sup, stage) == ["stage 2 port not copied"]
    problem = sup.ledger.rows[0][2]["problem"]
    assert problem.endswith("is not a regular file") is False and len(problem) == 300, len(problem)


def test_a_cache_directory_is_skipped_without_being_read_and_a_pyc_file_is_skipped(tmp_path, monkeypatch):
    sup, stage, src = seeder(tmp_path)
    (src / "port_model.py").write_text("import ttnn\n")
    (src / "stale.pyc").write_bytes(b"\0")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")
    opened = []
    real = os.scandir
    monkeypatch.setattr(supervisor.os, "scandir", lambda p=".": (opened.append(p) if isinstance(p, int) else None) or real(p))
    assert seed(sup, stage) == ["stage 2 port copied"]
    assert len(opened) == 1, f"a cache directory is not descended into: {len(opened)} directories were listed"
    assert [f["path"] for f in sup.ledger.rows[0][2]["files"]] == ["stages/3/port/port_model.py"]


@pytest.mark.timeout(60)
def test_a_resumed_stage_with_a_huge_port_directory_is_not_scanned_to_the_end(tmp_path, monkeypatch, fake_copy):
    sup, stage, src = seeder(tmp_path)
    (src / "port_model.py").write_text("from_stage_2\n")
    (stage / "port").mkdir()
    counter = fake_directory(monkeypatch, stage / "port", 300_000, kind="dir")
    seed(sup, stage)
    assert counter.pulled <= 1002, f"the resume check read {counter.pulled} entries"
    assert any(d.get("decision") == "stage 3 port directory too large to scan" for _, _, d in sup.ledger.rows)
    assert not (stage / "port" / "port_model.py").exists(), "the coder's directory is left as it was, nothing is copied over it"


CHILD = r"""
import json, resource, sys, time
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
sys.path.insert(0, sys.argv[1])
from types import SimpleNamespace
from orchard.supervisor import Supervisor
class L:
    def __init__(self): self.rows = []
    def append(self, event, stage, **data): self.rows.append(data)
run = sys.argv[2]
sup = SimpleNamespace(run_dir=__import__("pathlib").Path(run), ledger=L())
t = time.time()
Supervisor._seed_stage_3_port(sup, SimpleNamespace(number=3), __import__("pathlib").Path(run) / "stages" / "3")
print(json.dumps({"seconds": round(time.time() - t, 2), "rows": sup.ledger.rows}))
"""


@pytest.mark.timeout(240)
def test_a_real_directory_of_120000_files_is_refused_fast_in_a_2_gib_child(tmp_path):
    run = tmp_path / "run"
    (run / "stages" / "3").mkdir(parents=True)
    src = run / "stages" / "2" / "port"
    src.mkdir(parents=True)
    for i in range(120_000):
        os.close(os.open(src / f"f{i:06d}.py", os.O_CREAT | os.O_WRONLY))
    repo = str(Path(__file__).resolve().parent.parent)
    done = subprocess.run([sys.executable, "-I", "-c", CHILD, repo, str(run)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-600:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["rows"][0]["decision"] == "stage 2 port not copied" and "more than 200 files" in out["rows"][0]["problem"]
    assert out["seconds"] < 15, out
    assert not (run / "stages" / "3" / "port").exists()
