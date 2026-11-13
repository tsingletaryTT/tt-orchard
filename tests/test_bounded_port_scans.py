# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The full-port gate, the repair fingerprint and the evidence listing read a directory lazily and stop at a bound (issue 457).

Each of the three used to walk the whole directory before any limit applied: the gate's port check listed every link (`os.walk`) and every `.py`
file (`rglob`), the repair fingerprint hashed every file under the port directory into one dictionary, and the wrap-up listed and sorted every file
under `evidence/`. The port and evidence directories are written by agents, so a runaway or hostile directory made the supervisor spend memory and
time on it. These tests count the entries pulled from the operating system with a fake `os.scandir` that serves 300,000 entries one at a time (the
shape of `tests/test_supervisor_full_port_stage3_scan.py`), and one real hostile directory runs in a child with a 2 GiB address-space limit and a
hard timeout.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import boundedwalk, full_port, port_repair, supervisor


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
    """Make the listing of the directory `root` serve `count` entries one at a time and count how many were asked for. The walk lists through a
    descriptor, so the fake recognises `root` by its device and inode; other directories are real."""
    counter, real = Counter(), os.scandir
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
                yield FakeEntry(os.fspath(root), f"f{i:07d}.py" if kind == "file" else f"d{i:07d}", kind)

    def scandir(path="."):
        if isinstance(path, int):
            st = os.fstat(path)
            if (st.st_dev, st.st_ino) == (root_stat.st_dev, root_stat.st_ino):
                return Scan()
        return real(path)

    monkeypatch.setattr(os, "scandir", scandir)
    return counter


def deep_tree(root: Path, levels: int) -> None:
    here = root
    for _ in range(levels):
        here = here / "d"
        here.mkdir()
    (here / "deep.py").write_text("x = 1\n")


# ---- the gate's port check ------------------------------------------------------------------------------------------------------

def port_run(tmp_path):
    """A run directory with stages/2/port/port_model.py and a record that lists it."""
    run = tmp_path / "run"
    port = run / "stages" / "2" / "port"
    port.mkdir(parents=True)
    text = b"import ttnn\nx = ttnn.matmul(a, b)\n"
    (port / "port_model.py").write_bytes(text)
    record = {"port_sources": [{"path": full_port.PORT_MODULE, "role": full_port.MODEL_ROLE, "sha256": hashlib.sha256(text).hexdigest()}]}
    return run, run / "stages" / "2", port, record


def source_reasons(run, stage_dir, record):
    reasons = []
    full_port._source_reasons(record, stage_dir, run, reasons)
    return reasons


def test_a_port_inside_the_bounds_gives_no_scan_reason(tmp_path):
    run, stage_dir, _port, record = port_run(tmp_path)
    assert source_reasons(run, stage_dir, record) == []


@pytest.mark.timeout(60)
def test_the_gate_refuses_a_port_of_300000_files_after_about_the_entry_bound(tmp_path, monkeypatch):
    run, stage_dir, port, record = port_run(tmp_path)
    counter = fake_directory(monkeypatch, port, 300_000)
    reasons = source_reasons(run, stage_dir, record)
    assert counter.pulled <= boundedwalk.PORT_ENTRIES + 2, f"the gate read {counter.pulled} entries"
    hit = [r for r in reasons if f"more than {boundedwalk.PORT_ENTRIES} entries" in r]
    assert hit and "after reading" in hit[0], reasons


@pytest.mark.timeout(60)
def test_the_gate_refuses_a_port_of_300000_subdirectories(tmp_path, monkeypatch):
    run, stage_dir, port, record = port_run(tmp_path)
    counter = fake_directory(monkeypatch, port, 300_000, kind="dir")
    reasons = source_reasons(run, stage_dir, record)
    assert counter.pulled <= boundedwalk.PORT_ENTRIES + 2, f"the gate read {counter.pulled} entries"
    assert any(f"more than {boundedwalk.PORT_ENTRIES} entries" in r for r in reasons), reasons


def test_the_gate_refuses_a_port_nested_past_the_depth_bound(tmp_path):
    run, stage_dir, port, record = port_run(tmp_path)
    deep_tree(port, boundedwalk.PORT_DEPTH + 20)
    reasons = source_reasons(run, stage_dir, record)
    assert any(f"nested deeper than {boundedwalk.PORT_DEPTH} levels" in r for r in reasons), reasons


def test_the_gate_still_names_a_link_and_an_unlisted_python_file(tmp_path):
    run, stage_dir, port, record = port_run(tmp_path)
    (port / "sub").mkdir()
    (port / "sub" / "extra.py").write_text("y = 2\n")
    reasons = source_reasons(run, stage_dir, record)
    assert any("stages/2/port/sub/extra.py is in stages/2/port/ but not in port_sources" in r for r in reasons), reasons
    (port / "sub" / "extra.py").unlink()
    os.symlink(tmp_path, port / "sub" / "away")
    reasons = source_reasons(run, stage_dir, record)
    assert any(r.startswith("stages/2/port/sub/away is a link under stages/2/port/") for r in reasons), reasons


def test_the_gate_still_finds_a_link_inside_a_cache_directory(tmp_path):
    # the gate's old walk went into __pycache__; a link planted there is still named
    run, stage_dir, port, record = port_run(tmp_path)
    (port / "__pycache__").mkdir()
    os.symlink("/etc/hostname", port / "__pycache__" / "port_model.cpython-312.pyc")
    reasons = source_reasons(run, stage_dir, record)
    assert any("__pycache__/port_model.cpython-312.pyc is a link" in r for r in reasons), reasons


def test_the_gate_names_a_link_standing_where_the_port_directory_should_be(tmp_path):
    # a link to a directory outside the run is refused earlier (port_sources must name a file inside the run); this one stays inside it
    run, stage_dir, port, record = port_run(tmp_path)
    elsewhere = run / "elsewhere"
    port.rename(elsewhere)
    os.symlink(elsewhere, port)
    reasons = source_reasons(run, stage_dir, record)
    assert any(r.startswith("stages/2/port is a link") for r in reasons), reasons


# ---- the repair fingerprint -----------------------------------------------------------------------------------------------------

@pytest.mark.timeout(60)
def test_the_fingerprint_of_300000_files_stops_at_the_entry_bound(tmp_path, monkeypatch):
    counter = fake_directory(monkeypatch, tmp_path, 300_000)
    fp = port_repair.fingerprint(tmp_path)
    assert counter.pulled <= boundedwalk.PORT_ENTRIES + 2, f"the fingerprint read {counter.pulled} entries"
    assert list(fp) == [port_repair.SCAN_LIMIT_KEY], "past the bound the fingerprint holds one entry that says so, never a partial listing"
    assert f"more than {boundedwalk.PORT_ENTRIES} entries" in fp[port_repair.SCAN_LIMIT_KEY]


def test_the_fingerprint_of_a_tree_past_the_depth_bound_says_so(tmp_path):
    (tmp_path / "a.py").write_text("1")
    deep_tree(tmp_path, boundedwalk.PORT_DEPTH + 5)
    fp = port_repair.fingerprint(tmp_path)
    assert list(fp) == [port_repair.SCAN_LIMIT_KEY]
    assert f"nested deeper than {boundedwalk.PORT_DEPTH} levels" in fp[port_repair.SCAN_LIMIT_KEY]


def test_a_port_that_grows_past_the_bound_is_a_change(tmp_path):
    (tmp_path / "a.py").write_text("1")
    before = port_repair.fingerprint(tmp_path)
    deep_tree(tmp_path, boundedwalk.PORT_DEPTH + 5)
    assert port_repair.changed(before, port_repair.fingerprint(tmp_path)) == sorted(["a.py", port_repair.SCAN_LIMIT_KEY])


def test_the_fingerprint_reads_the_file_that_was_listed(tmp_path, monkeypatch):
    # the bytes are read beneath the walk's descriptor with the identity taken at the listing; a file swapped after the listing is not hashed
    (tmp_path / "a.py").write_text("1")
    calls = []
    real = boundedwalk.safeio.read_beneath

    def spy(root_fd, parts, *, max_bytes, expect_identity=None):
        calls.append((tuple(parts), expect_identity is not None))
        return real(root_fd, parts, max_bytes=max_bytes, expect_identity=expect_identity)

    monkeypatch.setattr(boundedwalk.safeio, "read_beneath", spy)
    port_repair.fingerprint(tmp_path)
    assert calls == [(("a.py",), True)]


# ---- the evidence listing -------------------------------------------------------------------------------------------------------

def evidence_stage(tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    (stage / "evidence").mkdir(parents=True)
    return run, stage


def test_the_evidence_listing_inside_the_bound_is_every_file_sorted(tmp_path):
    run, stage = evidence_stage(tmp_path)
    (stage / "evidence" / "b.txt").write_text("b")
    (stage / "evidence" / "sub").mkdir()
    (stage / "evidence" / "sub" / "a.txt").write_text("a")
    files, left_out = supervisor.evidence_listing(run, stage)
    assert files == ["stages/2/evidence/b.txt", "stages/2/evidence/sub/a.txt"]
    assert left_out is None


def test_the_evidence_listing_of_a_missing_directory_is_empty(tmp_path):
    run = tmp_path / "run"
    (run / "stages" / "2").mkdir(parents=True)
    assert supervisor.evidence_listing(run, run / "stages" / "2") == ([], None)


@pytest.mark.timeout(60)
def test_the_evidence_listing_of_300000_files_stops_and_says_what_was_left_out(tmp_path, monkeypatch):
    run, stage = evidence_stage(tmp_path)
    counter = fake_directory(monkeypatch, stage / "evidence", 300_000)
    files, left_out = supervisor.evidence_listing(run, stage)
    assert counter.pulled <= supervisor.EVIDENCE_LIST_ENTRIES + 2, f"the listing read {counter.pulled} entries"
    assert 0 < len(files) <= supervisor.EVIDENCE_LIST_ENTRIES and files == sorted(files)
    assert left_out and f"more than {supervisor.EVIDENCE_LIST_ENTRIES} entries" in left_out, left_out


def test_the_wrap_up_text_names_what_the_listing_left_out(tmp_path):
    spec = SimpleNamespace(number=2)
    text = supervisor.wrapup_text(spec, "result.json", ["stages/2/evidence/a.txt"], left_out="the listing stopped: holds more than 5 entries")
    assert "the listing stopped: holds more than 5 entries" in text
    assert "the listing stopped" not in supervisor.wrapup_text(spec, "result.json", ["stages/2/evidence/a.txt"])


# ---- one real hostile directory in a limited child ------------------------------------------------------------------------------

CHILD = r"""
import json, resource, sys, time
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from orchard import full_port, port_repair, supervisor
run = Path(sys.argv[2])
stage = run / "stages" / "2"
out = {}
t = time.time()
reasons = []
full_port._source_reasons({"port_sources": [{"path": full_port.PORT_MODULE, "role": "model", "sha256": "0"}]}, stage, run, reasons)
out["gate"] = [r for r in reasons if "entries" in r]
out["fingerprint"] = list(port_repair.fingerprint(stage / "port"))
out["evidence"] = supervisor.evidence_listing(run, stage)[1]
out["seconds"] = round(time.time() - t, 2)
print(json.dumps(out))
"""


@pytest.mark.timeout(240)
def test_a_real_directory_of_60000_files_is_refused_fast_in_a_2_gib_child(tmp_path):
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    for sub in ("port", "evidence"):
        (stage / sub).mkdir(parents=True)
        for i in range(30_000):
            os.close(os.open(stage / sub / f"f{i:06d}.py", os.O_CREAT | os.O_WRONLY))
    (stage / "port" / "port_model.py").write_text("import ttnn\n")      # the gate reaches the directory scan only with a readable model file
    repo = str(Path(__file__).resolve().parent.parent)
    done = subprocess.run([sys.executable, "-I", "-c", CHILD, repo, str(run)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-600:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["gate"], out
    assert out["fingerprint"] == [port_repair.SCAN_LIMIT_KEY], out
    assert out["evidence"] and "more than" in out["evidence"], out
    assert out["seconds"] < 15, out
