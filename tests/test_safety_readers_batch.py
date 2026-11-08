# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Safety-reader batch (issues 493, 494, 464, 503, 502, 461): the guards the hostile-layout harness (tests/hostile_fs.py) could not state
by itself. The harness cases for the ledger lock link, the residency link out of the cache and the sidecar link now pass; these tests add
what each fix promises beyond a case: a FIFO named like the lock file, a refused sidecar link with no size, a link inside the snapshot,
a link into blobs/, and a plain file."""
import importlib.util
import os
from pathlib import Path

import pytest

from orchard import ledger as ledger_mod

DT = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"
_spec = importlib.util.spec_from_file_location("delta_triage_batch_under_test", DT)
dt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dt)


# ---- 493: the ledger's lock file ----------------------------------------------------------------------------------------------------

def test_a_lock_file_that_is_a_link_is_refused_and_its_target_keeps_its_content(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"precious\n")
    os.symlink(victim, tmp_path / "ledger.jsonl.lock")
    with pytest.raises(ledger_mod.LedgerLockRefused):
        ledger_mod.Ledger(tmp_path / "ledger.jsonl")
    assert victim.read_bytes() == b"precious\n"


def test_a_pipe_named_like_the_lock_file_is_refused_and_does_not_block(tmp_path):
    os.mkfifo(tmp_path / "ledger.jsonl.lock")
    with pytest.raises(ledger_mod.LedgerLockRefused):       # a blocking open would hang here; the suite's timeout would end it
        ledger_mod.Ledger(tmp_path / "ledger.jsonl")


def test_the_refusal_is_a_ledger_locked_so_existing_handlers_still_catch_it(tmp_path):
    assert issubclass(ledger_mod.LedgerLockRefused, ledger_mod.LedgerLocked)


def test_a_plain_lock_file_still_works_and_a_second_opener_is_locked_out(tmp_path):
    first = ledger_mod.Ledger(tmp_path / "ledger.jsonl")
    try:
        with pytest.raises(ledger_mod.LedgerLocked):
            ledger_mod.Ledger(tmp_path / "ledger.jsonl")
    finally:
        first.close()
    ledger_mod.Ledger(tmp_path / "ledger.jsonl").close()


# ---- 464: the stage 0 sidecar scan --------------------------------------------------------------------------------------------------

def _scan(model_dir, tmp_path):
    empty = tmp_path / "empty_base"
    empty.mkdir(exist_ok=True)
    return dt.compare_sidecars(model_dir, empty, set())


def test_a_gguf_link_to_an_outside_file_is_listed_by_name_with_no_size(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    outside = tmp_path / "other.bin"
    outside.write_bytes(b"x" * 12345)
    os.symlink(outside, snap / "extra.gguf")
    got = _scan(snap, tmp_path)
    entry = next(e for e in got["ignored_files"] if e["file"] == "extra.gguf")
    assert entry["size"] is None and "outside" in entry["reason"]


def test_a_safetensors_sidecar_link_to_an_outside_file_is_a_problem_with_no_hash(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    outside = tmp_path / "other.safetensors"
    outside.write_bytes(b"x" * 100)
    (snap / "model.safetensors").write_bytes(b"backbone")          # with a model*.safetensors present, extra.safetensors is a sidecar
    os.symlink(outside, snap / "extra.safetensors")
    got = _scan(snap, tmp_path)
    entry = next(e for e in got["sidecars"] if e["file"] == "extra.safetensors")
    assert entry["sha256"] is None and entry["size"] is None
    assert any("extra.safetensors" in p and "outside" in p for p in got["problems"])


def test_a_gguf_link_into_the_caches_blobs_directory_keeps_its_size(tmp_path):
    model = tmp_path / "models--org--name"
    blobs, snap = model / "blobs", model / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (blobs / "abc").write_bytes(b"y" * 777)
    os.symlink("../../blobs/abc", snap / "extra.gguf")
    entry = next(e for e in _scan(snap, tmp_path)["ignored_files"] if e["file"] == "extra.gguf")
    assert entry["size"] == 777


def test_a_gguf_link_inside_the_snapshot_and_a_plain_file_keep_their_size(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "real.gguf").write_bytes(b"z" * 55)
    os.symlink(snap / "real.gguf", snap / "alias.gguf")
    sizes = {e["file"]: e["size"] for e in _scan(snap, tmp_path)["ignored_files"]}
    assert sizes == {"real.gguf": 55, "alias.gguf": 55}


def test_a_pipe_that_has_a_reader_opens_but_is_refused_because_it_is_not_a_regular_file(tmp_path):
    """With a reader on the other end the open of a pipe succeeds, so only the regular-file check on the descriptor stops it."""
    fifo = tmp_path / "ledger.jsonl.lock"
    os.mkfifo(fifo)
    reader = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        with pytest.raises(ledger_mod.LedgerLockRefused, match="not a regular file"):
            ledger_mod.Ledger(tmp_path / "ledger.jsonl")
    finally:
        os.close(reader)


# ---- 503: port_repair.current_sources walks the port directory through a pinned descriptor -----------------------------------------------

import hashlib

from orchard import port_repair, safeio


def _port_tree(tmp_path):
    stage = tmp_path / "stages" / "2"
    (stage / "port" / "pkg").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text("A = 1\n")
    (stage / "port" / "pkg" / "helper.py").write_text("B = 2\n")
    return stage


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def test_current_sources_lists_every_py_file_with_its_digest(tmp_path):
    got = port_repair.current_sources(_port_tree(tmp_path))
    assert got == {"stages/2/port/port_model.py": _digest("A = 1\n"), "stages/2/port/pkg/helper.py": _digest("B = 2\n")}


def test_a_subdirectory_swapped_for_a_link_with_the_same_sources_after_the_walk_is_refused(tmp_path, monkeypatch):
    """The window of issue 503: the sources are listed, then pkg/ is replaced by a link to a directory that holds the same files. A read by
    pathname follows it and the manifest still matches; a read beneath the pinned descriptor refuses the link."""
    stage = _port_tree(tmp_path)
    twin = tmp_path / "twin"
    twin.mkdir()
    (twin / "helper.py").write_text("B = 2\n")
    real = safeio.read_beneath
    swapped = []

    def swapping(fd, parts, **kw):
        if not swapped:
            swapped.append(True)
            os.rename(stage / "port" / "pkg", stage / "port" / "pkg.moved")
            os.symlink(twin, stage / "port" / "pkg")
        return real(fd, parts, **kw)

    monkeypatch.setattr(safeio, "read_beneath", swapping)
    assert port_repair.current_sources(stage) is None


def test_a_port_root_swapped_for_another_directory_after_the_pin_is_not_reported_as_current(tmp_path, monkeypatch):
    stage = _port_tree(tmp_path)
    real = safeio.read_beneath
    swapped = []

    def swapping(fd, parts, **kw):
        if not swapped:
            swapped.append(True)
            os.rename(stage / "port", stage / "port.moved")
            os.makedirs(stage / "port" / "pkg")
            (stage / "port" / "port_model.py").write_text("A = 1\n")
            (stage / "port" / "pkg" / "helper.py").write_text("B = 2\n")
        return real(fd, parts, **kw)

    monkeypatch.setattr(safeio, "read_beneath", swapping)
    assert port_repair.current_sources(stage) is None


def test_a_link_or_a_linked_root_in_the_port_tree_is_still_refused(tmp_path):
    stage = _port_tree(tmp_path)
    os.symlink(stage / "port" / "port_model.py", stage / "port" / "alias.py")
    assert port_repair.current_sources(stage) is None
    stage2 = tmp_path / "other" / "stages" / "2"
    stage2.mkdir(parents=True)
    os.symlink(stage / "port", stage2 / "port")
    assert port_repair.current_sources(stage2) is None


def test_pycache_is_left_out_and_a_missing_port_directory_gives_none(tmp_path):
    stage = _port_tree(tmp_path)
    (stage / "port" / "__pycache__").mkdir()
    (stage / "port" / "__pycache__" / "x.py").write_text("ignored\n")
    assert "stages/2/port/__pycache__/x.py" not in port_repair.current_sources(stage)
    assert port_repair.current_sources(tmp_path / "nowhere") is None


# ---- 502: restore_best checks nested supporting directories before it commits port_model.py -----------------------------------------------

from orchard import repair_help as rh


def _saved_port_with_a_nested_file(tmp_path):
    skeleton = b"# the skeleton\n"
    stage = tmp_path / "stages" / "2"
    (stage / "port" / "pkg").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_bytes(b"import pkg.helper\n")
    (stage / "port" / "pkg" / "helper.py").write_bytes(b"H = 1\n")
    entry = rh.save_best(stage)
    import shutil
    shutil.rmtree(stage / "port")
    (stage / "port").mkdir()
    (stage / "port" / "port_model.py").write_bytes(skeleton)
    return stage, entry, skeleton


def test_a_nested_directory_replaced_after_its_file_was_restored_refuses_the_restore_and_keeps_the_skeleton(tmp_path, monkeypatch):
    stage, entry, skeleton = _saved_port_with_a_nested_file(tmp_path)
    real = safeio.write_beneath_if

    def write_then_swap(fd, parts, data, **kw):
        out = real(fd, parts, data, **kw)
        if tuple(parts) == ("pkg", "helper.py"):
            os.rename(stage / "port" / "pkg", stage / "port" / "pkg.moved")       # renamed away and recreated empty
            os.mkdir(stage / "port" / "pkg")
        return out

    monkeypatch.setattr(safeio, "write_beneath_if", write_then_swap)
    placed, why = rh.restore_best(stage, entry, skeleton)
    assert placed is False and "pkg/helper.py" in why
    assert (stage / "port" / "port_model.py").read_bytes() == skeleton          # the entry module was not committed


def test_a_supporting_file_whose_bytes_changed_after_the_write_refuses_the_restore(tmp_path, monkeypatch):
    stage, entry, skeleton = _saved_port_with_a_nested_file(tmp_path)
    real = safeio.write_beneath_if

    def write_then_edit(fd, parts, data, **kw):
        out = real(fd, parts, data, **kw)
        if tuple(parts) == ("pkg", "helper.py"):
            (stage / "port" / "pkg" / "helper.py").write_bytes(b"H = 999\n")
        return out

    monkeypatch.setattr(safeio, "write_beneath_if", write_then_edit)
    placed, why = rh.restore_best(stage, entry, skeleton)
    assert placed is False and "does not hold" in why
    assert (stage / "port" / "port_model.py").read_bytes() == skeleton


def test_an_undisturbed_restore_still_places_every_file(tmp_path):
    stage, entry, skeleton = _saved_port_with_a_nested_file(tmp_path)
    assert rh.restore_best(stage, entry, skeleton) == (True, "")
    assert (stage / "port" / "pkg" / "helper.py").read_bytes() == b"H = 1\n"
    assert (stage / "port" / "port_model.py").read_bytes() == b"import pkg.helper\n"


# ---- 461 (first part): shared helpers for a JSON number, and a count that may not grow -----------------------------------------------------

import itertools
import json
import math

HUGE = [10 ** 400, -10 ** 400]
NOT_NUMBERS = [None, "1", [1], {"a": 1}, True, False]


def test_is_finite_number_never_raises_and_says_what_a_number_is():
    for v in HUGE + [0, 1, -5, 1.5, 0.0]:
        assert safeio.is_finite_number(v) is True
    for v in [float("inf"), float("-inf"), float("nan")] + NOT_NUMBERS:
        assert safeio.is_finite_number(v) is False


def test_math_isfinite_really_does_raise_on_a_huge_int_which_is_why_the_helper_exists():
    with pytest.raises(OverflowError):
        math.isfinite(10 ** 400)


def test_as_float_converts_what_fits_and_refuses_what_does_not_without_an_overflow_error():
    assert safeio.as_float(3) == 3.0 and safeio.as_float(2.5) == 2.5
    for v in HUGE + [float("inf"), float("nan")] + NOT_NUMBERS:
        with pytest.raises(safeio.BadValue):
            safeio.as_float(v)


ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "isfinite_baseline.json"


def _count_isfinite(path: Path) -> int:
    import ast
    n = 0
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "isfinite"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "math"):
            n += 1
    return n


def _counts() -> dict:
    out = {}
    for f in sorted((ROOT / "orchard").rglob("*.py")):
        rel = f.relative_to(ROOT).as_posix()
        if rel == "orchard/safeio.py":
            continue                       # the helpers themselves
        n = _count_isfinite(f)
        if n:
            out[rel] = n
    return out


def test_no_file_gained_a_math_isfinite_call_that_a_huge_integer_can_crash():
    """Issue 461: math.isfinite(x) raises OverflowError for an integer of more than about 308 digits. New code uses safeio.is_finite_number
    (or as_float). A count may fall; it may not rise or appear in a new file. Lower tests/isfinite_baseline.json when a call site is replaced."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    grew = {rel: (baseline.get(rel, 0), n) for rel, n in _counts().items() if n > baseline.get(rel, 0)}
    assert not grew, "math.isfinite calls grew (was, now): " + ", ".join(f"{r} {a} -> {b}" for r, (a, b) in sorted(grew.items())) + \
        ". Use safeio.is_finite_number or safeio.as_float."


def test_the_counter_counts_what_it_says(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("import math\na = math.isfinite(x)\nb = math.isfinite(y) and math.isfinite(z)\nc = other.isfinite(w)\n# math.isfinite(q)\n")
    assert _count_isfinite(f) == 3
