# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Readers and writers of a tree something else controls, against the hostile layouts of tests/hostile_fs.py.

Each registered reader runs once in a child process with an address-space limit and a per-case alarm; the per-pair tests read that
verdict. A reader that fails today is not weakened here: its pair is in `KNOWN_DEFECTS` with the issue that tracks it and is marked
xfail(strict=True), so the test turns red when the reader is fixed until the entry is removed and the list cannot go stale.

To add a reader: append a `PathReader` to `hostile_fs.PATH_READERS` and run this file. See docs/pr-hardening-checklist.md, the
symlink and special-file recipe.
"""
from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import hostile_fs as hf

# (reader name, case) -> the issue that tracks the defect and what is wrong
KNOWN_DEFECTS: dict[tuple[str, str], tuple[int, str]] = {
}

PAIRS = [(i, r.name, case) for i, r in enumerate(hf.PATH_READERS) for case in hf.EXPECT[r.family]]
_cache: dict[int, list[dict]] = {}


@pytest.fixture(scope="module")
def verdicts():
    """Every registered reader run once, in parallel, each in its own child process."""
    with tempfile.TemporaryDirectory(prefix="hfs-", dir=None) as tmp:
        def run(i):
            d = Path(tmp) / f"r{i}"
            d.mkdir()
            return i, hf.run_reader(i, d)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, got in pool.map(run, range(len(hf.PATH_READERS))):
                _cache[i] = got
        yield _cache


def _param(i: int, name: str, case: str):
    defect = KNOWN_DEFECTS.get((name, case))
    marks = [pytest.mark.xfail(strict=True, reason=f"issue {defect[0]}: {defect[1]}")] if defect else []
    return pytest.param(i, case, id=f"{name}::{case}", marks=marks)


@pytest.mark.parametrize("index,case", [_param(i, n, c) for i, n, c in PAIRS])
def test_reader_survives_the_hostile_layout(verdicts, index, case):
    reader = hf.PATH_READERS[index]
    got = {v["case"]: v for v in verdicts[index]}[case]
    want = hf.EXPECT[reader.family][case]
    assert got["verdict"] in want, (
        f"{reader.name} on {case}: {got['verdict']} (want one of {sorted(want)}). {got['detail']}. Case: {hf.CASES[case].why}")


# ---- the harness itself: each review mistake is seen to be caught

@pytest.fixture(scope="module")
def control_verdicts():
    with tempfile.TemporaryDirectory(prefix="hfs-ctl-") as tmp:
        out = {}
        for i, r in enumerate(hf.CONTROLS):
            d = Path(tmp) / f"c{i}"
            d.mkdir()
            out[r.name] = {v["case"]: v["verdict"] for v in hf.run_reader(i, d, controls=True)}
        yield out


def test_a_plain_open_follows_a_link_and_blocks_on_a_fifo(control_verdicts):
    got = control_verdicts["control: builtin open"]
    assert got["plain"] == "original"
    assert got["leaf_link"] == "leaked"                      # mistake 3 family: the link is followed
    assert got["fifo_named_file"] == "hang"                  # the FIFO blocks the read until the alarm


def test_o_nofollow_on_the_last_part_alone_misses_a_link_higher_up(control_verdicts):
    got = control_verdicts["control: O_NOFOLLOW on the last part only"]
    assert got["leaf_link"] == "refused"                     # the last part is protected
    assert got["ancestor_link"] == "leaked"                  # mistake 1: a link on the way is followed


def test_reading_by_pathname_after_a_pin_follows_the_swap(control_verdicts):
    got = control_verdicts["control: pin, then read by pathname"]
    assert got["plain"] == "original"
    assert got["swap_after_pin"] == "leaked"                 # mistake 2: the pin did not help


def test_walking_by_pathname_after_a_pin_follows_the_swap(control_verdicts):
    got = control_verdicts["control: pin, then walk by pathname"]
    assert got["plain"] == "original"
    assert got["swap_after_pin"] == "leaked"                 # the scan check can see a decoy: DECOY_MARK is matched on the right field


def test_deleting_a_file_or_a_directory_outside_the_root_is_a_leak(control_verdicts):
    # Review of PR 495, round 3: a reader that deletes the prepared outside objects must not pass as 'original' or 'refused' because the byte
    # comparison is skipped when the object is gone.
    for name in ("control: deletes the file outside the root", "control: deletes the directory outside the root",
                 "control: replaces the directory outside the root with a link"):
        got = control_verdicts[name]
        assert got and set(got.values()) == {"leaked"}, (name, got)


def test_replacing_the_file_outside_the_root_with_a_link_is_a_leak(control_verdicts):
    got = control_verdicts["control: replaces the file outside the root with a link"]
    assert got and set(got.values()) == {"leaked"}, got


def test_an_empty_listing_or_scan_is_not_a_pass(control_verdicts):
    # Review of PR 495: a reader that lists nothing used to be reported as `original` because nothing in the listing was a decoy.
    for name in ("control: lists nothing", "control: scans nothing"):
        got = control_verdicts[name]
        assert got["plain"] == "empty", (name, got)
    assert hf.classify_listing([]) == "empty" and hf.classify_listing(["data.json"]) == "original" and hf.classify_listing(["DECOY_MARK"]) == "decoy"
    assert hf.classify_scan([]) == "empty" and hf.classify_scan([("sub", "data.json")]) == "original"
    assert hf.classify_sidecars([], expect_listed="extra.gguf") == "empty" and hf.classify_sidecars([], expect_listed=None) == "original"


def test_a_blanket_handler_does_not_turn_a_fifo_timeout_into_a_refusal(control_verdicts):
    # Review of PR 495: the alarm was an Exception, so `except Exception` in a reader swallowed it.
    got = control_verdicts["control: swallows the alarm"]
    assert got["fifo_named_file"] == "hang", got


def test_a_change_to_any_fixture_file_outside_the_root_is_a_leak(control_verdicts):
    for name in ("control: rewrites a fixture file outside the root", "control: adds a file outside the root"):
        got = control_verdicts[name]
        assert got and set(got.values()) == {"leaked"}, (name, got)


def test_a_writer_that_publishes_nothing_is_not_reported_as_written(control_verdicts):
    got = control_verdicts["control: write that publishes nothing"]
    assert got and set(got.values()) == {"unpublished"}, got


def test_a_writer_that_omits_the_publication_check_is_not_reported_as_written(control_verdicts):
    # Review of PR 495, round 4: success with no way to verify the destination must not pass.
    got = control_verdicts["control: write that omits the publication check"]
    assert got and set(got.values()) == {"unpublished"}, got


def test_published_regular_wants_a_regular_file_holding_the_bytes(tmp_path):
    good = tmp_path / "good"
    good.write_bytes(b"new")
    assert hf.published_regular(good, b"new")
    assert not hf.published_regular(good, b"old")                           # wrong bytes
    assert not hf.published_regular(tmp_path / "missing", b"new")           # nothing there
    (tmp_path / "link").symlink_to(good)
    assert not hf.published_regular(tmp_path / "link", b"new")              # a link with the right bytes behind it is not a regular replacement
    (tmp_path / "dir").mkdir()
    assert not hf.published_regular(tmp_path / "dir", b"new")


def test_published_beneath_reads_through_the_pinned_descriptor(tmp_path):
    from orchard import safeio
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    with safeio.pinned_root(root) as pin:
        assert not hf.published_beneath(pin.fd, ("sub", "w.txt"), b"w")       # nothing written yet
        safeio.write_beneath(pin.fd, ("sub", "w.txt"), b"w")
        assert hf.published_beneath(pin.fd, ("sub", "w.txt"), b"w")
        assert not hf.published_beneath(pin.fd, ("sub", "w.txt"), b"other")


def test_the_cache_lock_name_is_the_one_the_lock_case_prepares(tmp_path):
    from orchard import cache_lock
    env = hf.layout(tmp_path)
    assert env["cache_lock"] == cache_lock.lock_path(env["root"], "ledger.jsonl")


def test_the_safe_readers_are_not_caught_by_the_same_cases(verdicts):
    names = {r.name: i for i, r in enumerate(hf.PATH_READERS)}
    got = {v["case"]: v["verdict"] for v in verdicts[names["safeio.read_bytes(anchor=pin)"]]}
    assert got["ancestor_link"] == "refused" and got["swap_after_pin"] == "original" and got["fifo_named_file"] == "refused"


# ---- the registry and the cases

def test_every_case_in_an_expectation_exists_and_every_family_has_one_that_must_succeed():
    for family, cases in hf.EXPECT.items():
        assert set(cases) <= set(hf.CASES), family
        assert any(v & {"original", "written"} and not v & {"refused"} for v in cases.values()), f"{family} has no control case that must succeed"


def test_readers_are_registered_once_and_use_a_known_family():
    names = [r.name for r in hf.PATH_READERS]
    assert len(names) == len(set(names))
    assert all(r.family in hf.EXPECT for r in hf.PATH_READERS + hf.CONTROLS)


def test_known_defects_name_a_registered_pair():
    pairs = {(n, c) for _, n, c in PAIRS}
    assert set(KNOWN_DEFECTS) <= pairs


@pytest.mark.parametrize("case", sorted(hf.CASES))
def test_each_case_builds_the_shape_it_says(tmp_path, case):
    env = hf.layout(tmp_path)
    hf.CASES[case].build(env)
    root = env["root"]
    if case == "leaf_link":
        assert env["file"].is_symlink()
    elif case == "ancestor_link":
        assert (root / "sub").is_symlink()
    elif case == "root_is_link":
        assert env["root"].is_symlink()
    elif case == "fifo_named_file":
        import stat
        assert stat.S_ISFIFO(env["file"].lstat().st_mode)
    elif case == "dir_for_file":
        assert env["file"].is_dir()
    elif case == "sparse_huge":
        assert env["file"].stat().st_size == hf.HUGE_SPARSE
    elif case == "lock_link":
        assert env["lock"].is_symlink() and env["cache_lock"].is_symlink()
        assert env["cache_lock"].name == ".ledger.jsonl.lock"
    elif case == "swap_after_pin":
        env["between"]()
        assert root.is_symlink() and (Path(str(root) + ".orig") / "sub" / "data.json").read_bytes() == hf.OK_BYTES
    elif case == "deep_tree":
        assert (root / "deep").is_dir()
    elif case == "hf_blob_inside":
        assert (env["snapshot"] / "model.safetensors").is_symlink()
        assert env["snapshot"].parent.name == "snapshots" and os.readlink(env["snapshot"] / "model.safetensors") == "../../blobs/abc123"


def test_a_refused_link_that_stays_listed_without_the_target_size_is_not_a_leak():
    ok = [{"file": "extra.gguf", "size": 17, "reason": "a link: not read"}]           # 17 is the length of the link's own text, not the target's size
    assert hf.sidecar_leaked(ok) is False
    assert hf.sidecar_leaked([{"file": "extra.gguf", "size": len(hf.VICTIM_BYTES), "reason": "x"}]) is True
    assert hf.sidecar_leaked([{"file": "extra.gguf", "size": 3, "sha256": "ab"}]) is True
    assert hf.sidecar_leaked([{"file": "other.gguf", "size": len(hf.VICTIM_BYTES)}]) is False
    assert hf.sidecar_leaked([]) is False
