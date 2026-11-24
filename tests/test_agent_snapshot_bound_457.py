# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 457, the site named last: `AgentStep._snapshot` lists `evidence/` after every tool call.

It used `sorted(rglob("*"))` with no bound and kept one entry per file for the whole step. The directory is written by the agent, so a runaway
step could make the supervisor read and hold millions of entries. The snapshot now reads through `boundedwalk.Walk` with the same bounds as the
wrap-up's evidence listing (`boundedwalk.EVIDENCE_ENTRIES`, `EVIDENCE_DEPTH`). Links are never followed or listed, so a link inside `evidence/`
is not evidence the agent produced there. Past a bound the files read so far are kept and one ledger decision names the bound, once per step.
"""
import os

import pytest

from fake_model import FakeModel, final
from orchard import boundedwalk, supervisor
from test_agent import run, step  # noqa: F401
from test_bounded_port_scans import deep_tree, fake_directory


def make_step(run):  # noqa: F811
    run_dir, ledger = run
    s, _events, _ = step(run_dir, ledger, "http://127.0.0.1:1")
    return s, run_dir, ledger


def notes(ledger):
    return [e["data"] for e in ledger.read() if e["event"] == "decision" and e["data"].get("decision") == "the evidence scan stopped"]


def test_the_two_listings_of_evidence_share_their_bounds():
    assert supervisor.EVIDENCE_LIST_ENTRIES == boundedwalk.EVIDENCE_ENTRIES == 2000
    assert supervisor.EVIDENCE_LIST_DEPTH == boundedwalk.EVIDENCE_DEPTH == 12


def test_a_snapshot_inside_the_bounds_has_every_file_with_its_size_and_mtime(run):  # noqa: F811
    s, run_dir, _ = make_step(run)
    ev = run_dir / "stages" / "0" / "evidence"
    (ev / "sub").mkdir()
    (ev / "b.txt").write_text("bb")
    (ev / "sub" / "a.txt").write_text("a")
    snap = s._snapshot()
    assert list(snap) == sorted([str(ev / "b.txt"), str(ev / "sub" / "a.txt")])
    st = os.stat(ev / "b.txt")
    assert snap[str(ev / "b.txt")] == (st.st_size, st.st_mtime_ns)
    assert s._snapshot_limit is None


def test_a_link_inside_evidence_is_neither_followed_nor_listed(run, tmp_path):  # noqa: F811
    s, run_dir, _ = make_step(run)
    ev = run_dir / "stages" / "0" / "evidence"
    (ev / "real.txt").write_text("x")
    outside = tmp_path / "outside.txt"
    outside.write_text("y")
    (ev / "to-outside").symlink_to(outside)
    (ev / "to-inside").symlink_to(ev / "real.txt")
    (ev / "to-dir").symlink_to(tmp_path)
    assert list(s._snapshot()) == [str(ev / "real.txt")]


def test_an_evidence_directory_that_is_a_link_is_not_read(run, tmp_path):  # noqa: F811
    s, run_dir, _ = make_step(run)
    ev = run_dir / "stages" / "0" / "evidence"
    ev.rmdir()
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "secret.txt").write_text("s")
    ev.symlink_to(other)
    assert s._snapshot() == {}


@pytest.mark.timeout(60)
def test_a_directory_of_300000_entries_is_read_only_to_the_bound(run, monkeypatch):  # noqa: F811
    s, run_dir, _ = make_step(run)
    ev = run_dir / "stages" / "0" / "evidence"
    counter = fake_directory(monkeypatch, ev, 300_000)
    snap = s._snapshot()
    assert counter.pulled <= boundedwalk.EVIDENCE_ENTRIES + 1, counter.pulled
    assert len(snap) <= boundedwalk.EVIDENCE_ENTRIES and s._snapshot_limit and "2000" in s._snapshot_limit


@pytest.mark.timeout(60)
def test_a_tree_deeper_than_the_bound_stops_the_scan(run):  # noqa: F811
    s, run_dir, _ = make_step(run)
    deep_tree(run_dir / "stages" / "0" / "evidence", boundedwalk.EVIDENCE_DEPTH + 3)
    assert s._snapshot() == {} and "deeper" in s._snapshot_limit


def test_the_ledger_names_the_bound_once_per_step_and_not_on_every_call(run, monkeypatch):  # noqa: F811
    s, run_dir, ledger = make_step(run)
    ev = run_dir / "stages" / "0" / "evidence"
    for i in range(7):
        (ev / f"f{i}.txt").write_text("x")
    monkeypatch.setattr(boundedwalk, "EVIDENCE_ENTRIES", 3)
    monkeypatch.setattr("orchard.agent.boundedwalk.EVIDENCE_ENTRIES", 3)
    s._record_new_evidence()
    s._record_new_evidence()
    got = notes(ledger)
    assert len(got) == 1 and got[0]["phase"] == "run" and got[0]["bound"] and got[0]["read"] >= 3


def test_files_read_before_the_bound_are_still_recorded_as_evidence(run, monkeypatch):  # noqa: F811
    s, run_dir, ledger = make_step(run)
    ev = run_dir / "stages" / "0" / "evidence"
    for i in range(7):
        (ev / f"f{i}.txt").write_text("x")
    monkeypatch.setattr("orchard.agent.boundedwalk.EVIDENCE_ENTRIES", 3)
    s._record_new_evidence()
    recorded = [e for e in ledger.read() if e["event"] == "evidence"]
    assert 1 <= len(recorded) <= 3 and notes(ledger)


def test_a_new_file_is_recorded_once_and_an_unchanged_file_is_not_recorded_again(run):  # noqa: F811
    s, run_dir, ledger = make_step(run)
    ev = run_dir / "stages" / "0" / "evidence"
    (ev / "a.txt").write_text("a")
    s._record_new_evidence()
    s._record_new_evidence()
    assert len([e for e in ledger.read() if e["event"] == "evidence"]) == 1
    (ev / "a.txt").write_text("aa")
    s._record_new_evidence()
    assert len([e for e in ledger.read() if e["event"] == "evidence"]) == 2


def test_the_snapshot_is_sorted_by_path_whatever_order_the_directory_serves(run):  # noqa: F811
    s, run_dir, _ = make_step(run)
    ev = run_dir / "stages" / "0" / "evidence"
    for name in ["k", "c", "m", "a", "z", "e", "q", "b"]:
        (ev / f"{name}.txt").write_text(name)
    keys = list(s._snapshot())
    assert keys == sorted(keys) and len(keys) == 8
