# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A disk-full block names the largest things on the disk (issue 27, first wanted item).

On 2026-10-25 the preflight said "needs 202 GB, 116 GB free" and nothing else. The space to free was in the Hugging Face hub
cache (finished runs' weights), which `--prune-caches` does not touch. The hints are read-only: they list, and remove nothing.
"""
from pathlib import Path

from orchard import preflight as pf


def snap(hub: Path, repo: str, nbytes: int) -> None:
    d = hub / f"models--{repo.replace('/', '--')}" / "blobs"
    d.mkdir(parents=True)
    with open(d / "w", "wb") as f:
        f.truncate(nbytes)


def test_hub_models_are_listed_largest_first_with_sizes(tmp_path):
    hub = tmp_path / "hub"
    snap(hub, "a/small", 2_000_000_000)
    snap(hub, "b/big", 9_000_000_000)
    snap(hub, "c/mid", 5_000_000_000)
    got = pf.disk_hints(tmp_path, tmp_path / "nocache", exclude=[])
    assert [h[0] for h in got] == ["b/big", "c/mid", "a/small"]
    assert got[0][1] == 9.0


def test_the_model_being_brought_up_and_its_base_are_left_out(tmp_path):
    hub = tmp_path / "hub"
    snap(hub, "a/new", 9_000_000_000)
    snap(hub, "b/base", 8_000_000_000)
    snap(hub, "c/other", 1_000_000_000)
    got = pf.disk_hints(tmp_path, tmp_path / "nocache", exclude=["a/new", "b/base"])
    assert [h[0] for h in got] == ["c/other"]


def test_at_most_six_and_none_under_one_gb(tmp_path):
    hub = tmp_path / "hub"
    for i in range(9):
        snap(hub, f"o/m{i}", (2 + i) * 1_000_000_000)
    got = pf.disk_hints(tmp_path, tmp_path / "nocache", exclude=[])
    assert len(got) == 6 and [h[0] for h in got][0] == "o/m8"


def test_a_model_under_one_gb_is_not_listed(tmp_path):
    snap(tmp_path / "hub", "o/tiny", 900_000_000)
    assert pf.disk_hints(tmp_path, tmp_path / "nocache", exclude=[]) == []


def test_a_missing_hub_gives_no_hints_and_no_error(tmp_path):
    assert pf.disk_hints(tmp_path / "none", tmp_path / "nocache", exclude=[]) == []


def test_the_block_detail_lists_them_and_says_nothing_is_removed():
    hints = [("b/big", 9.0), ("c/mid", 5.0)]
    c = pf.check_disk(free_gb={"hf_home": 50.0, "cache_root": 500.0}, model_gb=100.0, local=False, min_free_gb=40.0,
                      same_device=False, hints=hints)
    assert c.status == "block" and c.reason == "disk-full"
    assert "b/big 9 GB" in c.detail and "c/mid 5 GB" in c.detail
    assert "removes none" in c.detail


def test_a_passing_check_and_a_block_without_hints_are_unchanged():
    ok = pf.check_disk(free_gb={"hf_home": 500.0, "cache_root": 500.0}, model_gb=1.0, local=False, min_free_gb=40.0,
                       same_device=False, hints=[("x/y", 9.0)])
    assert ok.status == "ok" and "x/y" not in ok.detail
    blk = pf.check_disk(free_gb={"hf_home": 50.0, "cache_root": 500.0}, model_gb=100.0, local=False, min_free_gb=40.0,
                        same_device=False)
    assert "removes none" not in blk.detail


def test_a_hint_failure_gives_no_hints_and_the_wiring_uses_it(monkeypatch):
    import inspect
    monkeypatch.setattr(pf, "disk_hints", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    assert pf.safe_disk_hints("h", "c", ["a/b", None]) == []
    assert "hints=safe_disk_hints(" in inspect.getsource(pf.run_preflight)


def test_the_exclude_list_drops_empty_entries(monkeypatch):
    seen = []
    monkeypatch.setattr(pf, "disk_hints", lambda h, c, ex: seen.append(ex) or [])
    pf.safe_disk_hints("h", "c", ["a/b", None, ""])
    assert seen == [["a/b"]]
