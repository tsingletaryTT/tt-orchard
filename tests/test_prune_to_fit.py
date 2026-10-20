# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`bringup --prune-caches`: clear a disk-full block by removing regenerable tensor caches (autonomy, 2026-10-18).

A full disk was the most common stop that only a person could clear (RAG-Gate-27B: 109.6 GB free of 110 needed; the
Sonexa dry run: 89 of 137). The harness already knows which caches can be rebuilt. These tests prove what must never
happen: removing a cache an unfinished run uses, the shared kernel cache, anything outside the cache root, a partial
removal that still does not fit, and any removal on a dry run or without the flag."""
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import caches, cli
from orchard import preflight as pf
from test_caches import DAY, cache, run_using
from test_cli import Pipe, Recorder, conf, signals  # noqa: F401  (conf is a fixture)


@pytest.fixture
def box(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    runs = tmp_path / "runs"
    runs.mkdir()
    sizes = {}

    def measure(d):                                   # the GB of a directory is whatever the test says
        return sizes.get(Path(d).name, 1.0), os.stat(d).st_mtime
    monkeypatch.setattr(caches, "_measure", measure)
    return SimpleNamespace(root=root, runs=runs, cfg=SimpleNamespace(cache_root=root, runs_root=runs, lab=None),
                           sizes=sizes)


def make(box, name, gb, age_days):
    d = cache(box.root, name, f"org/{name}", age_days=age_days)
    box.sizes[name] = gb
    return d


def test_the_oldest_unused_caches_go_first_and_removal_stops_once_the_shortfall_is_covered(box):
    old, mid, new = make(box, "old", 30, 90), make(box, "mid", 30, 40), make(box, "new", 30, 5)
    ok, gone = caches.prune_to_fit(box.cfg, 45.0, say=lambda s: None)
    assert ok and [Path(c.path).name for c in gone] == ["old", "mid"]
    assert not old.exists() and not mid.exists() and new.exists()


def test_a_cache_an_unfinished_run_uses_is_never_removed_even_when_it_would_make_the_disk_fit(box):
    used, spare = make(box, "used", 100, 90), make(box, "spare", 10, 5)
    run_using(box.runs, "live", used)
    ok, gone = caches.prune_to_fit(box.cfg, 50.0, say=lambda s: None)
    assert not ok and gone == [] and used.exists() and spare.exists()


def test_a_cache_of_a_finished_run_is_regenerable_and_may_go(box):
    done = make(box, "done", 60, 90)
    run_using(box.runs, "past", done, finished=True)
    ok, gone = caches.prune_to_fit(box.cfg, 50.0, say=lambda s: None)
    assert ok and not done.exists()


def test_the_shared_kernel_cache_is_never_removed(box):
    kernels = box.root / "kernels"
    kernels.mkdir(parents=True)
    (kernels / "k.bin").write_bytes(b"x")
    box.sizes["kernels"] = 500
    ok, gone = caches.prune_to_fit(box.cfg, 50.0, say=lambda s: None)
    assert not ok and gone == [] and kernels.exists()


def test_nothing_is_removed_when_everything_unused_would_still_not_cover_the_shortfall(box):
    a, b = make(box, "a", 10, 90), make(box, "b", 10, 40)
    ok, gone = caches.prune_to_fit(box.cfg, 50.0, say=lambda s: None)
    assert not ok and gone == [] and a.exists() and b.exists()


def test_a_cache_outside_the_cache_root_is_never_removed(box, monkeypatch, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setattr(caches, "local_caches", lambda root: [caches.Cache(str(outside), "org/x", 100.0, 1.0)])
    ok, gone = caches.prune_to_fit(box.cfg, 50.0, say=lambda s: None)
    assert not ok and outside.exists()


def test_a_dry_run_names_what_would_go_and_removes_nothing(box):
    old = make(box, "old", 60, 90)
    said = []
    ok, gone = caches.prune_to_fit(box.cfg, 50.0, say=said.append, dry_run=True)
    assert ok and [Path(c.path).name for c in gone] == ["old"] and old.exists()
    assert any("would remove" in s for s in said)


def test_each_removal_is_said_with_its_size_and_path(box):
    old = make(box, "old", 60, 90)
    said = []
    caches.prune_to_fit(box.cfg, 50.0, say=said.append)
    assert any(str(old) in s and "60" in s and "removed" in s for s in said)


# ---- the shortfall the disk check reports ----------------------------------------------------

def test_a_disk_block_reports_how_short_each_disk_is_and_whether_they_share_one():
    c = pf.check_disk(free_gb={"hf_home": 100.0, "cache_root": 90.0}, model_gb=5.0, local=False, min_free_gb=110.0,
                      same_device=False)
    assert c.status == pf.BLOCK and c.data == {"short_gb": {"hf_home": 15.0, "cache_root": 20.0}, "same_device": False}


def test_pruning_can_fix_a_shortfall_on_the_cache_disk_or_on_a_disk_both_share():
    cache_only = pf.Check("disk", pf.BLOCK, "x", "disk-full", {"short_gb": {"cache_root": 20.0}, "same_device": False})
    assert pf.prunable_shortfall(cache_only) == 20.0
    shared = pf.Check("disk", pf.BLOCK, "x", "disk-full", {"short_gb": {"hf_home": 30.0}, "same_device": True})
    assert pf.prunable_shortfall(shared) == 30.0


def test_pruning_cannot_fix_a_shortfall_on_the_hub_disk_when_the_caches_are_elsewhere():
    both = pf.Check("disk", pf.BLOCK, "x", "disk-full",
                    {"short_gb": {"hf_home": 15.0, "cache_root": 20.0}, "same_device": False})
    assert pf.prunable_shortfall(both) is None
    hf_only = pf.Check("disk", pf.BLOCK, "x", "disk-full", {"short_gb": {"hf_home": 15.0}, "same_device": False})
    assert pf.prunable_shortfall(hf_only) is None
    assert pf.prunable_shortfall(pf.Check("disk", pf.OK, "x")) is None


# ---- the command ------------------------------------------------------------------------------

class World:
    """free_gb falls short until caches are removed, as the real disk would."""

    def __init__(self, box, start_free):
        self.box, self.start = box, start_free

    def free(self, path):
        gone = sum(box_gb for name, box_gb in self.box.sizes.items() if not (self.box.root / name).exists())
        return self.start + gone


def go(conf, box, world, *extra, **sig):
    out, rec = Pipe(), Recorder()
    sig.setdefault("same_device", lambda a, b: True)             # one disk holds the hub cache and the caches
    s = signals(free_gb=world.free, **sig)
    code = cli.main(["--config", str(conf), "bringup", "Cloudflare/clef", *extra], env={}, stdout=out, signals=s,
                    supervisor_main=rec.supervisor, fetcher=rec.fetcher, sleep=lambda s: None)
    return code, out.getvalue(), rec


@pytest.fixture
def real_caches(box, conf, monkeypatch):
    """The config's cache_root is the box's root; the used-by scan reads the box's runs."""
    text = conf.read_text().replace(f'cache_root = "{conf.parent}/cache"', f'cache_root = "{box.root}"')
    text = text.replace(f'runs_root = "{conf.parent}/runs"', f'runs_root = "{box.runs}"')
    conf.write_text(text)
    return conf


def test_with_the_flag_a_disk_only_block_is_cleared_by_pruning_and_the_run_starts(real_caches, box):
    old = make(box, "old", 130, 90)
    code, out, rec = go(real_caches, box, World(box, 20.0), "--prune-caches")
    assert code == 0 and rec.argv is not None and not old.exists()
    assert "removed" in out


def test_without_the_flag_nothing_is_removed_and_the_run_is_refused(real_caches, box):
    old = make(box, "old", 130, 90)
    code, out, rec = go(real_caches, box, World(box, 20.0))
    assert code == 2 and rec.argv is None and old.exists()


def test_a_dry_run_with_the_flag_removes_nothing(real_caches, box):
    old = make(box, "old", 130, 90)
    code, out, rec = go(real_caches, box, World(box, 20.0), "--prune-caches", "--dry-run")
    assert code == 2 and old.exists() and "would remove" in out


def test_another_block_means_nothing_is_pruned(real_caches, box):
    old = make(box, "old", 130, 90)
    code, out, rec = go(real_caches, box, World(box, 20.0), "--prune-caches", credentials=lambda: [Path("/h/.netrc")],
                        port_in_use=lambda p: True)
    assert code == 2 and old.exists()


def test_a_prune_that_cannot_cover_the_shortfall_removes_nothing_and_refuses(real_caches, box):
    small = make(box, "small", 5, 90)
    code, out, rec = go(real_caches, box, World(box, 20.0), "--prune-caches")
    assert code == 2 and rec.argv is None and small.exists()


def test_a_shortfall_on_a_hub_disk_that_the_caches_do_not_share_is_not_pruned(real_caches, box):
    old = make(box, "old", 130, 90)
    code, out, rec = go(real_caches, box, World(box, 20.0), "--prune-caches", same_device=lambda a, b: False)
    assert code == 2 and old.exists()
