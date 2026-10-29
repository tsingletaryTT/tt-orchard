# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 179: a stale entry in the audio reference cache must not block saving the fresh reference.

`audio_stages.cached_run` refuses an entry whose model revision, configuration, library commit or settings are no longer the run's. Stage 1 then
makes the CPU reference fresh. Before the fix the save step skipped the copy when `cache/<model>` existed, so every later run with the new identity
made its CPU references again. `supervisor.save_reference` now replaces the entry, safely: the new copy is complete before the old entry moves."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import shutil
import threading
import time

from orchard import audio_plan, cache_lock, supervisor
from orchard import audio_stages as st
from test_audio_stages import COMMIT, REV, make_run
from test_supervisor_audio_runtime import arig, make_environment  # noqa: F401  (arig is a fixture)
from orchard.supervisor import EXIT_READY

pytestmark = pytest.mark.usefixtures("stub_tools")

NEW_CONFIG = "a" * 64


def entries(cache):
    """The names in the cache directory, without the lock files (`.<model>.lock`) that the entry locks leave."""
    return sorted(p.name for p in cache.iterdir() if not p.name.endswith(cache_lock.LOCK_SUFFIX))


def identity(config_sha256=NEW_CONFIG):
    return {"revision": REV, "config_sha256": config_sha256}


def test_the_fresh_reference_replaces_a_stale_entry_and_the_next_run_uses_it(tmp_path):
    cache, fresh = tmp_path / "cache", tmp_path / "run" / "fresh"
    make_run(cache / "small-music", config_sha256="0" * 64)               # made for an earlier configuration
    make_run(fresh, config_sha256=NEW_CONFIG)
    assert st.cached_run(cache, "small-music", identity(), COMMIT) is None            # the stale entry is refused, as before
    assert supervisor.save_reference(fresh, cache, "small-music") is True
    assert st.cached_run(cache, "small-music", identity(), COMMIT) == cache / "small-music"      # the second run hits the cache
    assert st.cached_run(cache, "small-music", identity("0" * 64), COMMIT) is None
    assert entries(cache) == ["small-music"]                  # no temporary directory is left behind


def test_a_stale_entry_for_another_library_commit_is_replaced_too(tmp_path):
    cache, fresh = tmp_path / "cache", tmp_path / "fresh"
    make_run(cache / "small-music", config_sha256=NEW_CONFIG, commit="b" * 40)
    make_run(fresh, config_sha256=NEW_CONFIG)
    assert st.cached_run(cache, "small-music", identity(), COMMIT) is None
    assert supervisor.save_reference(fresh, cache, "small-music")
    assert st.cached_run(cache, "small-music", identity(), COMMIT) is not None


def test_an_entry_that_is_not_a_run_at_all_is_replaced(tmp_path):
    cache, fresh = tmp_path / "cache", tmp_path / "fresh"
    (cache / "small-music").mkdir(parents=True)
    (cache / "small-music" / "manifest.json").write_text("not json")
    make_run(fresh, config_sha256=NEW_CONFIG)
    assert supervisor.save_reference(fresh, cache, "small-music")
    assert st.cached_run(cache, "small-music", identity(), COMMIT) is not None


def test_a_link_in_the_cache_is_replaced_and_its_target_is_left_alone(tmp_path):
    cache, fresh, elsewhere = tmp_path / "cache", tmp_path / "fresh", tmp_path / "elsewhere"
    cache.mkdir()
    make_run(elsewhere, config_sha256="0" * 64)
    (cache / "small-music").symlink_to(elsewhere, target_is_directory=True)
    make_run(fresh, config_sha256=NEW_CONFIG)
    assert supervisor.save_reference(fresh, cache, "small-music")
    assert not (cache / "small-music").is_symlink() and st.cached_run(cache, "small-music", identity(), COMMIT) is not None
    assert (elsewhere / "manifest.json").is_file()
    assert entries(cache) == ["small-music"]            # the moved-aside link was removed


def test_the_other_models_entry_is_not_touched(tmp_path):
    cache, fresh = tmp_path / "cache", tmp_path / "fresh"
    make_run(cache / "small-music-base", "small-music-base", config_sha256=NEW_CONFIG)
    make_run(fresh, config_sha256=NEW_CONFIG)
    supervisor.save_reference(fresh, cache, "small-music")
    assert st.cached_run(cache, "small-music-base", identity(), COMMIT) is not None


def test_a_failed_save_leaves_the_old_entry_in_place(tmp_path, monkeypatch):
    """The old entry is kept until the new one is in place: when the final rename fails the old entry is put back."""
    cache, fresh = tmp_path / "cache", tmp_path / "fresh"
    make_run(cache / "small-music", config_sha256="0" * 64)
    make_run(fresh, config_sha256=NEW_CONFIG)
    real = os.rename

    def failing(src, dst, *a, **k):
        if Path(src).name.startswith(".small-music.tmp-"):
            raise OSError("disk went away")
        return real(src, dst, *a, **k)
    monkeypatch.setattr(os, "rename", failing)
    assert supervisor.save_reference(fresh, cache, "small-music") is False
    assert st.cached_run(cache, "small-music", identity("0" * 64), COMMIT) == cache / "small-music"      # the old entry is back and valid
    assert entries(cache) == ["small-music"]


def test_a_failed_copy_leaves_the_old_entry_and_no_temporary_directory(tmp_path):
    cache = tmp_path / "cache"
    make_run(cache / "small-music", config_sha256="0" * 64)
    assert supervisor.save_reference(tmp_path / "no-such-run", cache, "small-music") is False
    assert st.cached_run(cache, "small-music", identity("0" * 64), COMMIT) is not None
    assert entries(cache) == ["small-music"]


def test_leftovers_of_an_earlier_crash_in_the_same_process_id_do_not_block_the_save(tmp_path):
    cache, fresh = tmp_path / "cache", tmp_path / "fresh"
    (cache / f".small-music.tmp-{os.getpid()}").mkdir(parents=True)
    (cache / f".small-music.tmp-{os.getpid()}" / "junk").write_text("x")
    (cache / f".small-music.old-{os.getpid()}").mkdir()
    make_run(fresh, config_sha256=NEW_CONFIG)
    assert supervisor.save_reference(fresh, cache, "small-music")
    assert st.cached_run(cache, "small-music", identity(), COMMIT) is not None


def test_stage_1_replaces_a_stale_cache_entry_in_a_real_run(arig):          # noqa: F811
    """The whole path: the cache holds a reference made for another configuration, stage 1 makes a fresh one, and the cache ends up with an entry
    that a later run accepts."""
    make_environment(arig)
    cache = Path(arig.args.run_dir).parent / "cache" / "audio-reference"
    make_run(cache / "small-music", config_sha256="0" * 64)
    make_run(cache / "small-music-base", "small-music-base", config_sha256="0" * 64)
    assert arig.run() == EXIT_READY
    assert arig.calls["cpu"], "the stale entries were refused, so the CPU references were made"
    snap = [v.split("=", 1)[1] for v in arig.args.input if v.startswith("model=")][-1]
    base = [v.split("=", 1)[1] for v in arig.args.input if v.startswith("base=")][-1]
    assert st.cached_run(cache, "small-music", audio_plan.snapshot_identity(snap), supervisor.audio_reference_env.LIBRARY_COMMIT) is not None
    assert st.cached_run(cache, "small-music-base", audio_plan.snapshot_identity(base), supervisor.audio_reference_env.LIBRARY_COMMIT) is not None


# ---- the entry lock: a reader never sees the gap between the two renames (review of PR 191) ----------------------------------------------------

def run_in_thread(fn):
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("result", fn()), daemon=True)
    t.start()
    return t, box


def test_a_replacement_waits_for_a_run_that_is_copying_the_entry(tmp_path):
    cache, fresh = tmp_path / "cache", tmp_path / "fresh"
    make_run(cache / "small-music", config_sha256="0" * 64)
    make_run(fresh, config_sha256=NEW_CONFIG)
    with cache_lock.entry_lock(cache, "small-music", exclusive=False):          # a reader holds the shared lock
        t, box = run_in_thread(lambda: supervisor.save_reference(fresh, cache, "small-music"))
        time.sleep(0.4)
        assert t.is_alive() and "result" not in box, "the replacement went ahead while a run held the entry"
        assert st.cached_run(cache, "small-music", identity("0" * 64), COMMIT) is not None          # the old entry is untouched
    t.join(10)
    assert box["result"] is True and st.cached_run(cache, "small-music", identity(), COMMIT) is not None


def test_a_run_that_validates_and_copies_waits_for_a_replacement_in_progress(tmp_path):
    cache, out = tmp_path / "cache", tmp_path / "out"
    make_run(cache / "small-music", config_sha256=NEW_CONFIG)
    with cache_lock.entry_lock(cache, "small-music", exclusive=True):           # a writer is between its two renames
        t, box = run_in_thread(lambda: st.copy_cached_run(cache, "small-music", identity(), COMMIT, out))
        time.sleep(0.4)
        assert t.is_alive() and "result" not in box
    t.join(10)
    assert box["result"] == cache / "small-music" and (out / "manifest.json").is_file()


def test_copy_cached_run_gives_none_for_a_stale_entry_and_for_no_entry(tmp_path):
    cache = tmp_path / "cache"
    make_run(cache / "small-music", config_sha256="0" * 64)
    assert st.copy_cached_run(cache, "small-music", identity(), COMMIT, tmp_path / "o1") is None and not (tmp_path / "o1").exists()
    assert st.copy_cached_run(cache, "small-music-base", identity(), COMMIT, tmp_path / "o2") is None
    assert st.copy_cached_run(None, "small-music", identity(), COMMIT, tmp_path / "o3") is None


def test_a_copy_that_fails_leaves_nothing_behind_and_the_run_is_made_fresh(tmp_path, monkeypatch):
    cache, out = tmp_path / "cache", tmp_path / "out"
    make_run(cache / "small-music", config_sha256=NEW_CONFIG)
    real = shutil.copytree

    def failing(src, dst, *a, **k):
        real(src, dst, *a, **k)
        raise OSError("the entry went away half way")
    monkeypatch.setattr(shutil, "copytree", failing)
    assert st.copy_cached_run(cache, "small-music", identity(), COMMIT, out) is None
    assert not out.exists()


def test_the_lock_does_nothing_without_a_cache_directory_or_with_a_bad_name(tmp_path):
    for cache, name in ((None, "small-music"), (tmp_path / "nowhere", "small-music"), (tmp_path, ""), (tmp_path, "a/b")):
        with cache_lock.entry_lock(cache, name, exclusive=True) as held:
            assert held is False
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(cache_lock.LOCK_SUFFIX)] == []          # no lock file was made for any of them


def test_the_lock_file_is_made_in_the_cache_and_a_link_in_its_place_is_not_followed(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    target = tmp_path / "victim"
    target.write_text("keep")
    cache_lock.lock_path(cache, "small-music").symlink_to(target)
    with cache_lock.entry_lock(cache, "small-music", exclusive=True) as held:
        assert held is False                                     # O_NOFOLLOW refused the link, and the caller went on without the lock
    assert target.read_text() == "keep"
