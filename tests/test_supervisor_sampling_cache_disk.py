# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fourth review of PR 114, finding B (issue 107): the sampling trial writes a new tensor cache beside the stage's configured `tt_cache`, which can be on another
filesystem than the run directory. The disk check reads the nearest existing parent of that cache path for the conversion, and budgets the run directory
separately (a small amount) for evidence."""
import json
from collections import namedtuple
from pathlib import Path

import pytest

from orchard import defaults, sampling_choice as sc, supervisor
from orchard.supervisor import EXIT_READY
from run_fakes import bringup
from test_supervisor_sampling import TrialSpy, decision_file, overrides, rig, sd, spy  # noqa: F401
from test_supervisor import Rig

pytestmark = pytest.mark.usefixtures("stub_tools")

U = namedtuple("U", "total used free")


def with_cache(tmp_path, rig_, cache_rel="cachevol/models/tt_cache"):
    cache = tmp_path / cache_rel
    (tmp_path / "cachevol").mkdir(exist_ok=True)           # the volume exists; the cache directories under it are made later
    rig_.script = lambda req: bringup(req, overrides=overrides(extra2={"swap_config.json": {"bundle_dir": "/nonexistent/bundle", "tt_cache": str(cache)}}))
    return cache


def usage_by_path(rig_, cache, *, cache_gb, run_gb=10 ** 6):
    """Free space by filesystem: anything under the cache volume has `cache_gb`, everything else is plentiful. Records each path asked."""
    asked = []
    vol = cache.parent.parent

    def usage(path):
        asked.append(Path(path))
        return U(0, 0, int((cache_gb if str(path).startswith(str(vol)) else run_gb) * 1e9))
    rig_.usage = usage
    return asked


def test_a_full_cache_volume_skips_the_trial_although_the_run_directory_has_room(tmp_path, rig, spy):
    cache = with_cache(tmp_path, rig)
    usage_by_path(rig, cache, cache_gb=5)
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert d["flag"] == "variant-not-tried" and "configured tensor cache's disk" in d["reason"] and "5.0 GB is free" in d["reason"]
    assert spy.sampling_trials == []


def test_a_roomy_cache_volume_runs_the_trial_although_the_run_directory_is_short_for_the_trials_cache(tmp_path, rig, spy, monkeypatch):
    """The trial's threshold is made 300 GB. The run directory has 200 GB (enough for every stage entry), the cache volume 500 GB: the cache decides."""
    cache = with_cache(tmp_path, rig)
    usage_by_path(rig, cache, cache_gb=500, run_gb=200)
    monkeypatch.setattr(supervisor, "SAMPLING_TRIAL_DISK_GB", 300.0)
    assert rig.run() == EXIT_READY
    assert len(spy.sampling_trials) == 1 and decision_file(rig)["choice"] == "drafter-off"


def test_the_path_asked_for_the_cache_is_an_existing_parent_of_the_configured_cache(tmp_path, rig, spy):
    cache = with_cache(tmp_path, rig)
    assert not cache.exists() and (tmp_path / "cachevol").is_dir()
    asked = usage_by_path(rig, cache, cache_gb=500)
    rig.run()
    under_vol = [p for p in asked if str(p).startswith(str(tmp_path / "cachevol"))]
    assert under_vol and all(p.exists() for p in under_vol)
    assert tmp_path / "cachevol" in under_vol


def test_the_run_directory_is_checked_separately_for_evidence(tmp_path, rig, spy, monkeypatch):
    cache = with_cache(tmp_path, rig)
    usage_by_path(rig, cache, cache_gb=500)
    real = supervisor.check_disk
    monkeypatch.setattr(supervisor, "SAMPLING_TRIAL_RUN_DISK_GB", 54321.0)
    monkeypatch.setattr(supervisor, "check_disk", lambda path, gb, usage=None: (False, 3.0) if gb == 54321.0 and Path(path) == rig.run_dir else real(path, gb, usage=usage))
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert d["flag"] == "variant-not-tried" and "run directory" in d["reason"] and "54321 GB" in d["reason"] and "3.0 GB is free" in d["reason"]
    assert spy.sampling_trials == []


def test_a_missing_cache_setting_falls_back_to_the_run_directory(tmp_path, rig, spy, monkeypatch):
    """A swap_config.json with no tt_cache (or none at all) leaves the one check that existed: the run directory."""
    real = supervisor.check_disk
    monkeypatch.setattr(supervisor, "SAMPLING_TRIAL_DISK_GB", 12345.0)
    monkeypatch.setattr(supervisor, "check_disk", lambda path, gb, usage=None: (False, 7.0) if gb == 12345.0 else real(path, gb, usage=usage))
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert "run directory's disk" in d["reason"] and "7.0 GB is free" in d["reason"]


def test_the_trial_that_already_ran_is_not_checked_for_either_disk(tmp_path, rig, spy, monkeypatch):
    from test_supervisor_sampling_review import short_disk          # noqa: F401
    from fakes import Crash
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "evidence" and e["data"].get("what") == "sampling trial test")
    real = supervisor.check_disk
    monkeypatch.setattr(supervisor, "SAMPLING_TRIAL_DISK_GB", 12345.0)
    monkeypatch.setattr(supervisor, "SAMPLING_TRIAL_RUN_DISK_GB", 54321.0)
    monkeypatch.setattr(supervisor, "check_disk", lambda path, gb, usage=None: (False, 1.0) if gb in (12345.0, 54321.0) else real(path, gb, usage=usage))
    assert rig.run(pid=200) == EXIT_READY
    assert decision_file(rig)["choice"] == "drafter-off"


def test_the_evidence_budget_is_a_labelled_choice():
    assert defaults.SAMPLING_TRIAL_RUN_DISK_GB == 1.0
    line = next(ln for ln in open(defaults.__file__, encoding="utf-8").read().splitlines() if ln.startswith("SAMPLING_TRIAL_RUN_DISK_GB"))
    assert "choice" in line


def test_a_relative_cache_setting_is_read_against_the_run_directory(tmp_path, rig, spy):
    rig.script = lambda req: bringup(req, overrides=overrides(extra2={"swap_config.json": {"bundle_dir": "/nonexistent/bundle", "tt_cache": "relvol/tt_cache"}}))
    asked = []
    rig.usage = lambda path: asked.append(Path(path)) or U(0, 0, 10 ** 15)
    (rig.run_dir / "relvol").mkdir(parents=True)
    rig.run()
    assert rig.run_dir / "relvol" in asked
