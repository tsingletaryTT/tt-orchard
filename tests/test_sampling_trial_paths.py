# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 125 and 126 (from the review of PR 114): the sampling trial resolves inherited relative paths against the run directory before it builds the
variant, and the supervisor checks free space on the filesystem of the suffixed (`-nodraft`) cache the trial writes, not on the original cache's."""
import importlib.util
import json
from collections import namedtuple
from pathlib import Path

import pytest

from orchard import sampling_choice
from orchard.supervisor import EXIT_READY
from run_fakes import bringup
from test_sampling_trial_script import SCRIPT, rig as script_rig  # noqa: F401  (the script's own fixture)
from test_supervisor_sampling import TrialSpy, decision_file, overrides, rig, sd, spy  # noqa: F401

pytestmark = pytest.mark.usefixtures("stub_tools")

U = namedtuple("U", "total used free")

RELATIVE = {"bundle_dir": "inputs/bundle", "base_snapshot": "inputs/base", "new_snapshot": "inputs/new", "hf_home": "hf",
            "tt_cache": "caches/tt_cache", "suite_control": "stages/1/evidence/swap-check.json"}


# ---- issue 125: the template script -------------------------------------------------------------------------------------------

def _prepared_config(r):
    return next(c for c in r.calls() if c["tool"] == "prepare")["config"]


def test_relative_path_fields_become_absolute_against_the_run_directory_in_the_variant_config(script_rig):
    r = script_rig
    (r.stage / "swap_config.json").write_text(json.dumps({**r.cfg, **RELATIVE}))
    assert r.run().returncode == 0
    cfg = _prepared_config(r)
    for key, rel in RELATIVE.items():
        if key != "tt_cache":
            assert cfg[key] == str(r.run_dir / rel), key


def test_a_relative_tt_cache_gets_the_suffix_after_it_is_resolved_and_the_result_records_that_path(script_rig):
    r = script_rig
    (r.stage / "swap_config.json").write_text(json.dumps({**r.cfg, **RELATIVE}))
    r.run()
    want = str(r.run_dir / "caches" / "tt_cache-nodraft")
    assert _prepared_config(r)["tt_cache"] == want
    assert r.result()["variant"]["tt_cache"] == want


def test_an_absolute_path_field_is_left_as_it_is(script_rig):
    script_rig.run()
    assert _prepared_config(script_rig)["hf_home"] == "/hf"


def test_a_field_that_is_not_text_is_left_for_prepare_swap_to_refuse(script_rig):
    r = script_rig
    (r.stage / "swap_config.json").write_text(json.dumps({**r.cfg, "bundle_dir": 7, "suite_control": None}))
    assert r.run().returncode == 0
    cfg = _prepared_config(r)
    assert cfg["bundle_dir"] == 7 and cfg["suite_control"] is None


def test_the_cache_suffix_is_the_one_the_supervisor_checks():
    spec = importlib.util.spec_from_file_location("sampling_trial_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.CACHE_SUFFIX == sampling_choice.CACHE_SUFFIX == "-nodraft"


# ---- issue 126: the supervisor's disk check --------------------------------------------------------------------------------------

def _config(rig_, cache):
    rig_.script = lambda req: bringup(req, overrides=overrides(extra2={"swap_config.json": {"bundle_dir": "/nonexistent/bundle", "tt_cache": str(cache)}}))


def test_a_cache_symlinked_to_a_big_volume_is_checked_where_its_nodraft_sibling_lands(tmp_path, rig, spy):
    """`tt_cache` is a link into a big volume, but `<tt_cache>-nodraft` is made beside the link, on the small one."""
    small, big = tmp_path / "small", tmp_path / "big"
    (big / "real").mkdir(parents=True)
    small.mkdir()
    link = small / "tt_cache"
    link.symlink_to(big / "real")
    _config(rig, link)
    rig.usage = lambda path: U(0, 0, int((5 if Path(path).resolve().is_relative_to(small.resolve()) else 10 ** 6) * 1e9))   # statvfs follows links
    assert rig.run() == EXIT_READY
    d = decision_file(rig)
    assert d["flag"] == "variant-not-tried" and "configured tensor cache's disk" in d["reason"] and "5.0 GB is free" in d["reason"]
    assert spy.sampling_trials == []


def test_a_cache_on_a_small_volume_whose_sibling_is_on_a_big_one_is_not_refused(tmp_path, rig, spy):
    """The reverse: the link's target is small, the sibling lands on the big volume. The sibling's volume decides."""
    small, big = tmp_path / "small", tmp_path / "big"
    (small / "real").mkdir(parents=True)
    big.mkdir()
    link = big / "tt_cache"
    link.symlink_to(small / "real")
    _config(rig, link)
    rig.usage = lambda path: U(0, 0, int((5 if Path(path).resolve().is_relative_to(small.resolve()) else 10 ** 6) * 1e9))
    assert rig.run() == EXIT_READY
    assert len(spy.sampling_trials) == 1 and decision_file(rig)["choice"] == "drafter-off"


@pytest.mark.parametrize("cache", ["", "/"])
def test_a_cache_path_with_no_final_name_falls_back_to_the_run_directory(tmp_path, rig, spy, cache):
    _config(rig, cache)
    asked = []
    rig.usage = lambda path: asked.append(Path(path)) or U(0, 0, 10 ** 15)
    assert rig.run() == EXIT_READY
    assert rig.run_dir in asked and rig.run_dir.parent not in asked and Path("/") not in asked     # no sibling of the run directory is asked about


def test_the_path_asked_is_the_nearest_existing_ancestor_of_the_sibling_not_the_original_cache(tmp_path, rig, spy):
    cache = tmp_path / "vol" / "tt_cache"
    cache.mkdir(parents=True)                              # the original exists; the sibling does not
    _config(rig, cache)
    asked = []
    rig.usage = lambda path: asked.append(Path(path)) or U(0, 0, 10 ** 15)
    rig.run()
    assert cache not in asked and cache.parent in asked
