# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 73, review round 3 of PR 442: every place that picks a package or bundle filters by eligibility BEFORE it ranks or narrows.

The draft used to narrow the installed containers to one chip count (the exact count, else the smallest larger one) and only then drop
the refused ones. A refused 2-chip package therefore hid a usable 4-chip package that the supported larger-chip fallback would have taken,
and the stage reported that nothing serves the base. The same order mattered in the other selectors, and the config check accepted any
package or bundle that served the right repository, whether or not it could serve swapped weights."""
import json

import pytest

from orchard import preflight as pf
from orchard import swap_draft as sd
from orchard import weights_source as ws
from orchard.nearest import Installed, read_installed
from test_weights_source_round2 import (BAD_NEAREST, GOOD_NEAREST, HARD, MODEL, NEAREST, OPEN, RECORDS, _bundle_dir,  # noqa: F401
                                        _containers, records, run)

OTHER_OPEN = "org/second-open-pkg"


def stage4(run, home, counts, **kw):
    return sd.draft_stage4(run_dir=run["run_dir"], tt_model_root=run["tt_model_root"], cache_root=run["cache_root"],
                           hf_home=run["hf_home"], operator_home=home, inputs=run["inputs"], counts=counts, **kw)


# ---- 1. eligibility before the chip-count fallback --------------------------------------------------------------------------------

def test_a_refused_two_chip_package_does_not_hide_a_usable_four_chip_one(run, tmp_path, records):
    home = _containers(tmp_path / "op", {HARD: (NEAREST, 2), OPEN: (NEAREST, 4)})
    cfg, problems = sd.draft(**run, chips=2, operator_home=home)
    assert problems == [], problems
    assert cfg["package"] == OPEN and cfg["chips"] == 4


def test_the_exact_chip_count_still_wins_among_the_usable_packages(run, tmp_path, records):
    home = _containers(tmp_path / "op", {HARD: (NEAREST, 2), OPEN: (NEAREST, 4), OTHER_OPEN: (NEAREST, 2)})
    cfg, problems = sd.draft(**run, chips=2, operator_home=home)
    assert problems == [] and cfg["package"] == OTHER_OPEN and cfg["chips"] == 2


def test_the_smallest_larger_count_is_taken_from_the_usable_packages_only(run, tmp_path, records):
    home = _containers(tmp_path / "op", {HARD: (NEAREST, 4), OPEN: (NEAREST, 8)})
    cfg, problems = sd.draft(**run, chips=2, operator_home=home)
    assert problems == [] and cfg["package"] == OPEN and cfg["chips"] == 8


def test_when_every_package_is_refused_nothing_is_chosen_and_the_message_says_so(run, tmp_path, records):
    home = _containers(tmp_path / "op", {HARD: (NEAREST, 2)})
    cfg, problems = sd.draft(**run, chips=2, operator_home=home)
    assert cfg is None and any("no installed" in p for p in problems)


# ---- 2. both draft selectors agree with the preflight about what can serve -------------------------------------------------------

def test_both_selectors_choose_only_what_the_preflight_judges_to_serve(run, tmp_path, records):
    good = _bundle_dir(run["tt_model_root"], "org/good-bundle", NEAREST, 2, GOOD_NEAREST)
    _bundle_dir(run["tt_model_root"], "org/bad-bundle", NEAREST, 1, BAD_NEAREST)
    home = _containers(tmp_path / "op", {HARD: (NEAREST, 2), OPEN: (NEAREST, 2)})
    installed = read_installed(run["tt_model_root"]) + [Installed(HARD, NEAREST, "5.1", 2), Installed(OPEN, NEAREST, "5.1", 2)]
    serving, refused = ws.judge(installed, NEAREST, RECORDS)       # the preflight's own judgement, bundles read from their directories
    served = {p.name for p in serving}
    assert served == {"org/good-bundle", OPEN} and {p.name for p, _ in refused} == {"org/bad-bundle", HARD}
    cfg2, problems = sd.draft(**run, chips=2, operator_home=home)
    assert problems == [] and cfg2["bundle_dir"] == str(good)                       # a bundle that fits is preferred, and it is a served one
    configs, _tests, notes = stage4(run, home, (1, 2))
    assert 2 in configs and configs[2].get("bundle_dir") == str(good)
    assert 1 not in configs and any("1-chip" in n for n in notes)                   # the refused 1-chip bundle is not taken
    assert set(c.get("package", c.get("bundle_dir")) for c in configs.values()) <= {str(good), OPEN}


# ---- 3. the config check and the note about a named package -----------------------------------------------------------------------

def _config(run, **named):
    cfg = {"run_dir": str(run["run_dir"]), "nearest_model_id": NEAREST, "new_model_id": MODEL, **named}
    path = run["run_dir"] / "stages" / "2" / "swap_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg))
    return path


def test_a_config_that_names_a_hard_coded_package_is_refused_with_the_reason(run, tmp_path, records):
    home = _containers(tmp_path / "op", {HARD: (NEAREST, 2)})
    path = _config(run, package=HARD, profile="default", chips=2, operator_home=str(home))
    msg = sd.nearest_problem(run["run_dir"], path, tt_model_root=run["tt_model_root"], operator_home=home)
    assert msg and HARD in msg and "swapped weights" in msg and "hard" in msg.lower()


def test_a_config_that_names_a_usable_package_is_still_accepted(run, tmp_path, records):
    home = _containers(tmp_path / "op", {OPEN: (NEAREST, 2)})
    path = _config(run, package=OPEN, profile="default", chips=2, operator_home=str(home))
    assert sd.nearest_problem(run["run_dir"], path, tt_model_root=run["tt_model_root"], operator_home=home) is None


def test_a_config_that_names_a_bundle_whose_run_sh_cannot_take_swapped_weights_is_refused(run, tmp_path, records):
    d = _bundle_dir(run["tt_model_root"], "org/bad-bundle", NEAREST, 2, BAD_NEAREST)
    path = _config(run, bundle_dir=str(d))
    msg = sd.nearest_problem(run["run_dir"], path, tt_model_root=run["tt_model_root"], operator_home=None)
    assert msg and "run.sh" in msg


def test_a_config_that_names_a_usable_bundle_is_still_accepted(run, tmp_path, records):
    d = _bundle_dir(run["tt_model_root"], "org/good-bundle", NEAREST, 2, GOOD_NEAREST)
    path = _config(run, bundle_dir=str(d))
    assert sd.nearest_problem(run["run_dir"], path, tt_model_root=run["tt_model_root"], operator_home=None) is None


def test_stage_4_says_a_named_four_chip_package_is_refused_and_not_that_it_is_missing(run, tmp_path, records):
    home = _containers(tmp_path / "op", {HARD: (NEAREST, 4)})
    configs, _tests, notes = stage4(run, home, (4,), four_chip_package=HARD)
    assert configs == {} and any(HARD in n and "swapped weights" in n for n in notes), notes


def test_a_bundle_named_relative_to_the_run_directory_is_judged_where_the_test_will_run_it(run, tmp_path, records):
    _bundle_dir(run["run_dir"], "rel-bundle", NEAREST, 2, BAD_NEAREST)
    path = _config(run, bundle_dir="rel-bundle")                                    # the test runs from the run directory
    msg = sd.nearest_problem(run["run_dir"], path, tt_model_root=run["tt_model_root"], operator_home=None)
    assert msg and "rel-bundle" in msg and "run.sh" in msg


def test_a_broken_record_file_is_named_by_the_config_check_and_does_not_raise(run, tmp_path, monkeypatch):
    def broken():
        raise ValueError("weights_source record: schema must be 1, got 2")
    monkeypatch.setattr(ws, "_shipped", broken)
    home = _containers(tmp_path / "op", {OPEN: (NEAREST, 2)})
    path = _config(run, package=OPEN, profile="default", chips=2, operator_home=str(home))
    msg = sd.nearest_problem(run["run_dir"], path, tt_model_root=run["tt_model_root"], operator_home=home)
    assert msg and "schema must be 1" in msg


def test_a_config_that_names_neither_a_package_nor_a_bundle_does_not_read_the_record(run, tmp_path, monkeypatch):
    def boom():
        raise AssertionError("the record must not be read for a config that names neither")
    monkeypatch.setattr(ws, "_shipped", boom)
    path = _config(run)
    assert sd.nearest_problem(run["run_dir"], path, tt_model_root=run["tt_model_root"], operator_home=None) is None
