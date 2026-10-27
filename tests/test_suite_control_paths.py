# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 80, item 6: paths in the stage's config that are relative belong to the run directory, and the control runs from
`suite-control/`, so each inherited path is made absolute against the run directory first."""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_suite_control import CHECKOUT, _prep, ok_run, prep, stage_cfg  # noqa: E402,F401

from orchard import suite_control as sc  # noqa: E402

PATH_KEYS = ("bundle_dir", "base_snapshot", "new_snapshot", "tt_cache", "hf_home")


def relativize(prep, keys=PATH_KEYS):
    run = prep["stage"].parents[1]
    cfg = stage_cfg(prep)
    orig = dict(cfg)
    for k in keys:
        cfg[k] = os.path.relpath(cfg[k], run)
    cfg["run_dir"] = "."
    (prep["stage"] / "swap_config.json").write_text(json.dumps(cfg))
    return orig


def test_relative_paths_are_resolved_against_the_run_directory(prep):
    orig = relativize(prep)
    out = sc.prepare(prep["stage"], orchard_dir=CHECKOUT)     # the real prepare_swap.py runs from suite-control/
    cfg = out["config"]
    for k in ("bundle_dir", "base_snapshot", "hf_home"):
        assert cfg[k] == str(Path(orig[k]).resolve()), k
    assert cfg["new_snapshot"] == cfg["base_snapshot"]       # the control serves the nearest model
    assert cfg["tt_cache"] == str(Path(orig["tt_cache"]).resolve()) + sc.CACHE_SUFFIX
    assert cfg["run_dir"] == str(prep["stage"].parents[1].resolve())
    assert (Path(out["control_dir"]) / "model-dir" / "config.json").is_file()


def test_each_path_key_is_normalized_alone(prep):
    for key in PATH_KEYS:
        orig = relativize(prep, keys=(key,))
        cfg = sc.control_config(json.loads((prep["stage"] / "swap_config.json").read_text()), run_dir=prep["stage"].parents[1])
        assert os.path.isabs(cfg[key]), key
        (prep["stage"] / "swap_config.json").write_text(json.dumps(orig))


def test_operator_home_is_normalized_for_a_container(prep):
    c = stage_cfg(prep)
    c.pop("bundle_dir")
    run = prep["stage"].parents[1]
    c.update(package="org/pkg", profile="p", chips=4, operator_home="home")
    cfg = sc.control_config(c, run_dir=run)
    assert cfg["operator_home"] == str(run.resolve() / "home")


def test_absolute_paths_are_left_as_they_are(prep):
    cfg = sc.control_config(stage_cfg(prep), run_dir=prep["stage"].parents[1])
    assert cfg["bundle_dir"] == stage_cfg(prep)["bundle_dir"]


def test_a_relative_snapshot_equal_to_the_base_is_still_refused(prep):
    c = stage_cfg(prep)
    run = prep["stage"].parents[1]
    c["base_snapshot"] = os.path.relpath(c["base_snapshot"], run)
    c["new_snapshot"] = c["base_snapshot"]
    with pytest.raises(sc.ControlError, match="already serves the nearest"):
        sc.control_config(c, run_dir=run)


def test_the_run_directory_is_found_from_the_stage_path(tmp_path):
    assert sc.run_dir_of(tmp_path / "r" / "stages" / "2") == (tmp_path / "r").resolve()
    assert sc.run_dir_of(tmp_path / "r" / "stages" / "4" / "configs" / "2") == (tmp_path / "r").resolve()
    with pytest.raises(sc.ControlError, match="stages"):
        sc.run_dir_of(tmp_path / "elsewhere")


def test_the_last_stages_component_names_the_run(tmp_path):
    assert sc.run_dir_of(tmp_path / "stages" / "r" / "stages" / "2") == (tmp_path / "stages" / "r").resolve()
