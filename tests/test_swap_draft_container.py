# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 on a container (v5.1) package (issue 60).

49 of the 50 packages in the tt-model-manager catalog are container packages. Stage 4 already serves swapped weights through one
(serve_and_compare_container.py) for its 4-chip test, but stage 2 drafted only a v6 bundle's config, so a model whose nearest model is
container-packaged stopped at the base check. The draft now falls back to the one installed container package that serves the nearest
model on the stage's chips. A v6 bundle that fits is always preferred."""
import json
from pathlib import Path

import pytest

from orchard import swap_draft as sd
from test_swap_draft import MODEL, NEAREST, bundle, containers, run  # noqa: F401


def draft(run, home, **kw):
    return sd.draft(**run, operator_home=home, **kw)


def test_a_container_package_that_serves_the_nearest_model_on_the_chips_is_drafted(run, tmp_path):
    home = containers(tmp_path / "op", {"someone/qwen3.8-27b-p300": (NEAREST, 2, "default")})
    cfg, problems = draft(run, home)
    assert problems == [] and set(cfg) == set(sd.CONTAINER_KEYS)
    assert (cfg["package"], cfg["profile"], cfg["chips"]) == ("someone/qwen3.8-27b-p300", "default", 2)
    assert cfg["operator_home"] == str(home) and cfg["nearest_model_id"] == NEAREST and cfg["new_model_id"] == MODEL
    assert cfg["base_snapshot"] == run["inputs"]["base"] and cfg["new_snapshot"] == run["inputs"]["model"]
    assert cfg["hf_home"] == str(run["hf_home"]) and cfg["prompt_suite"] is True and cfg["run_dir"] == str(run["run_dir"])
    assert cfg["tt_cache"] == str(run["cache_root"] / "altworld--hemmingway-1" / "2chip-someone--qwen3.8-27b-p300" / "tt_cache")
    assert isinstance(cfg["port"], int) and 1024 < cfg["port"] < 65536
    assert "bundle_dir" not in cfg


def test_a_v6_bundle_that_fits_is_preferred_to_a_container_package(run, tmp_path):
    b = bundle(run["tt_model_root"], "episod/qwen3.8-27b-dflash2-p300", NEAREST, 2)
    home = containers(tmp_path / "op", {"someone/qwen3.8-27b-p300": (NEAREST, 2, "default")})
    cfg, problems = draft(run, home)
    assert problems == [] and cfg["bundle_dir"] == str(b) and "package" not in cfg and set(cfg) == set(sd.KEYS)


def test_without_an_operator_home_no_container_is_considered(run, tmp_path):
    containers(tmp_path / "op", {"someone/qwen3.8-27b-p300": (NEAREST, 2, "default")})
    cfg, problems = sd.draft(**run)
    assert cfg is None and any("no installed v6 bundle" in p for p in problems)


def test_a_container_for_another_chip_count_or_another_model_is_not_chosen(run, tmp_path):
    home = containers(tmp_path / "op", {"a/one-p300x2": (NEAREST, 4, "default"), "b/other-p300": ("other/Model", 2, "default")})
    cfg, problems = draft(run, home)
    assert cfg is None
    assert any("no installed v6 bundle" in p and "container package" in p and NEAREST in p and "2 chips" in p for p in problems), problems


def test_two_container_packages_that_both_fit_are_left_to_the_agent(run, tmp_path):
    home = containers(tmp_path / "op", {"a/one-p300": (NEAREST, 2, "x"), "b/two-p300": (NEAREST, 2, "y")})
    cfg, problems = draft(run, home)
    assert cfg is None and any("a/one-p300" in p and "b/two-p300" in p for p in problems), problems


def test_a_container_package_with_no_default_profile_is_not_drafted(run, tmp_path):
    home = containers(tmp_path / "op", {"a/one-p300": (NEAREST, 2, None)})
    cfg, problems = draft(run, home)
    assert cfg is None and any("a/one-p300" in p and "profile" in p for p in problems), problems


def test_several_v6_bundles_still_block_even_when_one_container_fits(run, tmp_path):
    bundle(run["tt_model_root"], "a/one", NEAREST, 2)
    bundle(run["tt_model_root"], "b/two", NEAREST, 2)
    home = containers(tmp_path / "op", {"c/three-p300": (NEAREST, 2, "default")})
    cfg, problems = draft(run, home)
    assert cfg is None and any("a/one" in p and "b/two" in p for p in problems)


def test_the_cache_name_matches_the_one_stage_4_uses_for_the_same_package_and_count(run, tmp_path):
    home = containers(tmp_path / "op", {"someone/qwen3.8-27b-p300": (NEAREST, 2, "default")})
    cfg, _ = draft(run, home)
    configs, _, notes = sd.draft_stage4(run_dir=run["run_dir"], tt_model_root=run["tt_model_root"], cache_root=run["cache_root"],
                                        hf_home=run["hf_home"], operator_home=home, inputs=run["inputs"], counts=(2,))
    assert notes == [] and configs[2]["tt_cache"] == cfg["tt_cache"]


def test_a_v6_bundle_that_fits_wins_even_when_several_container_packages_fit_too(run, tmp_path):
    b = bundle(run["tt_model_root"], "episod/qwen3.8-27b-dflash2-p300", NEAREST, 2)
    home = containers(tmp_path / "op", {"a/one-p300": (NEAREST, 2, "x"), "b/two-p300": (NEAREST, 2, "y")})
    cfg, problems = draft(run, home)
    assert problems == [] and cfg["bundle_dir"] == str(b)
