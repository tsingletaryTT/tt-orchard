# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 663 (the folded tests of the PR 660 follow-up group): bundle directories are compared by real path.

`swap_draft._named_bundle` compares the run's `bundle` input with the installed bundles by `os.path.realpath`, and `nearest_problem` does the same for
an agent-written config's `bundle_dir`. A link to the same bundle matches. Replacing either comparison with a plain string comparison left every test
green, and a legitimate path that goes through a link would then be refused.
"""
import json
import os

from orchard import swap_draft as sd
from test_bundle_identity import CAPS, QWEN4_4, REPO, bundle, run  # noqa: F401


def test_the_run_bundle_input_may_be_a_link_to_an_installed_bundle(run, tmp_path):  # noqa: F811
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 2, CAPS)
    link = tmp_path / "linked-bundle"
    os.symlink(q4, link)
    cfg, problems = sd.draft(**{**run, "inputs": {**run["inputs"], "bundle": str(link)}})
    assert problems == [] and cfg["bundle_dir"] == str(q4)


def test_an_agent_written_bundle_dir_through_a_link_is_accepted_when_it_points_into_the_family(run, tmp_path):  # noqa: F811
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    stage = run["run_dir"] / "stages" / "2"
    stage.mkdir(parents=True)
    os.symlink(q4, stage / "bundle-link")
    (stage / "swap_config.json").write_text(json.dumps({"nearest_model_id": REPO, "bundle_dir": "stages/2/bundle-link"}))
    # a relative bundle_dir is read from the run directory, through the link
    (run["run_dir"] / "stages" / "2" / "swap_config.json").write_text(json.dumps({"nearest_model_id": REPO, "bundle_dir": "stages/2/bundle-link"}))
    assert sd.nearest_problem(run["run_dir"], stage / "swap_config.json", tt_model_root=run["tt_model_root"], bundle=str(q4)) is None


def test_a_link_to_a_bundle_outside_the_family_is_still_refused(run, tmp_path):  # noqa: F811
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    other = bundle(run["tt_model_root"], "someone/other-bundle", REPO, 4)         # the same weights repo, other capabilities: another family
    stage = run["run_dir"] / "stages" / "2"
    stage.mkdir(parents=True)
    os.symlink(other, stage / "bundle-link")
    (stage / "swap_config.json").write_text(json.dumps({"nearest_model_id": REPO, "bundle_dir": "stages/2/bundle-link"}))
    msg = sd.nearest_problem(run["run_dir"], stage / "swap_config.json", tt_model_root=run["tt_model_root"], bundle=str(q4))
    assert msg and "not" in msg and "family" in msg


def test_the_models_root_may_itself_be_reached_through_a_link(run, tmp_path):  # noqa: F811
    """The installed bundles are listed under the link, and the run's bundle input names the real directory: they are the same bundle."""
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 2, CAPS)
    root_link = tmp_path / "models-link"
    os.symlink(run["tt_model_root"], root_link)
    cfg, problems = sd.draft(**{**run, "tt_model_root": root_link, "inputs": {**run["inputs"], "bundle": str(q4)}})
    assert problems == [] and cfg["bundle_dir"] == str(root_link / QWEN4_4)


def test_an_agent_written_bundle_dir_under_a_linked_models_root_is_the_run_bundle(run, tmp_path):  # noqa: F811
    q4 = bundle(run["tt_model_root"], QWEN4_4, REPO, 4, CAPS)
    root_link = tmp_path / "models-link"
    os.symlink(run["tt_model_root"], root_link)
    stage = run["run_dir"] / "stages" / "2"
    stage.mkdir(parents=True)
    (stage / "swap_config.json").write_text(json.dumps({"nearest_model_id": REPO, "bundle_dir": str(q4)}))
    assert sd.nearest_problem(run["run_dir"], stage / "swap_config.json", tt_model_root=root_link, bundle=str(q4)) is None
