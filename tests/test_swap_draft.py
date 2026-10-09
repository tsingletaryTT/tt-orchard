# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/swap_draft.py: the supervisor drafts stage 2's swap_config.json before the agent starts.

Every field of the config is a fact the run already holds (stage 0's delta.json, the run's inputs and
paths, the installed bundles). On the lab run Coder-Next spent whole attempts re-deriving them, and on
the way kept investigating why the nearest bundle's architecture name differs from the new model's,
until the watchdog stopped it. Drafted by code, the agent starts from a correct config.
"""
import json
import socket
from pathlib import Path

import pytest

from orchard import swap_draft as sd

NEAREST = "Qwen/Qwen3.8-27B"
MODEL = "Altworld/Hemmingway-1@1a5f363a"


def bundle(root: Path, name: str, repo: str, chips: int, schema="6", run_sh=True):
    d = root / name
    d.mkdir(parents=True)
    (d / "tt_kernel_manifest.json").write_text(json.dumps(
        {"schema_version": schema, "device_count": chips, "weights": {"repo_id": repo}}))
    if run_sh:
        (d / "run.sh").write_text("#!/bin/bash\n")
    return d


@pytest.fixture
def run(tmp_path):
    run_dir = tmp_path / "runs" / "altworld--hemmingway-1"
    (run_dir / "stages" / "0").mkdir(parents=True)
    (run_dir / "stages" / "0" / "delta.json").write_text(json.dumps(
        {"model": MODEL, "nearest_model": NEAREST + "@1d4bf0f2", "path": "weights-only"}))
    models = tmp_path / "tt-model" / "models"
    base = tmp_path / "hf" / "hub" / "models--Qwen--Qwen3.8-27B" / "snapshots" / "1d4bf0f2"
    new = tmp_path / "hf" / "hub" / "models--Altworld--Hemmingway-1" / "snapshots" / "1a5f363a"
    base.mkdir(parents=True)
    new.mkdir(parents=True)
    return dict(run_dir=run_dir, tt_model_root=models, cache_root=tmp_path / "cache", hf_home=tmp_path / "hf",
                inputs={"base": str(base), "model": str(new)}, chips=2)


def test_the_draft_names_every_fact_the_swap_templates_need(run):
    b = bundle(run["tt_model_root"], "episod/qwen3.8-27b-dflash2-p300", NEAREST, 2)
    bundle(run["tt_model_root"], "episod/qwen3.8-27b-dflash2-p150", NEAREST, 1)
    bundle(run["tt_model_root"], "episod/qwen3.8-27b-p300x2", NEAREST, 4)
    bundle(run["tt_model_root"], "other/gemma-p300", "google/gemma-4-12B", 2)
    cfg, problems = sd.draft(**run)
    assert problems == []
    assert set(cfg) == set(sd.KEYS)
    assert cfg["run_dir"] == str(run["run_dir"])
    assert cfg["bundle_dir"] == str(b)                          # serves the nearest model on the stage's chips
    assert cfg["nearest_model_id"] == NEAREST                   # the repo, without a revision
    assert cfg["new_model_id"] == MODEL                         # stage 0's value verbatim, with its revision
    assert cfg["base_snapshot"] == run["inputs"]["base"] and cfg["new_snapshot"] == run["inputs"]["model"]
    assert cfg["tt_cache"] == str(run["cache_root"] / "altworld--hemmingway-1" / "tt_cache")
    assert cfg["hf_home"] == str(run["hf_home"])
    assert isinstance(cfg["port"], int) and 1024 < cfg["port"] < 65536


def test_the_keys_are_the_ones_the_skills_tell_the_agent_to_write():
    for skill in ("weights-swap-check.md", "weights-sidecar-check.md"):
        text = (Path(sd.__file__).parent / "skills" / skill).read_text()
        for key in sd.KEYS:
            assert f'"{key}"' in text, (skill, key)


def test_a_container_package_or_a_bundle_without_run_sh_is_not_chosen(run):
    bundle(run["tt_model_root"], "episod/qwen-v51", NEAREST, 2, schema="5.1")
    bundle(run["tt_model_root"], "episod/qwen-broken", NEAREST, 2, run_sh=False)
    cfg, problems = sd.draft(**run)
    assert cfg is None and any("no installed v6 bundle" in p and NEAREST in p for p in problems)


def test_two_bundles_that_both_fit_are_left_to_the_agent(run):
    bundle(run["tt_model_root"], "a/one", NEAREST, 2)
    bundle(run["tt_model_root"], "b/two", NEAREST, 2)
    cfg, problems = sd.draft(**run)
    assert cfg is None and any("a/one" in p and "b/two" in p for p in problems)


@pytest.mark.parametrize("missing", ["delta", "base", "model"])
def test_a_missing_fact_means_no_draft_and_says_which(run, missing):
    bundle(run["tt_model_root"], "episod/qwen3.8-27b-dflash2-p300", NEAREST, 2)
    if missing == "delta":
        (run["run_dir"] / "stages" / "0" / "delta.json").unlink()
    else:
        run["inputs"] = {k: v for k, v in run["inputs"].items() if k != missing}
    cfg, problems = sd.draft(**run)
    assert cfg is None and problems
    assert any((missing if missing != "delta" else "delta.json") in p for p in problems)


def test_a_snapshot_input_that_does_not_exist_is_a_problem(run):
    bundle(run["tt_model_root"], "episod/qwen3.8-27b-dflash2-p300", NEAREST, 2)
    run["inputs"]["model"] = str(Path(run["inputs"]["model"]).parent / "nope")
    cfg, problems = sd.draft(**run)
    assert cfg is None and any("does not exist" in p for p in problems)


def test_the_port_is_one_nothing_listens_on(run):
    bundle(run["tt_model_root"], "episod/qwen3.8-27b-dflash2-p300", NEAREST, 2)
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", sd.FIRST_PORT))
        s.listen()
        cfg, _ = sd.draft(**run)
    assert cfg["port"] != sd.FIRST_PORT
