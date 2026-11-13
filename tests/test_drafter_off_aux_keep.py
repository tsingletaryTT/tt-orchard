# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""With the drafter off, stage 7 removes only the drafter's entry from TT_AUXILIARY_WEIGHTS (issue 672).

Stage 2's prepare_swap.py removes the repo that `export DFLASH_WEIGHTS=` names, at any revision, from the comma-separated
`TT_AUXILIARY_WEIGHTS` list and keeps the other entries (weights the bundle's launcher downloads before vLLM starts). Stage 7 used to
drop the whole variable, so a bundle with a second, non-drafter entry was tested in stage 2 with that repo and packaged without it. These
tests hold stage 7 to the stage 2 rule: the package keeps the other entries, the package check refuses only a drafter entry, and the card
names no kept entry as the drafter."""
import json
import os
import sys
from pathlib import Path

import pytest

from orchard import package
from orchard.package import serving_env, serving_problems
from orchard.stages import gate_package
from package_fakes import DRAFTER, DRAFTER_REV, fake_bin, fake_boot_result, make_clef_like, make_run, make_source

DRAFTER_AT = f"{DRAFTER}@{DRAFTER_REV}"
OTHER = "acme/vision-tower@" + "1" * 40
ENV = {"ARCH_NAME": "blackhole", "QWEN36_DRAFTER": "dflash2", "DFLASH_WEIGHTS": DRAFTER_AT, "QWEN36_DFLASH_BLOCK": "8"}


# ---- serving_env -----------------------------------------------------------------------------------------------------------------

def test_a_two_entry_list_keeps_the_entry_that_is_not_the_drafter():
    env = serving_env({**ENV, "TT_AUXILIARY_WEIGHTS": f"{DRAFTER_AT}, {OTHER}"}, drafter_off=True)
    assert env["TT_AUXILIARY_WEIGHTS"] == OTHER
    assert "DFLASH_WEIGHTS" not in env and "QWEN36_DFLASH_BLOCK" not in env and env["QWEN36_DRAFTER"] == ""


def test_the_drafter_entry_goes_at_any_revision():
    """prepare_swap.py matches the drafter's repo at any revision; so does stage 7."""
    env = serving_env({**ENV, "TT_AUXILIARY_WEIGHTS": f"{OTHER},{DRAFTER}@" + "e" * 40 + f",{DRAFTER}"}, drafter_off=True)
    assert env["TT_AUXILIARY_WEIGHTS"] == OTHER


def test_a_list_that_names_only_the_drafter_leaves_the_package():
    env = serving_env({**ENV, "TT_AUXILIARY_WEIGHTS": DRAFTER_AT}, drafter_off=True)
    assert "TT_AUXILIARY_WEIGHTS" not in env


def test_without_a_drafter_repo_the_list_is_left_as_it_is():
    """prepare_swap.py leaves the list alone when there is no DFLASH_WEIGHTS to tell the drafter's entry apart."""
    env = {k: v for k, v in ENV.items() if k != "DFLASH_WEIGHTS"}
    assert serving_env({**env, "TT_AUXILIARY_WEIGHTS": f"{DRAFTER_AT},{OTHER}"}, drafter_off=True)["TT_AUXILIARY_WEIGHTS"] == \
        f"{DRAFTER_AT},{OTHER}"


# ---- the package check -----------------------------------------------------------------------------------------------------------

def test_the_package_check_accepts_a_kept_entry_that_is_not_the_drafter():
    env = serving_env({**ENV, "TT_AUXILIARY_WEIGHTS": f"{DRAFTER_AT},{OTHER}"}, drafter_off=True)
    run_sh = f'export TT_AUXILIARY_WEIGHTS="{OTHER}"\n'
    assert serving_problems(run_sh, env, drafter_off=True, host_sampling=False, drafters={DRAFTER}) == []


def test_the_package_check_refuses_a_drafter_entry_in_the_environment_or_run_sh():
    env = {"QWEN36_DRAFTER": "", "TT_AUXILIARY_WEIGHTS": f"{OTHER},{DRAFTER}@abc"}
    problems = serving_problems("", env, drafter_off=True, host_sampling=False, drafters={DRAFTER})
    assert any(DRAFTER in p for p in problems), problems
    run_sh = f'export TT_AUXILIARY_WEIGHTS="{OTHER},{DRAFTER_AT}"\n'
    problems = serving_problems(run_sh, {"QWEN36_DRAFTER": ""}, drafter_off=True, host_sampling=False, drafters={DRAFTER})
    assert any("run.sh" in p and DRAFTER in p for p in problems), problems


def test_an_unknown_drafter_refuses_every_auxiliary_entry():
    """When the caller cannot say which repo is the drafter, the check keeps the earlier rule and refuses any entry."""
    env = {"QWEN36_DRAFTER": "", "TT_AUXILIARY_WEIGHTS": OTHER}
    assert serving_problems("", env, drafter_off=True, host_sampling=False)
    assert serving_problems(f'export TT_AUXILIARY_WEIGHTS="{OTHER}"\n', {"QWEN36_DRAFTER": ""}, drafter_off=True, host_sampling=False)


# ---- the card and the boot check -------------------------------------------------------------------------------------------------

def test_the_card_does_not_name_a_kept_entry_as_the_drafter():
    env = serving_env({**ENV, "TT_AUXILIARY_WEIGHTS": f"{DRAFTER_AT},{OTHER}"}, drafter_off=True)
    assert package.aux_repos(env, nearest_model="Qwen/Qwen3.8-27B") == []


def test_the_boot_check_links_every_repo_the_list_names():
    env = {**ENV, "TT_AUXILIARY_WEIGHTS": f"{DRAFTER_AT}, {OTHER}"}
    assert package.download_repos(env, nearest_model="Qwen/Qwen3.8-27B") == sorted({DRAFTER, OTHER.split("@")[0]})


def test_stage_2s_drafter_is_read_from_its_run_script(tmp_path):
    assert package.stage_2_drafter_repos(tmp_path) is None
    s2 = tmp_path / "stages/2/run.sh"
    s2.parent.mkdir(parents=True)
    s2.write_text('export QWEN36_DRAFTER=""\n')
    assert package.stage_2_drafter_repos(tmp_path) == set()
    s2.write_text(f'export QWEN36_DRAFTER=""\nexport DFLASH_WEIGHTS="{DRAFTER_AT}"\n')
    assert package.stage_2_drafter_repos(tmp_path) == {DRAFTER}


# ---- end to end: staging and the gate --------------------------------------------------------------------------------------------

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")


@pytest.fixture
def world(tmp_path, monkeypatch, stub_tools):
    models = tmp_path / "models"
    src = make_source(models, env={"ARCH_NAME": "blackhole", "QWEN36_DRAFTER": "dflash2", "DFLASH_WEIGHTS": DRAFTER_AT,
                                   "QWEN36_DFLASH_TP": "1", "TT_AUXILIARY_WEIGHTS": f"{DRAFTER_AT},{OTHER}"})
    r = make_run(tmp_path, src)
    make_clef_like(r["run"], src)
    # Stage 2's copy has the drafter's entry removed from the list, as prepare_swap.py leaves it.
    s2 = r["run"] / "stages/2/run.sh"
    s2.write_text(s2.read_text().replace(f'TT_AUXILIARY_WEIGHTS="{DRAFTER_AT},{OTHER}"', f'TT_AUXILIARY_WEIGHTS="{OTHER}"'))
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    env = dict(os.environ, FAKE_TT_MODEL_PYTHON=sys.executable, FAKE_SWAP_SERVER=str(FAKE_SERVER),
               FAKE_SWAP_CONFIG=str(tmp_path / "server.json"))
    return {**r, "models": models, "src": src, "env": env}


def staged(w):
    stage = w["run"] / "stages/7"
    stage.mkdir(parents=True)
    package.stage_all(w["run"], stage, namespace="episod", models_root=w["models"], env=w["env"])
    fake_boot_result(stage)
    package.finish(w["run"], stage)
    return stage, stage / "package/hemmingway-1-p300"


def test_the_package_keeps_the_other_entry_and_the_gate_passes(world):
    stage, pkg = staged(world)
    m = json.loads((pkg / "tt_kernel_manifest.json").read_text())
    assert m["env"]["TT_AUXILIARY_WEIGHTS"] == OTHER
    card = (pkg / "README.md").read_text()
    assert DRAFTER not in card and "Speculative decoding uses" not in card
    g = gate_package(stage, world["run"])
    assert g.ok, g.reasons


def test_the_gate_refuses_a_package_that_kept_the_drafter_entry(world):
    stage, pkg = staged(world)
    mpath = pkg / "tt_kernel_manifest.json"
    m = json.loads(mpath.read_text())
    m["env"]["TT_AUXILIARY_WEIGHTS"] = f"{OTHER},{DRAFTER_AT}"
    mpath.write_text(json.dumps(m, indent=2))
    reasons = gate_package(stage, world["run"]).reasons
    assert any(DRAFTER in r and "TT_AUXILIARY_WEIGHTS" in r for r in reasons), reasons
