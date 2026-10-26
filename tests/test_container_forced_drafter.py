# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 80, item 5: a control of a run served by a container package with the drafter off must itself be served with it off.
`"drafter": "off"` in swap_config.json is honoured by the container template as it is by prepare_swap.py, and the control reads the
stage's effective serving state from the stage's own evidence (a container stage has no run.sh)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_container_template import crig, docker_runs, run_env_values, set_index  # noqa: E402,F401

from orchard import suite_control as sc  # noqa: E402


def test_the_key_clears_the_drafter_of_a_model_that_has_the_head(crig):
    set_index(crig, ["model.embed_tokens.weight", "mtp.fc.weight"])
    r = crig["start"](drafter="off")
    assert r.returncode == 0, r.stdout + r.stderr
    assert run_env_values(crig, "QWEN36_DRAFTER") == [""]
    assert json.loads((crig["cdir"] / "evidence" / "swap-check.json").read_text())["serving"]["drafter"] == "off"


def test_without_the_key_the_drafter_stays(crig):
    set_index(crig, ["model.embed_tokens.weight", "mtp.fc.weight"])
    r = crig["start"]()
    assert r.returncode == 0, r.stdout + r.stderr
    assert run_env_values(crig, "QWEN36_DRAFTER") == ["mtp"]


def test_the_key_gives_a_package_that_sets_no_drafter_an_explicit_off(crig):
    r = crig["start"](printed={"drafter": False}, drafter="off")
    assert r.returncode == 0, r.stdout + r.stderr
    assert run_env_values(crig, "QWEN36_DRAFTER") == [""]


@pytest.mark.parametrize("value", ["on", True, "", None])
def test_any_other_value_exits_2(crig, value):
    r = crig["start"](drafter=value)
    assert r.returncode == 2 and "drafter" in (r.stdout + r.stderr)
    assert docker_runs(crig) == []


# ---- the control reads a container stage's serving state ------------------------------------------------------------

def write_swap_check(stage: Path, drafter):
    (stage / "evidence").mkdir(exist_ok=True)
    (stage / "evidence" / "swap-check.json").write_text(json.dumps({"serving": {"drafter": drafter, "sampling": "host"}}))


def test_a_container_stage_served_with_the_drafter_off_is_read_from_its_evidence(tmp_path):
    write_swap_check(tmp_path, "off")
    assert sc.stage_drafter_off(tmp_path) is True


@pytest.mark.parametrize("drafter", ["on (mtp)", "not set by the package", "unknown", "", None, True])
def test_other_serving_states_are_not_off(tmp_path, drafter):
    write_swap_check(tmp_path, drafter)
    assert sc.stage_drafter_off(tmp_path) is False


def test_no_run_script_and_no_evidence_is_not_off(tmp_path):
    assert sc.stage_drafter_off(tmp_path) is False
    (tmp_path / "evidence").mkdir()
    (tmp_path / "evidence" / "swap-check.json").write_text("{nope")
    assert sc.stage_drafter_off(tmp_path) is False


def test_the_run_script_wins_over_the_evidence(tmp_path):
    write_swap_check(tmp_path, "off")
    (tmp_path / "run.sh").write_text('export QWEN36_DRAFTER="dflash2"\n')
    assert sc.stage_drafter_off(tmp_path) is False
