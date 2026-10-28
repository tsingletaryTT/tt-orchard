# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The coder package: the fused-decode Qwen3-Coder-Next bundle is the default, the v1-era package is the fallback.

Issue 180. The two packages differ in name and image, so their containers differ in name. The supervisor
stops only a container it started (PR 158), so the stop check for one package must ignore a running container
of the other (an agent on the other board may be serving it).
"""
import tomllib
from pathlib import Path

import pytest

from fakes import LEASE
from orchard import setup_machine as sm
from orchard.server import ServerSpec
from test_server import control

REPO = Path(__file__).resolve().parent.parent
NEW = "raahemnabeel/qwen3-coder-next"
OLD = "raahemnabeel/qwen3-coder-next-blackhole"
# Each container as docker lists it: id, image, name, ports.
OLD_LINE = ("9f8e7d6c5b4a tt-model/qwen3-coder-next-blackhole:bd0ca927a46b "
            "tt-model-qwen3-coder-next-blackhole-p300 0.0.0.0:8011->8011/tcp\n")
NEW_LINE = ("1a2b3c4d5e6f tt-model/qwen3-coder-next:0c170421ca1b "
            "tt-model-qwen3-coder-next-p300 0.0.0.0:8001->8001/tcp\n")


def spec(package, image_id, port=8001):
    return ServerSpec(package, "container", port, "Qwen/Qwen3-Coder-Next", profile="p300", image_id=image_id)


def test_the_two_packages_get_different_container_names():
    assert spec(NEW, None).container_name == "tt-model-qwen3-coder-next-p300"
    assert spec(OLD, None).container_name == "tt-model-qwen3-coder-next-blackhole-p300"


def test_the_start_command_names_the_new_package_and_the_p300_profile():
    ctl, run, *_ = control(spec(NEW, "0c170421ca1b"))
    ctl.start(LEASE)
    argv = run.argvs()[0]
    assert argv[:3] == ["tt-model", "serve", NEW]
    assert argv[argv.index("--profile") + 1] == "p300"
    assert argv[argv.index("--device-id") + 1] == "0,1"


def test_stopping_one_package_ignores_a_running_container_of_the_other():
    # The new coder is stopped; the old package's container (another agent's, port 8011) still runs.
    ctl, *_ = control(spec(NEW, "0c170421ca1b"), **{"docker ps": [(0, OLD_LINE, "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    assert ctl.confirm_stopped().checks["docker_ps"] is True
    # and the reverse
    ctl, *_ = control(spec(OLD, "bd0ca927a46b", port=8011), **{"docker ps": [(0, NEW_LINE, "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    assert ctl.confirm_stopped().checks["docker_ps"] is True


def test_its_own_container_still_blocks_the_stop_check():
    ctl, *_ = control(spec(NEW, "0c170421ca1b"), **{"docker ps": [(0, NEW_LINE, "")]})
    ctl.adopt({"dev_indices": [0, 1]})
    assert ctl.confirm_stopped().checks["docker_ps"] is False


def test_the_default_coder_is_the_fused_decode_bundle_and_the_old_package_is_the_fallback():
    assert sm.CODERS["coder-next"]["package"] == NEW
    assert sm.CODERS["coder-next-v1"]["package"] == OLD
    assert "fallback" in sm.CODERS["coder-next-v1"]["note"]


@pytest.mark.parametrize("layout,package", [("coder-next", NEW), ("coder-next-v1", OLD)])
def test_each_coder_layout_ships_matching_bringup_and_tiers_files(layout, package):
    raw = (REPO / "config" / f"bringup.qb2-{layout}.toml").read_text()
    bring = tomllib.loads(sm.render(raw, RUNS_ROOT="/r", REFERENCE_PYTHON="/p", SKILLS_LINE=""))
    assert (REPO / "config" / f"tiers.qb2-{layout}.toml").is_file()
    assert bring["coder"]["target"] == package
    assert bring["coder"]["kind"] == "container" and bring["coder"]["profile"] == "p300"
    assert bring["coder"]["chips"] == 2


@pytest.mark.parametrize("layout", ["coder-next", "coder-next-v1"])
def test_the_tiers_files_send_the_weights_id_not_the_package_name(layout):
    text = (REPO / "config" / f"tiers.qb2-{layout}.toml").read_text()
    assert 'model = "Qwen/Qwen3-Coder-Next"' in text
