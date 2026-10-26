# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Review of PR 80, item 2: the chip count comes from the bundle's or package's own metadata, an override that disagrees is refused, and
the control's config and its lease always hold the same number."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_suite_control import CHECKOUT, NEAREST, _prep, build, ok_run, prep, stage_cfg  # noqa: E402,F401

from orchard import suite_control as sc  # noqa: E402


def set_manifest(prep, value):
    (prep["bundle"] / "tt_kernel_manifest.json").write_text(json.dumps({"device_count": value}))


def test_a_four_chip_bundle_gets_a_four_chip_lease_and_config(prep):
    set_manifest(prep, 4)
    out = build(prep)
    assert out["chips"] == 4 and out["config"]["chips"] == 4


def test_a_two_chip_bundle_gets_two(prep):
    out = build(prep)
    assert out["chips"] == 2 and out["config"]["chips"] == 2


def test_an_override_that_disagrees_with_the_bundle_is_refused(prep):
    with pytest.raises(sc.ControlError, match="needs 2 chips"):
        build(prep, chips=4)
    with pytest.raises(sc.ControlError, match="needs 2 chips"):
        build(prep, chips=1)


def test_an_override_that_agrees_is_accepted(prep):
    assert build(prep, chips=2)["chips"] == 2


def test_a_container_override_cannot_change_its_config(prep):
    c = stage_cfg(prep)
    c.pop("bundle_dir")
    c.update(package="org/pkg", profile="p", chips=4, operator_home=str(prep["stage"].parent))
    (prep["stage"] / "swap_config.json").write_text(json.dumps(c))
    with pytest.raises(sc.ControlError, match="needs 4 chips"):
        sc.prepare(prep["stage"], orchard_dir=CHECKOUT, chips=2, run=ok_run)
    out = sc.prepare(prep["stage"], orchard_dir=CHECKOUT, run=ok_run)
    assert out["chips"] == 4 and out["config"]["chips"] == 4


@pytest.mark.parametrize("manifest", [None, "{nope", json.dumps({}), json.dumps({"device_count": 0}), json.dumps({"device_count": True}),
                                       json.dumps({"device_count": "many"})])
def test_a_bundle_whose_chip_count_cannot_be_read_is_refused(prep, manifest):
    f = prep["bundle"] / "tt_kernel_manifest.json"
    if manifest is None:
        f.unlink()
    else:
        f.write_text(manifest)
    with pytest.raises(sc.ControlError, match="chip count"):
        build(prep, run=ok_run)


def test_a_container_config_without_a_chip_count_is_refused(prep):
    c = stage_cfg(prep)
    c.pop("bundle_dir")
    c.update(package="org/pkg", profile="p", operator_home=str(prep["stage"].parent))
    (prep["stage"] / "swap_config.json").write_text(json.dumps(c))
    with pytest.raises(sc.ControlError, match="chip count"):
        sc.prepare(prep["stage"], orchard_dir=CHECKOUT, run=ok_run)


def test_the_cli_chip_flag_is_optional_and_checked(prep, capsys):
    code = sc.main(["prepare", "--stage-dir", str(prep["stage"]), "--orchard-dir", str(CHECKOUT), "--chips", "4"])
    assert code == 2 and "needs 2 chips" in capsys.readouterr().err
