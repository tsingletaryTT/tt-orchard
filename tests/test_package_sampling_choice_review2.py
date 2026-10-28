# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Second review of PR 114, finding 1 (issue 107): stage 2 may have been built with `graft_mtp`. The sampling trial removes the graft and can choose a healthy
drafter-off variant. The sampling decision then supersedes the historical graft inference, so stage 7 packages the chosen configuration, while an explicit
`--graft-mtp` still conflicts with it."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_package_graft import stage, stage_2_with_the_real_graft, world  # noqa: E402,F401  (the fixture)
from test_package_sampling_choice import drafter_on_run, section  # noqa: E402
from test_package_graft_auto import finished_auto  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402
from orchard.package import PackageError, read_run  # noqa: E402
from orchard.stages import gate_package  # noqa: E402


def grafted_stage_2_then_drafter_off(world):
    stage_2_with_the_real_graft(world)
    dec = drafter_on_run(world)
    assert dec["choice"] == "drafter-off"
    return dec


def test_a_stage_2_that_grafted_is_read_as_drafter_off_when_the_sampling_decision_chose_drafter_off(world):
    grafted_stage_2_then_drafter_off(world)
    facts = read_run(world["run"])
    assert facts.drafter_off is True and facts.host_sampling is True and facts.graft is None
    assert facts.stage_2_grafted is True                       # the provenance record stays


def test_the_package_is_staged_drafter_off_with_no_graft_file(world):
    grafted_stage_2_then_drafter_off(world)
    _, package = stage(world)
    pkg = world["run"] / package["profiles"][0]["dir"]
    assert "mtp_graft" not in package and not (pkg / "graft.json").exists()
    assert "sample_on_device_mode" not in (pkg / "run.sh").read_text()


def test_the_finished_package_passes_the_gate(world):
    grafted_stage_2_then_drafter_off(world)
    stage_dir, package, pkg = finished_auto(world)
    g = gate_package(stage_dir, world["run"])
    assert g.ok, g.reasons


def test_an_explicit_graft_request_still_conflicts_with_the_decision(world):
    grafted_stage_2_then_drafter_off(world)
    with pytest.raises(PackageError, match="drafter-off"):
        read_run(world["run"], graft_mtp=True)


def test_no_graft_mtp_is_allowed_when_the_decision_chose_drafter_off(world):
    grafted_stage_2_then_drafter_off(world)
    assert read_run(world["run"], no_graft_mtp=True).drafter_off is True


def test_a_graft_decision_beside_a_drafter_off_decision_still_conflicts(world):
    from test_package_sampling_choice import write
    grafted_stage_2_then_drafter_off(world)
    write(world["run"] / "stages/2/graft-decision.json", {"choice": "graft", "reason": "r"})
    with pytest.raises(PackageError):
        read_run(world["run"])


def test_a_stage_2_that_grafted_with_a_keep_decision_is_still_a_grafted_package(world):
    """Only a drafter-off decision supersedes the historical inference."""
    stage_2_with_the_real_graft(world)
    drafter_on_run(world, needed=False)
    facts = read_run(world["run"])
    assert facts.graft is not None and facts.drafter_off is False
