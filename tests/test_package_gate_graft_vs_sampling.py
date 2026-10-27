# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fourth review of PR 114, finding A (issue 107, high): when stage 2's sampling decision chose the drafter-off configuration, a package that carries graft
metadata (`mtp_graft` in package.json, or a `graft.json` in a profile) turns the drafter on, which the decision and the card say is off. The gate does not
rely on `read_run`'s conflict check: it refuses that metadata itself, and the graft exemption no longer switches the serving checks off."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_package_graft import world  # noqa: E402,F401  (the fixture)
from test_package_sampling_choice import drafter_on_run, stage_card  # noqa: E402

from orchard.stages import gate_package  # noqa: E402

RECORD = {"repo": "Qwen/Qwen3.8-27B", "revision": "a" * 40, "file": "model-00018-of-00018.safetensors", "shard": "model-mtp-from-base.safetensors",
          "tensors": ["mtp.fc.weight"], "sha256": "b" * 64}


def test_a_drafter_off_package_passes_before_any_forging(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    assert gate_package(stage_dir, world["run"]).ok


def test_package_json_graft_metadata_is_refused_when_the_decision_chose_drafter_off(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    p = stage_dir / "package.json"
    d = json.loads(p.read_text())
    p.write_text(json.dumps({**d, "mtp_graft": RECORD}))
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("mtp_graft" in r and "drafter-off" in r for r in g.reasons), g.reasons


def test_a_profile_graft_file_is_refused_when_the_decision_chose_drafter_off(world):
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    (pkg / "graft.json").write_text(json.dumps(RECORD))
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("graft.json" in r and "drafter-off" in r for r in g.reasons), g.reasons


def test_graft_metadata_does_not_switch_off_the_serving_checks_for_a_drafter_off_decision(world):
    """A grafted package with the drafter on must also be named by the serving check, not only by the graft check."""
    drafter_on_run(world)
    stage_dir, pkg, card = stage_card(world)
    p = stage_dir / "package.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "mtp_graft": RECORD}))
    run_sh = pkg / "run.sh"
    run_sh.write_text(run_sh.read_text().replace("CMD=(", 'export QWEN36_DRAFTER="dflash2"\nCMD=(', 1))
    g = gate_package(stage_dir, world["run"])
    assert any("turns the drafter on" in r for r in g.reasons), g.reasons


def test_a_keep_decision_still_allows_graft_metadata(world):
    """Only a drafter-off decision conflicts with a graft; a kept configuration is judged as before (graft checks apply to the graft's own files)."""
    drafter_on_run(world, needed=False)
    stage_dir, pkg, card = stage_card(world)
    p = stage_dir / "package.json"
    p.write_text(json.dumps({**json.loads(p.read_text()), "mtp_graft": RECORD}))
    g = gate_package(stage_dir, world["run"])
    assert not any("drafter-off" in r and "mtp_graft" in r for r in g.reasons), g.reasons
