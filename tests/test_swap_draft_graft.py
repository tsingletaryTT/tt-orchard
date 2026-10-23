# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4's drafted configs follow stage 2's graft decision (orchard/graft_trial.py): a configuration served by the nearest model's own
v6 bundle gets `graft_mtp` when stage 2 chose the graft. The trial measured that bundle's server. A container package is another server,
so its configuration is left as it was."""
import json
from pathlib import Path

import pytest

from orchard import swap_draft as sd
from orchard.graft_trial import DECISION_FILE
from test_swap_draft import NEAREST, bundle, containers, run, stage4          # noqa: F401  (the fixtures and helpers)


def decide(run, choice):
    d = run["run_dir"] / "stages" / "2"
    d.mkdir(parents=True, exist_ok=True)
    (d / DECISION_FILE).write_text(json.dumps({"choice": choice, "reason": "because"}))


@pytest.fixture
def machine(run, tmp_path):
    bundle(run["tt_model_root"], "episod/qwen3.8-27b-dflash2-p150", NEAREST, 1)
    bundle(run["tt_model_root"], "episod/qwen3.8-27b-dflash2-p300", NEAREST, 2)
    home = containers(tmp_path / "op", {"changh95/qwen3.8-27b-p300x2": (NEAREST, 4, "batch32")})
    return home


def test_a_graft_decision_puts_graft_mtp_in_each_bundle_configuration(run, machine):
    decide(run, "graft")
    configs, tests, notes = stage4(run, machine, counts=(1, 2, 4))
    assert notes == []
    assert configs[1]["graft_mtp"] is True and configs[2]["graft_mtp"] is True


def test_the_container_configuration_is_left_without_the_graft(run, machine):
    decide(run, "graft")
    configs, tests, notes = stage4(run, machine, counts=(1, 2, 4))
    assert "graft_mtp" not in configs[4] and "package" in configs[4]


@pytest.mark.parametrize("choice", ["safe", None])
def test_no_graft_decision_or_a_safe_one_leaves_the_configs_as_they_were(run, machine, choice):
    if choice:
        decide(run, choice)
    configs, tests, notes = stage4(run, machine, counts=(1, 2, 4))
    assert all("graft_mtp" not in c for c in configs.values())


def test_a_malformed_decision_file_leaves_the_configs_as_they_were(run, machine):
    d = run["run_dir"] / "stages" / "2"
    d.mkdir(parents=True)
    (d / DECISION_FILE).write_text("{")
    configs, tests, notes = stage4(run, machine, counts=(1, 2, 4))
    assert all("graft_mtp" not in c for c in configs.values())


def test_the_rest_of_each_config_is_the_same_with_or_without_the_graft(run, machine):
    plain, _, _ = stage4(run, machine, counts=(1, 2, 4))
    decide(run, "graft")
    grafted, _, _ = stage4(run, machine, counts=(1, 2, 4))
    for n in (1, 2, 4):
        assert {k: v for k, v in grafted[n].items() if k != "graft_mtp"} == plain[n]
