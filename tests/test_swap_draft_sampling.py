# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4's drafted configs follow stage 2's sampling decision (orchard/sampling_choice.py, issue 107): when stage 2 chose the drafter-off configuration,
every drafted configuration says `"drafter": "off"`, so prepare_swap.py (bundle) and the container script write the same serving state that stage 2 tested."""
import json

import pytest

from orchard import sampling_choice as sc
from test_swap_draft import NEAREST, bundle, containers, run, stage4          # noqa: F401  (the fixtures and helpers)


def decide(run, choice):
    d = run["run_dir"] / "stages" / "2"
    d.mkdir(parents=True, exist_ok=True)
    (d / sc.DECISION_FILE).write_text(json.dumps({"choice": choice, "reason": "because", "flag": None}))


@pytest.fixture
def machine(run, tmp_path):
    bundle(run["tt_model_root"], "episod/qwen3.8-27b-dflash2-p150", NEAREST, 1)
    bundle(run["tt_model_root"], "episod/qwen3.8-27b-dflash2-p300", NEAREST, 2)
    return containers(tmp_path / "op", {"changh95/qwen3.8-27b-p300x2": (NEAREST, 4, "batch32")})


def test_a_drafter_off_decision_puts_drafter_off_in_every_configuration(run, machine):
    decide(run, "drafter-off")
    configs, tests, notes = stage4(run, machine, counts=(1, 2, 4))
    assert notes == []
    assert {n: c.get("drafter") for n, c in configs.items()} == {1: "off", 2: "off", 4: "off"}


@pytest.mark.parametrize("choice", ["keep", None])
def test_a_keep_decision_or_none_leaves_the_configs_as_they_were(run, machine, choice):
    if choice:
        decide(run, choice)
    configs, _, _ = stage4(run, machine, counts=(1, 2, 4))
    assert all("drafter" not in c for c in configs.values())


def test_a_malformed_decision_file_leaves_the_configs_as_they_were(run, machine):
    d = run["run_dir"] / "stages" / "2"
    d.mkdir(parents=True)
    (d / sc.DECISION_FILE).write_text("{")
    configs, _, _ = stage4(run, machine, counts=(1, 2, 4))
    assert all("drafter" not in c for c in configs.values())


def test_the_rest_of_each_config_is_the_same_with_or_without_the_decision(run, machine):
    plain, _, _ = stage4(run, machine, counts=(1, 2, 4))
    decide(run, "drafter-off")
    off, _, _ = stage4(run, machine, counts=(1, 2, 4))
    for n in (1, 2, 4):
        assert {k: v for k, v in off[n].items() if k != "drafter"} == plain[n]
