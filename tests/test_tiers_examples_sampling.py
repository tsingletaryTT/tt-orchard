# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The tiers files in the repository that name Qwen3-Coder-Next ask for sampling on the coder tiers. A paired test (one run per arm) on the tiny
Qwen4 rehearsal gave the coder written files and a port checked on chips with temperature 0.7 and repetition_penalty 1.05, and none in 37
turns under the harness default of greedy decoding (docs/coder-sampling.md)."""
import tomllib
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[1] / "config"
FILES = ["tiers.example.toml", "tiers.qb2-coder-next.toml", "tiers.qb2-coder-next-v1.toml"]


@pytest.mark.parametrize("name", FILES)
@pytest.mark.parametrize("tier", ["large", "small"])
def test_each_coder_tier_of_a_shipped_example_asks_for_the_tested_sampling(name, tier):
    table = tomllib.loads((CONFIG / name).read_text())["tiers"][tier]
    assert table["temperature"] == 0.7 and table["repetition_penalty"] == 1.05


@pytest.mark.parametrize("name", ["tiers.qb2-coder-next.toml", "tiers.qb2-coder-next-v1.toml"])
def test_the_coder_next_examples_still_load_with_the_sampling_keys(name):
    from orchard.tiers import load
    tiers = load(CONFIG / name).tiers
    assert tiers["small"]["temperature"] == 0.7 and tiers["large"]["repetition_penalty"] == 1.05


def test_the_cpu_stand_in_tier_keeps_the_defaults():
    table = tomllib.loads((CONFIG / "tiers.qb2-coder-next.toml").read_text())["tiers"]["cpu"]
    assert "temperature" not in table and "repetition_penalty" not in table
