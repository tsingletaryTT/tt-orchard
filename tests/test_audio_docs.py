# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The text that tells a person and the stage 0 agent about audio models (plan docs/superpowers/plans/2026-10-28-audio-bringup.md, task A5)
names the same things the code uses: the layout, the facts and their values, the block code and the evidence files."""
import re
from pathlib import Path

from orchard import audio_facts, blocked

ROOT = Path(__file__).resolve().parent.parent
SKILL = (ROOT / "orchard" / "skills" / "delta-triage.md").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_the_skill_describes_the_audio_branch_and_its_files():
    for needle in ("model_config.json", "audio-compare.json", "facts", "modality", "`repo_id`", "training", "Do not edit"):
        assert needle in SKILL, needle
    assert "Do not edit the script and do not change" in SKILL


def test_the_skill_names_every_value_of_the_facts():
    for value in audio_facts.FORMS + audio_facts.SERVINGS + audio_facts.MODALITIES:
        assert f"`{value}`" in SKILL, value


def test_the_skill_says_what_makes_an_audio_pair_a_full_port_and_unknown():
    section = SKILL[SKILL.index("## Audio models"):SKILL.index("## Resume notes")]
    assert "full-port" in section and "unknown" in section and "weights-only" in section


def test_the_readme_has_the_audio_row_and_the_block_code():
    assert "audio-path-unbuilt" in README and "audio-path-unbuilt" in blocked.REASONS
    assert re.search(r"\| `modality`", README) or "modality" in README


def test_the_readme_says_nothing_serves_stable_audio_yet():
    assert "There is no runtime for Stable Audio 3 on a Tenstorrent chip" in README
