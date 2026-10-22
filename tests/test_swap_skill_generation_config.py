# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the stage 2 swap skill tells the agent to do when prepare_swap.py warns about the generation config.

The script prints a warning when `generation_config.json` is in neither snapshot, and it goes on. The warning is a
prediction from one bundle's reading rule: the copied `config.json` may still hold a top-level `eos_token_id`. The
supervisor reads `hw_test.json` and `handoff.json` after the prepare step and has no outcome for "the agent wrote
`result.json` and stopped" (it reports the two files missing), so the agent keeps going and the hardware test decides."""
from pathlib import Path

SKILL = Path(__file__).parent.parent / "orchard" / "skills" / "weights-swap-check.md"


def bullet(start: str) -> str:
    """One bullet of the skill, whitespace folded, from the line that starts with `start` to the next bullet."""
    text = SKILL.read_text(encoding="utf-8")
    first = text.index(start)
    end = text.find("\n- ", first + 1)
    return " ".join(text[first:end if end != -1 else len(text)].split())


def test_a_generation_config_in_neither_snapshot_is_advisory_and_the_agent_goes_on():
    b = bullet("- A new model with no `generation_config.json`")
    assert "in neither snapshot" in b and "advisory" in b
    assert "`hw_test.json` and `handoff.json`" in b and "do not stop" in b.lower()


def test_the_skill_no_longer_sends_the_agent_to_result_json_and_a_stop_for_that_warning():
    b = bullet("- A new model with no `generation_config.json`")
    assert "`result.json`" not in b and "and stop" not in b
