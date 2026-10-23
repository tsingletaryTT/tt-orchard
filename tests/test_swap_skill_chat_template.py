# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the stage 2 swap skill says about the chat template that prepare_swap.py records.

The script writes the served chat template and special tokens into `model-dir-sources.json` and prints one line. The
swap check sends plain text to /v1/completions, so it never applies the template; the agent has nothing to fix and must
not edit the record."""
from pathlib import Path

SKILL = Path(__file__).parent.parent / "orchard" / "skills" / "weights-swap-check.md"


def bullet(start: str) -> str:
    text = SKILL.read_text(encoding="utf-8")
    first = text.index(start)
    end = text.find("\n- ", first + 1)
    return " ".join(text[first:end if end != -1 else len(text)].split())


def test_the_skill_says_the_record_holds_the_chat_template_and_special_tokens_and_to_leave_them():
    b = bullet("- `prepare_swap.py` also writes `stages/2/model-dir-sources.json`")
    assert "also holds the chat template the directory serves" in b and "and the special tokens" in b
    assert "Do not edit" in b


def test_the_skill_says_the_swap_check_applies_no_chat_template():
    b = bullet("- `prepare_swap.py` also writes `stages/2/model-dir-sources.json`")
    assert "/v1/completions" in b and "applies no chat template" in b
