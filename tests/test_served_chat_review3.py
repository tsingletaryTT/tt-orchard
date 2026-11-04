# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 297, Copilot round 4 on PR 466: the credential removal must read only the display range of the ORIGINAL text.

`_scrubbed_prefix` replaced every credential in the scan window and then cut the result at the limit. A replacement is shorter than the
credential it replaces, so the cut moved right in the original text and pulled in the start of a second credential whose payload the
window had already truncated below the pattern's 30-character minimum. `hf_` and 29 payload characters were shown."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_served_chat import reply, swap_check  # noqa: E402

from orchard import served_chat as sc  # noqa: E402

# The reviewer's case: a first credential 300 characters long, then a second one that the 256-character window cuts to 29 payload characters.
TWO = "hf_" + "A" * 300 + " hf_" + "B" * 40


def clean(text: str) -> bool:
    return "hf_BBBB" not in text and "BBBBBBBBBB" not in text and "AAAAAAAAAA" not in text


def test_shown_does_not_pull_a_second_truncated_credential_into_the_display():
    out = sc._shown(TWO)
    assert clean(out) and "hf_" not in out


def test_scrubbed_prefix_matches_the_reviewers_reproduction():
    out = sc._scrubbed_prefix(TWO, 80)
    assert clean(out) and "hf_" not in out


def test_cut_text_does_not_pull_a_second_truncated_credential_into_the_display():
    out = sc._cut_text(TWO, 200)
    assert clean(out) and out.endswith(f"({len(TWO)} characters)")


@pytest.mark.parametrize("gap", [0, 1, 20, 60])
def test_several_short_credentials_before_the_cut_leave_no_fragment(gap):
    """Many replacements before the cut shift it far to the right; the text it reaches must still be only the original display range."""
    text = (("hf_" + "C" * 31 + " " * (gap + 1)) * 12) + "hf_" + "D" * 40
    out = sc._scrubbed_prefix(text, 100)
    assert "hf_" not in out and "DDDDDDDD" not in out and "CCCCCCCC" not in out


def test_the_reply_lead_in_a_problem_does_not_leak_the_second_credential(tmp_path):
    text = "hf_" + "A" * 3 + " hf_" + "A" * 33 + " hf_" + "B" * 40     # a lead of 30 characters ends in the first credential
    res = sc.check_replies(json.loads(swap_check(tmp_path, [reply(content="assistant: " + text)]).read_text())["prompt_suite"])
    detail = next(p["detail"] for p in res["problems"] if p["name"] == "role_header_start")
    assert "BBBBBBBB" not in detail


@pytest.mark.parametrize("text,limit", [("plain text " * 20, 80), ("a" * 500, 80), ("short", 80), ("x" * 79 + "hf_" + "E" * 40, 80)])
def test_text_whose_credentials_all_start_at_or_after_the_cut_is_cut_as_before(text, limit):
    out = sc._scrubbed_prefix(text, limit)
    assert out == text[:limit]


def test_a_credential_inside_the_display_range_is_replaced_by_one_label():
    out = sc._scrubbed_prefix("see hf_" + "F" * 40 + " then more text", 80)
    assert out == "see <token removed> then more text"


def test_the_shown_part_is_never_longer_than_the_limit():
    for start in range(0, 120, 7):
        assert len(sc._scrubbed_prefix("x" * start + "hf_" + "G" * 400 + " tail", 80)) <= 80


def test_text_after_a_replaced_credential_is_copied_only_up_to_the_original_limit():
    """The label is shorter than the credential, so the shown part ends at original offset `limit` and is shorter than `limit` characters."""
    out = sc._scrubbed_prefix("hf_" + "A" * 40 + " " + "t" * 200, 80)
    assert out == "<token removed>" + " " + "t" * 36


def test_a_credential_at_the_cut_is_cut_with_the_shown_text():
    out = sc._scrubbed_prefix("x" * 69 + " hf_" + "H" * 40, 80)
    assert out == "x" * 69 + " <token rem" and len(out) == 80
