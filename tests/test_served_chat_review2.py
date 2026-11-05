# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 297, Copilot round 2 on PR 466: a credential must be removed from a whole string before the string is cut.

Cutting first leaves a fragment of a token that straddles the cut (a Hugging Face token starting at character 480 leaves `hf_` and 17
payload characters when the line is cut at 500), and no redactor matches it: the pattern needs 30 payload characters. Two places cut text
that came from outside files: the writer (`orchard/served_chat.py`: `_shown`, `_cut_text` and the 30-character reply lead) and the bundle
builder (`served_chat_verdict` in the operator-bundle template, four lines copied into RISKS.md)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_served_chat import all_good, chat_block, reply, suite, swap_check, pair  # noqa: E402

from orchard import served_chat as sc  # noqa: E402

TOKEN = " hf_" + "A" * 40           # a fixture shaped like a Hugging Face token after a space (the pattern needs a word boundary): 3 + 40 characters, over its 30-character minimum
LEAK = "hf_AAAA"                    # any piece of the token this long in an output is a leak
PAD = "p" * 480                     # the reviewer's case: the token starts at character 480, the cut is at 500


def clean(text: str) -> bool:
    return LEAK not in text and "AAAAAAAAAAAA" not in text


# ---- the writer: the cut sites of orchard/served_chat.py --------------------------------------------------------------------

@pytest.mark.parametrize("start", [60, 70, 77, 79])
def test_shown_removes_a_token_that_straddles_its_cut(start):
    out = sc._shown("x" * start + TOKEN)
    assert clean(out) and "hf_" not in out


def test_shown_removes_a_token_whose_payload_runs_far_past_the_cut():
    out = sc._shown("x" * 70 + " hf_" + "A" * (1 << 20))
    assert clean(out) and "hf_" not in out
    assert str((1 << 20) + 74) in out                                  # the full length is still reported


@pytest.mark.parametrize("start", [170, 190, 198])
def test_cut_text_removes_a_token_that_straddles_its_cut(start):
    out = sc._cut_text("p" * start + TOKEN, 200)
    assert clean(out) and "hf_" not in out
    assert out.endswith(f"({start + len(TOKEN)} characters)")


def test_the_written_record_removes_a_token_that_straddles_a_cut():
    rec = {"files": {"end_tokens": ["p" * 190 + TOKEN], "special_tokens": {"served": {"bos": "p" * 190 + TOKEN}}},
           "problems": [{"name": "n", "detail": "p" * 590 + TOKEN}], "incomplete": ["p" * 590 + TOKEN]}
    assert clean(json.dumps(sc.bounded_record(rec)))


def test_the_reply_lead_in_a_problem_removes_a_token_that_straddles_its_cut(tmp_path):
    text = "assistant: " + "p" * 10 + TOKEN                              # the 30-character lead ends inside the token
    res = sc.check_replies(json.loads(swap_check(tmp_path, [reply(content=text)]).read_text())["prompt_suite"])
    detail = next(p["detail"] for p in res["problems"] if p["name"] == "role_header_start")
    assert clean(detail)


@pytest.mark.parametrize("text", ["short", "a" * 80])
def test_shown_text_without_a_credential_is_unchanged(text):
    assert sc._shown(text) == repr(text)


def test_shown_cut_text_without_a_credential_is_cut_as_before():
    r = repr("a" * 80)
    assert sc._shown("a" * 300) == f"{r[:-1]}...{r[-1]} (300 characters)"


@pytest.mark.parametrize("text,limit", [("short", 200), ("a" * 200, 200), ("a" * 300, 200)])
def test_cut_text_without_a_credential_is_cut_exactly_as_before(text, limit):
    cut = sc._cut_text(text, limit)
    assert cut == (text if len(text) <= limit else f"{text[:limit]}... ({len(text)} characters)")


def test_the_writer_scan_is_bounded_on_huge_text():
    start = time.monotonic()
    out = sc._shown("x" * (40 << 20))
    assert len(out) < 200 and time.monotonic() - start < 5


# ---- the bundle: the four lines of served_chat_verdict ------------------------------------------------------------------------

def stage2(tmp_path, **changes):
    records = all_good(tmp_path / "rec")
    records["stage2"] = {**records["stage2"], **changes}
    return records


@pytest.mark.parametrize("items", [[PAD + TOKEN], [PAD + TOKEN, "second"], ["a", "b", PAD + TOKEN], ["p" * 490 + TOKEN]])
def test_an_incomplete_line_is_redacted_whole_before_it_is_cut(tmp_path, items):
    block, _ = chat_block(tmp_path / "run", stage2(tmp_path, incomplete=items, ok=False))
    assert "is incomplete" in block and clean(block)


def test_a_problem_name_is_redacted_whole_before_it_is_cut(tmp_path):
    block, _ = chat_block(tmp_path / "run", stage2(tmp_path, problems=[{"name": PAD + TOKEN, "detail": "d"}], ok=False))
    assert "found:" in block and clean(block)


def test_a_why_not_is_redacted_whole_before_it_is_cut(tmp_path):
    block, _ = chat_block(tmp_path / "run", stage2(tmp_path, replies={"checked": False, "why_not": PAD + TOKEN}, ok=False))
    assert "checked no chat reply" in block and clean(block)


def test_a_failed_write_reason_is_redacted_whole_before_it_is_cut(tmp_path):
    def mutate(run):
        from orchard.ledger import Ledger
        with Ledger(run / "ledger.jsonl") as led:
            led.append("decision", 2, decision="served chat check not written", reason=PAD + TOKEN)
    block, _ = chat_block(tmp_path / "run", all_good(tmp_path / "rec"), mutate=mutate)
    assert "failed to write" in block and clean(block)


def test_a_redacted_line_still_shows_the_text_before_the_token_and_a_label(tmp_path):
    block, _ = chat_block(tmp_path / "run", stage2(tmp_path, incomplete=[PAD + TOKEN], ok=False))
    assert "ppppp" in block and "<token removed>" in block


def test_a_long_line_without_a_credential_is_still_cut_with_its_length(tmp_path):
    block, _ = chat_block(tmp_path / "run", stage2(tmp_path, incomplete=["q" * 900], ok=False))
    assert "900 characters" in block and "q" * 501 not in block


def test_a_text_over_the_record_cap_is_redacted_and_cut_without_a_long_scan(tmp_path):
    start = time.monotonic()
    block, _ = chat_block(tmp_path / "run", stage2(tmp_path, incomplete=["z" * (3 << 20) + TOKEN], ok=False))
    assert "characters" in block and clean(block) and time.monotonic() - start < 60


# ---- the shared table-cell helper `txt` had the same order (cut, then redact) ------------------------------------------------

@pytest.mark.parametrize("start", [40, 55, 60])
def test_txt_redacts_the_whole_string_before_cutting_to_its_limit(start, tmp_path):
    import importlib.util
    path = Path(__file__).parent.parent / "orchard/skills/operator-bundle-templates/build_bundle.py"
    spec = importlib.util.spec_from_file_location("bb_review2", path)
    bb = importlib.util.module_from_spec(spec)
    sys.modules["bb_review2"] = bb
    spec.loader.exec_module(bb)
    cell = bb.txt(bb.Redactor({}, tmp_path / 'run', __import__('orchard.scrub', fromlist=['x'])), "q" * start + TOKEN, 80)
    assert clean(cell)


def test_record_text_hands_the_redactor_at_most_the_read_cap():
    import importlib.util
    path = Path(__file__).parent.parent / "orchard/skills/operator-bundle-templates/build_bundle.py"
    spec = importlib.util.spec_from_file_location("bb_review2b", path)
    bb = importlib.util.module_from_spec(spec)
    sys.modules["bb_review2b"] = bb
    spec.loader.exec_module(bb)
    seen = []

    def redact(text):
        seen.append(len(text))
        return text

    out = bb._record_text("z" * (bb.SERVED_CHAT_READ_MAX + 1000), redact)
    assert seen == [bb.SERVED_CHAT_READ_MAX] and "characters" in out
