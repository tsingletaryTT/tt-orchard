# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 297, Copilot round 1 on PR 466: token text from a tokenizer file must never make the served-chat record large, and a ledger-named record
must reach the missing-record checks even when stage 2 has none.

The tokenizer files are outside input (the served directory and the new model's snapshot), each read up to 64 MiB. The bundle reads
`served-chat.json` with an 8 MiB cap and treats a record over the cap as unreadable, so one 9 MiB token text would have hidden every mismatch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_served_chat import GOOD_GEN, TOK, TOKENIZER_JSON, all_good, chat_block, make, names, pair, reply, suite, swap_check  # noqa: E402
from test_served_chat_issue297 import BASE, added, tokenizer_with  # noqa: E402

from orchard import served_chat as sc  # noqa: E402

HUGE = 3 << 20                    # 3 MiB of token text: far below the 64 MiB tokenizer cap, far above anything a record may carry
DETAIL_MAX = 700                  # what one problem detail or incomplete line may hold


def detail_of(rec, name):
    return next(p["detail"] for p in rec["problems"] if p["name"] == name)


# ---- finding 1: the length of a displayed token ------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["missing", "extra", "id", "special"])
def test_an_added_token_with_huge_text_gives_a_bounded_detail_and_keeps_the_count(tmp_path, kind):
    big = "x" * HUGE
    with_big = tokenizer_with(added(*BASE, (big, 20, True)))
    plain = tokenizer_with(added(*BASE))
    other_id = tokenizer_with(added(*BASE, (big, 21, True)))
    other_flag = tokenizer_with(added(*BASE, (big, 20, False)))
    served_tok, new_tok, name = {"missing": (plain, with_big, "added_token_missing_in_served"),
                                 "extra": (with_big, plain, "added_token_extra_in_served"),
                                 "id": (other_id, with_big, "added_token_id_differs"),
                                 "special": (other_flag, with_big, "added_token_special_differs")}[kind]
    new = make(tmp_path / "new", gen=GOOD_GEN, tokenizer=new_tok)
    served = make(tmp_path / "served", gen=GOOD_GEN, tokenizer=served_tok)
    detail = detail_of(sc.check_files(served, new), name)
    assert len(detail) < DETAIL_MAX
    assert detail.startswith("1 added tokens")                       # the count of differences is kept
    assert str(HUGE) in detail                                       # and the cut says how long the text was


def test_a_named_special_token_with_huge_text_gives_a_bounded_detail(tmp_path):
    big = "y" * HUGE
    new = make(tmp_path / "new", gen=GOOD_GEN, tokens={**TOK, "bos_token": big})
    served = make(tmp_path / "served", gen=GOOD_GEN, tokens={**TOK, "bos_token": "<s>"})
    assert len(detail_of(sc.check_files(served, new), "special_token_differs")) < DETAIL_MAX


def test_an_eos_token_with_huge_text_that_the_vocabulary_lacks_gives_a_bounded_incomplete_line(tmp_path):
    big = "z" * HUGE
    new = make(tmp_path / "new", gen=GOOD_GEN, tokens={**TOK, "eos_token": big})
    served = make(tmp_path / "served", gen=GOOD_GEN, tokens={**TOK, "eos_token": big})
    rec = sc.check_files(served, new)
    lines = [i for i in rec["incomplete"] if "eos token" in i]
    assert lines and all(len(i) < DETAIL_MAX for i in lines)


def test_an_eos_token_and_named_tokens_with_huge_text_and_different_ids_give_bounded_details(tmp_path):
    big = "w" * HUGE
    new_tok = tokenizer_with(added(*BASE, (big, 30, True)))
    served_tok = tokenizer_with(added(*BASE, (big, 31, True)))
    new = make(tmp_path / "new", gen=GOOD_GEN, tokenizer=new_tok, tokens={**TOK, "eos_token": big, "bos_token": big})
    served = make(tmp_path / "served", gen=GOOD_GEN, tokenizer=served_tok, tokens={**TOK, "eos_token": big, "bos_token": big})
    rec = sc.check_files(served, new)
    assert {"eos_id_differs", "special_token_id_differs"} <= names(rec)
    assert all(len(p["detail"]) < DETAIL_MAX for p in rec["problems"])


def test_an_end_id_whose_token_has_huge_text_gives_bounded_details(tmp_path):
    big = "v" * HUGE
    new = make(tmp_path / "new", gen={"eos_token_id": [7]}, tokenizer=tokenizer_with(added(("<|im_end|>", 9, True), (big, 7, True))))
    served = make(tmp_path / "served", gen={"eos_token_id": [7]}, tokenizer=tokenizer_with(added(("<|im_end|>", 7, True))))
    rec = sc.check_files(served, new)
    assert "end_id_token_differs" in names(rec)
    assert all(len(p["detail"]) < DETAIL_MAX for p in rec["problems"])
    lacking = make(tmp_path / "served2", gen={"eos_token_id": [7]}, tokenizer=tokenizer_with(added(("<|im_end|>", 9, True))))
    rec = sc.check_files(lacking, new)
    assert all(len(i) < DETAIL_MAX for i in rec["incomplete"])


def test_the_served_chat_record_stays_small_with_huge_token_text_and_still_catches_a_stop_token_in_a_reply(tmp_path):
    big = "u" * HUGE
    medium = "<|end|>" + "m" * 400                                     # long, but a reply can hold it
    tok = tokenizer_with(added(*BASE, (medium, 40, True)))
    new = make(tmp_path / "new", gen={"eos_token_id": [7, 8, 40]}, tokenizer=tok, tokens={**TOK, "bos_token": big})
    served = make(tmp_path / "served", gen={"eos_token_id": [7, 8, 40]}, tokenizer=tok, tokens={**TOK, "bos_token": big})
    sw = swap_check(tmp_path, [reply(content="Paris " + medium + " and more"), reply()])
    rec = sc.build_record(served, new, sw)
    assert len(json.dumps(rec)) < 200_000
    bos = rec["files"]["special_tokens"]["new"]["bos_token"]
    assert isinstance(bos, str) and len(bos) < 400 and str(HUGE) in bos          # shown cut, with its length
    assert "stop_token_in_reply" in {p["name"] for p in rec["problems"]}         # the check itself used the whole token
    assert all(len(p["detail"]) < DETAIL_MAX for p in rec["problems"])


def test_a_stop_token_found_in_a_reply_is_shown_cut(tmp_path):
    token = "<|stop|>" + "q" * 5000
    rec = suite([reply(content="x " + token), reply()])
    out = sc.check_replies(rec, end_tokens=[token])
    hit = next(p for p in out["problems"] if p["name"] == "stop_token_in_reply")
    assert len(hit["detail"]) < DETAIL_MAX


def test_a_short_token_is_shown_whole(tmp_path):
    new = make(tmp_path / "new", gen=GOOD_GEN, tokens={**TOK, "bos_token": "<s>"})
    served = make(tmp_path / "served", gen=GOOD_GEN, tokens={**TOK, "bos_token": "<bos>"})
    detail = detail_of(sc.check_files(served, new), "special_token_differs")
    assert "'<bos>'" in detail and "'<s>'" in detail and "characters" not in detail


def test_the_record_lists_are_capped(tmp_path):
    rec = sc.build_record(*pair(tmp_path), swap_check(tmp_path, [reply(), reply()]))
    many = {**rec, "problems": [{"name": f"p{i}", "detail": "d" * 5000} for i in range(500)], "incomplete": ["i" * 5000] * 500}
    out = sc.bounded_record(many)
    assert len(out["problems"]) <= sc.RECORD_LIST_MAX and all(len(p["detail"]) <= sc.RECORD_TEXT_MAX + 50 for p in out["problems"])
    assert len(out["incomplete"]) <= sc.RECORD_LIST_MAX and len(json.dumps(out)) < 400_000


# ---- finding 2: ledger-named records reach the missing-record checks ------------------------------------------------------

def test_a_stage_4_record_deleted_after_the_ledger_wrote_it_is_missing_even_without_a_stage_2_record(tmp_path):
    recs = {k: v for k, v in all_good(tmp_path).items() if k != "stage2"}
    chips = sorted(recs)[0]

    def delete(run):
        (run / f"stages/4/configs/{chips}" / sc.FILE).unlink()
    block, _ = chat_block(tmp_path / "run", recs, mutate=delete)
    assert "Dealt with: Not shown" in block and "missing" in block and "the ledger recorded it" in block
    assert "No stage sent a chat request" not in block


def test_a_ledger_named_record_that_is_present_but_has_no_stage_2_record_is_not_dealt_with(tmp_path):
    recs = {k: v for k, v in all_good(tmp_path).items() if k != "stage2"}
    block, _ = chat_block(tmp_path / "run", recs)
    assert "Dealt with: Not shown" in block and "stages/2/served-chat.json" in block


def test_a_run_with_no_record_and_no_ledger_entry_still_says_no_stage_sent_a_chat_request(tmp_path):
    block, _ = chat_block(tmp_path / "run", {}, ledger=False)
    assert "No stage sent a chat request" in block


# ---- the bundle's own copies of record text are capped too ------------------------------------------------------------------

@pytest.mark.parametrize("field", ["incomplete", "problem_name", "why_not"])
def test_the_bundle_cuts_huge_record_text_it_copies_into_risks_md(tmp_path, field):
    rec = all_good(tmp_path)
    big = "k" * (1 << 20)                                   # under the 8 MiB read cap: a record from an older or an unbounded writer
    s2 = dict(rec["stage2"])
    if field == "incomplete":
        s2["incomplete"] = [big, big]
    elif field == "problem_name":
        s2["problems"] = [{"name": big, "detail": "d"}]
    else:
        s2["replies"] = {**s2["replies"], "checked": False, "why_not": big}
    block, _ = chat_block(tmp_path / "run", {**rec, "stage2": s2})
    assert "Dealt with: Not shown" in block
    assert len(block) < 8_000

