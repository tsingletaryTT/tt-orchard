# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 297: the follow-ups from the reviews of PRs 292, 294 and 295 that were still open after PR 396.

Items covered here: (1) added-token ids and the `special` flag, (2) the stop-id rule per serving backend, (3) a deleted record that the ledger
names is reported as missing, (4) digests taken again after the last reply. Items 5 and 6 are in tests/test_floor_report.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_served_chat import (GOOD_GEN, TOK, TOKENIZER_JSON, all_good, chat_block, make, names, pair, reply, suite, swap_check, TEMPLATE_A)  # noqa: E402

from orchard import served_chat as sc  # noqa: E402


def tokenizer_with(added):
    return {**TOKENIZER_JSON, "added_tokens": added}


def added(*rows):
    return [{"id": i, "content": c, "special": s} for c, i, s in rows]


BASE = (("<|im_end|>", 7, True), ("<|endoftext|>", 8, True), ("<|im_start|>", 9, True))


# ---- item 1: added tokens -------------------------------------------------------------------------------------------------

def test_an_added_token_with_another_id_in_the_served_tokenizer_is_a_problem(tmp_path):
    new = make(tmp_path / "new", gen=GOOD_GEN, tokenizer=tokenizer_with(added(*BASE, ("<think>", 20, False))))
    served = make(tmp_path / "served", gen=GOOD_GEN, tokenizer=tokenizer_with(added(*BASE, ("<think>", 21, False))))
    rec = sc.check_files(served, new)
    assert "added_token_id_differs" in names(rec)
    assert "<think>" in next(p["detail"] for p in rec["problems"] if p["name"] == "added_token_id_differs")


def test_an_added_token_whose_special_flag_differs_is_a_problem(tmp_path):
    new = make(tmp_path / "new", gen=GOOD_GEN, tokenizer=tokenizer_with(added(*BASE, ("<think>", 20, True))))
    served = make(tmp_path / "served", gen=GOOD_GEN, tokenizer=tokenizer_with(added(*BASE, ("<think>", 20, False))))
    assert "added_token_special_differs" in names(sc.check_files(served, new))


def test_an_added_token_the_served_tokenizer_lacks_or_has_extra_is_a_problem(tmp_path):
    new = make(tmp_path / "new", gen=GOOD_GEN, tokenizer=tokenizer_with(added(*BASE, ("<think>", 20, False))))
    served = make(tmp_path / "served", gen=GOOD_GEN, tokenizer=tokenizer_with(added(*BASE, ("<extra>", 21, False))))
    got = names(sc.check_files(served, new))
    assert {"added_token_missing_in_served", "added_token_extra_in_served"} <= got


def test_the_same_added_tokens_pass(tmp_path):
    served, new = pair(tmp_path)
    assert not {n for n in names(sc.check_files(served, new)) if n.startswith("added_token")}


def test_a_huge_added_token_list_gives_a_bounded_record(tmp_path):
    rows = [(f"<t{i}>", 1000 + i, False) for i in range(200_000)]
    moved = [(c, i + 1, s) for c, i, s in rows]
    new = make(tmp_path / "new", gen=GOOD_GEN, tokenizer=tokenizer_with(added(*BASE, *rows)))
    served = make(tmp_path / "served", gen=GOOD_GEN, tokenizer=tokenizer_with(added(*BASE, *moved)))
    rec = sc.check_files(served, new)
    hits = [p for p in rec["problems"] if p["name"] == "added_token_id_differs"]
    assert len(hits) == 1 and len(hits[0]["detail"]) < 1500 and "200000" in hits[0]["detail"]
    assert len(json.dumps(rec)) < 60_000


# ---- item 2: the stop-id rule depends on the serving backend ---------------------------------------------------------------

def write_swap_check(tmp_path, served, **extra):
    p = tmp_path / "swap-check.json"
    p.write_text(json.dumps({"prompt_suite": suite([reply(), reply()], files=sc.file_digests(served)), **extra}))
    return p


def test_a_bundle_swap_check_uses_the_bundle_stop_id_rule(tmp_path):
    served, new = pair(tmp_path)
    rec = sc.build_record(served, new, write_swap_check(tmp_path, served))
    assert rec["ok"] is True and rec["files"]["stop_id_rule"].startswith("generation_config.json")


def test_a_container_swap_check_is_incomplete_because_its_stop_id_rule_is_not_known(tmp_path):
    served, new = pair(tmp_path)
    rec = sc.build_record(served, new, write_swap_check(tmp_path, served, kind="container"))
    assert rec["ok"] is False and any("container" in i and "stop id" in i for i in rec["incomplete"])
    assert rec["files"]["stop_id_rule"] is None


def test_a_container_record_raises_no_stop_id_problem_it_cannot_judge(tmp_path):
    served, new = pair(tmp_path, gen={"eos_token_id": [8]})                      # would be stop_id_missing under the bundle rule
    rec = sc.build_record(served, new, write_swap_check(tmp_path, served, kind="container"))
    assert "stop_id_missing" not in names(rec) and rec["ok"] is False


def test_an_unrecognised_or_hostile_kind_is_treated_as_an_unknown_backend(tmp_path):
    served, new = pair(tmp_path)
    for kind in ("somethingelse", 5, ["bundle"]):
        rec = sc.build_record(served, new, write_swap_check(tmp_path, served, kind=kind))
        assert rec["ok"] is False and any("stop id" in i for i in rec["incomplete"])


def test_check_files_takes_the_backend_and_defaults_to_the_bundle(tmp_path):
    served, new = pair(tmp_path, gen={"eos_token_id": [8]})
    assert "stop_id_missing" in names(sc.check_files(served, new))
    assert "stop_id_missing" not in names(sc.check_files(served, new, backend="container"))


# ---- item 4: digests again after the last reply ----------------------------------------------------------------------------

def test_a_swap_check_with_no_digests_after_the_last_reply_is_incomplete(tmp_path):
    served, new = pair(tmp_path)
    p = tmp_path / "swap-check.json"
    p.write_text(json.dumps({"prompt_suite": suite([reply(), reply()], files=sc.file_digests(served), after=None)}))
    rec = sc.build_record(served, new, p)
    assert rec["ok"] is False and any("served_files_after" in i for i in rec["incomplete"])


def test_a_file_that_changed_between_the_first_and_the_last_digest_is_a_problem(tmp_path):
    served, new = pair(tmp_path)
    before = sc.file_digests(served)
    after = {**before, "generation_config.json": "0" * 64}
    p = tmp_path / "swap-check.json"
    p.write_text(json.dumps({"prompt_suite": suite([reply(), reply()], files=before, after=after)}))
    rec = sc.build_record(served, new, p)
    assert "inputs_changed_during_replies" in names(rec) and rec["ok"] is False
    assert "generation_config.json" in next(x["detail"] for x in rec["problems"] if x["name"] == "inputs_changed_during_replies")


def test_a_tokenizer_that_could_not_be_hashed_makes_the_record_incomplete(tmp_path):
    served, new = pair(tmp_path)
    dig = {**sc.file_digests(served), "tokenizer.json": None}                      # refused or unreadable when the test ran, and again after
    p = tmp_path / "swap-check.json"
    p.write_text(json.dumps({"prompt_suite": suite([reply(), reply()], files=dig, after=dig)}))
    rec = sc.build_record(served, new, p)
    assert rec["ok"] is False and any("tokenizer.json" in i and "hashed" in i for i in rec["incomplete"])


def test_a_non_dict_served_files_after_is_incomplete_and_never_raises(tmp_path):
    served, new = pair(tmp_path)
    for bad in ("x", 5, ["a"]):
        p = tmp_path / "swap-check.json"
        p.write_text(json.dumps({"prompt_suite": suite([reply(), reply()], files=sc.file_digests(served), after=bad)}))
        rec = sc.build_record(served, new, p)
        assert rec["ok"] is False and any("served_files_after" in i for i in rec["incomplete"])


# ---- item 3: a deleted record the ledger names is "missing" -----------------------------------------------------------------

def test_a_stage_2_record_deleted_after_the_ledger_wrote_it_is_reported_as_missing(tmp_path):
    def delete(run):
        (run / "stages/2" / sc.FILE).unlink()
    block, _ = chat_block(tmp_path / "run", all_good(tmp_path), mutate=delete)
    assert "Dealt with: Not shown" in block and "missing" in block and "the ledger recorded it" in block
    assert "No stage sent a chat request" not in block
