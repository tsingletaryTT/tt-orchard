# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/repair_help.py and the draft checks it relies on (orchard/phone_a_friend.py): the request chain and progress, when the helper is asked
in a repair round, what is sent, when a reply is placed, and the completeness checks that keep a stub out of the port."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from orchard import phone_a_friend as paf
from orchard import port_facts, port_repair, repair_help as rh
from port_check_fakes import HONEST


def entry(seq, event, stage=2, **data):
    return {"seq": seq, "event": event, "stage": stage, "data": data}


def decision(seq, name, **data):
    return entry(seq, "decision", decision=name, **data)


# ---- the request chain -----------------------------------------------------------------------------------------------------------------------

def test_the_chain_runs_init_load_prefill_decode_and_a_failure_before_any_request_is_first():
    assert [rh.rank(op) for op in (None, "init", "load", "prefill", "decode")] == [0, 1, 2, 3, 4]
    assert rh.rank("close") is None and rh.rank(5) is None


def test_progress_is_a_failure_later_in_the_chain_and_nothing_else():
    assert rh.progressed("load", "prefill") and rh.progressed(None, "load") and rh.progressed("prefill", "decode")
    assert not rh.progressed("prefill", "prefill") and not rh.progressed("prefill", "load")
    assert not rh.progressed("load", "mystery") and not rh.progressed("mystery", "decode")      # an op nobody knows never counts


# ---- when to ask -----------------------------------------------------------------------------------------------------------------------------

def rounds(*ops):
    """A ledger with one attempt and a `port repair round` entry per op (round 1, 2, ...)."""
    out = [entry(1, "stage_start")]
    for i, op in enumerate(ops, start=1):
        out.append(decision(1 + i, port_repair.DECISION, round=i, op=op))
    return out


def test_the_first_round_of_an_attempt_asks():
    assert rh.start_trigger(rounds("load"), 2, 1, {"op": "load"})[0] is True


def test_a_round_whose_failure_did_not_move_asks_again():
    ask, why = rh.start_trigger(rounds("load", "load"), 2, 2, {"op": "load"})
    assert ask is True and "did not move on" in why


def test_a_round_whose_failure_moved_further_lets_the_coder_go_first():
    ask, why = rh.start_trigger(rounds("load", "prefill"), 2, 2, {"op": "prefill"})
    assert ask is False and "further" in why


def test_a_failure_that_moved_back_asks():
    assert rh.start_trigger(rounds("prefill", "load"), 2, 2, {"op": "load"})[0] is True


def test_a_round_the_ledger_does_not_know_asks():
    assert rh.start_trigger([entry(1, "stage_start")], 2, 3, {"op": "load"})[0] is True


def test_the_previous_round_is_read_from_this_attempt_only():
    entries = rounds("prefill") + [entry(9, "stage_start"), decision(10, port_repair.DECISION, round=1, op="load")]
    assert rh.previous_op(entries, 2, 2) == (True, "load")


# ---- the limits ------------------------------------------------------------------------------------------------------------------------------

def test_one_ask_per_round_and_the_ledger_is_what_counts():
    entries = rounds("load") + [decision(5, rh.ASKED, round=1)]
    ok, why = rh.due(entries, 2, 1)
    assert ok is False and "round 1" in why
    assert rh.due(entries, 2, 2)[0] is True


def test_a_round_asked_in_an_earlier_attempt_does_not_block_the_same_round_now():
    entries = rounds("load") + [decision(5, rh.ASKED, round=1), entry(6, "stage_start")]
    assert rh.due(entries, 2, 1)[0] is True


def test_six_asks_in_the_stage_count_the_attempt_start_asks_and_the_round_asks_together():
    entries = [entry(1, "stage_start")] + [decision(2 + i, paf.ASKED) for i in range(2)] + [decision(10 + i, rh.ASKED, round=i + 1) for i in range(3)]
    assert rh.asks_in_stage(entries, 2) == 5 and rh.due(entries, 2, 9)[0] is True
    entries.append(decision(30, rh.ASKED, round=4))
    ok, why = rh.due(entries, 2, 9)
    assert ok is False and "6" in why


def test_another_stage_does_not_count():
    entries = [decision(i, rh.ASKED, stage=3, round=i) for i in range(1, 9)]
    assert rh.asks_in_stage(entries, 2) == 0


# ---- what is sent ----------------------------------------------------------------------------------------------------------------------------

def test_the_request_holds_the_whole_port_the_facts_and_the_excerpts_and_no_home_path():
    port = "# /home/someone/secret/path\n" + HONEST
    task, sent = rh.build_task(facts="facts text", excerpts="excerpt text", port_text=port, listing="port_model.py  100 bytes", homes=("/home/someone",))
    assert task is not None and "def prefill(model, ids)" in task and "facts text" in task and "excerpt text" in task
    assert "/home/someone" not in task and "COMPLETE corrected" in task
    assert sent["port"]["cut"] is False and sent["port"]["chars"] > 0 and sent["facts"]["cut"] is False


def test_a_port_over_the_cap_is_not_sent_because_the_answer_to_a_cut_file_would_lose_the_rest(monkeypatch):
    monkeypatch.setattr(rh, "PORT_CAP", 50)
    task, sent = rh.build_task(facts=None, excerpts=None, port_text=HONEST, listing="", homes=())
    assert task is None and sent["port"]["cut"] is True


def test_the_failure_names_the_round_and_the_request():
    text = rh.failure(2, 3, "prefill", "FAILED: the port failed at 'prefill': boom\n", ())
    assert "round 2 of 3" in text and "'prefill'" in text and "boom" in text
    assert "before any request" in rh.failure(1, 3, None, "", ())


# ---- is a reply a whole port ----------------------------------------------------------------------------------------------------------------

def test_a_whole_port_has_no_problem():
    assert rh.reply_problems(HONEST) == []


@pytest.mark.parametrize("source,needle", [
    (HONEST.replace("def close(model):\n    pass\n", ""), "no close function"),
    (HONEST.replace("return {\"dir\": model_dir, \"device\": mesh_device}", "raise NotImplementedError('later')"), "load raises NotImplementedError"),
    (HONEST.replace("return _peak(100)", "pass"), "prefill is empty"),
    (HONEST.replace("def decode_step(model, token_id, position):", "def decode_step(model, token_id):"), "decode_step takes"),
    (HONEST.replace("def load(model_dir, mesh_device):", "def load(model_dir, mesh_device, extra):"), "load takes"),
    (HONEST.replace("return _peak(token_id + 1)", "return _peak(helper_nobody_defined(token_id))"), "uses names nothing defines: helper_nobody_defined"),
])
def test_a_reply_that_is_not_a_whole_port_says_why(source, needle):
    assert any(needle in p for p in rh.reply_problems(source)), rh.reply_problems(source)


def test_extra_defaulted_arguments_and_star_args_are_fine():
    assert rh.reply_problems(HONEST.replace("def prefill(model, ids):", "def prefill(model, ids, *, chunk=8):")) == []
    assert rh.reply_problems(HONEST.replace("def prefill(model, ids):", "def prefill(*args, **kw):")) == []


def test_a_name_bound_anywhere_is_not_undefined_and_builtins_and_comprehension_variables_are_known():
    src = "import os\nfrom math import sqrt as root\ndef f(a, *r, **k):\n    t = [x * 2 for x in range(a)]\n    with open(os.devnull) as fh:\n        n = len(t) + root(4)\n    return n, fh, r, k\n"
    assert paf.undefined_names(src) == []


def test_a_star_import_is_not_judged_and_a_text_that_does_not_parse_has_no_names():
    assert paf.undefined_names("from x import *\nprint(whatever)\n") == []
    assert paf.undefined_names("def (") == []


def test_the_merge_into_the_skeleton_refuses_a_draft_that_uses_a_name_nothing_defines():
    skeleton = Path(port_facts.SKELETON).read_text()
    merged, why = paf.merge_into_skeleton(skeleton, HONEST.replace("return _peak(100)", "return missing_helper(100)"))
    assert merged is None and "missing_helper" in why
    assert paf.merge_into_skeleton(skeleton, HONEST)[0] is not None


def test_each_problem_id_is_one_run_so_the_per_problem_caps_do_not_carry_over():
    assert paf.problem_id("/x/orchard-runs/tiny-qwen4-run12").startswith("full-port-stage-2:tiny-qwen4-run12-")      # the run's name, then a digest of its path
    assert paf.problem_id("/x/a") != paf.problem_id("/x/b")
    assert "/" not in paf.problem_id("/x/we ird/na me") and len(paf.problem_id("/x/" + "n" * 300)) < 80


# ---- putting the reply in place --------------------------------------------------------------------------------------------------------------

def make_port(tmp_path, text=HONEST):
    stage = tmp_path / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text(text)
    return stage, hashlib.sha256(text.encode()).hexdigest()


NEW = HONEST.replace("return _peak(100)", "return _peak(101)")


def test_a_reply_replaces_the_file_that_was_sent(tmp_path):
    stage, sha = make_port(tmp_path)
    placed, why, found = rh.place(stage, NEW, sha, coder_failed=False, call_id="c1", model="m")
    assert placed is True and why == "" and found == sha
    text = (stage / "port" / "port_model.py").read_text()
    assert text.startswith("# Replaced by the supervisor") and "_peak(101)" in text


def test_a_file_the_coder_changed_while_the_helper_answered_is_left_alone(tmp_path):
    stage, sha = make_port(tmp_path)
    (stage / "port" / "port_model.py").write_text(HONEST + "# the coder was here\n")
    placed, why, found = rh.place(stage, NEW, sha, coder_failed=False, call_id="c1", model="m")
    assert placed is False and "coder changed" in why and found != sha
    assert (stage / "port" / "port_model.py").read_text().endswith("# the coder was here\n")


def test_a_failed_coder_step_lets_the_reply_replace_whatever_the_coder_left(tmp_path):
    stage, sha = make_port(tmp_path)
    (stage / "port" / "port_model.py").write_text("def half(\n")          # the coder's step ended with a broken file
    placed, why, found = rh.place(stage, NEW, sha, coder_failed=True, call_id="c1", model="m")
    assert placed is True and found != sha and "_peak(101)" in (stage / "port" / "port_model.py").read_text()


def test_a_reply_equal_to_the_port_changes_nothing(tmp_path):
    stage, sha = make_port(tmp_path)
    before = (stage / "port" / "port_model.py").stat().st_mtime_ns
    placed, why, _ = rh.place(stage, HONEST, sha, coder_failed=False, call_id="c1", model="m")
    assert placed is False and "as it already is" in why and (stage / "port" / "port_model.py").stat().st_mtime_ns == before


def test_a_link_in_place_of_the_port_file_is_never_written_through(tmp_path):
    stage, sha = make_port(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("outside")
    os.remove(stage / "port" / "port_model.py")
    os.symlink(outside, stage / "port" / "port_model.py")
    placed, why, _ = rh.place(stage, NEW, sha, coder_failed=True, call_id="c1", model="m")
    assert placed is False and "link" in why and outside.read_text() == "outside"


def test_a_missing_port_file_is_not_created(tmp_path):
    stage = tmp_path / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    assert rh.place(stage, NEW, "0" * 64, coder_failed=True, call_id="c", model="m")[0] is False
    assert not (stage / "port" / "port_model.py").exists()


# ---- verdict bookkeeping ---------------------------------------------------------------------------------------------------------------------

def test_an_answer_without_a_verdict_is_listed_with_its_round_and_request():
    entries = [entry(1, "stage_start"), decision(2, rh.ASKED, round=1, failure_op="load"),
               decision(3, rh.ANSWERED, ok=True, call_id="aa", cached=False)]
    assert rh.unreported(entries, 2) == [{"call_id": "aa", "round": 1, "failure_op": "load", "failure_agreement": None, "seq": 3, "placed": False}]
    entries.append(decision(4, rh.PROGRESS, call_id="aa"))
    assert rh.unreported(entries, 2) == []


def test_an_answer_that_was_not_ok_or_was_cached_needs_no_verdict():
    entries = [decision(2, rh.ASKED, round=1, failure_op="load"), decision(3, rh.ANSWERED, ok=False, call_id="aa"),
               decision(4, rh.ANSWERED, ok=True, call_id="bb", cached=True)]
    assert rh.unreported(entries, 2) == []


def test_the_check_ran_again_only_when_a_tested_entry_follows_the_answer():
    entries = [decision(3, rh.ANSWERED, ok=True, call_id="aa")]
    assert rh.tested_after(entries, 2, 3) is False
    entries.append(decision(5, port_repair.TESTED))
    assert rh.tested_after(entries, 2, 3) is True and rh.tested_after(entries, 2, 5) is False
