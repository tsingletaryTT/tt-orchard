# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Third review of PR 114 (issue 107). Finding 1: the gate compares the whole saved decision with the recomputed one, field by field, because the card and the
bundle read `needed`, `numbers` and `thresholds` and not only `choice` and `flag`. Findings 2 and 3 (card and bundle wording) are in later sections."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_sampling_choice import NEEDED, NOT_NEEDED, side, variant  # noqa: E402
from test_sampling_choice_files import make_run, report, save, write  # noqa: E402

from orchard import sampling_choice as sc  # noqa: E402


def saved(tmp_path, **kw):
    run, s2 = make_run(tmp_path, variant=report(drafter="off", ignored=False, seconds=6.0), **kw)
    d = save(run, s2)
    return run, s2, d


def forge(s2, d, **edit):
    out = copy.deepcopy(d)
    for k, v in edit.items():
        if k == "numbers":
            out["numbers"].update(v)
        else:
            out[k] = v
    sc.write_decision(s2, out, trial=True)


def test_the_genuine_decision_passes(tmp_path):
    run, s2, d = saved(tmp_path)
    assert sc.decision_problems(s2, run) == []


@pytest.mark.parametrize("field,value", [
    ("needed", False), ("need_source", "operator"), ("need_evidence", ["made up"]), ("thresholds", {"min_tok_s": 1.0}),
    ("reason", "everything is fine"),
])
def test_a_forged_top_level_field_is_refused_and_named(tmp_path, field, value):
    run, s2, d = saved(tmp_path)
    forge(s2, d, **{field: value})
    problems = sc.decision_problems(s2, run)
    assert any(f"`{field}`" in p for p in problems), problems


@pytest.mark.parametrize("key,value", [("served_verdict", "sampled-full-speed"), ("served_tok_s", 999.0), ("variant_tok_s", 999.0),
                                       ("variant_verdict", "sampled-ignored"), ("served_speed_basis", "greedy"), ("served_drafter", "off")])
def test_a_forged_number_is_refused_and_named(tmp_path, key, value):
    run, s2, d = saved(tmp_path)
    forge(s2, d, numbers={key: value})
    problems = sc.decision_problems(s2, run)
    assert any(f"`numbers.{key}`" in p for p in problems), problems


def test_the_review_example_needed_true_and_applied_with_choice_and_flag_unchanged_is_refused(tmp_path):
    run, s2 = make_run(tmp_path, needed=False)
    d = save(run, s2)
    assert d["choice"] == "keep" and d["flag"] is None and d["needed"] is False
    forge(s2, d, needed=True, numbers={"served_verdict": "sampled-full-speed"})
    problems = sc.decision_problems(s2, run)
    assert any("`needed`" in p for p in problems) and any("`numbers.served_verdict`" in p for p in problems)


def test_the_choice_and_the_flag_are_still_named(tmp_path):
    run, s2, d = saved(tmp_path)
    forge(s2, d, choice="keep")
    assert any("chose 'keep'" in p for p in sc.decision_problems(s2, run))
    run2, s22, d2 = saved(tmp_path / "b")
    forge(s22, d2, flag="unverified")
    assert any("flag" in p for p in sc.decision_problems(s22, run2))


def test_a_skipped_trial_with_its_note_still_passes(tmp_path):
    run, s2 = make_run(tmp_path)
    d = sc.decide(sc.need(run), sc.served_side(s2), None, variant_note="the run uses a lab")
    sc.write_decision(s2, d, trial=False)
    assert d["variant_note"] == "the run uses a lab" and sc.decision_problems(s2, run) == []


def test_a_forged_note_with_the_reason_left_alone_is_refused(tmp_path):
    run, s2 = make_run(tmp_path)
    d = sc.decide(sc.need(run), sc.served_side(s2), None, variant_note="the run uses a lab")
    d["variant_note"] = "something else"
    sc.write_decision(s2, d, trial=False)
    assert any("`reason`" in p for p in sc.decision_problems(s2, run))


def test_only_the_decision_file_is_read_for_the_note_when_it_is_a_string(tmp_path):
    run, s2 = make_run(tmp_path)
    d = sc.decide(sc.need(run), sc.served_side(s2), None, variant_note="x")
    d["variant_note"] = ["not", "a", "string"]
    sc.write_decision(s2, d, trial=False)
    assert any("`variant_note`" in p or "`reason`" in p for p in sc.decision_problems(s2, run))


def test_a_decision_with_an_extra_unknown_key_is_not_refused_for_it(tmp_path):
    run, s2, d = saved(tmp_path)
    forge(s2, d, comment="a person added this")
    assert sc.decision_problems(s2, run) == []


def test_a_note_that_is_not_a_string_is_refused_even_when_the_reason_does_not_use_it(tmp_path):
    run, s2, d = saved(tmp_path)
    forge(s2, d, variant_note=5)
    assert any("`variant_note`" in p for p in sc.decision_problems(s2, run))


def test_a_missing_number_is_refused(tmp_path):
    run, s2, d = saved(tmp_path)
    out = copy.deepcopy(d)
    del out["numbers"]["served_tok_s"]
    sc.write_decision(s2, out, trial=True)
    assert any("`numbers.served_tok_s`" in p for p in sc.decision_problems(s2, run))


@pytest.mark.parametrize("bad", [[], "x", None, 3])
def test_numbers_that_are_not_an_object_are_refused_without_a_crash(tmp_path, bad):
    run, s2, d = saved(tmp_path)
    forge(s2, d, numbers={})
    out = copy.deepcopy(d)
    out["numbers"] = bad
    sc.write_decision(s2, out, trial=True)
    assert any("`numbers." in p for p in sc.decision_problems(s2, run))
