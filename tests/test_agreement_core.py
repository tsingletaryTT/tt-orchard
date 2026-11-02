# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The agreement set's shared arithmetic (issues 15 and 18): orchard/agreement_core.py and its template copy, the shipped prompt set, the
damage control, the control verdict, the gate's reading of a record, and the long-context sweep design. No chip and no network."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from orchard import agreement_core as core
from orchard import defaults

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "weights-swap-templates"


def ref(i, n=32):
    return [(j * 3 + i) % 97 for j in range(n)]


def entry(i, wrong_every=None, n=32):
    r = ref(i, n)
    chip = list(r)
    if wrong_every:
        for k in range(wrong_every - 1, n, wrong_every):
            chip[k] = r[k] + 1
    return {"id": f"p{i}", "kind": "chat", "reference_ids": r, "chip_ids": chip}


def good_entries(n_prompts=6):
    return [entry(i) for i in range(n_prompts)]


# ---- the file the templates copy ----------------------------------------------------------------------------------------

def test_the_template_copy_is_byte_identical_to_the_shipped_file():
    assert (TEMPLATES / "agreement_core.py").read_bytes() == (REPO / "orchard" / "agreement_core.py").read_bytes()


def test_the_template_copy_imports_only_the_standard_library():
    import ast
    tree = ast.parse((TEMPLATES / "agreement_core.py").read_text(encoding="utf-8"))
    names = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert names <= {"__future__", "hashlib", "json", "math"}


def test_the_bars_here_equal_the_bars_in_defaults():
    assert core.BASELINE_TOP1_MIN == defaults.SWAP_TOP1_MIN
    assert core.POOLED_TOP1_MIN == defaults.AGREEMENT_POOLED_MIN
    assert core.PROMPT_TOP1_MIN == defaults.AGREEMENT_PROMPT_MIN
    assert core.MIN_PROMPTS == defaults.AGREEMENT_MIN_PROMPTS
    assert core.MIN_TOKENS == defaults.AGREEMENT_MIN_TOKENS
    assert core.CONTROL_MARGIN == defaults.AGREEMENT_CONTROL_MARGIN


# ---- the shipped prompt set -----------------------------------------------------------------------------------------------

def shipped_set():
    return json.loads((TEMPLATES / "agreement_prompts.json").read_text(encoding="utf-8"))


def test_the_shipped_prompt_set_is_valid_and_covers_the_kinds_issue_15_asks_for():
    data = shipped_set()
    assert core.prompt_set_problems(data) == []
    kinds = {p["kind"] for p in data["prompts"]}
    assert {"chat", "code", "tool-call", "non-english", "long-context"} <= kinds
    assert len(data["prompts"]) >= defaults.AGREEMENT_MIN_PROMPTS


def test_the_shipped_long_prompt_is_the_one_the_generator_makes():
    long = next(p for p in shipped_set()["prompts"] if p["kind"] == "long-context")
    assert long["messages"][0]["content"] == core.long_prompt_text(core.LONG_PROMPT_TOKENS)
    assert long["approx_tokens"] == core.LONG_PROMPT_TOKENS


def test_the_long_prompt_is_deterministic_and_grows_with_its_argument():
    assert core.long_prompt_text(500) == core.long_prompt_text(500)
    assert len(core.long_prompt_text(2000)) > 3 * len(core.long_prompt_text(500))
    assert "violet-4417" in core.long_prompt_text(500)
    for bad in (0, 99, True, 2.5, "500", None):
        with pytest.raises(ValueError):
            core.long_prompt_text(bad)


@pytest.mark.parametrize("data", [None, [], {}, {"prompts": "x"}, {"prompts": []}, {"prompts": [None] * 5},
                                  {"prompts": [{"id": "a", "kind": "k", "messages": []}] * 5}])
def test_a_bad_prompt_set_is_refused(data):
    assert core.prompt_set_problems(data)


def test_a_duplicate_prompt_id_is_refused():
    data = copy.deepcopy(shipped_set())
    data["prompts"][1]["id"] = data["prompts"][0]["id"]
    assert any("duplicate" in p for p in core.prompt_set_problems(data))


# ---- scoring ------------------------------------------------------------------------------------------------------------

def test_score_counts_matches_and_names_the_first_mismatch():
    s = core.score_prompt([1, 2, 3, 4], [1, 2, 9, 4])
    assert s == {"n": 4, "matches": 3, "top1": 0.75, "first_mismatch": 2}
    assert core.score_prompt([1, 2], [1, 2])["first_mismatch"] is None
    assert core.score_prompt([1, 2], [None, 2])["matches"] == 1


@pytest.mark.parametrize("ref_ids,chip", [([], []), ([1], [1, 2]), ([1, 2], [1]), ("ab", "ab"), ([True], [True]), ([1.5], [1.5]), (None, None)])
def test_score_refuses_malformed_ids(ref_ids, chip):
    with pytest.raises(ValueError):
        core.score_prompt(ref_ids, chip)


def test_a_good_set_passes():
    s = core.summarise(good_entries())
    assert s["passes"] is True and s["pooled_top1"] == 1.0 and s["reasons"] == []


def test_a_set_with_too_few_prompts_fails():
    s = core.summarise(good_entries(4))
    assert s["passes"] is False and any("at least 5" in r for r in s["reasons"])


def test_a_prompt_with_too_few_tokens_fails():
    entries = good_entries()
    entries[2] = entry(2, n=8)
    assert any("compared 8 tokens" in r for r in core.summarise(entries)["reasons"])


def test_one_drifting_prompt_cannot_hide_in_the_pool():
    entries = good_entries(8)
    entries[3]["chip_ids"] = [x + 1 for x in entries[3]["reference_ids"]]     # that prompt agrees on 0 of 32
    s = core.summarise(entries)
    assert s["pooled_top1"] > core.POOLED_TOP1_MIN            # the pool alone passes
    assert s["passes"] is False and s["worst_id"] == "p3"
    assert any("'p3'" in r and "per-prompt bar" in r for r in s["reasons"])


def test_a_malformed_entry_is_one_reason_and_the_rest_are_scored():
    entries = good_entries() + [None, {"id": "x"}, "text"]
    s = core.summarise(entries)
    assert s["passes"] is False and len(s["per_prompt"]) == 6
    assert sum(1 for r in s["reasons"] if "not an object" in r) == 2           # None and "text"
    assert any("'x'" in r for r in s["reasons"])                                 # the entry with no ids


@pytest.mark.parametrize("entries", [None, "x", 5, {}])
def test_entries_that_are_not_a_list_are_a_reason(entries):
    assert core.summarise(entries)["passes"] is False


# ---- the damage control -----------------------------------------------------------------------------------------------------------

def test_the_damaged_copy_changes_every_fourth_token_and_nothing_else():
    r = list(range(100, 132))
    d = core.damaged_copy(r)
    assert [k for k in range(32) if d[k] != r[k]] == list(range(3, 32, 4))
    assert core.damaged_copy(r, vocab_size=140) != r
    assert all(0 <= x < 140 for x in core.damaged_copy([139] * 8, vocab_size=140))


def test_the_shipped_bars_reject_the_damage_control():
    cal = core.calibrate(good_entries())
    assert cal["usable"] is True and cal["real_passes"] is True
    assert cal["damaged_pooled_top1"] == 0.75 and cal["damaged_passes"] is False


def test_bars_that_pass_the_damage_control_are_reported_unusable():
    cal = core.calibrate(good_entries(), pooled_min=0.5, prompt_min=0.5)
    assert cal["usable"] is False and "cannot tell" in cal["reason"]


def test_calibrate_with_nothing_scorable_is_unusable():
    assert core.calibrate([None, "x"])["usable"] is False


# ---- the control ------------------------------------------------------------------------------------------------------------------

def test_a_control_that_agrees_as_well_as_the_new_model_does_not_separate():
    new = core.summarise(good_entries())
    same = core.summarise(good_entries())
    v = core.control_verdict(new, same)
    assert v["present"] is True and v["separates"] is False and "cannot tell" in v["reason"]


def test_a_control_well_below_the_new_model_separates():
    v = core.control_verdict(core.summarise(good_entries()), core.summarise([entry(i, wrong_every=2) for i in range(6)]))
    assert v["separates"] is True


def test_no_control_is_reported_as_absent():
    assert core.control_verdict(core.summarise(good_entries()), None) == {"present": False, "separates": None, "margin": core.CONTROL_MARGIN}


# ---- the gate's judgement of a record -----------------------------------------------------------------------------------------------

def record(entries=None, **extra):
    return {"ran": True, "entries": entries if entries is not None else good_entries(), **extra}


def test_a_record_that_did_not_run_has_no_reasons():
    for r in (None, {}, {"ran": False}, "x", 3, []):
        assert core.judge(r)["reasons"] == []


def test_a_good_record_passes_and_a_base_weights_run_by_mistake_fails():
    assert core.judge(record())["reasons"] == []
    # The base weights stood in: agreement drops on every prompt (issue 15's 0.78 against 0.94).
    bad = record([entry(i, wrong_every=4) for i in range(6)])
    verdict = core.judge(bad)
    assert verdict["passes"] is False and any("pooled top-1" in r for r in verdict["reasons"])


def test_a_record_judged_with_bars_that_pass_the_damage_control_is_refused_as_unusable():
    verdict = core.judge(record(), pooled_min=0.5, prompt_min=0.5)
    assert verdict["passes"] is False and any("bars are unusable" in r for r in verdict["reasons"])


def test_a_control_that_is_not_separated_fails_the_record():
    ctl = {"entries": good_entries()}
    verdict = core.judge(record(control=ctl))
    assert verdict["passes"] is False and any("cannot tell" in r for r in verdict["reasons"])


def weak_control(**change):
    entries = [entry(i, wrong_every=2) for i in range(6)]
    for e in entries:
        e.update(change)
    return {"entries": entries}


def test_a_complete_control_that_agrees_poorly_lets_the_new_model_pass():
    assert core.judge(record(control=weak_control()))["reasons"] == []


def test_a_control_with_failed_requests_is_never_read_as_a_weak_model():
    ctl = weak_control()
    for e in ctl["entries"]:
        e["chip_ids"] = [None] * len(e["chip_ids"])          # every request of the control failed: agreement 0, a "perfect" margin
        e["error"] = "HTTP 500"
    v = core.judge(record(control=ctl))
    assert v["passes"] is False and any("control cannot be used" in r and "failed request" in r for r in v["reasons"])


def test_a_control_with_one_missing_id_or_other_reference_ids_or_other_prompts_is_refused():
    base = weak_control()
    missing = json.loads(json.dumps(base))
    missing["entries"][1]["chip_ids"][5] = None
    other_ref = json.loads(json.dumps(base))
    other_ref["entries"][2]["reference_ids"][0] += 1
    short = {"entries": base["entries"][:5]}
    extra = json.loads(json.dumps(base))
    extra["entries"].append({**extra["entries"][0], "id": "zzz"})
    twice = {"entries": base["entries"] + [base["entries"][0]]}
    for ctl, word in ((missing, "missing chip ids"), (other_ref, "other reference ids"), (short, "is missing"), (extra, "not in the run's set"),
                      (twice, "more than once")):
        v = core.judge(record(control=ctl))
        assert v["passes"] is False and any(word in r for r in v["reasons"]), (word, v["reasons"])


@pytest.mark.parametrize("ctl", [None, 3, "x", [], {"entries": None}, {"entries": [None]}, {"entries": [{"id": ["a"]}]}, {"entries": "abc"}])
def test_a_hostile_control_is_a_reason_not_a_crash(ctl):
    v = core.judge(record(control=ctl)) if ctl is not None else core.judge(record())
    assert isinstance(v["reasons"], list)
    if ctl is not None:
        assert v["passes"] is False


def test_the_prompt_ids_must_be_the_expected_set_each_once():
    ids = [f"p{i}" for i in range(6)]
    assert core.judge(record(), expected_ids=ids)["reasons"] == []
    five_copies = [dict(entry(0), id="p0") for _ in range(6)]
    v = core.judge(record(five_copies), expected_ids=ids)
    assert any("more than once" in r for r in v["reasons"]) and any("missing" in r for r in v["reasons"])
    assert any("not in the shipped set" in r for r in core.judge(record(), expected_ids=ids[:5])["reasons"])
    assert core.id_problems([{"id": ["x"]}], ids) == ["every entry of the agreement set needs a string id"]
    assert core.id_problems(None, ids)


def test_a_prompt_with_a_recorded_error_is_a_reason():
    entries = good_entries()
    entries[1]["error"] = "HTTP 500"
    assert any("failed request" in r for r in core.judge(record(entries))["reasons"])


def test_a_record_that_asked_for_a_control_and_holds_none_fails():
    assert any("asked for a control" in r for r in core.judge(record(control_expected=True))["reasons"])


@pytest.mark.parametrize("junk", [None, 7, "x", [], {}, [None], {"id": 1}, [{"reference_ids": "abc", "chip_ids": 5}], 10 ** 400])
def test_a_hostile_record_gives_reasons_and_never_raises(junk):
    for field in ("entries", "control"):
        r = judge_input(field, junk)
        v = core.judge(r)
        assert isinstance(v["reasons"], list)


def judge_input(field, junk):
    r = record()
    r[field] = junk
    return r


def test_the_baseline_record_keeps_the_old_bar():
    assert core.baseline_record(0.94, 32) == {"check": "32-token teacher-forced, one prompt", "top1_agreement": 0.94, "n_tokens": 32,
                                              "bar": defaults.SWAP_TOP1_MIN}


# ---- the long-context sweep design (issue 18) -----------------------------------------------------------------------------------------------

def test_the_default_sweep_covers_8k_to_200k_and_four_sequences():
    plan = core.sweep_plan()
    lengths = {c["input_tokens"] for c in plan}
    assert {8192, 32768, 100000, 200000} <= lengths and {c["concurrency"] for c in plan} == {1, 4}
    assert plan == sorted(plan, key=lambda c: (c["input_tokens"], c["concurrency"]))


def test_long_cells_are_checked_against_the_base_on_the_chip_and_short_ones_against_the_cpu():
    by = {c["input_tokens"]: c["reference"] for c in core.sweep_plan()}
    assert by[1024] == "cpu" and by[8192] == "base-on-chip" and by[200000] == "base-on-chip"


@pytest.mark.parametrize("lengths,conc", [([], [1]), ([1], []), ([0], [1]), ([1], [-1]), ([True], [1]), ([1.5], [1])])
def test_a_bad_sweep_request_is_refused(lengths, conc):
    with pytest.raises(ValueError):
        core.sweep_plan(lengths, conc)


def test_the_sweep_summary_flags_a_decode_drop_and_counts_missing_cells():
    cells = [{"input_tokens": 1024, "concurrency": 1, "prefill_tok_s": 900.0, "decode_tok_s": 20.0},
             {"input_tokens": 8192, "concurrency": 1, "prefill_tok_s": 800.0, "decode_tok_s": 18.0},
             {"input_tokens": 32768, "concurrency": 1, "prefill_tok_s": 500.0, "decode_tok_s": 6.0},
             {"input_tokens": 100000, "concurrency": 1, "decode_tok_s": None, "error": "timeout"},
             "junk"]
    s = core.sweep_summary(cells)
    assert s["missing_cells"] == 1
    assert [d["input_tokens"] for d in s["decode_drops"]] == [32768] and s["decode_drops"][0]["share_of_shortest"] == 0.3
    assert s["cells"][3]["decode_tok_s"] is None and s["cells"][3]["error"] == "timeout"
