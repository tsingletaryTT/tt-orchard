# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 433, review of PR 444 (limit round): a number in a report never raises, whatever its size or type.

An independent review found that `top1_agreement` written as an integer of 400 digits crashed the gate with `OverflowError`: `math.isfinite`
and `float()` both convert an integer to a float, and an integer over about 308 digits does not fit. JSON parses integers up to 4300
digits and the safe reader does not refuse them, so an agent could put one in swap-check.json, in the report's `result_draft` or in
result.json. The old gate compared with `!=` and could not raise. The exception also escaped the supervisor, because neither the gate
call in `_check_gate` nor the one in `_finished_check` caught it.

This file checks, with no device, no server and no model:
- the number helpers are total and compare integers exactly (no float conversion of an integer);
- every place the module reads a number out of a report or a result refuses a hostile value with a reason, and never raises;
- all three swap gates (stage 2, the sidecar gate, stage 4) return a refusal for each hostile value in each place;
- the supervisor turns an exception from any gate into a failed gate with the exception's class, and lets an interrupt through."""
import functools
import json
from types import SimpleNamespace

import pytest

from orchard import sampling_choice, token_check
from orchard.ledger import Ledger
from orchard.stages import GateResult, gate_mesh_swap, gate_weights_swap, gate_weights_swap_sidecar
from orchard.supervisor import Supervisor
from test_stages import swap_config
from test_token_check_gate import REF, mentions, mesh, served_report, sidecar_stage, stage2

HUGE = 10 ** 400
HOSTILE = [HUGE, -HUGE, 10 ** 4000, float("inf"), float("-inf"), float("nan"), True, False, "0.5", [0.5], {"v": 1}, None]
HOSTILE_IDS = ["huge", "minus-huge", "huge-4000-digits", "inf", "minus-inf", "nan", "true", "false", "text", "list", "object", "null"]
SOME = [HUGE, -HUGE, float("inf"), float("nan"), "0.5", 10 ** 99, 2, -1]
SOME_IDS = ["huge", "minus-huge", "inf", "nan", "text", "99-digits", "two", "minus-one"]


def refused(got):
    """A gate refused the file: it names the field, or the safe reader refused the file first. The safe reader (`safeio`, MAX_INT_DIGITS = 100,
    no NaN or Infinity) stands in front of every gate, so an integer over 100 digits and a non-finite number never reach the recompute through a
    gate; the recompute is still total on its own (the tests above call it directly), and the reader's refusal is a refusal."""
    return isinstance(got, GateResult) and not got.ok and bool(mentions(got.reasons, "top1_agreement") or mentions(got.reasons, "not readable JSON"))


# ---- the helpers ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", HOSTILE + [0, 1, 0.5, 3], ids=HOSTILE_IDS + ["zero", "one", "half", "three"])
def test_number_never_raises(value):
    assert token_check._number(value) in (True, False)


def test_an_integer_of_any_size_is_a_finite_number_and_a_boolean_or_a_non_finite_float_is_not():
    assert token_check._number(HUGE) is True and token_check._number(-HUGE) is True and token_check._number(7) is True
    assert token_check._number(0.5) is True
    for bad in (True, False, float("inf"), float("-inf"), float("nan"), "1", None, [1]):
        assert token_check._number(bad) is False


@pytest.mark.parametrize("recorded", HOSTILE, ids=HOSTILE_IDS)
def test_same_fraction_never_raises_and_refuses_a_hostile_value(recorded):
    assert token_check._same_fraction(recorded, 1.0) is False


def test_same_fraction_compares_integers_exactly():
    assert token_check._same_fraction(1, 1.0) is True and token_check._same_fraction(0, 0.0) is True
    assert token_check._same_fraction(2, 1.0) is False and token_check._same_fraction(-1, 1.0) is False
    assert token_check._same_fraction(1, 0.9375) is False and token_check._same_fraction(0, 0.0625) is False
    assert token_check._same_fraction(HUGE, 1.0) is False and token_check._same_fraction(-HUGE, 0.0) is False


def test_same_fraction_keeps_the_rounding_tolerance_for_floats():
    assert token_check._same_fraction(0.9375 + 1e-12, 0.9375) is True
    assert token_check._same_fraction(0.9375 + 1e-6, 0.9375) is False


# ---- every number the module reads from a report ---------------------------------------------------------------------------------

def perfect():
    """A swap-check.json whose replies match the reference everywhere: top1 1.0, no mismatches."""
    return served_report(bad=())


def check(report, results=()):
    return token_check.problems(report, list(results), list(REF), "stage 2: ", required_positions=32)


def test_the_perfect_report_has_no_problems():
    assert check(perfect()) == []


@pytest.mark.parametrize("value", HOSTILE, ids=HOSTILE_IDS)
@pytest.mark.parametrize("where", ["report", "result_draft", "result"])
def test_a_hostile_top1_agreement_is_refused_with_a_reason_in_every_place_it_can_be_written(where, value):
    r = perfect()
    results = []
    if where == "report":
        r["top1_agreement"] = value
    elif where == "result_draft":
        r["result_draft"]["top1_agreement"] = value
    else:
        results = [("result.json", {"n_tokens": 32, "top1_agreement": value})]
    got = check(r, results)
    assert mentions(got, "top1_agreement"), got
    assert all(len(x) <= token_check.MAX_REASON_CHARS + 200 for x in got)


@pytest.mark.parametrize("field", ["n_tokens", "matches"])
@pytest.mark.parametrize("value", HOSTILE, ids=HOSTILE_IDS)
def test_a_hostile_count_in_the_report_is_refused_and_never_raises(field, value):
    r = perfect()
    r[field] = value
    assert mentions(check(r), field)


@pytest.mark.parametrize("value", HOSTILE, ids=HOSTILE_IDS)
def test_a_hostile_n_tokens_in_a_result_is_refused_and_never_raises(value):
    got = check(perfect(), [("result.json", {"n_tokens": value, "top1_agreement": 1.0})])
    assert mentions(got, "result.json n_tokens")


@pytest.mark.parametrize("key", ["first_id", "position"])
@pytest.mark.parametrize("value", [HUGE, -HUGE, float("inf"), float("nan"), "7", [1], None, True], ids=["huge", "minus-huge", "inf", "nan", "text", "list", "null", "true"])
def test_a_hostile_value_inside_a_teacher_forced_record_is_refused_and_never_raises(key, value):
    r = perfect()
    r["teacher_forced"][3][key] = value
    got = check(r)
    # A null first id is the server replying with nothing: a mismatch the report does not list, so it is refused as well.
    assert got, got


@pytest.mark.parametrize("value", [HUGE, -HUGE, float("inf"), float("nan"), "7", [1], {"a": 1}, None, True], ids=["huge", "minus-huge", "inf", "nan", "text", "list", "object", "null", "true"])
def test_a_hostile_value_in_the_mismatch_list_is_refused_and_never_raises(value):
    r = served_report()
    r["mismatches"].append({"position": value})
    assert mentions(check(r), "mismatches")


def test_a_huge_integer_in_the_copy_of_the_reference_ids_is_refused_and_never_raises():
    r = perfect()
    r["reference_generated_ids"][5] = HUGE
    assert mentions(check(r), "reference_generated_ids")


def test_the_recompute_never_raises_even_if_something_inside_it_does(monkeypatch):
    def boom(*a, **k):
        raise OverflowError("int too large to convert to float")
    monkeypatch.setattr(token_check, "_problems", boom)
    got = check(perfect())
    assert len(got) == 1 and "OverflowError" in got[0] and "cannot be recomputed" in got[0]
    assert "int too large" not in got[0]            # the class only: the message may carry the report's own text


# ---- the three swap gates --------------------------------------------------------------------------------------------------------

def rewrite(path, change):
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))                # allow_nan: Infinity and NaN are written the way the template's json.dump would


def put(where, value):
    """A function that puts `value` in the place named by `where` of a swap-check.json (or, for "result", of a result file)."""
    def change(doc):
        if where == "result_draft":
            doc.setdefault("result_draft", {})["top1_agreement"] = value
        else:
            doc["top1_agreement"] = value
    return change


@pytest.mark.parametrize("value", SOME, ids=SOME_IDS)
@pytest.mark.parametrize("where", ["report", "result_draft", "result"])
@pytest.mark.parametrize("gate_name", ["stage2", "sidecar"])
def test_the_stage_2_gates_refuse_a_hostile_top1_agreement_and_never_raise(tmp_path, gate_name, where, value):
    if gate_name == "stage2":
        g, sd, run = stage2(tmp_path)
        gate = gate_weights_swap
    else:
        sd, run = sidecar_stage(tmp_path)
        gate = gate_weights_swap_sidecar
    target = sd / "result.json" if where == "result" else sd / "evidence" / "swap-check.json"
    rewrite(target, put(where, value))
    got = gate(sd, run)
    assert refused(got), getattr(got, "reasons", got)


@pytest.mark.parametrize("value", SOME, ids=SOME_IDS)
@pytest.mark.parametrize("where", ["report", "result_draft", "result"])
def test_the_stage_4_gate_refuses_a_hostile_top1_agreement_and_never_raises(tmp_path, where, value):
    g, sd, run = mesh(tmp_path, served_report(bad=()))
    assert g.ok, g.reasons
    if where == "result":
        rewrite(sd / "result.json", lambda d: d["configs"][0].__setitem__("top1_agreement", value))
    else:
        rewrite(run / "stages" / "4" / "configs" / "2" / "evidence" / "swap-check.json", put(where, value))
    got = gate_mesh_swap(sd, run)
    assert refused(got), getattr(got, "reasons", got)


def test_a_huge_top1_agreement_is_named_by_its_size_and_never_copied():
    got = check(perfect(), [("result.json", {"n_tokens": 32, "top1_agreement": 10 ** 4000})])
    assert mentions(got, "result.json top1_agreement", "an integer of about 4001 digits"), got
    assert max(len(r) for r in got) < token_check.MAX_REASON_CHARS + 200


def test_the_safe_reader_stands_in_front_of_the_gates_for_a_huge_integer(tmp_path):
    """Documents the layering: through a gate a 400-digit integer is refused by the reader, with the file named, before any recompute."""
    g, sd, run = stage2(tmp_path)
    rewrite(sd / "evidence" / "swap-check.json", put("report", HUGE))
    got = gate_weights_swap(sd, run)
    assert not got.ok and mentions(got.reasons, "swap-check.json", "not readable JSON"), got.reasons


# ---- the sampling trial's number check -------------------------------------------------------------------------------------------

def test_the_sampling_choice_number_helper_is_total_and_refuses_an_integer_no_float_can_hold():
    assert sampling_choice._number(3) is True and sampling_choice._number(0.5) is True and sampling_choice._number(10 ** 300) is True
    for bad in (HUGE, -HUGE, 10 ** 4000, True, False, float("inf"), float("-inf"), float("nan"), "1", None, [1]):
        assert sampling_choice._number(bad) is False


def test_the_speed_readers_of_the_sampling_choice_never_raise_on_a_huge_integer():
    """`speed` converts what `_number` accepts to a float: an accepted integer must fit one."""
    assert sampling_choice.speed({"sampled_tok_s": HUGE, "greedy_tok_s": 18.5}) == 18.5
    assert sampling_choice.speed({"sampled_tok_s": HUGE}) is None
    assert sampling_choice.speed_basis({"sampled_tok_s": HUGE, "greedy_tok_s": 18.5}) == "greedy"
    assert sampling_choice.speed_basis({"sampled_tok_s": -HUGE}) is None


# ---- the supervisor ---------------------------------------------------------------------------------------------------------------

def stub(tmp_path, gate):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    s = SimpleNamespace(ledger=ledger, run_dir=tmp_path, _gate=lambda spec: gate)
    s._note = functools.partial(Supervisor._note, s)
    s._run_gate = functools.partial(Supervisor._run_gate, s)
    s._note_unverified = functools.partial(Supervisor._note_unverified, s)      # _check_gate notes unverified numbers (issue 445)
    return s, ledger


def raising(exc):
    def gate(stage_dir, run_dir):
        raise exc
    return gate


SPEC = SimpleNamespace(number=2, tests=False, skill="decoder-check", gate_file="result.json")


def test_a_gate_that_raises_is_a_failed_gate_that_names_the_class_and_not_the_message(tmp_path):
    s, ledger = stub(tmp_path, raising(OverflowError("int too large to convert to float: " + "9" * 50)))
    try:
        got = Supervisor._run_gate(s, SPEC, tmp_path)
        entries = ledger.read()
    finally:
        ledger.close()
    assert isinstance(got, GateResult) and got.ok is False
    text = " ".join(got.reasons)
    assert "OverflowError" in text and "stage 2" in text and "9" * 20 not in text
    notes = [e for e in entries if e["event"] == "notice" and e["data"].get("what") == "a gate raised"]
    assert len(notes) == 1 and notes[0]["data"]["error_class"] == "OverflowError" and notes[0]["data"]["for_stage"] == 2
    assert "9" * 20 not in json.dumps(notes[0])


def test_a_gate_that_returns_a_result_is_passed_through_unchanged(tmp_path):
    result = GateResult(True, (), ("stages/2/result.json",))
    s, ledger = stub(tmp_path, lambda stage_dir, run_dir: result)
    try:
        assert Supervisor._run_gate(s, SPEC, tmp_path) is result
        assert ledger.read() == []
    finally:
        ledger.close()


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_in_a_gate_is_not_swallowed(tmp_path, interrupt):
    s, ledger = stub(tmp_path, raising(interrupt()))
    try:
        with pytest.raises(interrupt):
            Supervisor._run_gate(s, SPEC, tmp_path)
    finally:
        ledger.close()


def test_a_failure_in_choosing_the_gate_is_not_turned_into_a_gate_refusal(tmp_path):
    """The choice of gate can read the ledger (the rehearsal flag). A ledger that cannot be read is a supervisor failure and has to propagate."""
    s, ledger = stub(tmp_path, lambda *a: None)
    s._gate = lambda spec: (_ for _ in ()).throw(RuntimeError("the ledger cannot be read"))
    try:
        with pytest.raises(RuntimeError):
            Supervisor._run_gate(s, SPEC, tmp_path)
    finally:
        ledger.close()


def test_the_stage_check_turns_a_raising_gate_into_a_failed_result(tmp_path):
    s, ledger = stub(tmp_path, raising(OverflowError("x")))
    try:
        got = Supervisor._check_gate(s, SPEC, tmp_path)
    finally:
        ledger.close()
    assert isinstance(got, GateResult) and got.ok is False and mentions(got.reasons, "OverflowError")


def test_the_finished_check_of_an_agent_step_is_false_when_the_gate_raises(tmp_path):
    s, ledger = stub(tmp_path, raising(OverflowError("x")))
    try:
        finished = Supervisor._finished_check(s, SPEC, "finish", tmp_path)
        assert finished() is False
    finally:
        ledger.close()
