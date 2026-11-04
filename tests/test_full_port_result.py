# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 456: stage 2 of a full port, `result.json` derived by the supervisor from the check's record."""
import json
import random

import pytest

from orchard import full_port, full_port_result as fpr
from test_full_port import REF_IDS, TEXT, build, put

PROSE = ("The sea was calm that morning and the boat left the harbour early, carrying nets, bread and a small brass lamp "
         "that the old fisherman had kept since his youth.")


def derived(tmp_path, **kw):
    """Build a passing stage 2 directory, take its result.json away, and derive one."""
    stage, run = build(tmp_path, **{k: v for k, v in kw.items() if k != "rehearsal"})
    (stage / "result.json").unlink()
    return fpr.derive(stage, run, rehearsal=kw.get("rehearsal", False)), stage, run


def gate_reasons(stage, run, result, *, rehearsal=False):
    (stage / "result.json").write_text(json.dumps(result))
    return full_port.check(stage, run, rehearsal=rehearsal)[0]


def test_a_derived_result_passes_the_gate_and_says_the_supervisor_wrote_it(tmp_path):
    (result, why), stage, run = derived(tmp_path)
    assert why == "derived" and result["derived_by"] == "supervisor" and "form only" in result["coherent_basis"]
    assert gate_reasons(stage, run, result) == []


def test_the_numbers_are_the_records_and_agreement_is_recomputed_from_its_ids(tmp_path):
    (result, _), stage, run = derived(tmp_path, record={"top1_agreement": 1.0})       # the record's own field lies
    assert result["top1_agreement"] == pytest.approx(30 / 32) and result["n_tokens"] == 32
    assert result["load_s"] == 120.5 and result["free_run_text"] == TEXT and result["runs"] is True and result["coherent"] is True
    assert result["evidence"] == ["stages/2/evidence/port-check.json", "stages/2/evidence/hw-test-output.txt"]


def test_agreement_below_the_bar_is_still_derived_and_the_gate_refuses_it(tmp_path):
    chip = [0] * 32
    (result, _), stage, run = derived(tmp_path, record={"chip_ids": chip})
    assert result["top1_agreement"] == 0.0
    assert any("below the minimum" in r for r in gate_reasons(stage, run, result))


@pytest.mark.parametrize("marker,fragment", [({"returncode": 1}, "did not pass"), ({"timed_out": True}, "did not pass"),
                                              ({"returncode": None}, "did not pass"), ({"returncode": True}, "did not pass"),
                                              ({"timed_out": None}, "did not pass")])
def test_a_failed_or_odd_hardware_test_is_left_to_the_agent(tmp_path, marker, fragment):
    (result, why), _, _ = derived(tmp_path, marker=marker)
    assert result is None and fragment in why


def test_a_missing_marker_or_record_is_left_to_the_agent(tmp_path):
    stage, run = build(tmp_path)
    (stage / "test-result.json").unlink()
    assert fpr.derive(stage, run)[0] is None and "test-result.json" in fpr.derive(stage, run)[1]
    stage, run = build(tmp_path / "b")
    (stage / "evidence" / "port-check.json").unlink()
    assert fpr.derive(stage, run)[0] is None and "port-check.json" in fpr.derive(stage, run)[1]


@pytest.mark.parametrize("record", [{"chip_ids": [1, 2]}, {"chip_ids": "x"}, {"reference_ids": []}, {"load_s": -1}, {"load_s": True},
                                    {"load_s": "fast"}, {"free_run_text": ""}, {"free_run_text": None}, {"chip_ids": [True] * 32}])
def test_a_malformed_record_is_left_to_the_agent_and_never_a_crash(tmp_path, record):
    (result, why), _, _ = derived(tmp_path, record=record)
    assert result is None and why


def test_a_free_run_text_below_the_floor_is_left_to_the_agent(tmp_path):
    (result, why), _, _ = derived(tmp_path, record={"free_run_text": "the " * 40})
    assert result is None and "readable floor" in why


def test_a_rehearsal_with_unreadable_text_is_derived_with_coherent_false_and_the_gate_waives_it(tmp_path):
    junk = " ".join("".join(random.Random(i).choice("#%&*@") for _ in range(5)) for i in range(40))
    (result, why), stage, run = derived(tmp_path, record={"free_run_text": junk}, rehearsal=True)
    assert why == "derived" and result["coherent"] is False and "rehearsal model" in result["coherent_basis"]
    assert gate_reasons(stage, run, result, rehearsal=True) == []
    assert any("coherent" in r for r in full_port.check(stage, run, rehearsal=False)[0])


def test_a_hostile_files_are_a_none_and_not_a_crash(tmp_path):
    stage, run = build(tmp_path)
    (stage / "evidence" / "port-check.json").write_text("[" * 100000)
    assert fpr.derive(stage, run)[0] is None


# ---- the readable floor ----

def test_ordinary_prose_passes_the_floor_and_the_basis_states_what_was_measured():
    ok, basis = fpr.readable(PROSE)
    assert ok and "words" in basis and "printable" in basis and "compresses to" in basis


@pytest.mark.parametrize("text", ["", "word " * 5, "the " * 60, "a b c d e f g h i j k l m n o p", "#" * 400,
                                  " ".join(f"{i}{i}{i}" for i in range(50)), "\x00\x01\x02 " * 40, None, 12, PROSE * 30])
def test_text_below_the_floor_is_refused_with_a_reason(text):
    ok, basis = fpr.readable(text)
    assert not ok and basis


def test_a_long_repeated_paragraph_fails_on_repetition():
    ok, basis = fpr.readable((PROSE + " ") * 40)
    assert not ok and ("repeats itself" in basis or "differ" in basis or "commonest" in basis or "one word holds" in basis)


# Each floor on its own: the text is built so that every other measurement passes (checked first), and only the floor under test fails.

def only_fails(text, fragment):
    ok, basis = fpr.readable(text)
    assert not ok and fragment in basis, basis


def test_one_word_holding_a_third_of_the_text_fails_only_the_commonest_word_floor():
    text = " ".join([f"alpha{i}" for i in range(70)] + ["the"] * 40)         # 65% different, compresses to 0.22: nothing else fails
    only_fails(text, "one word holds more than")


def test_a_few_words_repeated_in_a_shuffled_order_fail_only_the_distinct_word_floor():
    rnd = random.Random(9)
    words = [f"term{i}" for i in range(20)] * 5
    rnd.shuffle(words)
    text = " ".join(words)                                                     # the commonest word holds 5%, compresses to 0.24
    only_fails(text, "fewer than 35% of the words differ")


def test_control_characters_in_the_text_fail_only_the_printable_floor():
    text = " ".join((f"alpha{i}\x01" if i % 2 else f"alpha{i}") for i in range(80))
    only_fails(text, "printable floor")


def test_a_long_regular_text_with_all_different_words_fails_only_the_repetition_floor():
    text = " ".join(f"aaaaaaaaa{i}" for i in range(150))
    ok, basis = fpr.readable(text)
    assert not ok and "it repeats itself" in basis
