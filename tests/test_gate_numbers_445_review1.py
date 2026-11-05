# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Copilot round 1 on PR 469 (issue 445). Two findings, each pinned here.

1. The gate sets the probability-difference bar from `noise_floor` in result.json's copy of the sidecar parity object, and it reads
   `measured`, `head_sha256` and `code_sha256` from the same copy. The recompute folded the numbers and compared `wiring_check`, but not these
   four fields, so the finish step could raise the bar (or name another head file) while the hashed evidence said otherwise. Every field the gate
   reads from the copy must now be the evidence file's own, compared by type and value.
2. A run made on PR 444's base has the reference ids pinned in the ledger, but its boot check was written before `verify_bundle.py` recorded
   `teacher_forced`, and the supervisor never hashed its verify.json. `bound or ids_bound` refused it, although the compatibility rule says such a
   run passes with an agent-reported label. The records are required only when verify.json itself is bound; the reference file is checked
   against its own hash separately."""
import pytest

from orchard import sidecar_numbers, stages
from test_gate_numbers_445 import (boot_report, boot_reasons, mentions, parity_doc, rec, sidecar_stage)

# One record with a large probability difference: the evidence's own noise floor of 0.002 gives a bar of 0.25 (the larger of 0.25 and 4 x 0.002),
# so a difference of 0.5 fails. A floor of 0.2 would give a bar of 0.8 and let it pass.
BIG = [rec(0), rec(1, diff=0.5), rec(2, n=4, agree=4), rec(3, pcc=0.9975, diff=0.003), rec(4, n=2, agree=2), rec(5, n=2, agree=2)]


def summary(ev):
    return {k: v for k, v in ev.items() if k != "per_record"}


# ---- finding 1: every copied field is the evidence's ---------------------------------------------------------------------------------

def test_the_unedited_big_difference_record_fails_on_its_own_numbers(tmp_path):
    """Control: with the evidence's noise floor the 0.5 difference is above the bar, whatever else is true."""
    ev = parity_doc(BIG)
    g, _ = sidecar_stage(tmp_path, evidence=ev, bind=True)
    assert not g.ok and mentions(g.reasons, "prob_max_abs_diff", "above the bar")


@pytest.mark.parametrize("bind", [True, False])
def test_raising_the_noise_floor_in_result_json_only_is_refused(tmp_path, bind):
    """The reviewer's case: bound records report 0.5 and a floor of 0.002; result.json says the floor is 0.2. The gate's bar would be 0.8."""
    ev = parity_doc(BIG)
    sp = {**summary(ev), "noise_floor": {"prob_max_abs_diff": 0.2}}
    g, _ = sidecar_stage(tmp_path, evidence=ev, result_sp=sp, bind=bind)
    assert not g.ok
    assert mentions(g.reasons, "noise_floor", "sidecar-parity.json"), g.reasons


@pytest.mark.parametrize("field, bad", [
    ("noise_floor", {"prob_max_abs_diff": 0.2}),
    ("noise_floor", {"prob_max_abs_diff": 0.002, "extra": 1}),
    ("noise_floor", 0.2),
    ("measured", 1),                       # True == 1 in Python: a copy that is not the evidence's own boolean
    ("head_sha256", "0" * 64),
    ("code_sha256", "1" * 64),
    ("wiring_check", {"cosine": 0.9999, "passed": 1}),
])
def test_each_field_the_gate_reads_from_the_copy_must_be_the_evidences(tmp_path, field, bad):
    ev = parity_doc()
    g, _ = sidecar_stage(tmp_path, evidence=ev, result_sp={**summary(ev), field: bad}, bind=True)
    assert mentions(g.reasons, field, "sidecar-parity.json"), (field, g.reasons)


def test_a_noise_floor_in_the_copy_that_the_evidence_does_not_have_is_refused(tmp_path):
    ev = {k: v for k, v in parity_doc(BIG).items() if k != "noise_floor"}
    sp = {**summary(ev), "noise_floor": {"prob_max_abs_diff": 0.2}}
    g, _ = sidecar_stage(tmp_path, evidence=ev, result_sp=sp, bind=True)
    assert mentions(g.reasons, "noise_floor", "sidecar-parity.json")


def test_a_copy_without_the_field_is_not_a_new_refusal(tmp_path):
    """Leaving the floor out only makes the bar stricter (0.25), so it is not a way to weaken the gate and it is not refused here."""
    ev = parity_doc()
    sp = {k: v for k, v in summary(ev).items() if k != "noise_floor"}
    g, _ = sidecar_stage(tmp_path, evidence=ev, result_sp=sp, bind=True)
    assert g.ok, g.reasons


def test_an_unedited_copy_still_passes(tmp_path):
    for bind in (True, False):
        g, _ = sidecar_stage(tmp_path / str(bind), bind=bind)
        assert g.ok, g.reasons


@pytest.mark.parametrize("hostile", [10 ** 400, float("1e308"), [10 ** 400] * 50, {"prob_max_abs_diff": 10 ** 5000}, "x" * 100_000, None])
def test_a_hostile_copied_value_is_refused_without_raising(tmp_path, hostile):
    ev = parity_doc()
    sp = {**summary(ev), "noise_floor": hostile}
    reasons, verified = sidecar_numbers.problems([("sidecar-parity.json", ev), ("result.json sidecar_parity", sp)], ev, bound=True)
    assert verified and mentions(reasons, "noise_floor")


@pytest.mark.parametrize("field", ["noise_floor", "measured", "wiring_check", "head_sha256", "code_sha256"])
@pytest.mark.parametrize("hostile", [10 ** 400, [[[[1]]]] * 3, {"a": {"b": {"c": [None] * 2000}}}, "y" * 200_000])
@pytest.mark.parametrize("bind", [True, False])
def test_the_whole_gate_refuses_a_hostile_copied_field_without_raising(tmp_path, field, hostile, bind):
    """Through the gate itself (result.json read through the bounded reader), whatever the field holds, the gate returns a refusal."""
    ev = parity_doc()
    g, _ = sidecar_stage(tmp_path, evidence=ev, result_sp={**summary(ev), field: hostile}, bind=bind)
    assert not g.ok and g.reasons


def test_identical_compares_by_type_and_value_and_is_bounded():
    same = sidecar_numbers.identical
    assert same({"a": [1, 2.5, "x", None, True]}, {"a": [1, 2.5, "x", None, True]})
    assert not same(True, 1) and not same(1, 1.0) and not same(0, False) and not same("1", 1)
    assert not same({"a": 1}, {"a": 1, "b": 2}) and not same([1], [1, 1])
    deep_a = deep_b = []
    for _ in range(5000):
        deep_a, deep_b = [deep_a], [deep_b]
    assert not same(deep_a, deep_b)                      # deeper than the bound: not identical, and no RecursionError
    wide = list(range(200_000))
    assert not same(wide, list(wide))                    # more nodes than the budget: not identical, and no long walk


# ---- finding 2: a legacy boot check with pinned reference ids ------------------------------------------------------------------------

def without_records(report):
    return {k: v for k, v in report.items() if k != "teacher_forced"}


def test_a_legacy_boot_check_with_pinned_ids_and_no_raw_entry_passes_with_the_agent_reported_label(tmp_path):
    """The ledger holds the reference ids' hash (the run was made on PR 444's base), verify.json has no teacher_forced and was never hashed."""
    reasons, unverified = boot_reasons(tmp_path, without_records(boot_report()), ids=True, raw=False)
    assert reasons == []
    assert any("boot check" in u for u in unverified), unverified


def test_the_label_is_not_a_way_around_the_records_when_they_are_there(tmp_path):
    """Pinned ids, no raw entry, but the records are in the file: they are recomputed, so edited numbers are still refused."""
    r = boot_report()
    r["top1_agreement"], r["matches"], r["mismatches"] = 1.0, 32, []
    reasons, unverified = boot_reasons(tmp_path, r, ids=True, raw=False)
    assert mentions(reasons, "verify.json", "top1_agreement", "31 of 32")
    assert unverified == ()


def test_a_hashed_boot_check_without_records_is_still_refused(tmp_path):
    reasons, _ = boot_reasons(tmp_path, without_records(boot_report()), ids=True, raw=True)
    assert mentions(reasons, "teacher_forced")


def test_a_hashed_boot_check_with_ids_unpinned_and_no_records_is_still_refused(tmp_path):
    reasons, _ = boot_reasons(tmp_path, without_records(boot_report()), ids=False, raw=True)
    assert mentions(reasons, "teacher_forced")


def test_a_swapped_reference_file_is_refused_whether_or_not_verify_json_is_bound(tmp_path):
    """The reference file's own hash is checked independently of verify.json's binding."""
    from test_gate_numbers_445 import REF_PATH, REF
    from test_stages import write
    for raw in (False, True):
        run = tmp_path / str(raw) / "run"
        write(run, REF_PATH, {"generated_ids": REF})
        write(run, "stages/7/verify/evidence/verify.json", without_records(boot_report()))
        from test_gate_numbers_445 import boot_ledger
        boot_ledger(run, ids=True, raw=raw)
        write(run, REF_PATH, {"generated_ids": [i + 1 for i in REF]})
        reasons, _ = stages.boot_check_reasons(run / "stages" / "7", run, without_records(boot_report()))
        assert mentions(reasons, "reference", "sha256") or mentions(reasons, "generated-ids", "ledger"), (raw, reasons)
