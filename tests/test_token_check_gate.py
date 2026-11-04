# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 2 and stage 4 swap gates recompute the 32-token check from the raw replies (issue 433).

The template writes `matches`, `top1_agreement`, `n_tokens` and the mismatches into swap-check.json and copies the numbers into
`result_draft`; the agent's result.json repeats them. The gate recomputes them from `teacher_forced[*].first_id` and stage 1's reference
ids, which the supervisor hashed into the ledger before the test. A swap-check.json and result.json edited together must not pass.
No device, no server and no model: the records are written by the tests."""
import copy
import hashlib
import json
import math

import pytest

from orchard import stages, token_check
from orchard.stages import gate_mesh_swap, gate_weights_swap, gate_weights_swap_sidecar
from test_stages import GOOD_DELTA, SWAP, mesh_stage, stage, swap_config, write
from test_stages_class import RESULT, SIDE_DELTA

REF = [1000 + 7 * i for i in range(32)]              # stage 1's generated ids
BAD = (3, 17)                                        # the two positions the served model disagrees on
REF_PATH = "stages/1/evidence/reference/generated-ids.json"


def served_report(ref=REF, bad=BAD):
    """The swap-check.json the template writes for a served model that disagrees with the reference at `bad`."""
    forced, mismatches = [], []
    for k, want in enumerate(ref):
        got = want + 1 if k in bad else want
        forced.append({"position": k, "text": f"t{k}", "first_id": got})
        if k in bad:
            mismatches.append({"position": k, "expected_id": want, "got_id": got, "expected_text": "e", "got_text": "g"})
    matches = len(ref) - len(bad)
    top1 = matches / len(ref)
    draft = {"serves": True, "top1_agreement": top1, "n_tokens": len(ref)}
    return {"label": "measured", "n_tokens": len(ref), "matches": matches, "top1_agreement": top1,
            "prompt_ids": [1, 2], "reference_generated_ids": list(ref), "teacher_forced": forced, "mismatches": mismatches,
            "free_run_text": "The sea was calm.", "reference_generated_text": "The sea was calm.", "result_draft": draft}


def result_of(report):
    return {**{k: v for k, v in SWAP.items() if k != "evidence"}, "top1_agreement": report["top1_agreement"],
            "n_tokens": report["n_tokens"], "evidence": ["stages/2/evidence/diff.txt"]}


def record_ids(run, path=None):
    """What Supervisor._draft_stage_files writes: the sha256 of stage 1's generated-ids.json, in the ledger, before the stage 2 test."""
    from orchard.ledger import Ledger
    path = path or run / REF_PATH
    led = Ledger(run / "ledger.jsonl")
    try:
        led.append("decision", 2, decision=stages.REFERENCE_IDS_LEDGER_DECISION, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    finally:
        led.close()


def stage2(tmp_path, report=None, result=None, ref=REF, record=True, gate=gate_weights_swap):
    report = served_report() if report is None else report
    sd, run = stage(tmp_path, 2, "result.json", result if result is not None else result_of(report))
    write(run, "stages/0/delta.json", GOOD_DELTA)
    write(run, "stages/2/evidence/swap-check.json", report)
    if ref is not None:
        write(run, REF_PATH, {"generated_ids": ref})
        if record:
            record_ids(run)
    return gate(sd, run), sd, run


def mentions(reasons, *words):
    return [r for r in reasons if all(w in r for w in words)]


# ---- the unedited record -------------------------------------------------------------------------------------------------------------

def test_the_recompute_gives_the_counts_the_template_wrote():
    got, why = token_check.recompute(served_report()["teacher_forced"], REF)
    assert why == [] and got == {"n_tokens": 32, "matches": 30, "top1_agreement": 30 / 32, "mismatch_positions": [3, 17]}


def test_an_unedited_stage_2_record_passes(tmp_path):
    g, *_ = stage2(tmp_path)
    assert g.ok, g.reasons


# ---- the edit the issue names -------------------------------------------------------------------------------------------------------

def hide_and_raise(report):
    """Hide the mismatches and raise the number everywhere it is written (the pair of edits issue 433 describes)."""
    r = copy.deepcopy(report)
    for k in BAD:
        r["teacher_forced"][k]["first_id"] = REF[k]          # the replies now agree
    return r


def test_raising_top1_everywhere_while_the_raw_replies_stay_the_same_is_refused(tmp_path):
    """The edit: top1_agreement 1.0 in swap-check.json, result_draft and result.json; matches 32; the mismatch list emptied. The raw replies
    still disagree at two positions, so the gate refuses."""
    r = served_report()
    r["top1_agreement"], r["matches"], r["mismatches"] = 1.0, 32, []
    r["result_draft"]["top1_agreement"] = 1.0
    res = result_of(r)
    g, *_ = stage2(tmp_path, report=r, result=res)
    assert not g.ok
    assert mentions(g.reasons, "swap-check.json", "top1_agreement", "30 of 32")
    assert mentions(g.reasons, "result.json", "top1_agreement")


def test_listing_positions_skipped_to_hide_the_mismatches_is_refused(tmp_path):
    r = served_report()
    r["positions_skipped"] = list(BAD)
    r["top1_agreement"], r["matches"] = 1.0, 32
    r["result_draft"]["top1_agreement"] = 1.0
    g, *_ = stage2(tmp_path, report=r, result=result_of(r))
    assert mentions(g.reasons, "positions_skipped", "rebuilt skip mask")


def test_a_result_that_lists_positions_skipped_is_refused_even_when_the_report_does_not(tmp_path):
    r = served_report()
    res = {**result_of(r), "positions_skipped": [3]}
    g, *_ = stage2(tmp_path, report=r, result=res)
    assert mentions(g.reasons, "result.json lists positions_skipped")


def test_result_json_edited_alone_is_refused_and_names_the_recomputed_value(tmp_path):
    r = served_report()
    res = {**result_of(r), "top1_agreement": 0.99}
    g, *_ = stage2(tmp_path, report=r, result=res)
    assert mentions(g.reasons, "result.json top1_agreement 0.99", "0.9375")


def test_the_result_draft_edited_alone_is_refused(tmp_path):
    r = served_report()
    r["result_draft"]["top1_agreement"] = 0.99
    g, *_ = stage2(tmp_path, report=r, result=result_of(served_report()))
    assert mentions(g.reasons, "result_draft top1_agreement")


def test_an_edited_matches_count_is_refused(tmp_path):
    r = served_report()
    r["matches"] = 31
    g, *_ = stage2(tmp_path, report=r)
    assert mentions(g.reasons, "matches 31", "30 of 32")


def test_a_mismatch_list_that_leaves_a_position_out_is_refused(tmp_path):
    r = served_report()
    r["mismatches"] = r["mismatches"][:1]
    g, *_ = stage2(tmp_path, report=r)
    assert mentions(g.reasons, "mismatches are not the positions", "17")


def test_a_mismatch_list_that_names_a_position_that_matches_is_refused(tmp_path):
    r = served_report()
    r["mismatches"].append({"position": 5, "expected_id": 1, "got_id": 2})
    g, *_ = stage2(tmp_path, report=r)
    assert mentions(g.reasons, "mismatches are not the positions", "5")


def test_a_changed_n_tokens_is_refused(tmp_path):
    r = served_report()
    r["n_tokens"] = 16
    g, *_ = stage2(tmp_path, report=r, result={**result_of(served_report()), "n_tokens": 16})
    assert mentions(g.reasons, "n_tokens 16", "32 records")


def test_a_reference_copy_that_is_not_stage_1s_is_refused(tmp_path):
    """Editing the copy of the reference in swap-check.json and the raw replies to match it does not help: the ids come from stage 1."""
    r = served_report()
    other = [i + 1 for i in REF]
    r["reference_generated_ids"] = other
    for k, f in enumerate(r["teacher_forced"]):
        f["first_id"] = other[k]
    r["mismatches"], r["matches"], r["top1_agreement"] = [], 32, 1.0
    r["result_draft"]["top1_agreement"] = 1.0
    g, *_ = stage2(tmp_path, report=r, result=result_of(r))
    assert not g.ok
    assert mentions(g.reasons, "top1_agreement", "swap-check.json")


def test_a_changed_copy_of_the_reference_ids_alone_is_refused(tmp_path):
    r = served_report()
    r["reference_generated_ids"] = [i + 1 for i in REF]
    g, *_ = stage2(tmp_path, report=r)
    assert mentions(g.reasons, "reference_generated_ids are not the first 32 ids of stage 1's reference")


def test_removing_the_raw_replies_does_not_dodge_the_recompute(tmp_path):
    r = served_report()
    del r["teacher_forced"]
    g, *_ = stage2(tmp_path, report=r)
    assert mentions(g.reasons, "no teacher_forced records")


@pytest.mark.parametrize("bad_forced, why", [
    ([], "non-empty list"),
    ("many", "non-empty list"),
    ([{"position": 1, "first_id": 5}] * 32, "in order"),
    ([{"position": k, "first_id": "x"} for k in range(32)], "in order"),
    ([{"position": k, "first_id": True} for k in range(32)], "in order"),
])
def test_unreadable_raw_records_are_refused_with_a_reason(tmp_path, bad_forced, why):
    r = served_report()
    r["teacher_forced"] = bad_forced
    g, *_ = stage2(tmp_path, report=r)
    assert mentions(g.reasons, why), g.reasons


def test_a_missing_first_id_counts_as_a_mismatch():
    forced = served_report(bad=())["teacher_forced"]
    forced[0]["first_id"] = None
    got, why = token_check.recompute(forced, REF)
    assert why == [] and got["mismatch_positions"] == [0] and got["matches"] == 31


def test_a_reference_shorter_than_the_record_is_refused(tmp_path):
    g, *_ = stage2(tmp_path, ref=REF[:20])
    assert mentions(g.reasons, "stage 1's reference holds 20 ids")


def test_numbers_that_are_booleans_or_not_finite_never_match():
    r = served_report()
    for bogus in (True, float("nan"), float("inf"), "0.9375", None):
        r2 = {**r, "top1_agreement": bogus}
        assert any("top1_agreement" in x for x in token_check.problems(r2, [], REF)), bogus
    assert token_check.problems({**r, "matches": True}, [], REF)
    assert token_check.problems({**r, "n_tokens": True}, [], REF)


def test_the_recorded_fraction_may_differ_by_rounding_only():
    r = served_report()
    assert token_check.problems({**r, "top1_agreement": r["top1_agreement"] + 1e-12}, [], REF) == []
    assert token_check.problems({**r, "top1_agreement": r["top1_agreement"] + 1e-6}, [], REF)


def test_a_reason_names_at_most_eight_positions():
    r = served_report(bad=tuple(range(12)))
    r["mismatches"] = []
    text = " ".join(token_check.problems(r, [], REF))
    assert "and 4 more" in text


# ---- the reference ids come from the ledger-bound file -------------------------------------------------------------------------

def test_stage_1_ids_replaced_after_the_supervisor_recorded_them_are_refused(tmp_path):
    """The agent edits stage 1's generated-ids.json to the edited replies. The ledger holds the hash of the original."""
    r = served_report()
    g, sd, run = stage2(tmp_path, report=r)
    assert g.ok
    forged = list(REF)
    for k in BAD:
        forged[k] = r["teacher_forced"][k]["first_id"]
    write(run, REF_PATH, {"generated_ids": forged})
    g2 = gate_weights_swap(sd, run)
    assert not g2.ok and mentions(g2.reasons, "not the file the supervisor recorded")


def test_stage_1_ids_deleted_after_the_supervisor_recorded_them_are_refused(tmp_path):
    g, sd, run = stage2(tmp_path)
    (run / REF_PATH).unlink()
    g2 = gate_weights_swap(sd, run)
    assert mentions(g2.reasons, "missing, though the supervisor recorded it")


def test_a_run_without_a_ledger_record_is_still_recomputed(tmp_path):
    """A run from before this check has no ledger record. The recompute still runs against the file; only the hash binding is absent."""
    r = served_report()
    r["top1_agreement"], r["matches"], r["mismatches"] = 1.0, 32, []
    g, *_ = stage2(tmp_path, report=r, result=result_of(r), record=False)
    assert mentions(g.reasons, "top1_agreement", "30 of 32")
    ok, *_ = stage2(tmp_path / "again", record=False)
    assert ok.ok


def test_a_report_without_raw_replies_is_judged_as_before_when_nothing_binds_the_reference(tmp_path):
    """No ledger record (a run from before this check): the absence of teacher_forced is not invented into a refusal. With the record it is
    one (test_removing_the_raw_replies_does_not_dodge_the_recompute)."""
    r = served_report()
    del r["teacher_forced"]
    g, *_ = stage2(tmp_path, report=r, record=False)
    assert g.ok, g.reasons


def test_a_stage_2_with_no_stage_1_reference_and_no_ledger_record_is_judged_as_before(tmp_path):
    r = {"result_draft": {"top1_agreement": SWAP["top1_agreement"]}, "free_run_text": "The sea was calm.",
         "reference_generated_text": "The sea was calm."}
    g, *_ = stage2(tmp_path, report=r, result=dict(SWAP), ref=None)
    assert g.ok, g.reasons


@pytest.mark.parametrize("raw, why", [
    (b"[1, 2", "cannot be read"),
    (b'{"generated_ids": "x"}', "no generated_ids list"),
    (b"[1, 2, 3]", "no generated_ids list"),
])
def test_a_reference_file_that_cannot_be_used_is_refused(tmp_path, raw, why):
    g, sd, run = stage2(tmp_path)
    (run / REF_PATH).write_bytes(raw)
    g2 = gate_weights_swap(sd, run)
    assert mentions(g2.reasons, why), g2.reasons


# ---- the same recompute on the other gates ------------------------------------------------------------------------------------------

def test_the_sidecar_gate_recomputes_too(tmp_path):
    r = served_report()
    r["top1_agreement"], r["matches"], r["mismatches"] = 1.0, 32, []
    sd, run = stage(tmp_path, 2, "result.json", {**RESULT, **result_of(r), "top1_agreement": 1.0})
    write(run, "stages/0/delta.json", SIDE_DELTA)
    write(run, "stages/2/evidence/swap-check.json", r)
    write(run, REF_PATH, {"generated_ids": REF})
    record_ids(run)
    g = gate_weights_swap_sidecar(sd, run)
    assert mentions(g.reasons, "swap-check.json", "top1_agreement", "30 of 32")


def mesh(tmp_path, report, n=2, ref=REF):
    cfg = swap_config(n, top1_agreement=report["top1_agreement"], n_tokens=report["n_tokens"])
    sd, run = mesh_stage(tmp_path, [cfg])
    write(run, "stages/0/delta.json", GOOD_DELTA)
    write(run, f"stages/4/configs/{n}/evidence/swap-check.json", report)
    write(run, REF_PATH, {"generated_ids": ref})
    record_ids(run)
    return gate_mesh_swap(sd, run), sd, run


def test_an_unedited_stage_4_configuration_passes(tmp_path):
    g, *_ = mesh(tmp_path, served_report())
    assert g.ok, g.reasons


def test_a_stage_4_configuration_with_raised_numbers_is_refused_and_names_the_configuration(tmp_path):
    r = served_report()
    r["top1_agreement"], r["matches"], r["mismatches"] = 1.0, 32, []
    r["result_draft"]["top1_agreement"] = 1.0
    g, *_ = mesh(tmp_path, r)
    assert mentions(g.reasons, "the 2-chip configuration", "swap-check.json", "top1_agreement")
    assert mentions(g.reasons, "the 2-chip configuration", "the configuration's result", "top1_agreement")


def test_the_sampling_trials_correctness_check_recomputes_the_variants_numbers(tmp_path):
    from orchard import sampling_choice
    r = served_report()
    r["top1_agreement"], r["matches"], r["mismatches"] = 1.0, 32, []
    r["result_draft"] = {"serves": True, "coherent": True, "server_ready_s": 10.0, "top1_agreement": 1.0, "n_tokens": 32}
    run = tmp_path / "run"
    write(run, REF_PATH, {"generated_ids": REF})
    record_ids(run)
    out = sampling_choice._correctness_failures(r, run)
    assert mentions(out, "swap-check.json", "top1_agreement", "30 of 32")
    served = served_report()
    served["result_draft"] = {"serves": True, "coherent": True, "server_ready_s": 10.0, "top1_agreement": served["top1_agreement"], "n_tokens": 32}
    assert not mentions(sampling_choice._correctness_failures(served, run), "teacher_forced")


# ---- the supervisor records the hash ----------------------------------------------------------------------------------------------

def test_reference_ids_record_hashes_the_file_and_is_none_when_it_is_absent(tmp_path):
    run = tmp_path / "run"
    assert stages.reference_ids_record(run) is None
    p = write(run, REF_PATH, {"generated_ids": REF})
    got = stages.reference_ids_record(run)
    assert got["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest() and got["path"].endswith("generated-ids.json")


# ---- review of PR 444, round 2 ------------------------------------------------------------------------------------------------------
# A run whose ledger holds the reference ids' hash was governed by the supervisor from the start. Its swap-check.json cannot go missing,
# and cannot be replaced by a JSON value that is not an object, without a refusal: the recompute is what the ledger binding is for.

def sidecar_stage(tmp_path):
    r = served_report()
    sd, run = stage(tmp_path, 2, "result.json", {**RESULT, **result_of(r)})
    write(run, "stages/0/delta.json", SIDE_DELTA)
    write(run, "stages/2/evidence/swap-check.json", r)
    write(run, REF_PATH, {"generated_ids": REF})
    record_ids(run)
    return sd, run


@pytest.mark.parametrize("gate_name", ["stage2", "sidecar"])
def test_a_bound_run_refuses_a_swap_check_that_is_missing(tmp_path, gate_name):
    if gate_name == "stage2":
        g, sd, run = stage2(tmp_path)
        assert g.ok, g.reasons
        gate = gate_weights_swap
    else:
        sd, run = sidecar_stage(tmp_path)
        gate = gate_weights_swap_sidecar
    (sd / "evidence" / "swap-check.json").unlink()
    got = gate(sd, run)
    assert not got.ok
    assert mentions(got.reasons, "swap-check.json is missing", "ledger"), got.reasons


@pytest.mark.parametrize("gate_name", ["stage2", "sidecar"])
@pytest.mark.parametrize("raw", ["null", "[]", "3", '"x"'])
def test_a_bound_run_refuses_a_swap_check_that_is_not_an_object(tmp_path, gate_name, raw):
    if gate_name == "stage2":
        g, sd, run = stage2(tmp_path)
        gate = gate_weights_swap
    else:
        sd, run = sidecar_stage(tmp_path)
        gate = gate_weights_swap_sidecar
    (sd / "evidence" / "swap-check.json").write_text(raw)
    got = gate(sd, run)
    assert not got.ok
    assert mentions(got.reasons, "swap-check.json is not an object", "recomputed"), got.reasons


@pytest.mark.parametrize("raw", ["{not json", '{"a": NaN}', "x" * 10])
def test_a_bound_run_refuses_when_the_recompute_cannot_read_the_swap_check(tmp_path, raw):
    """Review of PR 444, round 4: the suite check and the recompute read swap-check.json separately. A file that read fine for the suite check and
    is unreadable for the recompute (an edit between the two reads) must not end the recompute with no reasons on a run the ledger governs."""
    g, sd, run = stage2(tmp_path)
    path = sd / "evidence" / "swap-check.json"
    path.write_text(raw)
    got = stages._token_check_reasons_in(path, [], run)
    assert mentions(got, "swap-check.json", "could not be read", "ledger"), got


def test_an_unbound_run_still_judges_an_unreadable_swap_check_as_before(tmp_path):
    g, sd, run = stage2(tmp_path, record=False)
    path = sd / "evidence" / "swap-check.json"
    path.write_text("{not json")
    assert stages._token_check_reasons_in(path, [], run) == []


def test_an_unbound_run_keeps_the_old_judgement_for_a_missing_swap_check(tmp_path):
    """A run from before the check has no ledger record: a missing swap-check.json is judged as before (by the other checks), not by this one."""
    g, sd, run = stage2(tmp_path, record=False)
    (sd / "evidence" / "swap-check.json").unlink()
    assert not mentions(gate_weights_swap(sd, run).reasons, "recomputed")


@pytest.mark.parametrize("position", [[], {}, None, "7", 7.5, True, [3], {"a": 1}])
def test_a_mismatch_list_with_any_json_value_as_a_position_is_refused_not_raised(position):
    """The positions are JSON: a list or an object is unhashable and a mix of null, text and numbers cannot be sorted. Neither may raise."""
    r = served_report()
    r["mismatches"] = [*r["mismatches"], {"position": position}]
    out = token_check.problems(r, [], REF)
    assert mentions(out, "mismatches are not the positions"), out


def test_a_mismatch_list_with_mixed_junk_positions_is_refused_and_names_each_once():
    r = served_report()
    r["mismatches"] = [{"position": None}, {"position": "x"}, {"position": []}, {"position": None}, 5, {"position": 99}]
    out = token_check.problems(r, [], REF)
    text = " ".join(mentions(out, "mismatches are not the positions"))
    # a list is named by its size, never copied (review of PR 444, limit round: a reason is bounded whatever the report holds)
    assert "None" in text and "'x'" in text and "a list of 0 items" in text and "99" in text, out
    assert text.count("None") == 1, text                                     # the repeat of a value is named once


def test_the_real_mismatch_list_still_agrees():
    assert token_check.problems(served_report(), [], REF) == []


@pytest.mark.parametrize("raw", ["null", "[]", "3"])
def test_stage_4_refuses_a_swap_check_that_is_not_an_object_on_a_bound_run(tmp_path, raw):
    g, sd, run = mesh(tmp_path, served_report())
    assert g.ok, g.reasons
    (sd / "configs" / "2" / "evidence" / "swap-check.json").write_text(raw)
    got = gate_mesh_swap(sd, run)
    assert not got.ok and mentions(got.reasons, "the 2-chip configuration", "swap-check.json"), got.reasons


@pytest.mark.parametrize("bad", [None, [], 3, "x"])
def test_the_sampling_correctness_check_fails_a_variant_report_that_is_not_an_object(tmp_path, bad):
    from orchard import sampling_choice
    run = tmp_path / "run"
    write(run, REF_PATH, {"generated_ids": REF})
    record_ids(run)
    assert sampling_choice._correctness_failures(bad, run)


@pytest.mark.parametrize("raw", ["null", "[]", "3"])
def test_an_unbound_run_keeps_the_old_judgement_for_a_swap_check_that_is_not_an_object(tmp_path, raw):
    """No ledger record (a run from before the check): the recompute is not demanded, and this check adds no reason of its own."""
    g, sd, run = stage2(tmp_path, record=False)
    (sd / "evidence" / "swap-check.json").write_text(raw)
    assert not mentions(gate_weights_swap(sd, run).reasons, "recomputed")
