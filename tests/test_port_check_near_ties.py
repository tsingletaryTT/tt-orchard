# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 2 check reports which of the port's top-1 disagreements are near-ties of the reference (a lesson from a skills-only port of the tiny
Qwen4: its three argmax mismatches were all positions where the reference's own top two logits were equal in bf16). The block is report-only. The
gate recomputes it from the record's ids and the stage 1 top-5 file and fails a block that disagrees with them; it never fails a run for the
number of disagreements."""
import json

import pytest

from orchard import full_port
from orchard.stages import gate_full_port
from port_check_fakes import HONEST, REF_IDS, WRONG, make_run, run_template, write_result
from test_port_check_fidelity import pc, record

# A port that is right everywhere except at step 5, where it picks the reference's second choice.
FLIP_AT_5 = HONEST.replace("return _peak(token_id + 1)", "return _peak(token_id + 1 if token_id != %d else 0)" % REF_IDS[4])


def write_top5(run, gaps: dict, second: int = 0):
    """A stage 1 top-5 file whose step k has the chosen token and, as the second choice, `second`, `gaps[k]` logits below it."""
    steps = []
    for k, t in enumerate(REF_IDS):
        top5 = [{"id": t, "logit": 12.0, "text": ""}, {"id": second, "logit": 12.0 - gaps.get(k, 5.0), "text": ""}]
        steps.append({"step": k, "chosen": t, "logprob": -1.0, "top5": top5})
    (run / "stages" / "1" / "evidence" / "reference" / "top5-logits.json").write_text(json.dumps({"steps": steps}))


def reasons_for(rec, run):
    out = []
    full_port._near_tie_reasons(rec, run, rec["reference_ids"], rec["chip_ids"], out)
    return out


def flipped(tmp_path, gap):
    run, stage, _ = make_run(tmp_path, FLIP_AT_5)
    write_top5(run, {5: gap})
    assert run_template(tmp_path, run, stage).returncode == 0
    return run, stage


def test_the_two_template_and_gate_constants_are_equal():
    assert pc.TIE_GAP == full_port.TIE_GAP


def test_a_flip_at_a_tie_is_counted_as_a_near_tie_and_the_agreement_without_it_is_one(tmp_path):
    run, stage = flipped(tmp_path, 0.0625)
    rec = record(stage)
    assert rec["top1_agreement"] < 1.0
    t = rec["near_ties"]
    assert (t["disagreements"], t["near_ties"], t["near_tie_positions"]) == (1, 1, [5])
    assert t["agreement_excluding_near_ties"] == 1.0 and t["tie_gap"] == pc.TIE_GAP


def test_a_flip_with_a_clear_gap_is_not_a_near_tie(tmp_path):
    run, stage = flipped(tmp_path, 3.0)
    t = record(stage)["near_ties"]
    assert (t["disagreements"], t["near_ties"], t["near_tie_positions"]) == (1, 0, [])
    assert t["agreement_excluding_near_ties"] == pytest.approx(1 - 1 / len(REF_IDS))


def test_a_flip_to_a_token_outside_the_top_five_is_not_a_near_tie(tmp_path):
    run, stage, _ = make_run(tmp_path, FLIP_AT_5)
    write_top5(run, {5: 0.0}, second=77)            # the port picks 0, which is not in the reference's top 5 at step 5
    assert run_template(tmp_path, run, stage).returncode == 0
    assert record(stage)["near_ties"]["near_ties"] == 0


def test_a_reference_without_a_top_five_gives_no_block(tmp_path):
    run, stage, _ = make_run(tmp_path, FLIP_AT_5, reference_logprob=None)
    assert run_template(tmp_path, run, stage).returncode == 0
    assert "near_ties" not in record(stage)


def test_the_gate_accepts_an_honest_block_and_never_fails_a_run_for_having_disagreements(tmp_path):
    run, stage = flipped(tmp_path, 3.0)
    assert reasons_for(record(stage), run) == []


@pytest.mark.parametrize("field,value", [("near_ties", 0), ("disagreements", 0), ("near_tie_positions", []), ("tie_gap", 9.0),
                                         ("agreement_excluding_near_ties", 0.5)])
def test_the_gate_refuses_a_block_that_disagrees_with_the_record_and_the_reference(tmp_path, field, value):
    run, stage = flipped(tmp_path, 0.0625)
    rec = record(stage)
    rec["near_ties"][field] = value
    assert reasons_for(rec, run)


def test_the_gate_refuses_a_block_when_the_reference_on_disk_says_it_was_not_a_tie(tmp_path):
    run, stage = flipped(tmp_path, 0.0625)
    write_top5(run, {5: 3.0})                       # the file changed after the check: the block now claims a tie the reference does not show
    assert any("does not match" in r for r in reasons_for(record(stage), run))


def test_the_gate_reaches_the_block_through_the_whole_stage_gate(tmp_path):
    run, stage = flipped(tmp_path, 0.0625)
    write_result(stage)
    rec = record(stage)
    rec["near_ties"]["near_ties"] = 0
    (stage / "evidence" / "port-check.json").write_text(json.dumps(rec))
    g = gate_full_port(stage, run)
    assert any("near_ties" in r for r in g.reasons)
