# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 433, review of PR 444, round 3: a ledger-bound run must carry every position the template records.

The weights-swap template teacher-forces `N_TOKENS` = 32 positions. The recompute trusted the number of records it was given, so with the
ledger binding in place an edit could delete a failing suffix of `teacher_forced`, set `n_tokens`, `matches` and `top1_agreement` to the
shorter record's values, cut `reference_generated_ids` the same way and pass: 24 matches followed by eight mismatches became 1.0. A bound run
(the supervisor hashed stage 1's reference ids into the ledger before the test) now needs all 32 positions; a run from before the check, which
nothing binds, is judged as before."""
import ast
import copy
from pathlib import Path

from orchard import token_check
from orchard.stages import gate_mesh_swap, gate_weights_swap, gate_weights_swap_sidecar
from test_stages import GOOD_DELTA, mesh_stage, stage, swap_config, write
from test_stages_class import RESULT, SIDE_DELTA
from test_token_check_gate import REF, REF_PATH, mentions, record_ids, result_of, served_report, stage2

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "orchard" / "skills" / "weights-swap-templates" / "serve_and_compare.py"


def truncated(keep=24):
    """The edit from the review: the model disagrees at every position from `keep` on, those records are deleted, and every number that
    repeats them is rewritten to match the shorter record."""
    r = served_report(bad=tuple(range(keep, len(REF))))
    r["teacher_forced"] = r["teacher_forced"][:keep]
    r["mismatches"] = []
    r["reference_generated_ids"] = list(REF[:keep])
    r["n_tokens"], r["matches"], r["top1_agreement"] = keep, keep, 1.0
    r["result_draft"] = {"serves": True, "top1_agreement": 1.0, "n_tokens": keep}
    return r


def test_the_required_count_is_the_one_the_template_records():
    tree = ast.parse(TEMPLATE.read_text())
    values = [n.value.value for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "N_TOKENS" for t in n.targets)
              and isinstance(n.value, ast.Constant)]
    assert values == [token_check.TEMPLATE_POSITIONS] == [32]


def test_a_bound_report_with_its_failing_suffix_deleted_and_every_number_rewritten_is_refused(tmp_path):
    r = truncated()
    g, *_ = stage2(tmp_path, report=r, result=result_of(r))
    assert not g.ok
    assert mentions(g.reasons, "teacher_forced", "24", "32")


def test_the_same_edit_on_a_run_nothing_binds_is_judged_as_before(tmp_path):
    r = truncated()
    g, *_ = stage2(tmp_path, report=r, result=result_of(r), record=False)
    assert g.ok, g.reasons


def test_a_bound_report_with_more_positions_than_the_template_records_is_refused(tmp_path):
    ref = list(REF) + [5000 + i for i in range(8)]
    r = served_report(ref=ref, bad=())
    g, *_ = stage2(tmp_path, report=r, result=result_of(r), ref=ref)
    assert mentions(g.reasons, "teacher_forced", "40", "32")


def test_an_unedited_bound_report_with_all_positions_still_passes(tmp_path):
    g, *_ = stage2(tmp_path)
    assert g.ok, g.reasons


def test_the_sidecar_gate_refuses_a_truncated_bound_report(tmp_path):
    r = truncated()
    sd, run = stage(tmp_path, 2, "result.json", {**RESULT, **result_of(r)})
    write(run, "stages/0/delta.json", SIDE_DELTA)
    write(run, "stages/2/evidence/swap-check.json", r)
    write(run, REF_PATH, {"generated_ids": REF})
    record_ids(run)
    assert mentions(gate_weights_swap_sidecar(sd, run).reasons, "teacher_forced", "24", "32")


def test_the_stage_4_gate_refuses_a_truncated_bound_configuration_and_names_it(tmp_path):
    r = truncated()
    cfg = swap_config(2, top1_agreement=1.0, n_tokens=24)
    sd, run = mesh_stage(tmp_path, [cfg])
    write(run, "stages/0/delta.json", GOOD_DELTA)
    write(run, "stages/4/configs/2/evidence/swap-check.json", r)
    write(run, REF_PATH, {"generated_ids": REF})
    record_ids(run)
    assert mentions(gate_mesh_swap(sd, run).reasons, "the 2-chip configuration", "teacher_forced", "24", "32")


def test_the_sampling_trials_correctness_check_refuses_a_truncated_bound_report(tmp_path):
    from orchard import sampling_choice
    r = truncated()
    run = tmp_path / "run"
    write(run, REF_PATH, {"generated_ids": REF})
    record_ids(run)
    assert mentions(sampling_choice._correctness_failures(r, run), "teacher_forced", "24", "32")


def test_problems_takes_the_required_count_as_an_argument_and_only_applies_it_when_given():
    r = truncated()
    with_count = token_check.problems(r, [], REF, "", require_records=True, required_positions=32)
    assert mentions(with_count, "teacher_forced", "24", "32")
    assert token_check.problems(r, [], REF, "", require_records=True) == []
    assert token_check.problems(copy.deepcopy(served_report()), [], REF, "", require_records=True, required_positions=32) == []
