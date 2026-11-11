# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 2 check reports a fidelity score next to the top-1 agreement (issue 731, from Compression-Bench): 10 * PPL_reference / PPL_port on the
reference's own tokens. It is report-only: the gate recomputes it from the record and compares the reference numbers with the stage 1 file,
and never fails a run for a low value."""
import importlib.util
import json
import math

import pytest

from orchard import full_port
from orchard.stages import gate_full_port
from port_check_fakes import HONEST, HONEST_LP, REF_IDS, WRONG, make_run, run_template, write_result

def _load_template():
    """The template imported from a copy under a `stages/` directory, where it finds its run directory the way it does under the supervisor."""
    import shutil
    import tempfile
    from pathlib import Path
    stage = Path(tempfile.mkdtemp(prefix="pc-fid-")) / "run" / "stages" / "2"
    stage.mkdir(parents=True)
    shutil.copy2(full_port.TEMPLATE, stage / full_port.TEMPLATE_NAME)
    spec = importlib.util.spec_from_file_location("port_check_under_test", stage / full_port.TEMPLATE_NAME)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pc = _load_template()


def record(stage):
    return json.loads((stage / "evidence" / "port-check.json").read_text())


def fidelity_reasons(rec, run, n=len(REF_IDS)):
    out = []
    full_port._fidelity_reasons(rec, run, n, out)
    return out


@pytest.fixture
def honest(tmp_path):
    run, stage, _ = make_run(tmp_path)
    assert run_template(tmp_path, run, stage).returncode == 0
    return run, stage, tmp_path


# ---- the record ------------------------------------------------------------------------------------------------------

def test_an_honest_port_scores_ten_and_the_record_says_it_is_reported_not_gated(honest):
    run, stage, _ = honest
    f = record(stage)["fidelity"]
    assert f["kind"] == "reference-greedy-text" and f["tokens"] == len(REF_IDS)
    assert f["value"] == pytest.approx(10.0, abs=1e-6)
    assert f["chip_logprobs"] == pytest.approx([HONEST_LP] * len(REF_IDS))
    assert "reported, not gated" in f["note"]                              # 32 tokens: below the noise floor, and the record says so
    write_result(stage)
    g = gate_full_port(stage, run)
    assert g.ok, g.reasons


def test_a_port_whose_probabilities_for_the_reference_tokens_are_lower_scores_below_ten_and_the_gate_does_not_add_a_fidelity_reason(tmp_path):
    run, stage, _ = make_run(tmp_path, WRONG)                               # its peak is token 0, so it gives each reference token a lower probability
    assert run_template(tmp_path, run, stage).returncode == 0
    lp = 0.0 - math.log(math.e + 199.0)
    assert record(stage)["fidelity"]["value"] == pytest.approx(10 * math.exp(lp - HONEST_LP))
    write_result(stage)
    assert not [r for r in gate_full_port(stage, run).reasons if "fidelity" in r]


def test_a_reference_made_before_the_log_probabilities_existed_gives_no_fidelity_block_and_the_gate_still_passes(tmp_path):
    run, stage, _ = make_run(tmp_path, reference_logprob=None)
    assert run_template(tmp_path, run, stage).returncode == 0
    assert "fidelity" not in record(stage)
    write_result(stage)
    g = gate_full_port(stage, run)
    assert g.ok, g.reasons


def test_the_check_line_names_the_score_and_says_it_is_not_gated(honest):
    run, stage, tmp_path = honest
    done = run_template(tmp_path / "again", *make_run(tmp_path / "again")[:2])
    assert "fidelity 10.000" in done.stdout and "reported not gated" in done.stdout


# ---- the gate's recomputation -----------------------------------------------------------------------------------------

def test_the_gate_accepts_the_record_the_template_wrote(honest):
    run, stage, _ = honest
    assert fidelity_reasons(record(stage), run) == []


def test_the_gate_refuses_a_value_that_is_not_what_its_own_log_probabilities_give(honest):
    run, stage, _ = honest
    rec = record(stage)
    rec["fidelity"]["value"] = 9.9
    assert any("recomputed" in r for r in fidelity_reasons(rec, run))


def test_the_gate_refuses_reference_log_probabilities_that_are_not_the_stage_one_files(honest):
    run, stage, _ = honest
    rec = record(stage)
    rec["fidelity"]["reference_logprobs"] = [-0.5] * len(REF_IDS)           # a flattering reference: the value is recomputed to match
    rec["fidelity"]["value"] = full_port.fidelity_value(rec["fidelity"]["reference_logprobs"], rec["fidelity"]["chip_logprobs"])
    assert any("numbers the reference did not produce" in r for r in fidelity_reasons(rec, run))


@pytest.mark.parametrize("field,bad", [("chip_logprobs", [-1.0] * 3), ("chip_logprobs", [float("nan")] * len(REF_IDS)),
                                       ("chip_logprobs", [0.5] * len(REF_IDS)), ("reference_logprobs", "x")])
def test_the_gate_refuses_log_probabilities_of_the_wrong_length_or_that_are_not_finite_numbers_at_most_zero(honest, field, bad):
    run, stage, _ = honest
    rec = record(stage)
    rec["fidelity"][field] = bad
    assert any("finite log-probabilities" in r for r in fidelity_reasons(rec, run))


def test_the_gate_refuses_a_token_count_that_differs_from_the_positions_compared(honest):
    run, stage, _ = honest
    rec = record(stage)
    rec["fidelity"]["tokens"] = 5
    assert any("tokens 5" in r for r in fidelity_reasons(rec, run))


def test_the_gate_refuses_a_fidelity_block_that_is_not_an_object(honest):
    run, stage, _ = honest
    rec = record(stage)
    rec["fidelity"] = [1]
    assert fidelity_reasons(rec, run) == ["fidelity must be an object"]


# ---- the worker's arithmetic and the two copies of the formula -----------------------------------------------------------

def test_logprob_is_the_softmax_log_probability_for_a_list_and_for_an_array_like():
    v = [0.0, 1.0, 2.0, -1.0]
    want = 2.0 - math.log(sum(math.exp(x) for x in v))
    assert pc.logprob(v, 2) == pytest.approx(want)

    class Arr:
        def __init__(self, data): self.data = data
        def reshape(self, *a): return self
        def tolist(self): return list(self.data)
    assert pc.logprob(Arr(v), 2) == pytest.approx(want)


def test_logprob_does_not_overflow_on_large_logits_and_refuses_a_target_outside_the_vocabulary():
    assert pc.logprob([1000.0, 999.0], 0) == pytest.approx(-math.log(1 + math.exp(-1)))
    with pytest.raises(ValueError, match="outside the 2 logits"):
        pc.logprob([0.0, 1.0], 2)


def test_the_templates_formula_and_the_gates_formula_agree():
    ref, chip = [-1.0, -2.0, -0.5, -3.0], [-1.2, -2.5, -0.4, -3.3]
    assert pc.fidelity_block(ref, chip)["value"] == pytest.approx(full_port.fidelity_value(ref, chip))
    assert full_port.fidelity_value(ref, list(ref)) == pytest.approx(10.0)


def test_the_template_leaves_out_a_reference_log_probability_that_is_not_a_finite_number_at_most_zero(tmp_path):
    run, stage, _ = make_run(tmp_path, reference_logprob=0.5)
    assert run_template(tmp_path, run, stage).returncode == 0
    assert "fidelity" not in record(stage)
