# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 789: the stage 2 check scores the port against the sampled reference of stage 1 (`fidelity-ids.json`) when there is one.

The greedy pass, the `fidelity` block and the `positions` block are as they were. A second teacher-forced pass over the sampled ids gives the
`sampled_fidelity` block: report-only, bound to the stage 1 file by its sha256, and recomputed by the gate. The template runs for real on the CPU
with a fake `ttnn`; nothing touches a chip."""
import hashlib
import json
import math

import pytest

from orchard import full_port
from orchard.stages import gate_full_port
from test_port_check_fidelity import pc
from port_check_fakes import HONEST, HONEST_LP, REF_IDS, WRONG, make_run, run_template, sampled_reference, write_result

FILE = "stages/1/evidence/reference/fidelity-ids.json"


def record(stage):
    return json.loads((stage / "evidence" / "port-check.json").read_text())


def go(tmp_path, *, port=HONEST, cap=None, **kw):
    run, stage, _ = make_run(tmp_path, port, **kw)
    if cap is not None:
        cfg = json.loads((stage / full_port.CONFIG_NAME).read_text())
        (stage / full_port.CONFIG_NAME).write_text(json.dumps({**cfg, "max_positions": cap}))
    done = run_template(tmp_path, run, stage)
    return run, stage, done


def sampled_reasons(rec, run, stage):
    out = []
    full_port._sampled_fidelity_reasons(rec, run, stage, out)
    return out


# ---- the record --------------------------------------------------------------------------------------------------------

def test_a_sampled_reference_gives_a_sampled_fidelity_block_of_the_capped_length_and_the_greedy_blocks_are_as_before(tmp_path):
    run, stage, done = go(tmp_path, sampled=sampled_reference())
    assert done.returncode == 0, done.stdout + done.stderr
    rec = record(stage)
    b = rec["sampled_fidelity"]
    assert b["kind"] == "reference-sampled-text" and b["tokens"] == 64 and b["value"] == pytest.approx(10.0, abs=1e-6)       # 90 sampled, cap 64
    assert b["source"] == {"path": FILE, "sha256": hashlib.sha256((run / FILE).read_bytes()).hexdigest(), "length": 90}
    assert rec["fidelity"]["kind"] == "reference-greedy-text" and rec["fidelity"]["tokens"] == len(REF_IDS)
    assert rec["positions"] == {"cap": 64, "scored": len(REF_IDS), "reference_length": len(REF_IDS)}
    assert "cap of 64 positions limits" in b["note"]


def test_a_higher_cap_scores_the_whole_sampled_reference(tmp_path):
    run, stage, done = go(tmp_path, cap=90, sampled=sampled_reference())
    assert done.returncode == 0, done.stdout + done.stderr
    b = record(stage)["sampled_fidelity"]
    assert b["tokens"] == 90 and len(b["chip_logprobs"]) == 90 and len(b["reference_logprobs"]) == 90


def test_a_short_sampled_reference_says_it_is_the_limit(tmp_path):
    run, stage, done = go(tmp_path, sampled=sampled_reference(range(100, 120)))
    b = record(stage)["sampled_fidelity"]
    assert b["tokens"] == 20 and "reference holds 20 tokens" in b["note"]


def test_the_port_is_teacher_forced_on_the_sampled_ids_not_on_its_own_or_the_greedy_ones(tmp_path):
    ids = list(range(100, 190))
    ids[5], ids[6] = 150, 151                      # the honest port expects 105 at position 5; after 150 it expects 151, which position 6 holds
    run, stage, done = go(tmp_path, sampled=sampled_reference(ids))
    lp = record(stage)["sampled_fidelity"]["chip_logprobs"]
    low = 0.0 - math.log(math.e + 199.0)
    assert lp[5] == pytest.approx(low) and lp[4] == pytest.approx(HONEST_LP) and lp[6] == pytest.approx(HONEST_LP) and lp[0] == pytest.approx(HONEST_LP)


def test_a_port_that_gives_the_sampled_tokens_less_probability_scores_below_ten(tmp_path):
    run, stage, done = go(tmp_path, port=WRONG, sampled=sampled_reference())
    assert done.returncode == 0
    assert record(stage)["sampled_fidelity"]["value"] == pytest.approx(10 * math.exp((0.0 - math.log(math.e + 199.0)) - HONEST_LP))


def test_without_a_sampled_reference_the_record_has_no_sampled_block(tmp_path):
    run, stage, done = go(tmp_path)
    assert done.returncode == 0 and "sampled_fidelity" not in record(stage)


@pytest.mark.parametrize("over,why", [({"complete": False}, "not complete"),
                                      ({"prompt_ids_sha256": "0" * 64}, "prompt"),
                                      ({"temperature": 0.7}, "temperature"),
                                      ({"logprobs": [0.5] * 90}, "log-probabilities"),
                                      ({"logprobs": [HONEST_LP] * 89}, "same length")])
def test_a_sampled_reference_that_cannot_be_used_is_recorded_as_skipped_with_the_reason_and_the_run_still_passes(tmp_path, over, why):
    run, stage, done = go(tmp_path, sampled=sampled_reference(**over))
    assert done.returncode == 0, done.stdout + done.stderr
    b = record(stage)["sampled_fidelity"]
    assert b["status"] == "skipped" and why in b["reason"] and "value" not in b
    write_result(stage)
    assert gate_full_port(stage, run).ok


def test_an_unreadable_sampled_file_is_skipped_too(tmp_path):
    run, stage, _ = make_run(tmp_path)
    (run / FILE).write_text("{not json")
    assert run_template(tmp_path, run, stage).returncode == 0
    assert record(stage)["sampled_fidelity"]["status"] == "skipped"


def test_the_check_line_names_the_sampled_score(tmp_path):
    run, stage, done = go(tmp_path, sampled=sampled_reference())
    assert "sampled fidelity 10.000 over 64 tokens" in done.stdout and "reported not gated" in done.stdout


def test_the_limit_constants_are_the_same_in_the_template_and_in_stage_1():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rg", full_port.TEMPLATE.parent.parent / "reference-gate-templates" / "reference_gate.py")
    rg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rg)
    assert rg.FIDELITY_MAX == full_port.POSITIONS_LIMIT and rg.FIDELITY_FILE == FILE.rsplit("/", 1)[1]
    assert full_port.SAMPLED_FILE == FILE


# ---- the gate ------------------------------------------------------------------------------------------------------------

@pytest.fixture
def honest(tmp_path):
    run, stage, done = go(tmp_path, sampled=sampled_reference())
    assert done.returncode == 0
    return run, stage


def test_the_gate_accepts_the_record_the_template_wrote(honest):
    run, stage = honest
    write_result(stage)
    g = gate_full_port(stage, run)
    assert g.ok, g.reasons
    assert sampled_reasons(record(stage), run, stage) == []


def test_the_gate_refuses_a_value_that_is_not_what_the_log_probabilities_give(honest):
    run, stage = honest
    rec = record(stage)
    rec["sampled_fidelity"]["value"] = 9.5
    assert any("recomputed" in r for r in sampled_reasons(rec, run, stage))


def test_the_gate_refuses_reference_log_probabilities_that_are_not_the_stage_one_files(honest):
    run, stage = honest
    rec = record(stage)
    b = rec["sampled_fidelity"]
    b["reference_logprobs"] = [-0.5] * b["tokens"]
    b["value"] = full_port.fidelity_value(b["reference_logprobs"], b["chip_logprobs"])
    assert any("numbers the sampled reference did not produce" in r for r in sampled_reasons(rec, run, stage))


def test_the_gate_refuses_a_stage_one_file_that_changed_after_the_check(honest):
    run, stage = honest
    (run / FILE).write_text(json.dumps(sampled_reference(logprob=-0.1)))
    assert any("sha256" in r for r in sampled_reasons(record(stage), run, stage))


def test_the_gate_refuses_a_block_that_scored_fewer_positions_than_the_cap_and_the_file_allow(honest):
    run, stage = honest
    rec = record(stage)
    b = rec["sampled_fidelity"]
    b["tokens"], b["reference_logprobs"], b["chip_logprobs"] = 20, b["reference_logprobs"][:20], b["chip_logprobs"][:20]
    b["value"] = full_port.fidelity_value(b["reference_logprobs"], b["chip_logprobs"])
    assert any("not the smaller of" in r for r in sampled_reasons(rec, run, stage))


def test_the_gate_refuses_a_block_whose_stage_one_file_is_gone(honest):
    run, stage = honest
    (run / FILE).unlink()
    assert any("not a file in the run directory" in r for r in sampled_reasons(record(stage), run, stage))


@pytest.mark.parametrize("bad", ["x", [1], {"kind": "reference-sampled-text"}, {"status": "skipped"}, {"status": "skipped", "reason": 5}])
def test_the_gate_refuses_a_block_of_the_wrong_shape(honest, bad):
    run, stage = honest
    rec = record(stage)
    rec["sampled_fidelity"] = bad
    assert sampled_reasons(rec, run, stage)


def test_the_gate_accepts_a_skipped_block_that_names_a_reason(honest):
    run, stage = honest
    rec = record(stage)
    rec["sampled_fidelity"] = {"status": "skipped", "reason": "x"}
    assert sampled_reasons(rec, run, stage) == []


def test_a_record_with_no_sampled_block_passes_the_gate_when_the_file_exists(tmp_path):
    # a check made before this option existed: nothing to recompute
    run, stage, _ = go(tmp_path, sampled=sampled_reference())
    rec = record(stage)
    del rec["sampled_fidelity"]
    assert sampled_reasons(rec, run, stage) == []


# ---- a port that fails the extra pass ---------------------------------------------------------------------------------------

# prefill is called once by the greedy pass and then once by the sampled pass: the second call is the one that fails
SECOND_PREFILL = HONEST.replace("def prefill(model, ids):\n    return _peak(100)",
                                "CALLS = [0]\ndef prefill(model, ids):\n    CALLS[0] += 1\n    if CALLS[0] == 2:\n        @@BOOM@@\n    return _peak(100)")


def test_a_port_that_raises_in_the_sampled_pass_is_recorded_as_skipped_and_the_rest_of_the_record_is_kept(tmp_path):
    run, stage, done = go(tmp_path, port=SECOND_PREFILL.replace("@@BOOM@@", "raise RuntimeError('sampled boom')"), sampled=sampled_reference())
    assert done.returncode == 0, done.stdout + done.stderr
    rec = record(stage)
    assert rec["sampled_fidelity"]["status"] == "skipped" and "sampled boom" in rec["sampled_fidelity"]["reason"]
    assert rec["top1_agreement"] == 1.0 and "fidelity" in rec


def test_a_port_that_dies_in_the_sampled_pass_fails_the_run_like_any_request_that_gets_no_answer(tmp_path):
    run, stage, done = go(tmp_path, port=SECOND_PREFILL.replace("@@BOOM@@", "import os; os._exit(3)"), sampled=sampled_reference())
    assert done.returncode == 2 and "did not answer 'prefill'" in done.stdout


def test_a_pass_that_gets_no_log_probability_from_the_port_is_skipped_with_that_reason():
    class Worker:
        last = {}

        def ask(self, **kw):
            self.last = {"lp": None}
            return 0
    out = pc.sampled_fidelity(Worker(), [1, 2, 3], 64, {"ids": [5, 6], "logprobs": [-1.0, -1.0], "sha256": "x", "length": 2})
    assert out["status"] == "skipped" and "no log-probability" in out["reason"]


def test_a_sampled_file_that_is_a_link_is_skipped(tmp_path):
    run, stage, _ = make_run(tmp_path)
    real = tmp_path / "elsewhere.json"
    real.write_text(json.dumps(sampled_reference()))
    (run / FILE).symlink_to(real)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0 and record(stage)["sampled_fidelity"]["status"] == "skipped" and "link" in record(stage)["sampled_fidelity"]["reason"]


def test_the_gate_refuses_a_source_length_that_is_not_the_files(honest):
    run, stage = honest
    rec = record(stage)
    rec["sampled_fidelity"]["source"]["length"] = 91
    assert any("source.length" in r for r in sampled_reasons(rec, run, stage))


def test_the_gate_refuses_a_source_path_that_is_not_the_stage_one_file(honest):
    run, stage = honest
    rec = record(stage)
    rec["sampled_fidelity"]["source"]["path"] = "stages/1/evidence/reference/top5-logits.json"
    assert any("source.path" in r for r in sampled_reasons(rec, run, stage))


def test_the_gate_refuses_a_stage_one_file_with_no_log_probabilities_even_when_the_hash_was_updated(honest):
    run, stage = honest
    (run / FILE).write_text(json.dumps({"ids": [1], "logprobs": "x"}))
    rec = record(stage)
    rec["sampled_fidelity"]["source"]["sha256"] = hashlib.sha256((run / FILE).read_bytes()).hexdigest()
    assert any("holds no list of log-probabilities" in r for r in sampled_reasons(rec, run, stage))


def test_the_full_gate_calls_the_sampled_check(honest):
    # the wiring: a forged value in the record on disk is named by gate_full_port itself, not only by the helper
    run, stage = honest
    rec = record(stage)
    rec["sampled_fidelity"]["value"] = 9.5
    (stage / "evidence" / "port-check.json").write_text(json.dumps(rec))
    write_result(stage)
    assert any("sampled_fidelity value" in r for r in gate_full_port(stage, run).reasons)


def test_a_sampled_file_that_is_a_fifo_is_skipped_and_does_not_block_the_check(tmp_path):
    import os
    run, stage, _ = make_run(tmp_path)
    os.mkfifo(run / FILE)
    done = run_template(tmp_path, run, stage)           # the run helper's timeout is 120 s: a read of the pipe would hit it
    assert done.returncode == 0 and record(stage)["sampled_fidelity"]["status"] == "skipped"


@pytest.mark.parametrize("field,bad", [("chip_logprobs", [-10 ** 400] * 64), ("reference_logprobs", [-10 ** 400] * 64), ("scale", 10 ** 400),
                                       ("scale", 5.0), ("tokens", 10 ** 400), ("value", 10 ** 400)])
def test_the_gate_refuses_huge_numbers_in_the_block_without_raising(honest, field, bad):
    run, stage = honest
    rec = record(stage)
    rec["sampled_fidelity"][field] = bad
    assert sampled_reasons(rec, run, stage)


def test_a_huge_logprob_in_the_stage_one_file_is_skipped_by_the_template_and_does_not_crash_it(tmp_path):
    lps = sampled_reference()
    lps["logprobs"][3] = -10 ** 400
    run, stage, done = go(tmp_path, sampled=lps)
    assert done.returncode == 0 and record(stage)["sampled_fidelity"]["status"] == "skipped"
