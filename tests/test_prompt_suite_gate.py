# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The stage 2 and stage 4 swap gates read the prompt suite beside the 32-token check (issue 15).

The gate reads evidence/swap-check.json, which the template wrote. It ignores any suite summary in result.json, because an agent
types result.json and the record in swap-check.json holds the raw replies, which orchard/prompt_suite.py judges. A swap-check.json
without a suite record (an older run, or a config that did not ask for the suite) changes nothing."""
import copy
import json

import pytest

from orchard import prompt_suite as ps
from orchard.stages import gate_mesh_swap, gate_weights_swap, gate_weights_swap_sidecar
from test_prompt_suite_verdict import LOOP, control_of, make_record, with_entry
from test_stages import GOOD_DELTA, SWAP, mesh_stage, stage, swap_config, write
from test_stages_class import RESULT, SIDE_DELTA

HEALTHY_FREE = "Paris is the capital of France and a large city"


def swap_check(record=None, **extra):
    rep = {"result_draft": {"top1_agreement": SWAP["top1_agreement"]}, "free_run_text": HEALTHY_FREE,
           "reference_generated_text": HEALTHY_FREE, **extra}
    if record is not None:
        rep["prompt_suite"] = record
    return rep


def stage2(tmp_path, report=None, gate=gate_weights_swap):
    sd, run = stage(tmp_path, 2, "result.json", SWAP)
    write(run, "stages/0/delta.json", GOOD_DELTA)         # names the nearest model, which a control has to have been served from
    if report is not None:
        write(run, "stages/2/evidence/swap-check.json", report)
    return gate(sd, run)


def test_a_stage_2_result_with_no_swap_check_file_is_judged_as_before(tmp_path):
    assert stage2(tmp_path).ok


def test_a_swap_check_without_the_suite_changes_nothing(tmp_path):
    assert stage2(tmp_path, swap_check()).ok


def test_a_healthy_suite_passes(tmp_path):
    g = stage2(tmp_path, swap_check(make_record()))
    assert g.ok, g.reasons


def _stage2_with_raw_swap_check(tmp_path, raw: bytes, link: bool = False):
    sd, run = stage(tmp_path, 2, "result.json", SWAP)
    write(run, "stages/0/delta.json", GOOD_DELTA)
    path = run / "stages/2/evidence/swap-check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if link:
        target = tmp_path / "elsewhere.json"
        target.write_bytes(raw)
        path.symlink_to(target)
    else:
        path.write_bytes(raw)
    return gate_weights_swap(sd, run), gate_weights_swap_sidecar


def test_a_swap_check_file_that_is_not_json_fails_the_stage_and_says_so(tmp_path):
    g, _ = _stage2_with_raw_swap_check(tmp_path, b"{not json")
    assert not g.ok and any("swap-check.json is not readable JSON" in r for r in g.reasons), g.reasons


@pytest.mark.parametrize("raw,link,why", [
    pytest.param(b'{"hub_fetch": ["x/y"], "tok_s": NaN}', False, "an unrelated NaN", id="nan"),
    pytest.param(b'{"hub_fetch": ["x/y"], "pad": "' + b"a" * (17 * 1024 * 1024) + b'"}', False, "a file over the size cap", id="oversize"),
    pytest.param(b'{"hub_fetch": ["x/y"]}', True, "a link", id="link"),
])
def test_a_swap_check_the_safe_reader_refuses_cannot_hide_a_failing_report(tmp_path, raw, link, why):
    """The report holds a non-empty hub_fetch, which fails the gate when it is read. A refusal of the file must not drop that."""
    g, _ = _stage2_with_raw_swap_check(tmp_path, raw, link)
    assert not g.ok and any("swap-check.json is not readable JSON" in r for r in g.reasons), (why, g.reasons)


def test_a_blocking_finding_the_control_does_not_share_fails_the_stage_and_names_it(tmp_path):
    g = stage2(tmp_path, swap_check(with_entry(make_record(), "arithmetic", content=LOOP)))
    assert not g.ok
    assert any("prompt suite" in r and "arithmetic" in r and "mechanical_repetition" in r for r in g.reasons)


def test_the_same_finding_in_the_control_passes(tmp_path):
    rec = with_entry(make_record(), "arithmetic", content=LOOP)
    rec["control"] = control_of(rec)
    assert stage2(tmp_path, swap_check(rec)).ok


def test_an_advisory_finding_passes(tmp_path):
    assert stage2(tmp_path, swap_check(with_entry(make_record(), "chat-capital", content=""))).ok


def test_the_suite_is_read_from_the_template_file_and_not_from_the_agents_result(tmp_path):
    # The agent's result.json carries a prompt_suite summary that says all is well; the swap-check.json says otherwise.
    sd, run = stage(tmp_path, 2, "result.json", {**SWAP, "prompt_suite": {"ran": True, "findings": []}})
    write(run, "stages/2/evidence/swap-check.json", swap_check(with_entry(make_record(), "arithmetic", content=LOOP)))
    assert not gate_weights_swap(sd, run).ok


def test_a_looping_free_run_the_cpu_reference_does_not_share_fails_the_stage(tmp_path):
    g = stage2(tmp_path, swap_check(make_record(), free_run_text=LOOP))
    assert not g.ok and any("stage 1 free run" in r for r in g.reasons)


def test_a_record_of_other_prompts_fails_the_stage(tmp_path):
    g = stage2(tmp_path, swap_check(make_record(suite_sha256="0" * 64)))
    assert not g.ok and any("not the committed prompt suite" in r for r in g.reasons)


def test_the_32_token_reasons_and_the_suite_reasons_are_both_given(tmp_path):
    sd, run = stage(tmp_path, 2, "result.json", {**SWAP, "top1_agreement": 0.5})
    write(run, "stages/2/evidence/swap-check.json", swap_check(with_entry(make_record(), "arithmetic", content=LOOP)))
    reasons = gate_weights_swap(sd, run).reasons
    assert any("top1_agreement" in r for r in reasons) and any("prompt suite" in r for r in reasons)


# ---- the sidecar variant --------------------------------------------------------------------------------------------

def sidecar_stage2(tmp_path, report):
    run = tmp_path / "run"
    write(run, "stages/0/delta.json", SIDE_DELTA)
    write(run, "stages/2/evidence/sidecar-parity.json", "{}")
    sd, run = stage(tmp_path, 2, "result.json", RESULT)
    write(run, "stages/2/evidence/swap-check.json", report)
    return gate_weights_swap_sidecar(sd, run)


def test_the_sidecar_gate_reads_the_suite_too(tmp_path):
    assert sidecar_stage2(tmp_path / "a", swap_check(make_record())).ok
    g = sidecar_stage2(tmp_path / "b", swap_check(with_entry(make_record(), "arithmetic", content=LOOP)))
    assert not g.ok and any("prompt suite" in r for r in g.reasons)


# ---- stage 4 --------------------------------------------------------------------------------------------------------

def mesh_with_suite(tmp_path, records, configs=None):
    """mesh_stage, then each configuration's swap-check.json gets the record given for its chip count."""
    configs = configs or [swap_config(n) for n in records]
    sd, run = mesh_stage(tmp_path, configs)
    write(run, "stages/0/delta.json", GOOD_DELTA)         # names the nearest model, which a control has to have been served from
    for n, rec in records.items():
        write(run, f"stages/4/configs/{n}/evidence/swap-check.json", swap_check(rec))
    return sd, run


def test_a_stage_4_configuration_with_a_healthy_suite_passes(tmp_path):
    g = gate_mesh_swap(*mesh_with_suite(tmp_path, {2: make_record(), 4: make_record()}), required=(2, 4))
    assert g.ok, g.reasons


def test_a_stage_4_configuration_with_a_finding_fails_and_says_which_configuration(tmp_path):
    bad = with_entry(make_record(), "arithmetic", content=LOOP)
    g = gate_mesh_swap(*mesh_with_suite(tmp_path, {2: make_record(), 4: bad}), required=(2, 4))
    assert not g.ok
    assert any(r.startswith("the 4-chip configuration: ") and "arithmetic" in r for r in g.reasons)
    assert not any(r.startswith("the 2-chip configuration: ") and "prompt suite" in r for r in g.reasons)


def test_a_stage_4_swap_check_without_the_suite_is_judged_as_before(tmp_path):
    sd, run = mesh_stage(tmp_path, [swap_config(2), swap_config(4)])
    assert gate_mesh_swap(sd, run, required=(2, 4)).ok


def test_a_stage_4_configuration_that_does_not_claim_a_pass_is_not_read_for_the_suite(tmp_path):
    bad = with_entry(make_record(), "arithmetic", content=LOOP)
    sd, run = mesh_with_suite(tmp_path, {2: make_record(), 1: bad},
                              configs=[swap_config(2), swap_config(1, ok=False)])
    g = gate_mesh_swap(sd, run, required=(2,))
    assert g.ok, g.reasons


def test_the_top1_check_of_stage_4_still_works_next_to_the_suite(tmp_path):
    sd, run = mesh_with_suite(tmp_path, {2: make_record()})
    write(run, "stages/4/configs/2/evidence/swap-check.json",
          {**swap_check(make_record()), "result_draft": {"top1_agreement": 0.5}})
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok and any("result_draft" in r for r in g.reasons)


def test_a_stage_4_control_shared_finding_passes(tmp_path):
    rec = with_entry(make_record(), "arithmetic", content=LOOP)
    rec = copy.deepcopy(rec)
    rec["control"] = control_of(rec)
    g = gate_mesh_swap(*mesh_with_suite(tmp_path, {2: rec}), required=(2,))
    assert g.ok, g.reasons


def test_the_suite_gate_does_not_read_a_stage_4_file_it_cannot_parse(tmp_path):
    sd, run = mesh_with_suite(tmp_path, {2: make_record()})
    (run / "stages/4/configs/2/evidence/swap-check.json").write_text("{not json")
    g = gate_mesh_swap(sd, run, required=(2,))
    assert not g.ok            # the existing check of result_draft still fails it
    assert not any("prompt suite" in r for r in g.reasons)


def test_ps_is_the_module_the_gates_use():
    import orchard.stages as stages
    assert stages.prompt_suite is ps


# ---- the base weights stood in by mistake (issue 15) ----------------------------------------------------------------------

def test_a_stage_2_run_whose_replies_equal_the_control_on_every_prompt_fails_with_a_reason_that_says_why(tmp_path):
    rec = make_record()
    rec["control"] = control_of(rec, differ=False)
    g = stage2(tmp_path, swap_check(rec))
    assert not g.ok and any("identical to the control" in r and "weights" in r for r in g.reasons)


def test_a_stage_4_configuration_whose_replies_equal_the_control_fails_that_configuration(tmp_path):
    rec = make_record()
    rec["control"] = control_of(rec, differ=False)
    g = gate_mesh_swap(*mesh_with_suite(tmp_path, {2: make_record(), 4: rec}), required=(2, 4))
    assert not g.ok and any(r.startswith("the 4-chip configuration: ") and "identical" in r for r in g.reasons)
