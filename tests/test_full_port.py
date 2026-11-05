# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The full-port route, slice 1 (docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md): the stage 2 gate and its
fabrication guards, the stage table's routing, the block code and the skill text. No test here touches a chip."""
import hashlib
import json
from pathlib import Path

import pytest

from orchard import defaults, full_port
from orchard.stages import STAGES, gate_full_port, spec_for

SKILLS = Path(__file__).resolve().parent.parent / "orchard" / "skills"
CHIPS = ["0000:01:00.0", "0000:02:00.0"]
REF_IDS = list(range(100, 132))                      # 32 reference ids
TEXT = "The sea was calm that morning and the boat left early."


def sha(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def put(run, rel, content):
    p = Path(run) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content if isinstance(content, str) else json.dumps(content))
    return p


def build(tmp_path, *, result=None, record=None, marker=None, output=None, model_src="import ttnn\n", ref_ids=REF_IDS, stage=2):
    """A stage 2 directory (or, with `stage=3`, a stage 3 directory: issue 326) a passing full-port run would leave. Each argument replaces a part (a dict is merged over the good one)
    so a test changes one fact and watches the gate name it."""
    run = tmp_path / "run"
    ref = put(run, "stages/1/evidence/reference/generated-ids.json", {"generated_ids": ref_ids, "generated_text": "x"})
    prm = put(run, "stages/1/evidence/reference/prompt-ids.json", {"prompt_ids": [1, 2, 3], "prompt_text": "p"})
    model = put(run, f"stages/{stage}/port/port_model.py", model_src)
    template = put(run, f"stages/{stage}/" + full_port.TEMPLATE_NAME, full_port.TEMPLATE.read_text())      # the supervisor's copy of the shipped file
    config = put(run, f"stages/{stage}/" + full_port.CONFIG_NAME, {"model_snapshot": "snapshot"})
    good_record = {
        "schema": full_port.RECORD_SCHEMA,
        "reference": {"path": "stages/1/evidence/reference/generated-ids.json", "sha256": sha(ref.read_bytes())},
        "prompt": {"path": "stages/1/evidence/reference/prompt-ids.json", "sha256": sha(prm.read_bytes())},
        "reference_ids": list(ref_ids), "chip_ids": list(ref_ids[:30]) + [0, 0],
        "top1_agreement": 30 / 32, "chips": list(CHIPS), "load_s": 120.5,
        "free_run_ids": [ref_ids[0]] + list(range(1, 48)), "free_run_text": TEXT, "template_sha256": sha(template.read_bytes()),
        "config": {"path": f"stages/{stage}/" + full_port.CONFIG_NAME, "sha256": sha(config.read_bytes())},
        "port_sources": [{"path": f"stages/{stage}/port/port_model.py", "sha256": sha(model.read_bytes()), "role": "model"}]}
    rec = {**good_record, **(record or {})}
    rec_path = put(run, f"stages/{stage}/evidence/port-check.json", rec)
    out_text = f"loading\nPORT_CHECK_SHA256 {sha(rec_path.read_bytes())}\ndone\n" if output is None else output
    out_path = put(run, f"stages/{stage}/evidence/hw-test-output.txt", out_text)
    good_marker = {"command": full_port.command(stage), "returncode": 0, "timed_out": False, "seconds": 5.0, "chips": list(CHIPS),
                   "output": {"path": f"stages/{stage}/evidence/hw-test-output.txt", "sha256": sha(out_path.read_bytes())}}
    put(run, f"stages/{stage}/test-result.json", {**good_marker, **(marker or {})})
    good_result = {"runs": True, "coherent": True, "n_tokens": 32, "top1_agreement": 30 / 32, "load_s": 120.5, "free_run_text": TEXT,
                   "evidence": [f"stages/{stage}/evidence/port-check.json", f"stages/{stage}/evidence/hw-test-output.txt"]}
    put(run, f"stages/{stage}/result.json", {**good_result, **(result or {})})
    return run / "stages" / str(stage), run


def reasons(tmp_path, **kw):
    stage_dir, run = build(tmp_path, **kw)
    return gate_full_port(stage_dir, run)


def test_a_complete_result_passes_and_lists_its_evidence(tmp_path):
    g = reasons(tmp_path)
    assert g.ok, g.reasons
    assert "stages/2/evidence/port-check.json" in g.evidence and "stages/2/evidence/hw-test-output.txt" in g.evidence


def test_the_bar_is_the_weights_only_stage_2_bar():
    assert full_port.TOP1_MIN == defaults.SWAP_TOP1_MIN and full_port.MIN_TOKENS == defaults.SWAP_MIN_TOKENS


# ---- each guard, one at a time ----------------------------------------------------------------------

def one(g, word):
    """The reasons of a failing gate that mention `word`."""
    assert not g.ok, "the gate passed a result it must refuse"
    return [r for r in g.reasons if word in r]


def test_a_result_the_agent_inflated_is_refused_because_the_gate_recomputes_agreement(tmp_path):
    # The chip ids match 30 of 32 reference ids (0.9375); the agent writes 1.0.
    assert one(reasons(tmp_path, result={"top1_agreement": 1.0}), "recomputed")


def test_agreement_below_the_bar_is_refused_even_when_reported_correctly(tmp_path):
    bad = [0] * 32
    g = reasons(tmp_path, record={"chip_ids": bad}, result={"top1_agreement": 0.0})
    assert one(g, f"below the minimum of {full_port.TOP1_MIN}")


def test_too_few_positions_are_refused(tmp_path):
    ids = REF_IDS[:8]
    g = reasons(tmp_path, ref_ids=ids, record={"chip_ids": list(ids), "reference_ids": list(ids)}, result={"n_tokens": 8, "top1_agreement": 1.0})
    assert one(g, "at least")


def test_reference_ids_that_differ_from_the_stage_1_file_are_refused(tmp_path):
    # A port check that compares with ids it made up agrees with itself.
    fake = [i + 1000 for i in REF_IDS]
    g = reasons(tmp_path, record={"reference_ids": fake, "chip_ids": fake}, result={"top1_agreement": 1.0})
    assert one(g, "stage 1")


def test_a_reference_file_that_is_not_the_stage_1_file_is_refused(tmp_path):
    g = reasons(tmp_path, record={"reference": {"path": "stages/2/port/port_model.py", "sha256": sha("import ttnn\n")}})
    assert one(g, "reference.path")


def test_a_reference_sha_that_does_not_match_the_file_is_refused(tmp_path):
    g = reasons(tmp_path, record={"reference": {"path": "stages/1/evidence/reference/generated-ids.json", "sha256": "0" * 64}})
    assert one(g, "reference.sha256")


def test_chips_that_are_not_the_leased_chips_are_refused(tmp_path):
    assert one(reasons(tmp_path, record={"chips": ["0000:09:00.0"]}), "chips")


def test_a_failed_hardware_test_is_refused_and_its_failure_text_is_quoted(tmp_path):
    g = reasons(tmp_path, marker={"returncode": 4}, result={"runs": False, "failure": "device init failed"})
    assert one(g, "device init failed") and one(g, "exited with code 4")


def test_a_timed_out_test_is_refused(tmp_path):
    assert one(reasons(tmp_path, marker={"timed_out": True, "returncode": None}), "deadline")


def test_a_record_the_test_output_does_not_name_is_refused(tmp_path):
    # An agent that writes port-check.json by hand cannot also make the supervisor's output file carry its sha256.
    assert one(reasons(tmp_path, output="loading\ndone\n"), "PORT_CHECK_SHA256")


def test_an_output_file_that_is_not_the_one_the_supervisor_recorded_is_refused(tmp_path):
    assert one(reasons(tmp_path, marker={"output": {"path": "stages/2/evidence/hw-test-output.txt", "sha256": "1" * 64}}), "hw-test-output")


def test_port_sources_that_changed_after_the_record_are_refused(tmp_path):
    stage_dir, run = build(tmp_path)
    (stage_dir / "port" / "port_model.py").write_text("import ttnn\n# edited\n")
    assert one(gate_full_port(stage_dir, run), "port_sources")


def test_a_port_with_no_ttnn_is_refused(tmp_path):
    assert one(reasons(tmp_path, model_src="import torch\n"), "ttnn")


def test_a_model_source_that_reads_the_reference_is_refused(tmp_path):
    src = "import ttnn\nids = open('stages/1/evidence/reference/generated-ids.json')\n"
    assert one(reasons(tmp_path, model_src=src), "reference")


def test_a_record_with_no_model_source_is_refused(tmp_path):
    g = reasons(tmp_path, record={"port_sources": []})
    assert one(g, "port_sources")


def test_free_run_text_that_differs_from_the_record_is_refused(tmp_path):
    assert one(reasons(tmp_path, result={"free_run_text": TEXT + " extra"}), "free_run_text")


def test_free_run_text_that_is_one_repeated_word_is_refused(tmp_path):
    t = "the " * 30
    assert one(reasons(tmp_path, record={"free_run_text": t}, result={"free_run_text": t}), "free_run_text")


def test_coherent_must_be_true(tmp_path):
    assert one(reasons(tmp_path, result={"coherent": False}), "coherent")


def test_a_missing_result_is_one_reason(tmp_path):
    stage_dir, run = build(tmp_path)
    (stage_dir / "result.json").unlink()
    g = gate_full_port(stage_dir, run)
    assert not g.ok and "result.json is missing" in g.reasons[0]


def test_a_missing_record_is_a_reason_and_not_a_crash(tmp_path):
    stage_dir, run = build(tmp_path)
    (stage_dir / "evidence" / "port-check.json").unlink()
    g = gate_full_port(stage_dir, run)
    assert not g.ok and any("port-check.json" in r for r in g.reasons)


def test_a_hostile_record_is_a_reason_and_not_a_crash(tmp_path):
    stage_dir, run = build(tmp_path)
    (stage_dir / "evidence" / "port-check.json").write_text('{"chip_ids": NaN}')
    g = gate_full_port(stage_dir, run)
    assert not g.ok


# ---- routing ---------------------------------------------------------------------------------------

def test_without_the_opt_in_the_full_port_table_is_unchanged():
    assert spec_for(2, "full-port").skill == "functional-decoder"
    assert spec_for(2, "full-port", None, "full-port").skill == "functional-decoder"


def test_with_the_opt_in_a_full_port_class_gets_the_full_port_stage_2():
    s = spec_for(2, "full-port", None, "full-port", full_port=True)
    assert s.skill == "full-port-check" and s.gate is gate_full_port and s.boards == 1 and s.gate_file == "result.json"
    assert s.refs == ("tt-device-usage",)


def test_the_opt_in_changes_no_other_stage_and_no_other_class():
    for n in (0, 1, 4, 5, 7, 8):      # stage 3 changes too since issue 326 (tests/test_full_port_stage3.py), stage 6 since issue 329 (tests/test_supervisor_full_port_quality.py)
        assert spec_for(n, "full-port", None, "full-port", full_port=True) == spec_for(n, "full-port", None, "full-port")
    assert spec_for(2, "full-port", None, "unknown", full_port=True).skill == "functional-decoder"
    assert spec_for(2, "weights-only", None, "weights-only", full_port=True).skill == "weights-swap-check"


def test_the_table_still_holds_the_generic_stage_2():
    assert STAGES[2].skill == "functional-decoder"


# ---- the skill --------------------------------------------------------------------------------------

def skill_text():
    return (SKILLS / "full-port-check.md").read_text()


def test_the_skill_is_a_marked_draft_with_the_name_the_table_uses():
    front = skill_text().split("---")[1]
    assert "\nname: full-port-check\n" in front and "\nstatus: draft." in front


def test_the_skill_names_the_gate_file_the_bar_and_the_record():
    flat = " ".join(skill_text().split())
    assert "result.json" in flat and "evidence/port-check.json" in flat and "PORT_CHECK_SHA256" in flat
    assert f"`top1_agreement` of at least {full_port.TOP1_MIN}" in flat
    assert f"`n_tokens` of at least {full_port.MIN_TOKENS}" in flat
    assert full_port.RECORD_SCHEMA in flat


def test_the_skill_says_results_written_in_prepare_are_set_aside():
    flat = " ".join(skill_text().split())
    assert "moved aside unread" in flat and "evidence/port-check.json" in flat


def test_the_skill_names_no_lease_tool_and_no_machine_path():
    text = skill_text()
    assert "gozer" not in text.lower()
    assert "/home/" not in text and "/mnt/" not in text


def test_the_skill_shows_how_a_failed_test_is_recorded():
    flat = " ".join(skill_text().split())
    assert "`runs` false" in flat and "`failure`" in flat


# ---- review findings (PR 331) ---------------------------------------------------------------------

@pytest.mark.parametrize("chips", [2, "0000:01:00.0", [], [1, 2], None])
def test_a_record_whose_chips_are_not_a_list_of_names_is_a_reason_and_not_a_crash(tmp_path, chips):
    assert one(reasons(tmp_path, record={"chips": chips}), "chips")


def test_a_marker_whose_chips_are_not_a_list_is_a_reason_and_not_a_crash(tmp_path):
    assert one(reasons(tmp_path, marker={"chips": 2}), "chips")


def test_a_nul_in_a_port_source_path_is_a_reason_and_not_a_crash(tmp_path):
    bad = {"path": "stages/2/port/port_model\u0000.py", "sha256": "0" * 64, "role": "model"}
    g = reasons(tmp_path, record={"port_sources": [bad]})
    assert one(g, "port_sources")


def test_a_nul_in_an_evidence_path_is_a_reason_and_not_a_crash(tmp_path):
    g = reasons(tmp_path, result={"evidence": ["stages/2/evidence/port-check.json", "stages/2/evidence/hw-test-output.txt", "a\u0000b"]})
    assert one(g, "evidence")


def test_the_marker_record_is_read_with_a_bound_and_a_fifo_is_refused_without_blocking(tmp_path):
    import os
    run = tmp_path / "run"
    (run / "stages" / "2").mkdir(parents=True)
    fifo = run / "stages" / "2" / "test-result.json"
    os.mkfifo(fifo)
    assert full_port.file_record(run, fifo) is None          # a blocking open would hang here
    plain = run / "stages" / "2" / "ok.json"
    plain.write_text("{}")
    assert full_port.file_record(run, plain) == {"path": "stages/2/ok.json", "sha256": sha("{}")}
    link = run / "stages" / "2" / "zero.json"
    os.symlink("/dev/zero", link)
    assert full_port.file_record(run, link) is None


@pytest.mark.parametrize("load", [None, "120.5", 0, -1, True])
def test_a_record_without_a_positive_numeric_load_time_is_refused(tmp_path, load):
    # result.json carries 120.5; with no number in the record nothing supports it.
    assert one(reasons(tmp_path, record={"load_s": load}), "load_s")


# ---- slice 2 (issue 325): the supervisor-owned check ----------------------------------------------------

def test_a_template_copy_that_differs_from_the_shipped_file_is_refused(tmp_path):
    stage_dir, run = build(tmp_path)
    (stage_dir / full_port.TEMPLATE_NAME).write_text("print('mine')\n")
    assert one(gate_full_port(stage_dir, run), "differs from the shipped template")


def test_a_record_that_carries_another_template_sha256_is_refused(tmp_path):
    assert one(reasons(tmp_path, record={"template_sha256": "1" * 64}), "template_sha256")


def test_a_record_with_no_template_sha256_is_refused(tmp_path):
    stage_dir, run = build(tmp_path)
    rec = json.loads((stage_dir / "evidence" / "port-check.json").read_text())
    del rec["template_sha256"]
    put(run, "stages/2/evidence/port-check.json", rec)
    assert one(gate_full_port(stage_dir, run), "template_sha256")


def test_a_hardware_test_command_that_is_not_the_templates_is_refused(tmp_path):
    assert one(reasons(tmp_path, marker={"command": "python3 stages/2/port/check.py"}), "command for this stage")


def test_two_lines_that_begin_with_the_check_mark_are_refused(tmp_path):
    out = f"PORT_CHECK_SHA256 {'0' * 64}\nPORT_CHECK_SHA256 {'1' * 64}\n"
    assert one(reasons(tmp_path, output=out), "begin with PORT_CHECK_SHA256")


def test_a_port_source_with_the_old_check_role_is_refused(tmp_path):
    src = [{"path": "stages/2/port/port_model.py", "sha256": sha("import ttnn\n"), "role": "check"}]
    assert one(reasons(tmp_path, record={"port_sources": src}), "role of 'model'")


def test_a_record_that_does_not_list_the_port_module_is_refused(tmp_path):
    stage_dir, run = build(tmp_path)
    other = put(run, "stages/2/port/other.py", "import ttnn\n")
    rec = json.loads((stage_dir / "evidence" / "port-check.json").read_text())
    rec["port_sources"] = [{"path": "stages/2/port/other.py", "sha256": sha(other.read_bytes()), "role": "model"}]
    put(run, "stages/2/evidence/port-check.json", rec)
    assert one(gate_full_port(stage_dir, run), "does not list stages/2/port/port_model.py")


def test_a_config_that_changed_since_the_record_is_refused(tmp_path):
    stage_dir, run = build(tmp_path)
    (stage_dir / full_port.CONFIG_NAME).write_text('{"model_snapshot": "elsewhere"}')
    assert one(gate_full_port(stage_dir, run), full_port.CONFIG_NAME)


def test_the_shipped_template_is_where_the_supervisor_and_the_gate_look():
    assert full_port.TEMPLATE == SKILLS / "full-port-templates" / "port_check.py" and full_port.TEMPLATE.is_file()
    assert full_port.COMMAND == f"python3 -P -s stages/2/{full_port.TEMPLATE_NAME}"


def test_the_skill_says_the_coder_writes_only_the_port_module_and_names_its_four_functions():
    flat = " ".join(skill_text().split())
    for name in ("load(model_dir, mesh_device)", "prefill(model, ids)", "decode_step(model, token_id, position)", "close(model)"):
        assert name in flat, name
    assert "stages/2/port/port_model.py" in flat and "You do not write the check" in flat
    assert full_port.COMMAND in flat and "port_check_config.json" in flat and "template_sha256" in flat


def test_the_skill_no_longer_has_the_coder_write_a_check_script_or_mark_a_check_role():
    flat = " ".join(skill_text().split())
    assert '"role": "check"' not in flat and "python3 stages/2/port/check.py" not in flat
    assert "Do not run `port_check.py` yourself" in flat and "Do not edit `port_check.py`" in flat
