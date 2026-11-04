# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 3 of the full-port route (issue 326): the port that passed stage 2 on one board, on the chip count the run is configured for.
This file holds the gate, the routing and the skill text. The supervisor's side is in tests/test_supervisor_full_port_stage3.py. No test
here touches a chip."""
from pathlib import Path

import pytest

from orchard import full_port
from orchard.stages import FULL_PORT_STAGE_2, FULL_PORT_STAGE_3, STAGES, SKIP_3_FULL_PORT, gate_full_port, gate_full_port_3, spec_for
from test_full_port import REF_IDS, build, sha

SKILLS = Path(__file__).resolve().parent.parent / "orchard" / "skills"
FOUR = ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"]


def stage3(tmp_path, *, chips=FOUR, **kw):
    """A stage 3 directory a passing run would leave: the record and the supervisor's marker both list `chips`."""
    record = {**kw.pop("record", {}), "chips": list(chips)}
    marker = {**kw.pop("marker", {}), "chips": list(chips)}
    return build(tmp_path, stage=3, record=record, marker=marker, **kw)


def gate3(tmp_path, *, expected=4, **kw):
    stage_dir, run = stage3(tmp_path, **kw)
    return gate_full_port_3(stage_dir, run, expected_chips=expected)


def one(g, word):
    assert not g.ok, "the gate passed a result it must refuse"
    return [r for r in g.reasons if word in r]


# ---- the gate ---------------------------------------------------------------------------------------

def test_a_stage_3_result_on_the_configured_chips_passes_and_lists_its_stage_3_evidence(tmp_path):
    g = gate3(tmp_path)
    assert g.ok, g.reasons
    assert "stages/3/evidence/port-check.json" in g.evidence and "stages/3/evidence/hw-test-output.txt" in g.evidence


def test_a_test_that_ran_on_fewer_chips_than_the_stage_is_configured_for_is_refused(tmp_path):
    # The port passed on a mesh of two and the stage is configured for four: nothing shows it on four.
    assert one(gate3(tmp_path, chips=FOUR[:2]), "configured for 4")


def test_a_test_that_ran_on_more_chips_than_configured_is_refused_too(tmp_path):
    assert one(gate3(tmp_path, expected=2, chips=FOUR), "configured for 2")


def test_without_a_configured_count_the_gate_checks_only_that_the_record_and_the_marker_agree(tmp_path):
    stage_dir, run = stage3(tmp_path, chips=FOUR[:2])
    assert gate_full_port_3(stage_dir, run).ok
    stage_dir, run = build(tmp_path / "other", stage=3, record={"chips": FOUR[:2]}, marker={"chips": FOUR})
    assert one(gate_full_port_3(stage_dir, run), "not the chips the supervisor leased")


def test_the_marker_and_the_record_must_both_carry_the_configured_count(tmp_path):
    # A marker that lists four chips next to a record that lists two is the old agreement failure, whatever the count.
    stage_dir, run = build(tmp_path, stage=3, record={"chips": FOUR[:2]}, marker={"chips": FOUR})
    assert one(gate_full_port_3(stage_dir, run, expected_chips=4), "not the chips the supervisor leased")


def test_the_stage_2_command_in_a_stage_3_marker_is_refused(tmp_path):
    g = gate3(tmp_path, marker={"command": full_port.COMMAND})
    assert one(g, full_port.command(3))


def test_a_port_listed_under_stage_2_is_refused_for_stage_3(tmp_path):
    stage_dir, run = stage3(tmp_path)
    # The record names the stage 2 port module: the files this stage ran are not the files it lists.
    rec = full_port._json(stage_dir / "evidence" / "port-check.json", "r", [])
    rec["port_sources"] = [{"path": "stages/2/port/port_model.py", "sha256": sha("import ttnn\n"), "role": "model"}]
    (stage_dir / "evidence" / "port-check.json").write_text(__import__("json").dumps(rec))
    g = gate_full_port_3(stage_dir, run, expected_chips=4)
    assert one(g, "needs a file under stages/3/port/")


def test_the_gate_names_this_stages_port_directory_when_no_source_is_listed(tmp_path):
    g = gate3(tmp_path, record={"port_sources": []})
    assert one(g, "stages/3/port/")


def test_rehearsal_waives_only_the_coherent_field_in_stage_3_as_in_stage_2(tmp_path):
    stage_dir, run = stage3(tmp_path, result={"coherent": False})
    assert one(gate_full_port_3(stage_dir, run, expected_chips=4), "coherent")
    waived = gate_full_port_3(stage_dir, run, expected_chips=4, rehearsal=True)
    assert waived.ok and waived.waived == ("coherent",)
    bad_stage, bad_run = stage3(tmp_path / "b", result={"coherent": False, "top1_agreement": 0.1})
    assert not gate_full_port_3(bad_stage, bad_run, expected_chips=4, rehearsal=True).ok     # every other check still applies


def test_the_stage_2_gate_is_unchanged_by_the_new_arguments(tmp_path):
    stage_dir, run = build(tmp_path)
    assert gate_full_port(stage_dir, run).ok
    assert gate_full_port(stage_dir, run, expected_chips=2).ok         # the stage 2 test of these fixtures ran on two chips
    assert one(gate_full_port(stage_dir, run, expected_chips=4), "configured for 4")


def test_a_stage_other_than_2_or_3_is_not_checked_as_a_full_port_stage(tmp_path):
    stage_dir, run = build(tmp_path)
    reasons, _ = full_port.check(stage_dir, run, stage=4)
    assert reasons and "stages [2, 3]" in reasons[0]


def test_the_stage_helpers_name_each_stages_own_paths():
    assert full_port.port_dir(3) == "stages/3/port/" and full_port.port_module(3) == "stages/3/port/port_model.py"
    assert full_port.command(3) == "python3 -P -s stages/3/port_check.py"
    assert (full_port.port_dir(), full_port.port_module(), full_port.command()) == (full_port.PORT_DIR, full_port.PORT_MODULE, full_port.COMMAND)


def test_the_stage_3_template_is_the_same_shipped_file(tmp_path):
    # One template serves both stages: a stage 3 copy that differs from the shipped file is refused as it is for stage 2.
    stage_dir, run = stage3(tmp_path)
    (stage_dir / full_port.TEMPLATE_NAME).write_text(full_port.TEMPLATE.read_text() + "\n# edited\n")
    assert one(gate_full_port_3(stage_dir, run, expected_chips=4), "differs from the shipped template")


# ---- routing ----------------------------------------------------------------------------------------

def test_with_the_opt_in_a_full_port_class_gets_the_full_port_stage_3():
    s = spec_for(3, "full-port", None, "full-port", full_port=True)
    assert s is FULL_PORT_STAGE_3
    assert s.skill == "full-port-mesh" and s.gate is gate_full_port_3 and s.draft == "full_port" and s.boards == 1 and s.gate_file == "result.json"
    assert s.refs == ("tt-device-usage",) and s.skip is None


def test_without_the_opt_in_stage_3_of_a_full_port_is_the_generic_stage():
    assert spec_for(3, "full-port", None, "full-port") == STAGES[3]
    assert spec_for(3, "full-port", None, "unknown", full_port=True) == STAGES[3]       # `unknown` and `restricted-head` keep their pauses
    assert spec_for(3, "full-port", None, "restricted-head", full_port=True) == STAGES[3]


def test_other_routes_keep_their_stage_3():
    assert spec_for(3, "weights-only", None, "weights-only", full_port=True).skip           # weights-only skips stage 3
    assert spec_for(3, "stock-runtime", None, None, full_port=True).skip


def test_a_scoring_model_of_class_full_port_keeps_the_generic_stage_3():
    # Its own pause comes after stage 2 (`SCORING_STAGES_UNBUILT`); it never reaches the generative stage 3.
    assert spec_for(3, "full-port", None, "full-port", full_port=True, kind="reranker") == STAGES[3]


def test_stage_2_and_stage_3_are_different_specs_with_the_same_template():
    assert FULL_PORT_STAGE_2 != FULL_PORT_STAGE_3 and FULL_PORT_STAGE_2.draft == FULL_PORT_STAGE_3.draft == "full_port"
    assert SKIP_3_FULL_PORT and "required-chips" in SKIP_3_FULL_PORT


# ---- the skill --------------------------------------------------------------------------------------

def skill_text():
    return (SKILLS / "full-port-mesh.md").read_text()


def test_the_skill_names_stage_3_paths_and_the_fixed_command():
    text = skill_text()
    assert full_port.command(3) in text and "stages/3/port/" in text and "stages/3/evidence/port-check.json" in text
    assert full_port.COMMAND not in text                      # it never tells the coder to run stage 2's command


def test_the_skill_says_what_the_gate_requires_of_the_chip_count_and_that_there_are_no_repair_rounds():
    text = skill_text()
    assert "exactly the configured number of chips" in text and "no repair rounds" in text.lower()


def test_the_skill_forbids_editing_stage_2_and_points_at_the_stage_2_skill():
    text = skill_text()
    assert "Do not edit anything under `stages/2/`" in text and "full-port-check.md" in text
