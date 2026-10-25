# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""An agent-written swap config must name stage 0's nearest model (issue 67).

When the supervisor drafted no config for a fine-tune of Qwen3-Coder-30B-A3B-Instruct, the stage 2 agent wrote one that served the Qwen3.8-27B bundle and
named `Qwen/Qwen3.8-27B` as the nearest model. The server failed with `No TT model architecture is registered`, after a coder boot, a lease and about
15 minutes of chip time. The check runs where the supervisor reads the agent's files (`_read_test` for stage 2, the plan for stage 4), before any hardware
is leased, and its message goes back to the agent as the reason the step is not finished."""
import json

from orchard import swap_draft as sd
from orchard.supervisor import Supervisor


def write_delta(run, nearest="Qwen/Qwen3-Coder-30B-A3B-Instruct"):
    d = run / "stages" / "0"
    d.mkdir(parents=True, exist_ok=True)
    (d / "delta.json").write_text(json.dumps({"model": "a/new@" + "0" * 40, "nearest_model": nearest}))


def write_cfg(path, nearest):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"nearest_model_id": nearest}))
    return path


def test_a_config_that_names_another_model_is_a_problem(tmp_path):
    write_delta(tmp_path)
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3.8-27B")
    msg = sd.nearest_problem(tmp_path, p)
    assert msg and "Qwen/Qwen3.8-27B" in msg and "Qwen/Qwen3-Coder-30B-A3B-Instruct" in msg


def test_the_matching_model_with_or_without_a_revision_is_fine(tmp_path):
    write_delta(tmp_path, "Qwen/Qwen3-Coder-30B-A3B-Instruct@" + "a" * 40)
    for named in ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen/Qwen3-Coder-30B-A3B-Instruct@" + "b" * 40):
        p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", named)
        assert sd.nearest_problem(tmp_path, p) is None


def test_a_similar_name_is_not_a_match(tmp_path):
    write_delta(tmp_path)
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    assert sd.nearest_problem(tmp_path, p)


def test_a_missing_config_or_missing_stage_0_is_no_problem_here(tmp_path):
    assert sd.nearest_problem(tmp_path, tmp_path / "none.json") is None          # other checks say the file is missing
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "x/y")
    assert sd.nearest_problem(tmp_path, p) is None                                # no stage 0 record to compare with


def test_a_config_with_no_nearest_model_key_is_left_to_prepare_swap(tmp_path):
    write_delta(tmp_path)
    p = tmp_path / "stages" / "2" / "swap_config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"bundle_dir": "/x"}))
    assert sd.nearest_problem(tmp_path, p) is None


def test_stage_2_refuses_the_test_before_any_lease(tmp_path):
    """_read_test is what the supervisor and the step's `finished` check both read."""
    write_delta(tmp_path)
    stage = tmp_path / "stages" / "2"
    write_cfg(stage / "swap_config.json", "Qwen/Qwen3.8-27B")
    (stage / "hw_test.json").write_text(json.dumps({"command": "true", "deadline_s": 60}))
    (stage / "handoff.json").write_text(json.dumps({k: "x" for k in __import__("orchard.supervisor", fromlist=["NOTE_KEYS"]).NOTE_KEYS}))
    sup = Supervisor.__new__(Supervisor)
    sup.run_dir = tmp_path
    test, problems = sup._read_test(stage)
    assert any("Qwen/Qwen3.8-27B" in p and "nearest" in p for p in problems), problems


def test_stage_4_refuses_a_configuration_that_names_another_model(tmp_path):
    write_delta(tmp_path)
    stage = tmp_path / "stages" / "4"
    write_cfg(stage / "configs" / "2" / "swap_config.json", "Qwen/Qwen3.8-27B")
    write_cfg(stage / "configs" / "4" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    problems = sd.stage4_nearest_problems(tmp_path, stage)
    assert len(problems) == 1 and "2-chip" in problems[0]


def test_a_name_that_is_part_of_the_nearest_model_is_not_a_match(tmp_path):
    write_delta(tmp_path)
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder")
    assert sd.nearest_problem(tmp_path, p)


def test_every_stage_4_configuration_is_checked(tmp_path):
    write_delta(tmp_path)
    stage = tmp_path / "stages" / "4"
    for n in (1, 2, 4):
        write_cfg(stage / "configs" / str(n) / "swap_config.json", "Qwen/Qwen3.8-27B")
    assert len(sd.stage4_nearest_problems(tmp_path, stage)) == 3


def test_the_stage_4_plan_the_supervisor_reads_carries_the_problem(tmp_path, monkeypatch):
    import orchard.supervisor as sup_mod
    write_delta(tmp_path)
    stage = tmp_path / "stages" / "4"
    write_cfg(stage / "configs" / "2" / "swap_config.json", "Qwen/Qwen3.8-27B")
    monkeypatch.setattr(sup_mod, "read_plan", lambda *a, **k: ([], []))
    sup = Supervisor.__new__(Supervisor)
    sup.run_dir, sup.required_chips, sup.home = tmp_path, None, tmp_path
    spec = type("S", (), {"boards": 2, "budget_s": 60})()
    _, problems = sup._read_plan(spec, stage)
    assert len(problems) == 1 and "2-chip" in problems[0]
