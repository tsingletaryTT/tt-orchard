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
    sup.paths = type("P", (), {"tt_model_root": tmp_path / "models", "operator_home": tmp_path / "op"})()
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
    monkeypatch.setattr(sup_mod, "read_plan", lambda *a, **k: ([type("T", (), {"chips": 2})()], []))
    sup = Supervisor.__new__(Supervisor)
    sup.run_dir, sup.required_chips, sup.home = tmp_path, None, tmp_path
    sup.paths = type("P", (), {"tt_model_root": tmp_path / "models", "operator_home": tmp_path / "op"})()
    spec = type("S", (), {"boards": 2, "budget_s": 60})()
    _, problems = sup._read_plan(spec, stage)
    assert len(problems) == 1 and "2-chip" in problems[0]


# ---- the package or bundle itself must serve stage 0's nearest model (review of PR 70) ------------------------------------

def write_installed(tmp_path, name, repo, container=True):
    """An installed container package (installed.json plus its manifest) or a v6 bundle under a models root."""
    home = tmp_path / "op"
    if container:
        mf = home / ".cache" / "tt-model" / "pulled" / name.replace("/", "__") / "tt_kernel_manifest.json"
        mf.parent.mkdir(parents=True, exist_ok=True)
        mf.write_text(json.dumps({"schema_version": "5.1", "device_count": 4, "weights": {"repo_id": repo}}))
        inst = home / ".cache" / "tt-model" / "installed.json"
        inst.write_text(json.dumps({name: {"container": True, "image": "x:y", "profile": "default", "manifest": str(mf)}}))
        return home
    root = tmp_path / "models"
    b = root / name
    b.mkdir(parents=True, exist_ok=True)
    (b / "tt_kernel_manifest.json").write_text(json.dumps({"schema_version": "6", "device_count": 2, "weights": {"repo_id": repo}}))
    return root


def test_a_container_package_that_serves_another_repository_is_a_problem_even_with_the_right_label(tmp_path):
    write_delta(tmp_path)
    home = write_installed(tmp_path, "x/qwen38-p300x2", "Qwen/Qwen3.8-27B")
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["package"] = "x/qwen38-p300x2"; p.write_text(json.dumps(cfg))
    msg = sd.nearest_problem(tmp_path, p, operator_home=home)
    assert msg and "x/qwen38-p300x2" in msg and "Qwen/Qwen3.8-27B" in msg


def test_a_container_package_that_serves_the_nearest_model_is_fine(tmp_path):
    write_delta(tmp_path)
    home = write_installed(tmp_path, "x/coder-p300x2", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["package"] = "x/coder-p300x2"; p.write_text(json.dumps(cfg))
    assert sd.nearest_problem(tmp_path, p, operator_home=home) is None


def test_a_bundle_directory_that_serves_another_repository_is_a_problem(tmp_path):
    write_delta(tmp_path)
    root = write_installed(tmp_path, "org/q38", "Qwen/Qwen3.8-27B", container=False)
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["bundle_dir"] = str(root / "org" / "q38"); p.write_text(json.dumps(cfg))
    msg = sd.nearest_problem(tmp_path, p, tt_model_root=root)
    assert msg and "Qwen/Qwen3.8-27B" in msg and "org/q38" in msg


def test_a_package_the_check_cannot_find_is_left_to_the_template(tmp_path):
    write_delta(tmp_path)
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["package"] = "nobody/none"; p.write_text(json.dumps(cfg))
    assert sd.nearest_problem(tmp_path, p, operator_home=tmp_path / "empty") is None


def test_the_supervisor_passes_its_roots_to_the_check(tmp_path):
    write_delta(tmp_path)
    home = write_installed(tmp_path, "x/qwen38-p300x2", "Qwen/Qwen3.8-27B")
    stage = tmp_path / "stages" / "2"
    p = write_cfg(stage / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["package"] = "x/qwen38-p300x2"; p.write_text(json.dumps(cfg))
    (stage / "hw_test.json").write_text(json.dumps({"command": "true", "deadline_s": 60}))
    from orchard.supervisor import NOTE_KEYS
    (stage / "handoff.json").write_text(json.dumps({k: "x" for k in NOTE_KEYS}))
    sup = Supervisor.__new__(Supervisor)
    sup.run_dir = tmp_path
    sup.paths = type("P", (), {"tt_model_root": tmp_path / "models", "operator_home": home})()
    _, problems = sup._read_test(stage)
    assert any("x/qwen38-p300x2" in m for m in problems), problems


def test_the_named_package_is_the_one_checked_when_several_are_installed(tmp_path):
    write_delta(tmp_path)
    home = write_installed(tmp_path, "x/coder-p300x2", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    inst = home / ".cache" / "tt-model" / "installed.json"
    data = json.loads(inst.read_text())
    mf2 = home / ".cache" / "tt-model" / "pulled" / "y__qwen38" / "tt_kernel_manifest.json"
    mf2.parent.mkdir(parents=True)
    mf2.write_text(json.dumps({"schema_version": "5.1", "device_count": 4, "weights": {"repo_id": "Qwen/Qwen3.8-27B"}}))
    data["y/qwen38"] = {"container": True, "image": "x:y", "profile": "default", "manifest": str(mf2)}
    inst.write_text(json.dumps(data))
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["package"] = "y/qwen38"; p.write_text(json.dumps(cfg))
    assert sd.nearest_problem(tmp_path, p, operator_home=home)               # the second package serves the wrong model
    cfg["package"] = "x/coder-p300x2"; p.write_text(json.dumps(cfg))
    assert sd.nearest_problem(tmp_path, p, operator_home=home) is None


def test_a_config_that_names_a_different_operator_home_is_a_problem(tmp_path):
    """Review of PR 70: the container template reads installed.json and runs tt-model under cfg["operator_home"]. A different home can hold a
    same-named package that serves another repository, so a config whose home differs from the run's is refused."""
    write_delta(tmp_path)
    home = write_installed(tmp_path, "x/coder-p300x2", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    other = tmp_path / "other-home"
    other.mkdir()
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["package"] = "x/coder-p300x2"; cfg["operator_home"] = str(other); p.write_text(json.dumps(cfg))
    msg = sd.nearest_problem(tmp_path, p, operator_home=home)
    assert msg and "operator_home" in msg


def test_the_run_s_own_operator_home_is_accepted_even_when_spelled_through_a_link(tmp_path):
    write_delta(tmp_path)
    home = write_installed(tmp_path, "x/coder-p300x2", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    link = tmp_path / "home-link"
    link.symlink_to(home)
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["package"] = "x/coder-p300x2"; cfg["operator_home"] = str(link); p.write_text(json.dumps(cfg))
    assert sd.nearest_problem(tmp_path, p, operator_home=home) is None


def test_a_bundle_config_is_not_held_to_an_operator_home(tmp_path):
    write_delta(tmp_path)
    root = write_installed(tmp_path, "org/q", "Qwen/Qwen3-Coder-30B-A3B-Instruct", container=False)
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["bundle_dir"] = str(root / "org" / "q"); cfg["operator_home"] = "/anywhere"; p.write_text(json.dumps(cfg))
    assert sd.nearest_problem(tmp_path, p, tt_model_root=root, operator_home=tmp_path / "op") is None


# ---- review of PR 70, third round ---------------------------------------------------------------------------------------

def test_a_relative_bundle_dir_is_read_from_the_run_directory(tmp_path):
    """The hardware command runs from the run directory, so a relative bundle_dir means a path under it."""
    write_delta(tmp_path)
    b = tmp_path / "bundles" / "q38"
    b.mkdir(parents=True)
    (b / "tt_kernel_manifest.json").write_text(json.dumps({"schema_version": "6", "weights": {"repo_id": "Qwen/Qwen3.8-27B"}}))
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["bundle_dir"] = "bundles/q38"; p.write_text(json.dumps(cfg))
    msg = sd.nearest_problem(tmp_path, p)
    assert msg and "Qwen/Qwen3.8-27B" in msg


def test_a_resumed_stage_4_checks_its_saved_plan_before_any_hardware(tmp_path):
    """_run_test_list loaded tests/plan.json on a resume and never read the configurations again."""
    import orchard.supervisor as sup_mod
    write_delta(tmp_path)
    stage = tmp_path / "stages" / "4"
    write_cfg(stage / "configs" / "4" / "swap_config.json", "Qwen/Qwen3.8-27B")
    sup = Supervisor.__new__(Supervisor)
    sup.run_dir = tmp_path
    sup.paths = type("P", (), {"tt_model_root": tmp_path / "models", "operator_home": tmp_path / "op"})()
    sup.lab = None
    spec = type("S", (), {"number": 4})()
    sup_mod.load_plan = lambda d: [type("T", (), {"chips": 4})()]       # a saved plan with a 4-chip test exists
    try:
        status, reasons, gate = sup._run_test_list(spec, stage, False, True)
    finally:
        import importlib
        importlib.reload(sup_mod)
    assert status == "fail" and any("4-chip" in r and "Qwen/Qwen3.8-27B" in r for r in reasons), (status, reasons)


# ---- review of PR 70, fourth round --------------------------------------------------------------------------------------

def test_a_config_naming_both_a_package_and_a_bundle_is_a_problem(tmp_path):
    """prepare_swap.py prefers the bundle, so a matching package could hide another model's bundle."""
    write_delta(tmp_path)
    home = write_installed(tmp_path, "x/coder-p300x2", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["package"] = "x/coder-p300x2"; cfg["bundle_dir"] = "/some/bundle"; p.write_text(json.dumps(cfg))
    msg = sd.nearest_problem(tmp_path, p, operator_home=home)
    assert msg and "package" in msg and "bundle_dir" in msg and "both" in msg


def test_a_relative_operator_home_is_read_from_the_run_directory(tmp_path):
    """The container test runs from the run directory, so a relative operator_home means a path under it."""
    write_delta(tmp_path)
    home = tmp_path / "homes" / "op"
    write_installed(tmp_path / "homes", "x/coder-p300x2", "Qwen/Qwen3-Coder-30B-A3B-Instruct")        # creates homes/op
    p = write_cfg(tmp_path / "stages" / "2" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    cfg = json.loads(p.read_text()); cfg["package"] = "x/coder-p300x2"; cfg["operator_home"] = "homes/op"; p.write_text(json.dumps(cfg))
    assert sd.nearest_problem(tmp_path, p, operator_home=home) is None                            # the same place, spelled from the run directory
    cfg["operator_home"] = "homes/elsewhere"; p.write_text(json.dumps(cfg))
    assert sd.nearest_problem(tmp_path, p, operator_home=home)


def test_a_configuration_that_is_not_in_the_plan_is_not_judged(tmp_path):
    """Review of PR 70: an optional configuration dropped from hw_tests.json, its directory left behind, must not reject the plan."""
    write_delta(tmp_path)
    stage = tmp_path / "stages" / "4"
    write_cfg(stage / "configs" / "2" / "swap_config.json", "Qwen/Qwen3.8-27B")                 # left over, unused
    write_cfg(stage / "configs" / "4" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    assert sd.stage4_nearest_problems(tmp_path, stage, counts=[4]) == []
    assert len(sd.stage4_nearest_problems(tmp_path, stage, counts=[2, 4])) == 1
    assert len(sd.stage4_nearest_problems(tmp_path, stage)) == 1                                 # no plan given: every directory, as before


def test_the_plan_the_supervisor_reads_judges_only_its_own_counts(tmp_path, monkeypatch):
    import orchard.supervisor as sup_mod
    write_delta(tmp_path)
    stage = tmp_path / "stages" / "4"
    write_cfg(stage / "configs" / "2" / "swap_config.json", "Qwen/Qwen3.8-27B")
    write_cfg(stage / "configs" / "4" / "swap_config.json", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
    monkeypatch.setattr(sup_mod, "read_plan", lambda *a, **k: ([type("T", (), {"chips": 4})()], []))
    sup = Supervisor.__new__(Supervisor)
    sup.run_dir, sup.required_chips, sup.home = tmp_path, None, tmp_path
    sup.paths = type("P", (), {"tt_model_root": tmp_path / "models", "operator_home": tmp_path / "op"})()
    _, problems = sup._read_plan(type("S", (), {"boards": 2, "budget_s": 60})(), stage)
    assert problems == []
