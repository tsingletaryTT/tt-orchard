# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard suite-control` (issue 15): builds the prompt suite's control from the nearest model.

The control is the stage's config with the nearest model as the model served, in its own directory with its own tensor cache.
The prepare step runs the real shipped prepare_swap.py on the same fake snapshots the template tests use. The serve step opens a
chip, so these tests only look at the lease command it would run, and at the result of a fake runner."""
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weights_swap_templates import LINKED, NEAREST, put  # noqa: E402
from test_weights_swap_templates import prep as _prep  # noqa: E402

from orchard import prompt_suite, suite_control as sc  # noqa: E402

CHECKOUT = Path(__file__).resolve().parent.parent


@pytest.fixture
def prep(_prep):
    """The template tests' stage, with the files a real nearest-model snapshot holds (a control is built from it)."""
    for name in LINKED:
        put(_prep["base"], name, f"base {name}")
    (_prep["bundle"] / "tt_kernel_manifest.json").write_text(json.dumps({"device_count": 2, "weights": {"repo_id": NEAREST}}))
    return _prep


def stage_cfg(p):
    return json.loads((p["stage"] / "swap_config.json").read_text())


def build(p, **kw):
    return sc.prepare(p["stage"], orchard_dir=CHECKOUT, **kw)


def ok_run(*a, **k):
    return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()


def test_the_control_serves_the_nearest_model_with_its_own_cache(prep):
    out = build(prep)
    cfg = json.loads((Path(out["control_dir"]) / "swap_config.json").read_text())
    assert cfg["new_model_id"] == NEAREST and cfg["new_snapshot"] == stage_cfg(prep)["base_snapshot"]
    assert cfg["tt_cache"] == stage_cfg(prep)["tt_cache"] + "-control" != stage_cfg(prep)["tt_cache"]
    assert cfg["prompt_suite"] is True and cfg["nearest_model_id"] == NEAREST


def test_the_control_has_no_graft_and_no_control_of_its_own(prep):
    c = stage_cfg(prep)
    (prep["stage"] / "swap_config.json").write_text(json.dumps({**c, "graft_mtp": True, "suite_control": "x.json"}))
    cfg = build(prep)["config"]
    assert "graft_mtp" not in cfg and "suite_control" not in cfg


def test_the_real_prepare_script_builds_the_control_model_dir_from_the_base(prep):
    out = build(prep)
    ctl = Path(out["control_dir"])
    assert json.loads((ctl / "model-dir" / "config.json").read_text()) == {"base": True}
    assert (ctl / "run.sh").is_file()
    assert not (prep["stage"] / "model-dir").exists(), "the stage's own directory must not be touched"


def test_the_stages_own_files_are_left_alone(prep):
    before = (prep["stage"] / "swap_config.json").read_text()
    build(prep)
    assert (prep["stage"] / "swap_config.json").read_text() == before


def test_the_control_passes_the_identity_check_the_gate_and_the_template_apply(prep):
    cfg = build(prep)["config"]
    assert prompt_suite.control_identity_problem(stage_cfg(prep)["nearest_model_id"], cfg["new_model_id"]) is None
    assert prompt_suite.control_identity_problem(stage_cfg(prep)["nearest_model_id"], stage_cfg(prep)["new_model_id"])


def test_a_stage_that_already_serves_the_nearest_model_is_refused(prep):
    c = stage_cfg(prep)
    (prep["stage"] / "swap_config.json").write_text(json.dumps({**c, "new_model_id": NEAREST + "@abc"}))
    with pytest.raises(sc.ControlError, match="already serves the nearest"):
        build(prep)
    (prep["stage"] / "swap_config.json").write_text(json.dumps({**c, "new_snapshot": c["base_snapshot"]}))
    with pytest.raises(sc.ControlError, match="already serves the nearest"):
        build(prep)


@pytest.mark.parametrize("key", sc.REQUIRED)
def test_a_missing_key_is_named(prep, key):
    c = stage_cfg(prep)
    c.pop(key)
    (prep["stage"] / "swap_config.json").write_text(json.dumps(c))
    with pytest.raises(sc.ControlError, match=key):
        build(prep)


def test_a_config_with_no_bundle_or_package_is_refused(prep):
    c = stage_cfg(prep)
    c.pop("bundle_dir")
    (prep["stage"] / "swap_config.json").write_text(json.dumps(c))
    with pytest.raises(sc.ControlError, match="neither a bundle_dir nor a package"):
        build(prep, run=ok_run)      # the refusal is this command's own, not prepare_swap.py's


def test_an_unreadable_or_non_object_config_is_refused(prep):
    (prep["stage"] / "swap_config.json").write_text("{nope")
    with pytest.raises(sc.ControlError, match="cannot read"):
        build(prep)
    (prep["stage"] / "swap_config.json").write_text("[1]")
    with pytest.raises(sc.ControlError, match="not a JSON object"):
        build(prep)


def test_a_container_package_uses_the_container_serve_script(prep):
    c = stage_cfg(prep)
    c.pop("bundle_dir")
    c.update(package="org/pkg", profile="p", chips=4, operator_home="/x")
    (prep["stage"] / "swap_config.json").write_text(json.dumps(c))
    assert sc.scripts_for(c) == ("prepare_swap.py", "serve_and_compare_container.py")
    assert sc.scripts_for({"bundle_dir": "b"}) == ("prepare_swap.py", "serve_and_compare.py")
    calls = []
    out = sc.prepare(prep["stage"], orchard_dir=CHECKOUT, run=lambda argv, **k: calls.append(argv) or ok_run())
    assert out["serve"][-1].endswith("serve_and_compare_container.py") and out["chips"] == 4
    assert (Path(out["control_dir"]) / "serve_and_compare_container.py").is_file()
    assert calls and calls[0][1].endswith("prepare_swap.py")


def test_the_shipped_files_are_copied_from_the_checkout_not_the_stage(prep, tmp_path):
    fake = tmp_path / "fake"
    t = fake / "orchard" / "skills" / "weights-swap-templates"
    t.mkdir(parents=True)
    for n in ("prepare_swap.py", "serve_and_compare.py", "prompt_suite.json"):
        (t / n).write_text("# from the checkout\n")
    (prep["stage"] / "serve_and_compare.py").write_text("# an agent's edited copy\n")
    out = sc.prepare(prep["stage"], orchard_dir=fake, run=ok_run)
    assert (Path(out["control_dir"]) / "serve_and_compare.py").read_text() == "# from the checkout\n"
    assert (Path(out["control_dir"]) / "prompt_suite.json").is_file()


def test_a_checkout_missing_a_template_is_named(prep, tmp_path):
    fake = tmp_path / "fake"
    (fake / "orchard" / "skills" / "weights-swap-templates").mkdir(parents=True)
    with pytest.raises(sc.ControlError, match="prompt_suite.json"):
        sc.prepare(prep["stage"], orchard_dir=fake)


def test_a_failing_prepare_script_is_reported_with_its_output(prep):
    run = lambda *a, **k: type("R", (), {"returncode": 2, "stdout": "", "stderr": "prepare_swap: boom"})()  # noqa: E731
    with pytest.raises(sc.ControlError, match="exited with code 2.*boom"):
        build(prep, run=run)


def test_an_earlier_control_directory_is_cleared_first(prep):
    d = prep["stage"] / sc.CONTROL_DIR
    (d / "evidence").mkdir(parents=True)
    (d / "evidence" / "swap-check.json").write_text("old")
    out = build(prep)
    assert not Path(out["evidence"]).exists()


def test_port_and_chips_can_be_set(prep):
    out = build(prep, port=8150, chips=2)
    assert out["config"]["port"] == 8150 and out["chips"] == 2
    again = build(prep)
    assert again["config"]["port"] == stage_cfg(prep)["port"] and again["chips"] == 2


def test_wire_value_is_relative_to_the_run_directory(prep):
    assert build(prep)["wire"] == "stages/2/suite-control/evidence/swap-check.json"


def test_the_lease_command_is_one_gozer_run_with_the_serve_script(prep):
    out = build(prep)
    argv = sc.gozer_argv(out, who="claude:suite-control", reason="why")
    assert argv[:4] == ["gozer", "run", "--chips", "2"]
    assert argv[argv.index("--who") + 1] == "claude:suite-control" and argv[argv.index("--reason") + 1] == "why"
    assert argv[argv.index("--") + 1:] == out["serve"]
    assert "reset" not in argv and "release" not in argv


def run_main(prep, *args, call=None):
    out = io.StringIO()
    code = sc.main(["--stage-dir", str(prep["stage"]), "--orchard-dir", str(CHECKOUT), *args], out=out,
                   call=call or (lambda *a, **k: pytest.fail("a chip command ran")))
    return code, out.getvalue()


def test_prepare_prints_the_command_and_runs_nothing_on_a_chip(prep):
    code, text = run_main(prep, "prepare")
    assert code == 0
    printed = json.loads(text)
    assert printed["gozer"][:2] == ["gozer", "run"] and printed["suite_control"].endswith("swap-check.json")


def test_run_calls_gozer_in_the_control_directory_and_returns_its_status(prep):
    seen = {}

    def call(argv, cwd=None):
        seen.update(argv=argv, cwd=cwd)
        return 0
    code, _ = run_main(prep, "run", call=call)
    assert code == 0 and seen["argv"][:2] == ["gozer", "run"] and seen["cwd"].endswith("suite-control")
    assert run_main(prep, "run", call=lambda *a, **k: 7)[0] == 7


def test_a_queued_request_exits_3_and_says_to_wait(prep, capsys):
    code, _ = run_main(prep, "run", call=lambda *a, **k: 10)
    assert code == 3 and "gozer wait" in capsys.readouterr().err


def test_a_refusal_exits_2_without_calling_gozer(prep, capsys):
    (prep["stage"] / "swap_config.json").write_text("{nope")
    code, _ = run_main(prep, "run")
    assert code == 2 and "suite-control:" in capsys.readouterr().err


def test_the_cli_routes_suite_control(prep, capsys):
    from orchard import cli
    assert cli.main(["suite-control", "prepare", "--stage-dir", str(prep["stage"])]) == 0
    assert "suite_control" in capsys.readouterr().out


def test_the_module_runs_as_a_script(prep):
    r = subprocess.run([sys.executable, "-m", "orchard.suite_control", "prepare", "--stage-dir", str(prep["stage"])],
                       capture_output=True, text=True, timeout=120, cwd=str(CHECKOUT))
    assert r.returncode == 0, r.stderr




def stage_run_sh(prep, drafter_line):
    (prep["stage"] / "run.sh").write_text("#!/bin/bash\n" + drafter_line + "\nexec true\n")


def test_a_stage_served_with_the_drafter_off_gets_a_control_with_the_drafter_off(prep):
    stage_run_sh(prep, 'export QWEN36_DRAFTER=""')
    out = build(prep)
    assert out["config"]["drafter"] == "off"
    assert 'export QWEN36_DRAFTER=""' in (Path(out["control_dir"]) / "run.sh").read_text()
    assert out["mirrors_drafter_off"] is True


def test_a_stage_served_with_the_drafter_on_gets_no_drafter_key(prep):
    stage_run_sh(prep, 'export QWEN36_DRAFTER="dflash2"')
    out = build(prep)
    assert "drafter" not in out["config"] and out["mirrors_drafter_off"] is False


def test_a_stage_with_no_run_script_gets_no_drafter_key(prep):
    assert "drafter" not in build(prep)["config"]


def test_the_last_drafter_line_of_the_stage_script_decides(prep):
    stage_run_sh(prep, 'export QWEN36_DRAFTER=""\nexport QWEN36_DRAFTER="dflash2"')
    assert "drafter" not in build(prep)["config"]
    stage_run_sh(prep, '# export QWEN36_DRAFTER=""')
    assert "drafter" not in build(prep)["config"]
