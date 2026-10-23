# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A control has to be the nearest model (PR 49 review, finding 1).

The gate used to reject a control only when its model id equalled the new model's. A same-suite control from an unrelated model
could then explain a blocking finding away, because it showed the same failure on that prompt. The template now checks the
control's repository (and its pinned revision when the config pins one) against `nearest_model_id` before any server starts,
and the gates check the record they read against stage 0's `nearest_model`, which an agent does not write."""
import json
import shutil

import pytest

pytest.importorskip("tokenizers")

from orchard import prompt_suite as ps  # noqa: E402
from orchard.stages import gate_mesh_swap, gate_weights_swap  # noqa: E402
from test_prompt_suite_gate import swap_check  # noqa: E402
from test_prompt_suite_template import control_file, run, stage_for_suite  # noqa: E402,F401  (a fixture: swap)
from test_prompt_suite_verdict import LOOP, control_of, make_record, with_entry  # noqa: E402
from test_stages import GOOD_DELTA, SWAP, mesh_stage, stage, swap_config, write  # noqa: E402
from test_weights_swap_templates import NEAREST, TEMPLATES, load_swap_template, swap  # noqa: E402,F401  (a fixture)

REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
OTHER_REV = "0" * 40


def looping(control_model_id):
    """A record whose arithmetic reply loops, with a control that loops the same way and was served from `control_model_id`."""
    rec = with_entry(make_record(), "arithmetic", content=LOOP)
    ctl = control_of(rec)
    ctl["record"]["model_id"] = control_model_id
    rec["control"] = ctl
    return rec


# ---- the rule, in orchard/prompt_suite.py ----------------------------------------------------------------------------

@pytest.mark.parametrize("expected, actual", [
    ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
    ("Qwen/Qwen3.8-27B", f"Qwen/Qwen3.8-27B@{REV}"),            # the config pins nothing: any revision of that repository
    (f"Qwen/Qwen3.8-27B@{REV}", f"Qwen/Qwen3.8-27B@{REV}"),
])
def test_the_nearest_model_is_a_control_of_itself(expected, actual):
    assert ps.control_identity_problem(expected, actual) is None


@pytest.mark.parametrize("expected, actual, word", [
    ("Qwen/Qwen3.8-27B", "Other/Model", "Other/Model"),
    ("Qwen/Qwen3.8-27B", "qwen/qwen3.8-27b", "qwen/qwen3.8-27b"),         # repository ids are case sensitive on disk
    (f"Qwen/Qwen3.8-27B@{REV}", f"Qwen/Qwen3.8-27B@{OTHER_REV}", "revision"),
    (f"Qwen/Qwen3.8-27B@{REV}", "Qwen/Qwen3.8-27B", "revision"),        # a pin the control does not carry cannot be checked
    ("Qwen/Qwen3.8-27B", None, "no model id"),
    ("Qwen/Qwen3.8-27B", "", "no model id"),
    ("Qwen/Qwen3.8-27B", 7, "no model id"),
    (None, "Qwen/Qwen3.8-27B", "cannot be checked"),                     # the run does not say which model is the nearest one
    ("", "Qwen/Qwen3.8-27B", "cannot be checked"),
])
def test_anything_else_is_not_a_control_and_the_problem_says_why(expected, actual, word):
    assert word in (ps.control_identity_problem(expected, actual) or "")


def test_a_branch_after_an_at_sign_that_is_not_a_revision_is_part_of_the_repository():
    assert ps.control_identity_problem("org/name@main", "org/name") is not None
    assert ps.control_identity_problem("org/name@main", "org/name@main") is None
    assert ps.control_identity_problem("org/name", "org/name@main") is not None      # another repository id, not a pin on this one


def test_an_unrelated_control_does_not_explain_a_blocking_finding_away():
    v = ps.evaluate({"prompt_suite": looping("Other/Model")}, nearest_model="Qwen/Qwen3.8-27B")
    assert any(r.startswith("control:") and "Other/Model" in r for r in v.reasons), v.reasons
    assert any("arithmetic" in r and "mechanical_repetition" in r for r in v.reasons), v.reasons      # the finding stands


def test_the_nearest_models_control_still_explains_a_shared_finding():
    v = ps.evaluate({"prompt_suite": looping("Qwen/Qwen3.8-27B")}, nearest_model="Qwen/Qwen3.8-27B")
    assert v.reasons == []


def test_without_a_stated_nearest_model_the_old_behaviour_is_unchanged():
    assert ps.evaluate({"prompt_suite": looping("Other/Model")}).reasons == []


def test_a_nearest_model_that_is_not_known_leaves_a_control_unverified_and_the_finding_stands():
    v = ps.evaluate({"prompt_suite": looping("Qwen/Qwen3.8-27B")}, nearest_model=None)
    assert any(r.startswith("control:") and "cannot be checked" in r for r in v.reasons), v.reasons
    assert any("arithmetic" in r for r in v.reasons)


def test_a_run_without_a_control_does_not_need_to_know_the_nearest_model():
    assert ps.evaluate({"prompt_suite": make_record()}, nearest_model=None).reasons == []


def test_a_control_with_no_model_id_is_not_verified():
    rec = looping("Qwen/Qwen3.8-27B")
    del rec["control"]["record"]["model_id"]
    assert any("no model id" in r for r in ps.evaluate({"prompt_suite": rec}, nearest_model="Qwen/Qwen3.8-27B").reasons)


def test_the_command_line_takes_the_nearest_model(tmp_path, capsys):
    path = tmp_path / "swap-check.json"
    path.write_text(json.dumps({"prompt_suite": looping("Other/Model")}))
    assert ps.main([str(path)]) == 0
    assert ps.main([str(path), "--nearest-model", "Qwen/Qwen3.8-27B"]) == 1
    assert "Other/Model" in capsys.readouterr().out


# ---- the template, before any server starts --------------------------------------------------------------------------

def test_the_template_refuses_a_control_of_another_model_before_any_server_starts(swap, tmp_path):
    stage_for_suite(swap)
    path = control_file(swap, tmp_path, model_id="Other/Model")
    shutil.rmtree(swap["stage"] / "evidence", ignore_errors=True)
    swap["pid_file"].unlink(missing_ok=True)
    r = run(swap, tmp_path, suite_control=str(path))
    assert r.returncode == 2 and "Other/Model" in r.stdout + r.stderr and "nearest" in r.stdout + r.stderr
    assert not swap["pid_file"].exists()


def test_the_template_accepts_a_control_of_the_nearest_model(swap, tmp_path):
    path = control_file(swap, tmp_path, model_id=NEAREST)
    assert run(swap, tmp_path, suite_control=str(path)).returncode == 0


def test_the_template_checks_the_pinned_revision_the_config_carries(swap, tmp_path):
    path = control_file(swap, tmp_path, model_id=f"{NEAREST}@{OTHER_REV}")
    r = run(swap, tmp_path, suite_control=str(path), nearest_model_id=f"{NEAREST}@{REV}")
    assert r.returncode == 2 and "revision" in r.stdout + r.stderr


def test_the_template_refuses_a_control_when_the_config_names_no_nearest_model(swap, tmp_path):
    path = control_file(swap, tmp_path, model_id=NEAREST)
    r = run(swap, tmp_path, suite_control=str(path), nearest_model_id="")
    assert r.returncode == 2 and "cannot be checked" in r.stdout + r.stderr


def test_the_template_and_the_gate_apply_the_same_rule():
    template = load_swap_template()
    cases = [("Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B"), ("Qwen/Qwen3.8-27B", f"Qwen/Qwen3.8-27B@{REV}"), ("Qwen/Qwen3.8-27B", "Other/Model"),
             (f"Qwen/Qwen3.8-27B@{REV}", f"Qwen/Qwen3.8-27B@{OTHER_REV}"), (f"Qwen/Qwen3.8-27B@{REV}", "Qwen/Qwen3.8-27B"),
             ("Qwen/Qwen3.8-27B", None), (None, "Qwen/Qwen3.8-27B"), ("org/name@main", "org/name"), ("org/name@main", "org/name@main"),
             ("org/name", "org/name@main")]
    for expected, actual in cases:
        assert (template.control_identity_problem(expected, actual) is None) == (ps.control_identity_problem(expected, actual) is None), \
            (expected, actual)


# ---- the gates, from stage 0's nearest model -------------------------------------------------------------------------

def stage2(tmp_path, record, nearest="Qwen/Qwen3.8-27B", gate=gate_weights_swap):
    sd, run_dir = stage(tmp_path, 2, "result.json", SWAP)
    write(run_dir, "stages/2/evidence/swap-check.json", swap_check(record))
    if nearest is not None:
        write(run_dir, "stages/0/delta.json", {**GOOD_DELTA, "nearest_model": nearest})
    return gate(sd, run_dir)


def test_the_stage_2_gate_fails_a_finding_an_unrelated_control_would_have_explained(tmp_path):
    g = stage2(tmp_path, looping("Other/Model"))
    assert not g.ok
    assert any("control" in r and "Other/Model" in r for r in g.reasons) and any("arithmetic" in r for r in g.reasons)


def test_the_stage_2_gate_accepts_the_nearest_models_control(tmp_path):
    g = stage2(tmp_path, looping("Qwen/Qwen3.8-27B"))
    assert g.ok, g.reasons


def test_the_stage_2_gate_takes_the_pinned_revision_from_stage_0(tmp_path):
    ok = stage2(tmp_path / "a", looping(f"Qwen/Qwen3.8-27B@{REV}"), nearest=f"Qwen/Qwen3.8-27B@{REV}")
    bad = stage2(tmp_path / "b", looping(f"Qwen/Qwen3.8-27B@{OTHER_REV}"), nearest=f"Qwen/Qwen3.8-27B@{REV}")
    assert ok.ok, ok.reasons
    assert not bad.ok and any("revision" in r for r in bad.reasons)


def test_a_stage_0_that_names_no_nearest_model_leaves_the_control_unverified(tmp_path):
    g = stage2(tmp_path, looping("Qwen/Qwen3.8-27B"), nearest=None)
    assert not g.ok and any("cannot be checked" in r for r in g.reasons)


def test_a_swap_config_the_agent_wrote_does_not_decide_which_model_is_the_nearest_one(tmp_path):
    # The agent can write any nearest_model_id into swap_config.json; the gate asks stage 0.
    sd, run_dir = stage(tmp_path, 2, "result.json", SWAP)
    write(run_dir, "stages/2/swap_config.json", {"nearest_model_id": "Other/Model"})
    write(run_dir, "stages/2/evidence/swap-check.json", swap_check(looping("Other/Model")))
    write(run_dir, "stages/0/delta.json", {**GOOD_DELTA, "nearest_model": "Qwen/Qwen3.8-27B"})
    assert not gate_weights_swap(sd, run_dir).ok


def test_the_stage_4_gate_checks_each_configurations_control_too(tmp_path):
    sd, run_dir = mesh_stage(tmp_path, [swap_config(2), swap_config(4)])
    for n in (2, 4):
        report = swap_check(looping("Other/Model" if n == 4 else "Qwen/Qwen3.8-27B"))
        report["result_draft"] = {"top1_agreement": swap_config(n)["top1_agreement"]}
        write(run_dir, f"stages/4/configs/{n}/evidence/swap-check.json", report)
    write(run_dir, "stages/0/delta.json", {**GOOD_DELTA, "nearest_model": "Qwen/Qwen3.8-27B"})
    g = gate_mesh_swap(sd, run_dir, required=(2, 4))
    assert not g.ok
    assert any(r.startswith("the 4-chip configuration: ") and "control" in r and "Other/Model" in r for r in g.reasons), g.reasons
    assert not any(r.startswith("the 2-chip configuration: ") and "control" in r for r in g.reasons), g.reasons
