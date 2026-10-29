# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Staged construction in the kernel probe (orchard/kernelstaged.py, issue 176).

A fake coder (a FakeModel that writes scripted kernel.py files) and a fake `run_watched` that answers each stage's
authoritative check from a table stand in for the model and the simulator, so no test needs a chip, a model or the
simulator. The `needs_sim` test runs the real simulator on one fixture kernel per stage and skips, saying so, when the
ttlang venv is absent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from fake_model import FakeModel, call, final, turn
from orchard import kernelprobe as kp, kernelstaged as ks
from test_kernelprobe import (FACTS, VENV, SRC, needs_sim, fake_kenv, kenv, env_ready, probe_args,  # noqa: F401
                              ttlang_skill, skills_current, gozer_tripwire)

PASS = {"passed": True, "category": None, "first_line": "", "sets": []}


def failing(category="numeric_mismatch", line="set seed0: pcc 0.01"):
    return {"passed": False, "category": category, "first_line": line, "sets": []}


def stage_number(request) -> int:
    first_user = next(m for m in request["messages"] if m["role"] == "user")["content"]
    return int(re.match(r"Stage (\d+) of", first_user).group(1))


def writes_one_kernel_per_stage(request):
    """The fake coder: on its first turn it writes a kernel.py that names its stage, then it ends."""
    k = stage_number(request)
    if turn(request) == 0:
        return call("write_file", path="kernel.py", content=f"# kernel written in stage {k}\n")
    return final("done")


def no_run(argv, **kw):
    raise AssertionError(f"nothing may run: {argv}")


def table_watcher(results: dict, calls: list | None = None):
    """A fake run_watched. It reads the spec the check was given and answers with results[spec's reference] (PASS when
    the table has none), so each stage is judged by what its spec names."""
    def fake(argv, **kw):
        spec_path = Path(argv[argv.index("--spec") + 1])
        spec = json.loads(spec_path.read_text())
        if calls is not None:
            calls.append({"spec": spec, "spec_path": spec_path, "argv": list(argv),
                          "kernel": (Path(argv[argv.index("--attempt-dir") + 1]) / "kernel.py").read_text()})
        result = results.get(spec["reference"], PASS)
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps(result))
        return kp.Watched(0 if result["passed"] else 1, None, False, 0.1)
    return fake


def run_staged(tmp_path, env, server, op="center", results=None, calls=None, **kw):
    args = dict(op=op, n=1, probe_dir=tmp_path / "probe", env=env, endpoint=server.endpoint, model="fake",
                coder_model="fake-coder", probe_id="t", tt_lang=FACTS, temperature=0.7, seed=2026110301,
                run_watched=table_watcher(results or {}, calls))
    args.update(kw)
    return ks.run_attempt(**args)


# ---- the stage table ---------------------------------------------------------------------------------------------

def test_the_table_holds_center_and_rmsnorm_and_other_ops_can_be_added_as_data():
    assert [s.name for s in ks.STAGES["center"]] == ["copy", "scale", "row_mean", "center"]
    assert [s.name for s in ks.STAGES["rmsnorm"]] == ["copy", "square", "row_sum_of_squares", "scale_rows", "rmsnorm"]
    assert set(ks.STAGES) <= set(kp.OPS)


@pytest.mark.parametrize("op", sorted(ks.STAGES))
def test_every_stage_names_a_reference_that_exists_and_only_the_last_uses_the_ops_own(op):
    import ast
    tree = ast.parse((kp.TEMPLATES / "staged_references.py").read_text(encoding="utf-8"))
    table = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "STAGE_REFERENCES" for t in n.targets))
    keys = {k.value for k in table.keys}              # read without importing the file: it imports torch
    stages = ks.STAGES[op]
    for s in stages[:-1]:
        assert s.reference in keys, s.name
    assert stages[-1].reference is None


@pytest.mark.parametrize("op", sorted(ks.STAGES))
def test_the_last_stage_is_the_real_op_and_no_instruction_holds_a_worked_solution(op):
    stages = ks.STAGES[op]
    assert ks.stage_spec(op, stages[-1]) == kp.load_spec(op)
    for s in stages:
        assert s.instruction.strip()
        assert "ttl." not in s.instruction and "def " not in s.instruction and "```" not in s.instruction


@pytest.mark.parametrize("op", sorted(ks.STAGES))
def test_the_stages_of_an_op_share_the_whole_kernel_budget_and_never_exceed_it(op):
    stages = ks.STAGES[op]
    assert sum(s.max_turns for s in stages) <= kp.BUDGET["max_turns"] and sum(s.max_turns for s in stages) > kp.BUDGET["max_turns"] - len(stages)
    assert sum(s.wall_s for s in stages) == pytest.approx(kp.BUDGET["wall_s"])


def test_a_stage_spec_changes_the_reference_the_inputs_and_the_output_and_nothing_else_in_the_bars():
    real = kp.load_spec("rmsnorm")
    stage = ks.STAGES["rmsnorm"][2]
    spec = ks.stage_spec("rmsnorm", stage)
    assert spec["reference"] == "stage_row_sum_of_squares" and spec["inputs"] == {"x": [64, 256]}
    assert spec["output"] == [64, 32] and spec["op"] == "rmsnorm.row_sum_of_squares"
    for key in ("pcc_bar_sim", "rel_l2_bar_sim", "slope_band_sim", "seeds", "dtype"):
        assert spec[key] == real[key]


# ---- the files a stage runs against ------------------------------------------------------------------------------

def test_stage_templates_hold_the_stage_spec_and_references_that_know_the_stage(tmp_path):
    stage = ks.STAGES["center"][1]
    tdir = ks.write_stage_templates(tmp_path / "t", "center", stage)
    assert json.loads((tdir / "ops" / "center.json").read_text())["reference"] == stage.reference
    refs = (tdir / "references.py").read_text()
    assert (kp.TEMPLATES / "references.py").read_text() in refs and "STAGE_REFERENCES" in refs
    assert (tdir / "kernel_check.py").read_bytes() == (kp.TEMPLATES / "kernel_check.py").read_bytes()
    assert (tdir / "kernel_harness.py").read_bytes() == (kp.TEMPLATES / "kernel_harness.py").read_bytes()
    assert (tdir / "staged_references.py").is_file()


@needs_sim
def test_the_merged_references_answer_a_stage_key(tmp_path):
    import subprocess, sys
    stage = ks.STAGES["rmsnorm"][2]
    tdir = ks.write_stage_templates(tmp_path / "t", "rmsnorm", stage)
    code = ("import torch, references as r; x = torch.arange(64 * 256, dtype=torch.float32).reshape(64, 256) / 1000;"
            f"y = r.REFERENCES[{stage.reference!r}]({{'x': x}}); assert y.shape == (64, 32);"
            "assert abs(float(y[3, 0]) - float((x[3] ** 2).sum())) < 1e-2 and float(y[:, 1:].abs().sum()) == 0;"
            "assert 'rmsnorm' in r.REFERENCES")
    got = subprocess.run([str(VENV / "bin" / "python"), "-c", code], cwd=tdir, capture_output=True, text=True)
    assert got.returncode == 0, got.stderr


# ---- stage progression -------------------------------------------------------------------------------------------

def test_every_stage_passing_runs_all_stages_in_order_and_the_attempt_passes(tmp_path, fake_kenv):
    calls: list = []
    with FakeModel(writes_one_kernel_per_stage) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, op="rmsnorm", calls=calls)
    assert attempt["passed_sim"] is True and attempt["category"] is None
    st = attempt["staged"]
    assert st["stage_names"] == ["copy", "square", "row_sum_of_squares", "scale_rows", "rmsnorm"]
    assert st["stages_passed"] == 5 and st["reached"] == 5 and st["failed_stage"] is None
    assert [s["name"] for s in st["stages"]] == st["stage_names"] and all(s["passed"] for s in st["stages"])
    assert [c["spec"]["reference"] for c in calls] == ["stage_copy", "stage_square", "stage_row_sum_of_squares",
                                                      "stage_scale_rows_by_inverse_rms", "rmsnorm"]
    assert [stage_number(r) for r in server.requests if turn(r) == 0] == [1, 2, 3, 4, 5]


def test_the_last_stage_is_judged_by_the_unmodified_op_spec(tmp_path, fake_kenv):
    calls: list = []
    with FakeModel(writes_one_kernel_per_stage) as server:
        run_staged(tmp_path, fake_kenv, server, op="center", calls=calls)
    assert calls[-1]["spec_path"] == kp.TEMPLATES / "ops" / "center.json"
    assert all(c["spec_path"] != kp.TEMPLATES / "ops" / "center.json" for c in calls[:-1])


def test_a_failure_at_stage_k_stops_the_attempt_and_names_the_stage(tmp_path, fake_kenv):
    results = {"stage_row_sum_of_squares": failing("shape_mismatch", "Tensor shape (32, 32) does not match Block shape (1, 8)")}
    with FakeModel(writes_one_kernel_per_stage) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, op="rmsnorm", results=results)
    st = attempt["staged"]
    assert attempt["passed_sim"] is False and attempt["category"] == "shape_mismatch"
    assert st["failed_stage"] == "row_sum_of_squares" and st["stages_passed"] == 2 and st["reached"] == 3
    assert [s["passed"] for s in st["stages"]] == [True, True, False]
    assert [stage_number(r) for r in server.requests if turn(r) == 0] == [1, 2, 3]       # stages 4 and 5 never started
    assert "row_sum_of_squares" in attempt["status_detail"] or "row_sum_of_squares" in attempt["first_line"]


def test_a_pass_of_the_last_stage_with_an_earlier_failure_is_impossible_because_the_attempt_stops(tmp_path, fake_kenv):
    results = {"stage_copy": failing()}
    with FakeModel(writes_one_kernel_per_stage) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, results=results)
    assert attempt["passed_sim"] is False and attempt["staged"]["reached"] == 1


def test_the_verified_prefix_is_the_next_stages_starting_file(tmp_path, fake_kenv):
    def reads_then_writes(request):
        k = stage_number(request)
        t = turn(request)
        if t == 0:
            return call("read_file", path="kernel.py")
        if t == 1:
            return call("write_file", path="kernel.py", content=f"# kernel written in stage {k}\n")
        return final("done")
    with FakeModel(reads_then_writes) as server:
        run_staged(tmp_path, fake_kenv, server, op="center")
    for r in server.requests:
        if turn(r) == 1 and stage_number(r) > 1:
            shown = [m["content"] for m in r["messages"] if m["role"] == "tool"][-1]
            assert f"# kernel written in stage {stage_number(r) - 1}" in shown, stage_number(r)


def test_the_starting_file_is_the_checked_bytes_even_if_the_attempt_directory_copy_was_changed(tmp_path, fake_kenv):
    """The prefix that is kept is what the check passed, not whatever sits in kernel.py afterwards."""
    adir = tmp_path / "probe" / "center" / "attempt-1"
    calls: list = []
    inner = table_watcher({}, calls)

    def tampering(argv, **kw):
        got = inner(argv, **kw)
        (adir / "kernel.py").write_text("# TAMPERED after the check\n")
        return got
    seen = []

    def script(request):
        k, t = stage_number(request), turn(request)
        if t == 0:
            seen.append((k, (adir / "kernel.py").read_text() if (adir / "kernel.py").exists() else None))
            return call("write_file", path="kernel.py", content=f"# kernel written in stage {k}\n")
        return final("done")
    with FakeModel(script) as server:
        run_staged(tmp_path, fake_kenv, server, op="center", run_watched=tampering)
    assert seen[1][1] == "# kernel written in stage 1\n"


def test_a_stage_that_spends_its_turns_ends_the_attempt_at_that_stage(tmp_path, fake_kenv):
    def never_finishes(request):
        return call("read_file", path="kernel_spec.json", call_id=f"c{turn(request)}")
    with FakeModel(never_finishes) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, stage_turns=3)
    st = attempt["staged"]
    assert st["reached"] == 1 and st["stages"][0]["turns"] <= 3 and st["failed_stage"] == "copy"
    assert attempt["status"] in ("turns", "escalate", "pause")
    assert [stage_number(r) for r in server.requests if turn(r) == 0] == [1]


def test_a_stage_that_spends_its_wall_clock_ends_the_attempt_at_that_stage(tmp_path, fake_kenv):
    with FakeModel(writes_one_kernel_per_stage) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, stage_wall_s=0)
    assert attempt["status"] == "wall-clock" and attempt["staged"]["reached"] == 1
    assert attempt["passed_sim"] is False


def test_each_stage_has_its_own_turn_and_wall_budget_from_the_table(tmp_path, fake_kenv, monkeypatch):
    small = tuple(ks.Stage(**{**s.__dict__, "max_turns": 2 + i}) for i, s in enumerate(ks.STAGES["center"]))
    monkeypatch.setitem(ks.STAGES, "center", small)

    def never_finishes(request):
        return call("read_file", path="kernel_spec.json", call_id=f"c{turn(request)}")
    with FakeModel(never_finishes) as server:
        attempt = run_staged(tmp_path, fake_kenv, server)
    assert attempt["staged"]["stages"][0]["turns"] <= 2


def test_the_seed_is_fixed_for_every_stage_of_an_attempt_and_recorded(tmp_path, fake_kenv):
    with FakeModel(writes_one_kernel_per_stage) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, seed=2026110307, temperature=0.7)
    assert {r.get("seed") for r in server.requests} == {2026110307}
    assert {r.get("temperature") for r in server.requests} == {0.7}
    assert attempt["sampling"]["seed"] == 2026110307


def test_the_system_prompt_is_the_probe_skill_plus_the_staged_note_and_the_user_message_names_only_this_step(tmp_path, fake_kenv):
    with FakeModel(writes_one_kernel_per_stage) as server:
        run_staged(tmp_path, fake_kenv, server, op="rmsnorm")
    first = [r for r in server.requests if turn(r) == 0]
    system = first[0]["messages"][0]["content"]
    assert kp.SKILL.read_text(encoding="utf-8") in system and "stage" in system[len(kp.SKILL.read_text("utf-8")):].lower()
    users = [next(m for m in r["messages"] if m["role"] == "user")["content"] for r in first]
    assert users[0].startswith("Stage 1 of 5") and ks.STAGES["rmsnorm"][0].instruction in users[0]
    assert ks.STAGES["rmsnorm"][1].instruction not in users[0] and ks.STAGES["rmsnorm"][4].instruction not in users[0]


def test_each_stage_is_checked_against_its_own_spec_in_the_agents_directory(tmp_path, fake_kenv):
    adir = tmp_path / "probe" / "rmsnorm" / "attempt-1"
    seen = {}

    def script(request):
        k = stage_number(request)
        if turn(request) == 0:
            seen[k] = json.loads((adir / "kernel_spec.json").read_text())["reference"]
            return call("write_file", path="kernel.py", content=f"# {k}\n")
        return final("done")
    with FakeModel(script) as server:
        run_staged(tmp_path, fake_kenv, server, op="rmsnorm")
    assert seen == {1: "stage_copy", 2: "stage_square", 3: "stage_row_sum_of_squares",
                    4: "stage_scale_rows_by_inverse_rms", 5: "rmsnorm"}


def test_the_record_is_saved_and_has_the_fields_the_probe_report_reads(tmp_path, fake_kenv):
    with FakeModel(writes_one_kernel_per_stage) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, op="center")
    saved = json.loads((tmp_path / "probe" / "center" / "attempt-1" / "attempt.json").read_text())
    assert saved["staged"] == attempt["staged"]
    for key in ("op", "attempt", "status", "turns", "agent_wall_s", "passed_sim", "category", "first_line", "sets",
                "sampling", "provenance"):
        assert key in attempt


def test_turns_and_wall_are_summed_over_the_stages(tmp_path, fake_kenv):
    with FakeModel(writes_one_kernel_per_stage) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, op="center")
    assert attempt["turns"] == sum(s["turns"] for s in attempt["staged"]["stages"])


# ---- the report --------------------------------------------------------------------------------------------------

def record(n, stages, passed_count, failed=None, category=None, line="", op="rmsnorm"):
    names = [s.name for s in ks.STAGES[op]]
    reached = len(names) if failed is None else names.index(failed) + 1
    return {"op": op, "attempt": n, "passed_sim": failed is None, "category": category, "first_line": line,
            "staged": {"stage_names": names, "stages_passed": passed_count, "reached": reached, "failed_stage": failed}}


def test_failure_class_splits_deadlock_and_dataflow_steps_from_other_runtime_errors():
    assert ks.failure_class("tt_lang_error", "Deadlock detected: all generators blocked") == "deadlock"
    assert ks.failure_class("tt_lang_error", "pop(): not a valid next dataflow step") == "dataflow step"
    assert ks.failure_class("tt_lang_error", "KeyError: 'x'") == "tt_lang_error"
    assert ks.failure_class("shape_mismatch", "Tensor shape") == "shape_mismatch"
    assert ks.failure_class(None, "") == "passed"


def test_the_report_counts_the_stage_reached_the_passes_and_the_failure_classes_per_stage():
    rows = [record(1, 5, 5), record(2, 5, 2, "row_sum_of_squares", "shape_mismatch", "Tensor shape (32, 32) does not match Block shape (1, 8)"),
            record(3, 5, 2, "row_sum_of_squares", "tt_lang_error", "Deadlock detected: all generators blocked"),
            record(4, 5, 0, "copy", "numeric_mismatch", "set seed0: pcc 0.1"),
            record(5, 5, 4, "rmsnorm", "numeric_mismatch", "set seed0: pcc 0.5")]
    text = "\n".join(ks.report_lines(rows, "rmsnorm"))
    assert "passed 1 of 5" in text
    hist = ks.reached_histogram(rows, "rmsnorm")
    assert hist == {"copy": 1, "square": 0, "row_sum_of_squares": 2, "scale_rows": 0, "rmsnorm": 2}
    per = ks.failure_classes_by_stage(rows, "rmsnorm")
    assert per["row_sum_of_squares"] == {"shape_mismatch": 1, "deadlock": 1}
    assert per["copy"] == {"numeric_mismatch": 1} and per["rmsnorm"] == {"numeric_mismatch": 1}
    assert "| row_sum_of_squares | 2 | 2 | deadlock 1, shape_mismatch 1 |" in text
    assert "deadlock 1" in text and "shape_mismatch 1" in text


def test_the_report_skips_attempts_with_no_stage_record():
    rows = [{"op": "rmsnorm", "attempt": 1, "passed_sim": False}, record(2, 5, 5)]
    assert ks.reached_histogram(rows, "rmsnorm")["rmsnorm"] == 1


# ---- the probe option --------------------------------------------------------------------------------------------

def test_the_probe_refuses_staged_for_an_op_with_no_stage_table(tmp_path, capsys):
    import argparse
    args = argparse.Namespace(op="add", staged=True, attempts=1, out=str(tmp_path / "o"))
    assert kp.main(args) == kp.EXIT_REFUSED
    assert "no stage table" in capsys.readouterr().err


def test_main_runs_staged_attempts_and_writes_the_staged_report(tmp_path, fake_kenv, env_ready):
    import io
    out = io.StringIO()
    with FakeModel(writes_one_kernel_per_stage) as server:
        code = kp.main(probe_args(tmp_path, op="center", staged=True, no_agent_hw=True, attempts=2, temperature="0.7",
                                  seed="2026110300", endpoint=server.endpoint, stage_turns=None, stage_wall_s=None),
                       env=fake_kenv, out=out, probe=lambda e, m: True, run_watched_fn=table_watcher(
                           {"stage_row_mean": failing("shape_mismatch", "Tensor shape (32, 32) does not match Block shape (1, 8)")}))
    assert code == kp.EXIT_OK, out.getvalue()
    text = (tmp_path / "out" / "kernel-probe-staged-report.md").read_text()
    assert "passed 0 of 2" in text and "| row_mean | 2 | 2 | shape_mismatch 2 |" in text
    seeds = sorted({r.get("seed") for r in server.requests})
    assert seeds == [2026110301, 2026110302]


def test_main_refuses_staged_with_the_agent_hardware_tool_or_the_hardware_phase(tmp_path, fake_kenv, env_ready, capsys):
    import io
    for extra, words in (({}, "--no-agent-hw"), ({"no_agent_hw": True, "phase": "hw"}, "simulator only")):
        code = kp.main(probe_args(tmp_path, op="center", staged=True, **extra), env=fake_kenv, out=io.StringIO(),
                       probe=lambda e, m: True, run_watched_fn=no_run)
        assert code == kp.EXIT_REFUSED and words in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_the_cli_accepts_staged_and_the_stage_budgets():
    from orchard import cli
    p = cli._parser()
    a = p.parse_args(["kernel-probe", "--op", "center", "--staged", "--stage-turns", "9", "--stage-wall-s", "300"])
    assert a.staged is True and a.stage_turns == 9 and a.stage_wall_s == 300.0


# ---- review findings (PR 199) ------------------------------------------------------------------------------------

def test_the_watchdog_is_told_each_stage_so_its_rung_counts_are_per_stage(tmp_path, fake_kenv):
    seen = []

    class Dog:
        stage = None

        def feed(self, ev):
            seen.append(self.stage)
            return []
    with FakeModel(writes_one_kernel_per_stage) as server:
        run_staged(tmp_path, fake_kenv, server, op="center", watchdog=Dog())
    assert sorted(set(seen)) == [1, 2, 3, 4]


def test_the_attempts_agent_time_is_the_sum_of_the_stages_agent_times(tmp_path, fake_kenv):
    with FakeModel(writes_one_kernel_per_stage) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, op="center")
    assert attempt["agent_wall_s"] == pytest.approx(sum(s["agent_wall_s"] for s in attempt["staged"]["stages"]), abs=0.02)


def test_an_edited_agent_check_is_untrusted_when_the_final_stage_fails(tmp_path, fake_kenv):
    def script(request):
        k, t = stage_number(request), turn(request)
        if t == 0:
            return call("write_file", path="kernel.py", content=f"# {k}\n")
        if t == 1 and k == 4:
            return call("write_file", path="check.json", content=json.dumps(
                {"passed": True, "category": None, "sha256": {"kernel_check.py": "0" * 64}}))
        return final("done")
    with FakeModel(script) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, op="center", results={"center": failing()})
    assert attempt["staged"]["failed_stage"] == "center" and attempt["agent_check_untrusted"] is True


def test_a_stage_that_spends_its_turns_ends_the_attempt_even_if_its_check_passes(tmp_path, fake_kenv):
    def writes_then_loops(request):
        k, t = stage_number(request), turn(request)
        if t == 0:
            return call("write_file", path="kernel.py", content=f"# {k}\n")
        return call("read_file", path="kernel_spec.json", call_id=f"c{t}")
    with FakeModel(writes_then_loops) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, op="center", stage_turns=3)
    st = attempt["staged"]
    assert attempt["passed_sim"] is False and st["failed_stage"] == "copy" and st["reached"] == 1 and st["stages_passed"] == 0
    assert attempt["category"] == "stage_budget" and st["stages"][0]["passed"] is False
    assert [stage_number(r) for r in server.requests if turn(r) == 0] == [1]


# ---- review round 2 (PR 199) ------------------------------------------------------------------------------------

def test_installing_a_stage_does_not_write_through_a_link_the_agent_left(tmp_path):
    adir = tmp_path / "adir"
    adir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me")
    for name in ("kernel_check.py", "references.py", "kernel_spec.json", "staged_references.py"):
        (adir / name).symlink_to(outside)
    stage = ks.STAGES["center"][1]
    tdir = ks.write_stage_templates(tmp_path / "t", "center", stage)
    ks._install_in_attempt(adir, tdir, "center")
    assert outside.read_text() == "keep me"
    assert not (adir / "kernel_check.py").is_symlink() and (adir / "kernel_spec.json").read_text().startswith("{")


def test_a_relative_probe_directory_works(tmp_path, fake_kenv, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls: list = []
    with FakeModel(writes_one_kernel_per_stage) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, op="center", probe_dir=Path("rel-probe"), calls=calls)
    assert attempt["passed_sim"] is True
    assert all(Path(c["argv"][1]).is_absolute() and Path(c["spec_path"]).is_absolute() for c in calls)


def test_the_last_stage_gets_the_shipped_spec_byte_for_byte(tmp_path):
    tdir = ks.write_stage_templates(tmp_path / "t", "rmsnorm", ks.STAGES["rmsnorm"][-1])
    assert (tdir / "ops" / "rmsnorm.json").read_bytes() == (kp.TEMPLATES / "ops" / "rmsnorm.json").read_bytes()


def test_an_unedited_agent_check_of_a_passing_attempt_is_trusted(tmp_path, fake_kenv):
    shipped = {"kernel_check.py": kp._sha256(kp.TEMPLATES / "kernel_check.py"),
               "kernel_harness.py": kp._sha256(kp.TEMPLATES / "kernel_harness.py"),
               "references.py": kp._sha256(kp.TEMPLATES / "references.py"),
               "spec": kp._sha256(kp.TEMPLATES / "ops" / "center.json")}

    def script(request):
        k, t = stage_number(request), turn(request)
        if t == 0:
            return call("write_file", path="kernel.py", content=f"# {k}\n")
        if t == 1 and k == 4:
            adir = tmp_path / "probe" / "center" / "attempt-1"
            spec_hash = kp._sha256(adir / "kernel_spec.json")
            refs_hash = kp._sha256(adir / "references.py")
            return call("write_file", path="check.json", content=json.dumps(
                {"passed": True, "category": None, "sha256": {**shipped, "spec": spec_hash, "references.py": refs_hash}}))
        return final("done")
    with FakeModel(script) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, op="center")
    assert attempt["passed_sim"] is True and attempt["agent_check_untrusted"] is False


def test_the_probe_refuses_stage_budgets_that_are_not_finite_or_not_positive(tmp_path, fake_kenv, env_ready, capsys):
    import io
    for extra, words in (({"stage_wall_s": float("nan")}, "finite"), ({"stage_wall_s": float("inf")}, "finite"),
                         ({"stage_wall_s": 0.0}, "finite"), ({"stage_turns": 0}, "at least 1"), ({"stage_turns": -3}, "at least 1")):
        code = kp.main(probe_args(tmp_path, op="center", staged=True, no_agent_hw=True, **extra), env=fake_kenv, out=io.StringIO(),
                       probe=lambda e, m: True, run_watched_fn=no_run)
        assert code == kp.EXIT_REFUSED and words in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_a_final_stage_that_spends_its_budget_with_a_matching_agent_check_is_still_trusted(tmp_path, fake_kenv):
    shipped = {n: kp._sha256(kp.TEMPLATES / f) for n, f in (("kernel_check.py", "kernel_check.py"), ("kernel_harness.py", "kernel_harness.py"),
                                                            ("references.py", "references.py"), ("spec", "ops/center.json"))}

    def script(request):
        k, t = stage_number(request), turn(request)
        if k < 4:
            return call("write_file", path="kernel.py", content=f"# {k}\n") if t == 0 else final("done")
        if t == 0:
            return call("write_file", path="kernel.py", content="# 4\n")
        if t == 1:
            return call("write_file", path="check.json", content=json.dumps({"passed": True, "category": None, "sha256": shipped}))
        return call("read_file", path="kernel_spec.json", call_id=f"c{t}")
    with FakeModel(script) as server:
        attempt = run_staged(tmp_path, fake_kenv, server, op="center", stage_turns=4)
    assert attempt["staged"]["failed_stage"] == "center" and attempt["category"] == "stage_budget"
    assert attempt["agent_check_untrusted"] is False


# ---- the real simulator ------------------------------------------------------------------------------------------

STAGED_FIXTURES = Path(__file__).parent / "fixtures" / "kernel-loop" / "staged"


@needs_sim
@pytest.mark.parametrize("op,stage_name,fixture", [
    ("center", "copy", "center_copy.py"), ("center", "scale", "center_scale.py"),
    ("center", "row_mean", "center_row_mean.py"), ("rmsnorm", "copy", "center_copy.py"),
    ("rmsnorm", "square", "rmsnorm_square.py"), ("rmsnorm", "row_sum_of_squares", "rmsnorm_row_sum_of_squares.py"),
    ("rmsnorm", "scale_rows", "rmsnorm_scale_rows.py")])
def test_a_fixture_kernel_passes_its_stage_on_the_real_simulator(tmp_path, kenv, op, stage_name, fixture):
    stage = next(s for s in ks.STAGES[op] if s.name == stage_name)
    tdir = ks.write_stage_templates(tmp_path / "t", op, stage)
    adir = tmp_path / "probe" / op / "attempt-1"
    kp.prepare_attempt(adir, op=op, env=kenv, tt_lang=kp.tt_lang_facts(kenv))
    (adir / "kernel.py").write_bytes((STAGED_FIXTURES / fixture).read_bytes())
    got = kp.authoritative_check(adir, op=op, env=kenv, tt_lang=kp.tt_lang_facts(kenv), templates=tdir)
    assert got["passed"] is True, got.get("first_line")
