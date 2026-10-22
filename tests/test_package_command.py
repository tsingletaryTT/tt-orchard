# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard package RUN`: stage 7 on a finished run, recorded in its ledger (issue 22).

The hardware part is a `gozer run` command line. A test passes a runner that records it and, like the
real boot check, leaves test-result.json and verify.json behind. Nothing here touches a chip or runs
gozer; the stub `gozer` fails the test if it is ever reached."""
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from orchard import package_cmd
from orchard.ledger import Ledger, read_entries
from orchard.paths import PATHS_RECORDED
from orchard.server import proc_identity
from orchard.stages import gate_package
from package_fakes import fake_bin, fake_boot_result, make_run, make_source, write

FAKE_SERVER = os.path.join(os.path.dirname(__file__), "fake_swap_server.py")
pytestmark = pytest.mark.usefixtures("stub_tools")


def finished_ledger(run, *, models, cache, ready=True, stage7="skipped"):
    """The ledger of a run that ended `ready-for-operator-review` with stage 7 skipped, the way
    AREX-2's did (it was started without --package-format)."""
    with Ledger(run / "ledger.jsonl") as led:
        led.append("run_start", None, model="Altworld/Hemmingway-1", package=None,
                   paths={"orchard_dir": "/o", "hf_home": "/h", "operator_home": "/home/op",
                          "tt_model_root": str(models), "cache_root": str(cache)})
        led.append("stage_end", 7, result=stage7, reason="no package format")
        if ready:
            led.append("decision", None, decision="ready for operator review")


@pytest.fixture
def rig(tmp_path, monkeypatch, stub_tools):
    models = tmp_path / "models"
    r = make_run(tmp_path, make_source(models))
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    cache = tmp_path / "cache"
    finished_ledger(r["run"], models=models, cache=cache)
    extra = {"FAKE_TT_MODEL_PYTHON": sys.executable, "FAKE_SWAP_SERVER": FAKE_SERVER,
             "FAKE_SWAP_CONFIG": str(tmp_path / "server.json")}
    return {**r, "models": models, "cache": cache, "extra_env": extra, "stage": r["run"] / "stages/7",
            "tmp": tmp_path, "stubs": stub_tools}


class Runner:
    """Stands in for the `gozer run` call. It records the command line and, as the real boot check
    does, leaves the boot check's files (or a failure) in stage 7."""

    def __init__(self, stage, *, code=0, boot_code=0):
        self.stage, self.code, self.boot_code, self.calls = stage, code, boot_code, []

    def __call__(self, argv, env):
        self.calls.append((list(argv), dict(env)))
        if self.code == 0:
            fake_boot_result(self.stage, returncode=self.boot_code)
        return self.code


def args(rig, **kw):
    base = dict(run_dir=str(rig["run"]), step="all", namespace="episod", models_root=None, cache_root=None,
                restage=False, gozer="gozer", exact=None, python=sys.executable)
    base.update(kw)
    return SimpleNamespace(**base)


def run_cmd(rig, runner=None, **kw):
    out = io.StringIO()
    runner = runner or Runner(rig["stage"])
    code = package_cmd.main(args(rig, **kw), out=out, runner=runner, extra_env=rig["extra_env"],
                            hostname="testhost")
    return code, out.getvalue(), runner


def events(rig):
    return [(e["event"], e["data"].get("decision") or e["data"].get("what")) for e in read_entries(rig["run"] / "ledger.jsonl")]


# ---- the whole flow -----------------------------------------------------------------------------

def test_a_finished_run_is_staged_boot_checked_finished_and_gated(rig):
    code, text, runner = run_cmd(rig)
    assert code == 0, text
    assert (rig["stage"] / "package/hemmingway-1-p300/README.md").is_file()
    assert gate_package(rig["stage"], rig["run"]).ok
    assert len(runner.calls) == 1
    assert "gate: ok" in text
    # the publish commands are printed as text for the operator to read
    assert "hf upload --repo-type model --private episod/hemmingway-1-p300" in text
    assert "nothing was run, uploaded or published" in text


def test_the_ledger_records_what_the_command_did_and_the_chain_stays_valid(rig):
    run_cmd(rig)
    names = events(rig)
    assert ("decision", "package command") in names
    assert ("evidence", "package staged") in names
    assert ("evidence", "package boot check") in names
    assert ("decision", "package gate") in names
    gate = [e for e in read_entries(rig["run"] / "ledger.jsonl")
            if e["data"].get("decision") == "package gate"][0]
    assert gate["data"]["ok"] is True and gate["data"]["reasons"] == []
    boot = [e for e in read_entries(rig["run"] / "ledger.jsonl") if e["data"].get("what") == "package boot check"][0]
    assert boot["data"]["path"] == "stages/7/test-result.json" and len(boot["data"]["sha256"]) == 64


def test_the_run_still_reads_as_ready_for_review_afterwards(rig):
    from orchard.status import collect
    run_cmd(rig)
    facts = collect(rig["run"], pid_alive=lambda p: False, lock_holder=lambda r: None,
                    disk_free_gb=lambda p: 500.0, gozer_status=lambda: "")
    assert facts["state"] == "ready-for-operator-review"


# ---- the lease -----------------------------------------------------------------------------------

def test_the_boot_check_runs_under_a_gozer_lease_for_the_required_profiles_chips(rig):
    _, _, runner = run_cmd(rig)
    argv, env = runner.calls[0]
    assert argv[:2] == ["gozer", "run"]
    assert argv[argv.index("--chips") + 1] == "2"
    assert "--who" in argv and "--reason" in argv
    tail = argv[argv.index("--") + 1:]
    assert tail[0] == "timeout" and tail[2:9] == [sys.executable, "-m", "orchard.cli", "package",
                                                   str(rig["run"]), "--step", "boot"]
    assert float(tail[1]) > 0


def test_a_chip_to_pin_the_lease_to_is_passed_through(rig):
    _, _, runner = run_cmd(rig, exact=2)
    argv = runner.calls[0][0]
    assert argv[argv.index("--exact") + 1] == "2"
    _, _, runner = run_cmd(rig, restage=True)
    assert "--exact" not in runner.calls[0][0]


def test_the_lease_command_carries_the_paths_the_boot_check_needs_and_no_token(rig, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    _, _, runner = run_cmd(rig)
    argv, env = runner.calls[0]
    tail = argv[argv.index("--") + 1:]
    assert "--cache-root" in tail and tail[tail.index("--cache-root") + 1] == str(rig["cache"])
    assert "HF_TOKEN" not in env


def test_the_checkout_is_on_the_path_of_the_leased_command_so_it_imports_from_any_directory(rig, monkeypatch):
    checkout = os.path.dirname(os.path.dirname(package_cmd.__file__))
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    _, _, runner = run_cmd(rig)
    assert runner.calls[0][1]["PYTHONPATH"].split(os.pathsep) == [checkout, "/somewhere/else"]
    monkeypatch.delenv("PYTHONPATH")
    _, _, runner = run_cmd(rig, restage=True)
    assert runner.calls[0][1]["PYTHONPATH"] == checkout


def test_a_queued_lease_ends_the_command_with_the_reason_and_nothing_is_finished(rig):
    code, text, _ = run_cmd(rig, Runner(rig["stage"], code=10))
    assert code == 3
    assert "queued" in text and "run the same command again" in text
    assert not (rig["stage"] / "test-result.json").exists()
    assert ("decision", "package gate") not in events(rig)


def test_a_boot_check_that_fails_is_reported_gated_and_not_offered_for_publishing(rig):
    code, text, _ = run_cmd(rig, Runner(rig["stage"], boot_code=1))
    assert code == 1
    assert "gate: not ok" in text
    assert "hf upload" not in text
    gate = [e for e in read_entries(rig["run"] / "ledger.jsonl") if e["data"].get("decision") == "package gate"][0]
    assert gate["data"]["ok"] is False and gate["data"]["reasons"]


# ---- refusals -------------------------------------------------------------------------------------

def test_a_run_with_no_ledger_is_refused(rig):
    (rig["run"] / "ledger.jsonl").unlink()
    code, text, runner = run_cmd(rig)
    assert code == 2 and "no ledger.jsonl" in text and runner.calls == []


def test_a_run_that_has_not_finished_is_refused(rig):
    (rig["run"] / "ledger.jsonl").unlink()
    finished_ledger(rig["run"], models=rig["models"], cache=rig["cache"], ready=False)
    code, text, runner = run_cmd(rig)
    assert code == 2 and "not finished" in text and runner.calls == []
    assert not rig["stage"].exists()


def test_a_run_a_supervisor_still_holds_is_refused_and_untouched(rig):
    with Ledger(rig["run"] / "ledger.jsonl"):
        code, text, runner = run_cmd(rig)
    assert code == 2 and "supervisor" in text and runner.calls == []
    assert not rig["stage"].exists()


def test_a_staged_package_is_not_overwritten_without_restage(rig):
    run_cmd(rig)
    before = (rig["stage"] / "package.json").read_text()
    code, text, runner = run_cmd(rig)
    assert code == 2 and "--restage" in text and runner.calls == []
    assert (rig["stage"] / "package.json").read_text() == before


def test_restage_moves_the_old_stage_aside_and_starts_again(rig):
    run_cmd(rig)
    code, text, _ = run_cmd(rig, restage=True)
    assert code == 0, text
    aside = [p.name for p in (rig["run"] / "stages").iterdir() if p.name.startswith("7.old-")]
    assert len(aside) == 1


def test_a_test_command_that_could_publish_is_refused_before_anything_runs(rig):
    code, _, _ = run_cmd(rig, step="stage")
    assert code == 0
    test = json.loads((rig["stage"] / "hw_test.json").read_text())
    test["command"] = "hf upload --private some/repo ."
    write(rig["stage"] / "hw_test.json", test)
    code, text, runner = run_cmd(rig, step="boot")
    assert code == 2 and "refused" in text and runner.calls == []


def test_the_lease_command_line_is_checked_to_be_exactly_the_boot_step():
    ok = ["gozer", "run", "--chips", "2", "--who", "w", "--reason", "r", "--", "timeout", "60", sys.executable,
          "-m", "orchard.cli", "package", "/run", "--step", "boot"]
    package_cmd.check_lease_argv(ok, run_dir="/run", python=sys.executable)
    package_cmd.check_lease_argv(ok[:8] + ["--exact", "0"] + ok[8:], run_dir="/run", python=sys.executable)
    package_cmd.check_lease_argv(ok + ["--cache-root", "/c"], run_dir="/run", python=sys.executable)
    for bad in (ok + ["--also", "x"],
                ok[:-1] + ["finish"],
                ok[:10] + ["hf", "upload", "--private", "a/b", "."],
                ["gozer", "release", "abc"],
                ["gozer", "reset", "0"],
                ok[:1] + ["acquire"] + ok[2:],
                ok[:2] + ["--chips", "4"] + ok[2:],
                ok[:8] + ["--tag", "x"] + ok[8:],
                ok[:9] + ["env", "FOO=1"] + ok[11:],
                ok[:9] + ["sleep", "5", ";"] + ok[11:],
                ["gozer", "run", "--chips", "2"] + ok[8:9] + ok[10:]):
        with pytest.raises(package_cmd.PackageCommandError):
            package_cmd.check_lease_argv(bad, run_dir="/run", python=sys.executable)


def test_the_command_has_no_upload_code_and_never_calls_publish_tools():
    src = open(package_cmd.__file__, encoding="utf-8").read()
    for word in ("hf upload", "upload_file", "upload_folder", "create_repo", "git push", "publish_repo"):
        assert word not in src


def test_no_stub_tool_ran(rig):
    run_cmd(rig)
    assert rig["stubs"].calls() == []


# ---- the boot step, run under a lease ----------------------------------------------------------

def fake_spawn(command, run_dir, env, timeout, stdout):
    fake_spawn.seen = (command, env, timeout)
    stdout.write(b"boot output\n")
    return 0, False


def test_the_boot_step_runs_the_staged_command_with_the_leased_chips_and_the_kernel_cache(rig, monkeypatch):
    run_cmd(rig, step="stage")
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("TT_METAL_VISIBLE_DEVICES", "9")
    out = io.StringIO()
    code = package_cmd.main(args(rig, step="boot", cache_root=str(rig["cache"])), out=out, spawn=fake_spawn,
                            extra_env=rig["extra_env"], hostname="testhost")
    assert code == 0, out.getvalue()
    command, env, timeout = fake_spawn.seen
    assert env["TT_VISIBLE_DEVICES"] == "2,3" and "TT_METAL_VISIBLE_DEVICES" not in env
    assert env["TT_METAL_CACHE"] == str(rig["cache"] / "kernels")
    assert timeout == json.loads((rig["stage"] / "hw_test.json").read_text())["deadline_s"]
    rec = json.loads((rig["stage"] / "test-result.json").read_text())
    assert rec["returncode"] == 0 and rec["timed_out"] is False and rec["chips"] == ["2", "3"]
    assert rec["output"]["path"] == "stages/7/evidence/hw-test-output.txt"


def test_the_boot_step_refuses_to_run_without_a_lease(rig, monkeypatch):
    run_cmd(rig, step="stage")
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
    out = io.StringIO()
    code = package_cmd.main(args(rig, step="boot"), out=out, spawn=fake_spawn, extra_env=rig["extra_env"])
    assert code == 2 and "gozer" in out.getvalue()
    assert not (rig["stage"] / "test-result.json").exists()


def test_a_boot_step_that_exits_nonzero_exits_nonzero_and_records_it(rig, monkeypatch):
    run_cmd(rig, step="stage")
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    out = io.StringIO()
    code = package_cmd.main(args(rig, step="boot", cache_root=str(rig["cache"])), out=out,
                            spawn=lambda c, r, e, t, o: (4, False), extra_env=rig["extra_env"])
    assert code == 1
    assert json.loads((rig["stage"] / "test-result.json").read_text())["returncode"] == 4


# ---- paths ------------------------------------------------------------------------------------------

def test_the_models_root_defaults_to_the_one_the_run_recorded_and_a_flag_overrides_it(rig):
    run_cmd(rig, step="stage")
    first = [e for e in read_entries(rig["run"] / "ledger.jsonl") if e["data"].get("decision") == "package command"][-1]
    assert first["data"]["models_root"] == str(rig["models"])
    other = rig["tmp"] / "elsewhere"
    other.mkdir()
    run_cmd(rig, step="stage", restage=True, models_root=str(other))
    last = [e for e in read_entries(rig["run"] / "ledger.jsonl") if e["data"].get("decision") == "package command"][-1]
    assert last["data"]["models_root"] == str(other)


def test_a_run_with_no_recorded_paths_needs_them_on_the_command_line(rig):
    (rig["run"] / "ledger.jsonl").unlink()
    with Ledger(rig["run"] / "ledger.jsonl") as led:
        led.append("run_start", None, model="m", package=None)
        led.append("decision", None, decision="ready for operator review")
    code, text, runner = run_cmd(rig)
    assert code == 2 and "--models-root and --cache-root" in text and runner.calls == []
    code, text, _ = run_cmd(rig, models_root=str(rig["models"]), cache_root=str(rig["cache"]))
    assert code == 0, text


# ---- the card step -----------------------------------------------------------------------------------

BENCH = {
    "models": {"arex": "BAAI/AREX-2@abc", "base": "Qwen/Qwen3.8-27B@def"},
    "conditions": "same bundle, thinking on, temperature 0, max_gen_toks 32768 for the re-run",
    "tasks": {
        "gsm8k": {"arex": {"n": 1319, "passed": 1237, "score": 0.938, "stderr": 0.007, "cut_off": 3},
                  "base": {"n": 1319, "passed": 1199, "score": 0.909, "stderr": 0.008, "cut_off": 1},
                  "paired": {"items": 1319, "arex_only_right": 61, "base_only_right": 23, "sign_test_p": 3.2e-05}},
        "ifeval": {"arex": {"n": 541, "passed": 460, "score": 0.85, "stderr": 0.015, "cut_off": 0},
                   "base": {"n": 541, "passed": 470, "score": 0.869, "stderr": 0.014, "cut_off": 0}},
    }}

REPORTED = {"source": "BAAI/AREX-2 model card", "rows": [{"name": "GAIA", "value": "70.1", "note": "text-only"}]}


def card_args(rig, **kw):
    bench = rig["tmp"] / "benchmark.json"
    if not bench.exists():
        bench.write_text(json.dumps(BENCH))
    base = dict(run_dir=str(rig["run"]), benchmark=str(bench), model_key="arex", base_key="base", reported=None,
                replace_reported=False, namespace="episod", models_root=None, cache_root=None)
    base.update(kw)
    return SimpleNamespace(**base)


def run_card(rig, **kw):
    out = io.StringIO()
    code = package_cmd.card_main(card_args(rig, **kw), out=out, extra_env=rig["extra_env"], hostname="testhost")
    return code, out.getvalue()


@pytest.fixture
def packaged(rig):
    code, text, _ = run_cmd(rig)
    assert code == 0, text
    return rig


def card_text(rig):
    return (rig["stage"] / "package/hemmingway-1-p300/README.md").read_text()


def test_benchmark_results_become_measured_rows_in_the_card_and_the_gate_still_passes(packaged):
    code, text = run_card(packaged)
    assert code == 0, text
    card = card_text(packaged)
    assert "GSM8K" in card and "0.938" in card and "0.909" in card
    assert "IFEval" in card and "0.85" in card
    assert "paired sign test" in card
    assert "replies cut off at the token limit, this model" in card
    assert (packaged["stage"] / "package/hemmingway-1-p300/evidence/stages/7/benchmark/benchmark.json").is_file()
    assert gate_package(packaged["stage"], packaged["run"]).ok


def test_the_card_step_records_itself_in_the_ledger(packaged):
    run_card(packaged)
    assert ("decision", "package card") in events(packaged)
    e = [x for x in read_entries(packaged["run"] / "ledger.jsonl") if x["data"].get("decision") == "package card"][0]
    assert e["data"]["benchmark"] == "stages/7/benchmark/benchmark.json" and len(e["data"]["sha256"]) == 64


def test_the_nearest_model_rows_are_labelled_as_the_control(packaged):
    run_card(packaged)
    assert "nearest model" in card_text(packaged)


def test_a_reported_results_file_is_installed_and_shown_apart_from_the_measured_rows(packaged):
    rep = packaged["tmp"] / "reported.json"
    rep.write_text(json.dumps(REPORTED))
    code, text = run_card(packaged, reported=str(rep))
    assert code == 0, text
    card = card_text(packaged)
    assert "## Reported by the authors" in card and "GAIA" in card and "70.1" in card
    assert card.index("## Reported by the authors") > card.index("GSM8K")
    assert (packaged["run"] / "reported-results.json").is_file()
    assert gate_package(packaged["stage"], packaged["run"]).ok


def test_a_bad_reported_file_is_refused_and_leaves_the_card_alone(packaged):
    before = card_text(packaged)
    rep = packaged["tmp"] / "reported.json"
    rep.write_text(json.dumps({"source": "", "rows": []}))
    code, text = run_card(packaged, reported=str(rep))
    assert code == 2 and "reported" in text
    assert card_text(packaged) == before and not (packaged["run"] / "reported-results.json").exists()


def test_a_different_reported_file_is_not_replaced_without_the_flag(packaged):
    rep = packaged["tmp"] / "reported.json"
    rep.write_text(json.dumps(REPORTED))
    run_card(packaged, reported=str(rep))
    other = packaged["tmp"] / "other.json"
    other.write_text(json.dumps({"source": "elsewhere", "rows": [{"name": "HLE", "value": "5"}]}))
    code, text = run_card(packaged, reported=str(other))
    assert code == 2 and "--replace-reported" in text
    assert "70.1" in card_text(packaged)
    code, text = run_card(packaged, reported=str(other), replace_reported=True)
    assert code == 0, text
    assert "HLE" in card_text(packaged) and "70.1" not in card_text(packaged)


def test_a_model_key_the_file_does_not_have_is_refused(packaged):
    code, text = run_card(packaged, model_key="missing")
    assert code == 2 and "missing" in text and "arex" in text


@pytest.mark.parametrize("damage, word", [
    (lambda b: b.pop("tasks"), "tasks"),
    (lambda b: b["tasks"]["gsm8k"]["arex"].pop("score"), "score"),
    (lambda b: b["tasks"]["gsm8k"]["arex"].update(score="high"), "score"),
    (lambda b: b["tasks"]["gsm8k"]["arex"].update(score=True), "score"),
    (lambda b: b["tasks"]["gsm8k"]["arex"].update(score=float("nan")), "score"),
    (lambda b: b["tasks"].update(empty={}), "empty"),
])
def test_a_benchmark_file_that_is_wrong_is_refused_with_the_reason(packaged, damage, word):
    bench = json.loads(json.dumps(BENCH))
    damage(bench)
    path = packaged["tmp"] / "bad.json"
    path.write_text(json.dumps(bench))
    before = card_text(packaged)
    code, text = run_card(packaged, benchmark=str(path))
    assert code == 2 and word in text
    assert card_text(packaged) == before


def test_the_card_step_needs_a_package_that_was_staged_and_boot_checked(rig):
    code, text = run_card(rig)
    assert code == 2 and "no staged and boot-checked package" in text
    assert ("decision", "package card") not in events(rig) and not (rig["stage"] / "benchmark").exists()
    run_cmd(rig, step="stage")                      # staged, but never boot-checked
    code, text = run_card(rig)
    assert code == 2 and "no staged and boot-checked package" in text
    assert ("decision", "package card") not in events(rig) and not (rig["stage"] / "benchmark").exists()


def test_the_card_step_is_refused_while_a_supervisor_holds_the_run(packaged):
    with Ledger(packaged["run"] / "ledger.jsonl"):
        code, text = run_card(packaged)
    assert code == 2 and "supervisor" in text


def test_a_second_card_step_replaces_the_earlier_rows(packaged):
    run_card(packaged)
    changed = json.loads(json.dumps(BENCH))
    changed["tasks"]["gsm8k"]["arex"]["score"] = 0.5
    path = packaged["tmp"] / "second.json"
    path.write_text(json.dumps(changed))
    code, text = run_card(packaged, benchmark=str(path))
    assert code == 0, text
    card = card_text(packaged)
    assert "0.5" in card and "0.938" not in card


def test_the_benchmark_numbers_file_is_read_back_by_the_package_code(packaged):
    from orchard.package import read_benchmark_numbers
    run_card(packaged)
    nums = read_benchmark_numbers(packaged["run"])
    assert nums and all(n.label == "measured" and n.evidence == ("stages/7/benchmark/benchmark.json",) for n in nums)
    assert read_benchmark_numbers(packaged["tmp"] / "nowhere") == ()


def test_staging_needs_a_namespace_and_changes_nothing_without_one(rig):
    for step in ("all", "stage"):
        code, text, runner = run_cmd(rig, namespace=None, step=step)
        assert code == 2 and "--namespace" in text and runner.calls == []
        assert not rig["stage"].exists()
        assert ("decision", "package command") not in events(rig)


def test_an_aborted_run_is_refused(rig):
    with Ledger(rig["run"] / "ledger.jsonl") as led:
        led.append("decision", None, decision="abort", reason="operator")
    code, text, runner = run_cmd(rig)
    assert code == 2 and "not finished" in text and runner.calls == []


@pytest.mark.parametrize("test", [{"command": "", "deadline_s": 60}, {"command": "true", "deadline_s": 0},
                                  {"command": "true", "deadline_s": -5}, {"command": "true"}])
def test_a_hw_test_file_that_is_wrong_is_refused(rig, test):
    run_cmd(rig, step="stage")
    write(rig["stage"] / "hw_test.json", test)
    code, text, runner = run_cmd(rig, step="boot")
    assert code == 2 and "hw_test.json" in text and runner.calls == []


def test_a_boot_check_that_timed_out_is_a_failure_even_with_exit_zero(rig, monkeypatch):
    run_cmd(rig, step="stage")
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    out = io.StringIO()
    code = package_cmd.main(args(rig, step="boot", cache_root=str(rig["cache"])), out=out,
                            spawn=lambda c, r, e, t, o: (0, True), extra_env=rig["extra_env"])
    assert code == 1
    assert json.loads((rig["stage"] / "test-result.json").read_text())["timed_out"] is True


def test_a_lease_that_ends_without_a_boot_result_finishes_nothing(rig):
    code, text, _ = run_cmd(rig, Runner(rig["stage"], code=1))
    assert code == 1 and "did not run to a result" in text
    assert ("decision", "package gate") not in events(rig)
    assert not (rig["stage"] / PUBLISH).exists()


PUBLISH = "PUBLISH_COMMANDS.txt"


def test_finish_without_a_staged_package_is_refused(rig):
    code, text, _ = run_cmd(rig, step="finish")
    assert code == 2 and "nothing is staged" in text


def test_the_finish_step_alone_gates_a_package_staged_and_boot_checked_earlier(rig):
    run_cmd(rig, step="stage")
    fake_boot_result(rig["stage"])
    code, text, runner = run_cmd(rig, step="finish")
    assert code == 0 and "gate: ok" in text and runner.calls == []


def test_a_benchmark_without_a_base_key_has_no_comparison_rows(packaged):
    code, text = run_card(packaged, base_key=None)
    assert code == 0, text
    card = card_text(packaged)
    assert "GSM8K" in card and "paired sign test" not in card and "nearest model, same conditions" not in card


@pytest.mark.parametrize("row, word", [
    ({"name": "", "value": 1, "unit": "x", "evidence": ["stages/7/benchmark/benchmark.json"]}, "name"),
    ({"name": "a", "value": "1", "unit": "x", "evidence": ["stages/7/benchmark/benchmark.json"]}, "number"),
    ({"name": "a", "value": True, "unit": "x", "evidence": ["stages/7/benchmark/benchmark.json"]}, "number"),
    ({"name": "a", "value": 1, "unit": 3, "evidence": ["stages/7/benchmark/benchmark.json"]}, "unit"),
    ({"name": "a", "value": 1, "unit": "x", "evidence": []}, "evidence"),
    ({"name": "a", "value": 1, "unit": "x", "evidence": [3]}, "evidence"),
    ({"name": "a", "value": 1, "unit": "x", "evidence": ["stages/7/nowhere.json"]}, "nowhere"),
])
def test_benchmark_numbers_that_are_wrong_are_refused_by_the_package_code(tmp_path, row, word):
    from orchard.package import PackageError, read_benchmark_numbers
    write(tmp_path / "stages/7/benchmark/benchmark.json", "{}")
    write(tmp_path / "stages/7/benchmark-numbers.json", {"rows": [row]})
    with pytest.raises(PackageError, match=word):
        read_benchmark_numbers(tmp_path)
    write(tmp_path / "stages/7/benchmark-numbers.json", {"rows": "no"})
    with pytest.raises(PackageError, match="list"):
        read_benchmark_numbers(tmp_path)


def test_the_default_runner_runs_the_command_with_the_given_environment_and_returns_its_exit_code(tmp_path):
    marker = tmp_path / "seen.txt"
    script = tmp_path / "s.sh"
    script.write_text(f'#!/bin/sh\nprintf "%s" "$PROBE" > {marker}\nexit 7\n')
    script.chmod(0o755)
    assert package_cmd._default_runner([str(script)], {"PROBE": "hello", "PATH": "/usr/bin:/bin"})[0] == 7
    assert marker.read_text() == "hello"


def test_the_command_line_the_lease_runs_is_a_real_entry_point_that_refuses_without_a_lease(rig):
    """`python -m orchard.cli package RUN --step boot`, as gozer would start it but with no lease."""
    run_cmd(rig, step="stage")
    env = {k: v for k, v in os.environ.items() if k != "TT_VISIBLE_DEVICES"}
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(package_cmd.__file__))
    done = subprocess.run([sys.executable, "-m", "orchard.cli", "package", str(rig["run"]), "--step", "boot"],
                          capture_output=True, text=True, env=env, cwd=str(rig["tmp"]), timeout=60)
    assert done.returncode == 2 and "runs under a lease" in done.stdout
    assert not (rig["stage"] / "test-result.json").exists()


# ---- review: a requested control that the benchmark does not hold --------------------------------------

def test_a_base_key_that_is_not_in_the_benchmark_is_refused_and_no_comparison_is_claimed(packaged):
    """`--base-key bas` (a typo) used to skip the control's scores but still print the paired p-value."""
    code, text = run_card(packaged, base_key="bas")
    assert code == 2 and "'bas'" in text and "arex, base" in text      # names the key and what the file holds
    assert not (packaged["stage"] / "benchmark-numbers.json").exists()
    assert "paired sign test" not in card_text(packaged)


def test_a_base_key_missing_from_one_task_is_refused_naming_the_task(packaged):
    bench = json.loads(json.dumps(BENCH))
    del bench["tasks"]["ifeval"]["base"]
    (packaged["tmp"] / "benchmark.json").write_text(json.dumps(bench))
    code, text = run_card(packaged)
    assert code == 2 and "ifeval" in text and "'base'" in text
    assert not (packaged["stage"] / "benchmark-numbers.json").exists()


# ---- review: the leased child needs the cache root and nothing else -------------------------------------

def test_the_boot_step_of_a_run_with_no_recorded_paths_runs_with_only_the_cache_root(rig, monkeypatch):
    """The outer command is given --models-root and --cache-root, but `lease_argv` forwards only --cache-root, so the
    child must not ask for the models root it never uses."""
    (rig["run"] / "ledger.jsonl").unlink()
    with Ledger(rig["run"] / "ledger.jsonl") as led:
        led.append("run_start", None, model="m", package=None)
        led.append("decision", None, decision="ready for operator review")
    code, text, _ = run_cmd(rig, step="stage", models_root=str(rig["models"]), cache_root=str(rig["cache"]))
    assert code == 0, text
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    seen = []
    out = io.StringIO()
    code = package_cmd.main(args(rig, step="boot", cache_root=str(rig["cache"])), out=out,
                            spawn=lambda c, r, e, t, o: (seen.append(e) or (0, False)), extra_env=rig["extra_env"])
    assert code == 0, out.getvalue()
    assert seen[0]["TT_METAL_CACHE"] == str(rig["cache"] / "kernels")


def test_the_boot_step_still_asks_for_the_cache_root_when_the_run_recorded_none(rig, monkeypatch):
    (rig["run"] / "ledger.jsonl").unlink()
    with Ledger(rig["run"] / "ledger.jsonl") as led:
        led.append("run_start", None, model="m", package=None)
        led.append("decision", None, decision="ready for operator review")
    run_cmd(rig, step="stage", models_root=str(rig["models"]), cache_root=str(rig["cache"]))
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    out = io.StringIO()
    code = package_cmd.main(args(rig, step="boot"), out=out, spawn=lambda c, r, e, t, o: (0, False),
                            extra_env=rig["extra_env"])
    assert code == 2 and "--cache-root" in out.getvalue() and "--models-root" not in out.getvalue()


# ---- review: a queued gozer request is cancelled, not abandoned ----------------------------------------

class QueuedRunner(Runner):
    """`gozer run` that queued a ticket: exit 10 and the line gozer prints (the runner returns what it printed too)."""

    def __init__(self, stage, text="queued: ticket T0k3n\nwaiting for chips 2,3\n"):
        super().__init__(stage, code=10)
        self.text = text

    def __call__(self, argv, env):
        self.calls.append((list(argv), dict(env)))
        return 10, self.text


def run_cmd_cancel(rig, runner, cancel):
    out = io.StringIO()
    code = package_cmd.main(args(rig), out=out, runner=runner, extra_env=rig["extra_env"], hostname="testhost",
                            cancel=cancel)
    return code, out.getvalue()


def test_a_queued_request_has_its_ticket_cancelled_and_the_ledger_and_the_operator_are_told(rig):
    cancelled = []
    code, text = run_cmd_cancel(rig, QueuedRunner(rig["stage"]), lambda t: cancelled.append(t))
    assert code == 3 and cancelled == ["T0k3n"]
    assert "T0k3n" in text and "cancelled" in text
    e = [x for x in read_entries(rig["run"] / "ledger.jsonl") if x["data"].get("outcome") == "queued"][0]
    assert e["data"]["ticket"] == "T0k3n" and e["data"]["cancel_error"] is None


def test_a_ticket_that_cannot_be_cancelled_is_reported_with_the_reason_and_where_to_look(rig):
    code, text = run_cmd_cancel(rig, QueuedRunner(rig["stage"]), lambda t: "gozer cancel exited 3")
    assert code == 3 and "could not be cancelled" in text and "gozer cancel exited 3" in text and "gozer status" in text
    e = [x for x in read_entries(rig["run"] / "ledger.jsonl") if x["data"].get("outcome") == "queued"][0]
    assert e["data"]["cancel_error"] == "gozer cancel exited 3"


def test_a_queued_exit_with_no_ticket_in_the_output_cancels_nothing_and_says_so(rig):
    cancelled = []
    code, text = run_cmd_cancel(rig, QueuedRunner(rig["stage"], text="busy\n"), lambda t: cancelled.append(t))
    assert code == 3 and cancelled == [] and "no ticket" in text


def test_an_exit_ten_that_is_not_a_queue_request_is_not_cancelled(rig):
    """A runner that reports a plain int (no output) gets no ticket to cancel."""
    cancelled = []
    code, text = run_cmd_cancel(rig, Runner(rig["stage"], code=10), lambda t: cancelled.append(t))
    assert code == 3 and cancelled == [] and "no ticket" in text


def test_the_default_runner_returns_the_exit_code_and_what_the_command_printed(tmp_path, capsys):
    script = tmp_path / "s.sh"
    script.write_text('#!/bin/sh\necho "queued: ticket Abc9"\necho "oops" >&2\nexit 10\n')
    script.chmod(0o755)
    code, text = package_cmd._default_runner([str(script)], {"PATH": "/usr/bin:/bin"})
    assert code == 10 and "queued: ticket Abc9" in text and "oops" in text
    assert "queued: ticket Abc9" in capsys.readouterr().out          # the operator still sees it as it happens


def test_a_runner_that_returns_a_bare_exit_code_still_works(rig):
    code, text, _ = run_cmd(rig, Runner(rig["stage"]))
    assert code == 0, text


def test_without_a_cancel_hook_the_command_cancels_through_gozer_cancel(rig, monkeypatch):
    asked = []
    monkeypatch.setattr(package_cmd, "cancel_ticket", lambda t: asked.append(t))
    out = io.StringIO()
    code = package_cmd.main(args(rig), out=out, runner=QueuedRunner(rig["stage"]), extra_env=rig["extra_env"],
                            hostname="testhost")
    assert code == 3 and asked == ["T0k3n"]


def test_the_default_runner_keeps_only_the_tail_of_a_long_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(package_cmd, "OUTPUT_KEEP", 200)
    script = tmp_path / "s.sh"
    script.write_text('#!/bin/sh\nfor i in $(seq 1 100); do echo "filler line $i xxxxxxxxxxxxxxxx"; done\necho "queued: ticket Late1"\n')
    script.chmod(0o755)
    code, text = package_cmd._default_runner([str(script)], {"PATH": "/usr/bin:/bin"})
    assert code == 0 and len(text) <= 200 and "queued: ticket Late1" in text and "filler line 1 " not in text


# ---- review: a server the verifier started must not outlive the boot step ------------------------------

RECORD = "verify/evidence/server.pid"


def server_like(script="sleep 300", record=None, stage=None):
    """A process in its own session standing in for the server the verifier started. With `stage`, the verifier's record
    of it is written the way verify_bundle.py writes it (pid, boot id, start time)."""
    proc = subprocess.Popen(["sh", "-c", script], start_new_session=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    boot_id, start = proc_identity(proc.pid)
    if stage is not None:
        write(stage / RECORD, record or {"pid": proc.pid, "boot_id": boot_id, "start_time": start})
    return proc


@pytest.fixture
def servers():
    made = []
    yield made
    for proc in made:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def boot_with(rig, monkeypatch, spawn, **kw):
    run_cmd(rig, step="stage")
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    out = io.StringIO()
    code = package_cmd.main(args(rig, step="boot", cache_root=str(rig["cache"]), **kw), out=out, spawn=spawn,
                            extra_env=rig["extra_env"])
    return code, out.getvalue()


def result(rig) -> dict:
    return json.loads((rig["stage"] / "test-result.json").read_text())


def test_a_server_left_behind_by_a_timed_out_check_is_stopped_before_the_step_returns(rig, monkeypatch, servers):
    def spawn(command, run_dir, env, timeout, stdout):
        servers.append(server_like(stage=rig["stage"]))
        return None, True
    code, text = boot_with(rig, monkeypatch, spawn)
    assert code == 1
    assert servers[0].wait(timeout=5) in (-signal.SIGTERM, -signal.SIGKILL)       # gone, and reaped here
    assert result(rig)["server"] == {"recorded": True, "outcome": "stopped"}
    assert not (rig["stage"] / RECORD).exists()
    assert "stopped" in text


def test_a_server_that_ignores_sigterm_is_killed_after_the_wait(rig, monkeypatch, servers):
    monkeypatch.setattr(package_cmd, "TERM_WAIT_S", 0.5)
    def spawn(command, run_dir, env, timeout, stdout):
        servers.append(server_like('trap "" TERM; while true; do sleep 1; done', stage=rig["stage"]))
        time.sleep(0.3)                                                          # let sh install its trap
        return None, True
    code, _ = boot_with(rig, monkeypatch, spawn)
    assert servers[0].wait(timeout=5) == -signal.SIGKILL
    assert result(rig)["server"]["outcome"] == "killed" and code == 1


def test_a_record_whose_process_start_time_differs_is_not_signalled(rig, monkeypatch, servers):
    """A pid can be reused. The record holds the start time, and a process that does not match it is left alone."""
    def spawn(command, run_dir, env, timeout, stdout):
        proc = server_like()
        servers.append(proc)
        boot_id, start = proc_identity(proc.pid)
        write(rig["stage"] / RECORD, {"pid": proc.pid, "boot_id": boot_id, "start_time": start + 1})
        return 0, False
    code, text = boot_with(rig, monkeypatch, spawn)
    assert servers[0].poll() is None                                              # still running: not ours to stop
    assert result(rig)["server"] == {"recorded": True, "outcome": "not the recorded process"}
    assert code == 0 and not (rig["stage"] / RECORD).exists()


def test_a_server_that_cannot_be_stopped_fails_the_step_and_says_where_to_look(rig, monkeypatch, servers):
    monkeypatch.setattr(package_cmd, "_signal_group", lambda pgid, sig: None)
    monkeypatch.setattr(package_cmd, "TERM_WAIT_S", 0.2)
    monkeypatch.setattr(package_cmd, "KILL_WAIT_S", 0.2)
    def spawn(command, run_dir, env, timeout, stdout):
        servers.append(server_like(stage=rig["stage"]))
        return 0, False
    code, text = boot_with(rig, monkeypatch, spawn)
    assert code == 1 and "still running" in text and "gozer status" in text
    assert result(rig)["server"]["outcome"] == "still running"
    assert servers[0].poll() is None


def test_a_clean_check_leaves_no_server_record_and_the_result_says_so(rig, monkeypatch):
    code, _ = boot_with(rig, monkeypatch, lambda c, r, e, t, o: (0, False))
    assert code == 0 and result(rig)["server"] == {"recorded": False}


def test_an_interrupted_check_still_stops_the_server_and_the_interrupt_goes_on(rig, monkeypatch, servers):
    def spawn(command, run_dir, env, timeout, stdout):
        servers.append(server_like(stage=rig["stage"]))
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        boot_with(rig, monkeypatch, spawn)
    assert servers[0].wait(timeout=5) in (-signal.SIGTERM, -signal.SIGKILL)


def test_sigterm_from_the_leases_timeout_stops_the_server_too_and_the_handler_is_put_back(rig, monkeypatch, servers):
    """`timeout` in the lease command sends SIGTERM to the boot step. The baseline handler only counts, so a boot step
    with no handler of its own fails this test instead of ending the test run."""
    got = []
    baseline = lambda sig, frame: got.append(sig)                         # noqa: E731
    original = signal.signal(signal.SIGTERM, baseline)
    try:
        def spawn(command, run_dir, env, timeout, stdout):
            servers.append(server_like(stage=rig["stage"]))
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(2)
            return None, True
        with pytest.raises(SystemExit) as stopped:
            boot_with(rig, monkeypatch, spawn)
        assert stopped.value.code == 128 + signal.SIGTERM                 # the conventional exit of a SIGTERM
        assert signal.getsignal(signal.SIGTERM) is baseline
    finally:
        signal.signal(signal.SIGTERM, original)
    assert servers[0].wait(timeout=5) in (-signal.SIGTERM, -signal.SIGKILL)


def test_the_handler_is_put_back_after_a_normal_step(rig, monkeypatch):
    baseline = lambda sig, frame: None                                    # noqa: E731
    original = signal.signal(signal.SIGTERM, baseline)
    try:
        boot_with(rig, monkeypatch, lambda c, r, e, t, o: (0, False))
        assert signal.getsignal(signal.SIGTERM) is baseline
    finally:
        signal.signal(signal.SIGTERM, original)


@pytest.mark.parametrize("junk", ["not json", '{"pid": "x"}', '{"pid": 0, "boot_id": "b", "start_time": 1}',
                                  '{"pid": 1, "boot_id": "b"}', '[]', '{"pid": true, "boot_id": "b", "start_time": 1}',
                                  '{"pid": 4242424, "boot_id": "b", "start_time": true}',
                                  '{"pid": 4242424, "boot_id": 7, "start_time": 1}'])
def test_a_damaged_server_record_signals_nothing_and_is_reported(rig, monkeypatch, junk):
    def spawn(command, run_dir, env, timeout, stdout):
        (rig["stage"] / RECORD).parent.mkdir(parents=True, exist_ok=True)
        (rig["stage"] / RECORD).write_text(junk)
        return 0, False
    code, text = boot_with(rig, monkeypatch, spawn)
    assert result(rig)["server"] == {"recorded": True, "outcome": "unreadable record"}
    assert "gozer status" in text


def test_a_real_timed_out_boot_check_leaves_no_server_running(rig, monkeypatch):
    """The whole path: the staged verifier runs under the real spawn, the fake server is too slow for the deadline,
    spawn_checked kills the verifier's session, and the server, which has a session of its own, is stopped by the step."""
    from package_fakes import GENERATED, PROMPT_IDS, VOCAB
    run_cmd(rig, step="stage")
    pid_file = rig["tmp"] / "server-pid.json"
    (rig["tmp"] / "server.json").write_text(json.dumps({"mode": "slow", "delay_s": 3, "vocab": VOCAB,
                                                        "prompt_ids": PROMPT_IDS, "generated_ids": GENERATED,
                                                        "pid_file": str(pid_file)}))
    hw = json.loads((rig["stage"] / "hw_test.json").read_text())
    hw["deadline_s"] = 20
    write(rig["stage"] / "hw_test.json", hw)
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    out = io.StringIO()
    server = None
    try:
        code = package_cmd.main(args(rig, step="boot", cache_root=str(rig["cache"])), out=out,
                                extra_env=rig["extra_env"])
        server = json.loads(pid_file.read_text())
        assert code == 1 and result(rig)["timed_out"] is True, out.getvalue()
        assert result(rig)["server"]["outcome"] in ("stopped", "killed")
        with pytest.raises(ProcessLookupError):
            os.killpg(server["pgid"], 0)
    finally:
        if pid_file.exists():
            try:
                os.killpg(json.loads(pid_file.read_text())["pgid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_a_record_from_another_boot_is_not_signalled_even_with_the_same_start_time(rig, monkeypatch, servers):
    """Start times count clock ticks since boot, so the same number on another boot is a different process."""
    def spawn(command, run_dir, env, timeout, stdout):
        proc = server_like()
        servers.append(proc)
        boot_id, start = proc_identity(proc.pid)
        write(rig["stage"] / RECORD, {"pid": proc.pid, "boot_id": "another-boot", "start_time": start})
        return 0, False
    boot_with(rig, monkeypatch, spawn)
    assert servers[0].poll() is None
    assert result(rig)["server"]["outcome"] == "not the recorded process"


def test_a_server_whose_leader_is_gone_but_whose_group_still_runs_is_stopped(rig, monkeypatch, servers):
    """The server's worker outlived its leader: nothing else can hold that process group id, so the members are the
    old group's and are stopped."""
    def spawn(command, run_dir, env, timeout, stdout):
        leader = server_like("sleep 300 & sleep 0.3", stage=rig["stage"])
        servers.append(leader)
        leader.wait(timeout=10)                                                  # the leader is gone, its child is not
        assert proc_identity(leader.pid) is None
        return 0, False
    code, text = boot_with(rig, monkeypatch, spawn)
    from orchard.package_cmd import _group_members
    assert result(rig)["server"]["outcome"] == "stopped"
    assert _group_members(servers[0].pid) == []


def test_a_record_whose_process_and_group_are_both_gone_is_reported_as_already_gone(rig, monkeypatch, servers):
    def spawn(command, run_dir, env, timeout, stdout):
        proc = server_like("exit 0", stage=rig["stage"])
        proc.wait(timeout=10)
        return 0, False
    boot_with(rig, monkeypatch, spawn)
    assert result(rig)["server"] == {"recorded": True, "outcome": "already gone"}
    assert not (rig["stage"] / RECORD).exists()


@pytest.mark.parametrize("which", ["pid", "group"])
def test_a_record_naming_the_commands_own_process_or_group_signals_nothing(rig, monkeypatch, which):
    """The step runs inside its own process group. A record that names it would have the step signal itself."""
    own = os.getpid() if which == "pid" else os.getpgrp()
    def spawn(command, run_dir, env, timeout, stdout):
        boot_id, start = proc_identity(own)
        write(rig["stage"] / RECORD, {"pid": own, "boot_id": boot_id, "start_time": start})
        return 0, False
    code, text = boot_with(rig, monkeypatch, spawn)                      # still here: nothing signalled this process
    assert result(rig)["server"] == {"recorded": True, "outcome": "unreadable record"}


def test_the_record_reader_refuses_the_steps_own_pid_and_group_separately(tmp_path, monkeypatch):
    """Pid and group id are told apart here by patching both, because a test run in a session of its own has them equal."""
    monkeypatch.setattr(os, "getpid", lambda: 111111)
    monkeypatch.setattr(os, "getpgrp", lambda: 222222)
    rec = tmp_path / "server.pid"
    for pid, ok in ((111111, False), (222222, False), (333333, True)):
        rec.write_text(json.dumps({"pid": pid, "boot_id": "b", "start_time": 5}))
        assert (package_cmd._read_server_record(rec) is not None) is ok, pid


def test_a_server_that_takes_a_moment_to_exit_on_sigterm_is_stopped_not_killed(rig, monkeypatch, servers):
    """It gets TERM_WAIT_S to leave on its own; only one that does not leave in that time is killed."""
    def spawn(command, run_dir, env, timeout, stdout):
        servers.append(server_like("trap 'sleep 0.6; exit 0' TERM; while true; do sleep 0.1; done", stage=rig["stage"]))
        time.sleep(0.3)                                                          # let sh install its trap
        return None, True
    boot_with(rig, monkeypatch, spawn)
    assert result(rig)["server"]["outcome"] == "stopped" and servers[0].wait(timeout=5) == 0


# ---- review: a standalone boot step is as guarded as the other steps ------------------------------------

from orchard.ledger import LedgerLocked                                              # noqa: E402

HANDOFF = "boot-handoff"
TOKEN = "0123456789abcdef0123456789abcdef"


def spawn_must_not_run(*a):
    raise AssertionError("the boot check started")


def boot_bare(rig, monkeypatch, spawn=spawn_must_not_run, **kw):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    out = io.StringIO()
    code = package_cmd.main(args(rig, step="boot", cache_root=str(rig["cache"]), **kw), out=out, spawn=spawn,
                            extra_env=rig["extra_env"])
    return code, out.getvalue()


def test_a_bare_boot_step_is_refused_while_a_supervisor_holds_the_run(rig, monkeypatch):
    run_cmd(rig, step="stage")
    with Ledger(rig["run"] / "ledger.jsonl"):                                       # the supervisor
        code, text = boot_bare(rig, monkeypatch)
    assert code == 2 and "supervisor holds" in text
    assert not (rig["stage"] / "test-result.json").exists()


def test_a_bare_boot_step_is_refused_on_a_run_that_has_not_finished(rig, monkeypatch):
    run_cmd(rig, step="stage")
    (rig["run"] / "ledger.jsonl").unlink()
    with Ledger(rig["run"] / "ledger.jsonl") as led:
        led.append("run_start", None, model="m", package=None,
                   paths={"tt_model_root": str(rig["models"]), "cache_root": str(rig["cache"])})
    code, text = boot_bare(rig, monkeypatch)
    assert code == 2 and "has not finished" in text and not (rig["stage"] / "test-result.json").exists()


def test_a_bare_boot_step_is_refused_on_an_aborted_run(rig, monkeypatch):
    run_cmd(rig, step="stage")
    with Ledger(rig["run"] / "ledger.jsonl") as led:
        led.append("decision", None, decision="abort", reason="operator")
    code, text = boot_bare(rig, monkeypatch)
    assert code == 2 and "has not finished" in text


def test_a_bare_boot_step_holds_the_run_while_it_boots_and_lets_go_afterwards(rig, monkeypatch):
    run_cmd(rig, step="stage")
    held = []
    def spawn(command, run_dir, env, timeout, stdout):
        try:
            Ledger(rig["run"] / "ledger.jsonl")
            held.append(False)
        except LedgerLocked:
            held.append(True)
        return 0, False
    code, text = boot_bare(rig, monkeypatch, spawn)
    assert code == 0, text
    assert held == [True]
    with Ledger(rig["run"] / "ledger.jsonl"):                                       # released
        pass
    boots = [e for e in read_entries(rig["run"] / "ledger.jsonl")
             if e["data"].get("decision") == "package command" and e["data"].get("step") == "boot"]
    assert len(boots) == 1


def handoff_record(rig, token=TOKEN, pid=None, **over):
    me = proc_identity(os.getpid())
    rec = {"token": token, "pid": pid or os.getpid(), "boot_id": me[0], "start_time": me[1], **over}
    write(rig["stage"] / ("." + HANDOFF), rec)


def boot_as_child(rig, monkeypatch, token=TOKEN):
    """The boot step the `package` command's lease runs: it is given the handoff token, and the parent (this test)
    holds the run's ledger."""
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "2,3")
    out = io.StringIO()
    ran = []
    code = package_cmd.main(args(rig, step="boot", cache_root=str(rig["cache"]), handoff=token), out=out,
                            spawn=lambda c, r, e, t, o: (ran.append(1) or (0, False)), extra_env=rig["extra_env"])
    return code, out.getvalue(), bool(ran)


def test_the_boot_step_of_the_lease_runs_under_its_parents_lock_with_a_valid_handoff(rig, monkeypatch):
    run_cmd(rig, step="stage")
    with Ledger(rig["run"] / "ledger.jsonl"):                                       # the parent holds the run
        handoff_record(rig)
        code, text, ran = boot_as_child(rig, monkeypatch)
    assert code == 0 and ran, text


@pytest.mark.parametrize("what", ["wrong token", "no record", "parent gone", "parent is not the lock holder",
                                  "lock not held", "damaged record"])
def test_the_boot_step_refuses_a_handoff_that_does_not_check_out(rig, monkeypatch, what):
    run_cmd(rig, step="stage")
    holder = Ledger(rig["run"] / "ledger.jsonl") if what != "lock not held" else None
    try:
        if what == "wrong token":
            handoff_record(rig, token="f" * 32)
        elif what == "parent gone":
            handoff_record(rig, start_time=1)                                       # not this process's start time
        elif what == "parent is not the lock holder":
            me = proc_identity(os.getppid())                                        # alive, with its own identity
            handoff_record(rig, pid=os.getppid(), boot_id=me[0], start_time=me[1])
        elif what == "lock not held":
            handoff_record(rig)
        elif what == "damaged record":
            (rig["stage"] / ("." + HANDOFF)).write_text("not json")
        code, text, ran = boot_as_child(rig, monkeypatch)
    finally:
        if holder is not None:
            holder.close()
    assert code == 2 and not ran and "handoff" in text
    assert not (rig["stage"] / "test-result.json").exists()


def test_a_boot_step_with_a_handoff_is_still_refused_on_an_unfinished_run(rig, monkeypatch):
    run_cmd(rig, step="stage")
    (rig["run"] / "ledger.jsonl").unlink()
    with Ledger(rig["run"] / "ledger.jsonl") as led:
        led.append("run_start", None, model="m", package=None,
                   paths={"tt_model_root": str(rig["models"]), "cache_root": str(rig["cache"])})
        handoff_record(rig)
        code, text, ran = boot_as_child(rig, monkeypatch)
    assert code == 2 and not ran and "has not finished" in text


def test_the_full_flow_hands_its_lock_to_the_lease_command_with_a_token_that_is_checked(rig):
    seen = {}
    class Spy(Runner):
        def __call__(self, argv, env):
            seen["argv"] = list(argv)
            seen["record"] = json.loads((rig["stage"] / ("." + HANDOFF)).read_text())
            return super().__call__(argv, env)
    code, text, runner = run_cmd(rig, Spy(rig["stage"]))
    assert code == 0, text
    argv = seen["argv"]
    token = argv[argv.index("--handoff") + 1]
    assert re.fullmatch(r"[0-9a-f]{32}", token) and seen["record"]["token"] == token
    assert seen["record"]["pid"] == os.getpid()
    assert (seen["record"]["boot_id"], seen["record"]["start_time"]) == proc_identity(os.getpid())
    assert not (rig["stage"] / ("." + HANDOFF)).exists()                            # taken away afterwards


def test_the_handoff_record_is_taken_away_even_when_the_lease_command_fails(rig):
    class Boom(Runner):
        def __call__(self, argv, env):
            raise RuntimeError("gozer blew up")
    with pytest.raises(RuntimeError):
        run_cmd(rig, Boom(rig["stage"]))
    assert not (rig["stage"] / ("." + HANDOFF)).exists()


def test_the_lease_command_check_allows_the_handoff_token_and_nothing_else_new(tmp_path):
    ok = package_cmd.lease_argv(tmp_path, chips=2, deadline_s=60, python=sys.executable, cache_root=tmp_path / "c",
                                handoff=TOKEN)
    package_cmd.check_lease_argv(ok, run_dir=tmp_path, python=sys.executable)
    assert ok[-2:] == ["--handoff", TOKEN]
    for bad in (ok + ["--extra", "x"], ok[:-1] + ["not a token"], ok[:-2] + ["--handoff"]):
        with pytest.raises(package_cmd.PackageCommandError):
            package_cmd.check_lease_argv(bad, run_dir=tmp_path, python=sys.executable)


def test_each_handoff_gets_a_new_unguessable_token(tmp_path):
    (tmp_path / "stage").mkdir()
    first = package_cmd._write_handoff(tmp_path / "stage")
    second = package_cmd._write_handoff(tmp_path / "stage")
    assert first != second and re.fullmatch(r"[0-9a-f]{32}", first) and re.fullmatch(r"[0-9a-f]{32}", second)
    assert json.loads((tmp_path / "stage" / ".boot-handoff").read_text())["token"] == second
