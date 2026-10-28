# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""An audio run whose model the harness's own runtime serves goes past stage 0 (orchard/audio_runtime.py, orchard/audio_stages.py, stages.py
`AUDIO_TABLE`). The CPU commands of stage 1 and the chip command of stage 2 are replaced by fakes that write recorded runs; everything else is the
supervisor as it runs."""
import json
import os
import stat
from pathlib import Path

import pytest

from orchard import audio_cli, audio_plan, audio_remote, audio_runtime, audio_stages as st, blocked, status, supervisor
from orchard.stages import AUDIO_TABLE, GateResult, spec_for
from orchard.supervisor import EXIT_BLOCKED, EXIT_READY, Control, parse
from run_fakes import DELTA, FILES, argv, bringup
from test_audio_chip_config import REAL_CONFIG
from test_audio_stages import REV, music, make_run, twin
from test_supervisor import Rig, Stop

pytestmark = pytest.mark.usefixtures("stub_tools")

AUDIO = {(0, "run"): {**FILES[(0, "run")], "delta.json": {**DELTA, "class": "weights-only", "facts": {"form": "full", "modality": "audio"}}}}
MODEL = "stabilityai/stable-audio-3-small-music@" + REV
BASE = "stabilityai/stable-audio-3-small-music-base@" + REV


@pytest.fixture
def arig(tmp_path, monkeypatch):
    r = Rig(tmp_path)
    r.on_sleep = lambda: (_ for _ in ()).throw(Stop("the run waited for the operator"))
    r.script = lambda req: bringup(req, overrides=AUDIO)
    # the snapshots the run reads: the model's and the base's, each with a supported model_config.json
    snaps = {}
    for key, name in (("model", "small-music"), ("base", "small-music-base")):
        d = tmp_path / "hub" / name / "snapshots" / REV
        d.mkdir(parents=True)
        (d / "model_config.json").write_text(json.dumps(REAL_CONFIG))
        snaps[key] = d
    a = argv(tmp_path, r.tiers, r.chip_server.endpoint, chips=2)
    a[a.index("--model") + 1] = "stabilityai/stable-audio-3-small-music"
    r.args = parse(a + ["--input", f"model={snaps['model']}", "--input", f"base={snaps['base']}"])
    r.calls = {"cpu": [], "chip": []}
    monkeypatch.setattr(supervisor.Supervisor, "_audio_models", lambda self: (MODEL, BASE))

    def cpu(argv_, env, timeout_s, log):
        r.calls["cpu"].append((argv_, env))
        model = argv_[argv_.index("--model") + 1]
        out = Path(argv_[argv_.index("--out") + 1])
        host = argv_[0].endswith(".tenstorrent-venv/bin/python")
        seed = {"small-music": 1, "small-music-base": 2}[model]
        make_run(out, model, wave=music(seed=seed) + (0.002 if host else 0), seed=seed, commit=supervisor.audio_reference_env.LIBRARY_COMMIT,
                 config_sha256=audio_plan.snapshot_identity(snaps["model" if model == "small-music" else "base"])["config_sha256"])
        return 0

    real_spawn = supervisor.Supervisor._spawn

    def spawn(self, n, command, chips, ids, deadline, out_path, env_extra=None, **record):
        if "audio_chip_test.py" not in command:         # a text run's hardware test runs as it does everywhere else
            return real_spawn(self, n, command, chips, ids, deadline, out_path, env_extra, **record)
        r.calls["chip"].append((command, list(chips)))
        cfg_path = Path(command.split("--config ")[1])
        cfg = json.loads(cfg_path.read_text())
        run_dir = cfg_path.parent.parent.parent
        for item in cfg["runs"]:
            ref = run_dir / "stages" / "1" / "reference" / item["model"]
            dst = cfg_path.parent / "chip" / item["name"]
            twin(ref, dst, noise=0.001, commit=supervisor.audio_reference_env.LIBRARY_COMMIT)
            (dst / "chip.json").write_text(json.dumps({"device": "fake chip", "total_s": 9.0, "timings": {}}))
        out_path.write_text("fake chip output\n")
        from orchard.stages import evidence_record
        return {"command": command, "returncode": 0, "timed_out": False, "seconds": 1.0, "chips": list(chips), "output": evidence_record(run_dir, out_path)}

    monkeypatch.setattr(st, "subprocess_runner", lambda env: cpu)
    monkeypatch.setattr(supervisor.Supervisor, "_spawn", spawn)

    def in_process(python, verb, args, *, timeout_s, env=None):
        """The numeric verbs run in this process (the real interpreter hop is tested in test_audio_remote.py)."""
        out = audio_cli.dispatch(verb, args)
        if "error" in out:
            raise audio_plan.AudioStageError(out["error"])
        return out["result"]
    monkeypatch.setattr(audio_remote, "call", in_process)

    r.real_bundle_body = supervisor.Supervisor._bundle_body

    def bundle(self, spec, stage_dir):
        (stage_dir / "bundle").mkdir(parents=True, exist_ok=True)
        (stage_dir / "bundle" / "RESULTS.md").write_text("fake bundle\n")
        return "pass", [], GateResult(True, (), ())
    monkeypatch.setattr(supervisor.Supervisor, "_bundle_body", bundle)
    yield r
    r.close()


def make_environment(rig):
    paths = audio_runtime.audio_paths(rig.home)
    for exe in (paths.reference_python, paths.chip_python):
        Path(exe).parent.mkdir(parents=True, exist_ok=True)
        Path(exe).write_text("#!/bin/sh\n")
        os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR)
    lib = Path(paths.src_dir) / "stable_audio_3"
    lib.mkdir(parents=True, exist_ok=True)
    (lib / "model.py").write_text("")
    return paths


def started(rig, n):
    return [e for e in rig.entries() if e["event"] == "stage_start" and e["stage"] == n]


def decisions(rig, name):
    return [d for d in rig.decisions() if d.get("decision") == name]


def test_a_supported_audio_run_selects_the_runtime_and_reaches_the_bundle(arig):
    make_environment(arig)
    assert arig.run() == EXIT_READY
    sel = decisions(arig, audio_runtime.AUDIO_RUNTIME_SELECTED)
    assert len(sel) == 1 and sel[0]["runtime"] == audio_runtime.RUNTIME and set(sel[0]["scripts"]) >= {"chip_dit.py", "chip_generate.py"}
    assert not decisions(arig, "pause")
    ends = {n: arig.ends(n)[0] for n in range(9)}
    assert ends[1]["result"] == "pass" and ends[2]["result"] == "pass" and ends[8]["result"] == "pass"
    assert [ends[n]["result"] for n in (3, 4, 5, 6, 7)] == ["skipped"] * 5
    assert "audio runtime path" in ends[3]["reason"]


def test_the_stage_commands_are_the_ones_the_design_says(arig):
    make_environment(arig)
    arig.run()
    paths = audio_runtime.audio_paths(arig.home)
    assert [(a[0], a[a.index("--model") + 1]) for a, _ in arig.calls["cpu"]] == [
        (paths.reference_python, "small-music"), (paths.chip_python, "small-music"), (paths.reference_python, "small-music-base")]
    assert len(arig.calls["chip"]) == 1
    command, chips = arig.calls["chip"][0]
    assert command.endswith("audio_chip_config.json") and "audio_chip_test.py" in command and len(chips) == 2


def test_the_audio_stages_do_not_boot_the_coder_on_a_resume(arig):
    """Stage 0 of a fresh run still uses the coder. A run resumed after a pause at stage 1 has given the coder back, and the audio stages leave it down."""
    with pytest.raises(Stop):                      # the environment is missing: stage 0 runs, then the run pauses before stage 1
        arig.run()
    pause = decisions(arig, "pause")[-1]["reason"]
    assert pause.startswith(supervisor.AUDIO_ENV_MISSING) and "audio-reference-env" in pause
    starts_after_stage_0 = arig.m.coder_starts
    assert starts_after_stage_0 == 1 and not started(arig, 1)
    make_environment(arig)
    seen = []

    def resume():
        seen.append(1)
        if len(seen) == 1:
            Control(arig.run_dir).write("resume")
        else:
            raise Stop("a second wait")
    arig.on_sleep = resume
    assert arig.run() == EXIT_READY
    assert arig.m.coder_starts == starts_after_stage_0, "the audio stages must not start the coder again"
    assert len(started(arig, 2)) == 1 and arig.ends(2)[0]["result"] == "pass"
    assert any(d.get("decision") == "hardware phase" and d["action"] == "use_free" for d in arig.decisions())


def test_a_missing_environment_pauses_with_a_reason_that_classifies_as_the_audio_stop(arig):
    arig.args.unattended = True
    assert arig.run() == EXIT_BLOCKED
    assert decisions(arig, "blocked")[-1]["code"] == "audio-path-unbuilt"
    assert status.classify_pause({"reason": supervisor.AUDIO_ENV_MISSING + "x"})[0] == "audio_unbuilt"
    assert blocked.block_code({"reason": supervisor.AUDIO_ENV_MISSING + "x"}) == "audio-path-unbuilt"


def test_a_model_the_runtime_does_not_implement_still_stops_before_stage_1(arig):
    make_environment(arig)
    cfg = json.loads(json.dumps(REAL_CONFIG))
    cfg["model"]["diffusion"]["config"]["conformer"] = True
    snap = [v.split("=", 1)[1] for v in arig.args.input if v.startswith("model=")][0]
    (Path(snap) / "model_config.json").write_text(json.dumps(cfg))
    with pytest.raises(Stop):
        arig.run()
    assert decisions(arig, "pause")[-1]["reason"] == supervisor.AUDIO_UNBUILT
    assert not decisions(arig, audio_runtime.AUDIO_RUNTIME_SELECTED) and not started(arig, 1)


def test_a_text_run_never_selects_the_runtime(arig):
    make_environment(arig)
    arig.script = lambda req: bringup(req)
    assert arig.run() == EXIT_READY
    assert not decisions(arig, audio_runtime.AUDIO_RUNTIME_SELECTED)
    assert spec_for(2, "weights-only", None, "weights-only").name != AUDIO_TABLE[2].name


def test_a_failed_chip_test_fails_the_stage_and_names_the_output(arig, monkeypatch):
    make_environment(arig)
    def failing(self, n, command, chips, ids, deadline, out_path, env_extra=None, **record):
        out_path.write_text("boom\n")
        from orchard.stages import evidence_record
        return {"command": command, "returncode": 1, "timed_out": False, "seconds": 1.0, "chips": list(chips),
                "output": evidence_record(self.run_dir, out_path)}
    monkeypatch.setattr(supervisor.Supervisor, "_spawn", failing)
    with pytest.raises(Stop):
        arig.run()
    end = arig.ends(2)[0]
    assert end["result"] == "fail" and any("hw-test-output.txt" in r for r in end["reasons"])
    assert not started(arig, 8)


def test_a_failed_cpu_run_fails_stage_1_and_leaves_no_chip_test(arig, monkeypatch):
    make_environment(arig)
    monkeypatch.setattr(st, "subprocess_runner", lambda env: (lambda a, e, t, l: 2))
    with pytest.raises(Stop):
        arig.run()
    assert arig.ends(1)[0]["result"] == "fail" and not started(arig, 2) and arig.calls["chip"] == []


def test_stage_1_fills_the_reference_cache_and_a_second_run_uses_it(arig, tmp_path):
    make_environment(arig)
    assert arig.run() == EXIT_READY
    cache = Path(arig.args.run_dir).parent / "cache" / "audio-reference"
    assert (cache / "small-music" / "manifest.json").is_file() and (cache / "small-music-base" / "manifest.json").is_file()
    snap = [v.split("=", 1)[1] for v in arig.args.input if v.startswith("model=")][-1]
    assert st.cached_run(cache, "small-music", audio_plan.snapshot_identity(snap), supervisor.audio_reference_env.LIBRARY_COMMIT) is not None


def test_every_audio_stage_spec_needs_no_coder_and_the_text_table_still_does():
    assert all(not spec.needs_coder for spec in AUDIO_TABLE.values())
    assert spec_for(2, "weights-only").needs_coder and spec_for(1, "weights-only").needs_coder


def build_supervisor(rig, ledger):
    return supervisor.build(rig.args, ledger, adapter=rig.adapter_cls(rig.m, owner_pid=100), coder=rig.coder_cls(rig.m), versions={"tt_model": "test"},
                            clock=rig.clock, sleep=rig.sleep, disk_usage=lambda p: rig.usage(p), home=rig.home, containers=rig.containers)


def test_the_audio_body_refuses_lab_mode_and_a_ledger_without_a_selection(arig, tmp_path):
    from orchard.ledger import Ledger
    from orchard.stages import AUDIO_STAGE_1
    with Ledger(arig.run_dir / "ledger.jsonl") as led:
        sup = build_supervisor(arig, led)
        stage_dir = arig.run_dir / "stages" / "1"
        status_, reasons, gate = sup._audio_body(AUDIO_STAGE_1, stage_dir, False)
        assert status_ == "fail" and "no audio runtime selection" in reasons[0] and gate is None
        sup.lab = object()
        status_, reasons, gate = sup._audio_body(AUDIO_STAGE_1, stage_dir, False)
        assert status_ == "fail" and "lab mode" in reasons[0]


# ---- cases added after the first mutation run ------------------------------------------------------------------------------------

def selected_ledger(rig, led, **override):
    paths = audio_runtime.audio_paths(rig.home)
    rec = {**paths.record(), **override}
    led.append("decision", 1, decision=audio_runtime.AUDIO_RUNTIME_SELECTED, runtime=audio_runtime.RUNTIME, paths=rec, scripts={}, model=MODEL)


def test_a_coder_booted_for_stage_0_is_adopted_by_the_audio_stages_and_one_that_was_released_is_not(arig, monkeypatch):
    make_environment(arig)
    seen = []
    real = supervisor.Supervisor._coder_in_use
    monkeypatch.setattr(supervisor.Supervisor, "_coder_in_use", lambda self: seen.append(real(self)) or seen[-1])
    arig.run()
    assert seen and all(seen[:1]), "stage 1 of a fresh run finds the coder stage 0 booted still in use"


def test_a_second_pass_through_stage_1_does_not_select_again(arig, monkeypatch):
    make_environment(arig)
    good = st.subprocess_runner
    monkeypatch.setattr(st, "subprocess_runner", lambda env: (lambda a, e, t, l: 2))
    with pytest.raises(Stop):
        arig.run()
    monkeypatch.setattr(st, "subprocess_runner", good)
    seen = []

    def resume():
        seen.append(1)
        if len(seen) == 1:
            Control(arig.run_dir).write("resume")
        else:
            raise Stop("a second wait")
    arig.on_sleep = resume
    arig.run()
    assert len(decisions(arig, audio_runtime.AUDIO_RUNTIME_SELECTED)) == 1


def test_stage_1_and_stage_2_do_not_repeat_finished_work_on_a_resume(arig):
    from orchard.ledger import Ledger
    from orchard.stages import AUDIO_STAGE_1, AUDIO_STAGE_2
    make_environment(arig)
    arig.run()
    with Ledger(arig.run_dir / "ledger.jsonl") as led:
        sup = build_supervisor(arig, led)
        cpu_before, chip_before = len(arig.calls["cpu"]), len(arig.calls["chip"])
        assert sup._audio_body(AUDIO_STAGE_1, arig.run_dir / "stages" / "1", True)[0] == "pass"
        assert sup._audio_body(AUDIO_STAGE_2, arig.run_dir / "stages" / "2", True)[0] == "pass"
        assert (len(arig.calls["cpu"]), len(arig.calls["chip"])) == (cpu_before, chip_before)


def test_a_chip_command_the_runner_refuses_fails_the_stage_before_any_lease(arig):
    from orchard.ledger import Ledger
    from orchard.stages import AUDIO_STAGE_2
    with Ledger(arig.run_dir / "ledger.jsonl") as led:
        selected_ledger(arig, led, chip_python="/bin/python $(id)")
        sup = build_supervisor(arig, led)
        (arig.run_dir / "stages" / "2").mkdir(parents=True)
        status_, reasons, _ = sup._audio_body(AUDIO_STAGE_2, arig.run_dir / "stages" / "2", False)
    assert status_ == "fail" and "refused" in reasons[0] and arig.calls["chip"] == []


def test_the_audio_children_get_the_offline_cache_environment_and_no_token(arig, monkeypatch, tmp_path):
    """The environment the numeric verb hands to the CPU runs: offline Hub, the cache directory, no token, no chip."""
    from orchard.ledger import Ledger
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    with Ledger(arig.run_dir / "ledger.jsonl") as led:
        sup = build_supervisor(arig, led)
        env = sup._audio_env()
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1" and env["HF_HOME"] == sup.paths.hf_home
    assert "HF_TOKEN" not in env and "hf_secret_value" not in " ".join(env.values())


def test_a_repository_the_planners_do_not_know_is_not_selected_even_with_a_supported_configuration(arig):
    """The configuration check passes for any audio repository with this network, but the planners know two repositories; selecting another would
    record a choice that stage 1 cannot carry out, and a retry could never reach the unsupported-audio pause."""
    make_environment(arig)
    arig.args.model = "someone/else-audio-model"
    with pytest.raises(Stop):
        arig.run()
    assert decisions(arig, "pause")[-1]["reason"] == supervisor.AUDIO_UNBUILT
    assert not decisions(arig, audio_runtime.AUDIO_RUNTIME_SELECTED) and not started(arig, 1)


def test_a_base_without_a_local_snapshot_has_no_control_and_the_run_still_reaches_the_bundle(arig):
    make_environment(arig)
    arig.args.input = [v for v in arig.args.input if not v.startswith("base=")]
    assert arig.run() == EXIT_READY
    assert [a[a.index("--model") + 1] for a, _ in arig.calls["cpu"]] == ["small-music", "small-music"]
    cfg = json.loads(next(arig.run_dir.glob("stages/2/audio_chip_config.json")).read_text())
    assert {r["model"] for r in cfg["runs"]} == {"small-music"}


def test_an_audio_runs_bundle_builder_is_the_running_checkouts_and_a_text_runs_is_the_recorded_one(arig, tmp_path):
    import dataclasses
    from orchard.ledger import Ledger
    from orchard.stages import STAGES
    recorded = tmp_path / "recorded-checkout"
    builder = recorded / "orchard" / "skills" / "operator-bundle-templates"
    builder.mkdir(parents=True)
    (builder / "build_bundle.py").write_text("raise SystemExit(3)  # the recorded checkout's builder\n")
    spec = STAGES[8]
    with Ledger(arig.run_dir / "ledger.jsonl") as led:
        sup = build_supervisor(arig, led)
        sup.paths = dataclasses.replace(sup.paths, orchard_dir=str(recorded))
        stage_dir = arig.run_dir / "stages" / "8"
        arig.real_bundle_body(sup, spec, stage_dir)
        assert "the recorded checkout's builder" in (stage_dir / "build_bundle.py").read_text()          # a text run keeps the recorded checkout
        selected_ledger(arig, led)
        arig.real_bundle_body(sup, spec, stage_dir)
        running = Path(supervisor.__file__).resolve().parent
        assert (stage_dir / "build_bundle.py").read_bytes() == (running / "skills" / "operator-bundle-templates" / "build_bundle.py").read_bytes()
        assert json.loads((stage_dir / "bundle_config.json").read_text())["orchard_dir"] == str(running.parent)


def test_a_base_that_is_another_revision_of_the_models_own_repository_does_not_replace_the_models_expectation(arig):
    from orchard.ledger import Ledger
    base_snapshot = [v.split("=", 1)[1] for v in arig.args.input if v.startswith("base=")][-1]
    with Ledger(arig.run_dir / "ledger.jsonl") as led:
        sup = build_supervisor(arig, led)
        model_snapshot = sup.inputs["model"]
        sup.inputs["base"] = str(Path(base_snapshot).parent / "another-revision")
        (Path(sup.inputs["base"])).mkdir()
        (Path(sup.inputs["base"]) / "model_config.json").write_text(json.dumps({**REAL_CONFIG, "x": 1}))
        expected, base = sup._audio_expected("stabilityai/stable-audio-3-small-music@" + REV, "stabilityai/stable-audio-3-small-music@another-revision")
    assert expected == {"small-music": audio_plan.snapshot_identity(model_snapshot)} and base is None
