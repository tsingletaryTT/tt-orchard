# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A speech run (orchard/speech_route.py, stages.py `SPEECH_TABLE`) goes from stage 0 to the bundle without booting the coder. The reference interpreter is a
fake program written under the test's directory, the bundle's server is a fake on a port the system chose, and `tt-model` is a fake script. Everything else is the
supervisor as it runs. No chip, no real `tt-model`, no network."""
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from orchard import blocked, speech_hw, speech_route as sr, supervisor
from orchard.stages import GateResult, evidence_record, spec_for
from orchard.supervisor import EXIT_READY, parse
from fakes import Crash
from run_fakes import argv
from test_speech_check import Fake, wav_bytes
from test_speech_hw import make_tt_model
from test_supervisor import Rig, Stop

pytestmark = pytest.mark.usefixtures("stub_tools")

ASR_ID = "mistralai/Voxtral-Mini-3B-2507"
TTS_ID = "mistralai/Voxtral-4B-TTS-2603"
SPOKEN = "I have a dream that one day"

FAKE_PYTHON = """#!{python}
import json, os, sys
args = sys.argv[1:]
mode = args[args.index("--mode") + 1]
cfg = json.load(open(args[args.index("--config") + 1]))
with open({log!r}, "a") as f:
    f.write(json.dumps({{"argv": args, "mode": mode, "env": sorted(os.environ)}}) + "\\n")
answers = json.load(open({answers!r}))
if answers.get("fail"):
    sys.stderr.write("the fake producer failed: " + answers["fail"])
    sys.exit(1)
texts = {{f["name"]: answers.get(f["name"], {spoken!r}) for f in cfg["files"]}}
json.dump({{"schema": 1, "engine": {{"python": sys.executable, "transformers": "5.24.0", "torch": "2.15.0+cpu", "model_id": cfg["model_id"], "decoding": "greedy"}},
           "texts": texts}}, open(args[args.index("--out") + 1], "w"))
"""


def install_fake_python(tmp_path, **answers):
    py = tmp_path / "fake-ref-python"
    (tmp_path / "answers.json").write_text(json.dumps(answers))
    py.write_text(FAKE_PYTHON.format(python=sys.executable, log=str(tmp_path / "producer.log"), answers=str(tmp_path / "answers.json"), spoken=SPOKEN))
    py.chmod(py.stat().st_mode | stat.S_IXUSR)
    return py


def producer_calls(tmp_path):
    p = tmp_path / "producer.log"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def build(tmp_path, monkeypatch, request, *, task="speech-to-text", model_id=ASR_ID, clips=True, with_python=True, answers=None, extra_inputs=(), clips_input=True):
    r = Rig(tmp_path)
    r.on_sleep = lambda: (_ for _ in ()).throw(Stop("the run waited for the operator"))
    snap = tmp_path / "hub" / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"architectures": ["VoxtralForConditionalGeneration"], "model_type": "voxtral"}))
    (snap / "README.md").write_text("---\nlicense: apache-2.0\n---\n# card\n")
    clip_dir = tmp_path / "my-clips"
    clip_dir.mkdir()
    if clips:
        (clip_dir / "mlk.wav").write_bytes(wav_bytes(1.0, rate=16000))
        (clip_dir / "mlk.txt").write_text(SPOKEN + "\n")
    py = install_fake_python(tmp_path, **(answers or {}))
    a = argv(tmp_path, r.tiers, r.chip_server.endpoint, chips=2)
    a[a.index("--model") + 1] = model_id
    inputs = ["--input", f"model={snap}", "--input", f"speech_route={task}"]
    if clips_input:
        inputs += ["--input", f"speech_clips={clip_dir}"]
    if with_python:
        inputs += ["--input", f"reference_python={py}"]
    for k, v in extra_inputs:
        inputs += ["--input", f"{k}={v}"]
    r.args = parse(a + inputs)
    r.calls = {"hw": [], "chip": []}
    r.snap, r.clip_dir, r.py, r.tmp = snap, clip_dir, py, tmp_path
    monkeypatch.setattr(supervisor.Supervisor, "model_id", model_id + "@" + "a" * 40, raising=False)
    r.fake = Fake(text=SPOKEN)
    r.fake.__enter__()
    r.tt_model = make_tt_model(tmp_path, r.fake.url)
    real_spawn = supervisor.Supervisor._spawn

    def spawn(self, n, command, chips, ids, deadline, out_path, env_extra=None, **record):
        if "speech_hw.py" not in command:
            return real_spawn(self, n, command, chips, ids, deadline, out_path, env_extra, **record)
        r.calls["chip"].append((command, list(chips)))
        cfg_path = Path(command.split("--config ")[1])
        cfg = json.loads(cfg_path.read_text())
        cfg["tt_model"] = str(r.tt_model)
        cfg_path.write_text(json.dumps(cfg))
        code = speech_hw.run(cfg_path)
        out_path.write_text(f"fake hardware output, exit {code}\n")
        return {"command": command, "returncode": 0, "timed_out": False, "seconds": 1.0, "chips": list(chips), "output": evidence_record(self.run_dir, out_path)}
    monkeypatch.setattr(supervisor.Supervisor, "_spawn", spawn)

    def bundle(self, spec, stage_dir):
        (stage_dir / "bundle").mkdir(parents=True, exist_ok=True)
        (stage_dir / "bundle" / "RESULTS.md").write_text("fake bundle\n")
        return "pass", [], GateResult(True, (), ())
    if not request.node.get_closest_marker("real_bundle"):
        monkeypatch.setattr(supervisor.Supervisor, "_bundle_body", bundle)
    request.addfinalizer(lambda: r.fake.__exit__(None, None, None))
    return r


@pytest.fixture
def asr_rig(tmp_path, monkeypatch, request):
    r = build(tmp_path, monkeypatch, request)
    yield r
    r.close()


@pytest.fixture
def tts_rig(tmp_path, monkeypatch, request):
    prompts = {"p0.wav": sr.DEFAULT_PROMPTS[0], "p1.wav": sr.DEFAULT_PROMPTS[1], "p2.wav": sr.DEFAULT_PROMPTS[2]}
    r = build(tmp_path, monkeypatch, request, task="text-to-speech", model_id=TTS_ID, answers=prompts)
    r.tt_model = make_tt_model(tmp_path, r.fake.url)
    yield r
    r.close()


def decisions(rig, name):
    return [d for d in rig.decisions() if d.get("decision") == name]


def started(rig, n):
    return [e for e in rig.entries() if e["event"] == "stage_start" and e["stage"] == n]


def test_a_speech_to_text_run_reaches_the_bundle_and_never_boots_the_coder(asr_rig):
    r = asr_rig
    assert r.run() == EXIT_READY
    assert r.m.coder_starts == 0
    ends = {n: r.ends(n)[0] for n in range(9)}
    assert [ends[n]["result"] for n in (0, 1, 2, 8)] == ["pass"] * 4
    assert [ends[n]["result"] for n in (3, 4, 5, 6, 7)] == ["skipped"] * 5
    assert "speech route" in ends[3]["reason"]
    assert ends[0]["path"] == "full-port" and not decisions(r, "pause")


def test_the_selection_decision_records_the_task_the_interpreter_and_the_script_hashes(asr_rig):
    r = asr_rig
    r.run()
    sel = decisions(r, sr.SPEECH_ROUTE_SELECTED)
    assert len(sel) == 1 and sel[0]["task"] == "speech-to-text" and sel[0]["python"] == str(r.py)
    assert set(sel[0]["scripts"]) == {"speech_hw.py", "speech_reference_run.py", "speech_route.py"} and all(len(h) == 64 for h in sel[0]["scripts"].values())
    assert sel[0]["bundle"] == "apande-TT/mistralai-voxtral-mini-3b-2507-tt"


def test_stage_0_records_the_licence_line_in_the_ledger_and_in_delta_json(asr_rig):
    r = asr_rig
    r.run()
    lic = decisions(r, "speech licence")
    assert len(lic) == 1 and lic[0]["license"]["id"] == "apache-2.0" and "did not copy or republish any weights" in lic[0]["license"]["line"]
    delta = json.loads((r.run_dir / "stages" / "0" / "delta.json").read_text())
    assert delta["speech"]["license"] == lic[0]["license"] and delta["facts"] == {"form": "full", "serving": "container", "modality": "audio"}


def test_stage_1_runs_the_producer_isolated_with_a_scrubbed_environment_and_saves_the_clips(asr_rig, monkeypatch):
    r = asr_rig
    monkeypatch.setenv("HF_TOKEN", "should-not-reach-the-producer")
    monkeypatch.setenv("HOME", str(r.tmp / "somewhere"))
    r.run()
    calls = producer_calls(r.tmp)
    assert len(calls) == 1 and calls[0]["mode"] == "asr" and calls[0]["argv"][0] == "-I"
    assert "HF_TOKEN" not in calls[0]["env"] and "HOME" not in calls[0]["env"]
    ref = json.loads((r.run_dir / "stages" / "1" / sr.REFERENCE_FILE).read_text())
    assert ref["clips"][0]["text"] == SPOKEN and ref["clips"][0]["expected"] == SPOKEN
    assert (r.run_dir / "stages" / "1" / "clips" / "mlk.wav").is_file()
    assert ref["reference"]["transformers"] == "5.24.0"


def test_stage_2_serves_the_bundle_under_a_lease_for_one_board_and_records_the_answers(asr_rig):
    r = asr_rig
    r.run()
    assert len(r.calls["chip"]) == 1 and "speech_hw.py" in r.calls["chip"][0][0]
    res = json.loads((r.run_dir / "stages" / "2" / sr.RESULT_FILE).read_text())
    assert res["served"] == {"serve_exit": 0, "stop_exit": 0} and res["bundle"] == "apande-TT/mistralai-voxtral-mini-3b-2507-tt"
    assert [c["status"] for c in res["checks"]] == ["pass"] * len(res["checks"])
    assert (r.run_dir / "stages" / "2" / "test-result.json").is_file()


def test_the_hardware_files_name_the_goal_and_the_check_on_return(asr_rig):
    r = asr_rig
    r.run()
    hand = json.loads((r.run_dir / "stages" / "2" / "handoff.json").read_text())
    assert "speech" in hand["goal"] and hand["check_on_return"] == "stages/2/test-result.json"
    test = json.loads((r.run_dir / "stages" / "2" / "hw_test.json").read_text())
    assert test["deadline_s"] == sr.CHIP_DEADLINE_S and "speech_hw.py" in test["command"]


def test_a_text_to_speech_run_scores_the_saved_audio_on_the_host_after_the_lease(tts_rig):
    r = tts_rig
    assert r.run() == EXIT_READY
    assert r.m.coder_starts == 0
    modes = [c["mode"] for c in producer_calls(r.tmp)]
    assert modes == ["score", "score"], "one calibration run in stage 1 and one scoring run in stage 2"
    res = json.loads((r.run_dir / "stages" / "2" / sr.RESULT_FILE).read_text())
    assert [p["heard"] for p in res["prompts"]] == list(sr.DEFAULT_PROMPTS)
    ref = json.loads((r.run_dir / "stages" / "1" / sr.REFERENCE_FILE).read_text())
    assert ref["calibration"][0]["wer"] == 0.0 and ref["license"]["non_commercial"] is True
    haz = json.loads((r.run_dir / "stages" / "0" / "delta.json").read_text())["hazards"]
    assert any(h["area"] == "license" and "non-commercial" in h["finding"] for h in haz)


def test_a_resume_after_stage_1_keeps_the_selection_and_does_not_run_the_producer_again(asr_rig):
    r = asr_rig
    with pytest.raises(Crash):
        r.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 1)
    assert not started(r, 2)
    assert r.run() == EXIT_READY
    assert len(decisions(r, sr.SPEECH_ROUTE_SELECTED)) == 1
    assert len(producer_calls(r.tmp)) == 1 and r.m.coder_starts == 0


def test_a_resume_after_the_hardware_phase_does_not_serve_the_bundle_again(asr_rig):
    r = asr_rig
    with pytest.raises(Crash):
        r.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 2)
    assert r.run() == EXIT_READY
    assert len(r.calls["chip"]) == 1


def test_a_missing_reference_interpreter_fails_stage_0_with_the_fix(tmp_path, monkeypatch, request):
    r = build(tmp_path, monkeypatch, request, with_python=False)
    try:
        with pytest.raises(Stop):
            r.run()
        end = r.ends(0)[0]
        assert end["result"] == "fail" and "reference_python" in " ".join(end["reasons"])
        assert r.m.coder_starts == 0
    finally:
        r.close()


def failing_stage0(tmp_path, monkeypatch, request, **kw):
    r = build(tmp_path, monkeypatch, request, **kw)
    try:
        with pytest.raises(Stop):
            r.run()
        return r.ends(0)[0], r.m.coder_starts
    finally:
        r.close()


def test_no_clips_input_fails_stage_0_and_names_the_input(tmp_path, monkeypatch, request):
    end, coders = failing_stage0(tmp_path, monkeypatch, request, clips_input=False)
    assert end["result"] == "fail" and "speech_clips" in " ".join(end["reasons"]) and coders == 0


def test_a_clips_directory_with_no_clips_fails_stage_0(tmp_path, monkeypatch, request):
    end, _ = failing_stage0(tmp_path, monkeypatch, request, clips=False)
    assert end["result"] == "fail" and "no clips" in " ".join(end["reasons"])


def test_a_model_with_no_known_bundle_fails_stage_0(tmp_path, monkeypatch, request):
    end, _ = failing_stage0(tmp_path, monkeypatch, request, model_id="someone/else-voxtral")
    assert end["result"] == "fail" and "bundle" in " ".join(end["reasons"])


def test_a_producer_that_fails_fails_stage_1_with_the_end_of_its_stderr(tmp_path, monkeypatch, request):
    r = build(tmp_path, monkeypatch, request, answers={"fail": "no weights in the snapshot"})
    try:
        with pytest.raises(Stop):
            r.run()
        end = r.ends(1)[0]
        assert end["result"] == "fail" and "no weights in the snapshot" in " ".join(end["reasons"])
        assert not started(r, 2) and r.calls["chip"] == []
    finally:
        r.close()


def test_a_bundle_that_transcribes_wrongly_fails_stage_2_and_names_the_clip(tmp_path, monkeypatch, request):
    r = build(tmp_path, monkeypatch, request)
    r.fake.cfg["text"] = "nothing like the words in the clip at all"
    try:
        with pytest.raises(Stop):
            r.run()
        end = r.ends(2)[0]
        assert end["result"] == "fail" and "mlk.wav" in " ".join(end["reasons"])
    finally:
        r.close()


def test_a_bundle_that_will_not_stop_fails_stage_2(tmp_path, monkeypatch, request):
    r = build(tmp_path, monkeypatch, request)
    r.tt_model = make_tt_model(tmp_path, r.fake.url, stop_exit=7)
    try:
        with pytest.raises(Stop):
            r.run()
        assert "stop" in " ".join(r.ends(2)[0]["reasons"])
    finally:
        r.close()


def test_a_hardware_test_that_fails_stops_stage_2_with_its_problem(tmp_path, monkeypatch, request):
    r = build(tmp_path, monkeypatch, request)
    real = supervisor.Supervisor._spawn

    def bad(self, n, command, chips, ids, deadline, out_path, env_extra=None, **record):
        out_path.write_text("died\n")
        return {"command": command, "returncode": 1, "timed_out": False, "seconds": 1.0, "chips": list(chips), "output": evidence_record(self.run_dir, out_path)}
    monkeypatch.setattr(supervisor.Supervisor, "_spawn", bad)
    try:
        with pytest.raises(Stop):
            r.run()
        assert "exited with code 1" in " ".join(r.ends(2)[0]["reasons"])
    finally:
        r.close()


def test_a_resume_with_a_different_speech_route_input_is_refused(asr_rig):
    r = asr_rig
    with pytest.raises(Crash):
        r.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 0)
    r.args.input = [i.replace("speech_route=speech-to-text", "speech_route=text-to-speech") for i in r.args.input]
    with pytest.raises(ValueError, match="started with speech_route"):
        r.run()


def test_a_saved_clip_that_changes_between_stages_fails_stage_2_on_the_reference_hash(asr_rig):
    r = asr_rig
    with pytest.raises(Crash):
        r.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 1)
    (r.run_dir / "stages" / "1" / "clips" / "mlk.wav").write_bytes(b"edited after the reference was made")
    with pytest.raises(Stop):
        r.run()
    assert r.ends(2)[0]["result"] == "fail" and "stage 1" in " ".join(r.ends(2)[0]["reasons"])
    assert r.calls["chip"] == []


def test_a_speech_run_never_asks_the_coder_for_a_port_or_a_repair(asr_rig):
    r = asr_rig
    r.run()
    assert r.m.coder_starts == 0
    assert not [e for e in r.entries() if e["event"] in ("request", "turn", "model_request")]


def test_no_speech_stage_calls_the_packager_or_an_upload(asr_rig):
    r = asr_rig
    r.run()
    for e in r.entries():
        text = json.dumps(e)
        for word in ("tt-orchard can", "hf upload", "tt-model push", "tt-model publish", "upload_file", "upload_folder"):
            assert word not in text, (word, e["event"])


# ---- the operator bundle ----------------------------------------------------------------------------------------------------

LICENCE_SENTENCES = {
    "asr": "Voxtral-Mini-3B-2507 is used under its licence, apache-2.0, as the bundle's card states it. This run served the bundle "
           "apande-TT/mistralai-voxtral-mini-3b-2507-tt and did not copy or republish any weights.",
    "tts": "Voxtral-4B-TTS-2603 is used under its licence, cc-by-nc-4.0 (non-commercial use only), as the bundle's card states it. This run served the bundle "
           "apande-TT/mistralai-voxtral-4b-tts-2603-tt and did not copy or republish any weights.",
}


def bundle_files(rig):
    d = rig.run_dir / "stages" / "8" / "bundle"
    return (d / "RESULTS.md").read_text(), (d / "RISKS.md").read_text()


@pytest.mark.real_bundle
def test_the_speech_to_text_bundle_holds_the_licence_line_in_results_and_in_risks(asr_rig):
    r = asr_rig
    assert r.run() == EXIT_READY
    results, risks = bundle_files(r)
    assert LICENCE_SENTENCES["asr"] in results and LICENCE_SENTENCES["asr"] in risks
    assert "## Speech model on a chip" in results and "## Speech run" in risks
    assert "mlk.wav" in results and SPOKEN in results
    assert "stages/1/speech_reference.json" in results and "stages/2/speech_result.json" in results


@pytest.mark.real_bundle
def test_the_text_to_speech_bundle_holds_the_non_commercial_licence_line(tts_rig):
    r = tts_rig
    assert r.run() == EXIT_READY
    results, risks = bundle_files(r)
    assert LICENCE_SENTENCES["tts"] in results and LICENCE_SENTENCES["tts"] in risks
    assert "non-commercial" in risks and "p0" in results and sr.DEFAULT_PROMPTS[0] in results


@pytest.mark.real_bundle
def test_the_bundle_says_the_bars_are_provisional_and_what_was_not_checked(asr_rig):
    r = asr_rig
    r.run()
    results, risks = bundle_files(r)
    assert "provisional" in risks and "What these numbers do not show" in results
    assert "speed" in results and "/v1/chat/completions" in results


def rebuild_bundle(run_dir, tmp_path):
    """Run the real builder again on a finished run, as the supervisor's stage 8 does."""
    import shutil
    import subprocess
    script = tmp_path / "build_bundle_copy.py"
    checkout = Path(supervisor.__file__).resolve().parent.parent
    shutil.copyfile(checkout / "orchard" / "skills" / "operator-bundle-templates" / "build_bundle.py", script)
    (tmp_path / "bundle_config_copy.json").write_text(json.dumps({"run_dir": str(run_dir), "orchard_dir": str(checkout)}))
    stage8 = run_dir / "stages" / "8"
    shutil.copyfile(tmp_path / "bundle_config_copy.json", stage8 / "bundle_config.json")
    shutil.copyfile(script, stage8 / "build_bundle.py")
    proc = subprocess.run([sys.executable, str(stage8 / "build_bundle.py")], cwd=run_dir, capture_output=True, text=True, timeout=300,
                          env={"PATH": os.environ["PATH"], "PYTHONPATH": str(checkout)})
    assert proc.returncode == 0, proc.stderr[-1500:]


@pytest.mark.real_bundle
def test_a_licence_record_that_differs_from_the_bundle_table_is_a_risk_and_the_table_line_is_printed(asr_rig):
    r = asr_rig
    assert r.run() == EXIT_READY
    ref_path = r.run_dir / "stages" / "1" / sr.REFERENCE_FILE
    ref = json.loads(ref_path.read_text())
    ref["license"]["id"] = "mit"
    ref["license"]["line"] = "tampered line"
    ref_path.write_text(json.dumps(ref))
    rebuild_bundle(r.run_dir, r.tmp)
    results, risks = bundle_files(r)
    assert LICENCE_SENTENCES["asr"] in results and "tampered line" not in results and "tampered line" not in risks
    assert "licence recorded in stage 1 differs from the bundle table" in risks
