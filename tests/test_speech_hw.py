# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The command stage 2 runs under the lease (orchard/speech_hw.py): start the bundle with `tt-model serve`, run the checks of orchard/speech_check.py against
it, save the answers, and stop the bundle whatever happened. The tests use a fake `tt-model` script and a fake server on a port the system chose."""
import json
import os
import stat
import sys
from pathlib import Path

import pytest

from orchard import speech_hw as hw, speech_route as sr
from test_speech_check import Fake
from speech_fixtures import ASR_BUNDLE, TTS_BUNDLE, asr_reference, tts_reference

FAKE_TT_MODEL = """#!{python}
import sys, time
log = open({log!r}, "a")
log.write(" ".join(sys.argv[1:]) + "\\n"); log.close()
verb = sys.argv[1]
if verb == "serve":
    mode = {serve_mode!r}
    if mode == "hang":
        time.sleep(30)
    print({serve_text!r})
    sys.exit({serve_exit})
if verb == "stop":
    sys.exit({stop_exit})
sys.exit(9)
"""


def make_tt_model(tmp_path, url, *, serve_text=None, serve_exit=0, stop_exit=0, serve_mode="ok"):
    script = tmp_path / "fake-tt-model"
    script.write_text(FAKE_TT_MODEL.format(python=sys.executable, log=str(tmp_path / "tt-model.log"), serve_mode=serve_mode, serve_exit=serve_exit, stop_exit=stop_exit,
                                           serve_text=serve_text if serve_text is not None else f"model ready: {url}/v1\n"))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def calls(tmp_path):
    p = tmp_path / "tt-model.log"
    return p.read_text().splitlines() if p.exists() else []


def config(tmp_path, task, bundle, **over):
    s1, s2 = tmp_path / "stage1", tmp_path / "stage2"
    s2.mkdir(exist_ok=True)
    if task == "speech-to-text":
        ref = asr_reference(s1)
        body = {"clips": [{"name": c["name"], "path": str(s1 / "clips" / c["name"]), "text": c["text"]} for c in ref["clips"]]}
    else:
        ref = tts_reference(s1)
        body = {"prompts": [{"id": p["id"], "text": p["text"]} for p in ref["prompts"]]}
    cfg = {"schema": 1, "task": task, "bundle": bundle, "stage_dir": str(s2), "serve_timeout_s": 20, "timeout_s": 30, **body, **over}
    path = tmp_path / "speech_config.json"
    path.write_text(json.dumps(cfg))
    return path, ref, s2


def result_of(s2):
    return json.loads((s2 / sr.RESULT_FILE).read_text())


def test_a_speech_to_text_run_serves_checks_saves_and_stops(tmp_path):
    with Fake(text="I have a dream that one day") as f:
        path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, f.url)))
        assert hw.run(path) == 0
    assert calls(tmp_path) == [f"serve {ASR_BUNDLE}", f"stop {ASR_BUNDLE}"]
    result = result_of(s2)
    assert result["served"] == {"serve_exit": 0, "stop_exit": 0}
    assert result["url"] == f.url
    assert result["bundle"] == ASR_BUNDLE
    assert sr.result_problems(result, ref, s2) == []


def test_a_text_to_speech_run_saves_each_prompts_audio_and_leaves_the_scorer_transcript_empty(tmp_path):
    with Fake() as f:
        path, ref, s2 = config(tmp_path, "text-to-speech", TTS_BUNDLE, tt_model=str(make_tt_model(tmp_path, f.url)))
        assert hw.run(path) == 0
    result = result_of(s2)
    assert [p["id"] for p in result["prompts"]] == ["p0", "p1"]
    assert all((s2 / p["audio"]["file"]).is_file() for p in result["prompts"])
    assert all(p["heard"] is None for p in result["prompts"])
    problems = sr.result_problems(result, ref, s2)
    assert problems and all("scorer" in p for p in problems), problems          # only the host's scoring step is still to come
    assert calls(tmp_path)[-1] == f"stop {TTS_BUNDLE}"


def test_each_text_to_speech_prompt_is_sent_as_its_own_request(tmp_path):
    with Fake() as f:
        path, ref, s2 = config(tmp_path, "text-to-speech", TTS_BUNDLE, tt_model=str(make_tt_model(tmp_path, f.url)))
        hw.run(path)
        sent = [json.loads(b) for m, p, b in f.posts() if p == "/v1/audio/speech"]
    texts = {s.get("input") for s in sent if s.get("input")}
    assert {p["text"] for p in ref["prompts"]} <= texts


def test_the_serve_output_is_saved_as_evidence(tmp_path):
    with Fake(text="I have a dream that one day") as f:
        path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, f.url)))
        hw.run(path)
    assert f"model ready: {f.url}" in (s2 / "evidence" / "serve-output.txt").read_text()


def test_a_wrong_transcript_is_recorded_for_the_gate_and_does_not_fail_the_command(tmp_path):
    with Fake(text="nothing like it") as f:
        path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, f.url)))
        assert hw.run(path) == 0
    assert any("mlk.wav" in p for p in sr.result_problems(result_of(s2), ref, s2))


def test_a_serve_command_that_fails_still_stops_the_bundle_and_fails_the_command(tmp_path):
    path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, "http://127.0.0.1:1", serve_exit=3)))
    assert hw.run(path) != 0
    assert calls(tmp_path) == [f"serve {ASR_BUNDLE}", f"stop {ASR_BUNDLE}"]
    assert result_of(s2)["served"]["serve_exit"] == 3


def test_serve_output_with_no_address_fails_and_stops(tmp_path):
    path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, "x", serve_text="started\n")))
    assert hw.run(path) != 0
    assert calls(tmp_path)[-1] == f"stop {ASR_BUNDLE}"
    assert "address" in result_of(s2)["error"]


@pytest.mark.parametrize("port", [8000, 8100, 8101, 8102, 8103])
def test_an_address_on_a_port_the_machine_keeps_is_refused_and_the_bundle_is_stopped(tmp_path, port):
    path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, f"http://127.0.0.1:{port}")))
    assert hw.run(path) != 0
    assert calls(tmp_path)[-1] == f"stop {ASR_BUNDLE}"
    error = result_of(s2)["error"]
    assert str(port) in error and "the address in the serve output is refused" in error, "refused before any check was sent"
    assert result_of(s2).get("checks") is None


def test_an_address_that_is_not_on_this_machine_is_not_used(tmp_path):
    path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, "x", serve_text="see http://example.com:20000 and http://203.0.113.5:20000\n")))
    assert hw.run(path) != 0
    assert "address" in result_of(s2)["error"]


def test_the_first_local_address_in_the_output_is_used(tmp_path):
    with Fake(text="I have a dream that one day") as f:
        text = f"docs at http://example.com:9\nserving http://localhost:{f.port} and http://127.0.0.1:{f.port + 1}\n"
        path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, "x", serve_text=text)))
        assert hw.run(path) == 0
    assert result_of(s2)["url"] == f"http://localhost:{f.port}"


def test_a_serve_command_that_never_returns_is_stopped_at_the_deadline_and_the_bundle_is_stopped(tmp_path):
    path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, serve_timeout_s=1, tt_model=str(make_tt_model(tmp_path, "x", serve_mode="hang")))
    assert hw.run(path) != 0
    assert calls(tmp_path)[-1] == f"stop {ASR_BUNDLE}"
    assert "deadline" in result_of(s2)["error"]


def test_a_stop_that_fails_is_recorded_and_fails_the_command(tmp_path):
    with Fake(text="I have a dream that one day") as f:
        path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, f.url, stop_exit=5)))
        assert hw.run(path) != 0
    assert result_of(s2)["served"]["stop_exit"] == 5


def test_an_unreadable_clip_still_stops_the_bundle(tmp_path):
    with Fake() as f:
        path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, f.url)))
        cfg = json.loads(path.read_text())
        cfg["clips"][0]["path"] = str(tmp_path / "no-such-clip.wav")
        path.write_text(json.dumps(cfg))
        assert hw.run(path) != 0
    assert calls(tmp_path)[-1] == f"stop {ASR_BUNDLE}"
    assert "clip" in result_of(s2)["error"]


BAD_CONFIGS = {
    "not an object": lambda c: [],
    "schema": lambda c: c.update(schema=2),
    "task": lambda c: c.update(task="x"),
    "bundle not text": lambda c: c.update(bundle=3),
    "bundle with an option": lambda c: c.update(bundle="--rm"),
    "bundle with a space": lambda c: c.update(bundle="a/b c"),
    "bundle unknown": lambda c: c.update(bundle="someone/else"),
    "bundle of the other task": lambda c: c.update(bundle=TTS_BUNDLE),
    "stage dir missing": lambda c: c.pop("stage_dir"),
    "stage dir relative": lambda c: c.update(stage_dir="rel"),
    "serve timeout zero": lambda c: c.update(serve_timeout_s=0),
    "serve timeout a string": lambda c: c.update(serve_timeout_s="9"),
    "serve timeout a bool": lambda c: c.update(serve_timeout_s=True),
    "timeout huge": lambda c: c.update(timeout_s=10 ** 7),
    "no clips": lambda c: c.update(clips=[]),
    "clip text missing": lambda c: c["clips"][0].pop("text"),
    "tt_model not text": lambda c: c.update(tt_model=3),
}


@pytest.mark.parametrize("name", sorted(BAD_CONFIGS))
def test_a_bad_config_is_refused_before_anything_is_started(tmp_path, name):
    path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, "http://127.0.0.1:1")))
    cfg = json.loads(path.read_text())
    new = BAD_CONFIGS[name](cfg)
    path.write_text(json.dumps(cfg if new is None else new))
    assert hw.run(path) == 2
    assert calls(tmp_path) == []


def test_a_config_file_that_is_not_json_is_refused(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("{nope")
    assert hw.run(p) == 2


def test_a_missing_config_file_is_refused(tmp_path):
    assert hw.run(tmp_path / "none.json") == 2


def test_text_to_speech_needs_prompts(tmp_path):
    path, ref, s2 = config(tmp_path, "text-to-speech", TTS_BUNDLE, tt_model=str(make_tt_model(tmp_path, "http://127.0.0.1:1")))
    cfg = json.loads(path.read_text())
    cfg["prompts"] = []
    path.write_text(json.dumps(cfg))
    assert hw.run(path) == 2 and calls(tmp_path) == []


def test_the_command_line_runs_the_same_function(tmp_path):
    import subprocess
    with Fake(text="I have a dream that one day") as f:
        path, ref, s2 = config(tmp_path, "speech-to-text", ASR_BUNDLE, tt_model=str(make_tt_model(tmp_path, f.url)))
        proc = subprocess.run([sys.executable, str(Path(hw.__file__)), "--config", str(path)], capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert proc.returncode == 0, proc.stderr
    assert (s2 / sr.RESULT_FILE).is_file()
