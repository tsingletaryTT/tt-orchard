# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What stage 2 assembles around the in-lease command (orchard/speech_route.py): its config, its command, the producer call and the scoring merge."""
import json
import os
import shlex
import stat
import sys
from pathlib import Path

import pytest

from orchard import speech_hw, speech_route as sr
from speech_fixtures import ASR_BUNDLE, TTS_BUNDLE, asr_reference, tts_reference, tts_result


def test_the_hardware_config_for_speech_to_text_names_the_saved_clips_and_their_reference_transcripts(tmp_path):
    ref = asr_reference(tmp_path / "s1")
    cfg = sr.hw_config(ref, tmp_path / "s1", tmp_path / "s2")
    assert speech_hw.config_problems(cfg) == []
    assert cfg["bundle"] == ASR_BUNDLE and cfg["task"] == "speech-to-text" and cfg["stage_dir"] == str((tmp_path / "s2").resolve())
    assert cfg["clips"] == [{"name": "mlk.wav", "path": str((tmp_path / "s1" / "clips" / "mlk.wav").resolve()), "text": ref["clips"][0]["text"]}]
    assert "tt_model" not in cfg


def test_the_hardware_config_for_text_to_speech_names_the_prompts(tmp_path):
    ref = tts_reference(tmp_path / "s1")
    cfg = sr.hw_config(ref, tmp_path / "s1", tmp_path / "s2")
    assert speech_hw.config_problems(cfg) == []
    assert cfg["bundle"] == TTS_BUNDLE and [p["id"] for p in cfg["prompts"]] == ["p0", "p1"]


def test_a_reference_with_no_known_bundle_gives_no_hardware_config(tmp_path):
    ref = asr_reference(tmp_path / "s1")
    ref["model"] = "someone/else"
    with pytest.raises(sr.SpeechRouteError, match="bundle"):
        sr.hw_config(ref, tmp_path / "s1", tmp_path / "s2")


def test_the_hardware_command_runs_the_in_lease_script_with_the_config(tmp_path):
    cmd = shlex.split(sr.hw_command(tmp_path / "s2"))
    assert cmd[0] == sys.executable and Path(cmd[1]) == Path(speech_hw.__file__)
    assert cmd[2:] == ["--config", str((tmp_path / "s2").resolve() / "speech_config.json")]


def test_the_hardware_command_has_no_upload_packaging_or_publish_word(tmp_path):
    text = sr.hw_command(tmp_path / "s2")
    for word in ("upload", "push", "publish", "can", "hf "):
        assert word not in text.split(), word


def test_the_score_config_lists_each_saved_audio_file(tmp_path):
    ref = tts_reference(tmp_path / "s1")
    s2 = tmp_path / "s2"
    result = tts_result(s2, ref)
    cfg = sr.score_config(result, s2)
    assert srr_ok(cfg) and [f["name"] for f in cfg["files"]] == ["p0.wav", "p1.wav"]
    assert cfg["model_id"] == sr.SCORER_MODEL and cfg["model_dir"] is None


def srr_ok(cfg):
    from orchard import speech_reference_run as srr
    return srr._config_problems(cfg) == []


def test_a_prompt_with_no_audio_is_not_scored(tmp_path):
    ref = tts_reference(tmp_path / "s1")
    s2 = tmp_path / "s2"
    result = tts_result(s2, ref)
    result["prompts"][1]["audio"] = None
    assert [f["name"] for f in sr.score_config(result, s2)["files"]] == ["p0.wav"]


def test_no_audio_at_all_is_an_error(tmp_path):
    ref = tts_reference(tmp_path / "s1")
    result = tts_result(tmp_path / "s2", ref)
    for p in result["prompts"]:
        p["audio"] = None
    with pytest.raises(sr.SpeechRouteError, match="no audio"):
        sr.score_config(result, tmp_path / "s2")


def test_scores_are_merged_by_audio_file_name(tmp_path):
    ref = tts_reference(tmp_path / "s1")
    result = tts_result(tmp_path / "s2", ref, heard={"p0": "x", "p1": "y"})
    for p in result["prompts"]:
        p["heard"] = None
    merged = sr.merge_scores(result, {"p0.wav": "first words", "p1.wav": "second words"})
    assert [p["heard"] for p in merged["prompts"]] == ["first words", "second words"]
    assert result["prompts"][0]["heard"] is None, "the input is not changed"


def test_a_missing_score_is_an_error_naming_the_prompt(tmp_path):
    ref = tts_reference(tmp_path / "s1")
    result = tts_result(tmp_path / "s2", ref)
    with pytest.raises(sr.SpeechRouteError, match="p1"):
        sr.merge_scores(result, {"p0.wav": "first"})


def fake_python(tmp_path, body):
    p = tmp_path / "fake-python"
    p.write_text(f"#!{sys.executable}\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


def test_the_producer_runner_returns_the_exit_code_and_the_end_of_stderr(tmp_path):
    py = fake_python(tmp_path, "import sys\nsys.stderr.write('line1\\n' + 'x' * 9000 + 'END')\nsys.exit(3)\n")
    code, err = sr.run_producer([str(py)], timeout_s=30, env={"PATH": os.environ["PATH"]})
    assert code == 3 and err.endswith("END") and len(err) <= sr.STDERR_TAIL


def test_the_producer_runner_stops_a_process_at_the_deadline(tmp_path):
    py = fake_python(tmp_path, "import time\ntime.sleep(30)\n")
    code, err = sr.run_producer([str(py)], timeout_s=1, env={"PATH": os.environ["PATH"]})
    assert code is None and "deadline" in err


def test_the_producer_runner_reports_a_program_that_cannot_start(tmp_path):
    code, err = sr.run_producer([str(tmp_path / "nothing")], timeout_s=5, env={})
    assert code is None and "cannot" in err


def test_the_producer_gets_only_the_environment_it_is_given(tmp_path):
    py = fake_python(tmp_path, "import os, sys\nsys.stderr.write(','.join(sorted(os.environ)))\nsys.exit(1)\n")
    code, err = sr.run_producer([str(py)], timeout_s=30, env={"ONLY": "1"})
    assert "ONLY" in err and "HF_TOKEN" not in err


def test_the_producer_environment_keeps_path_and_the_cache_location_and_drops_everything_else():
    env = sr.producer_env({"PATH": "/bin", "HF_HOME": "/h", "HF_TOKEN": "secret", "HOME": "/home/x", "OPENAI_API_KEY": "k", "TRANSFORMERS_OFFLINE": "1"})
    assert env == {"PATH": "/bin", "HF_HOME": "/h", "TRANSFORMERS_OFFLINE": "1"}


def test_the_selection_decision_is_read_from_the_ledger():
    rec = {"event": "decision", "stage": 0, "data": {"decision": sr.SPEECH_ROUTE_SELECTED, "task": "speech-to-text", "python": "/p"}}
    assert sr.selection([{"event": "run_start", "data": {}}, rec])["python"] == "/p"
    assert sr.selection([{"event": "run_start", "data": {}}]) is None
    assert sr.selection("x") is None
