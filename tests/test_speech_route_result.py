# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2's gate (orchard/speech_route.py `result_problems`): every number is recomputed from the saved answers and stage 1's record."""
import copy

import pytest

from orchard import speech_route as sr
from test_speech_check import wav_bytes
from speech_fixtures import PROMPTS, asr_reference, asr_result, put_json, sha, tts_reference, tts_result


@pytest.fixture
def asr(tmp_path):
    s1, s2 = tmp_path / "1", tmp_path / "2"
    s1.mkdir(), s2.mkdir()
    ref = asr_reference(s1, clips=(("a.wav", "one two three four five six seven eight nine ten", "one two three four five six seven eight nine ten"),
                                   ("b.wav", "good morning to you all", "good morning to you all")))
    return ref, s2


@pytest.fixture
def tts(tmp_path):
    s1, s2 = tmp_path / "t1", tmp_path / "t2"
    s1.mkdir(), s2.mkdir()
    return tts_reference(s1), s2


# ---- speech to text ---------------------------------------------------------------------------------------------------


def test_a_matching_speech_to_text_result_has_no_problems(asr):
    ref, s2 = asr
    assert sr.result_problems(asr_result(ref), ref, s2) == []


def test_a_transcript_at_the_reference_limit_passes(asr):
    ref, s2 = asr
    # 0.05 of 20 words is 1 error: use a 20-word clip
    twenty = " ".join(f"w{i}" for i in range(20))
    ref["clips"][0].update(text=twenty, expected=twenty)
    heard = twenty.replace("w3 ", "x3 ")
    assert sr.result_problems(asr_result(ref, {"a.wav": heard}), ref, s2) == []


def test_a_transcript_over_the_reference_limit_fails_and_names_the_clip(asr):
    ref, s2 = asr
    problems = sr.result_problems(asr_result(ref, {"a.wav": "one two three four five six seven eight nine eleven"}), ref, s2)
    assert any("a.wav" in p and "reference" in p for p in problems), problems


def test_a_transcript_far_from_the_expected_text_fails_even_when_it_matches_the_reference(asr):
    ref, s2 = asr
    ref["clips"][1].update(expected="something else entirely here today")       # the reference itself is wrong about this clip
    problems = sr.result_problems(asr_result(ref), ref, s2)
    assert any("b.wav" in p and "expected" in p for p in problems), problems


def test_a_clip_with_no_expected_text_is_judged_against_the_reference_only(asr):
    ref, s2 = asr
    ref["clips"][1].update(expected=None)
    assert sr.result_problems(asr_result(ref), ref, s2) == []


def test_a_recorded_error_rate_that_disagrees_with_the_transcript_is_refused(asr):
    ref, s2 = asr
    result = asr_result(ref, {"a.wav": "one two three four five six seven eight nine eleven"})
    for c in result["checks"]:
        if c["name"] == "transcribe a.wav":
            c["data"]["wer"] = 0.0
    problems = sr.result_problems(result, ref, s2)
    assert any("recorded" in p and "a.wav" in p for p in problems), problems


def test_a_clip_with_no_transcribe_check_is_a_problem(asr):
    ref, s2 = asr
    result = asr_result(ref)
    result["checks"] = [c for c in result["checks"] if c["name"] != "transcribe b.wav"]
    assert any("b.wav" in p and "no transcribe check" in p for p in sr.result_problems(result, ref, s2))


def test_a_transcribe_check_that_failed_is_a_problem(asr):
    ref, s2 = asr
    result = asr_result(ref)
    for c in result["checks"]:
        if c["name"] == "transcribe a.wav":
            c["status"] = "fail"
    assert any("a.wav" in p and "fail" in p for p in sr.result_problems(result, ref, s2))


@pytest.mark.parametrize("name", ["health", "models", "bad request"])
def test_a_failed_structural_check_is_a_problem(asr, name):
    ref, s2 = asr
    result = asr_result(ref)
    for c in result["checks"]:
        if c["name"] == name:
            c["status"] = "fail"
    assert any(name in p for p in sr.result_problems(result, ref, s2))


@pytest.mark.parametrize("name", ["health", "models", "bad request"])
def test_a_missing_structural_check_is_a_problem(asr, name):
    ref, s2 = asr
    result = asr_result(ref)
    result["checks"] = [c for c in result["checks"] if c["name"] != name]
    assert any(name in p for p in sr.result_problems(result, ref, s2))


COMMON_BREAKS = {
    "schema": (lambda d: d.update(schema=0), "schema"),
    "task": (lambda d: d.update(task="text-to-speech"), "task"),
    "bundle": (lambda d: d.update(bundle="someone/else"), "bundle"),
    "bundle missing": (lambda d: d.pop("bundle"), "bundle"),
    "url on a forbidden port": (lambda d: d.update(url="http://127.0.0.1:8000"), "url"),
    "url not local": (lambda d: d.update(url="http://example.com:20000"), "url"),
    "url missing": (lambda d: d.pop("url"), "url"),
    "serve exit": (lambda d: d["served"].update(serve_exit=1), "serve"),
    "stop exit": (lambda d: d["served"].update(stop_exit=1), "stop"),
    "stop never ran": (lambda d: d["served"].pop("stop_exit"), "stop"),
    "served missing": (lambda d: d.pop("served"), "served"),
}


@pytest.mark.parametrize("name", sorted(COMMON_BREAKS))
def test_a_speech_to_text_result_that_breaks_a_common_rule_is_refused(asr, name):
    ref, s2 = asr
    result = asr_result(ref)
    COMMON_BREAKS[name][0](result)
    problems = sr.result_problems(result, ref, s2)
    assert any(COMMON_BREAKS[name][1] in p for p in problems), (name, problems)


def test_a_result_that_is_not_an_object_is_refused(asr):
    ref, s2 = asr
    for bad in (None, [], "x"):
        assert sr.result_problems(bad, ref, s2)


def test_a_checks_list_that_is_not_a_list_of_objects_is_refused(asr):
    ref, s2 = asr
    for bad in (None, "x", ["x"], [3]):
        result = asr_result(ref)
        result["checks"] = bad
        assert sr.result_problems(result, ref, s2)


def test_a_transcript_that_is_not_text_is_refused(asr):
    ref, s2 = asr
    result = asr_result(ref)
    for c in result["checks"]:
        if c["name"] == "transcribe a.wav":
            c["data"]["text"] = 5
    assert any("a.wav" in p for p in sr.result_problems(result, ref, s2))


# ---- text to speech -----------------------------------------------------------------------------------------------------


def test_a_good_text_to_speech_result_has_no_problems(tts):
    ref, s2 = tts
    assert sr.result_problems(tts_result(s2, ref), ref, s2) == []


def test_a_scorer_transcript_at_the_limit_passes(tts):
    ref, s2 = tts
    text = "w0 w1 w2 w3 w4 w5 w6"       # a seven-word prompt: 1 error is 0.143, under 0.15
    ref["prompts"][0]["text"] = text
    assert sr.result_problems(tts_result(s2, ref, heard={"p0": "w0 w1 w2 w3 w4 w5 x6"}), ref, s2) == []


def test_a_scorer_transcript_over_the_limit_fails_and_names_the_prompt(tts):
    ref, s2 = tts
    problems = sr.result_problems(tts_result(s2, ref, heard={"p1": "nothing like the prompt at all"}), ref, s2)
    assert any("p1" in p and "word error rate" in p for p in problems), problems


def test_a_prompt_with_no_scorer_transcript_is_a_problem(tts):
    ref, s2 = tts
    result = tts_result(s2, ref)
    result["prompts"][0]["heard"] = None
    assert any("p0" in p and "scorer" in p for p in sr.result_problems(result, ref, s2))


def test_a_prompt_missing_from_the_result_is_a_problem(tts):
    ref, s2 = tts
    result = tts_result(s2, ref)
    result["prompts"] = result["prompts"][:1]
    assert any("p1" in p for p in sr.result_problems(result, ref, s2))


def test_an_audio_file_that_changed_is_refused(tts):
    ref, s2 = tts
    result = tts_result(s2, ref)
    (s2 / "audio" / "p0.wav").write_bytes(wav_bytes(2.9))
    assert any("p0" in p and "sha256" in p for p in sr.result_problems(result, ref, s2))


def test_an_audio_file_that_is_gone_is_refused(tts):
    ref, s2 = tts
    result = tts_result(s2, ref)
    (s2 / "audio" / "p0.wav").unlink()
    assert any("p0" in p and "audio" in p for p in sr.result_problems(result, ref, s2))


def test_audio_that_fails_the_wav_checks_is_refused_even_when_the_record_says_it_passed(tts):
    ref, s2 = tts
    result = tts_result(s2, ref)
    bad = wav_bytes(3.0, silent=True)
    (s2 / "audio" / "p0.wav").write_bytes(bad)
    result["prompts"][0]["audio"]["sha256"] = sha(bad)
    assert any("p0" in p and "silent" in p for p in sr.result_problems(result, ref, s2))


def test_a_wav_at_the_wrong_sample_rate_is_refused(tts):
    ref, s2 = tts
    result = tts_result(s2, ref)
    bad = wav_bytes(3.0, rate=16000)
    (s2 / "audio" / "p0.wav").write_bytes(bad)
    result["prompts"][0]["audio"]["sha256"] = sha(bad)
    assert any("p0" in p and "Hz" in p for p in sr.result_problems(result, ref, s2))


def test_a_recorded_duration_that_disagrees_with_the_file_is_refused(tts):
    ref, s2 = tts
    result = tts_result(s2, ref)
    for c in result["prompts"][0]["checks"]:
        if c["name"] == "speech":
            c["data"]["seconds"] = 9.0
    assert any("p0" in p and "seconds" in p for p in sr.result_problems(result, ref, s2))


@pytest.mark.parametrize("name", ["health", "models", "speech", "bad voice", "missing input"])
def test_a_failed_text_to_speech_check_is_a_problem(tts, name):
    ref, s2 = tts
    result = tts_result(s2, ref)
    for c in result["prompts"][1]["checks"]:
        if c["name"] == name:
            c["status"] = "fail"
    assert any("p1" in p and name in p for p in sr.result_problems(result, ref, s2))


@pytest.mark.parametrize("name", ["health", "models", "speech", "bad voice", "missing input"])
def test_a_missing_text_to_speech_check_is_a_problem(tts, name):
    ref, s2 = tts
    result = tts_result(s2, ref)
    result["prompts"][1]["checks"] = [c for c in result["prompts"][1]["checks"] if c["name"] != name]
    assert any("p1" in p and name in p for p in sr.result_problems(result, ref, s2))


@pytest.mark.parametrize("name", sorted(COMMON_BREAKS))
def test_a_text_to_speech_result_that_breaks_a_common_rule_is_refused(tts, name):
    ref, s2 = tts
    result = tts_result(s2, ref)
    COMMON_BREAKS[name][0](result)
    key = "task" if name == "task" else COMMON_BREAKS[name][1]
    if name == "task":
        result["task"] = "speech-to-text"
    assert any(key in p for p in sr.result_problems(result, ref, s2)), name


def test_an_audio_path_that_leaves_the_stage_directory_is_refused(tts):
    ref, s2 = tts
    result = tts_result(s2, ref)
    outside = s2.parent / "outside.wav"
    outside.write_bytes(wav_bytes(3.0))
    result["prompts"][0]["audio"] = {"file": "../outside.wav", "sha256": sha(outside.read_bytes())}
    assert any("p0" in p and "audio" in p for p in sr.result_problems(result, ref, s2))


def test_a_reference_of_the_wrong_task_for_the_result_is_refused(asr, tts):
    ref, s2 = asr
    tref, _ = tts
    assert sr.result_problems(asr_result(ref), tref, s2)
