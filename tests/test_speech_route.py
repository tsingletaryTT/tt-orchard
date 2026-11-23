# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The speech route's records and what its gates recompute (orchard/speech_route.py, docs/superpowers/specs/2026-11-23-speech-route-design.md).

A gate recomputes each number from the saved clips, the saved audio and the transcripts. A recorded number that disagrees with the recomputed one fails
the gate, so a result file cannot say more than its evidence shows."""
import copy
import json

import pytest

from orchard import speech_models, speech_route as sr
from speech_fixtures import (ASR_BUNDLE, ASR_LICENSE, ASR_MODEL, TTS_BUNDLE, TTS_LICENSE, TTS_MODEL, asr_reference, asr_result, put_json, tts_reference,
                             tts_result)


def entries(**inputs):
    return [{"event": "run_start", "data": {"inputs": inputs}}]


# ---- which run takes the route --------------------------------------------------------------------------------------


@pytest.mark.parametrize("task", speech_models.TASKS)
def test_a_run_started_with_a_speech_task_takes_the_route(task):
    assert sr.selected(entries(speech_route=task)) == task


@pytest.mark.parametrize("value", ["", "tts", None, 3, ["speech-to-text"], "Speech-To-Text"])
def test_any_other_value_does_not_take_the_route(value):
    assert sr.selected(entries(speech_route=value)) is None


def test_a_run_without_the_input_does_not_take_the_route():
    assert sr.selected(entries(model="x")) is None
    assert sr.selected([]) is None


def test_only_the_first_run_start_decides():
    both = entries(speech_route="speech-to-text") + entries(speech_route="text-to-speech")
    assert sr.selected(both) == "speech-to-text"


# ---- the bundle and the licence -----------------------------------------------------------------------------------------


def test_the_bundle_for_a_task_is_the_known_one_for_that_task():
    assert sr.bundle_for_task("speech-to-text", ASR_MODEL).bundle == ASR_BUNDLE
    assert sr.bundle_for_task("text-to-speech", TTS_MODEL).bundle == TTS_BUNDLE


def test_a_model_of_the_other_task_has_no_bundle_for_this_one():
    assert sr.bundle_for_task("text-to-speech", ASR_MODEL) is None
    assert sr.bundle_for_task("speech-to-text", TTS_MODEL) is None
    assert sr.bundle_for_task("speech-to-text", "someone/else") is None


def test_the_licence_record_carries_the_id_and_the_non_commercial_flag_from_the_table():
    assert sr.license_record(sr.bundle_for_task("speech-to-text", ASR_MODEL)) == ASR_LICENSE
    assert sr.license_record(sr.bundle_for_task("text-to-speech", TTS_MODEL)) == TTS_LICENSE


def test_the_licence_line_names_the_licence_says_terms_apply_and_that_no_weights_were_copied():
    line = sr.license_line(sr.bundle_for_task("text-to-speech", TTS_MODEL))
    assert "cc-by-nc-4.0" in line and "non-commercial" in line
    assert "as the bundle's card states it" in line
    assert "did not copy or republish any weights" in line
    assert TTS_BUNDLE in line


def test_a_commercial_licence_line_does_not_say_non_commercial():
    assert "non-commercial" not in sr.license_line(sr.bundle_for_task("speech-to-text", ASR_MODEL))


def test_the_line_gives_no_legal_opinion():
    for task, model in (("speech-to-text", ASR_MODEL), ("text-to-speech", TTS_MODEL)):
        line = sr.license_line(sr.bundle_for_task(task, model)).lower()
        assert not any(w in line for w in ("legal", "lawful", "permitted", "allowed", "advice", "compliant"))


# ---- stage 1: the reference record --------------------------------------------------------------------------------------


def test_a_good_speech_to_text_reference_has_no_problems(tmp_path):
    assert sr.REFERENCE_FILE == "speech_reference.json"
    asr_reference(tmp_path)
    assert sr.reference_problems(json.loads((tmp_path / sr.REFERENCE_FILE).read_text()), tmp_path) == []


def test_a_good_text_to_speech_reference_has_no_problems(tmp_path):
    tts_reference(tmp_path)
    assert sr.reference_problems(json.loads((tmp_path / sr.REFERENCE_FILE).read_text()), tmp_path) == []


def test_a_reference_that_is_not_an_object_is_refused(tmp_path):
    for bad in ([], "x", None, 3):
        assert sr.reference_problems(bad, tmp_path)


def break_asr(doc, tmp_path):
    return doc


ASR_BREAKS = {
    "schema": (lambda d: d.update(schema=2), "schema"),
    "task": (lambda d: d.update(task="text-to-speech"), "task"),
    "unknown task": (lambda d: d.update(task="x"), "task"),
    "model": (lambda d: d.update(model=3), "model"),
    "model with no bundle": (lambda d: d.update(model="someone/else"), "bundle"),
    "licence id": (lambda d: d["license"].update(id="mit"), "licence"),
    "licence flag": (lambda d: d["license"].update(non_commercial=True), "licence"),
    "licence line": (lambda d: d["license"].update(line="whatever"), "licence"),
    "licence missing": (lambda d: d.pop("license"), "licence"),
    "no clips": (lambda d: d.update(clips=[]), "clips"),
    "clips not a list": (lambda d: d.update(clips={}), "clips"),
    "clip not an object": (lambda d: d["clips"].__setitem__(0, "x"), "clip"),
    "clip name with a slash": (lambda d: d["clips"][0].update(name="../x.wav"), "name"),
    "clip name empty": (lambda d: d["clips"][0].update(name=""), "name"),
    "clip hash": (lambda d: d["clips"][0].update(sha256="0" * 64), "sha256"),
    "clip size": (lambda d: d["clips"][0].update(bytes=1), "bytes"),
    "transcript missing": (lambda d: d["clips"][0].pop("text"), "transcript"),
    "transcript with no words": (lambda d: d["clips"][0].update(text="  ... "), "transcript"),
    "expected with no words": (lambda d: d["clips"][0].update(expected="--"), "expected"),
    "expected a number": (lambda d: d["clips"][0].update(expected=3), "expected"),
    "no engine": (lambda d: d.pop("reference"), "reference"),
    "engine without a version": (lambda d: d["reference"].pop("transformers"), "transformers"),
    "duplicate clip": (lambda d: d["clips"].append(copy.deepcopy(d["clips"][0])), "twice"),
}


@pytest.mark.parametrize("name", sorted(ASR_BREAKS))
def test_a_broken_speech_to_text_reference_is_refused_with_the_reason(tmp_path, name):
    doc = asr_reference(tmp_path)
    ASR_BREAKS[name][0](doc)
    problems = sr.reference_problems(doc, tmp_path)
    assert problems, name
    assert any(ASR_BREAKS[name][1] in p for p in problems), problems


def test_a_clip_file_that_is_gone_is_refused(tmp_path):
    doc = asr_reference(tmp_path)
    (tmp_path / "clips" / "mlk.wav").unlink()
    assert any("mlk.wav" in p for p in sr.reference_problems(doc, tmp_path))


def test_a_clip_file_that_changed_is_refused(tmp_path):
    doc = asr_reference(tmp_path)
    (tmp_path / "clips" / "mlk.wav").write_bytes(b"other bytes")
    assert any("sha256" in p for p in sr.reference_problems(doc, tmp_path))


def test_a_clip_that_is_a_link_is_refused(tmp_path):
    doc = asr_reference(tmp_path)
    real = tmp_path / "real.wav"
    (tmp_path / "clips" / "mlk.wav").rename(real)
    (tmp_path / "clips" / "mlk.wav").symlink_to(real)
    assert any("mlk.wav" in p for p in sr.reference_problems(doc, tmp_path))


TTS_BREAKS = {
    "no prompts": (lambda d: d.update(prompts=[]), "prompts"),
    "prompt without words": (lambda d: d["prompts"][0].update(text="..."), "prompt"),
    "duplicate prompt id": (lambda d: d["prompts"][1].update(id="p0"), "twice"),
    "prompt id missing": (lambda d: d["prompts"][0].pop("id"), "prompt"),
    "no calibration": (lambda d: d.update(calibration=[]), "calibration"),
    "calibration without expected": (lambda d: d["calibration"][0].update(expected=None), "expected"),
    "calibration wer recorded wrong": (lambda d: d["calibration"][0].update(wer=0.5), "wer"),
    "scorer too poor": (lambda d: d["calibration"][0].update(text="completely different words here today", wer=1.0), "scorer"),
    "licence": (lambda d: d["license"].update(non_commercial=False), "licence"),
}


@pytest.mark.parametrize("name", sorted(TTS_BREAKS))
def test_a_broken_text_to_speech_reference_is_refused_with_the_reason(tmp_path, name):
    doc = tts_reference(tmp_path)
    TTS_BREAKS[name][0](doc)
    problems = sr.reference_problems(doc, tmp_path)
    assert any(TTS_BREAKS[name][1] in p for p in problems), (name, problems)


def test_a_scorer_that_is_just_good_enough_passes(tmp_path):
    # 1 error in 10 words is 0.10, the limit
    text = "one two three four five six seven eight nine ten"
    doc = tts_reference(tmp_path, calibration=(("mlk.wav", text, "one two three four five six seven eight nine eleven", 0.1),))
    assert sr.reference_problems(doc, tmp_path) == []


def test_a_scorer_just_over_the_limit_is_refused(tmp_path):
    text = "one two three four five six seven eight nine"
    doc = tts_reference(tmp_path, calibration=(("mlk.wav", text, "one two three four five six seven eight ten", 1 / 9),))
    assert any("scorer" in p for p in sr.reference_problems(doc, tmp_path))
