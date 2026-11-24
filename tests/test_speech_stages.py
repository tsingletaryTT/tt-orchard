# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The speech route's stage table and gates (orchard/stages.py `SPEECH_TABLE`, orchard/speech_route.py)."""
import json
from pathlib import Path

import pytest

from orchard import speech_route as sr, stages
from orchard.stages import SPEECH_TABLE, spec_for
from speech_fixtures import (ASR_BUNDLE, ASR_LICENSE, ASR_MODEL, TTS_BUNDLE, TTS_LICENSE, TTS_MODEL, asr_reference, asr_result, put_json, tts_reference,
                             tts_result)


def triage(run_dir, task="speech-to-text", **over):
    """A stage 0 record written the way the supervisor writes it (`delta.json` plus the evidence file it names)."""
    asr = task == "speech-to-text"
    model, bundle, lic = (ASR_MODEL, ASR_BUNDLE, ASR_LICENSE) if asr else (TTS_MODEL, TTS_BUNDLE, TTS_LICENSE)
    s0 = Path(run_dir) / "stages" / "0"
    (s0 / "evidence").mkdir(parents=True, exist_ok=True)
    (s0 / "evidence" / "triage.json").write_text("{}")
    doc = {"model": model, "nearest_model": f"none: the published bundle {bundle} serves it", "path": "full-port", "class": "full-port",
           "path_reasons": ["a published bundle serves it"],
           "differences": [{"area": "architecture", "evidence": ["stages/0/evidence/triage.json"], "finding": f"a {task} model served by {bundle}"}],
           "hazards": [] if asr else [{"area": "license", "evidence": ["stages/0/evidence/triage.json"], "finding": "non-commercial licence"}],
           "facts": {"form": "full", "serving": "container", "modality": "audio"},
           "speech": {"task": task, "upstream": model, "bundle": bundle, "license": lic}}
    doc.update(over)
    put_json(s0 / "delta.json", doc)
    return s0


# ---- the table -------------------------------------------------------------------------------------------------------


def test_the_table_holds_stages_0_to_8_and_no_stage_boots_the_coder():
    assert sorted(SPEECH_TABLE) == list(range(9))
    assert all(spec.needs_coder is False for spec in SPEECH_TABLE.values())


def test_stages_0_to_2_are_supervisor_code_and_stage_2_takes_one_board():
    assert [SPEECH_TABLE[n].harness for n in (0, 1, 2)] == [True, True, True]
    assert [SPEECH_TABLE[n].boards for n in (0, 1, 2)] == [0, 0, 1]
    assert SPEECH_TABLE[1].gate_file == sr.REFERENCE_FILE and SPEECH_TABLE[2].gate_file == sr.RESULT_FILE


def test_stages_3_to_7_are_skipped_each_with_a_reason_that_names_the_route():
    for n in range(3, 8):
        assert SPEECH_TABLE[n].skip and "speech" in SPEECH_TABLE[n].skip


def test_spec_for_uses_the_table_only_when_the_run_is_on_the_route():
    assert spec_for(2, "full-port", cls="full-port", speech_route=True) is SPEECH_TABLE[2]
    assert spec_for(0, None, speech_route=True) is SPEECH_TABLE[0]
    assert spec_for(2, "full-port", cls="full-port") is not SPEECH_TABLE[2]
    assert spec_for(2, "weights-only", cls="weights-only").gate is not SPEECH_TABLE[2].gate


def test_the_text_table_still_wins_for_a_text_run():
    assert spec_for(1, "full-port", cls="full-port", text_runtime=True) is stages.TEXT_TABLE[1]


# ---- stage 0 ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["speech-to-text", "text-to-speech"])
def test_a_good_triage_record_passes(tmp_path, task):
    s0 = triage(tmp_path, task)
    assert stages.gate_speech_triage(s0, tmp_path).ok


TRIAGE_BREAKS = {
    "no speech record": (lambda d: d.pop("speech"), "speech"),
    "task": (lambda d: d["speech"].update(task="x"), "task"),
    "task and model disagree": (lambda d: d["speech"].update(task="text-to-speech"), "bundle"),
    "bundle": (lambda d: d["speech"].update(bundle="someone/else"), "bundle"),
    "licence": (lambda d: d["speech"]["license"].update(id="mit"), "licence"),
    "licence line": (lambda d: d["speech"]["license"].update(line="x"), "licence"),
    "modality": (lambda d: d["facts"].update(modality="text"), "modality"),
    "serving": (lambda d: d["facts"].update(serving="thin-bundle"), "serving"),
    "path": (lambda d: d.update(path="weights-only", **{"class": "weights-only"}), "path"),
}


@pytest.mark.parametrize("name", sorted(TRIAGE_BREAKS))
def test_a_triage_record_that_breaks_a_rule_is_refused_with_the_reason(tmp_path, name):
    s0 = triage(tmp_path)
    doc = json.loads((s0 / "delta.json").read_text())
    TRIAGE_BREAKS[name][0](doc)
    put_json(s0 / "delta.json", doc)
    gate = stages.gate_speech_triage(s0, tmp_path)
    assert not gate.ok and any(TRIAGE_BREAKS[name][1] in r for r in gate.reasons), gate.reasons


def test_a_non_commercial_licence_must_appear_as_a_hazard(tmp_path):
    s0 = triage(tmp_path, "text-to-speech")
    doc = json.loads((s0 / "delta.json").read_text())
    doc["hazards"] = []
    put_json(s0 / "delta.json", doc)
    gate = stages.gate_speech_triage(s0, tmp_path)
    assert not gate.ok and any("non-commercial" in r for r in gate.reasons)


def test_a_missing_delta_is_refused(tmp_path):
    (tmp_path / "stages" / "0").mkdir(parents=True)
    assert not stages.gate_speech_triage(tmp_path / "stages" / "0", tmp_path).ok


# ---- stage 1 ---------------------------------------------------------------------------------------------------------


def test_a_good_reference_passes_stage_1_and_names_the_file_as_evidence(tmp_path):
    s1 = tmp_path / "stages" / "1"
    asr_reference(s1)
    gate = stages.gate_speech_reference(s1, tmp_path)
    assert gate.ok and any(sr.REFERENCE_FILE in e for e in gate.evidence)


def test_a_broken_reference_fails_stage_1_with_the_modules_reasons(tmp_path):
    s1 = tmp_path / "stages" / "1"
    doc = asr_reference(s1)
    doc["clips"][0]["sha256"] = "0" * 64
    put_json(s1 / sr.REFERENCE_FILE, doc)
    gate = stages.gate_speech_reference(s1, tmp_path)
    assert not gate.ok and any("sha256" in r for r in gate.reasons)


def test_a_missing_reference_fails_stage_1(tmp_path):
    (tmp_path / "stages" / "1").mkdir(parents=True)
    assert not stages.gate_speech_reference(tmp_path / "stages" / "1", tmp_path).ok


# ---- stage 2 ---------------------------------------------------------------------------------------------------------


def test_a_good_result_passes_stage_2(tmp_path):
    s1, s2 = tmp_path / "stages" / "1", tmp_path / "stages" / "2"
    ref = asr_reference(s1)
    put_json(s2 / sr.RESULT_FILE, asr_result(ref))
    gate = stages.gate_speech_check(s2, tmp_path)
    assert gate.ok and any(sr.RESULT_FILE in e for e in gate.evidence)


def test_a_good_text_to_speech_result_passes_stage_2(tmp_path):
    s1, s2 = tmp_path / "stages" / "1", tmp_path / "stages" / "2"
    ref = tts_reference(s1)
    put_json(s2 / sr.RESULT_FILE, tts_result(s2, ref))
    assert stages.gate_speech_check(s2, tmp_path).ok


def test_a_result_that_disagrees_with_the_reference_fails_stage_2(tmp_path):
    s1, s2 = tmp_path / "stages" / "1", tmp_path / "stages" / "2"
    ref = asr_reference(s1)
    put_json(s2 / sr.RESULT_FILE, asr_result(ref, {"mlk.wav": "something quite different here"}))
    gate = stages.gate_speech_check(s2, tmp_path)
    assert not gate.ok and any("mlk.wav" in r for r in gate.reasons)


def test_stage_2_fails_when_stage_1_left_no_reference(tmp_path):
    s1, s2 = tmp_path / "stages" / "1", tmp_path / "stages" / "2"
    ref = asr_reference(tmp_path / "other")
    s1.mkdir(parents=True)
    put_json(s2 / sr.RESULT_FILE, asr_result(ref))
    gate = stages.gate_speech_check(s2, tmp_path)
    assert not gate.ok and any("reference" in r for r in gate.reasons)


def test_stage_2_fails_when_the_result_is_missing(tmp_path):
    (tmp_path / "stages" / "2").mkdir(parents=True)
    assert not stages.gate_speech_check(tmp_path / "stages" / "2", tmp_path).ok


def test_stage_2_judges_a_reference_that_no_longer_passes_stage_1_as_unusable(tmp_path):
    s1, s2 = tmp_path / "stages" / "1", tmp_path / "stages" / "2"
    ref = asr_reference(s1)
    put_json(s2 / sr.RESULT_FILE, asr_result(ref))
    (s1 / "clips" / "mlk.wav").write_bytes(b"changed")
    gate = stages.gate_speech_check(s2, tmp_path)
    assert not gate.ok and any("stage 1" in r for r in gate.reasons)
