# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What stage 1 assembles before its gate runs (orchard/speech_route.py): the operator's clips, the producer's config and command, and the reference record."""
import json
import os
import sys
from pathlib import Path

import pytest

from orchard import speech_reference_run as srr, speech_route as sr
from test_speech_check import wav_bytes
from speech_fixtures import ASR_MODEL, ENGINE, SCORER, TTS_MODEL


def put(d, name, data=b"RIFFx"):
    p = Path(d) / name
    p.write_bytes(data)
    return p


@pytest.fixture
def clips_dir(tmp_path):
    d = tmp_path / "clips-in"
    d.mkdir()
    put(d, "mlk.wav", wav_bytes(1.0, rate=16000))
    put(d, "mlk.txt", b"I have a dream that one day\n")
    put(d, "weather.flac", b"fLaC-ish")
    return d


def test_clips_are_found_by_suffix_with_the_expected_text_from_the_txt_next_to_them(clips_dir):
    found = sr.discover_clips(clips_dir)
    assert [c["name"] for c in found] == ["mlk.wav", "weather.flac"]
    assert found[0]["expected"] == "I have a dream that one day" and found[1]["expected"] is None
    assert found[0]["path"] == str(clips_dir / "mlk.wav")


def test_a_txt_with_no_clip_and_other_files_are_ignored(clips_dir):
    put(clips_dir, "orphan.txt", b"words")
    put(clips_dir, "notes.md", b"words")
    put(clips_dir, "x.exe", b"MZ")
    assert [c["name"] for c in sr.discover_clips(clips_dir)] == ["mlk.wav", "weather.flac"]


@pytest.mark.parametrize("suffix", sr.CLIP_SUFFIXES)
def test_each_audio_suffix_is_accepted(tmp_path, suffix):
    d = tmp_path / "c"
    d.mkdir()
    put(d, "a" + suffix)
    assert [c["name"] for c in sr.discover_clips(d)] == ["a" + suffix]


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(sr.SpeechRouteError, match="directory"):
        sr.discover_clips(tmp_path / "none")


def test_a_directory_with_no_clips_is_refused(tmp_path):
    (tmp_path / "e").mkdir()
    with pytest.raises(sr.SpeechRouteError, match="no clips"):
        sr.discover_clips(tmp_path / "e")


def test_a_directory_that_is_a_link_is_refused(tmp_path, clips_dir):
    (tmp_path / "link").symlink_to(clips_dir)
    with pytest.raises(sr.SpeechRouteError, match="directory"):
        sr.discover_clips(tmp_path / "link")


def test_a_clip_that_is_a_link_is_refused(tmp_path, clips_dir):
    (clips_dir / "ln.wav").symlink_to(clips_dir / "mlk.wav")
    with pytest.raises(sr.SpeechRouteError, match="ln.wav"):
        sr.discover_clips(clips_dir)


def test_a_clip_over_the_size_limit_is_refused(clips_dir, monkeypatch):
    monkeypatch.setattr(sr, "CLIP_LIMIT", 10)
    with pytest.raises(sr.SpeechRouteError, match="mlk.wav"):
        sr.discover_clips(clips_dir)


def test_more_than_the_clip_count_limit_is_refused(tmp_path):
    d = tmp_path / "many"
    d.mkdir()
    for i in range(sr.MAX_CLIPS + 1):
        put(d, f"c{i}.wav")
    with pytest.raises(sr.SpeechRouteError, match="at most"):
        sr.discover_clips(d)


def test_an_expected_text_with_no_words_is_refused(clips_dir):
    put(clips_dir, "mlk.txt", b" -- \n")
    with pytest.raises(sr.SpeechRouteError, match="mlk.txt"):
        sr.discover_clips(clips_dir)


def test_an_expected_text_over_the_limit_is_refused(clips_dir):
    put(clips_dir, "mlk.txt", b"word " * 2000)
    with pytest.raises(sr.SpeechRouteError, match="mlk.txt"):
        sr.discover_clips(clips_dir)


def test_an_expected_text_that_is_not_utf8_is_refused(clips_dir):
    put(clips_dir, "mlk.txt", b"\xff\xfe\xfa")
    with pytest.raises(sr.SpeechRouteError, match="mlk.txt"):
        sr.discover_clips(clips_dir)


def test_saved_clips_are_copied_with_their_hash_and_size(clips_dir, tmp_path):
    entries = sr.save_clips(sr.discover_clips(clips_dir), tmp_path / "s1")
    for e in entries:
        data = (tmp_path / "s1" / "clips" / e["name"]).read_bytes()
        assert e["bytes"] == len(data) and len(e["sha256"]) == 64
    assert entries[0]["expected"] == "I have a dream that one day"


def test_default_prompts_are_used_when_no_file_is_given():
    prompts = sr.read_prompts(None)
    assert [p["id"] for p in prompts] == ["p0", "p1", "p2"]
    assert [p["text"] for p in prompts] == list(sr.DEFAULT_PROMPTS) and all(p["voice"] is None for p in prompts)


def test_a_prompts_file_gives_one_prompt_per_nonblank_line(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("First line here.\n\n  Second line here.  \n")
    assert [p["text"] for p in sr.read_prompts(f)] == ["First line here.", "Second line here."]


@pytest.mark.parametrize("content", [b"", b"\n\n", b" -- \n", b"\xff\xfe"])
def test_an_unusable_prompts_file_is_refused(tmp_path, content):
    f = tmp_path / "p.txt"
    f.write_bytes(content)
    with pytest.raises(sr.SpeechRouteError, match="prompts"):
        sr.read_prompts(f)


def test_a_prompts_file_with_too_many_prompts_is_refused(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("word\n" * (sr.MAX_PROMPTS + 1))
    with pytest.raises(sr.SpeechRouteError, match="at most"):
        sr.read_prompts(f)


def test_the_producer_config_matches_what_the_producer_accepts(tmp_path):
    entries = [{"name": "a.wav", "path": "/x/a.wav"}]
    cfg = sr.producer_config(ASR_MODEL, "/snap", entries, language="en")
    assert srr._config_problems(cfg) == []
    assert cfg["model_dir"] == "/snap" and cfg["files"] == entries and cfg["language"] == "en"


def test_the_scorer_config_names_whisper_and_no_snapshot():
    cfg = sr.producer_config(sr.SCORER_MODEL, None, [{"name": "a.wav", "path": "/x/a.wav"}])
    assert cfg["model_id"] == sr.SCORER_MODEL == "openai/whisper-large-v3-turbo" and cfg["model_dir"] is None
    assert srr._config_problems(cfg) == []


def test_the_producer_command_runs_the_script_isolated_in_the_given_interpreter(tmp_path):
    cmd = sr.producer_command("/ref/bin/python", "asr", tmp_path / "c.json", tmp_path / "o.json")
    assert cmd[:2] == ["/ref/bin/python", "-I"]
    assert Path(cmd[2]) == Path(srr.__file__)
    assert cmd[3:] == ["--mode", "asr", "--config", str(tmp_path / "c.json"), "--out", str(tmp_path / "o.json")]


def test_a_producer_mode_that_is_not_known_is_refused():
    with pytest.raises(sr.SpeechRouteError):
        sr.producer_command("/p", "other", "c", "o")


# ---- the reference record -------------------------------------------------------------------------------------------------


def test_a_speech_to_text_reference_is_built_from_the_clips_and_the_producers_transcripts(clips_dir, tmp_path):
    entries = sr.save_clips(sr.discover_clips(clips_dir), tmp_path / "s1")
    texts = {"mlk.wav": "I have a dream that one day", "weather.flac": "yesterday it was warm"}
    doc = sr.build_reference("speech-to-text", ASR_MODEL, entries, ENGINE, texts)
    assert sr.reference_problems(doc, tmp_path / "s1") == []
    assert doc["clips"][1]["text"] == "yesterday it was warm" and doc["clips"][1]["expected"] is None
    assert doc["license"]["id"] == "apache-2.0" and doc["reference"] == ENGINE


def test_a_clip_the_producer_gave_no_transcript_for_is_refused(clips_dir, tmp_path):
    entries = sr.save_clips(sr.discover_clips(clips_dir), tmp_path / "s1")
    with pytest.raises(sr.SpeechRouteError, match="weather.flac"):
        sr.build_reference("speech-to-text", ASR_MODEL, entries, ENGINE, {"mlk.wav": "I have a dream"})


def test_a_text_to_speech_reference_records_the_scorers_error_on_each_clip(clips_dir, tmp_path):
    (clips_dir / "weather.flac").unlink()
    entries = sr.save_clips(sr.discover_clips(clips_dir), tmp_path / "s1")
    doc = sr.build_reference("text-to-speech", TTS_MODEL, entries, SCORER, {"mlk.wav": "I have a dream that one day"}, prompts=sr.read_prompts(None))
    assert doc["calibration"][0]["wer"] == 0.0
    assert sr.reference_problems(doc, tmp_path / "s1") == []
    assert [p["id"] for p in doc["prompts"]] == ["p0", "p1", "p2"]


def test_a_text_to_speech_clip_with_no_expected_text_is_refused(clips_dir, tmp_path):
    entries = sr.save_clips(sr.discover_clips(clips_dir), tmp_path / "s1")
    with pytest.raises(sr.SpeechRouteError, match="weather.flac"):
        sr.build_reference("text-to-speech", TTS_MODEL, entries, SCORER, {"mlk.wav": "I have a dream", "weather.flac": "x y"}, prompts=sr.read_prompts(None))


def test_a_reference_for_a_model_without_a_bundle_is_refused(clips_dir, tmp_path):
    entries = sr.save_clips(sr.discover_clips(clips_dir), tmp_path / "s1")
    with pytest.raises(sr.SpeechRouteError, match="bundle"):
        sr.build_reference("speech-to-text", "someone/else", entries, ENGINE, {})
