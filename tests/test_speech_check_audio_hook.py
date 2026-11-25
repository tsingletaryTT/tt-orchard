# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`check_text_to_speech(on_audio=...)` hands each accepted answer's WAV bytes to the caller, so the speech route can save what the bundle said and score it
later on the host. The hook sees only answers that passed the WAV checks."""
import pytest

from orchard import speech_check as sc
from test_speech_check import Fake, wav_bytes


def test_each_accepted_answer_is_handed_to_the_hook_with_its_name_and_bytes():
    seen = []
    with Fake() as f:
        results = sc.check_text_to_speech(f.url, "Hello there my friend.", voices=("casual_male", "fr_male"), on_audio=lambda name, data: seen.append((name, data)))
    assert [n for n, _ in seen] == ["speech casual_male", "speech fr_male"]
    assert all(d == wav_bytes() for _, d in seen)
    assert all(r.status == "pass" for r in results)


def test_the_default_voice_answer_is_named_speech():
    seen = []
    with Fake() as f:
        sc.check_text_to_speech(f.url, "Hello there my friend.", on_audio=lambda name, data: seen.append(name))
    assert seen == ["speech"]


def test_an_answer_that_fails_the_wav_checks_is_not_handed_over():
    seen = []
    with Fake(wav=wav_bytes(silent=True)) as f:
        results = sc.check_text_to_speech(f.url, "Hello there my friend.", on_audio=lambda name, data: seen.append(name))
    assert seen == []
    assert any(r.name == "speech" and r.status == "fail" for r in results)


def test_without_a_hook_nothing_changes():
    with Fake() as f:
        results = sc.check_text_to_speech(f.url, "Hello there my friend.")
    assert all(r.status == "pass" for r in results)


def test_a_hook_that_is_not_callable_is_refused_before_anything_is_sent():
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="on_audio"):
            sc.check_text_to_speech(f.url, "Hello there my friend.", on_audio="no")
        assert f.seen == []
