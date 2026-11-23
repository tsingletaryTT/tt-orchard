# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard speech-check` (orchard/speech_check.py), against fake servers on localhost.

Every server here binds port 0 and the test reads the port the system chose (ephemeral ports are far from 8000 and 8100 to 8103). The fakes
copy what the two Voxtral bundles' code does, as read on 2026-11-23: the transcription route takes a multipart `file` and answers
`{"text": ...}` or 400 when `file` is missing; the speech route takes JSON `input` and `voice` and answers a 24 kHz 16-bit mono WAV, or 400 for an
unknown voice. No test needs a chip.
"""
from __future__ import annotations

import functools
import io
import json
import math
import os
import struct
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from orchard import cli, speech_check as sc


@functools.lru_cache(maxsize=None)
def wav_bytes(seconds=3.0, rate=24000, channels=1, width=2, amp=8000, silent=False) -> bytes:
    frames = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        if width == 2:
            data = b"".join(struct.pack("<h", 0 if silent else int(amp * math.sin(2 * math.pi * 220 * i / rate))) * channels for i in range(frames))
        else:
            data = bytes([128]) * frames * channels
        w.writeframes(data)
    return buf.getvalue()


class Fake:
    """A server on a port the system picks. `cfg` holds the behaviour; `seen` records every request as (method, path, body)."""

    def __init__(self, **cfg):
        self.cfg = {"text": "I have a dream", "voices": ["casual_male", "fr_male"], "wav": wav_bytes(), "models": ["fake/model"], **cfg}
        self.seen = []
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def send(self, code, body=b"", ctype="application/json", extra=()):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in extra:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                fake.seen.append(("GET", self.path, b""))
                if fake.cfg.get("redirect"):
                    return self.send(302, b"", extra=[("Location", fake.cfg["redirect"])])
                if self.path == "/health":
                    return self.send(fake.cfg.get("health", 200), b'{"status": "ok"}')
                if self.path == "/v1/models":
                    return self.send(200, json.dumps({"object": "list", "data": [{"id": m} for m in fake.cfg["models"]]}).encode())
                self.send(404, b'{"error": "not found"}')

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                fake.seen.append(("POST", self.path, body))
                if fake.cfg.get("post_status"):
                    return self.send(fake.cfg["post_status"], b'{"error": "boom"}')
                if self.path == "/v1/audio/transcriptions":
                    if b'name="file"' not in body:
                        return self.send(fake.cfg.get("missing_file_status", 400), b'{"error": {"message": "missing file"}}')
                    raw = fake.cfg.get("transcript_raw")
                    if raw is not None:
                        return self.send(200, raw)
                    return self.send(200, json.dumps({"text": fake.cfg["text"]}).encode())
                if self.path == "/v1/audio/speech":
                    req = json.loads(body or b"{}")
                    voice = req.get("voice")
                    if voice is not None and voice not in fake.cfg["voices"]:
                        return self.send(fake.cfg.get("bad_voice_status", 400), b'{"error": "voice is not served"}')
                    if not req.get("input"):
                        return self.send(400, b'{"error": "missing input"}')
                    if req.get("stream"):
                        pcm = fake.cfg.get("pcm", b"\x01\x02" * 72000)
                        return self.send(200, pcm, "audio/pcm")
                    return self.send(200, fake.cfg["wav"], fake.cfg.get("ctype", "audio/wav"))
                self.send(404, b'{"error": "not found"}')

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.02), daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def posts(self):
        return [s for s in self.seen if s[0] == "POST"]


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "mlk.wav"
    p.write_bytes(wav_bytes(1.0, rate=16000))
    return p


def statuses(results):
    return {r.name: r.status for r in results}


# ---- the address: what is refused before anything is sent ----------------------------------------------------------


@pytest.mark.parametrize("url,expected", [("http://127.0.0.1:20000", ("127.0.0.1", 20000)), ("http://127.0.0.1:20000/", ("127.0.0.1", 20000)),
                                          ("http://127.0.0.1:20000/v1", ("127.0.0.1", 20000)), ("http://127.0.0.1:20000/v1/", ("127.0.0.1", 20000)),
                                          ("http://localhost:20001", ("localhost", 20001)), ("http://[::1]:20002/v1", ("::1", 20002)),
                                          ("http://127.0.0.1:8104", ("127.0.0.1", 8104)), ("http://127.0.0.1:7999", ("127.0.0.1", 7999))])
def test_a_local_address_with_a_port_is_accepted(url, expected):
    host, port = sc.parse_base_url(url)
    assert (host, port) == expected


@pytest.mark.parametrize("port", [8000, 8100, 8101, 8102, 8103])
@pytest.mark.parametrize("suffix", ["", "/v1", "/"])
def test_the_ports_the_machine_keeps_for_other_services_are_refused(port, suffix):
    with pytest.raises(sc.SpeechCheckError, match="8000|8100"):
        sc.parse_base_url(f"http://127.0.0.1:{port}{suffix}")


@pytest.mark.parametrize("url", ["http://127.0.0.1", "http://127.0.0.1/", "http://localhost/v1"])
def test_a_missing_port_is_refused_and_no_default_is_assumed(url):
    with pytest.raises(sc.SpeechCheckError, match="port"):
        sc.parse_base_url(url)


@pytest.mark.parametrize("url", ["http://example.com:20000", "http://10.0.0.5:20000", "http://127.0.0.1.example.com:20000", "http://0.0.0.0:20000",
                                 "http://192.168.1.2:20000"])
def test_a_server_that_is_not_on_this_machine_is_refused(url):
    with pytest.raises(sc.SpeechCheckError, match="this machine"):
        sc.parse_base_url(url)


@pytest.mark.parametrize("url", ["https://127.0.0.1:20000", "ftp://127.0.0.1:20000", "127.0.0.1:20000", "//127.0.0.1:20000", "file:///etc/passwd"])
def test_only_plain_http_is_accepted(url):
    with pytest.raises(sc.SpeechCheckError):
        sc.parse_base_url(url)


@pytest.mark.parametrize("url", ["http://user:pw@127.0.0.1:20000", "http://127.0.0.1:20000?x=1", "http://127.0.0.1:20000#frag",
                                 "http://127.0.0.1:20000/other", "http://127.0.0.1:20000/v1/audio", "http://127.0.0.1:0", "http://127.0.0.1:65536",
                                 "http://127.0.0.1:abc", "http://127.0.0.1:2000\x000", "http://127.0.0.1:20000/v1\n"])
def test_credentials_queries_other_paths_and_odd_ports_are_refused(url):
    with pytest.raises(sc.SpeechCheckError):
        sc.parse_base_url(url)


@pytest.mark.parametrize("bad", [None, 3, [], {}, b"http://127.0.0.1:20000", ""])
def test_an_address_that_is_not_text_is_refused(bad):
    with pytest.raises(sc.SpeechCheckError):
        sc.parse_base_url(bad)


# ---- word error rate: the exact rule ----------------------------------------------------------------------------------


def test_identical_text_has_no_errors():
    assert sc.wer("I have a dream", "I have a dream") == (0, 4)


def test_case_and_punctuation_do_not_count():
    assert sc.wer("I have a dream.", "i HAVE a dream!!") == (0, 4)
    assert sc.wer("Yesterday it was 35 degrees, in Barcelona.", "yesterday it was 35 degrees in barcelona") == (0, 7)


def test_each_substitution_insertion_and_deletion_costs_one():
    assert sc.wer("one two three four", "one two three five") == (1, 4)
    assert sc.wer("one two three four", "one two three") == (1, 4)
    assert sc.wer("one two three four", "one two three four five") == (1, 4)
    assert sc.wer("one two three four", "four three two one") == (4, 4)       # a reversal costs four edits, whichever word is matched


def test_an_apostrophe_inside_a_word_stays_and_one_at_the_edge_goes():
    assert sc.wer("don't stop", "don't stop") == (0, 2)
    assert sc.wer("don't stop", "'don't' stop") == (0, 2)
    assert sc.wer("don't stop", "dont stop") == (1, 2)


def test_combining_marks_stay_inside_a_word():
    # Devanagari vowel signs are marks and are not letters; splitting at them would count one word as several.
    assert sc.words("लोमड़ी ऊपर") == ["लोमड़ी", "ऊपर"]
    assert sc.wer("लोमड़ी ऊपर", "लोमडी उपर") == (2, 2)


def test_an_empty_reference_is_an_error_and_an_empty_hypothesis_is_all_deletions():
    with pytest.raises(sc.SpeechCheckError):
        sc.wer("...", "anything")
    assert sc.wer("a b c", "") == (3, 3)


def test_words_are_compared_after_unicode_normalisation():
    assert sc.wer("café", "café") == (0, 1)


# ---- speech to text ----------------------------------------------------------------------------------------------------


def stt(fake, clip, expect="I have a dream", **kw):
    return sc.check_speech_to_text(fake.url, [(clip, expect)], **kw)


def test_a_correct_transcript_passes_every_check(clip):
    with Fake() as f:
        res = stt(f, clip)
    assert statuses(res) == {"health": "pass", "models": "pass", "transcribe mlk.wav": "pass", "bad request": "pass"}
    t = next(r for r in res if r.name.startswith("transcribe"))
    assert t.data["errors"] == 0 and t.data["reference_words"] == 4 and t.data["wer"] == 0.0 and t.data["text"] == "I have a dream"


def test_the_clip_is_sent_as_the_multipart_file_field(clip):
    with Fake() as f:
        stt(f, clip, language="en")
        body = f.posts()[0][2]
    assert b'name="file"; filename="mlk.wav"' in body and clip.read_bytes() in body and b'name="language"' in body and b"en" in body


def test_no_language_field_is_sent_unless_asked(clip):
    with Fake() as f:
        stt(f, clip)
        body = f.posts()[0][2]
    assert b'name="language"' not in body


def test_a_wrong_transcript_fails_with_the_error_rate_and_the_limit(clip):
    with Fake(text="I have a nightmare today") as f:
        res = stt(f, clip, max_wer=0.10)
    t = next(r for r in res if r.name.startswith("transcribe"))
    assert t.status == "fail" and t.data["wer"] > 0.10 and "0.10" in t.detail and "I have a nightmare today" in t.detail


def test_a_transcript_at_the_limit_passes(clip):
    with Fake(text="I have a dreamt") as f:      # one substitution in four words: 0.25
        assert next(r for r in stt(f, clip, max_wer=0.25) if r.name.startswith("transcribe")).status == "pass"
        assert next(r for r in stt(f, clip, max_wer=0.2499) if r.name.startswith("transcribe")).status == "fail"


@pytest.mark.parametrize("raw", [b"not json", b"[]", b'{"text": 5}', b"{}", b'{"text": null}', b"\xff\xfe", b'{"text": "x", "n": NaN}',
                                 b'{"text": "x", "text": "y"}', b'{"text": "\\ud800"}', b'{"text": "x", "n": 1e999}', b"[" * 200 + b"]" * 200])
def test_an_answer_that_is_not_a_transcript_fails(clip, raw):
    with Fake(transcript_raw=raw) as f:
        res = stt(f, clip)
    t = next(r for r in res if r.name.startswith("transcribe"))
    assert t.status == "fail" and "text" in t.detail


@pytest.mark.parametrize("code", [400, 404, 500, 503])
def test_a_server_error_fails_the_transcription(clip, code):
    with Fake(post_status=code) as f:
        t = next(r for r in stt(f, clip) if r.name.startswith("transcribe"))
    assert t.status == "fail" and str(code) in t.detail


def test_a_server_that_answers_200_to_a_request_without_a_file_fails_the_bad_request_check(clip):
    with Fake(missing_file_status=200) as f:
        res = statuses(stt(f, clip))
    assert res["bad request"] == "fail"


def test_a_server_that_crashes_on_a_request_without_a_file_fails_the_bad_request_check(clip):
    with Fake(missing_file_status=500) as f:
        assert statuses(stt(f, clip))["bad request"] == "fail"


def test_a_failing_health_route_is_a_failure_and_the_other_checks_still_run(clip):
    with Fake(health=500) as f:
        res = statuses(stt(f, clip))
    assert res["health"] == "fail" and res["transcribe mlk.wav"] == "pass"


def test_the_models_list_is_recorded(clip):
    with Fake(models=["a/b", "c/d"]) as f:
        m = next(r for r in stt(f, clip) if r.name == "models")
    assert m.status == "pass" and m.data["ids"] == ["a/b", "c/d"]


def test_an_empty_models_list_fails(clip):
    with Fake(models=[]) as f:
        assert statuses(stt(f, clip))["models"] == "fail"


def test_two_clips_are_checked_one_after_the_other(clip, tmp_path):
    other = tmp_path / "other.wav"
    other.write_bytes(wav_bytes(1.0, rate=16000))
    with Fake() as f:
        res = sc.check_speech_to_text(f.url, [(clip, "I have a dream"), (other, "something else entirely")])
    s = statuses(res)
    assert s["transcribe mlk.wav"] == "pass" and s["transcribe other.wav"] == "fail"


def test_an_unreachable_server_fails_every_check_and_does_not_raise(clip):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                        # nothing listens on this port now
    res = sc.check_speech_to_text(f"http://127.0.0.1:{port}", [(clip, "x")], timeout=2)
    assert res and all(r.status == "fail" for r in res)


def test_a_server_that_never_answers_fails_after_the_timeout(clip):
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    try:
        res = sc.check_speech_to_text(f"http://127.0.0.1:{srv.getsockname()[1]}", [(clip, "x")], timeout=0.5)
    finally:
        srv.close()
    assert res and all(r.status == "fail" for r in res)
    assert any("timed out after 0.5 s" in r.detail for r in res)


def test_a_redirect_is_a_failure_and_is_not_followed(clip):
    with Fake() as target, Fake(redirect=target.url + "/health") as f:
        res = stt(f, clip)
        assert target.seen == []
    assert any(r.status == "fail" and "redirect" in r.detail for r in res)


def test_an_answer_larger_than_the_limit_fails_instead_of_filling_memory(clip, monkeypatch):
    monkeypatch.setattr(sc, "JSON_LIMIT", 64)
    with Fake(transcript_raw=b'{"text": "' + b"a " * 200 + b'"}') as f:
        t = next(r for r in stt(f, clip) if r.name.startswith("transcribe"))
    assert t.status == "fail" and "larger than" in t.detail


# ---- clips: files something else controls ---------------------------------------------------------------------------------


def test_a_missing_clip_is_refused_before_anything_is_sent(tmp_path):
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="clip"):
            sc.check_speech_to_text(f.url, [(tmp_path / "none.wav", "x")])
        assert f.seen == []


def test_a_clip_that_is_a_link_is_refused(tmp_path, clip):
    link = tmp_path / "link.wav"
    os.symlink(clip, link)
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="clip"):
            sc.check_speech_to_text(f.url, [(link, "x")])
        assert f.seen == []


def test_a_clip_that_is_a_fifo_is_refused_without_blocking(tmp_path):
    fifo = tmp_path / "pipe.wav"
    os.mkfifo(fifo)
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="clip"):
            sc.check_speech_to_text(f.url, [(fifo, "x")])


def test_a_clip_over_the_size_limit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "CLIP_LIMIT", 100)
    big = tmp_path / "big.wav"
    big.write_bytes(b"x" * 101)
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="clip"):
            sc.check_speech_to_text(f.url, [(big, "x")])
        assert f.seen == []


def test_an_expected_text_with_no_words_is_refused_before_anything_is_sent(clip):
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="expected"):
            sc.check_speech_to_text(f.url, [(clip, "... !!")])
        assert f.seen == []


def test_no_clip_is_refused(clip):
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="at least one clip"):
            sc.check_speech_to_text(f.url, [])


# ---- text to speech -------------------------------------------------------------------------------------------------------


TEXT = "The quick brown fox jumps over the lazy dog."


def tts(fake, **kw):
    return sc.check_text_to_speech(fake.url, TEXT, **kw)


def test_good_speech_passes_every_check():
    with Fake() as f:
        res = tts(f)
    s = statuses(res)
    assert s == {"health": "pass", "models": "pass", "speech": "pass", "bad voice": "pass", "missing input": "pass"}
    sp = next(r for r in res if r.name == "speech")
    assert sp.data["sample_rate"] == 24000 and sp.data["channels"] == 1 and sp.data["sample_width"] == 2 and sp.data["seconds"] == pytest.approx(3.0)


def test_the_request_carries_the_text_and_the_voice():
    with Fake() as f:
        tts(f, voices=["fr_male"])
        speech = [json.loads(b) for m, p, b in f.posts() if p == "/v1/audio/speech"]
    assert {"input": TEXT, "voice": "fr_male"} in speech


def test_no_voice_field_is_sent_when_none_is_asked_for():
    with Fake() as f:
        tts(f)
        first = json.loads(f.posts()[0][2])
    assert "voice" not in first and first["input"] == TEXT


def test_each_voice_asked_for_gets_its_own_row():
    with Fake() as f:
        s = statuses(tts(f, voices=["casual_male", "fr_male"]))
    assert s["speech casual_male"] == "pass" and s["speech fr_male"] == "pass" and "speech" not in s


@pytest.mark.parametrize("kw,word", [(dict(rate=16000), "24000"), (dict(channels=2), "mono"), (dict(width=1), "16-bit")])
def test_a_wav_with_the_wrong_format_fails(kw, word):
    with Fake(wav=wav_bytes(**kw)) as f:
        sp = next(r for r in tts(f) if r.name == "speech")
    assert sp.status == "fail" and word in sp.detail


def test_the_expected_sample_rate_can_be_changed():
    with Fake(wav=wav_bytes(rate=16000)) as f:
        assert next(r for r in tts(f, sample_rate=16000) if r.name == "speech").status == "pass"


@pytest.mark.parametrize("raw", [b"", b"RIFF", b"not a wav file at all", b"\x00" * 100])
def test_an_answer_that_is_not_a_wav_fails(raw):
    with Fake(wav=raw) as f:
        sp = next(r for r in tts(f) if r.name == "speech")
    assert sp.status == "fail" and "WAV" in sp.detail


def test_silence_fails():
    with Fake(wav=wav_bytes(silent=True)) as f:
        sp = next(r for r in tts(f) if r.name == "speech")
    assert sp.status == "fail" and "silent" in sp.detail


@pytest.mark.parametrize("seconds", [0.1, 60.0])
def test_a_duration_that_does_not_fit_the_text_fails(seconds):
    with Fake(wav=wav_bytes(seconds=seconds)) as f:
        sp = next(r for r in tts(f) if r.name == "speech")
    assert sp.status == "fail" and "seconds per character" in sp.detail


@pytest.mark.parametrize("seconds", [1.0, 3.0, 8.0])
def test_a_plausible_duration_passes(seconds):
    with Fake(wav=wav_bytes(seconds=seconds)) as f:
        assert next(r for r in tts(f) if r.name == "speech").status == "pass"


def test_a_content_type_that_is_not_audio_fails():
    with Fake(ctype="text/html") as f:
        sp = next(r for r in tts(f) if r.name == "speech")
    assert sp.status == "fail" and "Content-Type" in sp.detail


@pytest.mark.parametrize("code", [400, 500])
def test_a_server_error_fails_the_speech_check(code):
    with Fake(post_status=code) as f:
        assert next(r for r in tts(f) if r.name == "speech").status == "fail"


def test_a_server_that_serves_an_unknown_voice_fails_the_bad_voice_check():
    with Fake(bad_voice_status=200) as f:
        assert statuses(tts(f))["bad voice"] == "fail"


def test_a_server_that_crashes_on_an_unknown_voice_fails_the_bad_voice_check():
    with Fake(bad_voice_status=500) as f:
        assert statuses(tts(f))["bad voice"] == "fail"


def test_the_unknown_voice_is_not_one_the_server_could_serve():
    with Fake() as f:
        tts(f)
        voices = [json.loads(b).get("voice") for m, p, b in f.posts()]
    assert "no-such-voice-for-speech-check" in voices


def test_streaming_is_checked_only_when_asked():
    with Fake() as f:
        assert "stream" not in statuses(tts(f))
        assert statuses(tts(f, stream=True))["stream"] == "pass"


@pytest.mark.parametrize("pcm,word", [(b"", "no audio"), (b"\x01\x02\x03", "whole number of 16-bit samples")])
def test_a_stream_that_is_empty_or_cut_in_a_sample_fails(pcm, word):
    with Fake(pcm=pcm) as f:
        st = next(r for r in tts(f, stream=True) if r.name == "stream")
    assert st.status == "fail" and word in st.detail


def test_the_streamed_request_asks_for_raw_pcm():
    with Fake() as f:
        tts(f, stream=True)
        streamed = [json.loads(b) for m, p, b in f.posts() if json.loads(b or b"{}").get("stream")]
    assert streamed and streamed[0]["response_format"] == "pcm" and streamed[0]["stream"] is True


# ---- speech back through a speech-to-text server --------------------------------------------------------------------------


def test_the_speech_is_transcribed_by_the_asr_server_and_compared_with_the_text():
    with Fake() as tts_server, Fake(text=TEXT) as asr:
        res = tts(tts_server, asr_url=asr.url)
        sent = asr.posts()[0][2]
    r = next(r for r in res if r.name == "transcribe speech")
    assert r.status == "pass" and r.data["wer"] == 0.0
    assert tts_server.cfg["wav"] in sent


def test_speech_the_asr_server_cannot_read_fails_with_the_error_rate():
    with Fake() as tts_server, Fake(text="completely different words here") as asr:
        r = next(r for r in tts(tts_server, asr_url=asr.url, max_wer=0.15) if r.name == "transcribe speech")
    assert r.status == "fail" and r.data["wer"] > 0.15


def test_the_asr_address_follows_the_same_rules():
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="8000|8100"):
            tts(f, asr_url="http://127.0.0.1:8100")
        with pytest.raises(sc.SpeechCheckError, match="this machine"):
            tts(f, asr_url="http://example.com:20000")
        assert f.seen == []


def test_without_an_asr_server_the_content_of_the_speech_is_not_checked_and_the_output_says_so():
    with Fake() as f:
        s = statuses(tts(f))
    assert "transcribe speech" not in s
    out = io.StringIO()
    with Fake() as f:
        sc.run(sc.parser().parse_args(["--task", "text-to-speech", f.url]), out)
    assert "not checked" in out.getvalue() and "words" in out.getvalue()


# ---- the command ----------------------------------------------------------------------------------------------------------


def run(argv):
    out = io.StringIO()
    code = sc.run(sc.parser().parse_args(argv), out)
    return code, out.getvalue()


def test_exit_zero_when_every_check_passes(clip):
    with Fake() as f:
        code, text = run(["--task", "speech-to-text", f.url, "--clip", str(clip), "--expect", "I have a dream"])
    assert code == 0 and "pass" in text and "fail" not in text.replace("0 fail", "")


def test_exit_one_when_a_check_fails(clip):
    with Fake(text="nothing like it") as f:
        code, text = run(["--task", "speech-to-text", f.url, "--clip", str(clip), "--expect", "I have a dream"])
    assert code == 1 and "fail" in text


def test_exit_two_and_a_message_on_stderr_for_a_refused_address(clip, capsys):
    code, text = run(["--task", "speech-to-text", "http://127.0.0.1:8000", "--clip", str(clip), "--expect", "x"])
    assert code == 2 and text == ""
    assert "8000" in capsys.readouterr().err


def test_exit_two_when_the_clips_and_the_expected_texts_do_not_pair_up(clip, capsys):
    with Fake() as f:
        code, _ = run(["--task", "speech-to-text", f.url, "--clip", str(clip)])
        assert code == 2 and f.seen == []
        code, _ = run(["--task", "speech-to-text", f.url, "--clip", str(clip), "--expect", "a", "--expect", "b"])
        assert code == 2 and f.seen == []
    assert "--expect" in capsys.readouterr().err


def test_a_task_is_required_and_must_be_a_known_one(clip):
    with pytest.raises(SystemExit):
        sc.parser().parse_args(["http://127.0.0.1:20000"])
    with pytest.raises(SystemExit):
        sc.parser().parse_args(["--task", "image-to-text", "http://127.0.0.1:20000"])


def test_clips_belong_to_speech_to_text_and_text_to_speech_options_to_the_other(clip, capsys):
    with Fake() as f:
        assert run(["--task", "text-to-speech", f.url, "--clip", str(clip), "--expect", "x"])[0] == 2
        assert run(["--task", "speech-to-text", f.url, "--clip", str(clip), "--expect", "x", "--voice", "a"])[0] == 2
        assert run(["--task", "speech-to-text", f.url, "--clip", str(clip), "--expect", "x", "--asr-url", f.url])[0] == 2
        assert f.seen == []


def test_the_json_output_holds_every_check_and_the_rule_that_was_applied(clip):
    with Fake() as f:
        code, text = run(["--task", "speech-to-text", f.url, "--clip", str(clip), "--expect", "I have a dream", "--json"])
    doc = json.loads(text)
    assert code == 0 and doc["task"] == "speech-to-text" and doc["passed"] is True
    assert [c["name"] for c in doc["checks"]] == ["health", "models", "transcribe mlk.wav", "bad request"]
    assert "max_wer" in doc["rule"] and "word error rate" in doc["rule"]["wer"]
    assert doc["checks"][2]["data"]["wer"] == 0.0


def test_the_output_names_what_a_pass_rests_on(clip):
    with Fake() as f:
        _, text = run(["--task", "speech-to-text", f.url, "--clip", str(clip), "--expect", "I have a dream"])
    assert "word error rate 0.00" in text and "0 of 4" in text


def test_the_text_output_has_no_right_hand_border(clip):
    with Fake() as f:
        _, text = run(["--task", "speech-to-text", f.url, "--clip", str(clip), "--expect", "I have a dream"])
    assert not any(line.rstrip().endswith(("║", "│", "╗", "╝")) for line in text.splitlines())


def test_the_subcommand_is_registered_in_the_cli(clip):
    out = io.StringIO()
    with Fake() as f:
        code = cli.main(["speech-check", "--task", "speech-to-text", f.url, "--clip", str(clip), "--expect", "I have a dream"], stdout=out)
    assert code == 0 and "transcribe mlk.wav" in out.getvalue()


def test_the_default_limits_are_the_documented_ones():
    assert sc.DEFAULT_MAX_WER_TEXT == 0.10 and sc.DEFAULT_MAX_WER_SPEECH == 0.15
    assert sc.SECONDS_PER_CHARACTER == (0.02, 0.4)
    assert sc.DEFAULT_SAMPLE_RATE == 24000
    assert sc.FORBIDDEN_PORTS == frozenset({8000, 8100, 8101, 8102, 8103})


def test_the_help_and_messages_avoid_the_phrases_the_writing_rules_forbid(capsys):
    text = sc.parser().format_help()
    for phrase in (", not ", "rather than", "instead of", "reads as", "honest", "the one", "which is why", "This is what"):
        assert phrase not in text, phrase


# ---- option values ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), float("inf"), True])
def test_a_max_wer_outside_zero_to_one_is_refused_before_anything_is_sent(clip, bad):
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="max-wer"):
            sc.check_speech_to_text(f.url, [(clip, "x")], max_wer=bad)
        with pytest.raises(sc.SpeechCheckError, match="max-wer"):
            sc.check_text_to_speech(f.url, TEXT, max_wer=bad)
        assert f.seen == []


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), 86401, True])
def test_a_timeout_that_is_not_a_sensible_number_of_seconds_is_refused(clip, bad):
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="timeout"):
            sc.check_speech_to_text(f.url, [(clip, "x")], timeout=bad)
        assert f.seen == []


@pytest.mark.parametrize("bad", ["", "   ", "...", None, 5, "a " * 3000])
def test_text_to_speak_must_hold_a_word_and_stay_short(bad):
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="--text"):
            sc.check_text_to_speech(f.url, bad)
        assert f.seen == []


@pytest.mark.parametrize("bad", [999, 200000, 0, True, 24000.5, "24000"])
def test_a_sample_rate_outside_the_range_is_refused(bad):
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="sample-rate"):
            sc.check_text_to_speech(f.url, TEXT, sample_rate=bad)
        assert f.seen == []


@pytest.mark.parametrize("bad", ["", "a\nb", "x" * 101, 5])
def test_a_voice_name_must_be_short_printable_text(bad):
    with Fake() as f:
        with pytest.raises(sc.SpeechCheckError, match="voice"):
            sc.check_text_to_speech(f.url, TEXT, voices=[bad])
        assert f.seen == []


def test_the_json_exit_code_follows_the_checks(clip):
    with Fake(text="nothing like it") as f:
        code, text = run(["--task", "speech-to-text", f.url, "--clip", str(clip), "--expect", "I have a dream", "--json"])
    assert code == 1 and json.loads(text)["passed"] is False
