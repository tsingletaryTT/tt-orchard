"""The parts of the canning step that a music model needs and a chat model did not (orchard/can.py): a directory of code in the spec
(`[[extra_tree]]`), a boot check that makes a short clip and tests the audio (`[check] mode = "music"`), and a health path that is not `/health`.

The boot check runs against a stand-in server that returns a generated WAV file, so no chip and no install is used."""
import io
import json
import math
import os
import struct
import sys
import textwrap
import wave
from pathlib import Path

import pytest

from orchard import can

REV = "53fc7557ea3101d3efbe5e50598bd0bea7f44344"

SPEC = textwrap.dedent('''
    [can]
    name = "toy-music-blackhole"
    kind = "tt-dit-server"
    app = "toy_asgi:app"
    model_py = "toy_model.py"
    readme = "README.src.md"
    port = 20123
    requirements = ["ttnn==0.80.0", "numpy==1.26.4", "fastapi==0.136.3", "uvicorn==0.45.0", "pydantic==2.12.5", "pillow==10.4.0"]
    [weights]
    repo = "someone/toy-music"
    revision = "%s"
    [[extra_tree]]
    src = "code"
    dst = "."
    [check]
    mode = "music"
    health_path = "/v1/health"
    lyrics = "[verse]\\nla la la"
    caption = "pop"
    seconds = 4.0
    sample_rate = 32000
''' % REV)


def make_base(tmp_path, spec_text=SPEC):
    b = tmp_path / "base"
    (b / "code" / "pkg" / "sub").mkdir(parents=True)
    (b / "code" / "pkg" / "__init__.py").write_text("")
    (b / "code" / "pkg" / "sub" / "mod.py").write_text("X = 1\n")
    (b / "code" / "pkg" / "data.json").write_text("{}\n")
    (b / "toy_model.py").write_text("")
    (b / "README.src.md").write_text("# toy\n")
    (b / "can.toml").write_text(spec_text)
    return b


def load(b, text=None):
    if text is not None:
        (b / "can.toml").write_text(text)
    return can.load_spec(b / "can.toml", base=b)


# ---- [[extra_tree]]

def test_an_extra_tree_becomes_one_extra_code_entry_per_regular_file(tmp_path):
    b = make_base(tmp_path)
    s = load(b)
    assert sorted(dst for _, dst in s.extra_code) == ["pkg/__init__.py", "pkg/data.json", "pkg/sub/mod.py"]
    assert all(src.is_file() and src.is_relative_to(b.resolve()) for src, _ in s.extra_code)


def test_an_extra_tree_skips_python_caches(tmp_path):
    b = make_base(tmp_path)
    (b / "code" / "pkg" / "__pycache__").mkdir()
    (b / "code" / "pkg" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"x")
    (b / "code" / "pkg" / "stray.pyc").write_bytes(b"x")
    assert not [d for _, d in load(b).extra_code if "pyc" in d or "__pycache__" in d]


def test_an_extra_tree_skips_a_pycache_directory_whatever_it_holds(tmp_path):
    b = make_base(tmp_path)
    (b / "code" / "pkg" / "__pycache__").mkdir()
    (b / "code" / "pkg" / "__pycache__" / "note.txt").write_text("x")
    assert not [d for _, d in load(b).extra_code if "__pycache__" in d]


def test_an_extra_tree_that_is_a_link_to_another_directory_inside_base_is_refused(tmp_path):
    b = make_base(tmp_path)
    (b / "alias").symlink_to(b / "code")
    with pytest.raises(can.CanError, match="not a directory|link"):
        load(b, SPEC.replace('src = "code"', 'src = "alias"'))


def test_an_extra_tree_destination_is_a_directory_inside_extra_code(tmp_path):
    b = make_base(tmp_path)
    s = load(b, SPEC.replace('dst = "."', 'dst = "vendor"'))
    assert sorted(dst for _, dst in s.extra_code) == ["vendor/pkg/__init__.py", "vendor/pkg/data.json", "vendor/pkg/sub/mod.py"]


@pytest.mark.parametrize("old,new,words", [
    ('src = "code"', 'src = "../code"', "relative path inside"),
    ('src = "code"', 'src = "/etc"', "relative path inside"),
    ('src = "code"', 'src = "missing"', "is not a directory"),
    ('src = "code"', 'src = "toy_model.py"', "is not a directory"),
    ('dst = "."', 'dst = "../x"', "no '..'"),
    ('dst = "."', 'dst = "/x"', "no '..'"),
])
def test_a_bad_extra_tree_is_refused(tmp_path, old, new, words):
    b = make_base(tmp_path)
    with pytest.raises(can.CanError, match=words):
        load(b, SPEC.replace(old, new))


def test_an_extra_tree_with_a_link_inside_is_refused(tmp_path):
    b = make_base(tmp_path)
    (tmp_path / "outside.py").write_text("x = 1\n")
    (b / "code" / "pkg" / "link.py").symlink_to(tmp_path / "outside.py")
    with pytest.raises(can.CanError, match="link"):
        load(b)


def test_an_extra_tree_that_is_a_link_to_outside_is_refused(tmp_path):
    b = make_base(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.py").write_text("x = 1\n")
    (b / "linked").symlink_to(outside)
    with pytest.raises(can.CanError, match="outside|link"):
        load(b, SPEC.replace('src = "code"', 'src = "linked"'))


def test_an_empty_extra_tree_is_refused(tmp_path):
    b = make_base(tmp_path)
    (b / "empty").mkdir()
    with pytest.raises(can.CanError, match="no files"):
        load(b, SPEC.replace('src = "code"', 'src = "empty"'))


def test_a_file_named_by_both_extra_code_and_a_tree_is_refused(tmp_path):
    b = make_base(tmp_path)
    (b / "one.py").write_text("")
    text = SPEC + '[[extra_code]]\nsrc = "one.py"\ndst = "pkg/data.json"\n'
    with pytest.raises(can.CanError, match="twice"):
        load(b, text)


# ---- the check keys

def test_the_music_check_keys_load(tmp_path):
    ck = load(make_base(tmp_path)).check
    assert (ck.mode, ck.health_path, ck.seconds, ck.sample_rate) == ("music", "/v1/health", 4.0, 32000)


def test_an_unknown_check_mode_is_refused(tmp_path):
    with pytest.raises(can.CanError, match="mode"):
        load(make_base(tmp_path), SPEC.replace('mode = "music"', 'mode = "video"'))


def test_a_health_path_must_be_a_path(tmp_path):
    with pytest.raises(can.CanError, match="health_path"):
        load(make_base(tmp_path), SPEC.replace('"/v1/health"', '"http://elsewhere/health"'))


def test_the_chat_check_still_defaults_to_health_and_chat(tmp_path):
    b = make_base(tmp_path)
    s = load(b, SPEC.split("[check]")[0])
    assert (s.check.mode, s.check.health_path) == ("chat", "/health")


# ---- wav_stats

def wav_bytes(seconds=1.0, rate=32000, channels=2, amp=0.5, silent_after=None, sampwidth=2):
    n = int(seconds * rate)
    frames = bytearray()
    for i in range(n):
        v = 0 if (silent_after is not None and i / rate >= silent_after) else int(amp * 32767 * math.sin(2 * math.pi * 440 * i / rate))
        frames += struct.pack("<h", v) * channels
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def test_wav_stats_reads_rate_length_peak_and_loud_fraction():
    st = can.wav_stats(wav_bytes(seconds=2.0, rate=32000, amp=0.5))
    assert (st["sample_rate"], st["channels"]) == (32000, 2)
    assert abs(st["seconds"] - 2.0) < 1e-6
    assert 0.49 < st["peak"] < 0.51
    assert st["loud_fraction"] == 1.0


def test_wav_stats_counts_the_silent_half_of_a_clip():
    st = can.wav_stats(wav_bytes(seconds=2.0, amp=0.5, silent_after=1.0))
    assert 0.45 <= st["loud_fraction"] <= 0.55


def test_wav_stats_of_silence_has_zero_peak_and_no_loud_windows():
    st = can.wav_stats(wav_bytes(seconds=1.0, amp=0.0))
    assert st["peak"] == 0.0 and st["loud_fraction"] == 0.0


def test_wav_stats_refuses_a_file_whose_header_promises_more_audio_than_it_holds():
    good = wav_bytes(seconds=2.0, rate=8000)
    short = good[:44] + good[44:44 + 1600]            # the header still declares 2 s; 100 ms of PCM follows
    with pytest.raises(can.CanError, match="holds"):
        can.wav_stats(short)


@pytest.mark.parametrize("data", [b"", b"not a wav", b"RIFF" + b"\0" * 40])
def test_wav_stats_refuses_data_that_is_not_a_pcm16_wav(data):
    with pytest.raises(can.CanError, match="WAV"):
        can.wav_stats(data)


def test_wav_stats_refuses_eight_bit_audio():
    with pytest.raises(can.CanError, match="16-bit"):
        can.wav_stats(_wav8())


def _wav8():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(b"\x80" * 800)
    return buf.getvalue()


# ---- the music boot check, against a stand-in server

SERVER = textwrap.dedent('''
    import io, json, math, os, struct, sys, wave
    from http.server import BaseHTTPRequestHandler, HTTPServer
    RATE = int(os.environ.get("TOY_RATE", "32000"))
    SECONDS = float(os.environ.get("TOY_SECONDS", "4.0"))
    AMP = float(os.environ.get("TOY_AMP", "0.5"))
    SOUND_S = float(os.environ.get("TOY_SOUND_SECONDS", "999"))      # sound for this long, then silence
    STATUS = int(os.environ.get("TOY_STATUS", "200"))
    HEALTH = os.environ.get("TOY_HEALTH", "/v1/health")
    def make():
        n = int(SECONDS * RATE); frames = bytearray()
        for i in range(n):
            a = AMP if i / RATE < SOUND_S else 0.0
            frames += struct.pack("<h", int(a * 32767 * math.sin(2 * math.pi * 440 * i / RATE))) * 2
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(2); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(bytes(frames))
        return buf.getvalue()
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path == HEALTH:
                self._send(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": []})
            else:
                self._send(404, {"error": "no"})
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n))
            with open(os.environ["TOY_LAST_REQUEST"], "w") as f:
                json.dump(req, f)
            if self.path != "/v1/audio/speech":
                self._send(404, {"error": "no"})
            elif STATUS != 200:
                self._send(STATUS, {"detail": "busy"})
            else:
                self._send(200, make(), "audio/wav")
        def log_message(self, *a): pass
    print(json.dumps({"ready": True}), flush=True)
    HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
''')


def fake_bundle(tmp_path, **env):
    b = tmp_path / "fake-bundle"
    b.mkdir()
    (b / "server.py").write_text(SERVER)
    (b / "install.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    exports = "".join(f'export {k}="{v}"\n' for k, v in env.items())
    (b / "run.sh").write_text(f'#!/usr/bin/env bash\n{exports}export TOY_LAST_REQUEST="{tmp_path}/last-request.json"\n'
                              f'exec "{sys.executable}" "$(dirname "$0")/server.py"\n')
    return b


@pytest.fixture
def spec(tmp_path):
    return load(make_base(tmp_path))


def run_check(spec, tmp_path, **env):
    return can.boot_check(fake_bundle(tmp_path, **env), tmp_path / "work", spec, install=False, health_timeout=30)


def test_the_music_boot_check_passes_a_clip_that_is_loud_and_the_right_rate_and_length(spec, tmp_path):
    ev = run_check(spec, tmp_path)
    assert ev["passed"], ev
    assert {"health", "models", "music_request", "music_decodes", "music_sample_rate", "music_length", "music_not_silent", "health_after"} <= set(ev["checks"])
    assert ev["audio"]["sample_rate"] == 32000 and abs(ev["audio"]["seconds"] - 4.0) < 0.01
    assert "chat" not in ev["checks"]


def test_the_music_boot_check_sends_the_lyrics_caption_duration_seed_and_a_wav_format(spec, tmp_path):
    run_check(spec, tmp_path)
    req = json.loads((tmp_path / "last-request.json").read_text())
    assert req["input"] == "[verse]\nla la la" and req["instructions"] == "pop"
    assert req["audio_duration"] == 4.0 and req["response_format"] == "wav" and req["sample_rate"] == 32000
    assert isinstance(req["seed"], int)


def test_the_music_boot_check_fails_a_silent_clip(spec, tmp_path):
    ev = run_check(spec, tmp_path, TOY_AMP="0")
    assert not ev["passed"] and not ev["checks"]["music_not_silent"]["ok"]


def test_the_music_boot_check_fails_a_clip_that_is_quiet_all_through(spec, tmp_path):
    ev = run_check(spec, tmp_path, TOY_AMP="0.02")       # every window is above -40 dBFS, and the loudest sample is 2% of full scale
    assert not ev["passed"] and not ev["checks"]["music_not_silent"]["ok"] and ev["audio"]["loud_fraction"] == 1.0


def test_the_music_boot_check_fails_a_clip_that_is_a_click_and_then_silence(spec, tmp_path):
    ev = run_check(spec, tmp_path, TOY_SOUND_SECONDS="0.2")
    assert not ev["passed"] and not ev["checks"]["music_not_silent"]["ok"] and ev["audio"]["peak"] > 0.4


def test_the_music_boot_check_fails_a_clip_at_the_wrong_sample_rate(spec, tmp_path):
    ev = run_check(spec, tmp_path, TOY_RATE="16000")
    assert not ev["passed"] and not ev["checks"]["music_sample_rate"]["ok"]


def test_the_music_boot_check_fails_a_clip_that_is_too_short(spec, tmp_path):
    ev = run_check(spec, tmp_path, TOY_SECONDS="1.0")
    assert not ev["passed"] and not ev["checks"]["music_length"]["ok"]


def test_the_music_boot_check_fails_a_clip_that_is_too_long(spec, tmp_path):
    ev = run_check(spec, tmp_path, TOY_SECONDS="9.0")
    assert not ev["passed"] and not ev["checks"]["music_length"]["ok"]


def test_the_music_boot_check_fails_when_the_server_refuses_the_request(spec, tmp_path):
    ev = run_check(spec, tmp_path, TOY_STATUS="429")
    assert not ev["passed"] and not ev["checks"]["music_request"]["ok"]
    assert "429" in ev["checks"]["music_request"]["detail"]


def test_the_music_boot_check_waits_on_the_spec_health_path_and_not_on_slash_health(spec, tmp_path):
    ev = can.boot_check(fake_bundle(tmp_path, TOY_HEALTH="/health"), tmp_path / "work", spec, install=False, health_timeout=6)
    assert not ev["passed"] and not ev["checks"]["health"]["ok"] and "/v1/health" in ev["checks"]["health"]["detail"]


def test_the_music_boot_check_stops_the_server(spec, tmp_path):
    ev = run_check(spec, tmp_path)
    assert not Path(f"/proc/{ev['server_pid']}").exists() or Path(f"/proc/{ev['server_pid']}/stat").read_text().split()[2] == "Z"


def test_the_music_boot_check_checks_the_start_up_log(spec, tmp_path):
    text = (tmp_path / "base" / "can.toml").read_text() + "ready_contains = ['\"ready\": true', 'never printed']\n"
    s = load(tmp_path / "base", text)
    ev = run_check(s, tmp_path)
    assert not ev["passed"] and ev["checks"]['log contains "ready": true']["ok"] and not ev["checks"]["log contains never printed"]["ok"]


def test_a_failed_decode_of_the_audio_is_reported_and_not_raised(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(can, "wav_stats", lambda data: (_ for _ in ()).throw(can.CanError("not a WAV file")))
    ev = run_check(spec, tmp_path)
    assert not ev["passed"] and not ev["checks"]["music_decodes"]["ok"] and "not a WAV file" in ev["checks"]["music_decodes"]["detail"]


CHAT_SERVER = textwrap.dedent('''
    import json, os
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class H(BaseHTTPRequestHandler):
        def _s(self, code, body, raw=None):
            d = raw or json.dumps(body).encode(); self.send_response(code); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
        def do_GET(self):
            self._s(200 if self.path in ("/v1/health", "/v1/models") else 404, {"status": "ok", "data": []})
        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if req.get("stream"):
                self._s(200, None, b"data: [DONE]\\n\\n")
            else:
                self._s(200, {"choices": [{"message": {"content": "Paris"}}], "timings": {"decode_tok_s": 99.0}})
        def log_message(self, *a): pass
    HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
''')


def test_the_chat_boot_check_also_uses_the_health_path_for_its_final_probe(tmp_path):
    text = SPEC.split("[check]")[0] + '[check]\nhealth_path = "/v1/health"\n'
    s = load(make_base(tmp_path), text)
    fb = tmp_path / "chatb"
    fb.mkdir()
    (fb / "server.py").write_text(CHAT_SERVER)
    (fb / "install.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (fb / "run.sh").write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$(dirname "$0")/server.py"\n')
    ev = can.boot_check(fb, tmp_path / "work", s, install=False, health_timeout=30)
    assert ev["checks"]["chat"]["ok"] and ev["checks"]["health_after"]["ok"], ev
