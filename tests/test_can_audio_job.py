"""The audio mode of the canning step: the spec keys (`[check] mode = "audio-job"`, `[[extra_tree]]`) and the job-server boot check.

The boot check is run against a stand-in job server (standard library only), so no chip is used and nothing is installed."""
import struct
import sys
import textwrap
from pathlib import Path

import pytest

from orchard import can

REV = "8bdfb53c6b0fbc62113cd571fd37373efe15fa14"

SPEC = textwrap.dedent('''
    [can]
    name = "toy-audio-blackhole"
    kind = "tt-dit-server"
    app = "toy_asgi:app"
    model_py = "toy_model.py"
    readme = "README.src.md"
    port = 20171
    requirements = ["ttnn==0.80.0", "numpy==1.26.4", "fastapi==0.136.3", "uvicorn==0.45.0", "pydantic==2.12.5", "pillow==10.4.0"]
    [weights]
    repo = "someone/toy-audio"
    revision = "%s"
    [[extra_tree]]
    src = "lib/toylib"
    dst = "toylib"
    [check]
    mode = "audio-job"
    prompt = "A short tune"
    audio_seconds = 2
    audio_steps = 8
    audio_seed = 42
    max_rtf = 5.0
    ready_contains = ['"ready": true']
''' % REV)


@pytest.fixture
def base(tmp_path):
    b = tmp_path / "base"
    (b / "lib" / "toylib" / "sub").mkdir(parents=True)
    (b / "lib" / "toylib" / "__init__.py").write_text("X = 1\n")
    (b / "lib" / "toylib" / "sub" / "m.py").write_text("Y = 2\n")
    (b / "lib" / "toylib" / "sub" / "data.json").write_text("{}\n")
    (b / "lib" / "toylib" / "__pycache__").mkdir()
    (b / "lib" / "toylib" / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\0")
    (b / "lib" / "toylib" / "__pycache__" / "notes.txt").write_text("a file in a cache directory is not copied\n")
    (b / "lib" / "toylib" / "stray.pyc").write_bytes(b"\0")          # a compiled file outside a cache directory is not copied either
    (b / "toy_model.py").write_text("x = 1\n")
    (b / "README.src.md").write_text("# toy\n")
    (b / "can.toml").write_text(SPEC)
    return b


def load(base, text=None):
    if text is not None:
        (base / "can.toml").write_text(text)
    return can.load_spec(base / "can.toml", base=base)


# ---- [[extra_tree]]

def test_an_extra_tree_becomes_one_extra_code_entry_per_file_without_caches(base):
    s = load(base)   # the fixture also holds __pycache__/notes.txt and stray.pyc, which must not be listed
    assert sorted(d for _, d in s.extra_code) == ["toylib/__init__.py", "toylib/sub/data.json", "toylib/sub/m.py"]
    assert all(src.is_file() for src, _ in s.extra_code)


def test_an_extra_tree_refuses_a_link_inside_it(base):
    (base / "lib" / "toylib" / "evil.py").symlink_to(base / "toy_model.py")
    with pytest.raises(can.CanError, match="link"):
        load(base)


@pytest.mark.parametrize("src", ["../outside", "/etc", "lib/missing"])
def test_an_extra_tree_must_be_a_directory_inside_base(base, src):
    with pytest.raises(can.CanError, match="extra_tree"):
        load(base, SPEC.replace('src = "lib/toylib"', f'src = "{src}"'))


def test_an_extra_tree_cannot_reach_outside_base_through_a_linked_directory(base, tmp_path):
    out = tmp_path / "elsewhere"
    out.mkdir()
    (out / "a.py").write_text("1\n")
    (base / "lib" / "linked").symlink_to(out, target_is_directory=True)
    with pytest.raises(can.CanError, match="extra_tree"):
        load(base, SPEC.replace('src = "lib/toylib"', 'src = "lib/linked"'))


def test_an_extra_tree_dst_must_be_relative_without_dotdot(base):
    with pytest.raises(can.CanError, match="no '..'"):
        load(base, SPEC.replace('dst = "toylib"', 'dst = "../toylib"'))


def test_an_extra_tree_dst_colliding_with_an_extra_code_file_is_refused(base):
    text = SPEC.replace("[check]", '[[extra_code]]\nsrc = "toy_model.py"\ndst = "toylib/__init__.py"\n[check]')
    with pytest.raises(can.CanError, match="twice"):
        load(base, text)


def test_a_tree_with_no_files_is_refused(base):
    (base / "empty").mkdir()
    with pytest.raises(can.CanError, match="no files"):
        load(base, SPEC.replace('src = "lib/toylib"', 'src = "empty"'))


# ---- the check keys

def test_an_extra_tree_inside_a_linked_directory_that_leads_outside_base_is_refused(base, tmp_path):
    out = tmp_path / "elsewhere2"
    (out / "sub").mkdir(parents=True)
    (out / "sub" / "a.py").write_text("1\n")
    (base / "dirlink").symlink_to(out, target_is_directory=True)        # the tree itself is a plain directory below a linked one
    with pytest.raises(can.CanError, match="extra_tree"):
        load(base, SPEC.replace('src = "lib/toylib"', 'src = "dirlink/sub"'))


def test_an_audio_check_needs_its_own_prompt(base):
    with pytest.raises(can.CanError, match="needs a prompt"):
        load(base, SPEC.replace('prompt = "A short tune"\n', ""))


def test_the_audio_check_keys_are_read(base):
    ck = load(base).check
    assert (ck.mode, ck.audio_seconds, ck.audio_steps, ck.audio_seed, ck.max_rtf) == ("audio-job", 2, 8, 42, 5.0)


@pytest.mark.parametrize("old,new,words", [
    ('mode = "audio-job"', 'mode = "video"', "mode must be"),
    ("audio_seconds = 2", "audio_seconds = 0", "audio_seconds"),
    ("audio_steps = 8", "audio_steps = 2.5", "audio_steps"),
    ("max_rtf = 5.0", "max_rtf = -1", "max_rtf"),
])
def test_a_bad_audio_check_is_refused(base, old, new, words):
    with pytest.raises(can.CanError, match=words):
        load(base, SPEC.replace(old, new))


def test_chat_keys_in_an_audio_check_and_audio_keys_in_a_chat_check_are_refused(base):
    with pytest.raises(can.CanError, match="chat"):
        load(base, SPEC.replace("max_rtf = 5.0", "max_rtf = 5.0\nmin_decode_tok_s = 30.0"))
    with pytest.raises(can.CanError, match="audio"):
        load(base, SPEC.replace('mode = "audio-job"\n', ""))


# ---- the boot check

SERVER = textwrap.dedent('''
    import json, os, struct, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    MODE = os.environ.get("TOY_MODE", "good")
    JOBS = {}
    def flac(seconds, rate=44100):
        packed = (rate << 44) | (1 << 41) | (15 << 36) | int(seconds * rate)
        return b"fLaC" + bytes([0x80, 0, 0, 34]) + b"\\0" * 10 + struct.pack(">Q", packed) + b"\\0" * 16
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok"})
            parts = self.path.strip("/").split("/")
            if parts[:3] == ["v1", "audio", "jobs"] and len(parts) >= 4 and parts[3] in JOBS:
                j = JOBS[parts[3]]
                if len(parts) == 5:
                    secs = j["settings"]["duration"] * (0.5 if MODE == "short" else 1)
                    return self._send(200, b"not audio" if MODE == "notflac" else flac(secs), "audio/flac")
                if MODE == "second_fails" and len(JOBS) >= 2:
                    return self._send(200, {"id": parts[3], "status": "error", "error": "second boom", "settings": j["settings"]})
                if MODE == "garbage":
                    return self._send(200, b'{"status": "done", "status": "running", "x": NaN}')     # a repeated key and NaN: the guarded parser refuses it
                if MODE == "error":
                    return self._send(200, {"id": parts[3], "status": "error", "error": "boom", "settings": j["settings"]})
                if MODE == "never":
                    return self._send(200, {"id": parts[3], "status": "running", "settings": j["settings"]})
                total = 50.0 if MODE == "slow" else 1.5
                return self._send(200, {"id": parts[3], "status": "done", "settings": j["settings"],
                                        "stats": {"audio_s": j["settings"]["duration"], "total_s": total, "audio_format": "flac", "stages": {}}})
            self._send(404, {"detail": "not found"})
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n))
            if MODE == "reject":
                return self._send(429, {"detail": "the queue is full"})
            jid = "%032x" % (len(JOBS) + 1)
            JOBS[jid] = {"settings": {"prompt": body["prompt"], "duration": body["duration"], "steps": body["steps"], "seed": body["seed"]}}
            self._send(202, {"id": jid, "status": "queued"})
        def log_message(self, *a): pass
    print(json.dumps({"ready": True}), flush=True)
    HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
''')


def fake_bundle(tmp_path, mode="good"):
    b = tmp_path / "fake-bundle"
    b.mkdir(parents=True)
    (b / "server.py").write_text(SERVER)
    (b / "install.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (b / "run.sh").write_text(f'#!/usr/bin/env bash\nexport TOY_MODE="{mode}"\nexec "{sys.executable}" "$(dirname "$0")/server.py"\n')
    return b


def boot(tmp_path, base, mode="good", **kw):
    return can.boot_check(fake_bundle(tmp_path, mode), tmp_path / "work", load(base), install=False, health_timeout=30, **kw)


def test_the_audio_boot_check_passes_a_healthy_job_server_and_records_both_job_times(tmp_path, base):
    ev = boot(tmp_path, base)
    assert ev["passed"], ev
    assert {"health", "audio_job", "audio_file", "audio_length", "audio_rate", "audio_second_job", 'log contains "ready": true'} <= set(ev["checks"])
    assert "chat" not in ev["checks"] and "models" not in ev["checks"]
    assert ev["audio"]["first"]["audio_s"] == 2 and ev["audio"]["first"]["server_total_s"] == 1.5
    assert ev["audio"]["second"]["server_total_s"] == 1.5 and ev["audio"]["first"]["settings"]["seed"] == 42


def test_the_audio_boot_check_sends_the_spec_settings(tmp_path, base):
    ev = boot(tmp_path, base)
    s = ev["audio"]["first"]["settings"]
    assert (s["prompt"], s["duration"], s["steps"], s["seed"]) == ("A short tune", 2, 8, 42)


@pytest.mark.parametrize("mode,failed", [("error", "audio_job"), ("never", "audio_job"), ("notflac", "audio_file"), ("short", "audio_length"),
                                         ("slow", "audio_rate"),
                                         ("second_fails", "audio_second_job"), ("reject", "audio_job"), ("garbage", "audio_job")])
def test_the_audio_boot_check_fails_each_fault(tmp_path, base, mode, failed):
    ev = boot(tmp_path, base, mode, job_timeout=3)
    assert not ev["passed"] and not ev["checks"][failed]["ok"], ev["checks"]


def test_the_audio_boot_check_stops_the_server(tmp_path, base):
    ev = boot(tmp_path, base)
    assert not Path(f"/proc/{ev['server_pid']}").exists() or Path(f"/proc/{ev['server_pid']}/stat").read_text().split()[2] == "Z"


def test_the_error_text_of_a_failed_job_is_in_the_evidence(tmp_path, base):
    ev = boot(tmp_path, base, "error", job_timeout=3)
    assert "boom" in ev["checks"]["audio_job"]["detail"]


def test_without_max_rtf_the_rate_is_recorded_and_not_judged(tmp_path, base):
    spec = load(base, SPEC.replace("max_rtf = 5.0\n", ""))
    out = can.boot_check(fake_bundle(tmp_path, "slow"), tmp_path / "work2", spec, install=False, health_timeout=30)
    assert out["passed"] and "audio_rate" not in out["checks"] and out["audio"]["first"]["rtf"] == 25.0


def test_a_truncated_wav_is_refused_like_a_cut_short_clip():
    import io, wave
    from orchard import can_audio
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000); w.writeframes(b"\0\0" * 8000)
    good = buf.getvalue()
    assert can_audio.clip_seconds(good) == 1.0
    with pytest.raises(ValueError, match="cut short"):
        can_audio.clip_seconds(good[:-100])


def test_a_malformed_wav_is_a_value_error_and_never_another_exception():
    from orchard import can_audio
    for bad in (b"RIFF\0\0\0\0WAVE", b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0", b"RIFF\0\0\0\0WAVEdata\0\0\0\0", b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0" + b"\0" * 16 + b"data\0\0\0\0"):
        with pytest.raises(ValueError):
            can_audio.clip_seconds(bad)


def test_the_rate_uses_the_validated_file_length_and_not_the_servers_audio_seconds(tmp_path, base):
    """The server says 2 s of audio but the clip's own header says 1 s (mode short halves it): rtf = 1.5 / 1.0 = 1.5, not 0.75."""
    ev = boot(tmp_path, base, "short", job_timeout=3)
    assert ev["audio"]["first"]["rtf"] == 1.5 and ev["audio"]["first"]["file_s"] == 1.0


def test_the_second_jobs_rate_also_uses_the_file_length_and_an_unreadable_clip_has_no_rate(tmp_path, base):
    spec = load(base, SPEC.replace("audio_seconds = 2", "audio_seconds = 4"))
    out = can.boot_check(fake_bundle(tmp_path / "a", "short"), tmp_path / "w1", spec, install=False, health_timeout=30)
    assert out["audio"]["second"]["file_s"] == 2.0 and out["audio"]["second"]["rtf"] == 0.75      # 1.5 s of generation over the 2 s clip, not over the 4 s the server claims
    out = can.boot_check(fake_bundle(tmp_path / "b", "notflac"), tmp_path / "w2", load(base), install=False, health_timeout=30)
    assert out["audio"]["second"]["rtf"] is None and not out["checks"]["audio_rate"]["ok"]
