"""The audio boot check of the canning step (orchard/can.py, `[check] mode = "audio"`): a model that answers a request with a WAV clip.

The server is a stand-in that returns a synthesised clip, so no chip is used and nothing is installed. Each fault changes the clip or the
server and the matching check must fail; the healthy clip must pass every check."""
import json
import math
import struct
import sys
import textwrap
from pathlib import Path

import pytest

from orchard import can

SPEC = textwrap.dedent('''
    [can]
    name = "toy-music-blackhole"
    kind = "tt-dit-server"
    app = "toy_asgi:app"
    model_py = "toy_model.py"
    readme = "README.src.md"
    port = 20124
    requirements = ["ttnn==0.80.0"]
    [weights]
    repo = "someone/toy-music"
    revision = "%s"
    [check]
    mode = "audio"
    path = "/v1/music"
    body = { prompt = "calm piano", duration = 2, seed = 42 }
    expect_seconds = 2.0
    sample_rate = 48000
    channels = 2
    min_loud_fraction = 0.4
    repeat_identical = true
    ready_contains = ['"event": "ready"']
''' % ("d" * 40))


def wav(seconds=2.0, rate=48000, channels=2, level=0.3, fmt="f32"):
    """A WAV of a sine at `level` (a fraction of full scale), 32-bit float or 16-bit integer, as ACE-Step writes the first."""
    n = int(seconds * rate)
    samples = [level * math.sin(2 * math.pi * 440 * i / rate) for i in range(n)]
    frames = [s for s in samples for _ in range(channels)]
    if fmt == "f32":
        data, tag, bits = struct.pack(f"<{len(frames)}f", *frames), 3, 32
    else:
        data, tag, bits = struct.pack(f"<{len(frames)}h", *[int(s * 32767) for s in frames]), 1, 16
    fmt_chunk = struct.pack("<HHIIHH", tag, channels, rate, rate * channels * bits // 8, channels * bits // 8, bits)
    body = b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt_chunk + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


SERVER = textwrap.dedent('''
    import json, os, sys, time, itertools
    from http.server import BaseHTTPRequestHandler, HTTPServer
    CLIP = open(os.environ["TOY_CLIP"], "rb").read()
    STATUS = int(os.environ.get("TOY_STATUS", "200"))
    DELAY = float(os.environ.get("TOY_DELAY", "0"))
    COUNT = itertools.count()
    class H(BaseHTTPRequestHandler):
        def _send(self, code, data, ctype="application/json"):
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            self._send(200, b'{"status": "ok"}')
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n))
            assert req == {"prompt": "calm piano", "duration": 2, "seed": 42}, req
            k = next(COUNT)
            time.sleep(DELAY if k else 0)                        # only the warm request is slow when asked to be
            same = os.environ.get("TOY_SAME", "1")
            if same == "header" and k:                           # same samples, one more chunk before the data: a different file
                clip = CLIP[:12] + b"LIST" + (4).to_bytes(4, "little") + b"tag!" + CLIP[12:]
            else:
                clip = CLIP if same == "1" or k == 0 else CLIP[:-4] + b"\\x01\\x00\\x00\\x00"
            self._send(STATUS, clip if STATUS == 200 else b'{"error": "x"}', "audio/wav" if STATUS == 200 else "application/json")
        def log_message(self, *a): pass
    print(json.dumps({"event": "ready"}), flush=True)
    HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
''')


@pytest.fixture
def base(tmp_path):
    b = tmp_path / "base"
    b.mkdir()
    (b / "toy_model.py").write_text("app = None\n")
    (b / "README.src.md").write_text("# toy\n")
    (b / "can.toml").write_text(SPEC)
    return b


def load(base, old=None, new=None):
    p = base / "can.toml"
    if old is not None:
        text = p.read_text()
        assert old in text
        p.write_text(text.replace(old, new))
    return can.load_spec(p, base=base)


def bundle(tmp_path, clip, **env):
    b = tmp_path / "toy-bundle"
    b.mkdir()
    (b / "server.py").write_text(SERVER)
    (b / "clip.wav").write_bytes(clip)
    (b / "install.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    exports = "".join(f'export {k}="{v}"\n' for k, v in env.items())
    (b / "run.sh").write_text(f'#!/usr/bin/env bash\nexport TOY_CLIP="$(dirname "$0")/clip.wav"\n{exports}exec "{sys.executable}" "$(dirname "$0")/server.py"\n')
    return b


def boot(tmp_path, base, clip=None, spec=None, **env):
    return can.boot_check(bundle(tmp_path, clip if clip is not None else wav(), **env), tmp_path / "work", spec or load(base), install=False,
                          health_timeout=30)


# ---- the spec

def test_the_audio_spec_loads(base):
    s = load(base)
    assert s.check.mode == "audio" and s.check.path == "/v1/music" and s.check.body == {"prompt": "calm piano", "duration": 2, "seed": 42}
    assert (s.check.expect_seconds, s.check.sample_rate, s.check.channels, s.check.repeat_identical) == (2.0, 48000, 2, True)


@pytest.mark.parametrize("old,new,words", [
    ('mode = "audio"', 'mode = "video"', "mode must be"),
    ('path = "/v1/music"', 'path = "v1/music"', "path must start with"),
    ('body = { prompt = "calm piano", duration = 2, seed = 42 }', 'body = "text"', "body must be a table"),
    ("expect_seconds = 2.0", "expect_seconds = 0", "expect_seconds"),
    ("min_loud_fraction = 0.4", "min_loud_fraction = 2", "min_loud_fraction"),
])
def test_a_bad_audio_check_is_refused_with_a_message_that_names_the_fault(base, old, new, words):
    with pytest.raises(can.CanError, match=words):
        load(base, old, new)


def test_an_audio_check_needs_a_path_a_body_and_a_length(base):
    for line in ('path = "/v1/music"\n', 'expect_seconds = 2.0\n'):
        text = SPEC.replace(line, "")
        (base / "can.toml").write_text(text)
        with pytest.raises(can.CanError, match="audio check needs"):
            can.load_spec(base / "can.toml", base=base)


def test_a_chat_check_is_unchanged_by_the_audio_fields(base):
    (base / "can.toml").write_text(SPEC.split("[check]")[0] + '[check]\nexpect = "Paris"\n')
    c = can.load_spec(base / "can.toml", base=base).check
    assert c.mode == "chat" and c.body is None and c.expect_seconds is None


ROOT = Path(__file__).resolve().parent.parent


def test_the_acestep_spec_loads_and_every_file_it_names_exists():
    s = can.load_spec(ROOT / "config" / "can.acestep.toml", base=ROOT)
    assert (s.kind, s.app, s.mesh, s.device_count, s.port) == ("tt-dit-server", "acestep_asgi:app", "P150", 1, 20140)
    assert s.revision == "19671f406d603126926c1b7e2adc169acbcade22" and s.check.mode == "audio" and s.check.repeat_identical
    assert all(src.is_file() for src, _ in s.extra_code + s.bundle_files)
    assert all("==" in p for p in s.requirements) and any(p.startswith("ttnn==") for p in s.requirements)


def test_the_acestep_pins_file_names_the_revision_the_spec_names():
    pins = json.loads((ROOT / "pkgsrc" / "can" / "acestep-pins.json").read_text())
    assert pins["model_revision"] == can.load_spec(ROOT / "config" / "can.acestep.toml", base=ROOT).revision


# ---- reading a WAV

def test_wav_facts_reads_float32_and_sixteen_bit_clips():
    for fmt in ("f32", "i16"):
        facts = can.wav_facts(wav(1.0, fmt=fmt))
        assert (facts["sample_rate"], facts["channels"], round(facts["seconds"], 3)) == (48000, 2, 1.0)
        assert facts["loud_window_fraction"] == 1.0 and facts["finite"]


def test_wav_facts_sees_silence_and_full_scale_and_non_finite_samples():
    assert can.wav_facts(wav(1.0, level=0.0))["loud_window_fraction"] == 0.0
    assert can.wav_facts(wav(1.0, level=1.5))["full_scale_fraction"] > 0.05
    clip = bytearray(wav(1.0))
    clip[-4:] = struct.pack("<f", float("nan"))
    assert not can.wav_facts(bytes(clip))["finite"]


@pytest.mark.parametrize("data", [b"", b"RIFF", b"not a wav at all", wav(1.0)[:30], b"RIFF\x04\x00\x00\x00WAVE"])
def test_wav_facts_refuses_what_is_not_a_clip(data):
    with pytest.raises(can.CanError):
        can.wav_facts(data)


# ---- the boot check

def test_a_healthy_clip_passes_every_check_and_the_server_is_stopped(tmp_path, base):
    ev = boot(tmp_path, base)
    assert ev["passed"], ev
    assert {"health", "audio_request", "audio_format", "audio_length", "audio_not_silent", "audio_finite", "repeat_identical", "health_after",
            'log contains "event": "ready"'} <= set(ev["checks"])
    assert ev["audio"]["sample_rate"] == 48000 and ev["first_request_s"] >= 0 and ev["warm_request_s"] >= 0
    assert not Path(f"/proc/{ev['server_pid']}").exists() or Path(f"/proc/{ev['server_pid']}/stat").read_text().split()[2] == "Z"


@pytest.mark.parametrize("clip,failing", [
    (lambda: wav(rate=44100), "audio_format"),
    (lambda: wav(channels=1), "audio_format"),
    (lambda: wav(seconds=1.0), "audio_length"),
    (lambda: wav(level=0.0), "audio_not_silent"),
    (lambda: wav(level=1.5), "audio_not_clipped"),
])
def test_a_clip_with_a_fault_fails_the_matching_check(tmp_path, base, clip, failing):
    ev = boot(tmp_path, base, clip=clip())
    assert not ev["passed"] and not ev["checks"][failing]["ok"], ev


def test_a_very_quiet_clip_fails_the_silence_check(tmp_path, base):
    ev = boot(tmp_path, base, clip=wav(level=0.002))          # about -54 dBFS: not digital silence, still not music
    assert not ev["passed"] and not ev["checks"]["audio_not_silent"]["ok"]


def test_a_clip_with_a_sample_that_is_not_finite_fails_the_finite_check(tmp_path, base):
    clip = bytearray(wav(2.0))
    clip[-4:] = struct.pack("<f", float("nan"))
    ev = boot(tmp_path, base, clip=bytes(clip))
    assert not ev["passed"] and not ev["checks"]["audio_finite"]["ok"]


def test_a_clip_that_is_not_a_wav_fails_the_request_check(tmp_path, base):
    ev = boot(tmp_path, base, clip=b"this is text")
    assert not ev["passed"] and not ev["checks"]["audio_request"]["ok"]


def test_a_503_fails_the_request_check(tmp_path, base):
    ev = boot(tmp_path, base, TOY_STATUS="503")
    assert not ev["passed"] and not ev["checks"]["audio_request"]["ok"] and "503" in ev["checks"]["audio_request"]["detail"]


def test_a_slow_warm_request_fails_when_the_spec_sets_a_limit(tmp_path, base):
    spec = load(base, "repeat_identical = true", "repeat_identical = true\nmax_warm_s = 0.5")
    ev = boot(tmp_path, base, spec=spec, TOY_DELAY="1.5")
    assert not ev["passed"] and not ev["checks"]["warm_time"]["ok"]


def test_a_fast_warm_request_passes_the_limit(tmp_path, base):
    spec = load(base, "repeat_identical = true", "repeat_identical = true\nmax_warm_s = 30")
    ev = boot(tmp_path, base, spec=spec)
    assert ev["passed"] and ev["checks"]["warm_time"]["ok"]


def test_no_warm_limit_means_no_warm_time_check(tmp_path, base):
    assert "warm_time" not in boot(tmp_path, base)["checks"]


def test_a_repeat_that_differs_fails_when_the_spec_asks_for_identical_clips(tmp_path, base):
    ev = boot(tmp_path, base, TOY_SAME="0")
    assert not ev["passed"] and not ev["checks"]["repeat_identical"]["ok"]


def test_a_repeat_with_the_same_samples_and_a_different_header_passes(tmp_path, base):
    """ACE-Step's WAV files differ byte for byte between identical requests (measured on the chip) while every sample is equal."""
    ev = boot(tmp_path, base, TOY_SAME="header")
    assert ev["passed"] and ev["checks"]["repeat_identical"]["ok"], ev


def test_a_repeat_that_differs_is_not_checked_when_the_spec_does_not_ask(tmp_path, base):
    spec = load(base, "repeat_identical = true", "repeat_identical = false")
    ev = boot(tmp_path, base, spec=spec, TOY_SAME="0")
    assert ev["passed"] and "repeat_identical" not in ev["checks"]


def test_the_start_up_log_is_checked(tmp_path, base):
    spec = load(base, "ready_contains = ['\"event\": \"ready\"']", "ready_contains = ['not in the log']")
    ev = boot(tmp_path, base, spec=spec)
    assert not ev["passed"] and not ev["checks"]["log contains not in the log"]["ok"]


def test_wav_facts_refuses_a_truncated_data_chunk():
    clip = wav(1.0)
    with pytest.raises(can.CanError, match="cut short"):
        can.wav_facts(clip[:-1000])


def test_a_truncated_clip_fails_the_boot_check(tmp_path, base):
    ev = boot(tmp_path, base, clip=wav(2.0)[:-1000])
    assert not ev["passed"] and not ev["checks"]["audio_request"]["ok"] and "cut short" in ev["checks"]["audio_request"]["detail"]


@pytest.mark.parametrize("port", [8000, 8011, 8012, 8780, 8100, 8103, 80, 70000, "20140", True])
def test_the_boot_port_override_follows_the_spec_port_rules(tmp_path, base, monkeypatch, port):
    from types import SimpleNamespace
    monkeypatch.setattr(can, "boot_check", lambda *a, **kw: pytest.fail("booted on a refused port"))
    args = SimpleNamespace(action="boot", spec=str(base / "can.toml"), base=str(base), bundle="b", work=str(tmp_path / "w"), hf_home=None,
                           health_timeout=5.0, port=port, run_dir=None)
    assert can.main(args) == 2


def test_the_spec_and_the_override_share_one_port_check():
    assert can.check_port(20140, "x") == 20140
    with pytest.raises(can.CanError, match="--port 8000 is not usable"):
        can.check_port(8000, "--port")


def test_the_boot_command_serves_on_the_port_it_is_given(tmp_path, base, monkeypatch, capsys):
    from types import SimpleNamespace
    seen = {}
    monkeypatch.setattr(can, "boot_check", lambda *a, **kw: seen.update(kw) or {"checks": {}, "passed": True})
    args = SimpleNamespace(action="boot", spec=str(base / "can.toml"), base=str(base), bundle="b", work=str(tmp_path / "w"), hf_home=None,
                           health_timeout=5.0, port=20149, run_dir=None)
    assert can.main(args) == 0 and seen["port"] == 20149


def test_the_evidence_file_is_written(tmp_path, base):
    boot(tmp_path, base)
    ev = json.loads((tmp_path / "work" / "boot-check.json").read_text())
    assert ev["passed"] and "audio" in ev
