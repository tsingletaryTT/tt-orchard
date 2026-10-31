"""The bundle's command-line client (pkgsrc/tools/audio_client.py), run against the job server's own stand-alone HTTP server with the fake generator."""
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod            # dataclasses looks its module up here
    spec.loader.exec_module(mod)
    return mod


client = load("audio_client_under_test", "pkgsrc/tools/audio_client.py")
jobs = load("stable_audio_jobs_for_client_test", "pkgsrc/server/stable_audio_jobs.py")


@pytest.fixture
def server():
    js = jobs.JobServer(jobs.FakeGenerator(delay_s=0), max_waiting=0)
    http = jobs.make_http_server(js, "127.0.0.1", 0)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    js.start()
    import time
    t0 = time.time()
    while js.health()["status"] == "loading" and time.time() - t0 < 10:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{http.server_address[1]}"
    js.stop()
    http.shutdown()


def test_the_client_runs_repeated_jobs_writes_the_clips_and_the_numbers(server, tmp_path):
    out = tmp_path / "numbers.json"
    rc = client.main(["--url", server, "--prompt", "a tune", "--duration", "2", "--seed", "5", "--repeat", "2", "--out", str(tmp_path / "last.wav"),
                      "--out-dir", str(tmp_path / "clips"), "--json", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert [j["ok"] for j in data["jobs"]] == [True, True] and data["jobs"][0]["audio_s"] == 2.0 and data["jobs"][0]["seed"] == 5
    assert abs(data["jobs"][0]["file_s"] - 2.0) < 0.01 and data["health"]["status"] == "ok"
    assert (tmp_path / "clips" / "clip-1.wav").exists() is False and (tmp_path / "clips" / "clip-1.flac").exists()
    assert (tmp_path / "last.wav").read_bytes()[:4] == b"RIFF"


def test_a_refused_job_makes_the_exit_status_1_and_says_why(server, capsys):
    rc = client.main(["--url", server, "--prompt", "x", "--duration", "9999"])
    assert rc == 1 and "FAILED" in capsys.readouterr().err


def test_clip_seconds_reads_flac_and_wav_headers_and_refuses_anything_else():
    import struct
    packed = (44100 << 44) | (1 << 41) | (15 << 36) | 441000
    flac = b"fLaC" + bytes([0x80, 0, 0, 34]) + b"\0" * 10 + struct.pack(">Q", packed) + b"\0" * 16
    assert client.clip_seconds(flac) == 10.0
    with pytest.raises(ValueError):
        client.clip_seconds(b"nope")
