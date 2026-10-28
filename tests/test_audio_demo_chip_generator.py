# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The chip generator (audio_server.SubprocessChipGenerator) against a stand-in for chip_generate.py. No chip, no ttnn: the stand-in is a small Python
script that takes the same arguments, writes the same files (clips/audio.flac, chip.json) and exits with the code the test asks for."""
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchard" / "audio_demo"))
import audio_server as srv  # noqa: E402


def flac_header(seconds: float, rate: int = 44100) -> bytes:
    """A FLAC file that is only a STREAMINFO block: enough for the length to be read from it."""
    total = int(seconds * rate)
    packed = (rate << 44) | (1 << 41) | (15 << 36) | total       # 2 channels, 16 bits
    info = struct.pack(">HH", 4096, 4096) + b"\x00" * 6 + struct.pack(">Q", packed) + b"\x00" * 16
    return b"fLaC" + bytes([0x80, 0, 0, 34]) + info


STAND_IN = r'''
import argparse, json, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
for a in ("--model", "--out", "--prompt", "--duration", "--steps", "--cfg-scale", "--seed"):
    p.add_argument(a)
p.add_argument("--no-layers", action="store_true")
a = p.parse_args()
mode = os.environ.get("STAND_IN_MODE", "ok")
Path(os.environ["STAND_IN_ARGS"]).write_text(json.dumps({"argv": sys.argv[1:], "pythonpath": os.environ.get("PYTHONPATH", "")}))
if mode == "fail":
    print("Traceback line one"); print("RuntimeError: no device"); sys.exit(3)
if mode == "hang":
    import time
    Path(os.environ["STAND_IN_PID"]).write_text(str(os.getpid()))
    time.sleep(60)
out = Path(a.out); (out / "clips").mkdir(parents=True)
if mode != "noclip":
    (out / "clips" / "audio.flac").write_bytes(Path(os.environ["STAND_IN_FLAC"]).read_bytes())
(out / "chip.json").write_text(json.dumps({"timings": {"model_load_s": 5.4, "generate_and_decode_s": 5.7, "note": "x", "ok": True, "chip_passes": 8, "ok_s": True}}))
'''


@pytest.fixture
def gen(tmp_path, monkeypatch):
    script = tmp_path / "chip_generate.py"
    script.write_text(STAND_IN)
    src = tmp_path / "src"
    (src / "stable_audio_3").mkdir(parents=True)
    flac = tmp_path / "clip.flac"
    flac.write_bytes(flac_header(10.0))
    monkeypatch.setenv("STAND_IN_FLAC", str(flac))
    monkeypatch.setenv("STAND_IN_ARGS", str(tmp_path / "args.json"))
    g = srv.SubprocessChipGenerator(sys.executable, str(script), str(src), work_dir=str(tmp_path / "work"), timeout_s=5)
    g.load()
    return g, tmp_path


REQ = srv.JobRequest("A warm acoustic guitar melody with soft percussion, 100 BPM", 10, 8, 1.0, 42)


def test_flac_length_is_read_from_the_header():
    assert srv.flac_seconds(flac_header(10.0)) == pytest.approx(10.0, abs=1e-3)
    assert srv.flac_seconds(flac_header(3.5, rate=48000)) == pytest.approx(3.5, abs=1e-3)
    for bad in (b"", b"RIFF" + b"\x00" * 60, b"fLaC" + b"\x01" + b"\x00" * 60, flac_header(0)):
        with pytest.raises(ValueError):
            srv.flac_seconds(bad)


def test_the_child_gets_the_request_and_the_library_on_its_path(gen):
    g, tmp = gen
    stages = []
    clip = g.generate(REQ, stages.append)
    seen = json.loads((tmp / "args.json").read_text())
    argv = seen["argv"]
    assert f"--prompt={REQ.prompt}" in argv                    # one token, so a prompt that starts with a dash is not read as an option
    for pair in (["--model", "small-music"], ["--duration", "10.0"], ["--steps", "8"], ["--cfg-scale", "1.0"], ["--seed", "42"]):
        i = argv.index(pair[0])
        assert argv[i + 1] == pair[1]
    assert "--no-layers" in argv
    assert seen["pythonpath"].split(":")[0] == str(tmp / "src")
    assert clip.audio_format == "flac" and clip.audio_s == pytest.approx(10.0, abs=1e-3)
    assert clip.stages == {"model_load_s": 5.4, "generate_and_decode_s": 5.7}            # only times: a count (chip_passes), a bool, a bool named *_s and a string are left out
    assert stages and "chip" in stages[0]


def test_a_failed_child_gives_its_last_lines(gen, monkeypatch):
    g, _ = gen
    monkeypatch.setenv("STAND_IN_MODE", "fail")
    with pytest.raises(RuntimeError, match=r"code 3.*RuntimeError: no device"):
        g.generate(REQ, lambda s: None)


def test_a_child_without_a_clip_is_an_error(gen, monkeypatch):
    g, _ = gen
    monkeypatch.setenv("STAND_IN_MODE", "noclip")
    with pytest.raises(RuntimeError, match="without writing a clip"):
        g.generate(REQ, lambda s: None)


def test_a_hung_child_is_stopped_at_the_timeout(gen, monkeypatch):
    g, tmp = gen
    g.timeout_s = 1.5
    monkeypatch.setenv("STAND_IN_MODE", "hang")
    monkeypatch.setenv("STAND_IN_PID", str(tmp / "child.pid"))
    with pytest.raises(RuntimeError, match="did not finish.*reset"):
        g.generate(REQ, lambda s: None)
    pid = int((tmp / "child.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)                                   # the child is gone, not left holding the chip


def test_two_clips_do_not_share_a_run_directory(gen):
    g, _ = gen
    g.keep_runs = True
    g.generate(REQ, lambda s: None)
    g.generate(REQ, lambda s: None)                      # chip_generate.py refuses an existing --out; a reused directory would fail here
    assert len(list(g.work.glob("run-*/"))) == 2


def test_load_names_what_is_missing(tmp_path):
    g = srv.SubprocessChipGenerator(str(tmp_path / "nopython"), str(tmp_path / "noscript.py"), str(tmp_path / "nosrc"))
    with pytest.raises(RuntimeError) as e:
        g.load()
    for name in ("nopython", "noscript.py", "nosrc"):
        assert name in str(e.value)


def test_the_chip_generator_says_what_it_is_and_the_length_limit_is_configurable(gen):
    g, _ = gen
    assert "Blackhole" in g.device_name
    g2 = srv.SubprocessChipGenerator("p", "s", "d", max_seconds=30)
    js = srv.JobServer(g2)
    with pytest.raises(ValueError, match="between 1 and 30"):
        js.validate({"prompt": "x", "duration": 31})


def left_over(g):
    """What a finished request left in the work directory: run directories and logs."""
    return sorted(p.name for p in g.work.iterdir()) if g.work.exists() else []


@pytest.mark.parametrize("mode", ["ok", "fail", "noclip", "hang"])
def test_a_request_leaves_nothing_on_disk_whatever_happened(gen, monkeypatch, mode):
    """Run directories and logs are removed after success, failure, a missing clip and a timeout, so repeated use does not fill the disk."""
    g, tmp = gen
    g.timeout_s = 1.5
    monkeypatch.setenv("STAND_IN_MODE", mode)
    monkeypatch.setenv("STAND_IN_PID", str(tmp / "child.pid"))
    try:
        g.generate(REQ, lambda s: None)
    except RuntimeError:
        assert mode != "ok"
    assert left_over(g) == []


def test_a_failed_request_still_reports_the_log_tail_after_cleanup(gen, monkeypatch):
    g, _ = gen
    monkeypatch.setenv("STAND_IN_MODE", "fail")
    with pytest.raises(RuntimeError, match="no device"):
        g.generate(REQ, lambda s: None)
    assert left_over(g) == []


def test_keep_runs_keeps_the_run_directory_for_debugging(gen):
    g, _ = gen
    g.keep_runs = True
    g.generate(REQ, lambda s: None)
    assert any(n.startswith("run-") for n in left_over(g))


def test_close_stops_a_running_child_and_the_request_ends(gen, monkeypatch):
    """The server's shutdown must not leave the chip run alive after the process that holds the lease exits."""
    import threading
    import time
    g, tmp = gen
    g.timeout_s = 60
    monkeypatch.setenv("STAND_IN_MODE", "hang")
    monkeypatch.setenv("STAND_IN_PID", str(tmp / "child.pid"))
    seen = {}

    def work():
        try:
            g.generate(REQ, lambda s: None)
        except RuntimeError as e:
            seen["error"] = str(e)
    t = threading.Thread(target=work)
    t.start()
    end = time.monotonic() + 10
    while not (tmp / "child.pid").exists() and time.monotonic() < end:
        time.sleep(0.05)
    pid = int((tmp / "child.pid").read_text())
    g.close()
    t.join(10)
    assert not t.is_alive() and "stopped" in seen["error"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert left_over(g) == []


def test_close_removes_a_work_directory_the_generator_made(tmp_path):
    g = srv.SubprocessChipGenerator(sys.executable, str(tmp_path / "s.py"), str(tmp_path))
    work = g.work
    assert work.is_dir()
    g.close()
    assert not work.exists()
    kept = srv.SubprocessChipGenerator(sys.executable, "s", "d", keep_runs=True)
    (kept.work / "run-abc").mkdir()
    kept.close()
    assert (kept.work / "run-abc").is_dir()                  # --keep-runs: the run directories outlive the server for inspection
    shutil.rmtree(kept.work)
    explicit = tmp_path / "mine"
    explicit.mkdir()
    g2 = srv.SubprocessChipGenerator(sys.executable, "s", "d", work_dir=str(explicit))
    g2.close()
    assert explicit.is_dir()                              # a directory the caller named is the caller's


@pytest.mark.parametrize("prompt", ["--help", "-ambient pads", "--steps 99"])
def test_a_prompt_that_starts_with_a_dash_reaches_the_child_as_its_prompt(gen, prompt):
    g, tmp = gen
    g.generate(srv.JobRequest(prompt, 10, 8, 1.0, 1), lambda s: None)       # the stand-in's argparse fails on a prompt it reads as an option
    seen = json.loads((tmp / "args.json").read_text())["argv"]
    assert f"--prompt={prompt}" in seen and "8" == seen[seen.index("--steps") + 1]


def test_the_model_name_follows_the_model_chosen(tmp_path):
    g = srv.SubprocessChipGenerator("p", "s", "d", model="small-music-base", work_dir=str(tmp_path))
    assert "small-music-base" in g.model_name and "Blackhole" in g.model_name
    assert "small-music-base" not in srv.SubprocessChipGenerator("p", "s", "d", work_dir=str(tmp_path)).model_name


@pytest.mark.parametrize("bad", [0, -5, 121, 240])
def test_a_length_limit_outside_what_the_model_and_client_accept_is_refused_before_anything_is_made(bad):
    before = set(Path(tempfile.gettempdir()).glob("audio-demo-*"))
    with pytest.raises(ValueError, match="max_seconds"):
        srv.SubprocessChipGenerator("p", "s", "d", max_seconds=bad)
    assert set(Path(tempfile.gettempdir()).glob("audio-demo-*")) == before          # no work directory was left behind
