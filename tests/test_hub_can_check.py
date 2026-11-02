# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/hub_can_check.py: the parts that run without a board.

The script serves a published can from the Hub under a lease. These tests cover what a mistake would hide:
the isolation of every cache under one directory, the stop that targets only the process group the script
started, the port table against the cards, and the audio check on a real file."""
from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import wave
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hub_can_check.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("hub_can_check", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_every_cache_variable_points_inside_the_root(mod, tmp_path):
    env = mod.env_for(tmp_path)
    for name in ("XDG_CACHE_HOME", "TT_MODEL_MODELS_DIR", "HF_HOME", "TT_METAL_CACHE"):
        assert Path(env[name]).resolve().is_relative_to(tmp_path.resolve()), name


def test_the_environment_keeps_the_callers_token_without_printing_it(mod, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_for_the_test")
    env = mod.env_for(tmp_path)
    assert env["HF_TOKEN"] == "hf_secret_for_the_test"      # the server needs it
    # the script never writes the environment: its printable result holds no env and no token
    text = SCRIPT.read_text()
    assert "print(env" not in text and "environ[\"HF_TOKEN\"]" not in text


def test_each_can_has_the_port_its_card_gives(mod):
    assert {k: v["port"] for k, v in mod.CANS.items()} == {
        "granite-4.0-h-1b-blackhole": 20110, "minicpm3-4b-blackhole": 20130, "ace-step-1.5-turbo-blackhole": 20140,
        "stable-audio-3-small-music-blackhole": 20170, "minimax-music3-fiona-crapple-blackhole": 20150,
        "qwen3.5-35b-a3b-blackhole-p300": 20160}


def test_stop_group_kills_the_group_the_script_started_and_nothing_else(mod, tmp_path):
    # a shell that starts a child which would outlive it; the group stop must take both down
    marker = tmp_path / "child.pid"
    proc = subprocess.Popen(["bash", "-c", f"sleep 300 & echo $! > {marker}; wait"], start_new_session=True)
    for _ in range(100):
        if marker.exists() and marker.read_text().strip():
            break
        subprocess.run(["sleep", "0.05"])
    child = int(marker.read_text())
    bystander = subprocess.Popen(["sleep", "300"])          # another process of ours, outside the group
    try:
        assert mod.stop_group(proc, wait_s=10) == "stopped on SIGTERM"
        gone = False
        for _ in range(100):                                 # init collects the orphaned child shortly after the group dies
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                gone = True
                break
            subprocess.run(["sleep", "0.05"])
        assert gone, "the child of the started process survived the group stop"
        assert bystander.poll() is None                      # untouched
    finally:
        bystander.send_signal(signal.SIGKILL)
        bystander.wait()


@pytest.mark.skipif(subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0, reason="needs ffmpeg and ffprobe")
def test_audio_facts_reads_duration_rate_and_silence_from_a_real_wav(mod, tmp_path):
    np = pytest.importorskip("numpy")
    rate = 8000
    t = np.arange(rate) / rate
    loud = (0.5 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    silence = np.zeros(rate, dtype="float32")
    x = np.concatenate([loud, silence])
    pcm = (x * 32767).astype("<i2")
    path = tmp_path / "clip.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    facts = mod.audio_facts(path)
    assert facts["sample_rate"] == rate and facts["channels"] == 1
    assert facts["duration_s"] == pytest.approx(2.0, abs=0.01)
    assert facts["windows_above_-40dBFS"] == pytest.approx(0.5, abs=0.06)     # one loud second, one silent second
    assert facts["peak"] == pytest.approx(0.5, abs=0.01)


def test_the_script_refuses_an_unknown_can_name(tmp_path):
    cp = subprocess.run([sys.executable, str(SCRIPT), "not-a-can", "--root", str(tmp_path), "--print-env"],
                        capture_output=True, text=True)
    assert cp.returncode == 2 and "unknown can" in cp.stderr and "Traceback" not in cp.stderr
