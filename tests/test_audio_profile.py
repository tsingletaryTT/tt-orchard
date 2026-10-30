# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/audio_profile.py and the profile argument of audio_stages.check_run_dir: a 48 kHz model with a normalised clip passes under its own profile
and fails under the Stable Audio values. Runs are built with the recorder at a small duration; no chip and no model."""
import numpy as np
import pytest

from orchard import audio_profile as prof
from orchard import audio_reference as ar
from orchard import audio_stages as st

sf = pytest.importorskip("soundfile")
DUR = 4.0


def _tone(rate, seconds, channels=2, amp=0.3):
    t = np.arange(int(rate * seconds)) / rate
    return np.stack([amp * np.sin(2 * np.pi * (220 + 110 * c) * t) for c in range(channels)]).astype(np.float32)


def _make_run(tmp_path, rate=48000, steps=8, skip=None, clip_gain=1.0, name="run"):
    d = tmp_path / name
    s = {"prompt": "p", "duration": DUR, "steps": steps, "cfg_scale": 1.0, "sampler": "turbo-ode-shift1", "seed": 1, "dtype": "float32",
         "sample_size": int(DUR * rate), "device": "cpu"}
    rec = ar.Recorder(d, model={"name": "ACE-Step/Ace-Step1.5", "revision": "r"}, library={"name": "lib", "commit": "c"}, settings=s)
    frames = int(DUR * 25)
    arrays = {("noise", "x0"): np.zeros((64, frames), np.float32), ("conditioning", "encoder_states"): np.zeros((5, 8), np.float32),
              ("conditioning", "context"): np.zeros((frames, 128), np.float32), ("latent", "final"): np.zeros((64, frames), np.float32)}
    for k in range(steps):
        arrays[("step_latent", f"x{k}")] = np.zeros((64, frames), np.float32)
        arrays[("step_denoised", f"v{k}")] = np.zeros((64, frames), np.float32)
    wave = _tone(rate, DUR)
    arrays[("waveform", "audio")] = wave
    for (kind, nm), a in arrays.items():
        if (kind, nm) != skip:
            rec.record(kind, nm, a)
    rec.finish()
    (d / "clips").mkdir()
    sf.write(str(d / "clips" / "audio.wav"), (wave * clip_gain).T, rate)
    return d


def test_acestep_run_passes_its_profile(tmp_path):
    d = _make_run(tmp_path)
    assert prof.run_problems(d, prof.ACESTEP_TURBO) == []


def test_the_same_run_fails_at_the_stable_audio_rate(tmp_path):
    d = _make_run(tmp_path)
    problems = prof.run_problems(d, prof.STABLE_AUDIO)
    assert any("44100" in x for x in problems), problems


def test_missing_per_step_array_is_named(tmp_path):
    d = _make_run(tmp_path, skip=("step_denoised", "v3"))
    assert "missing array step_denoised/v3" in prof.layout_problems(ar.load_run(d).manifest, prof.ACESTEP_TURBO)


def test_wrong_step_count_is_refused(tmp_path):
    d = _make_run(tmp_path, steps=4)
    assert any("4 steps" in x for x in prof.layout_problems(ar.load_run(d).manifest, prof.ACESTEP_TURBO))


def test_waveform_recorded_at_the_wrong_rate_is_refused(tmp_path):
    d = _make_run(tmp_path, rate=44100)
    assert any("48000" in x for x in prof.layout_problems(ar.load_run(d).manifest, prof.ACESTEP_TURBO))


def test_for_model_ignores_revision_and_unknown_is_none():
    assert prof.for_model("ACE-Step/Ace-Step1.5@abc") is prof.ACESTEP_TURBO
    assert prof.for_model("stabilityai/stable-audio-3-small-music") is prof.STABLE_AUDIO
    assert prof.for_model("someone/else") is None


def _check(d, profile):
    problems = []
    entry = {"dir": d.name, **st.run_facts(d.parent, d.name), "clip": f"{d.name}/clips/{profile.clip_name}"}
    st.check_run_dir(d.parent, entry, None, "chip", problems, expected={"revision": "r", "config_sha256": None}, library_commit="c", profile=profile)
    return [x for x in problems if "config" not in x and "snapshot" not in x]


def test_check_run_dir_accepts_a_normalised_wav_clip_under_the_profile(tmp_path):
    d = _make_run(tmp_path, clip_gain=3.0)            # the clip is louder than the recorded waveform, as after peak normalisation
    assert _check(d, prof.ACESTEP_TURBO) == []


def test_check_run_dir_without_the_profile_refuses_the_same_run(tmp_path):
    d = _make_run(tmp_path, clip_gain=3.0)
    problems = []
    entry = {"dir": d.name, **st.run_facts(d.parent, d.name), "clip": f"{d.name}/clips/audio.wav"}
    st.check_run_dir(d.parent, entry, None, "chip", problems)
    assert any("clip path" in x for x in problems) and any("sound checks" in x for x in problems)


def test_a_stable_audio_clip_must_still_equal_the_waveform(tmp_path):
    d = _make_run(tmp_path, rate=44100, clip_gain=3.0)
    flac = d / "clips" / "audio.flac"
    sf.write(str(flac), (_tone(44100, DUR) * 3.0).T, 44100)
    wave = _tone(44100, DUR)
    assert st.clip_matches_waveform(d, wave, 44100) != []
    assert st.clip_matches_waveform(d, wave, 44100, exact=False) == []
