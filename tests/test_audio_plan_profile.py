# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 269, the part that does not need a runner: `audio_plan` chooses the model family from `orchard.audio_profile`, and plan items carry the profile's
key so the stage code reads the sample rate and the clip file from it. Stable Audio plans are unchanged. No chip, no model."""
import pytest

from orchard import audio_plan as ap
from orchard import audio_profile as prof

ACE = "ACE-Step/Ace-Step1.5"
SA = "stabilityai/stable-audio-3-small-music"
SA_BASE = "stabilityai/stable-audio-3-small-music-base"


def test_a_profile_model_has_a_reference_plan_with_the_profile_key():
    plan = ap.reference_plan(ACE + "@abc", None)
    assert [p["role"] for p in plan] == [ap.ROLE_REFERENCE]
    assert plan[0]["profile"] == prof.ACESTEP_TURBO.key and plan[0]["name"] == prof.ACESTEP_TURBO.key
    assert plan[0]["interpreter"] == "reference" and plan[0]["dir"] == f"reference/{prof.ACESTEP_TURBO.key}"


def test_a_profile_model_has_a_chip_plan_with_the_profile_key():
    assert ap.chip_plan(ACE, None) == [{"role": ap.ROLE_CHIP, "name": prof.ACESTEP_TURBO.key, "profile": prof.ACESTEP_TURBO.key}]


def test_stable_audio_plans_carry_no_profile_key_and_are_unchanged():
    assert all("profile" not in p for p in ap.reference_plan(SA, SA_BASE))
    assert [p["role"] for p in ap.reference_plan(SA, SA_BASE)] == [ap.ROLE_REFERENCE, ap.ROLE_HOST, ap.ROLE_CONTROL_REFERENCE]
    assert all("profile" not in p for p in ap.chip_plan(SA, SA_BASE))


def test_a_model_with_neither_a_library_name_nor_a_profile_is_still_refused():
    for fn in (ap.reference_plan, ap.chip_plan):
        with pytest.raises(ap.AudioStageError):
            fn("someone/else-model", None)


def test_the_settings_of_a_profile_model_come_from_its_profile():
    s = ap.settings_for(prof.ACESTEP_TURBO.key)
    assert (s["steps"], s["cfg_scale"], s["sampler"], s["duration"]) == (8, 1.0, "turbo-ode-shift1", 30.0)
    assert s["sample_size"] == int(30.0 * prof.ACESTEP_TURBO.sample_rate)


def test_the_sample_rate_and_clip_file_of_an_item_come_from_its_profile():
    assert ap.sample_rate_of({"name": "small-music"}) == ap.SAMPLE_RATE and ap.clip_name_of({"name": "small-music"}) == "audio.flac"
    item = ap.chip_plan(ACE, None)[0]
    assert (ap.sample_rate_of(item), ap.clip_name_of(item)) == (48000, "audio.wav")


def test_an_unknown_profile_key_in_an_item_is_an_error_not_the_stable_audio_default():
    with pytest.raises(ap.AudioStageError):
        ap.sample_rate_of({"name": "x", "profile": "no-such-profile"})


def test_a_profile_item_has_no_cpu_command_yet_and_says_so():
    item = ap.reference_plan(ACE, None)[0]
    with pytest.raises(ap.AudioStageError, match="acestep_run.py"):
        ap.reference_command(item, None, __import__("pathlib").Path("stages/1"), __import__("pathlib").Path("x.py"))


def test_write_clip_names_the_file_and_the_rate_it_is_given(tmp_path):
    import numpy as np
    sf = pytest.importorskip("soundfile")
    from orchard import audio_reference as ar
    from orchard import audio_stages as st
    rec = ar.Recorder(tmp_path / "r", model={"name": "m"}, library={"name": "l"},
                      settings={"prompt": "p", "duration": 1.0, "steps": 1, "cfg_scale": 1.0, "sampler": "s", "seed": 1, "dtype": "float32", "sample_size": 48000,
                                "device": "cpu"})
    rec.record("waveform", "audio", (0.3 * np.sin(np.arange(48000) / 20.0)).astype(np.float32)[None, :].repeat(2, 0))
    rec.finish()
    clip = st.write_clip(tmp_path / "r", 48000, "audio.wav")
    assert clip.name == "audio.wav" and sf.info(str(clip)).samplerate == 48000
    assert not (tmp_path / "r" / "clips" / "audio.flac").exists()


def test_the_stage_code_takes_the_clip_path_and_rate_from_the_plan_item():
    import inspect
    from orchard import audio_stages as st
    src = inspect.getsource(st.run_references) + inspect.getsource(st.finish_chip)
    assert "clip_name_of(item)" in src and "sample_rate_of(item)" in src and "clips/audio.flac" not in src
