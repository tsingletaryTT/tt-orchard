# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/audio_stages.py: the sound checks, the comparisons, stage 1's CPU runs and stage 2's result with their gates. Every run here is a small
synthetic recorded run (orchard/audio_reference.py); no model, chip or interpreter other than this one is used."""
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orchard import audio_reference as ar
from orchard import audio_stages as st

SR = 44100
COMMIT = "c" * 40
REV = "rev-small"
REPEATS = 2
EXP = {"revision": REV, "config_sha256": "0" * 64}          # what a run's snapshot gives (make_run records the same config hash)


def music(seconds=10.0, seed=0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    g = np.random.default_rng(seed)
    base = sum(np.sin(2 * np.pi * f * t) for f in (220, 330, 440)) / 3 * amp
    return np.stack([base + 0.01 * g.standard_normal(t.size), base * 0.9 + 0.01 * g.standard_normal(t.size)]).astype(np.float32)


def make_run(path, name="small-music", *, dtype="float32", device="cpu", wave=None, latent=None, layer=None, revision=REV, commit=COMMIT,
             settings_override=None, seed=1, config_sha256="0" * 64):
    s = {**st.settings_for(name), "dtype": dtype, "device": device, **(settings_override or {})}
    rec = ar.Recorder(path, model={"name": name, "revision": revision, "config_sha256": config_sha256, "diffusion_objective": "rf_denoiser",
                                   "checkpoint_bytes": 1},
                      library={"commit": commit, "name": "stable-audio-3", "source_verified": True}, settings=s, seeds={"torch": 42})
    g = np.random.default_rng(seed)
    rec.record("latent", "final", latent if latent is not None else g.standard_normal((1, 4, 8)))
    rec.record("layer", "layer_00", layer if layer is not None else g.standard_normal((1, 6, 8)))
    rec.record("noise", "initial", np.arange(8.0))
    rec.record("waveform", "audio", wave if wave is not None else music(seed=seed))
    rec.timings["generate_and_decode_s"] = 1.5
    rec.finish()
    return Path(path)


def twin(src, dst, *, noise=0.001, scale=1.0, dtype="bfloat16", device="tt", name=None, **kw):
    """A candidate run near `src`: every array plus a little noise (and optionally scaled)."""
    with ar.load_run(src) as r:
        arrs = {k: np.asarray(r.array(*k)) for k in r.entries}
        m = r.manifest
    g = np.random.default_rng(7)
    out = {k: (v if k[0] == "noise" else (v * scale + noise * g.standard_normal(v.shape))) for k, v in arrs.items()}
    return make_run(dst, name or m["model"]["name"], dtype=dtype, device=device, wave=out[("waveform", "audio")], latent=out[("latent", "final")],
                    layer=out[("layer", "layer_00")], **{"revision": m["model"]["revision"], "config_sha256": m["model"]["config_sha256"], **kw})


# ---- sound checks -------------------------------------------------------------------------------------------------------------

def test_sound_checks_pass_on_music_of_the_requested_length():
    r = st.sound_checks(music(), SR, 10.0)
    assert r["ok"] and r["duration_s"] == 10.0 and r["channels"] == 2 and r["loud_window_fraction"] > 0.99


@pytest.mark.parametrize("wave, seconds, why", [
    (music(seconds=9.0), 10.0, "duration"),
    (music(seconds=10.6), 10.0, "duration"),
    (np.zeros((2, 441000), dtype=np.float32), 10.0, "windows"),
    (np.clip(music(amp=5.0), -1.0, 1.0), 10.0, "full scale"),
])
def test_each_sound_check_can_fail(wave, seconds, why):
    r = st.sound_checks(wave, SR, seconds)
    assert not r["ok"] and any(why in x for x in r["reasons"])


def test_sound_check_edges_are_inclusive_where_the_bars_say():
    # exactly 5 percent off in duration passes; just over fails
    assert st.sound_checks(music(seconds=9.5), SR, 10.0)["ok"]
    assert not st.sound_checks(music(seconds=9.49), SR, 10.0)["ok"]


def test_sound_checks_refuse_a_waveform_with_a_value_that_is_not_finite_or_the_wrong_shape():
    w = music()
    w[0, 5] = np.nan
    assert not st.sound_checks(w, SR, 10.0)["ok"]
    assert not st.sound_checks(music()[0], SR, 10.0)["ok"]
    assert not st.sound_checks(np.zeros((2, 0)), SR, 10.0)["ok"]


def test_the_loud_share_counts_windows_not_samples():
    w = music()
    w[:, : int(SR * 7)] = 0                      # 70 percent silent: 30 percent of windows loud, below the 40 percent bar
    r = st.sound_checks(w, SR, 10.0)
    assert not r["ok"] and 0.29 < r["loud_window_fraction"] < 0.31


# ---- clips --------------------------------------------------------------------------------------------------------------------

def test_a_written_clip_matches_the_recorded_waveform(tmp_path):
    run = make_run(tmp_path / "r")
    clip = st.write_clip(run)
    assert clip.name == "audio.flac" and clip.is_file()
    with ar.load_run(run) as r:
        wave = np.asarray(r.array("waveform", "audio"))
    assert st.clip_matches_waveform(run, wave) == []


def test_a_clip_that_is_another_signal_is_refused(tmp_path):
    run = make_run(tmp_path / "r")
    st.write_clip(run)
    other = music(seed=99)
    assert st.clip_matches_waveform(run, other)


def test_a_missing_or_wrong_rate_clip_is_refused(tmp_path):
    run = make_run(tmp_path / "r")
    with ar.load_run(run) as r:
        wave = np.asarray(r.array("waveform", "audio"))
    assert "cannot be read" in st.clip_matches_waveform(run, wave)[0]
    import soundfile
    (run / "clips").mkdir()
    soundfile.write(str(run / "clips" / "audio.flac"), wave.T, 22050)
    assert "Hz" in st.clip_matches_waveform(run, wave)[0]


# ---- comparing ----------------------------------------------------------------------------------------------------------------

def test_a_near_copy_agrees_and_the_numbers_are_summarised(tmp_path):
    ref = make_run(tmp_path / "ref")
    cand = twin(ref, tmp_path / "cand")
    s = st.compare(ref, cand)
    assert s["ok"] and s["failed"] == [] and s["latent_final"]["pcc"] > 0.99 and s["waveform"]["log_spectrum_distance"] is not None
    assert s["worst_layer"]["name"] == "layer_00" and "provisional" in s["bars"]


def test_a_scaled_output_does_not_agree_although_its_correlation_is_perfect(tmp_path):
    ref = make_run(tmp_path / "ref")
    cand = twin(ref, tmp_path / "cand", noise=0.0, scale=1.6)
    s = st.compare(ref, cand)
    assert not s["ok"] and s["latent_final"]["pcc"] > 0.999


def test_runs_with_different_settings_are_not_comparable(tmp_path):
    ref = make_run(tmp_path / "ref")
    cand = make_run(tmp_path / "cand", dtype="bfloat16", device="tt", settings_override={"seed": 43})
    s = st.compare(ref, cand)
    assert not s["ok"] and not s["comparable"] and s["problems"]


def test_the_control_must_differ_from_the_fine_tune(tmp_path):
    fine = make_run(tmp_path / "fine", seed=1)
    other = make_run(tmp_path / "base", "small-music-base", seed=2, settings_override={"steps": 8, "cfg_scale": 1.0, "sampler": "pingpong"})
    same = twin(fine, tmp_path / "same", noise=0.0, name="small-music-base", settings_override=None)
    assert st.control_difference(fine, other)["ok"]
    assert not st.control_difference(fine, same)["ok"]


def test_agree_compares_numbers_loosely_and_everything_else_exactly():
    assert st.agree({"a": 1.0, "b": [1, 2], "c": "x"}, {"a": 1.0 + 1e-12, "b": [1, 2], "c": "x"})
    assert not st.agree({"a": 1.0}, {"a": 1.1})
    assert not st.agree({"a": 1.0}, {"a": 1.0, "b": 2})
    assert not st.agree({"a": True}, {"a": 1})
    assert not st.agree([1, 2], [1, 2, 3])
    assert not st.agree({"c": "x"}, {"c": "y"})


def test_inside_refuses_paths_that_leave_the_run_directory(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "link").symlink_to(tmp_path / "a")
    assert st.inside(tmp_path / "out", "x/y") == tmp_path / "out" / "x" / "y"
    for bad in ("../a", "/etc", "", None, "link", "link/x", 3):
        assert st.inside(tmp_path / "out", bad) is None


# ---- names and plans ----------------------------------------------------------------------------------------------------------

def test_library_names_and_settings():
    assert st.library_name("stabilityai/stable-audio-3-small-music@abc") == "small-music"
    assert st.library_name("stabilityai/stable-audio-3-small-music-base") == "small-music-base"
    assert st.library_name("someone/else") is None and st.library_name(None) is None
    assert st.settings_for("small-music")["steps"] == 8 and st.settings_for("small-music-base")["steps"] == 50
    assert st.settings_for("small-music-base")["cfg_scale"] == 7.0 and st.settings_for("small-music-base")["sampler"] == "euler"


def test_the_reference_plan_has_a_control_only_for_a_known_distinct_base():
    plan = st.reference_plan("stabilityai/stable-audio-3-small-music@x", "stabilityai/stable-audio-3-small-music-base@y")
    assert [(p["role"], p["name"], p["interpreter"]) for p in plan] == [
        ("reference", "small-music", "reference"), ("host_same_env", "small-music", "chip"), ("control_reference", "small-music-base", "reference")]
    assert len(st.reference_plan("stabilityai/stable-audio-3-small-music", "other/base")) == 2
    assert len(st.reference_plan("stabilityai/stable-audio-3-small-music", "stabilityai/stable-audio-3-small-music")) == 2
    with pytest.raises(st.AudioStageError):
        st.reference_plan("someone/else", None)


def test_the_chip_plan_follows_the_same_rule():
    assert [p["role"] for p in st.chip_plan("stabilityai/stable-audio-3-small-music", "stabilityai/stable-audio-3-small-music-base")] == ["chip", "control_chip"]
    assert [p["role"] for p in st.chip_plan("stabilityai/stable-audio-3-small-music", None)] == ["chip"]


# ---- the cache ----------------------------------------------------------------------------------------------------------------

def test_a_cached_reference_is_used_only_when_everything_it_depends_on_matches(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    make_run(cache / "small-music")
    assert st.cached_run(cache, "small-music", EXP, COMMIT) == cache / "small-music"
    assert st.cached_run(cache, "small-music", {**EXP, "revision": "other-rev"}, COMMIT) is None
    assert st.cached_run(cache, "small-music", REV, "d" * 40) is None
    assert st.cached_run(cache, "small-music", None, COMMIT) is None
    assert st.cached_run(cache, "small-music-base", EXP, COMMIT) is None
    assert st.cached_run(None, "small-music", EXP, COMMIT) is None


@pytest.mark.parametrize("kw", [{"dtype": "bfloat16"}, {"device": "tt"}, {"settings_override": {"steps": 9}}, {"settings_override": {"seed": 1}}])
def test_a_cached_run_made_with_other_settings_is_not_used(tmp_path, kw):
    cache = tmp_path / "cache"
    cache.mkdir()
    make_run(cache / "small-music", **kw)
    assert st.cached_run(cache, "small-music", EXP, COMMIT) is None


def test_a_cached_run_with_a_changed_array_is_not_used(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    run = make_run(cache / "small-music")
    f = run / "arrays" / "latent__final.npy"
    f.write_bytes(f.read_bytes()[:-4] + b"\x00\x00\x00\x00")
    assert st.cached_run(cache, "small-music", EXP, COMMIT) is None


def test_a_cache_entry_that_is_a_link_is_not_used(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    real = make_run(tmp_path / "real")
    (cache / "small-music").symlink_to(real)
    assert st.cached_run(cache, "small-music", EXP, COMMIT) is None


# ---- stage 1 ------------------------------------------------------------------------------------------------------------------

PATHS = SimpleNamespace(reference_python="/ref/python", chip_python="/chip/python", src_dir="/src/lib")
MODEL, BASE = "stabilityai/stable-audio-3-small-music@x", "stabilityai/stable-audio-3-small-music-base@y"


class FakeRunner:
    """Stands in for the CPU scripts: makes the run the command asks for."""

    def __init__(self, fail=None, timeout=None):
        self.calls, self.fail, self.timeout = [], fail, timeout

    def __call__(self, argv, env, timeout_s, log_path):
        self.calls.append((argv, env, timeout_s))
        model = argv[argv.index("--model") + 1]
        out = Path(argv[argv.index("--out") + 1])
        if self.timeout == model:
            return None
        if self.fail == model:
            return 3
        seed = {"small-music": 1, "small-music-base": 2}[model]
        host = argv[0] == PATHS.chip_python
        make_run(out, model, wave=music(seed=seed) + (0.002 * np.ones((2, 1), dtype=np.float32) if host else 0), seed=seed)
        return 0


def stage1(tmp_path, runner=None, cache=None, expected=None, plan=None):
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stages" / "1"
    stage_dir.mkdir(parents=True)
    runner = runner or FakeRunner()
    plan = plan or st.reference_plan(MODEL, BASE)
    rec = st.run_references(stage_dir, run_dir, PATHS, plan, runner, expected=expected or {"small-music": EXP, "small-music-base": EXP},
                            library_commit=COMMIT, cache_dir=cache, script=Path("/skills/make_reference.py"))
    return run_dir, stage_dir, rec, runner


def test_stage_1_makes_every_run_and_writes_its_record(tmp_path):
    run_dir, stage_dir, rec, runner = stage1(tmp_path)
    assert set(rec["runs"]) == {"reference", "host_same_env", "control_reference"}
    assert [c[0][0] for c in runner.calls] == ["/ref/python", "/chip/python", "/ref/python"]
    assert all(r["from_cache"] is False for r in rec["runs"].values())
    assert rec["environment_floor"]["comparable"] is True
    assert (stage_dir / "audio_reference.json").is_file()
    for r in rec["runs"].values():
        assert (run_dir / r["clip"]).is_file() and r["manifest_sha256"] and r["revision"] == REV
    assert st.reference_problems(stage_dir, run_dir, expect_roles=("reference", "host_same_env")) == []


def test_stage_1_passes_the_expected_source_and_the_interpreter_specific_environment(tmp_path):
    _, _, _, runner = stage1(tmp_path)
    ref, host = runner.calls[0], runner.calls[1]
    assert ref[0][ref[0].index("--expect-src") + 1] == "/src/lib" and ref[1] == {}
    assert host[1] == {"PYTHONPATH": "/src/lib"}
    assert ref[0][ref[0].index("--steps") + 1] == "8" and runner.calls[2][0][runner.calls[2][0].index("--steps") + 1] == "50"
    assert runner.calls[2][0][runner.calls[2][0].index("--cfg-scale") + 1] == "7.0"


@pytest.mark.parametrize("fail, word", [("small-music", "exited 3"), ("small-music-base", "exited 3")])
def test_a_failed_cpu_run_stops_the_stage_with_its_log_named(tmp_path, fail, word):
    with pytest.raises(st.AudioStageError) as e:
        stage1(tmp_path, runner=FakeRunner(fail=fail))
    assert word in str(e.value) and "log" in str(e.value)


def test_a_timed_out_cpu_run_is_reported_as_a_timeout(tmp_path):
    with pytest.raises(st.AudioStageError) as e:
        stage1(tmp_path, runner=FakeRunner(timeout="small-music"))
    assert "timed out" in str(e.value)


def test_a_cached_reference_is_copied_and_not_made_again(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    make_run(cache / "small-music", seed=1)
    make_run(cache / "small-music-base", "small-music-base", seed=2)
    _, _, rec, runner = stage1(tmp_path, cache=cache)
    assert rec["runs"]["reference"]["from_cache"] is True and rec["runs"]["control_reference"]["from_cache"] is True
    assert rec["runs"]["host_same_env"]["from_cache"] is False
    assert [c[0][0] for c in runner.calls] == ["/chip/python"]


def test_the_same_environment_run_is_never_taken_from_the_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    make_run(cache / "small-music", seed=1)
    _, _, rec, runner = stage1(tmp_path, cache=cache)
    assert rec["runs"]["host_same_env"]["from_cache"] is False and any(c[0][0] == "/chip/python" for c in runner.calls)


def test_the_stage_1_gate_refuses_a_changed_array_a_wrong_setting_and_a_missing_role(tmp_path):
    run_dir, stage_dir, rec, _ = stage1(tmp_path)
    base = st.reference_problems(stage_dir, run_dir)
    assert base == []
    f = run_dir / rec["runs"]["reference"]["dir"] / "arrays" / "latent__final.npy"
    good = f.read_bytes()
    f.write_bytes(good[:-4] + b"\x01\x02\x03\x04")
    assert any("reference" in p for p in st.reference_problems(stage_dir, run_dir))
    f.write_bytes(good)
    assert st.reference_problems(stage_dir, run_dir) == []
    data = json.loads((stage_dir / "audio_reference.json").read_text())
    del data["runs"]["host_same_env"]
    (stage_dir / "audio_reference.json").write_text(json.dumps(data))
    assert any("host_same_env" in p for p in st.reference_problems(stage_dir, run_dir))


def test_the_stage_1_gate_refuses_a_recorded_floor_that_is_not_what_the_runs_give(tmp_path):
    run_dir, stage_dir, _, _ = stage1(tmp_path)
    data = json.loads((stage_dir / "audio_reference.json").read_text())
    data["environment_floor"]["latent_final"]["rel_l2"] = 0.5
    (stage_dir / "audio_reference.json").write_text(json.dumps(data))
    assert any("environment floor" in p for p in st.reference_problems(stage_dir, run_dir))


def test_the_stage_1_gate_refuses_a_silent_reference_clip(tmp_path):
    run_dir, stage_dir, rec, _ = stage1(tmp_path, runner=None)
    # replace the reference run with one whose waveform is silent, and keep its manifest hash in the record
    d = run_dir / rec["runs"]["reference"]["dir"]
    shutil.rmtree(d)
    make_run(d, wave=np.zeros((2, 441000), dtype=np.float32))
    data = json.loads((stage_dir / "audio_reference.json").read_text())
    data["runs"]["reference"]["manifest_sha256"] = st.sha256_file(d / "manifest.json")
    (stage_dir / "audio_reference.json").write_text(json.dumps(data))
    assert any("sound checks" in p for p in st.reference_problems(stage_dir, run_dir))


def test_a_run_dir_outside_the_run_is_refused_by_the_stage_1_gate(tmp_path):
    run_dir, stage_dir, _, _ = stage1(tmp_path)
    data = json.loads((stage_dir / "audio_reference.json").read_text())
    data["runs"]["reference"]["dir"] = "../../elsewhere"
    (stage_dir / "audio_reference.json").write_text(json.dumps(data))
    assert any("inside the run directory" in p for p in st.reference_problems(stage_dir, run_dir))


# ---- stage 2 ------------------------------------------------------------------------------------------------------------------

def stage2(tmp_path, *, chip_noise=0.001, control="different"):
    run_dir, ref_dir, rec, _ = stage1(tmp_path)
    s2 = run_dir / "stages" / "2"
    (s2 / "chip").mkdir(parents=True)
    plan = st.chip_plan(MODEL, BASE)
    st.write_chip_config(s2, PATHS, "/hf", plan, repeats=REPEATS)
    for item in plan:
        ref_role = {"chip": "reference", "control_chip": "control_reference"}[item["role"]]
        src = run_dir / rec["runs"][ref_role]["dir"]
        for k in range(REPEATS):
            dst = s2 / "chip" / st.repeat_name(item["name"], k)
            twin(src, dst, noise=chip_noise)
            (dst / "chip.json").write_text(json.dumps({"device": "test chip", "total_s": 12.5, "timings": {}}))
    roles = {"chip": "reference", "control_chip": "control_reference"}
    result = st.finish_chip(s2, run_dir, ref_dir, plan, roles)
    return run_dir, ref_dir, s2, result


def test_stage_2_writes_the_comparisons_and_the_gate_passes_them(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    assert set(res["runs"]) == {"chip", "control_chip"}
    assert set(res["agreement"]) == {"chip_vs_reference", "control_chip_vs_control_reference", "chip_vs_host_same_env"}
    assert res["control_difference"]["ok"] and res["runs"]["chip"]["device"] == "test chip"
    assert all(Path(run_dir / r["clip"]).is_file() for r in res["runs"].values())
    assert st.chip_problems(s2, run_dir, ref_dir) == []
    assert any("listened" in x for x in res["limits"]) and any("provisional" in x for x in res["limits"])


def test_stage_2_names_a_run_that_left_no_manifest(tmp_path):
    run_dir, ref_dir, rec, _ = stage1(tmp_path)
    s2 = run_dir / "stages" / "2"
    (s2 / "chip").mkdir(parents=True)
    with pytest.raises(st.AudioStageError) as e:
        st.finish_chip(s2, run_dir, ref_dir, st.chip_plan(MODEL, BASE), {"chip": "reference", "control_chip": "control_reference"})
    assert "no manifest" in str(e.value)


def test_the_gate_refuses_a_chip_run_that_does_not_agree(tmp_path):
    run_dir, ref_dir, s2, _ = stage2(tmp_path, chip_noise=0.8)
    problems = st.chip_problems(s2, run_dir, ref_dir)
    assert any("does not agree with its reference" in p for p in problems)


def test_the_gate_refuses_a_result_whose_numbers_were_edited(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path, chip_noise=0.8)
    res["agreement"]["chip_vs_reference"]["ok"] = True
    res["agreement"]["chip_vs_reference"]["failed"] = []
    (s2 / "audio_result.json").write_text(json.dumps(res))
    problems = st.chip_problems(s2, run_dir, ref_dir)
    assert any("not what the runs give now" in p for p in problems) and any("does not agree" in p for p in problems)


def test_the_gate_refuses_a_control_that_matches_the_fine_tune(tmp_path):
    run_dir, ref_dir, rec, _ = stage1(tmp_path)
    s2 = run_dir / "stages" / "2"
    (s2 / "chip").mkdir(parents=True)
    plan = st.chip_plan(MODEL, BASE)
    fine = twin(run_dir / rec["runs"]["reference"]["dir"], s2 / "chip" / "small-music")
    # the "base" chip run is the fine-tune's output again, which is what serving the wrong weights would give
    with ar.load_run(fine) as r:
        wave = np.asarray(r.array("waveform", "audio"))
        lat = np.asarray(r.array("latent", "final"))
        lay = np.asarray(r.array("layer", "layer_00"))
    make_run(s2 / "chip" / "small-music-base", "small-music-base", dtype="bfloat16", device="tt", wave=wave, latent=lat, layer=lay)
    res = st.finish_chip(s2, run_dir, ref_dir, plan, {"chip": "reference", "control_chip": "control_reference"})
    assert res["control_difference"]["ok"] is False
    problems = st.chip_problems(s2, run_dir, ref_dir)
    assert any("control:" in p and "does not differ" in p for p in problems)


def test_the_gate_refuses_a_chip_clip_that_fails_the_sound_checks(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    d = run_dir / res["runs"]["chip"]["dir"]
    shutil.rmtree(d)
    make_run(d, dtype="bfloat16", device="tt", wave=np.zeros((2, 441000), dtype=np.float32), revision=REV)
    res["runs"]["chip"]["manifest_sha256"] = st.sha256_file(d / "manifest.json")
    (s2 / "audio_result.json").write_text(json.dumps(res))
    assert any("sound checks" in p for p in st.chip_problems(s2, run_dir, ref_dir))


def test_the_gate_refuses_a_changed_chip_array_and_a_swapped_manifest(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    f = run_dir / res["runs"]["chip"]["dir"] / "arrays" / "latent__final.npy"
    f.write_bytes(f.read_bytes()[:-4] + b"\x01\x02\x03\x04")
    assert any(p.startswith("chip:") for p in st.chip_problems(s2, run_dir, ref_dir))


def test_the_gate_refuses_reference_runs_that_are_not_stage_1s(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    res["reference_runs"]["reference"]["wall_seconds"] = 1.0
    (s2 / "audio_result.json").write_text(json.dumps(res))
    assert any("reference_runs" in p for p in st.chip_problems(s2, run_dir, ref_dir))


def test_the_gate_refuses_a_missing_or_empty_result(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    (s2 / "audio_result.json").write_text("{}")
    assert st.chip_problems(s2, run_dir, ref_dir)
    (s2 / "audio_result.json").unlink()
    assert "cannot be read" in st.chip_problems(s2, run_dir, ref_dir)[0]


def test_the_gate_refuses_a_model_the_runtime_does_not_know(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    res["runs"]["chip"]["model"] = "some-other-model"
    (s2 / "audio_result.json").write_text(json.dumps(res))
    assert any("not one the audio runtime knows" in p for p in st.chip_problems(s2, run_dir, ref_dir))


def test_the_chip_config_and_command(tmp_path):
    path = st.write_chip_config(tmp_path, PATHS, "/hf", st.chip_plan(MODEL, BASE))
    cfg = json.loads(path.read_text())
    assert cfg["src_dir"] == "/src/lib" and cfg["hf_home"] == "/hf" and cfg["seed"] == 42 and cfg["duration"] == 10.0
    assert [(r["name"], r["model"], r["steps"], r["cfg_scale"], r["sampler"]) for r in cfg["runs"]] == [
        ("small-music", "small-music", 8, 1.0, "pingpong"), ("small-music-r1", "small-music", 8, 1.0, "pingpong"),
        ("small-music-r2", "small-music", 8, 1.0, "pingpong"), ("small-music-base", "small-music-base", 50, 7.0, "euler"),
        ("small-music-base-r1", "small-music-base", 50, 7.0, "euler"), ("small-music-base-r2", "small-music-base", 50, 7.0, "euler")]
    assert cfg["repeats"] == 3
    cmd = st.chip_test_command(PATHS, tmp_path, Path("/skills/audio_chip_test.py"))
    assert cmd == f"/chip/python /skills/audio_chip_test.py --config {tmp_path}/audio_chip_config.json"


def test_the_chip_command_quotes_paths_with_spaces_and_shell_characters(tmp_path):
    import shlex
    paths = SimpleNamespace(reference_python="/r", chip_python="/my venv/bin/python", src_dir="/s")
    cmd = st.chip_test_command(paths, tmp_path / "a b; echo $HOME", Path("/skills dir/audio_chip_test.py"))
    assert shlex.split(cmd) == ["/my venv/bin/python", "/skills dir/audio_chip_test.py", "--config", f"{tmp_path}/a b; echo $HOME/audio_chip_config.json"]


# ---- cases added after the first mutation run ------------------------------------------------------------------------------------

def test_a_quiet_clip_fails_the_level_check():
    quiet = (0.001 * np.sin(2 * np.pi * 440 * np.arange(441000) / SR) * np.ones((2, 1))).astype(np.float32)      # about -63 dBFS
    r = st.sound_checks(quiet, SR, 10.0)
    assert not r["ok"] and any("windows" in x for x in r["reasons"])


def test_a_clip_of_another_length_is_refused(tmp_path):
    import soundfile
    run = make_run(tmp_path / "r")
    with ar.load_run(run) as r:
        wave = np.asarray(r.array("waveform", "audio"))
    (run / "clips").mkdir()
    soundfile.write(str(run / "clips" / "audio.flac"), wave[:, :SR].T, SR)
    assert "samples" in st.clip_matches_waveform(run, wave)[0]


def test_an_existing_clip_is_kept(tmp_path):
    run = make_run(tmp_path / "r")
    (run / "clips").mkdir()
    (run / "clips" / "audio.flac").write_bytes(b"placeholder")
    st.write_clip(run)
    assert (run / "clips" / "audio.flac").read_bytes() == b"placeholder"


def test_the_summary_lists_the_failing_arrays(tmp_path):
    ref = make_run(tmp_path / "ref")
    s = st.compare(ref, twin(ref, tmp_path / "cand", noise=0.0, scale=1.6))
    assert "latent/final" in s["failed"]


def test_a_cache_entry_for_another_model_is_not_used(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    make_run(cache / "small-music", "small-music-base", settings_override=st.settings_for("small-music"))
    assert st.cached_run(cache, "small-music", EXP, COMMIT) is None


def test_the_chip_plan_has_no_control_when_the_base_is_the_model():
    assert [p["role"] for p in st.chip_plan(MODEL, MODEL)] == ["chip"]


def test_write_json_leaves_no_temporary_file(tmp_path):
    st.write_json(tmp_path / "a.json", {"x": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.json"]


def test_the_stage_1_gate_names_a_model_it_does_not_know_a_changed_manifest_hash_and_wrong_settings(tmp_path):
    run_dir, stage_dir, rec, _ = stage1(tmp_path)
    path = stage_dir / "audio_reference.json"
    good = json.loads(path.read_text())
    bad = json.loads(path.read_text())
    bad["runs"]["reference"]["model"] = "unknown-model"
    path.write_text(json.dumps(bad))
    assert any("not one the audio runtime knows" in p for p in st.reference_problems(stage_dir, run_dir))
    bad = json.loads(json.dumps(good))
    bad["runs"]["reference"]["manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(bad))
    assert any("manifest is not the one" in p for p in st.reference_problems(stage_dir, run_dir))
    path.write_text(json.dumps(good))
    d = run_dir / good["runs"]["control_reference"]["dir"]
    shutil.rmtree(d)
    make_run(d, "small-music-base", settings_override={"steps": 9})
    good["runs"]["control_reference"]["manifest_sha256"] = st.sha256_file(d / "manifest.json")
    path.write_text(json.dumps(good))
    assert any("its settings are" in p for p in st.reference_problems(stage_dir, run_dir))


def test_the_stage_1_gate_refuses_a_clip_that_is_not_the_waveform(tmp_path):
    import soundfile
    run_dir, stage_dir, rec, _ = stage1(tmp_path)
    clip = run_dir / rec["runs"]["reference"]["clip"]
    soundfile.write(str(clip), music(seed=50).T, SR)
    assert any("clip" in p for p in st.reference_problems(stage_dir, run_dir))


def test_the_gate_refuses_an_edited_control_difference_a_missing_chip_run_and_a_missing_control_reference(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    path = s2 / "audio_result.json"
    edited = json.loads(json.dumps(res))
    edited["control_difference"]["rel_l2"] = 0.0
    path.write_text(json.dumps(edited))
    assert any("recorded difference" in p for p in st.chip_problems(s2, run_dir, ref_dir))
    nochip = json.loads(json.dumps(res))
    del nochip["runs"]["chip"]
    path.write_text(json.dumps(nochip))
    assert st.chip_problems(s2, run_dir, ref_dir) == ["audio_result.json has no chip run"]
    path.write_text(json.dumps(res))
    ref = json.loads((ref_dir / "audio_reference.json").read_text())
    del ref["runs"]["control_reference"]
    (ref_dir / "audio_reference.json").write_text(json.dumps(ref))
    assert any("stage 1 made no control_reference" in p for p in st.chip_problems(s2, run_dir, ref_dir))


# ---- the base control is judged by discrimination, and its agreement is informational ------------------------------------------------

def test_the_control_agreement_is_recorded_as_informational_and_does_not_decide(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    key = "control_chip_vs_control_reference"
    assert res["agreement"][key]["bars"].startswith("informational")
    assert res["control_discrimination"]["ok"] is True and res["control_discrimination"]["own_rel_l2"] < res["control_discrimination"]["other_rel_l2"]


def test_a_control_run_that_misses_the_provisional_bars_but_is_nearer_its_own_reference_still_passes(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    d = s2 / "chip" / "small-music-base"
    ref = run_dir / res["reference_runs"]["control_reference"]["dir"]
    shutil.rmtree(d)
    with ar.load_run(ref) as r:                    # the latent well outside the provisional bars, still nearer its own reference than the fine-tune's
        lat, lay, wave = (np.asarray(r.array(*k)) for k in (("latent", "final"), ("layer", "layer_00"), ("waveform", "audio")))
    make_run(d, "small-music-base", dtype="bfloat16", device="tt", wave=wave, latent=lat + 0.45 * np.random.default_rng(3).standard_normal(lat.shape),
             layer=lay, revision=REV)
    (d / "chip.json").write_text(json.dumps({"device": "test chip", "total_s": 12.5}))
    res = st.finish_chip(s2, run_dir, ref_dir, st.chip_plan(MODEL, BASE), {"chip": "reference", "control_chip": "control_reference"})
    assert res["agreement"]["control_chip_vs_control_reference"]["ok"] is False
    assert st.chip_problems(s2, run_dir, ref_dir) == []


def test_a_control_nearer_the_fine_tunes_reference_than_its_own_is_refused(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    d = s2 / "chip" / "small-music-base"
    fine_ref = run_dir / res["reference_runs"]["reference"]["dir"]
    shutil.rmtree(d)
    twin(fine_ref, d, noise=0.01, name="small-music-base")       # the base's run made from the fine-tune's arrays
    (d / "chip.json").write_text(json.dumps({"device": "test chip", "total_s": 12.5}))
    res = st.finish_chip(s2, run_dir, ref_dir, st.chip_plan(MODEL, BASE), {"chip": "reference", "control_chip": "control_reference"})
    assert res["control_discrimination"]["ok"] is False
    assert any("nearer to the base's own reference" in p or "not nearer" in p for p in st.chip_problems(s2, run_dir, ref_dir))


def test_the_gate_refuses_an_edited_discrimination_record(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    res["control_discrimination"]["other_rel_l2"] = 99.0
    (s2 / "audio_result.json").write_text(json.dumps(res))
    assert any("recorded discrimination" in p for p in st.chip_problems(s2, run_dir, ref_dir))


def test_the_fine_tune_pair_is_still_held_to_the_bars(tmp_path):
    run_dir, ref_dir, s2, _ = stage2(tmp_path, chip_noise=0.8)
    assert any(p.startswith("chip_vs_reference") and "does not agree" in p for p in st.chip_problems(s2, run_dir, ref_dir))


def test_a_latent_of_another_shape_cannot_show_discrimination(tmp_path):
    a = make_run(tmp_path / "a", "small-music-base")
    b = make_run(tmp_path / "b", "small-music-base", latent=np.zeros((1, 4, 9)))
    c = make_run(tmp_path / "c", "small-music-base")
    d = st.control_discrimination(a, c, b)
    assert d == {"ok": False, "own_rel_l2": None, "other_rel_l2": None}
    assert st.closeness(a, b) is None


# ---- identity of the snapshot a run loaded (Copilot review of PR 165) ----------------------------------------------------------------

def test_the_snapshot_identity_is_the_directory_name_and_the_hash_of_the_config_behind_a_link(tmp_path):
    import hashlib
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "abc").write_text('{"x": 1}')
    snap = tmp_path / "snapshots" / "rev123"
    snap.mkdir(parents=True)
    (snap / "model_config.json").symlink_to(blobs / "abc")
    assert st.snapshot_identity(snap) == {"revision": "rev123", "config_sha256": hashlib.sha256(b'{"x": 1}').hexdigest()}
    with pytest.raises(st.AudioStageError):
        st.snapshot_identity(tmp_path / "nowhere")


def run_facts_of(tmp_path, **kw):
    return st.run_facts(tmp_path, str(make_run(tmp_path / "r", **kw).relative_to(tmp_path)))


def test_identity_problems_name_each_difference_and_a_missing_expectation(tmp_path):
    facts = run_facts_of(tmp_path)
    assert st.identity_problems(facts, EXP, COMMIT, "x") == []
    assert any("revision" in p for p in st.identity_problems(facts, {**EXP, "revision": "other"}, COMMIT, "x"))
    assert any("model_config.json" in p for p in st.identity_problems(facts, {**EXP, "config_sha256": "1" * 64}, COMMIT, "x"))
    assert any("library commit" in p for p in st.identity_problems(facts, EXP, "d" * 40, "x"))
    for missing in (None, {}, {"revision": REV}, {"config_sha256": "0" * 64}, "x"):
        assert any("no expected model revision" in p for p in st.identity_problems(facts, missing, COMMIT, "x")), missing


def test_a_fresh_run_of_another_revision_is_refused_by_stage_1(tmp_path):
    with pytest.raises(st.AudioStageError) as e:
        stage1(tmp_path, expected={"small-music": {**EXP, "revision": "the-selected-snapshot"}, "small-music-base": EXP})
    assert "the-selected-snapshot" in str(e.value) and "reference" in str(e.value)


def test_a_fresh_run_with_another_model_config_is_refused_by_stage_1(tmp_path):
    with pytest.raises(st.AudioStageError) as e:
        stage1(tmp_path, expected={"small-music": {**EXP, "config_sha256": "2" * 64}, "small-music-base": EXP})
    assert "model_config.json" in str(e.value)


def test_stage_1_refuses_a_run_when_no_expectation_exists_for_its_model(tmp_path):
    with pytest.raises(st.AudioStageError) as e:
        stage1(tmp_path, expected={"small-music": EXP})
    assert "no expected model revision" in str(e.value)


def test_stage_1_refuses_a_run_made_with_another_library_commit(tmp_path):
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stages" / "1"
    stage_dir.mkdir(parents=True)
    with pytest.raises(st.AudioStageError) as e:
        st.run_references(stage_dir, run_dir, PATHS, st.reference_plan(MODEL, BASE), FakeRunner(), expected={"small-music": EXP, "small-music-base": EXP},
                          library_commit="e" * 40, script=Path("/skills/make_reference.py"))
    assert "library commit" in str(e.value)


def test_the_cache_is_not_used_for_a_snapshot_with_another_config(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    make_run(cache / "small-music")
    assert st.cached_run(cache, "small-music", {**EXP, "config_sha256": "3" * 64}, COMMIT) is None
    assert st.cached_run(cache, "small-music", {"revision": REV}, COMMIT) is None


def test_the_stage_1_gate_refuses_a_run_of_another_revision_and_a_record_without_expectations(tmp_path):
    run_dir, stage_dir, rec, _ = stage1(tmp_path)
    path = stage_dir / "audio_reference.json"
    good = json.loads(path.read_text())
    bad = json.loads(json.dumps(good))
    bad["expected"]["small-music"]["revision"] = "another"
    path.write_text(json.dumps(bad))
    assert any("it loaded revision" in p for p in st.reference_problems(stage_dir, run_dir))
    bad = json.loads(json.dumps(good))
    del bad["expected"]
    path.write_text(json.dumps(bad))
    assert any("no expected model revision" in p for p in st.reference_problems(stage_dir, run_dir))
    bad = json.loads(json.dumps(good))
    bad["library_commit"] = "f" * 40
    path.write_text(json.dumps(bad))
    assert any("library commit" in p for p in st.reference_problems(stage_dir, run_dir))


def test_stage_2_refuses_a_chip_run_that_loaded_another_revision(tmp_path):
    run_dir, ref_dir, rec, _ = stage1(tmp_path)
    s2 = run_dir / "stages" / "2"
    (s2 / "chip").mkdir(parents=True)
    plan = st.chip_plan(MODEL, BASE)
    for item in plan:
        ref_role = {"chip": "reference", "control_chip": "control_reference"}[item["role"]]
        twin(run_dir / rec["runs"][ref_role]["dir"], s2 / "chip" / item["name"], noise=0.001, **({"revision": "wrong"} if item["role"] == "chip" else {}))
    with pytest.raises(st.AudioStageError) as e:
        st.finish_chip(s2, run_dir, ref_dir, plan, {"chip": "reference", "control_chip": "control_reference"})
    assert "wrong" in str(e.value)


def test_the_stage_2_gate_refuses_a_chip_run_of_another_revision(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    d = run_dir / res["runs"]["chip"]["dir"]
    shutil.rmtree(d)
    make_run(d, dtype="bfloat16", device="tt", revision="wrong", wave=music(seed=1))
    res["runs"]["chip"]["manifest_sha256"] = st.sha256_file(d / "manifest.json")
    (s2 / "audio_result.json").write_text(json.dumps(res))
    assert any("it loaded revision" in p for p in st.chip_problems(s2, run_dir, ref_dir))


# ---- the informational comparisons and the repeats are checked too ---------------------------------------------------------------------

def test_the_gate_recomputes_the_host_comparison_and_the_environment_floor(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    path = s2 / "audio_result.json"
    a = json.loads(json.dumps(res))
    a["agreement"]["chip_vs_host_same_env"]["latent_final"]["rel_l2"] = 0.0
    path.write_text(json.dumps(a))
    assert any(p.startswith("chip_vs_host_same_env") for p in st.chip_problems(s2, run_dir, ref_dir))
    b = json.loads(json.dumps(res))
    b["environment_floor"]["waveform"]["pcc"] = 0.5
    path.write_text(json.dumps(b))
    assert any(p.startswith("environment_floor") for p in st.chip_problems(s2, run_dir, ref_dir))
    c = json.loads(json.dumps(res))
    del c["agreement"]["chip_vs_host_same_env"]
    path.write_text(json.dumps(c))
    assert any(p.startswith("chip_vs_host_same_env") for p in st.chip_problems(s2, run_dir, ref_dir))


def test_a_failing_informational_comparison_does_not_fail_the_gate_when_it_is_recorded_truthfully(tmp_path):
    """No pass bar is applied to the host comparison or the floor: a chip run that is far from the host run still passes when it agrees with the reference."""
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    assert res["agreement"]["chip_vs_host_same_env"]["ok"] is True           # the fixture's runs are near each other; the gate does not read this value
    assert st.chip_problems(s2, run_dir, ref_dir) == []


def test_the_result_lists_every_repeat_and_the_gate_checks_them(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    assert [len(res["repeats"][r]) for r in ("chip", "control_chip")] == [REPEATS, REPEATS]
    assert [e["dir"].rsplit("/", 1)[1] for e in res["repeats"]["chip"]] == ["small-music", "small-music-r1"]
    path = s2 / "audio_result.json"
    short = json.loads(json.dumps(res))
    short["repeats"]["chip"] = short["repeats"]["chip"][:1]
    path.write_text(json.dumps(short))
    assert any("lists 1 repeats; the config asked for 2" in p for p in st.chip_problems(s2, run_dir, ref_dir))
    wrong = json.loads(json.dumps(res))
    wrong["repeats"]["chip"][1]["dir"] = res["repeats"]["chip"][0]["dir"]
    path.write_text(json.dumps(wrong))
    assert any("directory is not the expected one" in p for p in st.chip_problems(s2, run_dir, ref_dir))
    none = json.loads(json.dumps(res))
    del none["repeats"]
    path.write_text(json.dumps(none))
    assert any("no repeats record" in p for p in st.chip_problems(s2, run_dir, ref_dir))
    path.write_text(json.dumps(res))
    f = run_dir / res["repeats"]["chip"][1]["dir"] / "arrays" / "latent__final.npy"
    f.write_bytes(f.read_bytes()[:-4] + b"\x01\x02\x03\x04")
    assert any(p.startswith("chip repeat 1") for p in st.chip_problems(s2, run_dir, ref_dir))


def test_stage_2_names_a_repeat_that_left_no_manifest(tmp_path):
    run_dir, ref_dir, rec, _ = stage1(tmp_path)
    s2 = run_dir / "stages" / "2"
    (s2 / "chip").mkdir(parents=True)
    plan = st.chip_plan(MODEL, BASE)
    st.write_chip_config(s2, PATHS, "/hf", plan, repeats=2)
    for item in plan:
        ref_role = {"chip": "reference", "control_chip": "control_reference"}[item["role"]]
        twin(run_dir / rec["runs"][ref_role]["dir"], s2 / "chip" / item["name"], noise=0.001)
    with pytest.raises(st.AudioStageError) as e:
        st.finish_chip(s2, run_dir, ref_dir, plan, {"chip": "reference", "control_chip": "control_reference"})
    assert "repeat 1" in str(e.value) and "no manifest" in str(e.value)


def test_stage_2_refuses_a_repeat_that_loaded_another_revision(tmp_path):
    run_dir, ref_dir, rec, _ = stage1(tmp_path)
    s2 = run_dir / "stages" / "2"
    (s2 / "chip").mkdir(parents=True)
    plan = st.chip_plan(MODEL, BASE)
    st.write_chip_config(s2, PATHS, "/hf", plan, repeats=2)
    for item in plan:
        ref_role = {"chip": "reference", "control_chip": "control_reference"}[item["role"]]
        for k in range(2):
            twin(run_dir / rec["runs"][ref_role]["dir"], s2 / "chip" / st.repeat_name(item["name"], k), noise=0.001,
                 **({"revision": "wrong-repeat"} if (item["role"] == "chip" and k == 1) else {}))
    with pytest.raises(st.AudioStageError) as e:
        st.finish_chip(s2, run_dir, ref_dir, plan, {"chip": "reference", "control_chip": "control_reference"})
    assert "wrong-repeat" in str(e.value) and "repeat 1" in str(e.value)


# ---- copies in the record, clip paths, a missing control, cancellation (Copilot review of PR 165, third round) ------------------------------

def test_the_gate_refuses_edited_timings_and_other_copies_that_differ_from_the_manifest(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    path = s2 / "audio_result.json"
    for key, value in (("timings", {"generate_and_decode_s": 0.001}), ("model", "small-music-base"), ("revision", "other"), ("config_sha256", "9" * 64),
                       ("library_commit", "9" * 40), ("settings", {**res["runs"]["chip"]["settings"], "steps": 1})):
        edited = json.loads(json.dumps(res))
        edited["runs"]["chip"][key] = value
        path.write_text(json.dumps(edited))
        assert any(f"its recorded {key} is not what its manifest says" in p for p in st.chip_problems(s2, run_dir, ref_dir)), key


def test_the_gate_refuses_edited_repeat_timings(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    edited = json.loads(json.dumps(res))
    edited["repeats"]["chip"][1]["timings"] = {"generate_and_decode_s": 0.001}
    (s2 / "audio_result.json").write_text(json.dumps(edited))
    assert any("chip repeat 1: its recorded timings" in p for p in st.chip_problems(s2, run_dir, ref_dir))


def test_the_stage_1_gate_refuses_edited_timings(tmp_path):
    run_dir, stage_dir, rec, _ = stage1(tmp_path)
    data = json.loads((stage_dir / "audio_reference.json").read_text())
    data["runs"]["reference"]["timings"] = {"generate_and_decode_s": 0.001}
    (stage_dir / "audio_reference.json").write_text(json.dumps(data))
    assert any("its recorded timings" in p for p in st.reference_problems(stage_dir, run_dir))


def test_the_gate_refuses_a_clip_path_that_points_at_another_models_clip(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    res["runs"]["chip"]["clip"] = res["runs"]["control_chip"]["clip"]
    (s2 / "audio_result.json").write_text(json.dumps(res))
    assert any("its clip path is" in p for p in st.chip_problems(s2, run_dir, ref_dir))


def test_the_gate_requires_the_control_run_when_stage_1_made_a_control_reference(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    del res["runs"]["control_chip"]
    (s2 / "audio_result.json").write_text(json.dumps(res))
    assert any("needs a control_chip run" in p for p in st.chip_problems(s2, run_dir, ref_dir))


def test_a_run_without_a_control_reference_needs_no_control_chip_run(tmp_path):
    run_dir, ref_dir, rec, _ = stage1(tmp_path, plan=[p for p in st.reference_plan(MODEL, BASE) if p["role"] != "control_reference"])
    s2 = run_dir / "stages" / "2"
    (s2 / "chip").mkdir(parents=True)
    plan = st.chip_plan(MODEL, None)
    st.write_chip_config(s2, PATHS, "/hf", plan, repeats=1)
    twin(run_dir / rec["runs"]["reference"]["dir"], s2 / "chip" / "small-music", noise=0.001)
    st.finish_chip(s2, run_dir, ref_dir, plan, {"chip": "reference"})
    assert st.chip_problems(s2, run_dir, ref_dir) == []


def test_a_cancelled_numeric_verb_takes_its_cpu_runs_down_with_it(tmp_path):
    """The CPU runs live in sessions of their own. The verb is sent SIGTERM first; its cleanup kills those sessions, so nothing outlives it."""
    import time
    from orchard import audio_remote, audio_runtime
    marker = tmp_path / "cpu-run-survived"
    fake = tmp_path / "fake_python"
    fake.write_text(f"#!/bin/sh\nsleep 3\ntouch {marker}\n")
    fake.chmod(0o755)
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stages" / "1"
    stage_dir.mkdir(parents=True)
    paths = audio_runtime.AudioPaths(str(fake), str(fake), "/src")
    args = {"stage_dir": str(stage_dir), "run_dir": str(run_dir), "paths": paths.record(), "plan": st.reference_plan(MODEL, None),
            "expected": {"small-music": EXP}, "library_commit": COMMIT, "cache_dir": None, "env": {"PATH": "/usr/bin:/bin"}, "timeout_s": 60.0}
    with pytest.raises(st.AudioStageError):
        audio_remote.call(sys.executable, "references", args, timeout_s=1.5, env={"PATH": "/usr/bin:/bin"})
    time.sleep(4.0)                                       # the CPU run would have written its file at 3 s
    assert not marker.exists()
