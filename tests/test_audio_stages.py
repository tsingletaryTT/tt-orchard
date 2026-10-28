# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/audio_stages.py: the sound checks, the comparisons, stage 1's CPU runs and stage 2's result with their gates. Every run here is a small
synthetic recorded run (orchard/audio_reference.py); no model, chip or interpreter other than this one is used."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orchard import audio_reference as ar
from orchard import audio_stages as st

SR = 44100
COMMIT = "c" * 40
REV = "rev-small"


def music(seconds=10.0, seed=0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    g = np.random.default_rng(seed)
    base = sum(np.sin(2 * np.pi * f * t) for f in (220, 330, 440)) / 3 * amp
    return np.stack([base + 0.01 * g.standard_normal(t.size), base * 0.9 + 0.01 * g.standard_normal(t.size)]).astype(np.float32)


def make_run(path, name="small-music", *, dtype="float32", device="cpu", wave=None, latent=None, layer=None, revision=REV, commit=COMMIT,
             settings_override=None, seed=1):
    s = {**st.settings_for(name), "dtype": dtype, "device": device, **(settings_override or {})}
    rec = ar.Recorder(path, model={"name": name, "revision": revision, "config_sha256": "0" * 64, "diffusion_objective": "rf_denoiser",
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
                    layer=out[("layer", "layer_00")], revision=m["model"]["revision"], **kw)


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
    assert st.cached_run(cache, "small-music", REV, COMMIT) == cache / "small-music"
    assert st.cached_run(cache, "small-music", "other-rev", COMMIT) is None
    assert st.cached_run(cache, "small-music", REV, "d" * 40) is None
    assert st.cached_run(cache, "small-music", None, COMMIT) is None
    assert st.cached_run(cache, "small-music-base", REV, COMMIT) is None
    assert st.cached_run(None, "small-music", REV, COMMIT) is None


@pytest.mark.parametrize("kw", [{"dtype": "bfloat16"}, {"device": "tt"}, {"settings_override": {"steps": 9}}, {"settings_override": {"seed": 1}}])
def test_a_cached_run_made_with_other_settings_is_not_used(tmp_path, kw):
    cache = tmp_path / "cache"
    cache.mkdir()
    make_run(cache / "small-music", **kw)
    assert st.cached_run(cache, "small-music", REV, COMMIT) is None


def test_a_cached_run_with_a_changed_array_is_not_used(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    run = make_run(cache / "small-music")
    f = run / "arrays" / "latent__final.npy"
    f.write_bytes(f.read_bytes()[:-4] + b"\x00\x00\x00\x00")
    assert st.cached_run(cache, "small-music", REV, COMMIT) is None


def test_a_cache_entry_that_is_a_link_is_not_used(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    real = make_run(tmp_path / "real")
    (cache / "small-music").symlink_to(real)
    assert st.cached_run(cache, "small-music", REV, COMMIT) is None


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


def stage1(tmp_path, runner=None, cache=None, revisions=None, plan=None):
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stages" / "1"
    stage_dir.mkdir(parents=True)
    runner = runner or FakeRunner()
    plan = plan or st.reference_plan(MODEL, BASE)
    rec = st.run_references(stage_dir, run_dir, PATHS, plan, runner, revisions=revisions or {"small-music": REV, "small-music-base": REV},
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
    for item in plan:
        ref_role = {"chip": "reference", "control_chip": "control_reference"}[item["role"]]
        src = run_dir / rec["runs"][ref_role]["dir"]
        dst = s2 / "chip" / item["name"]
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
    assert [(r["name"], r["steps"], r["cfg_scale"], r["sampler"]) for r in cfg["runs"]] == [
        ("small-music", 8, 1.0, "pingpong"), ("small-music-base", 50, 7.0, "euler")]
    cmd = st.chip_test_command(PATHS, tmp_path, Path("/skills/audio_chip_test.py"))
    assert cmd == f"/chip/python /skills/audio_chip_test.py --config {tmp_path}/audio_chip_config.json"
