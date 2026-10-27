# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/audio_reference.py: the record of a reference run and the check of one run against another. Everything runs on small random
arrays in a temporary directory; no model, no chip."""
import io
import json
import math

import numpy as np
import pytest

from orchard import audio_reference as ar
from orchard.audio_reference import (Bars, Recorder, ReferenceError, compare_control, compare_runs, load_run, metrics, verdict)

SETTINGS = {"prompt": "a piano", "duration": 10, "steps": 8, "cfg_scale": 1.0, "sampler": "pingpong", "seed": 42,
            "dtype": "float32", "sample_size": 705600}
MODEL = {"id": "org/model", "revision": "a" * 40}
LIB = {"name": "stable-audio-3", "commit": "b" * 40}


def make_run(tmp_path, name="run", settings=None, arrays=None, seeds=None, **kw):
    rec = Recorder(tmp_path / name, model=MODEL, library=LIB, settings={**SETTINGS, **(settings or {})}, seeds=seeds or {"torch": 42}, **kw)
    rng = np.random.default_rng(0)
    for (kind, nm), arr in (arrays if arrays is not None else {
            ("conditioning", "prompt"): rng.normal(size=(1, 8, 4)), ("layer", "0"): rng.normal(size=(1, 20, 16)),
            ("step_latent", "0"): rng.normal(size=(1, 4, 12)), ("latent", "final"): rng.normal(size=(1, 4, 12)),
            ("waveform", "audio"): rng.normal(size=(2, 4096))}).items():
        rec.record(kind, nm, arr)
    rec.finish()
    return tmp_path / name


# ---- Recorder -------------------------------------------------------------------------------------

def test_a_recorded_run_loads_and_the_arrays_come_back_as_float32(tmp_path):
    run = load_run(make_run(tmp_path))
    arr = run.array("layer", "0")
    assert arr.dtype == np.float32 and arr.shape == (1, 20, 16)
    assert run.settings["seed"] == 42 and run.manifest["seeds"] == {"torch": 42}
    assert run.manifest["model"] == MODEL and run.manifest["library"] == LIB


def test_the_manifest_holds_each_files_sha256_shape_and_finiteness(tmp_path):
    root = make_run(tmp_path)
    m = json.loads((root / "manifest.json").read_text())
    import hashlib
    for e in m["artifacts"]:
        assert e["sha256"] == hashlib.sha256((root / e["file"]).read_bytes()).hexdigest()
        assert e["dtype"] == "float32" and e["finite"] is True and isinstance(e["shape"], list)


def test_a_non_finite_array_is_recorded_and_flagged(tmp_path):
    root = make_run(tmp_path, arrays={("layer", "0"): np.array([1.0, np.nan])})
    assert json.loads((root / "manifest.json").read_text())["artifacts"][0]["finite"] is False


def test_the_same_kind_and_name_twice_is_refused(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.record("layer", "0", np.zeros(2))
    with pytest.raises(ReferenceError, match="already recorded"):
        rec.record("layer", "0", np.zeros(2))
    rec.record("step_latent", "0", np.zeros(2))        # the same name under another kind is fine


@pytest.mark.parametrize("kind,name", [("mystery", "a"), ("layer", ""), ("layer", "../x"), ("layer", "a/b"), ("layer", "a b"), ("layer", "x" * 121)])
def test_a_bad_kind_or_name_is_refused(tmp_path, kind, name):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    with pytest.raises(ReferenceError):
        rec.record(kind, name, np.zeros(2))


def test_a_non_numeric_array_is_refused(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    with pytest.raises(ReferenceError, match="not numeric"):
        rec.record("layer", "0", np.array(["a", "b"]))


def test_settings_need_every_match_key_and_allow_no_unknown_key(tmp_path):
    with pytest.raises(ReferenceError, match="missing"):
        Recorder(tmp_path / "a", model=MODEL, library=LIB, settings={k: v for k, v in SETTINGS.items() if k != "seed"})
    with pytest.raises(ReferenceError, match="unknown settings"):
        Recorder(tmp_path / "b", model=MODEL, library=LIB, settings={**SETTINGS, "surprise": 1})


@pytest.mark.parametrize("model,lib", [({}, LIB), (MODEL, {}), (None, LIB), (MODEL, "x")])
def test_the_model_and_library_must_be_named(tmp_path, model, lib):
    with pytest.raises(ReferenceError):
        Recorder(tmp_path / "r", model=model, library=lib, settings=SETTINGS)


def test_an_existing_manifest_is_never_replaced(tmp_path):
    root = make_run(tmp_path)
    with pytest.raises(ReferenceError, match="never overwritten"):
        Recorder(root, model=MODEL, library=LIB, settings=SETTINGS)


def test_no_temporary_file_is_left_behind(tmp_path):
    root = make_run(tmp_path)
    assert not list(root.rglob("*.tmp"))


def test_float16_and_integer_arrays_are_stored_as_float32(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    e1 = rec.record("layer", "h", np.ones(3, dtype=np.float16))
    e2 = rec.record("layer", "i", np.arange(3))
    assert e1["dtype"] == e2["dtype"] == "float32"


# ---- load_run -------------------------------------------------------------------------------------

def edit_manifest(root, fn):
    p = root / "manifest.json"
    m = json.loads(p.read_text())
    fn(m)
    p.write_text(json.dumps(m))


def test_an_array_that_changed_after_it_was_recorded_is_refused(tmp_path):
    root = make_run(tmp_path)
    np.save(root / "arrays" / "layer__0.npy", np.zeros((1, 20, 16), dtype=np.float32))
    with pytest.raises(ReferenceError, match="sha256"):
        load_run(root)


def test_verify_can_be_skipped_but_the_shape_is_still_checked_on_read(tmp_path):
    root = make_run(tmp_path)
    np.save(root / "arrays" / "layer__0.npy", np.zeros((3,), dtype=np.float32))
    run = load_run(root, verify=False)
    with pytest.raises(ReferenceError, match="shape"):
        run.array("layer", "0")


def test_a_missing_manifest_or_array_is_refused(tmp_path):
    with pytest.raises(ReferenceError):
        load_run(tmp_path / "nothing")
    root = make_run(tmp_path)
    (root / "arrays" / "layer__0.npy").unlink()
    with pytest.raises(ReferenceError, match="missing"):
        load_run(root)


@pytest.mark.parametrize("text", ["not json", "[]", '{"schema": 2}', "{}"])
def test_a_manifest_of_the_wrong_kind_is_refused(tmp_path, text):
    root = make_run(tmp_path)
    (root / "manifest.json").write_text(text)
    with pytest.raises(ReferenceError):
        load_run(root)


@pytest.mark.parametrize("fn", [
    lambda m: m.update(model="x"), lambda m: m.update(library=[]), lambda m: m.update(settings=3), lambda m: m.update(seeds=[1]),
    lambda m: m.update(artifacts="x"), lambda m: m["settings"].pop("steps"),
    lambda m: m["artifacts"][0].update(kind="nonsense"), lambda m: m["artifacts"][0].update(name="../x"),
    lambda m: m["artifacts"][0].update(sha256="zz"), lambda m: m["artifacts"][0].update(shape=[-1]),
    lambda m: m["artifacts"][0].update(shape=[True]), lambda m: m["artifacts"][0].update(shape="x"),
    lambda m: m["artifacts"][0].update(dtype="float64"), lambda m: m["artifacts"][0].update(file=3),
    lambda m: m["artifacts"].append(dict(m["artifacts"][0])), lambda m: m["artifacts"].append("x"),
])
def test_a_malformed_manifest_is_refused(tmp_path, fn):
    root = make_run(tmp_path)
    edit_manifest(root, fn)
    with pytest.raises(ReferenceError):
        load_run(root)


def test_an_array_path_that_leaves_the_arrays_directory_is_refused(tmp_path):
    root = make_run(tmp_path)
    outside = root / "elsewhere.npy"
    np.save(outside, np.zeros(2, dtype=np.float32))
    edit_manifest(root, lambda m: m["artifacts"][0].update(file="elsewhere.npy"))
    with pytest.raises(ReferenceError, match="outside"):
        load_run(root, verify=False)


def test_a_symlinked_array_is_refused(tmp_path):
    root = make_run(tmp_path)
    target = tmp_path / "real.npy"
    (root / "arrays" / "layer__0.npy").rename(target)
    (root / "arrays" / "layer__0.npy").symlink_to(target)
    with pytest.raises(ReferenceError, match="link"):
        load_run(root)


def test_a_symlinked_manifest_is_refused(tmp_path):
    root = make_run(tmp_path)
    (root / "manifest.json").rename(tmp_path / "m.json")
    (root / "manifest.json").symlink_to(tmp_path / "m.json")
    with pytest.raises(ReferenceError, match="link"):
        load_run(root)


def test_an_oversized_manifest_is_refused(tmp_path, monkeypatch):
    root = make_run(tmp_path)
    monkeypatch.setattr(ar, "MAX_MANIFEST_BYTES", 10)
    with pytest.raises(ReferenceError, match="larger"):
        load_run(root)


def test_an_oversized_array_is_refused_on_read(tmp_path, monkeypatch):
    run = load_run(make_run(tmp_path))
    monkeypatch.setattr(ar, "MAX_ARRAY_BYTES", 10)
    with pytest.raises(ReferenceError, match="larger"):
        run.array("layer", "0")


def test_a_pickled_array_is_not_loaded(tmp_path):
    root = make_run(tmp_path)
    np.save(root / "arrays" / "layer__0.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(Exception):
        load_run(root, verify=False).array("layer", "0")


# ---- metrics --------------------------------------------------------------------------------------

def test_identical_arrays_give_perfect_numbers():
    x = np.random.default_rng(1).normal(size=100).astype(np.float32)
    m = metrics(x, x.copy())
    assert m["pcc"] == pytest.approx(1.0) and m["rel_l2"] == 0.0 and m["slope"] == pytest.approx(1.0) and m["max_abs_error"] == 0.0


def test_a_scaled_copy_has_perfect_correlation_and_the_wrong_slope():
    x = np.random.default_rng(1).normal(size=200).astype(np.float32)
    m = metrics(x, 964.0 * x)
    assert m["pcc"] == pytest.approx(1.0) and m["slope"] == pytest.approx(964.0) and m["rel_l2"] == pytest.approx(963.0, rel=1e-4)


def test_an_offset_copy_is_caught_by_the_error_and_not_by_the_correlation():
    x = np.random.default_rng(1).normal(size=200).astype(np.float32)
    m = metrics(x, x + 5.0)
    assert m["pcc"] == pytest.approx(1.0) and m["slope"] == pytest.approx(1.0) and m["rel_l2"] > 1.0


def test_an_anticorrelated_copy_has_negative_pcc_and_slope():
    x = np.random.default_rng(1).normal(size=200).astype(np.float32)
    m = metrics(x, -x)
    assert m["pcc"] == pytest.approx(-1.0) and m["slope"] == pytest.approx(-1.0)


def test_shapes_that_differ_are_not_comparable():
    m = metrics(np.zeros((2, 3)), np.zeros((3, 2)))
    assert m["comparable"] is False and "shapes differ" in m["reason"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_value_on_either_side_is_not_comparable(bad):
    good = np.ones(4)
    broken = np.array([1.0, bad, 1.0, 1.0])
    assert metrics(good, broken)["comparable"] is False
    assert metrics(broken, good)["comparable"] is False


def test_empty_arrays_are_not_comparable():
    assert metrics(np.zeros(0), np.zeros(0))["comparable"] is False


def test_equal_constants_agree_and_unequal_constants_do_not():
    assert metrics(np.full(5, 2.0), np.full(5, 2.0))["pcc"] == 1.0
    m = metrics(np.full(5, 2.0), np.full(5, 3.0))
    assert m["pcc"] is None and m["slope"] is None and m["rel_l2"] == pytest.approx(0.5)


def test_a_zero_reference_has_infinite_relative_error_unless_the_candidate_is_zero_too():
    assert metrics(np.zeros(4), np.ones(4))["rel_l2"] == math.inf
    assert metrics(np.zeros(4), np.zeros(4))["rel_l2"] == 0.0


def test_a_constant_candidate_against_a_varying_reference_has_no_pcc():
    m = metrics(np.arange(5.0), np.full(5, 1.0))
    assert m["pcc"] is None


# ---- verdict --------------------------------------------------------------------------------------

BARS = Bars(0.99, 0.1, 0.05)


def test_each_number_can_fail_the_verdict_alone():
    x = np.random.default_rng(1).normal(size=500)
    ok, why = verdict(metrics(x, x), BARS)
    assert ok and why == []
    ok, why = verdict(metrics(x, 3 * x), BARS)           # pcc 1, rel_l2 2, slope 3: error and slope fail, pcc passes
    assert not ok and any("relative L2" in w for w in why) and any("slope" in w for w in why) and not any("pcc" in w for w in why)
    ok, why = verdict(metrics(x, x + 0.5 * np.random.default_rng(2).normal(size=500)), Bars(0.99, 10.0, 10.0))
    assert not ok and len(why) == 1 and "pcc" in why[0]
    ok, why = verdict(metrics(x, x * 1.2), Bars(0.99, 10.0, 0.05))     # only the slope is out
    assert not ok and len(why) == 1 and "slope" in why[0]
    ok, why = verdict(metrics(x, x + 3.0), Bars(0.99, 0.5, 5.0))       # only the error is out
    assert not ok and len(why) == 1 and "relative L2" in why[0]


def test_a_value_exactly_on_a_bar_passes():
    assert verdict({"comparable": True, "pcc": 0.99, "rel_l2": 0.1, "slope": 1.0625}, Bars(0.99, 0.1, 0.0625))[0]


def test_a_missing_pcc_fails_and_says_why():
    ok, why = verdict({"comparable": True, "pcc": None, "rel_l2": 0.0, "slope": None}, BARS)
    assert not ok and "constant" in why[0]


def test_a_comparison_that_could_not_be_made_fails():
    ok, why = verdict({"comparable": False, "reason": "x"}, BARS)
    assert not ok and why == ["x"]


# ---- compare_runs ---------------------------------------------------------------------------------

def perturbed(tmp_path, name, scale=0.0, noise=0.0, skip=()):
    ref = load_run(tmp_path / "ref")
    rec = Recorder(tmp_path / name, model=MODEL, library=LIB, settings=SETTINGS, seeds={"torch": 42})
    rng = np.random.default_rng(5)
    for (kind, nm) in ref.entries:
        if (kind, nm) in skip:
            continue
        a = ref.array(kind, nm)
        rec.record(kind, nm, a * (1 + scale) + noise * rng.normal(size=a.shape))
    rec.finish()
    return load_run(tmp_path / name)


def test_a_run_compared_with_a_copy_of_itself_passes(tmp_path):
    make_run(tmp_path, "ref")
    report = compare_runs(load_run(tmp_path / "ref"), perturbed(tmp_path, "cand"))
    assert report["ok"] and report["comparable"] and len(report["results"]) == 5 and report["problems"] == []


def test_a_scaled_candidate_fails_on_every_kind_with_a_bar_on_scale(tmp_path):
    make_run(tmp_path, "ref")
    report = compare_runs(load_run(tmp_path / "ref"), perturbed(tmp_path, "cand", scale=2.0))
    assert not report["ok"] and all(not r["ok"] for r in report["results"])


def test_a_candidate_that_skipped_an_array_fails(tmp_path):
    make_run(tmp_path, "ref")
    report = compare_runs(load_run(tmp_path / "ref"), perturbed(tmp_path, "cand", skip={("step_latent", "0")}))
    assert not report["ok"] and any("step_latent/0 is only in the reference" in p for p in report["problems"])


def test_an_array_only_the_candidate_has_fails(tmp_path):
    make_run(tmp_path, "ref", arrays={("layer", "0"): np.arange(8.0)})
    make_run(tmp_path, "cand", arrays={("layer", "0"): np.arange(8.0), ("layer", "1"): np.arange(8.0)})
    report = compare_runs(load_run(tmp_path / "ref"), load_run(tmp_path / "cand"))
    assert not report["ok"] and any("only in the candidate" in p for p in report["problems"])


def test_runs_with_no_shared_array_fail(tmp_path):
    make_run(tmp_path, "ref", arrays={("layer", "0"): np.arange(8.0)})
    make_run(tmp_path, "cand", arrays={("layer", "1"): np.arange(8.0)})
    report = compare_runs(load_run(tmp_path / "ref"), load_run(tmp_path / "cand"))
    assert not report["ok"] and any("share no array" in p for p in report["problems"])


@pytest.mark.parametrize("key,value", [("seed", 43), ("steps", 9), ("prompt", "other"), ("duration", 11), ("sampler", "euler"),
                                       ("cfg_scale", 7.0), ("dtype", "bfloat16"), ("sample_size", 1)])
def test_runs_with_different_settings_are_not_compared(tmp_path, key, value):
    make_run(tmp_path, "ref")
    make_run(tmp_path, "cand", settings={key: value})
    report = compare_runs(load_run(tmp_path / "ref"), load_run(tmp_path / "cand"))
    assert report["ok"] is False and report["comparable"] is False and key in report["problems"][0]


def test_the_device_setting_may_differ(tmp_path):
    rec = Recorder(tmp_path / "ref", model=MODEL, library=LIB, settings={**SETTINGS, "device": "cpu"})
    rec.record("layer", "0", np.arange(8.0))
    rec.finish()
    rec = Recorder(tmp_path / "cand", model=MODEL, library=LIB, settings={**SETTINGS, "device": "tt"})
    rec.record("layer", "0", np.arange(8.0))
    rec.finish()
    assert compare_runs(load_run(tmp_path / "ref"), load_run(tmp_path / "cand"))["ok"]


def test_custom_bars_replace_the_defaults_for_their_kind(tmp_path):
    make_run(tmp_path, "ref")
    cand = perturbed(tmp_path, "cand", noise=0.3)
    loose = {k: Bars(0.0, 10.0, 10.0) for k in ar.KINDS}
    assert compare_runs(load_run(tmp_path / "ref"), cand, loose)["ok"]
    strict = compare_runs(load_run(tmp_path / "ref"), cand, {"layer": Bars(0.9999, 0.001, 0.001)})
    assert not strict["ok"] and [r for r in strict["results"] if r["kind"] == "layer"][0]["ok"] is False


def test_the_noise_draws_must_come_back_unchanged(tmp_path):
    make_run(tmp_path, "ref", arrays={("noise", "0"): np.arange(8.0)})
    make_run(tmp_path, "cand", arrays={("noise", "0"): np.arange(8.0) + 1e-3})
    assert not compare_runs(load_run(tmp_path / "ref"), load_run(tmp_path / "cand"))["ok"]


def test_a_non_finite_candidate_array_fails(tmp_path):
    make_run(tmp_path, "ref", arrays={("layer", "0"): np.arange(8.0)})
    bad = np.arange(8.0)
    bad[3] = np.nan
    make_run(tmp_path, "cand", arrays={("layer", "0"): bad})
    report = compare_runs(load_run(tmp_path / "ref"), load_run(tmp_path / "cand"))
    assert not report["ok"] and "not finite" in report["results"][0]["reasons"][0]


def test_the_report_names_the_bars_each_result_used(tmp_path):
    make_run(tmp_path, "ref")
    report = compare_runs(load_run(tmp_path / "ref"), perturbed(tmp_path, "cand"))
    assert all("pcc" in r["bars"] for r in report["results"])


# ---- the noise floor ------------------------------------------------------------------------------

def test_bars_come_from_the_worst_floor_result_loosened_by_the_margin(tmp_path):
    make_run(tmp_path, "ref")
    floor = compare_runs(load_run(tmp_path / "ref"), perturbed(tmp_path, "low", noise=0.01))
    bars = ar.bars_from_floor(floor, margin=3.0)
    layer = [r for r in floor["results"] if r["kind"] == "layer"][0]
    assert bars["layer"].rel_l2 == pytest.approx(3.0 * layer["rel_l2"])
    assert bars["layer"].pcc == pytest.approx(1.0 - 3.0 * (1.0 - layer["pcc"]))
    assert "three" in bars["layer"].source or "3.0 times" in bars["layer"].source
    assert set(bars) == {"conditioning", "layer", "step_latent", "latent", "waveform"}


def test_a_floor_run_that_could_not_be_compared_gives_no_bars(tmp_path):
    with pytest.raises(ReferenceError):
        ar.bars_from_floor({"comparable": False, "results": []})
    with pytest.raises(ReferenceError):
        ar.bars_from_floor({"comparable": True, "results": []})
    with pytest.raises(ReferenceError, match="not comparable"):
        ar.bars_from_floor({"comparable": True, "results": [{"kind": "layer", "name": "0", "comparable": False}]})


def test_a_margin_below_one_is_refused():
    with pytest.raises(ReferenceError):
        ar.bars_from_floor({"comparable": True, "results": [{"kind": "layer", "name": "0", "comparable": True, "pcc": 1.0, "rel_l2": 0.0, "slope": 1.0}]}, margin=0.5)


# ---- the control ----------------------------------------------------------------------------------

def test_two_models_with_different_waveforms_pass_the_control(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base", arrays={("waveform", "audio"): np.random.default_rng(9).normal(size=(2, 4096))}, settings={"steps": 50, "sampler": "euler", "cfg_scale": 7.0})
    report = compare_control(load_run(tmp_path / "ft"), load_run(tmp_path / "base"))
    assert report["ok"] and report["rel_l2"] > 0.05


def test_equal_waveforms_fail_the_control(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base")
    report = compare_control(load_run(tmp_path / "ft"), load_run(tmp_path / "base"))
    assert not report["ok"] and "same weights" in report["problems"][0]


def test_the_control_needs_the_same_prompt_duration_and_seed(tmp_path):
    make_run(tmp_path, "ft")
    for key, value in (("prompt", "x"), ("duration", 3), ("seed", 1)):
        make_run(tmp_path, f"b{key}", settings={key: value})
        report = compare_control(load_run(tmp_path / "ft"), load_run(tmp_path / f"b{key}"))
        assert report["ok"] is False and report["comparable"] is False and key in report["problems"][0]


def test_the_control_needs_a_waveform_in_both_runs(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base", arrays={("layer", "0"): np.arange(4.0)})
    assert "no waveform/audio" in compare_control(load_run(tmp_path / "ft"), load_run(tmp_path / "base"))["problems"][0]
    assert "no waveform/audio" in compare_control(load_run(tmp_path / "base"), load_run(tmp_path / "ft"))["problems"][0]


def test_the_control_refuses_waveforms_of_different_length(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base", arrays={("waveform", "audio"): np.zeros((2, 100))})
    report = compare_control(load_run(tmp_path / "ft"), load_run(tmp_path / "base"))
    assert report["ok"] is False and "shapes differ" in report["problems"][0]


def test_the_control_threshold_is_a_parameter(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base", arrays={("waveform", "audio"): np.random.default_rng(0).normal(size=(2, 4096)) * 1.001})
    a, b = load_run(tmp_path / "ft"), load_run(tmp_path / "base")
    assert compare_control(a, b, min_rel_l2=0.05)["ok"] is True
    assert compare_control(a, b, min_rel_l2=100.0)["ok"] is False


# ---- the log-spectrum distance --------------------------------------------------------------------

def tone(freq, n=8192, phase=0.0):
    t = np.arange(n) / 8192.0
    return np.sin(2 * np.pi * freq * t + phase)[None, :]


def test_a_small_phase_shift_barely_moves_the_spectrum_distance_and_a_new_tone_does():
    same = ar.log_spectrum_distance(tone(440), tone(440, phase=0.3))
    other = ar.log_spectrum_distance(tone(440), tone(1200))
    assert same is not None and other is not None and same < 0.05 < other


def test_identical_signals_have_zero_distance():
    assert ar.log_spectrum_distance(tone(440), tone(440)) == 0.0


def test_the_distance_is_none_for_short_non_finite_or_mismatched_signals():
    assert ar.log_spectrum_distance(np.zeros((1, 100)), np.zeros((1, 100))) is None
    bad = tone(440)
    bad[0, 5] = np.nan
    assert ar.log_spectrum_distance(tone(440), bad) is None
    assert ar.log_spectrum_distance(tone(440), tone(440, n=4096)) is None
    assert ar.log_spectrum_distance(np.zeros((1, 4096)), np.zeros((1, 4096))) is None


# ---- command line ---------------------------------------------------------------------------------

def run_cli(*argv):
    out = io.StringIO()
    code = ar.main(list(argv), out=out)
    return code, out.getvalue()


def test_the_check_command_exits_0_for_a_match_and_1_for_a_mismatch(tmp_path):
    make_run(tmp_path, "ref")
    perturbed(tmp_path, "good")
    perturbed(tmp_path, "bad", scale=3.0)
    code, text = run_cli("check", str(tmp_path / "ref"), str(tmp_path / "good"))
    assert code == 0 and text.strip().endswith("ok")
    code, text = run_cli("check", str(tmp_path / "ref"), str(tmp_path / "bad"))
    assert code == 1 and "FAIL" in text and "slope" in text


def test_the_check_command_can_print_json(tmp_path):
    make_run(tmp_path, "ref")
    perturbed(tmp_path, "good")
    code, text = run_cli("check", str(tmp_path / "ref"), str(tmp_path / "good"), "--json")
    assert code == 0 and json.loads(text)["ok"] is True


def test_a_run_that_cannot_be_read_exits_2(tmp_path):
    code, text = run_cli("check", str(tmp_path / "x"), str(tmp_path / "y"))
    assert code == 2 and text.startswith("refused:")


def test_the_control_command_exits_1_for_equal_waveforms(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base")
    code, text = run_cli("control", str(tmp_path / "ft"), str(tmp_path / "base"))
    assert code == 1 and "same weights" in text


def test_the_verify_command_re_hashes(tmp_path):
    root = make_run(tmp_path)
    assert run_cli("verify", str(root))[0] == 0
    np.save(root / "arrays" / "layer__0.npy", np.zeros((1, 20, 16), dtype=np.float32))
    code, text = run_cli("verify", str(root))
    assert code == 2 and "sha256" in text


# ---- guards found by the first mutation table -----------------------------------------------------

def test_the_number_of_arrays_is_capped_on_write_and_on_read(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "MAX_ARTIFACTS", 2)
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.record("layer", "0", np.zeros(2))
    rec.record("layer", "1", np.zeros(2))
    with pytest.raises(ReferenceError, match="more than"):
        rec.record("layer", "2", np.zeros(2))
    monkeypatch.setattr(ar, "MAX_ARTIFACTS", 1000)
    root = make_run(tmp_path, "big", arrays={("layer", str(i)): np.zeros(2) for i in range(3)})
    monkeypatch.setattr(ar, "MAX_ARTIFACTS", 2)
    with pytest.raises(ReferenceError, match="at most"):
        load_run(root)


def test_a_manifest_with_another_schema_number_is_refused(tmp_path):
    root = make_run(tmp_path)
    edit_manifest(root, lambda m: m.update(schema=2))
    with pytest.raises(ReferenceError, match="schema 1"):
        load_run(root)


def test_a_malformed_sha_is_refused_even_without_verification(tmp_path):
    root = make_run(tmp_path)
    edit_manifest(root, lambda m: m["artifacts"][0].update(sha256="zz"))
    with pytest.raises(ReferenceError, match="malformed"):
        load_run(root, verify=False)


def test_a_shape_that_is_a_number_is_refused_as_malformed(tmp_path):
    root = make_run(tmp_path)
    edit_manifest(root, lambda m: m["artifacts"][0].update(shape=5))
    with pytest.raises(ReferenceError, match="malformed"):
        load_run(root, verify=False)


def test_a_link_to_another_file_inside_arrays_is_refused(tmp_path):
    import shutil
    root = make_run(tmp_path)
    target = root / "arrays" / "layer__0.npy"
    copy = root / "arrays" / "copy.npy"
    shutil.copy(target, copy)
    target.unlink()
    target.symlink_to(copy)
    with pytest.raises(ReferenceError, match="link"):
        load_run(root)


def test_an_object_array_is_never_unpickled(tmp_path):
    root = make_run(tmp_path)
    marker = tmp_path / "ran"

    class Evil:
        def __reduce__(self):
            return (marker.write_text, ("x",))
    obj = np.empty(1, dtype=object)
    obj[0] = Evil()
    np.save(root / "arrays" / "layer__0.npy", obj, allow_pickle=True)
    with pytest.raises(Exception):
        load_run(root, verify=False).array("layer", "0")
    assert not marker.exists()


def test_the_largest_absolute_error_is_reported():
    m = metrics(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0, 2.0, 3.5]))
    assert m["max_abs_error"] == pytest.approx(0.5)


def test_a_slope_below_one_fails_too():
    x = np.random.default_rng(1).normal(size=500)
    ok, why = verdict(metrics(x, 0.5 * x), Bars(0.99, 10.0, 0.1))
    assert not ok and len(why) == 1 and "slope" in why[0]


def test_floor_bars_take_the_worst_of_each_number_across_results():
    floor = {"comparable": True, "results": [
        {"kind": "layer", "name": "0", "comparable": True, "pcc": 0.999, "rel_l2": 0.02, "slope": 1.02},
        {"kind": "layer", "name": "1", "comparable": True, "pcc": 0.9995, "rel_l2": 0.05, "slope": 0.97},
        {"kind": "layer", "name": "2", "comparable": True, "pcc": 0.9999, "rel_l2": 0.01, "slope": 0.995}]}
    bar = ar.bars_from_floor(floor, margin=2.0)["layer"]
    assert bar.pcc == pytest.approx(1.0 - 2.0 * 0.001) and bar.rel_l2 == pytest.approx(0.1) and bar.slope_tol == pytest.approx(0.06)
    flipped = ar.bars_from_floor({"comparable": True, "results": floor["results"][::-1]}, margin=2.0)["layer"]
    assert flipped == bar


def test_the_spectrum_distance_matches_an_independent_calculation():
    x, y = tone(440), 2.0 * tone(440)
    w = np.hanning(1024)

    def spec(sig):
        frames = [np.abs(np.fft.rfft(sig[0, s:s + 1024] * w)) for s in range(0, sig.shape[-1] - 1024 + 1, 512)]
        return np.log1p(np.array(frames))
    expected = np.abs(spec(x) - spec(y)).mean() / spec(x).mean()
    assert ar.log_spectrum_distance(x, y) == pytest.approx(expected)


# ---- cross-precision comparison (review of PR 123) ------------------------------------------------

def test_runs_that_differ_only_in_dtype_are_refused_by_default_and_compared_when_asked(tmp_path):
    make_run(tmp_path, "ref")
    make_run(tmp_path, "low", settings={"dtype": "bfloat16"})
    ref, low = load_run(tmp_path / "ref"), load_run(tmp_path / "low")
    assert compare_runs(ref, low)["comparable"] is False
    report = compare_runs(ref, low, cross_precision=True)
    assert report["comparable"] is True and report["ok"] is True and report["cross_precision"] is True


def test_a_strict_comparison_says_it_is_not_cross_precision(tmp_path):
    make_run(tmp_path, "ref")
    make_run(tmp_path, "cand")
    assert compare_runs(load_run(tmp_path / "ref"), load_run(tmp_path / "cand"))["cross_precision"] is False


@pytest.mark.parametrize("key,value", [("seed", 43), ("steps", 9), ("prompt", "other"), ("duration", 11), ("sampler", "euler"),
                                       ("cfg_scale", 7.0), ("sample_size", 1)])
def test_cross_precision_still_needs_every_other_setting_to_match(tmp_path, key, value):
    make_run(tmp_path, "ref")
    make_run(tmp_path, "low", settings={"dtype": "bfloat16", key: value})
    report = compare_runs(load_run(tmp_path / "ref"), load_run(tmp_path / "low"), cross_precision=True)
    assert report["comparable"] is False and key in report["problems"][0] and report["cross_precision"] is True


def test_cross_precision_reports_the_numbers_that_feed_the_noise_floor(tmp_path):
    make_run(tmp_path, "ref")
    ref = load_run(tmp_path / "ref")
    rec = Recorder(tmp_path / "low", model=MODEL, library=LIB, settings={**SETTINGS, "dtype": "bfloat16"}, seeds={"torch": 42})
    rng = np.random.default_rng(5)
    for (kind, nm) in ref.entries:
        a = ref.array(kind, nm)
        rec.record(kind, nm, a + 0.01 * rng.normal(size=a.shape))
    rec.finish()
    report = compare_runs(ref, load_run(tmp_path / "low"), cross_precision=True)
    bars = ar.bars_from_floor(report, margin=3.0)
    assert set(bars) >= {"layer", "latent", "waveform"}


def test_the_check_command_takes_a_cross_precision_flag(tmp_path):
    make_run(tmp_path, "ref")
    make_run(tmp_path, "low", settings={"dtype": "bfloat16"})
    code, text = run_cli("check", str(tmp_path / "ref"), str(tmp_path / "low"))
    assert code == 1 and "dtype" in text
    code, text = run_cli("check", str(tmp_path / "ref"), str(tmp_path / "low"), "--cross-precision")
    assert code == 0 and text.strip().endswith("ok")
