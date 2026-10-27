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
    edit_manifest(root, lambda m: [e.update(shape=[3]) for e in m["artifacts"] if (e["kind"], e["name"]) == ("layer", "0")])
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
    with pytest.raises(ReferenceError, match="must be"):
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
    floor = compare_runs(load_run(tmp_path / "ref"), perturbed(tmp_path, "low", noise=0.01), cross_precision=True)
    bars = ar.bars_from_floor(floor, margin=3.0)
    layer = [r for r in floor["results"] if r["kind"] == "layer"][0]
    assert bars["layer"].rel_l2 == pytest.approx(3.0 * layer["rel_l2"])
    assert bars["layer"].pcc == pytest.approx(1.0 - 3.0 * (1.0 - layer["pcc"]))
    assert "three" in bars["layer"].source or "3.0 times" in bars["layer"].source
    assert set(bars) == {"conditioning", "layer", "step_latent", "latent", "waveform"}


def test_a_floor_run_that_could_not_be_compared_gives_no_bars(tmp_path):
    with pytest.raises(ReferenceError):
        ar.bars_from_floor({"comparable": False, "cross_precision": True, "results": []})
    with pytest.raises(ReferenceError):
        ar.bars_from_floor({"comparable": True, "cross_precision": True, "results": []})
    with pytest.raises(ReferenceError, match="not comparable"):
        ar.bars_from_floor({"comparable": True, "cross_precision": True, "results": [{"kind": "layer", "name": "0", "comparable": False}]})


def test_a_margin_below_one_is_refused():
    with pytest.raises(ReferenceError):
        ar.bars_from_floor({"comparable": True, "results": [{"kind": "layer", "name": "0", "comparable": True, "pcc": 1.0, "rel_l2": 0.0, "slope": 1.0}]}, margin=0.5)


# ---- the control ----------------------------------------------------------------------------------

def test_two_models_with_different_waveforms_pass_the_control(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base", arrays={("waveform", "audio"): np.random.default_rng(9).normal(size=(2, 4096))})
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
    floor = {"comparable": True, "cross_precision": True, "results": [
        {"kind": "layer", "name": "0", "comparable": True, "pcc": 0.999, "rel_l2": 0.02, "slope": 1.02},
        {"kind": "layer", "name": "1", "comparable": True, "pcc": 0.9995, "rel_l2": 0.05, "slope": 0.97},
        {"kind": "layer", "name": "2", "comparable": True, "pcc": 0.9999, "rel_l2": 0.01, "slope": 0.995}]}
    bar = ar.bars_from_floor(floor, margin=2.0)["layer"]
    assert bar.pcc == pytest.approx(1.0 - 2.0 * 0.001) and bar.rel_l2 == pytest.approx(0.1) and bar.slope_tol == pytest.approx(0.06)
    flipped = ar.bars_from_floor({"comparable": True, "cross_precision": True, "results": floor["results"][::-1]}, margin=2.0)["layer"]
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


# ---- review of PR 123, round three: concurrent runs, floor reports, reading what was verified -----------

def test_a_second_recorder_on_the_same_directory_is_refused_even_before_a_manifest_exists(tmp_path):
    Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    with pytest.raises(ReferenceError, match="never overwritten"):
        Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)


def test_an_existing_empty_directory_is_refused_too(tmp_path):
    (tmp_path / "r").mkdir()
    with pytest.raises(ReferenceError, match="never overwritten"):
        Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)


def test_an_interrupted_runs_files_are_left_alone(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.record("layer", "0", np.arange(3.0))
    with pytest.raises(ReferenceError):
        Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    assert (tmp_path / "r" / "arrays" / "layer__0.npy").is_file()


def test_exactly_one_of_many_concurrent_recorders_gets_the_directory(tmp_path):
    import threading
    results, barrier = [], threading.Barrier(8)

    def go():
        barrier.wait()
        try:
            Recorder(tmp_path / "race", model=MODEL, library=LIB, settings=SETTINGS)
            results.append("won")
        except ReferenceError:
            results.append("refused")
    threads = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count("won") == 1 and results.count("refused") == 7


def test_settings_are_checked_before_any_directory_is_made(tmp_path):
    with pytest.raises(ReferenceError):
        Recorder(tmp_path / "r", model={}, library=LIB, settings=SETTINGS)
    assert not (tmp_path / "r").exists()


def floor_report(tmp_path, noise_ok=True):
    make_run(tmp_path, "ref", arrays={("noise", "0"): np.arange(8.0), ("layer", "0"): np.arange(8.0)})
    make_run(tmp_path, "low", settings={"dtype": "bfloat16"},
             arrays={("noise", "0"): np.arange(8.0) + (0.0 if noise_ok else 0.5), ("layer", "0"): np.arange(8.0) + 0.01})
    return compare_runs(load_run(tmp_path / "ref"), load_run(tmp_path / "low"), cross_precision=True)


def test_a_floor_whose_noise_draws_differ_gives_no_bars(tmp_path):
    with pytest.raises(ReferenceError, match="noise"):
        ar.bars_from_floor(floor_report(tmp_path, noise_ok=False))


def test_noise_bars_are_never_loosened_by_a_floor(tmp_path):
    bars = ar.bars_from_floor(floor_report(tmp_path, noise_ok=True))
    assert "noise" not in bars and "layer" in bars
    assert ar.DEFAULT_BARS["noise"].rel_l2 == 1e-6


def test_a_floor_report_with_missing_arrays_gives_no_bars(tmp_path):
    report = floor_report(tmp_path)
    report["problems"] = ["layer/1 is only in the reference"]
    with pytest.raises(ReferenceError, match="problems"):
        ar.bars_from_floor(report)


@pytest.mark.parametrize("report", [None, [], "x", {"comparable": True, "results": "x"}, {"comparable": True, "results": [1]},
                                    {"comparable": True, "results": [{"kind": "layer"}]},
                                    {"comparable": True, "results": [{"kind": "layer", "name": "0", "comparable": True, "pcc": "x", "rel_l2": 0.1, "slope": 1.0}]},
                                    {"comparable": True, "results": [{"kind": "layer", "name": "0", "comparable": True, "pcc": 1.0, "rel_l2": float("nan"), "slope": 1.0}]},
                                    {"comparable": True, "results": [{"kind": "layer", "name": "0", "comparable": True, "pcc": 1.0, "rel_l2": 0.1, "slope": None}]},
                                    {"comparable": True, "results": [{"kind": "mystery", "name": "0", "comparable": True, "pcc": 1.0, "rel_l2": 0.1, "slope": 1.0}]}])
def test_a_malformed_floor_report_is_refused_without_an_exception_of_another_kind(report):
    with pytest.raises(ReferenceError):
        ar.bars_from_floor(report)


def test_a_floor_that_is_not_cross_precision_gives_no_bars(tmp_path):
    make_run(tmp_path, "ref")
    make_run(tmp_path, "cand")
    strict = compare_runs(load_run(tmp_path / "ref"), load_run(tmp_path / "cand"))
    with pytest.raises(ReferenceError, match="cross-precision"):
        ar.bars_from_floor(strict)


def test_an_array_changed_after_the_run_was_loaded_is_refused_when_it_is_read(tmp_path):
    root = make_run(tmp_path)
    run = load_run(root)
    np.save(root / "arrays" / "layer__0.npy", np.zeros((1, 20, 16), dtype=np.float32))
    with pytest.raises(ReferenceError, match="sha256"):
        run.array("layer", "0")


def test_an_array_replaced_by_a_link_after_the_run_was_loaded_is_refused_when_it_is_read(tmp_path):
    root = make_run(tmp_path)
    run = load_run(root)
    target = root / "arrays" / "layer__0.npy"
    copy = tmp_path / "copy.npy"
    copy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(copy)
    with pytest.raises(ReferenceError, match="link"):
        run.array("layer", "0")


def test_a_manifest_file_name_must_be_the_one_the_kind_and_name_give(tmp_path):
    import shutil
    root = make_run(tmp_path)
    shutil.copy(root / "arrays" / "layer__0.npy", root / "arrays" / "other.npy")
    edit_manifest(root, lambda m: m["artifacts"][0].update(file="arrays/other.npy"))
    with pytest.raises(ReferenceError, match="must be"):
        load_run(root, verify=False)


def test_reading_an_array_without_verification_at_load_still_checks_it_at_read(tmp_path):
    root = make_run(tmp_path)
    run = load_run(root, verify=False)
    np.save(root / "arrays" / "layer__0.npy", np.zeros((1, 20, 16), dtype=np.float32))
    with pytest.raises(ReferenceError, match="sha256"):
        run.array("layer", "0")


def good_result(**change):
    return {"kind": "layer", "name": "0", "comparable": True, "pcc": 0.999, "rel_l2": 0.05, "slope": 1.01, **change}


def floor_of(*results, **change):
    return {"comparable": True, "cross_precision": True, "problems": [], "results": list(results), **change}


@pytest.mark.parametrize("report,match", [
    (None, "must be a comparison report"), ([], "must be a comparison report"), ("x", "must be a comparison report"),
    (floor_of(good_result(), comparable=False) | {"comparable": False}, "not comparable or lists problems"),
    (floor_of(good_result()) | {"results": "x"}, "no results"),
    (floor_of(good_result()) | {"results": []}, "no results"),
    (floor_of(1), "malformed"), (floor_of({"kind": "layer"}), "malformed"),
    (floor_of({**good_result(), "kind": "mystery"}), "malformed"),
    (floor_of({**good_result(), "name": 3}), "malformed"),
    (floor_of(good_result(comparable=False)), "not comparable in the floor run"),
    (floor_of({k: v for k, v in good_result().items() if k != "comparable"}), "not comparable in the floor run"),
    (floor_of(good_result(pcc="x")), "not comparable in the floor run"),
    (floor_of(good_result(pcc=True)), "not comparable in the floor run"),
    (floor_of(good_result(rel_l2=float("nan"))), "not comparable in the floor run"),
    (floor_of(good_result(slope=float("inf"))), "not comparable in the floor run"),
    (floor_of(good_result(slope=None)), "not comparable in the floor run"),
    (floor_of(good_result()) | {"cross_precision": False}, "not a cross-precision"),
    (floor_of(good_result()) | {"problems": ["x"]}, "not comparable or lists problems"),
])
def test_each_kind_of_bad_floor_report_is_refused_for_its_own_reason(report, match):
    with pytest.raises(ReferenceError, match=match):
        ar.bars_from_floor(report)


@pytest.mark.parametrize("margin", [0.5, float("nan"), True, "3", None])
def test_a_bad_margin_is_refused(margin):
    with pytest.raises(ReferenceError, match="margin"):
        ar.bars_from_floor(floor_of(good_result()), margin=margin)


def test_a_failed_noise_result_is_refused_and_a_passing_one_is_skipped():
    noise_ok = {"kind": "noise", "name": "0", "comparable": True, "pcc": 1.0, "rel_l2": 0.0, "slope": 1.0, "ok": True}
    assert set(ar.bars_from_floor(floor_of(good_result(), noise_ok))) == {"layer"}
    with pytest.raises(ReferenceError, match="noise"):
        ar.bars_from_floor(floor_of(good_result(), {**noise_ok, "ok": False}))
    with pytest.raises(ReferenceError, match="noise"):
        ar.bars_from_floor(floor_of(good_result(), {k: v for k, v in noise_ok.items() if k != "ok"}))


def test_the_bytes_that_were_hashed_are_the_bytes_that_are_parsed(tmp_path, monkeypatch):
    run = load_run(make_run(tmp_path))
    seen = []
    real = np.load
    monkeypatch.setattr(ar.np, "load", lambda f, **kw: (seen.append(type(f).__name__), real(f, **kw))[1])
    run.array("layer", "0")
    assert seen == ["BytesIO"]


def test_an_array_over_the_cap_is_refused_before_it_is_hashed_or_parsed(tmp_path, monkeypatch):
    run = load_run(make_run(tmp_path))
    monkeypatch.setattr(ar, "MAX_ARRAY_BYTES", 10)
    monkeypatch.setattr(ar.np, "load", lambda *a, **k: pytest.fail("parsed an oversized array"))
    with pytest.raises(ReferenceError, match="larger than 10 bytes"):
        run.array("layer", "0")


def test_an_array_exactly_at_the_cap_is_read(tmp_path, monkeypatch):
    root = make_run(tmp_path, arrays={("layer", "0"): np.arange(4.0)})
    size = (root / "arrays" / "layer__0.npy").stat().st_size
    run = load_run(root)
    monkeypatch.setattr(ar, "MAX_ARRAY_BYTES", size)
    assert run.array("layer", "0").shape == (4,)
    monkeypatch.setattr(ar, "MAX_ARRAY_BYTES", size - 1)
    with pytest.raises(ReferenceError, match="larger"):
        run.array("layer", "0")


def test_a_missing_array_file_is_a_reference_error_at_read(tmp_path):
    root = make_run(tmp_path)
    run = load_run(root)
    (root / "arrays" / "layer__0.npy").unlink()
    with pytest.raises(ReferenceError, match="is missing"):
        run.array("layer", "0")


# ---- round four: payloads that do not decode, size limits before hashing, strict manifest JSON --------------

def replace_payload(root, kind, name, data: bytes):
    """Overwrite an array file with `data` and make the manifest's sha256 match, so only decoding can notice."""
    import hashlib
    path = root / "arrays" / f"{kind}__{name}.npy"
    path.write_bytes(data)
    edit_manifest(root, lambda m: [e.update(sha256=hashlib.sha256(data).hexdigest()) for e in m["artifacts"] if (e["kind"], e["name"]) == (kind, name)])


@pytest.mark.parametrize("payload", [b"not numpy at all", b"", b"\x93NUMPY\x01\x00", b"\x93NUMPY\x01\x00\x05\x00{bad}\n" + b"\0" * 8, b"PK\x03\x04" + b"\0" * 20])
def test_a_payload_that_does_not_decode_is_a_reference_error_even_with_a_matching_digest(tmp_path, payload):
    root = make_run(tmp_path)
    replace_payload(root, "layer", "0", payload)
    run = load_run(root)
    with pytest.raises(ReferenceError, match="not a readable array"):
        run.array("layer", "0")


def test_a_renamed_npz_archive_is_refused_instead_of_returning_a_non_array(tmp_path):
    import io as _io
    root = make_run(tmp_path)
    buf = _io.BytesIO()
    np.savez(buf, a=np.zeros(2))
    replace_payload(root, "layer", "0", buf.getvalue())
    with pytest.raises(ReferenceError, match="not a readable array|not an array"):
        load_run(root).array("layer", "0")


def test_an_object_array_with_a_matching_digest_is_refused_without_unpickling(tmp_path):
    root = make_run(tmp_path)
    marker = tmp_path / "ran"

    class Evil:
        def __reduce__(self):
            return (marker.write_text, ("x",))
    obj = np.empty(1, dtype=object)
    obj[0] = Evil()
    import io as _io
    buf = _io.BytesIO()
    np.save(buf, obj, allow_pickle=True)
    replace_payload(root, "layer", "0", buf.getvalue())
    with pytest.raises(ReferenceError):
        load_run(root).array("layer", "0")
    assert not marker.exists()


def test_the_cli_reports_an_undecodable_array_and_does_not_raise(tmp_path):
    make_run(tmp_path, "ref")
    root = make_run(tmp_path, "cand")
    replace_payload(root, "layer", "0", b"garbage")
    code, text = run_cli("check", str(tmp_path / "ref"), str(root))
    assert code == 2 and text.startswith("refused:") and "not a readable array" in text


def test_the_control_command_reports_an_undecodable_waveform(tmp_path):
    make_run(tmp_path, "ft")
    root = make_run(tmp_path, "base")
    replace_payload(root, "waveform", "audio", b"garbage")
    code, text = run_cli("control", str(tmp_path / "ft"), str(root))
    assert code == 2 and text.startswith("refused:")


def test_an_oversized_array_is_refused_before_it_is_hashed_when_a_run_is_loaded(tmp_path, monkeypatch):
    root = make_run(tmp_path)
    monkeypatch.setattr(ar, "MAX_ARRAY_BYTES", 100)
    monkeypatch.setattr(ar.hashlib, "sha256", lambda *a, **k: pytest.fail("hashed an oversized array"))
    with pytest.raises(ReferenceError, match="larger than 100 bytes"):
        load_run(root)


def test_the_verify_command_refuses_an_oversized_array_without_reading_it_to_the_end(tmp_path, monkeypatch):
    root = make_run(tmp_path)
    monkeypatch.setattr(ar, "MAX_ARRAY_BYTES", 100)
    code, text = run_cli("verify", str(root))
    assert code == 2 and "larger than 100 bytes" in text


def test_a_symlinked_array_is_refused_by_the_hashing_pass_too(tmp_path):
    root = make_run(tmp_path)
    target = root / "arrays" / "layer__0.npy"
    copy = root / "arrays" / "copy.npy"
    copy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(copy)
    with pytest.raises(ReferenceError, match="link"):
        load_run(root)


def test_the_recorder_refuses_an_array_the_loader_would_refuse(tmp_path, monkeypatch):
    monkeypatch.setattr(ar, "MAX_ARRAY_BYTES", 1000)
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.record("layer", "small", np.zeros(4))
    with pytest.raises(ReferenceError, match="larger than"):
        rec.record("layer", "big", np.zeros(1000))
    assert not (tmp_path / "r" / "arrays" / "layer__big.npy").exists()


@pytest.mark.parametrize("text", ['{"schema": 1, "schema": 1, "model": {"a": 1}, "library": {"a": 1}, "settings": {}, "artifacts": []}',
                                  '{"schema": 1, "model": {"a": NaN}, "library": {"a": 1}, "settings": {}, "artifacts": []}',
                                  '{"schema": 1, "model": {"a": Infinity}, "library": {"a": 1}, "settings": {}, "artifacts": []}'])
def test_a_manifest_with_a_repeated_key_or_a_non_finite_number_is_refused(tmp_path, text):
    root = make_run(tmp_path)
    (root / "manifest.json").write_text(text)
    with pytest.raises(ReferenceError, match="not JSON"):
        load_run(root)


def test_a_manifest_nested_absurdly_deep_is_refused(tmp_path):
    root = make_run(tmp_path)
    (root / "manifest.json").write_text('{"schema": 1, "x": ' + "[" * 100000)
    with pytest.raises(ReferenceError):
        load_run(root)


def test_a_manifest_artifact_list_with_a_repeated_key_in_an_entry_is_refused(tmp_path):
    root = make_run(tmp_path)
    text = (root / "manifest.json").read_text()
    assert '"kind": "conditioning",' in text
    (root / "manifest.json").write_text(text.replace('"kind": "conditioning",', '"kind": "conditioning", "kind": "conditioning",', 1))
    with pytest.raises(ReferenceError, match="not JSON"):
        load_run(root)


# ---- boundaries the first mutation table found unguarded (reference) ------------------------------------------

def test_the_recorder_accepts_a_file_exactly_at_the_cap_and_refuses_one_byte_more(tmp_path, monkeypatch):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.record("layer", "probe", np.zeros(4))
    size = (tmp_path / "r" / "arrays" / "layer__probe.npy").stat().st_size
    monkeypatch.setattr(ar, "MAX_ARRAY_BYTES", size)
    rec.record("layer", "exact", np.zeros(4))
    monkeypatch.setattr(ar, "MAX_ARRAY_BYTES", size - 1)
    with pytest.raises(ReferenceError, match="larger than"):
        rec.record("layer", "over", np.zeros(4))


def manifest_with(root, extra_json):
    (root / "manifest.json").write_text('{"schema": 1, "model": {"a": 1}, "library": {"a": 1}, "settings": {}, "artifacts": [], "x": ' + extra_json + "}")


@pytest.mark.parametrize("extra,ok", [("1e999", False), ("-1e999", False), ("9" * 31, False), ("-" + "9" * 31, False), ("9" * 30, True),
                                      ("-" + "9" * 30, True), ("1.5", True)])
def test_manifest_numbers_must_be_finite_and_not_huge(tmp_path, extra, ok):
    root = make_run(tmp_path)
    manifest_with(root, extra)
    if ok:
        with pytest.raises(ReferenceError, match="settings are missing|artifacts"):
            load_run(root)                     # parsed fine; refused later for its own reasons
    else:
        with pytest.raises(ReferenceError, match="not JSON"):
            load_run(root)


@pytest.mark.parametrize("opening,closing", [("[", "]"), ('{"a": ', "}")])
def test_manifest_nesting_is_limited_whether_the_containers_are_lists_or_objects(tmp_path, opening, closing):
    root = make_run(tmp_path)
    for levels, ok in ((31, True), (32, False)):          # the manifest object is level 1, so 31 inner levels make 32
        manifest_with(root, opening * levels + "1" + closing * levels)
        if ok:
            with pytest.raises(ReferenceError, match="settings are missing|artifacts"):
                load_run(root)
        else:
            with pytest.raises(ReferenceError, match="not JSON"):
                load_run(root)


# ---- round five: reads anchored to descriptors, a manifest that is never replaced, a control with matched settings ------

def test_a_symlinked_arrays_directory_is_refused(tmp_path):
    root = make_run(tmp_path)
    real = tmp_path / "real-arrays"
    (root / "arrays").rename(real)
    (root / "arrays").symlink_to(real)
    with pytest.raises(ReferenceError, match="link"):
        load_run(root)


def test_a_missing_arrays_directory_is_refused(tmp_path):
    import shutil
    root = make_run(tmp_path)
    shutil.rmtree(root / "arrays")
    with pytest.raises(ReferenceError, match="missing"):
        load_run(root)


def test_reads_stay_in_the_directory_that_was_opened_even_if_arrays_is_replaced_afterwards(tmp_path):
    root = make_run(tmp_path)
    run = load_run(root)
    evil = tmp_path / "evil-arrays"
    evil.mkdir()
    np.save(evil / "layer__0.npy", np.zeros((1, 20, 16), dtype=np.float32))
    (root / "arrays").rename(tmp_path / "moved-arrays")
    (root / "arrays").symlink_to(evil)
    assert run.array("layer", "0").any()                 # the original, not the zeros the link now leads to


def test_reads_stay_in_the_run_directory_that_was_opened_even_if_it_is_replaced_afterwards(tmp_path):
    root = make_run(tmp_path)
    run = load_run(root)
    root.rename(tmp_path / "moved-run")
    make_run(tmp_path, "run", arrays={("layer", "0"): np.zeros(3)})
    assert run.array("layer", "0").shape == (1, 20, 16)


def test_a_run_can_be_closed_and_then_refuses_reads(tmp_path):
    run = load_run(make_run(tmp_path))
    run.array("layer", "0")
    run.close()
    run.close()                                             # closing twice is fine
    with pytest.raises(ReferenceError, match="closed"):
        run.array("layer", "0")


def test_a_run_works_as_a_context_manager(tmp_path):
    with load_run(make_run(tmp_path)) as run:
        assert run.array("layer", "0").shape == (1, 20, 16)
    with pytest.raises(ReferenceError, match="closed"):
        run.array("layer", "0")


def test_a_run_directory_reached_through_a_link_is_read_as_given(tmp_path):
    root = make_run(tmp_path)
    link = tmp_path / "link-to-run"
    link.symlink_to(root)
    assert load_run(link).array("layer", "0").shape == (1, 20, 16)


def test_finishing_twice_never_replaces_the_manifest(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.record("layer", "0", np.arange(3.0))
    rec.finish()
    before = (tmp_path / "r" / "manifest.json").read_bytes()
    rec.record("layer", "1", np.arange(3.0)) if False else None
    with pytest.raises(ReferenceError, match="already exists"):
        rec.finish()
    assert (tmp_path / "r" / "manifest.json").read_bytes() == before
    assert not list((tmp_path / "r").glob("*.tmp")) and not [p for p in (tmp_path / "r").iterdir() if p.name.endswith(".tmp")]


def test_a_manifest_that_appears_after_the_recorder_was_made_is_not_replaced(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.record("layer", "0", np.arange(3.0))
    (tmp_path / "r" / "manifest.json").write_text("someone else's")
    with pytest.raises(ReferenceError, match="already exists"):
        rec.finish()
    assert (tmp_path / "r" / "manifest.json").read_text() == "someone else's"


def test_a_manifest_that_is_a_dangling_link_is_not_followed_or_replaced(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    (tmp_path / "r" / "manifest.json").symlink_to(tmp_path / "target")
    with pytest.raises(ReferenceError):
        rec.finish()
    assert not (tmp_path / "target").exists()


def test_an_array_file_that_appears_before_it_is_published_is_not_replaced(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    (tmp_path / "r" / "arrays" / "layer__0.npy").write_bytes(b"someone else's")
    with pytest.raises(ReferenceError, match="already exists"):
        rec.record("layer", "0", np.arange(3.0))
    assert (tmp_path / "r" / "arrays" / "layer__0.npy").read_bytes() == b"someone else's"
    assert not [p for p in (tmp_path / "r" / "arrays").iterdir() if p.name.endswith(".tmp")]


def test_an_array_name_that_is_a_dangling_link_is_not_written_through(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    (tmp_path / "r" / "arrays" / "layer__0.npy").symlink_to(tmp_path / "target.npy")
    with pytest.raises(ReferenceError):
        rec.record("layer", "0", np.arange(3.0))
    assert not (tmp_path / "target.npy").exists()


def test_the_recorder_writes_into_the_directory_it_created_even_if_the_path_is_replaced(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    (tmp_path / "r").rename(tmp_path / "moved")
    evil = tmp_path / "evil"
    (evil / "arrays").mkdir(parents=True)
    (tmp_path / "r").symlink_to(evil)
    rec.record("layer", "0", np.arange(3.0))
    rec.finish()
    assert (tmp_path / "moved" / "manifest.json").is_file() and not list(evil.rglob("*.npy")) and not (evil / "manifest.json").exists()


def test_the_recorded_sha256_is_the_hash_of_the_published_file(tmp_path):
    import hashlib
    root = make_run(tmp_path)
    for e in json.loads((root / "manifest.json").read_text())["artifacts"]:
        assert e["sha256"] == hashlib.sha256((root / e["file"]).read_bytes()).hexdigest()


# ---- the control needs matched settings --------------------------------------------------------------------

def test_the_control_needs_every_match_setting_equal_by_default(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base", arrays={("waveform", "audio"): np.random.default_rng(9).normal(size=(2, 4096))},
             settings={"steps": 50, "sampler": "euler", "cfg_scale": 7.0})
    report = compare_control(load_run(tmp_path / "ft"), load_run(tmp_path / "base"))
    assert report["ok"] is False and report["comparable"] is False and report["matched"] is False
    assert any("steps" in p for p in report["problems"]) and any("matched" in p for p in report["problems"])


def test_a_matched_control_with_different_weights_passes(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base", arrays={("waveform", "audio"): np.random.default_rng(9).normal(size=(2, 4096))})
    report = compare_control(load_run(tmp_path / "ft"), load_run(tmp_path / "base"))
    assert report["ok"] and report["matched"] is True and report["rel_l2"] > 0.05


def test_a_matched_control_with_equal_waveforms_means_the_same_weights_may_have_run_twice(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base")
    report = compare_control(load_run(tmp_path / "ft"), load_run(tmp_path / "base"))
    assert report["ok"] is False and report["matched"] is True and "same weights" in report["problems"][0]


def test_an_unmatched_control_is_available_on_request_and_says_what_it_cannot_show(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base", arrays={("waveform", "audio"): np.random.default_rng(9).normal(size=(2, 4096))}, settings={"steps": 50})
    report = compare_control(load_run(tmp_path / "ft"), load_run(tmp_path / "base"), require_matched=False)
    assert report["ok"] is True and report["matched"] is False and "cannot show" in report["note"]


def test_an_unmatched_control_still_needs_the_same_prompt_duration_and_seed(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base", settings={"seed": 1})
    report = compare_control(load_run(tmp_path / "ft"), load_run(tmp_path / "base"), require_matched=False)
    assert report["ok"] is False and "seed" in report["problems"][0]


def test_the_control_command_defaults_to_matched_and_has_an_unmatched_flag(tmp_path):
    make_run(tmp_path, "ft")
    make_run(tmp_path, "base", arrays={("waveform", "audio"): np.random.default_rng(9).normal(size=(2, 4096))}, settings={"steps": 50})
    code, text = run_cli("control", str(tmp_path / "ft"), str(tmp_path / "base"))
    assert code == 1 and "matched" in text
    code, text = run_cli("control", str(tmp_path / "ft"), str(tmp_path / "base"), "--unmatched")
    assert code == 0 and "cannot show" in text


# ---- descriptor anchoring: the recorder's setup, temporary files and cleanup ---------------------------

def _open_fds():
    import os
    return {int(n) for n in os.listdir("/proc/self/fd")}


def test_finishing_twice_leaves_the_working_directory_alone_and_says_so(tmp_path, monkeypatch):
    """A recorder whose descriptors are closed must refuse, not fall back to the working directory."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.record("layer", "0", np.arange(3.0))
    rec.finish()
    monkeypatch.chdir(cwd)
    with pytest.raises(ReferenceError, match="already exists"):
        rec.finish()
    assert list(cwd.iterdir()) == []


def test_recording_after_finish_is_refused_with_its_own_message(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.finish()
    with pytest.raises(ReferenceError, match="recorder is finished"):
        rec.record("layer", "0", np.arange(3.0))
    assert not [p for p in tmp_path.iterdir() if p.name != "r"]


def test_finishing_closes_both_descriptors(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    fds = (rec._root_fd, rec._arrays_fd)
    assert all(fd in _open_fds() for fd in fds)
    rec.finish()
    import os
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert rec._root_fd is None and rec._arrays_fd is None


def test_a_run_directory_swapped_for_a_link_after_it_was_opened_does_not_redirect_the_recorder(tmp_path, monkeypatch):
    """The recorder opens the run directory once. Replacing the path with a link to another directory afterwards changes nothing."""
    import os
    other = tmp_path / "other"
    other.mkdir()
    real_open = ar.pf.open_dir
    calls = []

    def swapping(path, **kw):
        fd = real_open(path, **kw)
        if not calls:
            calls.append(1)
            os.rename(tmp_path / "r", tmp_path / "moved")
            os.symlink(other, tmp_path / "r")
        return fd
    monkeypatch.setattr(ar.pf, "open_dir", swapping)
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.record("layer", "0", np.arange(3.0))
    rec.finish()
    assert (tmp_path / "moved" / "arrays" / "layer__0.npy").is_file() and (tmp_path / "moved" / "manifest.json").is_file()
    assert list(other.iterdir()) == []


def test_a_run_directory_that_is_a_link_when_it_is_opened_is_refused(tmp_path, monkeypatch):
    """The directory was just created, so the only way it is a link is a swap between the mkdir and the open."""
    import os
    other = tmp_path / "other"
    other.mkdir()
    real_mkdir = ar.Path.mkdir

    def mkdir_then_swap(self, *a, **kw):
        real_mkdir(self, *a, **kw)
        if self.name == "r":
            os.rmdir(self)
            os.symlink(other, self)
    monkeypatch.setattr(ar.Path, "mkdir", mkdir_then_swap)
    with pytest.raises(ReferenceError, match="cannot set up.*is a link"):
        Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    assert list(other.iterdir()) == []


def test_an_arrays_directory_that_is_a_link_when_it_is_opened_is_refused_and_the_recorder_closes_its_descriptors(tmp_path, monkeypatch):
    import os
    other = tmp_path / "other"
    other.mkdir()
    real_mkdir = os.mkdir
    before = _open_fds()

    def mkdir_then_swap(path, *a, **kw):
        real_mkdir(path, *a, **kw)
        if path == "arrays":
            os.rmdir(path, dir_fd=kw["dir_fd"])
            os.symlink(other, path, dir_fd=kw["dir_fd"])
    monkeypatch.setattr(ar.os, "mkdir", mkdir_then_swap)
    with pytest.raises(ReferenceError, match="cannot set up.*arrays is a link"):
        Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    assert _open_fds() <= before


def test_a_temporary_array_name_that_is_already_a_link_is_not_followed(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    target = tmp_path / "victim"
    target.write_bytes(b"keep me")
    (tmp_path / "r" / "arrays" / ".layer__0.npy.tmp").symlink_to(target)
    with pytest.raises(ReferenceError, match="already exists"):
        rec.record("layer", "0", np.arange(3.0))
    assert target.read_bytes() == b"keep me"


def test_a_temporary_manifest_name_that_is_already_a_link_is_not_followed(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    target = tmp_path / "victim"
    target.write_bytes(b"keep me")
    (tmp_path / "r" / ".manifest.json.tmp").symlink_to(target)
    with pytest.raises(ReferenceError, match="already exists"):
        rec.finish()
    assert target.read_bytes() == b"keep me"


def test_a_failed_finish_removes_its_temporary_manifest(tmp_path):
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    (tmp_path / "r" / "manifest.json").write_text("someone else's")
    with pytest.raises(ReferenceError):
        rec.finish()
    assert sorted(p.name for p in (tmp_path / "r").iterdir()) == ["arrays", "manifest.json"]


def test_every_array_and_the_manifest_is_flushed_to_disk(tmp_path, monkeypatch):
    import os
    synced = []
    real = os.fsync
    monkeypatch.setattr(ar.os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.record("layer", "0", np.arange(3.0))
    n_array = len(synced)
    rec.finish()
    assert n_array >= 1 and len(synced) > n_array


def test_a_failed_load_closes_the_descriptors_it_opened(tmp_path):
    root = make_run(tmp_path)
    m = json.loads((root / "manifest.json").read_text())
    m["artifacts"][0]["sha256"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(m))
    before = _open_fds()
    with pytest.raises(ReferenceError, match="sha256"):
        load_run(root)
    assert _open_fds() <= before


# ---- review of PR 142 -----------------------------------------------------------------------------

def test_a_recorder_closed_before_finish_publishes_nothing_anywhere(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    rec = Recorder(tmp_path / "r", model=MODEL, library=LIB, settings=SETTINGS)
    rec.close()
    monkeypatch.chdir(cwd)
    with pytest.raises(ReferenceError, match="closed"):
        rec.finish()
    assert list(cwd.iterdir()) == [] and not (tmp_path / "r" / "manifest.json").exists()


def test_a_run_loaded_without_verification_still_refuses_missing_linked_and_non_regular_arrays(tmp_path):
    import os
    root = make_run(tmp_path)
    arr = root / "arrays" / "layer__0.npy"
    saved = arr.read_bytes()
    arr.unlink()
    with pytest.raises(ReferenceError, match="layer__0.npy"):
        load_run(root, verify=False)
    other = tmp_path / "elsewhere.npy"
    other.write_bytes(saved)
    arr.symlink_to(other)
    with pytest.raises(ReferenceError, match="is a link"):
        load_run(root, verify=False)
    arr.unlink()
    os.mkfifo(arr)
    with pytest.raises(ReferenceError, match="not a regular file"):
        load_run(root, verify=False)
    arr.unlink()
    arr.mkdir()
    with pytest.raises(ReferenceError, match="not a regular file"):
        load_run(root, verify=False)
    arr.rmdir()
    arr.write_bytes(saved)
    assert load_run(root, verify=False).array("layer", "0").shape == (1, 20, 16)


def test_loading_without_verification_does_not_hash_the_arrays(tmp_path, monkeypatch):
    root = make_run(tmp_path)
    calls = []
    real = ar.pf.sha256_regular
    monkeypatch.setattr(ar.pf, "sha256_regular", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    load_run(root, verify=False)
    assert calls == []
    load_run(root)
    assert calls
