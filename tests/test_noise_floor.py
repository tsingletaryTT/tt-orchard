# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/noise_floor.py: agreement bars from the spread between CPU environments, and a known-bad control that shows whether the bars can tell a
good run from a damaged one. Runs are built from small random arrays in a temporary directory: the reference, a repeat of it, the same code in
a second environment (a fixed relative error on each array), a good candidate and damaged candidates. No model and no chip."""
import io
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

from orchard import audio_reference as ar
from orchard import noise_floor as nf

SETTINGS = {"prompt": "a piano", "duration": 10, "steps": 8, "cfg_scale": 1.0, "sampler": "pingpong", "seed": 42,
            "dtype": "float32", "sample_size": 705600}
MODEL = {"id": "org/model", "revision": "a" * 40}
LIB = {"name": "lib", "commit": "b" * 40}
LAYERS = [f"layer_{i:02d}" for i in range(4)]


def noisy(arr, rel, seed):
    """`arr` plus an error whose L2 norm is exactly `rel` times the norm of `arr`."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(arr.shape)
    return arr + d * (rel * np.linalg.norm(arr) / np.linalg.norm(d))


def base_arrays():
    rng = np.random.default_rng(0)
    out = {("noise", "initial"): rng.normal(size=(1, 8, 12)), ("conditioning", "prompt"): rng.normal(size=(1, 6, 8)),
           ("latent", "final"): rng.normal(size=(1, 8, 12)), ("step_latent", "0"): rng.normal(size=(1, 8, 12)),
           ("layer", "dit_output"): rng.normal(size=(1, 8, 12))}
    for n in LAYERS:
        out[("layer", n)] = rng.normal(size=(1, 30, 8)) * 5
    return out


def write(path, arrays, dtype="float32", model=None, library=None, **settings):
    rec = ar.Recorder(path, model=model or MODEL, library=library or LIB, settings={**SETTINGS, "dtype": dtype, **settings}, seeds={"torch": 42})
    for (kind, name), a in arrays.items():
        rec.record(kind, name, a)
    rec.finish()
    return ar.load_run(path)


def variant(arrays, rel_by_kind, seed):
    """The arrays with a relative error per kind; noise arrays are always unchanged."""
    out = {}
    for i, ((kind, name), a) in enumerate(arrays.items()):
        rel = 0.0 if kind == "noise" else rel_by_kind.get(kind, rel_by_kind.get("*", 0.0))
        out[(kind, name)] = noisy(a, rel, seed * 1000 + i) if rel else a
    return out


@pytest.fixture
def world(tmp_path):
    base = base_arrays()
    env = {"layer": 0.01, "latent": 0.05, "conditioning": 0.02, "step_latent": 0.04}
    good = {"layer": 0.02, "latent": 0.08, "conditioning": 0.02, "step_latent": 0.06}
    w = type("W", (), {})()
    w.tmp, w.base = tmp_path, base
    w.ref = write(tmp_path / "ref", base)
    w.repeat = write(tmp_path / "repeat", base)
    w.env_b = write(tmp_path / "env_b", variant(base, env, 1))
    # the same error direction as env_b at a larger size, so every number scales in a known way
    w.good = write(tmp_path / "good", variant(base, good, 1), dtype="bfloat16")
    return w


# ---- measuring the spread ---------------------------------------------------------------------------------------

def test_the_spread_inside_one_environment_and_between_environments_are_kept_apart(world):
    f = nf.measure_floor([world.ref, world.repeat], [world.env_b])
    a = f["arrays"]["layer/layer_01"]
    assert a["within_environment"]["rel_l2"] == 0.0
    assert a["between_environments"]["rel_l2"] == pytest.approx(0.01, rel=1e-6)
    assert a["floor"]["rel_l2"] == pytest.approx(0.01, rel=1e-6)
    assert f["kinds"]["latent"]["worst"]["rel_l2"] == pytest.approx(0.05, rel=1e-6)
    assert f["kinds"]["layer"]["worst"]["rel_l2"] == pytest.approx(0.01, rel=1e-6)
    assert f["runs"]["reference_environment"] and f["runs"]["chip_environment"]


def test_two_runs_that_differ_inside_one_environment_show_in_the_within_spread(world):
    jitter = write(world.tmp / "jit", variant(world.base, {"layer": 0.003}, 5))
    f = nf.measure_floor([world.ref, jitter], [world.env_b])
    assert f["arrays"]["layer/layer_00"]["within_environment"]["rel_l2"] == pytest.approx(0.003, rel=1e-6)


def test_the_floor_is_the_worst_pair_not_the_best(world):
    far = write(world.tmp / "far", variant(world.base, {"layer": 0.04}, 9))
    f = nf.measure_floor([world.ref, world.repeat], [world.env_b, far])
    assert f["arrays"]["layer/layer_00"]["floor"]["rel_l2"] >= 0.04 - 1e-9


def test_a_single_run_gives_no_floor(world):
    with pytest.raises(nf.NoiseFloorError, match="two runs"):
        nf.measure_floor([world.ref], [])


def test_runs_with_different_settings_are_refused(world):
    other = write(world.tmp / "o", world.base, seed=1)
    with pytest.raises(nf.NoiseFloorError, match="seed"):
        nf.measure_floor([world.ref], [other])


def test_runs_of_different_dtype_are_refused_as_a_floor(world):
    bf = write(world.tmp / "bf", world.base, dtype="bfloat16")
    with pytest.raises(nf.NoiseFloorError, match="dtype"):
        nf.measure_floor([world.ref], [bf])


def test_different_noise_draws_are_refused_because_the_floor_would_say_nothing(world):
    arrays = dict(world.base)
    arrays[("noise", "initial")] = arrays[("noise", "initial")] + 0.5
    other = write(world.tmp / "n", arrays)
    with pytest.raises(nf.NoiseFloorError, match="noise"):
        nf.measure_floor([world.ref], [other])


def test_an_array_missing_from_one_run_is_listed_and_left_out_of_the_floor(world):
    arrays = {k: v for k, v in world.base.items() if k != ("layer", "layer_02")}
    other = write(world.tmp / "m", variant(arrays, {"*": 0.01}, 3))
    f = nf.measure_floor([world.ref], [other])
    assert "layer/layer_02" not in f["arrays"] and any("layer/layer_02" in p for p in f["problems"])


# ---- bars from the floor ---------------------------------------------------------------------------------------

def test_bars_are_the_margin_times_the_worst_floor_and_carry_both(world):
    f = nf.measure_floor([world.ref, world.repeat], [world.env_b])
    bars = nf.derive_bars(f, margin=3.0)
    assert bars["latent"].rel_l2 == pytest.approx(3.0 * 0.05, rel=1e-6)
    assert bars["latent"].pcc == pytest.approx(1 - 3.0 ** 2 * (1 - f["kinds"]["latent"]["worst"]["pcc"]))   # the shortfall of PCC is quadratic in the error
    assert "3.0 times" in bars["latent"].source and "measured" in bars["latent"].source
    assert "noise" in bars and bars["noise"].rel_l2 <= 1e-6      # draws must come back unchanged


def test_a_margin_below_one_is_refused(world):
    f = nf.measure_floor([world.ref, world.repeat], [world.env_b])
    for bad in (0.5, float("nan"), True, "3"):
        with pytest.raises(nf.NoiseFloorError):
            nf.derive_bars(f, margin=bad)


def test_a_floor_of_zero_gives_a_small_bar_not_a_zero_bar(world):
    f = nf.measure_floor([world.ref], [world.repeat])      # bit-identical: the floor is zero everywhere
    bars = nf.derive_bars(f, margin=3.0)
    assert bars["layer"].rel_l2 == pytest.approx(3.0 * nf.MIN_FLOOR)
    assert bars["layer"].slope_tol > 0


def test_family_grouping_gives_each_kind_of_name_its_own_bar(world):
    f = nf.measure_floor([world.ref, world.repeat], [world.env_b], group="family")
    bars = nf.derive_bars(f, margin=3.0)
    assert "layer/layer_#" in bars and "layer/dit_output" in bars


def test_the_slope_rule_can_follow_the_relative_error_bar(world):
    f = nf.measure_floor([world.ref, world.repeat], [world.env_b])
    own = nf.derive_bars(f, margin=3.0)
    rel = nf.derive_bars(f, margin=3.0, slope_rule="rel_l2")
    assert own["layer"].slope_tol < own["layer"].rel_l2 and rel["layer"].slope_tol == rel["layer"].rel_l2
    assert "slope bar equal to the relative L2 bar" in rel["layer"].source and "slope floor" in own["layer"].source
    with pytest.raises(nf.NoiseFloorError):
        nf.derive_bars(f, margin=3.0, slope_rule="wild")


def test_a_chip_scale_error_that_no_cpu_floor_holds_fails_the_floor_rule_and_passes_the_relative_rule(world):
    """The Stable Audio chip layers are about 1 percent small. Build a good run like that: the usual error plus a 1 percent shrink."""
    shrunk = {k: (v * 0.99 if k[0] == "layer" else v) for k, v in variant(world.base, {"layer": 0.02, "latent": 0.08, "conditioning": 0.02, "step_latent": 0.06}, 1).items()}
    good = write(world.tmp / "shrunk", shrunk, dtype="bfloat16")
    dmg = nf.Damage(arrays=(("layer", "layer_02"),), mode="relative", size=0.5)
    strict = nf.analyse([world.ref, world.repeat], [world.env_b], good, margin=3.0, damage=dmg, work_dir=world.tmp / "a")
    assert strict["usable"] is False and any("slope" in r["reasons"][0] for r in strict["good"]["results"] if not r["ok"])
    loose = nf.analyse([world.ref, world.repeat], [world.env_b], good, margin=3.0, slope_rule="rel_l2", damage=dmg, work_dir=world.tmp / "b")
    assert loose["usable"] is True and loose["slope_rule"] == "rel_l2"


# ---- the known-bad control ------------------------------------------------------------------------------------

def test_a_good_run_passes_and_a_damaged_one_fails_so_the_bars_are_usable(world):
    res = nf.analyse([world.ref, world.repeat], [world.env_b], world.good, margin=3.0,
                     damage=nf.Damage(arrays=(("layer", "layer_02"),), mode="relative", size=0.5), work_dir=world.tmp / "w")
    assert res["usable"] is True and res["reasons"] == []
    assert res["good"]["ok"] is True
    assert res["controls"][0]["ok"] is False and res["controls"][0]["failing"] == ["layer/layer_02"]
    assert res["margin"] == 3.0 and res["floor"]["kinds"]["latent"]["worst"]["rel_l2"] > 0
    assert res["fallback"] is None


def test_damage_the_bars_cannot_see_makes_them_unusable_and_says_how_large_it_would_have_to_be(world):
    res = nf.analyse([world.ref, world.repeat], [world.env_b], world.good, margin=3.0,
                     damage=nf.Damage(arrays=(("layer", "layer_02"),), mode="bf16"), work_dir=world.tmp / "w")
    assert res["usable"] is False
    assert any("not detected" in r for r in res["reasons"])
    d = res["detectable"]["layer"]
    # the layer bar is 3 x 0.01 = 0.03; the good run is at 0.02; extra error adds in quadrature: sqrt(0.03^2 - 0.02^2)
    expected = min(math.sqrt(0.03 ** 2 - 0.02 ** 2) / r["norm_ratio"] for r in res["good"]["results"] if r["kind"] == "layer")
    assert d["min_relative_error"] == pytest.approx(expected, rel=1e-3)


def test_bars_the_good_run_itself_fails_are_unusable(world):
    res = nf.analyse([world.ref, world.repeat], [world.env_b], world.good, margin=1.0,
                     damage=nf.Damage(arrays=(("layer", "layer_02"),), mode="relative", size=0.5), work_dir=world.tmp / "w")
    assert res["usable"] is False and res["good"]["ok"] is False
    assert any("good run fails" in r for r in res["reasons"])
    assert res["controls"][0]["detected"] is False      # it fails, but only where the good run fails too: the bars reject everything there


def test_a_real_bad_run_must_fail_the_bars_too(world):
    bad = write(world.tmp / "bad", variant(world.base, {"layer": 0.02, "latent": 0.30, "conditioning": 0.02, "step_latent": 0.06}, 1), dtype="bfloat16")
    ok = nf.analyse([world.ref, world.repeat], [world.env_b], world.good, margin=3.0, bad_runs=[bad],
                    damage=nf.Damage(arrays=(("layer", "layer_02"),), mode="relative", size=0.5), work_dir=world.tmp / "w1")
    assert ok["usable"] is True and [c["label"] for c in ok["controls"]][-1].startswith("recorded bad run")
    mild = write(world.tmp / "mild", variant(world.base, {"layer": 0.02, "latent": 0.09, "conditioning": 0.02, "step_latent": 0.06}, 1), dtype="bfloat16")
    res = nf.analyse([world.ref, world.repeat], [world.env_b], world.good, margin=3.0, bad_runs=[mild],
                     damage=nf.Damage(arrays=(("layer", "layer_02"),), mode="relative", size=0.5), work_dir=world.tmp / "w2")
    assert res["usable"] is False and any("recorded bad run" in r and "passes" in r for r in res["reasons"])


def test_a_kind_with_no_measured_bar_is_not_passed_silently(world):
    arrays = dict(world.base)
    arrays[("step_denoised", "0")] = arrays[("step_latent", "0")]
    good = write(world.tmp / "g2", variant(arrays, {"layer": 0.02, "latent": 0.08, "conditioning": 0.02, "step_latent": 0.06, "step_denoised": 0.05}, 1), dtype="bfloat16")
    ref2 = write(world.tmp / "ref2", arrays)             # the floor runs do not hold step_denoised, so no bar can come from them
    res = nf.evaluate(ref2, good, nf.derive_bars(nf.measure_floor([world.ref, world.repeat], [world.env_b]), margin=3.0))
    bad = [r for r in res["results"] if r["kind"] == "step_denoised"]
    assert bad and bad[0]["ok"] is False and "no measured bar" in bad[0]["reasons"][0]
    assert res["ok"] is False


def test_an_array_only_one_run_holds_fails_the_evaluation(world):
    arrays = {k: v for k, v in world.base.items() if k != ("layer", "layer_01")}
    short = write(world.tmp / "short", variant(arrays, {"*": 0.01}, 3), dtype="bfloat16")
    res = nf.evaluate(world.ref, short, nf.derive_bars(nf.measure_floor([world.ref, world.repeat], [world.env_b]), margin=30.0))
    assert res["ok"] is False and any("layer_01" in p for p in res["problems"])


# ---- damage -------------------------------------------------------------------------------------------------------

def test_bf16_damage_rounds_each_value_to_eight_bits_of_mantissa():
    a = np.random.default_rng(0).normal(size=(4, 50)).astype(np.float32)
    b = nf.damage_array(a, "bf16", None, seed=0)
    assert b.dtype == np.float32 and not np.array_equal(a, b)
    assert (b.view(np.uint32) & 0xFFFF).sum() == 0                      # the low 16 bits are gone: a bfloat16 value
    assert np.abs(b - a).max() <= np.abs(a).max() * 2.0 ** -8
    rel = np.linalg.norm(b - a) / np.linalg.norm(a)
    assert 0.0005 < rel < 0.004                                         # about 2^-9 over the square root of 3 for random data


def test_bf16_damage_leaves_the_array_alone_when_it_is_already_bf16():
    a = np.random.default_rng(0).normal(size=(3, 9)).astype(np.float32)
    b = nf.damage_array(a, "bf16", None, seed=0)
    assert np.array_equal(nf.damage_array(b, "bf16", None, seed=0), b)


def test_relative_damage_has_the_stated_size_and_the_same_seed_gives_the_same_damage():
    a = np.random.default_rng(1).normal(size=(2, 40, 8)).astype(np.float32)
    b = nf.damage_array(a, "relative", 0.07, seed=3)
    assert np.linalg.norm(b - a) / np.linalg.norm(a) == pytest.approx(0.07, rel=1e-4)
    assert np.array_equal(b, nf.damage_array(a, "relative", 0.07, seed=3))
    assert not np.array_equal(b, nf.damage_array(a, "relative", 0.07, seed=4))


def test_damage_of_a_bad_kind_or_size_is_refused():
    a = np.ones((2, 2), dtype=np.float32)
    for mode, size in (("wrong", 0.1), ("relative", None), ("relative", -1.0), ("relative", float("nan")), ("relative", 0.0)):
        with pytest.raises(nf.NoiseFloorError):
            nf.damage_array(a, mode, size, seed=0)
    with pytest.raises(nf.NoiseFloorError, match="finite"):
        nf.damage_array(np.array([1.0, np.nan], dtype=np.float32), "bf16", None, seed=0)


def test_a_damaged_run_is_a_new_directory_built_from_the_good_run_and_the_good_run_is_untouched(world):
    before = {k: e["sha256"] for k, e in world.good.entries.items()}
    out = nf.make_damaged_run(world.good, world.tmp / "dmg", nf.Damage(arrays=(("layer", "layer_02"),), mode="relative", size=0.2))
    dmg = ar.load_run(out)
    assert {k: e["sha256"] for k, e in world.good.entries.items()} == before
    assert set(dmg.entries) == set(world.good.entries)
    assert dmg.settings == world.good.settings and dmg.manifest["model"] == world.good.manifest["model"]
    changed = [k for k in dmg.entries if dmg.entries[k]["sha256"] != before[k]]
    assert changed == [("layer", "layer_02")]
    assert "damaged control" in dmg.entries[("layer", "layer_02")]["note"]


def test_damage_naming_an_array_the_run_does_not_hold_is_refused(world):
    with pytest.raises(nf.NoiseFloorError, match="layer_99"):
        nf.make_damaged_run(world.good, world.tmp / "dmg2", nf.Damage(arrays=(("layer", "layer_99"),), mode="bf16"))


def test_the_default_damage_is_the_last_array_of_kind_layer(world):
    d = nf.default_damage(world.good)
    assert d.mode == "bf16" and d.arrays == (("layer", "layer_03"),)       # the highest-numbered block, not dit_output


# ---- bars out of a result, and the fallback -----------------------------------------------------------------------

def _result(world, **kw):
    return nf.analyse([world.ref, world.repeat], [world.env_b], world.good, margin=3.0, work_dir=world.tmp / "w",
                      damage=kw.get("damage") or nf.Damage(arrays=(("layer", "layer_02"),), mode="relative", size=0.5))


def test_a_work_directory_that_exists_is_refused(world):
    (world.tmp / "taken").mkdir()
    with pytest.raises(nf.NoiseFloorError, match="never reused"):
        nf.analyse([world.ref, world.repeat], [world.env_b], world.good, work_dir=world.tmp / "taken")


def test_usable_bars_come_back_from_the_result_with_their_label(world):
    res = _result(world)
    bars, label = nf.bars_or_fallback(res)
    assert bars["latent"].rel_l2 == pytest.approx(0.15, rel=1e-6) and label.startswith("measured")
    assert "3.0" in label


def test_unusable_bars_are_refused_unless_the_fallback_is_asked_for_and_then_it_is_labelled(world):
    res = _result(world, damage=nf.Damage(arrays=(("layer", "layer_02"),), mode="bf16"))
    assert res["usable"] is False
    with pytest.raises(nf.NoiseFloorError, match="unusable"):
        nf.bars_or_fallback(res)
    bars, label = nf.bars_or_fallback(res, allow_fallback=True)
    assert bars["latent"] == ar.DEFAULT_BARS["latent"]
    assert label.startswith("fallback: provisional") and "not detected" in label


def test_bars_from_a_result_that_was_edited_into_nonsense_are_refused(world):
    res = json.loads(json.dumps(_result(world)))
    for edit in (lambda r: r["bars"]["latent"].update(rel_l2=float("nan")), lambda r: r["bars"]["latent"].update(rel_l2=-1),
                 lambda r: r["bars"].update(latent="x"), lambda r: r.pop("bars"), lambda r: r.update(usable="yes")):
        bad = json.loads(json.dumps(res))
        edit(bad)
        with pytest.raises(nf.NoiseFloorError):
            nf.bars_or_fallback(bad)


def test_the_result_is_plain_json_and_names_the_runs(world):
    res = _result(world)
    again = json.loads(json.dumps(res, allow_nan=False))
    assert again["floor"]["runs"]["reference_environment"][0]["name"] == "ref"
    assert re.fullmatch(r"[0-9a-f]{64}", again["floor"]["runs"]["chip_environment"][0]["arrays_sha256"])
    assert again["bars"]["latent"]["source"].startswith("measured")


# ---- command line ------------------------------------------------------------------------------------------------

def _argv(world, *extra, tag="cli"):
    return ["--ref-env", str(world.ref.root), str(world.repeat.root), "--chip-env", str(world.env_b.root), "--good", str(world.good.root),
            "--work-dir", str(world.tmp / tag), *extra]


def test_the_command_line_exits_0_for_usable_bars_and_1_for_unusable_ones(world, tmp_path):
    out = io.StringIO()
    assert nf.main(_argv(world, "--damage", "layer/layer_02", "--damage-mode", "relative", "--damage-size", "0.5"), out=out) == 0
    assert "usable" in out.getvalue()
    out2 = io.StringIO()
    assert nf.main(_argv(world, "--damage", "layer/layer_02", tag="cli2"), out=out2) == 1
    assert "unusable" in out2.getvalue() and "not detected" in out2.getvalue()


def test_the_command_line_writes_a_json_result_and_refuses_to_replace_one(world, tmp_path):
    target = tmp_path / "floor.json"
    args = _argv(world, "--damage", "layer/layer_02", "--damage-mode", "relative", "--damage-size", "0.5", "--out", str(target))
    assert nf.main(args, out=io.StringIO()) == 0
    assert json.loads(target.read_text())["usable"] is True
    args2 = _argv(world, "--damage", "layer/layer_02", "--damage-mode", "relative", "--damage-size", "0.5", "--out", str(target), tag="again")
    assert nf.main(args2, out=io.StringIO()) == 2
    assert json.loads(target.read_text())["usable"] is True


def test_the_command_line_refuses_a_missing_run(world, tmp_path):
    out = io.StringIO()
    args = ["--ref-env", str(tmp_path / "nope"), "--chip-env", str(world.env_b.root), "--good", str(world.good.root)]
    assert nf.main(args, out=out) == 2 and out.getvalue().startswith("refused:")


def test_the_command_line_refuses_a_damage_that_is_not_kind_slash_name(world):
    out = io.StringIO()
    assert nf.main(_argv(world, "--damage", "layer_02"), out=out) == 2 and "kind/name" in out.getvalue()


BANNED = [r", not ", r"rather than", r"instead of", r"reads as", r"honest", r"\bthe one\b", r"which is why", r"This is what"]


def test_the_module_and_its_printed_text_avoid_the_phrases_the_project_does_not_use(world):
    src = Path(nf.__file__).read_text()
    for pat in BANNED:
        assert not re.search(pat, src, flags=re.I), pat
    out = io.StringIO()
    nf.main(_argv(world, "--damage", "layer/layer_02", tag="sweep"), out=out)
    for pat in BANNED:
        assert not re.search(pat, out.getvalue()), pat


def test_the_module_reads_runs_only_through_audio_reference_and_has_no_json_parser():
    src = Path(nf.__file__).read_text()
    assert not re.search(r"\bjson\.loads?\(", src) and "audio_reference" in src


# ---- review round 1: identity, missing noise, incomplete floors, what the headroom number means ------------------------

def test_floor_runs_of_another_model_revision_or_library_commit_are_refused(world):
    for name, kw, key in (("rev", dict(model={**MODEL, "revision": "c" * 40}), "model.revision"),
                          ("commit", dict(library={**LIB, "commit": "d" * 40}), "library.commit")):
        other = write(world.tmp / name, world.base, **kw)
        with pytest.raises(nf.NoiseFloorError, match=re.escape(key)):
            nf.measure_floor([world.ref], [other])


def test_environment_fields_of_the_library_may_differ_between_floor_runs(world):
    other = write(world.tmp / "env", variant(world.base, {"layer": 0.01}, 1), library={**LIB, "path_hash": "abc", "source_verified": False})
    assert nf.measure_floor([world.ref], [other])["arrays"]


def test_a_candidate_of_another_model_revision_is_refused_by_the_evaluation(world):
    other = write(world.tmp / "oc", variant(world.base, {"layer": 0.02}, 1), dtype="bfloat16", model={**MODEL, "revision": "e" * 40})
    with pytest.raises(nf.NoiseFloorError, match="model.revision"):
        nf.evaluate(world.ref, other, nf.derive_bars(nf.measure_floor([world.ref, world.repeat], [world.env_b]), margin=3.0))


def test_a_floor_run_without_the_noise_recording_is_refused_not_left_out(world):
    arrays = {k: v for k, v in world.base.items() if k[0] != "noise"}
    no_noise = write(world.tmp / "nn", variant(arrays, {"layer": 0.01}, 1))
    with pytest.raises(nf.NoiseFloorError, match=r"noise/initial is not in every run"):
        nf.measure_floor([world.ref], [no_noise])


def test_an_incomplete_floor_makes_the_bars_unusable_even_when_everything_else_passes(world):
    arrays = {k: v for k, v in world.base.items() if k != ("layer", "layer_02")}
    env_short = write(world.tmp / "es", variant(arrays, {"layer": 0.01, "latent": 0.05, "conditioning": 0.02, "step_latent": 0.04}, 1))
    res = nf.analyse([world.ref, world.repeat], [env_short], world.good, margin=3.0,
                     damage=nf.Damage(arrays=(("layer", "layer_03"),), mode="relative", size=0.5), work_dir=world.tmp / "inc")
    assert res["floor"]["problems"] and res["good"]["ok"] is True
    assert res["usable"] is False
    assert any("floor is incomplete" in r and "layer_02" in r for r in res["reasons"])
    assert any("no floor of their own" in r and "layer_02" in r for r in res["reasons"])


def test_the_headroom_is_labelled_as_a_relative_l2_estimate_everywhere_it_is_given(world):
    res = nf.analyse([world.ref, world.repeat], [world.env_b], world.good, margin=3.0,
                     damage=nf.Damage(arrays=(("layer", "layer_02"),), mode="bf16"), work_dir=world.tmp / "lab")
    assert all("PCC and slope" in d["basis"] and "relative L2" in d["basis"] for d in res["detectable"].values())
    text = "\n".join(nf.summarise(res))
    assert "assumes independent errors" in text and "PCC and slope bars may reject less" in text
    assert "passes unseen" not in text and not any("passes unseen" in r for r in res["reasons"])
    assert any("assumes independent errors" in r for r in res["reasons"])


# ---- review round 3: noise draws of the candidate and of the controls -------------------------------------------------

def test_a_recorded_bad_run_with_different_noise_draws_is_refused_not_counted_as_a_detection(world):
    base = dict(world.base)
    ref = write(world.tmp / "rn", base)
    rep = write(world.tmp / "rn2", base)
    env = write(world.tmp / "rn3", variant(base, {"layer": 0.01, "latent": 0.05, "conditioning": 0.02, "step_latent": 0.04}, 1))
    good = write(world.tmp / "rn4", variant(base, {"layer": 0.02, "latent": 0.08, "conditioning": 0.02, "step_latent": 0.06}, 1), dtype="bfloat16")
    other = dict(variant(base, {"layer": 0.02, "latent": 0.08, "conditioning": 0.02, "step_latent": 0.06}, 1))
    other[("noise", "initial")] = other[("noise", "initial")] + 1.0          # everything else is as good as the good run
    bad = write(world.tmp / "rn5", other, dtype="bfloat16")
    with pytest.raises(nf.NoiseFloorError, match="noise/initial differs"):
        nf.analyse([ref, rep], [env], good, margin=3.0, bad_runs=[bad], damage=nf.Damage((("layer", "layer_02"),), "bf16"), work_dir=world.tmp / "rnw")


def test_a_control_that_fails_only_on_a_noise_array_is_not_detected(world):
    bars = nf.derive_bars(nf.measure_floor([world.ref, world.repeat], [world.env_b]), margin=3.0)
    good_eval = nf.evaluate(world.ref, world.good, bars)
    ev = {"ok": False, "failing": ["noise/initial"], "results": []}
    c = nf._control("x", ev, good_eval, "recorded", None)
    assert c["detected"] is False and c["newly_failing"] == []


# ---- review round 4: one normalization for the headroom ----------------------------------------------------------------

def test_the_headroom_is_in_the_units_of_damage_size_for_a_scaled_candidate():
    """good = 0.8 * ref against a relative L2 bar of 0.5: the good run is 0.2 off. Orthogonal damage of size s (relative to the good array)
    adds an error of 0.8 s, so the bar is reached at 0.8 s = sqrt(0.5^2 - 0.2^2), s = 0.573, and not at 0.458."""
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(1, 50, 20))
    ev = {"ok": True, "problems": [], "failing": [], "results": [{"kind": "layer", "name": "layer_00", "group": "layer", "ok": True, "reasons": [],
          "comparable": True, "pcc": 1.0, "slope": 0.8, "rel_l2": 0.2, "norm_ratio": 0.8,
          "bars": {"pcc": 0.0, "rel_l2": 0.5, "slope_tol": 1.0}}]}
    assert nf.detectable(ev, {})["layer"]["min_relative_error"] == pytest.approx(math.sqrt(0.5 ** 2 - 0.2 ** 2) / 0.8)
    # and evaluate() reports the ratio from the arrays themselves
    r = write  # noqa: F841  (the ratio is measured in evaluate; see the next test)


def test_evaluate_reports_the_norm_ratio_of_candidate_to_reference(world):
    scaled = write(world.tmp / "sc", {k: v * 0.8 for k, v in world.base.items() if k[0] != "noise"} | {("noise", "initial"): world.base[("noise", "initial")]}, dtype="bfloat16")
    bars = nf.derive_bars(nf.measure_floor([world.ref, world.repeat], [world.env_b]), margin=3.0)
    res = nf.evaluate(world.ref, scaled, bars)
    ratio = next(r["norm_ratio"] for r in res["results"] if r["name"] == "layer_00")
    assert ratio == pytest.approx(0.8, rel=1e-6)
