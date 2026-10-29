# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Audio stage 2 with bars from a measured noise floor (issue 173, orchard/audio_bars.py). The runs are the small synthetic ones of
test_audio_stages; no model and no chip. The known-bad control is the first bfloat16 decoder run of the Stable Audio overnight log
(docs/run-logs/2026-10-29-overnight-stable-audio-decoder.md): a waveform at PCC -0.02 against the CPU decode, with layers 10 to 44 percent off."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from orchard import audio_bars as ab
from orchard import audio_reference as ar
from orchard import audio_stages as st
from orchard import first_divergence as fd
from orchard import noise_floor as nf
from test_audio_stages import BASE, MODEL, make_run, stage2, twin

ROLES = {"chip": "reference", "control_chip": "control_reference"}


def make_floor(tmp_path, *, margin=5.0):
    run_dir, ref_dir, s2, _ = stage2(tmp_path)
    rec = json.loads((ref_dir / "audio_reference.json").read_text())
    ref_path = run_dir / rec["runs"]["reference"]["dir"]
    repeat = twin(ref_path, run_dir / "floor-repeat", noise=0.0003, dtype="float32", device="cpu")
    good_path = next(p for p in sorted((s2 / "chip").iterdir()) if "base" not in p.name and "repeat" not in p.name)
    handles = [ar.load_run(p) for p in (ref_path, repeat, run_dir / rec["runs"]["host_same_env"]["dir"], good_path)]
    try:
        ref, rep, host, good = handles
        res = nf.analyse([ref, rep], [host], good, margin=margin, work_dir=tmp_path / "nf-work",
                         damage=nf.Damage(arrays=(("layer", "layer_00"),), mode="relative", size=0.5))
    finally:
        for h in handles:
            h.close()
    return run_dir, ref_dir, s2, ref_path, res


def put(ref_dir, res):
    (ref_dir / ab.FLOOR_FILE).write_text(fd.to_json(res))


def test_the_fixture_floor_is_usable(tmp_path):
    *_, res = make_floor(tmp_path)
    assert res["usable"] is True, res["reasons"]


def test_without_a_floor_record_the_result_says_no_floor_measured(tmp_path):
    run_dir, ref_dir, s2, res = stage2(tmp_path)
    assert res["noise_floor"]["measured"] is False and res["noise_floor"]["label"].startswith("no floor measured")
    assert res["agreement"]["chip_vs_reference"]["bars"].startswith("no floor measured")
    assert any(x.startswith("No floor measured") for x in res["limits"])
    assert st.chip_problems(s2, run_dir, ref_dir) == []


def test_with_a_floor_record_the_verdict_uses_the_measured_bars_and_the_gate_agrees(tmp_path):
    run_dir, ref_dir, s2, ref_path, floor_res = make_floor(tmp_path)
    put(ref_dir, floor_res)
    plan = st.chip_plan(MODEL, BASE)
    res = st.finish_chip(s2, run_dir, ref_dir, plan, ROLES)
    assert res["noise_floor"]["measured"] is True and res["noise_floor"]["sha256"]
    assert res["agreement"]["chip_vs_reference"]["bars"].startswith("measured: 5.0 times")
    assert res["agreement"]["chip_vs_reference"]["ok"] is True
    assert st.chip_problems(s2, run_dir, ref_dir) == []
    assert not any("No floor measured" in x for x in res["limits"])


def test_a_floor_record_changed_after_the_result_was_written_fails_the_gate(tmp_path):
    run_dir, ref_dir, s2, ref_path, floor_res = make_floor(tmp_path)
    put(ref_dir, floor_res)
    st.finish_chip(s2, run_dir, ref_dir, st.chip_plan(MODEL, BASE), ROLES)
    lax = json.loads(fd.to_json(floor_res))
    lax["margin"] = 50.0
    (ref_dir / ab.FLOOR_FILE).write_text(json.dumps(lax))
    assert any(p.startswith("noise_floor:") for p in st.chip_problems(s2, run_dir, ref_dir))
    (ref_dir / ab.FLOOR_FILE).unlink()
    assert any(p.startswith("noise_floor:") for p in st.chip_problems(s2, run_dir, ref_dir))


def broken_decoder_run(tmp_path, ref_path):
    """The first bfloat16 decoder run: waveform uncorrelated with the CPU decode (PCC about -0.02), layers 10 to 44 percent off."""
    with ar.load_run(ref_path) as r:
        wave, layer, latent = (np.asarray(r.array(*k)) for k in (("waveform", "audio"), ("layer", "layer_00"), ("latent", "final")))
        manifest = r.manifest
    g = np.random.default_rng(3)
    bad_wave = g.standard_normal(wave.shape).astype(np.float32) * 0.3
    d = g.standard_normal(layer.shape)
    bad_layer = layer + d * (0.3 * np.linalg.norm(layer) / np.linalg.norm(d))
    return make_run(tmp_path / "broken", "small-music", dtype="bfloat16", device="tt", wave=bad_wave, latent=latent, layer=bad_layer,
                    revision=manifest["model"]["revision"], config_sha256=manifest["model"]["config_sha256"])


def test_the_known_bad_decoder_run_is_rejected_by_the_measured_bars_and_the_good_run_is_not(tmp_path):
    run_dir, ref_dir, s2, ref_path, floor_res = make_floor(tmp_path)
    put(ref_dir, floor_res)
    floor = ab.load_floor(ref_dir)
    assert floor.measured
    bad = broken_decoder_run(tmp_path, ref_path)
    with ar.load_run(ref_path) as r, ar.load_run(bad) as b:
        pcc = ar.metrics(np.asarray(r.array("waveform", "audio")), np.asarray(b.array("waveform", "audio")))["pcc"]
    assert abs(pcc) < 0.05, pcc                               # the control has the broken run's waveform correlation
    got = st.compare(ref_path, bad, floor=floor)
    assert got["ok"] is False and "waveform/audio" in got["failed"] and "layer/layer_00" in got["failed"]
    good = next(p for p in sorted((s2 / "chip").iterdir()) if "base" not in p.name and "repeat" not in p.name)
    assert st.compare(ref_path, good, floor=floor)["ok"] is True


def test_an_error_the_provisional_bars_let_through_fails_the_measured_bars(tmp_path):
    """A layer 10 percent off passes the provisional bar of 0.15 and is far over a bar set from a floor of well under one percent."""
    run_dir, ref_dir, s2, ref_path, floor_res = make_floor(tmp_path)
    put(ref_dir, floor_res)
    with ar.load_run(ref_path) as r:
        layer = np.asarray(r.array("layer", "layer_00"))
        wave, latent, m = np.asarray(r.array("waveform", "audio")), np.asarray(r.array("latent", "final")), r.manifest
    d = np.random.default_rng(5).standard_normal(layer.shape)
    off = make_run(tmp_path / "off", "small-music", dtype="bfloat16", device="tt", wave=wave, latent=latent,
                   layer=layer + d * (0.1 * np.linalg.norm(layer) / np.linalg.norm(d)),
                   revision=m["model"]["revision"], config_sha256=m["model"]["config_sha256"])
    assert st.compare(ref_path, off)["ok"] is True                                     # provisional bars: passes
    assert st.compare(ref_path, off, floor=ab.load_floor(ref_dir))["failed"] == ["layer/layer_00"]


def test_a_floor_whose_bars_cannot_tell_good_from_bad_is_not_used_and_the_label_says_why(tmp_path):
    run_dir, ref_dir, s2, ref_path, floor_res = make_floor(tmp_path)
    unusable = json.loads(fd.to_json(floor_res))
    unusable["usable"], unusable["reasons"] = False, ["the damaged control is not detected"]
    (ref_dir / ab.FLOOR_FILE).write_text(json.dumps(unusable))
    floor = ab.load_floor(ref_dir)
    assert floor.measured is False and "not usable" in floor.label and "damaged control is not detected" in floor.label and floor.sha256


@pytest.mark.parametrize("text, word", [("not json", "cannot be read"), ("[]", "not usable")])
def test_an_unreadable_floor_record_falls_back_with_a_label(tmp_path, text, word):
    (tmp_path / ab.FLOOR_FILE).write_text(text)
    floor = ab.load_floor(tmp_path)
    assert floor.measured is False and word in floor.label


def test_a_floor_grouped_by_family_is_not_applied_by_kind(tmp_path):
    run_dir, ref_dir, s2, ref_path, floor_res = make_floor(tmp_path)
    fam = json.loads(fd.to_json(floor_res))
    fam["group"] = "family"
    (ref_dir / ab.FLOOR_FILE).write_text(json.dumps(fam))
    floor = ab.load_floor(ref_dir)
    assert floor.measured is False and "family" in floor.label


def test_a_kind_the_floor_does_not_cover_is_named_in_the_label(tmp_path):
    run_dir, ref_dir, s2, ref_path, floor_res = make_floor(tmp_path)
    put(ref_dir, floor_res)
    floor = ab.load_floor(ref_dir)
    assert "no measured bar for step_latent" in floor.describe({"layer", "step_latent"})
    assert "no measured bar" not in floor.describe({"layer"})
