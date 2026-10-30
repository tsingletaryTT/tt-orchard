# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/step_bars.py: bars from a teacher-forced floor, and the refusal when a control passes them. No chip, no model."""
import json

import numpy as np
import pytest

from orchard import step_bars as sb


def _doc(rels, vae=0.01):
    return {"dtype": "bfloat16", "dit_steps": [{"step": i, "pcc": 0.99, "rel_l2": r, "slope": 1.0} for i, r in enumerate(rels)],
            "vae_on_cpu_final_latent": {"rel_l2": vae}}


def _vels(n=4, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((64, 50))
    return [base * (1 + 0.5 * k) + 0.6 * rng.standard_normal(base.shape) for k in range(n)]


def test_bar_is_margin_times_floor_and_lifted_from_zero():
    bars = sb.derive_bars(_doc([0.02, 0.0]), margin=3.0)
    assert bars["dit"] == pytest.approx([0.06, sb.MIN_FLOOR * 3.0])


def test_margin_at_or_below_one_is_refused():
    with pytest.raises(ValueError):
        sb.derive_bars(_doc([0.02]), margin=1.0)


def test_good_chip_and_failing_controls_are_usable():
    floor = _doc([0.02] * 4)
    ctl = {"wrong_step": [0.2] * 4, "noise": [0.25] * 4}
    r = sb.analyse(floor, _doc([0.03] * 4), ctl)
    assert r["usable"] and not r["reasons"]


def test_noise_that_passes_a_bar_is_a_detection_limit_not_a_refusal():
    r = sb.analyse(_doc([0.2] * 4), _doc([0.1] * 4), {"wrong_step": [0.9] * 4, "noise": [0.25] * 4}, margin=1.5)
    assert r["usable"] and r["controls"]["noise"]["passed_the_bar_at_steps"] == [0, 1, 2, 3]
    assert r["detection_limit_rel_l2"] == pytest.approx([0.3] * 4)


def test_chip_above_bar_is_reported_by_step():
    r = sb.analyse(_doc([0.02] * 4), _doc([0.03, 0.03, 0.09, 0.03]), {"wrong_step": [0.3] * 4})
    assert not r["usable"] and any("chip step 2" in x for x in r["reasons"])


def test_a_control_that_passes_a_bar_refuses_the_bars():
    # a floor this wide gives a bar of 0.4, so a wrong-step control of 0.17 passes it
    r = sb.analyse(_doc([0.2] * 4), _doc([0.1] * 4), {"wrong_step": [0.17, 0.6, 0.6, 0.6]})
    assert not r["usable"] and r["controls"]["wrong_step"]["passed_the_bar_at_steps"] == [0]


def test_step_count_mismatch_is_an_error():
    with pytest.raises(ValueError):
        sb.analyse(_doc([0.02] * 4), _doc([0.02] * 3), {})


def test_controls_have_the_stated_size():
    c = sb.controls_from_velocities(_vels(), noise=0.25)
    assert c["noise"] == pytest.approx([0.25] * 4, rel=1e-6)
    assert all(x > 0.1 for x in c["wrong_step"])


def test_load_names_the_bad_field(tmp_path):
    d = _doc([0.02, 0.02])
    d["dit_steps"][1]["pcc"] = None
    p = tmp_path / "x.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match=r"dit_steps\[1\].pcc"):
        sb.load(p)
    d = _doc([0.02, 0.02])
    d["dit_steps"][1]["step"] = 5
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="in order"):
        sb.load(p)


def test_missing_required_control_refuses_the_bars():
    r = sb.analyse(_doc([0.02] * 4), _doc([0.03] * 4), {})
    assert not r["usable"] and any("wrong_step was not supplied" in x for x in r["reasons"])


def test_cli_refuses_files_made_from_another_run(tmp_path):
    from orchard import audio_reference as ar
    rec = ar.Recorder(tmp_path / "run", model={"name": "m"}, library={"name": "l"},
                      settings={"prompt": "p", "duration": 1.0, "steps": 2, "cfg_scale": 1.0, "sampler": "s", "seed": 1, "dtype": "float32", "sample_size": 10, "device": "cpu"})
    for k in range(2):
        rec.record("step_denoised", f"v{k}", np.ones((4, 4), np.float32) * (k + 1))
    rec.finish()
    import hashlib
    digest = hashlib.sha256((tmp_path / "run" / "manifest.json").read_bytes()).hexdigest()
    good, bad = _doc([0.02] * 2), _doc([0.02] * 2)
    good["ref_manifest_sha256"] = digest
    bad["ref_manifest_sha256"] = "0" * 64
    (tmp_path / "f.json").write_text(json.dumps(good))
    (tmp_path / "c.json").write_text(json.dumps(good))
    (tmp_path / "c_bad.json").write_text(json.dumps(bad))
    assert sb.main(["--floor", str(tmp_path / "f.json"), "--chip", str(tmp_path / "c.json"), "--run", str(tmp_path / "run")]) in (0, 1)
    assert sb.main(["--floor", str(tmp_path / "f.json"), "--chip", str(tmp_path / "c_bad.json"), "--run", str(tmp_path / "run")]) == 2


def test_a_report_that_is_not_a_bfloat16_cpu_floor_is_refused_as_a_floor():
    chip = _doc([0.03] * 4)
    del chip["dtype"]                      # a chip step check carries no dtype
    r = sb.analyse(chip, chip, {"wrong_step": [0.9] * 4})
    assert not r["usable"] and any("not a teacher-forced bfloat16" in x for x in r["reasons"])


# ---- issue 274: the reports are checked against the run they claim to come from -------------------------------------------

def _make_run(path, steps=2, dtype="float32", device="cpu", velocities=None):
    from orchard import audio_reference as ar
    rec = ar.Recorder(path, model={"name": "m"}, library={"name": "l"},
                      settings={"prompt": "p", "duration": 1.0, "steps": steps, "cfg_scale": 1.0, "sampler": "s", "seed": 1, "dtype": dtype,
                                "sample_size": 10, "device": device})
    for k in range(steps if velocities is None else velocities):
        rec.record("step_denoised", f"v{k}", np.ones((4, 4), np.float32) * (k + 1))
    rec.finish()
    import hashlib
    return hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()


def _reports(tmp_path, digest, n):
    good = _doc([0.02] * n)
    good["ref_manifest_sha256"] = digest
    (tmp_path / "f.json").write_text(json.dumps(good))
    (tmp_path / "c.json").write_text(json.dumps(good))
    return ["--floor", str(tmp_path / "f.json"), "--chip", str(tmp_path / "c.json"), "--run", str(tmp_path / "run")]


def test_reports_with_too_few_steps_for_the_run_are_refused(tmp_path, capsys):
    digest = _make_run(tmp_path / "run", steps=4)
    assert sb.main(_reports(tmp_path, digest, 3)) == 2
    assert "4 steps" in capsys.readouterr().out


def test_reports_with_too_many_steps_for_the_run_are_refused(tmp_path, capsys):
    digest = _make_run(tmp_path / "run", steps=2)
    assert sb.main(_reports(tmp_path, digest, 3)) == 2
    assert "2 steps" in capsys.readouterr().out


def test_a_run_that_is_not_a_float32_cpu_reference_is_refused(tmp_path, capsys):
    digest = _make_run(tmp_path / "run", steps=2, dtype="bfloat16")
    assert sb.main(_reports(tmp_path, digest, 2)) == 2
    assert "float32" in capsys.readouterr().out
    digest = _make_run(tmp_path / "run2", steps=2, device="tt")
    args = _reports(tmp_path, digest, 2)
    args[-1] = str(tmp_path / "run2")
    assert sb.main(args) == 2
    assert "cpu" in capsys.readouterr().out


def test_a_run_whose_recorded_velocities_are_fewer_than_its_steps_is_refused(tmp_path, capsys):
    digest = _make_run(tmp_path / "run", steps=3, velocities=2)
    assert sb.main(_reports(tmp_path, digest, 3)) == 2
    assert "velocit" in capsys.readouterr().out


def test_a_chip_file_with_another_step_count_than_the_floor_is_refused_not_a_crash(tmp_path, capsys):
    digest = _make_run(tmp_path / "run", steps=2)
    args = _reports(tmp_path, digest, 2)
    chip = _doc([0.02] * 3)
    chip["ref_manifest_sha256"] = digest
    (tmp_path / "c.json").write_text(json.dumps(chip))
    assert sb.main(args) == 2
