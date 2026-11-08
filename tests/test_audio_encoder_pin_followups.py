# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 555, 556 and 557 (found in the review of PR 545): the encoder pin (orchard/audio_encoder_pin.py, issue 162).

555: the pin read back only the top-level encoder config, so a child config that kept `sdpa` could be recorded as `eager`.
556: the stage 1 gate accepted a reference with missing or unpinned encoder settings, and a resume kept an existing record without looking.
557: a missing encoder field and an explicit null compared equal."""
import json
import shutil
from types import SimpleNamespace

import pytest
import torch

from audio_encoder_fakes import FakeEncoder, T5GemmaConditioner, library_model
from orchard import audio_cli, audio_encoder_pin as pin, audio_plan, audio_remote, audio_stages as st, supervisor
from test_audio_stages import make_run, stage1


# ---- 557: absent is not null --------------------------------------------------------------------------------------------------

def test_a_missing_field_and_a_recorded_null_differ():
    problems = pin.settings_problems({}, {"encoder_dtype": None, "encoder_attn": None})
    assert len(problems) == 2 and all("absent" in p for p in problems)


def test_two_runs_that_both_lack_the_fields_or_both_record_null_have_no_problem():
    assert pin.settings_problems({}, {}) == []
    assert pin.settings_problems({"encoder_dtype": None}, {"encoder_dtype": None}) == []


def test_the_shared_pin_state_does_not_depend_on_which_run_is_the_reference():
    a, b = {}, {"encoder_dtype": None, "encoder_attn": None}
    assert pin.settings_problems(a, b) and pin.settings_problems(b, a)


# ---- 555: every submodule config ---------------------------------------------------------------------------------------------

class ReadOnlyConfig:
    """A config that refuses the switch (a read-only config, or a transformers version that moved the attribute)."""
    _attn_implementation = "sdpa"

    def __setattr__(self, name, value):
        raise AttributeError("read-only")


def encoder_with_stubborn_child():
    enc = FakeEncoder(attn="sdpa")
    child = torch.nn.Linear(2, 2)
    child.config = ReadOnlyConfig()
    enc.add_module("child", child)
    return enc


def model_with(encoder):
    return SimpleNamespace(model=SimpleNamespace(conditioner=SimpleNamespace(conditioners={"p": T5GemmaConditioner(encoder)})))


def test_a_child_config_that_stays_sdpa_is_refused_before_the_pin_is_recorded():
    model = model_with(encoder_with_stubborn_child())
    with pytest.raises(pin.EncoderPinError, match="attention"):
        pin.apply(model)


def test_the_readback_reports_mixed_attention_when_a_child_differs():
    model = model_with(encoder_with_stubborn_child())
    with pytest.raises(pin.EncoderPinError):
        pin.apply(model)                           # the top-level config switched to eager, the child stayed sdpa
    assert pin.observed(model)[pin.ATTN_KEY] == "mixed"


def test_a_model_whose_every_config_switches_still_pins():
    seen = pin.apply(library_model())
    assert seen == pin.PINNED


def test_a_child_config_with_no_attention_setting_is_not_a_difference():
    enc = FakeEncoder(attn="sdpa")
    child = torch.nn.Linear(2, 2)
    child.config = SimpleNamespace()               # nothing to run attention with
    enc.add_module("child", child)
    assert pin.apply(model_with(enc)) == pin.PINNED


# ---- 556: the stage 1 gate and the resume --------------------------------------------------------------------------------------

def replace_reference_run(run_dir, stage_dir, rec, **override):
    d = run_dir / rec["runs"]["reference"]["dir"]
    shutil.rmtree(d)
    make_run(d, settings_override=override)
    data = json.loads((stage_dir / "audio_reference.json").read_text())
    entry = data["runs"]["reference"]
    entry["manifest_sha256"] = st.sha256_file(d / "manifest.json")
    entry["settings"] = st.run_facts(run_dir, entry["dir"])["settings"]
    (stage_dir / "audio_reference.json").write_text(json.dumps(data))


def test_the_stage_1_gate_refuses_a_run_that_did_not_record_the_pinned_encoder(tmp_path):
    run_dir, stage_dir, rec, _ = stage1(tmp_path)
    assert st.reference_problems(stage_dir, run_dir) == []
    replace_reference_run(run_dir, stage_dir, rec, encoder_dtype="bfloat16")
    problems = st.reference_problems(stage_dir, run_dir)
    assert any(pin.UNPINNED_PROBLEM in p and "reference" in p for p in problems)


def test_the_stage_1_gate_refuses_a_run_with_no_encoder_fields_at_all(tmp_path):
    run_dir, stage_dir, rec, _ = stage1(tmp_path)
    d = run_dir / rec["runs"]["reference"]["dir"]
    manifest = json.loads((d / "manifest.json").read_text())
    for k in pin.ENCODER_KEYS:
        manifest["settings"].pop(k, None)
    (d / "manifest.json").write_text(json.dumps(manifest))
    data = json.loads((stage_dir / "audio_reference.json").read_text())
    data["runs"]["reference"]["manifest_sha256"] = st.sha256_file(d / "manifest.json")
    data["runs"]["reference"]["settings"] = st.run_facts(run_dir, data["runs"]["reference"]["dir"])["settings"]
    (stage_dir / "audio_reference.json").write_text(json.dumps(data))
    assert any(pin.UNPINNED_PROBLEM in p for p in st.reference_problems(stage_dir, run_dir))


def in_process(python, verb, args, *, timeout_s, env=None):
    out = audio_cli.dispatch(verb, args)
    if "error" in out:
        raise audio_plan.AudioStageError(out["error"])
    return out["result"]


class FakeLedger:
    def __init__(self):
        self.rows = []

    def append(self, event, stage, **data):
        self.rows.append((event, stage, data))


def resume_fake(stage_dir, run_dir):
    calls = []
    fake = SimpleNamespace(run_dir=run_dir, ledger=FakeLedger(), stage_calls=calls)
    fake._audio_stage_1 = lambda *a, **k: calls.append(a)
    fake._audio_reference_pin_stale = lambda *a, **k: supervisor.Supervisor._audio_reference_pin_stale(fake, *a, **k)
    return fake


def test_a_resume_regenerates_a_stage_1_record_whose_runs_are_not_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_remote, "call", in_process)
    run_dir, stage_dir, rec, _ = stage1(tmp_path)
    replace_reference_run(run_dir, stage_dir, rec, encoder_attn="sdpa")
    fake = resume_fake(stage_dir, run_dir)
    paths = SimpleNamespace(reference_python="/ref/python")
    supervisor.Supervisor._audio_stage_1_or_resume(fake, stage_dir, paths, "m", "b", {}, resumed=True)
    assert len(fake.stage_calls) == 1
    assert any(row[2].get("decision") == "audio reference regenerated" for row in fake.ledger.rows)


def test_a_resume_keeps_a_stage_1_record_that_is_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_remote, "call", in_process)
    run_dir, stage_dir, _, _ = stage1(tmp_path)
    fake = resume_fake(stage_dir, run_dir)
    supervisor.Supervisor._audio_stage_1_or_resume(fake, stage_dir, SimpleNamespace(reference_python="/ref/python"), "m", "b", {}, resumed=True)
    assert fake.stage_calls == [] and fake.ledger.rows == []


def test_a_resume_does_not_regenerate_for_a_problem_that_is_not_about_the_pin(tmp_path, monkeypatch):
    """Another gate problem (a changed array) is the gate's to report, not a reason to throw the evidence away."""
    monkeypatch.setattr(audio_remote, "call", in_process)
    run_dir, stage_dir, rec, _ = stage1(tmp_path)
    f = run_dir / rec["runs"]["reference"]["dir"] / "arrays" / "latent__final.npy"
    f.write_bytes(f.read_bytes()[:-4] + b"\x01\x02\x03\x04")
    fake = resume_fake(stage_dir, run_dir)
    supervisor.Supervisor._audio_stage_1_or_resume(fake, stage_dir, SimpleNamespace(reference_python="/ref/python"), "m", "b", {}, resumed=True)
    assert fake.stage_calls == []


def test_a_fresh_start_always_makes_the_references(tmp_path):
    run_dir = tmp_path / "run"
    stage_dir = run_dir / "stages" / "1"
    stage_dir.mkdir(parents=True)
    fake = resume_fake(stage_dir, run_dir)
    supervisor.Supervisor._audio_stage_1_or_resume(fake, stage_dir, SimpleNamespace(reference_python="/ref/python"), "m", "b", {}, resumed=False)
    assert len(fake.stage_calls) == 1
