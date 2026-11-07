# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 162: the T5Gemma text encoder runs in float32 with eager attention wherever the reference or the chip run builds prompt embeddings.

Everything here uses fakes (tests/audio_encoder_fakes.py): no transformers, no weights, no device. Each guard of the change has a test that goes red when
the guard is removed (the mutations are listed in the pull request)."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from audio_encoder_fakes import FakeEncoder, library_model  # noqa: E402
from orchard import audio_encoder_pin as pin  # noqa: E402
from orchard import audio_reference as ar  # noqa: E402
from orchard import audio_stages as st  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MAKE_REFERENCE = REPO / "orchard" / "skills" / "audio-reference-templates" / "make_reference.py"
RUNTIMES = REPO / "orchard" / "skills" / "audio-chip-templates" / "audio_runtimes.py"


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mr = load_script(MAKE_REFERENCE, "make_reference_encoder_pin_test")


def encoders_of(model):
    return [c.model for c in model.model.conditioner.conditioners.values()]


# ---- the shared place ---------------------------------------------------------------------------------------------------------------

def test_the_pin_is_float32_with_eager_attention():
    assert (pin.ENCODER_DTYPE, pin.ENCODER_ATTN) == ("float32", "eager")
    assert pin.PINNED == {"encoder_dtype": "float32", "encoder_attn": "eager"}


@pytest.mark.parametrize("dtype, attn", [(torch.bfloat16, "sdpa"), (torch.bfloat16, "eager"), (torch.float32, "sdpa"), (torch.float32, "eager")])
def test_apply_puts_float32_and_eager_on_every_encoder(dtype, attn):
    model = library_model(dtype, attn, encoders=2)
    assert pin.apply(model) == pin.PINNED
    for e in encoders_of(model):
        assert next(e.parameters()).dtype == torch.float32 and e.config._attn_implementation == "eager"


def test_the_float32_encoder_holds_the_same_values_as_the_bfloat16_one():
    model = library_model(torch.bfloat16, "sdpa")
    before = next(encoders_of(model)[0].parameters()).detach().clone()
    pin.apply(model)
    assert torch.equal(next(encoders_of(model)[0].parameters()).detach(), before.float())


def test_a_model_with_no_encoder_is_refused_not_recorded_as_pinned():
    with pytest.raises(pin.EncoderPinError, match="no T5Gemma"):
        pin.apply(library_model(encoders=0))
    with pytest.raises(pin.EncoderPinError, match="no T5Gemma"):
        pin.observed(types.SimpleNamespace())


def test_an_encoder_that_ignores_the_change_is_not_reported_as_pinned():
    """A transformers version whose config refuses the switch must stop the run: the manifest would otherwise record a setting nobody had."""
    model = library_model(torch.bfloat16, "sdpa")
    encoder = encoders_of(model)[0]
    encoder.set_attn_implementation = lambda name: None

    class Frozen:
        _attn_implementation = "sdpa"

        def __setattr__(self, key, value):
            pass
    encoder.config = Frozen()
    with pytest.raises(pin.EncoderPinError, match="sdpa"):
        pin.apply(model)


def test_observed_reads_the_model_and_not_the_constants():
    assert pin.observed(library_model(torch.bfloat16, "sdpa")) == {"encoder_dtype": "bfloat16", "encoder_attn": "sdpa"}
    with pytest.raises(pin.EncoderPinError):
        pin.require(library_model(torch.bfloat16, "sdpa"))


# ---- every load site gets the pin ---------------------------------------------------------------------------------------------------

def fake_library(monkeypatch, loaded):
    """Fake `stable_audio_3` so that `make_reference.load_model` runs as written: the library hands back a model as it loads one, in bfloat16 with sdpa."""
    class Resolved:
        def resolve(self):
            return "/snap/model_config.json", "/snap/model.safetensors"

    class StableAudioModel:
        @staticmethod
        def from_pretrained(name, device=None, model_half=True):
            model = library_model(torch.bfloat16, "sdpa")
            loaded.append(model)
            return model
    pkg = types.ModuleType("stable_audio_3")
    model_mod = types.ModuleType("stable_audio_3.model")
    model_mod.StableAudioModel = StableAudioModel
    configs = types.ModuleType("stable_audio_3.model_configs")
    configs.all_models = {"small-music": Resolved()}
    for name, mod in (("stable_audio_3", pkg), ("stable_audio_3.model", model_mod), ("stable_audio_3.model_configs", configs)):
        monkeypatch.setitem(sys.modules, name, mod)


def test_the_reference_loader_returns_a_model_with_the_pinned_encoder(monkeypatch):
    loaded = []
    fake_library(monkeypatch, loaded)
    model, _, _ = mr.load_model("small-music")
    assert model is loaded[0] and pin.observed(model) == pin.PINNED


def test_the_chip_runs_host_encoder_is_loaded_through_the_same_pinned_loader(monkeypatch):
    """The chip run (ChipRuntime, which inherits LibraryRuntime.load) and the same-environment run both call `make_reference.load_model`."""
    loaded = []
    fake_library(monkeypatch, loaded)
    rts = load_script(RUNTIMES, "audio_runtimes_encoder_pin_test")
    assert "load" not in vars(rts.ChipRuntime), "the chip runtime must not load the model by another route"
    rt = rts.LibraryRuntime("small-music")
    rt.load()
    assert pin.observed(rt.model) == pin.PINNED and rt.model is loaded[0]


def test_the_loader_stops_when_the_pin_cannot_be_applied(monkeypatch):
    loaded = []
    fake_library(monkeypatch, loaded)
    monkeypatch.setattr(pin, "_encoders", lambda model: [])
    with pytest.raises(SystemExit, match="text encoder pin"):
        mr.load_model("small-music")


# ---- the manifest records what the encoder really had -------------------------------------------------------------------------------

def test_main_records_the_encoder_setting_in_the_manifest(tmp_path, monkeypatch):
    snap = tmp_path / "snapshots" / ("c" * 40)
    snap.mkdir(parents=True)
    (snap / "model_config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"1234")
    monkeypatch.setattr(mr, "load_model", lambda name, device="cpu": (pin_applied(), snap / "model_config.json", snap / "model.safetensors"))
    monkeypatch.setattr(mr, "library_commit", lambda: {"name": "stable-audio-3", "commit": "d" * 40})
    monkeypatch.setattr(mr, "run_reference", lambda model, settings, rec, torch_: rec.record("waveform", "audio", np.zeros((2, 4))))
    assert mr.main(["--model", "small-music", "--out", str(tmp_path / "out"), "--prompt", "p", "--duration", "2", "--steps", "3",
                    "--cfg-scale", "1.0", "--seed", "9", "--sample-size", "64"]) == 0
    settings = json.loads((tmp_path / "out" / "manifest.json").read_text())["settings"]
    assert settings["encoder_dtype"] == "float32" and settings["encoder_attn"] == "eager"


def pin_applied():
    model = library_model(torch.bfloat16, "sdpa")
    pin.apply(model)
    return model


def test_main_refuses_to_record_a_model_whose_encoder_is_not_the_pin(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "load_model", lambda name, device="cpu": (library_model(torch.bfloat16, "sdpa"), tmp_path / "c.json", tmp_path / "m.st"))
    (tmp_path / "c.json").write_text("{}")
    (tmp_path / "m.st").write_bytes(b"1")
    with pytest.raises(SystemExit, match="text encoder pin"):
        mr.main(["--model", "small-music", "--out", str(tmp_path / "out"), "--prompt", "p", "--steps", "3", "--cfg-scale", "1.0", "--seed", "9"])


def test_a_runtime_generation_records_the_encoder_setting(monkeypatch):
    rts = load_script(RUNTIMES, "audio_runtimes_encoder_pin_test2")
    rt = rts.LibraryRuntime("small-music")
    rt.mk = mr
    rt.model = pin_applied()
    s = rt._run_settings({"prompt": "p", "duration": 2.0, "steps": 3, "cfg_scale": 1.0, "seed": 1, "sampler": "pingpong", "sample_size": 64})
    assert s["encoder_dtype"] == "float32" and s["encoder_attn"] == "eager"


def test_the_recorder_accepts_the_encoder_keys_and_an_old_manifest_without_them_still_loads(tmp_path):
    base = {"prompt": "p", "duration": 1.0, "steps": 1, "cfg_scale": 1.0, "sampler": "euler", "seed": 1, "dtype": "float32", "sample_size": 8, "device": "cpu"}
    for name, extra in (("new", pin.PINNED), ("old", {})):
        rec = ar.Recorder(tmp_path / name, model={"name": "m"}, library={"name": "l"}, settings={**base, **extra})
        rec.record("waveform", "audio", np.zeros((2, 4)))
        rec.finish()
        with ar.load_run(tmp_path / name) as run:
            assert pin.state(run.settings) == ("pinned" if extra else "unrecorded")


# ---- a record with another encoder setting is refused -------------------------------------------------------------------------------

BASE = {"prompt": "p", "duration": 1.0, "steps": 1, "cfg_scale": 1.0, "sampler": "euler", "seed": 1, "dtype": "float32", "sample_size": 8, "device": "cpu"}


def run_with(tmp_path, name, **extra):
    rec = ar.Recorder(tmp_path / name, model={"name": "m"}, library={"name": "l"}, settings={**BASE, **extra})
    rec.record("waveform", "audio", np.ones((2, 8)))
    rec.record("latent", "final", np.ones((2, 8)))
    rec.finish()
    return ar.load_run(tmp_path / name)


@pytest.mark.parametrize("other", [{"encoder_dtype": "bfloat16", "encoder_attn": "eager"}, {"encoder_dtype": "float32", "encoder_attn": "sdpa"}, {}])
def test_two_runs_whose_encoders_differ_are_not_comparable(tmp_path, other):
    with run_with(tmp_path, "a", **pin.PINNED) as a, run_with(tmp_path, "b", **other) as b:
        report = ar.compare_runs(a, b)
        assert report["comparable"] is False and report["ok"] is False
        assert any("encoder" in p for p in report["problems"])
        assert any("encoder" in p for p in ar.settings_problems(a, b))


def test_the_same_check_applies_to_a_cross_precision_comparison_and_to_the_divergence_search(tmp_path):
    from orchard import first_divergence as fd
    with run_with(tmp_path, "a", **pin.PINNED) as a, run_with(tmp_path, "b", encoder_dtype="bfloat16", encoder_attn="sdpa") as b:
        assert ar.compare_runs(a, b, cross_precision=True)["comparable"] is False
        assert any("encoder" in p for p in fd._settings_problems(a, b, "candidate", cross_precision=True))


def test_matching_pinned_runs_compare_and_say_so_and_two_old_runs_are_flagged_unrecorded(tmp_path):
    with run_with(tmp_path, "a", **pin.PINNED) as a, run_with(tmp_path, "b", **pin.PINNED) as b:
        report = ar.compare_runs(a, b)
        assert report["comparable"] is True and report["encoder_pin"] == "pinned"
    with run_with(tmp_path, "c") as c, run_with(tmp_path, "d") as d:
        report = ar.compare_runs(c, d)
        assert report["comparable"] is True and report["encoder_pin"] == "unrecorded"
    with run_with(tmp_path, "e", encoder_dtype="bfloat16", encoder_attn="sdpa") as e, run_with(tmp_path, "f", encoder_dtype="bfloat16", encoder_attn="sdpa") as f:
        assert ar.compare_runs(e, f)["encoder_pin"] == "other"


def test_the_control_comparison_does_not_look_at_the_encoder(tmp_path):
    with run_with(tmp_path, "a", **pin.PINNED) as a, run_with(tmp_path, "b") as b:
        assert ar.settings_problems(a, b, ar.CONTROL_KEYS) == []


# ---- the reference cache key --------------------------------------------------------------------------------------------------------

def test_the_cache_key_changes_with_the_encoder_setting(monkeypatch):
    before = pin.cache_entry_name("small-music")
    assert before != "small-music" and "float32" in before and "eager" in before
    monkeypatch.setattr(pin, "ENCODER_ATTN", "sdpa")
    assert pin.cache_entry_name("small-music") != before
    monkeypatch.setattr(pin, "ENCODER_ATTN", "eager")
    monkeypatch.setattr(pin, "ENCODER_DTYPE", "bfloat16")
    assert pin.cache_entry_name("small-music") != before
    assert pin.cache_entry_name("ace", uses_encoder=False) == "ace"


def test_an_entry_made_before_the_pin_is_not_used_and_is_not_deleted(tmp_path):
    from test_audio_stages import COMMIT, EXP, make_run
    cache = tmp_path / "cache"
    make_run(cache / "small-music", settings_override={"encoder_dtype": "bfloat16", "encoder_attn": "sdpa"})      # the old entry, under the plain name
    assert st.cached_run(cache, "small-music", EXP, COMMIT) is None
    assert (cache / "small-music" / "manifest.json").is_file()                         # still on disk
    import orchard.supervisor as supervisor
    fresh = tmp_path / "fresh"
    make_run(fresh)
    assert supervisor.save_reference(fresh, cache, "small-music") is True
    assert (cache / "small-music" / "manifest.json").is_file()                         # the old entry was not replaced or deleted
    assert st.cached_run(cache, "small-music", EXP, COMMIT) == cache / pin.cache_entry_name("small-music")


def test_an_entry_under_the_new_name_that_holds_another_encoder_setting_is_refused(tmp_path):
    from test_audio_stages import COMMIT, EXP, make_run
    cache = tmp_path / "cache"
    make_run(cache / pin.cache_entry_name("small-music"), settings_override={"encoder_dtype": "bfloat16", "encoder_attn": "sdpa"})
    assert st.cached_run(cache, "small-music", EXP, COMMIT) is None
    (cache / pin.cache_entry_name("small-music")).rename(tmp_path / "set-aside")
    make_run(cache / pin.cache_entry_name("small-music"))                                # the same place, now with the pinned setting
    assert st.cached_run(cache, "small-music", EXP, COMMIT) == cache / pin.cache_entry_name("small-music")


def test_stage_1_refuses_a_cpu_run_that_did_not_record_the_pin(tmp_path):
    import test_audio_stages as tas
    from test_audio_stages import FakeRunner, make_run

    class Unpinned(FakeRunner):
        def __call__(self, argv, env, timeout_s, log_path):
            code = super().__call__(argv, env, timeout_s, log_path)
            out = Path(argv[argv.index("--out") + 1])
            if out.exists():
                import shutil
                shutil.rmtree(out)
                make_run(out, argv[argv.index("--model") + 1], settings_override={"encoder_dtype": "bfloat16", "encoder_attn": "sdpa"})
            return code
    with pytest.raises(st.AudioStageError, match="pinned text encoder"):
        tas.stage1(tmp_path, runner=Unpinned())


def test_stage_1_records_the_pin_state_of_each_run(tmp_path):
    import test_audio_stages as tas
    _, _, rec, _ = tas.stage1(tmp_path)
    assert {r["encoder_pin"] for r in rec["runs"].values()} == {"pinned"}
    assert all(r["settings"]["encoder_dtype"] == "float32" and r["settings"]["encoder_attn"] == "eager" for r in rec["runs"].values())


def test_the_environment_label_names_the_encoder_and_the_stage_record_carries_it(tmp_path):
    import test_audio_stages as tas
    assert "float32" in pin.LABEL and "eager" in pin.LABEL and "162" in pin.LABEL
    _, _, rec, _ = tas.stage1(tmp_path)
    assert pin.LABEL in rec["limits"]


def test_every_package_that_ships_audio_reference_ships_the_pin_module_it_imports():
    """audio_reference.py imports orchard.audio_encoder_pin, so a runtime copy that has one and not the other fails on import."""
    from orchard import acestep_package, audio_package
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import can_stable_audio_base
    finally:
        sys.path.remove(str(REPO / "scripts"))
    lists = {"audio_package": [d for _, d in audio_package.RUNTIME_FILES] if hasattr(audio_package, "RUNTIME_FILES") else None,
             "can_stable_audio_base": [d for _, d in can_stable_audio_base.RUNTIME_SRC],
             "acestep_package": [d for _, d in acestep_package.SRC_FILES]}
    for name, dests in lists.items():
        assert dests is not None, f"{name}: the list of shipped files was renamed; update this test"
        assert any(d.endswith("orchard/audio_reference.py") for d in dests), name
        assert any(d.endswith("orchard/audio_encoder_pin.py") for d in dests), f"{name} ships audio_reference.py without audio_encoder_pin.py"
