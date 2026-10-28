# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/skills/audio-reference-templates/make_reference.py against a tiny fake of the library's model (no weights, no chip). The fake has the
same attribute layout the script reads (StableAudioModel.model.model.model.transformer.layers, .conditioner, .pretransform.decode,
.generate with a callback) and draws noise through torch.randn and torch.randn_like the way the ping-pong sampler does."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from orchard.audio_reference import Recorder, ReferenceError, load_run  # noqa: E402

SCRIPT = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "audio-reference-templates" / "make_reference.py"
SPEC = importlib.util.spec_from_file_location("make_reference_under_test", SCRIPT)
mr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mr)

SETTINGS = {"prompt": "a piano", "duration": 2.0, "steps": 3, "cfg_scale": 1.0, "sampler": "pingpong", "seed": 7, "dtype": "float32",
            "sample_size": 64, "device": "cpu"}


class Layer(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return x * self.k + 0.01


class Transformer(torch.nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(1.0 + 0.1 * i) for i in range(depth)])


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()

    def forward(self, x, t):
        for layer in self.transformer.layers:
            x = layer(x)
        return x * 0.1


class DiTWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Network()

    def forward(self, x, t, **kw):
        return self.model(x, t)


class Pretransform(torch.nn.Module):
    def decode(self, z, **kw):
        return z.repeat_interleave(4, dim=-1)[:, :2]


class Conditioner:
    def __call__(self, conditioning, device):
        return {"prompt": [torch.arange(6.0).reshape(1, 3, 2), torch.ones(1, 3)], "seconds_total": [torch.tensor([[conditioning[0]["seconds_total"]]])],
                "note": "not a tensor"}


class Wrapper(torch.nn.Module):
    def __init__(self, objective="rf_denoiser"):
        super().__init__()
        self.model = DiTWrapper()
        self.pretransform = Pretransform()
        self.conditioner = Conditioner()
        self.diffusion_objective = objective


class FakeModel:
    """The surface of the library's StableAudioModel that the script uses."""
    device = "cpu"

    def __init__(self, objective="rf_denoiser", fail_at=None):
        self.model = Wrapper(objective)
        self.fail_at = fail_at
        self.calls = []

    def generate(self, *, conditioning, conditioning_tensors, duration, steps, cfg_scale, seed, sample_size, sampler_type, callback):
        self.calls.append({"conditioning": conditioning, "tensors": conditioning_tensors, "sampler": sampler_type, "seed": seed})
        torch.manual_seed(seed)
        x = torch.randn([1, 2, 8])
        for i in range(steps):
            if self.fail_at == i:
                raise RuntimeError("boom")
            den = x - 0.5 * self.model.model(x, torch.ones(1))
            callback({"x": x, "i": i, "denoised": den})
            x = 0.5 * den + 0.5 * torch.randn_like(x)
        return self.model.pretransform.decode(x)


def record(tmp_path, model=None, settings=None, name="run", layers=None):
    rec = Recorder(tmp_path / name, model={"name": "fake"}, library={"name": "fake"}, settings=settings or SETTINGS, seeds={"torch": 7})
    model = model or FakeModel()
    mr.run_reference(model, settings or SETTINGS, rec, torch, layers=layers)
    rec.finish()
    return load_run(tmp_path / name), model


def names(run, kind):
    return sorted(n for k, n in run.entries if k == kind)


def test_a_run_records_every_kind_of_intermediate(tmp_path):
    run, _ = record(tmp_path)
    assert names(run, "conditioning") == ["prompt.0", "prompt.1", "seconds_total.0"]
    assert names(run, "layer") == ["dit_output", "layer_00", "layer_01", "layer_02"]
    assert names(run, "step_latent") == ["0", "1", "2"] and names(run, "step_denoised") == ["0", "1", "2"]
    assert names(run, "latent") == ["final"] and names(run, "waveform") == ["audio"]


def test_the_noise_draws_are_recorded_in_order(tmp_path):
    run, _ = record(tmp_path)
    assert names(run, "noise") == ["initial", "step_0", "step_1", "step_2"]
    torch.manual_seed(7)
    expected_initial = torch.randn([1, 2, 8]).numpy()
    assert np.array_equal(run.array("noise", "initial"), expected_initial)


def test_recording_does_not_change_the_random_stream(tmp_path):
    run, _ = record(tmp_path)
    wave = FakeModel().generate(conditioning=[{}], conditioning_tensors={}, duration=2.0, steps=3, cfg_scale=1.0, seed=7, sample_size=64,
                                sampler_type="pingpong", callback=lambda info: None)
    assert np.array_equal(run.array("waveform", "audio"), wave[0].numpy())


def test_the_recorded_values_are_the_models_own(tmp_path):
    run, _ = record(tmp_path)
    x0 = run.array("step_latent", "0")
    first = Layer(1.0)(torch.from_numpy(x0)).numpy()
    assert np.allclose(run.array("layer", "layer_00"), first, atol=1e-6)
    assert np.allclose(run.array("step_denoised", "0"), x0 - 0.5 * run.array("layer", "dit_output"), atol=1e-6)
    assert run.array("latent", "final").shape == (1, 2, 8)


def test_layers_are_recorded_for_the_first_pass_only(tmp_path):
    run, model = record(tmp_path, settings={**SETTINGS, "steps": 4})
    assert names(run, "layer") == ["dit_output", "layer_00", "layer_01", "layer_02"]
    assert names(run, "step_latent") == ["0", "1", "2", "3"]


def test_the_waveform_is_channels_by_samples(tmp_path):
    run, _ = record(tmp_path)
    assert run.array("waveform", "audio").shape == (2, 32)


def test_the_conditioning_is_computed_once_and_handed_to_generate(tmp_path):
    run, model = record(tmp_path)
    call = model.calls[0]
    assert call["conditioning"] == [{"prompt": "a piano", "seconds_total": 2.0}]
    assert set(call["tensors"]) == {"prompt", "seconds_total", "note"} and call["sampler"] == "pingpong" and call["seed"] == 7


def test_non_tensor_conditioning_is_skipped(tmp_path):
    run, _ = record(tmp_path)
    assert not [n for n in names(run, "conditioning") if n.startswith("note")]


def test_hooks_the_decoder_and_the_random_functions_are_restored_after_a_run(tmp_path):
    model = FakeModel()
    original_decode = model.model.pretransform.decode
    before = (torch.randn, torch.randn_like)
    record(tmp_path, model=model)
    assert (torch.randn, torch.randn_like) == before
    assert model.model.pretransform.decode == original_decode
    assert all(not l._forward_hooks for l in model.model.model.model.transformer.layers)
    assert not model.model.model.model._forward_hooks


def test_a_failure_in_generate_restores_everything_and_records_no_manifest(tmp_path):
    model = FakeModel(fail_at=1)
    before = (torch.randn, torch.randn_like)
    original_decode = model.model.pretransform.decode
    rec = Recorder(tmp_path / "bad", model={"name": "fake"}, library={"name": "fake"}, settings=SETTINGS)
    with pytest.raises(RuntimeError, match="boom"):
        mr.run_reference(model, SETTINGS, rec, torch)
    assert (torch.randn, torch.randn_like) == before and model.model.pretransform.decode == original_decode
    assert all(not l._forward_hooks for l in model.model.model.model.transformer.layers)
    assert not (tmp_path / "bad" / "manifest.json").exists()


def test_a_duplicate_array_name_from_the_model_is_an_error_not_a_silent_overwrite(tmp_path):
    rec = Recorder(tmp_path / "dup", model={"name": "fake"}, library={"name": "fake"}, settings=SETTINGS)
    rec.record("waveform", "audio", np.zeros(2))
    with pytest.raises(ReferenceError):
        mr.run_reference(FakeModel(), SETTINGS, rec, torch)


def test_noise_wrapper_ignores_non_three_dimensional_first_draws():
    seen = []
    with mr.recording_noise(torch, lambda kind, name, t: seen.append((kind, name, tuple(t.shape)))):
        torch.randn(4)
        torch.randn(1, 2, 3)
        torch.randn(1, 2, 3)
        torch.randn_like(torch.zeros(2))
    assert seen == [("noise", "initial", (1, 2, 3)), ("noise", "step_0", (2,))]


# ---- the environment and the helpers --------------------------------------------------------------

def test_offline_environment_removes_every_token_and_sets_offline_mode():
    env = {"HF_TOKEN": "secret", "HUGGING_FACE_HUB_TOKEN": "s2", "HUGGINGFACE_HUB_TOKEN": "s3", "HF_API_TOKEN": "s4", "KEEP": "1"}
    mr.offline_environment(env)
    assert env == {"KEEP": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_main_removes_the_token_before_anything_is_imported(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    seen = {}

    def fake_load(name, device="cpu"):
        seen["token"] = os.environ.get("HF_TOKEN")
        seen["offline"] = os.environ.get("HF_HUB_OFFLINE")
        raise SystemExit("stop here")
    monkeypatch.setattr(mr, "load_model", fake_load)
    with pytest.raises(SystemExit):
        mr.main(["--model", "m", "--out", str(tmp_path / "o"), "--prompt", "p", "--steps", "3", "--cfg-scale", "1", "--seed", "1"])
    assert seen == {"token": None, "offline": "1"}


def test_main_writes_a_manifest_with_the_model_revision_and_settings(tmp_path, monkeypatch):
    snap = tmp_path / "snapshots" / ("c" * 40)
    snap.mkdir(parents=True)
    (snap / "model_config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"1234")
    monkeypatch.setattr(mr, "load_model", lambda name, device="cpu": (FakeModel(), snap / "model_config.json", snap / "model.safetensors"))
    monkeypatch.setattr(mr, "library_commit", lambda: {"name": "stable-audio-3", "commit": "d" * 40})
    monkeypatch.setattr(mr, "run_reference", lambda model, settings, rec, torch_: rec.record("waveform", "audio", np.zeros((2, 4))))
    code = mr.main(["--model", "small-music", "--out", str(tmp_path / "out"), "--prompt", "p", "--duration", "2", "--steps", "3",
                    "--cfg-scale", "1.0", "--seed", "9", "--sample-size", "64"])
    assert code == 0
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert m["model"]["revision"] == "c" * 40 and m["model"]["name"] == "small-music" and m["model"]["checkpoint_bytes"] == 4
    assert m["settings"]["sampler"] == "pingpong" and m["settings"]["seed"] == 9 and m["settings"]["device"] == "cpu"
    assert m["settings"]["dtype"] == "float32" and m["settings"]["sample_size"] == 64
    assert m["library"]["commit"] == "d" * 40 and m["seeds"]["torch"] == 9
    import hashlib
    assert m["model"]["config_sha256"] == hashlib.sha256(b"{}").hexdigest()


def test_the_default_sampler_follows_the_models_objective(tmp_path, monkeypatch):
    snap = tmp_path / "snapshots" / ("c" * 40)
    snap.mkdir(parents=True)
    (snap / "model_config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"1234")
    monkeypatch.setattr(mr, "library_commit", lambda: {"name": "x", "commit": "y"})
    monkeypatch.setattr(mr, "run_reference", lambda model, settings, rec, torch_: rec.record("waveform", "audio", np.zeros((2, 4))))
    for objective, expected in (("rectified_flow", "euler"), ("rf_denoiser", "pingpong")):
        monkeypatch.setattr(mr, "load_model", lambda name, device="cpu", o=objective: (FakeModel(o), snap / "model_config.json", snap / "model.safetensors"))
        out = tmp_path / objective
        mr.main(["--model", "m", "--out", str(out), "--prompt", "p", "--steps", "3", "--cfg-scale", "1", "--seed", "1"])
        assert json.loads((out / "manifest.json").read_text())["settings"]["sampler"] == expected
    out = tmp_path / "explicit"
    mr.main(["--model", "m", "--out", str(out), "--prompt", "p", "--steps", "3", "--cfg-scale", "1", "--seed", "1", "--sampler", "rk4"])
    assert json.loads((out / "manifest.json").read_text())["settings"]["sampler"] == "rk4"


@pytest.mark.parametrize("duration,steps", [("0", "3"), ("-1", "3"), ("2", "0")])
def test_a_non_positive_duration_or_steps_is_refused_before_the_model_loads(tmp_path, monkeypatch, duration, steps):
    argv = ["--model", "m", "--out", str(tmp_path / "o"), "--prompt", "p", "--duration", duration, "--steps", steps, "--cfg-scale", "1", "--seed", "1"]
    monkeypatch.setattr(mr, "load_model", lambda *a, **k: pytest.fail("the model was loaded"))
    with pytest.raises(SystemExit, match="duration must be a positive number"):
        mr.main(argv)


def test_an_unknown_model_name_is_refused_with_the_known_names(monkeypatch):
    import types
    fake_model = types.ModuleType("stable_audio_3.model")
    fake_model.StableAudioModel = object
    fake_configs = types.ModuleType("stable_audio_3.model_configs")
    fake_configs.all_models = {"small-music": None}
    monkeypatch.setitem(sys.modules, "stable_audio_3", types.ModuleType("stable_audio_3"))
    monkeypatch.setitem(sys.modules, "stable_audio_3.model", fake_model)
    monkeypatch.setitem(sys.modules, "stable_audio_3.model_configs", fake_configs)
    with pytest.raises(SystemExit, match="small-music"):
        mr.load_model("nonsense")


def test_the_library_commit_is_read_from_the_checkout_without_a_subprocess(tmp_path, monkeypatch):
    import types
    root = tmp_path / "stable-audio-3"
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (root / ".git" / "refs" / "heads" / "main").write_text("e" * 40 + "\n")
    pkg = root / "stable_audio_3"
    pkg.mkdir()
    mod = types.ModuleType("stable_audio_3")
    mod.__file__ = str(pkg / "__init__.py")
    monkeypatch.setitem(sys.modules, "stable_audio_3", mod)
    assert mr.library_commit()["commit"] == "e" * 40
    (root / ".git" / "HEAD").write_text("f" * 40 + "\n")
    assert mr.library_commit()["commit"] == "f" * 40
    (root / ".git" / "HEAD").unlink()
    assert mr.library_commit()["commit"] == "unknown"


def test_tensors_of_walks_lists_and_dicts_and_skips_the_rest():
    t = torch.ones(2)
    got = dict(mr._tensors_of({"a": [t, "x", (t,)], "b": 3, "c": t}, "k"))
    assert sorted(got) == ["k.a.0", "k.a.2.0", "k.c"]


def test_resolve_attr_follows_numeric_parts_as_indexes():
    class Box:
        items = [10, 20]
    assert mr.resolve_attr(Box(), ("items", "1")) == 20


# ---- review of PR 123: cleanup when conditioning fails, the sentinel seed, the snapshot revision ---------

class RaisingConditioner:
    def __call__(self, conditioning, device):
        raise RuntimeError("encoder failed")


def hooks_left(model):
    return [l for l in model.model.model.model.transformer.layers if l._forward_hooks] + ([model.model.model.model] if model.model.model.model._forward_hooks else [])


def test_a_failing_encoder_removes_the_hooks_and_restores_the_decoder(tmp_path):
    model = FakeModel()
    model.model.conditioner = RaisingConditioner()
    original_decode = model.model.pretransform.decode
    before = (torch.randn, torch.randn_like)
    rec = Recorder(tmp_path / "r", model={"name": "fake"}, library={"name": "fake"}, settings=SETTINGS)
    with pytest.raises(RuntimeError, match="encoder failed"):
        mr.run_reference(model, SETTINGS, rec, torch)
    assert not hooks_left(model) and model.model.pretransform.decode == original_decode and (torch.randn, torch.randn_like) == before


def test_a_failure_while_recording_the_conditioning_removes_the_hooks_and_restores_the_decoder(tmp_path):
    model = FakeModel()
    original_decode = model.model.pretransform.decode
    rec = Recorder(tmp_path / "r", model={"name": "fake"}, library={"name": "fake"}, settings=SETTINGS)
    rec.record("conditioning", "prompt.0", np.zeros(2))                 # the run's first conditioning array is now a duplicate
    with pytest.raises(ReferenceError, match="already recorded"):
        mr.run_reference(model, SETTINGS, rec, torch)
    assert not hooks_left(model) and model.model.pretransform.decode == original_decode


def test_the_same_model_can_be_run_again_after_a_failure_without_duplicate_layers(tmp_path):
    model = FakeModel()
    good_conditioner = model.model.conditioner
    model.model.conditioner = RaisingConditioner()
    with pytest.raises(RuntimeError):
        mr.run_reference(model, SETTINGS, Recorder(tmp_path / "a", model={"name": "fake"}, library={"name": "fake"}, settings=SETTINGS), torch)
    model.model.conditioner = good_conditioner
    run, _ = record(tmp_path, model=model, name="b")
    assert names(run, "layer") == ["dit_output", "layer_00", "layer_01", "layer_02"]


def run_main(tmp_path, monkeypatch, *extra, seed="1"):
    monkeypatch.setattr(mr, "load_model", lambda *a, **k: pytest.fail("the model was loaded"))
    argv = ["--model", "m", "--out", str(tmp_path / "o"), "--prompt", "p", "--steps", "3", "--cfg-scale", "1", "--seed", seed, *extra]
    return mr.main(argv)


@pytest.mark.parametrize("seed", ["-1", "-5"])
def test_a_negative_seed_is_refused_because_the_library_would_pick_a_random_one(tmp_path, monkeypatch, seed):
    with pytest.raises(SystemExit, match="seed must be 0 or more"):
        run_main(tmp_path, monkeypatch, seed=seed)


def test_seed_zero_is_accepted_by_the_argument_check(tmp_path, monkeypatch):
    with pytest.raises(pytest.fail.Exception, match="the model was loaded"):
        run_main(tmp_path, monkeypatch, seed="0")


def test_the_revision_is_the_snapshot_directory_name_even_when_the_checkpoint_is_a_link_into_blobs(tmp_path, monkeypatch):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "abc123").write_bytes(b"1234")
    (blobs / "cfg123").write_text("{}")
    snap = tmp_path / "snapshots" / ("c" * 40)
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blobs / "abc123")
    (snap / "model_config.json").symlink_to(blobs / "cfg123")
    monkeypatch.setattr(mr, "load_model", lambda name, device="cpu": (FakeModel(), snap / "model_config.json", snap / "model.safetensors"))
    monkeypatch.setattr(mr, "library_commit", lambda: {"name": "x", "commit": "y"})
    monkeypatch.setattr(mr, "run_reference", lambda model, settings, rec, torch_: rec.record("waveform", "audio", np.zeros((2, 4))))
    mr.main(["--model", "m", "--out", str(tmp_path / "out"), "--prompt", "p", "--steps", "3", "--cfg-scale", "1", "--seed", "1"])
    m = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert m["model"]["revision"] == "c" * 40 and m["model"]["checkpoint_bytes"] == 4


# ---- round three: numeric arguments and the library's source ---------------------------------------------

@pytest.mark.parametrize("extra,match", [(["--duration", "nan"], "duration"), (["--duration", "inf"], "duration"), (["--duration=-inf"], "duration"),
                                         (["--cfg-scale", "nan"], "cfg-scale"), (["--cfg-scale", "inf"], "cfg-scale"),
                                         (["--sample-size", "0"], "sample-size"), (["--sample-size", "-4"], "sample-size"),
                                         (["--threads", "-1"], "threads")])
def test_a_non_finite_or_out_of_range_number_is_refused_before_the_model_loads(tmp_path, monkeypatch, extra, match):
    monkeypatch.setattr(mr, "load_model", lambda *a, **k: pytest.fail("the model was loaded"))
    argv = ["--model", "m", "--out", str(tmp_path / "o"), "--prompt", "p", "--steps", "3", "--seed", "1"]
    if "--cfg-scale" not in extra:
        argv += ["--cfg-scale", "1"]
    argv = [a for a in argv if a not in ("--duration",)] if False else argv
    with pytest.raises(SystemExit, match=match):
        mr.main(argv + extra)
    assert not (tmp_path / "o").exists()


def test_a_negative_cfg_scale_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "load_model", lambda *a, **k: pytest.fail("the model was loaded"))
    with pytest.raises(pytest.fail.Exception, match="the model was loaded"):
        mr.main(["--model", "m", "--out", str(tmp_path / "o"), "--prompt", "p", "--steps", "3", "--seed", "1", "--cfg-scale", "0"])


def fake_library(monkeypatch, tmp_path, where):
    import types
    mod = types.ModuleType("stable_audio_3")
    mod.__file__ = str(where / "stable_audio_3" / "__init__.py")
    mod.__spec__ = importlib.util.spec_from_file_location("stable_audio_3", mod.__file__)
    monkeypatch.setitem(sys.modules, "stable_audio_3", mod)


def test_expect_src_accepts_a_library_found_inside_that_tree_without_importing_it(tmp_path, monkeypatch):
    src = tmp_path / "src"
    (src / "stable_audio_3").mkdir(parents=True)
    (src / "stable_audio_3" / "__init__.py").write_text("raise RuntimeError('imported')")
    monkeypatch.setattr(sys, "path", [str(src)] + sys.path)
    monkeypatch.delitem(sys.modules, "stable_audio_3", raising=False)
    mr.check_library_source(src)                      # must not execute the package's __init__


def test_expect_src_refuses_a_library_found_elsewhere(tmp_path, monkeypatch):
    other = tmp_path / "other"
    (other / "stable_audio_3").mkdir(parents=True)
    (other / "stable_audio_3" / "__init__.py").write_text("")
    src = tmp_path / "src"
    src.mkdir()
    monkeypatch.setattr(sys, "path", [str(other)] + sys.path)
    monkeypatch.delitem(sys.modules, "stable_audio_3", raising=False)
    with pytest.raises(SystemExit, match="not from the expected source"):
        mr.check_library_source(src)


def test_expect_src_refuses_when_the_library_cannot_be_found(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    monkeypatch.delitem(sys.modules, "stable_audio_3", raising=False)
    with pytest.raises(SystemExit, match="cannot find"):
        mr.check_library_source(tmp_path)


def test_a_sibling_directory_with_the_same_prefix_is_not_the_expected_source(tmp_path, monkeypatch):
    other = Path(str(tmp_path / "src") + "-old")
    (other / "stable_audio_3").mkdir(parents=True)
    (other / "stable_audio_3" / "__init__.py").write_text("")
    (tmp_path / "src").mkdir()
    monkeypatch.setattr(sys, "path", [str(other)] + sys.path)
    monkeypatch.delitem(sys.modules, "stable_audio_3", raising=False)
    with pytest.raises(SystemExit):
        mr.check_library_source(tmp_path / "src")


def test_main_checks_the_source_before_loading_and_records_that_it_did(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mr, "check_library_source", lambda d: calls.append(Path(d)))
    snap = tmp_path / "snapshots" / ("c" * 40)
    snap.mkdir(parents=True)
    (snap / "model_config.json").write_text("{}")
    (snap / "model.safetensors").write_bytes(b"1234")
    monkeypatch.setattr(mr, "load_model", lambda name, device="cpu": (calls.append("loaded"), (FakeModel(), snap / "model_config.json", snap / "model.safetensors"))[1])
    monkeypatch.setattr(mr, "library_commit", lambda: {"name": "x", "commit": "y"})
    monkeypatch.setattr(mr, "run_reference", lambda model, settings, rec, torch_: rec.record("waveform", "audio", np.zeros((2, 4))))
    base = ["--model", "m", "--prompt", "p", "--steps", "3", "--cfg-scale", "1", "--seed", "1"]
    mr.main([*base, "--out", str(tmp_path / "a"), "--expect-src", str(tmp_path / "src")])
    assert calls == [tmp_path / "src", "loaded"]
    assert json.loads((tmp_path / "a" / "manifest.json").read_text())["library"]["source_verified"] is True
    mr.main([*base, "--out", str(tmp_path / "b")])
    assert json.loads((tmp_path / "b" / "manifest.json").read_text())["library"]["source_verified"] is False


def test_an_existing_output_directory_is_refused_before_the_model_loads(tmp_path, monkeypatch):
    (tmp_path / "o").mkdir()
    monkeypatch.setattr(mr, "load_model", lambda *a, **k: pytest.fail("the model was loaded"))
    with pytest.raises(SystemExit, match="never overwritten"):
        mr.main(["--model", "m", "--out", str(tmp_path / "o"), "--prompt", "p", "--steps", "3", "--cfg-scale", "1", "--seed", "1"])


def test_expect_src_given_through_a_link_is_compared_by_where_it_leads(tmp_path, monkeypatch):
    src = tmp_path / "src"
    (src / "stable_audio_3").mkdir(parents=True)
    (src / "stable_audio_3" / "__init__.py").write_text("")
    link = tmp_path / "link"
    link.symlink_to(src)
    monkeypatch.setattr(sys, "path", [str(src)] + sys.path)
    monkeypatch.delitem(sys.modules, "stable_audio_3", raising=False)
    mr.check_library_source(link)


# ---- round five: files read through descriptors ------------------------------------------------------------------

def checkout(tmp_path, head, refs=None):
    import types
    root = tmp_path / "stable-audio-3"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text(head)
    for rel, text in (refs or {}).items():
        p = root / ".git" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    pkg = root / "stable_audio_3"
    pkg.mkdir()
    mod = types.ModuleType("stable_audio_3")
    mod.__file__ = str(pkg / "__init__.py")
    return root, mod


def commit_of(monkeypatch, mod):
    monkeypatch.setitem(sys.modules, "stable_audio_3", mod)
    return mr.library_commit()["commit"]


def test_a_symbolic_head_is_followed_to_a_loose_ref_inside_the_git_directory(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "ref: refs/heads/main\n", {"refs/heads/main": "a" * 40 + "\n"})
    assert commit_of(monkeypatch, mod) == "a" * 40


def test_a_detached_head_is_its_commit(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "b" * 40 + "\n")
    assert commit_of(monkeypatch, mod) == "b" * 40


@pytest.mark.parametrize("head", ["ref: ../../etc/passwd\n", "ref: refs/../../x\n", "ref: /etc/passwd\n", "ref: refs/heads/../../x\n", "ref: refs/heads/a b\n",
                                  "ref: heads/main\n", "zzzz\n", "", "ref: refs/heads/\n", "ref: refs//heads/x\n", "ref: refs/heads/main extra\n", "A" * 40 + "\n"])
def test_a_head_that_is_not_a_commit_or_a_plain_ref_name_gives_unknown(tmp_path, monkeypatch, head):
    root, mod = checkout(tmp_path, head, {"refs/heads/main": "a" * 40 + "\n"})
    outside = tmp_path / "x"
    outside.write_text("c" * 40)
    assert commit_of(monkeypatch, mod) == "unknown"


def test_a_ref_file_that_is_not_a_commit_gives_unknown(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "ref: refs/heads/main\n", {"refs/heads/main": "not a commit\n"})
    assert commit_of(monkeypatch, mod) == "unknown"


def test_a_missing_ref_file_gives_unknown(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "ref: refs/heads/main\n")
    assert commit_of(monkeypatch, mod) == "unknown"


def test_a_git_directory_that_is_a_link_is_not_read(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "b" * 40 + "\n")
    real = tmp_path / "real-git"
    (root / ".git").rename(real)
    (root / ".git").symlink_to(real)
    assert commit_of(monkeypatch, mod) == "unknown"


def test_a_head_file_or_ref_file_that_is_a_link_is_not_read(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "ref: refs/heads/main\n", {"refs/heads/main": "a" * 40 + "\n"})
    other = tmp_path / "other"
    other.write_text("d" * 40 + "\n")
    ref = root / ".git" / "refs" / "heads" / "main"
    ref.unlink()
    ref.symlink_to(other)
    assert commit_of(monkeypatch, mod) == "unknown"
    head = root / ".git" / "HEAD"
    head.unlink()
    head.symlink_to(other)
    assert commit_of(monkeypatch, mod) == "unknown"


def test_a_ref_directory_that_is_a_link_is_not_followed(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "ref: refs/heads/main\n", {"refs/heads/main": "a" * 40 + "\n"})
    heads = root / ".git" / "refs" / "heads"
    moved = tmp_path / "moved-heads"
    heads.rename(moved)
    heads.symlink_to(moved)
    assert commit_of(monkeypatch, mod) == "unknown"


def test_an_oversized_head_gives_unknown(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "ref: refs/heads/" + "a" * 5000 + "\n")
    assert commit_of(monkeypatch, mod) == "unknown"


def test_the_library_commit_comes_with_the_name_and_a_path_hash(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "b" * 40 + "\n")
    monkeypatch.setitem(sys.modules, "stable_audio_3", mod)
    info = mr.library_commit()
    assert info["name"] == "stable-audio-3" and len(info["path_hash"]) == 12


def test_file_facts_reads_the_size_and_the_hash_from_one_open_file(tmp_path, monkeypatch):
    import hashlib
    f = tmp_path / "f.bin"
    f.write_bytes(b"abcdef")
    real_stat = Path.stat

    def guard(self, *a, **k):
        if self.name == "f.bin":
            raise AssertionError("f.bin was looked at by name")
        return real_stat(self, *a, **k)
    monkeypatch.setattr(Path, "stat", guard)
    assert mr.file_facts(f) == {"sha256": hashlib.sha256(b"abcdef").hexdigest(), "bytes": 6}
    assert mr.file_facts(f, hash_it=False) == {"sha256": None, "bytes": 6}


def test_file_facts_follows_a_cache_link_but_refuses_a_directory_and_a_missing_file(tmp_path):
    blob = tmp_path / "blob"
    blob.write_bytes(b"xyz")
    link = tmp_path / "snap-link"
    link.symlink_to(blob)
    assert mr.file_facts(link)["bytes"] == 3
    with pytest.raises(SystemExit, match="not a regular file"):
        mr.file_facts(tmp_path)
    with pytest.raises(SystemExit, match="cannot read"):
        mr.file_facts(tmp_path / "missing")


# ---- descriptor anchoring: what each guard in the commit reader stops -------------------------------------

def _fds():
    return {int(n) for n in os.listdir("/proc/self/fd")}


def test_a_ref_name_that_climbs_out_of_refs_is_not_followed_even_when_it_would_find_a_commit(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "ref: refs/heads/../../commit\n", {"refs/heads/keep": "a" * 40 + "\n", "commit": "c" * 40 + "\n"})
    assert commit_of(monkeypatch, mod) == "unknown"
    (root / ".git" / "HEAD").write_text("ref: refs/../commit\n")
    assert commit_of(monkeypatch, mod) == "unknown"
    (root / ".git" / "HEAD").write_text("ref: refs/./heads/keep\n")
    assert commit_of(monkeypatch, mod) == "unknown"


@pytest.mark.parametrize("head,ref", [("ref: heads/main\n", "heads/main"), ("ref: refs/heads/a b\n", "refs/heads/a b"),
                                      ("ref: refs/heads/a:b\n", "refs/heads/a:b"), ("ref: refs/heads/a~b\n", "refs/heads/a~b")])
def test_a_ref_name_outside_the_plain_pattern_is_not_read_even_when_the_file_exists(tmp_path, monkeypatch, head, ref):
    root, mod = checkout(tmp_path, head, {ref: "c" * 40 + "\n"})
    assert commit_of(monkeypatch, mod) == "unknown"


def test_a_missing_loose_ref_is_none_and_not_an_error(tmp_path):
    from orchard import pinned_files as pf
    (tmp_path / "refs" / "heads").mkdir(parents=True)
    fd = pf.open_dir(tmp_path)
    try:
        assert mr._read_ref(fd, "refs/heads/nope") is None
        assert mr._read_ref(fd, "refs/nodir/x") is None
    finally:
        os.close(fd)


def test_reading_a_commit_closes_every_descriptor_it_opened(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "ref: refs/heads/feature/x\n", {"refs/heads/feature/x": "a" * 40 + "\n"})
    monkeypatch.setitem(sys.modules, "stable_audio_3", mod)
    before = _fds()
    assert mr.library_commit()["commit"] == "a" * 40
    assert _fds() <= before
    (root / ".git" / "refs" / "heads" / "feature" / "x").unlink()
    assert mr.library_commit()["commit"] == "unknown"
    assert _fds() <= before


def test_a_head_file_over_the_size_cap_is_refused_even_when_the_rest_is_blank(tmp_path, monkeypatch):
    root, mod = checkout(tmp_path, "b" * 40 + "\n" + " " * 3000)
    assert commit_of(monkeypatch, mod) == "unknown"


def test_a_checkout_directory_that_is_a_link_at_the_open_is_not_read(tmp_path, monkeypatch):
    """The path is resolved first, so a link at the open means it was swapped in between. Simulated by not resolving."""
    root, mod = checkout(tmp_path, "b" * 40 + "\n")
    linked = tmp_path / "linked"
    linked.symlink_to(root)
    mod.__file__ = str(linked / "stable_audio_3" / "__init__.py")
    monkeypatch.setattr(mr.Path, "resolve", lambda self, *a, **k: self)
    assert commit_of(monkeypatch, mod) == "unknown"


def test_file_facts_refuses_a_fifo_without_waiting_for_a_writer(tmp_path):
    import signal
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    signal.alarm(20)                      # a regression would block in open(); the alarm ends the test run instead of hanging it
    try:
        with pytest.raises(SystemExit, match="not a regular file"):
            mr.file_facts(fifo)
    finally:
        signal.alarm(0)
