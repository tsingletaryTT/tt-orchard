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
    with pytest.raises(SystemExit, match="duration must be positive"):
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


def test_sha256_file(tmp_path):
    import hashlib
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert mr.sha256_file(p) == hashlib.sha256(b"abc").hexdigest()
