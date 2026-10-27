# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Makes a CPU reference run of a Stable Audio 3 model with the model's own library and saves every intermediate value
(design docs/superpowers/specs/2026-10-28-stable-audio-runtime-design.md, step 0).

Run it with the interpreter of the dedicated environment that `tt-orchard audio-reference-env` builds, never the shared Tenstorrent venv:

    <venv>/bin/python make_reference.py --model small-music --out RUN_DIR --prompt "..." --duration 10 --steps 8 --cfg-scale 1.0 --seed 42

What it saves (orchard/audio_reference.py describes the directory; `python -m orchard.audio_reference check` compares two of them):
    conditioning/*   the text encoder's output and the other conditioning tensors, before sampling
    noise/initial    the starting noise; noise/step_N  the fresh noise the sampler draws after step N (the ping-pong sampler draws some)
    layer/*          the output of each transformer layer and of the whole network, for the FIRST network pass only
    step_latent/N    the latent before step N;  step_denoised/N  the model's estimate of the clean latent at step N
    latent/final     the latent the decoder receives
    waveform/audio   the decoded audio, shape [channels, samples], float32 in [-1, 1]

It uses the CPU in float32, never a chip. It never uses a Hugging Face token: the token variables are removed from the environment and the Hub is
set to offline mode, so the model files must already be in the local cache (a gated model is downloaded by a person, not by this script). It reads the
model's config and weights through the library and runs the library's own `generate` with callbacks and recording wrappers, so the numbers
are the library's.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOKEN_VARIABLES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN")
DEFAULT_SAMPLE_SIZE = 5292032          # the library's own default for `generate`
LAYERS_PATH = ("model", "model", "model", "transformer", "layers")   # StableAudioModel.model (conditioned wrapper) -> .model (DiTWrapper) -> .model (network) -> .transformer.layers


def offline_environment(environ=os.environ) -> None:
    """No token, no network: the Hub client reads the local cache only."""
    for name in TOKEN_VARIABLES:
        environ.pop(name, None)
    environ["HF_HUB_OFFLINE"] = "1"
    environ["TRANSFORMERS_OFFLINE"] = "1"


def _numpy(value):
    """A torch tensor (or anything array-like) as a numpy array."""
    if hasattr(value, "detach"):
        value = value.detach().to("cpu")
        try:
            return value.float().numpy()
        except Exception:                      # a dtype numpy cannot hold
            return value.numpy()
    import numpy as np
    return np.asarray(value)


def _tensors_of(value, name):
    """Yield (name, array) for a tensor, or for each tensor in a list, tuple or dict of them. Anything else is skipped."""
    if hasattr(value, "detach"):
        yield name, _numpy(value)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from _tensors_of(v, f"{name}.{i}")
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _tensors_of(v, f"{name}.{k}")


def resolve_attr(obj, path):
    for part in path:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def recording_noise(torch, record):
    """Wrap torch.randn and torch.randn_like so every draw made while sampling is saved. The wrappers call the originals with the same arguments, so
    the random stream is unchanged."""
    original_randn, original_randn_like = torch.randn, torch.randn_like
    state = {"initial": None, "step": 0}

    def randn(*args, **kwargs):
        out = original_randn(*args, **kwargs)
        if state["initial"] is None and out.dim() == 3:
            state["initial"] = True
            record("noise", "initial", out)
        return out

    def randn_like(x, *args, **kwargs):
        out = original_randn_like(x, *args, **kwargs)
        record("noise", f"step_{state['step']}", out)
        state["step"] += 1
        return out
    torch.randn, torch.randn_like = randn, randn_like
    try:
        yield
    finally:
        torch.randn, torch.randn_like = original_randn, original_randn_like


def run_reference(model, settings: dict, recorder, torch, *, layers=None) -> None:
    """Record one generation of `model` (a library StableAudioModel). `settings` holds prompt, duration, steps, cfg_scale, seed, sampler and
    sample_size. The recorder is an orchard.audio_reference.Recorder."""
    state = {"first_pass": True}
    handles = []
    pretransform = model.model.pretransform
    original_decode = pretransform.decode

    def hook(name):
        def fn(_module, _inputs, output):
            if state["first_pass"]:
                first = output[0] if isinstance(output, (tuple, list)) else output
                if hasattr(first, "detach"):
                    recorder.record("layer", name, _numpy(first))
        return fn

    def decode(latent, *args, **kwargs):
        recorder.record("latent", "final", _numpy(latent))
        return original_decode(latent, *args, **kwargs)

    def callback(info):
        i = int(info["i"])
        recorder.record("step_latent", str(i), _numpy(info["x"]))
        if "denoised" in info:
            recorder.record("step_denoised", str(i), _numpy(info["denoised"]))
        state["first_pass"] = False

    # Everything that changes the model sits inside this try, so a failure at any point (a missing layer, the encoder, recording the
    # conditioning, the sampler) removes the hooks and puts the decoder back. A model that is run again then records each layer once.
    try:
        layer_list = layers if layers is not None else resolve_attr(model, LAYERS_PATH)
        for i, layer in enumerate(layer_list):
            handles.append(layer.register_forward_hook(hook(f"layer_{i:02d}")))
        dit = model.model.model.model               # the network that the layers sit in
        handles.append(dit.register_forward_hook(hook("dit_output")))
        pretransform.decode = decode

        conditioning = [{"prompt": settings["prompt"], "seconds_total": settings["duration"]}]
        t0 = time.time()
        with torch.inference_mode():
            tensors = model.model.conditioner(conditioning, model.device)
        recorder.timings["conditioning_s"] = round(time.time() - t0, 3)
        for key, value in tensors.items():
            for name, arr in _tensors_of(value, str(key)):
                recorder.record("conditioning", name, arr)
        t0 = time.time()
        with recording_noise(torch, lambda kind, name, t: recorder.record(kind, name, _numpy(t))):
            audio = model.generate(conditioning=conditioning, conditioning_tensors=dict(tensors), duration=settings["duration"],
                                   steps=settings["steps"], cfg_scale=settings["cfg_scale"], seed=settings["seed"],
                                   sample_size=settings["sample_size"], sampler_type=settings["sampler"], callback=callback)
        recorder.timings["generate_and_decode_s"] = round(time.time() - t0, 3)
    finally:
        pretransform.decode = original_decode
        for h in handles:
            h.remove()
    wave = _numpy(audio)
    if wave.ndim == 3:
        wave = wave[0]
    recorder.record("waveform", "audio", wave)


def load_model(name: str, device: str = "cpu"):
    """The library's own loader. Returns (model, config_path, checkpoint_path)."""
    from stable_audio_3.model import StableAudioModel
    from stable_audio_3.model_configs import all_models
    if name not in all_models:
        raise SystemExit(f"unknown model {name!r}; the library knows {sorted(all_models)}")
    config_path, ckpt_path = all_models[name].resolve()
    return StableAudioModel.from_pretrained(name, device=device, model_half=False), Path(config_path), Path(ckpt_path)


def library_commit() -> dict:
    """The library's name and, when it is a checkout, its commit (read from the .git directory, no subprocess)."""
    import stable_audio_3
    root = Path(stable_audio_3.__file__).resolve().parents[1]
    commit = None
    head = root / ".git" / "HEAD"
    try:
        text = head.read_text().strip()
        commit = (root / ".git" / text.split(" ", 1)[1]).read_text().strip() if text.startswith("ref:") else text
    except (OSError, IndexError):
        pass
    return {"name": "stable-audio-3", "commit": commit or "unknown", "path_hash": hashlib.sha256(str(root).encode()).hexdigest()[:12]}


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="the library's model name, e.g. small-music or small-music-base")
    p.add_argument("--out", required=True, help="a new directory for the run")
    p.add_argument("--prompt", required=True)
    p.add_argument("--duration", type=float, default=10.0, help="seconds of audio")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--cfg-scale", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sampler", default=None, help="pingpong, euler, rk4 or dpmpp; default: the library's choice for the model's objective")
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: leave the default)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    offline_environment()
    sys.path.insert(0, str(REPO))
    import torch
    from orchard.audio_reference import Recorder
    if args.duration <= 0 or args.steps < 1:
        raise SystemExit("duration must be positive and steps at least 1")
    if args.seed < 0:
        # The library treats -1 as "pick a random seed" and the manifest would then record a seed that was never used.
        raise SystemExit("seed must be 0 or more (the library treats -1 as a request for a random seed)")
    if args.threads:
        torch.set_num_threads(args.threads)
    model, config_path, ckpt_path = load_model(args.model)
    sampler = args.sampler or ("pingpong" if model.model.diffusion_objective == "rf_denoiser" else "euler")
    settings = {"prompt": args.prompt, "duration": args.duration, "steps": args.steps, "cfg_scale": args.cfg_scale, "sampler": sampler,
                "seed": args.seed, "dtype": "float32", "sample_size": args.sample_size, "device": "cpu"}
    revision = ckpt_path.parent.name       # the snapshot directory; resolving the checkpoint link would give `blobs`
    recorder = Recorder(args.out, model={"name": args.model, "revision": revision, "config_sha256": sha256_file(config_path),
                                         "diffusion_objective": str(model.model.diffusion_objective), "checkpoint_bytes": ckpt_path.stat().st_size},
                        library=library_commit(), settings=settings, seeds={"torch": args.seed, "numpy": None})
    run_reference(model, settings, recorder, torch)
    path = recorder.finish()
    print(f"wrote {path} with {len(recorder.artifacts)} arrays; timings {recorder.timings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
