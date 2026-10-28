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
import math
import os
import re
import stat
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


def file_facts(path, *, hash_it: bool = True) -> dict:
    """{"sha256", "bytes"} of a regular file, from one open descriptor: the size is the size of the file that was opened and the hash covers exactly
    those bytes. Links are followed (a Hugging Face snapshot is made of links into blobs/). `hash_it=False` skips the hash (a 2 GB checkpoint)."""
    try:
        fd = os.open(os.path.realpath(path), os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NONBLOCK", 0))
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode):
        os.close(fd)
        raise SystemExit(f"{path} is not a regular file")
    with os.fdopen(fd, "rb") as f:
        digest = None
        if hash_it:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
            digest = h.hexdigest()
    return {"sha256": digest, "bytes": st.st_size}


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


REF_NAME = re.compile(r"refs/[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")


def _read_ref(git_fd: int, name: str):
    """The text of a loose ref, found by opening each directory of its name relative to the previous one without following a link. None when any step
    is missing, is a link, or is not what it should be."""
    from orchard import pinned_files as pf
    parts = name.split("/")
    if not REF_NAME.fullmatch(name) or any(p in (".", "..") for p in parts):
        return None
    opened = []
    try:
        d = git_fd
        for part in parts[:-1]:
            d = pf.open_dir(part, dir_fd=d, follow=False)
            opened.append(d)
        return pf.read_regular(d, parts[-1], 1024).decode("ascii", "replace").strip()
    except pf.PinnedError:
        return None
    finally:
        for fd in opened:
            os.close(fd)


def library_commit() -> dict:
    """The library's name and, when it is a checkout, its commit. HEAD and a loose ref are read through descriptors (the `.git` directory opened without
    following a link, every directory of the ref's name opened relative to the one before), the ref's name must be a plain `refs/...` path, and the answer
    must be 40 hex digits; anything else is "unknown". No subprocess runs."""
    import stable_audio_3
    from orchard import pinned_files as pf
    root = Path(stable_audio_3.__file__).resolve().parents[1]
    commit = None
    try:
        # `root` was resolved above, so it holds no link. A link found here means the path was swapped after the resolve.
        rfd = pf.open_dir(root, follow=False)
        try:
            gfd = pf.open_dir(".git", dir_fd=rfd, follow=False)
            try:
                head = pf.read_regular(gfd, "HEAD", 1024).decode("ascii", "replace").strip()
                m = re.fullmatch(r"ref: (\S+)", head)
                text = _read_ref(gfd, m.group(1)) if m else head
                commit = text if text and re.fullmatch(r"[0-9a-f]{40}", text) else None
            finally:
                os.close(gfd)
        finally:
            os.close(rfd)
    except pf.PinnedError:
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
    p.add_argument("--expect-src", default=None, help="the checkout the library must come from (the one `tt-orchard audio-reference-env` pins); "
                                                      "the run is refused when `stable_audio_3` would be imported from anywhere else")
    return p.parse_args(argv)


def check_library_source(expect) -> None:
    """Refuse to go on unless `stable_audio_3` would be imported from inside `expect`. It looks the package up without importing it, so code in a
    shadowing copy never runs."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("stable_audio_3")
    except (ImportError, ValueError):
        spec = None
    origin = getattr(spec, "origin", None)
    if not origin:
        raise SystemExit("cannot find the stable_audio_3 package on the import path")
    root, found = Path(expect).resolve(), Path(origin).resolve()
    if root != found and root not in found.parents:
        raise SystemExit(f"stable_audio_3 would be imported from {found.parent.parent}, not from the expected source {root}")


def main(argv=None) -> int:
    args = parse(argv)
    offline_environment()
    sys.path.insert(0, str(REPO))
    import torch
    from orchard.audio_reference import Recorder
    if not math.isfinite(args.duration) or args.duration <= 0 or args.steps < 1:
        raise SystemExit("duration must be a positive number and steps at least 1")
    if not math.isfinite(args.cfg_scale):
        raise SystemExit("cfg-scale must be a finite number")
    if args.sample_size < 1:
        raise SystemExit("sample-size must be at least 1")
    if args.threads < 0:
        raise SystemExit("threads must be 0 or more")
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists; a run is never overwritten (checked again, atomically, when the run starts)")
    if args.seed < 0:
        # The library treats -1 as "pick a random seed" and the manifest would then record a seed that was never used.
        raise SystemExit("seed must be 0 or more (the library treats -1 as a request for a random seed)")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.expect_src:
        check_library_source(args.expect_src)
    model, config_path, ckpt_path = load_model(args.model)
    sampler = args.sampler or ("pingpong" if model.model.diffusion_objective == "rf_denoiser" else "euler")
    settings = {"prompt": args.prompt, "duration": args.duration, "steps": args.steps, "cfg_scale": args.cfg_scale, "sampler": sampler,
                "seed": args.seed, "dtype": "float32", "sample_size": args.sample_size, "device": "cpu"}
    revision = ckpt_path.parent.name       # the snapshot directory; resolving the checkpoint link would give `blobs`
    recorder = Recorder(args.out, model={"name": args.model, "revision": revision, "config_sha256": file_facts(config_path)["sha256"],
                                         "diffusion_objective": str(model.model.diffusion_objective),
                                         "checkpoint_bytes": file_facts(ckpt_path, hash_it=False)["bytes"]},
                        library={**library_commit(), "source_verified": bool(args.expect_src)}, settings=settings, seeds={"torch": args.seed, "numpy": None})
    run_reference(model, settings, recorder, torch)
    path = recorder.finish()
    print(f"wrote {path} with {len(recorder.artifacts)} arrays; timings {recorder.timings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
