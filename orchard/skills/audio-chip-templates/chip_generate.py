# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Makes one Stable Audio 3 clip with the diffusion transformer on a Tenstorrent chip (runtime design, step 1) and records every intermediate in the
same layout the CPU reference uses, so `python -m orchard.audio_reference check --cross-precision REFERENCE THIS_RUN` compares the two.

Run it under a gozer lease with the shared Tenstorrent environment (it imports ttnn) and the library's source on the import path:

    PYTHONPATH=<stable-audio-3 checkout> python chip_generate.py --model small-music --out RUN_DIR --prompt "..." --duration 10 --steps 8 \
        --cfg-scale 1.0 --seed 42

What runs on the chip: the 20-layer stack of the diffusion transformer (chip_dit.py). What stays on the host, in float32, from the library's own
code: the T5Gemma encoder, the duration embedder, the timestep, conditioning and global embedders, the noise draws, the ping-pong sampler and the
autoencoder decoder. The run records the wall time of each part. The Hub is offline and no token is read (the model is in the local cache).

Output: RUN_DIR/ (arrays and manifest.json like the reference), RUN_DIR/clips/audio.flac, RUN_DIR/chip.json (timings, device and settings).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load_sibling(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="a new directory for the run")
    p.add_argument("--prompt", required=True)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--cfg-scale", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sampler", default=None)
    p.add_argument("--sample-size", type=int, default=5292032)
    p.add_argument("--device-id", type=int, default=0, help="the chip as the process sees it (a lease sets TT_VISIBLE_DEVICES, so 0)")
    p.add_argument("--weights-dtype", choices=("bfloat16", "float32"), default="bfloat16", help="the dtype of the weights and the attention tensors")
    p.add_argument("--no-layers", action="store_true", help="do not record per-layer outputs of the first pass (each costs a read back)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    mk = load_sibling(REPO / "orchard" / "skills" / "audio-reference-templates" / "make_reference.py", "make_reference_for_chip")
    mk.offline_environment()
    sys.path.insert(0, str(REPO))
    import torch
    from orchard.audio_reference import Recorder
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists; a run is never overwritten")
    if args.seed < 0 or args.steps < 1 or not math.isfinite(args.duration) or args.duration <= 0:
        raise SystemExit("seed must be 0 or more, steps at least 1 and duration a positive number")
    t_start = time.time()
    model, config_path, ckpt_path = mk.load_model(args.model)
    t_loaded = time.time()
    chip_dit_cfg = load_sibling(HERE / "chip_dit.py", "chip_dit_config")
    problems = chip_dit_cfg.config_problems(json.loads(Path(config_path).read_text(encoding="utf-8")))
    if problems:
        raise SystemExit("the chip path does not implement this model's configuration:\n  " + "\n  ".join(problems))
    sampler = args.sampler or ("pingpong" if model.model.diffusion_objective == "rf_denoiser" else "euler")
    settings = {"prompt": args.prompt, "duration": args.duration, "steps": args.steps, "cfg_scale": args.cfg_scale, "sampler": sampler,
                "seed": args.seed, "dtype": args.weights_dtype, "sample_size": args.sample_size, "device": "tt"}
    revision = ckpt_path.parent.name
    recorder = Recorder(args.out, model={"name": args.model, "revision": revision, "config_sha256": mk.file_facts(config_path)["sha256"],
                                         "diffusion_objective": str(model.model.diffusion_objective),
                                         "checkpoint_bytes": mk.file_facts(ckpt_path, hash_it=False)["bytes"]},
                        library={**mk.library_commit(), "source_verified": False}, settings=settings, seeds={"torch": args.seed, "numpy": None})

    import ttnn                                    # after the lease; this opens the runtime
    chip_dit = load_sibling(HERE / "chip_dit.py", "chip_dit_runtime")
    device = ttnn.open_device(device_id=args.device_id)
    try:
        dit = model.model.model.model              # StableAudioModel.model (conditioned wrapper) -> .model (DiTWrapper) -> .model (network)
        record = None if args.no_layers else (lambda name, t: recorder.record("layer", name, mk._numpy(t)))
        chip = chip_dit.ChipTransformer(ttnn, torch, device, dit.transformer, record=record,
                                        dtype=getattr(ttnn, args.weights_dtype))
        original = chip_dit.install(dit, chip)
        del original
        mk.run_reference(model, settings, recorder, torch, layers=[])
        # the stack's forward time is the part that ran on the chip; the rest of `generate_and_decode_s` ran on the host
        recorder.timings.update({"model_load_s": round(t_loaded - t_start, 3), "chip_weights_upload_s": chip.timings["upload_weights_s"],
                                 "chip_transformer_call_s": round(chip.timings["forward_s"], 3), "chip_passes": chip.timings["passes"]})
        path = recorder.finish()
    finally:
        ttnn.close_device(device)
    # the clip, written from the recorded waveform
    wave = None
    try:
        import numpy as np
        import soundfile
        from orchard.audio_reference import load_run
        with load_run(args.out) as run:
            wave = np.asarray(run.array("waveform", "audio"))
        clips = Path(args.out) / "clips"
        clips.mkdir(exist_ok=True)
        soundfile.write(str(clips / "audio.flac"), wave.T, 44100)
    except Exception as exc:                       # the run is still valid; say what failed
        print(f"clip not written: {exc!r}", file=sys.stderr)
    (Path(args.out) / "chip.json").write_text(json.dumps({"timings": recorder.timings, "settings": settings,
                                                          "device": "blackhole, one chip, device id %d" % args.device_id,
                                                          "total_s": round(time.time() - t_start, 3)}, indent=1))
    print(f"wrote {path}; timings {recorder.timings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
