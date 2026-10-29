# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Makes Stable Audio 3 clips with the diffusion transformer on a Tenstorrent chip (runtime design, step 1) and records every intermediate in the
same layout the CPU reference uses, so `python -m orchard.audio_reference check --cross-precision REFERENCE THIS_RUN` compares the two.

This is a thin caller of the resident runtime (audio_runtimes.py, the contract in orchard/resident.py): it loads the model and the chip once and
calls `generate` once, or, with `--repeats N` and `--warmup W`, W times without recording and then N times with recording, all on the single loaded
process. A repeat measures generation and not a reload. Run it under a gozer lease with the shared Tenstorrent environment (it imports ttnn) and the
library's source on the import path:

    PYTHONPATH=<stable-audio-3 checkout> python chip_generate.py --model small-music --out RUN_DIR --prompt "..." --duration 10 --steps 8 \
        --cfg-scale 1.0 --seed 42 [--warmup 1 --repeats 3]

What runs on the chip: the 20-layer stack of the diffusion transformer (chip_dit.py). What stays on the host, in float32, from the library's own
code: the T5Gemma encoder, the duration embedder, the timestep, conditioning and global embedders, the noise draws, the ping-pong sampler and the
autoencoder decoder. The run records the wall time of each part. The Hub is offline and no token is read (the model is in the local cache).

Output: RUN_DIR/ (arrays and manifest.json like the reference), RUN_DIR/clips/audio.flac, RUN_DIR/chip.json (timings, device and settings). Repeat
k of a run is recorded in a sibling directory `<name>-r<k>` (orchard/resident.py repeat_dir); the load's timings are in repeat 0 only, because the
load happened once. With `--repeats` above 1 or a warm-up, `<RUN_DIR>.resident.json` beside the run holds the spread of each timed part over the
repeats (a one-shot run writes nothing outside RUN_DIR, because the job server that calls it removes only RUN_DIR).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="a new directory for the run (repeats go to siblings named <name>-r<k>)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--cfg-scale", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sampler", default=None)
    p.add_argument("--sample-size", type=int, default=5292032)
    p.add_argument("--device-id", type=int, default=0, help="the chip as the process sees it (a lease sets TT_VISIBLE_DEVICES, so 0)")
    p.add_argument("--weights-dtype", choices=("bfloat16", "float32"), default="bfloat16", help="the dtype of the weights and the attention tensors")
    p.add_argument("--chip-decoder", action="store_true", help="also run the autoencoder decoder's transformer on the chip (chip_decoder.py)")
    p.add_argument("--no-layers", action="store_true", help="do not record per-layer outputs of the first pass (each costs a read back)")
    p.add_argument("--repeats", type=int, default=1, help="recorded runs on the single loaded process (default 1)")
    p.add_argument("--warmup", type=int, default=0, help="unrecorded runs before the recorded ones, discarded (default 0)")
    return p.parse_args(argv)


def _runtimes():
    spec = importlib.util.spec_from_file_location("audio_runtimes", HERE / "audio_runtimes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audio_runtimes"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None, runtime_factory=None) -> int:
    """`runtime_factory(args)` (tests) gives the runtime; the default is the chip runtime."""
    args = parse(argv)
    sys.path.insert(0, str(REPO))
    from orchard import resident
    out = Path(args.out)
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("repeats must be at least 1 and warmup 0 or more")
    for k in range(args.repeats):
        if resident.repeat_dir(out, k).exists():
            raise SystemExit(f"{resident.repeat_dir(out, k)} exists; a run is never overwritten")
    settings = {"prompt": args.prompt, "duration": args.duration, "steps": args.steps, "cfg_scale": args.cfg_scale, "sampler": args.sampler,
                "seed": args.seed, "sample_size": args.sample_size}
    if runtime_factory is None:
        rts = _runtimes()
        problems = rts.settings_problems(settings)
        if problems:
            raise SystemExit("; ".join(problems))
        runtime = rts.ChipRuntime(args.model, device_id=args.device_id, weights_dtype=args.weights_dtype, record_layers=not args.no_layers,
                                chip_decoder=args.chip_decoder)
    else:
        runtime = runtime_factory(args)
    try:
        load = runtime.load()                    # once: the model, the device and the weights
        result = resident.run_benchmark(runtime, settings, warmup=args.warmup, repeats=args.repeats, load_timings=load,
                                        out_dirs=lambda k: resident.repeat_dir(out, k), extra_timings=lambda k: load if k == 0 else None)
    finally:
        runtime.close()
    summary = {"label": result["label"], "warmup_runs": result["warmup_runs"], "repeats": result["repeats"], "load": result["load"],
               "parts": result["parts"], "host_loadavg_1m": [r["host_loadavg_1m"] for r in result["runs"]]}
    if args.repeats > 1 or args.warmup > 0:
        out.with_name(out.name + ".resident.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out / 'manifest.json'}; {args.repeats} recorded run(s) after {args.warmup} warm-up run(s) on one loaded process; "
          f"load {load}; spread {json.dumps(result['parts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
