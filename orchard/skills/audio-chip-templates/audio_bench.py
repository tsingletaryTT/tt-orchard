# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A length sweep of the Stable Audio 3 runtime on one loaded process (issue 174): the model is loaded once, then each stated clip length is run with the
warm-up and the repeats of orchard/resident.py, and the spread of every timed part is written to one JSON file.

    # on a chip, under a gozer lease, in the shared Tenstorrent environment:
    PYTHONPATH=<stable-audio-3 checkout> python audio_bench.py --device chip --model small-music --out bench-chip.json --prompt "..." \
        --steps 8 --cfg-scale 1.0 --seed 42 --sampler pingpong --sizes 10,30,60,120 --warmup 1 --repeats 3
    # the CPU comparison, in the reference environment:
    PYTHONPATH=<stable-audio-3 checkout> python audio_bench.py --device cpu ...

The sizes default to orchard/resident.py SWEEP_SECONDS (10, 30, 60 and 120 seconds); a size over 120 seconds is refused. Fewer than 3 repeats at a size is
labelled "single sample" in that size's record. Every measured run carries the host's one-minute load average, because the host is shared. The runs are
recorded into a directory that is removed after each run (no arrays or clips are kept), so a sweep leaves only the JSON file.
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
    p.add_argument("--device", choices=("chip", "cpu"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="the JSON file to write (a new file)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--cfg-scale", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sampler", default=None)
    p.add_argument("--sample-size", type=int, default=5292032)
    p.add_argument("--sizes", default=None, help="clip lengths in seconds, comma separated (default 10,30,60,120)")
    p.add_argument("--warmup", type=int, default=None, help="discarded runs before the repeats at each size (default: the rule in orchard/resident.py)")
    p.add_argument("--repeats", type=int, default=None, help="measured runs at each size (default: the rule in orchard/resident.py)")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads, on the CPU runtime and for the host work of the chip runtime (0: the default)")
    p.add_argument("--device-id", type=int, default=0)
    return p.parse_args(argv)


def load_runtimes():
    spec = importlib.util.spec_from_file_location("audio_runtimes", HERE / "audio_runtimes.py")
    rts = importlib.util.module_from_spec(spec)
    sys.modules["audio_runtimes"] = rts
    spec.loader.exec_module(rts)
    return rts


def main(argv=None, runtime_factory=None) -> int:
    args = parse(argv)
    sys.path.insert(0, str(REPO))
    from orchard import resident
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; a result is never overwritten")
    warmup = resident.WARMUP_RUNS if args.warmup is None else args.warmup
    repeats = resident.MIN_REPEATS if args.repeats is None else args.repeats
    try:
        sizes = resident.SWEEP_SECONDS if args.sizes is None else tuple(float(x) for x in args.sizes.split(","))
    except ValueError:
        raise SystemExit("--sizes is a comma separated list of numbers of seconds") from None
    settings = {"prompt": args.prompt, "duration": sizes[0], "steps": args.steps, "cfg_scale": args.cfg_scale, "sampler": args.sampler, "seed": args.seed,
                "sample_size": args.sample_size}
    for s in sizes:                                          # refuse before the model is loaded
        if not (s > 0 and s <= resident.MAX_SECONDS):
            raise SystemExit(f"a size must be above 0 and at most {resident.MAX_SECONDS:g} seconds, got {s:g}")
    if runtime_factory is None:
        rts = load_runtimes()
        problems = rts.settings_problems(settings)
        if problems:
            raise SystemExit("; ".join(problems))
        if args.device == "chip":
            runtime = rts.ChipRuntime(args.model, device_id=args.device_id, threads=args.threads, record_layers=False)
        else:
            runtime = rts.LibraryRuntime(args.model, threads=args.threads, record_layers=False)
    else:
        runtime = runtime_factory(args)
    start_load = resident.host_loadavg_1m()
    try:
        result = resident.run_sweep(runtime, settings, sizes=sizes, warmup=warmup, repeats=repeats)
    finally:
        runtime.close()
    record = {"device": args.device, "model": args.model, "settings": {k: v for k, v in settings.items() if k != "duration"}, "warmup_runs": warmup,
              "repeats": repeats, "host_loadavg_1m_at_start": start_load, "host_loadavg_1m_at_end": resident.host_loadavg_1m(), **result}
    out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    for row in result["sizes"]:
        main_part = row["parts"][resident.MAIN_PART]
        print(f"{args.device} {row['seconds']:g} s: {row['label']}, {main_part['n']} run(s), generate+decode min {main_part['min']} median {main_part['median']} "
              f"max {main_part['max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
