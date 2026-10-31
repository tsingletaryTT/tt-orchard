#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Time the model library's own float32 run on the host CPU, for the same clips the chip server makes (the CPU floor of a card's numbers).

    PYTHONPATH=extra_code venv/bin/python tools/cpu_floor.py --durations 10 30 --repeats 3 --json cpu-floor.json

It uses the same code and the same pinned weights as the server (`stable_audio_asgi.resolve_weights`, then `audio_runtimes.LibraryRuntime`, which does not
import ttnn and opens no device). It loads the model once, makes `--warmup` unrecorded clips, then `--repeats` recorded ones for each duration, and writes the
median, the least and the greatest generation time with the host's load average at each run. The clips are not kept.
"""
import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--durations", type=float, nargs="+", default=[10.0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default="A warm acoustic guitar melody with soft percussion, 100 BPM")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0: the library's default)")
    ap.add_argument("--json")
    args = ap.parse_args(argv)
    import stable_audio_asgi as server                       # from extra_code/ (PYTHONPATH)
    server.resolve_weights(os.environ, server._download)
    root = server._roots()
    sys.path.insert(0, str(root))
    rts = server._load_file(root / "orchard" / "skills" / "audio-chip-templates" / "audio_runtimes.py", "audio_runtimes")
    rt = rts.LibraryRuntime("small-music", threads=args.threads, record_layers=False)
    t0 = time.time()
    rt.load()
    out = {"load_s": round(time.time() - t0, 3), "settings": {"prompt": args.prompt, "steps": args.steps, "cfg_scale": args.cfg_scale, "seed": args.seed},
           "device": "cpu, float32", "threads": args.threads or "default", "runs": {}}
    for duration in args.durations:
        settings = {"prompt": args.prompt, "duration": duration, "steps": args.steps, "cfg_scale": args.cfg_scale, "seed": args.seed,
                    "sample_size": server.SAMPLE_SIZE}
        for _ in range(args.warmup):
            rt.generate(settings)
        times = []
        for _ in range(args.repeats):
            t = time.time()
            result = rt.generate(settings)
            times.append({"generate_and_decode_s": result["timings"]["generate_and_decode_s"], "wall_s": round(time.time() - t, 3),
                          "host_loadavg_1m": os.getloadavg()[0]})
        g = [r["generate_and_decode_s"] for r in times]
        out["runs"][str(duration)] = {"runs": times, "median_s": round(statistics.median(g), 3), "min_s": min(g), "max_s": max(g)}
        print(f"{duration} s clip on the CPU: median {out['runs'][str(duration)]['median_s']} s over {args.repeats} runs (min {min(g)}, max {max(g)})", flush=True)
    rt.close()
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
