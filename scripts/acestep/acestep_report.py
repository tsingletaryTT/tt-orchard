# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Summarises ACE-Step runs made by `acestep_run.py`: sound checks at 48 kHz, agreement of a chip run with a CPU reference, and a length sweep
as a table with the benchmark label of `orchard/resident.py`.

    python acestep_report.py sound RUN_DIR...                    sound checks of each run's recorded waveform (48 kHz)
    python acestep_report.py agree REF_RUN CAND_RUN [--bars FILE]  `audio_reference.compare_runs` (cross precision) plus the waveform's log-spectrum distance
    python acestep_report.py sweep SWEEP_JSON                    the sweep table: min, median, max of each part per length, real-time factor, label

Standard library, numpy and soundfile only (no torch, no ttnn).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from orchard import audio_reference as ar                    # noqa: E402
from orchard import audio_stages as st                       # noqa: E402
from orchard import resident                                 # noqa: E402

SAMPLE_RATE = 48000       # ACE-Step's output rate (orchard/audio_stages.py uses Stable Audio's 44.1 kHz)


def sound(run_dirs) -> int:
    for d in run_dirs:
        with ar.load_run(d) as run:
            wave = np.asarray(run.array("waveform", "audio"))
            dur = run.settings["duration"]
        r = st.sound_checks(wave, SAMPLE_RATE, dur)
        print(json.dumps({"run": Path(d).name, **r}))
    return 0


def agree(ref_dir, cand_dir, bars_file=None) -> int:
    bars = None
    if bars_file:
        floor = json.loads(Path(bars_file).read_text())
        bars = {k: ar.Bars(**v) for k, v in floor["bars"].items()} if "bars" in floor else None
    with ar.load_run(ref_dir) as ref, ar.load_run(cand_dir) as cand:
        rep = ar.compare_runs(ref, cand, bars, cross_precision=True)
        wd = ar.log_spectrum_distance(np.asarray(ref.array("waveform", "audio")), np.asarray(cand.array("waveform", "audio")))
    rep["waveform_log_spectrum_distance"] = wd
    print(json.dumps({"ok": rep["ok"], "failed": [f"{r['kind']}/{r['name']}" for r in rep["results"] if not r["ok"]], "problems": rep["problems"],
                      "waveform_log_spectrum_distance": wd,
                      "worst": {k: max((r[k] for r in rep["results"] if r.get("comparable")), default=None) for k in ("rel_l2",)}}, indent=1))
    return 0 if rep["ok"] else 1


def sweep(path) -> int:
    data = json.loads(Path(path).read_text())
    print(f"device {data['device']}  seed {data['seed']}  load {data.get('load', {}).get('chip_open_upload_vae_warm_s')} s (chip open, weights, VAE warm-up)")
    print(f"{'length_s':>8} {'n':>2} {'warmup':>6} {'total min/med/max':>22} {'dit med':>8} {'vae med':>8} {'rest med':>9} {'xRT':>6}  label")
    for row in data["rows"]:
        runs = row["runs"]
        tot = resident.spread([r["generate_and_decode_s"] for r in runs])
        dit = resident.spread([r["dit_s"] for r in runs]) if all("dit_s" in r for r in runs) else None
        vae = resident.spread([r["vae_s"] for r in runs])
        rest = [r["generate_and_decode_s"] - r.get("dit_s", 0) - r["vae_s"] for r in runs]
        label = resident.label_for(len(runs), row["warmup"])
        print(f"{row['duration_s']:>8g} {len(runs):>2} {row['warmup']:>6} {tot['min']:>7.2f}/{tot['median']:>6.2f}/{tot['max']:>6.2f} "
              f"{(dit['median'] if dit else float('nan')):>8.2f} {vae['median']:>8.2f} {float(np.median(rest)):>9.2f} {row['duration_s'] / tot['median']:>6.1f}  {label}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sound")
    s.add_argument("runs", nargs="+")
    a = sub.add_parser("agree")
    a.add_argument("ref")
    a.add_argument("cand")
    a.add_argument("--bars")
    w = sub.add_parser("sweep")
    w.add_argument("file")
    args = p.parse_args(argv)
    if args.cmd == "sound":
        return sound(args.runs)
    if args.cmd == "agree":
        return agree(args.ref, args.cand, args.bars)
    return sweep(args.file)


if __name__ == "__main__":
    raise SystemExit(main())
