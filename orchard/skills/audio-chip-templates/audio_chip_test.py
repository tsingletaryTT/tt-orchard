# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The hardware test of the audio stage 2: run `chip_generate.py` once for each model in a config file, in its own process, under the lease the
supervisor holds, and record how each run ended. A process loads the model and the chip once, discards the warm-up runs and records every repeat of
its model (the resident runtime, orchard/resident.py), so a repeat measures generation and not a reload.

    python audio_chip_test.py --config audio_chip_config.json

The config (written by the supervisor, orchard/audio_stages.py) holds the shared settings (prompt, duration, seed, sample size), the library
checkout and the Hugging Face cache directory, the number of warm-up runs (`warmup`), and a list of runs (name, model, steps, cfg_scale, sampler),
one for each repeat of each model. One process per model keeps a hang or a crash of one model from taking the other's runs with it. The result is
`chip_test.json` beside the config; the exit status is 0 only when every run ended with status 0 and wrote its manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PER_RUN_TIMEOUT_S = 1800


def groups(runs: list[dict]) -> list[list[dict]]:
    """The runs of the config grouped by model, in order. Each group is one process: its first run's name is the run directory and the later names must
    be the repeat names (`<name>-r<k>`), because chip_generate.py records repeat k in that sibling directory."""
    out: dict[str, list[dict]] = {}
    for run in runs:
        out.setdefault(run["model"], []).append(run)
    for model, group in out.items():
        first = group[0]["name"]
        for k, run in enumerate(group):
            want = first if k == 0 else f"{first}-r{k}"
            if run["name"] != want:
                raise SystemExit(f"run {run['name']!r} of {model} should be named {want!r}: repeats of one model are recorded as <name>-r<k>")
        if len({tuple(run.get(key) for key in ("steps", "cfg_scale", "sampler", "prompt") if key != "prompt" or "prompt" in run) for run in group}) != 1:
            raise SystemExit(f"the runs of {model} do not share their settings, so one process cannot repeat them")
    return list(out.values())


def run_group(cfg: dict, group: list[dict], out_root: Path, log_dir: Path, timeout_s: float) -> list[dict]:
    """One process for all the repeats of one model: it loads the model and the chip once, discards `warmup` runs and records `len(group)` runs
    (orchard/resident.py). A crash or a hang takes down that model's runs only."""
    first = group[0]
    out = out_root / first["name"]
    cmd = [sys.executable, str(HERE / "chip_generate.py"), "--model", first["model"], "--out", str(out), "--prompt", first.get("prompt", cfg["prompt"]),
           "--duration", str(cfg["duration"]), "--steps", str(first["steps"]), "--cfg-scale", str(first["cfg_scale"]), "--seed", str(cfg["seed"]),
           "--sampler", first["sampler"], "--sample-size", str(cfg["sample_size"]), "--repeats", str(len(group)),
           "--warmup", str(int(cfg.get("warmup", 0)))]
    if cfg.get("chip_decoder"):
        cmd.append("--chip-decoder")                # the autoencoder decoder's transformer on the chip too (chip_decoder.py)
    env = dict(os.environ)
    env["PYTHONPATH"] = cfg["src_dir"]
    env["HF_HOME"] = cfg["hf_home"]
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN"):
        env.pop(name, None)
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timed_out = False
    with open(log_dir / f"{first['name']}.log", "wb") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout_s)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    seconds = round(time.time() - t0, 3)
    return [{"name": run["name"], "returncode": code, "timed_out": timed_out, "seconds": seconds,
             "manifest": (out_root / run["name"] / "manifest.json").is_file(), "out": str(out_root / run["name"]), "process": first["name"]}
            for run in group]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out_root = cfg_path.parent / cfg.get("out_dir", "chip")
    per_run = float(cfg.get("per_run_timeout_s", PER_RUN_TIMEOUT_S))
    results: list[dict] = []
    for group in groups(cfg["runs"]):
        results += run_group(cfg, group, out_root, cfg_path.parent / "log", per_run * (len(group) + int(cfg.get("warmup", 0))))
    (cfg_path.parent / "chip_test.json").write_text(json.dumps({"runs": results}, indent=1) + "\n", encoding="utf-8")
    ok = all(r["returncode"] == 0 and r["manifest"] for r in results)
    print(json.dumps({"ok": ok, "runs": results}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
