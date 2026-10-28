# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The hardware test of the audio stage 2: run `chip_generate.py` once for each run in a config file, each in its own process, under the lease the
supervisor holds, and record how each ended.

    python audio_chip_test.py --config audio_chip_config.json

The config (written by the supervisor, orchard/audio_stages.py) holds the shared settings (prompt, duration, seed, sample size), the library
checkout and the Hugging Face cache directory, and a list of runs (name, model, steps, cfg_scale, sampler). One process per run keeps a hang or a
crash of one run from taking the others with it, and gives each its own device open. The result is `chip_test.json` beside the config; the exit
status is 0 only when every run ended with status 0 and wrote its manifest.
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


def run_one(cfg: dict, run: dict, out_root: Path, log_dir: Path, timeout_s: float) -> dict:
    out = out_root / run["name"]
    cmd = [sys.executable, str(HERE / "chip_generate.py"), "--model", run["model"], "--out", str(out), "--prompt", cfg["prompt"],
           "--duration", str(cfg["duration"]), "--steps", str(run["steps"]), "--cfg-scale", str(run["cfg_scale"]), "--seed", str(cfg["seed"]),
           "--sampler", run["sampler"], "--sample-size", str(cfg["sample_size"])]
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
    with open(log_dir / f"{run['name']}.log", "wb") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout_s)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    return {"name": run["name"], "returncode": code, "timed_out": timed_out, "seconds": round(time.time() - t0, 3),
            "manifest": (out / "manifest.json").is_file(), "out": str(out)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    out_root = cfg_path.parent / cfg.get("out_dir", "chip")
    results = [run_one(cfg, run, out_root, cfg_path.parent / "log", float(cfg.get("per_run_timeout_s", PER_RUN_TIMEOUT_S)))
               for run in cfg["runs"]]
    (cfg_path.parent / "chip_test.json").write_text(json.dumps({"runs": results}, indent=1) + "\n", encoding="utf-8")
    ok = all(r["returncode"] == 0 and r["manifest"] for r in results)
    print(json.dumps({"ok": ok, "runs": results}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
