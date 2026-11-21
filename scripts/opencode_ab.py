#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run the pre-registered pilot or batch of the OpenCode v2 against orchard comparison, one attempt at a time.

    python3 scripts/opencode_ab.py pilot|batch [--root DIR]

Each attempt is one `kernel-probe --attempts 1` invocation: `O` runs scripts/opencode_probe.py (OpenCode as the agent loop), `R` runs the same
script in `control` mode (orchard's own agent loop). Both run the probe code of this checkout. One line per attempt goes to
`ROOT/<phase>/timeline.jsonl` with its UTC start and end, the return code and the attempt's record (read back from the probe's attempt.json).
An attempt that is already in the timeline is skipped on a rerun. The driver stops after two attempts in a row whose status is `error`.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "opencode_probe.py"
BRINGUP = Path.home() / "code" / "tt-orchard" / "config" / "bringup.toml"        # read only: the probe wants a config path
COMMON = ["kernel-probe", "--config", str(BRINGUP), "--attempts", "1", "--temperature", "0.7", "--phase", "sim", "--library", "none",
          "--no-agent-hw", "--endpoint", "http://127.0.0.1:8011/v1", "--model", "Qwen/Qwen3-Coder-Next", "--allow-stale-skills",
          "--style", "plain"]
# (op, pair number, harness order). Harness order flips on every pair (ABBA).
PILOT = [("add", 1, "OR"), ("center", 1, "RO")]
BATCH = ([("add", i, "OR" if i % 2 else "RO") for i in (1, 2)] + [("center", i, "OR" if i % 2 else "RO") for i in (1, 2, 3, 4)]
         + [("row_sum", i, "OR" if i % 2 else "RO") for i in (1, 2, 3, 4)])
SEED_BASE = {"pilot": 2026112000, "batch": 2026112100}


def plan(phase: str) -> list:
    out = []
    for idx, (op, n, order) in enumerate(PILOT if phase == "pilot" else BATCH, start=1):
        for h in order:
            out.append({"phase": phase, "pair": idx, "op": op, "n": n, "harness": h, "seed": SEED_BASE[phase] + idx})
    return out


def read_attempt(outdir: Path) -> dict | None:
    found = sorted(glob.glob(str(outdir / "**" / "attempt.json"), recursive=True))
    for f in found:
        try:
            return json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
    return None


def main(argv: list) -> int:
    phase = argv[0] if argv else "pilot"
    root = Path(argv[argv.index("--root") + 1]) if "--root" in argv else Path.home() / "orchard-runs" / "ocprobe"
    tdir = root / phase
    tdir.mkdir(parents=True, exist_ok=True)
    timeline = tdir / "timeline.jsonl"
    done = set()
    if timeline.exists():
        for line in timeline.read_text().splitlines():
            d = json.loads(line)
            done.add((d["op"], d["n"], d["harness"]))
    errors_in_a_row = 0
    for step in plan(phase):
        key = (step["op"], step["n"], step["harness"])
        if key in done:
            continue
        outdir = tdir / ("opencode" if step["harness"] == "O" else "orchard") / f"{step['op']}-{step['n']}"
        args = ([] if step["harness"] == "O" else ["control"]) + COMMON + ["--op", step["op"], "--seed", str(step["seed"]), "--out", str(outdir)]
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t0 = time.monotonic()
        log = tdir / f"{step['harness']}-{step['op']}-{step['n']}.log"
        with open(log, "wb") as lf:
            rc = subprocess.run([sys.executable, str(SCRIPT), *args], stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(REPO)).returncode
        rec = {**step, "started_utc": started, "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "driver_wall_s": round(time.monotonic() - t0, 1), "returncode": rc, "out": str(outdir), "log": str(log),
               "attempt": read_attempt(outdir)}
        with open(timeline, "a") as f:
            f.write(json.dumps(rec) + "\n")
        status = (rec["attempt"] or {}).get("status")
        errors_in_a_row = errors_in_a_row + 1 if (status == "error" or rec["attempt"] is None) else 0
        if errors_in_a_row >= 2:
            print("stopping: two attempts in a row ended in error or left no record", flush=True)
            return 3
    print("plan finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
