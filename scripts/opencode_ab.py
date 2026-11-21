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
import re
import subprocess
import urllib.request
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
# Run log addendum 1: center and row_sum go from 4 to 6 attempts per harness when the pre-registered projection is at most 4.5 hours (it is: 4.17).
BATCH = ([("add", i, "OR" if i % 2 else "RO") for i in (1, 2)] + [("center", i, "OR" if i % 2 else "RO") for i in (1, 2, 3, 4, 5, 6)]
         + [("row_sum", i, "OR" if i % 2 else "RO") for i in (1, 2, 3, 4, 5, 6)])
MAX_HOURS = 5.0                      # the pre-registered cap: no new attempt starts after this many hours from the driver's start
SEED_BASE = {"pilot": 2026112000, "batch": 2026112100}
METRICS_URL = "http://127.0.0.1:8011/metrics"
_METRIC_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfNa]+)$")
# Cumulative counters of the model server (vLLM's /metrics). One client at a time, so the change across an attempt belongs to it; the number of
# requests the server counted is compared with the harness's own request count to show that no other client was served in between.
_METRICS = {"requests": ("vllm:request_success_total", None), "requests_hit_reply_cap": ("vllm:request_success_total", 'finished_reason="length"'),
            "prompt_tokens": ("vllm:prompt_tokens_total", None), "prompt_tokens_cache_hit": ("vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'),
            "prompt_tokens_computed": ("vllm:prompt_tokens_by_source_total", 'source="local_compute"'),
            "generation_tokens": ("vllm:generation_tokens_total", None), "prefill_s": ("vllm:request_prefill_time_seconds_sum", None),
            "decode_s": ("vllm:request_decode_time_seconds_sum", None), "queue_s": ("vllm:request_queue_time_seconds_sum", None),
            "first_token_s": ("vllm:time_to_first_token_seconds_sum", None), "preemptions": ("vllm:num_preemptions_total", None)}


def metrics_snapshot() -> dict:
    try:
        text = urllib.request.urlopen(METRICS_URL, timeout=10).read().decode("utf-8", "replace")
    except Exception as exc:                                  # a failed read is recorded, never fatal
        return {"error": f"{exc.__class__.__name__}"}
    sums = {k: 0.0 for k in _METRICS}
    for line in text.splitlines():
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", m.group(3)
        for key, (metric, label) in _METRICS.items():
            if name == metric and (label is None or label in labels):
                try:
                    sums[key] += float(value)
                except ValueError:
                    pass
    return sums


def metrics_delta(before: dict, after: dict) -> dict:
    if "error" in before or "error" in after:
        return {"error": before.get("error") or after.get("error")}
    return {k: round(after[k] - before[k], 3) for k in before}


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
    began = time.monotonic()
    for step in plan(phase):
        if time.monotonic() - began > MAX_HOURS * 3600:
            print("stopping: the pre-registered 5-hour cap is reached", flush=True)
            return 4
        key = (step["op"], step["n"], step["harness"])
        if key in done:
            continue
        outdir = tdir / ("opencode" if step["harness"] == "O" else "orchard") / f"{step['op']}-{step['n']}"
        args = ([] if step["harness"] == "O" else ["control"]) + COMMON + ["--op", step["op"], "--seed", str(step["seed"]), "--out", str(outdir)]
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        before = metrics_snapshot()
        load_start = Path("/proc/loadavg").read_text().split()[:3]
        t0 = time.monotonic()
        log = tdir / f"{step['harness']}-{step['op']}-{step['n']}.log"
        with open(log, "wb") as lf:
            rc = subprocess.run([sys.executable, str(SCRIPT), *args], stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                cwd=str(REPO)).returncode
        after = metrics_snapshot()
        rec = {**step, "server_delta": metrics_delta(before, after), "loadavg_start": load_start,
               "loadavg_end": Path("/proc/loadavg").read_text().split()[:3], "started_utc": started, "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
