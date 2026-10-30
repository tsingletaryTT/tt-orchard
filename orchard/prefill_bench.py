# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Guards for a hardware timing run of the Granite prefill paths (issue 290). No chip code lives here, so the guards are tested on the CPU.

A timing run once printed nothing for 400 seconds, was ended by its outer timeout and left a chip unreadable. This module keeps three rules:

  * every iteration (the warm-up too) writes a log line before it starts and another after it ends, each flushed to disk, so a hang
    leaves a log whose last line says which iteration it was in;
  * every iteration has its own time limit. A watchdog thread writes a `hang` line that names the iteration and ends the process
    (`os._exit`, which works while the main thread is blocked inside a native call, where a Python signal handler would not run);
  * a run with trace capture on refuses to time anything until the log holds a finished warm-up line.

The worker that drives the chip is `scripts/granite_prefill_bench.py`. The whole process is meant to run under `timeout PROCESS_TIMEOUT_S`.
"""
from __future__ import annotations

import json
import os
import statistics
import threading
import time
from pathlib import Path

HANG_EXIT = 97                  # the exit status of a process ended by the watchdog
PROCESS_TIMEOUT_S = 570         # the outer `timeout` for one process: under ten minutes


class TraceWithoutWarmup(RuntimeError):
    """A trace-on run was asked to time an iteration before any warm-up finished."""


class BenchLog:
    """An append-only JSON-lines log. Each line is flushed and synced before `line` returns."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def line(self, event: str, **fields) -> None:
        rec = {"t": round(time.time(), 3), "event": event, **fields}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(x) for x in self.path.read_text(encoding="utf-8").splitlines() if x.strip()]


class Watchdog:
    """Ends the process when an iteration runs past its limit. `kill` is replaceable so a test can observe it."""

    def __init__(self, log: BenchLog, kill=None):
        self.log = log
        self.kill = kill or os._exit

    def arm(self, label: str, iteration: int, timeout_s: float, phase: str) -> threading.Timer:
        def fire():
            self.log.line("hang", label=label, iteration=iteration, phase=phase, timeout_s=timeout_s)
            self.kill(HANG_EXIT)
        timer = threading.Timer(timeout_s, fire)
        timer.daemon = True
        timer.start()
        return timer


def run_iteration(log: BenchLog, wd: Watchdog, label: str, iteration: int, timeout_s: float, fn, phase: str = "timed"):
    """Log `before`, run `fn` under the watchdog, log `after`. Returns (fn's value, seconds)."""
    log.line("before", label=label, iteration=iteration, phase=phase, timeout_s=timeout_s)
    timer = wd.arm(label, iteration, timeout_s, phase)
    t0 = time.perf_counter()
    try:
        value = fn()
    finally:
        timer.cancel()
    seconds = time.perf_counter() - t0
    log.line("after", label=label, iteration=iteration, phase=phase, seconds=round(seconds, 4))
    return value, seconds


def require_warmup(log: BenchLog, trace: bool) -> None:
    """With trace capture on, the log must already hold a finished warm-up (an `after` line of phase `warmup`)."""
    if not trace:
        return
    if not any(e["event"] == "after" and e.get("phase") == "warmup" for e in log.events()):
        raise TraceWithoutWarmup("a trace-on run needs a finished warm-up line in the log before any timed iteration")


def run_plan(log: BenchLog, wd: Watchdog, label: str, fn, warmups: int, runs: int, trace: bool, warmup_timeout: float, timeout: float) -> list[float]:
    """`warmups` warm-up iterations, then `runs` timed ones. Returns the seconds of the timed runs."""
    require_warmup_possible = trace and warmups < 1
    if require_warmup_possible:
        raise TraceWithoutWarmup("a trace-on run needs at least one warm-up iteration")
    for i in range(warmups):
        run_iteration(log, wd, label, i, warmup_timeout, fn, phase="warmup")
    secs = []
    for i in range(1, runs + 1):
        require_warmup(log, trace)
        secs.append(run_iteration(log, wd, label, i, timeout, fn)[1])
    return secs


def summarize(prompt_tokens: int, seconds: list[float]) -> dict:
    """The label `n` is the count of timed runs. Every run is kept, so a reader can see the spread."""
    tok = [round(prompt_tokens / s, 2) for s in seconds]
    return {"n": len(seconds), "prompt_tokens": prompt_tokens, "seconds": [round(s, 4) for s in seconds], "tok_s": tok,
            "tok_s_median": statistics.median(tok), "tok_s_min": min(tok), "tok_s_max": max(tok)}
