# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The resident-runtime contract and the benchmark rules (issue 174). Standard library only.

A runtime that serves a model on a chip (or on a CPU, for a comparison) has two calls:

    load()                    once per process. Reads the model, opens the device, uploads the weights. Returns the timings of that work.
                              A second call is refused.
    generate(settings, out_dir=None, extra_timings=None)
                              as many times as wanted on the loaded model. Returns {"timings": {...}, "output_sha256": ...}. The same settings
                              (prompt, duration, steps, guidance scale, sampler, seed, sample size) give the same output on every call, whether
                              the call is the first or the hundredth. `out_dir` is where a runtime that records its run puts it. `extra_timings` are numbers
                              the caller wants written next to the run's own timings in that record (a one-shot script passes the load's);
                              they are not part of the returned timings.
    close()                   releases the device. A call to generate after close is refused.

The one-shot scripts (orchard/skills/audio-chip-templates/chip_generate.py) are thin callers of this contract: they call `load()` once and
`generate()` once. Repeats, sweeps and a server call `generate()` again on the same loaded runtime, so a repeat measures generation and not
a reload (a reload is about 6 s of model load and 1.2 s of weight upload per process on the audio chip path).

Benchmark rules, written down here and used by the stages and the scripts:

    WARMUP_RUNS   one discarded `generate` before the measured ones, for each size, because the first call at a size compiles kernels and fills
                  caches that depend on the size.
    MIN_REPEATS   three measured runs. Fewer is labelled "single sample" and nothing may call it a benchmark. Three or more with fewer than WARMUP_RUNS
                  discarded runs (or with no statement of how many) is labelled "unwarmed" and is not a benchmark either.
    spread        min, median and max over the measured runs, for each timed part the runtime names.
    host load     the host's one-minute load average, read next to each measured run, because the host is shared.
    SWEEP_SECONDS the stated sizes of a length sweep; MAX_SECONDS is the longest clip the model makes (its sample size is 120 s at 44.1 kHz).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path

WARMUP_RUNS = 1
MIN_REPEATS = 3
SWEEP_SECONDS = (10.0, 30.0, 60.0, 120.0)
MAX_SECONDS = 120.0
LABEL_BENCHMARK = "benchmark"
LABEL_SINGLE = "single sample"
LABEL_UNWARMED = "unwarmed"
LABEL_NONE = "no measurement"
BENCH_HEADING = "### Timing repeats: benchmark"          # the heading the operator bundle gives timings over at least MIN_REPEATS repeats ...
UNWARMED_HEADING = "### Timing repeats: unwarmed"         # three or more repeats with no warm-up stated
SINGLE_HEADING = "### Timing repeats: single sample"      # ... and the heading it gives anything less (orchard/skills/operator-bundle-templates/build_bundle.py)
MAIN_PART = "generate_and_decode_s"          # the part every runtime names: the whole generate call and decode


class ResidentError(Exception):
    """A misuse of the contract, or a benchmark that cannot be summarised."""


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def spread(values) -> dict:
    """`{"n", "min", "median", "max"}` of a non-empty list of finite numbers. Anything else is refused: a missing or odd timing is not averaged away."""
    values = list(values)
    if not values or not all(_is_number(v) for v in values):
        raise ResidentError("spread needs a non-empty list of finite numbers")
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def label_for(n_measured: int, warmup=None) -> str:
    """What a set of `n_measured` timed runs, made after `warmup` discarded runs, may be called. `warmup` None means it was not stated, which does not
    meet the rule."""
    if n_measured < 1:
        return LABEL_NONE
    if n_measured < MIN_REPEATS:
        return LABEL_SINGLE
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < WARMUP_RUNS:
        return LABEL_UNWARMED
    return LABEL_BENCHMARK


def repeat_dir(out, k: int) -> Path:
    """The directory of repeat `k` for the run directory `out`: `out` itself for 0, a sibling `<name>-r<k>` after it (orchard/audio_plan.py repeat_name)."""
    out = Path(out)
    return out if k == 0 else out.with_name(f"{out.name}-r{k}")


def host_loadavg_1m(loadavg=os.getloadavg):
    """The one-minute load average, or None when the host will not say."""
    try:
        return float(loadavg()[0])
    except (OSError, AttributeError, IndexError, TypeError, ValueError):
        return None


# ---- a fake runtime for tests -------------------------------------------------------------------------------------------------------

class FakeRuntime:
    """A runtime with no model: it keeps the contract and a deterministic output (a hash of the settings), and reports the timings it was told to.

    `times` is the `generate_and_decode_s` of each call in order (the last repeats); `parts` maps another part name to such a list; `drop_part_on_call`
    (1-based) leaves `generate_and_decode_s` out of that call's timings; `drift` mixes the call number into the output, to show a runtime that does
    not keep its promise; `no_digest` returns no output hash, as a runtime that does not compute one would."""
    SETTING_KEYS = ("prompt", "duration", "steps", "cfg_scale", "sampler", "seed", "sample_size")

    def __init__(self, *, times=None, parts=None, drop_part_on_call=None, drift=False, no_digest=False):
        self.times = list(times) if times else [1.0]
        self.parts = {k: list(v) for k, v in (parts or {}).items()}
        self.drop_part_on_call, self.drift, self.no_digest = drop_part_on_call, drift, no_digest
        self.loads = self.generate_calls = 0
        self.closed = False
        self.load_timings = {"model_load_s": 6.0, "chip_weights_upload_s": 1.2}
        self.seen_durations: list = []
        self.seen_seeds: list = []
        self.out_dirs: list = []
        self.extras: list = []

    def load(self) -> dict:
        if self.loads:
            raise ResidentError("the runtime is already loaded")
        self.loads += 1
        return dict(self.load_timings)

    def generate(self, settings: dict, out_dir=None, extra_timings=None) -> dict:
        if self.closed:
            raise ResidentError("the runtime is closed")
        if not self.loads:
            raise ResidentError("load() has not been called")
        i = self.generate_calls
        self.generate_calls += 1
        self.seen_durations.append(settings["duration"])
        self.seen_seeds.append(settings["seed"])
        self.out_dirs.append(out_dir)
        self.extras.append(extra_timings)
        timings = {MAIN_PART: (self.times[i] if i < len(self.times) else self.times[-1])}
        for name, values in self.parts.items():
            timings[name] = values[i] if i < len(values) else values[-1]
        if self.drop_part_on_call == i + 1:
            del timings[MAIN_PART]
        key = {k: settings[k] for k in self.SETTING_KEYS}
        if self.drift:
            key["call"] = i
        return {"timings": timings, "output_sha256": None if self.no_digest else hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()}

    def close(self) -> None:
        self.closed = True


def same_output(runtime, settings: dict, calls: int = 3) -> dict:
    """Call `generate` `calls` times with the same settings on one loaded runtime and say whether the output hashes agree. A call that returns no hash
    is refused: missing hashes would all look alike."""
    shas = [runtime.generate(settings).get("output_sha256") for _ in range(calls)]
    if not all(isinstance(s, str) and s for s in shas):
        raise ResidentError("the runtime returned no output hash, so the outputs cannot be compared")
    return {"calls": calls, "sha256": shas, "same": len(set(shas)) == 1}


LOAD_KEYS = ("model_load_s", "chip_weights_upload_s")        # reported once per process, in the first recorded run
NOT_PARTS = LOAD_KEYS + ("host_loadavg_1m",)                  # timings that are not a per-run part


def summarise_runs(timings_list, warmup=None) -> dict:
    """The record of a set of measured runs, from each run's timings (a list of dicts of numbers). The label says what the set may be called (fewer
    than MIN_REPEATS runs: "single sample"). `parts` holds min, median and max of every timed part that all the runs report; a part only some runs
    report is named in `incomplete_parts` and left out of `parts`. `load` is the load's timings, read from the first run, where the single loaded process
    reports them. `host_loadavg_1m` lists each run's load average (None where it was not recorded). `warmup` is the number of discarded runs that came
    first (it decides the label, with the count)."""
    runs = [t for t in timings_list if isinstance(t, dict)]
    n = len(runs)
    keys = sorted({k for t in runs for k, v in t.items() if isinstance(k, str) and k not in NOT_PARTS and _is_number(v)})
    complete = [k for k in keys if all(_is_number(t.get(k)) for t in runs)]
    first = runs[0] if runs else {}
    return {"label": label_for(n, warmup), "repeats": n, "parts": {k: spread([t[k] for t in runs]) for k in complete},
            "incomplete_parts": [k for k in keys if k not in complete],
            "load": {k: first[k] for k in LOAD_KEYS if _is_number(first.get(k))},
            "host_loadavg_1m": [t["host_loadavg_1m"] if _is_number(t.get("host_loadavg_1m")) else None for t in runs]}


# ---- the benchmark -----------------------------------------------------------------------------------------------------------------

def run_benchmark(runtime, settings: dict, *, warmup: int = WARMUP_RUNS, repeats: int = MIN_REPEATS, loadavg=os.getloadavg,
                  required_parts=(MAIN_PART,), out_dirs=None, extra_timings=None, load_timings: dict | None = None) -> dict:
    """Load the runtime if it is not loaded yet, discard `warmup` runs, measure `repeats` runs, and return the label, the spread of every timed part and
    each run's timings with the host load average read right after it. `out_dirs(k)` (optional) gives the directory a measured run records into and
    `extra_timings(k)` the numbers written next to its timings there.
    The runtime stays open for the caller. `load_timings` is the result of a `load()` the caller already made."""
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ResidentError("warmup must be 0 or more")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ResidentError("repeats must be at least 1")
    load = load_timings if load_timings is not None else runtime.load()
    for _ in range(warmup):
        runtime.generate(settings)                          # discarded: kernel compile and size-dependent caches
    runs = []
    for k in range(repeats):
        res = runtime.generate(settings, out_dir=None if out_dirs is None else out_dirs(k),
                               extra_timings=None if extra_timings is None else extra_timings(k))
        runs.append({"timings": dict(res["timings"]), "host_loadavg_1m": host_loadavg_1m(loadavg), "output_sha256": res.get("output_sha256")})
    names = sorted({name for run in runs for name in run["timings"]} | set(required_parts))
    for name in names:
        for k, run in enumerate(runs):
            if not _is_number(run["timings"].get(name)):
                raise ResidentError(f"part {name!r} is missing or not a number in measured run {k}")
    return {"label": label_for(repeats, warmup), "warmup_runs": warmup, "repeats": repeats, "load": dict(load),
            "parts": {name: spread([run["timings"][name] for run in runs]) for name in names}, "runs": runs}


def run_sweep(runtime, settings: dict, *, sizes=None, warmup: int = WARMUP_RUNS, repeats: int = MIN_REPEATS, loadavg=os.getloadavg,
              required_parts=(MAIN_PART,)) -> dict:
    """`run_benchmark` at each stated size (clip length in seconds) on one loaded runtime, the warm-up repeated for each size. The seed and every other
    setting stay the same. A size over MAX_SECONDS, a size that is not positive and a size that is not a finite number are is refused before anything runs."""
    sizes = SWEEP_SECONDS if sizes is None else tuple(sizes)
    if not sizes:
        raise ResidentError("a sweep needs at least one size")
    for s in sizes:
        if not _is_number(s) or s <= 0 or s > MAX_SECONDS:
            raise ResidentError(f"a sweep size must be a number of seconds above 0 and at most {MAX_SECONDS:g}, got {s!r}")
    load = runtime.load()
    rows = []
    for s in sizes:
        res = run_benchmark(runtime, {**settings, "duration": float(s)}, warmup=warmup, repeats=repeats, loadavg=loadavg,
                            required_parts=required_parts, load_timings=load)
        rows.append({"seconds": float(s), **{k: v for k, v in res.items() if k != "load"}})
    return {"load": dict(load), "sizes": rows}
