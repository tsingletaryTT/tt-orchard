# Benchmark rules for a new-architecture bring-up

These rules came out of issue 174. The code is in `orchard/resident.py`, and the audio stage 2 and the operator bundle use it.

## The resident-runtime contract

A runtime that serves a model has three calls.

| Call | What it does |
|---|---|
| `load()` | Once per process. Reads the model, opens the device and uploads the weights. Returns the timings of that work. A second call is refused. |
| `generate(settings, out_dir=None, extra_timings=None)` | As many times as wanted on the loaded model. Returns the timings of that call and a hash of its output. The same settings give the same output on every call. |
| `close()` | Releases the device. `generate` after `close` is refused. |

A one-shot script is a thin caller: `load()` once, `generate()` once. Repeats, sweeps and a server call `generate()` again on the same loaded
runtime, so a repeat measures generation and not a reload. `FakeRuntime` in `orchard/resident.py` keeps the contract with no model, for tests of
anything that calls it. The audio runtimes are in `orchard/skills/audio-chip-templates/audio_runtimes.py`.

## What a benchmark is

| Rule | Value | Where it is enforced |
|---|---|---|
| Warm-up | 1 discarded run in the same process, before the measured runs, and again at each size of a sweep | `WARMUP_RUNS`; the stage 2 config states it and the stage 2 gate compares it with the result |
| Measured repeats | At least 3 | `MIN_REPEATS` |
| Spread | min, median and max of every timed part, over the measured runs | `summarise_runs`; the stage 2 gate recomputes it from the repeats |
| Label | 3 or more measured repeats: "benchmark". Fewer: "single sample". | `label_for`; the bundle heads the timings "benchmark" or "single sample", and the stage 8 gate refuses the benchmark heading when the result file lists fewer than 3 measured repeats of a model |
| Load | Reported once per process, by the first recorded run, and never as a per-run part | the stage 2 gate refuses a load figure in a later repeat |
| Host load | The host's one-minute load average, recorded with each measured run | `host_loadavg_1m` in each run's timings |
| Length sweep | 10, 30, 60 and 120 seconds, on one loaded process; nothing above 120 seconds | `SWEEP_SECONDS`, `MAX_SECONDS`; `audio_bench.py` |

A timed part that one measured run lacks is named in the record as incomplete and is not given a spread.

## Queue and concurrency

`python -m orchard.queue_probe` sends 1, 2 and 4 requests at the same instant to any local server that speaks the audio job API (`POST /v1/audio/jobs`,
`GET /v1/audio/jobs/<id>`, `DELETE`, `GET /v1/health`). It records the status of each request, the queue positions seen, the wait of each job, whether the
jobs ran one at a time and in order, and the load average. It probes only a server on the loopback address, never on a port that other work uses.
