# Stable Audio 3 small music: resident runtime, warm-up rule, length sweep and queue probe (2026-10-28)

Issue 174. Raw results are in `docs/run-logs/evidence/2026-10-28-resident-benchmark/`. The rules are in `docs/benchmark-rules.md`. Paths are placeholders.

## What was built

| Piece | Where |
|---|---|
| The resident-runtime contract (`load()` once, `generate(settings)` many times, `close()`), a fake runtime, spread, labels, the benchmark and the length sweep | `orchard/resident.py` |
| The chip and CPU runtimes that keep the contract | `orchard/skills/audio-chip-templates/audio_runtimes.py` |
| A one-shot script, now a thin caller (`--warmup N --repeats M` run on a single loaded process) | `orchard/skills/audio-chip-templates/chip_generate.py` |
| The stage 2 hardware test, now one process for each model | `orchard/skills/audio-chip-templates/audio_chip_test.py` |
| The length sweep script | `orchard/skills/audio-chip-templates/audio_bench.py` |
| The queue and concurrency probe, for any local server that speaks `POST /v1/audio/jobs` | `orchard/queue_probe.py` |
| A timing record in `audio_result.json` and its gate, and the bundle's benchmark, unwarmed and single-sample sections and the stage 8 gate | `orchard/audio_stages.py`, `orchard/skills/operator-bundle-templates/build_bundle.py`, `orchard/stages.py` |

## The rules

One warm-up run is discarded in the same process before the measured runs, and again at each size of a sweep. At least 3 measured runs make a benchmark; 2 or fewer
are a single sample; 3 or more with no stated warm-up are labelled unwarmed. Every timed part gets min, median and max. The load (model load and weight upload) is reported
once, by the first recorded run. The host's one-minute load average is recorded with every measured run. Sizes of a sweep are 10, 30, 60 and 120 seconds, and 120 seconds is the maximum.

Why one warm-up run: the first run in a fresh process is slower in the transformer calls and about as fast as the rest in everything else. The table below
comes from four recorded runs on one loaded process with no warm-up (fine-tune, 8 steps; seconds).

| Clip length | Part | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|---|
| 10 s | generate and decode | 2.252 | 1.853 | 1.775 | 1.828 |
| 10 s | transformer calls | 0.709 | 0.295 | 0.300 | 0.308 |
| 120 s | generate and decode | 11.576 | 11.566 | 11.114 | 10.678 |
| 120 s | transformer calls | 0.938 | 0.813 | 0.807 | 0.818 |

Run 1 carries start-up cost in the transformer calls (0.4 s more at 10 s, 0.12 s more at 120 s). Runs 2 to 4 agree within the host noise. An earlier run of the same test
(made before the label rule below was added) showed the same pattern: transformer calls of 0.454 s and then 0.29 to 0.32 s at 10 s, and 0.959 s and then 0.83 to 0.84 s at 120 s. One warm-up run is enough for this model on this chip. A different model, a different
size range or a cold kernel cache may need more, and the rule is a number in one place (`WARMUP_RUNS`) to change then.

## The numbers do not change

The same seed gives the same arrays through the old one-shot path and the resident path. The check is bit for bit on every array in the run directory, comparing runs of the
code before this change (stage 2 of an earlier run, one process per run) with runs of the resident process (one warm-up, then three recorded runs).

| Model | Arrays compared | Resident repeat 1, 2 and 3 against the earlier one-shot run |
|---|---|---|
| Fine-tune, 8 steps | 56 | identical in every array |
| Base control, 50 steps | 132 | identical in every array |

The resident repeats also agree with one another. A test with the fake runtime states the same contract (`tests/test_resident.py`, `tests/test_audio_runtimes.py`).

## Fine-tune, length sweep on one chip, against the CPU

One loaded process on one Blackhole chip (`audio_bench.py`, one warm-up and 3 recorded runs at each size). The CPU runs are the same sweep in the reference environment on the
16-core host (`LibraryRuntime`, float32), one warm-up and 3 recorded runs at 10, 30 and 60 s. The 120 s CPU run was made once, with no warm-up, so it is a single sample. Prompt "A warm
acoustic guitar melody with soft percussion, 100 BPM", seed 42, 8 steps, guidance scale 1.0, ping-pong sampler. Seconds. The load average column is the host's one-minute load average when each recorded run ended.

| Clip length | Where | Runs | Generate and decode, min | median | max | Transformer calls, median | Host load average, per run |
|---|---|---|---|---|---|---|---|
| 10 s | chip | 3 (benchmark) | 1.563 | 1.702 | 1.742 | 0.281 | 6.2, 6.2, 6.2 |
| 10 s | CPU | 3 (benchmark) | 5.659 | 6.136 | 6.154 | n/a | 7, 7.5, 8.4 |
| 30 s | chip | 3 (benchmark) | 3.033 | 3.139 | 3.167 | 0.383 | 7, 8.1, 8.1 |
| 30 s | CPU | 3 (benchmark) | 9.509 | 10.73 | 11.16 | n/a | 10.6, 11.1, 11.3 |
| 60 s | chip | 3 (benchmark) | 6.157 | 6.169 | 6.201 | 0.534 | 11.5, 11, 11.3 |
| 60 s | CPU | 3 (benchmark) | 17.58 | 19.72 | 19.88 | n/a | 12.3, 14.3, 13.8 |
| 120 s | chip | 3 (benchmark) | 10.51 | 10.51 | 10.85 | 0.812 | 12.1, 12, 12.6 |
| 120 s | CPU | 1 (single sample) | 40.22 | 40.22 | 40.22 | n/a | 16 |

Spread of the other parts of the chip runs (seconds):

| Clip length | Part | min | median | max |
|---|---|---|---|---|
| 10 s | transformer calls | 0.279 | 0.281 | 0.286 |
| 10 s | T5Gemma encoder (host) | 0.038 | 0.038 | 0.048 |
| 30 s | transformer calls | 0.380 | 0.383 | 0.383 |
| 30 s | T5Gemma encoder (host) | 0.038 | 0.038 | 0.039 |
| 60 s | transformer calls | 0.533 | 0.534 | 0.535 |
| 60 s | T5Gemma encoder (host) | 0.039 | 0.039 | 0.041 |
| 120 s | transformer calls | 0.810 | 0.812 | 0.813 |
| 120 s | T5Gemma encoder (host) | 0.040 | 0.041 | 0.063 |

The load, once for the process: model load 5.7 s and weight upload 1.2 s. In every run the transformer made 8 passes.

What the table says. The chip run takes about 3.2 to 3.6 times less time than the CPU run at 10, 30 and 60 s (medians), and the single 120 s samples are 40.2 s on the CPU and 10.5 s on the chip.
The chip time grows with the clip length (1.7 s at 10 s, 10.5 s at 120 s), and the transformer calls are a small part of it (0.28 s of 1.7 s at 10 s, 0.81 s of 10.5 s at 120 s). The rest runs
on the host (the sampler and the autoencoder decoder), so the next speed step is the decoder (issue 161). The earlier single 120 s run in the first-runs log (16.5 s) is
about 6 s more than any of these three; the table cannot say why (see the limits).

## Base control on one chip

Same method (one warm-up, 3 recorded runs at each size, one loaded process), base model, 50 steps, guidance scale 7.0, Euler sampler (100 transformer passes). The host load average was 20 to 28 during
these runs.

| Clip length | Generate and decode, 3 runs | min | median | max | Transformer calls, 3 runs | Host load average, per run |
|---|---|---|---|---|---|---|
| 10 s | 6.686, 5.605, 5.484 | 5.484 | 5.605 | 6.686 | 4.075, 3.687, 3.711 | 28.2, 26.6, 25.3 |
| 120 s | 24.486, 24.548, 42.244 | 24.486 | 24.548 | 42.244 | 10.727, 10.309, 13.566 | 23, 20.7, 22.8 |

Load once: model load 7.9 s, weight upload 1.4 s. The third 120 s run took 42 s while the first two took 24.5 s. The cause was not found, and the load average was about the same for the three runs. The spread shows the jump, and a median of 3 hides it, so the table gives all three values.

## Stage 2 end to end

The stage 2 test (`audio_chip_test.py`) ran for both models from a real config (one process for each model, one warm-up, 3 recorded repeats), then `finish_chip` and the stage 2 gate ran on the
real directories with stage 1 copied from an earlier run. The gate returned no problems. The record has `warmup_runs` 1, each role labelled "benchmark" with 3 repeats, and the
spread of each part. This run's numbers (fine-tune generate and decode median 2.31 s, base control 6.26 s) were taken at a host load average of 11 to 12.6.

## Queue and concurrency probe

The probe (`orchard/queue_probe.py`) ran against the job server of the separate demo repository (read only, its default queue of 2 waiting jobs), started on a free port with a one-shot
`chip_generate.py` as its generator, so each clip is a new chip process that loads the model again. The server runs one job at a time. Requests are sent at the same instant from threads;
jobs are polled every 0.25 s.

| Requests at once | Accepted | Refused | Largest queue position seen | One at a time | Service time of each job (s) | Wait of each job (s) | Whole level (s) | Host load average, start to end |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | none | 0 | not judged (1 job) | 14.9 | 0.02 | 14.9 | 22.1 to 18.8 |
| 2 | 2 | none | 1 | yes | 17.1, 15.2 | 0.2, 17.4 | 32.5 | 18.8 to 17.3 |
| 4 | 3 | one, HTTP 429 | 2 | yes | 14.5, 14.7, 14.0 | 0.03, 14.7, 29.3 | 43.3 | 17.3 to 16.7 |

The server accepts one running job and two waiting ones, and refuses the next request at once with 429, as its documentation says. A waiting job waited about one service time for each job ahead of it.
The service time of about 14.5 s is a 10 s clip made by a process that loads the model. Inside a chip process the load, the upload and the generation add up to about 9 to 10 s in the runs above; the rest (starting the process, importing, opening and closing the device, writing the clip) was not timed.
A server that keeps the model loaded would not pay the load again, and this probe is the measurement to repeat once the server does (issue 168).

An earlier probe of the same server ran while the chip had not been reset after another agent's work (the firmware failed to start). Every job ended in an error after about 70 s. The queue behaved the same way
(one at a time, one 429 at 4 requests), and the server stayed up. The file is kept as `queue-probe-failed-device.json`; its timings are not chip timings.

## What could not be determined

- Whether the server starts jobs in the order they arrive. The first version of the probe inferred an order from queue positions seen after a job had started, which cannot show it (review of PR 187). The probe now has a separate order test that sends jobs one after another, and that test has not been run on hardware.
- Why the earlier single 120 s run took 16.5 s. It was a one-shot process on a busier host. The warm-up table shows run 1 costs 0.1 to 0.4 s more in the transformer, which does not explain 6 s.
- How much of the transformer-call time is device time. The part includes host preparation, uploads and the read back.
- Whether one warm-up run is enough for another model or a cold kernel cache. It was enough here, with a warm cache.
- The CPU time at 120 s is one sample. A single sample has no spread.
- The host was shared and its load average was 5 to 28 during these runs. The load average is recorded next to each timing, and it does not remove the effect of other processes; the third base 120 s run is the clearest case.
- The probe sees times by polling, so its times have a resolution of about 0.25 s. It judges "one at a time" from the server's own service times.
- The resident-server queue behaviour. The demo server was not changed, so each job reloaded the model; a server that keeps the model loaded was not measured.
- One prompt, one seed and one clip length at each level of the probe. No power or memory numbers. Nobody listened to the clips.
