# Stable Audio 3 small music: the first clips with the diffusion transformer on a chip (2026-10-28)

Design: `docs/superpowers/specs/2026-10-28-stable-audio-runtime-design.md`, step 1. Tracking issue: 160. Paths are placeholders.

## What ran

One Blackhole chip (one gozer lease per run, one chip), the shared Tenstorrent environment (torch 2.7.0, transformers 5.19.0) with the library checkout on
`PYTHONPATH`. The diffusion transformer (the 20-layer stack, its input and output projections and the 64 memory tokens) ran through
`orchard/skills/audio-chip-templates/chip_dit.py`. The T5Gemma encoder, the duration, timestep and global embedders, the noise draws, the ping-pong sampler and
the autoencoder decoder ran on the host with the library's own code, in float32. Prompt "A warm acoustic guitar melody with soft percussion, 100 BPM",
10 s, 8 steps, guidance scale 1.0, seed 42. The noise draws match the reference bit for bit (same seed, same host code).

## What changed between the attempts

| attempt | weights | residual stream and outputs added to it | final latent, PCC and relative L2 error against the reference | waveform PCC | bars |
|---|---|---|---|---|---|
| 2 | bfloat16 | bfloat16 | 0.9705 and 0.260 | 0.942 | failed (latent and the last three steps) |
| 3 | bfloat16 | float32 | 0.9905 and 0.138 | 0.981 | all pass |
| 4 | float32 | float32 | 0.9767 and 0.216 (against the same-environment CPU run) | 0.947 | all pass |

The last layer takes the latent rows from a norm of about 5,800 down to about 840, so a bfloat16 residual stream loses much of what is left. Attempt 3
keeps the stream and the outputs added to it in float32 and leaves the matrix multiply inputs in bfloat16. Attempt 4 shows that float32 weights do not
help. Attempt 3 is the default.

## Timings (attempt 3)

| part | seconds |
|---|---|
| model load on the host | 5.4 |
| chip weights upload | 1.2 |
| eight calls of the transformer, summed (host preparation, uploads, the layers on the chip and the read back; the first includes kernel compile) | 4.4 |
| `generate` and decode, whole | 5.7 |
| CPU reference, same call, reference environment | 24.8 |

## Agreement

Bars are the provisional ones in `orchard/audio_reference.py`. All 56 arrays compared; every layer output has PCC at least 0.9999 against the reference
for the first pass, with a relative L2 error of 1.1 to 1.7 percent.

| comparison | final latent PCC | final latent rel. L2 | waveform PCC | waveform rel. L2 |
|---|---|---|---|---|
| chip against the reference (reference environment) | 0.9905 | 0.138 | 0.981 | 0.198 |
| same-environment CPU run against the reference (no chip; the environment floor) | 0.9963 | 0.086 | 0.9915 | 0.132 |
| chip against the same-environment CPU run | 0.9872 | 0.161 | 0.972 | 0.235 |

The environment floor exists because the two CPU environments (transformers 5.19.0 against 5.21.0) give T5Gemma embeddings that differ by 1.7 percent
(issue 162), and the sampler amplifies the difference. The chip result is within about twice that floor.

## Where the layer error comes from (issue 163)

Layer 0, relative L2 error of the chip against the host decomposition on the same inputs: input embedding 0.0025, after self attention 0.0061, after cross
attention 0.0064, after the feed forward 0.0170 over all rows and 0.0052 over the 174 latent rows. The memory rows (norm about 21,000) carry the larger
number. No single operation stands out; each section adds a fraction of a percent, as bfloat16 operands give.

## Limits

Nobody has listened to the clips. One prompt, one duration, one seed. The bars are choices that have not been set from a measured noise floor. The base
control (50 steps, guidance scale 7) was not part of these first runs; the stage 2 test makes it.

## Benchmark: chip against CPU, three repeats (2026-10-28, after review of PR 165)

The model has been benchmarked on this one prompt, one duration and one seed only. Settings: 10 s of stereo audio at 44.1 kHz, prompt "A warm acoustic guitar melody
with soft percussion, 100 BPM", seed 42. The fine-tune uses its own settings (8 steps, guidance scale 1.0, ping-pong sampler); the base control uses the base's
(50 steps, guidance scale 7.0, Euler sampler, so two transformer passes per step). The chip runs are the three repeats of stage 2 in a fresh run
(`stages/2/chip/`, each repeat a separate process with its own device open; the kernel cache was warm, so the first repeat is not much slower than the others).
The CPU runs are three float32 runs of the library's own `generate` in the reference environment on the 16-core host, run one after another before the chip
runs. The host was shared: its load average was about 6 during the CPU runs, another agent held one board during the chip runs, and the three CPU
samples of each model spread widely (the table shows each). The CPU time is the library's own timer around `generate` and the decode, the same span as the chip
column `generate_and_decode_s`. All numbers are seconds.

| Model and settings | Where | Repeat 1 | Repeat 2 | Repeat 3 | Median |
|---|---|---|---|---|---|
| Fine-tune, 8 steps | chip: whole `generate` and decode | 1.755 | 1.608 | 1.578 | 1.608 |
| Fine-tune, 8 steps | CPU: whole `generate` and decode | 3.865 | 7.972 | 6.023 | 6.023 |
| Base control, 50 steps, guidance 7 | chip: whole `generate` and decode | 4.951 | 5.029 | 5.296 | 5.029 |
| Base control, 50 steps, guidance 7 | CPU: whole `generate` and decode | 43.337 | 66.831 | 43.327 | 43.337 |

Parts of the chip runs:

| Part | Fine-tune, 3 repeats | Median | Base control, 3 repeats | Median |
|---|---|---|---|---|
| Diffusion transformer calls, summed over all passes (host preparation, uploads, layers on the chip, read back) | 0.399, 0.354, 0.363 | 0.363 | 3.492, 3.508, 3.803 | 3.508 |
| T5Gemma encoder on the host | 0.039, 0.045, 0.044 | 0.044 | 0.041, 0.040, 0.041 | 0.041 |
| Passes of the transformer | 8 | 8 | 100 (50 steps, two per step) | 100 |
| Everything else in `generate` and decode (sampler, noise, autoencoder decoder on the host) | 1.3, 1.2, 1.2 | 1.2 | 1.4, 1.5, 1.5 | 1.5 |

One-off costs of each chip run: model load on the host about 6 s, weights upload to the chip about 1.2 s. A whole chip process takes about 11 s for the
fine-tune.

What the table says. The chip is about 3.7 times faster than the CPU for the fine-tune (median 1.6 s against 6.0 s) and about 8.6 times faster for the base control
(median 5.0 s against 43.3 s). In the fine-tune's 1.6 s the transformer is 0.36 s; most of the rest is host work (the sampler and the autoencoder decoder), so moving
the decoder to the chip (issue 161) is the next speed step. In the control the transformer is 3.5 s of 5.0 s. The transformer time includes host
preparation and transfers (see the Copilot note in PR 165), so it is an upper bound on device time. The stage 1 reference runs that the bundle prints (24.8 s and
57.2 s, one sample each) were made while the host was busier than during this benchmark; the three-sample CPU numbers above are the fairer comparison. No power
or memory numbers were taken.
