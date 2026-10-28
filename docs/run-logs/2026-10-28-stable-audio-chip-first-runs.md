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
