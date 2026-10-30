# ACE-Step 1.5 turbo: per-step bars from a teacher-forced floor, and an audio profile for a 48 kHz model (issues 262 and 263, 2026-10-30)

Operator instruction: continue the ACE-Step bring-up from issue 254. This log covers steps 1 and 2. Paths are placeholders (`<run>` is a scratch directory of recorded runs).

## Per-step bars (issue 262)

The 0.10 relative L2 bar of the first run was a choice. A teacher-forced bfloat16 CPU run now measures what bfloat16 arithmetic alone does on each step.

Method: `acestep_run.py --device cpu --cpu-dtype bfloat16 --teacher-forced <float32 run>` runs one generate call in which every transformer call receives the float32 run's recorded input of
the same step, and the VAE decoder receives the float32 run's final latent. Only the transformer and the decoder are in bfloat16, the same split as the chip. The velocities and the waveform
are compared with the float32 run's. `--stepcheck` does the same on the chip. `orchard/step_bars.py` sets each step's bar to a stated multiple (2.0) of that step's floor and reports, per step,
the chip's error, the floor and the bar.

Two prompts, seed 42 (lo-fi, instrumental) and seed 7 (folk with lyrics), 30 s, 8 steps. Relative L2 per step; the chip numbers are one call per step on the CPU run's input.

| Step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | decoder |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lo-fi: bfloat16 CPU floor | 0.172 | 0.126 | 0.051 | 0.032 | 0.038 | 0.021 | 0.017 | 0.022 | 0.0075 |
| lo-fi: chip | 0.078 | 0.054 | 0.023 | 0.020 | 0.023 | 0.020 | 0.022 | 0.028 | 0.0141 |
| lo-fi: bar (2.0 x floor) | 0.344 | 0.252 | 0.102 | 0.063 | 0.076 | 0.043 | 0.034 | 0.044 | 0.0151 |
| lo-fi: wrong-step control | 0.603 | 0.450 | 0.220 | 0.174 | 0.154 | 0.161 | 0.320 | 0.263 | |
| folk: bfloat16 CPU floor | 0.133 | 0.098 | 0.040 | 0.026 | 0.017 | 0.020 | 0.019 | 0.023 | 0.0079 |
| folk: chip | 0.042 | 0.033 | 0.022 | 0.018 | 0.020 | 0.020 | 0.023 | 0.028 | 0.0125 |
| folk: bar (2.0 x floor) | 0.265 | 0.197 | 0.081 | 0.052 | 0.034 | 0.040 | 0.037 | 0.045 | 0.0159 |
| folk: wrong-step control | 0.535 | 0.336 | 0.189 | 0.165 | 0.122 | 0.130 | 0.230 | 0.201 | |

Findings:
- The 0.10 bar was too tight for the first two steps: the bfloat16 CPU floor itself is 0.17 and 0.13 at step 0 and 0.13 and 0.10 at step 1. It was also loose for steps 5 to 7 (floor 0.017 to 0.023).
- The chip is below the bfloat16 CPU floor on steps 0 to 4 (0.4 to 0.6 of it) and at 1.0 to 1.3 times it on steps 5 to 7. The chip's decoder is 1.88 times (lo-fi) and 1.58 times (folk) the CPU decoder floor.
  So the margin has to be about 2 to hold the chip's decoder; at 1.5 the lo-fi decoder fails (0.0141 against 0.0113). The margin was chosen after seeing that ratio, from two prompts; it is a stated choice with this evidence behind it, and no derivation gives 2.
- The wrong-step control (the neighbouring step's velocity) fails the bar at every step on both prompts. Noise at 25 percent relative size gets under the bar at step 0 (and step 1 for lo-fi): the bar is the largest error that can hide,
  and for steps 0 and 1 it is 0.25 to 0.34. A fault smaller than that at those two steps is invisible to this check. The tool reports the bars as detection limits and refuses only when the wrong-step control passes.
- One floor run per prompt, one chip call per step, one seed each. The floor is not a distribution.

Evidence: `docs/run-logs/evidence/acestep-step-bars/` (six JSON files: floor, chip and bars for each prompt). `python3 -m orchard.step_bars --floor F --chip C --run RUN` reproduces the bars.
This does not change `audio_stages` stage 2 yet; the supervisor still compares end to end for Stable Audio. The profile below records that ACE-Step uses the per-step check.

## Audio profile (issue 263)

`orchard/audio_profile.py` holds, per model family, the sample rate, the clip file, whether the clip equals the recorded waveform, the recorded arrays a run must hold (as a function of the step count),
which stage-2 checks apply, and the runtime files of a package. Two profiles: Stable Audio (44.1 kHz, FLAC, end-to-end) and ACE-Step 1.5 turbo (48 kHz, WAV peak-normalised after recording, per-step).
`layout_problems` checks a run's manifest, `sound_problems` runs the sound checks at the profile's rate, and `python3 -m orchard.audio_profile RUN --profile acestep-1.5-turbo` checks a run directory.
`audio_stages.check_run_dir` takes an optional `profile`; without one nothing changes for Stable Audio.

Check on the recorded runs from the first bring-up: the CPU float32, chip and bfloat16 CPU runs pass the ACE-Step profile and fail the Stable Audio one (1,440,000 samples is not 30 s at 44.1 kHz).

Not done: `audio_plan.reference_plan`, the supervisor's stage commands and `audio_cli` still name the Stable Audio library. They need to read the profile and start `scripts/acestep/acestep_run.py`. Left to an issue.

## Review round 1 (PR 267)

Copilot found three High and one Medium defect, all fixed with tests that went red when the fix was removed: a non-exact clip check accepted a silent or unrelated file (it now needs correlation 0.999 with the recorded waveform, since peak normalisation is a gain);
`analyse` accepted a missing required control; the CLI combined files made from different reference runs (both reports now carry the sha256 of the reference manifest and the CLI refuses a mismatch); teacher forcing did not check that the conditioning
made from the current prompt equals the reference run's (it now refuses above relative L2 1e-4). The six evidence files were made before the digest existed; the digest was added to them afterwards from the recorded reference runs, which still existed.
