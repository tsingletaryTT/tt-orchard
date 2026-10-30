# ACE-Step 1.5 turbo package 0.2.0: the faster path, regated and boot-checked (2026-10-30, issue 254 step 3)

Operator instruction: rebuild the local package with the faster path, regate it, boot it on a chip and update its numbers table from fresh measurements (1 warm-up, 3 timed runs per length at 30, 60, 120, 240 and 480 s).
Paths are placeholders. The staged package is in `~/orchard-packages/acestep-1030-fast/ace-step-1.5-turbo-blackhole` on the machine that ran this (260 KB). Nothing was published. The path it packages is described in
`docs/run-logs/2026-10-30-acestep-chip-path.md` (PR 278); this change is stacked on that branch.

## What changed in the package

- Version 0.2.0. The runtime gains `runtime/acestep_fast.py` and `runtime/acestep_chip.py`; the server takes `--fast` (default `all`; `run.sh` passes `ACESTEP_FAST`, default `all`; `none` is the 0.1.0 path). The ready line lists the steps.
- Card: a section "What runs on the chip" (where each piece runs and which file holds it), an "Other" column in the numbers table (median total minus the two chip stages), a column with the median total of the same sweep with every
  step off, measured in the same hour, and updated limits. The old sentence that the encoders are most of each request is gone.
- Evidence: `sweep.json` (fast path) and `sweep-base.json` (0.1.0 path), per-step bars measured with `--own-conditioning`, the end-to-end agreement and its ablation, the profile, the boot record.
- The gate requires the new files, the new section and the new evidence files. New tests in `tests/test_acestep_package.py` (seven) and `tests/test_acestep_scripts.py` (one); each gate change was removed in turn and a test went red (chip files, section, other column, before column, run.sh flag, version, second-request line).

## Numbers (from `evidence/summary.json`; one chip of board 1, host load average 1.7 to 2.7)

1 warm-up and 3 timed calls per length, lo-fi prompt, seed 42. Total of one generate call, min / median / max.

| Length | 0.2.0 total | transformer (chip) | VAE (chip) | other | audio s per s | 0.1.0 path, median, same hour |
|---:|---|---:|---:|---:|---:|---:|
| 30 s | 0.93 / 0.93 / 0.93 | 0.39 | 0.46 | 0.07 | 32.3 | 3.00 |
| 60 s | 1.63 / 1.63 / 1.64 | 0.59 | 0.93 | 0.11 | 36.7 | 3.76 |
| 120 s | 2.95 / 2.95 / 2.99 | 0.83 | 1.88 | 0.23 | 40.7 | 5.06 |
| 240 s | 5.66 / 5.67 / 5.69 | 1.92 | 3.31 | 0.45 | 42.3 | 7.69 |
| 480 s | 12.20 / 12.26 / 12.27 | 4.67 | 6.63 | 0.96 | 39.2 | 14.57 |

Load: chip open and VAE warm-up 7.6 s; the server printed its ready line 18.0 s after the start (polled every 2 s; the server reported 15.26 s of its own load, kernel cache warm).

## Gate

`python -m orchard.acestep_package gate PKG` gave 0 problems on the staged package. The pins are the 0.1.0 pins: the bundle tree hash (1,307 files) and the three weight files match the earlier package.

## Boot check on a chip

A first staging with a placeholder boot record (the gate reported only "the boot check did not pass") was copied, `run.sh fetch` downloaded the bundle code as real files (21 s), `run.sh --self-test` passed, and `run.sh` started
the server on port 20121 (one chip pair of board 1 under the lease). Then the evidence was completed from the boot, and the final package was restaged. Review of PR 278 then changed two runtime files (`acestep_run.py`, `acestep_chip.py`), so the package was restaged and booted a second time from a copy of the restaged
package (ready line after 18.0 s; 30 s requests of 1.58 s and 0.96 s; both clips identical to each other and to the CLI clip, maximum absolute difference 0.0). The `boot.json` figures are from the first boot.
All runtime files, `run.sh`, the tools, `pins.json` and the card of the final package are byte-identical to the second booted copy (compared with `cmp`).

- Ready line after 18.0 s, listing all six steps.
- A 30 s request (lo-fi, seed 42) returned in 1.57 s (the first after the start); an identical second request in 0.97 s. Both pass the 48 kHz sound checks. The samples of the two are identical, and identical to the clip that
  `acestep_run.py --fast all` records for the same prompt and seed (maximum absolute difference 0.0).
- A 60 s request with a 207-token lyric (folk, seed 7) returned in 1.91 s; a 120 s request (seed 11) in 3.31 s. Both pass the sound checks.
- The server was stopped with SIGTERM (it was started in the background, where a shell sets SIGINT to ignored, so Ctrl-C style shutdown was not exercised).
- Not exercised on the chip this time: the 400 and 503 replies (unit tests only; the 0.1.0 boot checked the 400 replies by hand).

## Agreement

Per-step bars with the package's own conditioning (text encoder, lyric encoder and cross-attention keys and values on the chip): all steps and the decoder pass on both prompts; the closest is lo-fi step 4 at 1.57 times
the floor (0.059 against a bar of 0.076). End to end against the float32 CPU run: 16 of 21 recorded arrays pass the provisional bars (17 before), latent PCC 0.978 (0.982 before), waveform PCC 0.918 (0.939 before).
The ablation in the PR 278 log shows that this comparison moves by about 0.01 to 0.03 under changes of 0.0002 in the conditioning.

## Clips to listen to (A/B against the previous clips)

In `orchard-audio/acestep-2026-10-30-chip-path/`: `before-*` are the 0.1.0 path and `after-*` the 0.2.0 path for the same prompt and seed (30 s: lofi42, folk7, lofi11, folk3; lo-fi 42 at 60, 120, 240, 480 s).
The package server's own clips are `package-server-lofi-30s-seed42.wav`, `package-server-folk-60s-seed7-long-lyrics.wav` and `package-server-lofi-120s-seed11.wav` in the same directory. No one has listened to any of them.

## Not done

- No upload and no second boot of a pulled copy (nothing was published). `tt-model.yaml` is still a draft.
- The planner (issue 272) and the XL transformer (issue 273).
