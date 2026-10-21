# 2026-10-19: packaging bytkim/Qwen3.8-27B-pi (branch `bytkim-package`)

Directive (from the parent session): prepare the v6 package for `bytkim/Qwen3.8-27B-pi` from the finished run
`~/orchard-runs/bytkim--qwen3.8-27b-pi`, by hand, the way Clef and RAG-Gate were done. Measure first, then package, then
write the card. Nothing was published, uploaded or pushed. No cache, run directory or model was deleted.

## Measurements (before packaging)

One 2-chip board, the installed bundle `episod/qwen3.8-27b-dflash2-p300` (drafter on, sampling on the device), greedy decoding,
streamed, one warm-up and three repetitions per prompt, medians. The prompt-set sha256 is the same for both models
(`6d2c2d4f...bb2`); the launcher text differs in the two model-path lines only. Tool: `orchard/measure_serving.py`.

| Prompt | pi decode tok/s (range) | base decode tok/s (range) | pi / base | pi TTFT s | base TTFT s | tokens pi / base |
|---|---|---|---|---|---|---|
| pi-coding | 53.0 (52.8 to 53.1) | 54.4 (53.5 to 56.0) | 0.974 | 0.18 | 0.20 | 384 / 384 |
| tool-call | 68.1 (62.3 to 70.2) | 67.4 (66.8 to 68.1) | 1.010 | 0.22 | 0.21 | 55 / 55 |
| tool-call-write | 73.3 (72.7 to 73.9) | 63.8 (57.2 to 64.3) | 1.149 | 0.21 | 0.22 | 357 / 384 |
| chat | 48.7 (48.5 to 49.1) | 44.1 (43.2 to 44.4) | 1.104 | 0.16 | 0.18 | 224 / 206 |

- `tool-call` produced the same 55 tokens from both models, so its ratio (1.01) is a noise control for the two servers. The prompt is
  under the 100-token floor; `tool-call-write` was added to give a long call.
- On the other prompts the two models wrote different text. With a speculative drafter the decode rate depends on the text and on
  the drafter's acceptance, so these ratios do not support a claim that the fine-tune is faster or slower.
- Drafter acceptance was not measured: the server exports no spec-decode counters and prints none.
- Server ready: pi 104 s (warm tensor cache), base 236.1 s (empty cache). Not comparable.
- Not measured: concurrency, long-context prefill, sampling with a temperature, output quality. The author's benchmark charts were
  not read and are unverified.
- Files: `evidence/2026-10-19-bytkim-pi-packaging/measurements.json` (full) and `card-numbers.json` (the rounded values the card cites).

## Package

- Stage 7 by hand with `stage7_by_hand.py` (in the evidence directory). Profiles: `qwen3.8-27b-pi-p300` (2 chips, P150x2) staged and boot-checked.
  Skipped: 1 chip (stage 4 had no passing 1-chip configuration) and 4 chips (no 4-chip v6 bundle of the base is installed).
- Boot check on a fresh install and an empty tensor cache, under `gozer run --chips 2`: return code 0, 32 of 32 tokens match the CPU reference
  (top-1 1.0), coherent, ready in 1,992.4 s (stage 2's start of the same weights took 682.1 s; the two starts differ in more than the package,
  and the cause was not isolated). Whole check 2,003.6 s.
- `gate_package`: ok.
- Card: generated, then rewritten by hand with the measured speed rows, the base comparison, and a not-measured list. `package.finish` rewrites the
  card from its template, so the edit was applied after it, and `after_card.py` redid the evidence copy, the scrub and the gate. The gate
  requires a measured number's first cited file to contain the value as text and refuses a numeral with a unit in prose, so the speed numbers
  live in `packaging/card-numbers.json` (the full measurements file names the operator's namespace in its bundle field, which the scrub refuses).
- `PUBLISH_COMMANDS.txt` holds one live `hf upload --private` line as text. It was not run.

## Harness gap found and fixed

`tt-model package-thin` (0.1.0) now writes a `run.sh` that sets a default `HF_MODEL`, resolves the weights with `snapshot_download` and exports
`HF_MODEL` and `MODEL_WEIGHTS_DIR` from `$resolved_hf_model`. `wire_weights` expected exactly one `export HF_MODEL=` line and stopped with "found 2".
`orchard/package.py` now wires the two resolved exports (the resolver block stays), and `weights_wiring_problems` accepts the default line only when
it does not name the nearest model. Tests first (fixture is a real generated `run.sh`); mutations: 1 equivalent survivor, 1 real gap closed with a test.
The first staging attempt is kept at `stages/7.partial-1`.

## Slips and disk

- The measurement tool's first runs failed twice: the warm-up kept `min_tokens` above `max_tokens`, and the TT server refuses `min_tokens` at all and reports
  that inside the stream. Both are fixed with tests (`StreamError`).
- The base's 34 GB tensor cache is at `/mnt/bonus/models/orchard-runs/bytkim-package/base-tt-cache`, put there to protect the root drive.
- Disk: `stages/7` is 37 GB (mostly the boot check's cache under `verify/`). Root has 38 GB free and `/mnt/bonus` 46 GB.
