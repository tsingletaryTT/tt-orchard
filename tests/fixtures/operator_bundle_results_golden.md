# Results: Altworld/Hemmingway-1

## Summary

This run brought up Altworld/Hemmingway-1 on the weights-only path. Stages that passed: 0, 1, 2, 4, 7. Stages skipped: 3, 5, 6. Stage 7 staged 2 package(s); 1 passed its boot check. RISKS.md lists 6 TODO item(s) and 4 stage 0 hazard(s).

## Run facts

Every path in this file is relative to the run directory (the directory that holds `ledger.jsonl`). A number is labelled `measured` or `TODO`; `TODO` means not measured.

| Fact | Value |
|---|---|
| Model | Altworld/Hemmingway-1 |
| Revision | 1a5f363a3dd2d1cc456c28b8abbb403b9555efaf |
| Nearest model | Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 |
| Path | weights-only |
| Class | weights-only |
| License | cc-by-nc-4.0 (non-commercial) |

## The run

Counted from `ledger.jsonl`.

| Count | Value | Label | Evidence |
|---|---|---|---|
| Wall time (first to last ledger entry) | 0 s (0.00 h) | measured | `ledger.jsonl` |
| Escalations | 1 | measured | `ledger.jsonl` |
| Retries | 1 | measured | `ledger.jsonl` |
| Pauses | 1 | measured | `ledger.jsonl` |
| Operator commands | 1 | measured | `ledger.jsonl` |
| Ledger entries | 24 | measured | `ledger.jsonl` |

## Stage 0: intake and delta triage

Result: pass. Attempts: 1.

Path: `weights-only`. Model: Altworld/Hemmingway-1@1a5f363a3dd2d1cc456c28b8abbb403b9555efaf. Nearest model: Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.

### Differences

- tokenizer: 6 of 206 strings encode differently (Thai, Devanagari). Evidence: `stages/0/evidence/tokenizer-compare.json`.
- license: cc-by-nc-4.0 against apache-2.0. Read at <home>/.cache/huggingface/x/README.md. Evidence: `stages/0/evidence/genconfig-license.json`.

### Hazards

RISKS.md says whether a later stage dealt with each one.

- tensor_cache: The nearest model's cache at <OPERATOR_HOME>/.cache/src-build/tensors is keyed by layer name. Evidence: `stages/0/evidence/tensor-compare.json`.
- drafter: The drafter was trained on the nearest model. Evidence: `stages/0/evidence/config-compare.json`.
- disk: 120 GiB free. Evidence: `stages/0/evidence/disk-free.json`.
- license: The license changed to cc-by-nc-4.0. Evidence: `stages/0/evidence/genconfig-license.json`.

Evidence: `stages/0/delta.json`.

## Stage 1: environment and CPU reference

Result: pass. Attempts: 1.

Verdict: pass.
Environment: python 3.12, torch 2.11, transformers 5.17.

- loads: pass. 0 missing keys. Evidence: `stages/1/evidence/load-report.json`.
- decodes forward: pass. 32 tokens. Evidence: `stages/1/evidence/decode.txt`.
- matches the card: pass. This is a form check only. Evidence: `stages/1/evidence/card-check.txt`.

Evidence: `stages/1/reference.json`.

## Stage 2: weights swap check on one board

Result: pass. Attempts: 2.

| Field | Value | Label | Evidence |
|---|---|---|---|
| coherent | yes | - | `stages/2/result.json`, `stages/2/evidence/swap-check.json`, `stages/2/evidence/server.log` |
| n_tokens | 32 | measured | `stages/2/result.json`, `stages/2/evidence/swap-check.json`, `stages/2/evidence/server.log` |
| server_ready_s | 728.1 | measured | `stages/2/result.json`, `stages/2/evidence/swap-check.json`, `stages/2/evidence/server.log` |
| serves | yes | - | `stages/2/result.json`, `stages/2/evidence/swap-check.json`, `stages/2/evidence/server.log` |
| top1_agreement | 0.9375 | measured | `stages/2/result.json`, `stages/2/evidence/swap-check.json`, `stages/2/evidence/server.log` |

Free-run text: Let me work.
Tensor cache: <CACHE_ROOT>/hemmingway-1/tt_cache

## Stage 3: full model

Result: skipped. Reason (from the ledger): weights-only path: the stage 2 serve-and-compare covers the full model

## Stage 4: weights swap on each chip configuration

Result: pass. Attempts: 1.

Required chip counts: 2, 4.

| Chips | Pass | Serves | server_ready_s | Coherent | top1_agreement | n_tokens | Package | Tensor cache | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| 1 | fail | - | TODO | - | TODO | TODO | - | - | `stages/4/result.json`, `stages/4/tests/1/output.txt` |
| 2 | pass | yes | 152.0 (measured) | yes | 0.9375 (measured) | 32 (measured) | episod/p2 | <CACHE_ROOT>/hemmingway-1/2chip/tt_cache | `stages/4/result.json`, `stages/4/configs/2/evidence/swap-check.json`, `stages/4/tests/2/output.txt` |
| 4 | pass | yes | 154.0 (measured) | yes | 0.90625 (measured) | 32 (measured) | episod/p4 | <CACHE_ROOT>/hemmingway-1/4chip/tt_cache | `stages/4/result.json`, `stages/4/configs/4/evidence/swap-check.json`, `stages/4/tests/4/output.txt` |

Configurations that did not pass:

- 1 chip: the server did not become healthy. Evidence: `stages/4/tests/1/output.txt`.

## Stage 5: serving integration

Result: skipped. Reason (from the ledger): weights-only path: stage 4's serve-and-compare tests boot each configuration, check the output against the CPU reference and check it is coherent (the black-box serving check)

## Stage 6: qualitative check and benchmark

Result: skipped. Reason (from the ledger): weights-only path: the operator deferred the qualitative check and benchmark; stage 4 already shows the output is coherent

## Stage 7: package (v6 thin bundle) and boot check

Result: pass. Attempts: 1.

Stage 7 staged 2 package(s) of Altworld/Hemmingway-1 at revision `1a5f363a3dd2d1cc456c28b8abbb403b9555efaf` in format v6. License: cc-by-nc-4.0 (non-commercial).

| Package | Chips | Mesh | Required | Boot check | top1_agreement | server_ready_s | Evidence |
|---|---|---|---|---|---|---|---|
| hemmingway-1-p150 | 1 | P150 | no | not boot-checked | TODO | TODO | `stages/7/package.json`, `stages/7/package/hemmingway-1-p150/README.md` |
| hemmingway-1-p300 | 2 | P150x2 | yes | passed | 0.9375 (measured) | 2034.4 (measured) | `stages/7/package.json`, `stages/7/verify/evidence/verify.json`, `stages/7/package/hemmingway-1-p300/README.md` |

No package for 4 chips: no v6 bundle for 4 chips is installed.

The publish commands are in `stages/7/PUBLISH_COMMANDS.txt`, copied to this bundle's `PUBLISH_COMMANDS.txt`. The run never ran them.

Evidence: `stages/7/package.json`.

## Stage 8: operator bundle

Result: this bundle. The stage 8 gate checks it after the agent finishes.

Files: `RESULTS.md`, `RISKS.md`, `card.md`, `PUBLISH_COMMANDS.txt`, `package/` and `ledger.jsonl`. `package/` holds stage 7's package folders without venvs, caches, weights or wheels the package does not ship, plus stage 7's record, publish commands and cards.

`ledger.jsonl` is the operator's copy of the run record. It holds absolute paths from the machine and is not for publishing. The scrub skips it.

The build record is `stages/8/evidence/bundle-build.json`.
