---
name: stock-runtime-check
description: Stage 2 of a tt-orchard run when stage 0 found that tt-transformers already supports the model (class stock-runtime). Serve the model through tt-transformers on one leased board, as it is, and compare its teacher-forced top-1 and top-5 with the stage 1 CPU reference.
status: draft. A local tt-orchard copy, written for the issue 61 pilot (Llama-3.1-8B-Instruct). One tested template script does the work; you run it and write no model code.
---

# Stock-runtime check

## When to use this

Stage 0 wrote `delta.json` with `"class": "stock-runtime"`: the model's config equals a `model_params/<name>` config in
tt-metal's `models/tt_transformers`, and its tensors are in the layout tt-transformers loads. No weights swap is possible (no
installed bundle serves it) and no model code has to be written. This stage shows that tt-transformers serves the model on this box
and that its output agrees with the CPU reference from stage 1.

## Goal

The gate reads `evidence/stock-check.json`, which the template writes. You do not write it. The gate needs the teacher-forced top-1 at
or above `STOCK_TOP1_MIN` and top-5 at or above `STOCK_TOP5_MIN` (orchard/defaults.py; lower bars for a model of 3 billion parameters or
fewer), at least 128 compared tokens, the board name the runtime saw, the tt-metal tag, the optimization mode, a free run with no
blocking finding, a positive decode speed, no model fetched from the Hub, and the patch of the scratch tree as evidence.

## How to work

- The supervisor wrote `stages/2/stock_config.json` and copied `stock_check.py` into your stage directory. Every fact in the config
  is one the run already holds. Use it as it is. Do not look for the tt-metal tree, the snapshot or a board name yourself.
- Do not edit `stock_check.py`. The gate compares your copy with the shipped file and fails the stage if they differ.
- Do not read tt-metal model code, and do not write model code. If the check fails, the failure text is the result.
- The host tt-metal tree is read only. A copy of its `models/` directory is `<run>/tt-metal-scratch/`; `write_file` can write there
  with a path that starts with `tt-metal-scratch/`. Change it only when the template's failure text names a file that has to change.
- Never run `tt-smi`, `gozer` or the model yourself. The supervisor runs the hardware test on a leased board.

## Prepare phase: the steps

1. Read `stages/2/stock_config.json` once.
2. Run `python3 stages/2/stock_check.py prepare`. It copies the host tree's `models/` into the scratch directory (once), builds a model
   directory named after the tt-transformers model from the snapshot's files, makes the reference file from the run's own snapshot on the CPU (the
   file the host tree ships for a model name belongs to the model the tree lists, so it is not used), and writes `hw_test.json` and `handoff.json`. Exit 2 means a fact does not fit; its message names it.
3. Check that `stages/2/hw_test.json` and `stages/2/handoff.json` exist (`ls -l`), and reply with a short summary.

## Finish phase

The supervisor ran `python3 stages/2/stock_check.py run` under a lease. Read `stages/2/evidence/stock-check.json` once. If it exists, reply
with a short summary: the board name, top-1, top-5 and the decode speed. If the test failed, `evidence/hw-test-output.txt` holds the
output. Quote the first line that states an error and stop. Do not try to make a failed test pass, and do not write a result file: the gate
reads the template's evidence.
