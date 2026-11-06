---
name: full-port-quality
description: Stage 6 of a tt-orchard run started with --full-port, after the port passed the earlier stage. No agent runs this stage. The supervisor runs its own template on the port that passed, measures quality against the CPU reference and a benchmark with repeats, and writes result.json from the record. This file tells the operator and any reviewer what the stage does and what each number is.
status: draft. Written on 2026-11-04 for issue 329, stage 6 half (docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md, slice 6). No full port has passed stage 2 on a real model yet, so nothing here has met a chip.
---

# Full port quality and benchmark

## When this runs

The run was started with `--full-port`, stage 0 said `full-port`, and the port passed stage 2 (one board) and, when the run configures more
chips than one board holds, stage 3. This stage measures that same port. It asks the coder for nothing: `needs_coder` is false, there is no
prepare step and no finish step. The gate file is `result.json`, which the supervisor writes.

## What the supervisor does

1. Copies the port that passed into `stages/6/port/`. It is stage 3's port when stage 3 ran, otherwise stage 2's. The ledger holds the
   sha256 of each copied file (the decision `stage N port copied`).
2. Copies the shipped template `port_quality.py` and writes `port_quality_config.json` (the model snapshot) next to it. It writes
   `hw_test.json` with the fixed command `python3 -P -s stages/6/port_quality.py`.
3. Leases the chips (the same count the source stage used: the configured chips after stage 3, else one board), parks the coder when the
   board is needed, runs the command, and gives everything back.
4. Writes `result.json` from `evidence/port-quality.json`, runs the gate on it, and deletes it if the gate refuses.

## What the template measures

Everything runs on one loaded port. The port module is the one the coder wrote for stage 2 (`load`, `prefill`, `decode_step`, `close`). It may
also define a module-level `DTYPE_POLICY`, a dict that maps a weight group of `orchard/decode_floor.py` to a dtype name, for example
`{"default": "bfp8"}`. A port that defines none is taken as bf16 and the record says so.

| Part | What it does | What goes in `result.json` |
|---|---|---|
| Teacher-forced scoring | Feeds the stage 1 reference ids in one at a time (64 positions at most). At each position records the port's argmax, the log probability the port gives the reference's next id, and its logits at the reference's top five ids from `stages/1/evidence/reference/top5-logits.json`. | `perplexity` (exp of the mean negative log probability), `top1_agreement`, and `top5_gap_diff`, the mean absolute difference between the port's logit gaps and the reference's. |
| Free run | A fresh prefill and 64 of the port's own greedy ids, decoded to text. | `leading_match` (ids equal to the reference's before the first difference), `coherent` and its basis, the text. |
| Benchmark | One discarded warm-up run and three measured runs in the same process. Each run is one prefill and 32 decode steps. | Spread (min, median, max) of decode speed and prefill time, the label from `orchard/resident.py`, the decode floor from `orchard/decode_floor.py` for the dies the test ran on, and the median speed as a share of it. |

Every figure appears in `numbers` with a label, `measured` or `TODO`, as the operator bundle expects. Three things the stage does not measure are
listed there as `TODO`: quality on other prompts, decode speed at a batch size above one, and quality at a long context.

## What the gate requires

- The test exited 0, did not time out, ran the fixed command, and ran on the chips the record names. After stage 3 the chip count is the
  configured one.
- The reference ids, the prompt and the top-five file in the record are the stage 1 files on disk (path and sha256). The record's top-five
  entries are the file's.
- The stage's copy of the template is the shipped file, and the record carries its sha256. The output file carries the record's sha256 once.
- The port files in the record are the files the supervisor copied (the ledger's hashes), and they are the files the source stage's own record
  names. A model source must mention `ttnn` and must not read the stage 1 reference.
- Every number in `result.json` is what the gate recomputes from the record. The teacher-forced top-1 agreement is at least the stage 2 bar, and
  `top5_gap_diff` is at most 0.5 logits (a choice, not a measured noise floor).
- The benchmark has at least one warm-up run and three measured runs, and each run's decode speed is its steps over its decode time.
  The share of the decode floor is above 0 and at most 1.5.
- `coherent` is true. `--full-port-rehearsal` waives this field and no other, and the ledger says so (`full-port rehearsal: coherence waived`).

## What a pass does not show

Quality and speed come from one prompt. Perplexity is the port's perplexity of the reference's own greedy continuation, so it says how close the
port's distribution is on that path and not how it does on held-out text. Timings include a pipe round trip per request. The floor is an
estimate at the datasheet bandwidth. A port that returns the reference's ids without computing them cannot be told from one that computed them.
