---
name: full-port-mesh
description: Stage 3 of a tt-orchard run started with --full-port, after the port passed stage 2 on one board. Adapt the port to the chip count the run is configured for; the supervisor's own check runs it there and compares its tokens with the CPU reference. Not for a model whose stage 0 class is not full-port.
status: draft. Written on 2026-11-04 for issue 326 (docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md, slice 3). No full port has passed stage 2 on a real model yet, so nothing here has met a chip.
---

# Full port on the configured chips

## When to use this

Stage 2 of this run wrote a port under `stages/2/port/` and the supervisor's check showed it running on one board with tokens that agree
with the stage 1 reference. This stage shows the same port on the chip count the run is configured for: the largest of the run's
`--required-chips`, the large tier. The chip count is in the prepare task. When the run configures no count above one board, the
supervisor skips this stage and you are not called.

Read `full-port-check.md` first if you have not: the port module's four functions, the record's shape and the rules about what the check
may and may not see are the same here. This file lists only what differs.

## What differs from stage 2

- The port lives in `stages/3/port/`. The supervisor copied the files of the stage 2 port there before this step. Edit only
  `stages/3/port/`. Do not write under `stages/2/`: its record and its port are the evidence that stage 2 passed.
- The supervisor's check is the same shipped template, copied to `stages/3/port_check.py` with `stages/3/port_check_config.json`. It runs
  `python3 -P -s stages/3/port_check.py` under a lease of the configured chips. It opens a mesh of one row and N columns with the 1-D fabric
  on when N is more than one, and passes it to `load` as `mesh_device`. Your port gets the mesh and nothing else about it.
- The gate requires that the test ran on exactly the configured number of chips. A port that passes on a mesh of two while the stage is
  configured for four does not pass this stage.
- There are no repair rounds in this stage. A failed test ends the attempt, and the next attempt starts again with a fresh copy of the
  stage 2 port.

## Where you come in

The prepare phase adapts the port, then writes `hw_test.json` and `handoff.json`. The finish phase reads the test output and writes
`result.json`. Write `result.json` nowhere else. A `result.json`, `test-result.json`, `evidence/port-check.json` or
`evidence/hw-test-output.txt` found when the prepare phase ends is moved aside unread.

## Prepare phase

1. Read the stage 2 record (`stages/2/evidence/port-check.json`) for the load time and the agreement the port reached on one board, and
   the files in `stages/3/port/`. Read the nearest model's TT code only as an example of how a layer is placed on a mesh.
2. Change what a mesh needs. Decide for each weight whether it is replicated on every device or sharded across them, and where a layer
   needs a collective (an all-reduce after a sharded matrix product, an all-gather before a replicated one). Keep `load`, `prefill`,
   `decode_step` and `close` with the same signatures. `prefill` must still start the request afresh.
3. Test the parts that need no chip with ordinary shell commands (shard shapes, that every weight is placed once, a CPU run of your layer
   code against the reference layer). Do not import `ttnn` in your own shell: importing it opens the device, and only the supervisor's test
   holds a lease.
4. Write `hw_test.json`: `{"command": "python3 -P -s stages/3/port_check.py", "deadline_s": 3600}`. The supervisor runs that command
   whatever the file says; it keeps your `deadline_s` when it is a positive number up to 3600.
5. Write `handoff.json` (`goal`, `stage`, `evidence`, `next_action`, `check_on_return`) and reply with a short summary.

Do not run `port_check.py` yourself.

## Goal

The finish phase writes `result.json` in your stage directory, in stage 2's shape with stage 3's evidence paths:

    {"runs": true,
     "load_s": 180.0,
     "coherent": true,
     "free_run_text": "<the same text as in the record>",
     "top1_agreement": 0.94,
     "n_tokens": 32,
     "evidence": ["stages/3/evidence/port-check.json", "stages/3/evidence/hw-test-output.txt"]}

The gate does not take your numbers. It recomputes `top1_agreement` from the record's `reference_ids` and `chip_ids`, checks the reference ids
against the stage 1 file, checks that the record's chips are the chips the supervisor leased and that there are as many as the stage is
configured for, checks that the output carries the record's sha256 once, that the record and the check script are the shipped template's and
that the files under `stages/3/port/` match the files the record names. Copy the numbers from the record. A number you cannot copy from the
record is not measured: write null and say why in your summary.

## Finish phase

1. Read `test-result.json` first. If `returncode` is not 0 or `timed_out` is true, write `result.json` with `runs` false, a `failure` field
   that quotes the last lines of the output, and `coherent`, `n_tokens`, `top1_agreement` and `load_s` null. Then stop. Do not investigate
   and do not edit the port: the stage ends and the next attempt runs a fresh prepare and a fresh test.
2. Otherwise read `evidence/port-check.json` and `evidence/hw-test-output.txt` and write `result.json` from them. Set `coherent` true only
   if you read `free_run_text` and it is readable text in the language of the prompt. If it is not, write false and quote it.
3. Reply with a short summary and no tool call.

## Do not

- Do not write or edit `port-check.json`, `hw-test-output.txt`, `test-result.json`, `port_check.py` or `port_check_config.json`. The supervisor
  and its check write them.
- Do not change a model source after the test has run. The record holds each file's sha256.
- Do not edit anything under `stages/2/`.
- Do not publish, push or write outside your stage directory.
- Do not investigate anything this skill does not list.
