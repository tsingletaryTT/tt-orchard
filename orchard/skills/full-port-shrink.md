---
name: full-port-shrink
description: Stage 4 of a tt-orchard run started with --full-port, after the port passed stages 2 and 3. Adapt the port to one required chip count that stages 2 and 3 did not show; the supervisor's own check runs it there and compares its tokens with the CPU reference. One prepare step per chip count. Not for a model whose stage 0 class is not full-port.
status: draft. Written on 2026-11-04 for issue 454 (docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md, slice 4). No full port has passed stage 2 on a real model yet, so nothing here has met a chip.
---

# Full port on each required chip count

## When to use this

The port you are given passed stage 2 on one board and, when the run lists a count above one board, stage 3 on the largest count. The run
lists more chip counts (`--required-chips`, for example `4,2,1`). Stage 4 shows the port on each count that stage 2 and stage 3 did not
show. The supervisor makes one entry for each such count and calls you once for each entry, in order of chip count. A count that stage 2
or stage 3 already showed needs no test, and you are not called for it.

Read `full-port-check.md` first if you have not: the port module's four functions, the record's shape and the rules about what the check
may and may not see are the same here. Read `full-port-mesh.md` for how a port places weights on a mesh. This file lists only what differs.

## What differs from stages 2 and 3

- Your stage directory is the entry's directory, `stages/4/configs/<N>/`, where `<N>` is the chip count in the task. It holds your port
  (`port/`), the supervisor's copy of the check (`port_check.py` and `port_check_config.json`), and later its evidence and its marker.
  Write only inside it. Do not write under `stages/2/`, `stages/3/` or another entry's directory: their records are the evidence that the
  earlier stages passed.
- The supervisor copied the files of the port that passed the stage before (stage 3's, or stage 2's when stage 3 was skipped) into
  `port/` before this step.
- The check is the same shipped template. It runs `python3 -P -s stages/4/configs/<N>/port_check.py` under a lease of the entry's chips.
  With one chip it opens a single device and passes it to `load` as `mesh_device`. With more it opens a mesh of one row and N columns with
  the 1-D fabric on.
- The gate requires that the test ran on exactly N chips, and that the record, the config, the template copy and the port sources are all
  under this entry's directory.
- There are no repair rounds in this stage. A failed entry ends the stage, and the next attempt starts again with fresh copies of the
  earlier port.

## Prepare phase

1. Read the record of the stage whose port you were given (`stages/3/evidence/port-check.json`, or `stages/2/evidence/port-check.json`)
   for the load time and the agreement it reached, and read the files in `port/`.
2. Change what the smaller mesh needs. For each weight decide whether it is replicated on every device or sharded across the N devices
   (a count of 1 means no sharding and no collective). Remove a collective that has nothing to gather. Keep `load`, `prefill`,
   `decode_step` and `close` with the same signatures. `prefill` must still start the request afresh.
3. Test the parts that need no chip with ordinary shell commands (shard shapes, that every weight is placed once, a CPU run of your layer
   code against the reference layer). Do not import `ttnn` in your own shell: importing it opens the device, and only the supervisor's test
   holds a lease.
4. Write `hw_test.json`: `{"command": "python3 -P -s stages/4/configs/<N>/port_check.py", "deadline_s": 3600}`. The supervisor runs that
   command whatever the file says; it keeps your `deadline_s` when it is a positive number up to 3600.
5. Write `handoff.json` (`goal`, `stage`, `evidence`, `next_action`, `check_on_return`) and reply with a short summary.

Do not run `port_check.py` yourself. The supervisor writes `result.json` for the entry from the check's record, so there is no finish
phase. A `result.json`, `test-result.json`, `evidence/port-check.json` or `evidence/hw-test-output.txt` found when the prepare phase ends is
moved aside unread.

## What the gate checks

For each entry: that the supervisor's marker for the test lists exit code 0, no timeout and N chips; that the record's chips are the chips the
supervisor leased; the agreement recomputed from the record's `reference_ids` and `chip_ids`, checked against the stage 1 file; that the output
carries the record's sha256 once; that `port_check.py` is the shipped file; and that the files under `configs/<N>/port/` are the files the record
names. The stage's own `result.json` is the summary the gate rebuilds from the entries' records.

## Do not

- Do not write or edit `port-check.json`, `hw-test-output.txt`, `test-result.json`, `covered.json`, `result.json`, `port_check.py` or
  `port_check_config.json`. The supervisor and its check write them.
- Do not change a model source after the test has run. The record holds each file's sha256.
- Do not edit anything under `stages/2/` or `stages/3/`.
- Do not publish, push or write outside your stage directory.
- Do not investigate anything this skill does not list.
