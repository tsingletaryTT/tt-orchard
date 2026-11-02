---
name: full-port-check
description: Stage 2 of a tt-orchard run started with --full-port, when stage 0 found that the model needs new model code. Write the port, run it on one leased board, and compare the chip's greedy tokens with the stage 1 CPU reference.
status: draft. Written on 2026-11-02 for slice 1 of the full-port route (docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md) from the hand port of MiniCPM3-4B, which has not been repeated under the supervisor. Nothing here has run on a chip through the supervisor.
---

# Full port check

## When to use this

Stage 0 wrote `delta.json` with `"class": "full-port"` and the operator started the run with `--full-port`. The model has a tensor
layout, a layer type or a config that the nearest model's TT code cannot load as it is. This stage writes the missing model code,
runs it on one board, and compares its greedy tokens with the CPU reference from stage 1.

The stage has one hardware test and no retry loop of its own. If the port does not run, record that and stop (see "Finish phase").

## What the stage produces

Under `stages/2/port/`: the port's source. Mark each file with one of two roles in the record:

- `model`: the model code (weight loading, layers, prefill, decode step). It must use `ttnn`. It must not read the stage 1 reference
  files: a model file that opens `generated-ids.json` could copy the reference into the record, and the gate refuses it.
- `check`: the script the hardware test runs. It loads the model code, runs the teacher-forced comparison and writes the record.

The check script writes `evidence/port-check.json` and, as its last act, prints one line to its output:

    PORT_CHECK_SHA256 <the sha256 of evidence/port-check.json, 64 hex digits>

The supervisor writes the output file itself, so a record that the output does not name did not come from the test the supervisor ran.

The record has this shape (`orchard-port-check/1`):

    {"schema": "orchard-port-check/1",
     "reference": {"path": "stages/1/evidence/reference/generated-ids.json", "sha256": "<of that file as it is on disk>"},
     "prompt": {"path": "stages/1/evidence/reference/prompt-ids.json", "sha256": "<of that file>"},
     "reference_ids": [<the first N generated ids of the reference file>],
     "chip_ids": [<the chip's argmax id at each of those N positions>],
     "chips": ["<each chip the test ran on, from TT_VISIBLE_DEVICES>"],
     "load_s": 120.5,
     "free_run_text": "<the text the port produced when it ran free from the prompt>",
     "port_sources": [{"path": "stages/2/port/model.py", "sha256": "<of the file>", "role": "model"},
                      {"path": "stages/2/port/check.py", "sha256": "<of the file>", "role": "check"}]}

How `chip_ids` is made: teacher-forced. Feed the prompt ids from `prompt-ids.json`, then at position `i` take the chip's argmax and
append the reference id `i` (the chip's own id is never appended), so one wrong token does not change the later positions. Use at least 32 positions
when the reference has them. Read the reference ids from `generated-ids.json` (key `generated_ids`) in the check script and never in the
model code.

## Where you come in

The prepare phase writes the port, the check script, `hw_test.json` and `handoff.json`. The finish phase reads the test output and
writes `result.json`. Write `result.json` nowhere else. A `result.json`, `test-result.json`, `evidence/port-check.json` or
`evidence/hw-test-output.txt` found when the prepare phase ends is moved aside unread, and the values below are an example only.

## Prepare phase

1. Read `stages/0/delta.json` (`path_reasons`, `differences`, `hazards`, `alignment`) and the stage 1 evidence. They say what differs from
   the nearest model and what the reference produced. Read the nearest model's TT code only as an example of how a layer is written
   in ttnn. Do not edit files outside `stages/2/`.
2. Write the port under `stages/2/port/` and test the parts that need no chip (weight names, shapes, a CPU run of your own layer
   code against the reference layer) with ordinary shell commands. Do not import `ttnn` in your own shell: importing it opens the
   device, and only the supervisor's test holds a lease. The harness runs the check script on a leased board.
3. Write `hw_test.json`: `{"command": "python3 stages/2/port/check.py", "deadline_s": 3600}`.
4. Write `handoff.json` (`goal`, `stage`, `evidence`, `next_action`, `check_on_return`) and reply with a short summary.

Do not run `check.py` yourself.

## Goal

The finish phase writes `result.json` in your stage directory:

    {"runs": true,
     "load_s": 120.5,
     "coherent": true,
     "free_run_text": "<the same text as in the record>",
     "top1_agreement": 0.94,
     "n_tokens": 32,
     "evidence": ["stages/2/evidence/port-check.json", "stages/2/evidence/hw-test-output.txt"]}

The gate needs `runs` true, `coherent` true, `n_tokens` of at least 16, `top1_agreement` of at least 0.85, a positive `load_s`, and both
evidence files. The gate does not take your numbers. It recomputes `top1_agreement` from `reference_ids` and `chip_ids` in the record,
checks the reference ids against the stage 1 file on disk, checks that the record's chips are the chips the supervisor leased, that the
output carries the record's sha256 and that the port's source files are the ones the record names. Copy the numbers from the record.
A number you cannot copy from the record is not measured: write null and say why in your summary.

## Finish phase

1. Read `test-result.json` first. If `returncode` is not 0 or `timed_out` is true, write `result.json` with `runs` false, a `failure` field
   that quotes the last lines of the output, and `coherent`, `n_tokens`, `top1_agreement` and `load_s` null. Then stop. Do not
   investigate and do not edit the port: the stage ends and the next attempt runs a fresh prepare and a fresh test.
2. Otherwise read `evidence/port-check.json` and `evidence/hw-test-output.txt` and write `result.json` from them. Set `coherent` true
   only if you read `free_run_text` and it is readable text in the language of the prompt. If it is not, write false and quote it.
3. Reply with a short summary and no tool call.

## Do not

- Do not write or edit `port-check.json`, `hw-test-output.txt` or `test-result.json`. The test writes them.
- Do not change a model source after the test has run. The record holds each file's sha256.
- Do not publish, push or write outside your stage directory.
- Do not investigate anything this skill does not list.
