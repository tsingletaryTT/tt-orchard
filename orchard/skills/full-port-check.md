---
name: full-port-check
description: Stage 2 of a tt-orchard run started with --full-port, when stage 0 found that the model needs new model code. Write the port module (load, prefill, decode step, close); the supervisor's own check runs it on one leased board and compares the chip's greedy tokens with the stage 1 CPU reference.
status: draft. Written on 2026-11-02 for slices 1 and 2 of the full-port route (docs/superpowers/specs/2026-11-02-full-port-supervisor-design.md) from the hand port of MiniCPM3-4B, which has not been repeated under the supervisor. Nothing here has run on a chip through the supervisor.
---

# Full port check

## When to use this

Stage 0 wrote `delta.json` with `"class": "full-port"` and the operator started the run with `--full-port`. The model has a tensor
layout, a layer type or a config that the nearest model's TT code cannot load as it is. This stage writes the missing model code,
runs it on one board, and compares its greedy tokens with the CPU reference from stage 1.

The stage has one hardware test and no retry loop of its own. If the port does not run, record that and stop (see "Finish phase").

## What the stage produces

Under `stages/2/port/`: the port's source, and nothing else. The one file the check calls is `stages/2/port/port_model.py`. It defines
four functions:

    load(model_dir, mesh_device) -> model
        Load the weights from the snapshot directory onto the open device. The time this takes is the record's `load_s`.
    prefill(model, ids) -> logits_last
        Reset the cache, run the prompt (a list of ints), return the logits for the next token.
    decode_step(model, token_id, position) -> logits
        Append `token_id` at `position` (its index in the whole sequence, prompt included), return the logits for the token after it.
    close(model) -> None
        Release what `load` took. The check closes the device itself.

`prefill` must start the request afresh. The check calls it twice on the same prompt (before the teacher-forced run and before the
free run), and the gate refuses a record where the second call returns a different first id: the serving plugin reuses a decode slot for
a new request and expects prefill to reset that request's state
(`docs/superpowers/specs/2026-11-02-port-interface-vllm-plugin-mapping.md`).

`port_model.py` may import other files in `stages/2/port/`. Every `.py` file there is model code, and at least one must use `ttnn`.
No file there may read the stage 1 reference files: a model file that opens `generated-ids.json` could copy the reference, and the gate
refuses it. A worked example of the same shape is `scripts/minicpm3/chip_model.py`.

You do not write the check. The supervisor owns it. It copies `stages/2/port_check.py` and `stages/2/port_check_config.json` into the stage
directory (read them if you want to see what calls your code) and runs `python3 -P -s stages/2/port_check.py` under the lease. The check
starts your port in a second process, so nothing your module does can change the check's own code, record or output. It does the following:

1. Reads the prompt ids and the reference ids from stage 1 and calls `load`, timing it.
2. Teacher forced: calls `prefill` on the prompt, takes the argmax, then for position `i` feeds reference id `i - 1` through
   `decode_step` and takes the argmax again. It uses up to 64 positions. The port's own id is never fed back, so one wrong token does not
   change the later positions.
3. Free run: calls `prefill` again and decodes 48 tokens greedily from the port's own ids, then decodes them to text with the model's
   `tokenizer.json`.
4. Writes `evidence/port-check.json` itself and prints `PORT_CHECK_SHA256 <the sha256 of that file>` as its last line. The supervisor
   writes the output file. Everything your port prints appears in that output with the prefix `port| `.

The record has this shape (`orchard-port-check/1`):

    {"schema": "orchard-port-check/1",
     "reference": {"path": "stages/1/evidence/reference/generated-ids.json", "sha256": "<of that file as it is on disk>"},
     "prompt": {"path": "stages/1/evidence/reference/prompt-ids.json", "sha256": "<of that file>"},
     "reference_ids": [<the first N generated ids of the reference file>],
     "chip_ids": [<the port's argmax id at each of those N positions>],
     "top1_agreement": 0.94,
     "chips": ["<each chip the test ran on, from TT_VISIBLE_DEVICES>"],
     "load_s": 120.5,
     "free_run_ids": [<the port's own greedy ids>],
     "free_run_text": "<those ids decoded>",
     "port_sources": [{"path": "stages/2/port/port_model.py", "sha256": "<of the file>", "role": "model"}],
     "template_sha256": "<of the shipped port_check.py>",
     "config": {"path": "stages/2/port_check_config.json", "sha256": "<of that file>"}}

Do not edit `port_check.py` or `port_check_config.json`. The supervisor puts both back as shipped after your prepare step, and the gate
compares the copy and the record's `template_sha256` with the shipped file. A port that prints a line mentioning `PORT_CHECK` fails the
run. A port that writes `evidence/port-check.json` has it deleted before the check and replaced after it.

## Where you come in

The prepare phase writes the port and `hw_test.json`, then `handoff.json`. The finish phase reads the test output and writes
`result.json`. Write `result.json` nowhere else. A `result.json`, `test-result.json`, `evidence/port-check.json` or
`evidence/hw-test-output.txt` found when the prepare phase ends is moved aside unread, and the values below are an example only.

## Prepare phase

1. Read `stages/0/delta.json` (`path_reasons`, `differences`, `hazards`, `alignment`) and the stage 1 evidence. They say what differs
   from the nearest model and what the reference produced. Read the nearest model's TT code only as an example of how a layer is written
   in ttnn. Do not edit files outside `stages/2/`.
2. Write the port under `stages/2/port/` and test the parts that need no chip (weight names, shapes, a CPU run of your own layer
   code against the reference layer) with ordinary shell commands. Do not import `ttnn` in your own shell: importing it opens the
   device, and only the supervisor's test holds a lease.
3. Write `hw_test.json`: `{"command": "python3 -P -s stages/2/port_check.py", "deadline_s": 3600}`. The supervisor runs that command
   whatever the file says; it keeps your `deadline_s` when it is a positive number up to 3600.
4. Write `handoff.json` (`goal`, `stage`, `evidence`, `next_action`, `check_on_return`) and reply with a short summary.

Do not run `port_check.py` yourself.

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
output carries the record's sha256 once, that the record and the check script are the shipped template's and that the port's source files match the files the record names. Copy the numbers from the record.
A number you cannot copy from the record is not measured: write null and say why in your summary.

## Finish phase

When the hardware test passed, the record is complete and `free_run_text` is readable text, the supervisor writes `result.json` itself from
the record (it adds `derived_by: supervisor` and says how it judged `coherent`), checks it with the gate's own check, and does not start this
phase. You get this phase when it did not: the test failed or has no record, the text did not pass its floor, or its result did not pass the
check. Then write `result.json` as described below, and read `free_run_text` yourself.

1. Read `test-result.json` first. If `returncode` is not 0 or `timed_out` is true, write `result.json` with `runs` false, a `failure` field
   that quotes the last lines of the output, and `coherent`, `n_tokens`, `top1_agreement` and `load_s` null. Then stop. Do not
   investigate and do not edit the port: the stage ends and the next attempt runs a fresh prepare and a fresh test.
2. Otherwise read `evidence/port-check.json` and `evidence/hw-test-output.txt` and write `result.json` from them. Set `coherent` true
   only if you read `free_run_text` and it is readable text in the language of the prompt. If it is not, write false and quote it.
3. Reply with a short summary and no tool call.

## Do not

- Do not write a file longer than 6,000 lines or 400,000 bytes, and do not repeat a block of text: the supervisor refuses such a file, puts the
  old one back and tells you. A port is a few hundred lines. If your replies keep being cut off, you are repeating yourself: stop analysing,
  run one command or write one file, and keep the reply short.
- Do not write or edit `port-check.json`, `hw-test-output.txt`, `test-result.json`, `port_check.py` or `port_check_config.json`. The supervisor and its check write them.
- Do not change a model source after the test has run. The record holds each file's sha256.
- Do not publish, push or write outside your stage directory.
- Do not investigate anything this skill does not list.
