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

## Start here

The supervisor has already put two things in the stage directory, and the prepare task tells you so:

- `stages/2/port_facts.json`: the config's small fields, the layer schedule by type, every tensor's dtype and shape (read from the
  checkpoint's own headers, whether it is one `model.safetensors` or shards), and, for config fields that hold layer ids, the index
  convention the reference class states for them. Read it first, once. It is written from the checkpoint, so you do not need to read
  the config, the weight headers or the weights again to learn shapes. A checkpoint with one file has no `model.safetensors.index.json`;
  the file says which case this is. When a config id list says `one-indexed`, layer index = id - 1. When the file says the convention
  was not found, compare the ids with the layer numbers of the tensors before you use them. The file names what it left out for size:
  `tensors_left_out`, `config_left_out`, `text_config_left_out` (with `*_left_out_more` for names beyond the first 100),
  `config_keys_not_listed`, `text_config_keys_not_listed`, `conventions_left_out`, `layer_schedule_note` and `tensors_with_long_names`. A
  convention note names its reference class; use the class for the section the field is in, and read another class's different numbering as
  a warning to check the tensors. `layer_flag_fields` hold one 0 or 1 per layer and no layer ids. A field in a `*_left_out` list may well be in the checkpoint's config; read `config.json` for that one field.
- `stages/2/effective_construction.txt` (when the reference interpreter could build the model; `port_facts.json` carries its hash and the
  first lines of its list under `effective_construction`): what each module's constructor resolved from the config, and under "Resolved vs class
  default" every setting that differs from the default in the class's signature. The reference excerpts elide constructor bodies, so a default
  there (`activation: str = "silu"`) may not be the value the model uses (the tiny Qwen4's gated norm resolves to `sigmoid`). Read the lines for the
  classes you port before you write them, and use the resolved value.
- `stages/2/port/port_model.py`: a skeleton. Each of the four functions raises `NotImplementedError` until you replace its body, and
  `read_tensors(model_dir)` returns every tensor by name from one file or from shards. A port left as the skeleton fails the check at
  `load` and says so.

Write a first version of `port_model.py` within 12 turns, even if it is incomplete, and refine it after. A stage that reads for twenty
turns and writes nothing is stopped by the watchdog and costs a coder reboot. Do not page through the stage 1 reference files: the
check reads them and your port may not. Do not search the machine for other models' code; the worked example named below is enough.
If `port_facts.json` is not there, the supervisor could not read the checkpoint (the ledger says why); collect the same facts yourself
from `config.json` and the weight headers, once.

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

A port may also define one optional function, used only to localise a numerics failure:

    layer_outputs(model, ids) -> list
        Run the prompt `ids` once and return the hidden state after each decoder layer, in order, as torch tensors or numpy arrays.

The check calls it only when the agreement ends below the gate's bar, after the comparison, in the same hardware test and under the same lease.
It never changes the record, the exit code or the gate. The check saves the arrays to `stages/2/evidence/layer-port.npz` and prints one
`PORT_CHECK_LAYERS <sha256> <count>` line that the supervisor's capture of the output keeps. `python3 stages/2/stage_tools.py layer-diff` compares
that file with the CPU reference layer by layer (in the reference interpreter) and names the first layer whose relative error is over a threshold
(default 0.05). The same tool file also has `gate-check`, `last-check`, `facts` and `show`; the task text lists them. `gate-check` runs the board check's device-ops test on your port in seconds with no board: run it before you reply, because a port with no `ttnn.linear` or `ttnn.matmul` call fails the board check. `last-check` ends with a `next:` line (the one command to run) and shows the check from before the current step; it does not run the check again.

If `load` writes a converted copy of the weights to disk (a cache under the cache root that the first boot fills), say so in
`stages/2/port/converted_cache.json`: `{"writes_converted_cache": true, "cache_dir": "<where>"}`. The supervisor reads it before the hardware
test and blocks when the cache disk has less free space than the first boot needs. If `load` converts straight onto the chips and writes no
cache, write `{"writes_converted_cache": false}`. With no file the supervisor does not check the disk for a cache and records that nothing was
declared. The file is your statement and it is read once: the supervisor hashes it into the ledger before the test.

`prefill` must start the request afresh. The check calls it twice on the same prompt (before the teacher-forced run and before the
free run), and the gate refuses a record where the second call returns a different first id: the serving plugin reuses a decode slot for
a new request and expects prefill to reset that request's state
(`docs/superpowers/specs/2026-11-02-port-interface-vllm-plugin-mapping.md`).

`port_model.py` may import other files in `stages/2/port/`. Every `.py` file there is model code, and the port as a whole must make
`ttnn.<op>(` calls: an `import ttnn` line does not count. The check counts the calls the port makes (issue 541). It fails the run, before
the device is used, when no file has a reachable `ttnn.matmul`, `ttnn.linear` or similar call. It fails the first prefill when the pass
makes fewer matmul-family calls than the model has layers (one per layer is the floor). It fails a decode step that makes fewer, unless
the step replays a captured trace. It fails the run after `load` when most of the weight elements are still host torch or numpy tensors.
A port that computes in torch on the host and only imports `ttnn` therefore fails with a message that says to upload the weights with
`ttnn.from_torch(..., device=mesh_device)` and run the projections with `ttnn.linear` or `ttnn.matmul`. The record keeps the counts under
`device_ops`, and the gate checks them again.
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

0. If `stages/2/reference_excerpts.md` exists, read it first, before any other file. It holds the source of the reference classes your
   model uses (constructor signature, `forward` method and the small helper functions `forward` calls), taken from the installed
   `transformers` package, and its header lists what was cut. Write a first `stages/2/port/port_model.py` from it within your first
   turns, even if it is incomplete, and only then read other files. Do not page through the reference modeling module with `sed` or
   `read_file`: the excerpts file already holds the classes you need, and a step that reads for 20 turns without writing a file gets a
   nudge and then a pause.
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

## Repair phase

When the first check fails because the port did, the supervisor gives you up to three repair rounds in the same attempt (the run's
`--port-repair-rounds` sets the number). A failure of the port is one the check names: the port raised an error or did not answer at
`load`, `prefill` or `decode`, or its files are wrong. A failure of the run around the port (no chip, a stage 1 file that does not
read), a timeout and any other exit code earn no round.

A round gives you the round number, where the check failed and the end of its output. Read the failure and the port, find the
cause, and edit only the files under `stages/2/port/`. A mistake that the failure text points at is usually one line: a weight key the
checkpoint does not have, a missing import, an id read from zero when the config counts from one. Read the config and the checkpoint's
tensor names before you change the port. Do not write any other file and do not run the check. The supervisor runs the same check
again on the same chips and writes the new record. A change to `test-result.json` or to the output fails the attempt. If you change
nothing, the rounds end. After the last round the stage goes on as a failed test: the finish phase records it and the next attempt
starts again.

## Reading a disagreement

A low top-1 agreement does not always mean a wrong port. Look at the line that says how many disagreements are near-ties of the reference (the
record's `near_ties` block). At a near-tie the reference's own top two logits differ by at most 0.125, which is two bf16 steps at a logit of 8
to 16, so any bf16 port can flip it. A skills-only port of the tiny Qwen4 had 3 mismatches in about 300 positions, and all 3 were such ties.
The block is reported and not gated, and the gate recomputes it from the stage 1 file. A disagreement outside the ties is where to look.

To find where a real disagreement starts, compare each layer's output with the reference's at the same input. A good port keeps every layer
at a correlation of 0.9999 or better. The first layer that drops names the block to read. A block that picks discrete items from scores (a
sparse-attention indexer's top-k blocks, a mixture router's experts) can choose differently when two scores are close. Look at the score
margin of the row that flipped before you change the port. Computing the scores in float32 narrows the margin where this happens and does
not remove it.

A causal model gives the same predictions from one forward pass over the whole sequence as from feeding the tokens one at a time. A port can use
that to check itself: `prefill` of a prompt and `prefill` of a shorter prompt followed by `decode_step` must agree. Check lengths that sit on a
boundary of the tile size: 1, 2 and 3 tokens, exactly 32, and 33.

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
   investigate and do not edit the port: repair rounds, when the failure was the port's, have already been used, so the stage ends
   and the next attempt runs a fresh prepare and a fresh test.
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
