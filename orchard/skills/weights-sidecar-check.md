---
name: weights-sidecar-check
description: Stage 2 of a tt-orchard run when stage 0 found class weights+sidecar. Do everything the weights swap check does, and in the same hardware lease measure whether the sidecar head, run on the host from the chip's final hidden states, agrees with the same head run on the CPU reference.
status: draft. Written 2026-10-06 for Cloudflare/clef (a Qwen3.8-27B backbone plus a task head in a separate file). The scripts it names are tested with fakes; the parity script has been run on hardware only in the prototype recorded in docs/run-logs. It extends weights-swap-check and keeps that skill's facts.
---


# Weights swap check, and the sidecar parity

## When to use this

Stage 0 wrote `delta.json` with `"class": "weights+sidecar"` and `"path": "weights-only"`: the new model's
backbone has the same text architecture as a model that already runs on this machine, and only the weights
differ, and it also ships a sidecar (a head in its own weights file, listed in `delta.json` under
`sidecars`, with its sha256). The chips run the backbone. The sidecar runs on the host, on the chip's final
hidden states. Then no decoder needs to be written. Do not read or copy tt-metal model code. The existing TT
implementation loads the new weights as they are. This stage shows that it does, that the chip's
output agrees with the CPU reference from stage 1, and that the sidecar head gives the same answers from the
chip's hidden states as from the CPU reference's.

## Goal

Write `result.json` in your stage directory:

    {"serves": true,
     "server_ready_s": 280.5,
     "coherent": true,
     "free_run_text": "first 200 characters of the chip's greedy text",
     "top1_agreement": 0.94,
     "n_tokens": 32,
     "cache_dir": "<the tensor cache directory used>",
     "sidecar_parity": {"measured": true, "n_questions": 17, "hidden_pcc_min": 0.997,
                        "top1_agree_fraction": 1.0, "prob_max_abs_diff": 0.01, "...": "..."},
     "evidence": ["stages/2/evidence/swap-check.json", "stages/2/evidence/server.log",
                  "stages/2/evidence/sidecar-parity.json"]}

You do not type these numbers. A script writes them into `stages/2/evidence/sidecar-check.json`, and you
copy its `result_draft`. The gate needs `serves` true, `coherent` true, `n_tokens` of at least 16,
`top1_agreement` of at least 0.85, a positive `server_ready_s`, and a `sidecar_parity` object: measured,
its wiring check passed, enough questions, hidden-state correlation and top-option agreement above the
bars, and a probability difference below the bar. It must also name the exact head file and code file
stage 0 recorded (their sha256 values), and every evidence path must exist. Label each number measured
or TODO.

## How to work

Two tested scripts do all of the work. You copy them and give them one config file. You do not
write a script.

- Write `swap_config.json` FIRST, as soon as you know the facts below.
- Write each file as soon as you know its content. Do not wait until the end.
- Do not investigate anything this skill does not list. On an earlier run the model spent 60 turns
  grepping vLLM source for an unrelated timeout and wrote nothing.

## Why the scripts do what they do

Four facts decide whether a swap works. Each was found by failing first.

1. The model directory must look like the nearest model's. The bundle's TT model class is
   registered for the nearest model's architecture, which is a vision-language model. So the
   config files come from the nearest model, and the tokenizer and weights come from the new model.
2. The tensor cache must be empty or belong to this model. The bundle reads a cache keyed only by
   layer name, so another model's cache serves that model's weights without any error. The
   script refuses a non-empty cache that lacks its `.orchard-model` marker.
3. The bundle's `run.sh` is run from a copy with these edits: `HERE=` points at the bundle,
   `--model` points at the model directory, the `--revision` flags are removed (a local
   directory has no revision), and `HF_MODEL` is set to the model directory.
4. The server must be told where the new weights are. The TT runtime takes its weights directory
   from `MODEL_WEIGHTS_DIR`, then `HF_MODEL`, then the config path. The bundle's `run.sh` exports
   `HF_MODEL` as the nearest model's id, so `--model` alone gives vLLM the new config and tokenizer
   while the chip loads the nearest model's weights. The first hand prototype did exactly that: it
   agreed with the Hemmingway-1 CPU reference on 25 of 32 tokens. With `MODEL_WEIGHTS_DIR` set to
   the model directory it agreed on 30 of 32, and its free-run text matched the CPU text. The
   template sets `MODEL_WEIGHTS_DIR` and `HF_MODEL` for you and records both in `swap-check.json`.

## Prepare phase: the steps

1. Find four facts with a few commands:
   - `bundle_dir`: the installed `tt-model` bundle that serves the nearest model. Read
     `delta.json` for `nearest_model` and its `architecture`. Run `tt-model list`, then read
     `{{TT_MODEL_ROOT}}/<org>/<name>/vllm_models/*/vllm_metadata.json`. Pick the bundle whose
     `arch` equals the nearest model's architecture and whose chip count is the smallest that fits
     (2 chips on this machine). On this machine that is
     `{{TT_MODEL_ROOT}}/episod/qwen3.8-27b-dflash2-p300`. It has `run.sh` and `venv/`.
   - `base_snapshot`: the nearest model's snapshot directory,
     `{{HF_HOME}}/hub/models--<org>--<name>/snapshots/<sha>/` (use `ls` to find it).
   - `new_snapshot`: the new model's snapshot directory, found the same way.
   - `port`: a free port, for example 8100 (`ss -ltn` lists the ports in use).
2. Write `swap_config.json` in your stage directory (absolute paths):

       {"run_dir": "<the run directory>",
        "bundle_dir": "<bundle directory>",
        "nearest_model_id": "Qwen/Qwen3.8-27B",
        "base_snapshot": "<nearest model snapshot>",
        "new_snapshot": "<new model snapshot>",
        "new_model_id": "<the model value from stages/0/delta.json, for example Altworld/Hemmingway-1@<revision>>",
        "tt_cache": "<a new directory whose name contains the model's slug>",
        "hf_home": "{{HF_HOME}}",
        "port": 8100}

   Copy the `model` value from `stages/0/delta.json` verbatim, including the `@revision`, into
   `new_model_id`. Stage 7 compares this label with stage 0's model and the revision of the weights
   the test served, so the label must name the same weights.

   `tt_cache` must be a new directory, for example `{{CACHE_ROOT}}/<slug>/tt_cache`, or one this
   run already used for this model. Never point it at
   `{{OPERATOR_HOME}}/.cache/qwen36-src-build/...`. `hf_home` is the operator's Hugging Face cache
   on this machine. Your shell's HOME points somewhere else, so write the path out as shown. It
   holds the drafter the run script names in `DFLASH_WEIGHTS`.
3. Write `parity_config.json` in your stage directory. Every value comes from `stages/0/delta.json` or from
   the facts above:

       {"run_dir": "<the run directory>",
        "bundle_dir": "<the same bundle directory as in swap_config.json>",
        "model_snapshot": "<new model snapshot>",
        "head_file": "<the sidecar weights file name in delta.json sidecars, for example joint_head.safetensors>",
        "head_config": "<the sidecar's config file name in the snapshot, for example joint_head_config.json>",
        "code_file": "<the code file name in delta.json code_files, for example joint_schema_model.py>",
        "code_sha256": "<that code file's sha256 from delta.json code_files>",
        "tt_cache": "<a second new directory, next to the first, whose name contains the model's slug>",
        "mesh_shape": [1, 2]}

   Copy `code_sha256` from `delta.json` exactly. The parity script refuses to run when the code file's hash
   differs from it, because the sidecar code is third-party code and only the file stage 0 recorded may be
   used. The script hashes the head file itself, and the gate compares that hash with the one stage 0
   recorded. Do not open or run the code file yourself.
4. Copy the templates into your stage directory:

       cp {{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/prepare_swap.py {{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/serve_and_compare.py stages/2/
       cp {{ORCHARD_DIR}}/orchard/skills/sidecar-parity-templates/prepare_parity.py {{ORCHARD_DIR}}/orchard/skills/sidecar-parity-templates/hidden_parity.py {{ORCHARD_DIR}}/orchard/skills/sidecar-parity-templates/run_sidecar_checks.py stages/2/

5. Run `python3 stages/2/prepare_swap.py`, then `python3 stages/2/prepare_parity.py`. Each prints a summary.
   Exit 2 means a fact in the config does not fit; its message names the key or the edit. Fix that fact and
   run it again.
6. Check that `stages/2/model-dir`, `stages/2/run.sh`, `stages/2/parity-model-dir` and `stages/2/parity-run.sh`
   exist (`ls -l`). The two scripts build different directories on purpose; never copy files between them.
7. Write `hw_test.json`: `{"command": "python3 stages/2/run_sidecar_checks.py", "deadline_s": 7200}`.
8. Write `handoff.json` and reply with a short summary.

Do not run `serve_and_compare.py`, `hidden_parity.py` or `run_sidecar_checks.py` yourself. The supervisor runs
the last one on a leased board. It runs the swap check and then the parity check, one after the other.

## What serve_and_compare.py measures

It starts the server, waits for `/health` (up to 3300 s). The first boot converts the weights
(about 5 minutes) and compiles every kernel with a cold compile cache, which took more than 26 minutes
on 2026-10-03; a boot with warm caches takes about 2 minutes, and asks for 32 greedy tokens from the stage 1 prompt. Then, for each of the
32 reference positions, it asks for one token with the reference prefix and compares the first
re-tokenized id with the reference id. Near-synonyms differ, because the chip runs quantized
weights and the reference runs bf16. With the correct weights the prototype matched 30 of 32
(0.94). With the base model's weights standing in by mistake it matched 25 of 32 (0.78), because
the two models are close. Agreement under 0.85 usually means the server is not running the new
weights. It writes `stages/2/evidence/swap-check.json`, whose `result_draft` holds the fields of
`result.json`. Exit codes: 0 measured, 3 cache refused, 4 the server did not become healthy, 5 the
server answered with an HTTP error.

## What hidden_parity.py measures

It loads the new model's backbone on the leased board and, for a few fixed text records, reads back the
final hidden state of every token. It runs the sidecar head on the host from those states and from the CPU
reference's states, and compares the head's answers. It writes `stages/2/evidence/sidecar-parity.json`. A
low hidden-state correlation means the chip's backbone differs from the reference. A good correlation with
different head answers means the head was fed something other than the final hidden states. Exit 6 means its
wiring check failed: the loop it runs no longer matches the model, so its numbers would mean nothing.

## Finish phase

Read `stages/2/evidence/hw-test-output.txt` and `stages/2/evidence/sidecar-check.json`. If the test
measured, copy `result_draft` into `result.json`. It already holds `sidecar_parity` and the evidence list.

If the test failed (`test-result.json` shows a `returncode` other than 0, or `timed_out` true),
write `result.json` with `serves` false, the failure text from `hw-test-output.txt` in a `failure`
field, `coherent`, `n_tokens`, `top1_agreement` and `server_ready_s` null, and
`stages/2/evidence/hw-test-output.txt` in `evidence`. Do not write a `sidecar_parity` object you did not
get from the script. Then stop.
Do not investigate firmware or cache directories, and do not try to make the result pass. Do not
invent a number. The gate fails this result on purpose, and the next attempt runs a fresh test.

## Do not

- Do not modify the bundle, the nearest model's snapshot, or any tensor cache you did not create.
- Do not write model code or your own serving script.
- Do not open, read or run the sidecar's code file. The scripts import it only after its hash matches.
- Do not compute or type a parity number. Copy the script's `result_draft`.
- Do not spend turns reading tt-metal or vLLM source.
