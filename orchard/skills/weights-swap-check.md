---
name: weights-swap-check
description: Stage 2 of a tt-orchard run when stage 0 found a weights-only delta. Load the new weights into the existing TT implementation of the nearest model, serve them on one leased board, and compare the chip's output with the stage 1 CPU reference.
status: draft. A local tt-orchard copy, written from one hand prototype on 2026-10-03 (Altworld/Hemmingway-1 on the 2-chip Qwen3.8-27B bundle). Since 2026-10-03 the work is done by two tested template scripts that you copy. Its home is the tt-model-bringup plugin in tenstorrent/skills, after a harness run has used it.
---

# Weights swap check

## When to use this

Stage 0 wrote `delta.json` with `"path": "weights-only"`: the new model has the same text architecture
as a model that already runs on this machine, and only the weights (and maybe the tokenizer) differ.
Then no decoder needs to be written. Do not read or copy tt-metal model code. The existing TT
implementation loads the new weights as they are. This stage shows that it does, and that the chip's
output agrees with the CPU reference from stage 1.

## Goal

Write `result.json` in your stage directory:

    {"serves": true,
     "server_ready_s": 280.5,
     "coherent": true,
     "free_run_text": "first 200 characters of the chip's greedy text",
     "top1_agreement": 0.94,
     "n_tokens": 32,
     "cache_dir": "<the tensor cache directory used>",
     "evidence": ["stages/2/evidence/swap-check.json", "stages/2/evidence/server.log"]}

The gate needs `serves` true, `coherent` true, `n_tokens` of at least 16, `top1_agreement` of at
least 0.85, a positive `server_ready_s`, and every evidence path to exist. Label each number measured
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
     `~/.cache/tt-model/models/<org>/<name>/vllm_models/*/vllm_metadata.json`. Pick the bundle whose
     `arch` equals the nearest model's architecture and whose chip count is the smallest that fits
     (2 chips on this machine). On this machine that is
     `~/.cache/tt-model/models/episod/qwen3.8-27b-dflash2-p300`. It has `run.sh` and `venv/`.
   - `base_snapshot`: the nearest model's snapshot directory,
     `~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<sha>/` (use `ls` to find it).
   - `new_snapshot`: the new model's snapshot directory, found the same way.
   - `port`: a free port, for example 8100 (`ss -ltn` lists the ports in use).
2. Write `swap_config.json` in your stage directory (absolute paths):

       {"run_dir": "<the run directory>",
        "bundle_dir": "<bundle directory>",
        "nearest_model_id": "Qwen/Qwen3.8-27B",
        "base_snapshot": "<nearest model snapshot>",
        "new_snapshot": "<new model snapshot>",
        "new_model_id": "<new model id, for example Altworld/Hemmingway-1>",
        "tt_cache": "<a new directory whose name contains the model's slug>",
        "hf_home": "/home/ttuser/.cache/huggingface",
        "port": 8100}

   `tt_cache` must be a new directory, for example
   `/mnt/bonus/models/orchard-runs/cache/<slug>/tt_cache`, or one this run already used for this
   model. Never point it at `~/.cache/qwen36-src-build/...`. `hf_home` is the operator's Hugging
   Face cache on this machine. Your shell's HOME points somewhere else, so write the path out. It
   holds the drafter the run script names in `DFLASH_WEIGHTS`.
3. Copy the two templates into your stage directory:

       cp /home/ttuser/code/tt-orchard/orchard/skills/weights-swap-templates/prepare_swap.py /home/ttuser/code/tt-orchard/orchard/skills/weights-swap-templates/serve_and_compare.py stages/2/

4. Run `python3 stages/2/prepare_swap.py`. It builds `stages/2/model-dir/` and `stages/2/run.sh` and
   prints a summary. Exit 2 means an edit of `run.sh` did not apply; its message names the edit.
   Fix the fact in `swap_config.json` that it points to and run it again.
5. Check that `stages/2/model-dir` and `stages/2/run.sh` exist (`ls -l`).
6. Write `hw_test.json`: `{"command": "python3 stages/2/serve_and_compare.py", "deadline_s": 2400}`.
7. Write `handoff.json` and reply with a short summary.

Do not run `serve_and_compare.py` yourself. The supervisor runs it on a leased board.

## What serve_and_compare.py measures

It starts the server, waits for `/health` (up to 1500 s; the first boot converts the weights,
about 5 minutes here), and asks for 32 greedy tokens from the stage 1 prompt. Then, for each of the
32 reference positions, it asks for one token with the reference prefix and compares the first
re-tokenized id with the reference id. Near-synonyms differ, because the chip runs quantized
weights and the reference runs bf16. With the correct weights the prototype matched 30 of 32
(0.94). With the base model's weights standing in by mistake it matched 25 of 32 (0.78), because
the two models are close. Agreement under 0.85 usually means the server is not running the new
weights. It writes `stages/2/evidence/swap-check.json`, whose `result_draft` holds the fields of
`result.json`. Exit codes: 0 measured, 3 cache refused, 4 the server did not become healthy, 5 the
server answered with an HTTP error.

## Finish phase

Read `stages/2/evidence/hw-test-output.txt` and `stages/2/evidence/swap-check.json`. If the test
measured, copy `result_draft` into `result.json`.

If the test failed, write `result.json` with `serves` false and the failure text from
`hw-test-output.txt`. Do not investigate firmware or cache directories. Do not invent a number.

## Do not

- Do not modify the bundle, the nearest model's snapshot, or any tensor cache you did not create.
- Do not write model code or your own serving script.
- Do not spend turns reading tt-metal or vLLM source.
