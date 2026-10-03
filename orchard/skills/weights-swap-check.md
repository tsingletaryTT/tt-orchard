---
name: weights-swap-check
description: Stage 2 of a tt-orchard run when stage 0 found a weights-only delta. Load the new weights into the existing TT implementation of the nearest model, serve them on one leased board, and compare the chip's output with the stage 1 CPU reference.
status: draft. A local tt-orchard copy, written from one hand prototype on 2026-10-03 (Altworld/Hemmingway-1 on the 2-chip Qwen3.8-27B bundle). Its home is the tt-model-bringup plugin in tenstorrent/skills, after a harness run has used it.
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
     "top1_agreement": 0.78,
     "n_tokens": 32,
     "cache_dir": "<the tensor cache directory used>",
     "evidence": ["stages/2/evidence/swap-check.json", "stages/2/evidence/server.log"]}

The gate needs `serves` true, `coherent` true, `n_tokens` of at least 16, `top1_agreement` of at
least 0.6, a positive `server_ready_s`, and every evidence path to exist. Label each number measured
or TODO.

## What to find out first (a few commands, not a survey)

1. Which installed `tt-model` bundle serves the nearest model? Read `delta.json` for `nearest_model`
   and its `architecture`. Run `tt-model list`, then read
   `~/.cache/tt-model/models/<org>/<name>/vllm_models/*/vllm_metadata.json` for each bundle. Pick the
   bundle whose `arch` equals the nearest model's architecture and whose chip count is the smallest
   that fits (2 chips on this machine). On this machine that is `episod/qwen3.8-27b-dflash2-p300`.
   The bundle has `run.sh` and a `venv/`. Do not modify the bundle.
2. The stage 1 reference is in `stages/1/evidence/reference/`: `prompt-ids.json` (key `prompt_ids`) and
   `generated-ids.json` (keys `generated_ids`, `generated_text`).

## The recipe

Three facts decide whether this works. Each was found by failing first.

1. **The model directory must look like the nearest model's.** The bundle registers its TT model class
   for one architecture name, and the nearest model is a vision-language model. A text-only fine-tune
   declares a different architecture and a flat config, and vLLM fails in its multimodal setup. Build a
   directory `model-dir/` under your stage directory (or the shared cache area, see below) like this:
   copy `config.json`, `preprocessor_config.json` and `video_preprocessor_config.json` from the
   nearest model's snapshot (on this machine under
   `~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<sha>/`; use `ls` to find the one snapshot); symlink the NEW model's `tokenizer.json`, `tokenizer_config.json`,
   `chat_template.jinja`, `generation_config.json`, `model.safetensors.index.json` and every
   `model-*.safetensors` file (including `model-mtp.safetensors` if present). The runtime skips the
   vision tower and accepts both `model.*` and `model.language_model.*` tensor names, so the missing
   vision weights do not matter. Use `ln -s` through Python (`os.symlink`) so the paths are exact.
2. **The tensor cache must be empty and belong to this model.** The bundle converts weights into a
   cache keyed only by layer name. A cache made for another model is read without complaint and
   serves that model's weights. Use a new directory whose name contains the model's slug, for example
   `/mnt/bonus/models/orchard-runs/cache/<slug>/tt_cache`, and check it is empty (or was made by an
   earlier attempt of this same run) before the first boot. Set both `TT_CACHE_PATH` and
   `TT_CACHE_HOME` to it. The first boot converts the weights (about 34 GB for this model, about
   5 minutes here) and later boots reuse it. Never point at `~/.cache/qwen36-src-build/...`.
3. **Run a copy of the bundle's `run.sh`, with two edits.** Copy `run.sh` to your stage directory. Change
   `HERE=` to the bundle's directory (absolute path), and change `--model "<nearest model>"` to
   `--model <your model-dir>`. Delete `--revision <sha>` and `--tokenizer-revision <sha>` from the
   command, because a local directory has no revision. Leave everything else, including the
   environment lines. Add `--port <free port, for example 8100>` after the other arguments. Set
   `HF_HOME=$HOME/.cache/huggingface`, which holds the speculative-decoding drafter the run script
   names in `DFLASH_WEIGHTS`, and `HF_HUB_OFFLINE=1`.

Things the server does not do: it rejects `logprobs` (it owns its own sampling) and other sampling
parameters. Compare generated ids only.

## The hardware test script

Write `stages/2/serve_and_compare.py`, then give `python3 stages/2/serve_and_compare.py` as the
command in `hw_test.json` with `deadline_s` of 2400. The supervisor runs it on a leased board and
puts that board's chips in `TT_VISIBLE_DEVICES`. The script must:

1. Start your `run.sh` copy with `subprocess.Popen(["bash", run_copy], start_new_session=True, ...)`,
   passing the environment through unchanged (it carries `TT_VISIBLE_DEVICES`) plus `TT_CACHE_PATH`,
   `TT_CACHE_HOME`, `HF_HOME` and `HF_HUB_OFFLINE`. Send its output to `evidence/server.log`.
2. Poll `http://127.0.0.1:<port>/health` until it answers 200. Allow 1500 seconds. Stop with a clear
   message if the process exits first. Record the seconds taken as `server_ready_s`.
3. Free run: `POST /v1/completions` with `{"model": <model-dir>, "prompt": <prompt_ids>,
   "max_tokens": 32, "temperature": 0}`. Keep the text.
4. Teacher forced, for k from 0 to 31: `POST /v1/completions` with
   `prompt = prompt_ids + generated_ids[:k]`, `max_tokens = 1`, `temperature = 0`. Re-tokenize the
   returned text with the new model's `tokenizer.json` (the `tokenizers` package) and compare its
   first id with `generated_ids[k]`. The agreement is the fraction that match. Differences between
   near-synonyms are normal: the chip runs quantized weights and the CPU reference runs bf16. On
   the prototype 25 of 32 matched. Noise would give close to 0.
5. `coherent` is true when the free-run text is non-empty and at least 80 percent of its characters
   are letters, digits, spaces or common punctuation, and the 6-word phrase that repeats most often
   appears at most 3 times.
6. Always stop the server in a `finally`. `Popen` has no `pgid`. With `start_new_session=True` the
   child's pid is its process group id, so use `os.killpg(proc.pid, signal.SIGTERM)`, wait up to 60
   seconds, then `os.killpg(proc.pid, signal.SIGKILL)`. Start the `try` block right after `Popen`, so
   that a later error in your own script cannot leave the server running. Write `evidence/swap-check.json` with every number, the ids, the texts and
   the first 300 characters of each mismatch. Exit 0 when the script finished its measurements, whatever
   the numbers say. Exit non-zero only when it could not measure.

## Prepare, then finish

In the prepare phase you write the files above and `hw_test.json` and `handoff.json`; do not run the
test. After the supervisor has run it, read `stages/2/evidence/hw-test-output.txt` and
`evidence/swap-check.json` and write `result.json` from them. If the test failed, write what failed.
Do not invent a number.

## Do not

- Do not modify the bundle, the nearest model's snapshot, or any tensor cache you did not create.
- Do not write model code. If the server fails to start, read the last 60 lines of `server.log`, change
  the model directory or the run script, and try again in the next attempt; say what you saw.
- Do not spend turns reading tt-metal source. A failed start will point to a specific cause.
