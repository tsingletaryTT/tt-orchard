---
name: weights-swap-check
description: Stage 2 of a tt-orchard run when stage 0 found a weights-only delta. Load the new weights into the existing TT implementation of the nearest model, serve them on one leased board, and compare the chip's output with the stage 1 CPU reference.
status: draft. A local tt-orchard copy, written from one hand prototype on 2026-10-03 (Altworld/Hemmingway-1 on the 2-chip Qwen3.8-27B bundle). Since 2026-10-03 the work is done by two tested template scripts. Since 2026-11-01 the supervisor runs the prepare script itself and calls you only when it fails or has no config. It lives in tt-orchard.
---

# Weights swap check

## When to use this

Stage 0 wrote `delta.json` with `"path": "weights-only"`: the new model has the same text architecture
as a model that already runs on this machine, and only the weights (and maybe the tokenizer) differ.
Then no decoder needs to be written. Do not read or copy tt-metal model code. The existing TT
implementation loads the new weights as they are. This stage shows that it does, and that the chip's
output agrees with the CPU reference from stage 1.

## Where you come in

The supervisor does the prepare step itself whenever it holds every fact. When `stages/2/swap_config.json` exists, it runs
`python3 stages/2/prepare_swap.py` and writes `hw_test.json` and `handoff.json` from the output. You are asked to prepare only in
two cases:

1. `prepare_swap.py` exited nonzero. The task message ends with a "From the supervisor" section that holds the script's exit code
   and output. Exit 2 means a value in `swap_config.json` is wrong or a fact does not fit, and the message names it. Fix that value,
   run the script again, then write `hw_test.json` and `handoff.json` (see "Prepare phase: what is left"). Any other exit code means
   the script failed for a reason the config may not explain: read the output, fix a config value only if the output shows a wrong
   one, and otherwise say what failed and stop.
2. `swap_config.json` does not exist (the supervisor could not draft it). Follow "When there is no swap_config.json" first.

In the finish phase you read the test output and write `result.json`. Write it nowhere else: a `result.json` (or
`test-result.json`, `evidence/swap-check.json`, `evidence/hw-test-output.txt`) found when the prepare phase ends is moved aside
unread, and the example values below are an example only.

## Goal

The finish phase writes `result.json` in your stage directory:

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
or TODO. In a run the operator started with `--weights-only-rehearsal` the gate accepts `coherent` false, so copy
`coherent` as the draft has it, also when it is false.

## How to work

Two tested scripts do all of the work. You copy them and give them one config file. You do not
write a script.

- `swap_config.json` holds facts the run already has. Do not look for the bundle, the snapshots or a port yourself, do not run
  `tt-model list`, and do not copy the two swap templates again. Change a value only when `prepare_swap.py` exits 2 and its message
  names that value.
- The nearest model is a vision-language model: its bundle's `vllm_metadata.json` names
  `Qwen3_5ForConditionalGeneration`, and the new model's `config.json` may name
  `Qwen3_5ForCausalLM`. That difference is expected on this path and needs nothing from you:
  `prepare_swap.py` copies the nearest model's config files (fact 1 below). Do not investigate it.
- A new model with no `generation_config.json` is expected too. `prepare_swap.py` copies the nearest model's and
  prints "generation_config.json: new_snapshot has none; copied from base_snapshot". The bundle's server needs an
  `eos_token_id` from that file and dies at the end of model load without one. Do not investigate or edit it. If the
  script instead warns that the file is in neither snapshot, the warning is advisory: the copied `config.json` may still
  give the server a stop token, and the hardware test shows whether the server starts. Write `hw_test.json` and
  `handoff.json` as usual and do not stop.
- `prepare_swap.py` also writes `stages/2/model-dir-sources.json`: which files of `model-dir/` came from the
  nearest model and which from the new model, and whose generation config the server reads. The stage 8 bundle
  and the package card quote it. It also holds the chat template the directory serves (the jinja file first, then the
  template inside `tokenizer_config.json`), whether that is the nearest model's, and the special tokens. The swap check
  sends plain text to `/v1/completions`, which applies no chat template, so there is nothing to fix there. Do not edit
  the record or write one yourself.
- `prepare_swap.py` warns "no stop token" when the model directory has no top-level `eos_token_id` in
  `generation_config.json` or `config.json`. The Qwen3.8 bundle then stops after loading the weights. The warning is
  advisory, because it predicts from one bundle's reading rule. Do not try to fix it. Write `hw_test.json` and
  `handoff.json` as usual and let the hardware test show whether the server starts. If it does not, the finish step
  records the failure and the operator reads the stop tokens question in the block message.
- `prepare_swap.py` warns "the tokenizer's eos token ... is not among the stop tokens" when the model's tokenizer
  names an eos token that the model directory's `eos_token_id` list does not hold. A reply that ends on that token
  would run to the length limit. The warning is advisory, because the stop tokens may still be right for chat use.
  Do not try to fix it. Write `hw_test.json` and `handoff.json` as usual. The record `model-dir-sources.json` keeps
  the id and the answer for the operator.
- When the new model's `vocab_size` differs from the nearest model's (stage 0's class is `vocab-trim`), `prepare_swap.py` writes the
  model's own `vocab_size` into `model-dir/config.json` and switches the speculative drafter off in `run.sh`, because the drafter
  works on the nearest model's token ids. It prints both lines and records them in `model-dir-sources.json` (`vocab`). Do not edit
  the config or `run.sh` by hand, and do not set `graft_mtp`: the script refuses it for a different vocabulary.
- Write each file as soon as you know its content. Do not wait until the end.
- Do not investigate anything this skill does not list. On an earlier run the model spent 60 turns
  grepping vLLM source for an unrelated timeout and wrote nothing.

## Why the scripts do what they do

Four facts decide whether a swap works. Each was found by failing first.

1. The model directory must look like the nearest model's. The bundle's TT model class is
   registered for the nearest model's architecture, which is a vision-language model. So the
   config files come from the nearest model, and the tokenizer and weights come from the new model.
   One exception: a bundle that ships `orchard-capabilities.json` reads the model's configuration at run time, and stage 0
   checked every differing key against that file. For such a bundle the script writes the new model's own `config.json` with
   only `architectures` set to the nearest model's, and `model-dir-sources.json` says so (`config_json`). This is expected:
   do not replace that file with the nearest model's.
2. The tensor cache must be empty or belong to this model. The bundle reads a cache keyed only by
   layer name, so another model's cache serves that model's weights without any error. The
   script refuses a non-empty cache that lacks its `.orchard-model` marker.
3. The bundle's `run.sh` is run from a copy with these edits: `HERE=` points at the bundle,
   `--model` points at the model directory, the `--revision` flags are removed (a local
   directory has no revision), and `HF_MODEL` is set to the model directory. Two forms of
   `run.sh` are handled. In the older form the command has `--model "<nearest model id>"`. In the
   current form the command has `--model "$HERE/model-dir"`, and `run.sh` builds that directory at
   each start with the bundle's `prepare_model_dir.py`, from the weights named in
   `tt_kernel_manifest.json`. No environment variable changes those weights. For the current form the
   script therefore also replaces the line that runs `prepare_model_dir.py` with a comment, sets
   `MODEL_WEIGHTS_DIR` in the copy, and copies the configuration files from the bundle's `base_config/`
   (the configuration the bundle's model code was built for). The nearest model's snapshot supplies none of them.
   `model-dir-sources.json` says which source was used (`config_from`). A `run.sh` in neither form
   makes the script exit 2 and say so; report that message and do not edit `run.sh`.
4. The server must be told where the new weights are. The TT runtime takes its weights directory
   from `MODEL_WEIGHTS_DIR`, then `HF_MODEL`, then the config path. The bundle's `run.sh` exports
   `HF_MODEL` as the nearest model's id, so `--model` alone gives vLLM the new config and tokenizer
   while the chip loads the nearest model's weights. The first hand prototype did exactly that: it
   agreed with the Hemmingway-1 CPU reference on 25 of 32 tokens. With `MODEL_WEIGHTS_DIR` set to
   the model directory it agreed on 30 of 32, and its free-run text matched the CPU text. The
   template sets `MODEL_WEIGHTS_DIR` and `HF_MODEL` for you and records both in `swap-check.json`.

5. The bundle's speculative drafter needs the model's `mtp.*` tensors. If the new model has none (stage 0's
   `tensors` finding says which tensors it lacks), a drafter bundle's engine dies at start with "model has no
   MTP head". `prepare_swap.py` reads the new model's weight index and, when there is no `mtp.*` tensor,
   writes `export QWEN36_DRAFTER=""` into its `run.sh`, so the bundle serves plain decoding. That gives the
   same greedy tokens. You do nothing; the script prints what it did. Never edit `run.sh` yourself.
   `"graft_mtp": true` in `swap_config.json` adds the nearest model's `mtp.*` tensors as one more shard and keeps the
   drafter on. Do not set it in this stage: the supervisor tries it itself, in a separate test after your stage
   passes its gate, and keeps it only when the measurements favour it (`graft-decision.json`). Do not remove it from
   a config that has it, and do not touch `stages/2/graft-trial/`. The script refuses (exit 2) when the grafted head
   does not fit.

## Prepare phase: what is left

1. Fix the value `prepare_swap.py` named (or write `swap_config.json` and copy the templates, when it did not exist), then run
   `python3 stages/2/prepare_swap.py`. It builds `stages/2/model-dir/` and `stages/2/run.sh` and prints a summary. Repeat until it
   exits 0. Check that `stages/2/model-dir` and `stages/2/run.sh` exist (`ls -l`).
2. Write `hw_test.json`: `{"command": "python3 stages/2/serve_and_compare.py", "deadline_s": 3600}`. When `swap_config.json`
   has a `package` key (a container package, not a `bundle_dir`), the supervisor drafted it for a nearest model that is served by a
   container package: `prepare_swap.py` builds only `model-dir/` and writes no `run.sh`, so skip the `run.sh` check in step 1, and the command
   is `python3 stages/2/serve_and_compare_container.py` with the same deadline. That script exits 6 when the container fetched a model from the
   Hugging Face Hub (it runs with the Hub switched off, so a package that ignores `MODEL_WEIGHTS_DIR` fails or leaves a fetch behind): the test then
   measured another model and the stage fails. Report that text; do not try to make it pass.
3. Read `stages/0/hazard-scan.json` when it exists. It predicts kernel hazards from the model's config (an expert count above the
   256-bit mask, vocabulary columns per die above 65,536, a tile count a kernel rejects, a mixer the bundle lacks). A hit never stops
   the run. Each hit has a `stage2_check`: the targeted test that would show the problem first. This stage runs one hardware test and
   does not add others. If the output is wrong, look at the hit's dimension first. In your summary, list each hit by name and say that
   its check was not run, so the operator bundle does not read the hit as settled.
4. Write `handoff.json` (`goal`, `stage`, `evidence`, `next_action`, `check_on_return`) and reply with a short summary.

Do not run `serve_and_compare.py` yourself. The supervisor runs it on a leased board.

## When there is no swap_config.json

Only then. The supervisor writes it whenever it can, so this case is rare.

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
        "port": 8100,
        "prompt_suite": true,
        "sampling_probe": true}

   Copy the `model` value from `stages/0/delta.json` verbatim, including the `@revision`, into
   `new_model_id`. Stage 7 compares this label with stage 0's model and the revision of the weights
   the test served, so the label must name the same weights.

   `tt_cache` must be a new directory, for example `{{CACHE_ROOT}}/<slug>/tt_cache`, or one this
   run already used for this model. Never point it at
   `{{OPERATOR_HOME}}/.cache/qwen36-src-build/...`. `hf_home` is the operator's Hugging Face cache
   on this machine. Your shell's HOME points somewhere else, so write the path out as shown. It
   holds the drafter the run script names in `DFLASH_WEIGHTS`.
3. Copy the swap templates and the prompt suite into your stage directory:

       cp {{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/prepare_swap.py {{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/serve_and_compare.py {{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/prompt_suite.json stages/2/

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

After that check it sends the fixed prompt suite of `prompt_suite.json` (copied next to the script) and
records the raw replies in `swap-check.json` under `prompt_suite`. The format follows the model's chat
template: with a template it uses the chat endpoint, without one it sends plain continuation prompts.
You do not judge the replies. The stage gate reads them against the control and fails a reply only
when it shows a failure the control does not (a leaked control token, a loop, text that is not
language, a repeated or corrupt first token, a word from another prompt). It also fails a run whose
replies are identical to the control's on every prompt both answered, because that run has probably
not served the new weights. Do not edit
`prompt_suite.json`: the gate compares its sha256 with the shipped file. `"prompt_suite": true` in
`swap_config.json` turns the suite on; the supervisor writes it. To read a control, set
`"suite_control"` to the `swap-check.json` of a run of the same suite on the nearest model.

`"agreement_set": true` (issue 15) widens the 32-token check. It needs `agreement_core.py` and
`agreement_prompts.json` copied next to the script and a stage 1 reference file for the same prompt set
(`stages/1/evidence/reference/agreement-reference.json`: for each prompt its ids and the CPU reference's greedy ids,
plus the prompt set's sha256). The script teacher-forces each of the six prompts (chat, code, tool call, German,
Japanese, and one long prompt) and records the raw ids under `agreement_set`, with the 32-token result as
`baseline`. The gate recomputes everything from the ids: a pooled top-1 bar of 0.85, a per-prompt floor of 0.70, a
damage control (a run with every fourth token wrong must fail the same bars, or the bars are reported unusable),
and, when `"agreement_control"` names the `swap-check.json` of the same set run on the nearest model, a margin of
0.05 over that control. The control must be a complete run (no failed request or missing id) of the same prompts and
reference ids, served from the nearest model (its `model_id` is checked like the prompt-suite control's). Every number is a
provisional choice. When stage 1 has written the reference file, the supervisor sets `agreement_set` in the stage 2 config
and records the file's sha256 in the ledger before the test; the gate compares the file with that hash, and a run whose
config asks for the set needs a completed record. Stage 1 does not write the reference file yet, so nothing turns this on. Do not edit `agreement_core.py`: a test keeps it equal to `orchard/agreement_core.py`.

`python3 -m orchard.long_sweep run --port <port> --model <name> [--api chat|completions] --out sweep.json` (issue 18) measures prefill
and decode speed over input lengths (1K to 200K tokens) and concurrency (1 and 4) on a server that is already
running, and `compare` reads two such files side by side. It needs a board and hours; nothing has run it on one.

`"sampling_probe": true` makes the script send one greedy and two sampled requests after the checks above
and record them under `sampling_probe` in `swap-check.json`. It records what the server did with a request
that has temperature above 0: accepted or refused, how fast, and whether two seeds gave different text.
You do not judge it. A refused or slow sampled request is a fact about the configuration and not a failed
test, so do not change the configuration to make it pass.

## Finish phase

Read `stages/2/evidence/hw-test-output.txt` and `stages/2/evidence/swap-check.json`. If the test
measured, copy `result_draft` into `result.json`.

Write `result.json` only from those two files. If the test failed (`test-result.json` shows a `returncode` other than 0, or `timed_out` true),
write `result.json` with `serves` false, the failure text from `hw-test-output.txt` in a `failure`
field, `coherent`, `n_tokens`, `top1_agreement` and `server_ready_s` null, and
`stages/2/evidence/hw-test-output.txt` in `evidence`. Then stop.
Do not investigate firmware or cache directories, and do not try to make the result pass. Do not
invent a number. The gate fails this result on purpose, and the next attempt runs a fresh test.

## Do not

- Do not modify the bundle, the nearest model's snapshot, or any tensor cache you did not create.
- Do not write model code or your own serving script.
- Do not spend turns reading tt-metal or vLLM source.
