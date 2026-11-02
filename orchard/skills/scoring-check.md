---
name: scoring-check
description: Stage 2 for a reranker or an embedding model (a model whose output is a score or a vector). Write the port module (load, score_pairs or embed, close); the supervisor's own check runs it on one leased board and compares its scores or vectors with the stage 1 Hugging Face reference.
status: draft. Written on 2026-11-03 for issues 376 and 378 (docs/superpowers/specs/2026-11-02-scoring-model-check-design.md). The check has run against fakes on the CPU. No port has run through it on a chip (issue 375).
---

# Scoring check

## When to use this

Stage 0 recorded a model kind of `reranker` or `embedding` in the ledger (`model_kind` in its `stage_end`) and the class is `weights-only`, or
`full-port` with the operator's `--full-port`. The model produces numbers, not tokens. Stage 1 recorded a fixed set of inputs and the
Hugging Face scores or vectors for them. This stage writes the port that produces the same numbers on one board.

There is one hardware test and no retry loop of your own, and no finish phase: the check writes its own record, and that record is the
result. If the port does not run, the stage fails with the check's message and the next attempt runs a fresh prepare and a fresh test.

## What the stage produces

Under `stages/2/port/`: the port's source, and nothing else. The one file the check calls is `stages/2/port/port_model.py`. It defines:

    load(model_dir, mesh_device) -> model
        Load the weights from the snapshot directory onto the open device.
    score_pairs(model, pairs, media=None) -> list[float]       kind reranker
        `pairs` is a list of token-id lists, each a whole prompt (prefix, query, document and suffix already in place). Return one
        number per prompt: logit("yes") - logit("no") at the prompt's last position.
    embed(model, ids, media=None) -> list[float]               kind embedding
        Return the vector for one prompt: the last token's hidden state after the final norm, L2 normalised.
    close(model) -> None
        Release what `load` took. The check closes the device itself.

The `media` argument is passed only when the stage 1 pair file lists files for a row (an image or a video clip): then it is a list of
{"kind", "name", "sha256", "data"} entries for that prompt (for `score_pairs`, one such list for each prompt). Without media the functions
are called with the first two arguments only.

`port_model.py` may import other files in `stages/2/port/`. Every `.py` file there is model code, and at least one must use `ttnn`. No file
there may read the stage 1 reference files: a model file that opens `scoring-reference.json` could copy the reference, and the gate refuses it.
A worked example of a port with the same shape is the public bundle for the model (raahemnabeel/qwen3-reranker-4b for a reranker).

## What the check does, and what it refuses

You do not write the check. The supervisor copies `stages/2/scoring_check.py` and `stages/2/scoring_check_config.json` into the stage
directory (read them if you want to see what calls your code) and runs `python3 -P -s stages/2/scoring_check.py` under the lease. It starts
your port in a second process and takes one thing from it per request: a list of finite numbers. It computes the parity numbers itself:

- reranker: Pearson r, mean and maximum absolute score difference, the groups with the same top document, the share of document pairs that
  keep the reference's order, and NDCG@10 and Hit@1 against the pair file's relevance labels;
- embedding: the cosine to the Hugging Face vector, as a mean and a minimum, over all inputs and for each input kind.

The bounds are in `scoring_check_config.json`. A config can tighten a bound and never loosen one. Do not edit the template or the config:
the supervisor puts both back as shipped after your prepare step. A port that prints a line mentioning `SCORING_CHECK` fails the run. A
port that writes `evidence/scoring-check.json` has it deleted before the check and replaced after it. A port that starts a process that
outlives the check, adds a file after the record was written, or leaves bytecode in `stages/2/port/` fails the run.

## Prepare phase

1. Read `stages/0/delta.json` (`path_reasons`, `differences`, `hazards`, `head`) and `stages/0/hazard-scan.json`, and the stage 1 files
   `stages/1/evidence/reference/scoring-pairs.json` (the inputs: kinds, token ids) and `scoring-reference.json` (the reference numbers).
   The hazard scan leaves out the vocabulary-per-die hazards for this kind and says so in its notes. Do not edit files outside `stages/2/`.
2. Write the port under `stages/2/port/` and test the parts that need no chip (weight names and shapes, a CPU run of your own layer code
   against the reference layer) with ordinary shell commands. Do not import `ttnn` in your own shell: importing it opens the device, and
   only the supervisor's test holds a lease.
3. Write `hw_test.json`: `{"command": "python3 -P -s stages/2/scoring_check.py", "deadline_s": 3600}`. The supervisor runs that command
   whatever the file says; it keeps your `deadline_s` when it is a positive number up to 3600.
4. Write `handoff.json` (`goal`, `stage`, `evidence`, `next_action`, `check_on_return`) and reply with a short summary.

Do not run `scoring_check.py` yourself.

## Do not

- Do not write or edit `scoring-check.json`, `hw-test-output.txt`, `test-result.json`, `scoring_check.py` or `scoring_check_config.json`.
- Do not change a model source after the test has run. The record holds each file's sha256.
- Do not publish, push or write outside your stage directory.
- Do not investigate anything this skill does not list.
