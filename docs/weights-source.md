# Which installed packages can serve swapped weights

A weights-only bring-up puts the new model's weights in a model directory and serves an installed package against it. The test
measures the new model only if the package reads that directory. A container package whose generator downloads a constant
repository ignores the directory and serves the base model. The first run that hit this (a Qwen3-Coder-30B-A3B fine-tune, issue 60)
reported 32 of 32 tokens matching and had measured the base. Issue 73 tracks the fix.

## What the harness checks

- **While a test runs** (PR 72): the test container has the Hub off (`HF_HUB_OFFLINE=1`) and starts from an empty `/hf`. A test whose
  `/hf` or log shows a fetch fails with exit 6 and a `hub_fetch` entry in `swap-check.json`, and the stage 2 and stage 4 gates refuse it.
  A package that hard-codes its repository is found this way, but only after a coder boot and a container start.
- **Before a run starts** (this page): the `base` preflight row asks `orchard/weights_source.py` about every installed package that
  serves the base. A package it refuses does not count as serving the base. When no package is left, the row blocks with code
  `nearest-model-missing`, one sentence per refused package, and the way out: a patched generator, that is a new package or a bundle
  whose generator reads `MODEL_WEIGHTS_DIR`.

## The two kinds of package

**A v6 bundle** is judged by its `run.sh`. The check runs stage 2's own edit (`edit_run_script` in `prepare_swap.py`) on the text in
memory, with placeholder paths, and writes nothing. A `run.sh` the edit cannot rewrite has no single `--model` argument, has a second
`--model` option, or (in the current form) has no line that runs `prepare_model_dir.py`. Such a bundle cannot take swapped weights, and
the reason is the swap script's own sentence. A bundle whose directory or `run.sh` cannot be read is not refused for that.

The dry run asks the script to fail quietly for this caller (the template's `QUIET` context variable) and reads the failure from the
exception it raises, so the process's stderr is not touched. The web UI runs preflights on request threads, and a redirect of the shared
stderr lets two overlapping dry runs swallow each other's messages.

**A container package** (schema 5.1) is judged by a record, `orchard/data/weights_source.json`. Nothing in a container's manifest says
what its generator does, so each row holds a status and the evidence for it:

| Status | Meaning | Effect |
|---|---|---|
| `hard-coded` | A run saw it fetch its own repository | The base row refuses it |
| `honours` | A probe saw it read the model directory and fetch nothing | The row passes and says so |
| `unprobed` | No probe is recorded | The row passes and the warning says it was not probed |

A package that is not in the file is `unprobed`. The file is read through `safeio` and checked row by row; a malformed file stops the
`base` row with the reason, and `tests/test_weights_source.py` checks the shipped file.

The record is about container packages only. A v6 bundle that carries a recorded name (a patched replacement that took the container's
name) is judged by its own `run.sh`.

## Where the answer is used

- The `base` row drops a refused package from the packages that serve the base.
- No picker offers a refused package, and none offers a recorded hard-coded container that is not installed (its download would be
  refused by the next preflight): the search results, the ranked bundle choices, the installed candidates, the inferred-base
  confirmation and the closest-bundles text all leave it out. An installed v6 bundle with that name stays, to be judged by its script.
- Stages 2 and 4 apply the same check when they choose a package (`swap_draft.draft` and `draft_stage4`), so a refused bundle is never
  chosen over a usable container that the preflight accepted. A record file that cannot be read gives the stage a named problem.

## Probing a package by hand

A probe takes about five minutes on one board and loads no weights.

1. Take a lease on a free board with `gozer run --chips 2 ...` (or `gozer acquire`). Never start a container on chips you do not hold.
2. Make a stub model directory: copies (not links) of the base model's `config.json`, tokenizer files and `model.safetensors.index.json`,
   and no weight shard.
3. Build the container command the way the swap test does: `tt-model serve <package> --profile <profile> --device-id <ids> --print`,
   then the edits of `edit_docker_argv` in `serve_and_compare_container.py` (the stub as `model-dir`, an empty `/hf`, the Hub off,
   `MODEL_WEIGHTS_DIR` set). Run it detached.
4. Read the container log for the first telling line. A missing shard named inside the stub directory means the package reads
   `MODEL_WEIGHTS_DIR`. An offline-mode error that names the base repository, or `Fetching N files`, means it hard-codes its repository.
5. Stop and remove the container, release the lease, and change the row in `weights_source.json` with the evidence: what was run, the
   log line, and the date. A row marked `honours` or `hard-coded` needs a sentence a reviewer can check.

A probe shows what the package does with this one directory. It does not show that a full swap serves the new model correctly; the
swap check's token comparison does that.

## What the check cannot see

- A container package that is not in the record and fetches only some files (a tokenizer, for example) passes the preflight. The
  runtime check of PR 72 still catches a fetch.
- A `run.sh` that the edit can rewrite but whose runtime reads another path than `MODEL_WEIGHTS_DIR` and `HF_MODEL` passes. The
  runtime check catches it.
- The `unprobed` rows are a list of work and say nothing about the package. Probe the packages that come from the same publisher as a hard-coding
  package first. The first probe, on 2026-11-03, covered `raahemnabeel/qwen3-coder-next` (one image, one profile) and found that it reads the
  model directory. Its two siblings use other images and are still unprobed.
