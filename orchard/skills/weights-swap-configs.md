---
name: weights-swap-configs
description: Stage 4 of a tt-orchard run when stage 0 found a weights-only delta. Serve the new weights on each chip configuration the packages will ship for (2 and 4 chips required, 1 chip optional), using the packages that already serve the nearest model, and compare each with the stage 1 CPU reference.
status: draft. A local tt-orchard copy written on 2026-10-03 from the stage 2 weights-swap-check skill and its templates. No harness run has used it yet. It lives in tt-orchard.
---

# Weights swap on each chip configuration

## When to use this

Stage 0 found a weights-only delta and stage 2 showed the new weights serving on one
configuration. This stage shows them working on every chip configuration the packages will ship
for. No model code is written and nothing is parallelised: for each configuration an existing
package or bundle of the nearest model serves the new weights, and the same two tested template
scripts as stage 2 measure it. You copy the templates and write one small config file per
configuration. You do not write a script.

## Goal

Write `result.json` in your stage directory:

    {"configs": [
      {"chips": 2, "pass": true, "kind": "bundle", "package": "episod/qwen3.8-27b-dflash2-p300",
       "serves": true, "server_ready_s": 120.4, "coherent": true,
       "free_run_text": "first 200 characters", "top1_agreement": 0.94, "n_tokens": 32,
       "cache_dir": "<this configuration's tensor cache>",
       "evidence": ["stages/4/configs/2/evidence/swap-check.json",
                    "stages/4/configs/2/evidence/server.log", "stages/4/tests/2/output.txt"]},
      {"chips": 4, "pass": true, "kind": "container", "package": "changh95/qwen3.8-27b-p300x2", ...},
      {"chips": 1, "pass": false, "reason": "what failed, from the test output",
       "evidence": ["stages/4/tests/1/output.txt"]}]}

The context lists the required chip counts for this run (normally 2 and 4). Each required count
needs an entry with `pass` true. Any other count is optional. Try it, and if it does not work,
record it with `pass` false and the reason. If no package exists for an optional count, record
`"pass": false, "reason": "not run: no installed package serves <N> chips"` and give the
evidence file that shows it (for example the output of `tt-model list` saved under
`stages/4/evidence/`).

The gate checks, for every entry with `pass` true: the supervisor's record
`tests/<chips>/test-result.json` shows that the test ran on that many chips and exited 0; the entry
has `serves` true, `coherent` true, `n_tokens` of at least 16, `top1_agreement` of at least 0.85
and a positive `server_ready_s`; and its evidence includes
`stages/4/configs/<chips>/evidence/swap-check.json`, whose `result_draft` has the same
`top1_agreement`. A configuration whose test never ran cannot pass.

## How to work

- The supervisor has usually drafted every configuration (`hw_tests.json` and
  `configs/<N>/swap_config.json`). Start from those; see the steps below.
- When `hw_tests.json` exists, its counts are the whole list. Do not add a count to it and do not make
  a `configs/<N>` for any other count: the supervisor drafted every count this machine can run, and it
  puts its list back after your step.
- Do the configurations one at a time, from the largest chip count to the smallest.
- Write each file as soon as you know its content.
- Do not investigate anything this skill does not list.

## Prepare phase: the steps

1. Read `stages/4/hw_tests.json` and each `stages/4/configs/<N>/swap_config.json`. The supervisor wrote
   them before you started, from facts the run already holds, and copied the three templates into each
   `configs/<N>`. Use them as they are: do not look for packages, snapshots, caches or ports, and do not
   run `tt-model list`. Only when `hw_tests.json` does not exist, a count in "required chip
   configurations" (or 1, 2 and 4 when none are required) with no `configs/<N>/swap_config.json` was not
   drafted: only for that count, do the steps under "When a configuration has no drafted config" below.
2. For each configuration N in `hw_tests.json`, run `python3 stages/4/configs/N/prepare_swap.py`. It
   builds `stages/4/configs/N/model-dir/` and, for a bundle, `stages/4/configs/N/run.sh`. Exit 2 means an
   edit did not apply or a fact does not fit; its message names it. Fix that value in that
   configuration's `swap_config.json` and run it again.
3. Check that each `stages/4/configs/N/model-dir` exists (`ls -l`), and for a bundle `run.sh` too.
4. If `stages/4/hw_tests.json` does not exist (a count was not drafted), write it as the steps below
   describe, one entry per configuration you prepared.
5. Write `handoff.json` and reply with a short summary.

Do not run either serve script yourself. The supervisor runs each one under a lease, in order of
chip count, and records it.

## When a configuration has no drafted config

Only then. The supervisor drafts every configuration it can: the bundle of the nearest model with that
many chips, and for 4 chips the container package bringup.toml names in `four_chip_package` (or the only
one installed).

### The packages on this machine

| chips | kind | package | context | template script |
|---|---|---|---|---|
| 1 | bundle | `episod/qwen3.8-27b-dflash2-p150` at `{{TT_MODEL_ROOT}}/episod/qwen3.8-27b-dflash2-p150` | 16K | `serve_and_compare.py` |
| 2 | bundle | `episod/qwen3.8-27b-dflash2-p300` at `{{TT_MODEL_ROOT}}/episod/qwen3.8-27b-dflash2-p300` | 262K | `serve_and_compare.py` |
| 4 | container | `changh95/qwen3.8-27b-p300x2`, profile `batch32` | 262K | `serve_and_compare_container.py` |

To check a row, run `tt-model list` and read `{{OPERATOR_HOME}}/.cache/tt-model/installed.json`
(a container package has `"container": true` and an `"image"`). A bundle has `run.sh` and
`venv/` in its directory. If a required package is missing, write that in `result.json` and stop.

### The steps for one configuration with N chips

1. Make the directory and copy the three templates into it:

       mkdir -p stages/4/configs/N
       cp {{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/prepare_swap.py {{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/serve_and_compare.py {{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/serve_and_compare_container.py stages/4/configs/N/

2. Write `stages/4/configs/N/swap_config.json` (absolute paths). The snapshots, the model ids and
   `run_dir` are the same as in stage 2's `swap_config.json` (`stages/2/swap_config.json`); read it.

   For a bundle (1 or 2 chips):

       {"run_dir": "<the run directory>",
        "bundle_dir": "<bundle directory from the table>",
        "nearest_model_id": "Qwen/Qwen3.8-27B",
        "base_snapshot": "<nearest model snapshot>",
        "new_snapshot": "<new model snapshot>",
        "new_model_id": "<the model value from stages/0/delta.json, for example Altworld/Hemmingway-1@<revision>>",
        "tt_cache": "{{CACHE_ROOT}}/<slug>/<N>chip-<package name>/tt_cache",
        "hf_home": "{{HF_HOME}}",
        "port": <8100 + N>}

   For the container package (4 chips): no `bundle_dir`; add the package, its profile, the chip
   count and the operator's home:

       {"run_dir": "<the run directory>",
        "package": "changh95/qwen3.8-27b-p300x2",
        "profile": "batch32",
        "chips": 4,
        "nearest_model_id": "Qwen/Qwen3.8-27B",
        "base_snapshot": "<nearest model snapshot>",
        "new_snapshot": "<new model snapshot>",
        "new_model_id": "<the model value from stages/0/delta.json, for example Altworld/Hemmingway-1@<revision>>",
        "tt_cache": "{{CACHE_ROOT}}/<slug>/4chip-qwen3.8-27b-p300x2/tt_cache",
        "hf_home": "{{HF_HOME}}",
        "operator_home": "{{OPERATOR_HOME}}",
        "port": 8104}

   Copy the `model` value from `stages/0/delta.json` verbatim, including the `@revision`, into
   `new_model_id`. Stage 7 compares this label with stage 0's model and the revision of the weights
   the test served, so the label must name the same weights.

   Every configuration gets its own new `tt_cache`. Never reuse stage 2's cache, another
   configuration's cache, or anything under `{{OPERATOR_HOME}}/.cache/tt-model/` or
   `{{OPERATOR_HOME}}/.cache/qwen36-src-build/`.
   The tensor cache is keyed only by layer name and mesh, so a cache made by another model or
   another mesh is read without an error and serves the wrong tensors. The supervisor refuses a
   list in which two configurations share a cache.
3. Run `python3 stages/4/configs/N/prepare_swap.py`. It builds `stages/4/configs/N/model-dir/` and,
   for a bundle, `stages/4/configs/N/run.sh`. Exit 2 means an edit did not apply; its message names
   the fact in `swap_config.json` to fix.
4. Check that `stages/4/configs/N/model-dir` exists (`ls -l`), and for a bundle `run.sh` too.

When every configuration is prepared:

5. Write `hw_tests.json` in your stage directory, one entry per configuration you prepared:

       {"tests": [{"chips": 2, "script": "serve_and_compare.py", "deadline_s": 3600},
                  {"chips": 4, "script": "serve_and_compare_container.py", "deadline_s": 3600},
                  {"chips": 1, "script": "serve_and_compare.py", "deadline_s": 3600}]}

6. Write `handoff.json` and reply with a short summary.

Do not run either serve script yourself. The supervisor runs each one under a lease, in order of
chip count, and records it.

## What the two serve scripts do

`serve_and_compare.py` (bundles) is the stage 2 script: it starts the bundle's edited `run.sh` with
`MODEL_WEIGHTS_DIR` and `HF_MODEL` set to the model directory and a fresh tensor cache, then
measures.

`serve_and_compare_container.py` (the container package) asks `tt-model serve <package> --print`
for the docker command, edits it and runs it: the tensor cache becomes your `tt_cache`, the
operator's Hugging Face cache is replaced by an empty directory so the base weights cannot be
loaded, the model directory and the new model's weight files are mounted read-only at their own
paths, `MODEL_WEIGHTS_DIR` and `HF_MODEL` name the model directory, and only the leased chips are
mapped. It saves the edited command as `evidence/docker-argv.json`, and it always stops and
removes its container.

Both write `stages/4/configs/N/evidence/swap-check.json`, whose `result_draft` holds the fields of
one `configs` entry (without `chips`, `pass`, `kind` and `package`). Exit codes: 0 measured, 2 the
config or the docker command could not be used, 3 the tensor cache was refused, 4 the server did
not become healthy, 5 the server answered with an HTTP error. Both wait up to 3300 s for the
server. A first boot converts the weights into the new cache (about 5 minutes for 2 chips here; not
measured for 4 chips), and a bundle whose kernel compile cache is cold compiles every kernel first
(more than 26 minutes on 2026-10-03). The 2-chip bundle reuses the kernels stage 2 compiled; the
1-chip bundle compiles its own. The container keeps the package's own kernel cache.

## Finish phase

Read the stage's `test-result.json` (every configuration's record), and for each configuration
`stages/4/tests/N/output.txt` and `stages/4/configs/N/evidence/swap-check.json`.

- If the test exited 0, copy `result_draft` into the entry, add `chips`, `kind` and `package`, and
  add `stages/4/tests/N/output.txt` to its evidence. Set `pass` true only if the draft meets the
  bar above; otherwise set `pass` false and give the reason (for example "top1_agreement 0.75 is
  below 0.85").
- If the test did not exit 0, set `pass` false and copy the failure text from `output.txt` into
  `reason`. Do not investigate firmware or cache directories.

Do not invent a number.

## Do not

- Do not modify a bundle, a package, the nearest model's snapshot, or any tensor cache you did not
  create.
- Do not start, stop or remove a container, and do not run `tt-model serve` or `docker` yourself.
- Do not write model code or your own serving script.
