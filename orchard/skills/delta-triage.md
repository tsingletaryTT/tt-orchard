---
name: delta-triage
description: Stage 0 of a tt-orchard run. Compare a new Hugging Face model with the nearest model that already runs on Tenstorrent hardware, write down every difference with evidence, and name the path the run takes.
status: draft. A local tt-orchard copy. Since 2026-10-04 a tested template script does the measuring and drafts delta.json; you configure it, run it and review what it wrote. It lives in tt-orchard.
---

# Delta triage

## Goal

Write `delta.json` in your stage directory. It lists what differs between the new model and the
nearest supported model, gives evidence for each difference, and names the path:

- `weights-only`: the model code that runs the nearest model can run the new one with new weights.
- `full-port`: new model code is needed. The run stops for the operator before stage 2.

A tested script does all of the measuring and writes a draft `delta.json`. You give it one config
file, run it, read what it wrote, and correct the findings where a measured fact needs explaining.
You do not write a comparison script.

## How to work

- Write `triage_config.json` FIRST, as soon as you know the four facts below.
- Do not investigate anything outside the paths this skill lists: the two snapshot directories,
  the template, and your stage directory.
- Never read the run's ledger (`ledger.jsonl`), the step logs under `stages/*/log/`, or any
  transcripts. An earlier stage 0 agent spent 19 turns reading them and wrote nothing.
- Do not download anything. The run has no Hugging Face token, and the weights are on disk.

## Steps

1. Find four facts with a few `ls` commands:
   - `model_id`: the run's `model` value from the run facts in your task, for example
     `Altworld/Hemmingway-1@<revision>`. If it has no `@<revision>`, the script adds the revision
     from the snapshot directory's name.
   - `model_snapshot`: the new model's snapshot directory,
     `{{HF_HOME}}/hub/models--<org>--<name>/snapshots/<sha>/`. If the run facts list an input
     path for the model, use that path.
   - `nearest_model_id`: the nearest supported model. Use the run facts' `input base` when it
     names one. Otherwise read the `base_model:` line in the front matter of the new model's
     `README.md`. For a Qwen3.8-27B fine-tune it is `Qwen/Qwen3.8-27B`.
   - `nearest_snapshot`: the nearest model's snapshot directory, found the same way, or the
     `input base` path.
2. Write `triage_config.json` in your stage directory (absolute paths):

       {"run_dir": "<the run directory>",
        "model_id": "<the run's model value>",
        "model_snapshot": "<new model snapshot>",
        "nearest_model_id": "<nearest model id>",
        "nearest_snapshot": "<nearest model snapshot>"}

3. Copy the template into your stage directory:

       cp {{ORCHARD_DIR}}/orchard/skills/delta-triage-templates/delta_triage.py stages/0/

4. Run `python3 stages/0/delta_triage.py`. It reads only the safetensors headers and the small
   JSON files, so it never loads the weights. It prints a line for each evidence file it writes.
   Exit 2 means `triage_config.json` is incomplete or names a directory that does not exist; the
   message names the key. Fix that key and run it again.
5. Read `stages/0/delta.json` and the evidence files it cites under `stages/0/evidence/`:
   `config-compare.json`, `tensor-compare.json`, `tokenizer-compare.json`,
   `tokenizer-encode.json`, `files-compare.json`, `genconfig-license.json` and `disk-free.json`.
6. Review the draft. Edit a `finding` only where a measured fact needs explaining. Examples:
   - the new model keeps or drops the nearest model's vision tower: say what that means for a
     text-generation recipe;
   - a tokenizer difference that matters for the model's language, for example Thai or
     Devanagari strings that encode differently (`tokenizer-encode.json` lists them by category);
   - a license change: name both licenses and what the new one restricts.
   Keep each finding's measured numbers. Do not change `path`, `class`, `sidecars`, `code_files`, `model`
   or `nearest_model`. If the
   evidence shows that the path is wrong, add a difference with area `other` that says why and
   cites the evidence file. The operator reads it.
7. The draft has the standing hazards: `tensor_cache`, `drafter`, `disk`, and `license` when the
   license changed. Add any further hazard you can justify from the evidence, with an area from
   the list below, a finding and the evidence path.
8. Write the reviewed `delta.json` with write_file (the whole file). Then reply with a short
   summary and no tool call.

## The file

The script writes this shape. The gate checks it.

    {"model": "<org/name>@<revision>", "nearest_model": "<org/name>@<revision>",
     "path": "weights-only", "class": "weights-only", "path_reasons": [],
     "sidecars": [], "code_files": [],
     "differences": [{"area": "tokenizer", "finding": "...",
                      "evidence": ["stages/0/evidence/tokenizer-compare.json"]}],
     "hazards": [{"area": "tensor_cache", "finding": "...",
                  "evidence": ["stages/0/evidence/tensor-compare.json"]}]}

A difference's `area` is one of: config, architecture, tensors, tensor_names, tokenizer,
chat_template, files, generation_config, license, other. A hazard's `area` is one of:
tensor_cache, drafter, disk, license, other. Evidence paths are relative to the run directory and
must be files inside it.

## What the script decides

The class is one of `weights-only`, `weights+sidecar`, `full-port` and `unknown`. A sidecar is a weights
file the new model ships outside its backbone shards (the shards its index lists) that the nearest model
does not have, for example a task head. `sidecars` lists each one with its size, sha256, tensor count and
tensor names, and `code_files` lists each `.py` file in the repo with its sha256. The script reads headers
and hashes files. It never imports or runs a code file, and neither do you. When a model has a readable
sidecar whose tensor names do not overlap the backbone, and the backbone is weights-only, the class is
`weights+sidecar` and the path stays `weights-only`. An unreadable or overlapping sidecar makes the class
`unknown` and the path `full-port`.

The path is `weights-only` when the text config values that matter (layers, hidden size, heads,
KV heads, head_dim, layer types, vocabulary size) are equal, every tensor both models share has
the same shape and dtype, and neither model has a text tensor the other lacks. Vision-tower,
projector and `mtp.*` tensors may differ. Otherwise the path is `full-port`, and `path_reasons`
lists each reason. A tokenizer difference does not change the path, because the runtime loads the
new model's tokenizer.

## Resume notes

Write progress notes to `RESUME.md` in your stage directory as you go. If the run restarts, the
supervisor keeps your stage directory when that file exists and gives you the notes.
