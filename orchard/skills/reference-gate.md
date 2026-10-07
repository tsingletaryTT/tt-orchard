---
name: reference-gate
description: Stage 1 of a tt-orchard run. Build a CPU reference of the new model and show that it reproduces the model's published behavior before any device result is compared with it.
status: draft. A local tt-orchard copy. Since 2026-10-04 a tested template script loads the model, runs every check and drafts reference.json; you configure it, run it and review what it wrote. It lives in tt-orchard.
---

# Reference gate

## Why this stage exists

A failed bring-up on this machine built a CPU reference that decoded the wrong way, and every
later measurement was made against it. This stage checks the reference itself first.

## Goal

Write `reference.json` in your stage directory. A tested script does all of the work and writes a
draft. You give it one config file, run it, read what it wrote, and add notes where a measured
fact needs explaining. You do not write a script.

## How to work

- Write `reference_config.json` FIRST, as soon as you know the facts below.
- Do not investigate anything outside the paths this skill lists: the new model's snapshot
  directory, the template, `stages/0/delta.json` and your stage directory.
- Never read the run's ledger (`ledger.jsonl`), the step logs under `stages/*/log/`, or any
  transcripts. An earlier agent spent 19 turns reading them and wrote nothing.
- Do not download anything. The weights are on disk.

## Steps

1. Find three facts:
   - `model_snapshot`: the new model's snapshot directory. Stage 0 used it: read
     `model_snapshot` in `stages/0/triage_config.json`. Otherwise it is
     `{{HF_HOME}}/hub/models--<org>--<name>/snapshots/<sha>/`.
   - `python`: an interpreter that can import torch and transformers. If the run facts list an
     input named `reference_python`, use that path exactly and go on; it was checked before the run
     started. Otherwise check `python3 -c "import torch, transformers; print(torch.__version__,
     transformers.__version__)"`. If that fails, look for another interpreter on the machine (for
     example a venv's `bin/python`) and check it the same way. Never install or upgrade a package
     (pip, uv and conda installs outside the run directory are refused, and an upgrade of a shared
     environment broke tools on this machine once). If no interpreter works, write
     `reference.json` with verdict `fail` and the import error as the reason, then stop.
   - `run_dir`: the run directory from the run facts.
2. Write `reference_config.json` in your stage directory (absolute paths):

       {"run_dir": "<the run directory>",
        "model_snapshot": "<new model snapshot>",
        "python": "<the interpreter>"}

3. Copy the template into your stage directory:

       cp {{ORCHARD_DIR}}/orchard/skills/reference-gate-templates/reference_gate.py stages/1/

4. Run it with the interpreter you chose: `<python> stages/1/reference_gate.py`. A 27B model on
   this CPU takes several minutes to load and decode; the script prints a line for each step and
   writes each evidence file as soon as it has it. Exit 4 means torch or transformers is missing;
   the message names the package. Choose another interpreter and run it again. Exit 3 means the
   weights did not load; `evidence/load-report.json` holds the error, and reference.json says
   `fail`. Exit 2 means `reference_config.json` is incomplete.
5. Read `stages/1/reference.json` and the evidence it cites: `evidence/load-report.json`,
   `evidence/tokenizer-roundtrip.json`, `evidence/decode.txt`, `evidence/card-check.txt`,
   `evidence/summary.json`, and the files under `evidence/reference/`.
6. Review the draft. Keep every measured number. Edit a `note` only where a measured fact needs
   explaining, for example:
   - `loads`: keys the class ignores (such as `mtp.*`) and why that is expected;
   - `matches the card`: read the model card (`README.md` in the snapshot) and say what it
     describes. The script's check is a form check only: it records whether the text is
     non-empty and whether it begins with reasoning ("Let me", "We need", "<think"). Keep that
     sentence. If the card publishes an exact sample, compare it with the generated text and say
     what you found.
   Do not change a check's `pass` from false to true. If a check failed, leave it failed and say
   in its note what the evidence shows.
7. Write the reviewed `reference.json` with write_file (the whole file). Then reply with a short
   summary and no tool call.

## The file

The script writes this shape. The gate checks it.

    {"verdict": "pass",
     "environment": {"python": "...", "torch": "...", "transformers": "..."},
     "checks": [{"name": "loads", "pass": true, "note": "...",
                 "evidence": ["stages/1/evidence/load-report.json"]},
                {"name": "tokenizer round trip", ...},
                {"name": "decodes forward", ...},
                {"name": "matches the card", ...}]}

`verdict` is `pass` only when every check passes. Evidence paths are relative to the run
directory and must be files inside it.

## What the script checks

1. `loads`: the weights load on CPU in bf16 with low_cpu_mem_usage. Missing, unexpected and
   mismatched keys fail the check. Keys that match the class's
   `_keys_to_ignore_on_load_unexpected` patterns are listed as ignored.
2. `tokenizer round trip`: 14 fixed strings (Latin, accented, emoji, CJK, Devanagari, Thai,
   Korean, punctuation, whitespace) encode and decode back to the same text.
3. `decodes forward`: a greedy decode of 32 tokens from "Write the text I send my landlord about
   the broken boiler." rendered through the chat template. The argmax of the last prompt
   position must equal the first generated token, and each new token must be appended at the
   end. When the tokenizer has Thai tokens, a Thai prompt is decoded the same way.
4. `matches the card`: a form check of the generated text (see step 6).

Stages 2 and 4 compare the chip with `evidence/reference/prompt-ids.json` and
`evidence/reference/generated-ids.json`, which must hold 32 generated ids.

## Resume notes

Write progress notes to `RESUME.md` in your stage directory as you go. If the run restarts, the
supervisor keeps your stage directory when that file exists and gives you the notes.
