---
name: scoring-reference
description: Stage 1 for a reranker or an embedding model. Run the recorder that writes a fixed input set and the Hugging Face scores or vectors for it on the CPU.
status: draft. Written on 2026-11-03 for issues 376 and 378 (docs/superpowers/specs/2026-11-02-scoring-model-check-design.md). The recorder has run on a tiny random Qwen3 on the CPU; no real snapshot has been recorded through the supervisor (issue 375).
---

# Scoring reference

## When to use this

Stage 0 recorded a model kind of `reranker` or `embedding` (`model_kind` in its `stage_end` ledger entry). The model gives a score or a
vector, so the generated-token reference of the usual stage 1 does not apply. This stage records what the Hugging Face model gives for a
fixed input set. Stage 2 compares the port with it.

## What to do

1. Pick an interpreter that can import torch, transformers and tokenizers. If the run facts list an input named `reference_python`, use
   that path exactly. Otherwise check `python3 -c "import torch, transformers, tokenizers"` and use the venv that passes. Never install or
   upgrade a package.
2. Run the recorder from the run directory with the kind the ledger holds (`rerank` for a reranker, `embed` for an embedding model):

       PYTHONPATH={{ORCHARD_DIR}} <python> -s -m orchard.scoring_reference <rerank|embed> --model-dir <the run's model input> --out stages/1/evidence/reference

   It writes `scoring-pairs.json` (the inputs and their token ids) and `scoring-reference.json` (the Hugging Face numbers and the sha256 of
   the pair file). A 4B reranker takes minutes on the CPU. Write the time you measured in `stages/1/RESUME.md` with the command.
3. Read the first lines of both files. For a reranker, compare two or three scores with the model card's probability example: the card's
   number is a sigmoid of the recorded score. If a number is far off, say so in `RESUME.md`. Do not edit either file.

## What the gate checks

Both files exist as regular files, both are for the run's kind, the reference names the pair file as it is on disk, and the reference holds
a finite score (reranker) or vector (embedding) for every input. The supervisor records both files' sha256 in the ledger. Stage 2's gate
refuses a record made against files that differ from those.

## Do not

- Do not write the reference numbers yourself or edit the files after the recorder ran.
- Do not publish, push or write outside your stage directory and `stages/1/evidence/reference/`.
