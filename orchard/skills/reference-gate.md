---
name: reference-gate
description: Stage 1 of a tt-orchard run. Build a CPU reference of the new model and show that it reproduces the model's published behavior before any device result is compared with it.
status: draft. A local tt-orchard copy. Its home is the tt-model-bringup plugin in tenstorrent/skills (spec section 11). Move it there after a real run has used it.
---

# Reference gate

## Why this stage exists

A failed bring-up on this machine built a CPU reference that decoded the wrong way, and every
later measurement was made against it. This stage checks the reference itself first.

## Goal

Write `reference.json` in your stage directory:

    {"verdict": "pass",
     "environment": {"python": "...", "torch": "...", "transformers": "..."},
     "checks": [{"name": "decodes forward", "pass": true, "note": "...",
                 "evidence": ["stages/1/evidence/decode.txt"]}]}

`verdict` is `pass` only when every check passes.

## Checks

Run each check, save its output under `evidence/`, and write one entry for it.

1. `loads`: the weights load on CPU with no missing or unexpected keys. Save the load report.
2. `tokenizer round trip`: a set of test strings encodes and decodes back to the same text.
3. `decodes forward`: greedy decoding of a short prompt. The argmax of the last position's logits
   equals the first generated token, and each new token is appended at the end. Save the prompt,
   the token ids and the text.
4. `matches the card`: reproduce one behavior the model card states. If the card publishes an
   exact sample, compare with it. If it publishes none, render one prompt through the chat
   template, generate, and check that the output has the form the card describes. Say in `note`
   that the card gives no exact sample.

## Cost

Large models are slow on this CPU. A dense 27B model in bf16 decodes at about 0.7 tokens per
second here (measured 2026-10-02). Keep prompts short and generate at most 32 tokens per check.
Load the model once and run every check in the same Python process.

## Leave for later stages

Save the reference outputs under `evidence/reference/`: the prompt ids, the generated ids and the
top-5 logits of each position. Stages 2 and 3 compare device results with them.

## Resume notes

Write progress notes to `RESUME.md` in your stage directory as you go. If the run restarts, the
supervisor keeps your stage directory when that file exists and gives you the notes.
