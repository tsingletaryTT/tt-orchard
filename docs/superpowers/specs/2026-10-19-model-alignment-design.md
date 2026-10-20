# Stage 0 matches a model to its nearest one by what the tensors are, and stops when it cannot serve the result

Date: 2026-10-19. Prompt (operator): "we have proximal models right but the way we're matching isn't helping us yet"; after a hand analysis of
`perplexity-ai/pplx-decider-v1.2` against `Qwen/Qwen3.8-27B`, "stop and report now" and "build it in a worktree that has all our open PRs
still merged together" (this branch is based on local `main` with PRs 9 and 10 merged).

## Why
The decider was filed as an unknown full port because stage 0's one rewrite rule (`model.language_model.` to `model.`) matched none of its
names. By hand, in about a minute: every one of its 1183 tensors has an analogue (`language_model.X` and `visual.X` against
`model.language_model.X` and `model.visual.X`, same shapes), the base has exactly 16 more (`lm_head` and 15 `mtp.*`), the weights are the
base's trained a little (cosine 0.9999 on sampled tensors), and its 255-row `readout.safetensors` holds copies of the base head's rows for
the answer-code tokens A, B, ... AA, AB. Then stage 1 failed to load any weight (the names did not line up for the loader either), the
agent rewrote the failing verdict to `pass`, and the gate accepted it.

## What it does
1. **Name alignment** (`align_leftovers`): names left over after the old rule pair when they share a long enough run of trailing parts
   (at least three, or two when one name is the end of the other). The layer number is part of the run. A tie, or a base name wanted twice,
   leaves both unmatched. Pairs vote for prefix rewrites; a rewrite with three votes also pairs the short names. `delta.json` gets
   `alignment` (pairs, rules, leftovers); `compare_tensor_maps` counts aligned pairs as shared, so the research estimate benefits too.
2. **Value sampling** (`sample_value_diffs`, standard library only): the first 4096 values of up to 8 paired tensors, read as BF16, F16 or F32,
   classed identical, near (cosine at least 0.999), changed or unrelated. `evidence/value-compare.json`, `delta.json` `values`.
3. **Restricted head** (`detect_restricted_head`, needs numpy; says so when it is missing): a sidecar matrix with the head's width and fewer
   rows, every row of which is a copy (cosine at least 0.999) of a row of the base `lm_head.weight`. `evidence/head-compare.json`, `delta.json`
   `head`, with the token strings when `tokenizer.json` can be read.
4. **Class `restricted-head`** (`orchard/classes.py`): path `full-port`; needs a sidecar and `head.restricted` true. The supervisor pauses
   before stage 1 with `RESTRICTED_HEAD` (an operator's resume is one acknowledgement for the full-port stages too); unattended it ends
   blocked with the code `restricted-head` and an unblock text. No coder time goes on a reference for a model that will not be served.
5. **Code in subdirectories** is hashed (`code_files`, four levels, links followed because a cache snapshot is all links).
6. **Stage 1 gate**: the reference template's `evidence/summary.json` and `load-report.json` decide. A verdict or check the evidence failed,
   or backbone keys missing outside the ignore patterns, fails the stage. Where the template was used, a missing summary fails; an evidence
   file that is unreadable, over 2 MiB or a link out of the stage fails closed.

## Decisions
- The operator's ruling: when matching shows the backbone plus a restricted head, stop and report; do not try to serve it on the base's runtime.
  The reason is the failure mode: a causal mask or a last-row readout would give plausible, wrong probabilities.
- Alignment and sampling are deterministic template code, not agent judgment, as the other stage 0 and 1 steps are.
- The cut-offs (3 and 2 trailing parts, 3 votes, cosine 0.999, 8 tensors, 4096 values) are choices, labelled in the template.

## Not done
- A reference loader that uses the name map, so a model that is weights-only but names its tensors differently loads for stage 1 (the gate
  now fails such a run instead of passing it).
- The attention mask and marker-token readout of the decider: its `model.py` has not been read.
- A middle path for restricted heads (hidden states from the TT backbone checked against the producer's code on the host, as the Clef sidecar
  gate does). It needs the mask question answered first.
- Hashing a base's non-`lm_head` heads, or a restricted head built from `embed_tokens` when the base ties them.
