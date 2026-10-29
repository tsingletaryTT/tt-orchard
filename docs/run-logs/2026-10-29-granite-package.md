# Granite-4.0-H-1B: package and private upload (2026-10-29)

Status: IN PROGRESS. Update 1 (about 1 hour in), written while the measurements run.

Assignment from the operator: make the traced composed path usable as a model, write an OpenAI-compatible chat server, package it like the
audio package, boot-check it on the chip, and upload it as a private repository `episod/granite-4.0-h-1b-blackhole` for the operator to review.
The generic text-model route of the harness (issues 61, 90, 214) is being built elsewhere and is not part of this work.

## Findings so far

- The earlier port had been checked only to 40 positions. Run on a 1,024-token natural-language sequence against fp32 on the CPU, the bf16
  port has a mean relative L2 of 0.0414 (maximum 0.62) and top-1 agreement of 0.955; the bf16 CPU run has 0.0084 and 0.973. The error grows
  with position (0.013 for positions 0 to 127, 0.084 for 896 to 1,023).
- Cause: the state update rounds the decay (a value close to 1) and the Mamba-2 state to bf16 at every step. Keeping dt, the decay and the
  state in float32 on the chip gives 0.0134 and top-1 0.978 on the same sequence, and the short-prompt numbers stay at the earlier level.
  The cost in a calm host condition is about 8% of decode speed (37.0 to 34.0 tok/s).
- Decode speed against the context limit (bf16, calm host): 37.9 tok/s at 128, 37.4 at 512, 36.5 at 1,024, 33.9 at 4,096.
