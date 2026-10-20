# pplx-decider-v1.2: the middle path measured (2026-10-19)

Prompt (operator): "do the reference-loader and the attention mask. may as well follow that middle path." Branch `make-it-work-make-it-useful`.
Evidence: `evidence/2026-10-19-pplx-decider-parity/` (the result, the mask effect, the records, the script's own log lines).

## The attention-mask question, answered
Read from `source/src/pplx_decider/model.py` and `protocol.py` (not run) and the model's `decision_config.json` (`attention_mode:
noncausal_full_attention`, `pooling: last`, `temperature: 1.3823`):
- Only the 16 full-attention layers are bidirectional (every position sees every non-padding position). The 48 linear-attention layers stay causal.
- One sequence per request: a chat-templated state, then each question with its options and a single `<|fim_suffix|>` marker token. The readout
  (`Linear(5120, 255)`, the 255 rows of `readout.safetensors`) is applied to the final hidden state at each marker; the first `count` logits of a
  question are divided by the temperature and softmaxed.

## Measured on the real checkpoint (CPU, bf16; 6 requests, 14 questions)
A host reference written from that text (no import of the model's code) gives confident, sensible answers (support team 1.00, customer blocked
1.00, German 1.00, headphones included false 1.00). Without the non-causal mask the same model changes its probabilities by up to 0.4656 and flips
one answer in 14 (moderation: harassment 0.58 becomes feedback 0.57; "are paid fees refundable?" false 1.00 becomes 0.53 against 0.47). The
producer's own verification (`evaluation/inference-verification.json`, 2,324 questions on a GB200) puts its run-to-run noise at a largest
probability difference of 0.059 and its fp8 path at 0.241, so the mask effect is about eight times the noise.

## Measured on the chips (one p300 board, 2 chips, bfp8 weights, drafter off)
| | |
|---|---|
| top-1 agreement with the CPU | 14 of 14 |
| largest probability difference | 0.0480 (bar 0.25) |
| hidden-state correlation at the marker positions | 0.9975 or better |
| hidden-state correlation, all positions | 0.9558 or better (mean 0.9677) |
| wiring (device against CPU, both causal, last row) | cosine 0.9873 (bar 0.90) |
| causal SDPA calls turned non-causal | 16 on every record (the number of full-attention layers) |
| model load | 169 s with the tensor cache built (31 GB), 46 s after |
| device time per record | 0.2 s after the first (41 s, kernel compile) |
A repeat run after the operator's `tt-smi -r` gave identical numbers. The verdict passes every bar. The CPU noise floor (a rerun with other thread
counts) was exactly 0.0, so the probability bar stayed at its plain 0.25.

## Findings from the build
1. The stage 1 reference failure was an edit, not a template bug. The agent replaced the class lookup in its copy of `reference_gate.py` with
   `cls = transformers.AutoModelForCausalLM`; the shipped template resolves `Qwen3_5Model` and loads it. My first explanation (the class is not
   exported at the top level) was wrong: it is exported on both interpreters here. The gates now compare the stage's copy of `delta_triage.py` and
   `reference_gate.py` with the shipped files.
2. The TT loader has the same trap the agent fell into: its bf16 path calls `Qwen3_5ForCausalLM.from_pretrained`, which reports every
   `language_model.*` key missing and runs on random values with only a warning. Serving this checkpoint through the TT stack as it is would give
   random weights. The parity run avoids it with a key mapping from stage 0's alignment.
3. A non-causal masked SDPA works on this stack (the DFlash2 drafter already uses it); the main prefill attention has one hard-coded
   `is_causal=True` call, which the run wraps. Nothing in tt-metal changed.
4. Slips: `AutoProcessor` needs torchvision, which the reference venv lacks (a text-only adapter replaced it); the readout was first read from
   the model directory that holds only the backbone (a 3-minute boot lost); the first test for the readout loader asserted something that could not fail.

## Not done
A CLI wrapper (`tt-orchard head-check`); the check on images, long states or 255-option questions; serving. The bars are copied, not measured for
this head. One board only; no 4-chip run.
