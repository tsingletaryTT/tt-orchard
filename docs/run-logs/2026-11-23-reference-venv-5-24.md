# Reference venv: torch 2.15.0, transformers 5.24.0, tokenizers 0.23.3 (2026-11-23)

`tt-orchard updates` showed the reference venv behind. A change moves reference numerics, so it was tested on a copy first, then applied to the live venv, with a backup of the old one kept.

## Test (CPU only, copy of the venv)
Live: torch 2.14.1+cpu, transformers 5.19.0, tokenizers 0.23.2. Copy: torch 2.15.0+cpu, transformers 5.24.0, tokenizers 0.23.3. huggingface-hub stayed at 1.33.0 (transformers 5.24.0 accepts hub 1.31.0 up to below 3.0) and safetensors at 0.8.0. Same bf16 CPU load as stage 1, offline, 8 threads. Per model: a prefill with per-position argmax, the last positions' full logits, and 16 greedy tokens from whole-sequence forward passes. A second run on the live interpreter was bit-identical, so any difference would be real.

| Model | Prompt | Top-1 agreement | Largest logit difference | Greedy tokens |
| --- | --- | --- | --- | --- |
| Aion-1.0-Instruct-2.3B (remote code) | 68, 49 and 2,400 tokens | 100 percent (2,400 of 2,400 on the long one) | 0.0000 | 16 of 16 |
| SmolLM2-135M | 68 tokens | 68 of 68 | 0.0000 | 16 of 16 |
| Olmo-Hybrid-7B (gated delta net) | 68 tokens | 68 of 68 | 0.0000 | 16 of 16 |
| tiny-qwen4-moe | 120 tokens | 120 of 120 | 0.0000 | 16 of 16 |

The 2,400-token Aion prompt is past the 2,048-token sliding window, so that path ran too. Aion's stage 1 `reference_gate.py` gave the same verdict, generated ids, top-5 logits, tokenizer round trip and card check with the copy and with the live interpreter (only timing lines differed).

transformers 5.24.0 prints deprecation warnings (`past_key_values` is now `cache`; a sliding-window mask helper). They matter only if Aion's remote code is updated.

## Not tested
OLMoE and MiniCPM3 references; prompts longer than 2,400 tokens; sampling; the audio reference venvs (a pin file notes that 5.21.0 flips the audio encoder from `sdpa` to `eager` attention, so check audio separately). The tiny-qwen4-dense fixture disagreed between two runs of the live venv itself (the experts are randomly initialised on each load by both versions), so it says nothing about the versions.

## Applied
The live venv was upgraded on 2026-11-23 after this test; the 5.19.0 venv is kept as a backup. `orchard/setup_machine.py` now pins `transformers==5.24.0` for new machines. Evidence records that say "transformers 5.19.0" stay true for the runs they describe. `orchard/hazards.py` and `orchard/stack_pins.py` still cite 5.19.0 for qwen4_exp behaviour; whether 5.24.0 changes that is not tested.
