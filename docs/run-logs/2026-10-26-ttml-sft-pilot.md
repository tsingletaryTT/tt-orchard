# tt-train SFT pilot on Llama-3.2-1B (2026-10-26): not run, no chip was available

Goal: find out whether tt-train's Python `SFTTrainer` can fine-tune Llama-3.2-1B on this box's
Blackhole chips, as a first step toward sequence-level distillation. This entry is an experiment log.
No training ran. No loss, step time or memory figure exists.

## What was done (no chip used)

- Read `tt-train/docs/SFT_TRAINER.md`, the `lora_llama` SFT example and the GRPO README. The 1B
  architecture for ttml is in `configs/model_configs/llama3_2_1B.yaml` (16 layers, 32 heads, 8 kv heads,
  hidden 2048, MLP 8192, tied embeddings, rope scaling 32/4/1/8192, theta 500000). The Hugging Face
  vocabulary is 128,256, so the yaml's 32,000 does not apply.
- Checked imports with `importlib.util.find_spec` only: `ttml`, `ttnn`, `transformers`, `pyarrow`, `pandas`
  are present in the shared venv.
- Found a problem before any device use: `import datasets` fails in the shared venv
  (`ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`; huggingface_hub is 1.33.0 there).
  `ttml/common/data.py` imports `datasets`, and the lora example imports that module, so the stock example
  would fail here. The pilot script does not import it and reads the parquet file with pandas.
  Whether `import ttml.trainers` itself pulls `datasets` was not tested (a real import opens the device).
- numpy: the shared venv has 2.5.3 and ttnn 0.80.0 wants numpy below 2. A numpy 1.26.4 copy was
  installed with `pip install --target <scratch>` for use on `PYTHONPATH`; the shared venv was not changed.
- Downloaded `unsloth/Llama-3.2-1B-Instruct` and the `trl-lib/Capybara` parquet files with `hf download`.
- Wrote `docs/run-logs/evidence/ttml-sft/ttml_sft_pilot.py`. It builds the 1B Llama in ttml, loads the
  safetensors, tokenizes chat data (last assistant turn is the completion, the rest is masked), and runs
  `SFTTrainer` in LoRA or full mode with a per-step loss and time recorder and the device memory tracker.
  It has never been run, so it may have errors.

## Blocker

All four chips were held for the whole watch (about 12 minutes):
chips 0 and 1 by `claude:graft-accept`, chips 2 and 3 by `claude:kernel-probe` (the coder server).
`gozer acquire --chips 1 --who claude:ttml-sft --reason "tt-train SFT pilot"` was queued at position 2,
behind `claude:suite-control` (wants 2 chips, queued since 20:15Z). `gozer wait` did not return.
The ticket was cancelled (my own) so it would not be granted to nobody. No other lease or ticket was touched.

## Needed to continue

- One free chip: either `graft-accept` finishes, or the operator decides about the `suite-control` ticket.
- Then: `gozer acquire --chips 1 ...`, run the script with `PYTHONPATH=<numpy 1.26 dir>`, LoRA first
  (about 20 steps, batch 4, max length 512), then full fine-tuning.
- Distillation beyond sequence-level SFT needs a KL loss: `ttml.ops.loss` has only mse, cross_entropy and nll.
