# tt-train SFT pilot on Llama-3.2-1B (2026-10-26)

Goal: find out whether tt-train's Python `SFTTrainer` can fine-tune Llama-3.2-1B on this box's
Blackhole chips, as a first step toward sequence-level distillation. This is an experiment log.
It is a smoke test of the training path. It does not show that any fine-tune improved the model.

## Result

`SFTTrainer` runs on one Blackhole chip, in LoRA mode and in full fine-tuning mode, on
`unsloth/Llama-3.2-1B-Instruct`. Both ran 20 steps with no error once the data was shaped correctly.
Step time is about 0.45 s for a batch of 4 sequences of 512 tokens (about 4,500 tokens/s) in both modes.
Per-step numbers are in `evidence/ttml-sft/*.json`. The script is `evidence/ttml-sft/ttml_sft_pilot.py`.

Setup: 61 Capybara conversations (`trl-lib/Capybara`, the first 200 rows, kept only if prompt plus answer is
at most 512 tokens; 139 were dropped), batch 4, every sequence padded to 512, loss on the last assistant
turn only, AdamW, 20 steps (80 sequences, so about 1.3 passes over the 61). Held-out: 8 conversations from
the Capybara test split (12 of 20 dropped for length). Lease: `gozer acquire --chips 1` granted a whole
board (chips 0 and 1); the script opened a 1x1 mesh.

| run | LR | eval loss before | eval loss after | train loss step 1 / 10 / 20 | step time |
|---|---|---|---|---|---|
| LoRA (rank 16, alpha 32, q/kv/out) | 1e-4 | 1.371 | 1.262 | 1.31 / 1.02 / 1.16 | 0.44 s |
| full | 1e-4 | 1.371 | 1.973 | 1.31 / 1.45 / 1.48 | 0.45 s |
| full | 1e-5 | 1.371 | 1.250 | 1.31 / 1.00 / 1.04 | 0.45 s |

- Train loss is noisy (batches of 4 differ a lot). The eval set has 8 conversations, so a 0.1 change is
  within what one more batch could move. The two runs that lower eval loss (LoRA 1e-4, full 1e-5) lower it by
  about 0.1; full at 1e-4 is too high a learning rate and raises it.
- First step takes 0.6 to 0.7 s once kernels are cached and 19 s on a cold kernel cache (JIT compile).
- Model build and weight load: 13 to 17 s.
- Device memory (ttml's `MemoryUsageTracker`, DRAM peak during step 1, excluding the weights resident before
  the capture started: its model-load snapshot read 0): LoRA 1.7 GB, full 8.1 GB. These are increments over
  the loaded model, not totals; the model itself is about 2.5 GB in bf16. Treat them as indicative.
- Step 1 loss of 1.31 and initial eval loss of 1.37 for a pretrained instruct model on its own chat format
  is plausible, so the weight loader (HF to ttml, with tied embeddings and rope scaling) looks right.
  This was not checked by generating text.

## Things that went wrong, verbatim where useful

1. `MemoryUsageTracker: Cannot snapshot - capture is not active`: `begin_capture()` returns a guard and the
   capture ends when the guard is dropped. Fixed by keeping the guard.
2. SDPA kernel failure with a batch whose longest sequence was 413 tokens (`sft_collate_fn` pads only to the
   longest example):
   `TT_FATAL ... sdpa_fw_device_operation.cpp:79: (!check_last_dim || logical[-1] == padded[-1]) && logical[-2] == padded[-2]`
   `Tensor 'Query' must be tile-aligned in sequence and head_dim ... Got logical=Shape([4, 32, 413, 64]), padded=Shape([4, 32, 416, 64])`.
   The sequence length must be a multiple of 32. The script now pads every example to `--max-len`.
3. My own bug, worth recording: the first successful runs passed labels equal to the input ids, with no shift
   by one. ttml does not shift labels; `sft_collate_fn` takes labels already aligned to what each position
   should predict. With unshifted labels the task is "copy the current token". Signs: initial eval loss 17.3
   (above the 11.85 of a uniform guess over 128,256 tokens) and loss falling to 0.01 in 20 steps with a held-out
   loss of 0.03. Those runs are kept as `unshifted-labels-bug_*.json` and are not results. A first look
   at those numbers would have called the pilot a success.
4. `import datasets` fails in the shared venv (huggingface_hub 1.33.0 has no `HfFolder`). It did not stop
   this run: the script reads parquet with pandas, and `ttml.trainers` and `ttml.datasets` imported fine.
   The stock `lora_llama` example imports `ttml.common.data`, which imports `datasets`, so that example would fail here.
5. ttnn 0.80.0 with the venv's numpy 2.5.3: a numpy 1.26.4 copy was put on `PYTHONPATH` as the earlier
   agent did. Whether the venv's numpy alone also works was not tried.

## What a teacher-data (sequence-level distillation) run would need

- Teacher text: prompt and answer pairs, written as `messages` (user turn, assistant turn). The script already
  reads a JSONL file with a `messages` column. At 512 tokens per padded sequence and 4,500 tokens/s, one epoch
  over 1,000 pairs of about 400 tokens takes roughly 2 minutes. Both modes fit on one chip with a lot of headroom,
  so longer sequences and bigger batches are the next thing to try (the 8 GB full-mode figure was at batch 4, length 512).
- Fix the dropping of long examples: 139 of 200 Capybara rows did not fit in 512 tokens. Use 1,024 or 2,048
  and gradient accumulation instead of discarding data.
- A real before and after comparison: held-out loss on teacher answers (need more than 8 conversations),
  and generated text side by side. `ttml.common.generate.generate` has no stop token and runs to the full
  sequence length, and whether a LoRA-wrapped model accepts the KV-cache arguments was not tested, so
  generation needs a small custom loop, or merging the adapter first.
- Learning rate matters: 1e-4 was fine for LoRA and too high for full fine-tuning on this setup.
- Logit distillation (matching the teacher's probabilities) needs a KL loss, which tt-train lacks
  (`ttml.ops.loss` has only mse, cross_entropy and nll). `SFTTrainer` takes `compute_loss_func`, so one could be
  built from existing ops. A teacher's logits for Qwen3 and a Llama student also use different vocabularies,
  so logit distillation would need a shared-vocabulary teacher.
