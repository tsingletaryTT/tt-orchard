# Expert-use skew of Qwen3-Coder-Next (512 experts, top 10), 2026-11-25

Issue 739, step 2: how unevenly a model with many experts uses them. The first measurement (PR 746, 2026-11-11) used OLMoE-1B-7B, which has 64 experts.
The tier plan has used those figures as its estimate for every model since. This run measures a model with 512 experts per layer.

## Method

- Model: Qwen/Qwen3-Coder-Next, snapshot a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb, bfloat16 weights. 48 layers, all of them mixture-of-experts,
  512 routed experts and one shared expert per layer, top 10 routed experts per token, 3 of every 4 layers linear attention.
- The model is about 150 GiB, more than this host's free memory, so `scripts/expert_skew_stream.py` ran it one decoder layer at a time. It reads
  one layer's tensors from the checkpoint, runs every text chunk through that layer, counts the router's top-10 picks, frees the layer and reads the
  next. The process held about 5.3 GiB of resident memory when sampled mid-run. A test runs the same script and a whole-model forward pass on a tiny random checkpoint and requires equal
  counts in every layer.
- Texts, 16,384 tokens each as eight chunks of 2,048 tokens (chunks are separate sequences): the project documents of this repository (`docs/*.md`,
  18 files, the same kind of text as the OLMoE run), and the `orchard/*.py` modules (219 files), which is code.
- Host: 16 cores, 8 threads, `nice` 10 beside other jobs. 48 layers took 10 to 20 seconds each, about 15 minutes in all. The torch fallbacks for
  the linear-attention layers were used (`flash-linear-attention` and `causal_conv1d` are not installed), which is slower and gives the same
  result as the model class in transformers 5.19.
- The per-expert counts are in `2026-11-25-expert-skew-qwen3-coder-next.json`. Each text's counts add up to 163,840 picks per layer (16,384 tokens times
  10), checked for all 48 layers. After the run the `model`, `snapshot` and `text_note` fields were edited by hand, because the script had put the
  snapshot hash in `model` (it now takes `--model-name`).

## Result

Share of picks that go to the busiest 10, 25 and 50 percent of the experts, mean over the 48 layers:

| | busiest 10% (51 experts) | busiest 25% (128) | busiest 50% (256) | never picked | layers: busiest 25% |
| --- | --- | --- | --- | --- | --- |
| Project documents | 54.3% | 80.9% | 96.3% | 8.7% | 52% to 90% |
| Python code | 49.7% | 74.7% | 92.9% | 5.6% | 48% to 86% |
| OLMoE-1B-7B, project documents (PR 746) | 41.4% | 60.6% | 82.4% | 0.0% | 48% to 70% |
| Uniform router | 10% | 25% | 50% | 0% | |

- The 512-expert model is more skewed than the 64-expert model on both texts. Its busiest quarter of experts takes 75% to 81% of the picks, where
  OLMoE's takes 61%.
- The first layers are the least skewed (the busiest quarter takes 52% in layer 0 and 57% in layer 2 on documents) and the later layers settle at
  about 85%.
- Between 6% and 9% of the experts were never picked in 16,384 tokens, and that share is 0.2% to 21% depending on the layer.

### A hot set chosen on one text, used on the other

A placement that keeps the busiest experts of each layer on the chips was chosen from one text's counts and scored on the other text's picks.
The "own text" figure picks the experts and scores them on the same tokens, so it is higher than a fresh text would give.

| experts kept per layer | own text, documents | own text, code | chosen on documents, scored on code | chosen on code, scored on documents |
| --- | --- | --- | --- | --- |
| 51 (10%) | 53.8% | 49.2% | 40.7% | 44.2% |
| 128 (25%) | 80.9% | 74.7% | 64.0% | 71.5% |
| 256 (50%) | 96.3% | 92.9% | 86.4% | 92.4% |

A placement planned on documents keeps 64% of a code workload's picks on the chips where the same size of set keeps 75% when planned on code. The
cards of the published hybrid bundles say the same in words: unusual domains route to more host experts and decode slower.

## What it changes in the tier plan

`tt-orchard tier-plan --skew-file` reads this file. For the Qwen3-Coder-Next config on one chip with all weights in bfp4 (273 of 512 experts per layer
stay on the chip), the hit share and the host expert picks per token change as follows:

| skew used | hit share | host picks per token |
| --- | --- | --- |
| OLMoE estimate (the default) | 84% | 78.9 |
| measured on documents | 97% | 16.4 |
| measured on code | 93% | 31.7 |

The default estimate made the host path look five times as busy as the measurement does for this model, and a measurement on code gives about twice the
host load of one on documents.

## Limits

- One model, one checkpoint, two texts of 16,384 tokens each. A coding model measured on documents and on its own repository's code is not a
  measurement of what a user's prompts route to.
- Counts come from prefill over independent 2,048-token chunks. Decode steps route one token at a time and could differ, and long contexts were not
  tried.
- The hot set in the "own text" column is chosen and scored on the same tokens. A split between a selection half and a scoring half was not run.
- This says nothing about speed. The host path for experts is not written (issue 739, step 3), and the bundles' own decode numbers depend on their
  kernels, cache and drafter.
- Hy4-flash and Ornith-1.5-397B also have 512 experts per layer. Their routers are their own, so these figures are not theirs. A model with the same
  expert count and top-k as the file passes the file's shape check, and the plan's source sentence names the model that was measured.
