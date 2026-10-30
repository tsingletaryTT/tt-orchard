---
license: apache-2.0
base_model: Qwen/Qwen3.5-35B-A3B
pipeline_tag: text-generation
library_name: vllm
tags: [tenstorrent, blackhole, p300, moe, qwen3.5]
---

# qwen3.5-35b-a3b-blackhole-p300

A launch recipe for [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (35B parameters, about 3B active per token) on one Tenstorrent Blackhole
p300 board (two dies), using the fused decode ops of the `raahemnabeel/qwen3-coder-next` bundle. This package holds no weights and no image.
It holds a launcher, a 41-line patch for the bundle's weight loader, a script that fetches the text weights, and the checks and results behind
the numbers below. Text only: the vision tower of the checkpoint is not included.

## What you need

- One p300 board with the Tenstorrent driver, 1 GB hugepages mounted at `/dev/hugepages-1G`, and Docker.
- The bundle's image, pulled with `tt-model pull raahemnabeel/qwen3-coder-next` (image id `sha256:0c170421ca1b5b95f17ebf28d0c8256ccb8c5d320957bc8c4e72e7a07c43acfb`).
- About 70 GB of disk for the weights, and `pip install huggingface_hub`.

## Run it

```
python3 fetch_weights.py ./model            # 69.3 GB, text weights at the pinned revision
MODEL_DIR=$PWD/model DEVICES="0 1" PORT=8000 ./run.sh
```

`run.sh` extracts the image's weight loader, checks its sha256, applies `weights_qwen35_moe.patch`, mounts the result over the image's file
and starts `vllm serve` for one sequence and a 4096-token context. The served model name is `m`. See `docs/running.md`.

## Measured numbers

One p300 board (chips 2 and 3 of a four-chip machine), two dies, mesh P300, precision policy
`coder_next_routed_bfp4_mixers_bfp8` (bfp4 routed experts and head, bfp8 attention and shared experts), one sequence.

| Quantity | Value | Label |
| --- | --- | --- |
| Decode speed, median of 5 runs (128 greedy tokens each, after 1 discarded warm-up run) | 126.83 tok/s (min 126.81, max 127.07) | benchmark |
| Time to first token for a short prompt | 109 ms | benchmark |
| Estimated floor on decode speed | 428 tok/s (2.336 ms, 2392 MB read per token) | estimate |
| Share of the estimated floor | 29.6 percent | estimate |
| Start, kernel and tensor caches warm | 309 seconds on the boot check; 267 to 272 seconds in the earlier fusion-port runs (the kernel cache was seeded from the coder's) | single sample (one boot) |
| Start, empty caches | 728 seconds from container start to the first answer on a first boot with empty kernel and tensor caches | single sample |

The floor is `orchard/decode_floor.py` with these inputs: 2 dies at 512 GB/s each (datasheet peak), bfp8 attention and shared experts, bfp4 routed experts and head, bf16 router.
Its assumptions: batch of one, one token per pass (no speculative decoding); weights spread evenly over the dies so their bandwidth adds; datasheet peak bandwidth; the KV cache, recurrent state, norms, biases, rotary tables and the embedding lookup are not counted; the dtype policy is whatever the caller gives and is not read from the bundle. A share of 100 percent cannot be reached.

## Agreement with the Hugging Face reference

The reference is the same checkpoint in bfloat16 on the CPU (Hugging Face transformers, greedy). Each of 6 fixed prompts
(`prompts.json`) was run for 32 tokens on both. "Leading tokens" is how many tokens agree before the first difference.

| Prompt | Leading tokens that agree | Same token at the same position | First difference |
| --- | --- | --- | --- |
| 'The history of the printing press began ' | 8 of 32 | 0.781 | token 8: device token inside the reference top 5, reference gap 0.125 nats |
| 'def fibonacci(n):\n    """Return the n-th' | 5 of 32 | 0.344 | token 5: device token inside the reference top 5, reference gap 0.375 nats |
| 'Question: What is 17 multiplied by 23? A' | 10 of 32 | 0.312 | token 10: device token inside the reference top 5, reference gap 0.375 nats |
| 'La capital de Francia es París, y la cap' | 3 of 32 | 0.094 | token 3: device token inside the reference top 5, reference gap 0.438 nats |
| 'Photosynthesis is the process by which p' | 17 of 32 | 0.531 | token 17: device token inside the reference top 5, reference gap 0.125 nats |
| 'Q: Name three primary colors.\nA:' | 0 of 32 | 0.0 | token 0: device token inside the reference top 5, reference gap 0.375 nats |

In every prompt that differs, the first difference is a position where the device's token is inside the reference's top 5 (largest gap 0.438 nats). The device runs bfp4 routed experts, so a near tie between two words can flip. After a flip the two free-running texts diverge, so the
leading-token counts are short even when the device text is fluent. 0 of 6 prompts matched in full. The counts are what was measured; no claim is made
that a longer prefix would agree. The earlier 16-token check in `2026-10-29-fusion-port.md` (a different prompt) matched for 8 tokens, then flipped a tie of 0.125 nats and matched again from token 11.

A short chat request through `/v1/chat/completions` answered: 'Thinking Process:\n\n1.  **Analyze the Request:**\n    *   Target: Tenstorrent Blackhole card.\n    *   Constraint: One sentence.\n    *   Goal: Define/Describe what...' (finish reason length).
With thinking off, a request for the primary colors answered: 'Red, blue, and yellow.'.
A prompt of 3817 tokens with a code word at the start and the question at the end answered ' PELICAN.'.

## Two changes that were tried and not kept

| Change | Median decode speed | Result |
| --- | --- | --- |
| Baseline | 126.83 tok/s | |
| Sampler scan grid 10 x 5 (default 10 x 10) | 126.84 tok/s | inside the baseline spread, under the 3 percent bar, dropped |
| Router top-k constant 72 (default 80) | 126.78 tok/s | inside the baseline spread, under the 3 percent bar, dropped |

A scan grid wider than 10 columns fails with a `TT_FATAL` (the merge core sits at column 10).

## Limits and what was not tested

- Two dies of one p300 board only. The bundle's fabric layer boundary runs on four dies and is not used here.
- Text only. The vision tower and the multi-token-prediction head are not fetched or served.
- One sequence at a time (`--max-num-seqs 1`) and a 4096-token context. The longest prompt tested was 3817 tokens.
  Longer contexts, concurrent requests, tool calling and sampling at a temperature above zero were not tested.
- The numbers come from one machine and one day, with other jobs on the host (one-minute load average 8.5 to 8.7).
- Cold compile: 728 seconds from container start to the first answer on a first boot with empty kernel and tensor caches.
- The warm start was measured once. The board's lease ended during that boot check, so another job may have used the board at the same time; the figure was not re-measured.
- The patch was made for one image (id above). `run.sh` refuses a loader whose sha256 differs.
- The quality check covers 6 prompts of 32 tokens. It is not an evaluation benchmark. No accuracy benchmark was run.
- The default chat template starts a thinking section; a 96-token reply can end inside it (finish reason `length`). Pass `chat_template_kwargs {"enable_thinking": false}` to skip it.
- Raw completion prompts are not chat-formatted: one test prompt (Spanish) continued into a thinking block after its first line.
- The scores above are for the bfp4 routed-expert policy; a bfloat16 or bfp8 expert policy was not tried.

## License

The weights are Qwen/Qwen3.5-35B-A3B, released under the Apache License 2.0 by the Qwen team (Alibaba Cloud). The license file is in this package
(`LICENSE`, copied from the model repository at revision `59d61f3ce65a6d9863b86d2e96597125219dc754`). The image and the loader that the patch changes belong to the bundle
`raahemnabeel/qwen3-coder-next`, whose loader file carries an Apache-2.0 header. This package's own files are Apache-2.0. See `NOTICE`.
The weights revision pinned here is `59d61f3ce65a6d9863b86d2e96597125219dc754`.
