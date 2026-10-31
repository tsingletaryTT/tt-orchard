---
license: apache-2.0
base_model: Qwen/Qwen3.5-35B-A3B
pipeline_tag: text-generation
library_name: vllm
tags:
- blackhole
- tenstorrent
- p300c
- vllm
- thin
- tt-model-cache
- moe
- qwen3.5
- tt-model
---

# qwen3.5-35b-a3b-blackhole-p300 on Tenstorrent p300c

> Runs [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (revision `59d61f3ce65a6d9863b86d2e96597125219dc754`) by the Qwen team on one Tenstorrent Blackhole p300 board (two dies). The model weights are not in this repository. This is a community package.

This is a v6 thin bundle for `tt-model`: `tt-model` installs it into its own virtual environment and serves it with vLLM. The model has 35B parameters, about 3B active per token. The bundle holds a tt-metal build with the MoE ops, the model code, the vLLM plugin and a launcher. It holds no weights and no container image. Text only: the vision tower of the checkpoint is not used.

Runs on Tenstorrent **p300c** (mesh P300, two dies)

## At a glance

| | |
| --- | --- |
| Architecture | mixture of experts, 40 layers (gated delta net and gated attention), 256 experts, 8 used per token |
| Hardware | one p300c board, both dies |
| Context | 4096 tokens, one sequence at a time |
| Input and output | text |
| License | Apache License 2.0 |

## Intended use

Text generation and chat with a 35B mixture-of-experts model on one p300 board, for one user at a time, with contexts up to 4096 tokens. Not for concurrent serving, long contexts, tool use or images (see Limitations).

## Quickstart

```bash
tt-model serve <owner>/qwen3.5-35b-a3b-blackhole-p300     # installs the bundle into its own venv, then serves it
```

Without `tt-model`: run `bash install.sh`, then `bash run.sh`. `run.sh` serves an OpenAI-compatible API on 127.0.0.1, port 20160 (`PORT` and `HOST` change them, and any vLLM argument after `run.sh` overrides the defaults). The server has no sign-in; do not bind it to a public address.

The first start downloads the weights (about 72 GB) from the Hugging Face Hub at the pinned revision, builds `model-dir/` from them and compiles kernels into a cache inside the bundle directory; later starts reuse the cache. Plan the disk space: the weights, the venv (about 3.5 GB, CPU build of PyTorch) and the kernel cache (about 2 GB).

```bash
curl -s http://127.0.0.1:20160/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name the three primary colors."}],"max_tokens":96,"chat_template_kwargs":{"enable_thinking":false}}'
```

The served model name is `qwen3.5-35b-a3b` (`GET /v1/models` lists it). The default chat template starts a thinking section, so a short reply can end inside it (finish reason `length`). Pass `chat_template_kwargs {"enable_thinking": false}` to skip it.

## Prerequisites

One p300 board, both dies. Set `TT_VISIBLE_DEVICES` to the two dies (a gozer lease does this); `run.sh` refuses fewer. The host needs 1 GB hugepages and the Tenstorrent driver. The SFPI compiler the kernels need (7.74.0) is in the `ttnn` wheel.

## Capabilities

| Setting | Value | Where |
| --- | --- | --- |
| sequences at once | 1 | `--max_num_seqs` |
| context | 4096 tokens | `--max_model_len` |
| block size | 64 | `--block_size` |
| batching | prefix caching, chunked prefill, 2048 tokens per batch | vLLM arguments in `run.sh` |
| sampling | on the device | `additional-config` `sample_on_device_mode` |
| trace region, fabric | 768 MiB, `FABRIC_1D` | `additional-config` |
| fused decode | `QWEN35_*` variables (MoE, GDN and attention decode ops, chunked prefill, replicated decode) | `run.sh` |

## What is in the bundle

| File | What it is |
| --- | --- |
| `wheels/ttnn-0.79.0.dev20261030+qwen35moe-*.whl` | the tt-metal build of the image `tt-model/qwen3-coder-next:0c170421ca1b`. It has the MoE ops (`moe_sorted_dispatch`, `moe_gather_add_rows`) that the PyPI `ttnn` 0.80.0 lacks |
| `wheels/tt_metal_models-0.79.0.dev20261030+qwen35moe-*.whl` | that image's `models` tree, with a 41-line loader patch applied to `models/autoports/qwen_qwen3_coder_next/tt/weights.py` |
| `wheels/vllm_tt_plugin-0.1.0-*.whl` | the vLLM plugin of that image |
| `base_config/config.json` | the checkpoint's `config.json` with three edits: architecture `Qwen3NextForCausalLM`, no multi-token-prediction layers, no mrope keys |
| `prepare_model_dir.py` | builds `model-dir/` at start-up from the pinned snapshot (tokenizer, index and weight files, linked) and `base_config/` |
| `tools/bench.py`, `tools/prompt.txt` | the decode benchmark: `python tools/bench.py PORT 5 128 tools/prompt.txt MODEL` |

## Expected performance

Measured on 2026-10-30 on this bundle, installed in its own venv, on the two dies of board 0 (chips 0 and 1, p300c) of a four-chip Blackhole machine shared with other jobs. Software: `ttnn` 0.79.0.dev20261030+qwen35moe (the wheel in this bundle), the bundle's own SFPI 7.74.0, vLLM 0.26.0. Precision: bfp4 routed experts and head, bfp8 attention and shared experts, bf16 router. One sequence, greedy.

| Number | Value | Label | Evidence |
| --- | --- | --- | --- |
| decode speed, tt-model install, 128 greedy tokens, median of 5 runs after 1 warm-up (min 125.39, max 125.80), kernels already compiled, host load average 3.5 | 125.75 tok/s | measured | `tools/bench.py` output |
| decode speed, boot check of an installed copy, 128 streamed tokens, one run | 125.34 tok/s | measured | boot check |
| time to first token, 5 runs, short prompt, warm | 102 to 136 ms | measured | `tools/bench.py` output |
| prompt of 3,817 tokens with a code word at the start, question at the end, warm | answered `PELICAN` in 0.84 s | measured | quality script |
| first chat request after the server was ready, 32 tokens | 0.13 s | measured | one request |
| cold start, empty kernel cache, boot check, `run.sh` start to `/health` 200 (weights read from the Hugging Face cache on a local disk) | 2,334 s | single sample | boot check; host load average 20 to 50 during kernel compilation |
| start under `tt-model serve`, kernel cache copied from the boot check (it was not reused: kernels compiled again) | 2,052 s | single sample | server log; host load average about 7 |
| decode floor from datasheet bandwidth | 428 tok/s (2 dies at 512 GB/s, 2392 MB read per token) | estimate | recipe package log |
| share of the floor | 29.4 percent | estimate | 125.75 / 428 |

The start is slow because the kernels compile on the first start; the two samples ran on a host with a load average of 7 to 50, and a start on a quiet host with an empty kernel cache was not measured.

Agreement: six fixed prompts (greedy, 32 tokens, top-5 log-probabilities) gave the same tokens in all 32 positions as the sibling recipe package (see Related packages). That package's own comparison with a Hugging Face bfloat16 CPU reference found 0 of 6 prompts matching in full, every first difference a near tie (largest reference gap 0.438 nats) with the device token inside the reference top 5, because of the bfp4 routed experts. This is not an accuracy benchmark.

## Limitations

- The two dies of one p300 board only. One sequence at a time and a 4096-token context; the longest prompt tested was 3,817 tokens. Concurrent requests, tool calls and sampling above temperature 0 were not tested. The vision tower and the multi-token-prediction head are not used.
- The first start is slow (above: 2,052 to 2,334 s on a loaded host) and downloads about 72 GB. It needs that disk space plus the venv (about 3.5 GB) and the kernel cache (about 2 GB) inside the bundle directory.
- The `ttnn` wheel is a tt-metal development build (`0.79.0.dev20261030+qwen35moe`), not a release. It carries the SFPI 7.74.0 compiler (about 240 MB wheel): the model's kernels do not compile with the newer SFPI (7.89.0) a host may have. The compiler is a Tenstorrent build of GCC and binutils (GPL and LGPL); its source is the tag 7.74.0 of github.com/tenstorrent/sfpi (see NOTICE).
- Not published and not run through `tt-model pull`: tt-model installs a v6 bundle from the Hub only, so this bundle was installed with tt-model's installer function from a local directory.

## Licensing

The weights are Qwen/Qwen3.5-35B-A3B, released under the Apache License 2.0 by the Qwen team (Alibaba Cloud). The license file is in this bundle (`LICENSE`, copied from the model repository at revision `59d61f3ce65a6d9863b86d2e96597125219dc754`). The wheels are built from tt-metal (Apache License 2.0); see `NOTICE`. The weights revision pinned here is `59d61f3ce65a6d9863b86d2e96597125219dc754`.

## Related packages

- `episod/qwen3.5-35b-a3b-blackhole-p300` (recipe package): the same model code and settings, run in the published image `tt-model/qwen3-coder-next:0c170421ca1b` with the loader patch mounted at run time. It measured 126.83 tok/s decode (median of 5, 2026-10-29, chips 2 and 3, host load 8.5 to 8.7) and cold starts of 471 s and 727 s.
- [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B): the upstream model.

## Feedback

Open a Discussion on this repository's page, or write to support@tenstorrent.com.

## Provenance

| Component | Built from |
| --- | --- |
| `ttnn` wheel | tt-metal as built in `tt-model/qwen3-coder-next:0c170421ca1b`, version `0.79.0.dev20261030+qwen35moe`, with its SFPI 7.74.0 |
| `tt_metal_models` wheel | the `models` tree of that image, with the loader patch applied |
| `vllm_tt_plugin` wheel | the plugin installed in that image |
| weights | Qwen/Qwen3.5-35B-A3B at `59d61f3ce65a6d9863b86d2e96597125219dc754` |
