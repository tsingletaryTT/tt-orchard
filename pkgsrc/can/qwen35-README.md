---
license: apache-2.0
base_model: Qwen/Qwen3.5-35B-A3B
pipeline_tag: text-generation
library_name: vllm
tags:
- blackhole
- tenstorrent
- p300
- moe
- qwen3.5
- tt-model
---

# qwen3.5-35b-a3b-blackhole-p300

> Runs [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (revision `59d61f3ce65a6d9863b86d2e96597125219dc754`) by the Qwen team on one Tenstorrent Blackhole p300 board (two dies). The model weights are not in this repository. This is a community package.

This is a v6 thin bundle for `tt-model`: `tt-model` installs it into its own virtual environment and serves it with vLLM. The model has 35B parameters, about 3B active per token. The bundle holds a tt-metal build with the MoE ops, the model code, the vLLM plugin and a launcher. It holds no weights and no container image. Text only: the vision tower of the checkpoint is not used.

## Serve it

```bash
tt-model serve <owner>/qwen3.5-35b-a3b-blackhole-p300     # installs the bundle into its own venv, then serves it
```

Without `tt-model`: run `bash install.sh`, then `bash run.sh`. `run.sh` serves an OpenAI-compatible API on 127.0.0.1, port 20160 (`PORT` and `HOST` change them, and any vLLM argument after `run.sh` overrides the defaults). The server has no sign-in; do not bind it to a public address.

The first start downloads the weights (about 72 GB) from the Hugging Face Hub at the pinned revision, builds `model-dir/` from them, builds the tensor cache and compiles kernels. Later starts reuse them. Plan the disk space: the weights, the venv (several GB, vLLM pulls a CUDA build of PyTorch) and the caches.

```bash
curl -s http://127.0.0.1:20160/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name the three primary colors."}],"max_tokens":96,"chat_template_kwargs":{"enable_thinking":false}}'
```

The model name is the path of `model-dir`; `GET /v1/models` lists it. The default chat template starts a thinking section, so a short reply can end inside it (finish reason `length`). Pass `chat_template_kwargs {"enable_thinking": false}` to skip it.

## Hardware

One p300 board, both dies. Set `TT_VISIBLE_DEVICES` to the two dies (a gozer lease does this); `run.sh` refuses fewer. The host needs the SFPI compiler (`/opt/tenstorrent/sfpi`), 1 GB hugepages and the Tenstorrent driver.

## Settings

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
| `wheels/tt_metal_models-0.79.0.dev20261030+qwen35moe-*.whl` | that image's `models` tree, with the 41-line loader patch of the recipe package applied to `models/autoports/qwen_qwen3_coder_next/tt/weights.py` |
| `wheels/vllm_tt_plugin-0.1.0-*.whl` | the vLLM plugin of that image |
| `base_config/config.json` | the checkpoint's `config.json` with three edits: architecture `Qwen3NextForCausalLM`, no multi-token-prediction layers, no mrope keys |
| `prepare_model_dir.py` | builds `model-dir/` at start-up from the pinned snapshot (tokenizer, index and weight files, linked) and `base_config/` |
| `tools/bench.py`, `tools/prompt.txt` | the decode benchmark of the recipe package: `python tools/bench.py PORT 5 128 tools/prompt.txt MODEL` |

## Numbers

All numbers below were measured on the recipe package `qwen3.5-35b-a3b-blackhole-p300` (the same model code, wheels' source image and settings, run in that image with a mounted patch) on chips 2 and 3 of a four-chip Blackhole machine, on 2026-10-29, with other jobs on the host. Precision: bfp4 routed experts and head, bfp8 attention and shared experts, bf16 router. They were not measured on this bundle: this bundle's boot check did not run (see Limits), so none of these rows is a measurement of the installed bundle.

| Number | Value | Label | Evidence |
| --- | --- | --- | --- |
| decode speed, recipe package, 128 greedy tokens, median of 5 runs after 1 warm-up (min 126.81, max 127.07), warm kernel cache, host load 8.5 to 8.7 | 126.83 tok/s | measured | recipe package run log |
| time to first token, recipe package, 14-token prompt, warm | 109 ms | measured | recipe package run log |
| Prompt of 3,817 tokens, question at the end, warm | answered in 0.84 s | measured | recipe package run log |
| Cold start, empty kernel and tensor caches, container start to first answer | 727 s | single sample (recipe package) | recipe package run log |
| Cold start of the restaged recipe package, empty caches, container start to `/v1/models` | 471 s | single sample (recipe package) | recipe package run log |
| Warm start, caches built | 308 s | single sample, may have overlapped another job | recipe package run log |
| Decode floor from datasheet bandwidth | 428 tok/s (2 dies at 512 GB/s, 2392 MB read per token) | estimate | recipe package run log |
| Share of the floor | 29.6 percent | estimate | recipe package run log |

The first request after a cold start was not timed apart from the start itself: kernels compile during warm-up, and the 727 s covers the compile, the tensor-cache build and the first answer together.

Agreement with the Hugging Face reference (bfloat16 on the CPU, greedy, 6 prompts of 32 tokens, recipe package): 0 of 6 prompts matched in full. In every prompt that differs, the first difference is a near tie (largest reference gap 0.438 nats) where the device token is inside the reference top 5, and 3 to 17 leading tokens agree. The device runs bfp4 routed experts, so a near tie can flip. This is not an accuracy benchmark.

## Limits and what was not tested

- This bundle was staged, gated and checked by tests. Its install and boot on a board did not finish: the host ran out of disk for the 72 GB download, the venv and the caches. The bundle's decode speed, cold start and first-request time are therefore not measured.
- The two dies of one p300 board only. One sequence at a time and a 4096-token context; the longest prompt tested was 3,817 tokens. Concurrent requests, tool calls and sampling above temperature 0 were not tested. The vision tower and the multi-token-prediction head are not used.
- The `ttnn` wheel is a tt-metal development build (`0.79.0.dev20261030+qwen35moe`), not a release.
- The first start needs network access for the weights and about 72 GB of disk, plus the venv and the caches inside the bundle directory.

## License

The weights are Qwen/Qwen3.5-35B-A3B, released under the Apache License 2.0 by the Qwen team (Alibaba Cloud). The license file is in this bundle (`LICENSE`, copied from the model repository at revision `59d61f3ce65a6d9863b86d2e96597125219dc754`). The wheels are built from tt-metal (Apache License 2.0); see `NOTICE`. The weights revision pinned here is `59d61f3ce65a6d9863b86d2e96597125219dc754`.
