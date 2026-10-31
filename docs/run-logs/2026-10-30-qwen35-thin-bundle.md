# Qwen3.5-35B-A3B as a v6 thin bundle: install, boot and comparison with the recipe (2026-10-30)

Spec `config/can.qwen35.toml`, kind `vllm`, PR 304. Board 0 (chips 0 and 1, p300c) under `gozer acquire --exact 0 --owner-pid` held by a driver script for each run; every lease released and reset cleanly. Ports 20160 and 20161. The weights were in the Hugging Face cache (downloaded with `scripts/qwen35/fetch_snapshot.py`, sha256 checked per file, 71.9 GB).

## What went wrong first, and the fixes

1. The first install pulled the CUDA build of PyTorch (several GB). Fixed with `torch==2.11.0+cpu` and `torchvision==0.26.0+cpu` pins (the image's builds); the second install pulled no `nvidia-*` package.
2. Boot 1 (empty caches) loaded the 40 layers in about 11 minutes and then failed in warm-up: a JIT build of the kernel `gu_compute.cpp` (`sfpi::reinterpret` is not a member of `sfpi`) and one linker crash (`ncrisc ... Segmentation fault`). The host's SFPI is 7.89.0 and the image's tt-metal wants 7.74.0 (the log said so in a warning). Boot 2 failed the same way. Fix: the ttnn wheel now includes the image's SFPI 7.74.0 (`BUNDLE_SFPI = True` in the image's `setup.py`, 240 MB wheel); `run.sh` prefers `runtime/sfpi` in the wheel to the host's. The log then said `validated SFPI 7.74.0[897]`.
3. Three boot attempts stopped at the bundle-copy check or at the lease because I had restaged the bundle (new manifest timestamp, new card) after the copy was made. No chip time was used. The check did its job.

## Boot check (`tt-orchard can boot`, copy of the staged bundle, `--hf-home` the shared cache)

All 8 checks passed: health 200, models, chat answered "Paris" (thinking off), decode rate, stream ending `[DONE]`, health after, log contains "Application startup complete". Decode 125.34 tok/s over 128 streamed tokens (one run). Ready after 2,334 s: kernel cache empty, host load average 20 to 50 during compilation (other agents' jobs), weights read from a local disk.

## Installer path

`tt_kernel.cli._install_self_contained` into a scratch directory and index (XDG_CACHE_HOME and TT_MODEL_MODELS_DIR), then `tt-model serve local/qwen3.5-35b-a3b-blackhole-p300 --local-only --port 20161`. The boot check's kernel cache was copied into the install directory; the kernels were compiled again anyway. Health 200 after 2,052 s (host load average about 7). The first chat request after that took 0.13 s.

`tools/bench.py` (the recipe's benchmark: 1 warm-up, 5 runs of 128 greedy tokens), host load average 3.5: decode 125.39 / 125.75 / 125.80 tok/s (min, median, max), time to first token 102 to 136 ms. The recipe package: 126.81 / 126.83 / 127.07, 109 ms. The median is 0.85 percent lower.

## Same outputs as the recipe

The recipe's `device_quality.py` against this server (the model name changed to the bundle's served name): the six fixed prompts (greedy, 32 tokens) gave identical tokens in all 32 positions for every prompt against `2026-10-29-qwen35-package-data/device_quality.json`; the default chat reply and the thinking-off reply ("Red, blue, and yellow.") match; the 3,817-token prompt answered " PELICAN." in 0.84 s, the same as the recipe.

## Differences from the recipe package

| | Recipe | Bundle |
| --- | --- | --- |
| runtime | the published image, loader patch mounted at run time | wheels built from that image (ttnn with SFPI 7.74.0, models with the patch applied, plugin), a venv from exact pins |
| weights | `fetch_weights.py` writes a text-only model directory (69.3 GB) | the pinned snapshot (72 GB, vision shards included, not read) linked into `model-dir/` by `prepare_model_dir.py` with the edited `config.json` |
| model name | `m` | `qwen3.5-35b-a3b` |
| listen address | 127.0.0.1 | 127.0.0.1 (`HOST`), port 20160 (`PORT`) |
| cold start | 471 s and 727 s | 2,334 s and 2,052 s (loaded host; the cause was not isolated) |
| decode | 126.83 tok/s | 125.75 tok/s |

## Not done

- A start on a quiet host with an empty kernel cache. The 4x longer start is unexplained: the recipe's kernel cache had been seeded, and the bundle compiled with a load average of 7 to 50.
- `tt-model pull` of a published bundle (needs the Hub), tool calls, concurrency, accuracy against the CPU reference under these wheels (the identical-token result against the recipe is the evidence).
- The bundled compiler (GCC-based, from the image) is shipped in the wheel; its licence terms need a look before publishing.
