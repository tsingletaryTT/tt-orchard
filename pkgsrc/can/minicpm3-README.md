---
license: apache-2.0
base_model: openbmb/MiniCPM3-4B
library_name: ttnn
pipeline_tag: text-generation
tags:
- tt-model-cache
- tenstorrent
- blackhole
- thin
- p150
- text-generation
---

# minicpm3-4b-blackhole

A tt-model v6 thin bundle that serves openbmb/MiniCPM3-4B (revision `d6b14ddaefdb11c624dd75c3c779549bc90b08cb`) on 1 Tenstorrent Blackhole chip (mesh P150: one p150, or one chip of a p300c in a QuietBox 2). The tt-orchard canning step (`tt-orchard can`) staged it from the ttnn port in tt-orchard's `scripts/minicpm3`. The weights are not in this repository. This is a community package and not a release of OpenBMB or Tenstorrent.

## License

The model's repository card states the license `apache-2.0` (Apache License 2.0). `LICENSE` is that license's text. The authors' card says: "The usage of MiniCPM3-4B model weights must strictly follow MiniCPM Model License.md" (https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md). That link returned 404 on 2026-10-30, and the OpenBMB/MiniCPM repository holds only an Apache-2.0 LICENSE, so the weights' terms could not be read here. They are the model authors' to state; check their repository before use, including commercial use. The code in this bundle is covered by the Apache License 2.0 (tt-orchard). `tt-model` downloads the weights from openbmb/MiniCPM3-4B with your own Hugging Face account.

## What it runs

- Weights: openbmb/MiniCPM3-4B at revision `d6b14ddaefdb11c624dd75c3c779549bc90b08cb` (62 layers, hidden size 2560, 40 attention heads, multi-latent attention, a dense gated-SiLU MLP).
- Model code: the ttnn port in `extra_code/chip_model.py` (absorbed multi-latent attention with `flash_multi_latent_attention_decode`, a paged cache, the decode step captured as a trace, bf16 weights with HiFi2 for the MLP).
- Server: an OpenAI-compatible ASGI app (`extra_code/minicpm3_asgi.py`) that `run.sh` serves with uvicorn on 127.0.0.1, port 20130 (`PORT` and `HOST` change them).
- Environment: Python 3.12, `ttnn==0.80.0` from PyPI, numpy 1.26.4, CPU torch 2.7.0 and transformers 5.19.0 (the tokenizer and its chat template). Every line of `requirements.txt` is an exact pin.
- Context length 2048 tokens by default; one request at a time; greedy decoding only.

## Intended use

Chat and short-text work with a small Chinese and English model on one chip: a local assistant, drafting, rewriting and summarising short texts, question answering over a short pasted document, and a low-cost first pass in a pipeline. MiniCPM3-4B is a 4 billion parameter model from OpenBMB. Its model card lists Chinese and English, and says its overall performance is above Phi-3.5-mini-Instruct and GPT-3.5-Turbo-0125 and comparable with many 7B to 9B models. These are the authors' claims; this bundle did not run their benchmarks.

## Expected performance

Every number is labelled. A `measured` number names its evidence files, which are in the tt-orchard repository under `docs/run-logs/evidence/2026-10-30-minicpm3-thin/`; the first file holds the value. Conditions for every row: 2026-10-30, one chip of a p300c board in a QuietBox 2 (board with chips 2 and 3, one chip used), tt-metal runtime of `ttnn==0.80.0` (the PyPI wheel), bundle installed by tt-model's own installer into a fresh directory, bf16 weights, HiFi2 MLP, context 2048, trace on, greedy decoding. The host was shared with other jobs: its load average was 15 to 24 during these runs, and the recipe package's own card measured 33.87 tok/s at a load average of about 3.6 and 31.6 at 55, so the speed follows the host load.

| Number | Value | Label | Evidence |
|---|---|---|---|
| decode speed, `chat` prompt (21 tokens), 64 output tokens, warm kernel cache (second start), mean of 3 runs (range 33.24 to 33.74) | 33.5 tok/s | measured | `ttmodel-warm-cache.json` |
| decode speed, `chat, empty cache` prompt (21 tokens), 64 output tokens, empty kernel cache (first start of a fresh install), mean of 3 runs (range 33.21 to 33.31) | 33.3 tok/s | measured | `ttmodel-cold-cache.json` |
| time to first token, `chat` prompt (21 tokens) | 0.62 s | measured | `ttmodel-cold-cache.json`, `ttmodel-warm-cache.json` |
| prefill, 494-token prompt, 1 output token | 15.1 to 15.6 s (about 32 tok/s: the prompt goes through the chip one token per step) | measured | `ttmodel-cold-cache.json`, `ttmodel-warm-cache.json` |
| server ready after start, empty kernel cache, converted weights already on disk | 61.0 s | measured | `ttmodel-cold-cache.json` |
| server ready after start, warm kernel cache, converted weights on disk | 73.0 s (host load average 18 to 20 against 15 to 22 for the cold start; the two starts are within each other's noise) | measured | `ttmodel-warm-cache.json` |
| first request after ready, empty kernel cache, short answer | 0.71 s wall (0.63 s to the first token) | measured | `ttmodel-cold-cache.json` |
| server ready after the first start of a copy that still had to convert the 8 GB weights file (pickle already downloaded) | 66.0 s | measured | `can-boot-check.json` |
| same runtime as the recipe package (source build of ttnn, numpy 2.5.3), decode speed, 64 tokens, streamed | 32.7 to 32.8 tok/s against 33.3 to 33.5 tok/s for this bundle, measured with the same script on the same board | measured | `recipe-package.json`, `ttmodel-warm-cache.json` |
| teacher-forced next-token agreement with the float32 CPU reference, perplexity and block errors | the recipe package's numbers (16 of 16 tokens; perplexity 8.007 on the chip against 8.0574 on the CPU over 767 tokens) were measured with a source build of ttnn and numpy 2.5; they were not re-measured under this bundle's pins | not measured here | see the recipe package's card |

### p150 (1 chip), warm kernel cache, tt-model's installed bundle

One row per prompt, one user, means over the runs in `ttmodel-warm-cache.json` (3 runs for `chat`, 2 for `long prompt`). The `long prompt` row asks for one output token, so it has no decode rate.

| profile | ISL | OSL | concurrency | N | TTFT (ms) | prefill tok/s/u | decode tok/s/u | E2EL (s) |
|---|---|---|---|---|---|---|---|---|
| chat | 21 | 64 | 1 | 3 | 624 | 33.7 | 33.5 | 2.54 |
| long prompt | 494 | 1 | 1 | 2 | 15281 | 32.3 | - | 15.3 |

The first start downloads the pinned weights (about 8 GB, a PyTorch pickle) into the Hugging Face cache and converts them once to one safetensors file (another 8 GB, kept in `tt-orchard-minicpm3` inside `HF_HOME`). The download time was not measured: the weights were already in the cache. The server answers `/health` only after the model is loaded and the decode step is traced.

## Limitations

- One chip only; the two-chip tensor-parallel version is in tt-orchard and is not in this bundle.
- Decoding is greedy; `temperature` and `top_p` are accepted and ignored.
- The server feeds the prompt one token per decode step (about 30 ms each), so a 2,000-token prompt takes about a minute before the first token. The batched prefill is not in this bundle: in the recipe server it hung the chip on the second request after a trace.
- One request at a time: a second request gets `503` with `Retry-After: 5`. After a failure inside a chip step, `/health` and every request return `503` until the server is restarted.
- The default context is 2048 tokens (`MINICPM3_MAX_LEN` raises it, up to 32768). Speed was measured out to position 4000 on the recipe package (30.78 tok/s over 32 steps). Agreement and perplexity were measured over the first few hundred positions only.
- The 8-bit MLP variant (`MINICPM3_MLP_DTYPE=bfp8`, `MINICPM3_MLP_FIDELITY=lofi`) did not meet the stage 2 bar on the recipe package, and is not measured here.
- Function calls, the code interpreter and the 32k window that the authors describe are not tested.

## Risks and safety considerations

The checks above do not cover output quality beyond the agreement and perplexity numbers of the recipe package, long outputs, tool calling or safety behaviour. Output can differ from MiniCPM3-4B run on a CPU or GPU in ways those checks do not show. The server binds to 127.0.0.1 and has no sign-in; put authentication in front of it before exposing it.

## Not measured

- a download of the weights through `tt-model pull`, and a boot of the bundle from the Hub (it was installed and served from a local copy);
- agreement with the CPU reference under `ttnn==0.80.0` from PyPI and numpy 1.26.4;
- concurrent use and load.

## API

`GET /health`, `GET /v1/models` and `POST /v1/chat/completions` (`messages`, `max_tokens`, `stream`).

- The prompt is the model's own chat template applied to `messages`. Generation ends at an end-of-sequence token, at `max_tokens` (default 256, at most 4096) or when the context is full. `finish_reason` is `stop` or `length` accordingly.
- A non-streaming response carries a `timings` object (seconds to the first piece of text, and decode tokens per second) next to `usage`.
- A client that leaves ends its generation before the next chip step.
- A body over 1 MiB, malformed JSON, bad fields, and a prompt that leaves no room in the context get a `400`.

```bash
curl -s http://127.0.0.1:20130/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'
```

## Settings

The server reads its settings from the environment, because uvicorn passes no flags:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MINICPM3_DIR` | the pinned revision from the Hugging Face cache | a directory with `config.json`, the tokenizer files and `pytorch_model.bin` |
| `MINICPM3_WEIGHTS` | converted once from `pytorch_model.bin` | the weights as one safetensors file |
| `MINICPM3_CACHE` | `tt-orchard-minicpm3` inside `HF_HOME` | where the converted file goes |
| `MINICPM3_MAX_LEN` | 2048 | context length (prompt plus output), 128 to 32768, rounded up to a multiple of 128 |
| `MINICPM3_MLP_DTYPE` | `bf16` | `bfp8` and `bfp4` are also accepted |
| `MINICPM3_MLP_FIDELITY` | `hifi2` | `lofi` and `hifi4` are also accepted |

## Boot check

`tt-orchard can boot` copied the staged bundle, ran the copy's `install.sh`, served the copy's `run.sh` on one leased chip and checked the answer ("Paris"), the decode rate (33.2 tok/s against a floor of 25), the stream's `[DONE]` and the start-up log (`"traced": true`, `"context": 2048`). All 9 checks passed. The same bundle was then installed with tt-model's own installer and served by `tt-model serve local/minicpm3-4b-blackhole`; the numbers above come from that run.

## How to serve

    tt-model serve <owner>/minicpm3-4b-blackhole

`tt-model` installs the bundle into its own venv, then serves it. Without `tt-model`: run `bash install.sh`, then `bash run.sh`. On a shared machine, take a chip lease first. The first start can take several minutes (the download, the one-time conversion and the trace capture).
