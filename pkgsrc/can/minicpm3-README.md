---
license: apache-2.0
base_model: openbmb/MiniCPM3-4B
library_name: ttnn
pipeline_tag: text-generation
tags:
- blackhole
- p150
- tt-dit-server
- tt-model-cache
- thin
- tenstorrent
- text-generation
---

# minicpm3-4b-blackhole

A tt-model v6 thin bundle that serves openbmb/MiniCPM3-4B (revision `d6b14ddaefdb11c624dd75c3c779549bc90b08cb`) on 1 Tenstorrent Blackhole chip (mesh P150: one p150, or one chip of a p300c in a QuietBox 2). The tt-orchard canning step (`tt-orchard can`) staged it from the ttnn port in tt-orchard's `scripts/minicpm3`. The weights are not in this repository. This is a community package and not a release of OpenBMB or Tenstorrent.

Runs on **p150** (mesh `P150`).

## At a glance

| | |
| --- | --- |
| Architecture | 62 layers, hidden size 2560, 40 attention heads, multi-latent attention (absorbed form on the chip), a dense gated-SiLU MLP. The prompt goes through the decode step one token at a time; the decode step is captured as a trace. |
| Hardware | p150 |
| License | apache-2.0 (the authors also point to a weights license whose link was unreachable; see License) |
| Status | Beta. Pulled from the Hub and served with `tt-model serve` on one chip of a p300c board on 2026-11-08 (answer "Paris", decode 33.0 to 33.6 tok/s over 12 runs, `bench-from-hub.json` in the tt-orchard repository); not validated on other hosts. |

## License

The model's repository card states the license `apache-2.0` (Apache License 2.0). `LICENSE` is that license's text. The authors' card says: "The usage of MiniCPM3-4B model weights must strictly follow MiniCPM Model License.md" (https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md). That link returned 404 on 2026-10-30, and the OpenBMB/MiniCPM repository holds only an Apache-2.0 LICENSE, so the weights' terms could not be read here. They are the model authors' to state; check their repository before use, including commercial use. The code in this bundle is covered by the Apache License 2.0 (tt-orchard). `tt-model` downloads the weights from openbmb/MiniCPM3-4B with your own Hugging Face account.

## What it runs

- Weights: openbmb/MiniCPM3-4B at revision `d6b14ddaefdb11c624dd75c3c779549bc90b08cb` (62 layers, hidden size 2560, 40 attention heads, multi-latent attention, a dense gated-SiLU MLP).
- Model code: the ttnn port in `extra_code/chip_model.py` (absorbed multi-latent attention with `flash_multi_latent_attention_decode`, a paged cache, the decode step captured as a trace, bf16 weights with HiFi2 for the MLP).
- Server: an OpenAI-compatible ASGI app (`extra_code/minicpm3_asgi.py`) that `run.sh` serves with uvicorn on 127.0.0.1, port 20130 (`PORT` and `HOST` change them).
- Environment: Python 3.12, `ttnn==0.80.0` from PyPI, numpy 1.26.4, CPU torch 2.7.0 and transformers 5.19.0 (the tokenizer and its chat template). Every line of `requirements.txt` is an exact pin.
- Context length 2048 tokens by default; one request at a time; greedy decoding only.

## Intended use

**Direct use:** Chat and short-text work with a small Chinese and English model on one chip: a local assistant, drafting, rewriting and summarising short texts, question answering over a short pasted document, and a low-cost first pass in a pipeline. MiniCPM3-4B is a 4 billion parameter model from OpenBMB. Its model card lists Chinese and English, and says its overall performance is above Phi-3.5-mini-Instruct and GPT-3.5-Turbo-0125 and comparable with many 7B to 9B models. These are the authors' claims; this bundle did not run their benchmarks.

**Out-of-scope use:**

- Coding agents and other long-context work (the default context is 2048 tokens).
- More than one simultaneous request.
- Sampling with a temperature (decoding is greedy).
- Function calls and the code interpreter (not tested).
- Any use that needs the weights' license terms confirmed first.

## Quickstart

```bash
tt-model serve <owner>/minicpm3-4b-blackhole     # installs the bundle into its own venv, then serves it
```

`tt-model` installs the bundle, then starts `run.sh`, which serves the app on 127.0.0.1 port 20130. The first start downloads the pinned weights (about 8 GB) into your Hugging Face cache, converts them once, compiles kernels and captures the trace, which takes several minutes. The server is ready when it logs `Application startup complete`. Without `tt-model`: run `bash install.sh`, then `bash run.sh`. On a shared machine, take a chip lease first.

## Expected performance

Every number is labelled. A `measured` number names its evidence file; the files are in the tt-orchard repository under `docs/run-logs/evidence/2026-11-08-minicpm3-can/` and in this bundle's `evidence/` folder. Conditions for every row unless it says otherwise: 2026-11-08, one chip of a p300c board in a QuietBox 2 (one chip used, under a gozer lease; the second chip of the board was idle), `ttnn==0.80.0` (the PyPI wheel) and numpy 1.26.4 in the bundle's own virtual environment, bf16 weights, HiFi2 MLP, trace on, greedy decoding, one request at a time. The kernel cache was warm for every row except the one that says cold. The host was shared with other jobs: its one-minute load average was 5.5 to 10.3 during the decode and prefill runs, and each run's start and end load is in the evidence file. A run is one request. Every decode and prefill row has one discarded warm-up request and 3 measured runs per prompt.

| Number | Value | Label | Evidence |
|---|---|---|---|
| decode speed, 4 prompts (a story, a Python function, a French translation, a two-sentence summary), 53 to 64 output tokens each, 3 runs per prompt, 12 runs in all, context 2048, warm cache: minimum, median, maximum | 33.27, 33.50, 33.64 tok/s | measured | `bench-ctx2048.json` |
| decode speed reported by the cold-cache boot check (first start of a fresh install, empty kernel cache, 64-token answer, 1 run) | 33.64 tok/s | measured, single sample | `boot-check-cold.json` |
| time to first token for a 21-token prompt (1 prompt, 3 runs) | 0.62 s | measured | `bench-ctx2048.json` |
| prefill, 522-token prompt, 1 output token, 3 runs: minimum, median, maximum | 15.75, 15.79, 15.80 s (33 tok/s) | measured | `bench-ctx2048.json` |
| prefill, 1,910-token prompt, 1 output token, 3 runs | 58.94, 59.57, 60.29 s (32 tok/s) | measured | `bench-ctx2048.json` |
| prefill, 8,202-token prompt, 1 output token, context 8704 (`MINICPM3_MAX_LEN=8704`), 3 runs | 268.23, 268.53, 268.74 s (30.5 tok/s; host load average 5.0 to 6.9) | measured | `bench-ctx8704.json` |
| decode speed at context 8704, same 4 prompts, 12 runs: minimum, median, maximum (host load average 4.5 to 6.7) | 32.23, 33.60, 33.74 tok/s | measured | `bench-ctx8704.json` |
| server ready after the first start of a fresh install (empty kernel cache, and the one-time conversion of the weights to one safetensors file; the pickle was already downloaded) | 126 s | measured, single sample | `boot-check-cold.json` |
| next-token agreement with the float32 CPU reference over 16 teacher-forced steps, `ttnn==0.80.0` from PyPI and numpy 1.26.4 | 16 of 16 (0 flips) | measured | `agreement-pypi-pins.txt` |
| logits agreement, last prompt position (40-token prompt) | PCC 0.9992, relative L2 0.021 | measured | `agreement-pypi-pins.txt` |
| logits agreement, first decode step | PCC 0.9978, relative L2 0.050 | measured | `agreement-pypi-pins.txt` |
| block error against the float32 reference, layers 0, 30 and 61 | 0.0077, 0.0052, 0.0556 (the CPU bfloat16 floor at layer 61 is 0.0199) | measured | `agreement-pypi-pins.txt` |
| teacher-forced perplexity over 767 WikiText-2 tokens | 8.007 on the chip against 8.057 for float32, with a source build of ttnn (not repeated under this bundle's pins) | measured in the earlier run on this model's port | `docs/run-logs/2026-10-29-overnight-minicpm3-4b.md` in tt-orchard |

### p150 (1 chip), warm kernel cache, the served bundle

One row per prompt length, one user. The decode column is the server's own report; the prefill column is the prompt tokens divided by the whole request time with one output token (the prompt goes through the chip one token per step).

| profile | ISL | OSL | concurrency | N | TTFT (ms) | prefill tok/s/u | decode tok/s/u | E2EL (s) |
|---|---|---|---|---|---|---|---|---|
| chat (the story prompt) | 31 | 64 | 1 | 3 | 934 | 33.2 | 33.6 | 2.8 |
| short prompt | 522 | 1 | 1 | 3 | 15785 | 33.1 | - | 15.8 |
| long prompt | 1910 | 1 | 1 | 3 | 59571 | 32.1 | - | 59.6 |
| 8k prompt (context 8704) | 8202 | 1 | 1 | 3 | 268530 | 30.5 | - | 268.5 |

The first start downloads the pinned weights (about 8 GB, a PyTorch pickle) into the Hugging Face cache and converts them once to one safetensors file (another 8 GB, kept in `tt-orchard-minicpm3` inside `HF_HOME`). The server answers `/health` only after the model is loaded and the decode step is traced.

## Limitations

- One chip only. A two-chip tensor-parallel version exists in tt-orchard (about 38 tok/s, measured in the earlier run) and is not in this bundle.
- Decoding is greedy; `temperature` and `top_p` are accepted and ignored.
- The prompt goes through the chip one token per decode step, so prefill runs at about 32 tok/s: a 2,000-token prompt takes about one minute and an 8,000-token prompt about four minutes before the first token. A batched prefill exists in tt-orchard (0.37 s for a 922-token prompt in the earlier run) and is not in this bundle, because in the server it hung the chip on the second request after a trace was captured. The cause has not been found.
- One request at a time: a second request gets `503` with `Retry-After: 5`. After a failure inside a chip step, `/health` and every request return `503` until the server is restarted.
- The default context is 2048 tokens. `MINICPM3_MAX_LEN` raises it, up to 32768; the larger context sizes the cache and was measured here to 8,704 tokens. Decode speed at a long position was measured in the earlier run: 30.8 tok/s at position 4,000.
- Agreement was measured over 16 teacher-forced steps after a 40-token prompt, and perplexity (earlier run, source-build ttnn) over the first 767 tokens of one text. Neither says how the model behaves on other text, in Chinese, or over long outputs.
- Only 3 runs stand behind each speed figure, on a shared host whose load average moved the decode rate by about 0.4 tok/s in these runs and by about 2 tok/s between the recipe package's quiet and loaded measurements.
- The 8-bit MLP variant (`MINICPM3_MLP_DTYPE=bfp8`, `MINICPM3_MLP_FIDELITY=lofi`) is not measured under this bundle's pins.
- Function calls, the code interpreter and the 32k window that the authors describe are not tested.

## Risks and safety considerations

The checks above do not cover output quality beyond the agreement and perplexity numbers listed in Expected performance, long outputs, tool calling or safety behaviour. Output can differ from MiniCPM3-4B run on a CPU or GPU in ways those checks do not show. The server binds to 127.0.0.1 and has no sign-in; put authentication in front of it before exposing it.

## Not measured

- a pull of the bundle from the Hub on a host other than the one it was built on;
- concurrent use and load (the server takes one request at a time);
- perplexity and free-running text agreement under this bundle's pins (the figures above come from the earlier run's source-build ttnn);
- decode speed at a context between 2,048 and 8,704 tokens, and at positions beyond 8,704;
- the 8-bit MLP variant and the two-chip version under this bundle.

## Using it

This bundle serves an OpenAI-compatible chat API on port 20130. `GET /health`, `GET /v1/models` and `POST /v1/chat/completions` (`messages`, `max_tokens`, `stream`).

- The prompt is the model's own chat template applied to `messages`. Generation ends at an end-of-sequence token, at `max_tokens` (default 256, at most 4096) or when the context is full. `finish_reason` is `stop` or `length` accordingly.
- A non-streaming response carries a `timings` object (seconds to the first piece of text, and decode tokens per second) next to `usage`.
- A client that leaves ends its generation before the next chip step.
- A body over 1 MiB, malformed JSON, bad fields, and a prompt that leaves no room in the context get a `400`.

```bash
curl -s http://127.0.0.1:20130/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'
```

### Settings

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

`tt-orchard can boot` (2026-11-08) copied the staged bundle, ran the copy's `install.sh` (a new virtual environment from the exact pins, `ttnn==0.80.0` from PyPI), served the copy's `run.sh` on one leased chip with an empty kernel cache and checked the answer ("Paris"), the decode rate (33.64 tok/s over a 64-token answer, against a floor of 25), the stream's `[DONE]` and the start-up log (`"traced": true`, `"context": 2048`). All 9 checks passed, and the server was ready 126 s after it started, including the one-time conversion of the weights. The record is `boot-check-cold.json`. An earlier boot check of the same bundle (2026-10-31, host load average about 20) passed with 31.5 tok/s, and the bundle was also installed and served by tt-model's own installer from a local copy.
