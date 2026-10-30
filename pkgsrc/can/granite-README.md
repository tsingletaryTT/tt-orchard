---
license: apache-2.0
base_model: ibm-granite/granite-4.0-h-1b
pipeline_tag: text-generation
tags:
- blackhole
- tenstorrent
- mamba2
- granite
- tt-model
---

# granite-4.0-h-1b-blackhole

> Runs [ibm-granite/granite-4.0-h-1b](https://huggingface.co/ibm-granite/granite-4.0-h-1b) (revision `d18cca4c121edb87d022116d281ce212c9136f57`) by IBM on one Tenstorrent Blackhole chip. The model weights are not in this repository. This is a community package and not a release of IBM or Tenstorrent.

This is a v6 thin bundle for `tt-model`. It holds the ttnn port of Granite-4.0-H-1B (a 1.5B-parameter hybrid of 36 Mamba-2 layers and 4 attention layers) and an OpenAI-compatible chat server written as an ASGI app, so `tt-model` can install and serve it. The bundle was built by the canning step of tt-orchard (`tt-orchard can`). The same code ships as a recipe package with a plain `run.sh` (version 0.3.0 of this model's other repository, `granite-4.0-h-1b-blackhole` under the same owner, which also has the evidence files).

## Serve it

```bash
tt-model serve <owner>/granite-4.0-h-1b-blackhole     # installs the bundle into its own venv, then serves it
```

`install.sh` builds a Python 3.12 venv from `requirements.txt`. Every line there is an exact pin, and `ttnn==0.80.0` comes from PyPI. `run.sh` serves the app with uvicorn on 127.0.0.1, port 20110 (`PORT` and `HOST` change them). The first start downloads the pinned weights (about 3 GB) into the Hugging Face cache. On a shared machine, take a chip lease first.

Without `tt-model`: run `bash install.sh`, then `bash run.sh`.

## API

`GET /health`, `GET /v1/models` and `POST /v1/chat/completions` (`messages`, `max_tokens` or `max_completion_tokens`, `stream`, `stop`, `temperature`, `top_p`, `seed`).

- The prompt is the model's own chat template applied to `messages`. Generation ends at the end-of-text token, at a `stop` string, at `max_tokens` (default 256, at most 4096) or when the context is full. `finish_reason` is `stop` or `length` accordingly.
- Without `temperature`, or with 0, decoding is greedy (this is what was measured). A temperature above 0 samples on the host.
- A response carries a `timings` object (prefill seconds, first-token seconds, decode tokens per second) next to `usage`.
- One request runs at a time. A second one gets `503` with `Retry-After: 5`. After a failed trace capture `/health` and every request return `503` until the server is restarted.
- A body over 1 MiB, malformed JSON, bad fields, a prompt longer than the context, tool calls and document inputs get a `400`.

```bash
curl -s http://127.0.0.1:20110/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}'
```

`tools/smoke.py --port 20110` checks a running server end to end.

## Settings

The server reads its settings from the environment, because uvicorn passes no flags:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GRANITE_MODEL_DIR` | the pinned revision from the Hugging Face cache | a directory with the model files |
| `GRANITE_MAX_LEN` | 2048 | context length (prompt plus output) |
| `GRANITE_SSD_PRECISION` | `fp32` | state precision of the Mamba-2 layers (`bf16` is faster and less accurate on long text) |
| `GRANITE_CHUNKED_PREFILL` | on | `0` prefills one token per step |
| `GRANITE_NO_TRACE` | off | `1` runs the decode step without a trace (slower) |

## Numbers

The numbers of the recipe package (same runtime code) are on its card. In short, on one Blackhole p300c chip with tt-metal v0.80.0: about 34 tok/s traced decode, 0.32 s to the first token after a short prompt, 123, 406 and 651 tok/s prefill for prompts of 128, 512 and 1,024 tokens, and top-1 agreement with the fp32 CPU reference of 0.979 on four short prompts. The numbers for this bundle, measured through its own `run.sh`, are in the tt-orchard repository (`docs/run-logs/2026-10-30-granite-thin-bundle.md`).

## Limits

- One request at a time, one chip, no batching. Context of 2048 tokens by default; the model supports 131,072.
- Greedy decoding was measured. Sampled output was not evaluated. No tool calls and no document inputs.
- Tested on one Blackhole p300c board with tt-metal v0.80.0 (the PyPI `ttnn` 0.80.0 wheel), Python 3.12.
- The server has no sign-in, no TLS and no rate limit. It binds 127.0.0.1 by default; `HOST=0.0.0.0` listens on every interface, which is only for a network you trust.
- The optional tt-lang kernel modes (`GRANITE_SSD=state_full_ro` and others) need the tt-lang package, which is not in `requirements.txt`. They are not the default.

## License

The model is licensed under the Apache License 2.0 by IBM. The code in this bundle is Tenstorrent's, also under the Apache License 2.0 (`LICENSE`, `NOTICE`).

## Provenance

- Weights: ibm-granite/granite-4.0-h-1b at revision `d18cca4c121edb87d022116d281ce212c9136f57`, not part of this bundle.
- `extra_code/` holds the runtime (`granite_chip.py`, `granite_trace.py`), the server's request and generation code (`granite_serve.py`), the ASGI front end (`granite_asgi.py`) and three optional tt-lang kernels, all from the tt-orchard repository. `granite_model.py` at the bundle root is the runner file that tt-model's packager asks for; it re-exports the app.
- `SHA256SUMS` lists the hash of every file.
