---
license: apache-2.0
base_model: allenai/Olmo-Hybrid-7B
pipeline_tag: text-generation
tags:
- blackhole
- tenstorrent
- hybrid
- gated-delta-net
- olmo
- tt-model
---

# olmo-hybrid-7b-blackhole

> Runs [allenai/Olmo-Hybrid-7B](https://huggingface.co/allenai/Olmo-Hybrid-7B) (revision `4f1cc566f9fdf3ce68da2ab6a788a83d89896dcf`) by the Allen Institute for AI on a mesh of four Tenstorrent Blackhole chips (two p300c boards). The model weights are not in this repository. This is a community package and not a release of the Allen Institute for AI or of Tenstorrent.

Olmo-Hybrid-7B is a pretrained 7.4B-parameter model that mixes 24 gated delta net layers with 8 full-attention layers. This is a v6 thin bundle for `tt-model`. It holds a ttnn port of the model and an OpenAI-style server written as an ASGI app, so `tt-model` can install and serve it. A language model wrote the port inside the tt-orchard bring-up harness (a local coder model, with a hosted model drafting repairs) and the harness checked it against a CPU reference. The bundle was built by the canning step of tt-orchard (`tt-orchard can`).

## Serve it

```bash
tt-model serve <owner>/olmo-hybrid-7b-blackhole     # installs the bundle into its own venv, then serves it
```

`install.sh` builds a Python 3.12 venv from `requirements.txt`. Every line there is an exact pin, and `ttnn==0.80.0` comes from PyPI. `run.sh` serves the app with uvicorn on 127.0.0.1, port 20120 (`PORT` and `HOST` change them). The first start downloads the pinned weights (about 15 GB) into the Hugging Face cache. The server needs four chips: on a shared machine, take a lease on both boards first. Without `tt-model`: run `bash install.sh`, then `bash run.sh`.

## API

`GET /health`, `GET /v1/models`, `POST /v1/completions` and `POST /v1/chat/completions`.

- Olmo-Hybrid-7B is a base model with no chat template. Use `/v1/completions` with a prompt. The chat route writes the messages as `role: content` lines that end with `assistant:`, which a base model follows only loosely.
- Fields: `prompt` or `messages`, `max_tokens` (default 128, at most 65,536) or `max_completion_tokens`, `stream`, `stop` (up to four strings), `temperature`, `top_p`, `seed`, and `return_token_ids` (adds the generated token ids to the response). Decoding is greedy unless `temperature` is above 0; this card's numbers use greedy decoding.
- A response carries `usage` and a `timings` object (prefill seconds, decode seconds, decode tokens per second).
- One request runs at a time. A second one gets `503` with `Retry-After: 5`. A body over 1 MiB, malformed JSON, bad fields or a prompt longer than the context get a `400`.

```bash
curl -s http://127.0.0.1:20120/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt":"The capital of France is","max_tokens":16}'
```

## Settings

The server reads its settings from the environment, because uvicorn passes no flags:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLMO_MODEL_DIR` | the pinned revision from the Hugging Face cache | a directory with the model files |
| `OLMO_MAX_LEN` | 16384 | context limit in tokens, prompt plus answer (the model supports 65,536) |
| `OLMO_CHIPS` | 4 | chips in the mesh (a 1x4 mesh with the 1-D fabric) |

## Expected performance

Measured through the server's own `run.sh` on a mesh of four Blackhole p300c chips with `ttnn` 0.80.0, one request at a time, greedy decoding, with a warm kernel cache. Wall time through the server. Each figure is the median of the runs stated beside it. The host load average (1, 5 and 15 minutes) was 3.6, 7.6, 9.1 before the measurement and 9.2, 9.9, 9.9 after it. The port does much of its work on the host CPU, so decode speed falls when other jobs use the host: the figures below were measured while the load average was high, and a quiet host gave a higher decode rate in the harness row.

| Measure | Value |
| --- | --- |
| Served decode, batch size 1, greedy (headline) | 7.277 tok/s (min 3.589, max 9.295, 15 runs) |
| Prefill, 141 prompt tokens | 1.077 s (min 0.996, max 1.129, 3 runs), 130.9 tok/s |
| Prefill, 561 prompt tokens | 4.623 s (min 4.417, max 4.788, 3 runs), 121.4 tok/s |
| Prefill, 2261 prompt tokens | 22.368 s (min 22.256, max 22.6, 3 runs), 101.1 tok/s |
| Prefill, 9101 prompt tokens | 137.655 s (min 130.653, max 165.194, 3 runs), 66.1 tok/s |
| In-process decode in the bring-up harness (not the served rate; host load not recorded) | 13.9 tok/s (min 13.9, max 13.9, 3 runs, 32 steps, warm) |

A second measurement run, with an empty kernel cache at the start, gave a served decode median of 9.68 tok/s (min 7.963, max 11.467, 15 runs); the host load average was 5.9, 8.2, 8.0 before it and 7.8, 9.3, 9.0 after it. The two runs ran at different times with other jobs on the host, so the difference between their medians should not be read as an effect of the kernel cache.
With an empty kernel cache the server took 40 s to become ready, and the first request (8 tokens) took 6.74 s.

In 4 boot checks of this bundle run before this card was written (one 64-token request each, host load not recorded) the served decode rate was 2.2, 5.5, 7.6, 9.3 tok/s. Taken together, the served decode rate of this package has been anywhere from 2.2 to 11.5 tok/s across the benchmark runs above and these boot checks, depending on what else the host was doing.

Decode runs used 5 different prompts with 64 new tokens each, 3 runs per prompt, after one discarded warm-up request. Batching is not supported: a second request gets 503 while one is running.

## Quality

The bring-up harness's stage 6 check (teacher-forced top-1 agreement 1.0, perplexity 2.996 on the reference's own 32-token continuation, a free run that matched for all 32 tokens) covers one prompt only. The table below is a second check on four short prompts, made through the server: greedy tokens from the served port against the transformers CPU reference (bf16, greedy, 32 new tokens each).

| Prompt | Tokens compared | Result |
| --- | --- | --- |
| `The capital of France is` | 32 | first difference at token index 1 (1 of 32 positions equal) |
| `Water boils at a temperature of` | 32 | all match |
| `def fibonacci(n): """Return the n-th` | 32 | all match |
| `In 1969, the first humans to walk on the` | 32 | all match |

At the first difference (`The capital of France is`, token index 1) the reference's full-forward logits, teacher-forced on its own continuation, rank '.' (logit 9.3125) above ',' (logit 9.25) by 0.0625. Given the same prefix, the port's top two are '.' (logit 9.25) and ',' (logit 9.1875), a gap of 0.0625. The reference's own greedy generation chose ',' (token id 11) at this position. The served port gave the same token at this index on all 6 repeats of the prompt. A gap of 0.0625 is one step of a bf16 value near 10 (0.0625) and is within the 0.25 that neighbouring bf16 values can produce. The port ranks the same two tokens in the same order as the reference's full forward, and the reference's own generation took the other side of the tie (it is not consistent with its own full forward here), so this difference is a bf16 near tie.

These checks cover short prompts. Nothing here measures long-context quality, instruction following, or anything on held-out text.

## Limitations

- Four chips are needed. One request at a time, no batching: a second request that arrives while one is running gets `503` with `Retry-After: 5`.
- The decode floor in the can spec (3 tok/s) is a liveness check for the boot test. It is not a performance claim.
- The matrix multiplications (attention and gated delta net projections, the MLP and the output head) run on the chips through `ttnn.linear`. The norms, the convolution, the delta rule recurrence, the attention scores, rotary embeddings and activations run on the host in torch, and every projection moves its input to the chips and its result back. Decode speed is therefore well below the memory-bandwidth limit of the boards.
- The large projection weights are stored as bfloat8_b and the small ones (the output head and the gated delta net gate projections) as bfloat16; accuracy beyond the checks above was not evaluated.
- Greedy decoding was measured. Sampled output was not evaluated. No tool calls, no chat template, no streaming of usage.
- Tested on one machine with two Blackhole p300c boards, tt-metal v0.80.0 (the PyPI `ttnn` 0.80.0 wheel), Python 3.12.
- The server has no sign-in, no TLS and no rate limit. It binds 127.0.0.1 by default; `HOST=0.0.0.0` listens on every interface, which is only for a network you trust.
- The model is licensed for research and educational use under the Allen Institute's responsible-use guidelines (see its card).

## License

The model is licensed under the Apache License 2.0 by the Allen Institute for AI. The code in this bundle is Tenstorrent's, also under the Apache License 2.0 (`LICENSE`, `NOTICE`).

## Provenance

- Weights: allenai/Olmo-Hybrid-7B at revision `4f1cc566f9fdf3ce68da2ab6a788a83d89896dcf`, not part of this bundle.
- `extra_code/` holds the port (`olmo_hybrid_port.py`) and the ASGI front end (`olmo_asgi.py`), both from the tt-orchard repository. `olmo_model.py` at the bundle root is the runner file that tt-model's packager asks for; it re-exports the app.
- `SHA256SUMS` lists the hash of every file.
