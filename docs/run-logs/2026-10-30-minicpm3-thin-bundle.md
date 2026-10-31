# MiniCPM3-4B as a v6 thin bundle: a can after Granite (2026-10-30)

Goal (operator): "v6 thin bundles for all these ... this is like the canning step." The first can was Granite (`docs/run-logs/2026-10-30-granite-thin-bundle.md`). This run cans MiniCPM3-4B by the same recipe. Code: `config/can.minicpm3.toml`, `pkgsrc/server/minicpm3_asgi.py`, `pkgsrc/can/minicpm3-*`, small changes to `scripts/minicpm3/serve.py` and `convert_weights.py`. Section for this model: `docs/canning.md`.

## What was built

- `pkgsrc/server/minicpm3_asgi.py`: the ASGI front end. It imports `parse_request`, `encode_prompt`, `generate_ids`, the lock and the state from `scripts/minicpm3/serve.py` (shipped as `extra_code/minicpm3_serve.py`).
- `serve.py` changes (the stdlib server keeps working; its 20 existing tests pass): the request checks and prompt encoding are functions now; `generate_ids` takes a `should_stop` callable checked before every chip step (the prompt goes through the chip one token per step, so a client that leaves during a long prompt would otherwise hold the slot); `import ttnn` moved into `main()`; the single-chip model construction is `build_single_chip`. A body that is a JSON list or a `max_tokens` that is a list used to raise an uncaught TypeError in the handler; both are 400 now.
- `convert_weights.convert()`: the one-time pickle to safetensors conversion as a function. The adapter calls it on the first start (the repository ships only `pytorch_model.bin`, 8 GB), into `tt-orchard-minicpm3` inside `HF_HOME`.
- The adapter adds what the stdlib server lacked: `/health` (200 only after the model loaded and the trace was captured), `timings` in the answer (the boot check reads `decode_tok_s`), a disconnect watcher for streaming and non-streaming requests, and an unusable latch (a failure in a chip step makes `/health` and later requests answer 503 until restart; the recipe's own boot check found that a chip that timed out once failed every later request).
- The bundle (13 files, no weights, no venv): manifest schema 6, kind `tt-dit-server`, app `minicpm3_asgi:app`, weights `openbmb/MiniCPM3-4B` at `d6b14dda` (the revision the published card pins; the `config.json` hash `cf1d08cb` matches the published package record), mesh P150, 1 device, port 20130, 20 exact pins (Granite's list plus `transformers==5.19.0`).

## Tests and mutations

`tests/test_minicpm3_asgi.py` (53 tests, written first and seen failing when the module did not exist) and the 20 older MiniCPM3 script tests. 30 mutations, each removing one guard (lock release in both workers, the non-blocking acquire, the health and chat unusable checks, the second look under the lock, timings, the three disconnect paths, the prompt-room check, empty messages, `max_tokens` clamping, body size, the conversion cache and its per-directory key, setting validation, the failed-start report, the shutdown close, a top-level `import ttnn` in serve.py, the revision pin, the download revision, batched prefill on). In the second pass 29 turned a test red (one of them by hanging, which the script counts as red). The survivor is "no first look at unusable": the second look under the lock gives the same 503, so the first look only avoids taking the lock. The first pass had four more survivors (a body that is not an object, a TypeError from `max_tokens`, the body-size check, and a tokenizer error that is not a ValueError); each got a test and the second pass turned them red.

## Hardware (board 1, chips 2 and 3, via gozer; host shared with three other agents, load average 15 to 24)

1. `tt-orchard can boot` on a copy of the staged bundle (own venv from the pins, `ttnn==0.80.0` from PyPI): passed all 9 checks. Ready after 66.0 s including the one-time conversion of the weights (the pickle was already downloaded); answer "Paris"; decode 33.2 tok/s (floor 25); stream ended with `[DONE]`; log held `"traced": true` and `"context": 2048`. The PyPI wheel ran `flash_multi_latent_attention_decode` and the rest of the port without a change to the model code.
2. The same bundle installed with tt-model's own installer (`tt_kernel.cli._install_self_contained` into a scratch directory) and served with its `run.sh` through `tt-model`'s serve function, twice (empty kernel cache, then warm), then the recipe package's own `run.sh` (source build of ttnn, numpy 2.5.3). Same request series each time (`docs/run-logs/evidence/2026-10-30-minicpm3-thin/`):

| | bundle, empty kernel cache | bundle, warm cache | recipe package |
| --- | --- | --- | --- |
| ready after start | 61.0 s | 73.0 s | 340.9 s |
| first request (short answer) | 0.71 s | 0.70 s | 0.80 s |
| time to first token, 21-token prompt | 0.62 s | 0.62 s | 0.63 s (streamed) |
| decode, 64 tokens (3 runs, server-reported) | 33.21, 33.30, 33.31 tok/s | 33.24, 33.74, 33.55 | not reported by the server |
| decode, 64 tokens streamed (2 runs, measured by the client) | 33.31, 33.31 | 33.54, 33.34 | 32.72, 32.83 |
| 494-token prompt, 1 token out | 15.5, 15.6 s | 15.1, 15.5 s | 15.1, 15.1 s |
| empty `messages` | 400 | 400 | 400 |
| first 120 characters of the 64-token story | identical in all three | | |

Differences listed: the bundle is ASGI under uvicorn (settings are `MINICPM3_*` variables; no flags); its environment is built from exact pins (numpy 1.26.4, PyPI ttnn); it has `/health` and `timings`; a disconnect stops the chip before the next step; a chip failure latches 503; the recipe server refuses ports 8000, 8011, 8780 and 8100 to 8103 in `main()`, and the ASGI path cannot refuse a port because uvicorn binds first (the spec refuses those ports at load, default 20130); the two-chip variant and `--batched-prefill` are not in the bundle.

Speed is within noise of the recipe package (33.3 to 33.5 against 32.7 to 32.8 tok/s measured the same way; the recipe card gives 33.87 at a load average of 3.6 and 31.6 at 55). The prompt path is the same: 494 tokens take about 15 s in all three, about 32 tok/s, because the prompt goes through the chip one token per step.

## Not done, and things to know

- The recipe package took 340.9 s to be ready in this run, against 61 to 73 s for the bundle. The cause was not investigated (a different ttnn build with a cold cache of its own is one possibility).
- A warm kernel cache did not shorten the start (73.0 s against 61.0 s, with host load 18 to 20 against 15 to 22). The start is dominated by loading 62 layers and capturing the trace.
- The download time of the weights was not measured (they were already in the cache). The download, the conversion and the trace capture together can exceed a short health timeout in `tt-model serve`; the card says so.
- Agreement with the CPU reference under `ttnn==0.80.0` (PyPI) and numpy 1.26.4 was not re-measured. The recipe's numbers came from a source build and numpy 2.5.
- Leases: the boot check ran under `gozer run --exact 2`. For the later runs I took the lease with `gozer wait` and ran the scripts with the granted `TT_VISIBLE_DEVICES`; gozer reaped that detached lease while the runs were finishing, so `gozer release` found nothing to release. The measurement script's first version left the tt-model server running after a run (tt-model starts it in its own session); the next run on the same port refused to start, and the processes were stopped by exact pid. The final script stops every descendant.
- No upload, no `tt-model push` or `publish`, no merge. The publish text is `tt-orchard can commands config/can.minicpm3.toml --namespace ORG`.
- The staged card was graded with `cardview --no-infer` from tt-model-ranker (local, read only): window grade B on p150 (chat B+ for speed; the card has no consistency or concurrency numbers, so B is the cap). The coding-agent tile is outside the focus. The rows are in the viewer's two layouts (the `Number | Value | Label | Evidence` table and the `profile | ISL | OSL | concurrency | N | TTFT | prefill | decode | E2EL` table).

## Review of PR 308: fixes, and a rerun under a fresh lease (2026-10-31)

- Copilot found two defects in the adapter; each got a failing test first. (1) `MINICPM3_MAX_LEN=129` advertised a 256-token context while the model was built with 129, so its rotary tables could be shorter than the accepted prompts. `settings()` now rounds up to a multiple of 128 once, and the model and the advertised context use that value. (2) The non-streaming answer read the token counts and the finish reason from shared state after the slot was released, so a request that took the slot next could write its values into this answer. They are now read before the release.
- `gozer history` showed that the first lease of the measurement session (`93d4f6`, taken with `gozer wait`, detached) was reaped at 00:25:07Z. The first A run started inside it. The first B and C runs were started after the reap or after a stray server was stopped, so they cannot be shown to have run under a live lease. Those files are kept in `docs/run-logs/evidence/2026-10-30-minicpm3-thin/first-pass/`.
- The final staged bundle was installed again (own venv, and tt-model's installer into a fresh directory, so the kernel cache was empty again) and run under one `gozer run --exact 2` lease (`3b7dba`, not detached) after merging `origin/main`:

| | can boot | bundle via tt-model, empty kernel cache | bundle, warm cache | recipe package |
| --- | --- | --- | --- | --- |
| ready after start | 75.0 s | 68.1 s | 50.1 s | 25.0 s |
| first request (short answer) | | 0.73 s | 0.72 s | 0.70 s |
| time to first token, 21-token prompt | 0.64 s | 0.63, 0.64, 0.63 s | 0.63, 0.63, 0.64 s | 0.64 s (streamed) |
| decode, 64 tokens, server-reported | 33.35 and 31.54 tok/s | 32.72, 33.35, 32.57 | 32.67, 32.65, 32.67 | not reported |
| decode, 64 tokens streamed, client-measured | | 32.07, 32.62 | 32.80, 32.32 | 32.89, 32.87 |
| 494-token prompt, 1 token out | | 15.4, 15.6 s | 15.1, 15.2 s | 15.2, 16.1 s |
| host load average (start, end) | about 20 | 24, 21 | 21, 16 | 16, 22 |

- All 9 boot checks passed. The earlier 340.9 s start of the recipe package was a cold kernel cache for its own build under heavy load: in this run, with that cache warm, it was ready after 25.0 s. The bundle's start is 50 to 68 s with or without a kernel cache, because loading 62 layers and capturing the trace dominate.
- The speed of the bundle and the recipe package differ by less than the spread between runs (32.3 to 33.4 tok/s against 32.9). The earlier session measured 33.2 to 33.7 for the bundle at a similar load, and the recipe card measured 33.87 at a load average of 3.6. The host was shared with three other agents in both sessions.
- Card wording was compared with the live `episod/audio8-asr-infinite-p150` README (read only): the card now has the same tag set style (`tt-dit-server`, `tt-model-cache`), a "Runs on **p150** (mesh `P150`)" line, an "At a glance" table (Architecture, Hardware, License, Status), "Direct use" and "Out-of-scope use" in Intended use (the out-of-scope items are a bullet list, which the ranker reads as things the model is not for), a "Quickstart" and a "Using it" section. `cardview --no-infer` still gives window B on p150.
