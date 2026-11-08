# Olmo-Hybrid-7B: a v6 thin bundle, measured and published (2026-11-08)

The full-port bring-up of allenai/Olmo-Hybrid-7B (run `qwen-like-olmo-hybrid-7b`, stages 0 to 6 passed, stage 7 skipped because a full port builds no package)
was packaged by hand with `tt-orchard can`. The bundle is published at `episod/olmo-hybrid-7b-blackhole` on the Hugging Face Hub and listed in the tt-model
community catalog. The operator gave an explicit yes to publish this model. The coordinator set two conditions on the card, and both are met (below).

## What is in the bundle

`config/can.olmo-hybrid-7b.toml` (spec), `pkgsrc/server/olmo_asgi.py` (ASGI adapter: `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`),
`pkgsrc/runtime/olmo_hybrid_port.py` (the stage 3 port, a 1x4 mesh), `pkgsrc/can/olmo-README.md` (the card, written by `scripts/olmo_hybrid/make_card.py` from the
records below), `olmo-NOTICE`, `olmo_model.py`. The weights stay on the Hub (revision `4f1cc566f9fdf3ce68da2ab6a788a83d89896dcf`). Pins: `ttnn==0.80.0` (PyPI) with `numpy==1.26.4`.

## Measurements (through the server, 4 Blackhole chips, greedy, one request at a time)

Evidence: `docs/run-logs/evidence/olmo-hybrid-7b/`. The host shared its CPU with other jobs (load average 3.6 to 9.9), and the port runs part of its work on the host.

| Measure | Warm kernel cache | Empty kernel cache at start |
| --- | --- | --- |
| Served decode, batch 1 (15 runs: 5 prompts, 3 runs, 64 tokens) | median 7.28, min 3.59, max 9.30 tok/s | median 9.68, min 7.96, max 11.47 tok/s |
| Prefill 141 tokens | 1.08 s | 1.04 s |
| Prefill 561 tokens | 4.62 s | 7.53 s |
| Prefill 2,261 tokens | 22.4 s | 24.9 s |
| Prefill 9,101 tokens | 137.7 s | 141.4 s |
| Server ready | 30 s | 40 s (first request 6.74 s) |
| Two requests at once | 200 and 503 | 200 and 503 |

Boot checks of the bundle reported 7.6, 5.5, 9.3, 2.2, 11.6 and 8.3 tok/s for one 64-token request each. The harness's in-process decode benchmark (stage 6) gave 13.9 tok/s (host load not recorded).
The spread of the served rate is the finding: the same bundle ran between 2.2 and 11.6 tok/s depending on host load. The can spec's decode floor is 1.0 tok/s and is a liveness check only.
It was first set to 8.0, then 6.0 and 3.0 as boot checks came in under it; the coordinator asked that it stay a liveness floor, which is what it is now.

## The divergence on "The capital of France is"

Against the transformers CPU reference (bf16, greedy, 32 new tokens), three of four prompts matched all 32 tokens. "The capital of France is" matched its first token and then differed.
Teacher-forced on the reference's own continuation, the reference's full-forward logits at token index 1 rank "." (9.3125) above "," (9.25), a gap of 0.0625 (one bf16 step). The port, given
the same prefix on the chips, ranks them the same way ("." 9.25, "," 9.1875, gap 0.0625). The reference's own greedy generation (incremental decoding) chose ",". The served port gave "." on six
repeats. So this is a bf16 near tie that the reference itself resolves differently in its two paths. Records: `diag.json`, scripts `ref_logits.py`, `chip_logits.py`, `analyze_divergence.py`.
The card says this and lists each prompt's result. It also says the stage 6 agreement (top-1 1.0 over 32 tokens) covers one prompt.

## Publishing

`hf upload --private` of the staged bundle (11 files, no weights), `tt-model serve episod/olmo-hybrid-7b-blackhole` from the private repository (ready in 30 s on its own port, one completion
and one chat request answered), then the repository was made public and `tt-model publish` listed it. The `tt-model-catalog` tag is present on the Hub card.

## Things that went wrong on the way

- The first two attempts to start the server on the 1x4 mesh failed (a Fabric Router Sync timeout once; a failed systemd unit name left over the second time). The pipeline now resets the lease's chips before the first start and retries once after a reset.
- A health check of `tt-model serve` waited on the bundle's own port (20120) while `tt-model serve` listens on 20000, so one run held a four-chip lease for 90 minutes before it was stopped. The check now uses 20000.
- The boot check's decode floor and the card were changed several times as numbers came in; the card's headline is now the served rate with its spread.

## Republish after the review of PR 633 (2026-11-08)

Copilot found four problems in the published adapter and card; each was fixed test-first (`tests/test_olmo_asgi.py`, a mutation per guard went red):
the request parser now refuses fields it cannot honour (`n` other than 1, `tools`, `logprobs`, `response_format`, wrong types for `stream` and `stop`, unknown fields) with a 400;
text that could be the start of a stop string waits for the next token and is flushed if the string never completes; a non-streaming request watches for a disconnect and frees the single slot;
and the card writes the near-tie explanation only when the records back it (same two tokens in the same order, and the reference's generation took the second one), otherwise it says the difference is not explained.
The changed files were uploaded to `episod/olmo-hybrid-7b-blackhole`, `tt-model publish` was run again, and the catalog tag was checked.
`tt-model publish` edits the card on the Hub (it adds the tag), so the README there no longer matches the hash the staged `SHA256SUMS` holds. `SHA256SUMS` was therefore regenerated from the files as published and uploaded last; a fresh download verifies against it (10 of 10).
The re-check on hardware was not made: a four-chip mesh was not free (another bring-up held two chips). The adapter changes are in the request and generation layers, covered by the fakes; a boot check on four chips is still owed.
