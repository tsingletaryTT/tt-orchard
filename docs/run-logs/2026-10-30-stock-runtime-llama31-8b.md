# 2026-10-30: Llama-3.1-8B-Instruct through tt-transformers on one Blackhole board (issues 93 and 62)

Prompt (operator): start #93 together with #62 on Llama-3.1-8B-Instruct (fallback Qwen2.5-7B-Instruct), measure on hardware, test first.

This is the first real chip result for the `stock-runtime` class. Stage 2's template (`stock_check.py`) ran by hand under a gozer lease on board 1 (chips 2 and 3),
outside the supervisor. The fallback model was not needed: the account's token reads `meta-llama/Llama-3.1-8B-Instruct` and the snapshot was already in the cache.
Nothing was downloaded and nothing was published.

## Result

| Run | Board name | Teacher-forced tokens | Top-1 | Top-5 | Decode tok/s/user (free run) | First token |
|---|---|---:|---:|---:|---:|---:|
| 1 chip | P150 | 500 | 97.2 percent | 100.0 percent | 45.46 | 53 ms |
| 2 chips | P300 | 500 | 96.6 percent | 100.0 percent | 55.86 | 37 ms |

Both records pass `gate_stock_runtime` (the bars are 85 and 95 percent). The reference is a CPU float run of the model's own snapshot, made by tt-transformers'
`generate_reference_hf.py`. The runtime is the host tt-metal tree at v0.80.0 with accuracy optimizations (bfp8 weights, bf16 activations). Wall time of the hardware
test: 110 s for one chip, 100 s for two (weights converted from the safetensors each time; see the third finding). The records are in
`2026-10-30-stock-check-llama31-8b-p150.json` and `2026-10-30-stock-check-llama31-8b-p300.json`.

Stage 0 on the real snapshot gives `stock-runtime`: 8.03 billion parameters, the config equal to `model_params/Llama-3.1-8B-Instruct` on every decisive key, and every
tensor in the layout tt-transformers loads.

The two speeds come from different tests. The 1-chip speed is the batch-1 free run; the accuracy runs gave 43.6 (one chip) and 41.2 (two chips) tokens per second
per user while teacher forcing. Only one text and one batch size were measured. A 4-chip run needs both boards, and chips 0 and 1 belong to the coder lease.

## What the first run found

Four things stopped the run, in this order. Each has a test that failed first.

1. `accelerate` was missing from the demo venv. tt-transformers' reference script loads the model with `device_map="auto"`, and transformers refuses that without it. The CPU
   model load had already run for minutes before the error. `accelerate==1.15.0` is now pinned, and the template's default import check names it, so a missing package
   shows up in seconds.
2. `torchvision` (from the same CPU index as torch) and `pytz` were missing. The host tree's root `conftest.py` imports the first and the benchmarking helper imports the
   second. Both are pinned now. The reviewed-names test lists them.
3. The scratch copy lacked `tests/tests_common`. The mesh fixture imports it after the board opens. It is in the copy list.
4. ttnn 0.80.0 on this box never returns from `load_tensor_flatbuffer(..., device=mesh)` for a cached tensor of about 35 MB or more. The kernel time of the process grew
   while the wall clock ran; `py-spy` showed the main thread inside the call. tt-transformers reaches it on the second `ttnn.as_tensor` of the K and V caches, which
   share one cache file name. A direct probe showed writes from memory are fast (33 MB in 0.07 s) and loads of the cache files hang (a 17 MB file loaded; the 35 MB KV
   file and the 1 GB embedding did not). The template now has a config key `disable_tensor_cache`, which the draft sets. It appends a five-line patch to the scratch copy's
   `conftest.py` that drops `cache_file_name` from every `ttnn.as_tensor` call. The numbers do not change. `evidence/scratch.patch` used to cover only `models/`, so it
   missed this change; it now covers every copied path.

Two smaller things. `gozer wait` after a queued `gozer acquire --owner-pid` granted the lease under the waiting process's own pid, so the lease went stale
when `gozer wait` exited; the second lease was taken directly with `--owner-pid` once the board was free. And on P300 the tt-transformers test fails after printing
the accuracy, because its table of published targets has no row for Llama-3.1-8B on P300. The template now records that (`tt_transformers_target`) and does not fail the run;
any other failure after the accuracy line still fails it.

## What was built

- `orchard/skills/operator-bundle-templates/build_bundle.py`: the "Stock runtime on a chip" section of RESULTS.md (accuracy, speed, board name, reference source and
  snapshot, tt-metal tag and tree state, the scratch patch as a line count) and a "Stock runtime run" section of RISKS.md. The part of the gap in #93 that asked for a
  stage 8 section is closed.
- The template and venv changes above.

## Not done

- Stage 4 for the stock-runtime path is not wired into the supervisor. The hand run shows the template works with a 2-chip config. Wiring it means a stage 4 spec with the
  same gate, a chip-count loop and the handoff; it is listed in #93.
- No 4-chip run.
- #62: the staged `package-thin` bundle was not installed, import-smoked or booted, and there is no base-bundle command. The text route does not apply to this model: it
  serves models whose code is a set of scripts in this repository, and tt-transformers is the runtime here.
- Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3 and the other `model_params` entries were not run. Llama-3.1-8B-Instruct is one model on one text.
