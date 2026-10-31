# minimax-music3-fiona-crapple as a v6 thin bundle (2026-10-30)

Spec `config/can.fiona.toml`; code `pkgsrc/music3/`; pattern `docs/canning.md`. Weights `episod/minimax-music3-fiona-crapple` at `53fc7557ea3101d3efbe5e50598bd0bea7f44344` (the four rewritten shards matched the sha256 values in that repository's card). The staged bundle has 74 files and 1.6 MB, with no weights, venv or cache. The gate passed. Every run used a copy of the staged directory or tt-model's own install; the staged directory was never installed into or served.

Conditions: one chip of a p300c board (board 1), `ttnn` 0.80.0 from PyPI, torch 2.11.0 CPU, weights on local disk (the download is not timed), a host shared with three other agents. Each `gozer` lease released and reset cleanly.

| Run | Result |
| --- | --- |
| `can boot` on a copy (host load average about 7) | 11 of 11 checks passed. Ready in 195 s from empty caches (warmup 108.8 s). 5 s clip in 29.6 s: 32000 Hz, 4.99 s, peak 0.357, 82% of windows above -40 dBFS. |
| tt-model installer (`_install_self_contained`, scratch directory and index) then `tt-model serve local/minimax-music3-fiona-crapple-blackhole --local-only --port 20151` (load average 20 to 30) | Ready in 513 s from empty caches (warmup 285 s). `/ui` 200, `/` 307 to `/ui`, `/v1/models` names the merged weights. |
| 20 s, seed 42, first request | 100.2 s wall, 95.9 s compute, 4.79 s per second of audio. llm 31.2, depth 20.7, denoise 36.3, vocoder 29.0, VAE decode 7.0 s. |
| 10 s, seed 11, twice | 56.5 s then 42.3 s compute. The two files have the same sha256. |
| container package, 20 s, seed 42, 2026-10-15 | 64.8 s compute, 3.24 s per second of audio (host load not recorded). |

Same request through the bundle and the container: both 20.02 s at 32 kHz with peak 1.0; the samples differ (correlation 0.017 over the whole clip); rms 0.120 against 0.146 and 66% against 84% of 100 ms windows above 0.01 rms. Different samples are expected from an autoregressive sampler when the ttnn build differs. Whether the sound differs in a way a person hears was not listened to.

The first `tt-model serve` attempt under `gozer run` was queued (exit 10); the ticket was granted later with `gozer wait`, and the server was started under that lease.

Differences from the recipe (container) package: the venv is built from exact pins with PyPI `ttnn` (the container ran a tt-metal source build); caches live inside the bundle directory; `run.sh` does not pass `--timeout-keep-alive 75`; the page is mounted on the model server at `/ui` and runs in its process (the companion repository runs it as a separate process); `HF_MODEL` is required.

Open: speed on an idle host; a listening test; the cause of the slower compute (load or the ttnn build). Evidence: `docs/run-logs/evidence/fiona-can/`.
