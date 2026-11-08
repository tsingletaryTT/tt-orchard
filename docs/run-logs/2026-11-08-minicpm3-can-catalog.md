# MiniCPM3-4B thin bundle: catalog-measured benchmark and publication (2026-11-08)

Operator's instruction (2026-11-08): benchmark the bundle so it can be published for tt-model-manager and Hugging Face, with an explicit go-ahead to publish this model.

The can itself came from `docs/run-logs/2026-10-30-minicpm3-thin-bundle.md` (spec `config/can.minicpm3.toml`, ASGI adapter `pkgsrc/server/minicpm3_asgi.py`, card `pkgsrc/can/minicpm3-README.md`). Its card lacked the prefill lengths that matter for use, a measurement under the pinned PyPI environment, and the three statements the `catalog-measured` card profile asks for. This run added them. The spec now sets `card_profile = "catalog-measured"`.

## What was run

- Stage and gate on the current main: `tt-orchard can stage` wrote 13 files, `can gate` passed (before and after the card change).
- Cold boot check (`can boot --cache-state empty`, work directory and Hugging Face cache both new, pinned weights downloaded from the Hub revision `d6b14ddaefdb11c624dd75c3c779549bc90b08cb`): all 9 checks passed. Ready after 126 s (this start includes the one-time conversion of the weights); the reported decode rate was 33.64 tok/s against a floor of 25.
- Benchmark client `evidence/2026-11-08-minicpm3-can/bench.py` against the installed copy's `run.sh` on one chip of a p300c board, under a gozer lease. Decode: 5 prompts, 1 discarded warm-up and 3 measured runs each. Prefill: prompts of 522, 1,910 and 8,190 tokens (counted with the model's tokenizer), one output token, 1 warm-up and 3 measured runs. The 8,190-token run used `MINICPM3_MAX_LEN=8704`.
- Agreement with the float32 CPU reference under the bundle's own environment (`ttnn==0.80.0` from PyPI, numpy 1.26.4): `run_chip.py agree --opt 3 --upd paged` on the same prompt as the earlier runs, summarised with `summarize.py`.

## Results

See the card (`pkgsrc/can/minicpm3-README.md`, Expected performance) and the evidence folder `docs/run-logs/evidence/2026-11-08-minicpm3-can/`. In short: decode 33.27, 33.50 and 33.64 tok/s (minimum, median, maximum of 12 runs at context 2048; 32.23, 33.60 and 33.74 at context 8704); prefill 15.8 s for 522 tokens, 59.6 s for 1,910 tokens and 268.5 s for 8,202 tokens (30.5 to 33 tok/s); 16 of 16 teacher-forced tokens agree with the float32 reference with 0 flips under the PyPI pins.

## Publication

The staged bundle (18 files; `SHA256SUMS` verified before the upload) was uploaded to the operator's existing repository `episod/minicpm3-4b-blackhole` (Hub commit `0782529d`; a second commit, `fc61a041`, updated the card status line after the Hub check). `tt-model publish episod/minicpm3-4b-blackhole` was run afterwards, because the card upload had dropped the `tt-model-catalog` tag; the tag is back. The repository already held files from the 2026-10-30 recipe package (`LICENSE-runtime`, `docs/running.md`, `evidence/context-bench.json`, `evidence/text_*.json`); they were left in place. The publish step rewrites the card's front matter, so `README.md` no longer matches its line in `SHA256SUMS`.

Final check from the Hub: `tt-model pull episod/minicpm3-4b-blackhole` built the virtual environment, and `tt-model serve episod/minicpm3-4b-blackhole --refresh` served it on one chip (under tt-model the server listens on port 20000). The answer was "Paris". Decode over the same four prompts, 3 runs each, was 33.03, 33.21 and 33.63 tok/s (minimum, median, maximum; host load average 4.4 to 7.1); the record is `bench-from-hub.json`. The warm-cache boot check of the final staged bundle (`can boot`, 2026-11-08) passed all 9 checks at 33.61 tok/s (`boot-check-warm.json`). In short: decode 33.27, 33.50 and 33.64 tok/s (minimum, median, maximum of 12 runs at context 2048; 32.23, 33.60 and 33.74 at context 8704); prefill 15.8 s for 522 tokens, 59.6 s for 1,910 tokens and 268.5 s for 8,202 tokens (30.5 to 33 tok/s); 16 of 16 teacher-forced tokens agree with the float32 reference with 0 flips under the PyPI pins.



## Things to know

- The `capital` prompt of the benchmark answers in 2 tokens, so its decode rate is not a meaningful sample of decode speed; the card's decode figures leave it out and use the other four prompts.
- Prefill runs at about 32 to 33 tok/s because the prompt goes through the decode step one token at a time. The batched prefill in `scripts/minicpm3/chip_model.py` is switched off in the server (`serve.STATE["batched_prefill"] = False`) because it hung the chip on the second request after a trace was captured (2026-10-30). That hang was not investigated here.
- Perplexity was not re-measured under the PyPI pins; the card repeats the earlier figure and says so.
- A concurrency or consistency benchmark does not apply: the server accepts one request at a time and answers a second with 503.
