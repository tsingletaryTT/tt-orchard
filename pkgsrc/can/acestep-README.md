---
license: apache-2.0
base_model: ACE-Step/Ace-Step1.5
pipeline_tag: text-to-audio
tags:
- tt-model-cache
- blackhole
- tt-dit-server
- thin
- p150
- music
- ace-step
---

# ace-step-1.5-turbo-blackhole

A tt-model v6 thin bundle that serves [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) (revision `19671f406d603126926c1b7e2adc169acbcade22`), the ACE-Step 1.5 turbo text-to-music model, on one Tenstorrent Blackhole chip (mesh P150). It makes a 48 kHz stereo clip from a caption and optional lyrics. The tt-orchard harness staged it with the canning step (`tt-orchard can`). This is a community package and not a release of ACE-Step or Tenstorrent.

## License

The weights this bundle points to are licensed under the MIT License by the ACE-Step authors. The model code it fetches (`sunandbean/acestep-1.5-turbo-p100a`, revision `dabe15cd44f9fb14e948e111fb671d7ab89fda17`) is licensed under the Apache License 2.0 by its publisher. The server and runtime in this bundle are Tenstorrent's, under the Apache License 2.0 (`LICENSE`, `NOTICE`). The bundle ships no weights and no ACE-Step code. The ACE-Step model card says generated music may be used commercially; read it before you rely on that.

## What it runs

- Weights: ACE-Step/Ace-Step1.5 at revision `19671f406d603126926c1b7e2adc169acbcade22`. At start-up the server fetches only the turbo files (`acestep-v15-turbo`, `vae`, `Qwen3-Embedding-0.6B`, `config.json`; about 6.5 GB) into the Hugging Face cache and checks each weight file against its pinned size. The planner language model and the XL transformer are not fetched.
- Model code: the `code/` directory of `sunandbean/acestep-1.5-turbo-p100a` (ACE-Step's Python, the `models/` directory of tt-metal and the chip port `tt_acestep`; 1,307 files). The server downloads it at start-up into its work directory as real files and compares a hash over every file with `extra_code/acestep_pins.json`. A copy made of links is refused: ACE-Step finds its own model code through its resolved path and would run the Hub's copy instead, which gives different music without an error.
- On the chip: the diffusion transformer (24 layers, 8 steps), the Oobleck VAE decoder, the text encoder (Qwen3-Embedding-0.6B), the lyric encoder and the cross-attention keys and values. The three encoder pieces run in bfloat16 and are ports in `extra_code/acestep_chip.py`. On the CPU: the tokeniser, masks, sampler update, peak normalisation and the WAV write.
- ACE-Step rewrites model files inside its checkpoint directory, so the runtime builds a private checkpoint tree under the work directory (small files copied, large weight files linked) and never writes into the Hugging Face cache.
- The ASGI front end (`extra_code/acestep_asgi.py`) imports the request checks from `extra_code/acestep_serve.py`, the standard-library server of the recipe package, so the two share one copy of that code.

## Not included

The planner language model (no thinking mode and no audio-code conditioning), the XL transformer, and more than one chip.

## Intended use

Making music clips of 5 to 480 seconds from a text caption, with or without lyrics, through a local HTTP server started by `tt-model serve`. Out of scope: judging whether the music is good, and any use that needs more than one request at a time.

## How to serve

```bash
tt-model serve <owner>/ace-step-1.5-turbo-blackhole     # installs the bundle into its own venv, then serves it
```

`install.sh` builds a Python 3.12 venv from `requirements.txt`. Every line there is an exact pin, and `ttnn==0.80.0` comes from PyPI. `run.sh` serves the app with uvicorn on 127.0.0.1, port 20140 (`PORT` and `HOST` change them). The first start downloads the model code and the weights, and the first request compiles kernels (see the table). On a shared machine, take a chip lease first. Without `tt-model`: `bash install.sh`, then `bash run.sh`.

```bash
curl -s http://127.0.0.1:20140/v1/music -H 'Content-Type: application/json' \
  -d '{"prompt":"warm lo-fi hip hop, mellow rhodes piano","duration":30,"seed":42}' -o clip.wav
python tools/smoke.py --port 20140 --duration 30 --out clip.wav     # also checks the clip
```

## API

`GET /health` and `POST /v1/music` with a JSON body.

- `prompt` (required, at most 2,000 characters), `lyrics` (optional, at most 8,000), `duration` (5 to 480 seconds, default 30), `seed` (integer, default 42). Other fields get a `400`.
- The reply is a WAV (48 kHz, stereo, 32-bit float) with `X-Timings` (JSON) and `X-Sample-Rate` headers. The same request returns the same samples; the WAV files can differ byte for byte in their headers.
- A second request while one runs gets `503` with `Retry-After: 5` at once. A body over 64 KiB, malformed JSON or a bad field gets a `400`. A failed generation gets a `500` with the error type only. After a failed chip call `/health` and every request return `503` until the server is restarted.
- ACE-Step cannot stop a clip half way. A client that leaves does not end the generation; the slot is free again when the clip is finished.

## Settings

The server reads its settings from the environment, because uvicorn passes no flags:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ACESTEP_WORK` | `ace-step-1.5-turbo-blackhole` under `$XDG_CACHE_HOME` (`run.sh` sets that to `.cache` in the bundle) | a writable directory for the fetched model code, the private checkpoint tree and the clips in flight |
| `ACESTEP_BUNDLE_CODE` | fetched into `ACESTEP_WORK` | the `code/` directory of the model code, if you already have it as real files |
| `ACESTEP_SNAPSHOT` | the pinned revision in the Hugging Face cache | the weights snapshot directory, if you already have it |
| `ACESTEP_FAST` | `all` | `none`, or a comma-separated list of `gc_freeze`, `timbre_cache`, `cond_cache`, `kv_on_chip`, `text_on_chip`, `lyric_on_chip` |
| `HF_HOME` | `.hf` in the bundle | the Hugging Face cache (point it at an existing cache to avoid the 6.5 GB download) |

## Expected performance

Every number is labelled, states its conditions and names its evidence file (in the tt-orchard repository under `docs/run-logs/evidence/acestep-can/`). Hardware: p150 (one Blackhole chip; measured on one chip of a p300c board). Software: `ttnn` 0.80.0 from PyPI, torch 2.7.0 (CPU), numpy 1.26.4, Python 3.12. Date: 2026-10-30. The host was shared with other jobs and busy (load average 19 to 37 over these runs; the recipe package's card measured at 2 to 6). Clip length is 10 s unless stated. A cold kernel cache is a new copy of the bundle with an empty `.cache`; a warm one is the same copy, restarted.

| Number | Value | Label | Conditions | Evidence |
| --- | --- | --- | --- | --- |
| Server ready after start, cold kernel cache, empty work directory (`tt-orchard can boot`) | 306 s | measured, n=1 | load average 37 at the start; includes the download of 1,307 code files and loading the models | `boot-check-cold.json` |
| First 10 s request after that start | 93.7 s | measured, n=1 | same run; an earlier cold run took 216.8 s for the same request (ready in 108 s) | `boot-check-cold.json`, `boot-check-cold-first-try.json` |
| Server ready after start, fresh install through `tt-model`'s installer, cold kernel cache (`tt-model serve`) | 310 s | measured, n=1 | load average 22 | `measurements.txt` section 3 |
| First 10 s request after that start | 121 s | measured, n=1 | same run, load average 17 | `measurements.txt` section 3 |
| Server ready after start, warm kernel cache | 32 s | measured, n=1 | load average 22 | `measurements.txt` section 1 |
| First 10 s request after that start | 5.7 s | measured, n=1 | load average 19 | `measurements.txt` section 1 |
| 10 s clip, repeated request, warm (`run.sh`) | 1.13, 1.49 and 1.18 s | measured, n=3 | load average 19; the first request after the start excluded | `measurements.txt` section 1 |
| 10 s clip, repeated request, warm (`tt-model serve`) | 0.92 and 0.92 s | measured, n=2 | load average 16 | `measurements.txt` section 3 |
| 30 s clip, repeated request, warm (`run.sh`) | 1.76 to 1.96 s (median 1.80) | measured, n=4 | load average 19 | `measurements.txt` section 1 |
| 30 s clip, repeated request, warm (`tt-model serve`) | 1.20, 1.25 and 1.85 s | measured, n=3 | load average 16 | `measurements.txt` section 3 |
| Recipe package, 30 s clip, warm (same runtime code; a source build of ttnn and numpy 2.5.3) | 1.60 to 3.54 s | measured, n=4 | same hour as the first `run.sh` rows, load average 30 to 36; its own card gives 0.93 s at load 2 to 6 | `measurements.txt` section 2 |
| Agreement with the recipe package, same prompt, seed and length | waveform correlation 0.990 (10 s) and 0.983 (30 s) | measured, n=1 each | the two environments differ (PyPI wheel against a source build, numpy 1.26 against 2.5, other package versions); the samples are not identical | `measurements.txt` section 4 |
| Same request through `run.sh` and through `tt-model serve` | identical samples (10 s and 30 s) | measured, n=1 each | same bundle | `measurements.txt` section 4 |

Within one server the same request gives the same samples every time (4 of 4 requests at 10 s, checked sample by sample). A cold first request is dominated by kernel compilation, and a new clip length compiles again. Speed numbers on a loaded host move by a factor of two, so compare them only with numbers from the same hour. The numbers and the commands that produced them are in the tt-orchard repository, `docs/run-logs/2026-10-30-acestep-thin-bundle.md`.

## Limitations

- One request at a time, one chip, no batching. The longest clip measured on the recipe package is 480 s; this bundle was run at 10 s and 30 s only.
- The text encoder, the lyric encoder and the cross-attention projection run on the chip in bfloat16, which adds a conditioning error about the size of a bfloat16 run (relative L2 near 0.02 for the text encoder). Waveform correlation with a float32 CPU run is near 0.92 on the recipe package's first measurement; the per-step checks are in the recipe package's evidence.
- Text and lyric sequences are padded to a multiple of 128 tokens. A prompt or lyric that needs a new length compiles that shape on first use.
- Seeds do not reproduce across backends: a chip clip and a CPU clip of the same seed are different music of the same kind.
- The server has no sign-in, no TLS and no rate limit. It binds 127.0.0.1 by default; `HOST=0.0.0.0` listens on every interface, which is only for a network you trust.
- The server fetches code and weights from the Hugging Face Hub at start-up. A machine with no network needs `ACESTEP_BUNDLE_CODE` and `ACESTEP_SNAPSHOT`.

## Risks and safety considerations

ACE-Step's loader runs Python files that sit in the weights directory (`trust_remote_code`), so only use the pinned revision of the weights and the pinned model code; the server refuses a code directory whose hash differs from the pin. Generated music can resemble material the model was trained on. The checks here measure length and level and do not judge content.

## Not measured

- Whether the music is good, or whether sung words are intelligible.
- Clips longer than 30 s through this bundle, and any machine other than one Blackhole p300c board.
- A download through `tt-model pull` and a boot of the bundle from the Hub. The bundle was installed with tt-model's own installer from a local directory.
- Agreement with the CPU reference under numpy 1.26.4 and the PyPI wheel. The agreement figure above compares two chip builds. It is not a comparison with the CPU.

## Boot check

`tt-orchard can boot` installed a copy of the bundle, started its `run.sh`, asked for a 10 s clip twice and checked the answer: 48,000 Hz, 2 channels, 10.00 s long, 89 percent of 100 ms windows above -40 dBFS, no sample at full scale, the same samples both times, the second request within 5 s, and the start-up line naming the chip steps. All checks passed on 2026-10-30.

## Provenance

- `extra_code/` holds the runtime (`acestep_run.py`, `acestep_fast.py`, `acestep_chip.py`), the request checks (`acestep_serve.py`), the ASGI front end (`acestep_asgi.py`), the hash check (`check_bundle.py`) and the pins (`acestep_pins.json`), all from the tt-orchard repository. `acestep_model.py` at the bundle root is the runner file that tt-model's packager asks for; it re-exports the app.
- `SHA256SUMS` lists the hash of every file.
