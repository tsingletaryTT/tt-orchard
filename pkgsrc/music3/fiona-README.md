---
tags:
- blackhole
- p150
- tt-dit-server
- tt-model
- text-to-audio
- music-generation
- gradio
license: other
license_name: minimax-music3-community-license
license_link: https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE
pipeline_tag: text-to-audio
base_model:
- MiniMaxAI/MiniMax-Music3
---

# minimax-music3-fiona-crapple-blackhole

> Derived from [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) by MiniMax, with the language-model adapter [terminusresearch/minimax-music3-lm-lora-fiona-crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple) (`regularised/checkpoint-750`) merged into the weights at strength 1.0. See the original model cards for training and evaluation details. The weights are in [episod/minimax-music3-fiona-crapple](https://huggingface.co/episod/minimax-music3-fiona-crapple) (revision `53fc7557ea3101d3efbe5e50598bd0bea7f44344`) and are not in this repository. This is a community package and not a release of MiniMax or Tenstorrent.

MiniMax Music 3 turns lyrics and a text description of the music into a song: an 8B Qwen3-based language model, a 0.6B depth decoder for the residual RVQ codebooks, a 2.4B flow-matching DiT and a vocoder. The adapter was trained on nine tracks (about 52 minutes) in one style and answers to the trigger phrase `fiona crapple`. This package serves it on one Blackhole chip, makes one song at a time and returns 32 kHz stereo audio. It includes a Gradio page.

This is a v6 thin bundle for `tt-model`: the model code and server, and a venv built from exact pins. It needs no Docker image. It was built by the canning step of tt-orchard (`tt-orchard can`).

## At a glance

| | |
| --- | --- |
| Architecture | Language model (8B, bfp8), depth decoder (bfp8) and DiT (bf16) run on the chip; condition encoder and vocoder run on the host CPU. |
| Hardware | p150 (one chip). Measured on one chip of a p300c board. |
| License | minimax-music3-community-license |
| Status | Beta. One host, one chip, `ttnn` 0.80.0 from PyPI. Speed under load is slower than the container package and has not been re-measured on an idle host (see Expected performance). No quality evaluation. |
| Port | 20150 (`PORT` changes it). Binds 127.0.0.1 (`HOST` changes it). |
| Disk | about 27 GB for the weights, 1.6 GB for the venv, and about 22 GB of caches after the first start (8.4 GB tensor cache, 12 GB weight views and caches, 1.1 GB compiled kernels) |

## Intended use

Generating songs from lyrics and a text description of the music. Putting `fiona crapple` in the description steers the output toward the adapter's training style, which its author describes as an intimate cafe singer-songwriter sound with jazzy piano, upright bass, brushed drums and a raw female vocal.

## Quickstart

```bash
tt-model serve <owner>/minimax-music3-fiona-crapple-blackhole     # installs the bundle into its own venv, then serves it
```

`install.sh` builds a Python 3.12 venv from `requirements.txt`. Every line there is an exact pin, and `ttnn==0.80.0` comes from PyPI. `run.sh` serves the app with uvicorn. The first start downloads the pinned weights (about 27 GB) into the Hugging Face cache, builds the weight and tensor caches and compiles kernels; see Expected performance for how long that took. On a shared machine, take a chip lease first. Without `tt-model`: run `bash install.sh`, then `bash run.sh`.

The Gradio page is at `http://127.0.0.1:20150/ui`. It has no login.

## Using it

The model's own HTTP API (it has no OpenAI chat route):

- `GET /v1/health` (200 when the model is loaded), `GET /v1/models`.
- `POST /v1/audio/speech`: one request, one clip, synchronous. `input` is the lyrics, `instructions` the description, `response_format` is `wav`, `flac`, `mp3` or `pcm`, `sample_rate` is 32000 or 44100, `seed`, and `max_new_tokens` (audio frames, 25 per second) or `audio_duration` (seconds).
- `POST /v1/music/jobs`, then `GET /v1/music/jobs/{id}` (status, `frames_done`, stage times when done) and `GET /v1/music/jobs/{id}/audio`, for songs that take minutes. `DELETE /v1/music/jobs/{id}` cancels.

```bash
curl -s localhost:20150/v1/music/jobs -H 'Content-Type: application/json' -d '{
  "input": "[Verse 1]\nThe kettle sings at half past three\n[Chorus]\nStay a little longer",
  "instructions": "fiona crapple, jazzy piano, raw female vocal, upright bass, brushed drums, intimate",
  "audio_duration": 20, "seed": 11, "response_format": "flac"}'
```

The server keeps a queue of 8 jobs and answers `429` when it is full. A song ends when its lyrics are finished, which can be before the length asked for. The same seed, lyrics and description gave the same file when repeated on one server (one pair, below).

The settings are environment variables, because uvicorn passes no flags. `run.sh` sets `HF_MODEL`, `TT_MODEL_WEIGHTS_REVISION`, `HOST`, `PORT` and the values in the manifest's `env`.

| variable | meaning |
| --- | --- |
| `HF_MODEL` | the weights: a repository id (downloaded at `TT_MODEL_WEIGHTS_REVISION`) or a local directory. Required: the app refuses to start without it, because the model code would otherwise load the base MiniMax-Music3 weights. |
| `MUSIC3_SNAPSHOT` | a local directory with the weights; takes precedence |
| `MUSIC3_UI` | `0` turns the page off (default `1`) |
| `MUSIC_URL` | the server the page talks to (default this server) |
| `MUSIC3_MAX_SEQ_LEN`, `MUSIC3_MAX_FRAMES`, `MUSIC3_MAX_QUEUE`, `MUSIC3_WARMUP`, `MUSIC3_WARMUP_FRAMES`, `MUSIC3_TRACE_REGION_MB` | the model server's own settings (manifest defaults 16384, 9000, 8, 1, 200, 256) |
| `MUSIC3_API_KEY` | the model server's bearer key. The page does not send it, so the page stops working when it is set. |
| `TT_DIT_CACHE_DIR`, `TT_CACHE_PATH`, `TT_METAL_CACHE`, `HF_HOME` | where the weight views, the tensor cache, the kernel cache and the weights go (default: inside the bundle's directory) |

## Expected performance

Conditions for every number in this section: 2026-10-30; one chip of a p300c board (profile `p150`); `ttnn` 0.80.0 from PyPI, Python 3.12, torch 2.11.0 (CPU); the weights already on local disk (the 27 GB download is not timed); a shared host that other jobs were using, with the load average given for each row. Evidence files are in the tt-orchard repository under `docs/run-logs/evidence/fiona-can/`.

| Number | Value | Label | Evidence |
| --- | --- | --- | --- |
| Cold start to `/v1/health` 200, empty kernel and tensor caches, installed by `can boot`'s copy (host load average about 7 at the end) | 195 s (warmup of 200 frames: 109 s of it) | measured | `boot-check.json` |
| Cold start, empty caches, installed and served by `tt-model` (host load average 20 to 30) | 513 s (warmup: 285 s) | measured | `tt-model-serve-start.txt` |
| First request after the cold start: 5 s clip, seed 11 (`can boot`) | 29.6 s | measured | `boot-check.json` |
| First request after the cold start: 20 s clip, seed 42 (`tt-model`, load average 30) | 100 s wall, 95.9 s of compute: 4.79 s per second of audio (language model 31.2, depth decoder 20.7, diffusion 36.3, vocoder 29.0, VAE decode 7.0) | measured | `clip-seed42-20s.json` |
| Warm requests: 10 s clip, seed 11, run 1 then run 2 (`tt-model`, load average 22 to 24) | 56.5 s then 42.3 s of compute: 5.65 then 4.23 s per second of audio | measured | `clip-seed11-10s-run1.json`, `clip-seed11-10s-run2.json` |
| Same request, container package, same weights, 2026-10-15 (host load not recorded) | 64.8 s of compute for 20 s: 3.24 s per second of audio | measured | `recipe-container-seed42-20s.json` |
| Boot check clip: sample rate, length, level | 32000 Hz, 4.99 s of 5 s asked, peak 0.357, 82% of 100 ms windows above -40 dBFS | measured | `boot-check.json` |
| Repeat of one request on one server | bit-identical files (sha256 equal) | measured | `clip-seed11-10s-run1.json`, `clip-seed11-10s-run2.json` |
| Same request (seed 42, 20 s) through this bundle and through the container package | not the same samples: both 20.02 s at 32 kHz, peak 1.0; rms 0.120 against 0.146; 66% against 84% of windows above 0.01 rms | measured | `clip-seed42-20s.json`, `recipe-container-seed42-20s.json` |

The compute per second of audio here (4.2 to 5.7 s) is higher than the container package's 3.2 to 3.3 s. The bundle runs a different `ttnn` build, and the host was heavily loaded (the vocoder and the condition encoder run on the CPU), so the cause is not separated and the speed on an idle host is not measured. The container package's own card gives a median of 65.6 s of compute for 20 s of audio over 20 requests.

No quality evaluation against the base model has been run.

## Limitations

- One song is generated at a time. The queue holds 8 jobs.
- Slower than real time: 4 to 6 s of compute per second of audio in the runs above.
- A seeded request gives the same file on one server. It does not give the same file as the container package.
- Only the `p150` profile on one chip is tested, with clips of 5 to 20 s. The server allows up to 9000 frames (6 minutes).
- A song ends when its lyrics are finished, which can be before the requested length.
- A cold start takes 3 to 9 minutes on this host, with the caches empty.
- No quality evaluation against the base model has been run. The adapter was trained on nine tracks in one style; its effect without the trigger phrase is not measured.
- On some hosts, loading a cached tensor of about 32 MiB or more never returned in the container package (Linux 7.0.0-38, KMD 2.11.0). The bundle's cold start and restart did not show it on that host.

## Risks and safety considerations

- Not for imitating a specific real artist's voice or presenting generated audio as a recording of a real person.
- The server and the page have no authentication and no TLS. They bind 127.0.0.1. `HOST=0.0.0.0` listens on every interface, which is only for a network you trust.
- The weights are not scanned for content. Generated audio has not been reviewed.

## Licensing

Model weights: MiniMax-Music3 Community License; products built with the weights must display "MiniMax-Music3". The merged adapter is Apache-2.0. Code in this package: Apache-2.0 (see `LICENSE` and `NOTICE`).

## Related packages

- [jashansinghTT/minimax-music3-blackhole](https://huggingface.co/jashansinghTT/minimax-music3-blackhole): the container package whose server code this bundle uses.
- [episod/minimax-music3-fiona-crapple-blackhole](https://huggingface.co/episod/minimax-music3-fiona-crapple-blackhole): the companion repository with a launcher for that container and the Gradio page.
- [episod/minimax-music3-fiona-crapple](https://huggingface.co/episod/minimax-music3-fiona-crapple): the merged weights.

## Feedback

Questions or problems with this package: open a discussion on its Hugging Face page. A problem with the `tt` tooling itself: `tt report issue`. Product feedback: support@tenstorrent.com.

## Provenance

| component | source |
| --- | --- |
| weights | [episod/minimax-music3-fiona-crapple](https://huggingface.co/episod/minimax-music3-fiona-crapple) at `53fc7557ea3101d3efbe5e50598bd0bea7f44344`, which is [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) at `fbdf52fbaaca799592917417eb05f1899f1255ec` with the adapter merged into the 252 language-model matrices it targets |
| model code and server | [jashansinghTT/minimax-music3-blackhole](https://huggingface.co/jashansinghTT/minimax-music3-blackhole) at `5a789cd8099cec3a6dc4386db3b4a1ae3925504c`, `code/`, the files the server imports, unchanged |
| page | [episod/minimax-music3-fiona-crapple-blackhole](https://huggingface.co/episod/minimax-music3-fiona-crapple-blackhole) at `2a2ae57a63961fc13cbe1fcfac4f4f5360f1cae3`, `demo/`, unchanged |
| runtime | `ttnn==0.80.0` from PyPI. The container package ran a source build of tt-metal (`v0.79.0-dev20260907`). |
| built | by `tt-orchard can` from `config/can.fiona.toml`; `SHA256SUMS` lists the hash of every file |
