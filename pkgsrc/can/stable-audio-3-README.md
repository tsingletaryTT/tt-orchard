---
tags:
- blackhole
- p150
- tt-dit-server
- thin
- tt-model-cache
- text-to-audio
- music-generation
license: other
license_name: stability-ai-community-license
license_link: LICENSE.md
pipeline_tag: text-to-audio
base_model:
- stabilityai/stable-audio-3-small-music
---

# stable-audio-3-small-music-blackhole on Tenstorrent p150

> Derived from [stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music) by Stability AI, unmodified
> (revision `8bdfb53c6b0fbc62113cd571fd37373efe15fa14`). The weights are not in this repository.
> **Powered by Stability AI.**
> This is a community package. It is not a release of Stability AI.

Stable Audio 3 small music, a latent diffusion model that makes stereo music clips at 44.1 kHz from a text prompt (1 to 120 seconds a clip), served as a job
API. The diffusion transformer and the autoencoder decoder's transformer layers run on one Blackhole chip. The text encoder, the sampler and the rest of
the decoder run on the host CPU.

Runs on Tenstorrent **p150** (mesh `P150`), one chip.

Packaged with [tt-model-manager](https://github.com/tenstorrent/tt-model-manager) 0.1.0 (manifest schema 6, kind `tt-dit-server`).

## At a glance

| | |
| --- | --- |
| Architecture | Latent diffusion: a 459 M-parameter diffusion transformer (20 layers) and a 108 M-parameter transformer autoencoder; T5Gemma text encoder |
| Hardware | p150 (one chip of a p150, of a p300c or of a QuietBox 2) |
| Output | Stereo FLAC, 44,100 Hz, 1 to 120 seconds a clip |
| Default settings | 8 steps, guidance scale 1.0, ping-pong sampler |
| License | Stability AI Community License (see Licensing) |
| Weights | Gated: a Hugging Face account that accepted the model's terms downloads them |

## Intended use

**Direct use:** making music clips from a text prompt for research, evaluation, testing or personal use, and for the limited commercial use the license allows (see Licensing).

**Out-of-scope use:** audio input (inpainting or audio-to-audio: the job API takes a text prompt only), audio that is not music, several clips at the same time, and any use the license or Stability AI's acceptable use policy forbids.

## Prerequisites

- The [Tenstorrent CLI](https://github.com/tenstorrent/tt-cli), `tt`: `uv tool install tenstorrent` (or use tt-model alone, see the Quickstart)
- Tenstorrent hardware: **p150**
- A Hugging Face account that accepted the terms on the [model's page](https://huggingface.co/stabilityai/stable-audio-3-small-music), and `HF_TOKEN` set to a token of that account. The download holds the T5Gemma text encoder, which is provided under the Gemma Terms of Use.
- About 4 GB for the weights in the Hugging Face cache and about 5 GB for the bundle's virtual environment.

## Quickstart

```bash
export HF_TOKEN=<token of an account that accepted the model's terms>
tt serve episod/stable-audio-3-small-music-blackhole
```

`tt serve` installs the bundle into its own virtual environment from the pins in `requirements.txt` and starts the server on the port it reports (20170 when
you run `run.sh` yourself). The first start downloads the pinned weights revision (about 3.4 GB) and loads the model onto the chip; the server is ready when `GET /health`
answers 200 and the log shows `"ready": true`. The weights go into the Hugging Face cache that `HF_HOME` names; when `HF_HOME` is not set, `run.sh` uses a cache inside the
bundle's directory, so set `HF_HOME` to share a cache you already have. The first clip after a start compiles kernels and takes longer than the later ones (see Expected performance).

Without tt-cli: `tt-model serve episod/stable-audio-3-small-music-blackhole`.

Make a clip with the client tool in this bundle (standard library only):

```bash
python tools/audio_client.py --url http://127.0.0.1:20170 --prompt "A warm acoustic guitar melody with soft percussion, 100 BPM" \
    --duration 10 --seed 42 --out clip.flac
```

## Capabilities

This package exposes a job server on port 20170 (the port `run.sh` uses unless `PORT` is set), **not** an OpenAI-compatible chat API. A clip is made by a job: submit it, poll it, download the audio.

| Route | What it does |
| --- | --- |
| `GET /health` | 200 when the model is loaded; 503 while it loads or after a failed load |
| `GET /v1/health` | the server's state, the queue, the limits and the defaults |
| `POST /v1/audio/jobs` | `{prompt, duration?, steps?, cfg_scale?, seed?}` gives 202 `{id}`; 400 for a bad field; 413 for a body over 64 KiB; 415 unless the body is `application/json`; 429 when the queue is full; 503 while the model loads |
| `GET /v1/audio/jobs/<id>` | `status` (`queued`, `running`, `done`, `error`, `cancelled`), `queue_position`, `stage`, `settings`, `stats` (audio seconds, server seconds, stage times) and `error` |
| `GET /v1/audio/jobs/<id>/audio` | the clip as FLAC (409 until the job is done) |
| `DELETE /v1/audio/jobs/<id>` | drops a queued job; a running clip cannot be stopped and its clip is discarded |
| `GET /openapi.json`, `GET /docs` | the API description |

Fields: `prompt` (1 to 2,000 characters), `duration` (1 to 120 seconds, default 10), `steps` (1 to 100, default 8), `cfg_scale` (1.0 to 20.0, default 1.0), `seed` (0 to 4,294,967,295; a random one is chosen and returned in `settings` when none is given). The same settings give the same clip.

The server makes one clip at a time and keeps a short queue (2 waiting). A request that arrives when the queue is full gets 429 at once. It binds 127.0.0.1 and answers requests whose `Host` names the loopback address only, unless `HOST` is set.

Settings are environment variables (uvicorn passes no flags):

| Variable | Meaning | Default |
| --- | --- | --- |
| `HOST`, `PORT` | address and port | `127.0.0.1`, 20170 |
| `AUDIO_MAX_WAITING` | jobs that may wait while one runs | 2 |
| `AUDIO_MAX_SECONDS` | longest clip accepted, 1 to 120 | 120 |
| `AUDIO_HOST_DECODER=1` | keep the whole autoencoder decoder on the host | off (its transformer layers run on the chip) |
| `AUDIO_WARMUP_SECONDS` | make one clip of this many seconds at start, so the first request does not compile kernels | 0 (off) |
| `AUDIO_PKG_CACHE` | where the cache that holds only the pinned revision is made | under `XDG_CACHE_HOME` or `~/.cache` |

## Expected performance

{{NUMBERS}}

## Limitations

- The text encoder, the sampler, the decoder's first linear layer and final convolution, and the waveform unpatching run on the host CPU. A host under load makes clips slower.
- One clip at a time; a second request waits in a queue of 2 or gets 429.
- Agreement with the CPU reference was measured on a 10-second clip, one prompt and one seed. Longer clips were timed and were not compared with the reference.
- Sound quality was not evaluated by this package's measurements.
- Tested only on one chip of a Blackhole p300c board with ttnn 0.80.0.
- The job server has no sign-in. Run it on a network you trust; it binds the loopback address by default.
- Generated audio can resemble existing recordings. Check the output before you use or share it.

## Licensing

The model is licensed under the Stability AI Community License. `LICENSE.md` is a copy of the agreement from the model's repository at the revision above, and `NOTICE` holds the attribution notice it asks for. The text encoder in the model download is provided under the Gemma Terms of Use (`LICENSE_GEMMA.md`; <https://ai.google.dev/gemma/terms>). The runtime code in this package is Tenstorrent's, under the Apache License 2.0 (`LICENSE-runtime`). The model library `stable_audio_3` in `extra_code/` is Stability AI's, under the MIT License (`LICENSE-stable-audio-3-library`). This package does not include the model weights.

The agreement says, word for word:

> {{QUOTE:research}}

> {{QUOTE:register}}
>
> {{QUOTE:threshold}}

> {{QUOTE:distribution}}

> {{QUOTE:restrictions}}

{{ATTRIBUTION}}

Stability AI's acceptable use policy: <https://stability.ai/use-policy>. Registration for commercial use: <https://stability.ai/community-license>. Read the whole agreement in `LICENSE.md` before you use or share the model or this package.

## Related packages

- [stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music): the model this package serves. [stabilityai/stable-audio-3-small-music-base](https://huggingface.co/stabilityai/stable-audio-3-small-music-base) is its base model; this package does not serve it.
- [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3): the model library (commit `77e0398c26f7d22d801ec1bb036307e8b231f741` is in `extra_code/stable_audio_3/`).

## Feedback

Questions or problems with this package: open a discussion at https://huggingface.co/episod/stable-audio-3-small-music-blackhole/discussions, which reaches its author. A problem with the `tt` tooling itself: `tt report issue` (collects your environment and opens a prefilled issue against tenstorrent/tt-cli). Product feedback: support@tenstorrent.com.

## Provenance

| component | built from |
| --- | --- |
| weights | `stabilityai/stable-audio-3-small-music` at `8bdfb53c6b0fbc62113cd571fd37373efe15fa14` (not in this repository) |
| model library | `Stability-AI/stable-audio-3` at `77e0398c26f7d22d801ec1bb036307e8b231f741` (`extra_code/stable_audio_3/`) |
| ttnn | 0.80.0 from PyPI (`requirements.txt` pins every package) |
| runtime and server | `extra_code/` (Tenstorrent): the chip runtime, the job server and its ASGI front end |
| files | `SHA256SUMS` lists the hash of every file; every number above comes from a file in `evidence/` |
