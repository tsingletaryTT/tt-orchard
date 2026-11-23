# Voxtral speech models: what tt-orchard now knows (2026-11-23)

Two Voxtral models from Mistral AI have published Tenstorrent bundles on the Hub: a speech-to-text model (Voxtral Mini 3B 2507) and a
text-to-speech model (Voxtral 4B TTS 2603). This note records what a preflight did with them before this change, what the models and bundles
are, and what the change adds. No chip was used. Every claim says where it comes from: the bundle's card, the upstream model's configuration
files, or the model code the bundle ships (the code was read and was not run).

## What the harness did before this change

The check below ran the preflight functions (`parse_hub`, `check_hub`, `check_base`, `audio_facts.audio_evidence`) on the Hub's own answers for four
repositories, with no installed bundles. No device was touched.

| Repository | Hub pipeline tag | Hub row | Audio evidence | Base row before |
| --- | --- | --- | --- | --- |
| mistralai/Voxtral-Mini-3B-2507 | none | ok | none | warn: the card names no base model, pass `--base` |
| mistralai/Voxtral-4B-TTS-2603 | text-to-speech | ok (cc-by-nc-4.0) | none | block: no bundle serves `mistralai/Ministral-3-3B-Base-2512` |
| apande-TT/mistralai-voxtral-mini-3b-2507-tt | automatic-speech-recognition | warn: ships code | none | block: no installed bundle serves `mistralai/Voxtral-Mini-3B-2507` |
| apande-TT/mistralai-voxtral-4b-tts-2603-tt | text-to-speech | warn: ships code | none | block: no installed bundle serves `mistralai/Voxtral-4B-TTS-2603` |

Three problems follow. The audio rule (`audio_facts.py`) accepts only the stable-audio-tools layout, so no Voxtral repository is audio. The upstream
Mini model has no pipeline tag, so the harness treated it as a text model and a run would have gone on into the text stages. The TTS model's block
named a text model (`Ministral-3-3B-Base-2512`) as the model to find a bundle for. None of the four rows said that the model is a speech model, which
bundle serves it or how to check it.

## The two models

Where a fact has a source in brackets: card = the bundle's model card; config = the upstream repository's `config.json` or `params.json`;
code = the model code in the bundle's `code/` directory.

### Voxtral Mini 3B 2507 (speech to text, and questions about audio)

- **Parts** (config): a Whisper-style audio encoder (32 layers, width 1280, 20 heads, 128 mel bins, 16 kHz audio), a projector that turns every 4
  encoder frames into one audio token (375 tokens for each 30 seconds of audio), and a Llama-style text decoder (30 layers, width 3072, 32 heads with
  8 key-value heads, vocabulary 131072). The architecture name is `VoxtralForConditionalGeneration`. The Hub card has no pipeline tag.
- **On the chip** (code, card): the encoder, the projector and the decoder, as TTNN modules with one PCC test each. The decoder keeps a resident
  KV cache of 32768 positions and decodes with a captured trace. The encoder and projector run in bf16; the decoder's matmul weights are bfloat8_b
  except the output layer.
- **On the host** (code): audio decoding and the Hugging Face processor, which makes the 128-bin log-mel windows and the prompt tokens. The
  bundle pins `transformers==4.56.0`, `mistral_common`, `soundfile`, `soxr` and `librosa` inside its container.
- **Interface** (card, code): `GET /health`, `GET /v1/models`, `POST /v1/audio/transcriptions` (multipart `file`, optional `language`,
  `response_format` json or text) and `POST /v1/chat/completions` (text and audio parts). A missing file, an undecodable file or a request that
  does not fit the context is answered with HTTP 400. Requests run one at a time. Decoding is greedy. `stream` is refused.
- **Limits** (card): 32768 tokens of context shared by audio, prompt and output, which allows about 30 minutes of audio to transcribe and about
  40 minutes to ask about. Eight languages: English, French, German, Spanish, Italian, Portuguese, Dutch, Hindi.
- **Licence** (card, Hub): Apache-2.0.
- **How the card measured quality** (card): the whole greedy output is compared token by token with the upstream model in transformers 4.56.0
  (bf16, CPU). Four requests were identical. Over a 3.4-minute transcript of 630 tokens, 620 tokens agree and 2 of 521 words differ. Each component
  is compared with its PyTorch module by Pearson correlation (at least 0.99; the whole model's prefill logits 0.995).
- **Speed** (card, one run set, warm): a 13-second clip is transcribed in 0.76 s; about 37 tokens per second afterwards.

### Voxtral 4B TTS 2603 (text to speech)

- **Parts** (config): a text decoder (26 layers, width 3072, from the Ministral 3B base the card lists as `base_model`), a flow-matching acoustic
  transformer (3 layers) that produces one semantic code and 36 acoustic codes for each audio frame, and a neural audio codec that turns the codes
  into a 24 kHz waveform. The frame rate is 12.5 per second (80 ms of audio per frame). There are 20 preset voices, each a small tensor file in the
  upstream repository. The upstream repository has `params.json` and `consolidated.safetensors` and no `config.json`.
- **On the chip** (code): the bundle's own report lists 31 of 31 components on the device (decoder, acoustic stage and codec).
- **On the host** (code): the tokenizer, request batching, the clipping and conversion of samples to 16-bit PCM, and the WAV header.
- **Interface** (card, code): `GET /health`, `GET /v1/models` and `POST /v1/audio/speech` with JSON `input`, optional `voice` (one of 20),
  optional `response_format` (`wav` or `pcm`) and optional `stream` (raw 16-bit PCM, chunked). An empty input, an unknown voice or an unsupported
  format is answered with HTTP 400. Every request runs inside a batch of 32 rows. Audio is 24 kHz, 16-bit, mono, at most 20.48 s for each text
  segment; longer text is split every 18 tokens and joined.
- **Licence** (card, Hub): CC BY-NC 4.0, which allows non-commercial use only.
- **How the card measured quality** (card): each clip is transcribed with Whisper large-v3-turbo for word error rate (WER) and scored with UTMOS22,
  and compared with the upstream model run on CPU with the same prompts and sampling noise. Over 32 prompts WER is 0.021 against 0.023 for the
  original and MOS 3.67 against 3.74. The waveform of every row correlates with the reference at 0.9998 or more.
- **Speed** (card): first streamed audio after about 0.5 s; one request about 1.6 s; 32 requests together 2.6 to 3.9 s.

## What is different from the Stable Audio path

| | Stable Audio (existing audio track) | Voxtral Mini (speech to text) | Voxtral TTS (text to speech) |
| --- | --- | --- | --- |
| Repository layout | `model_config.json`, no `config.json` | `config.json` (Mini) | `params.json`, no `config.json`, no `model_config.json` |
| Hub pipeline tag | text-to-audio or audio-to-audio | none (the bundle's card says automatic-speech-recognition) | text-to-speech |
| Output | audio from a diffusion sampler | text | audio from a decoder, a flow-matching stage and a codec |
| Reference for a check | CPU run of the library with recorded step latents | token-for-token against transformers on CPU | Whisper word error rate, UTMOS, and a waveform correlation with the same noise |
| Served through | no bundle; the harness runs its own stages | container bundle (schema 5.1 manifest) | container bundle (schema 5.1 manifest) |
| Weights swap into a nearest bundle | not applicable | not applicable: each bundle is built for its model | not applicable |

A container bundle holds its code and compiled programs in an image, so the harness's weights-only route (swap new weights into the nearest
bundle) does not apply to either model, and neither needs a port written by the harness. The bundles were written by someone else and published.
What the harness can do for them is know that they exist, say how to run them safely and check a running server.

## What the change adds

1. `orchard/speech_models.py` recognises a speech model from the Hub's answer. Rule, in order: a pipeline tag decides alone
   (`automatic-speech-recognition` is speech-to-text, `text-to-speech` is text-to-speech); with no usable tag, a tag in the tag list; with neither, the
   architecture name `VoxtralForConditionalGeneration`. The architecture rule exists because the upstream Mini card has no pipeline tag. The stable-audio
   tags stay with the audio track. It holds a short table of the two known bundles (their card facts as of 2026-11-23).
2. The preflight's base row for a speech model is a block with the reason `speech-model`. It says what the model is, that the harness has no speech
   stage and starts no run, which bundle serves it, the bundle's endpoints, the licence (and non-commercial use for the TTS model), and the command
   that checks a server. A dry classification (`bringup --classify`) stops on this block, because there is no class to report.
3. `tt-orchard speech-check URL --task speech-to-text|text-to-speech` sends requests to a running server and reports one row per check (the checks and
   the exact word error rate rule are in the module's docstring and in `--json` output under `rule`). It opens no device and takes no lease. The address
   needs a port, must be on this machine, and may not use ports 8000 or 8100 to 8103. Redirects are not followed.

### Decisions

- **No speech kind in `model_kind.py` and no new `modality` in `audio_facts.py`.** Those two records feed stages 0 to 8. A speech model never reaches
  stage 0, because the base row blocks first, so a record for it would never be written or read. The task name lives in the base row's data
  (`speech: {task, bundle}`).
- **The row is a block.** For the Mini model the earlier row was a warning, and a run would have started on the text path with nothing to
  compare. A block states that no run is possible. The harness can lift the block later by adding a speech route, with no change to the rows' wording
  except the sentence that says there is none.
- **A fixed table of two bundles.** The Hub has no field that says "this bundle serves that model" beyond the card's `base_model`. The table is small,
  dated in the code (`KNOWN_ON`) and the row says "known" for a model with no entry. The alternative, asking the Hub's search at preflight time,
  adds a network call and a failure mode to a row that has to be fast.
- **Loopback only for `speech-check`.** The command sends audio and text to the address it is given. A server on this machine is the only case that
  has come up, so other hosts are refused with a message.
- **The duration bounds are wide.** A text-to-speech answer must run between 0.02 and 0.4 seconds per character. The card's English clips run at about
  0.06 and its Hindi clips at about 0.2. The bounds catch an empty, truncated or run-on answer and say nothing about what was said; the words are
  checked only when a speech-to-text server is named with `--asr-url`.

## Serving a bundle on this machine (from the cards, and from how the coder was started)

- `tt-model pull REPO` downloads the container image (about 1.3 GB for the Mini bundle) and the upstream weights (9.4 GB for the Mini bundle);
  `tt-model serve REPO` starts the server and prints its port. The cards say 20000. The bundles' manifests list port 8000 and the Mini code's default
  is 8000, so a container started by hand with the defaults would use 8000; use `tt-model serve` and read the port from its output.
- Take a gozer lease first, as for any chip use. A board is leased whole, so one chip of a two-chip board is held with its neighbour. The first start
  after the machine has been idle can fail with a fabric timeout; `gozer reset LEASE` under the lease and a second start cleared it for the coder.
- First start: about 30 s for the Mini bundle (the device programs are in the image) and about 4 minutes for the TTS bundle (the first start compiles
  them); later starts about 20 s and 55 s (card).
- The 4B TTS weights are non-commercial. Nothing the harness produces with them may be published or sold.

## Owed: a run on a chip

`speech-check` is tested against fake servers that copy what the two bundles' code does. It has not been run against a real bundle. The first real run
should serve one bundle under a lease and run, for the Mini model, three or more clips with known text (the card's public clips are named in its
Usage section), and for the TTS model a default-voice request, two other voices, `--stream`, and `--asr-url` pointing at the Mini server. The
two bundles together need two chips of different boards or one after the other. Until then the duration bounds, the default word error limits
(0.10 and 0.15) and the silence floor (a peak of 300) rest on the card's numbers and have not been seen on a real answer.

Other open points: the reference environment (`transformers` 5.24.0) was not tried against either upstream model; the Mini bundle pins 4.56.0 and
the TTS bundle pins 5.12.1 inside their containers, so the host version does not matter for serving them. Whether the harness should grow a real
speech route (stages that compare a speech model with a reference) is a separate decision.

## Tests

`tests/test_speech_models.py` (39 tests) and `tests/test_speech_check.py` (162 tests) were written before the code they test; the speech-check tests
run against fake servers on ports the system picks. Sixty-one mutations, each removing or weakening one guard, were applied one at a time and each
turned at least one test red (recognition rule, bundle lookup, the base row and its hook, dry-classification stop, address rules, redirect and size
limits, word error rate rule, clip reading through `safeio`, option pairing, the 4xx checks, each WAV rule, the stream rules and the exit codes). The
first pass of 57 mutations left five green (a dead guard in the bundle lookup, a test of the audio-track exclusion that could not fail, the timeout
message, `--max-wer` validation, the JSON exit code); the dead guard was removed and the other four got tests, and the second pass is the 61 above.
The coder (Qwen3-Coder-Next on the local server) was asked once to review `speech_check.py` for bugs. It gave six findings; none held (the file read it
described as unhandled is handled, the size limit it said was missing is enforced in `_request`, and two findings said themselves that the code was
correct). Nothing from it was used.

One existing test changed: `test_a_text_to_speech_model_is_not_audio_for_this_rule` in `tests/test_audio_preflight.py` asserted the old reason
(`nearest-model-missing`) for a text-to-speech model and now asserts `speech-model` and that the audio flag is not set. Every other existing test is unchanged.
