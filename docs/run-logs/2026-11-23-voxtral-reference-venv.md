# Voxtral on the CPU reference environment (2026-11-23)

The operator asked whether the reference environment (transformers 5.24.0, torch 2.15.0 CPU) can run the two upstream Voxtral models. This note records
what ran, what needed a change and what each run costs. It follows `2026-11-23-voxtral-speech-models.md`. No chip, gozer lease or `tt-smi` was used, and
`ttnn` was never imported. Weights came from the Hub's two upstream repositories (Voxtral-Mini-3B-2507 and Voxtral-4B-TTS-2603) and from the Whisper
large-v3-turbo repository. The Mini model is Apache-2.0. The TTS model is CC BY-NC 4.0 (non-commercial), and the runs here were for research only; no
output was published.

## Result

| Question | Answer |
| --- | --- |
| Does the Mini model run on CPU under transformers 5.24.0? | Yes, without a code change. |
| Does it give the same tokens as transformers 4.56.0, the version the bundle was checked against? | Yes. Every token matched on all four requests, including the 631 tokens of the 3.4-minute clip. |
| Is greedy decoding repeatable? | Yes. Two separate processes gave identical tokens on all four short requests. |
| Does the TTS model run on CPU under 5.24.0? | Yes, after one line of the bundle's reference loader was changed (see below). One sentence in English gave 62 frames, 4.96 s of audio and a Whisper word error rate of 0.000. |
| Is the TTS run repeatable? | Yes with a fixed noise seed. Two runs gave the same codes (checksum `1a11fbab79eb4850`) and byte-identical WAV files. |

## The environment

The environment is a copy of the reference environment, built beside it so the original stays as it was. Python 3.12 (uv picked 3.12.14 for the new
copy; the original uses 3.12.12), torch 2.15.0 (CPU build) and the same 36 pinned packages as the reference environment, plus these, each checked
before installing (Mistral AI's own `mistral-common`, Apache-2.0; `soundfile`, `soxr`, `scipy`, `jiwer` and their dependencies):

```bash
uv venv REFSPEECH --python 3.12
uv pip install --python REFSPEECH/bin/python --index-url https://download.pytorch.org/whl/cpu "torch==2.15.0"
uv pip install --python REFSPEECH/bin/python -r reference-freeze-without-torch.txt     # the 36 packages of the reference environment
uv pip install --python REFSPEECH/bin/python "mistral_common[audio]" jiwer scipy
```

What this added and changed:

- `mistral-common` 1.13.1 (the bundle pins 1.12.0 for transformers 4.56.0). Its audio extra brings `soundfile` 0.14.0 and `soxr` 1.1.0. The processor's
  transcription request needs it (transformers 5.24.0 reads `is_mistral_common_available()`).
- `mistral-common` 1.13.1 asks for `numpy<2.4` on Python 3.12, so the install moved numpy from 2.5.3 to 2.3.5 in this copy. The reference environment
  itself still has 2.5.3 and has no `mistral-common`. Adding speech support to that environment would change its numpy.
- `pydantic`, `pydantic-extra-types`, `pycountry`, `tiktoken`, `jsonschema`, `requests` and `rapidfuzz` came in as dependencies.
- `jiwer` 4.0.0 computes the word error rate. `scipy` 1.18.1 is what the bundle's scoring code uses to resample.
- `librosa` is not needed. The bundle's own code reads audio with `soundfile` and resamples with `soxr`, and so did these runs.

A second copy with transformers 4.56.0, huggingface-hub 0.36.2 and `mistral_common==1.12.0` (the bundle's pins) was built the same way for the version
comparison.

All model runs used a scrubbed environment (no token variables, `HF_HUB_OFFLINE=1`) and `python -I`, so the code that loads and runs the downloaded model
code could not reach the network or read a credential. The weights were downloaded first, in a separate step.

## Voxtral Mini 3B 2507

Load: `VoxtralForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.bfloat16)` and `AutoProcessor`. The input is built with
`processor.apply_transcription_request(audio=..., model_id=..., language=..., sampling_rate=16000, format=["wav"])`, the same call the bundle's own test
uses. Audio is read with `soundfile`, averaged to mono and resampled to 16 kHz with `soxr` (HQ), as the bundle does. Decoding is
`model.generate(do_sample=False)`. The clips are the three public ones named in the bundle's card (checksums in the table below).

| Clip | Audio | Language | Prompt tokens | New tokens (with the end token) | Wall time | Tokens per second | Peak memory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mlk.flac` (sha256 `88a3befa`) | 13.0 s | en | 383 | 24 | 7.8 s | 3.06 | 8.6 GiB |
| `bcn_weather.mp3` (`47f2f8c2`) | 11.0 s | en | 383 | 26 | 19.3 s | 1.35 | 8.7 GiB |
| `bcn_weather.mp3` | 11.0 s | es | 383 | 24 | 9.8 s | 2.46 | 8.7 GiB |
| `bcn_weather.mp3` | 11.0 s | none (detect) | 380 | 26 | 17.4 s | 1.49 | 8.7 GiB |
| `obama.mp3` (`e0059603`) | 203.5 s | en | 2633 | 631 | 304.8 s | 2.07 | 9.3 GiB |

Timings come from a shared machine with eight threads, so the spread between runs of the same size (7.8 s and 10.3 s for the first clip) is real. Loading takes
under 1 s because the weights are memory-mapped; the first token carries the cost of reading them.

Transcripts:

- `mlk.flac`: "I have a dream that one day this nation will rise up and live out the true meaning of its creed." This is the card's text, 23 tokens plus the end token.
- `bcn_weather.mp3`, English: "Yesterday it was 35 degrees in Barcelona, but today the temperature will go down to minus 20 degrees." This is the card's text.
- `bcn_weather.mp3`, `language=es`: "Ayer hacía 35 grados en Barcelona, pero hoy la temperatura bajará a menos 20 grados." This is the card's Spanish text, 23 tokens plus the end token.
- `bcn_weather.mp3`, no language: the English transcript. With no language the model detected English.
- `obama.mp3`: 630 text tokens and 521 words, which are the counts the card gives for its 3.4-minute transcript.

### Against transformers 4.56.0

The same script ran under 4.56.0 with the bundle's pins, on the three requests with a language and on the 3.4-minute clip. All four gave exactly the same token
sequence as 5.24.0 (24, 26, 24 and 631 tokens, no differing position, 0.000 word difference). So the card's statement that the bundle matches transformers
4.56.0 token for token also holds for 5.24.0 on these clips. The 4.56.0 wall times in the saved files (47 to 819 s) were taken while the TTS runs below used the
same cores, and they say nothing about the version's speed.

One difference in the processor interface: under 4.56.0 `apply_transcription_request` has `language` as a required argument, and passing `language=None` fails
with a `TypeError`. Under 5.24.0 `language` is optional and leaving it out lets the model detect the language. A stage 1 reference step that wants
language detection needs 5.x.

### Repeatability

The four short requests ran in two separate processes under 5.24.0. All four token sequences are identical. The 3.4-minute clip ran once under 5.24.0 and once
under 4.56.0, with identical tokens.

## Voxtral 4B TTS 2603

The upstream repository has `params.json`, `consolidated.safetensors`, `tekken.json` and 20 voice files, and no `config.json`. transformers has no class for it
(`model_type` is `voxtral_tts`). The upstream serving path is vLLM-Omni. The bundle ships its own CPU reference: `_reference_loader.py` (1391 lines) rebuilds the
model from the native checkpoint (the text backbone as transformers' `MistralForCausalLM`, and the flow-matching acoustic transformer and codec decoder vendored
from vLLM-Omni's Apache-2.0 source, so no vLLM install is needed), and `reference/golden.py` runs the generation loop: prefill with the voice, then per frame the
acoustic sampler and one backbone step, then the codec. The bundle's `common.py` and `golden.py` import `torch` at the top and `ttnn` only inside functions the
run did not call; the script asserts that `ttnn` is not in `sys.modules` after the imports.

### One change to run it under 5.24.0

The loader failed at `_rebuild_nonpersistent_buffers` with

```
TypeError: MistralRotaryEmbedding.__init__() got an unexpected keyword argument 'device'
```

The 5.24.0 signature is `MistralRotaryEmbedding(config)`. The change is one line in a private copy of the bundle's code:

```
-    model.model.rotary_emb = MistralRotaryEmbedding(config=cfg, device="cpu")
+    model.model.rotary_emb = MistralRotaryEmbedding(config=cfg)
```

The rotary table is built on the CPU either way. After that line the loader finished its own structural checks (it raises when a tensor is left unconsumed or on `meta`, or when the codec frame
rate is not 12.5 Hz; none raised) and the model was built in 17 s with a peak of 25 GiB. The bundle pins 5.12.1 for its container, so the failure belongs to the host version; the
shipped container is unaffected. A `FutureWarning` says `past_key_values` becomes `cache` in 5.40.0.

Two further changes in the run script, both choices and not fixes: the bundle's `_voice_embedding` calls `torch.load(..., weights_only=False)` on the voice file,
and the script replaced it with `weights_only=True` (the voice file is one tensor, and it loaded). The audio is cut at the row's own `end_audio` frame (frame 61,
which is 61 frames of 1920 samples).

### The run

Text: "The expedition mapped the cave system for three long weeks, marking every passage with white chalk." (one of the card's eight scored clips). Voice
`casual_male`, one row, prompt of 170 tokens (147 voice slots), noise drawn once from seed 0, classifier-free guidance 3.0, cap of 100 frames.

| Measure | Value |
| --- | --- |
| Frames until every row emitted `end_audio` | 62 (the stop frame is 61) |
| Audio | 117,120 samples after the cut, 24 kHz, 4.88 s (4.96 s before the cut) |
| Wall time | 106 s on the first run, 551 s on a repeat that shared the cores with the 4.56.0 runs |
| Peak memory | 25.0 GiB |
| Whisper large-v3-turbo transcript | "The expedition mapped the cave system for three long weeks, marking every passage with white chalk." |
| Word error rate (the card's rule: Whisper English normalizer on both sides, `jiwer`) | 0.000 |
| Repeat with the same seed | same codes checksum (`1a11fbab79eb4850`), byte-identical WAV |

The card's own clip for this sentence is 6.08 s long, and ours is 4.88 s. The card ran 32 rows in one batch with its own noise draws; the length of a clip depends on the
noise, so the two are not the same experiment. The word error rate agrees (0.000 in both).

A second voice, `fr_female`, spoke "Le renard brun saute par-dessus le chien paresseux, puis il disparaît dans la forêt." in 54 frames (4.32 s, stop frame 53, 350 s
wall while sharing the cores). It was not scored: the bundle's scoring code transcribes with the language fixed to English.

### Not run

- **UTMOS (the naturalness score).** The card's loader is `torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)`, which downloads and runs code
  and weights from a GitHub repository outside the Hub. It was not run. Intelligibility (Whisper word error rate) was.
- **Hindi and Arabic voices**, which the card reports as the weakest (word error rates 0.31 to 0.38 for Hindi).
- **A comparison with a chip waveform.** The bundle's server draws its own noise, so codes from a chip run do not line up with the seed-0 codes here. A comparison
  needs the noise handed to both sides, as the bundle's accuracy test does with `x0`.

## What a stage 1 reference step would need

| | Mini (speech to text) | TTS |
| --- | --- | --- |
| Inputs | a clip (any format `soundfile` reads), a language or none | text, a voice name, a noise seed, a frame cap |
| Output to record | the generated token ids and the decoded text | the codes (37 per frame), the waveform at 24 kHz and the stop frame |
| Packages | transformers 5.x, `mistral-common[audio]`, torch (CPU) | transformers 5.x, torch, `safetensors`, `tiktoken`; for scoring also `scipy`, `jiwer` and Whisper |
| Weights to download | 9.4 GB of safetensors (the repository also holds a second copy in Mistral's own format, so a full snapshot is 18 GB; `allow_patterns` for the sharded files halves it) | about 8 GB, plus the voice files it uses (about 1 MB each) and 1.6 GB for Whisper |
| Memory | 9 to 10 GiB | 25 GiB |
| Time on 8 threads | 8 to 20 s for a clip of 11 to 13 s, 5 minutes for a clip of 3.4 minutes | 17 s to build and about 2 minutes for a sentence of 5 s |
| Comparison with a chip | tokens against tokens (identical on every clip here) | waveform correlation and word error rate, with the noise shared |

The Mini model is cheap enough for a reference step on every run. The TTS model fits a one-off reference run kept in the ledger.

## Not done

- The 3.4-minute clip was not repeated under 5.24.0 in a second process.
- No float32 run of the Mini model; every run was bfloat16, as in the card.
- No Hindi, Arabic or batch-of-32 TTS runs, no UTMOS, no comparison of a TTS waveform with a chip.
- The audio reference environments for other models (Stable Audio and similar) were not tried on 5.24.0 here.
- `mistral-common` is not in the reference environment itself. Whether to add it, and with it the numpy change, is a decision for whoever adds a speech route.
