# Voxtral on a chip: `tt-orchard speech-check` against the two published bundles (2026-11-24)

First run of `tt-orchard speech-check` (PR 818) against real servers. Both bundles ran on Blackhole boards under gozer leases. Before this run the command had only met fake servers.

Licences: the speech-to-text bundle serves Voxtral Mini 3B 2507 (Apache 2.0). The text-to-speech bundle serves Voxtral 4B TTS 2603 under CC BY-NC 4.0, which allows non-commercial use only. orchard copied and published nothing; it served and checked what the bundles' author published (`apande-TT/mistralai-voxtral-mini-3b-2507-tt` and `apande-TT/mistralai-voxtral-4b-tts-2603-tt`).

## Setup

| Item | Mini (speech to text) | TTS (text to speech) |
| --- | --- | --- |
| Board | one two-chip board, lease held by a `sleep` process | the other board, its own lease |
| Device given to the container | `--device-id 0` | `--device-id 2` |
| Port | 20000 (set with `--port`) | 20001 |
| First start | answered `/health` in under 40 s (weights already cached) | compiled kernels on the first start, answered about 12 minutes after `tt-model serve` began, which included a weights download inside the container |
| Image | tag `tt-model/voxtral_mini_3b_2507:3f3dff15971a` | tag `tt-model/voxtral_4b_tts_2603:d434b52da3e5` |

The bundles are one-chip p150 packages and the boards here hold two chips each. A request for one chip gets the whole board. `tt-model serve --device-id N` pins which chip the container opens; without it both bundles would have opened `/dev/tenstorrent/0`, so two bundles at once need two boards and an explicit device for the second.

Both bundles list port 8000 in their manifests. `tt-model serve --port` replaces it, and this run used 20000 and 20001.

### A pull problem on a host without skopeo

`tt-model pull` loads a container bundle with `docker load` when `skopeo` is not installed. The image in these two bundles is named `latest`, so the loaded image carried the name `latest` and no `tt-model/...` tag. The second pull took the `latest` name from the first one, and `tt-model serve --local-only` then said the image was not loaded. Fix used here:

1. Pull each bundle and read the new image id from `docker events` (the `load` event).
2. `docker tag <id> <the tag tt-model names in its error>`.
3. Pull each bundle again, so tt-model records the loaded image id in its install record. The check that decides whether the image is current compares that id.
4. `tt-model serve --local-only` then starts the container.

Installing `skopeo` should avoid the problem, since `tt-model` uses it to name the image. This was not tried.

## Speech to text, Mini

`tt-orchard speech-check http://127.0.0.1:20000 --task speech-to-text --language en`, three clips from the Mini card (public sample clips). The expected text for the first two is the card's. The 3.4-minute clip has no human transcript; the expected text is the output of the Mini on CPU under transformers 5.24.0 (the reference-environment trial), so that row compares the chip with a CPU run of the same model, and says nothing about distance from the speech.

| Clip | Audio | Words checked | Word error | Seconds, three runs |
| --- | --- | --- | --- | --- |
| mlk.flac | 13.0 s | 21 | 0.00 | 0.76, 0.76, 0.76 |
| bcn_weather.mp3 | 11.0 s | 18 | 0.00 | 0.82, 0.82, 0.82 |
| obama.mp3 | 203.5 s | 524 | 0.00 | 20.45, 20.46, 20.44 |

Card figures: 0.76 s for the 13-second clip and 19.1 s for the long one. The long clip took 1.3 s longer here than on the card. The malformed request (no file) answered 400. `speech-check` passed.

## Text to speech, TTS bundle

`tt-orchard speech-check http://127.0.0.1:20001 --task text-to-speech --voice casual_male --voice fr_female --voice neutral_female --stream --asr-url http://127.0.0.1:20000`: passed. The words were transcribed by the Mini on the other board.

| Voice | Audio | Seconds per character | Word error through the Mini |
| --- | --- | --- | --- |
| casual_male | 2.88 s (44 characters) | 0.065 | 0.00 |
| fr_female (English text) | 3.12 s | 0.071 | 0.00 |
| neutral_female | 3.92 s | 0.089 | 0.00 |

The audio is 24 kHz, 16-bit, mono. A voice the server does not have and a request without `input` both answered 400. The stream check passed.

Further measurements with a separate script (not part of `speech-check`):

- 11 clips, nine voices, five languages: word error 0.00 on 10 of 11. The Italian sentence scored 0.111 (the Mini heard "La volpe marrone e veloce ..." with an extra "e"). Each request took 1.4 to 2.2 s for 2.2 to 4.1 s of audio. The two-sentence English paragraph (about 10 s of audio) also scored 0.00.
- Single request, same sentence, three runs: 1.72, 1.75 and 1.65 s (card: 1.6 s).
- Streamed request, three runs: first audio after 0.51, 0.50 and 0.62 s, complete after 1.82, 1.81 and 1.92 s (card: first audio about 0.5 s). The streamed raw PCM had the same length as the whole WAV (138,240 bytes, 2.88 s).
- Three extra voices in their own language, for the duration bound only: hi_male 0.126 seconds per character, hi_female 0.095, ar_male 0.105. The Mini has no Arabic, so Arabic words were not scored.

## What the default bounds did on real servers

| Bound | Default | Observed | Reading |
| --- | --- | --- | --- |
| Word error, speech to text | 0.10 | 0.00 on all three clips | Passes with room; the clips are short and clean. |
| Word error, text to speech through an ASR server | 0.15 | 0.00 on 10 of 11, 0.111 on one | The Italian clip passes by 0.04. A strict run could use 0.12. |
| Seconds per character | 0.02 to 0.4 | 0.058 to 0.126 (English, French, Spanish, German, Italian, Hindi, Arabic) | Both ends are about three times looser than any observed value. 0.03 to 0.25 would still pass every clip. |
| Silence floor | peak 300 of 32767 | peaks 5,022 to 15,936 | The floor is 16 times below the quietest voice. |

No bound failed a correct answer. The bounds cannot catch a clip that is half the right length or a quiet recording; tightening them is a decision for a later change. This run changed no code.

## What this run does not show

- It tests short clips and one sentence at a time. It does not test the 32-row batch, the 20.48 s segment limit or the run-on segment the TTS card describes.
- The Mini was checked with three English clips. The other seven languages and the question-answering route (`/v1/chat/completions` with audio) were not run.
- Quality of the voices (the card's UTMOS scores) was not measured. Listening to the saved clips is the only judgement made here.
- Firmware here is 19.17.1 with tt-kmd 2.12.0; the cards say they were tested on 19.15.0 and 2.11.0.
