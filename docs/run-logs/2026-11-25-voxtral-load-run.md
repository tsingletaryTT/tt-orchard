# Voxtral under load, and a singing probe (2026-11-25)

Follow-up to `2026-11-24-voxtral-chip-run.md`. That run sent one request at a time. This run sends several at once to the same two bundles (speech to text on one board, text to speech on the other, each under its own gozer lease; ports 20000 and 20001, `--device-id` 0 and 2). Both load tests ran at the same time on the two boards, so the host CPU was shared; the 1-minute load average was 4.4 to 8.2 during the TTS run and about 1.1 during the last Mini repeats.

Licences: the speech-to-text bundle serves Voxtral Mini 3B 2507 (Apache 2.0). The text-to-speech bundle serves Voxtral 4B TTS 2603 under CC BY-NC 4.0, which allows non-commercial use only. This run served what the bundles' author published and copied nothing. The WAV files stay on the machine that made them.

## Protocol

- TTS: `load_tts.py`. N threads each open a connection, wait on one start flag and POST `/v1/audio/speech` together. Each request has its own voice (13 presets, cycled) and its own sentence (English for the English voices, a native sentence for fr, es, de, it). N is 1, 4, 8, 16 and 32. Each N runs five times whole-file (WAV) and five times streamed (`stream: true`, raw PCM); one unrecorded warm-up request came first. Time per request is from sending to the last byte; first-chunk time is to the first body bytes. Total (wall) time is from the start flag to the last request finishing.
- Mini: `load_mini.py`. N threads POST `/v1/audio/transcriptions` together with the 13.0 s clip (mlk.flac) or the 203.5 s clip (obama.mp3), N of 1, 2 and 4, five repeats each. Word error is scored against the expected text from the first run log.
- The scripts and raw JSON are kept with the run notes on the machine (`tts_load.json`, `mini_load.json`); they use only the standard library.

## TTS results

Wall time for N simultaneous requests, five repeats each (seconds). Every request returned 200 in every run; no request failed.

| N | Whole file, five walls | Median | Streamed, five walls | Streamed first chunk, mean per run (range) |
| --- | --- | --- | --- | --- |
| 1 | 3.80, 1.79, 1.73, 1.78, 1.79 | 1.78 | 1.96, 1.89, 1.89, 1.89, 1.94 | 0.58 to 0.66 |
| 4 | 2.68, 9.45, 2.45, 2.53, 2.79 | 2.68 | 2.65, 2.67, 10.58, 2.77, 2.77 | 0.68 to 0.82 |
| 8 | 2.45, 2.87, 2.92, 9.72, 3.25 | 2.92 | 2.65, 10.49, 3.11, 3.06, 2.78 | 0.61 to 0.84 |
| 16 | 3.30, 3.17, 3.20, 2.78, 2.90 | 3.17 | 3.27, 3.53, 3.09, 4.42, 10.57 | 0.64 to 0.86 |
| 32 | 2.60, 3.21, 2.49, 9.31, 3.17 | 3.17 | 2.82, 2.68, 3.42, 10.17, 3.33 | 0.46 to 0.56 |

- The server answers a group together: in most runs all N requests finished within a few milliseconds of each other, so time per request equals the wall time. Thirty-two requests (about 110 s of audio) took a median 3.2 s, against 1.8 s for one request.
- Seven of the 50 whole-file and streamed runs took 9.3 to 10.6 s, about 7 s more than their neighbours, at every N from 4 to 32 and once at N of 1 (3.8 s). They did not line up with host load. The cause was not found. A reader who needs a latency figure should use the median and keep the 9 to 10 s runs in view.
- Streaming adds about 0.1 s to the total at N of 1 and returns the first audio after about 0.6 s. At N of 32 the first chunk came after 0.46 to 0.56 s.
- Long texts, one request, three repeats: a 3-sentence lighthouse text of 301 characters came back as 26.3 s of audio in 4.50 to 4.57 s. The same text three times (911 characters) came back as 71.6 s of audio in 4.87 to 4.91 s. The first text produces more audio than the card's 20.48 s per-segment limit, so the server splits long input on its own and returns one file. The segment limit itself was not probed with an input just above 20.48 s.

## Mini results

Seconds per request (sorted), first two of five repeats; the five wall times are given as a range.

| Clip | N | Request times | Wall range | Failures | Word error |
| --- | --- | --- | --- | --- | --- |
| mlk.flac, 13 s | 1 | 0.78 to 0.80 | 0.77 to 0.80 | 0 | 0.00 |
| mlk.flac, 13 s | 2 | 0.77 and 1.53; 0.90 and 1.77 | 1.54 to 1.77 | 0 | 0.00 |
| mlk.flac, 13 s | 4 | 0.78, 1.54, 2.29, 3.05 | 3.05 to 3.07 | 0 | 0.00 |
| obama.mp3, 203 s | 1 | 19.48 | 19.45 to 19.52 | 0 | 0.00 |
| obama.mp3, 203 s | 2 | 19.52 and 38.65 | 38.59 to 38.68 | 0 | 0.00 |
| obama.mp3, 203 s | 4 | 19.6, 38.7, 57.8, 77.0 | 76.9 to 77.0 | 0 | 0.00 |

The card says one request at a time, and the server does exactly that: the extra requests wait their turn and none fails or times out. The k-th request finishes after k times the single-request time, and every answer is correct (word error 0.00 against the expected text). A client that sends four long clips together sees the last one return after about 77 s.

## What the logs say about where a TTS request spends its time

The container log has no per-request or per-stage lines. It prints the start-up lines, a first warm-up synthesis of 52.6 s (kernel compile) and a background warm-up of 15.5 s covering every voice, and nothing after that for any request. This run therefore cannot say how the 1.7 s of one request divides between prefill, decode, codec and acoustic work. The load numbers give one bound: 32 requests cost about 1.4 s more than one, so the per-request fixed cost dominates and the batch absorbs the rest. Per-stage timing needs a profiling hook in the bundle or a run with the tt-metal profiler; neither was tried.

## Singing probe

Each of the 20 voice presets (ar_male, casual_female, casual_male, cheerful_female, de_female, de_male, es_female, es_male, fr_female, fr_male, hi_female, hi_male, it_female, it_male, neutral_female, neutral_male, nl_female, nl_male, pt_female, pt_male) received four lyric lines: "Happy birthday to you"; the four-line birthday verse; the first two lines of "Twinkle, twinkle, little star"; and the first lines of "Mary had a little lamb". The WAVs are kept with the run notes (80 files). The Mini transcribed each with `language=en`.

| Line | Median word error through the Mini | Worst voice |
| --- | --- | --- |
| Happy birthday to you | 0.00 | 0.00 |
| Twinkle, twinkle (two lines) | 0.00 | 0.00 |
| Birthday verse | 0.06 | 2.00 |
| Mary had a little lamb | 0.20 | 8.95 |

Short lines come back as understood speech from every voice. The longer lines went wrong in some voices: a few ran on for 25 to 30 s and the Mini heard words that are not in the lyric (casual_male on the birthday verse: "Happy birthday to, you know, oh, me, me, yo, yo, yo"; nl_female on Mary: "bibb de reiptes, dat a tast"). Repeated lyric lines are where the model drifts.

Sung or spoken: the output is spoken. No voice sang a tune.
- Duration matches speech: the 22-word Twinkle lines run 7.2 s in casual_male, about 3 words a second.
- A pitch check (40 ms frames, autocorrelation; `pitch_ctl.py`) measured the share of consecutive voiced frames that hold one pitch to within half a semitone. Lyrics gave 0.64 to 0.72; ordinary spoken sentences from the earlier run gave 0.52 to 0.76. Held notes would push the lyric figures well above the spoken ones, and they do not.
- Nobody listened to these files in this run. The measurements above rule out held notes; they cannot rate how natural the speech sounds. The files are in the saved folder for a person to listen to.
- Licence for these files: CC BY-NC 4.0, non-commercial use only, as for every output of this bundle.

## What this run does not show

- The cause of the 7 s stalls.
- Per-stage TTS timing (the log does not carry it).
- Behaviour above 32 simultaneous TTS requests or with texts longer than 71 s of audio.
- Mini with audio longer than the 3.4-minute clip, or in languages other than English.
- Firmware 19.17.1 and tt-kmd 2.12.0, as in the earlier run.
