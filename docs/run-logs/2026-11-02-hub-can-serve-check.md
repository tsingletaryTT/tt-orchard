# Hub serve check of the six published cans (2026-11-02)

Goal (operator, issue 24): confirm that someone else can pull the six cans we published to the Hub and serve them with
`tt-model serve episod/<name>`. Each can was pulled into an empty directory, served on a leased board, sent one real request of its
kind, and stopped. The card of each repository was the source for the serve command, the port and the expected numbers.

Granite had been served from the Hub once before. The other five had not.

## Result

Five cans serve from the Hub and answer a real request. The Stable Audio can fails at start-up when `HF_TOKEN` is the only
credential (issue 408). It serves once the Hub client can read a token file.

| Can | Revision (Hub, 12 chars) | Profile | Start (start to health 200, empty kernel cache) | Request | Measured against the card | Verdict |
|---|---|---|---|---|---|---|
| `granite-4.0-h-1b-blackhole` | `4e7a5427d539` | one chip (p150 mesh) | 303 s with the 3 GB weights download (load not recorded, 70 to 80 by `uptime`); 99 s on a quiet host (load 8.5), weights downloaded again | chat, 3 streamed answers of 64 tokens; the text is a coherent paragraph on mountain building; first request 19.0 s (cold kernels) | decode 32.0 tok/s (median of 3, load 8.5) against about 34; time to first token 0.48 s against 0.32 s. At load 75 to 80 the same can (second start, warm kernel cache) gave 8.7 tok/s | pass |
| `minicpm3-4b-blackhole` | `21b824d3d58e` | p150 | 640 s including the 8 GB download and conversion (load 99); card: 68 s with the weights on disk | chat, 41-token answer, then 3 streamed answers of 64 tokens; coherent; first request 2.6 s | decode 27.5 tok/s (median of 3, load 95 to 99) against 32.7; 0.84 of the card | pass (speed taken under a load of 95) |
| `ace-step-1.5-turbo-blackhole` | `072a117b5f9d` | p150 | 856 s including 6.5 GB of weights and 1,307 code files (load 58); card: 306 to 310 s | `POST /v1/music`: 10 s clip (first, 356 s), then 30 s twice (5.9 s and 4.4 s) | 30 s clip in 4.4 s warm, 0.148 s per second of audio, against 1.2 to 1.96 s (0.04 to 0.07). WAV 48 kHz stereo, 10.000 s and 30.000 s long. 89 percent of 100 ms windows above -40 dBFS at 10 s (card: 89 percent); peak 0.89; the two 30 s clips have identical samples | pass (speed taken under a load of 58 to 70) |
| `stable-audio-3-small-music-blackhole` (private repo) | `45db29f00f38` | p150 | failed twice with `HF_TOKEN` only (24 s and 159 s). With `HF_TOKEN_PATH` naming a token file: 105 s (load 95); card: 19.5 s | job server: 10 s clip (first, 213 s), 10 s again (20.5 s), 30 s (92 s). FLAC 44.1 kHz stereo, 10.000 s and 30.000 s long | 10 s warm clip in 19.7 s of server time, 1.97 s per second of audio, against 9.89 s (0.99). Load was 95 | fail as published (issue 408); pass with a token file |
| `qwen3.5-35b-a3b-blackhole-p300` | `710d5bcf2bed` | p300 (both dies of one board) | 4,553 s: the 72 GB download, loading 40 layers, and compiling kernels (load 11); card: 2,052 to 2,334 s with the weights on disk | chat with thinking off: a 96-token answer on the primary colors, then 3 streamed answers of 64 tokens; coherent | decode 125.66 tok/s (median of 3: 125.76, 125.66, 123.31) against 125.75; time to first token 127 to 146 ms against 102 to 136 ms | pass |
| `minimax-music3-fiona-crapple-blackhole` | `070e8b333593` | p150 | 1,417 s including the 27 GB download and building the tensor caches (load 58); card: 513 s with the weights on disk | job `POST /v1/music/jobs`, WAV: 5 s clip (90 s), then 10 s clip (169 s). 32 kHz stereo, 4.992 s and 9.996 s long | 16.9 to 18.0 s per second of audio at load 58 to 63, against 4.2 to 5.7 s at load 20 to 30. 82 and 64 percent of windows above -40 dBFS (card, 5 s boot clip: 82 percent); peak 1.0 | pass for serving and output; speed 3 to 4 times the card's time per second of audio, under load |

Every revision above equals the revision the Hub reported for the repository when I listed it, and equals the `revision` that
`tt-model` recorded for the install. The speed numbers are not like for like with the cards unless the host load matches: each row gives the load at the time
(the one-minute load average on 16 cores, shared with other sessions). Qwen was measured at a load of 11 and is within 0.1 percent of its card. Granite was measured twice,
at load 75 to 80 and at load 8.5, and the two decode speeds differ by a factor of 3.7. For that reason the speed numbers taken at a load of 58 or more carry no verdict on speed.

## What failed

### Stable Audio: 401 from the tokenizer when the only credential is `HF_TOKEN` (issue 408)

`tt-model serve episod/stable-audio-3-small-music-blackhole` with `HF_TOKEN` set in the environment (an account that has accepted the
model's terms; the token reads the private repository and downloads the 17 weight files) exits after 24 s with:

```
ERROR:    OSError: You are trying to access a gated repo.
401 Client Error.
Cannot access gated repo for url https://huggingface.co/stabilityai/stable-audio-3-small-music/resolve/8bdfb53c6b0fbc62113cd571fd37373efe15fa14/t5gemma-b-b-ul2/added_tokens.json.
ERROR:    Application startup failed. Exiting.
```

Evidence: `evidence/hub-can-serve-check/stable-audio-3-small-music-blackhole-run1-fail.json` (exit code 3, last 60 log lines, paths and request
ids redacted). The first attempt took 159 s because it also downloaded the weights; the second took 24 s and failed the same way.

What I tried, in order:

1. Confirmed the token has access: `hf_hub_download` of the same file at the same revision with the token gives 404 (the file does not exist), which shows the token passes the gate.
2. Confirmed the server process has the token: its environment holds `HF_TOKEN` (checked by key name only).
3. Reproduced it without a board, in the bundle's own venv: import `huggingface_hub`, point `HF_HUB_CACHE` at the private cache the server builds,
   run `offline_environment()` from `orchard/skills/audio-reference-templates/make_reference.py`, then load the tokenizer. With no token
   file: the same 401. With a token file readable (`HF_HOME` with `token`): it loads. A probe that I ran with the token present first wrote
   `.no_exist` markers into the private cache, which hid the fault until I removed them; a later start on a machine with those markers passes.
4. Re-ran the real serve with `HF_TOKEN_PATH` pointing at the operator's existing token file (the file is read by the Hub client; I did not copy
   or print it). It started in 105 s and made clips (table above).

Cause: `offline_environment()` removes the token variables and sets `HF_HUB_OFFLINE=1` after `huggingface_hub` was imported. The client reads
that variable once at import, so the process stays online and the tokenizer's lookups for optional files go out with no token. The Hub answers 401 for
a gated repository. The card's own sessions passed because the author's machine had a token file. The fix options are in issue 408.

I did not write a token file into the isolated cache: the sandbox refused that action, and the check does not need it.

## What each can did when it worked

- Granite, MiniCPM3, Qwen3.5: the reply was a coherent paragraph or list that answers the prompt (the three primary colors; how a mountain range forms). The three streamed answers of Granite are identical, as are MiniCPM3's (greedy decoding). Qwen's three differ in wording, so its default request samples; the check did not set a temperature.
- ACE-Step: three clips. `ffprobe` reports `pcm_f32le`, 48,000 Hz, 2 channels, 10.000 s and 30.000 s. The two 30 s clips decode to identical samples (sha256 of the f32 stream equal). All samples are finite.
- Stable Audio: three FLAC clips at 44,100 Hz, 2 channels, 10.000 s, 10.000 s and 30.000 s; the two 10 s clips have the same size (915,718 bytes). Peak is 1.0 in all three; the card gives no level.
- Fiona: two WAV clips, 32,000 Hz stereo, 4.992 s of 5 s and 9.996 s of 10 s asked. A song can end when its lyrics end; the card says so.

Nobody listened to any clip. The check measures duration, level, silence and finiteness, which the cards also limit themselves to.

## Writes outside the can's directory

Each can ran with `XDG_CACHE_HOME`, `TT_MODEL_MODELS_DIR`, `HF_HOME` and `TT_METAL_CACHE` set to directories under one root per can. `tt-model pull`
put the venv, the bundle and its install index there, and `run.sh` wrote its weights and caches there (the weights of Granite, MiniCPM3 and Qwen went to
`HF_HOME` inside the root).

After the last run I listed every file under `~/.cache`, `~/.local`, `~/.config`, `~/.tenstorrent-venv` and `~/tt_metal_cache` that was modified after the start of the check and
whose path names any of the six cans: none. The listing is by path name and other sessions write in the same directories, so it rules out a write that names the can and not
one that does not. The operator's own `tt-model` index changed during the check; its only entry for these cans is the Qwen3.5 bundle that it already listed before
this check (the check used its own index).

With default settings, `tt-model pull` writes its index and the bundle under `~/.cache/tt-model`. That is its documented location and was not tested here.

## Pull

`tt-model pull` (outside the lease, 22 to 155 s per thin bundle, 729 s for Qwen with its three wheels) builds the venv and installs the pinned requirements. The five thin bundles
that use the PyPI `ttnn` 0.80.0 print `SFPI at /opt/tenstorrent/sfpi is not the version TTNN declares (want 7.85.0, found 7.89.0); continuing` and then work. The Qwen bundle
carries its own SFPI 7.74.0 in its `ttnn` wheel (its card says its kernels do not compile with 7.89.0, which this host has in `/opt`); its log says `validated SFPI 7.74.0` from the
wheel's own directory and prints no warning.

## Gozer

One lease per run, taken with `gozer acquire --owner-pid <a long-lived sleep I started>`, as the supervisor does. A request for one chip granted the whole board
(chips 0 and 1) each time. The other board was leased by another session throughout. `gozer release` reset my chips each time. Three times the release answered
`chips [0, 1] still open by [pid]` for a pid that did not exist at the next look; retrying after 5 s worked, so the retry loop is in the commands below. Several
acquires queued for under a minute while something outside my lease touched the chips, then were granted by `gozer wait`.

## Commands

The driver is `scripts/hub_can_check.py` (new in this change). It starts `tt-model serve episod/<can> --port <the card's port> --no-update-check` in its own process group
with the four cache variables above, polls the health route (timeout stated per can in the result files), sends the can's requests, decodes the audio with `ffprobe` and `ffmpeg`, and stops the server by the exact pid it started (SIGTERM to the group, SIGKILL after 60 s).

```
ROOT=<empty scratch directory>
python3 scripts/hub_can_check.py <can> --root $ROOT --pull-only                       # tt-model pull in the isolated environment
python3 scripts/hub_can_check.py <can> --root $ROOT --print-env                       # the exports, to repeat a pull by hand

python3 -c "import subprocess; p = subprocess.Popen(['sleep','7200'], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print(p.pid)"
gozer acquire --chips 1 --who claude:hw1-cans --reason hub-can-serve-check --owner-pid <that pid> --expect 30m --json
gozer wait --timeout 300 <ticket>                                                    # when the acquire queued (exit 10)
TT_VISIBLE_DEVICES=<the chips the lease reports> python3 scripts/hub_can_check.py <can> --root $ROOT
TT_VISIBLE_DEVICES=... python3 scripts/hub_can_check.py stable-audio-3-small-music-blackhole --root $ROOT --hf-token-path <existing token file>   # second Stable Audio run
for i in 1 2 3 4 5 6; do gozer release <lease> | tail -1 | grep -q "still open" && sleep 5 || break; done
kill <that pid>
```

`<can>` is each of the six names, in this order: granite-4.0-h-1b-blackhole, minicpm3-4b-blackhole, ace-step-1.5-turbo-blackhole,
stable-audio-3-small-music-blackhole, minimax-music3-fiona-crapple-blackhole, qwen3.5-35b-a3b-blackhole-p300 (Qwen with `--ready-timeout 6000`).
The cards were read with `huggingface_hub` from `python3 -I` (text files only). The result of each run is in `evidence/hub-can-serve-check/<can>.json`.

## What was not checked

- The agreement check of issue 24 (the same agreement check as the pre-upload boot check, run on the pulled copy), and the three packages that issue names (`clef-p300`, `rag-gate-27b-p300`, `qwen3.8-27b-pi-p300`).
- `tt-orchard verify-package` on a pulled copy (issue 271).
- The 503 reply while a request runs, long clips, concurrency, and listening to any clip.
- Speed on an idle host for MiniCPM3 (load 95), ACE-Step (58 to 70), Stable Audio (95) and Fiona (58 to 63). Granite (8.5) and Qwen (11) were measured on a quiet host and match their cards.
- The ACE-Step response header `X-Timings`: the driver looked it up by its mixed-case name and the server sends it in lower case, so it was recorded as null. The driver now lower-cases header names.
- A start with `tt-model`'s default index and cache directories.
