# ACE-Step 1.5 turbo: a local weights-free package, gate and chip boot check (2026-10-30, issue 254 step 3)

Operator instruction: continue the ACE-Step bring-up; step 3 is a weights-free local package built like the Granite and audio packages, gate-clean and boot-checked on a chip.
Paths are placeholders. The staged package is in `~/orchard-packages/acestep-1030/ace-step-1.5-turbo-blackhole` on the machine that ran this (200 KB, 24 files). Nothing was published.

## What the package is

An HTTP server (`runtime/acestep_serve.py`) that keeps the diffusion transformer and the VAE decoder loaded on one chip and makes a clip per request (`POST /v1/music`, `GET /health`),
plus `tools/smoke.py` (a request and the sound checks at 48 kHz, standard library and numpy), `tools/check_bundle.py`, `run.sh` (serve, fetch, check, --self-test), the card, `LICENSE`
(Apache 2.0), `NOTICE`, `SECURITY.md`, `tt-model.yaml` (a draft, kind `http-server`), `requirements.txt`, `pins.json`, `package-record.json`, `SHA256SUMS` and `evidence/`.

It does not contain the weights (MIT, `ACE-Step/Ace-Step1.5` revision `19671f40`) or the bundle code (`sunandbean/acestep-1.5-turbo-p100a` revision `dabe15cd`, Apache 2.0: ACE-Step's Python, tt-metal's `models/` and the chip port).
`pins.json` holds a hash over the 1,307 bundle files and the size and sha256 of each weight file; `check_bundle.py` compares them and `run.sh` runs it before the server starts.
The module is `orchard/acestep_package.py` (stage, summary, gate), modelled on `orchard/granite_package.py`. The card's numbers are produced from `evidence/summary.json` and the gate looks for each row in the card.

## Gate

`python -m orchard.acestep_package gate PKG` gave 0 problems on the staged package. `tests/test_acestep_package.py` (13 tests) stages a package from synthetic evidence and checks each damage:
a weight file, an edited card number, a changed runtime file (record and sums), changed pins, a wrong licence in the front matter, a home path and a key-file name, an overclaim, a failed boot or refused bars,
a link, a ready-line marker missing from the server. Six gate stages were each removed in turn and a test went red each time (pins, record, card, sums, scrub, manifest).

## Boot check on a chip

One chip of board 1 (chips 2 and 3 leased through gozer with `--owner-pid` on a long-lived pid), the staged package copied to a new directory:
- `run.sh fetch` downloaded the bundle code as real files into `fetched/bundle` (26 s; the weights were already in the cache). `run.sh --self-test` passed (bundle hash and sizes, runtime imports).
- `run.sh` started the server: ready line after 16 s (polled every 2 s; the server reported 12.7 s of its own load with a warm kernel cache).
- A 30 s request (the lo-fi prompt, seed 42) returned in 3.3 s and passed the 48 kHz sound checks. A second request gave identical samples. The samples are identical to the clip recorded in the first bring-up run (maximum absolute difference 0.0).
- A 60 s folk clip with lyrics (seed 7) returned in 4.3 s and passed the sound checks.
- The server's 400 replies for a bad duration and an unknown field were checked by hand. The 503 reply for a busy server is covered by a unit test only, not on the chip.

## A finding: the bundle code must be a directory of real files

The first boot used the bundle code from the Hugging Face cache (a directory of links to blobs). The clip it made was different music (correlation 0.65 against the recorded chip clip; the CPU clip correlates 0.94 with the recorded chip clip).
Cause: ACE-Step finds its own model code with `Path(__file__).resolve().parent / "models"`. With links, `resolve()` leaves the directory, the check that copies ACE-Step's model code into the checkpoint finds no source directory and does nothing,
and the model repository's own copy of the model code runs. There is no error or warning. With real files the sync runs and the clip matches the recorded one exactly.
So an earlier statement that the bundle "ran with the weights as downloaded" needs this qualification: results depend on which copy of the model code runs. The guards added: `check_bundle.py` refuses a code directory with links, the runner (`Session`) refuses to start
when the model-code directory is not found, `run.sh fetch` uses `--local-dir`, and the card says so. Not checked: which of the two model-code versions makes better music, or whether the Hub's copy is the older or the newer one.

## Runner change

`scripts/acestep/acestep_run.py` now has a `Session` class (load once, `one()` per clip) that `main` and the server share. A regression run of the CLI on the chip against the first bring-up's recorded run gave identical noise, conditioning, step-0 velocity and waveform (maximum absolute difference 0.0).
Module lookup of `orchard/` works from the repository and from the package.

## Not done

- The package was not pulled from a Hub repository and booted a second time (it was never uploaded). Only a local copy was booted.
- `tt-model.yaml` is a draft; no container build was tried.
- No second chip, no listening judgment, no XL transformer or planner (step 4).
- The sound checks measure length and level. Clips to listen to (48 kHz stereo WAV, made through the package server): `orchard-audio/acestep-2026-10-30/package-server-lofi-30s-seed42.wav` and `package-server-folk-60s-seed7.wav` under the home directory of the machine that ran this.
