# ACE-Step 1.5 turbo as a v6 thin bundle: the second can (2026-10-30)

Goal (operator): "v6 thin bundles for all these". `tt-model serve episod/<package>` refuses the recipe packages because they have no `tt_kernel_manifest.json`. Granite was the first can (`docs/run-logs/2026-10-30-granite-thin-bundle.md`). This is the second, for the audio model `ACE-Step/Ace-Step1.5` (revision `19671f40`), recipe package `episod/ace-step-1.5-turbo-blackhole` 0.2.0. Code: `config/can.acestep.toml`, `pkgsrc/server/acestep_asgi.py`, `orchard/can.py` (the audio boot check). Pattern: `docs/canning.md`. Raw timings: `docs/run-logs/evidence/acestep-can/`.

## What was built

- `pkgsrc/server/acestep_asgi.py`: ASGI front end with `GET /health` and `POST /v1/music`. It imports `validate` and the limits from the unchanged `scripts/acestep/acestep_serve.py` (shipped as `extra_code/acestep_serve.py`) and makes the clip with the unchanged `acestep_run.Session`. The chip opens in the lifespan start-up. The clip is made in a worker thread, which releases the one-request slot in a `finally`. A busy request gets 503 at once; a chip call that failed (`session.worker.failed`) gives 503 on `/health` and on requests. A client that leaves does not stop the clip (ACE-Step has no way to), and the slot is free when the clip ends. 36 tests, each guard checked by a mutation (23 mutations, all turned a test red; the first run had 6 survivors, which were fixed with tests for the busy answer arriving fast, an unusable chip while the slot is taken, a chip that fails between the first look and the lock, an over-size valid body and the empty-body message).
- Model code and weights are fetched at start-up at pinned revisions (`pkgsrc/can/acestep-pins.json`, the recipe package's `pins.json`): the 1,307-file code repository as real files (`snapshot_download(local_dir=...)`, issue 270) and the turbo files of the weights repository. The code directory is compared with a tree hash and refused on a mismatch. `Session` keeps its private checkpoint tree (issue 264).
- `orchard/can.py`: `[check] mode = "audio"` (path, body, expect_seconds, sample_rate, channels, min_loud_fraction, max_warm_s, repeat_identical), `wav_facts` (standard library WAV reader for 32-bit float and 16-bit samples), and `can boot --port`. 34 tests in `tests/test_can_audio.py` against a stand-in server, 18 mutations, all red after two tests were added (a very quiet clip, a clip with a non-finite sample).
- The bundle: 17 files, 184 KB, no weights, no ACE-Step code, no venv. Manifest schema 6, kind `tt-dit-server`, app `acestep_asgi:app`, weights `ACE-Step/Ace-Step1.5` at `19671f40`, mesh P150, 1 device, 38 exact pins (`ttnn==0.80.0` from PyPI). The card is `pkgsrc/can/acestep-README.md`, written in the section names of the catalog cards (`episod/clef-p300`) and checked with `tt-model-ranker` (read only).

## What went wrong on the way

| Found | Action |
| --- | --- |
| install.sh failed: `torchmetrics==1.8.2` is not on the PyTorch CPU index, and uv takes a package from the first index that lists it | pinned `torchmetrics==1.0.3` and `setuptools==70.2.0` (the versions that index holds; `setuptools` must stay below 72) |
| First boot: the chip opened and the models loaded, then `peft>=0.17.0 is required ... found peft==0.15.1` (transformers 4.57.6 asks for it); the failed start-up left the chip open and the process aborted at exit | pinned `peft==0.21.2` (the bundle's lock file value); the check ran in the probe venv with `uv pip check` |
| Second boot check run: all checks passed except `repeat_identical`. The four WAV files of four identical requests differ byte for byte; all samples are equal (correlation 1.0, largest difference 0.0) | the check compares the hash of the sample data |
| The first hardware attempt failed with "failed to initialize FW" at device open, after a `gozer run` had been cut short. The chips were left `STALE`; `gozer reconcile` and a `gozer reset` of a new lease cleared it | none needed in the can; noted so the next agent runs `gozer reconcile` first |

## Hardware (board 0, chips 0 and 1; one chip opened; leases through gozer; no chip wedged)

Conditions: 2026-10-30, `ttnn` 0.80.0 from PyPI, torch 2.7.0 CPU, numpy 1.26.4, Python 3.12, host load average 15 to 37 (other agents). The staged directory was never installed into or served: every run used a copy or tt-model's own install.

1. `tt-orchard can boot` on a fresh copy with an empty kernel cache (and a work directory with nothing fetched): passed 16 of 16 checks. Ready in 306 s (it downloads the 1,307 code files and loads the models), first 10 s request 93.7 s, warm 10 s request 1.50 s, 48 kHz, 2 channels, 10.00 s, 89 percent of windows above -40 dBFS, same samples twice. An earlier cold run: ready in 108 s, first request 216.8 s.
2. The same copy restarted (warm kernel cache): ready in 32 s, first 10 s request 5.7 s, then 1.13, 1.49 and 1.18 s; 30 s clips 1.76 to 1.96 s (n=4). Load average 19.
3. tt-model's own installer (`tt_kernel.cli._install_self_contained` into a scratch directory and index) and `tt-model serve local/ace-step-1.5-turbo-blackhole --local-only --port 20143`: ready in 310 s on a fresh install, first 10 s request 121 s, then 0.92 and 0.92 s; 30 s clips 1.25, 1.20 and 1.85 s. The samples equal those of run 2 (correlation 1.0).
4. Recipe package in the same hour (`run.sh`, its own venv with the source build of ttnn and numpy 2.5.3): ready in 140 s, first request 86.8 s, 10 s clips 1.06 to 1.34 s, 30 s clips 1.60 to 3.54 s at load average 30 to 36. Its card gives 0.93 s for a 30 s clip at load average 2 to 6.
5. Agreement with the recipe package, same prompt and seed: waveform correlation 0.990 at 10 s and 0.983 at 30 s, relative L2 0.14 and 0.19. The samples are not identical. The two environments differ in the ttnn build (PyPI wheel against a source build), numpy (1.26.4 against 2.5.3), transformers, peft and other versions; which one causes the difference was not isolated. For scale, the recipe card's bfloat16 CPU run against the float32 CPU run has waveform correlation near 0.89 to 0.92.

## Against the recipe package

Same: endpoints, request checks, the generation code, the busy and 400 behaviours, the chip steps, 48 kHz stereo output, the same samples for the same request within one server, speed within the noise of a loaded host (warm 10 s clip about 1 s in both).
Different: (a) the server is ASGI and its settings are environment variables (no flags); (b) code and weights are fetched at start-up into the bundle's own cache directory (the recipe has a separate `run.sh fetch`); (c) the venv is built from exact pins with numpy 1.26.4 and the PyPI ttnn wheel; the output differs from the recipe's by correlation 0.98 to 0.99 (item 5 above); (d) the recipe's `check_bundle.py` runs at start-up inside the adapter and a mismatch stops the start; (e) no `SECURITY.md`, evidence directory or `tt-model.yaml` draft in the bundle; (f) the port default is 20140 (the recipe's was 20120); `serve.py`'s port refusals do not apply to uvicorn.

## Card

`pkgsrc/can/acestep-README.md` is the staged card (not published). `tt-model-ranker` (`python -m card_viewer CARD --no-infer`) reads it as an audio card for p150, p100a and p300c and reports "window grade n/a no hardware with numbers": the viewer has no grader for audio jobs yet and reads speed numbers in tokens per second only. It names `p100a` because the code repository's name contains it. Nothing on the card was changed to chase a grade.

## Not done

- No upload, no `tt-model push` or `publish`, no merge. `tt-model serve episod/ace-step-1.5-turbo-blackhole` still fails until the operator publishes the can (`tt-orchard can commands config/can.acestep.toml --namespace ORG`).
- Clips longer than 30 s through the bundle; agreement with the CPU reference under the new pins; the cause of the 0.98 to 0.99 difference from the recipe package.
- Start-up failure handling: when `Session.__init__` raises after the chip opened, the chip stays open until the process exits (the exit aborts with a UMD exception). The Session class is shared with the recipe package and was not changed.
