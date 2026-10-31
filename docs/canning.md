# Canning: ship a brought-up model as a v6 thin bundle

Another person cannot install a model just because it runs on the chip. The canning step (`orchard/can.py`, `tt-orchard can`) turns the pieces of a brought-up model into a **v6 thin bundle**: a directory with `tt_kernel_manifest.json`, `install.sh`, `run.sh`, pinned `requirements.txt`, the model's code and a card. `tt-model` installs such a bundle into its own venv and serves it. `tt-model serve ORG/NAME` refuses a package that has no `tt_kernel_manifest.json`, which is why the recipe packages (a `run.sh` plus a draft container `tt-model.yaml`) could not be served by it.

The first can is Granite-4.0-H-1B (`config/can.granite.toml`). It was staged, installed in a copy and served on one chip. The numbers are in `docs/run-logs/2026-10-30-granite-thin-bundle.md`.

## What you write

One spec file (TOML), and for a model whose server is not an ASGI app, a thin ASGI adapter.

```toml
[can]
name = "granite-4.0-h-1b-blackhole"
kind = "tt-dit-server"            # or "vllm"
app = "granite_asgi:app"          # kind tt-dit-server: module:attribute that uvicorn serves
model_py = "pkgsrc/can/granite_model.py"     # the runner file; copied to the bundle root. Nothing imports it for this kind
readme = "pkgsrc/can/granite-README.md"      # the card, copied as it is
mesh = "P150"
device_count = 1
port = 20110                      # the default port in run.sh; never 8000, 8011, 8012, 8780, 8100-8103
requirements = ["ttnn==0.80.0", "numpy==1.26.4", ...]   # every line name==version
verify = ["import granite_asgi"]  # run at the end of install.sh; must not open a device

[weights]
repo = "ibm-granite/granite-4.0-h-1b"
revision = "<40-character commit>"

[[extra_code]]                    # code that goes to extra_code/, which run.sh puts on PYTHONPATH
src = "pkgsrc/runtime/granite_chip.py"
dst = "granite_chip.py"

[[bundle_file]]                   # files at the bundle root (LICENSE, NOTICE, tools)
src = "LICENSE"
dst = "LICENSE"

[check]                           # what the boot check asks of the served bundle
expect = "Paris"
min_decode_tok_s = 30.0
ready_contains = ['"chunked_prefill": true']
```

Paths are relative to the directory the command runs in (`--base` changes that). A source must be a regular file inside it.

## The commands

```
tt-orchard can stage config/can.granite.toml --out OUT/granite-4.0-h-1b-blackhole
tt-orchard can gate  OUT/granite-4.0-h-1b-blackhole config/can.granite.toml
tt-orchard can boot  OUT/granite-4.0-h-1b-blackhole config/can.granite.toml --work WORK [--hf-home DIR]
tt-orchard can commands config/can.granite.toml --namespace ORG     # the publish commands, as text
```

Add `--run-dir RUN` to `stage` or `boot` to write `RUN/stages/7/can.json` and an `evidence` entry (`can staged`) in the run's ledger.

`stage` runs `tt-model package-thin --out` (never with a repo id; stage 7's `assert_no_publish` refuses any upload-capable call), then:

- replaces `producer.hostname` in the manifest and makes `run.sh` and `install.sh` executable;
- changes `run.sh` to bind uvicorn to 127.0.0.1 (package-thin writes 0.0.0.0) and to default to the spec's port (package-thin writes 8000); `HOST` and `PORT` still override;
- writes `SHA256SUMS`;
- runs the gate.

The gate (`can gate`) reuses stage 7's package scrub. It reports any weights file, cache, venv, link, home path, host name, token or namespace outside the card; a hash in `SHA256SUMS` that no longer matches; a manifest that `tt_kernel.manifest.Manifest.from_json` refuses or that disagrees with the spec; a `requirements.txt` line that is not an exact pin; a missing server module; a Python file that does not compile.

`boot` copies the bundle, runs the copy's `install.sh`, starts the copy's `run.sh` in its own session, waits for `/health`, sends one chat request and one streamed request, checks the answer against `[check] expect`, the decode rate against `min_decode_tok_s` and the start-up log against `ready_contains`, and stops the server by process group. The staged directory is never installed into or served. **A boot check opens the device, so run it under a lease:**

```
gozer run --chips 1 --who "claude:can" --reason "boot check of a can" --expect 20m -- \
  python bin/tt-orchard can boot OUT/NAME config/can.NAME.toml --work WORK
```

The install step does not need a lease. Run `install.sh` in the work copy first if you want it done before the lease (the boot check skips an install that is already there).

## Which kind

- **`vllm`**: the model runs behind vLLM through the tt-vllm plugin (a model class that the plugin loads, a `vllm_metadata.json`, a plugin wheel). Qwen3.5-35B is this case (`orchard/qwen35_package.py` already stages a plugin-based package, and stage 7's `orchard/package.py` shows the same `package-thin` call for a weights-swap model). The spec gives `metadata`, `plugin_wheel`, `ops_wheels`, `models_wheels`, `vllm_version`, `max_num_seqs` and `max_model_len`, and `model_py` is the file with the model class. The gate checks that every shipped wheel is byte-identical to the file the spec names. The vLLM argument handling of stage 7 (fixed extra arguments spliced into `run.sh`, weights wiring) is not in the can: add what the model needs to its spec, or extend `fix_run_sh`. No can of this kind has been built on hardware yet; the call is covered by a test only.
- **`tt-dit-server`**: `run.sh` serves an ASGI app with uvicorn. Use it for every model with its own server: Granite, MiniCPM3 (`scripts/minicpm3/serve.py`, a standard-library server), ACE-Step (`scripts/acestep/acestep_serve.py`), and the Gradio demo of minimax-music3-fiona-crapple (a Gradio app is an ASGI app: mount it with `gradio.mount_gradio_app` on a FastAPI app and name that app in `[can] app`).

The kind is not a statement about the model. It says which front end `tt-model` starts.

## Writing the ASGI adapter

Keep one copy of the request and generation code. `pkgsrc/server/granite_asgi.py` imports `parse_request`, `encode_prompt`, `generate`, the lock and the state from `serve.py` (shipped as `extra_code/granite_serve.py`). It adds only what the standard-library server did with `http.server`: routes, status codes, the event stream, and the settings.

Put the adapter in `extra_code/` (a `[[extra_code]]` entry) and keep the bundle root free of it. Make `model_py` a small stub that re-exports the app (`pkgsrc/can/granite_model.py`). `install.sh` runs each `verify` statement with only `extra_code/` on the path, so a module at the bundle root cannot be imported there. The first install of the Granite bundle passed when run from inside the bundle directory and failed under tt-model's own installer for this reason; the gate now reports it.

Points that were easy to get wrong:

- The device opens in the ASGI lifespan startup and closes in the shutdown. uvicorn does not accept connections until startup returns, so `/health` answers only after the model is loaded.
- Chip work runs in a worker thread, so `/health` answers while a request generates. Take the one-request lock without waiting and answer 503 at once when it is held. Release it in the worker, in a `finally`.
- A streaming client that leaves shows up as `http.disconnect` on `receive()`. Watch for it and stop the generation, or the slot stays taken.
- uvicorn passes no flags. Read the settings from environment variables, one per flag, and list them on the card.
- Resolve the weights in the app: `HF_MODEL` and `TT_MODEL_WEIGHTS_REVISION` are set by `run.sh`; use `snapshot_download(HF_MODEL, revision=...)`. For `tt-dit-server`, tt-model does not prefetch weights.

## Pins

`install.sh` runs `uv pip install -r requirements.txt` with the PyTorch CPU index added. Pin every line (`name==version`); the gate refuses anything else. ttnn 0.80.0 from PyPI requires `numpy<2`, so a model that ran on a source build with numpy 2 needs a numpy 1.x pin and a check that its code runs with it. The base HTTP stack (`fastapi`, `uvicorn`, `pydantic`, `pillow`) is added by package-thin when missing, without pins; list all four with pins yourself so the gate passes.

## What the can does not do

It does not publish. `can commands` prints text; the supervisor and the stage agents never run it. It does not make `tt-model serve ORG/NAME` work before the bundle is on the Hub, because tt-model installs a v6 bundle only from the Hub. The boot check serves the bundle's own `run.sh`, which is what `tt-model serve` runs after the install.

## MiniCPM3-4B (`config/can.minicpm3.toml`)

Another can after Granite. It needed four things that Granite did not.

- **The standard-library server did its request checks inline.** `scripts/minicpm3/serve.py` read the body, checked the messages and built the prompt inside `Handler.do_POST`. The adapter needs the same checks without an HTTP handler, so `serve.py` now has `parse_request`, `encode_prompt` and `generate_ids`, and its handler calls them. `import ttnn` moved into `main()`, so the install check `import minicpm3_asgi` loads no device library. `generate_ids` takes a `should_stop` callable that it checks before every chip step. The prompt goes through the chip one token per step (about 30 ms each), so a client that leaves during a long prompt would otherwise hold the slot for the whole prompt.
- **The weights are a PyTorch pickle** (`pytorch_model.bin`, 8 GB) and the chip model reads one safetensors file. The adapter downloads the pinned revision, then converts once with `convert_weights.convert` into `tt-orchard-minicpm3` inside `HF_HOME` (or `MINICPM3_CACHE`). The cache file name holds a hash of the snapshot path. The pickle is read with `weights_only=True`.
- **The server reported no timings.** The boot check reads `timings.decode_tok_s`, so the adapter adds `timings` (seconds to the first piece, decode tokens per second) to the non-streaming answer.
- **A failed chip step is not recoverable.** The recipe's own boot check found that a chip that timed out once failed every later request. The adapter marks the server unusable after a failure inside a chip step: `/health` and every later request answer 503 until restart.

The adapter, `chip_model.py`, `ref_model.py`, `convert_weights.py` and `serve.py` (as `minicpm3_serve.py`) go to `extra_code/`. The pins are Granite's plus `transformers==5.19.0`; the PyPI wheel `ttnn==0.80.0` with numpy 1.26.4 ran the model unchanged (the recipe ran on a source build with numpy 2.5). Settings are `MINICPM3_*` environment variables. Greedy decoding only, default context 2048, no batched prefill and no two-chip variant, as in the recipe package. Numbers and the comparison with the recipe package are in `docs/run-logs/2026-10-30-minicpm3-thin-bundle.md`.


## ACE-Step 1.5 turbo (`config/can.acestep.toml`, `pkgsrc/server/acestep_asgi.py`)

The second can, for a model that returns audio. What it needed that Granite did not:

- **An audio boot check.** `[check] mode = "audio"` sends `path` and `body` twice (the first request compiles kernels, the second is timed) and checks the WAV: sample rate, channels, length (within 5 percent), share of 100 ms windows above -40 dBFS (`min_loud_fraction`), full-scale samples, finite samples, optionally the warm time (`max_warm_s`) and that the repeat has the same samples (`repeat_identical`). The WAV is read with the standard library (`can.wav_facts`). Compare the samples: ACE-Step's WAV files differ byte for byte between identical requests while every sample is equal. `can boot --port N` picks the port.
- **Code and weights fetched at start-up, at pinned revisions.** The model code is a 1,307-file Hugging Face repository (`sunandbean/acestep-1.5-turbo-p100a`), not shipped in the bundle. The adapter downloads it with `local_dir` (real files; links into the cache make ACE-Step run other code without an error, issue 270), compares a tree hash with `acestep_pins.json` and refuses a mismatch. The weights come from the pinned revision of the weights repository. `acestep_run.Session` keeps its private checkpoint tree under the work directory (issue 264), so nothing writes into the Hugging Face cache.
- **No stop for a request in flight.** ACE-Step cannot stop a clip half way, so a client that leaves frees the slot only when the clip ends. The worker thread releases the slot in a `finally`; a test covers the disconnect.
- **Pins that the PyTorch index can break.** install.sh gives uv the PyTorch CPU index as an extra index without `--index-strategy unsafe-best-match`, so uv takes a package from the PyTorch index when that index lists it at all. A pin that is not on that index fails the install. `torchmetrics` (1.0.3 only) and `setuptools` (70.0.0, 70.2.0, 78.1.0) hit this; `numpy`, `pillow` and `networkx` happened to be there at the pinned versions. Check each pin against `https://download.pytorch.org/whl/cpu/<name>/` before staging, or run install.sh in a copy first. Dependencies also have lower bounds that a pin set can miss: `transformers` 4.57.6 needs `peft` 0.17 or newer (the first boot failed on `peft==0.15.1`, after the chip had opened, so the failure shows up in the boot check and not in install.sh).
- **A failed start-up leaves the chip open.** If `Session.__init__` raises after the chip opened, the process aborts at exit with a UMD exception and the board needs a reset. Take the lease with `gozer run` or reset the lease afterwards.
## Second can: minimax-music3-fiona-crapple (`config/can.fiona.toml`), a music model with a Gradio page

What it needed that Granite did not:

- **No adapter for the model, one small one for the page.** The model's server is already a FastAPI app (`models.autoports.minimaxai_minimax_music3.server.app:app`). `pkgsrc/music3/fiona_asgi.py` imports it and mounts the Gradio page at `/ui` with `gradio.mount_gradio_app` (the page's `app.py` and `music_client.py` are copied unchanged from the companion repository). The mount keeps the server's lifespan, so the chip still opens in startup; `tests/test_fiona_asgi.py` checks that. The adapter also refuses to start without `HF_MODEL` (the model code falls back to the base MiniMax-Music3 weights), passes `TT_MODEL_WEIGHTS_REVISION` on as `MUSIC3_WEIGHTS_REVISION`, and puts the weight cache next to the tensor cache that `run.sh` chose.
- **A tree of code from a Hub package.** The code was not in this repository, so the spec gained `[[extra_tree]]` (a directory copied into `extra_code/`, no links, `__pycache__` skipped). The 63 files the server imports are in `pkgsrc/music3/code/` with their source and revision in `pkgsrc/music3/PROVENANCE.md`.
- **A boot check for audio.** `[check] mode = "music"` sends one `POST /v1/audio/speech` for a short WAV clip and tests the sample rate, the length and the level (peak and the share of 100 ms windows above -40 dBFS), and `[check] health_path` names a health route other than `/health`. Chat specs are unchanged.
- **Caches inside the bundle.** `run.sh` points every cache at the bundle's directory, so a boot copy grows by about 22 GB. Pass `--hf-home` to `can boot` to reuse weights already on disk (27 GB).
- **ttnn 0.80.0 from PyPI ran the code unchanged.** The Hub package ran a source build of tt-metal `v0.79.0-dev`. No code change was needed; the output is not the same samples as the container's for the same seed, and the speed under a loaded host was lower (card and `docs/run-logs/2026-10-30-fiona-thin-bundle.md`).
- **UI tests need a current gradio.** The host's gradio 4.44.1 does not import with huggingface_hub 1.33, so six UI tests skip there and run in the bundle's own venv (`gradio==6.20.0`).

## Stable Audio 3 small music (`config/can.stable-audio-3.toml`)

A can for a model that makes audio behind a job API (it has no chat endpoint). What it needed that Granite did not:

- **A front end around a job server.** The stand-alone job server (`serve/serve_audio.py` of the tt-stable-audio-3-small-music repository, copied unchanged to `pkgsrc/server/stable_audio_jobs.py`) starts a new process for each clip and loads the model every time. `pkgsrc/server/stable_audio_asgi.py` imports its `JobServer`, `Generator`, `host_allowed` and `openapi_document` and serves the same routes under uvicorn. It keeps one resident chip runtime (`ResidentChipGenerator`, which uses `audio_runtimes.ChipRuntime`) loaded between clips. The device opens in the lifespan startup; shutdown waits for a running clip before it closes the device. The job server already has a queue, so there is no separate lock: a full queue answers 429 at once. A job does not belong to the connection that submitted it, so a client that leaves does not stop its clip.
- **A boot check for jobs.** `[check] mode = "audio-job"` (orchard/can_audio.py) submits two jobs with the same settings, downloads each clip, reads its length from its own header and records both times. The first job runs against a kernel cache that the boot check's copy has never filled; the second is warm. `max_rtf` bounds the warm job's generation time divided by the clip length.
- **Files that are not in the repository.** The model library (`stable_audio_3`, MIT, not on PyPI), the model's license files and the weights are outside the repository. `[[extra_tree]]` copies a directory into `extra_code/`, and `scripts/can_stable_audio_base.py` builds the base directory the spec is staged from: it checks the library checkout's commit and that it is clean, checks that the snapshot directory is named for the pinned revision, copies the license files from it and renders the card with the license paragraphs cut from that copy. Stage with `--base BASE`.
- **Pins that fit the PyTorch index.** `install.sh` adds the PyTorch CPU index as an extra index and uv takes a package from the first index that has it, so a pin of a package that index also carries (tqdm, packaging, numpy, pillow, networkx) must be a version that index has. The first install failed on `tqdm==4.70.1` for this reason (the index stops at 4.66.5).
- **A gated model.** The weights are downloaded by the app at start (`resolve_weights`): it downloads the pinned revision, builds a cache directory that holds only that revision (the library resolves the checkpoint and the text encoder through the cache's `main` reference) and requires the download to land on the pinned revision. The account behind `HF_TOKEN` must have accepted the model's terms.
- **A card in the tt-model vernacular.** See `docs/can-cards.md`.

Run log: `docs/run-logs/2026-10-30-stable-audio-3-thin-bundle.md`.
