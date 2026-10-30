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
