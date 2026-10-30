# Granite as a v6 thin bundle: the first can (2026-10-30)

Goal (operator): "v6 thin bundles for all these", then "build this into tt orchard, this is like the canning step". `tt-model serve episod/<package>` refused every published package with "is not a tt-model bundle (no tt_kernel_manifest.json)", because they are recipe packages (a `run.sh` and a draft container `tt-model.yaml`). Granite-4.0-H-1B was the pilot. Code: `orchard/can.py`, `config/can.granite.toml`, `pkgsrc/server/granite_asgi.py`. Pattern: `docs/canning.md`.

## What was built

- `tt-orchard can stage | gate | boot | commands`, driven by a TOML spec. Kind `tt-dit-server` (an ASGI app served by uvicorn) for Granite; kind `vllm` is supported by the call and covered by a test, and no bundle of that kind was built.
- `pkgsrc/server/granite_asgi.py`: the ASGI front end. It imports the request checks, the generation loop and the lock from the unchanged `serve.py` (shipped in the bundle as `extra_code/granite_serve.py`).
- The bundle (16 files, 172 KB, no weights, no venv, no cache): `tt_kernel_manifest.json` (schema 6, kind `tt-dit-server`, app `granite_asgi:app`, weights `ibm-granite/granite-4.0-h-1b` at `d18cca4c`, mesh P150, 1 device), `install.sh`, `run.sh`, 19 exact pins (`ttnn==0.80.0` from PyPI), the runtime and kernels in `extra_code/`, the card, LICENSE, NOTICE, `tools/smoke.py`, `SHA256SUMS`.

## What package-thin leaves wrong, and what the can does about it

| Found | Action |
| --- | --- |
| `producer.hostname` is the build host | replaced with `redacted`; the gate checks it |
| `run.sh` and `install.sh` have mode 0600 | made 0755; the gate checks it |
| `run.sh` runs `uvicorn --host 0.0.0.0` (no sign-in on the server) | binds `${HOST:-127.0.0.1}`; the gate refuses 0.0.0.0 |
| `run.sh` defaults to port 8000 | defaults to the spec's port (20110); `PORT` still overrides |
| requirements for `tt-dit-server` get unpinned `fastapi`, `uvicorn`, `pydantic`, `pillow` appended when missing | the spec pins all four; the gate refuses any line that is not `name==version` |
| `--verify` statements run with only `extra_code/` on the path | found on the first run of tt-model's own installer (it failed on `import granite_asgi` for a module at the bundle root; my own `bash install.sh` from inside the bundle directory had passed because the directory was the working directory). The adapter now lives in `extra_code/`, `model_py` is a stub, and the gate reports a verify import of a root module |
| ttnn 0.80.0 from PyPI requires `numpy<2` | the spec pins numpy 1.26.4; the recipe package's environment had numpy 2.5 |
| tt-model cannot serve a local v6 directory (`tt-model serve <dir>` and `<manifest path>` fail: "Nothing was installed"; v6 bundles come from the Hub) | boot check serves the copy's own `run.sh`, which is what tt-model runs after its install; and the install-and-serve was also done through tt-model's own code (below) |

## Tests

- `tests/test_can.py` (73) and `tests/test_granite_asgi.py` (19). The can's gate is run against a real `tt-model package-thin --out` bundle of a toy spec, then each fault is introduced (17 faults and 7 manifest edits); the boot check runs against a stand-in server.
- Mutations (each seen to turn a test red, then restored): host fix removed, checksum comparison removed, hostname left in the manifest, no chmod, scrub skipped, boot check not stopping the server, boot check ignoring the answer, boot check installing into the staged directory, ASGI lock not released by the worker, stream ignoring a disconnect, a busy request waiting for the lock, `/health` ignoring an unusable chip. One survived the first time (stream ignoring a disconnect): the test ended on its own after 40 tokens. The test now asserts the engine stopped before 20 steps.

## Hardware (board 1, chips 2 and 3, `gozer run --exact 2`, each `gozer run` released and reset cleanly; no chip wedged)

Staged with `tt-orchard can stage config/can.granite.toml --out ...`; the gate passed. The staged directory was never installed into or served: every run used a copy.

1. `tt-orchard can boot` on a copy (first layout of the bundle): ready in 24 s the first time (the first request took 16.5 s: kernels compile into the copy's empty cache) and 9 s the second time. Chat answered "Paris"; streaming ended with `[DONE]`; the server log held `"traced": true` and `"chunked_prefill": true`; decode 33.9 tok/s over 64 tokens. The first run of this check reported FAILED because the short answer is one token and has no decode rate; the check now times a separate 64-token request.
2. After the change that moved the adapter to `extra_code/`, the bundle was installed with tt-model's own installer (`tt_kernel.cli._install_self_contained` into a scratch install directory and a scratch index) and served with `tt-model serve local/granite-4.0-h-1b-blackhole --local-only --port 20112`. Healthy after 27 s. `tools/smoke.py`: 9 of 9 checks pass (short answer, stop, length, stop string, stream, 1,481-token prompt, two 400s). Decode 33.0 tok/s over 64 tokens (27.7 over 7 tokens); first streamed content 0.58 s.
3. Prefill. A 1,534-token prompt took 1.79 s and 1.77 s (runs 1 and 2, warm). The first long prompt after a start took 21.9 s, both through `can boot`'s copy and through tt-model's install: the chunked-prefill programs compile into the empty cache at that request. The recipe package's card gives 1.2 s for a 1,481-token prompt, measured with a warm kernel cache; the two numbers are not the same prompt length (1,534 and 1,481 tokens) and 1.79 s against 1.2 s was not investigated.

4. `can boot` on the final layout of the bundle (adapter in `extra_code/`), recorded with `--run-dir` (`stages/7/can.json` and an `evidence` ledger entry). First run, with the full test suite running on the same host (load average 19): decode 28.4 tok/s over 64 tokens, so the check reported FAILED against its 30 tok/s floor and the ledger entry says `booted: false`. Second run, host load average 3.4: decode 34.3 tok/s, all 8 checks passed, `booted: true`. The card gives 25.2 tok/s at host load 17.9 and 34.5 at 2.4, so the rate follows the host load, as it does for the recipe package. A boot check on a loaded host can fail on the rate alone.

## Against the recipe package

Same: endpoints, request checks, generation loop, `/health` and 503 behaviours, chunked prefill and trace on by default, the smoke checks, about 33 to 34 tok/s decode (card: 34.5), answer "Paris".
Different: (a) the server is ASGI under uvicorn, so settings are environment variables (there are no flags); (b) the venv is built from exact pins, with `numpy 1.26.4` and the CPU `torch 2.7.0` (the recipe environment had numpy 2.5.3); agreement with the CPU reference was not re-measured with these pins; (c) the first requests after a start are slow (kernel compile in the bundle's own cache), which the recipe's `run.sh` also has on a cold cache but its smoke numbers did not show; (d) the recipe refused ports 8000, 8011, 8012, 8780 and 8100 to 8103 in `serve.py`'s `main()`. The ASGI path cannot refuse a port, because uvicorn binds before the app starts; the bundle's default port is 20110 and the spec refuses those ports at load; (e) the optional tt-lang kernel modes are in the bundle but `tt-lang` is not in the pins.

## Not done

- No upload, no `tt-model push` or `publish`, no merge. `tt-model serve episod/granite-4.0-h-1b-blackhole` still fails until the operator publishes the can; the publish text is `tt-orchard can commands config/can.granite.toml --namespace ORG`.
- The supervisor does not run the can by itself. The stage flow gets a checklist (`orchard/skills/canning.md`, a pointer in the operator runbook) and a record (`stages/7/can.json` and an `evidence` ledger entry from `--run-dir`).
- Kind `vllm` was not built on hardware (Qwen3.5-35B needs it).
- Agreement with the CPU reference under numpy 1.26.4, and the 1.79 s against 1.2 s prefill difference, are open.
