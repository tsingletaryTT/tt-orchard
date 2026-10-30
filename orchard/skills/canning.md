---
name: canning
description: Use at the end of a model bring-up that produced new code (a runtime and a server of its own) to can it, which means to stage it as a v6 thin bundle that tt-model can install and serve. The step is `tt-orchard can`. It never publishes.
---

# Canning a brought-up model

A model that runs on the chip ends in a can. The can is a v6 thin bundle: `tt_kernel_manifest.json`, `install.sh`, `run.sh`, exact pins, the model's code and its card. `tt-model` installs it into its own venv and serves it. A recipe package (a `run.sh` and a draft container manifest) cannot be served by `tt-model`; this one can.

Full description: `docs/canning.md` in the repository. This page is the checklist.

## Steps

1. **Find the server.** If the model already runs behind vLLM through the tt-vllm plugin, the kind is `vllm`. Any other server (a standard-library HTTP server, a Gradio page) uses kind `tt-dit-server`, which serves an ASGI app with uvicorn.
2. **Write the adapter when the kind is `tt-dit-server` and the server is not ASGI.** Import the existing request and generation code; do not copy it. Open the device in the ASGI lifespan startup. Run chip work in a worker thread. Take the one-request lock without waiting and answer 503 when it is held. Stop generating when the client leaves. Read settings from environment variables (uvicorn passes no flags). The model for this is `pkgsrc/server/granite_asgi.py` and its tests, `tests/test_granite_asgi.py`: copy the test file and keep every behaviour it checks.
3. **Write the spec.** Copy `config/can.granite.toml`. Fill in the name, the app, the files, the exact pins, the weights repo and its 40-character revision, the mesh and the device count. Choose a port from 20110 to 20199. `[check]` holds a prompt, the answer the model must give, the lowest decode rate the bring-up measured (leave a margin below it) and the strings the start-up log must contain.
4. **Write the card.** Say what the bundle is and how to serve it, the API, the settings, the limits and the license. Take the numbers from the run's evidence files and label where each came from. Do not call anything fast, best or production-ready.
5. **Pin.** Every line of `requirements` is `name==version`. Take the versions from the environment the model was brought up in. If the bring-up used a source build of ttnn, check what the PyPI wheel of the same version needs (ttnn 0.80.0 needs numpy below 2) and run the model's code with those pins. List `fastapi`, `uvicorn`, `pydantic` and `pillow` with pins when the kind is `tt-dit-server`.
6. **Stage.** `tt-orchard can stage SPEC --out DIR/NAME --run-dir RUN`. It runs `tt-model package-thin --out`, fixes what that leaves wrong, writes `SHA256SUMS` and runs the gate. Fix every gate problem in the spec or the sources. Do not edit the staged directory by hand.
7. **Boot check.** Copy the staged directory, run the copy's `install.sh` (no lease needed), then run `tt-orchard can boot` under a gozer lease. The boot check serves the copy's own `run.sh`, sends one chat and one streamed request and checks the answer, the decode rate and the log. Never install into or serve the staged directory.
8. **Record.** `--run-dir RUN` writes `stages/7/can.json` and an `evidence` ledger entry. Put the boot check's numbers in the run log.
9. **Publish commands.** `tt-orchard can commands SPEC --namespace ORG` prints them as text. Hand the text to the operator. Do not run it. You do not publish, push, upload, merge or run `tt-model push` or `publish`.

## Stop and ask when

- the gate reports a hit you cannot trace to a source file;
- the boot check fails twice for the same reason;
- the PyPI ttnn wheel does not run the model's code and a source build is needed (that changes what the bundle can claim).
