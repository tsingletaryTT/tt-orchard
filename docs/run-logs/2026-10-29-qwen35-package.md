# 2026-10-29: package Qwen3.5-35B-A3B for one p300 board

Operator request: package the 40-layer run measured in `2026-10-29-fusion-port.md` (126.8 tok/s, 29.6 percent of the estimated floor).
Nothing is uploaded or published; the operator has not approved publishing this package.

## Decisions

Read first: `orchard/audio_package.py` (stage, card, gate and record pattern), tt-model-manager `main` (`docs/container_packages.md`,
`docs/publishing.md`, `examples/container-example.yaml`) and its open branches (read only, fetched, nothing pushed). The open branch
`fix/card-weights-wording` (2026-10-29, `src/tt_kernel/build.py` and its test) changes where the weights pin is worded on a generated card.
It does not change the manifest schema. Other open branches (`feat/serve-n-chips`, `feat/cohost-kind`, `feat/deploy`) do not touch the fields used here.

Package shape: a recipe package, not a built v5.1 container package and not a v6 thin package.
- A v5.1 package is built by `tt-model package --container`, which needs a tt-metal tree and copies an allowlist of model code into a new
  image. This package changes one file of another author's published bundle and has no tt-metal tree of its own to build from.
- A v6 thin package pins stock `ttnn` and ships model code as `extra_code/`. The fused ops live in the Coder-Next image (a tt-metal fork build),
  so a thin package cannot reproduce the measured setup.
- The recipe keeps the measured image untouched, pins it by image id, mounts a patched copy of the loader at run time, and checks the loader's
  sha256 before and after the patch. `tt-model.yaml` is a draft in the v5.1 vocabulary that records the serve profile (as the audio package's does).
- The weights are a pointer. `fetch_weights.py` reads only the text tensors by byte range at a pinned revision. `hf download` of the same files ran at
  4 to 6 MB/s on this machine and the ranged reads at 35 to 50 MB/s.

License: the model repository's license field is `apache-2.0` (not gated); its `LICENSE` file is the Apache License 2.0 and is copied into the package. Revision
pinned: `59d61f3ce65a6d9863b86d2e96597125219dc754` (read from the Hub API on 2026-10-29). The loader file in the image carries an Apache-2.0 header.

## Progress log

- Worktree `qwen35-package` off `origin/main` (includes PR 236). Weights fetch into a scratch directory started (disk had 233 GB free; 69.3 GB fetch leaves more than 150 GB).
- Module `orchard/qwen35_package.py` (stage, card, gate, publish text) and `tests/test_qwen35_package.py` written; 19 tests pass; 11 mutations of the gate each turned a test red
  (the size limit needed its own test first).
