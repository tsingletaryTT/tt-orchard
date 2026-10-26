# A bundle for a model that tt-transformers already serves (issue 62)

Date: 2026-10-26. Status: research note and proposal. Nothing here ran on a chip. Everything below was read from files
(tt-metal at tag v0.80.0, tt-model-manager `main`, the Hugging Face Hub) or produced by a staging-only command that opens no
device. Each claim says which.

## 1. What the issue asks and how it relates to PR 78

Issue 62 asks for a path that builds or finds a package, meaning a v6 thin bundle, from tt-metal model code that already
serves a model. The later stages need a package: stage 4 tests 2 and 4 chips through a package's `run.sh`, and stage 7
packages and boot-checks one.

PR 78 (issue 61, branch `full-port-pilot-design`) adds the `stock-runtime` class. Its stage 2 runs tt-transformers' own token
accuracy test from a copy of the host tree's `models/` directory and compares with the stage 1 CPU reference. It builds no
package, and its stages 4 to 7 are skipped with a stated reason. The two pieces fit together:

| | PR 78 `stock-runtime` | Issue 62 |
|---|---|---|
| Question answered | does the stock code run this model on this box, and does it agree with the CPU reference | can the same code be shipped as a bundle that serves through vLLM, on 1, 2 and 4 chips |
| Runs | tt-transformers pytest demo, host tree, shared virtual environment | the bundle's own `run.sh`: vLLM, the TT plugin, a venv built from pins |
| Stages | 0, 1, 2 | 4 and 7, and a reusable base bundle for later models |
| Needs a chip | stage 2 | the boot check, then stage 4 |

The proposal in section 6 builds on PR 78 and does not replace anything in it.

## 2. Question 1: can `tt-model package-thin` build a bundle from tt-transformers with no new code

Read: `tt-model package-thin --help`, `docs/thin_packages.md` and `docs/E2E_RECIPE_V6.md` on tt-model-manager `main` (commit
713eeb2), and the generated files below. The open branches that touch thin packaging (`feat/v6-extra-code-127`, authoring
controls, auto custom ops, run-weights revision) are already merged into `main`, so `main` has them.

Run (staging only, in a scratch directory, no chip, no network, `--arch` and `--tt-metal-version` given so nothing is
detected from the host): `tt-model package-thin --out ./bundle --model-py model.py --extra-code extra_code
--requirements requirements.txt --arch blackhole --arch-name LlamaForCausalLM --main-class model:LlamaForCausalLM --weights
unsloth/Llama-3.1-8B-Instruct --mesh P150 --device-count 1 --tt-metal-version 0.80.0 --max-model-len 8192 --max-num-seqs 32
--block-size 64`. Inputs used:

- `model.py`: one line, `from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM`.
- `extra_code/models/`: a copy of the host tree's `models/common` and `models/tt_transformers/{tt,model_params}` (8.3 MB, 396
  files; the tree has no `models/__init__.py`, so these are namespace packages).
- `requirements.txt`: `ttnn==0.80.0` and `torchvision`. PyPI has `ttnn` 0.80.0, the same version as the host tree's tag.
  PyPI has no `tt-metal-models` and no `tt-transformers`, so the model code has to travel in `--extra-code`.

Result: the command staged a bundle (`tt_kernel_manifest.json`, `install.sh`, `run.sh`, `vllm_models/<name>/vllm_metadata.json`,
`extra_code/`). It printed one warning: no `--plugin-wheel` was given. What the generated `run.sh` starts:

- `python -m vllm.entrypoints.openai.api_server --model <weights repo> --max_num_seqs 32 --block_size 64 --max_model_len 8192`.
- The model class is registered through `vllm_metadata.json` (`arch: LlamaForCausalLM`, `main_class: model:LlamaForCausalLM`) and
  `EXTRA_MODELS_DIR`. `PYTHONPATH` is `extra_code:` then the bundle. `MESH_DEVICE` defaults to the value of `--mesh` (`P150`).
- Before it starts, `run.sh` resolves `HF_MODEL` to a snapshot directory and exports it as `HF_MODEL` and `MODEL_WEIGHTS_DIR`.

So the packaging step needs no new model code. A bundle that boots is a separate matter, and a static read of the staged tree
found three things the bundle needs that the command does not supply:

1. `tt_py_test_utils_common` (268 KB, pure Python, at the root of the tt-metal tree). `models/common/utility_functions.py` and
   `tensor_utils.py` are shims that import it, and the PyPI `ttnn` 0.80.0 wheel holds only `tracy`, `triage`, `tt_lib`, `ttnn`.
   An import walk (Python `ast`, nothing executed) from `generator_vllm` reached 33 modules under `models/` and named this
   package as outside them. It must be added to `--extra-code`.
2. The vLLM TT plugin wheel (`--plugin-wheel`). It is not on PyPI. Copies of `vllm_tt_plugin-0.1.0` exist in the installed
   Qwen3.8 bundles' build directories. Which plugin and vLLM version suit tt-transformers' `LlamaForCausalLM` at v0.80.0 was not
   determined: the installed Qwen bundle used vLLM 0.26.0, the 3B container package below used 0.30.0, and `package-thin`
   defaults to 0.25.1. `generator_vllm.py` imports `vllm.model_executor.models.gemma3_mm` and `mistral3` at module level, so
   it needs a vLLM that has them.
3. Twelve names under `models.demos`, `models.experimental` and `models.tt_transformers.demo` that the walk could not
   resolve. All are inside other model classes' `initialize_vllm_model` bodies (gpt-oss, Cohere, Gemma 3, vision demos), not
   in `LlamaForCausalLM`'s path. A grep for module-level imports of them found none. They would fail only if one of those
   other classes were selected.

A fourth point matters for weight swaps. tt-transformers reads `HF_MODEL` as the checkpoint directory and takes the model name
from `config.json`'s `_name_or_path` when it is set (a hub cache path gives the repository name; any other path gives its last
segment), otherwise from the last segment of `HF_MODEL`. A swap directory named `model-dir` would be an unknown model name.
PR 78's `stock_check.py` already names the directory after the tt-transformers model. A bundle-based swap needs the same rule,
and the Qwen3.8 swap template (which sets `MODEL_WEIGHTS_DIR`, `QWEN36_DRAFTER` and other Qwen-specific values) does not do it.

Not determined: whether the staged bundle installs (`install.sh` builds vLLM from source and needs the network and a plugin
wheel) and whether it serves. `tt-model package-thin` itself says it is beta and unsupported.

Commands that would settle it. Step 1 needs no chip: copy the staged bundle, add `tt_py_test_utils_common` to `extra_code/`,
pass `--plugin-wheel` (copy from an installed bundle) and `--constraints` from a resolved list, then `bash install.sh` in the
copy and the import smoke test from the recipe (`import model` with `run.sh`'s `PYTHONPATH`, no device open). Step 2 needs one
board: `gozer acquire --chips 1 --who "claude:stock-bundle" --reason "boot check of a tt-transformers thin bundle"`, then
`gozer wait <ticket>`, then `TT_VISIBLE_DEVICES=<granted> bash <copy>/run.sh --port 20000`, one `temperature: 0` completion
compared with the CPU reference, then `gozer release <lease>`. About 20 to 40 minutes, including a cold kernel build.

## 3. Question 2: which tt-transformers and demo models can run on Blackhole at v0.80.0

Evidence kinds, strongest first: (a) a `models/model_targets.yaml` row for a Blackhole board with status `active` (the file says
CI targets; `PERF.md` is marked deprecated in its favour; nothing here confirms CI runs the row at v0.80.0 or that it
passes); (b) Blackhole names in `tt/model_config.py` tables (`P150`, `P300`, `P150x4`, `P150x8`), which show the code was
written with the board in mind and show nothing about a result; (c) the README's verified-models table (it names p100 / p150 for
Llama-3.1-8B; its Qwen3-32B row says "LoudBox / QuietBox", which does not say which generation). No model below was run by me. "Unverified" means no row of kind (a).

The 23 `model_params` entries:

| Entry | Blackhole evidence in v0.80.0 | State |
|---|---|---|
| Llama-3.1-8B-Instruct | targets: `bh_p100`, `bh_p150` (top-1 90, top-5 98, 512 tokens), `p300x2` (90.0, 97.6, plus an IFEval row), `bh_p300` is `TODO`. README: n150 / p100 / p150. | target rows, `active` |
| Qwen2.5-32B-Instruct | targets: `p300x2` (98, 99) | target row, `active` |
| Qwen3-32B | targets: `p300x2` (91, 99). A published container package exists (section 4). | target row, `active` |
| Qwen3.6-27B | targets: `bh_quietbox_2` (97, 99; many context lengths). The installed Qwen3.8 bundles register `models.demos.blackhole.qwen36...`, so this model is served by that demo code and not by `tt_transformers` classes. | target row, `active`, other code |
| Llama-3.1-70B-Instruct | no row for it. The targets file has `llama3.3-70b` on `p300x2` (96, 100), a different model. Code tables have `P150x4`. | unverified |
| EXAONE-4.5-33B | targets: `bh_loudbox` only. Code table: `P150x8`. | not this box |
| Qwen2.5-VL-32B-Instruct, Qwen2.5-VL-72B-Instruct | targets: `p300x2` with status `TODO` | not verified |
| Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct | no Blackhole target row. Code: a `P150x4` prefill entry. A published p150 container package exists for the 3B (section 4). README: n150 only. | unverified |
| Llama-3.2-11B-Vision-Instruct, Llama-3.2-90B-Instruct, Llama-3.2-90B-Vision-Instruct, Meta-Llama-3-8B | no row; code tables mention `P150x4` | unverified |
| Mistral-7B-Instruct-v0.3, phi-4, Qwen2.5-7B-Instruct, Qwen2.5-72B-Instruct, gemma-3-27b-it, LFM2.5-VL-1.6B, Qwen3-VL-32B-Instruct | no Blackhole row; code tables mention `P150x4` (or `P150x8`) for most | unverified |
| Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct | no row; the `P150x4` value in the code table is `None` | unverified |

Demos outside the list: `models/demos/qwen25_vl` (README and `PERF.md` name Wormhole boards only; its `rope.py` has a
`blackhole` branch and `generator_vllm.py` gates one 72B case on `is_wormhole_b0()`; no target row for its 3B or 7B) is
unverified. Targets with Blackhole rows and no `model_params` directory, listed for completeness: `gemma-4-26b-a4b`, `gemma-4-31b`,
`gemma-4-e2b`, `gemma-4-e4b`, `gpt-oss-20b` (`bh_p150`, `p300x2`), `gpt-oss-120b`, `mistral-small-3.1-24b`
(`bh_quietbox_2`), `qwen2.5-coder-32b` (`bh_quietbox_2`), `qwen38-27b-qb2` and `llama3.1-8b-qb2`.

The only entry with rows for the single-chip and the two-board configurations a weights-only package ships for (1, 2 and 4
chips) is Llama-3.1-8B-Instruct. That agrees with PR 78's choice of pilot.

## 4. Question 3: is there an installed or published bundle for the two Llama models

`tt-model list` on this box (2026-10-26): no Llama package is installed. The installed entries are Qwen3.8-27B bundles and
container packages, `episod/tt-tnt`, `episod/tt-tnt-1024`, the music and Qwen3-Coder container packages.
`tt-model search llama`, with and without `--catalog`, returns one repository, `stisiTT/llama-3.2-3b-blackhole-v51`;
searches for `llama-3.1-8b` and `llama3.1` return none; `tt-model search blackhole` lists 17 repositories, of which the
tt-transformers-family ones are the 3B, `mando2222/qwen3-32b-blackhole-v51` and `mando2222/vibethinker-3b-blackhole-v51`.

The 3B package, read from its manifest on the Hub (not pulled): schema 5.1 container package, weights pointer
`unsloth/Llama-3.2-3B-Instruct`, arch Blackhole, `device_count` 1, hardware `p150` with a `p100` profile, image kind
`vllm-plugin` with vLLM 0.30.0, `extra_models_dir` `models/tt_transformers/vllm_bundle`, created 2026-10-15. That shows
someone built a tt-transformers model for a p150 board through the container route. No measurement from it was read. Whether
this container reads weights from `MODEL_WEIGHTS_DIR` is not known (the issue 60 container hard-coded its repository).

For Llama-3.1-8B-Instruct no bundle or package is installed or published as far as these searches show.

## 5. What the evidence says about the class and the stage

- No new class. A model that tt-transformers serves is `stock-runtime` in PR 78. What issue 62 adds is a package, and a
  package is needed per base model once, not per run.
- A package for a base model such as Llama-3.1-8B-Instruct is reusable. Fine-tunes of it have the same architecture, so
  they would take the ordinary weights-only path (swap weights into the bundle) with stage 4 and stage 7 unchanged. That
  route needs a swap template for the tt-transformers env rules in section 2 (directory name, `HF_MODEL`, no Qwen values).
- The packaging command works. Whether the result installs and serves is open (section 2); every later stage depends on it.

## 6. Proposal and next steps

1. Run the two checks in section 2 (step 1 with no chip, step 2 on one board) on the 8B. Exit criterion: the staged bundle
   serves one `temperature: 0` completion that matches the PR 78 stage 2 result for the same model and the same tokens.
   This is the only evidence that settles the open question, and it is cheap.
2. If it passes, add a base-bundle command (working name `tt-orchard base-bundle <model>`), supervisor code in the style of
   `orchard/package.py` with no agent: stage the extra-code tree from the host tag (the files listed in section 2 plus
   `tt_py_test_utils_common`), run `package-thin` with the flags above, boot-check under a lease with the existing stage 7 hardware
   phase, and write the bundle where `nearest.read_installed` finds it. The preflight then finds a bundle for the base, so a
   fine-tune of a stock model passes `base` without PR 78's warning path. Stage 7's gate (`gate_package`) needs a second look:
   it reads Qwen-specific serving facts.
3. Add a tt-transformers swap template (a sibling of `weights-swap-templates/prepare_swap.py`) only after step 2, because it
   depends on what the boot check shows.
4. Leave stage 4 of the `stock-runtime` class skipped, as PR 78 has it, until step 1 has passed. If it passes, stage 4 can
   reuse the existing multi-configuration machinery with the base bundle as `bundle_dir`.

Cost and risk: step 1 is about an hour of work and one board for under an hour. Step 2 is a module the size of
`package.py`'s staging half plus tests. The main risk is vLLM and plugin version compatibility, which nothing read here
settles. A fallback is the container route (issue 60): the 3B package above would serve a 3B fine-tune with no new code if
its generator honours the model directory, which has to be checked the way issue 60's validation found the other one did not.

## 7. What was not determined

- Whether the staged bundle installs or serves (no install, no chip).
- Which vLLM and plugin versions work with tt-transformers' `LlamaForCausalLM` at v0.80.0.
- Whether the Blackhole target rows pass in CI at v0.80.0, and any Blackhole result for the models marked unverified.
- Whether the published 3B container package serves swapped weights.
- Whether the 8B runs on 1, 2 and 4 chips on this box. Its target rows are for one chip (`bh_p100`, `bh_p150`) and `p300x2`
  (two boards, which the targets file equates with `bh_quietbox_2`, this box's layout); no two-chip (one board) row exists.
