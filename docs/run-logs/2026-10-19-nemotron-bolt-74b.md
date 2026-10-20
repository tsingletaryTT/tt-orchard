# nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB: unattended bring-up, stopped at preflight (2026-10-19)

Prompt (operator): perform an unattended bring-up. The only help allowed is disk space. If it fails, log the failure and
report what would make it succeed. Do not make the changes.

Tree: local `main` with PR 9 and PR 10 merged (version 0.6.4, 4422 passed, 2 skipped). Nothing was pushed.

## Result
`tt-orchard bringup nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB --dry-run` ends "frost before planting". Two checks block.
No run directory was created, nothing was downloaded, no chip was leased. A non-dry run stops at the same place.

| check | state | detail |
|---|---|---|
| base | BLOCK `nearest-model-missing` | The card's base is `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16`. No installed bundle serves it, and `tt model search nvidia` found none. |
| cpu | BLOCK `coder-unusable` | ollama was not running (connection refused on 127.0.0.1:11434). |
| disk | ok | 141 GB free on the HF cache drive, 104 GB needed. No help was needed. |
| hub | warn | 24.1 GB, license `other`, ships 12 Python files (modeling, configuration, processing, a reasoning parser). |
| skills | warn | `~/code/skills` is 3 commits behind upstream main, on a branch with uncommitted changes. |

## Facts about the model (from its `config.json`)
- Architecture `NemotronH_Omni_Reasoning_V3`, model type `nemotron_h_omni`: a hybrid model that takes more than text input.
- Q8_0 quantization in a ggml packing (8.5 effective bits, group size 32), declared per tensor under `quantization_config`.
- The Hub lists bundles only for the sibling family `Nemotron-3.5-Lightning-30B-A3B` (a different size; 0 downloads, none installed).
- No tt-metal demo for Nemotron-H or Mamba-style layers was found in `~/code/tt-metal/models/demos`.

## What would be needed for a success (not done)
1. Base. No bundle serves this family at this size. Options: a Bolt-74B bundle published and installed, or `--base` pointing at the
   Lightning-30B bundle. The second would not be a weights-only swap: the sizes differ, so stage 0 would very likely classify it
   `full-port`, and an unattended full port always blocks (spec, `_end_blocked`).
2. Packaging. Weights in ggml Q8_0 blocks are not safetensors tensors in the bf16 shapes the triage, swap and parity templates read.
   Stage 0 would need to understand the quantization layout, or the harness would need a bf16 base to compare against.
3. Modality. The omni model adds vision and radio modules; the harness's facts have `modality`, but only the audio spec covers a non-text path.
4. Code in the repo. `trust_remote_code` files are fetched and recorded and never run outside the agent sandbox, so a CPU reference
   (stage 1) that needs them depends on the reference venv's transformers supporting `nemotron_h_omni`. Not checked.
5. Environment, once 1 to 4 are settled: start `OLLAMA_MODELS=/mnt/bonus/models/ollama ollama serve` (the CPU stand-in check blocks
   otherwise); the preflight could start or at least name this in `setup`. Disk is tight on both drives (132 GB and 79 GB free) but met the 104 GB need.
6. Size. 74B total parameters with 9B active, so a one-board (2-chip) tier is a question for stage 2's memory check; not measured.

Not tested: any of the above beyond reading the preflight, the Hub config and the bundle search.
