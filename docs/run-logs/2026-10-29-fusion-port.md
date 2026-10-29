# 2026-10-29: fusion port candidates (issue 184)

Question: for a model and its nearest fused bundle, which fused decode ops apply as they are, which apply with a changed
constant, and which need new code? And can the harness port one op and report the measured change with its floor share?

Answer for Phase 1 (no hardware): the harness now lists the candidates (`tt-orchard fusion-candidates MODEL_CONFIG BUNDLE_DIR`,
`orchard/fusion_port.py`). Phase 2 (the hardware port) was not run. The reason is stated at the end.

## The pair

The nearest fused bundle is the Qwen3-Coder-Next decode bundle (`raahemnabeel/qwen3-coder-next`, the p300 profile, two dies).
Its fused ops are not in the manifest. They are Python modules under `models/demos/blackhole/qwen36/tt/` and
`models/autoports/qwen_qwen3_coder_next/tt/` in the container image; this run copied those two directories out of the local image
with `docker create` and `docker cp` (a stopped container, removed afterwards; the coder's running container was not touched) and read them
as text.

Config.json files were downloaded alone (no weights) with the Hugging Face CLI. The token stays in the environment and is not printed.

| Candidate | Dimensions against Coder-Next | Verdict |
| --- | --- | --- |
| Qwen3-Next-80B-A3B Instruct and Thinking | Identical in every number the ops read: 48 layers, 512 experts, top-k 10, hidden 2048, 16 heads, 2 KV heads, head dim 256, vocabulary 151936, linear attention 16 and 32 heads of 128. | Rejected as the port pair. Every op applies as is, so there is no changed constant to port. The weights are 162.7 GB and the disk has 81 GB free. |
| Qwen3-Coder-30B-A3B Instruct | `qwen3_moe`: 128 experts, top-k 8, 32 heads, head dim 128, full rotary, no linear attention, no shared expert. | Rejected. It has its own bundle (`raahemnabeel/qwen3-coder-30b-a3b`), and the attention prep needs new code (below). |
| Qwen3.5-35B-A3B | Same family code path (`qwen3_5_moe`, gated delta net plus gated attention): 40 layers, 256 experts, top-k 8, hidden 2048, vocabulary 248320, same attention and linear-attention shapes. | Chosen. Same op set, different expert count, top-k and vocabulary, so some constants differ. |

Chosen pair: `Qwen/Qwen3.5-35B-A3B` against the Coder-Next bundle, split over 2 dies. (Qwen3.6-35B-A3B has the same dimensions.)

## Candidate list, measured from the real bundle code

Run: `tt-orchard fusion-candidates tests/fixtures/model-configs/qwen3.5-35b-a3b.json <bundle code dir> --bundle-config <Coder-Next config.json> --dies 2`.

| Op | Family | Gate (code default) | Status for Qwen3.5-35B-A3B | Depends on | Why |
| --- | --- | --- | --- | --- | --- |
| router_topk | router | `QWEN35_ROUTER_TOPK_FUSED` (1) | applies with a changed constant | experts, top-k | All checks pass. `QWEN35_ROUTER_TOPK_ROUTE_K` (80) was measured at 512 experts and top-k 10; this model has 256 and 8. |
| moe_decode | MoE decode | `QWEN35_MOE_FUSED_DECODE` (0) | applies as is | experts, hidden, expert width per die | All five dimension checks pass; the sixth reads the weight dtype, a run-time policy, and is listed as not evaluated. The expert mask limit (512) holds. |
| gdn_decode | mixer (gated delta net) | `QWEN35_GDN_FUSED_DECODE` (0) | applies as is | linear heads per die, per-head widths, projection width | All seven checks pass. |
| attn_decode | mixer (gated attention) | `QWEN35_ATTN_DECODE_OPS` (0) | applies as is | heads and KV heads per die, head dim, rotary dim | All checks pass (rotary dim 64 = two tiles). |
| sampler_rows_topk | sampler | `QWEN35_ROWS_TOPK` (0) | applies with a changed constant | vocabulary per die | All checks pass (width 124160 is under 1,048,576 and takes the 32-bit index path). `GRID` (10, 10 scan cores) was sized for the bundle's 75,968 columns per die; this model has 124,160. |
| fabric_boundary | layer boundary | `QWEN35_FABRIC_BOUNDARY` (0) | needs new code | hidden, dies | The bundle's card says it runs on four dies only. On the p300 profile (two dies) it is a gap. Recorded as a gap and not attempted. |
| addnorm | layer boundary | `QWEN35_ADDNORM_DECODE` (0) | applies as is | hidden | The one check reads a core count the interpreter cannot evaluate (a `len` of a list built in the function), so it is listed as not evaluated. |

Counts: 4 apply as is, 2 apply with a changed constant, 1 needs new code. Hazards from `orchard/hazards.py` for this pair:
vocabulary per die above 65,536 (the bundle's own sampler docstring says its 32-bit index path covers this case).

The same command on the other two candidates, for comparison:

| Model | As is | Changed constant | New code | Not used |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-Next-80B-A3B Instruct, 2 dies | 6 | 0 | 1 (fabric boundary, two dies) | 0 |
| Qwen3.5-35B-A3B, 2 dies | 4 | 2 | 1 (fabric boundary) | 0 |
| Qwen3-Coder-30B-A3B, 2 dies | 3 | 1 (route-k) | 2 (attention prep: rotary dim 128 against the two-tile rotate-half; fabric boundary) | 1 (gated delta net: no linear layers) |

What the status rules mean (in `orchard/fusion_port.py`): a failing check that guards kernel code is new code; a failing check on the
Python side, or a constant the bundle tuned for a dimension that differs here, is a changed constant. Which checks guard kernel
code is a table in the module (`OPS`); the choice is conservative, so an op lands in "new code" when in doubt. This is a prediction from
the config and the source. No chip ran.

One finding to carry into the port step: the route-k constant only takes effect in the unfused router path. The bundle's code takes the
fused one-row router (`ROUTER_TOPK_FUSED`) when the step has one row, so changing route-k leaves one-row decode output and speed identical.
That is the case issue 15's comment says to flag; for this constant it is expected and not a sign the change failed to load.

## What was built

- `orchard/fusion_port.py`: reads a bundle's source without running it (no links followed, 1 MiB per file, 20,000 entries,
  parsed with `ast`), extracts each fused op's dimension checks, module constants, the environment gate with its code default and the
  tuned constants, evaluates the checks against a model config with a small interpreter (arithmetic and comparisons; no calls),
  and classifies each op. Reuses `model_config.py` (dimensions) and `hazards.py` (hazard names).
- `tt-orchard fusion-candidates MODEL_CONFIG BUNDLE_DIR [--bundle-config FILE] [--dies N] [--json]`, read only.
- `tests/test_fusion_port.py`: 25 tests plus one that runs on the real bundle code when `ORCHARD_FUSED_BUNDLE_DIR` points at a copy.
  Fixtures: configs of the 80B sibling, the 35B model and the 30B model (public Hugging Face configs).
- Mutations seen red (each reverted): the ambiguity check, the kernel-side table, the mixer-not-used rule, the tunable comparison, the
  not-in-bundle rule, call refusal in the interpreter, the file size limit. The symlink check is covered twice (the directory scan and
  `safeio.read_bytes`), so removing only the scan check leaves the test green; the second layer still refuses the link.
- Limits: the table `OPS` holds the Coder-Next bundle's layout (module names, gate names, tuned constants). A bundle with another layout
  needs its own rows. The per-die rule (heads and expert width rounded up over dies) follows the loader's tensor parallel split and was
  not checked against a run.

## Phase 2 (the hardware port): not run

Reasons, in order:

1. The 35B checkpoint is 71.9 GB of safetensors (from the Hub file metadata). The disk holds 81 GB free of 3.6 TB (98 percent used) and other
   agents write caches to it. A download would leave about 9 GB.
2. The bundle's model code names its checkpoint (`HF_MODEL_ID = "Qwen/Qwen3-Coder-Next"` in `model_config.py`) and its weight loader streams that
   checkpoint's layout, so serving Qwen3.5-35B needs a copy of that code with a changed model id and a bring-up of its own (cold compile for
   this bundle took about 15.5 minutes on the coder run). The two ops that apply with changed constants (route-k and the sampler scan grid)
   cannot be measured without that bring-up.
3. Chips 0 and 1 belong to the coder lease. Chips 2 and 3 were free when checked, but a baseline plus two ablations on them is several hours.

Nothing was leased and no chip was touched. The blocker is filed as a GitHub issue (disk and bring-up) and linked from issue 184.
What would unblock it: about 150 GB free disk (or the checkpoint on another volume), then a copy of the bundle code with the model id changed,
a baseline run with the benchmark rules of `docs/benchmark-rules.md`, and one ablation each for route-k (expected: no change at one row)
and the sampler `GRID`.

## Phase 2 progress (hardware), first session

Status when this section was written: the model loads and serves on two dies; the speed measurements are not done yet and are added below
when they exist.

What worked:
- Weights. The bundle's loader (`weights.py`) knows the Qwen3-Next checkpoint layout. A 35-line patch (`scripts/fusion_port/weights_qwen35_moe.patch`)
  adds the Qwen3.5-MoE layout (`model.language_model.` prefix, split GDN projections, fused experts). It applies to a copy of the bundle's
  autoport directory that is bind-mounted over the image's; the image, the published bundle and the coder's container were not changed.
- Download. `hf download` ran at 4 to 6 MB/s. Range reads of only the needed tensors from the Hub (`scripts/fusion_port/slice.py`) took
  8.76 GB (embeddings, head and layers 0 to 3 of the 35B model) in about 6 minutes. This is a 4-layer slice of the 35B model, not the model.
- Serving. The 4-layer slice served through vLLM in the bundle's image on chips 2 and 3 (mesh P300, precision policy
  `coder_next_routed_bfp4_mixers_bfp8`, one sequence, 4096 context) and answered `/v1/models` about 5 minutes after the container started, with the
  coder's kernel cache copied read-only as the starting cache.

Per-op check on the two ops with changed constants (`scripts/fusion_port/ops_check.py`, one session on the 2-die mesh, CPU `torch.topk` as reference):
- Sampler `RowsTopK` at 124,160 columns per die: top-k values equal the reference for scan grids (10,10), (8,8), (10,5), (9,10) and (10,8); the top-1
  index is equal. Index sets differ from the reference only at ties on the 32nd value. Untraced timings were 93 to 120 microseconds and are dominated
  by host dispatch, so they do not show the grid's device-time effect.
- A grid wider than 10 columns fails: `TT_FATAL: !check_kernel_logical_cores.contains(coreCoord)` (program.cpp:559). The merge core is at (10,0), so
  the scan grid must stay at 10 columns or fewer. That is a limit on the `GRID` constant.
- Router `RouterTopK` check is INVALID. It disagreed with the reference in 40 of 40 trials, including at the bundle's own 512 experts and top-k 10,
  so the harness is wrong (the input layout is the suspect), not the op. No router claim comes from this check.
