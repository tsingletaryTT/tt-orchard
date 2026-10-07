# 2026-10-06: sidecar parity prototype on Cloudflare/clef

Contract: `docs/superpowers/plans/2026-10-06-sidecar-parity.md` (on the `bringup-command` branch). This log
covers the three templates, the hardware runs, the numbers and the values suggested for the gate.
Every number is one measurement of one fixed set of six records unless it says otherwise.

## What was built

`orchard/skills/sidecar-parity-templates/` (branch `worktree-agent-a68804351d82feba5`):

| File | Job |
|---|---|
| `prepare_parity.py` | Copies the nearest bundle's `run.sh` to `parity-run.sh` with the `HERE=` line fixed and the final `exec "${CMD[@]}"` replaced by an exec of the bundle's python on `hidden_parity.py`. Sets `QWEN36_DRAFTER=""`, `HF_MODEL` and `MODEL_WEIGHTS_DIR` (the model dir), `TT_CACHE_PATH`, `TT_CACHE_HOME` and `HF_HUB_OFFLINE=1` just before the exec. Builds `model-dir/` from the backbone files only. Exit 2, naming the key or edit, when a config key is missing, an edit does not happen exactly once, or a required file is absent; nothing is written in that case |
| `hidden_parity.py` | The check. Fixed text-only records, sha256 gate on the sidecar code, CPU reference in bf16 (run twice with 16 and 8 threads), device hidden states from a copy of the `prefill_tp` layer loop over all rows plus the model's own final norm, a wiring check against `prefill_tp`, the head on the host, `evidence/sidecar-parity.json`. Phases `cpu`, `device` or `all`; the CPU reference is saved and reused when the records and layer limit are the same |
| `run_sidecar_checks.py` | The one hardware command for stage 2. Runs `serve_and_compare.py`, then `parity-run.sh`, always both, each in its own session and process group, stops each group in a `finally`, turns SIGTERM into an exit, and writes `evidence/sidecar-check.json` with the merged `result_draft` (swap draft fields, `sidecar_parity` copied verbatim, `evidence` paths relative to the run dir) and a `failure` text when a half failed. The exit code is the swap code, else the parity code, else 1 |

Tests: 116 (`tests/test_sidecar_prepare_parity.py` 24, `test_sidecar_hidden_parity.py` 69,
`test_sidecar_run_checks.py` 23). None imports ttnn or opens a device. The mutation runs used 23, 60 and
26 mutations on the three files. The survivors each led to a test, except one: the line that attaches
per-question detail inside the device-only function cannot be tested without a board, and the hardware run
checked it. Commits: see the end of this file.

The hidden-state read-back handles both layouts (a full-width replica per device, or a slice per device).
The board returned full-width replicas, so one replica is taken.

## Hardware runs

Machine: QuietBox 2, board 0 (two dies, mesh 1x2), bundle `episod/qwen3.8-27b-dflash2-p300`, ttnn
`0.79.0.dev20260929+qwen36p150x2.g9f6b02d`, torch 2.14.1+cpu, transformers 5.17.0. Clef revision
`2f3de3dd85f379784083b0814d997ab627200f0c`. One board lease for each device run, owner pid recorded, released
with `gozer release`. Board 1 was held by another agent (`claude:rolefit`) the whole time and was not touched.

### 4-layer smoke (first device run)

The first 4 layers (three DeltaNet, one full attention) on both sides, to check the plumbing quickly.

- hidden PCC min 0.99953, mean 0.99960
- prob_max_abs_diff 0.00258, top-1 agreement 15 of 16 (0.9375)
- wiring check cosine 0.99994
- model load 22 s; first record 109.6 s (kernel compile), the rest 0.02 to 12.7 s; peak RSS 16.0 GB

### Full 64 layers

| Quantity | Run 1 (cold tensor cache) | Run 2 (warm cache) |
|---|---|---|
| hidden_pcc_min | 0.95399 | 0.95399 |
| hidden_pcc_mean | 0.96760 | 0.96760 |
| prob_max_abs_diff | 0.12333 | 0.12333 |
| top1_agree_fraction | 0.9375 (15 of 16) | 0.9375 |
| wiring cosine (record 0, last row) | 0.99990 | 0.99990 |
| noise floor, prob_max_abs_diff | 0.0 | 0.0 |
| model load | 168 s (weights converted) | 52 s |
| device time per record | 0.2 to 0.5 s | 0.2 to 0.3 s |
| peak RSS | 52.3 GB | 52.3 GB |

The two runs match to every printed digit, so the device path is repeatable on this input.

Per record (hidden PCC, tokens): outage 0.971 (363), invoice 0.982 (274), review 0.965 (410),
access 0.961 (281), support-thread 0.954 (501), deploy 0.972 (369). PCC is lower for the longest record.

CPU reference (full model, 16 cores, HF's pure-PyTorch fallback for the gated delta rule): 84, 54 and 53 s for
the first three records (cold page cache), then 7 to 12 s each; the second pass with 8 threads took 5 to 10 s
per record. About 5 minutes for the whole reference.

Why the full-model PCC is lower than the 4-layer one: the TT weights are bfp8 (the tensor cache directory is
`tensor_cache_bfp8_mesh1x2`) and the reference is bf16, so the rounding error grows with depth. The wiring
check agrees to 0.9999 at 64 layers, so the loop this script runs and the model's own `prefill_tp` give the
same last row; the lower PCC is the model's precision and not the script. This is a reading of the numbers.
No run isolated the quantization from other sources.

The one question where the top choice differs is `review` / `topic`. Reference: durability 0.588, service
0.389. Device: service 0.512, durability 0.466. That is a near tie (device margin 0.046) and it is also the
largest probability difference (0.123). The other 15 questions differ by at most 0.043 per option.

### Noise floor

The CPU reference was run twice, with 16 and 8 torch threads. The probabilities were identical (difference
exactly 0.0). So the noise-floor rule in the plan (`max(SIDECAR_PROB_DIFF_MAX, 4 * noise floor)`) adds
nothing on this machine for this input: four times zero is zero. The fixed threshold carries the check.

## Things that were not as predicted, or that the plan left open

- **The loader ignores the sidecar file in the weights directory.** A 4-layer run with `HF_MODEL` and
  `MODEL_WEIGHTS_DIR` pointing at the raw Clef snapshot (which holds `joint_head.safetensors`,
  `joint_schema_model.py` and the README) gave the same numbers as the run on `model-dir` (hidden PCC min
  0.99953, 15 of 16) and printed nothing about the extra files. The loader reads only the files named in the
  index. `model-dir` still leaves them out, so a future loader change cannot read them by accident.
- **`Qwen36Model` loads the Clef snapshot directly.** Clef's own `config.json` works; the nearest model's
  config is not needed in the model dir. The first full run converted the weights into
  `/mnt/bonus/models/orchard-runs/sidecar-parity-cache` in 168 s. That cache holds Clef's bfp8 tensors under
  `P300/tensor_cache_bfp8_mesh1x2` and carries the `.orchard-model` marker `Cloudflare/clef@<revision>`, the same
  label stage 0 uses. A swap check that uses the same `tt_cache` would reuse it; the label must match.
- **The plan's `model_id` is not a config key.** The cache marker id is taken from the snapshot path
  (`models--Cloudflare--clef/snapshots/<rev>` becomes `Cloudflare/clef@<rev>`); an explicit `model_id` in the
  config overrides it. Any other path is its own label.
- **A child that leaves a background process holds its output pipe open.** The first version of
  `run_sidecar_checks.py` read the pipe until end of file and waited for that process (a test hung for the
  length of a `sleep 300`). A reader thread, a 2 s grace after the child exits and a group stop fix it.
- **The CPU phase needs no lease.** `--phase cpu` imports no ttnn and opens no device, so a stage can run it
  before taking the board. `--phase all` (the default) does both, which keeps the board for the roughly
  5 minute reference. Splitting the phases between stage 1 and the lease is a possible saving.
- **The first record on a cold kernel cache costs minutes.** 109.6 s for 4 layers on the first run. The 64-layer
  runs were fast because the 4-layer run had already compiled the same kernels. A first-ever run on a fresh
  kernel cache should budget for the swap check's cold-compile figure (more than 26 minutes was measured for
  the server on 2026-10-03).
- **Run-to-run determinism.** Two full device runs gave identical numbers. That is two runs.

## Suggested gate values

`gate_sidecar_parity` needs three thresholds. They come from one model and one set of 16 questions, so the
margins below are judgement, not a distribution.

| Constant | Suggested | Measured | Reasoning |
|---|---|---|---|
| `SIDECAR_PCC_MIN` | 0.90 | min 0.954, mean 0.968 | About five points under the lowest record. PCC fell with length (0.982 at 274 tokens, 0.954 at 501), so a longer record needs room. A wrong loop, a wrong norm or another model's weights gave values far below this in the cases reasoned about; none of those was run. The 4-layer figure (0.9995) shows how close the layers are before quantization error builds up |
| `SIDECAR_TOP1_MIN` | 0.85 | 15 of 16 = 0.9375 | One question is 0.0625 of 16. 0.85 allows 2 flipped questions; the one seen was a near tie. With the gate's minimum of 10 questions a single flip costs 0.10, so 0.85 still passes one flip and fails two |
| `SIDECAR_PROB_DIFF_MAX` | 0.25 | 0.123 (the tie); next largest 0.043 | About twice the measured maximum. A near-tie flip can move one option's probability by more than 0.1, as seen. Because the noise floor is 0.0 here, the plan's `max(that, 4 * noise floor)` equals this value |

`WIRING_COSINE_MIN` in the script is 0.99 (measured 0.9999 at 64 layers and 0.99994 at 4); the plan's 0.999
would also have passed. It is a config key (`wiring_cosine_min`) with a default, not a gate constant.

Not measured: records longer than 501 tokens, image or video records, a 4-chip mesh, a second model, a
different set of records, the real stage 2 flow (`run_sidecar_checks.py` against a real `serve_and_compare.py`
on the same lease). The last one matters: whether the board is ready to be opened again right after the
swap check's server has been stopped was not tested. The parity runs here each started on a board that
`gozer release` had reset.

## Command lines

Stage directory `/mnt/bonus/models/orchard-runs/sidecar-parity-proto`; every device run took a lease first.

    cp prepare_parity.py hidden_parity.py <stage dir>/        # and write parity_config.json (keys as in the plan)
    python3 prepare_parity.py
    bash parity-run.sh --phase cpu                             # no lease needed
    gozer acquire --chips 1 --who "claude:sidecar-parity" --reason "..." --owner-pid <recorded sleep pid>
    TT_VISIBLE_DEVICES=0000:01:00.0,0000:02:00.0 bash parity-run.sh --phase device
    gozer release <lease>

`parity_config.json` for the full run: `run_dir`, `bundle_dir`, `model_snapshot`, `head_file`
(`joint_head.safetensors`), `head_config` (`joint_head_config.json`), `code_file` (`joint_schema_model.py`),
`code_sha256` (`0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3`), `tt_cache`,
`mesh_shape` `[1, 2]`, `n_layers` `null`. The head file's sha256 is
`a010ac04f078e699988e4049cbea5e62c962393f59fec366640b64e8d69a4953`.

Evidence kept in this repo: `docs/run-logs/samples/2026-10-sidecar-parity-clef-full.json` (run 2, with
per-question probabilities) and `2026-10-sidecar-parity-clef-4layer.json`. The rest is under the stage
directories `/mnt/bonus/models/orchard-runs/sidecar-parity-proto/evidence/` (logs, `cpu-reference.pt`) and
`/mnt/bonus/models/orchard-runs/sidecar-parity-direct/evidence/` (the raw-snapshot check).
