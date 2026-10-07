# Plan: the sidecar parity check (Phase 3 of the bringup spec, section 6)

For a model of class `weights+sidecar` (first case: `Cloudflare/clef`). The backbone is checked by the
existing weights-swap tests. This check covers the extra head: the device must produce the backbone's final
hidden states, and the host-side head run on those states must agree with the same head run on the CPU
reference's states.

## The script: `orchard/skills/sidecar-parity-templates/hidden_parity.py`

Runs inside the nearest model's tt-model bundle venv, through a copy of the bundle's `run.sh` whose last line
(`exec "${CMD[@]}"`) is replaced by `exec "$PYBIN" hidden_parity.py ...`. That gives it the exact environment
the server uses (LD_PRELOAD, TT_METAL_HOME, visible devices, hermetic caches). The launcher is written by a
`prepare_parity.py` template, in the way `prepare_swap.py` writes the edited `run.sh`.

Input, `parity_config.json` (all paths absolute):

    {"run_dir": "...", "model_snapshot": "<new model snapshot>", "base_snapshot": "<nearest snapshot>",
     "head_file": "joint_head.safetensors", "head_config": "joint_head_config.json",
     "code_file": "joint_schema_model.py", "code_sha256": "<from stage 0>", "records": "records.json",
     "tt_cache": "<new dir>", "mesh_shape": [1, 2], "n_layers": null}

What it does:
1. Load the records (`records.json`, fixed in the template: a small set of text-only `state` + `questions`
   records, 6 to 8 of them, written by a function, never by the agent). Encode them with the sidecar code's
   own `encode_record` (the code file is imported only after its sha256 matches `code_sha256`).
2. CPU reference: build the CPU model the way `load_release_model` does (Hugging Face backbone in bf16 plus
   the head) and record, per record, the backbone's `last_hidden_state` over all positions and the head's
   logits. This is the reference, and it is written to `evidence/cpu-reference.pt`.
3. Device: open a mesh (FABRIC_1D, `GDN_CONV1D_L1_SMALL_SIZE`), `Qwen36Model.from_pretrained` on the
   snapshot, then for each record run the layer loop of `prefill_tp` over all rows, apply `model.norm`, and
   read one replica back: a `[T, 5120]` tensor. No lm head. No change to tt-metal.
4. Wiring check: for the last row of one record, apply `model._lm_head` to the script's own normed row and
   compare the logits with `model.prefill_tp` on the same tokens (cosine similarity at or above 0.999). If
   they differ, stop with exit 6: the loop the script runs no longer matches the model.
5. Host head: run the head (CPU, bf16 or float32, same as the reference) on the device's hidden states, with
   the same `lm_head.weight` the reference uses.
6. Compare and write `evidence/sidecar-parity.json` (below). Exit 0 when measured, 3 if the self check
   fails, 4 if the device cannot be opened, 5 if the code hash does not match, 6 if the wiring check fails.

## Result: `evidence/sidecar-parity.json`

    {"measured": true,
     "n_records": 6, "n_questions": 17,
     "hidden_pcc_min": 0.0, "hidden_pcc_mean": 0.0,        # per record over all positions, then min and mean
     "prob_max_abs_diff": 0.0,                             # over every option of every question
     "top1_agree_fraction": 1.0,                           # per question, argmax of the head's probabilities
     "noise_floor": {"prob_max_abs_diff": 0.0},            # CPU vs CPU, different thread counts
     "wiring_check": {"cosine": 0.9999, "passed": true},
     "tt_metal_sha": "...", "code_sha256": "...", "head_sha256": "...", "mesh_shape": [1, 2],
     "label": {"hidden_pcc_min": "measured", "...": "measured"}}

`gate_sidecar_parity` (orchard/stages.py) needs: `measured` true, `wiring_check.passed` true,
`n_questions` at least 10, `hidden_pcc_min` at or above `SIDECAR_PCC_MIN`, `top1_agree_fraction` at or above
`SIDECAR_TOP1_MIN`, and `prob_max_abs_diff` at or below the larger of `SIDECAR_PROB_DIFF_MAX` and four times
the measured noise floor. The three thresholds are set in `orchard/defaults.py` as choices after the first
hardware measurement, with the measurement next to them.

## How the stages use it

On the `weights+sidecar` class stage 2 runs two hardware commands under one lease, one after the other:
the existing `serve_and_compare.py` and then `hidden_parity.py` through its launcher. A wrapper template,
`run_sidecar_checks.py`, runs both as subprocesses and exits non-zero if either fails, so the supervisor's
single `hw_test.json` command stays a single command. `result.json` gains a `sidecar_parity` field copied
from `evidence/sidecar-parity.json`, and `gate_weights_swap_sidecar` checks the swap fields as before and
then `gate_sidecar_parity`. Stage 4 repeats the parity check on the 2-chip configuration only in the first
version. Other configurations are listed in the bundle as not tested.
