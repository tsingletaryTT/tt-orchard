# Classifying the unmapped ops of the Qwen4 trace (2026-11-02)

Inventory run on the Flash-Next dimensions (`tests/fixtures/model-configs/qwen3.8-flash-next.json`, the text config flattened to the top level with `model_type` `qwen4_exp_text` and an `eos_token_id` added, because the PLE layers refuse a config without one). 4 layers, 64 positions, `--stub-data-dependent`, the reference interpreter (transformers 5.19.0), meta device, ttnn 0.80.0 name dump. Everything ran on the CPU. No chip was opened. The trace is committed as `tests/fixtures/ops-inventory/raw-qwen4-flash-next.json` (the op list is whole; the ordered sequence is cut to the layer-0 `mlp.experts` module).

## Summary before and after

| Status | Before | After |
|---|---:|---:|
| mapped | 39 | 39 |
| composable | 1 | 2 |
| verified | 10 | 10 |
| candidate | 27 | 27 |
| unmapped | 7 | 3 |
| unchecked | 0 | 0 |
| ignored | 3 | 3 |
| host | not a status | 3 |

Op groups before: `rms_norm` (145), `silu_mul` (96). After: the same two, plus `moe_experts` -> `experimental.moe_compute` (48), status `pattern`.

## What changed for each op

| Op | Count | Now | Why, and the evidence |
|---|---:|---|---|
| `aten.unflatten.int` | 20 | composable (leaf `aten.view.default`, ttnn `reshape`) | A view in the hyper-connection and PLE modules. The probe built the gated residual with `ttnn.reshape` and got PCC 0.99996 on the mixed output and 0.9995 on the injection weights (`2026-11-02-qwen4-ops-probe.md`). The PLE module was not run on its own. |
| `aten.eye.default` | 3 | host | A constant of the GDN chunk path. `ttnn.transformer.chunk_gated_delta_rule` takes optional `eye`, `tril`, `ones` and mask constants. The probe ran the op without them and matched the reference (PCC at least 0.99998), so passing them is an option nobody has tried. |
| `aten.cummax.default` | 2 | host | The EOS-aware shift of the n-gram embedding. ttnn 0.80.0 has no `cummax`. The upstream port runs the hash, the shift and the 102 GB table lookup on the host. |
| `aten.diff.default` | 1 | host | Source: `find_packed_sequence_indices` in transformers `masking_utils.py` (`torch.diff` of `position_ids`), called through `create_causal_mask` at the top level of `Qwen4ExpModel`. It prepares the attention mask and is not model math. It matters on the device only if serving packs several sequences into one batch row, because the host would then have to build the mask that implies. |

## What is still unmapped, and why

| Op | Count | Why |
|---|---:|---|
| `aten.histc.default` | 4 | Inside `mlp.experts`, the reference expert loop. |
| `aten.index_put_.default` | 4 | Same module. |
| `transformers.grouped_mm_fallback.default` | 8 | Same module. |

The device path replaces the whole `mlp.experts` module with `ttnn.experimental.moe_compute`, so these ops are not built from ttnn ops one by one. The group is recognised as `moe_experts` with status `pattern`. It is not verified. The probe did not run `moe_compute`, and a check needs 4 dies and routing at 512 experts, top 10. The three ops keep the status `unmapped` and each carries a note that points at the group. A dense Qwen4 has none of them.

## Limits

- `host` records a decision with a reason. Nothing timed the host path, so the table does not say the host keeps up.
- The `diff` source was found by reading the installed transformers code. The trace places the op in the top-level module, which matches.
- The `moe_experts` count of 48 is the layer-0 group times the 48 layers in the config. Only layer 0 is in the ordered sequence, so the count is an estimate.
