# Qwen4-style ops on a real chip (2026-11-02)

This run settles the open questions in `2026-11-02-qwen4-new-ops.md` ("Not settled") with a Blackhole chip. Each op was compared with the CPU torch reference from `transformers` 5.19.0 `qwen4_exp`. The probes are in `scripts/qwen4_ops_probe/`, one file per probe. Each takes shape arguments, prints one `PROBE_RESULT {json}` line, and has a CPU-only `--selftest`.

Build under test: the `ttnn` in `~/.tenstorrent-venv` (editable from a tt-metal checkout, built 2026-10-30, still reports 0.80.0). The box has two p300c boards. A one-chip request is granted a whole board, so each lease gave 2 chips. Every probe opened device 0 of its lease only.

## Answers

| Question | Answer |
|---|---|
| Does `ttnn.conv1d` accept dilation? | The argument exists and a dilated convolution with `groups=1` is correct (PCC 0.9999). |
| Does it work as a depthwise dilated conv (groups = channels, dilation 3)? | No. It returns a wrong result without raising (PCC 0.18 to 0.28 at C = 64, 128 and 2048). Depthwise with dilation 1 is correct (PCC 0.99999), so the fault needs both settings together. At C = 10240, T = 128 it raises "DRAM Auto slice could not find valid slice configuration". |
| Does the fallback (four shifted multiply-adds) work? | Yes. PCC 0.99999 at C = 64, 128, 2048 and 10240 (T = 128), max absolute error 0.06 on outputs up to 7.0. It needs no layout workaround beyond a ROW_MAJOR slice followed by `to_layout(TILE)`. |
| `chunk_gated_delta_rule` at Flash-Next head counts | Matches the reference. Output max abs error about 5e-4 on outputs up to 0.07 (relative to the largest value 0.5% to 0.9%), state max abs 4e-3 to 6e-3 on values up to 0.78. PCC at least 0.99998 in every setting, T = 128 and 512. |
| `fused_recurrent_gated_delta_rule`, one decode step, state carried from the chunk op | Matches. Output max abs 2.2e-4 to 3.1e-4, new state max abs 5e-3, PCC at least 0.99999. |
| Gated residual from ttnn ops | Matches. The mixed output has max abs 0.029 on values up to 1.3 (2% of the largest value, PCC 0.99996). The injection weights have max abs 0.010 (PCC 0.9995). No op needed a layout or dtype workaround. |
| `update_padded_kv_cache` | Runs on a 1x1 mesh and writes exactly the expected rows (max abs 0, two chunks at offsets 0 and 32). |

## Results

Error columns: max abs is the largest |ttnn - reference|. max rel is max abs divided by the largest |reference| value. PCC is the Pearson correlation. Reference arithmetic is fp32. Wall time is the first call (includes program compile) and the second call. For probes 1, 2 and 4 the call includes uploads and readback. For probe 3 it ends at `synchronize_device` and does not include readback of the result.

### Probe 1: depthwise dilated causal conv, kernel 4, dilation 3, left pad 9, then silu

Inputs and weights bf16. `conv1d` entries are compared before silu, `shifted` entries after silu.

| Shape (T, C) | Implementation | Max abs | Max rel | PCC | Wall (s) |
|---|---|---|---|---|---|
| 32, 64 | `ttnn.conv1d`, op pads (9, 0) | 4.37 | 1.14 | 0.180 | 0.63, 0.001 |
| 32, 64 | `ttnn.conv1d`, host pads | 4.37 | 1.14 | 0.180 | 0.16, 0.001 |
| 32, 64 | shifted multiply-adds | 0.020 | 0.0054 | 0.99999 | 0.010, 0.0005 |
| 128, 128 | `ttnn.conv1d` (both pad forms) | 7.86 | n/a | 0.282 | 0.70, 0.001 |
| 128, 128 | shifted multiply-adds | 0.032 | n/a | 0.99999 | 0.48, 0.0006 |
| 128, 2048 | `ttnn.conv1d` (both pad forms) | 9.48 | n/a | 0.250 | 0.71, 0.002 |
| 128, 2048 | shifted multiply-adds | 0.056 | n/a | 0.99998 | 0.55, 0.002 |
| 128, 10240 | `ttnn.conv1d` (both pad forms) | raised | | | |
| 128, 10240 | shifted multiply-adds | 0.060 | 0.0085 | 0.99999 | 0.54, 0.007 |

Diagnostic matrix (T = 32, C = 64, and T = 128, C = 128; `ttnn.conv1d` against `F.conv1d`):

| dilation | groups | PCC (C = 64) | PCC (C = 128) |
|---|---|---|---|
| 1 | 1 | 0.99997 | 0.99993 |
| 1 | C | 0.99999 | 0.99999 |
| 3 | 1 | 0.99997 | 0.99993 |
| 3 | C | 0.189 | 0.266 |

The first run of probe 1 failed at device open with an L1_SMALL out-of-memory error. `ttnn.conv1d` needs `l1_small_size` reserved at `open_device` (32768 was enough). The probe now passes it (`--l1-small`).

### Probe 2: `ttnn.transformer.chunk_gated_delta_rule`, reference `torch_chunk_gated_delta_rule` (chunk 64, fp32)

Flash-Next heads (16 key, 48 value, K = V = 128), batch 1. Rank-3 "flat" inputs are raw (the op normalises and scales) and need `num_key_heads=16`. Rank-4 "head" inputs are host-L2-normalised. q, k and v are cast to bf16 by the op in every setting. `g/beta` is the dtype of those two inputs. Output and state are fp32.

| T | Input form | Chunk | g/beta | o max abs | o max rel | o PCC | state max abs | state PCC | Wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| 128 | flat | 32 | bf16 | 5.0e-4 | 7.4e-3 | 0.99999 | 4.8e-3 | 0.99999 | 1.28, 0.005 |
| 128 | flat | 32 | fp32 | 5.8e-4 | 8.6e-3 | 1.00000 | 5.1e-3 | 1.00000 | 0.24, 0.004 |
| 128 | head | 32 | fp32 | 4.2e-4 | 6.1e-3 | 0.99999 | 5.0e-3 | 0.99999 | 1.09, 0.007 |
| 128 | head | 64 | fp32 | 5.3e-4 | 7.8e-3 | 0.99999 | 4.0e-3 | 0.99999 | 1.20, 0.006 |
| 512 | flat | 32 | bf16 | 5.7e-4 | 6.7e-3 | 0.99999 | 5.7e-3 | 1.00000 | 0.24, 0.017 |
| 512 | flat | 32 | fp32 | 5.7e-4 | 6.7e-3 | 0.99999 | 5.3e-3 | 1.00000 | 0.016, 0.014 |
| 512 | head | 32 | fp32 | 4.6e-4 | 5.5e-3 | 0.99999 | 5.3e-3 | 1.00000 | 0.43, 0.023 |
| 512 | head | 64 | fp32 | 5.8e-4 | 6.8e-3 | 0.99999 | 5.4e-3 | 1.00000 | 1.01, 0.023 |

Reference output magnitude is 0.07 to 0.08 (largest value) and state 0.77 to 0.78. Small shape (T = 64, 2 key heads, 4 value heads, K = V = 64) gave the same error level in all four settings (o max abs 3.9e-4 to 7.5e-4, PCC at least 0.99999).

A `compute_kernel_config` with LoFi and no fp32 accumulation is refused: "chunk_gated_delta_rule runs HiFi4 with fp32 destination accumulation and no approx mode on every path". There is no reduced-precision compute setting. The only precision choice is the g/beta dtype, and bf16 costs little (o max abs 5.0e-4 against 5.8e-4 for fp32, which is within noise). The error floor is set by the bf16 cast of q, k and v.

### Probe 3: `ttnn.transformer.fused_recurrent_gated_delta_rule`, one decode token

Reference `torch_recurrent_gated_delta_rule`, started from the torch chunk reference's state. `state_ttnn` feeds the device state that the chunk op (flat, chunk 32) returned. `state_torch` uploads the torch state as fp32. Each row was run with q/k at 16 key heads (the op expands) and expanded to 48 on the host; the two gave identical numbers.

| Prefill T | State source | g/beta | o max abs | o PCC | new state max abs | new state PCC |
|---|---|---|---|---|---|---|
| 128 | `state_ttnn` | fp32 | 2.6e-4 | 0.99999 | 5.7e-3 | 1.00000 |
| 128 | `state_ttnn` | bf16 | 2.7e-4 | 0.99999 | 5.7e-3 | 0.99999 |
| 128 | `state_torch` | fp32 | 1.7e-4 | 1.00000 | 3.5e-3 | 1.00000 |
| 128 | `state_torch` | bf16 | 2.1e-4 | 1.00000 | 4.2e-3 | 1.00000 |
| 512 | `state_ttnn` | fp32 | 2.2e-4 | 0.99999 | 5.2e-3 | 1.00000 |
| 512 | `state_ttnn` | bf16 | 3.1e-4 | 0.99999 | 5.2e-3 | 1.00000 |
| 512 | `state_torch` | fp32 | 2.2e-4 | 1.00000 | 3.7e-3 | 1.00000 |
| 512 | `state_torch` | bf16 | 3.1e-4 | 1.00000 | 3.6e-3 | 1.00000 |

Reference output magnitude 0.03 to 0.05, state 0.76 to 0.88. Wall time after the first call is 0.0004 to 0.0009 s (no readback in the timed span); the first call took 1.5 s at T = 128 and 0.02 s at T = 512 (program already compiled by then).

### Probe 4: gated residual composed from ttnn ops

Hidden 2560, hc_count 4, lowrank 320, bf16 activations and weights, random weights (norm weight 0.1 times a normal draw, linear layers at their default init, all rounded to bf16), against the torch `GatedResidualRef` (a copy of `Qwen4ExpTextGatedResidual`) in fp32. The grouped RMSNorm is `reshape` to (1, 1, 4T, 2560), `ttnn.rms_norm` without weight, `reshape` back, multiply by (1 + w).

| T | Output | Max abs | Max rel | PCC | Wall (s) |
|---|---|---|---|---|---|
| 128 | mixed input, `ttnn.mean(dim=2)` | 0.0285 | 0.0236 | 0.99997 | 3.04, 0.053 |
| 128 | mixed input, four slices times 1/4 | 0.0246 | 0.0204 | 0.99997 | same call |
| 128 | injection weights | 0.0090 | 0.0072 | 0.99955 | same call |
| 512 | mixed input, `ttnn.mean(dim=2)` | 0.0285 | 0.0214 | 0.99996 | 2.23, 0.149 |
| 512 | mixed input, four slices times 1/4 | 0.0268 | 0.0202 | 0.99996 | same call |
| 512 | injection weights | 0.0104 | 0.0083 | 0.99954 | same call |

Small shape (T = 32, hidden 256, lowrank 64) gave max abs 0.007 and 0.005 with the same PCC level. No step raised, so no workaround was needed: `rms_norm` on the reshaped tensor, `mean` over a size-4 axis (tile padded), `linear` to 4 columns and the activations all worked in bf16 TILE layout. The mean over the stream axis and the slice-sum form agree to the same error level, so the fused `mean` can be used. The error is larger than for the GDN ops because every stage rounds to bf16. The reference computes in fp32 and the injection weights have the lowest PCC (0.9995) because they sit near 1 and the sigmoid's slope there amplifies the bf16 input rounding.

### Probe 5: `ttnn.experimental.deepseek_prefill.update_padded_kv_cache`

Argument layout (binding doc string, scalar form): `(cache, input, slot_idx, layer_idx, num_layers, kv_actual_global, cluster_axis, valid_global=None, tp_axis=None, rope=None, scales=None)`. A second form takes `slot_idx` and `kv_actual_global` as 1-element uint32 tensors and is trace-safe. Constraints from the doc string: 4D cache and input in the same layout, dtype and head dimension (block-float needs TILE, FP8_E4M3 needs ROW_MAJOR on Blackhole); the cache is sharded across `cluster_axis` with the outermost dimension equal to `num_slots * num_layers`; sequence dimensions stay 32-aligned; `kv_actual_global` is tile-aligned; the op writes in place and returns the cache.

Setup run: 1x1 mesh, cache (4, 1, 128, 128) bf16 TILE zeros, slab (1, 1, 32, 128), `slot_idx` 0, `layer_idx` 1, `num_layers` 2, `cluster_axis` 0, two calls with `kv_actual_global` 0 and 32. The cache read back equals the CPU expectation exactly (max abs 0, `torch.equal` true). The multi-device offset logic (each device deriving its own offset from its mesh coordinate), the tensor form, `valid_global`, `tp_axis`, `rope` and `scales` were not run.

## Name dump

`orchard/data/ttnn_ops-0.80.0.json` has 821 names from an earlier build. `python3 -m orchard.cli inventory refresh-ttnn` would overwrite that file under the same name, and the new build reports the same version string, so two builds would share one name. The refresh was not run through the CLI. The same dump code ran under a one-chip lease with its output written to a scratch directory, and the result is stored as `orchard/data/ttnn_ops-0.80.0-build-20261030.json` (829 names, a `build_note` field). The inventory loader reads `ttnn_ops-<version>.json` only, so the new file is not used by anything yet.

The new dump adds 8 names (`attach_absent_mcast`, `experimental.can_use_fused_msda_from_offsets_l1_value`, `experimental.fused_msda`, `experimental.fused_msda_from_offsets`, `experimental_get_l1_occupied_ranges`, `fabric_connection_rt_args_size`, `get_num_links`, `index_fill_new`) and removes none. It does not list `experimental.deepseek_prefill.update_padded_kv_cache` either, although the op exists and ran above. The dump script walks one level into `experimental` and `transformer` and skips submodules, so ops inside `experimental.deepseek_prefill` are never listed, and `experimental.moe_compute_utils` is a module and is skipped for the same reason. The "missing" name in the Flash-Next study comes from the dump script skipping submodules. The build has the op. Walking submodules is a change to `orchard/ttnn_dump.py` that has not been made.

## Not run

- `moe_compute_utils`, `indexer_score_dsa`, `sparse_sdpa`: outside the brief's list of probes. Only their names were checked.
- Multi-chip use of any op (the Flash-Next port shards over 4 chips). All probes used one chip.
- `chunk_gated_delta_rule` with a non-zero `initial_state`, batch above 1, and T not a multiple of the chunk size.
- `fused_recurrent_gated_delta_rule` with more than one token, ring mode and per-token states (speculative verify).
- A direct comparison with the Flash-Next port's own GDN kernels, and any weights from the real model. Inputs are random and the decay and beta ranges are chosen (g between -0.52 and -0.02, beta a sigmoid of a normal draw), so the errors do not describe the trained model.
- `conv1d` at C = 10240 with other slice settings (`slice_config`) or sharding overrides. Since the dilated depthwise result is wrong at smaller C, this was not pursued.

## Commands

All probes ran under gozer leases of the form below. The first two lease runs executed a shell script that called the probes one after another (`timeout` per probe); the interpreter was `~/.tenstorrent-venv/bin/python` (ttnn 0.80.0).

```
gozer run --chips 1 --who "claude:qwen4-ops-probe" --reason "probe Qwen4-style ops" -- bash <script>
```

The script lines (small shapes, then Flash-Next shapes):

```
p1_conv1d.py --seq 32 --channels 64            p1_conv1d.py --seq 128 --channels 10240
p2_chunk_gdn.py --seq 64 --heads-k 2 --heads-v 4 --dim-k 64 --dim-v 64
p2_chunk_gdn.py --seq 128                       p2_chunk_gdn.py --seq 512
p3_recurrent_gdn.py --seq 64 --heads-k 2 --heads-v 4 --dim-k 64 --dim-v 64
p3_recurrent_gdn.py --seq 128                   p3_recurrent_gdn.py --seq 512
p4_gated_residual.py --seq 32 --hidden 256 --lowrank 64
p4_gated_residual.py --seq 128                  p4_gated_residual.py --seq 512
```

The diagnostic runs (`p1_conv1d.py --seq 32 --channels 64 --diagnose`, `--seq 128 --channels 128 --diagnose`, `--seq 128 --channels 2048`, and `p5_update_padded_kv_cache.py`) ran under a lease taken with `gozer run` that queued (exit 10), then `gozer wait <ticket>`, the granted `TT_VISIBLE_DEVICES`, and `gozer release <lease>`. The name dump used the repository's own `orchard.ttnn_dump.main`, which takes its own lease.

The first lease run of probe 1 at small shape failed with the L1_SMALL error described above; the rows in the table are from the rerun with `--l1-small 32768`.

## Reference code

`scripts/qwen4_ops_probe/common.py` holds torch copies of `torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`, the grouped RMSNorm, the gated residual and `_short_conv`. `tests/test_qwen4_ops_probe.py` runs each probe's `--selftest`, checks that no probe imports ttnn in selftest, and, when `ORCHARD_REFERENCE_PYTHON` points at an interpreter with `transformers` 5.19.0, checks the copies against the transformers functions.
