# Real port: Granite-4.0-H-1B (Mamba-2 hybrid) with a coder-written kernel

Started 2026-10-28. Model choice and rejected candidates: `2026-10-28-real-port-choice.md`. This file holds the run.
Updated through the night; the last section says where it stands.

Rule of the run: the missing kernel is written by the local coder (Qwen3-Coder-Next fused-decode package, p300 profile) through the
kernel probe. The outside helper was switched off for the whole run (`telegraph.enabled = false` in the config used) and was not
asked anything.

## Summary (state at the last update)

| step | result |
|---|---|
| harness `bringup` on the model | stops at preflight, `nearest-model-missing` (no bundle serves a Mamba-2 hybrid); the full-port route has no unattended path (issues 61 and 90), so stages 3 and 5 were driven by hand |
| (a) CPU reference | a hand-written one-token step model agrees with transformers fp32 to 1e-6 relative L2 on 4 prompts (all greedy tokens equal); op-level arrays recorded for two mamba layers |
| (b) port on one chip, conv and SSD on the host | greedy tokens: 3 of 3, 24 of 24, 8 of 24, 24 of 24; the bf16 CPU run gives the same 3, 24, 8, 24 |
| composed baseline (conv and SSD from stock ttnn ops on the chip) | works at the same agreement; 22.6 tok/s |
| (c) coder kernels, simulator | `ssd_state_update` 4 of 5; `ssd_readout` 4 of 10 (0 of 5 at temperature 0, 4 of 5 at 0.7) |
| (c) coder kernels, hardware | `ssd_state_update`: runs on the chip and is in the model. `ssd_readout`: 0 of 7 simulator-passing kernels compile on the chip's tt-lang |
| (c) compact state update (`ssd_state_update_bcast`) | simulator 2 of 10; hardware 1 of those 2 |
| (d) kernel in the model | state-update kernel: agreement at the bf16 floor, 16.7 tok/s; compact kernel 18.0 tok/s; composed baseline 22.6 to 23.5 tok/s. Both kernels are slower than the baseline. Floor share 9.5% to 10.3% against 12.9% to 13.4% |

## (a) CPU reference

* `granite_ref.py`: Granite-4.0-H-1B in plain torch, one token per step. Compared with transformers 5.19.0
  `GraniteMoeHybridForCausalLM` (fp32, CPU, no `mamba_ssm`) on 4 prompts, 24 new tokens each, teacher forced: relative L2
  of the logits 6e-7 to 3e-6, argmax agreement 1.0 on every prompt. (`hf_vs_step_model.json`.)
* Prompt 0: the residual stream after every layer and after every mixer at each prompt position, the final logits, and the inputs and
  outputs of the SSD update at layers 0 and 17 at one decode position, recorded as arrays.
* bf16 control on the CPU (transformers, bf16): free-running greedy tokens equal to fp32 for 3, 24, 8 and 24 tokens (prompt 2 differs
  at token 8 from rounding), teacher-forced top-1 agreement 1.0, 1.0, 0.875, 1.0. Any chip run is read against this.

## (b) the port, with the missing op on the host

One Blackhole chip, bf16 weights (3 GB), one token per step, prefill is a loop of steps. Norms, projections, the 4 attention layers
(a GQA decode from matmul, softmax and a row-major cache), the MLP and the tied lm_head run in ttnn. The conv step, the Mamba-2 state
update and the readout run in torch on the host (`ssd_mode host`).

| prompt | prompt tokens | teacher-forced rel L2 (mean / max) | top-1 | top-5 overlap | free-run tokens equal / total |
|---|---|---|---|---|---|
| 0 | 5 | 0.0088 / 0.0128 | 1.00 | 1.00 | 3 / 3 |
| 1 | 16 | 0.0172 / 0.0467 | 1.00 | 0.99 | 24 / 24 |
| 2 | 7 | 0.0064 / 0.0140 | 0.875 | 0.97 | 8 / 24 |
| 3 | 12 | 0.0173 / 0.0371 | 1.00 | 0.96 | 24 / 24 |

bf16 CPU control, same prompts: rel L2 mean 0.0047, 0.0121, 0.0043, 0.0111; top-1 1.0, 1.0, 0.875, 1.0; prefix 3, 24, 8, 24.
Prompt 2 diverges at the same token in the control, so the divergence is bf16 rounding.

Per layer (prompt 0, 5 positions): the relative L2 of the residual stream against fp32, next to the bf16 CPU run, is in the table
below (every layer within 2.7 times the bf16 CPU error; the first-divergence probe, with a tolerance of 1.5 times the worst bf16 CPU
layer, finds no block outside it). The noise-floor tool of `orchard/noise_floor.py` reads audio-layout runs, so it was not used; the
bf16 CPU run is the floor and the limits of that choice are in the issues below.

| layer group | chip / fp32 | bf16 CPU / fp32 |
|---|---|---|
| layers 0 to 4 | 0.0035 to 0.0063 | 0.0035 to 0.0059 |
| layer 5 (attention) | 0.0225 | 0.0084 |
| layers 6 to 24 | 0.0052 to 0.0171 | 0.0028 to 0.0099 |
| layer 25 (attention) | 0.0264 | 0.0222 |
| layers 26 to 39 | 0.0152 to 0.0239 | 0.0128 to 0.0199 |

## Composed baseline

The conv and the SSD update written from stock ttnn ops on the chip (`ssd_mode composed`: broadcast multiply, add, exp, softplus and a
matmul for the readout). It works, at the same agreement (teacher-forced rel L2 mean 0.0074, 0.0182, 0.0073, 0.0138). So a kernel is
not needed for correctness here. It is the way to avoid the ten or so small ops per layer, and the run measures whether the coder's
kernel is that.

## (c) the kernel loop

Probe settings: `kernel-probe --library others` (the worked kernels of the other ops are in each attempt's reference directory; the
op's own is not), simulator checks at 64 x 128, three seeds, PCC 0.999 and relative L2 bars. Coder: Qwen3-Coder-Next fused-decode
package, p300 profile.

| round | op | attempts | simulator pass | failure classes |
|---|---|---|---|---|
| state-r1 | `ssd_state_update` (elementwise) | 5, temperature 0 | 4 | 1 `no_kernel` |
| readout-r1 | `ssd_readout` (row reduction) | 10 (5 at 0, 5 at 0.7) | 4 (0 of 5 at 0, 4 of 5 at 0.7) | 3 `numeric_mismatch` (PCC 0.09 to 0.18), 3 `tt_lang_error` (incomplete dataflow-buffer operations, a store shape mismatch, a copy from a reserved block) |

The passing readout kernels follow the library's rmsnorm kernel (`reduce_sum` over the row, then `broadcast`). The library was on, so
the reduction pattern came from a worked example and was not derived from the specification.

Hardware. The probe's hardware phase leases `--exact 1`, which is chip 1 of board 0, where the coder runs. The first hardware round
(`readout-hw1`) made 6 hardware calls that all ended `busy` while board 1 was free; that is a harness defect (issue below).
Run by hand on a free chip (shared environment, tt-lang 1.1.6): all 7 simulator-passing readout kernels fail to compile with
`reduce_sum() got an unexpected keyword argument 'shape'`. The simulator build in the pin is tt-lang specification 0.20, and the
chip's package (1.1.6, the newest on PyPI) has the older signature. The state-update kernel uses only syntax both accept and runs.

## (d) the kernel in the model

The coder's `ssd_state_update` kernel runs on the chip at the model's full state shape (3072 x 128) with the decay, dt*x and B
expanded to that shape first (the specification of the op takes them expanded). The readout stays composed.

Decode speed, batch 1, no trace, 64 greedy tokens after a 12-token prompt, one warm-up and five timed repeats, one chip. The floor is
the one-die bf16 estimate from `orchard/decode_floor.py` (175.2 tok/s: 2.92 GB per token at 512 GB/s).

| mode | tok/s mean (min to max) | share of the floor |
|---|---|---|
| host (conv and SSD in torch) | 19.0 (17.4 to 20.9) | 10.8% |
| composed (stock ttnn ops) | 22.6 (21.3 to 23.2) | 12.9% |
| coder state-update kernel + composed readout | 16.7 (14.9 to 17.8) | 9.5% |

Agreement with the kernel in the model (4 prompts): teacher-forced rel L2 mean 0.0061, 0.0202, 0.0088, 0.0153; top-1 1.0, 1.0,
0.875, 0.958; free-run prefix 3, 24, 8, 15 of 3, 24, 24, 24. The difference from host mode is prompt 3 (15 instead of 24),
a near tie that bf16 rounding can flip; it was not rerun.

The kernel is slower than the composed path because it needs five full-size inputs, and making the expanded copies of the per-head
values costs more than the fused update saves. Dispatch dominates every mode (all three are within 25% of each other and about 10%
of the floor), so this benchmark cannot show what a fused kernel is worth; it is not traced.

## Where it stands

Update 21:45 Pacific, 2026-10-28.
* The reduction half is stopped by a toolchain mismatch. The pinned simulator (specification 0.20) requires
  `reduce_sum(x, dims=..., shape=...)`; the chip's tt-lang 1.1.6 rejects `shape`. Checked both ways: a passing kernel with `shape`
  removed fails on the simulator with `reduce_sum() missing 1 required positional argument: 'shape'`. No kernel can pass both with a reduction, so a coder-written
  readout cannot reach the chip until the hardware tt-lang is built from the pinned checkout (issues 95 and 205).
* A hardware round with the coder seeing hardware errors (`--phase hw`) could not run: board 1 was held by other work all night, and
  the probe's hardware lease named chip 1, which is on the coder's board (issue 204, PR 209). Two such rounds produced 12 hardware
  calls, all `busy`.
* `ssd_state_update_bcast`, the state update with compact inputs (per-row scalars in column 0 of a [rows, 32] tensor, B in row 0 of a
  [32, 128] tensor), was the next attempt to make the kernel pay off: it needs a broadcast inside the kernel and avoids the full-size expanded
  inputs. Ten attempts (5 at temperature 0, 5 at 0.7), simulator: 2 pass (attempts 3 and 8). Failures: 4 `numeric_mismatch` (PCC 0.21 to 0.30), 2
  `tt_lang_error` (shape mismatch in a binary operation; in `store()`), 1 `shape_mismatch` (a tensor shape against a block), 1 `no_kernel`.
  Hardware, by hand on a free board: attempt 3 fails to compile (`Unknown function: ttl.grid_size`, called inside a thread; the chip's
  tt-lang accepts it only outside the threads), attempt 8 compiles and is correct at 64 x 128 and at 3072 x 128 (PCC 1.0 and 0.99999,
  relative L2 0.0033). So 1 of the 2 simulator passes runs on the chip, a second kind of simulator-passes-hardware-fails mismatch.
* The attempt-8 kernel in the model (compact kernel for the state update, readout composed): teacher-forced rel L2 mean 0.0044, 0.0180,
  0.0086, 0.0183, top-1 1.0, 1.0, 0.917, 0.958; free-run prefix 3, 24, 14, 15 of 3, 24, 24, 24 (the host mode gave 3, 24, 8, 24 and the
  composed mode 3, 24, 14, 11 on the same prompts, so the differences are bf16 near ties). Decode 18.0 tok/s (17.8 to 18.2, 5 repeats) against
  23.5 (22.7 to 23.9) for the composed baseline in the same session: 76% of it, 10.3% of the one-die floor. The kernel is still slower
  than composed ttnn.
* PRs: 207 (dense hybrid config), 208 (probe ops), 209 (probe chip variable), and the run logs. Issues: 203 to 206.

## Result

The model runs on one chip with agreement at the bf16 floor in all three modes (host SSD, composed ttnn SSD, coder kernel for the state
update with the readout composed). The local coder wrote a state-update kernel that passes the simulator 4 of 5 and runs on the chip
(first form) and a compact form that passes 2 of 10 in the simulator and 1 of 2 of those on the chip. Neither is faster than writing the
update from stock ttnn ops (16.7 and 18.0 tok/s against 22.6 and 23.5). The coder wrote a readout (row reduction) kernel that passes the
simulator 4 of 10; none of those can run on the chip, because the pinned simulator and the chip's tt-lang disagree on `reduce_sum`.
The run stopped there: a person has to decide how to build a hardware tt-lang from the pinned checkout (issue 205). Whether a kernel makes
this model faster was not shown. Decode is dispatch bound at about 10 to 13% of the floor in every mode; tracing the step is the next
measurement, and it was not done.

Not done: the harness's stages 3, 5 and 6 on this model (blocked at preflight, `nearest-model-missing`); a traced decode; prefill as a
chunked scan (prefill is a loop of decode steps); the 4-chip or 2-chip shrink steps; a package. Nothing was published.

Machine state at the end: the coder container `tt-model-qwen3-coder-next-p300` is still running on board 0 under lease 30e607 (adopted;
other agents are clients of it). No other lease is held.
