# Granite-4.0-H-1B: the coder's readout kernel inside the port, on the tt-lang pair (2026-10-29)

Follow-up to `2026-10-28-granite-mamba2-real-port.md` (PR 215) and `2026-10-29-ttlang-pair-chip-check.md` (PR 223). The earlier run could
not put the coder's `ssd_readout` kernel in the model, because none of its simulator-passing kernels compiled on tt-lang 1.1.6. PR 223
showed that the pair in `<pair-venv>` (tt-lang-light 1.1.10.dev20261026, tt-lang `fec38c6de`, tt-metal `0e9d200`, host tt-metal v0.80.0)
compiles and runs all 11 of them standalone. This run puts one of them in the Granite port on the chip and measures it. Raw results, the
scripts and the kernel files are in `evidence/granite-readout-pair/`.

## Verdict

The kernel readout is slower than the composed readout, and slower than the composed model as a whole.

- It compiles and runs on the chip inside the model, and the model's relative L2 against the CPU reference stays within about 1.2 to
  2.0 times the bf16 CPU floor, as for the composed mode. Top-1 and free-run prefix differ between the modes on prompts 2 and 3
  (see the agreement section).
- Decode speed, same session, batch 1, no trace: composed 26.9 and 26.2 tok/s (two runs, before and after), coder state-update kernel with
  composed readout 13.9 tok/s, coder state-update kernel with coder readout kernel 8.45 tok/s. The kernel readout makes the state-update-kernel mode 39% slower, and
  the whole kernel path runs at 32% of the composed speed.
- Whether a fused kernel pays off in this model is still not shown. The step is dispatch bound and untraced; the kernel modes add
  program launches and data copies per layer, and that cost dominates.

## Setup

- Kernel chosen: `ssd_readout` from the coder's run 1, attempt 6 (`readout_r1_a6`), file hash
  `7e63a787fe3a3a9a34bee9f294624ebde44c236da818f1a3af67fa401750e274`. All 11 simulator-passing readout kernels have PCC 0.99999 and
  relative L2 0.0053 to 0.0054 on the pair (PR 223), so the evidence does not separate them. I took attempt 6 because it is the kernel the
  port's `kernel_mod.py` already loads as `readout.py`, so no glue changed. Its loop bound `-(-row_tiles // nodes)` sits in the operation
  body, where PR 223 found it compiles. Shorter files exist (`a9` has 63 lines, `a6` has 66); the difference is not a reason to change.
- State-update kernel: the coder's `ssd_state_update` (`state.py`), hash
  `2d14bedd619f71cf2018bca677344274b87fecc54ba427589ed9017723b0be9e`. It compiled on 1.1.6 and compiles on the pair, so it was used as is.
- Both kernel files are copied unchanged from the probe attempts (copies in `evidence/granite-readout-pair/kernels/`). I wrote no new
  kernel and did not call Telegraph.
- Glue: none was written. `kernel_mod.py` and `kernel_mod_state_only.py` are the files of the 2026-10-28 run, unchanged. They expand the
  decay, dt*x, B and C to the state shape with `ttnn.repeat` before the kernel, and read the result from column 0 of the readout output.
  The port, the benchmark and the comparison script are the earlier ones. Only the interpreter changed (and `-I` was dropped, because
  the host `ttnn` comes in through `PYTHONPATH`).
- Environment: `<pair-venv>` with the exports from `tt-lang-setup-external-tt-metal --tt-metal-dir <metal>`. `ttl.build_info()` printed on the
  chip: version `1.1.10.dev20261026+light`, tt-lang `fec38c6de4b6f24ce4cb359f04aecc8d4577d6f0`, tt-metal `0e9d200db976120c129ab0deb13aa3f6d972b723`.
  `safetensors` is not in that venv; the port needs it to read the weights. It was made importable by putting a directory with a link to
  the shared venv's `safetensors` 0.8.0 at the end of `PYTHONPATH`, so the venv itself was not changed. torch there is 2.15.0 (the
  earlier run used the shared venv's torch).
- The earlier run used the shared venv (tt-lang 1.1.6, tt-metal c49bb76) and the host ttnn of that venv. Its numbers are not comparable
  with this session's, and none are mixed into the tables below.

## Session conditions

One lease, board 1 (chips 2 and 3; `--exact 2`), 2026-10-29 08:03 to 08:07 Pacific, released with `gozer release` right after. One process
at a time, each under `timeout`. The coder's board (chips 0 and 1) was held by the coder under another lease; I did not touch it, its
container or its ports, and did not measure whether it was serving requests during the session. Host load average at the start 1.1
(1 minute); it rose to 1.9 to 3.1 by the end of the session, which includes this session's own processes. All three modes ran
back to back in the same session; the composed mode ran before and after the kernel modes. A pilot agreement run of the full kernel mode ran in an
earlier command under the same lease (it compiled the kernels, so the timed runs used a warm kernel cache; the pilot's
numbers are not used).

## Decode speed

Protocol as before: 64 greedy tokens after a 12-token prompt, one warm-up and five timed repeats, batch 1, no trace, one chip. Floor
175.2 tok/s (`orchard/decode_floor.py`, one die, bf16).

| Mode | tok/s mean (min to max) | Share of floor |
| --- | --- | --- |
| composed, first run | 26.9 (26.0 to 27.4) | 15.4% |
| coder state-update kernel + composed readout | 13.9 (13.8 to 14.0) | 7.9% |
| coder state-update kernel + coder readout kernel | 8.45 (8.27 to 8.57) | 4.8% |
| composed, second run (after the kernel modes) | 26.2 (25.6 to 26.5) | 14.9% |

The two composed runs differ by 3%, which is the session drift. The kernel modes sit 48% and 69% below composed, far outside that
drift. Per repeat, the runs inside a mode vary by 2% to 5%.

## Agreement with the CPU reference

Teacher forced over the reference's tokens, 4 prompts, 24 new tokens each (the same protocol as `run_port.py` in the earlier run).
The bf16 CPU run is the floor (from the 2026-10-28 log): relative L2 mean 0.0047, 0.0121, 0.0043, 0.0111; top-1 1.0, 1.0, 0.875, 1.0;
free-run prefix 3, 24, 8, 24.

| Mode | Relative L2 mean, prompts 0 to 3 | Times the floor | Top-1, prompts 0 to 3 | Top-5 overlap | Free-run prefix |
| --- | --- | --- | --- | --- | --- |
| bf16 CPU floor | 0.0047, 0.0121, 0.0043, 0.0111 | 1 | 1.0, 1.0, 0.875, 1.0 | not recorded | 3, 24, 8, 24 |
| composed | 0.0074, 0.0182, 0.0073, 0.0138 | 1.6, 1.5, 1.7, 1.2 | 1.0, 1.0, 0.917, 0.958 | 1.0, 0.975, 0.958, 0.95 | 3, 24, 14, 11 |
| state kernel + composed readout | 0.0061, 0.0202, 0.0088, 0.0153 | 1.3, 1.7, 2.0, 1.4 | 1.0, 1.0, 0.875, 0.958 | 1.0, 0.983, 0.958, 0.975 | 3, 24, 8, 15 |
| state kernel + readout kernel | 0.0075, 0.0160, 0.0067, 0.0172 | 1.6, 1.3, 1.6, 1.5 | 1.0, 1.0, 0.875, 0.958 | 1.0, 0.992, 0.967, 0.95 | 3, 24, 8, 11 |

The composed row matches the earlier run's composed row to the digit (0.0074, 0.0182, 0.0073, 0.0138). The three modes do not agree
with the reference in the same way, and the differences are recorded here:

- Top-1, prompt 2: composed 0.917, both kernel modes 0.875 (the bf16 CPU floor is 0.875). Prompt 3: composed and both kernel modes 0.958
  (the floor is 1.0).
- Relative L2, per prompt: the kernel readout mode is lower than composed on prompts 1 and 2 (0.0160 against 0.0182, 0.0067 against
  0.0073) and higher on prompt 3 (0.0172 against 0.0138); it equals composed on prompt 0 (0.0075 against 0.0074). The state kernel with
  composed readout is higher than composed on prompts 1, 2 and 3 (0.0202, 0.0088, 0.0153) and lower on prompt 0 (0.0061).
- Free-run prefix, prompt 2: composed 14, both kernel modes 8 (the floor is 8). Prompt 3: composed 11, state kernel with composed
  readout 15, state kernel with readout kernel 11 (the floor is 24). The modes diverge from the fp32 tokens at different positions.
- All of these are within about 2.0 times the bf16 CPU floor in relative L2, and none of the three modes is better or worse than the
  others on every prompt. With 4 prompts and one run each, this run cannot say that the kernels change agreement or leave it unchanged.
  It shows that the model with the kernel readout runs and stays at the same size of error as the composed model.

## Limits

- One session, one board, one seed of prompts (4). The decode numbers are untraced; the ratio between modes may change with tracing,
  because the kernel modes add launches that a trace could hide. The run does not test that.
- The state-update kernel mode ran at 13.9 tok/s here and at 16.7 in the earlier run, and composed ran at 26.5 here and 22.6 there.
  The environment changed (tt-metal c49bb76 with tt-lang 1.1.6 against tt-metal 0e9d200 on the host v0.80.0 runtime, a different torch),
  so the absolute numbers moved in both directions. Only the comparisons inside this session are used for the verdict.
- Coder busy or idle was not measured; its board has its own host-side load, and the host load average is given above.
- The kernel choice among the 11 was by the PR 223 evidence, which does not separate them. Another kernel of the 11 might differ in
  speed. Only attempt 6 was timed in the model.

## What this changes

1. The blocker of the earlier run is gone: a simulator-passing coder readout kernel runs inside the model on the chip with the pair.
2. The kernel path is still not worth using for this model. Issue 205 can record that the compile mismatch is resolved for this pair and
   that the next question is dispatch cost: a fused kernel needs fewer, larger launches than the composed ops, and this port adds five
   `ttnn.repeat` copies before each kernel call. A compact-input readout (as the compact state update did) or a traced decode would test that.
