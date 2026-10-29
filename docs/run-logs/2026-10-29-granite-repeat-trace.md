# Granite-4.0-H-1B: remove the expanded input copies and trace the decode step (2026-10-29)

Operator request after `2026-10-29-granite-readout-on-pair.md` (PR 224): "let's figure out the ttnn.repeat copies". That run found the
coder's readout kernel slower than composed ttnn and named two causes it did not test: the untraced step is dispatch bound, and the
port expands the kernel inputs with `ttnn.repeat` before each call. This run tests both.

Environment as before: the pair venv (tt-lang-light 1.1.10.dev20261026, tt-lang `fec38c6de`, tt-metal `0e9d200`, host tt-metal
v0.80.0), chips 2 and 3 only, one process at a time under `timeout`. Code, scripts and result files are in
`evidence/granite-repeat-trace/`. Placeholders: `<run-dir>` is the Granite run directory, `<pair-dir>` the pair venv directory.

## Verdict

Once the expansion copies are replaced and the step is traced, the kernel modes are no longer slower than composed, and two of them are
slightly faster. The gain is 3 to 5%.

- Traced, clean session (host load about 8 to 12): composed 37.5 tok/s; coder state-update kernel + coder readout kernel 39.3 (4.8%
  above composed); compact state-update kernel + readout kernel 38.8 (3.5% above); state kernel with composed readout 37.3 (the same as
  composed); compact state kernel with composed readout 36.4 (3% below). The two readout-kernel modes are above composed in each of 3
  rounds (their lowest round mean, 38.7, is above composed's highest, 37.9).
- The expansion method matters. With the earlier `ttnn.repeat` the same modes run at 33.1 (state kernel) and 35.3 (state and readout
  kernels), which is 12% and 6% below composed. Replacing `ttnn.repeat` with a multiply by a resident tensor of ones moves them to 37.3
  and 39.3.
- Untraced, every kernel mode is slower than composed (7.7 to 13.9 tok/s against 21.6). Tracing moves composed from 21.6 to 37.5 and the
  kernel modes by a factor of 2.7 to 5 (the readout-kernel modes from 8 to 39).
- Accuracy against the reference is the same for traced and untraced runs, to the printed digits, and each mode's agreement is
  within the spread of the composed mode and the bf16 CPU floor (tables below).
- The remaining gap to the floor is large. 37.5 tok/s is 26.7 ms per token against 5.7 ms for the floor. The best kernel modes and composed are
  within 1.5 ms of each other, so a further gain from kernels needs the per-layer op count or the per-op device time to come down. This
  run did not profile the device.

## Step 1: where the repeats came from

The earlier run's log said five `ttnn.repeat` copies. `kernel_mod.py` of that run has four repeats per layer and call (decay, dt*x,
B, C), plus a `ttnn.clone` of the new state and a slice of the readout output. The earlier composed mode had one more (C for the
readout matmul).

- The coder kernels (`ssd_state_update`, `ssd_readout`) work tile by tile on tensors of equal shape, so the port expands the per-head
  decay `[48,1,1]`, the per-element `dt*x` `[48,64,1]` and the grouped `B` and `C` `[1,1,128]` to `[3072,128]` before the call.
- All four values depend on the token (through `dt`, `x` and the conv output), so they cannot be built once at load time. Only the
  constants (`A`, `D`, the conv weights) are fixed, and they are not what gets expanded. What the run builds at load time is a resident
  tensor of ones in the expanded shape, so an expansion is one broadcast multiply and no `ttnn.repeat`.
- The compact kernel (`ssd_state_update_bcast`, attempt 8 of the bcast round, that compiled on the chip) reads decay and dt*x as
  `[3072,32]` tiles with the value in column 0 and `B` as a `[32,128]` tile with the value in row 0. In this run decay is built as
  `[48,64,1]` by one multiply with a ones tensor (`[3072,1]` after a reshape), dt*x and `B` are reshapes of tensors the model already
  has. A tile-layout tensor of logical shape `[3072,1]` has 32 columns in memory, so the kernel reads it as it is. No padding op was
  needed (the `--pad` fallback in the code was not used).
- The readout kernel reads `C` as a full `[3072,128]` tensor, one tile row per head row, so a readout kernel needs one expansion of
  `C` per layer in every variant. The compact state kernel removes the other three.

## Step 2: what was changed (all of it is glue, none of it is kernel code)

Kernel files, used unchanged (hashes of the files as run):

| Kernel | File | sha256 |
| --- | --- | --- |
| `ssd_state_update` | `evidence/granite-readout-pair/kernels/state.py` | `2d14bedd619f71cf2018bca677344274b87fecc54ba427589ed9017723b0be9e` |
| `ssd_readout` (run 1, attempt 6) | `evidence/granite-readout-pair/kernels/readout.py` | `7e63a787fe3a3a9a34bee9f294624ebde44c236da818f1a3af67fa401750e274` |
| `ssd_state_update_bcast` (bcast round, attempt 8) | `evidence/granite-repeat-trace/port2/bcast_kernel_attempt8.py` | `20c48b96a2d1f2fed3f5242aae50480dd034c79a9930e009a249d65b33a70dcc` |

No kernel was written or edited and Telegraph was not called. The glue is `granite_tt2.py` (the model step), `run2.py` (agreement and
speed in one process), and the session scripts. It contains:

- the expansion variants (`expand = repeat` as before, `mul` by a resident tensor of ones);
- the compact-input wiring for the bcast kernel;
- the readout in composed mode as one batched matmul with `C` broadcast over heads (the earlier composed readout used a repeat of `C`
  and a multiply and sum in the kernel modes);
- the changes that make the step traceable: see step 3.

## Step 3: tracing

A traced step replays recorded work with fixed addresses and shapes, so the earlier port could not be traced (its K/V cache grew by
concat, its conv and SSM state were replaced by new tensors, the embedding was uploaded inside the step). `granite_tt2.py` changes that:

- Conv state, SSM state and the K/V caches are persistent device buffers, updated in place with `ttnn.copy`.
- The K/V cache has a fixed length of 128 positions. The new key and value go in with `cache * keep + onehot (x) new` and the scores
  get an additive mask for unwritten positions. Prompt plus generated tokens must stay below 128 (the longest run here is 76).
- Per step the host writes four small tensors into persistent device buffers (embedding, one-hot position, its complement, mask) and
  then replays the trace. A step in the untraced mode runs the same program eagerly, so traced and untraced numbers differ only in
  tracing.
- Capture follows one eager run of the program (it compiles every op). The trace region is 300 MB.
- The tt-lang kernels run inside the captured trace. That was not known beforehand. All three kernels were captured and replayed
  with correct outputs.

The static program gives the same results as the earlier dynamic port: composed mode relative L2 0.0074, 0.0182, 0.0073, 0.0138 in
both. Untraced static composed ran at 19.6 and 18.9 tok/s in the matrix session against 19.2 for the earlier dynamic composed port in
the same session (`granite_tt.py`, `bench_decode.py`), so the static form costs nothing.

## Measurements

Speed protocol as before: 12-token prompt, 64 greedy tokens, one warm-up and five timed repeats per run, batch 1, floor 175.2 tok/s.
Agreement protocol as before: teacher forced over the four reference prompts (relative L2 against the fp32 logits, top-1, free-run
prefix against the fp32 tokens). Configuration names: `composed`; `sf` = coder state-update kernel with composed readout; `sfro` =
state-update kernel and readout kernel; `bcast` = compact state-update kernel with composed readout; `bcastro` = compact state kernel
and readout kernel. `repeat` and `mul` are the two expansion methods.

### Speed, session 3 (the comparison used for the verdict)

One lease on board 1, 10:38 to 10:51 Pacific. Host load average (1 minute) 7.5 to 12 for the traced rounds and 4 to 9 for the untraced
runs; the other load is other agents' jobs on the host. The configurations ran interleaved: three traced rounds of all seven, then
one untraced round. Each cell is the mean tok/s of one run (five repeats); the median of the three round means and the lowest and highest
repeat are given.

| Configuration | Traced, round means | Traced median (low to high) | Share of floor | Untraced (low to high) | Share of floor |
| --- | --- | --- | --- | --- | --- |
| composed | 37.9, 37.5, 37.4 | 37.5 (36.8 to 38.2) | 21.4% | 21.6 (21.3 to 21.8) | 12.3% |
| sf, repeat | 33.3, 33.1, 32.9 | 33.1 (32.5 to 33.6) | 18.9% | 13.7 (13.6 to 13.8) | 7.8% |
| sf, mul | 37.1, 37.5, 37.3 | 37.3 (36.7 to 37.7) | 21.3% | 13.9 (13.8 to 13.9) | 7.9% |
| sfro, repeat | 35.3, 35.3, 35.3 | 35.3 (34.7 to 35.9) | 20.2% | 8.2 (8.1 to 8.3) | 4.7% |
| sfro, mul | 39.0, 39.4, 39.3 | 39.3 (38.2 to 40.0) | 22.4% | 8.3 (8.2 to 8.4) | 4.8% |
| bcast | 36.0, 36.4, 36.6 | 36.4 (34.7 to 37.0) | 20.8% | 12.0 (12.0 to 12.1) | 6.9% |
| bcastro | 38.8, 38.7, 38.9 | 38.8 (38.2 to 39.4) | 22.2% | 7.7 (7.5 to 7.8) | 4.4% |

### Speed, sessions 1 and 2 (repeats of the comparison, with conditions that differ)

Each run was made once, traced and untraced, in the order of the table, 16 configurations in one lease.

| Configuration | Session 1 traced / untraced | Session 2 traced / untraced |
| --- | --- | --- |
| composed (first run) | 37.1 / 20.5 | 37.0 / 19.6 |
| composed (last run of the session) | not valid, see below | 34.2 / 18.9 |
| sf, repeat | 31.9 / 11.8 | 17.8 / 9.1 |
| sf, mul | 34.7 / 11.8 | 17.9 / 3.9 |
| sfro, repeat | 34.6 / 7.0 | 34.9 / 2.3 |
| sfro, mul | 39.7 / 7.0 | 37.7 / 5.6 |
| bcast | 36.1 / 10.3 | 32.4 / 10.3 |
| bcastro | not valid, see below | 34.7 / 5.7 |

Session 1 (load about 10) agrees with session 3 on the traced ordering. Session 2 had host load of 20 to 60 from about 09:43 to 09:56
(other agents), which depressed the `sf` runs (17.8 and 17.9 traced) and the untraced runs; those cells are noise from the host and are
not used. Traced runs are less sensitive to host load than untraced ones, because only the four input writes and the trace launch run
on the host. They are not insensitive: the same configuration varied by up to 10% between sessions.

### Agreement with the reference (4 prompts)

Teacher forced over the fp32 reference tokens. The bf16 CPU row is the floor from the 2026-10-28 log. Agreement was measured in
sessions 1 and 2, for every configuration, traced and untraced; the numbers are identical for traced and untraced, and for the two
expansion methods, so each row stands for all of them.

| Mode | Relative L2 mean, prompts 0 to 3 | Top-1, prompts 0 to 3 | Free-run prefix, prompts 0 to 3 (length 3, 24, 24, 24) |
| --- | --- | --- | --- |
| bf16 CPU floor | 0.0047, 0.0121, 0.0043, 0.0111 | 1.0, 1.0, 0.875, 1.0 | 3, 24, 8, 24 |
| composed | 0.0074, 0.0182, 0.0073, 0.0138 | 1.0, 1.0, 0.917, 0.958 | 3, 24, 14, 11 |
| sf | 0.0045, 0.0178, 0.0084, 0.0172 | 1.0, 1.0, 0.917, 0.958 | 3, 24, 8, 15 |
| sfro | 0.0075, 0.0160, 0.0067, 0.0172 | 1.0, 1.0, 0.875, 0.958 | 3, 24, 8, 11 |
| bcast | 0.0068, 0.0160, 0.0070, 0.0189 | 1.0, 1.0, 0.958, 0.917 | 3, 24, 14, 11 |
| bcastro | 0.0047, 0.0145, 0.0085, 0.0159 | 1.0, 1.0, 0.917, 1.0 | 3, 24, 8, 24 |

The modes differ from each other on prompts 2 and 3 in top-1 (0.875 to 0.958 and 0.917 to 1.0) and in free-run prefix (8 to 14 and 11
to 24), as they did in the earlier run. Relative L2 per prompt is within 0.6 to 1.4 times the composed value in every cell, and no mode
is better or worse than the others on every prompt. With four prompts and one run each, the table cannot say that any kernel mode
changes agreement; it shows the modes stay at the same size of error. The `sf` row differs from the earlier run's `sf` row (0.0061,
0.0202, 0.0088, 0.0153) because the readout is now a batched matmul and not a multiply and sum; the earlier `sfro` row (0.0075, 0.0160,
0.0067, 0.0172) is reproduced to the printed digits.

## Incidents and exclusions

- The first lease of the day for this run (a development lease taken with `gozer acquire`, detached) was reaped by gozer after 932
  seconds, which is its grace window. I did not notice, and the last four runs of the first matrix (bcastro untraced and traced, the
  second composed pair) ran from 09:33 to 09:36 while another agent held the same two chips. Those four results are deleted and not used.
  A benchmark of the old dynamic port started at 09:37 without any lease and ended in a traceback; I did not check whether it had
  opened the device before it failed. No other chips were touched. Later sessions used a lease owned by the script's pid
  (session 2) or a detached lease and a script that finished inside the 15-minute window (session 3: 10:38:17 to 10:51).
- Session 1's other runs finished before the reap and are kept.
- I did not touch the coder, its container, lease 93025f or the listed ports.

## Limits

- One board, 4 prompts, three rounds of one traced configuration set. The traced differences between the best kernel modes and
  composed are 3 to 5%; the same configuration varied by up to 10% between sessions under different host load. The ordering in session 3
  repeated in session 1 for the cells that were valid there.
- The device was not profiled. The step takes 26 ms and the floor is 5.7 ms; where the rest goes (op count, per-op latency) is not measured.
- The static program limits the context to 128 positions and writes the K/V cache with a one-hot multiply-add, which costs device time on the
  four attention layers in every mode alike.
- The state is copied (`ttnn.copy`) from the kernel's output buffer back into the persistent buffer. Writing the kernel's output
  directly into the state buffer would remove that copy and was not tried.

## What this changes

1. The blocker named in the earlier run was real for the repeat: `ttnn.repeat` cost about 10% of the traced step in the state-kernel
   modes. A broadcast multiply by a resident ones tensor removes it.
2. With both removed, the coder's kernels give a small gain in the modes that use the readout kernel. The state-update kernel alone
   does not beat composed ttnn in this model, traced or not.
3. The earlier port's tok/s numbers (22.6 composed in the shared venv, 26.5 in the pair venv) were dispatch-bound; the traced
   composed mode is 37.5 in the same venv.
4. Next measurements: profile the traced step per op; write the kernel output into the state buffer; fuse the in-projection slices
   and the conv step into one kernel so the op count per layer falls.
