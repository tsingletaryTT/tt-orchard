# tt-lang: the pinned simulator and the S3 light compiler wheel, run on the chip (2026-10-29)

Operator instruction: follow the recommendation of `2026-10-29-ttlang-compat-matrix.md`. Install `tt-lang-light 1.1.10.dev20261026` in a
new venv, point it at the host tt-metal v0.80.0 tree, and find out on a chip whether the kernel that passes the simulator and fails to
compile on the installed tt-lang 1.1.6 now compiles and runs correctly. This run also answers the open `ttl.grid_size` question of
issue 205 and gives the smoke-kernel result that issue 95 asks for.

`<venv-light>` is the new venv, `<venv>` the shared hardware venv (tt-lang 1.1.6, used only to run kernels, never modified),
`<sim>` the simulator venv (`tt-lang-sim 1.1.10+gfec38c6`, the pin), `<metal>` the host tt-metal tree (v0.80.0). Raw results, the
kernels and the scripts are in `evidence/ttlang-pair/`.

## Verdict

The pair lines up for every kernel tried, with one exception that is a limit of the compiler and not a disagreement with the simulator.

- All 11 simulator-passing `ssd_readout` kernels, the library `rmsnorm` kernel and the smoke kernel compile with the light wheel and
  run on the chip at PCC 0.99999 or better against the torch reference. The same 12 reduction kernels pass the pinned simulator.
  On tt-lang 1.1.6 the two reduction kernels run (`readout_r1_a10`, `library_rmsnorm`) fail to compile with the error of issue 205.
- `ttl.grid_size(dims=2)` is accepted inside `@ttl.compute` and `@ttl.datamovement` threads by the pair. tt-lang 1.1.6 rejects it in
  every thread with `Unknown function: ttl.grid_size` and accepts it only in the operation body.
- Where the pair and the simulator still differ: inside a thread, `ttl.grid_size` returns a symbolic value, and the expression
  `-(-row_tiles // grid_rows)` used as a loop bound fails in the compiler with `AttributeError: 'UnaryMinusOp' object has no attribute 'type'`.
  The simulator runs that kernel at PCC 1.0. The form `(row_tiles + grid_rows - 1) // grid_rows` compiles, so only the unary minus is
  affected. No kernel written by the coder in this evidence uses this form inside a thread. It is recorded here and not worked around.

So a simulator pass now predicts a compile on the chip for reductions and for `grid_size` placement, on this pair. The pair is not on
public PyPI, which has nothing after 1.1.6 (compat matrix report).

## What was installed

New venv `<venv-light>` (Python 3.12, `uv venv`), nothing else was changed. Wheels from `https://pypi.eng.aws.tenstorrent.com/tt-lang/`,
downloaded and hashed before installing; their `METADATA` was read first (the metapackage has one dependency,
`tt-lang==1.1.10.dev20261026+light`; the light wheel has no `ttnn` requirement).

| Wheel | sha256 |
| --- | --- |
| `tt_lang-1.1.10.dev20261026+light-cp312-cp312-manylinux_2_34_x86_64.whl` | `56d8f3e33b41fb41dcd013b05760f7c0deb6bcec61a4a7e8f58f417135bfdafc` |
| `tt_lang_light-1.1.10.dev20261026-py3-none-any.whl` | `5f28a7180c6d6866a2c57e42a586f25023a132d5470ce7c12153029220b99888` |

The dependencies came from the default package index (torch 2.15.0, numpy and others, as `uv` resolved them). `ttnn` is not installed in
`<venv-light>`. The host `ttnn` (editable install of `<metal>`, v0.80.0) is found through the exports printed by
`tt-lang-setup-external-tt-metal --tt-metal-dir <metal>` (`TT_METAL_HOME`, `PYTHONPATH`, `LD_LIBRARY_PATH` into `<metal>/build`). Before the
lease, only `importlib.util.find_spec` ran: `ttl` found, `ttnn` not found. The first import of `ttnn` and `ttl` happened under the lease.
The fallback wheel from `ttmetal/0e9d200/` was not needed.

`ttl.build_info()`, printed on the chip by each of the two environments:

| Environment | `ttl.build_info()` |
| --- | --- |
| `<venv-light>` (the pair) | version `1.1.10.dev20261026+light`, tt-lang `fec38c6de4b6f24ce4cb359f04aecc8d4577d6f0` (equal to `TTLANG_PIN`), tt-metal tag `v0.79.0-dev20260917`, tt-metal `0e9d200db976120c129ab0deb13aa3f6d972b723` |
| `<venv>` (shared) | version `1.1.6`, tt-lang `dede2973f1c738ef531ef4a57c47369964dc146c`, tt-metal tag `v0.74.0`, tt-metal `c49bb7625e6a495337483353a5228339fece2c11` |

The host `ttnn` is v0.80.0; the pair's compiler was built against `0e9d200`, which is in the history of v0.80.0 (compat matrix report).
Both run on the host runtime.

## Lease and hardware

One lease, board 1 (chips 2 and 3, `--exact 2` expanded to the whole board), `TT_VISIBLE_DEVICES` taken from the grant, one process at
a time, each under `timeout 300`. Released with `gozer release`, which reset those two chips. The coder's chips, its container and the
ports named in the instruction were not touched. Total lease time was under 15 minutes.

## Kernel by kernel

Pinned simulator: `<sim> tt-lang-sim` on the pinned checkout's build. Shapes: smoke 256 x 256; readout and rmsnorm 64 x 128 and
3072 x 128 (both shapes gave the same PCC, shown once). Reference: torch in float32 from the same bfloat16 inputs, seed 0.
"1.1.6" is a compile and run in `<venv>` with the same script.

| Kernel | Pinned simulator | Compiler on 1.1.6 | Compiler on the pair | Chip result on the pair (PCC, relative L2) |
| --- | --- | --- | --- | --- |
| `smoke_eltwise_add` (the checkout's `eltwise_add.py`, the kernel the probe's smoke test runs) | pass, PCC 1.0 | compiles, runs, PCC 0.99999 | compiles | PCC 0.99999, 0.0018 |
| `readout_r1_a6`, `a7`, `a9`, `a10` (`ssd_readout`, coder run 1) | pass, PCC 1.0 | `a10`: `error: reduce_sum() got an unexpected keyword argument 'shape'`; the other three not rerun, same call | compiles | PCC 0.99999, 0.0054 (64 x 128) and 0.0053 (3072 x 128) |
| `readout_hw1_a1`, `a2`, `a3` | pass, PCC 1.0 | not rerun here, same `shape=(1, 1)` call; issue 205 records the failure | compiles | PCC 0.99999, 0.0054 and 0.0053 |
| `readout_hw2_a1`, `a2`, `a3`, `a4` | pass, PCC 1.0 | not rerun here, same call | compiles | PCC 0.99999, 0.0054 and 0.0053 |
| `library_rmsnorm` (library kernel with `reduce_sum`, `rsqrt`, `fill`, `broadcast`) | pass, PCC 1.0 | `error: reduce_sum() got an unexpected keyword argument 'shape'` | compiles | 0.99999, 0.0034 (64 x 128); 1.0, 0.0033 (3072 x 128) |
| `gs_call_compute` (`grid_size` called, result unused, in `@ttl.compute`) | not run | `error: Unknown function: ttl.grid_size` | compiles | PCC 0.99999, 0.0018 |
| `gs_call_datamovement` (same, in `read` and `write`) | not run | `error: Unknown function: ttl.grid_size` | compiles | PCC 0.99999, 0.0018 |
| `gs_use_compute`, `gs_use_datamovement` (`rpn = (row_tiles + gr - 1) // gr` in the thread) | not run | not run | compiles | PCC 0.99999, 0.0018 |
| `gs_compute`, `gs_datamovement`, `gs_threads_only` (`rows_per_node = -(-row_tiles // grid_rows)` after a thread's own `grid_size` call; the last one has no call in the body) | pass, PCC 1.0 | `gs_compute`, `gs_datamovement`: `error: Unknown function: ttl.grid_size` | **fails**: `AttributeError: 'UnaryMinusOp' object has no attribute 'type'` | no result |

Counts on the pair: 17 of 20 kernels compile and run on the chip with PCC at or above 0.99999; the 3 that fail are the unary-minus
variants. Every kernel that passed the pinned simulator and compiled on the pair ran correctly.

The `shape` error text on 1.1.6, in full for `library_rmsnorm`:

```
ttl.diagnostics.TTLangCompileError: error: reduce_sum() got an unexpected keyword argument 'shape'
  --> <scratch>/k/library_rmsnorm.py:44:26
   |
44 |                     sum_sq = ttl.math.reduce_sum(x_blk * x_blk, dims=[-1], shape=(1, 1))
```

The `grid_size` error text on 1.1.6, for the call inside `@ttl.compute`:

```
ttl.diagnostics.TTLangCompileError: error: Unknown function: ttl.grid_size
  --> <scratch>/k/gs_call_compute.py:27:16
   |
27 |         _gc, _gr = ttl.grid_size(dims=2)
```

The failure on the pair, from the compiler's own traceback (`ttl/pykernel/_src/kernel_ast.py`, `visit_For`, line 588):
`if isinstance(upper_bound.type, memref.MemRefType)` reads `.type` of a `UnaryMinusOp`.

## Answers to the open questions

- Issue 205, `grid_size` in a thread: on 1.1.6 the compiler rejects `ttl.grid_size` in every thread (compute and datamovement) and
  accepts it in the operation body. On the pair a thread call compiles and runs; its result is a symbolic value, so arithmetic with
  unary minus on it fails as a loop bound while addition, subtraction of a constant and floor division work.
- Issue 205, `reduce_sum(..., shape=)`: accepted by the pair, rejected by 1.1.6.
- Issue 95, preflight row: the smoke kernel compiled and ran on the pair. `ttl.build_info()` gives the tt-lang commit and the tt-metal
  commit, so a row can compare both with the pin. At present `TT_METAL_REF` is `None`; the value to pin for this pair is
  tt-metal `0e9d200db976120c129ab0deb13aa3f6d972b723` (v0.79.0-dev20260917), which runs on the host v0.80.0 runtime for these kernels.

## Other findings

- The light wheel's own simulator (`<venv-light>/bin/tt-lang-sim`) failed on all 16 kernels it was given with
  `RuntimeError: could not locate atom_rules.py to run a unified @ttl.operation body`. The file exists at `ttl/_src/atom_rules.py`; the
  simulator looks in `ttl/sim/` and `ttl/ttl/_src/`. The simulator column above is the pinned `<sim>` venv, as the instruction said. A
  `tt-lang-sim` from this wheel is not usable as a replacement until upstream fixes the lookup path.
- The pair's chip PCC is 0.99999 where the simulator gives 1.0 exactly, which is the bfloat16 rounding of the chip. The readout relative
  L2 error is 0.0053 to 0.0054, inside the probe's hardware bars (rel L2 5e-2, `kernel_check.py`).

## Limits

- One chip board, one shape pair per kernel, one seed. The probe's other op specs (softmax, center, silu_mul, tile scan) were not run.
- The 1.1.6 column holds a fresh run for the smoke kernel, `readout_r1_a10`, `library_rmsnorm` and four `grid_size` kernels. The ten other
  readout kernels contain the same `shape=(1, 1)` call and were not rerun on 1.1.6.
- The test kernels `gs_*` are edits of `eltwise_add.py` and have not been written by the coder; the real coder-written failure
  (`ssd_state_update_bcast` attempt 3, issue 205) was not rerun, because its source is not in this evidence.
- Only the hardware path with the host `ttnn` v0.80.0 was tried. The bundled wheel (ships its own ttnn) was not.

## What this changes

1. Hardware pin: `ttlang_pin.py` can record `tt-lang-light 1.1.10.dev20261026` and tt-metal `0e9d200db976` for the hardware path. The
   consistency row that compares `ttl.build_info()` with the pin is the next code change; it needs no chip.
2. The coder's reference material stays correct for this pair: the specification at the pin documents `reduce_sum(..., shape=)`, says `grid_size` "can be used anywhere in an operation", and the pair accepts both.
3. The shared hardware venv still holds 1.1.6. Moving the probe's hardware phase to `<venv-light>` is an operator decision; this run
   changed nothing outside the new venv.
