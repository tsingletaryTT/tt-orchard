# The rmsnorm kernel written by a Claude agent, with the coder's materials (2026-10-27)

## What was asked

The local coder (Qwen3-Coder-Next) has passed 0 of 9 `rmsnorm` attempts in the kernel probe (rounds rms1 to rms3). This run
asks whether a Claude agent, given the same files, can write a `kernel.py` that passes the shipped check in the
simulator. No chip, no lease and no coder server were used. The phase was `sim`.

## What the attempt started with

The attempt directory is a copy of the files a probe attempt starts with: `kernel_check.py`, `kernel_harness.py`,
`references.py`, `kernel_spec.json` (the shipped `rmsnorm` spec), `check_config.json` (tt-lang 1.1.10+gfec38c6, the
simulator venv, 300 s timeout) and `reference/` (`eltwise_add.py`, `eltwise_broadcast_reduce.py`,
`TTLangSpecification.md`, `guide.md`, `examples.md`). The three scripts and the spec were compared byte for byte with the
shipped templates and are identical. The skill text is `orchard/skills/kernel-probe.md`.

Read: the skill text, the two `.py` references, the specification rows for `reduce_sum`, `broadcast` and `fill`, and the
check's source. The broadcast-and-reduce example computes a sqrt of sums of squares and is not an RMSNorm; it was read.
Not used: any existing RMSNorm or LayerNorm kernel, tt-metal's norm kernels, earlier coder attempts, the tt-lang
examples directory beyond the two reference files. The tt-lang skill's own files were not opened beyond what is in
`reference/`.

The check is the same command the probe's agent loop runs: `python3 kernel_check.py` in the attempt directory.

## Result

Passed on attempt 2 of the allowed 12, all six input sets (three seeds, each plain and heavy-tailed). The check was run
once more on the final file and said `PASS` again.

| Attempt | Verdict | Numbers | Failure text and change |
|---|---|---|---|
| 1 | `FAIL: tt_lang_error` | no output sets | `TypeError: unsupported operand for block mul: float. Both operands must be blocks; materialize a scalar with ttl.block.fill() first.` I had written `total * inv_n + 1e-6` with Python numbers. |
| 2 | `PASS` | every set: PCC 1.0 (0.99999994 on two heavy-tail sets), relative L2 0.0, slope 1.0, max abs error 0.0 | Replaced both numbers with one-tile blocks made by `ttl.block.fill(value, shape=(1, 1))`. |

Attempt 1 was written from my reading of the specification and the example, and its failure was the first one the
simulator reported. Attempt 1's `check.json` was overwritten by attempt 2 and is not kept; the failure text above is
copied from the check's output. The final kernel and verdict files are in `docs/run-logs/evidence/rmsnorm-claude/`
(`kernel.py`, `attempt-2.verdict.txt`, `final-check.json`).

## How the kernel works

- **A row used twice.** One block holds a whole row of tiles (shape `(1, col_tiles)` = `(1, 8)` for the spec's 64 x 256
  input). The reader copies that row once into the block and pushes it. The compute thread waits for the block once and
  uses it twice inside the same `with`: once to form the sum of squares, once to scale. The block is popped after both
  uses. Nothing is sent or read a second time.
- **Units for slicing.** Tiles. The slice `x[r : r + 1, 0:col_tiles]` is one tile row and all 8 tile columns, with
  `row_tiles = x.shape[0] // 32` and `col_tiles = x.shape[1] // 32`. The mean divides by `x.shape[1]`, which counts
  elements (256), not tiles.
- **The weight.** The spec gives `w` at full size (64 x 256), so it is read exactly like `x`: its own buffer of shape
  `(1, 8)`, the same slice, the same loop. It is multiplied in as a block of the same shape.
- **Reduction.** `reduce_sum(x * x, dims=[-1], shape=(1, 1))` sums the 8 tiles of the row and then the columns inside the
  tile; the specification says the per-row result ends up in column 0 of the tile. `rsqrt(total * (1/256) + 1e-6)` is
  computed on that block, and `ttl.block.broadcast(scale, dims=[-1], shape=(1, 8))` spreads it across the row (the same
  form the reference example uses for its `(N, 1)` column).
- **Grid.** `grid=(1, 1)`, one node, one row at a time. This is the simplest form and is not tuned.

## Numeric error against the reference

Zero in every set that matters here: max abs error 0.0, relative L2 0.0 and slope 1.0 on all six sets, and PCC 1.0
(0.99999994 on two sets, which is float rounding of the correlation). That is exact agreement with the torch float
reference at the printed precision. This is the simulator, which the check ran with a tt-lang whose results here match
float arithmetic; hardware bf16 will differ (the hardware bars are looser for that reason: PCC 0.99, relative L2 0.05).
Nothing was run on a chip.

## What the skill text lacked

Each item is a general fact I needed or checked and that `kernel-probe.md` or the references did not state. None is an
RMSNorm solution.

1. A Python number cannot be combined with a block. `block * 0.5` and `block + 1e-6` fail with the `TypeError` above.
   A constant has to become a block with `ttl.block.fill(value, shape=...)` first. The skill mentions `fill` only as a
   way to get zeros. This was the one failure in the run.
2. A reduction over the last dimension leaves its result in column 0 of each result tile, not spread over the tile. The
   specification says so in its `reduce_sum` row, but a reader of the skill does not learn that a following
   `broadcast(dims=[-1])` is what turns it back into a value for every column.
3. The element count for a mean is `x.shape[1]` (elements), while buffer and slice sizes are tiles. The skill states the
   two units but gives no example of using both in one expression.
4. A block that is held between `wait()` and the end of its `with` can be used any number of times in compute
   expressions. The skill says a row used twice "must stay in one block across both uses", and does not say that using
   it in two statements inside one `with` is the way to do that.
5. A buffer shape can cover a whole row of tiles (`(1, col_tiles)`) and a reduction can then take the whole row in one
   call. The references use per-tile or small blocks; neither shows a full-row block with a reduction.
6. Compile-time Python arithmetic (`1.0 / x.shape[1]`, `x.shape[0] // 32`) is allowed outside the thread functions and
   can be captured by them. Not stated in the skill; the reference `eltwise_add.py` shows the integer form only.
7. Wiring detail that was not an obstacle but was inferred: the reader pushes by leaving the `with`, and the compute
   thread waits with the same `with` form. The skill says "copy the example's three thread functions", which covers it.

## What this does and does not show

- It is one agent, one run, with the coder's files. It shows that `rmsnorm` is writable in the simulator from those
  materials, in two attempts.
- It does not show why the coder failed. Claude and Qwen3-Coder-Next differ in many ways, and the facts in the list
  above are what I needed; the coder's recorded failures (deadlocks, a mis-scaled output, a bad `pop()`) are different
  errors from the one I made, and this run did not test whether adding these facts to the skill would help the coder.
- I know the RMSNorm formula and the simulator's behaviour from before this task; the skill text does not say that
  formula, and the spec states it in `kernel_spec.json`.
- No hardware phase. The kernel was not compiled for a chip, and the hardware compiler may reject constructs the simulator
  accepts (the reference example is marked as expected to fail on it).
- The check's bars are provisional (marked so in the spec).
