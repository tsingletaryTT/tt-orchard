# TT-Lang cheat sheet: chip claims judged by a compile (2026-11-24, issue 801)

Issue 801 asked for three things: the sheet's `hardware` source should be the package the hardware phase uses, `ttl.block.fill` and
`ttl.block.broadcast` should replace the refused `ttl.math` forms, and `op_cheats.verify` should judge a claim about a call used inside a
thread by a compile result. A signature found in the package's files is not enough. This run log covers the third. The first two were
already on main from PRs 803 and 812 (2026-11-22); this log checks them again.

## What main already did (checked today)

- `roots.hardware` names the managed pair (tt-lang 1.1.10.dev20261026+light) and `roots.shared` names tt-lang 1.1.6 as the fallback.
- The sheet has entries for `ttl.block.fill` and `ttl.block.broadcast`, both tagged `compiles on hardware`. It has no entry for
  `ttl.math.fill` or `ttl.math.broadcast`, and a mistake that quotes both compilers' refusal.
- `tests/test_sheet_chip_claims.py` pins the text and runs the real compilers (16 tests, all pass on this machine).

## The table of the issue, reproduced before any change

Kernel variants of the `row-reduce-spread` skeleton compiled with each package's own compiler, no device
(`orchard/ttlang_hwcompile.py`, the child process shuts `/dev/tenstorrent` out of itself and proves it):

| Form | tt-lang 1.1.6 | pair 1.1.10.dev20261026+light |
| --- | --- | --- |
| `ttl.block.fill`, `ttl.block.broadcast`, `reduce_sum(shape=)` | refused: `reduce_sum() got an unexpected keyword argument 'shape'` | compiled |
| same, `reduce_sum` without `shape` | compiled | compiled |
| `ttl.math.fill` | refused: `ttl.math.fill is not available; use ttl.block.fill` | the same line |
| `ttl.math.broadcast` | refused: `ttl.math.broadcast is not available; use ttl.block.broadcast` | the same line |

The table matches the issue. The pair also compiles `reduce_sum` without `shape`.

## What this change adds

- `orchard/cheat_probes.py` holds 15 probe kernels: 13 that use one call of the sheet inside a thread and 2 that use a refused form. Two base
  kernels carry them (an elementwise kernel and a row reduction). A change to a base that does not apply exactly once is an error.
- `scripts/verify_cheat_compiles.py` compiles every probe with every chip package on the machine and writes
  `orchard/skills/kernel-loop-templates/ttlang_cheat_compiles.json`: the probe text and, per package, the compile record. It refuses to write
  when a probe does not give the verdict its `expect` says. `--check` compares with the file on disk.
- `orchard/cheat_compiles.py` reads the file and judges the sheet by it. `op_cheats.verify` takes the records and the identity of the hardware
  package. Without them it says so as a problem.
- `scripts/verify_cheat_sheet.py` reads the identity from the `ttl/` directory given for the root `hardware` (new `identity_of` in
  `ttlang_hwcompile.py`) and loads the records.

The rules, judged against records made with exactly the hardware package's compiler (version and compiler bytes):

- A `compiles on hardware` claim needs a covering probe that compiled. A refusal and a check that could not be made are not compiles.
- A `hardware signature only` claim needs no record, because the tag says no compile was shown. A refusal of every covering probe contradicts it.
- A form that a probe records as refused may not be an entry, whatever its tag.
- A refusal counts only when the compiler's line names the form. A probe that fails for another reason does not show a refusal.
- The records have to match the probe text by sha256, and the hardware package has to be one of the data's targets, so a package upgrade is noticed.

## Results of the probes (compiled on this machine, 2026-11-24)

| Probe | Covers | pair | 1.1.6 |
| --- | --- | --- | --- |
| elementwise | `ttl.operation`, `ttl.compute`, `ttl.datamovement`, `ttl.make_dataflow_buffer_like`, `DataflowBuffer.wait`, `DataflowBuffer.reserve`, `TensorBlock.store`, `ttl.copy` | compiled | compiled |
| make-dfb | `ttl.make_dfb` | compiled | refused: no attribute `make_dfb` |
| grid-size, node-in-compute | `ttl.grid_size`, `ttl.node` | compiled | compiled |
| reduce-sum-broadcast-fill | `ttl.math.reduce_sum`, `ttl.block.broadcast`, `ttl.block.fill` | compiled | refused: `shape` on `reduce_sum` |
| reduce-max | `ttl.math.reduce_max` | compiled | refused: `shape` on `reduce_max` |
| exp, rsqrt, silu, transpose, matmul, signpost, typecast-torch-dtype | one call each | compiled | compiled |
| math-fill (expected refused) | refuses `ttl.math.fill` | refused, line names it | refused, line names it |
| math-broadcast (expected refused) | refuses `ttl.math.broadcast` | refused, line names it | refused, line names it |

Every `compiles on hardware` entry of the sheet (12) has a compiled record for the pair, and `verify` passes against the real trees of the
pair, tt-lang 1.1.6 and the checkout with the shipped records.

## Entries the same check would now fail

None. The old failure replayed: with `ttl.math.fill` added to the sheet as `hardware signature only`, the static half finds the name in the
pair's files and has nothing to say, and the recorded refusal makes `verify` report it
(`tests/test_cheat_compiles.py::test_the_static_check_alone_passes_math_fill_and_the_compile_records_catch_it`).

## Entries that could be raised (not changed here)

Seven entries are tagged `hardware signature only` and a kernel that uses them compiled with the pair: `ttl.matmul`, `ttl.math.transpose`,
`ttl.math.typecast`, `ttl.math.rsqrt`, `ttl.math.exp`, `ttl.math.silu` and `ttl.signpost`. `scripts/verify_cheat_sheet.py` prints each as a
note. Raising the tags changes the rendered sheet, so it is a separate change.

Four entries have no probe: `ttl.Pipe`, `ttl.PipeNet`, `PipeNet.if_src` and `PipeNet.if_dst`. A probe needs a multi-node grid. They stay
`hardware signature only`.

`ttl.math.typecast` needs a note. The sheet's usage line passes `ttnn.bfloat16`. With the compile check's stand-in for `ttnn`, the pair's
dtype conversion recurses (`RecursionError: maximum recursion depth exceeded`) while 1.1.6 accepts it. With `torch.bfloat16` both compile, and
the probe uses that form. So this check cannot judge the sheet's own usage line for `typecast` on the pair, and a run on a chip would
settle it.

## What the first version of the probes got wrong

The generator's expectation check caught two mistakes of mine before anything was written:

- The row-reduction probe multiplied a 1 by 8 block by a 1 by 1 constant, which the compiler rejects (`shape mismatch`). Both packages
  refused the probe.
- The first `math-broadcast` probe failed with the same shape mismatch, and a refused probe accepted any compile error, so it counted as
  refused for the wrong reason. A refusal now has to name the form (`tests/test_cheat_compiles.py::test_a_refusal_that_does_not_name_the_form_is_not_a_refusal_of_it`).

## Comparability of runs

The rendered sheet is unchanged: the pin in `tests/test_ttlang_recipes.py` (`bc9e27bc801f1a9d99728f0387d2b066acc1e42badcd24c554bffd77c9968c53`,
133 lines) passes without an edit, and the sheet file is byte for byte what PR 812 left. A run with this change is comparable with a run
after PR 812. The three-arm A/B that `docs/run-logs/2026-11-22-recipes-dialect-gate.md` pre-registers runs against the sheet PR 812
produced; this change adds nothing to carry as a fourth arm. A later change that raises the seven tags would change the digest again.

## What a record does not show

A compile record means the chip package's compiler accepted the kernel with no device and wrote kernel source. It does not show that
tt-metal builds that source, that the buffers fit the chip's free L1, or that the numbers are right on a chip. The records are named
`compile` and never `chip`.

## To reproduce

    python3 scripts/verify_cheat_compiles.py            # compile the probes, write the records
    python3 scripts/verify_cheat_compiles.py --check    # compile and compare with the file on disk
    python3 scripts/verify_cheat_sheet.py ttlang --root hardware=<pair ttl dir> --root shared=<1.1.6 ttl dir> --root tree=<checkout python/ttl> --root sim=<checkout python/sim>

Run the first after a chip package changes. `verify` reports a hardware package that the records were not made with.
