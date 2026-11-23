# Recipes and the chip: what was wrong, what is checked now, and the next A/B (2026-11-22)

Question from the operator: the recipes in the TT-Lang cheat sheet (PR 796) look more effective and risky. Every pass is a simulator pass,
the sheet grows from 141 to 399 lines, and the recipes were verified on the simulator only. Is the risk real, and what removes it?

Everything below ran on the CPU. No chip was used and no device was opened: an Aion benchmark held the chips.

## Findings

1. **The sheet teaches a chip form that both chip compilers refuse.** The cheat sheet and the recipes said the chip package has no
   `ttl.block` namespace and that `ttl.math.fill` and `ttl.math.broadcast` are the chip's names. Compiled with the package's own compiler:

   | form | tt-lang 1.1.6 (shared venv) | tt-lang 1.1.10.dev20261026+light (pair) |
   |---|---|---|
   | `ttl.block.fill(0.5, shape=(1, 1))` | compiles | compiles |
   | `ttl.math.fill(0.5, shape=(1, 1))` | `ttl.math.fill is not available; use ttl.block.fill` | the same message |
   | `ttl.math.broadcast(...)` | (the recipe that uses it is refused earlier) | `ttl.math.broadcast is not available; use ttl.block.broadcast` |

   The cause is in the compiler: `_src/ttl_ast.py` holds `_NAMESPACE_OVERRIDES = {"broadcast": "ttl.block", "fill": "ttl.block"}` in both
   packages (the installed 1.1.6 file has the sha256 its wheel's RECORD lists, so it is the published compiler). PR 796 printed a
   `chip:` line with `ttl.math.fill` under each of the four recipes. All four chip forms are refused by both compilers (8 of 8).

2. **The check that vouched for those lines looked at the wrong layer.** PR 796's static check found `fill` and `broadcast` exported by
   `ttl_math.py`. They are exported there as Python functions. A call inside a `@ttl.compute` or `@ttl.datamovement` thread is never
   resolved through Python: the compiler reads the thread's source and resolves `ttl.x.y` in its own table. A name can exist in the
   package and be refused in a kernel.

3. **"The chip package" is two packages, and the probe uses the newer one.** The kernel probe's hardware phase runs in the managed pair
   venv when it exists (`ttlang_pair.resolve_hardware_python`) and falls back to the shared venv (1.1.6). The pair's compiler is built
   from the same commit as the simulator (`fec38c6de`). It takes `reduce_sum(..., shape=...)`; issue 205 records 11 of 11 reduction
   kernels compiling and running on a chip with it. The figure "0 of 7 compiled on the chip" is a fact about 1.1.6.

4. **One recipe did not compile for the pair either, in the form that passed the simulator.** The row-reduction recipe wrote
   `inv = ttl.math.rsqrt(total + eps)`. The pair's compiler refuses it: `cannot lower tensor store to ttl.compute: store input is not
   dataflow-buffer-backed`. So does `rsqrt(total)`. `rsqrt(total * one + eps)`, with `one` a (1, 1) block, compiles (the library's
   rmsnorm kernel, which ran on a chip, multiplies the sum by a block first). The recipe now uses that form and says why, and it passes
   the simulator check again (6 of 6 sets).

5. **Two of the three simulator passes of the A/B cannot run on a chip.** `scripts/hwcompile_probe_passes.py` compiled every kernel that
   passed the simulator in the A/B of 2026-11-21 (`evidence/recipes-dialect-gate/recipes-ab-compile.json`):

   | arm | attempts | simulator passes | compile with the pair | compile with 1.1.6 |
   |---|---|---|---|---|
   | recipes on (`center`, `softmax`, `rmsnorm`) | 7 | 3 | 1 | 0 |
   | recipes off | 8 | 0 | 0 | 0 |

   Both `center` passes are refused by the pair's compiler (`<ast.Div object at 0x...> not implemented`: the coder wrote `1.0 / 256.0`
   inside a thread, which the simulator runs as Python). The `softmax` pass compiles. With 1.1.6 none compiles (the `shape` keyword).
   On passes that a chip compiler accepts, the A/B is 1 of 7 against 0 of 8 (Fisher exact p = 0.47). It shows no effect of the recipes.

6. **The plain sheet's chip claims explain part of the off arm's failures.** 5 of the 15 attempts ended on a simulator error that comes
   from writing the form the sheet calls the hardware form: `'_TTLMathNamespace' object has no attribute 'fill'` (center off 1),
   `... 'broadcast'` (center off 2), and `reduce_sum()` or `reduce_max()` `missing 1 required positional argument: 'shape'` (rmsnorm off
   1, softmax off 3, softmax on 1). The recipes section began with a paragraph that told the coder which form the check accepts. So the
   A/B compared a sheet that misleads with a sheet that misleads and then corrects itself. The recipes' own content was not isolated.

## What is checked now

**A compile with the chip package's own compiler, with no device** (`orchard/ttlang_hwcompile.py`, child
`orchard/ttlang_compile_child.py`). tt-lang has a compile-only mode (`TTLANG_COMPILE_ONLY=1`): it parses the threads, builds its MLIR,
runs its passes and writes one C++ source per thread, and stops before anything runs. Upstream's tests still open a device to make
tensors. The check gives the compiler stand-in tensors and never imports `ttnn`:

- The child first shuts `/dev/tenstorrent` out of itself with Landlock (file access is refused by the kernel for the process and every
  process it starts), and proves it by asking for a listing of the directory, which must be refused, before tt-lang is imported.
- `ttnn` in `sys.modules` is a stand-in: a tensor (shape, bfloat16, tile layout, DRAM), a device that reports a 13 x 10 Blackhole grid,
  and permissive descriptor classes. Each name tt-lang asks of it is recorded.
- A pass needs the proved guard, a call that returned, and at least one kernel source written by the compiler. A check that could not
  run (`timeout`, `guard_failed`, `bad_result`) has its own category and is never counted as a refusal.

**How far it can be trusted.** `scripts/calibrate_hwcompile.py` replays the 21 kernels of `evidence/ttlang-pair`, whose results on a chip
are recorded for both packages (2026-10-29, under a lease). The check agrees with the chip on 41 of 41 rows
(`evidence/hwcompile-calibration/calibration.json`): 30 kernels the chip compiled and ran all compile here, and 11 the chip refused are
refused here with the same first line, for three different messages (`reduce_sum() got an unexpected keyword argument 'shape'`,
`Unknown function: ttl.grid_size`, `'UnaryMinusOp' object has no attribute 'type'`).

**What it does not show.** Every chip failure in that evidence is a tt-lang compiler error. The check does not build the kernel source
with tt-metal, does not check that the buffers fit the chip's free L1 (each compiler uses its built-in budget), and says nothing about
numbers on a chip. PR 240 recorded a kernel that passed the simulator, compiled for the pair and scored PCC 0.249 on the chip. A compile
pass is the first of three steps.

## What changed in the sheet

- No recipe carries hand-written chip lines (`hardware_replace` is refused by the loader). A translation layer between a simulator
  form and a chip form was considered and not built. For the package the probe uses, the form that passes the simulator is the form its
  compiler takes, so there is nothing to translate. For 1.1.6 a translation would help one recipe: with `shape=` dropped from
  `reduce_sum`, the row-reduction skeleton compiles and the column-reduction skeleton still fails to lower (`failed to legalize operation
  'ttl.reduce'`). That form fails the simulator, which requires `shape`, so a coder judged by the simulator cannot use it. On 1.1.6
  the sheet says the compiler refused the two reduction recipes and gives its message.
- Each recipe's record holds, per chip package (version and sha256 of the compiler's files), the compile of exactly the skeleton the
  sheet prints and of each `compile_negatives` form, which that compiler must refuse. The section is rendered for the run's chip
  package. Tags: `[compiles with tt-lang X, no chip]`, `[refused by the compiler of tt-lang X]`, `[not compiled with tt-lang X]`,
  `[not compiled with a chip package]`, and `[ran on a chip with tt-lang X]` for a record only a chip run can write (none exists).
- Lines such as "The chip compiler refuses `ttl.math.fill`: write `ttl.block.fill`" are printed only when that compiler refused that
  form, with its message.

| recipe | simulator | pair (1.1.10.dev20261026+light) | 1.1.6 |
|---|---|---|---|
| row reduction, then spread | pass, 6 of 6 | compiles | refused (`shape`) |
| column reduction with a running total | pass, 6 of 6 | compiles | refused (`shape`) |
| elementwise with a constant and a row operand | pass, 6 of 6 | compiles | compiles |
| running total down the rows | pass, 6 of 6 | compiles | compiles |

`--cheat-sheet-recipes auto` turns the recipes on only when every recipe compiles with the run's package and every wrong form was
refused. Today that holds for the pair and does not hold for 1.1.6 or for a machine with no chip package. The default stays `off`:
the gate removes one risk, and the A/B below has to show the recipes help before they become the default.

## The next A/B (pre-registration)

Three arms, one variable each, on the same coder and server, temperature 0.7, `--library none --no-agent-hw --phase sim`, 40 turns:

| arm | flag | what the coder gets |
|---|---|---|
| plain | `--cheat-sheet-recipes off` | the sheet |
| control | `--cheat-sheet-recipes control` | the sheet and a section of the same framing and length (247 lines against 247, 14,506 characters against 14,656, rendered for the pair) with four elementwise kernels: no reduction, no broadcast, no constant block, no carried value |
| recipes | `--cheat-sheet-recipes on` | the sheet and the recipes |

- Ops: `center` and `softmax` (held out), with `rmsnorm` reported apart because it is close to the row-reduction recipe.
- Order: the three arms rotate within each block of three attempts, so each arm runs first, second and third equally often.
- Outcome: an attempt counts when its kernel passes the unedited simulator check and compiles with the pair's compiler
  (`scripts/hwcompile_probe_passes.py`). Simulator passes are reported beside it.
- Comparisons: recipes against control (what the recipes' content adds) and control against plain (what the extra text and the
  section's opening paragraphs add).
- Size: at least 10 attempts per arm and op. With 10 per arm, 5 passes against 0 is the smallest difference that Fisher's exact test
  separates at 0.05 (p = 0.033); 4 against 0 gives p = 0.087. Three per arm, as on 2026-11-21, cannot separate any outcome.
- Not answered by it: a chip run, more than one coder model, or ops outside these families.

A fourth arm would isolate finding 6: the plain sheet with its chip claims corrected. That needs the sheet itself changed, which is
filed apart (issue 801), because it changes the bytes of every attempt.

## Open

- No recipe and no A/B kernel has run on a chip since this change. The chip run that closes it is issue 800.
- The plain sheet still says `ttl.math.fill` and `ttl.math.broadcast` are the chip's names (issue 801).
- The stand-in device reports a fixed grid and no L1 figure. A kernel that only fails for a harvested grid or a full L1 passes here.
- A recipe has one skeleton for every chip package. A variant for one package (for 1.1.6: the row reduction without `shape=`), compiled
  by that package's compiler before it is printed, is not built.
- The probe does not run the compile check itself. After a run, `scripts/hwcompile_probe_passes.py DIR` says which simulator passes a
  chip compiler accepts.
