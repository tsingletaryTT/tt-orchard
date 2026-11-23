# The plain TT-Lang cheat sheet and the chip compilers (issue 801, 2026-11-22)

Question from the operator: keep chasing the recipes, and do not make things worse than before the new ideas began. Every kernel-probe attempt
gets the plain sheet, with recipes on or off, and PR 802 found that its chip claims are wrong. This change corrects only those claims.

Everything here ran on the CPU: the simulator, and the chip packages' own compilers through the device-less check of PR 802
(`orchard/ttlang_hwcompile.py`). No chip, no device and no coder was used. The sheet is verified against the trees of both chip packages
with `scripts/verify_cheat_sheet.py` (26 entries and 7 mistakes, all found).

## What was wrong

The sheet called `ttl.math.fill` and `ttl.math.broadcast` the hardware names, called `ttl.block.fill` simulator only, and said the hardware
package has no `shape` argument on `reduce_sum`. Compiled with each package's own compiler, with the row-reduction skeleton of PR 802's
first recipe (`evidence/plain-sheet-chip-claims/form-compiles.json`):

| form in the kernel | simulator | pair (tt-lang 1.1.10.dev20261026+light) | tt-lang 1.1.6 |
|---|---|---|---|
| `ttl.block.fill`, `ttl.block.broadcast`, `reduce_sum(..., shape=(1, 1))` | passes (recipes verifier, 6 of 6 sets) | compiles | refused: `reduce_sum() got an unexpected keyword argument 'shape'` |
| the same with `reduce_sum(...)` and no `shape` | fails: `reduce_sum() missing 1 required positional argument: 'shape'` (A/B, rmsnorm off 1) | compiles | compiles |
| `ttl.math.fill` | fails: `'_TTLMathNamespace' object has no attribute 'fill'` (A/B, center off 1) | refused: `ttl.math.fill is not available; use ttl.block.fill` | the same message |
| `ttl.math.broadcast` | fails: `... has no attribute 'broadcast'` (A/B, center off 2) | refused: `ttl.math.broadcast is not available; use ttl.block.broadcast` | the same message |
| `ttl.make_dfb(...)` | accepted (checkout) | compiles | `AttributeError: module 'ttl' has no attribute 'make_dfb'` |

For the package the probe compiles with (the pair), the form that passes the simulator is the form its compiler takes: `ttl.block.*` and
`shape=`. The old sheet told the coder to write the forms that both the simulator and both compilers refuse. The 1.1.6 package is the
one case with no common form: it refuses `shape`, and the simulator requires it.

## What changed in the sheet

The plain sheet goes from 141 lines to 133 (the limit set for this change was 148). Nothing new was added except one mistake entry. The
sheet's bytes change, so runs with it are not comparable with runs before it (the sha256 pin in `tests/test_ttlang_recipes.py` records
both digests).

| line | before | after | evidence |
|---|---|---|---|
| source `hardware` | "the tt-lang package installed for the chip (1.1.6)" | the managed pair, when it is installed | `ttlang_pair.resolve_hardware_python`: the probe's hardware phase uses the pair when it exists and falls back to 1.1.6 |
| source `shared` | none | tt-lang 1.1.6, the fallback | the same function |
| source `tree` | "newer than the package" | "built from the same commit as the pair" | PR 802 finding 3 (commit `fec38c6de`) |
| tag text | `[compiles on hardware]`: an elementwise kernel ran on the chip | a kernel ran on the chip or compiled with the chip package's own compiler, with no device | PR 802 calibration: the check agrees with the chip on 41 of 41 rows |
| `ttl.math.reduce_sum` | `(input, *, dims)`, "no shape argument in the hardware package", hardware signature only | `(input, *, dims, shape)`, shape passed; the simulator requires it, the pair takes it, 1.1.6 refuses it; compiles on hardware | table above; issue 205: 11 of 11 reduction kernels compiled and ran on a chip with the pair |
| `ttl.math.reduce_max` | the same, no shape | the same, with shape | the table (softmax A/B kernels) and the verifier |
| `ttl.math.broadcast` | `ttl.math.broadcast(input, *, dims, shape)`, hardware signature only | `ttl.block.broadcast(input, *, dims, shape)`, compiles on hardware | table above |
| `ttl.math.fill` | `ttl.math.fill(value, *, shape, dtype=None)`, hardware signature only | `ttl.block.fill(value, *, shape)`, compiles on hardware | table above |
| section "Simulator-only forms" | `reduce_sum` with shape, `ttl.block.fill`, `ttl.make_dfb` | removed: the first two are now the main entries; `make_dfb` moved to "Kernel structure", compiles on hardware | table above |
| mistake 0 | "call reduce_sum with no shape, then ttl.math.broadcast the result back" | keep `shape=`; only 1.1.6 refuses it and no one form passes both; build the pair with `tt-orchard kernel-env --pair --apply` | table above; signature checks on the `shared`, `hardware` and `sim` trees |
| mistake 1 | none | `ttl.math.fill` and `ttl.math.broadcast`: the simulator message and both compilers' messages, and the fix | table above |

What did not change: the simulator-side guidance. Every example the simulator needs is as before or more explicit (`shape=` on the
reductions, `ttl.block.fill` and `ttl.block.broadcast`, `make_dataflow_buffer_like` with `block_count`). A test pins them.

A line that cannot be correct for both: the shape argument of a reduction under tt-lang 1.1.6. It is labelled with its package in the
sheet (the mistake entry names 1.1.6, the simulator and the pair separately).

## The 15 attempts of the recipes A/B (2026-11-21)

`scripts/ab_sheet_counterfactual.py` reads the judged kernel of each attempt, counts the forms, and for a kernel with a wrong-claim form
rewrites it mechanically (`ttl.math.fill` and `ttl.math.broadcast` become `ttl.block.*`, and a reduction with no `shape` gets
`shape=(1, 1)`) and runs the simulator's unedited check and the chip compilers on the rewrite
(`evidence/plain-sheet-chip-claims/ab-counterfactual.json`).

| attempt | recorded simulator result | wrong-claim forms in the judged kernel | rewrite: simulator | rewrite: chip compilers |
|---|---|---|---|---|
| center off 1 | fail: `'_TTLMathNamespace' ... 'fill'` | `ttl.math.fill` x1 | fail: deadlock | not run |
| center off 2 | fail: `... 'broadcast'` | `ttl.math.fill` x1, `ttl.math.broadcast` x3 | pass | pair: refused (`<ast.Div ...> not implemented`); 1.1.6: refused (`shape`) |
| center off 3 | fail: `Block.__getitem__` validation | `ttl.math.broadcast` x1 | fail: the same validation error | not run |
| rmsnorm off 1 | fail: `reduce_sum() missing ... 'shape'` | reduction without shape x1 | fail: `unsupported operand for block truediv: float` | not run |
| softmax off 3 | fail: `reduce_max() missing ... 'shape'` | `ttl.math.broadcast` x2, reductions without shape x2 | fail: deadlock | not run |
| softmax on 1 | fail: `reduce_max() missing ... 'shape'` | the same | pass | pair: refused (`TTL subscript base must be ...`); 1.1.6: refused |
| center on 1, center on 3, softmax on 3 (the three passes) | pass | none | not needed | (recorded: center x2 refused by the pair for a division inside a thread; softmax compiles with the pair) |
| the other 6 | fail for other reasons | none | not needed | not needed |

Counts. 6 of the 15 judged kernels contain a form the old sheet taught wrongly; 5 of them ended on a simulator error that names the
form (the 6th, center off 3, ended on a different error before reaching it). After the mechanical rewrite 2 of the 6 pass the
simulator and none compiles with either chip compiler; the other 4 still fail, so the wrong claim was one cause among several in them.

**None of the three passes uses a wrong-claim form.** All three use `ttl.block.fill` or `ttl.block.broadcast` and `shape=`, which the
corrected sheet teaches. So the correction cannot turn one of these passes into a failure by changing what a form means. What this data
cannot show: that a coder given the new text writes the same forms again. Only a run shows that (the next A/B below).

## The risk of making anything worse

1. **The sheet's bytes changed.** Runs with it are not comparable with runs before 2026-11-22 (the A/B of 2026-11-21 ran with the old
   sheet in both arms). The pin in `tests/test_ttlang_recipes.py` keeps both digests.
2. **A run with tt-lang 1.1.6 as the chip package now gets text that points at a form 1.1.6 refuses** (`shape=`). Before, it pointed at a
   form the simulator refuses. The probe judges by the simulator, so a 1.1.6 run could pass the simulator and then fail the chip compile
   either way; the mistake entry says so and names the pair as the fix. A run with no pair venv still prints a warning.
3. **Two of six rewrites pass the simulator and are refused by the chip compilers for other reasons.** A coder that stops writing the wrong
   names may reach more simulator passes that cannot run on a chip. That is why the A/B below counts passes that a chip compiler accepts,
   and not simulator passes alone.
4. **Not checked here:** a model's behaviour with the new text. No coder ran.

## The next A/B (needs the coder after the Aion benchmark)

Three arms, as pre-registered in `2026-11-22-recipes-dialect-gate.md`, with the plain arm now the corrected sheet:

| arm | flag | what the coder gets |
|---|---|---|
| plain | `--cheat-sheet-recipes off` | the corrected sheet (133 lines) |
| control | `--cheat-sheet-recipes control` | the corrected sheet and the length-matched section with no recipe content |
| recipes | `--cheat-sheet-recipes on` | the corrected sheet and the recipes |

- Same coder and server, temperature 0.7, `--library none --no-agent-hw --phase sim`, 40 turns; at least 10 attempts per arm and op (`center`
  and `softmax`, `rmsnorm` reported apart); rotate the arms within blocks of three. At 10 per arm, 5 passes against 0 is the smallest
  difference separable at 0.05.
- A pre-flight before it starts: the pair venv exists (`tt-orchard kernel-env --pair --apply`), and `hwcompile_probe_passes.py` runs on
  one finished attempt. The outcome that counts is a simulator pass that the pair's compiler accepts.
- Old arms cannot be reused: the sheet changed. A fourth arm, the old plain sheet, would measure what this change is worth.
- Time on the coder. The 15 attempts of 2026-11-21 took 5,127 s of agent time in all (a mean of 342 s, from 64 s to 799 s each). Ten attempts
  per arm on `center` and `softmax` is 60 attempts for three arms (about 5.7 hours at that mean) and 80 with the fourth arm (about 7.6 hours),
  without `rmsnorm`. Count load windows and run no suite during it.

## Not verified

- No kernel written from the corrected sheet has run: no coder was used.
- No recipe or A/B kernel has run on a chip (issue 800 lists the chip run).
- The compile check does not build with tt-metal, check L1 fit or run numbers (PR 802).
- The simulator column of the table comes from the recorded A/B errors and from PR 802's 6 of 6 recipe checks; this change did not
  re-run the simulator on the skeleton variants.
