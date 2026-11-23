What the cheat sheet's recipes say about the chip is now compiled, and a gate decides `auto`

The recipes of PR 796 passed the simulator and were tagged "hardware form not verified on chip", with hand-written `chip:` lines beside each one. This change checks what can be checked without a chip, removes what turned out to be false, and adds the control arm the A/B lacked. It does not turn the recipes on by default. Run log: `docs/run-logs/2026-11-22-recipes-dialect-gate.md`.

## What the risk was (checked on the CPU; no device was opened)
- **The `chip:` lines were wrong.** Both chip compilers refuse `ttl.math.fill` and `ttl.math.broadcast`: `ttl.math.fill is not available; use ttl.block.fill` (tt-lang 1.1.6 and the pair, 1.1.10.dev20261026+light). All four recipes' chip forms are refused by both (8 of 8). PR 796's static check passed them because it read the package's Python exports; a call inside a thread is resolved by the compiler's own table (`_NAMESPACE_OVERRIDES` in `_src/ttl_ast.py`).
- **The probe's chip package is the pair**, built from the simulator's commit. It takes `reduce_sum(..., shape=...)`. "0 of 7 compiled on the chip" (issue 205) is a fact about 1.1.6.
- **One recipe did not compile for the pair** in the form that passed the simulator: `rsqrt(total + eps)` gives `cannot lower tensor store to ttl.compute: store input is not dataflow-buffer-backed`. `rsqrt(total * one + eps)` compiles. The recipe is corrected and passes the simulator again.
- **Two of the A/B's three simulator passes cannot run on a chip.** Both `center` passes divide inside a thread, which the pair's compiler refuses. On passes a chip compiler accepts, the A/B of 2026-11-21 is 1 of 7 (recipes) against 0 of 8 (plain), Fisher exact p = 0.47.
- **The plain sheet misleads on the simulator too.** 5 of 15 A/B attempts ended on an error that comes from writing the form the sheet calls the hardware form. The recipes section opened with a paragraph that corrected it, so the A/B did not isolate the recipes' content. The plain sheet is not changed here (issue 801).

## What this adds
- **A compile with the chip package's own compiler, with no device** (`orchard/ttlang_hwcompile.py`, child `orchard/ttlang_compile_child.py`). The child shuts `/dev/tenstorrent` out of itself with Landlock and proves it (a listing must be refused) before tt-lang is imported; `ttnn` is a stand-in; tt-lang runs with `TTLANG_COMPILE_ONLY=1` and writes its kernel sources. A pass needs the proved guard and a written kernel source. A check that could not run has its own category and is never a refusal.
- **Calibration against the chip** (`scripts/calibrate_hwcompile.py`, `docs/run-logs/evidence/hwcompile-calibration/calibration.json`): the 21 kernels of `evidence/ttlang-pair` have recorded chip results for both packages. The check agrees on 41 of 41 rows: 30 the chip ran, 11 it refused, with the same first line each time, for three different messages.
- **Recipes verified per chip package.** No hand-written chip lines (`hardware_replace` is refused). Each recipe lists `compile_negatives`, forms the chip compiler must refuse. Each record holds, per package (version and sha256 of the compiler's files), the compile of exactly the printed skeleton and of each wrong form. The sheet is rendered for the run's package:

  | recipe | simulator | pair | 1.1.6 |
  |---|---|---|---|
  | row reduction, then spread | pass | compiles | refused (`shape`) |
  | column reduction, running total | pass | compiles | refused (`shape`) |
  | elementwise, constant and row operand | pass | compiles | compiles |
  | running total down the rows | pass | compiles | compiles |

  Tags: `[compiles with tt-lang X, no chip]`, `[refused by the compiler of tt-lang X]`, `[not compiled with tt-lang X]`, `[not compiled with a chip package]`, and `[ran on a chip with tt-lang X]` for a record only a chip run can write. None exists.
- **`--cheat-sheet-recipes auto`**: on only when every recipe compiled with the run's package and that compiler refused every wrong form. It holds for the pair today. Otherwise the run uses the plain sheet and the report says why. The default stays `off`.
- **`--cheat-sheet-recipes control`**: a section with the same framing and length (247 lines against 247, 14,506 characters against 14,656 for the pair) and four elementwise kernels: no reduction, no broadcast, no constant block, no carried value. Verified the same way.
- **`scripts/hwcompile_probe_passes.py DIR`**: which simulator passes of finished probe runs each chip compiler accepts.

## Considered and not built
- A translation layer from a simulator form to a chip form. For the pair there is nothing to translate. For 1.1.6, dropping `shape=` makes one of the two reduction recipes compile, in a form the simulator refuses.
- `auto` as the default. The gate removes one risk. The three-arm A/B pre-registered in the run log has to show the recipes help first.
- The compile check inside the probe's loop. It would change what the coder sees and is a second variable.

## Not verified, and what would close it
- No recipe has run on a chip. Issue 800 lists the run: one leased chip, about 15 minutes.
- The check does not build the kernel source with tt-metal, does not check L1, and uses a fixed 13 x 10 grid. All 11 chip refusals in the calibration are tt-lang compiler errors, so a kernel that compiles and then fails in tt-metal's build is not covered.
- Whether the recipes help. Three attempts per arm cannot separate any outcome; with 10 per arm, 5 against 0 is the smallest difference Fisher's exact test separates at 0.05.

## Tests
- New: `tests/test_ttlang_hwcompile.py` (96), `tests/test_recipes_dialect_gate.py` (56), `tests/test_hwcompile_calibration.py` (13). Changed: `tests/test_ttlang_recipes.py` (75), `tests/test_verify_ttlang_recipes.py` (13). 253 in all.
- The guard, the stand-in and the result are tested with the real child in a real small venv whose `ttl` is a fake compiler. Eight tests use the chip packages of the machine when they are installed and skip elsewhere; they ran here against tt-lang 1.1.6 and the pair.
- 57 mutations, each turned a test red. The first table had four survivors and three tests were tightened. Two survivors came from the tests' own skip condition: it asked the guard under test whether Landlock exists, so a broken guard skipped every guard test and the run was green. The condition now asks the kernel.
- Related files: 628 passed and 6 skipped (sheet, CLI, pair, ratchets, paths, README, version), then 733 passed (kernel probe, stage tools, port templates, hostile file readers). The full suite was not run: timing measurements were running on the host.

## Checklist (docs/pr-hardening-checklist.md)
1. Wrong type or shape in input: the child's result file is type-checked field by field before use (13 malformed shapes are tests); shapes, identities, compile entries and chip entries have their own checks (`identity_problems`, `record_problems`, `_compile_entries_problems`); a malformed entry is a named problem that stops the section.
2. Files that something else controls: the kernel file, the package's METADATA and compiler files, the result file and the probe reports are read through `safeio` with byte caps; the records are written with `safeio.atomic_write`. A FIFO or an oversized file as the kernel is a test. The readers are not registered in `tests/hostile_inputs.py`; they add no reader of their own beyond `safeio`.
3. Check, then use: the package's identity and the kernel's sha256 are taken before the compiler runs, and the compiler reads a private copy with those bytes (a test swaps the caller's file during the run). A stale result file is removed before the run (test).
4. Evidence an agent can write: none added. The records come from the verifier; the probe's coder cannot write them.
5. A verdict stronger than its evidence: a compile is named `compile` everywhere and tagged "no chip"; the `chip` level needs an entry with a package, a skeleton hash, a date, a PCC and an evidence path, and nothing writes one yet. A timeout or a failed guard is not a refusal, and a wrong form counts as refused only when the compiler refused it.
6. The check looks at a different object than the action uses: the compiled bytes are the bytes the sheet prints (same hash, tested); a record counts only for the same package version and compiler bytes and the same skeleton.
7. A parser or matcher too broad or narrow: the package is identified by the `tt_lang` distribution only (the simulator wheel `tt_lang_sim` is refused, test); `operation_name` takes exactly one top-level `ttl.operation` function.
8. Failure paths that leave state behind: the scratch directory and the compiler's output directory are removed on every path; `auto` that cannot read the records is off with the reason.
9. Numbers and sizes: kernel 256 KiB, result 256 KiB, 16 tensors, dimensions to 2^20, first line 300 characters, 8 packages per record, 6 wrong forms per recipe; `pcc_min` goes through `safeio.is_finite_number`. Both finite ratchets pass.
10. Secrets, paths and host names: the scratch path is replaced by `<work>` and memory addresses by `0x...` in every recorded line; a test refuses `/home/`, `/tmp/` and `/mnt/` in the four data files. The paths test passes.
11. Tests that do not test the claim: 57 mutations, four survivors fixed; the test of the committed calibration file reads the chip's own result files again and ignores the file's `agree` flags; a test compiles the shipped skeletons again and compares with the records where the compilers are installed.
12. Housekeeping: `changes/recipes-dialect-gate.toml` (0.9.192), a dated entry at the end of `docs/project-log.md`, the flag's four values in the README (`tests/test_readme.py` passes), `scripts/version.py check` passes.

Issues: 800 (the chip run to schedule), 801 (the plain sheet's chip claims). Part of issue 205.
