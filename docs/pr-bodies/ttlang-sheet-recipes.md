Adds recipes by kind of op to the TT-Lang cheat sheet, behind a switch, so an A/B can change one thing.

## What it is
`kernel-probe --cheat-sheet-recipes on` writes four recipes after the cheat sheet in each attempt's directory (`ttlang_cheat_sheet.md`). Default `off`: the sheet is byte for byte what it was (sha256 pin `f425cee8…d96b`, 142 lines).

Each recipe is a short plan and a complete skeleton for a kind of op, written for a worked op that is not a probe op:

| recipe | worked op |
|---|---|
| reduce each row to one value, then use it across the row | L2-normalise each row |
| reduce down the rows over more tile rows than one block holds | mean of each column |
| elementwise chain with a constant and a row operand reused for every row block | `x * y * 0.5 + r` |
| a value carried from one row block to the next | running sum down tile rows |

The headings name no probe op, and a test keeps it that way (`center`, `rmsnorm`, `softmax`, `row_sum`, `linear_scan` are the ops the A/B measures).

## Verification (scripts/verify_ttlang_recipes.py, run on 2026-11-21, tt-lang-sim 1.1.10+gfec38c6)
- Each skeleton passes the unedited kernel check on all 6 input sets (PCC at least 0.9999998).
- Each wrong variant fails the same check: 5 variants, all `numeric_mismatch`. One recipe's wrong variant is the claim in its text: broadcasting the row operand down two tile rows fails (PCC 0.24 in the scratch run that measured it).
- The chip form (each recipe's `hardware_replace` pairs applied) is read statically from the installed tt-lang 1.1.6 package with `ast`: every `ttl.*` call exists under the same namespace, and every keyword is a parameter of that function. `resolve` in `op_cheats` matches by the last name only, so it cannot tell `ttl.block.fill` from `ttl.math.fill`; the new check can, and has a test that it refuses the simulator's names.
- **No recipe has been compiled on a chip.** The sheet says so beside each recipe, with the issue 205 figure (0 of 7 simulator-passing reduction kernels compiled on the chip).

A recipe reaches the coder only with a record for exactly that skeleton (sha256, pass on 6 sets, every wrong variant failed, chip-form calls found). A stale or failed record stops the run before any attempt directory exists.

## One paragraph resolves the skill/sheet conflict, in the sheet only
The skill (`orchard/skills/kernel-probe.md`) teaches the simulator's forms (`ttl.block.*`, `shape=`), and the sheet teaches the chip package's. The recipes section opens by saying which form the unedited check accepts (simulator) and which compiles for the chip. The skill is not edited here: that would be a second variable.

Proposal for a follow-up PR: make the skill's "rules" list say the same thing in one line, and point to the recipes for the kind of op, after the A/B has been read.

## Pre-registration of the A/B (to run after the opencode batch, on the same coder server, one client at a time)
- One variable: `--cheat-sheet-recipes on` against `off`. Same coder and server, same skill text, temperature 0.7, `--library none --no-agent-hw --phase sim`, 40 turns, same wall budget.
- Held-out ops: `center` and `softmax`, N attempts each, alternating on and off in ABBA order. `rmsnorm` is reported on its own because it is close to the L2-normalise recipe.
- Readout, counts only: attempts, simulator passes, failure classes, turns, agent wall seconds, and for passing kernels the margins and the static check against the sheet's chip forms. The earlier ladder (2026-10-28, 2-chip coder) and the opencode batch (4-chip coder, `center` 1 of 6 on the orchard harness) are context and do not serve as controls.
- What it cannot show: a chip compile, more than one model, or anything about ops outside these families.

## Tests
- `tests/test_ttlang_recipes.py` (57) and `tests/test_verify_ttlang_recipes.py` (9). Against the unchanged code, 49 of the first 54 failed; the 5 that passed pin today's behaviour (the default sheet's hash, an attempt without the switch, `main` with the switch off or unnamed).
- 45 mutations, each turned a test red: 37 on the recipes code, the probe plumbing and the CLI, and 8 on the verifier. The first table had two survivors (a loader that skipped its own shape check, and a per-recipe evidence line hidden by the intro's sentence); both tests were tightened and the table rerun. V3 and V8 fail through the same test, so the verifier's `**kwargs` branch has no test of its own.
- 87 test files that mention the changed modules, the ratchets, the README test, the paths test and the four files that broke `main` after PR 788: 4193 passed, 6 skipped. After the last wording edits the four files, the probe, CLI and cheat-sheet files were rerun: 810 passed, 4 skipped. The full suite was not run.
- The simulator runs are not in the tests. They are `scripts/verify_ttlang_recipes.py`, and the tests check its records against the skeletons.

## Checklist (docs/pr-hardening-checklist.md)
1. Wrong type or shape in input: the recipes file and the records are read through `safeio.read_json` with a byte cap; `recipes_shape_problems` checks every type and bound (16 malformed shapes are tests); the switch takes `on` or `off` only, and `main` refuses anything else.
2. Files that something else controls: neither data file is written by the coder or the supervisor. Both are read through the bounded reader. The verifier reads the chip package through `safeio.read_bytes` (4 MB cap) with `ast` and never imports it.
3. Check, then use: the sheet writer renders from the same loaded objects it checked. Preflight and each attempt both load and check; the files are repository files, and the provenance hash is read when the attempt records it.
4. Evidence an agent can write: not touched. The records come from the verifier, and the attempt directory's sheet is written by the probe.
5. A verdict stronger than its evidence: each recipe carries `[passes the simulator]` and `[hardware form not verified on chip]`. The chip-form check is named as a name and keyword check. No text says a recipe compiled.
6. The check looks at a different object than the action uses: the record's sha256 is over the skeleton lines the renderer prints, and the verifier writes exactly those lines to `kernel.py` (a test pins the hash rule). The chip form is the skeleton with the printed line pairs applied, and the static check uses those same pairs.
7. A parser or matcher too broad or narrow: the static check is namespace-aware (`ttl.block.*` is refused, `ttl.math.*` accepted) and tested against a fake package for the names it must refuse. The sheet's `resolve` matches by last name; it is not used for this.
8. Failure paths that leave state behind: a refusal happens before the output directory exists (tests). A sheet that cannot be written leaves no file (atomic write). The verifier works in a temporary directory removed on exit.
9. Numbers and sizes: file 200 KB, at most 12 recipes, 120 skeleton lines of at most 200 characters, 10 plan steps, 900 characters of prose; no `float()` on a JSON value (the finite ratchet passes; the record's PCC is shown through `safeio.is_finite_number`).
10. Secrets, paths and host names: none added. The verified file holds the simulator and package versions only. The paths test passes.
11. Tests that do not test the claim: red run 49 of 54, 45 mutations, two survivors fixed. No clock decides a result.
12. Housekeeping: change fragment `changes/ttlang-sheet-recipes.toml` (0.9.191), a dated entry at the end of `docs/project-log.md`, the flag in the README (`tests/test_readme.py` passes), `scripts/version.py check` passes.

Part of the work on how the coder is helped on ops with no worked kernel. No issue number: the A/B above is the follow-up.
