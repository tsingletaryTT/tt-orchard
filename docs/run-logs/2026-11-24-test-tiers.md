# Test tiers and the diff selector: measurements and checks (2026-11-23 and 2026-11-24)

The operator asked whether every test has to run every time or whether the tests could be classified. This page records what was measured
and how the markers and the selector were checked. The rules and the commands are in [test-tiers.md](../test-tiers.md).

## 1. The full run that sets the tiers

`pytest -n 6 --durations=0 --junitxml` on main (58352a33), 2026-11-23: 31,826 passed, 91 skipped, 1 xfailed in 19 minutes 20 seconds. Two
other suites ran on the machine at times, so every time is a little high. The report gave 31,918 test results in 753 files and 5,796
seconds of test time (the sum over workers).

| Threshold | Files above it | Test seconds in them |
| --- | --- | --- |
| 20 s per file | 72 | 4,504 |
| 30 s per file (chosen) | 51 | 3,987 (69 percent), 8,547 tests |
| 45 s per file | 38 | 3,530 |
| 60 s per file | 27 | 2,962 |

The 30 second line keeps 7 percent of the files in `slow` and leaves about 1,800 seconds of test time in the default tier. Above 45 seconds
the gain per file drops, and below 20 seconds the files are many and short. Single tests above 10 seconds number 58 and hold 1,145 seconds, but
they sit in the same files, so the marker is per file. The slowest single tests are two in `test_audio_reference_script` (93 s each) and
one case of `test_run_e2e` (88 s). `test_hostile_readers` has 5,015 tests and 133 s.

No test in the suite opens a device, so the `hardware` marker is registered and has no files.

## 2. Which tests are guards: a fault run

A scratch copy of main got one fault of each kind: a module with raw file access, an unchecked float, direct sqlite use and a reader that is not
registered; a new flag the README does not name; a home path in a skill file and in a package template; a malformed change fragment. One full
suite ran (31,818 passed, 10 failed, 29 minutes under load). The failures:

- `test_finite_ratchet::test_no_file_gained_a_float_check_that_a_huge_integer_can_crash`
- `test_hostile_readers::test_the_registry_covers_these_modules_and_the_rest_are_listed`
- `test_no_sql_outside_the_database_layer::test_no_module_but_the_database_layer_holds_sql_or_imports_sqlite3`
- `test_hardening_ratchet::test_no_file_gained_raw_file_access`
- `test_readme::test_every_flag_of_the_tt_orchard_command_is_in_the_readme`
- `test_skills::test_no_agent_facing_file_names_a_machine_path` (two cases)
- `test_version::test_every_fragment_in_the_repository_passes_check` and `test_no_fragment_in_the_repository_is_at_or_below_the_release`
- `test_version_cli::test_the_script_check_passes_on_this_repository`

The fault that adds a home path to a plain document under `docs/` and the one that leaves out the SPDX header failed no test, so no guard
exists for them. A scanner check in `tests/test_tier_lists.py` found two more test files that walk the repository
(`test_runaway`, `test_block_family_authoring_skill`). `WHOLE_REPO` holds 11 files; the guard tests are 349 of the 32,001 collected, about 16
seconds of test time.

## 3. The history check

For each of the 15 most recent merged pull requests on first-parent main (835 to 822, as of 2026-11-24), a scratch worktree was checked out
at the merge commit and the selector was asked for the diff of the merge against its first parent. The registry of that worktree held the
guards that exist there. The check compared the selection with the test files the pull request added or changed itself ("own tests").

| Pull request | Source files changed | Own test files | Default selection | `--eager` | `--direct` | Own tests left out |
| --- | --- | --- | --- | --- | --- | --- |
| #835 | 10 | 11 | full suite (template) | - | - | none |
| #834 | 0 | 0 | 10 of 772 | 10 | 10 | none |
| #833 | 1 | 1 | 715 of 772 | 563 | 395 | none |
| #832 | 1 | 1 | 574 of 772 | 512 | 18 | none |
| #831 | 0 | 0 | 10 of 771 | 10 | 10 | none |
| #830 | 1 | 1 | 714 of 771 | 563 | 395 | none |
| #829 | 1 | 1 | 572 of 770 | 16 | 16 | none |
| #828 | 0 | 0 | 10 of 769 | 10 | 10 | none |
| #827 | 0 | 0 | 10 of 769 | 10 | 10 | none |
| #826 | 1 | 1 | 712 of 769 | 561 | 395 | none |
| #825 | 4 | 1 | 712 of 768 | 543 | 418 | none |
| #824 | 4 | 2 | 578 of 767 | 516 | 41 | none |
| #816 | 1 | 1 | 397 of 766 | 134 | 19 | none |
| #823 | 5 | 2 | 553 of 766 | 362 | 41 | none |
| #822 | 3 | 2 | 563 of 765 | 498 | 181 | none |

- Four of 15 pull requests changed only documents and change fragments; each selected 10 files, the guards.
- Ten changed source. The default selected 397 to 715 files of about 770 (52 to 93 percent). The package is tightly linked, so a change
  to one module often sits in a chain of imports behind hundreds of tests.
- One changed a skill template and selected the full suite by rule.
- In every row and every mode, the tests the pull request wrote were in the selection.

What this does not show: that the tests a pull request did not change would still pass. Showing that needs the full suite on the reverted
change, which this check did not run. The narrow modes leave out tests on purpose.

### The check repeated on 2026-11-25 over the 10 newest merged pull requests (868 to 852)

After 15 more pull requests merged, with the final selector (the registry holds the guards that exist at each merge):

| Pull request | Source files | Own tests | Default | `--eager` | `--direct` | Own tests left out |
| --- | --- | --- | --- | --- | --- | --- |
| #868 | 4 | 1 | 598 of 798 | 533 | 72 | none |
| #866 | 4 | 3 | 738 of 797 | 644 | 428 | none |
| #863 | 2 | 1 | 737 of 796 | 583 | 428 | none |
| #857 | 2 | 2 | full suite (`tests/hostile_fs.py` is a shared helper) | - | - | none |
| #851 | 4 | 1 | 589 of 793 | 523 | 207 | none |
| #861 | 1 | 1 | full suite (template under `orchard/skills/`) | - | - | none |
| #862 | 1 | 1 | full suite (template) | - | - | none |
| #860 | 1 | 2 | full suite (template) | - | - | none |
| #853 | 1 | 1 | 13 of 790 | 13 | 13 | none |
| #852 | 7 | 2 | full suite (template) | - | - | none |

Five of the ten fall back to the full suite, four because a skill or package template changed and one because a shared test helper changed.
Of the other five, the default selected 74 to 93 percent of the files for four and 13 files for one. In every row the tests the pull request
wrote were selected. The check does not know which of the other tests each pull request made fail, because failing-test sets of those pull
requests were not recorded; the guards were in every selection.

## 4. The #818 failures and the faults

The three repository checks that failed on #818 (`test_finite_ratchet`, `test_hostile_readers` registry, `test_readme` flag list) were
replayed by creating each fault in a scratch copy of this branch and asking the selector for the diff, then running only the guard tests the
selection contained.

Each fault was made in a scratch copy of this branch. The selector ran on the diff, and only the guard tests among the selected ones ran
(`-m whole_repo`).

| Fault | Selection | Guards that failed |
| --- | --- | --- |
| A new module that reads a file and is not registered as a hostile-input reader | 10 of 768 files (the guards) | `test_hostile_readers` registry test, `test_hardening_ratchet` |
| A new module with an unchecked `math.isfinite` | 10 of 768 files | `test_finite_ratchet` |
| A new `--flag` on `bringup` that the README does not name | 398 of 768 files, with the guards | `test_readme` flag test |

All three are caught by the selection. In the first two the guards are the whole selection; the new module is imported by no test.
Before the selector read conftest one step at a time and ignored docstrings, the third selection reached 548 files; the narrowing came from
those two corrections, and the guards were in every version.

## 5. Tests and mutations

`tests/test_tests_for_diff.py` has 90 tests (selection rules, import forms, string references, the fallbacks, conftest and its fixtures, the
git layer and the arguments), and `tests/test_tier_lists.py` has 24 (the lists, the markers, strict checking, the scanner check, the
conftest hook). Fifty-eight mutations were applied one at a time and each turned a test red: the import walk (one hop, plain, from, relative,
lazy), string references and program names, every fallback, the guards (whole file, by name, the fault-run list), the dynamic and unparseable
rules, `--direct`, `--eager` and `--fast`, the git layer (untracked files, base ref, not a repository, the default root), the conftest rules, the
markers (a guard that is also slow, an emptied slow list, the hook, strict checking, the hardware skip). The first pass left 9 green, which
showed missing cases: a full-suite rule that another rule also triggered, messages that were not checked, a scanner that did not follow scripts,
a parametrized name. Each got a test. One rule (a branch for `tests/tier_lists.py`) turned out to be covered by the rule for test helpers
and was removed.

Corrections made on the way, each from a measurement: docstrings and comments are not references (a docstring that names `tt-orchard` linked
`orchard/__init__.py` to the command and so every test to every module); a program name in a message of a module is not an edge; a document is
matched by its file name and not by the folder it sits in; conftest imports are followed one step. The tools report the explanation for each
choice on standard error.

## 6. What the selector does not do

The default narrows little for a change to a core module, because the graph says which tests load a file and not which check it. A map of the
files each test executes, taken from one full run with coverage, would narrow it. That is not built. `--eager` and `--direct` give fast
loops and leave out tests that reach a change through a function call or another module, so they are for feedback. The full suite on the
integration state stays the gate before a merge.
