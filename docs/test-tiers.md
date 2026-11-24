# Test tiers and the diff selector

The full suite has about 32,000 tests in 758 files and takes about 19 minutes with `pytest -n 6`. Most changes touch a few modules.
This page describes a fast loop for development and keeps the full suite as the gate before a merge.

## The rule that does not change

Before a merge, the full suite runs once on the integration state: main plus the branch, with `origin/main` shown to be an ancestor of
`HEAD`, one suite at a time on the machine. The selector and the markers below are for the loop while developing. A green selection is not a
green suite.

## What was measured

A full run on 2026-11-23 (six workers, other suites running on the same machine, so each time is a little high):

| | Files | Tests | Test seconds |
| --- | --- | --- | --- |
| Whole suite | 753 | 31,910 | 5,796 |
| Files above 30 s of test time (`slow`) | 51 | 8,547 | 3,987 (69 percent) |
| The rest (the default tier, `-m "not slow"`) | 702 | 23,455 collected | about 1,800 |

The ten slowest files hold 1,705 s. They spawn worker pools or subprocesses (`test_make_franken`, `test_hardware_check`,
`test_audio_reference_script`, `test_run_e2e`, `test_phone_a_friend_ladder`) or run thousands of parametrized cases
(`test_hostile_readers`: 5,015 tests, 133 s). No test in the suite needs a chip: the suite never opens a device.

## The markers

Registered in `pyproject.toml` with `--strict-markers`, so a misspelt marker fails collection.

| Marker | Meaning | How a test gets it |
| --- | --- | --- |
| `slow` | The file is above 30 s of test time | `SLOW_FILES` in `tests/tier_lists.py` |
| `whole_repo` | A guard that scans the repository | `WHOLE_REPO` in `tests/tier_lists.py` |
| `hardware` | Needs a chip; skipped unless `ORCHARD_RUN_HARDWARE=1` | `HARDWARE_FILES` in `tests/tier_lists.py` (none today) |
| `real_bundle` | Runs the real stage 8 bundle builder | `@pytest.mark.real_bundle` (already in use) |

`tests/conftest.py` applies the lists when pytest collects. A test file in no list is in the default tier: it runs under the full suite and
under `-m "not slow"`. A new test file needs no marker. A guard that lives inside a slow file is named by function in `WHOLE_REPO`, keeps the
`whole_repo` marker and is not marked `slow`.

`pytest -m "not slow"` runs the default tier. `pytest -m whole_repo` runs the 349 guard tests (about 16 seconds of test time, plus the
collection of the files they live in).

## The guards

A guard fails when something unrelated to its own subject changes: a new module with raw file access, a float check without `finite: ok`, a
reader that is not registered as hostile-input, direct sqlite use, a new command flag the README does not name, a home path in a skill or
package template, a malformed change fragment, a `CLAUDE.md` over its size. A change to the new code never fails the tests of that
code, so only a test that scans the whole repository sees it. They are cheap (the whole set is about 16 s), so every selection includes them.

The list was found with a fault run, then checked with a scanner:

1. On 2026-11-23 a scratch copy of main got one fault of each kind above, and one full suite ran. Ten tests failed. Nine of them (the
   functions in `FAULT_RUN_FAILURES` in `tests/test_tier_lists.py`) are the core of the list.
2. `tests/test_tier_lists.py` also scans the test files with a syntax-tree check. A test file that walks the repository (`glob`, `rglob`,
   `iterdir`, `os.walk`, `listdir`, `scandir` on the repository root or its source directories), or loads a script by name that does,
   has to be in `WHOLE_REPO` or in `NOT_GUARDS` with a reason. That check found `test_runaway` and `test_block_family_authoring_skill`,
   which the fault run did not reach.

The scanner sees directory walks. It does not see a guard that checks the repository another way (the README flag check parses the command's
argument parser, the version check reads `changes/`). Those come from the fault run. When you add a rule that reads the whole repository by
some other route, add its test to `WHOLE_REPO`.

## The selector

```
python3 -m pytest $(python3 scripts/tests_for_diff.py)             # the tests this branch can affect
python3 -m pytest $(python3 scripts/tests_for_diff.py --fast)      # the same, without the slow files
python3 -m pytest $(python3 scripts/tests_for_diff.py --direct)    # only tests that import a changed file themselves
python3 scripts/tests_for_diff.py --base HEAD~3                    # a different starting point (default origin/main)
```

Standard output is one line of arguments. Standard error lists what changed, each selected test with the chain that led to it
(`tests/test_c.py -> orchard/c.py -> orchard/b.py -> orchard/a.py`), and the reason for any fall back to the full suite.

The diff is the working tree against the merge base with `--base`, plus untracked files. The files are read as syntax trees. Nothing is
imported, so a file with a syntax error does not stop the script.

A file depends on another when it imports it (any form, at any depth inside a function or `try`), when a string in it names the other file
(`"scripts/tool.py"`, `Path(...) / "tool.py"`), or when a string holds code that imports a module (`-m orchard.a`, `"from orchard import a"`).
The program `bin/tt-orchard` is a file in the graph, and a test that names `tt-orchard` depends on it. A test is selected when a chain of
these links reaches a changed file. `--direct` keeps only the tests with a link to the changed file themselves; it is faster, and it leaves
out the tests that reach the change through other modules.

### What it does not guess

The script selects the full suite, and names the file, for:

- `tests/conftest.py` and every other test helper, `tests/tier_lists.py`
- `pyproject.toml`, requirements files, CI configuration under `.github/`
- a deleted or renamed module under `orchard/`, `scripts/` or `bin/`
- anything under `orchard/skills/` or `orchard/package_templates/` (templates that modules load by path)
- a data file that no test names (a fixture, a config file), a base ref that does not exist, a directory that is not a git repository
- a `tests/tier_lists.py` that is missing or names a guard that does not exist

It ignores `changes/` fragments (the guards read them). A document (`*.md`, `docs/`) selects the tests whose strings name it, plus the guards.
A test file that computes an import (`importlib.import_module(name)`) or does not parse is selected for every change to Python source,
because its dependencies cannot be read.

### Where it can miss

- A link made at run time by a computed path or a name built from pieces (`"orchard." + name`) is invisible. The computed-import rule covers
  `import_module(name)` in test files. A test that reaches a module only through an environment variable or a configuration file is not covered.
- A change to behaviour that a test checks through a subprocess is found only when a string in the test names the file or module.
- A change whose only effect is on a guard's rule (a new rule in a scanner) is caught when the scanner file is changed, not before.

The history check below shows where this happened in practice.

## Adding a test, a slow file, a guard

- A new test file needs nothing. It runs by default.
- A file that grows past 30 s of test time: add it to `SLOW_FILES` (the full run's `--durations` report shows it). Keep `slow` for files, not
  single tests, so a file's tests stay together.
- A new repository-wide rule: add its test to `WHOLE_REPO`, by function name when the file is large.
