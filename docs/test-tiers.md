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
python3 -m pytest $(python3 scripts/tests_for_diff.py)             # every test this branch can reach (the safe default)
python3 -m pytest $(python3 scripts/tests_for_diff.py) -m "not slow"   # the same, without the slow files
python3 -m pytest $(python3 scripts/tests_for_diff.py --eager)     # follow only the imports that run when a module loads
python3 -m pytest $(python3 scripts/tests_for_diff.py --direct)    # only tests that import a changed file themselves
python3 scripts/tests_for_diff.py --base HEAD~3                    # a different starting point (default origin/main)
```

`--fast` adds `-m 'not slow'` to the printed arguments. The script works on the repository of the current directory (or `--root`).

Standard output is one line of arguments. Standard error lists what changed, each selected test with the chain that led to it
(`tests/test_c.py -> orchard/c.py -> orchard/b.py -> orchard/a.py`), and the reason for any fall back to the full suite.

The diff is the working tree against the merge base with `--base`, plus untracked files. The files are read as syntax trees. Nothing is
imported, so a file with a syntax error does not stop the script.

A file depends on another when it imports it (any form, at any depth inside a function or `try`), when a string in it names the other file
(`"scripts/tool.py"`, `Path(...) / "tool.py"`), or when a string holds code that imports a module (`-m orchard.a`, `"from orchard import a"`).
Docstrings and comments are not strings that name a file. A test names the program `bin/tt-orchard` by its name, and that program is a file
in the graph. Every test depends on `tests/conftest.py`; its autouse fixtures and hooks import modules, and a fixture a test has to ask for
adds its imports only to the tests that name it (as a parameter, through another fixture, or in `usefixtures`). A module that conftest
imports is followed one step: a change to `orchard/skillsfresh.py` reaches every test, and a change to a module that `skillsfresh` imports
reaches only the tests that depend on it themselves. A test is selected when a chain of these links reaches a changed file.

### How much it narrows

The package is tightly linked: `orchard/stages.py`, `supervisor.py`, `ledger.py` and `safeio.py` are imported, directly or through a chain,
by hundreds of test files. A change to a module that sits in one of those chains reaches most tests. Measured on the 15 most recent
merged pull requests (see [the run log](run-logs/2026-11-24-test-tiers.md)):

| Kind of change | Test files selected by the default |
| --- | --- |
| Only documents and change fragments (4 of 15 pull requests) | 10 of about 770: the guards |
| One to a few source files and their tests (10 of 15) | 397 to 715 of about 770 (52 to 93 percent) |
| A skill or package template (1 of 15) | the full suite, by rule |

The default is the safe choice. Two flags narrow it, and each says what it leaves out:

- `--eager` follows only imports that run when a module loads. Inside a function, `from orchard import speech_check` runs only when the
  function is called, so a test that reaches `speech_check` through `cli.main(...)` is left out. Measured on the same 10 source pull
  requests: 16 to 563 files.
- `--direct` keeps the tests that import the changed file themselves (through a test helper at most). Measured: 16 to 418 files.

When the default selects more than half of the test files, the explanation prints how many files `--eager` and `--direct` would select.
Choose the narrow loop for quick feedback, and run the default (or the full suite) before you push.

### What it does not guess

The script selects the full suite, and names the file, for:

- `tests/conftest.py` and every other test helper, `tests/tier_lists.py`
- `pyproject.toml`, requirements files, CI configuration under `.github/`
- a deleted or renamed module under `orchard/`, `scripts/` or `bin/`
- anything under `orchard/skills/` or `orchard/package_templates/` (templates that modules load by path)
- a data file that no test names (a fixture, a config file), a base ref that does not exist, a directory that is not a git repository
- a `tests/tier_lists.py` that is missing or names a guard that does not exist

It ignores `changes/` fragments (the guards read them). A document (`*.md`, `docs/`) selects the tests whose strings name its file name, plus the guards.
A test file that computes an import (`importlib.import_module(name)`) or does not parse is selected for every change to Python source,
because its dependencies cannot be read.

### Where it can miss

- A link made at run time by a computed path or a name built from pieces (`"orchard." + name`) is invisible. The computed-import rule covers
  `import_module(name)` in test files. A test that reaches a module only through an environment variable or a configuration file is not covered.
- A change to behaviour that a test checks through a subprocess is found only when a string in the test names the file or module.
- `--eager` and `--direct` leave out tests that reach a change through a function call or through another module. They are for feedback, not
  for a decision to merge.
- The graph says which tests can load a changed file. It does not say which tests check its behaviour, so the default selects far more
  tests than a change needs. Recording which files each test executes (coverage with one context per test file) from one full run would
  give a map that narrows the selection; this page does not build it.

The history check in the run log compares the selection with the tests each pull request changed itself. It found none missing in
any mode.

## Adding a test, a slow file, a guard

- A new test file needs nothing. It runs by default.
- A file that grows past 30 s of test time: add it to `SLOW_FILES` (the full run's `--durations` report shows it). Keep `slow` for files, not
  single tests, so a file's tests stay together.
- A new repository-wide rule: add its test to `WHOLE_REPO`, by function name when the file is large.
