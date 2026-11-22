Fixes the 16 test failures on `main` after the merge of PR 788.

## What failed
PR 788 made `Supervisor._pin_full_port` read `self.port_check_positions`. Seven hand-built `SimpleNamespace` stand-ins in four test files call that method and had no such attribute, so 16 tests raised `AttributeError`. They fail the same way when run alone on `main` at `c4b3bdc6`. The real constructor always sets the attribute, so no production code changes.

## Change
Each of the seven stand-ins now carries `port_check_positions=full_port.DEFAULT_POSITIONS`, the value the constructor gives a run that does not name `--port-check-positions`. Files: `tests/test_cheat_sheet_wiring.py` (2), `tests/test_decoder_idioms.py` (3), `tests/test_run_state_followups.py` (1), `tests/test_run_file_readers.py` (1, inside a child-process script that already imports `full_port`). Plus a change fragment and a log entry.

## Evidence
- Before: 16 failed on `main`, alone and in the full suite.
- After: 248 passed across those four files and the two other files that mention `_pin_full_port`.
- Full parallel suite on this branch: 30,353 passed, 147 skipped, 1 xfailed, 0 failed.

## Checklist (docs/pr-hardening-checklist.md)
1. Wrong type or shape in input: not touched; tests only.
2. Files that something else controls: not touched; no file access added.
3. Check, then use: not touched.
4. Evidence an agent can write: not touched.
5. A verdict stronger than its evidence: not touched.
6. The check looks at a different object than the action uses: not touched.
7. A parser or matcher too broad or narrow: not touched.
8. Failure paths that leave state behind: not touched.
9. Numbers and sizes: the stand-ins use the module's own `DEFAULT_POSITIONS` (64). The tests do not repeat the number.
10. Secrets, paths and host names: none added.
11. Tests that do not test the claim: the failing tests are the check. Each was red on `main` for this cause and is green with the change. No clock decides a result.
12. Housekeeping: change fragment `changes/fix-pin-stubs.toml`, dated log entry at the end of `docs/project-log.md`, `scripts/version.py check` passes. No new flag, no README change.

## Process note
The 788 branch was checked with targeted test files that do not include these four, and the earlier integration suite did not contain 788. A pull request that adds an attribute to `Supervisor` needs the full suite on its own head before merge.
