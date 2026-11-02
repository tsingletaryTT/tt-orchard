## What

## Evidence
Tests run, mutation results, anything measured (say what was not run).

## Hardening checklist
Answer each item with `yes (test name)` or `n/a (one sentence)`. The items and their rules are in `docs/pr-hardening-checklist.md`.

1. Wrong type or shape in input: every parsed field is type-checked and a bad one is one refused item:
2. Files something else controls: reads and writes go through `safeio` (FIFO, link, oversize, growing file tested):
3. Check, then use: identity is captured before the operation and the result is bound to it:
4. Evidence an agent can write: trusted only through a ledger entry with its hash; deletion fails the check:
5. Verdicts and labels: each names its evidence; a missing input gives "not checked":
6. The check inspects the same object the action uses (tested in two setups where they differ):
7. Parsers and matchers: near-miss cases in both directions:
8. Failure paths: state is recoverable if the next step raises or the process is killed:
9. Numbers and sizes: bounds and 0, -1, 1, max, 10**20 tested:
10. Secrets, paths and host names are redacted before output:
11. Tests: skips name the feature they need, each guard has a mutation that turns a test red:
12. Housekeeping: change fragment, project log entry, README flags, `python3 scripts/version.py check`:
