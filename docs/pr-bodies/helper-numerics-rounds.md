# A port that runs but disagrees is a failed round (run16)

Draft body for the pull request (no pull request is open yet). Branch `helper-numerics-rounds`, off the local run commit `3ae2f924` (the branch
`helper-drives-port`).

## What it does

Run16 of the tiny Qwen4 rehearsal produced the first port that loaded, ran prefill and decode on the chips and wrote 32 ids, with top1_agreement
0.5625 against the CPU reference. The check exits 0 when it runs to the end, so the helper-driven repair loop read it as a pass and stopped; the gate
then failed the attempt on the agreement and the coder took over. In the helper-driven loop a check that ran to the end with an agreement below the
gate's bar (`full_port.TOP1_MIN`) is now a failed round with its own request, `numerics`. The next ask carries a report the supervisor builds from the
check's record: the agreement, where the mismatches begin (position 0 is a prefill problem, later positions a decode problem), a table of the first 16
mismatching positions with the reference id and the id the port produced, the previous round's agreement, and the port's hash. The request tells the
helper the model is a random-weight rehearsal checkpoint, that the check is teacher forced, and lists the usual suspects without prescribing a fix.
The best port ranks by the request reached and then by agreement, and an answer is helped only when the agreement rises or the check passes.

## The twelve items of docs/pr-hardening-checklist.md

1. **Wrong type or shape in input.** `numerics_state` accepts a record only when the test record is a JSON object with exit code exactly 0 (an int) and
   `timed_out` false, the check's record has the shipped schema, the id lists are lists of whole numbers (not bools) of one non-zero length, and
   `ref` equals the stage 1 reference's first ids. Anything else is `unreadable` (tests: exit code 2, timed out, schema, short and long id lists,
   changed reference ids, no record).
2. **Files that something else controls.** Both JSON files and the port file are read with `safeio.read_json` and `safeio.read_bytes` (bounded by
   `full_port.JSON_MAX` and `SOURCE_MAX`). No raw `open(` or `.read_text(` was added to `orchard/` (`tests/test_hardening_ratchet.py`).
3. **Check, then use.** The record's reference and prompt entries are checked against the stage 1 files and their current sha256
   (`full_port._reference_ids`, the gate's own function) before any number is used, and the record must list the port file that is on disk now.
4. **Evidence that an agent can write.** The record lives under `evidence/`, where the coder's step could write. The agreement is therefore recomputed
   from the id lists and never read from the record's `top1_agreement` (test: a record that claims 1.0 still gives 0.5312); the lists must be the
   stage 1 reference's prefix and the record must be about the port that is on disk. The helper's reply is data, as before.
5. **A verdict stronger than its evidence.** `helped` is true only when the recomputed agreement rises or the check passes the bar; "ran to the end" is
   its own outcome (a crash that now runs) and the ledger names the agreement before and after. A port that prints its own agreement changes nothing
   (test).
6. **The check looks at a different object than the action uses.** The record is tied to the port by sha256: `port_sources` must name the file as it is
   on disk (test: edited after the check ran gives `unreadable`). The best port's rank and agreement are recorded for the port the check ran.
7. **A parser or matcher that is too broad or too narrow.** The failure line is built from the numbers only, so two rounds with the same agreement and
   the same first mismatch are "the same failure" for the existing hand-back rule, and any change in either is not (tests). Unknown ops still rank 0.
8. **Failure paths that leave state behind.** An unreadable record leaves no round and no pass; the ask is written before the call as before; a coder
   step that follows a hand back gets the same report in its text.
9. **Numbers and sizes.** The table is capped at 16 rows (`NUMERIC_ROWS`) and the report at 4,000 characters (`NUMERIC_REPORT_CAP`); rounds (10), asks (16)
   and the telegraph caps are unchanged. Agreement is a fraction checked to lie between 0 and 1 before it counts as a rise.
10. **Secrets, paths and host names.** The report holds ids, positions, counts and a hash; it goes through `port_repair.safe_text` like the rest of the
    request. Nothing in it comes from the port's own words.
11. **Tests that do not test the claim.** 31 tests in `tests/test_helper_numerics_rounds.py` with the shipped check template run against the fake ttnn;
    31 mutations, one per guard, restored in a `finally` and on SIGINT and SIGTERM with the tree clean at the start (script outside the repository:
    `~/orchard-runs/tmp/mutate_numerics.py`). The first run left two not red (a length check no test covered, and one mutation text that did not match);
    a test for a record with one id too many and a corrected mutation made both red.
12. **Housekeeping.** Change fragment `changes/helper-numerics-rounds.toml` (target 0.9.120), a project-log entry, a README sentence on the
    `--helper-drives-port` row. One older test changed (the exact dict of `unreported`). Not run: the full suite, any chip, any real helper call.

## Found while testing

Python reuses a cached `.pyc` when the source's size and its modification time in whole seconds are unchanged, so two ports of the same size written
within one second run as the first. A real round takes tens of seconds, so the loop is not affected; the fixtures use ports of different sizes.

## Not done

No run on chips. Whether the helper can close a 0.56 agreement on the tiny Qwen4 port within ten rounds is what the next run shows.
