# The helper drives stage 2 of a rehearsal (run12)

Draft body for the pull request (no pull request is open yet). Branch `helper-drives-port`, off the local run commit `21d30539`.

## What it does

Run12 of the tiny Qwen4 rehearsal reached prefill on the chips with a helper repair and then lost that port when the attempt ended, spent most of its
wall time on a coder that looped after every placed file, and was served a stub from the helper's cache a second time at no cost. Three changes:
(A) the furthest port an attempt reached is kept in `stages/2.best/port_model.py` (the ledger has its op, rank and sha256) and put back at the start of
the next attempt; (B) an answer reported as not helpful is not served from the cache again; (C) in a rehearsal (or with `--helper-drives-port`) the
repair rounds of stage 2 are driven by the helper: each round sends it the whole port, the failure and what the previous round changed, places its
reply and runs the supervisor's check, with no coder step, until the check passes, a cap is reached, or two rounds in a row fail at the same request
with the same message (the coder then repairs that round). `--no-helper-drives-port` keeps the coder in charge.

## The twelve items of docs/pr-hardening-checklist.md

1. **Wrong type or shape in input.** The helper's reply is checked as in the repair help (`extract_draft`, `parses`, `reply_problems`). The ledger
   fields read back (`rank`, `port_sha256`, `round`, `op`, `failure`) are type-checked in `best_state` (a string rank is ignored, tested) and
   `same_failure_twice` (an empty message never counts as the same).
2. **Files that something else controls.** The best copy is read with `safeio.read_bytes` after a link and regular-file test and written with
   `safeio.atomic_write`; a link in its place is refused (`test_a_link_in_place_of_the_best_copy_is_refused`). The port file goes through the existing
   `read_port` and `place`. No raw `open(` or `.read_text(` was added to `orchard/` (`tests/test_hardening_ratchet.py`).
3. **Check, then use.** `restore_best` reads the port and compares it with the skeleton, reads the copy and compares its hash with the ledger's, and only
   then writes; a coder-written file is never replaced. A driven round places the reply through `place` with the hash that was sent.
4. **Evidence that an agent can write.** The best copy is outside `stages/2`, so the coder cannot write it; its hash is in the ledger and is checked on
   restore. The helper's reply is data with its path and sha256. The check, its template and its record stay the supervisor's.
5. **A verdict stronger than its evidence.** The best port records the op the supervisor's own check reported (`read_failure`, which verifies the
   output's hash). A reply is not called helpful here; the existing verdict by progress stands. The ledger entry `helper-driven round` states what was done and makes no claim that it worked.
6. **The check looks at a different object than the action uses.** The hash on the ask, on `placed`, on `best port recorded` and on the restore are of the
   file the check ran, read in the same step; `_record_best` runs on the port as tested.
7. **A parser or matcher that is too broad or too narrow.** `same_failure_twice` needs the same op and the same non-empty message in rounds of the same
   attempt (a round of an earlier attempt does not count, tested); an unknown op ranks 0 for the best port. `change_summary` survives text that does
   not parse.
8. **Failure paths that leave state behind.** The ask is written before the call (existing). A failed copy, an unreadable file or a mismatch leaves the
   ledger entry `not placed` with the reason and the stage goes on. The best copy is replaced atomically.
9. **Numbers and sizes.** Rounds are bounded by `HELPER_ROUNDS` (10) and asks by `HELPER_TOTAL` (16) and by the configured telegraph caps; the previous-round
   section is cut at 1,500 characters; the port sent is still capped at 90,000 characters.
10. **Secrets, paths and host names.** The previous-round section and the failure go through `port_repair.safe_text` like the rest of the request. The
   change summary holds counts and function names only.
11. **Tests that do not test the claim.** 27 tests; 25 mutations, one per guard, each red on the first run, restored in a `finally` and on SIGINT and
   SIGTERM, tree clean at the start (`scratchpad/mutate_hdrive.py`).
12. **Housekeeping.** Change fragment `changes/helper-drives-port.toml` (target 0.9.119), a project-log entry, README rows for the two switches. Not run:
   the full suite, any chip, any real helper call.

## Not done

The keeping of the best port applies only to a helper-driven stage 2. `bringup` has no flag for the switches; a rehearsal turns the default on.
