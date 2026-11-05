# The outside helper inside the port repair rounds (issue 427, run11)

Pull request body. Branch `stack-5-repair-help`, link 5 of the stack, on `stack-4-chain`. Version fragment target 0.9.120.

## What it does

In a full-port stage 2 the supervisor now asks the helper (Telegraph) for the whole corrected `port_model.py` inside the port repair rounds, not
only at the start of an attempt. It asks at the start of a round when it is the attempt's first round or the failure is no further along the
request chain (init, load, prefill, decode) than the round before, and after the coder's repair step when that step ended in a watchdog pause or
escalation or on capped replies. At most one ask per round and six per stage counting the asks at attempt starts. A reply becomes `port_model.py`
only when it is a whole port and the file is still the one that was sent (or the coder's step failed). Each answer gets a verdict by progress
along the chain. The helper's problem id is one per run. Run11 of the tiny Qwen4 rehearsal is the evidence: a placed draft ran on the chips and
failed at prefill, the coder fixed it, the check failed at load, the coder looped and the two asks of the stage were spent.

## The twelve items of docs/pr-hardening-checklist.md

1. **Wrong type or shape in input.** The helper's reply is text from outside. `extract_draft` takes a fenced block or a parsing reply; `parses`
   and `draft_problems` read it with `ast` (a `SyntaxError`, `ValueError`, `RecursionError` or `MemoryError` is a refusal, tests cover a reply with
   no block and one that does not parse). The failure ops come from the check's own line through `port_repair.parse_failure`; an op that is not in
   `OPS` has no rank and is never progress (`test_progress_is_a_failure_later...`).
2. **Files that something else controls.** The port file is read with `safeio.read_bytes` after a link and regular-file check and written with
   `safeio.atomic_write`; a link or missing file is never written through (`test_a_link_in_place_of_the_port_file...`). The reply is saved with
   `write_draft` (mode 0600, never replacing a file, refusing a link in place of `telegraph/`). No raw `open(` or `.read_text(` was added to
   `orchard/` (`tests/test_hardening_ratchet.py` is green).
3. **Check, then use.** The port's sha256 is taken when the request is built and written on the ask; `place` reads the file again and compares.
   A coder edit made while the helper answered is not overwritten (`test_a_port_the_coder_changed_while_the_helper_answered...`). With a failed
   coder step the file may be replaced and both hashes are recorded.
4. **Evidence that an agent can write.** The coder can write `stages/2/port/` and `stages/2/telegraph/`. The helper's reply is data in a ledger
   entry with its path and sha256 (`evidence_record`); the verdict comes from the supervisor's own test record (`_repair_outcome` reads the marker
   and `port_repair.read_failure`, which checks the output's hash), never from a file the coder wrote.
5. **A verdict stronger than its evidence.** "Helped" means the check passed or now fails later in the request chain; a different message at the
   same request is not progress and a stub is not helped. The ledger names both requests. The verdict does not say the port is right.
6. **The check looks at a different object than the action uses.** The sha256 on the ask is of the file that is sent; the file that is replaced
   is checked against it; the file the check then runs is the one placed. The completeness check reads the reply that is placed.
7. **A parser or matcher that is too broad or too narrow.** `undefined_names` is scope-blind by design (a name bound anywhere counts), so it has no
   false alarms on ordinary code, and a star import is not judged (tests for comprehension variables, `with ... as`, `except ... as`, arguments and
   star imports). Signatures allow extra defaulted arguments and `*args`.
8. **Failure paths that leave state behind.** The ask is written before the call, so a crash cannot ask twice; the limits come from the ledger. A
   refusal, an empty reply, a reply that does not parse, a write error and a helper that raises are each a ledger decision and the round goes on
   (tests for no key, no code, a port over the size cap). An answer with no verdict at the end of an attempt gets the attempt's outcome.
9. **Numbers and sizes.** The port sent is capped at 90,000 characters and read at most 400,000 bytes; a larger port is not sent. The helper's
   own caps (calls, tokens, cost, per problem) apply and a refusal is a decision. `TOTAL_PER_STAGE` is 6 and `PER_ROUND` 1.
10. **Secrets, paths and host names.** The port and the failure go through `port_repair.safe_text` before the helper's own redaction (a test with a
    home path in the port). The ledger holds hashes and run-relative paths. No machine path in the skills or templates (`tests/test_paths.py`).
11. **Tests that do not test the claim.** 32 mutations, one per guard, each restored in a `finally` block and on SIGINT/SIGTERM
    (`scratchpad/mutate_tgrepair.py`), all red in the end. The first run had one green (a first-round clause that the previous-round lookup already
    covered); the two paths became one. The scenario tests run the real template against a fake `ttnn` and a real `Telegraph` with a recording
    transport; no clock decides a result.
12. **Housekeeping.** Change fragment `changes/telegraph-in-repair.toml` (target 0.9.120); dated project-log entry; README flag row
    `--no-repair-help` and a README section. The per-problem cap note: the helper counts a problem across its whole ledger, so the one shared id
    had five of its six calls used by three runs; the id is now per run.

## Tests

- `tests/test_repair_help.py` (36), `tests/test_supervisor_telegraph_in_repair.py` (21).
- Older tests changed: `tests/test_supervisor_phone_a_friend.py` and `tests/test_phone_a_friend_ladder.py` (the attempt-start asks, round asks off; per-run problem id).
- Not run: the full suite, any chip, any real helper call.
