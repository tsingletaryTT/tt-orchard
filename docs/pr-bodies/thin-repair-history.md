# Thin repair keeps what its earlier rounds did (issue 783)

Fixes #783.

## What

In the Aion-1.0-Instruct-2.3B bring-up (full port, `--thin-port`) the thin repair spliced `_rms_norm_layer` in three rounds in a row. Each round
fixed the error it was shown (a `RuntimeError`, a `TypeError`, an `AttributeError`) and the function still treated the port's model object as a
dict. A round's request held the current function and the latest error and nothing from the rounds before it. Issue 698 fixed which function is
blamed. This change is about what the request holds and when to stop.

- Each spliced round is recorded in `stages/2/thin_repair_history.json`: the function, the error the round was asked to fix, the failing line and
  message, and the diff of the function. The newest six records are kept.
- A request after the first carries the last four records in a section of at most 9,000 characters (each diff is cut at 2,500). The oldest record
  is left out until the section fits.
- When the last two records are consecutive rounds on one function and their errors differ, a third round on that function sends no request. It
  returns `fallback` with a reason that names the function and the errors, and a note for the agent's repair step: read the function together
  with the functions that call it and correct them as one unit. The same holds for every later round on that function. A different function is
  still repaired by request. Rounds with the same error keep the old behaviour.
- The ledger decision `thin repair round` carries `diff_sha256`. `thin repair handed to the coder` carries the stop reason.

## Evidence

- `tests/test_thin_repair_history.py`: 30 tests (unit tests with a fake endpoint, and two supervisor wiring tests with the stage machine, the check
  template and the ledger real). Each new test was seen to fail before the code existed (9 of the first 14 failed; the other 5 pin behaviour that
  must not change).
- Mutations (one guard removed each, run against the new file): 18, all red. Three survived at first (history field types, the file size cap, the
  value of the ledger hash) and got tests.

  | # | Mutation | Test that went red |
  |---|---|---|
  | M1 | request leaves out the history section | `test_the_next_request_carries_the_earlier_error_line_and_diff` |
  | M2 | history not saved after a splice | `test_a_spliced_round_records_the_error_the_failing_line_and_the_diff` |
  | M3 | load does not cut a diff | `test_history_in_a_request_is_bounded` |
  | M4 | recording does not cut a long diff | `test_a_long_diff_is_cut_when_it_is_recorded` |
  | M5 | section not fitted to its cap | `test_history_in_a_request_is_bounded` |
  | M6 | stop rule never fires | `test_a_third_round_on_the_same_function_after_a_changed_error_is_not_sent` |
  | M7 | stop ignores whether the error changed | `test_two_rounds_with_the_same_error_keep_the_old_behaviour` |
  | M8 | stop ignores which function | `test_a_different_function_is_still_repaired_after_the_stop` |
  | M9 | stop ignores consecutive rounds | `test_rounds_that_are_not_consecutive_do_not_count_as_two_in_a_row` |
  | M10 | stop after three rounds | `test_a_third_round_on_the_same_function_after_a_changed_error_is_not_sent` |
  | M11 | history file not validated | `test_a_malformed_history_file_counts_as_no_history` |
  | M12 | field types not checked | `test_one_record_with_a_wrong_field_voids_the_whole_file[code-5]` |
  | M13 | file size not capped | `test_a_valid_history_file_above_the_size_cap_counts_as_no_history` |
  | M14 | callers not named | `test_a_third_round_on_the_same_function_after_a_changed_error_is_not_sent` |
  | M15 | supervisor drops the note | `test_a_third_round_on_one_function_after_two_changed_errors_goes_to_the_agent_loop_with_a_note` |
  | M16 | supervisor drops the diff hash | same wiring test |
  | M17 | supervisor keeps a stale note | `test_the_note_of_one_round_does_not_reach_the_next_round` |
  | M18 | stop reason left out of the ledger | `test_a_third_round_on_the_same_function_after_a_changed_error_is_not_sent` |

- Neighbouring files run together: the thin repair and thin port files, `test_supervisor_port_repair.py`, `test_cli.py`, `test_bringup_config.py`,
  `test_cpu_run_before_board.py`, `test_lazy_coder_start.py`, the hardening and finite ratchets, `test_readme.py` and the version tests: 727 passed,
  2 skipped. `tests/test_hostile_readers.py -k thin_repair`: 42 passed. `python3 scripts/version.py check` is ok.
- Not run: the full suite (the operator runs it once on the integration branch). No chip run. Whether the coder repairs the Aion port once it gets
  the note is not shown by this change.

## Hardening checklist

1. **Wrong type or shape in input:** yes. `load_history` checks every field of every record (types, identifier and error-type patterns, whole
   numbers in range, bool refused); one bad record voids the file and the round works as before
   (`test_one_record_with_a_wrong_field_voids_the_whole_file`, 9 cases; `test_a_malformed_history_file_counts_as_no_history`).
2. **Files something else controls:** yes. Read with `safeio.read_json` (size cap 1 MiB), written with `safeio.atomic_write`; both are in the
   `PATH_READERS` registry already (`safeio.read_json(path)`, `safeio.atomic_write(path)`). `load_history` is in the hostile-input corpus
   (`tests/hostile_inputs.py`, 42 cases pass). No raw `open(` or `.read_text(` was added (`test_hardening_ratchet.py` is green).
3. **Check, then use:** n/a. A round reads the port and the history once at its start and writes the port and then the history; the history is
   advice to the coder and decides nothing about a verdict.
4. **Evidence an agent can write:** yes. The history file sits in a directory the agent can write. It carries no verdict. The ledger holds
   `diff_sha256` of each round's diff, so a rewritten file can be told from the file the supervisor wrote. A deleted or rewritten file means
   fewer notes or a different request; the check, the gate and the board test do not read it. The size and field caps bound what it can put in a
   request.
5. **Verdicts and labels:** yes. The stop reason names the function, both rounds and the three errors it rests on; the note names the callers it
   found, or says that no function calls the function. A missing or unreadable history gives no stop and no section.
6. **Same object checked and acted on:** yes. The stop rule compares the function the current failure names (after the generated-helper
   attribution of issue 698) with the functions in the records; `test_a_different_function_is_still_repaired_after_the_stop` and
   `test_two_rounds_on_different_functions_do_not_stop_a_third` are the second setup.
7. **Parsers and matchers:** yes. Stop: same error twice, one earlier round, a different function, non-consecutive rounds, two functions
   alternating, and the stop holding for later rounds.
8. **Failure paths:** yes. The port is written before the history; a history write that fails loses the advice and nothing else
   (`save_history` catches `OSError`). The note is cleared at the start of each round
   (`test_the_note_of_one_round_does_not_reach_the_next_round`). A crash between the two writes leaves a port with no record, which counts as no history.
9. **Numbers and sizes:** yes. Round and line numbers must be whole numbers between 0 and 1,000,000 (`10**9`, `-1`, `True` refused); a file over
   1 MiB or with more than 24 records is refused; fields, diffs and the section are capped
   (`test_history_in_a_request_is_bounded`, `test_a_long_diff_is_cut_when_it_is_recorded`).
10. **Secrets, paths and host names:** n/a. The history holds code lines, error messages and diffs of the port, the same text the check report
    already sends. The ledger gets a hash and no text. The README and log text hold no path or host name.
11. **Tests:** yes. No skips. Each guard has a mutation (table above). The wiring tests assert the request body the endpoint received and the text
    of the agent step's request.
12. **Housekeeping:** yes. `changes/thin-repair-history.toml`; a dated entry at the end of `docs/project-log.md`; no new flag (the `--thin-port`
    row in the README describes the history and the stop); `python3 scripts/version.py check` is ok.
