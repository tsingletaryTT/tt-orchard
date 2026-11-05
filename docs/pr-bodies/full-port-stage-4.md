# Stage 4 for a full port: the port on each required chip count (issue 454)

Draft body for the pull request (no pull request is open yet). Branch `full-port-stage-4`, off the local `integration` branch.

## What it does

Stage 4 of the full-port route runs the port on each chip count in `--required-chips` that stage 2 and stage 3 did not already show. The
supervisor writes `stages/4/tests/plan.json` from the option, and each count has its own directory `stages/4/configs/<N>/` with its copy of the
shipped `port_check.py`, its config, its port (seeded from the port that passed before), its evidence, its marker and its `result.json`. A count
that stage 3 or stage 2 ran on is recorded as covered (`covered.json`) and runs no test. The gate has a second form that takes an entry
directory (`full_port.check(..., entry=N)`). The pause for the unbuilt stages moves to before stage 5. New skill `full-port-shrink`, README and
design-spec text (section 3.7), a project-log entry and the change fragment `changes/full-port-stage-4.toml`.

Not in this change: stages 5 and 6 (issue 329), any run on chips, and the `stage4-missing-config` branch (weights-only config drafting).

## The twelve items of docs/pr-hardening-checklist.md

1. **Wrong type or shape in input.** The counts come from the operator's option and are checked as whole numbers of at least 1, bounded by what
   stage 4 can lease and by 16 entries (`plan_entries`, `check_stage`); tests cover `0`, `True`, `"4"`, `None`, `()`, `10**20` and `2**63`. The
   plan, `covered.json`, the markers and the records are read through `safeio.read_json_checked` with `expect=dict`, and every field is
   type-checked before use (`load_plan`, `marker_problem`, `_covered_reasons`). A plan entry of the wrong shape, order or script is refused as a
   whole (`test_a_plan_file_that_is_not_the_supervisors_shape_is_not_used`).
2. **Files that something else controls.** Reads and writes go through `safeio` (`read_json_checked`, `read_bytes`, `atomic_write`) and
   `full_port.file_record`, which returns None for a FIFO, a link out of the run, or an oversize file. The port seed is the stage 3 seed
   (descriptor-anchored scan, regular files only, capped); stage 4 changed only the source and the destination names. The entry's note is copied
   to `stages/4/handoff.json` with `atomic_write`. No raw `open(` or `.read_text(` was added to `orchard/` (`tests/test_hardening_ratchet.py` is green).
3. **Check, then use.** The covering stage's marker and record are read once, their path and sha256 go into `covered.json`, and the gate and
   the ledger half compare the current bytes with those. The entry's marker is hashed into the ledger as the supervisor writes it; the entry's
   `result.json` is hashed when derived. The plan's sha256 is in the ledger before any entry starts.
4. **Evidence that an agent can write.** The coder can write inside an entry directory. Every file the gate trusts is vouched for by a ledger entry
   with its path and sha256 (`full_port_stage4.vouch`): the plan, each marker, each `covered.json`, each entry result and the stage result. A
   missing or changed file fails; it does not skip. On a resume, a marker or a `covered.json` that the ledger does not hold is deleted and the entry
   runs (tests `test_a_marker_the_ledger_does_not_hold...` and `test_a_covered_file_the_ledger_does_not_hold...`). Results written during prepare are set aside.
5. **A verdict stronger than its evidence.** A covered entry names the stage, and the ledger must show that stage passed with a marker the
   supervisor recorded and exactly N chips. The stage result is rebuilt from the records by the gate, so a number the supervisor or an agent wrote
   that differs is refused. The skill and the spec say what is not shown (speed, how the port splits its weights, a covered entry beyond the
   covering record). `coherent` is the only field the rehearsal option waives, and the ledger says so.
6. **The check looks at a different object than the action uses.** The check reads `stages/4/configs/<N>/` and the supervisor runs
   `python3 -P -s stages/4/configs/<N>/port_check.py` from that directory; the marker's command, the record's chips, the port sources and the
   config path are all compared with that entry. The test asks for the first N chips of its lease, and the record and the marker must list N chip
   names. Tests: a record for two chips in the three-chip entry; a source in stage 3's directory; a source in another entry's directory. The
   template found the run directory two levels above itself; it now takes the nearest `stages` above itself, and a test runs that line for stages 2 and 4.
7. **A parser or matcher that is too broad or too narrow.** `entry=` is accepted only with stage 4 and a whole count (a `True` is refused). A
   port source must start with `stages/4/configs/<N>/port/`, so the entry for 2 does not accept a source under the entry for 20. A covering
   stage must be 3 or 2 as an `int`.
8. **Failure paths that leave state behind.** The supervisor takes no lease itself in the new code. Leases, the park, the sweep and the restore
   are `_run_listed`'s, which the weights-only stage 4 already tests. A kill between steps is tested: after the plan, after the prepare step, in the
   middle of the first entry's test, after the first entry's result. `prepared.json` is written after the prepare step ends, so a kill during it
   runs the step again. A kill between a marker and its ledger entry runs that entry's test again.
9. **Numbers and sizes.** Counts of 0, -1, `True`, `10**20` and `2**63` are refused or named as a missing entry; the plan holds at most 16 entries
   and its deadlines are checked against the stage budget. An entry that needs more boards than the box has blocks before any entry runs.
10. **Secrets, paths and host names.** The ledger entries added hold run-relative paths and hashes. The seed's problem text has no absolute
    path (the stage 3 rule, kept). The skill has no machine path (`tests/test_paths.py` is green).
11. **Tests that do not test the claim.** 26 mutations, one per guard, each restored in a `finally` block, each turning a test red (table below).
    One mutation (the prepare step on resume) first showed no red test, and a test for it was added. The scenario tests run the real supervisor,
    the real template and the real park path against a fake `ttnn`; no clock decides a result.
12. **Housekeeping.** Change fragment `changes/full-port-stage-4.toml` (target 0.9.115, the default next patch; another branch may claim the
    same target and one of the two needs another). Dated project-log entry at the end of `docs/project-log.md`. No new command-line flag, so no
    README flag row; the `full-port` class row and the `--full-port` help now describe stage 4. `scripts/version.py check` is green.

## Tests

- `tests/test_full_port_entries.py` (45): the gate, the ledger half, the plan, the entry form of the check, routing, the skill text.
- `tests/test_supervisor_full_port_stage4.py` (18): a pass with two real counts (1 and 3) and one covered (4); the plan; the skill; the seed; the
  prepare task; the fixed command; the park and restore of the coder for the three-chip entry; a failing count ends the stage and the next entry
  never starts; no repair rounds; a resume between entries, after the prepare step, with a forged marker and with a forged covered file; a count
  covered by stage 3 and one covered by stage 2; a run with no counts; the rehearsal option and its absence.
- Changed older tests: the pause before stage 5 (`test_supervisor_full_port.py`, `test_supervisor_full_port_stage3.py`), the opt-in test
  (`test_full_port.py`), the skill list (`test_skills.py`).
- Targeted runs only, under the memory-limited wrapper. The full suite was not run.

## Mutations (each restored; each turned a test red)

record chips equal to the entry's count; port sources under the entry directory; template copy is the shipped file; marker command is the
entry's; plan lists exactly the required counts; stage result equals the rebuilt summary; covered entry's covering marker shows N chips; covered
entry's file hashes match; covered entry has no marker; ledger holds each marker; ledger holds the plan; ledger holds `covered.json`; ledger shows
the covering stage passed (gate and covering rule); covering marker is the supervisor's; pause before stage 5; skip with no counts; rehearsal
flag reaches the gate; seed from the stage that passed; a failed entry ends the stage; a marker the ledger lacks is not trusted; a covered file
the ledger lacks is not trusted; a finished entry is not run again; the prepare step is not run again; the entry's note reaches the park; the
entry's marker goes to the entry directory.

## What could not be tested without hardware

A run on chips: the real mesh open for one and three chips, the fabric reset behaviour after a one-chip test, the park and restore of the real
coder container for a wide entry, and whether a real coder adapts a port to a smaller mesh. The rehearsal run on the tiny Qwen4 checkpoint
(`--full-port-rehearsal`) is the check the issue names and is not done here. The lab path (`_run_listed_on_lab`) is the weights-only list's code and has no stage 4 test.
