# A project database for tt-orchard: what moves, what gets indexed, what stays a file

Status: the telegraph part is implemented (section 14 says how, and where the implementation differs from this note). The note was
written as a design with a proof of the riskiest claim: `orchard/_proofs/telegraph_sqlite.py` with
`tests/test_proof_telegraph_sqlite.py` and `tests/test_proof_telegraph_sqlite_invariants.py`; the product does not import the proof. The
speed baseline (section 1, row 3), the virtual spend index and the runs index are not built.
Date: 2026-10-28. Version of the repository when written: 0.9.37.

The operator's question: "if a little sqlite DB inside this project helps everything run smoothly THEN BY ALL MEANS let's do it."

## 0. Invariants

This note is the specification for the real migration. Each invariant below came out of the Copilot review of PR 132 (one
review pass found 7 problems in the first version of this note) and is a requirement. Each has a named test in the proof
(`tests/test_proof_telegraph_sqlite_invariants.py`, test names start with the invariant), and each test was seen to fail
when its guard was removed (section 9). A change to the real migration that drops an invariant has to say so here.

| # | Invariant | Named tests in the proof |
|---|---|---|
| I1 | `scope` is part of every key and every lookup. A cached answer, a call row, a cap and a total belong to one scope; an answer cached for one scope is never served to another. | `test_i1_a_cached_answer_is_served_only_to_its_own_scope`, `test_i1_the_same_request_in_two_scopes_keeps_two_cache_rows`, `test_i1_caps_and_totals_are_counted_per_scope` |
| I2 | After any event that can lose history (a restore from a backup, a database moved aside, a new file next to existing backups) the ledger is unconfirmed: every reservation is refused until a person confirms. Reply files and an export can add conservative rows; they never confirm. | `test_i2_after_a_restore_from_an_older_backup_sending_is_refused_until_a_person_confirms`, `test_i2_a_moved_aside_database_starts_unconfirmed`, `test_i2_a_new_file_beside_existing_backups_starts_unconfirmed`, `test_i2_a_first_ever_open_is_not_unconfirmed`, `test_i2_the_unconfirmed_state_survives_reopening_and_is_cleared_only_by_confirming`, `test_i2_replies_without_a_row_are_counted_at_their_worst_case_and_do_not_confirm` |
| I3 | After the migration an old writer cannot recreate `telegraph.jsonl` or build a second history, and new code refuses a directory that holds both the migration marker and a live `telegraph.jsonl`. The cutover is drained: it takes the old lock, compares totals inside the import transaction, and rolls back on any difference. | `test_i3_after_migration_an_old_telegraph_refuses_every_call_and_sends_nothing`, `test_i3_the_settlement_of_a_request_in_flight_during_the_cutover_cannot_recreate_the_ledger`, `test_i3_new_code_refuses_a_directory_that_holds_the_marker_and_a_live_jsonl`, `test_i3_a_migration_whose_totals_differ_changes_nothing`, `test_i3_migrating_twice_is_refused` |
| I4 | The state directory is a real directory owned by the user and mode 0700; a loose mode on our own directory is tightened, a link or another user's directory is refused and left alone, the database file is not a link, and creating the directory is safe when two processes race (including the first switch to WAL). | `test_i4_an_existing_directory_with_loose_permissions_is_tightened`, `test_i4_a_directory_that_is_a_link_is_refused`, `test_i4_a_directory_owned_by_someone_else_is_refused_and_left_alone`, `test_i4_a_database_file_that_is_a_link_is_refused`, `test_i4_losing_the_race_to_create_the_directory_is_not_an_error`, `test_i4_four_processes_opening_a_missing_directory_at_once_all_succeed`, `test_i4_a_locked_answer_to_the_wal_switch_is_retried_until_the_timeout` |
| I5 | Only confirmed corruption moves a database aside: a failed `integrity_check`, or SQLite's own "malformed" or "not a database" error. Busy, locked, disk-full, I/O, read-only and cannot-open errors raise a retryable `LedgerUnavailable`, leave the file and its siblings untouched, and make telegraph refuse the call. A moved-aside file starts unconfirmed (I2). | `test_i5_a_lock_or_io_error_is_retryable_and_leaves_the_file_untouched` (6 messages at 3 places), `test_i5_confirmed_corruption_moves_the_file_aside_and_starts_unconfirmed` (3 messages at 2 places), `test_i5_an_integrity_check_that_fails_moves_the_file_aside`, `test_i5_a_write_that_finds_the_database_locked_raises_a_retryable_error_and_changes_nothing` |
| I6 | An offset index reuses a stored total only after validating the bytes it consumed: the tail chunk on every poll, every chunk on a full check; a half-written last line is not consumed. The residual (a same-size edit far behind the offset is seen only by a full check) is stated and tested. | `test_i6_an_appended_log_is_read_from_its_offset_and_the_total_is_kept`, `test_i6_a_half_written_last_line_is_not_consumed_until_it_is_whole`, `test_i6_a_truncate_and_regrow_that_keeps_the_start_is_detected`, `test_i6_a_same_size_edit_in_the_last_chunk_is_detected_by_an_ordinary_poll`, `test_i6_a_same_size_edit_in_an_early_chunk_is_missed_by_an_ordinary_poll_and_caught_by_a_full_check`, `test_i6_a_smaller_file_restarts_from_zero`, `test_i6_a_new_file_under_the_same_name_restarts_from_zero_even_when_it_is_larger` |

### Requirements for the real migration (second review pass)

The second review pass found four more problems that are about order of steps and crash recovery. They were recorded as
requirements and were not built in the proof. Each had a placeholder in the proof's test file, marked `xfail(strict=True)`. The migration
replaced each placeholder with a real test of the same name (see section 14). Where sections 8 and 10 describe a shorter sequence, the
step orders below take precedence.

Two mediums from the same pass were fixed in the proof: an `OperationalError` from a statement inside a transaction body
(disk full, I/O, read-only) is now translated to `LedgerUnavailable` after the rollback (I5), and the offset index
validates the whole last chunk when the offset is an exact multiple of the chunk size (I6). Tests:
`test_i5_an_error_inside_the_transaction_body_is_retryable_and_leaves_no_open_transaction`,
`test_i6_an_edit_in_the_last_chunk_is_detected_when_the_offset_is_an_exact_multiple_of_the_chunk`; each was seen red with its
guard removed.

**I7. Recovery state is on disk before a replacement database exists.**
Failure: the first version marked a moved-aside or recreated database unconfirmed after the replacement had been created and
migrated. A crash in between leaves a valid, empty file that the next open accepts, and another opener can reserve in the
gap, so spend made available again.
Rule: a durable marker file `RECOVERY-REQUIRED` (JSON: reason, time, the name the old file moved to) is the authority. It is
written, with `fsync` of the file and of the directory, before anything is moved or created. Every open takes an exclusive
`flock` on `open.lock` in the state directory for the whole open sequence, reads the marker first, and refuses every
reservation while the marker exists, whatever the database holds. The `unreconciled` row in the database is a mirror.
Confirming clears the row, then removes the marker (and syncs the directory); both steps are safe to repeat.

| Step | Action | On disk afterwards | Restart after a crash here |
|---|---|---|---|
| 1 | Take `open.lock`; decide recovery is needed (failed check, or file missing beside backups) | nothing changed | Repeat from 1 |
| 2 | Write `RECOVERY-REQUIRED`, `fsync` file and directory | marker | Marker found: refuse all reservations, go on to 3 |
| 3 | Rename the damaged file and its `-wal` and `-shm` to `.corrupt-*` | marker, no database | Marker found, no file: go on to 4 |
| 4 | Create the new file, migrate it | marker, valid empty database | Marker found: refuse; the empty database is never accepted on its own |
| 5 | Write the `unreconciled` row (mirror) | marker, database with row | Same |
| 6 | A person confirms: delete the row, then the marker | no marker | Crash between the two: marker remains, refusal continues, confirm again |

test: written with the migration (`test_I7_recovery_state_is_on_disk_before_a_replacement_database_exists_and_every_opener_honours_it`). It kills
the opener with SIGKILL after each step, opens again from a second process, and asserts that no reservation is granted before step 6.

**I8. A restore is staged, marked, and installed under exclusion.**
Failure: the first version copied the backup into place and marked it unconfirmed afterwards. A crash or a concurrent opener
between the two sees a valid file with no mark, and allowance spent after the backup is silently available.
Rule: a restore needs no other process to have the database open. Every opener holds a shared `flock` on `db.users` while its
connection is open; restore takes it exclusively without waiting and refuses ("close the other processes") if it cannot.
The backup is copied to a private staging file in the same directory, checked, and marked unconfirmed inside the staging
file, and only then published with one atomic `rename`. The marker file of I7 is written before anything else.

| Step | Action | On disk afterwards | Restart after a crash here |
|---|---|---|---|
| 1 | Take `db.users` exclusively (no waiting) | nothing changed | Repeat |
| 2 | Write `RECOVERY-REQUIRED` (reason: restore), `fsync` | marker | Refuse all reservations; go on |
| 3 | Copy the backup to `project.db.restore-<id>` (0600), `fsync`; run `integrity_check` on it; insert the `unreconciled` row into it | marker, complete staged file | Staged file complete: reuse it. Partial: delete it and copy again |
| 4 | Move `-wal`, `-shm`, then the live file, to `.replaced-*` as one group | marker, staged file, no live file | No live file: create nothing; go on to 5 |
| 5 | `rename` the staged file to `project.db`, `fsync` the directory | marker, restored file with the row | Restored file with row and marker: refuse until confirmed |
| 6 | Release `db.users` | | |

At no step is there a valid live file that is neither marked nor covered by the marker. The old live file and its WAL are kept
(never deleted) so a person can compare.
test: written with the migration (`test_I8_a_staged_restore_is_marked_before_installation_and_a_crash_at_any_step_leaves_reservations_refused`).

**I9. The migration recovers sidecars and torn fragments, or refuses.**
Failure: the migration and its equality check ignore `telegraph.jsonl.torn-*` sidecars with no note and a malformed
trailing fragment, so the cutover can succeed while dropping their conservative charges (the old code counts each as one call
of unknown cost before it applies caps).
Rule: under the old lock, before the import, run the old recovery: a last line with no line break is moved to a sidecar
(synced) and the file cut back; every sidecar without a `torn` note gets one (tokens = the configured `max_output_tokens`,
passed to the command). Then import, so every `torn` record is a row in `torn`. The equality check adds two comparisons: the
number of sidecar files equals the number of `torn` rows, and the accounting totals include them (the check already compares
`accounted_calls`, which counts torn rows). A malformed line that is not the last line is not recoverable: the migration
refuses and prints the line numbers, and a person can pass `--accept-malformed N` with the count they inspected.
test: written with the migration (`test_I9_orphaned_sidecars_and_malformed_fragments_are_recovered_or_block_the_migration`). It builds an orphan sidecar with
no note, a trailing fragment, and a malformed middle line, with the real old `Telegraph`.

**I10. The cutover is resumable at every step, and no phase lets a writer send into a second history.**
Failure: the first version did the import in a transaction, then renamed, then made the sentinel and the marker. A crash or a
`mkdir` failure after the rename leaves `telegraph.jsonl` absent and the lock released, so an old writer recreates an empty
file, while the database already says `migrated` and a retry refuses before repairing anything. The external marker was also
written last, so new code could not tell the half-finished state.
Rule: the order is below. The external marker `MIGRATING-TO-SQLITE` is written first. The database records the phase in
`meta` (`cutover:<scope>` = `imported`, then `archived`, then `sentinel`, then `complete`). New code refuses to reserve for a
scope whose phase is not `complete`, and refuses a directory that has either marker and a live `telegraph.jsonl` file. A
rerun of the command resumes from the recorded phase; it never refuses because an earlier phase finished. The old lock is held from step 1 to step 7, so an old writer
cannot append meanwhile.

| Step | Action | On disk afterwards | Restart after a crash here |
|---|---|---|---|
| 1 | Take the old `telegraph.lock` | nothing changed | Repeat |
| 2 | Write `MIGRATING-TO-SQLITE` (scope, sha256 of the ledger), `fsync` file and directory | marker; old ledger live | Resume at 3. Old code can still write the ledger (it has not seen a change); step 3 imports anything it added |
| 3 | Recover sidecars and fragments (I9); one transaction: import every record, compare totals, set phase `imported` and `imported_bytes` with the sha256 of that prefix | database has the history; phase `imported` | If the ledger grew since (an old writer ran after the crash): verify the stored prefix digest, import the suffix, compare again, then go on. If the prefix digest differs: refuse, a person decides |
| 4 | `rename` `telegraph.jsonl` to `telegraph.jsonl.archived`, `fsync` the directory; phase `archived` | no `telegraph.jsonl` | Resume at 5. Until it is done a regular file named `telegraph.jsonl` appearing here means an old writer got through: new code refuses (guard), and the person reconciles the late records |
| 5 | Make the directory `telegraph.jsonl` with its README; phase `sentinel` | sentinel in place | Resume at 6 |
| 6 | Write `MIGRATED-TO-SQLITE`, `fsync`; phase `complete`; remove `MIGRATING-TO-SQLITE` | complete | Crash before the remove: both markers; resume removes `MIGRATING-TO-SQLITE` |
| 7 | Release the old lock | | |

The only window the lock does not cover is a crash between steps 4 and 5, when no file of that name exists. The rule above
closes it from the new-code side (the guard refuses) and from the resume side (step 5 is repeated first). An old process that
creates the file in that window is detected and is not prevented; the check is the guard of step 4.
test: written with the migration (`test_I10_a_crash_at_any_step_of_the_cutover_resumes_and_no_phase_lets_a_writer_send_into_a_second_history`). It kills the
command with SIGKILL after each step, runs the real old `Telegraph` and the new one against the directory at each crash point, and resumes.

## 1. Answer and cost-benefit table

A SQLite file helps in three places and does not help in the others. The strongest case is the telegraph budget ledger. It
is the only store that holds money, several processes write to it, and its hand-written locking and recovery produced the
largest group of review findings. A SQLite version of its accounting gives the same totals and the same budget decisions
(section 9), survives four processes and a SIGKILL, and removes about 130 lines of file handling from `telegraph.py`.

| Candidate | Verdict | What it saves | What it costs | Order |
|---|---|---|---|---|
| Telegraph ledger, cache index, verdicts (authoritative) | Move to `project.db` | Closes the classes of finding about locks, torn tails, short writes, durability, unreadable ledger read as empty, cache published before its call is recorded (14 of 42 inline findings on PR 106). Removes `_ledger_lock`, `_quarantine_torn_tail`, `_append`, `_write`, the ledger parser (about 130 lines). One consistent snapshot for the web view (PR 111 had a finding on two reads seeing two snapshots). | The ledger stops being a text file: `grep` is replaced by `tt-orchard db export telegraph`. A schema and migrations to maintain. A backup to take (the file holds money records). The recovery story after a lost file is weaker than a text file in an editor (section 8). | 1 |
| Virtual spend scan (PR 115) | Index only, in `index.db` | A poll reads new bytes only: per file a stored offset and running totals replace the re-read of every log. Closes the "each poll walks and re-reads" cost and the finding about concurrent cache misses each scanning. | A derived table that must be invalidated correctly (inode, size, a shrink, a rewrite). It does not touch the symlink and traversal findings (section 11). | 2, after PR 115 merges |
| Coder-speed baseline (`speed-baseline.json`) | Move to `project.db` | `Baselines.update` reads the file, then replaces it, with no lock: two writers lose an update. One `INSERT ... ON CONFLICT DO UPDATE SET tok_s = MAX(...)` closes it. | One more table. Tiny. | 3 |
| Runs index for the web page and `status`, and the verdict peers | Index only, in `index.db`, only when a poll is measured to be slow | The per-process mtime cache in `webui.py` becomes one cache shared by the web server, `status` and `narrate`. A cold start no longer re-reads every run. | Live facts (pid alive, lock holder, gozer status) cannot be cached, so the index holds only what comes from ledgers and result files. The benefit is not measured; this machine has 14 run ledgers. | 4, optional |
| Benchmark numbers copied into READMEs and cards | Index only, later | A `measurements` table (model, config, metric, value, source file, sha256) lets a test check each README number against its source. | A writer for every bench output. No review finding in the history points at this today. | Later |
| Research caches (`orchard/research`) | Keep as files | Nothing. One JSON file per URL and range, a bad file is a cache miss, a run reads the same answers on resume. | | Leave |
| Kernel probe records (`attempt.json`, reports) | Keep as files | Nothing. These are agent-run evidence in the run directory and travel with it. | | Leave |
| Per-run result JSON, supervisor `ledger.jsonl`, `narrate.py` inputs | Keep as files | Nothing. They are the authoritative record of a run and a person reads them with a text tool. | | Leave |

Recommendation, in order:

1. Telegraph first. It has the largest finding history, the only money, and the proof is done.
2. The virtual spend offset index second, after PR 115 merges. The change is small if the scanner already returns per-file
   totals. It must be written as a cache that can be deleted.
3. The speed baseline third. It is small and the race is real.
4. A runs index only if a measured poll cost calls for it. Do not build it on the expectation of growth.
5. Leave research caches, kernel probe records, per-run results and the supervisor ledger as files.

The supervisor's `ledger.jsonl` stays out of scope. I found no strong reason to include it: it is read by a restart
that replays it, it has kill-at-every-event tests, and a person reads it during a run. The runs index would read it and
never write it.

One departure from "one SQLite file": use two files, `project.db` for authoritative data and `index.db` for derived data.
Section 5 gives the reason. Putting both in one file is possible; the cost is that every backup and every
corruption recovery has to separate the two kinds of table.

## 2. Evidence from the code

### 2.1 Telegraph (`orchard/telegraph.py`, 885 lines; `telegraph_view.py`, 509; `telegraph_cmd.py`, 199)

Holds today: `telegraph.jsonl` (one line per reservation, settled call, refusal, cache hit, report and quarantined torn
record), `telegraph.lock`, `cache/<sha256>.json` (answer plus the reply file name), `replies/<call_id>.txt` (the redacted
answer text), `telegraph.jsonl.torn-*` sidecar files.

Read and written: every `ask` reads the whole ledger (`_read_ledger_strict`), takes `flock` on `telegraph.lock`, repairs a
torn tail, reads again, applies the caps, appends the reservation with `fsync`, sends the request, appends the settlement,
writes the reply file, appends, then writes the cache entry. `report` appends. The web page and `usage` read the whole file.

Cost in review: PR 106 had 11 Copilot review passes and 42 inline comments. By my reading of each comment, 14 concern the file
mechanics: no lock across processes, a torn last line glued to the next record, writes silently dropped, a short
`os.write` treated as a reservation, `report()` ignoring a failed append, the cache visible before its call is recorded,
`0o600` applied only at creation, a read error treated as an empty history, no `fsync`, the truncate before the note is
durable, appends without the lock, a short write in the quarantine, lock errors escaping as `OSError`, and a flaky thread in a
test. The other 28 are validation of configuration, redaction, accounting rules and CLI wiring. PR 111 (the web view)
had 7 passes and 23 comments; about 8 concern reading the ledger: a read failure shown as zero spend, two reads of
different snapshots, NaN and infinity inside a ledger, a poll that rebuilds every row. `telegraph.py` has 20 commits and
`telegraph_view.py` 15.

Authoritative or derived: authoritative. A lost ledger is lost money information. The cache entries and the summary are derived.

Does a person need a text tool: sometimes. The operator greps the ledger and a backfill kept each source line verbatim.
`tt-orchard db export telegraph` has to produce the same lines.

Verdict: move. The reply files stay files (section 8 gives the reason).

### 2.2 Virtual spend (PR 115, `orchard/virtualspend.py`, 652 lines on branch `virtual-spend`)

Holds today: nothing. It walks run directories (up to 5000 files, depth 8, 200,000 entries), reads agent logs
(`log/*.jsonl`) and `requests.jsonl` with byte budgets, sums tokens, skips turns that came from the helper, and prices the
sum at a reference model's list rate. The view caches the result for a short time.

Cost in review: 6 Copilot passes and 18 inline comments, plus one from the operator. About 11 concern the walk and the
bounded reading: byte budgets, an over-long line, an entry budget still spent after it hits zero, `MAX_DEPTH` silently
dropped, `O_NOFOLLOW` covering only the last path component (three findings on symlink replacement), concurrent cache misses
each scanning. The remaining 7 concern totals, `NaN` and huge integers, and the CLI.

Authoritative or derived: derived. The logs are the record. Delete the index and a rescan rebuilds it.

Person needs a text tool: the logs, yes; the index, no.

Verdict: index only. A table `log_files(path, dev, inode, size, mtime_ns, offset, turns, prompt_tokens, completion_tokens,
skipped_helper, skipped_other, tail_digest, chunk_digests)` lets a poll read only the bytes after `offset`, up to the last
newline. A stored total is reused only after the bytes it was built from are validated (invariant I6). The first version of
this note used a hash of the first 4 KiB; the review showed that it cannot tell a same-size edit later in the file, or a
truncate and regrow that keeps the start, from an unchanged file. The contract is now:

* the index keeps a digest per 1 MiB chunk of the consumed prefix and a digest of the tail (from the start of the chunk that
  holds `offset` up to `offset`);
* every poll checks the inode and device, that the size is not below `offset`, and the tail digest. This reads at most one
  chunk and catches a truncate and regrow and an edit in the last chunk. Any failure restarts from zero;
* a full check re-hashes every chunk of the consumed prefix. It reads the whole prefix, so it runs on a slow schedule (the
  default is once a day per file and on `--rebuild`) and does not run on each poll;
* what a poll cannot see is an edit of the same size in an earlier chunk, between full checks. The index records when each
  file was last fully checked, the view says so, and no budget decision reads the index. The proof tests this residual
  as a documented miss;
* a half-written last line is not consumed (the offset stops at the last newline).

Agent logs are written by the harness and by agents. Where a log can be sealed (an attempt that has finished), sealing records
its full digest once and the index can then trust it. That is an option for later and is outside this design.

The walk over directories stays, so the symlink and depth findings stay (section 11).

### 2.3 Runs index for the web page and `status`

Holds today: nothing shared. `WebApp.list_runs` calls `status.collect` for each run on every poll. `WebApp._peers`
keeps a per-process dictionary keyed on the ledger's `(mtime_ns, size)`. `_compare` re-reads every ledger of the same model
on every run detail request with no cache. `status` (the command) and `narrate` start cold each time.

Cost in review: no review comment so far is about this cache. The project log has one entry for it (the peer cache added with
the verdict line). I did not measure a poll. With 14 run ledgers on this machine the cost is small; the claim is only that
the cost grows with the number of runs, and every process pays it separately.

Authoritative or derived: derived from `ledger.jsonl` and the result files. Rebuilt if lost.

Verdict: index only, when a measurement shows a poll over about 200 ms. The index would hold, per run: ledger mtime and size,
model, state parsed from the ledger, and the verdict inputs (`results_json`). Live facts (pid, lock, lease) are not stored.

### 2.4 Verdict peers and benchmark numbers

`orchard/verdict.py` does not read files itself; `webui.py` passes it the peers. So the peers are part of the runs index.
Benchmark outputs (`results_*.json`, `summary.json`, `bfcl-*.json`, `.control.json`) are files written by tools that run
for a long time and by shell scripts; READMEs and cards carry numbers copied from them by hand.

Verdict: keep the files. A `measurements` table that mirrors them is a later option, worth building when someone wants a test
that checks each README number against its source. Nothing in the review history asks for it yet.

### 2.5 Research caches (`orchard/research/http.py`)

Holds: one JSON file per URL and byte range under a cache directory, with the fetch time and the body. A bad or missing file
is a cache miss. The write is not atomic (`write_text`) and nothing locks; both are safe because a torn file fails to parse and
is a miss. The cache is the run's evidence ("a resume reads the same answers").

Verdict: keep. No review finding in the history is about its storage, and a database file would hide evidence a person reads.

### 2.6 Kernel probe records and the coder-speed baseline

Kernel probe: `attempt.json` and verdict files per attempt directory, written by `_write_new` and read by
`_read_regular` with `lstat` checks and `RECORD_MAX_BYTES`. They are evidence written next to what the agent produced.
Verdict: keep.

Speed baseline (`orchard/speedwatch.py`, `Baselines`): one JSON object keyed by `endpoint model` holding the best tok/s and a
date, read whole, changed in memory and swapped in with `os.replace`. Two processes that call `update` at once can each
read the old file and the later replace loses the earlier update (the larger of the two speeds may vanish). The file keeps
only the best value, so a lost update lowers a baseline that "only goes up". Authoritative in the weak sense (a person can
re-measure). Verdict: move; `INSERT ... ON CONFLICT (endpoint, model) DO UPDATE SET tok_s = MAX(tok_s, excluded.tok_s)`.

### 2.7 Other files that look like candidates

`skills-freshness.json` is a cache with an age limit: keep. The telegraph `cache/` directory moves with telegraph.
`narrate.py` follows ledgers and logs by reading them; it keeps reading them.

## 3. Location, files and permissions

* Directory: `~/.cache/tt-orchard/db/`, mode 0700, overridable with `[database] dir` in the bring-up config and with the
  environment variable `ORCHARD_DB_DIR`. The user-level state that exists today (the speed baseline, the telegraph directory,
  the skills cache) is already under `~/.cache/tt-orchard/`. A cache cleaner can delete this directory; backups live next to the
  file, so one sweep removes both. The operator may prefer `$XDG_STATE_HOME/tt-orchard/` for that reason. The code has one
  function that resolves the directory, so the choice is a one-line change.
* The path comes from configuration or `ORCHARD_DB_DIR`, never from `Path.home()` of the calling process. Agent shells run with
  a `HOME` inside the run directory; a `telegraph ask` that resolved `~` there would open a second database and the caps
  would not be shared.
* `project.db`: authoritative data (telegraph events and calls, cache index, speed baselines). Backed up.
* `index.db`: derived data (log offsets, runs index, measurements). Not backed up. Deleted on any integrity failure or schema
  mismatch and rebuilt on demand.
* The file is opened with `os.open(..., O_NOFOLLOW, 0o600)` and `fchmod`, so an older file with loose permissions is tightened
  and a link in its place is refused. SQLite gives `-wal` and `-shm` the permissions of the main file; the proof test checks
  all three are 0600.
* The directory is not trusted because `mkdir` was given a mode (invariant I4). Every open does this, in order: create the
  directory with `os.makedirs(..., exist_ok=True)` (two processes may both create it; losing the race is not an error); open it
  with `O_DIRECTORY | O_NOFOLLOW` (a link or a file in its place is `UnsafeDirectory`); `fstat` the descriptor; refuse if the
  owner is not the effective user (the directory is left as it is, since it is not ours to change); `fchmod` to 0700 when the mode
  has any group or other bit set. A directory that other users can write could let them replace the database and its
  sidecar files even though the files are 0600, so a loose mode on a directory we own is tightened and not tolerated. The
  same function is used for the backup directory. The checks run on a descriptor, so a swap of the path after the check does
  not change what was checked.
* The database must be on a local disk. WAL needs shared memory and does not work over a network file system. After opening, the code reads
  back `PRAGMA journal_mode`; if it is not `wal` the open fails and telegraph refuses calls (the same fail-closed rule as
  "the budget ledger cannot be written or read" today).

## 4. Pragmas, versioning and the access layer

On every connection, in this order:

| Pragma | Value | Reason |
|---|---|---|
| `journal_mode` | `WAL` | Readers (the web page) do not block the single writer. Persisted in the file. |
| `synchronous` | `FULL` | Each commit syncs the log. This matches `fsync` of the reservation today. `NORMAL` survives a killed process but can lose the last commits on a power cut. The test for this setting is a pragma read-back; a power cut was not tested. |
| `busy_timeout` | 30000 ms | A writer waits for another writer. A reservation or settlement holds the lock for microseconds; 30 s only matters if a process stalls inside a transaction. |
| `foreign_keys` | `ON` | Set per connection. |

Schema version: `PRAGMA user_version`. `orchard/db.py` holds an ordered list of migration functions; entry `i` takes the
file from version `i` to `i + 1`, inside one `BEGIN IMMEDIATE` together with the `user_version` update, and re-reads the version
after taking the lock so two processes cannot both migrate. A file whose version is higher than the list is refused with a
message that names both versions. Entries that have shipped are never edited. Before a migration runs, `project.db` is backed
up. Tables are `STRICT` (SQLite 3.37 or newer; the code checks `sqlite3.sqlite_version_info` at open and refuses an older one).
This machine has 3.50.4.

Access layer: `orchard/db.py` owns connections (one per thread, `isolation_level=None`, explicit transactions), pragmas,
migrations, `integrity_check`, backups, export and move-aside. Query code lives in `orchard/dbq/<store>.py`
(`telegraph.py`, `spend.py`, `speed.py`, `runs.py`). A repository test fails when a SQL keyword statement appears in any other
module. That test must be shown to fail by adding a statement to a scratch module.

Typed validation by the schema. `STRICT` tables and `CHECK` constraints reject some of what the review rounds on PR 111
found in jsonl: a token count below zero, a cost of `Infinity` or above 1e12, the string `"NaN"` in a REAL column, an integer
beyond 64 bits (the binding raises `OverflowError`). SQLite stores a Python `nan` as NULL, which for a cost means "unknown" and is counted
at the reservation. That is the conservative result, but it is a silent conversion and the writer should validate first. A
STRICT REAL column accepts the string `"-5"` and stores `-5.0`, so a numeric string is not caught by the schema. The
writer's own validation (the parallel typed-validation task) remains necessary.

## 5. Why two files

Authoritative and derived data differ in what must happen when the file is damaged. For derived data the answer is `rm` and
rebuild. For telegraph the answer is restore from a backup and then reconcile against the reply files (section 8). One file
forces every recovery routine to separate the tables and every backup to carry derived rows. Two files cost one more path
and one more connection. The code treats them through the same `db.open(kind)` function.

## 6. Telegraph schema and transactions

`project.db`, migration 1 for telegraph:

* `telegraph_events(seq INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT, ts REAL, kind TEXT, call_id TEXT, raw_json TEXT)`.
  Append only. `kind` is one of `reserve`, `settle`, `refuse`, `cache_hit`, `report`, `torn`. `raw_json` is the record as
  the jsonl line has it, so `db export telegraph` writes the old format. This table is the source of truth.
* `telegraph_calls(scope, call_id, has_reserve, rproblem, rmodel, reserved_tokens, reserved_cost, settled, sproblem, smodel,
  outcome, tokens_in, tokens_out, cost, usage_known, ...)`, one row per call, the projection the proof uses, with
  `CHECK (has_reserve = 1 OR settled = 1)`. Written by `dbq/telegraph.py` in the same transaction as the event, never by a
  trigger and never by another module. A view `telegraph_effective` gives each call's counted tokens and cost.
* `telegraph_torn(scope, name, tokens)`: imported sidecar notes. New data cannot produce a torn record.
* `telegraph_cache(scope, identity_sha256, call_id, reply_file, PRIMARY KEY (scope, identity_sha256))` (invariant I1). The
  request identity (a hash of the URL and body) has no scope in it, so the same question in two ledger directories has the
  same hash. With the scope in the key and in every lookup, the second scope neither hits a uniqueness error nor overwrites the
  first scope's row, and an answer cached for one scope is never served to another. Every table that holds calls, torn
  notes, events and cache rows has `scope` as the first column of its key.
* `speed_baseline(endpoint, model, tok_s, at, PRIMARY KEY (endpoint, model))`.

`scope` replaces the ledger directory. Today the caps are per directory, and a run can have its own `telegraph.jsonl` in its
run directory (`run_spend`). With a `scope` column the default ledger has scope `default` and a run's ledger has the run's
name. Whether per-run ledgers should stay files in the run directory (so they travel with the run) is an open question
(section 13).

Transactions, one per step:

1. Reserve: `BEGIN IMMEDIATE`; compute the accounting; apply the caps; insert the `reserve` event and the call row; `COMMIT`.
   A refusal inserts a `refuse` event in the same transaction and returns the reason. The write lock is released here.
2. Send the request. No transaction is open. The write lock is never held across the network call.
3. Write the reply file (as today: private, redacted).
4. Settle: one transaction that inserts the `settle` event, updates the call row, and inserts the `telegraph_cache` row. The
   cache is visible only with its call, which closes the PR 106 finding "the cache becomes visible before its originating
   call is recorded".
5. Report: one transaction.

A crash after step 1 leaves a reservation with no settlement, counted at its worst case, as the jsonl does. A crash between
3 and 4 leaves an orphan reply file and an unsettled reservation: the call counts at its worst case and the reply file is
evidence for recovery. The proof tests steps 1 to 2 with SIGKILL (section 9).

An unsettled reservation counts forever, as today ("until a person reconciles it"). A `telegraph reconcile` command that
writes the provider's real usage is a later addition; this note changes none of the accounting rules.

## 7. Concurrent writers

The writers are the supervisor, the web server (threads), agent tools, and `tt-orchard telegraph ask` or `report` from a shell.
All of them go through `db.write()`, which runs `BEGIN IMMEDIATE`. SQLite then has one writer at a time and queues others
for up to `busy_timeout`. With a deferred `BEGIN`, a transaction that has read and then tries to write fails
immediately with a snapshot error when another process committed in between; the proof's mutation shows this as red
tests. Readers use WAL snapshots and never block the writer. The web server gives each request thread its own connection.
A reader that takes two queries uses one read transaction, so the summary and the accounting describe the same moment.

If `busy_timeout` expires the writer raises `LedgerUnavailable`, and telegraph returns "the budget ledger cannot be written
or read" and sends nothing, the existing fail-closed path.

A finding from the proof: when several processes open a new file at once, the switch to WAL can answer "database is locked"
immediately, without waiting for `busy_timeout`. Four spawned processes opening a missing directory failed on most runs until
the open retried that statement (and the integrity check) until the timeout passed. The real code needs the same retry.

## 8. Corruption, backups, and what can be rebuilt

Open sequence: make the directory private (section 3), open the file without following links, set pragmas, run
`PRAGMA integrity_check`, run migrations.

What moves a file aside (invariant I5). Only confirmed corruption: a failed `integrity_check`, or an error SQLite reports as
SQLITE_CORRUPT (11) or SQLITE_NOTADB (26) ("database disk image is malformed", "file is not a database", "malformed database
schema"). `sqlite3.OperationalError` is a subclass of `sqlite3.DatabaseError`, so catching `DatabaseError` would also move a
healthy file aside on a lock, a full disk, an I/O error, a read-only file or a file that cannot be opened, and the next
call would start from an empty ledger with every cap reset. Those errors raise a retryable `LedgerUnavailable` that leaves
the file and its siblings untouched, and telegraph refuses the call. The proof injects six such messages at three places in the
open sequence and checks that every byte of the directory is unchanged, and injects three corruption messages at two places and
checks that the file is moved. A moved-aside file is moved with its `-wal` and `-shm` to `<name>.corrupt-<timestamp>-<pid>`.

The unconfirmed state (invariant I2). A restored backup can lack reservations and settled calls committed after the backup was
taken; restoring an older valid backup would make allowance that was spent available again. So any of these events makes the new
database unconfirmed: a restore from a backup, a database moved aside, and a new file created next to existing backups
(the file was deleted and the backups were not). The state is a row in `meta`, it survives reopening, and `reserve` refuses
every call while it is set, with a message that names the cause and the command that clears it (`tt-orchard db confirm-restore`).
Only a person's confirmation clears it. Two things help the person decide, and neither confirms by itself:
`reconcile_replies` adds, for every reply file with no call row, an unsettled reservation at the worst case (so the counted
spend only goes up), and an export of the archived jsonl (section 10) or the provider's usage page shows what is missing.
Calls that failed or timed out left no reply file, so they cannot be found from the files; that is why the person confirms.
For `index.db` none of this applies: it holds derived data, is deleted and rebuilt, and has no unconfirmed state. For
`project.db` the open restores the newest backup that passes the integrity check, marks the result unconfirmed, and if there
is none it starts empty, marked unconfirmed.

Backups of `project.db`: `Connection.backup()` (the online backup API; `VACUUM INTO` also works) to
`backups/project-<timestamp>.db`, mode 0600, taken when the newest backup is older than one hour and a write happens, and always
before a migration. The newest 14 are kept. The file is small (the real backfilled ledger, 77 records, is 48 KiB), so a copy
costs milliseconds. `index.db` has no backup.

What telegraph can rebuild if `project.db` and every backup are lost:

| Datum | Rebuildable from `replies/<call_id>.txt`? |
|---|---|
| That a call was answered, and its call id | Yes, from the file name |
| The answer text | Yes |
| Approximate time | Roughly, from the file's modification time |
| Problem, model | No |
| Tokens in and out, cost, whether usage was known | No |
| Verdicts (helped or not): they drive the model ladder | No |
| Calls that failed, timed out, were refused, or hit the cache | No (no reply file) |
| A reservation of a call that never settled | No |

So a rebuild from reply files can only produce one row per answered call, with unknown usage and the worst-case reservation
for the configured model. That is a conservative total, and it loses the cost split by model, the verdicts, and the existence
of failed and timed-out calls (which may have been billed). The reply files stay files for that reason: they are the only
copy of the data that does not live in the database. The provider's own usage page remains the final record of what was
billed. Backups are therefore part of the design and not an extra: an hourly backup bounds the loss to the calls of one hour.
Keeping the old `telegraph.jsonl` as a read-only archive after the migration costs nothing and covers everything before the move.

## 9. The proof

`orchard/_proofs/telegraph_sqlite.py` implements the reserve, settle, torn and import operations on SQLite with the
accounting rules of `Telegraph._accounting` and `_over_budget`, and, after the review of PR 132, the scope key, the
unconfirmed state, the migration cutover, the directory checks, the corruption rule and an offset index. It is not used by
the product. 137 tests (71 in `tests/test_proof_telegraph_sqlite.py`, 66 in `tests/test_proof_telegraph_sqlite_invariants.py`, of which 4 are strict xfail placeholders for I7 to I10)
pass in about 5 seconds, repeated six times without a failure after the retry described in section 7. Results:

* Same totals. Ten scenarios taken from `tests/test_telegraph.py` run through the real `Telegraph` (two plain calls; a timeout
  with a token cap; an answer with no usage under a cost cap; a 401 before processing; a process that dies mid-request; a
  torn tail; an unpriced legacy record; per-problem caps; a cache hit with reports and the model ladder; an unreadable
  answer). The ledger each one wrote is imported into SQLite; calls, tokens, unknown costs and per-problem numbers are
  equal, and dollars match to a billionth. A check that the scenario set reaches the corners (torn, reserved, refused,
  cached, unsettled) guards against both sides agreeing by being empty.
* Same decisions. 40 random sequences of 70 operations (reserve, settle with known and unknown usage, torn record, legacy
  record) under random caps give, at every reserve, the same refusal text as `Telegraph._over_budget` and the same accounting.
  Costs in the sequences are multiples of 1/16 so that sums are exact in any order.
* Four processes, 200 reservations each (spawned processes), against five cap settings (global calls 150, per-problem 60 for
  three problems, both, tokens 20000, cost $10): exactly 150, 180, 150, 20 and 40 reservations are granted, none twice, none
  beyond a cap, and the accounting equals the grants. A second run settles every grant and ends with 800 rows and 80,000 tokens.
* Kill tests. A child killed with SIGKILL after its reservation committed: the reservation is counted at its worst case
  (1500 tokens, $0.5), the next reserve under `max_calls = 1` is refused with "call cap", and the integrity check passes. A
  child killed inside the transaction, after the insert and before the commit: nothing remains, there is no stale lock,
  and a new writer succeeds within a 1 s timeout. Six children run a reserve-and-settle storm and are killed at different
  points: every reservation a child reported after its commit is in the file, every reported settlement is settled, no row
  is half written (`CHECK (has_reserve = 1 OR settled = 1)`), and the counted tokens equal the sum recomputed from the rows.
* Invariant tests (section 0), each seen red when its guard was removed. The cutover test runs the real, unchanged
  `Telegraph` against a migrated directory: every call is refused, no request is sent, and `telegraph.jsonl` is not
  recreated; a request in flight during the cutover returns and its settlement cannot be written, and the database counts
  its reservation at the worst case (3 calls: two settled before, one reserved). A restore from a backup that lacks five
  reservations refuses a call that a cap of 3 would otherwise allow, until `confirm_reconciled`. Errors injected as
  `sqlite3.OperationalError` ("database is locked", "disk I/O error", "database or disk is full", "unable to open database
  file", "attempt to write a readonly database", "database table is locked") at connect, at the WAL pragma and at the integrity
  check leave every file of the directory byte-identical; a real lock held by another connection makes `reserve` raise the
  retryable error and changes nothing. The log index test set includes the miss that the review pointed out (a same-size edit
  in an early chunk) as a test that expects the ordinary poll to miss it and a full check to catch it.
* Mutations, first file. 22 guards removed one at a time, each by a script that restores the file in a `finally`; all 22
  turned the tests red: deferred `BEGIN`; check and insert in separate transactions; unknown usage counted at the settlement;
  each of the six caps removed; unknown cost taken as zero; torn records not counted; `busy_timeout` zero; the reservation
  not written before the send; `synchronous = NORMAL`; no `chmod` of an existing file; the integrity check skipped; a newer
  schema accepted; the settlement always marking usage known; the settled problem ignored; a bool taken as a number; imported
  lines not kept; a legacy settlement ignored. The first run of this set left the skipped-integrity-check mutation green,
  because the damaged file used in the test also failed on open; the test now keeps page 1 intact.
* Mutations, invariants. 28 more: the cache lookup ignoring the scope; the cache key without the scope; totals not filtered by
  scope; `reserve` ignoring the unconfirmed state; a moved-aside file, a restore, and a new file beside backups not marked;
  confirming not clearing; reply files not reconciled; no sentinel after the rename; a live jsonl beside the marker accepted;
  totals not compared before the cutover; a second migration allowed; a loose directory mode not tightened; a linked directory
  followed; another user's directory accepted; a linked database file followed; non-idempotent directory creation; the WAL
  switch not retried; every database error treated as corruption; corruption not moved aside; a busy database not converted
  to the retryable error; the tail chunk not checked; the full check never run; a half-written line consumed. 26 turned the
  tests red. Two survived: removing the inode check and removing the "file smaller than the offset" check of the index. Both
  are covered by the tail-chunk digest (a replaced or shortened file fails it too), so they are cheap early exits and not
  independent guards. They stay in the design for that reason. The mutation that removed the 0700 `mode` argument of
  `makedirs` was dropped, because the tightening step makes the mode correct whatever `makedirs` did; the tightening has its
  own mutation. Two of my mutations first targeted the wrong place (one hit the retry helper and not `BEGIN IMMEDIATE`; the
  WAL retry mutation survived a probabilistic test) and the test set was strengthened until each mutation went red.
* Migration check on the real data. `python -m orchard._proofs.telegraph_sqlite import-check <telegraph dir> --db <file>`
  read the backfilled ledger without changing it: 77 records imported; 40 calls, 1,101,006 tokens in, 99,570 tokens out,
  $1.2106117, per-problem numbers (17 problems) equal before and after; exit status 0.

What the proof does not show. The timing: this machine's temporary directory may be memory backed, so the speed of
`synchronous = FULL` was not measured on a disk. A power cut was not tested (SIGKILL does not drop the operating system's
write cache). A network file system was not tested. Threads in one process, which the web server will add, were not tested.

## 10. Migration plan for existing data

Step 0, before any code: copy the telegraph directory aside (read only, with `sha256` recorded).

The cutover must be drained and must stop old writers for good (invariant I3). The first version of this note held the old
lock only while importing. An old process, or a stale checkout, could then append to `telegraph.jsonl` (the old code opens it
with `O_CREAT`, so it recreates a missing file) and build a second, incomplete budget history; so could the settlement of a
request that was in flight during the import. The migration therefore replaces the file with something the old code cannot
write to, and the proof runs the real old `Telegraph` against the result.

Step 1, migration command: `tt-orchard db migrate-telegraph [--dir <telegraph dir>]`. Stop the harness first. The command should
refuse while a supervisor has an open run that uses this directory (that check is a requirement for the real command; the
proof does not test it). A request that is in flight anyway is covered by step 5.
1. Take an exclusive `flock` on the old `telegraph.lock` (the lock the old code takes) for the whole command.
2. In one `BEGIN IMMEDIATE` transaction: refuse if this scope was imported before (a `meta` row `migrated:<scope>`);
   parse `telegraph.jsonl` as today (a malformed line is skipped and counted, and the count is printed); insert every record
   into `telegraph_events` in order, verbatim, and project calls, torn notes and the cache index (`cache/*.json`).
   Sidecar `telegraph.jsonl.torn-*` files without a note get one, as `_quarantine_torn_tail` would.
3. In the same transaction run the equality check (below). If it fails, roll back (the proof checks that the database has no
   rows and the old directory is byte for byte as it was) and exit non-zero.
4. Commit. Compute the sha256 of the ledger, then `rename` `telegraph.jsonl` to `telegraph.jsonl.archived` (atomic).
5. Make a directory named `telegraph.jsonl` in its place, with a short `README.txt` inside. An old process that opens it
   for appending gets `EISDIR`; the old code's reads fail the same way. The old code treats both as "the budget ledger cannot
   be written or read" and sends nothing (the test `test_a_ledger_that_cannot_take_a_record_refuses_the_call` of the old code
   relies on the same behaviour). A request that was already in flight can still return, and its settlement cannot be
   written; the old code says "the ledger could not record this call". The reservation of that request is in the import and
   counts at its worst case, so no spend is lost.
6. Write `MIGRATED-TO-SQLITE` (date, archive name, sha256, record count, scope). The reply files stay where they are. Nothing
   is deleted.

New code calls `refuse_if_legacy_present(directory)` before it uses a telegraph directory. If the marker exists and
`telegraph.jsonl` is a regular file or a link again, an old writer got through: the new code refuses to run against the
directory, and a person stops the writer and reconciles before any call. A migration run a second time is refused. Old
writers must stay stopped until SQLite-aware code is installed; the sentinel makes a forgotten one fail closed and not
quietly.

The equality check, written and tested in the proof (`compare` and `import_and_compare`): the jsonl accounting and summary
(`telegraph.accounting` and `summarise_records`) and the SQLite accounting and totals must be equal for calls, tokens
in, tokens out, accounted calls, accounted tokens, unknown-cost count, cost to a billionth, and every per-problem row. It was run on the
backfilled ledger of 40 helper calls ($1.21): equal. The test `test_import_check_fails_when_one_call_is_changed_after_the_import`
shows that the check can fail. A second check compares `export telegraph` with the archived file line by line.

Step 2: ship the new code with `telegraph.jsonl` still read when no database exists, then remove that read after one release.
Step 3: `tt-orchard telegraph usage` reads from the database; its output is unchanged and a test compares the two on the
same data. The web view reads one snapshot.

A note on the real data: the backfilled problem ids in this ledger contain the literal text `None` (for example
`linear_scan-None-r1`). That comes from how the helper evaluation named its problems. The migration keeps ids verbatim.

## 11. What does not get better with SQLite

* Symlinks and path traversal when scanning run directories. The virtual spend scanner opens files under run directories that
  a person or an agent can write into. A database changes none of this: the walk, the `O_NOFOLLOW` handling of every path
  component, the directory pinning, and the depth and entry budgets stay in the scanner. An index of offsets adds one more
  hazard: a path recorded in `log_files` has to be checked again when it is read (a file replaced by a link after it was
  indexed). The parallel `safeio` module owns this.
* Validation of what is stored. SQLite stores what it is given. `STRICT` and `CHECK` catch some types and ranges (section 4),
  but a cost that is a plausible number and wrong, a model name that does not exist, or a problem id that holds a credential
  passes. The redaction that `telegraph.py` applies before writing a ledger line must apply before every insert, including
  `raw_json`. The database never stores a credential, and a test inserts the test key through each public write function and greps
  the file, its `-wal`, and an export.
* Logic findings of the Copilot kind. The PR 106 findings on `extra_body`, the ladder moving twice for one answer, a cached
  id that cannot be reported, the numeric IPv4 host, and a negative token count are about the rules. A
  database does not change them.
* Disclosure to the agent. Agent shells run as the same user. An agent can open `project.db` as it can open a jsonl ledger. A
  read-only mount or another user belongs to the outer layers in the command-runner plans.
* The human-readable record. A jsonl ledger can be read with `less`. A SQLite file cannot. `db export` covers
  the need, and the supervisor's run ledger stays readable.
* Disk space and write cost per record. A SQLite file with indexes is larger than the equivalent text. At 77 records the file is 48 KiB
  (the page size is 4 KiB and the file has three tables and their indexes), against 18 KiB of jsonl.

## 12. Export, retention, testing, security

Export: `tt-orchard db export <table> [--db project|index] [--scope S]` writes jsonl to standard output. For `telegraph` it
writes the `raw_json` of each event in order, the same lines the ledger had. For the other tables it writes one object per row
with the column names. `telegraph usage` keeps its output and reads the database.

Retention. `telegraph_events` and `telegraph_calls` are kept (the data is small and is the budget). `index.db` tables are
pruned: `log_files` rows whose path no longer exists are deleted at the next scan; `runs` rows whose run directory is gone are
deleted. `backups/` keeps 14. `db vacuum` runs `VACUUM` on request; there is no automatic vacuum. A size warning is shown by
`db status` when `project.db` passes 64 MiB, which at the size of the real ledger is on the order of a million calls.

Testing, beyond the proof:
* A hostile-input test for each public write function: an integer beyond 64 bits, `NaN`, `Infinity`, a very long string, a
  string with NUL, a lone surrogate, a key, a home path. The pass condition is a refusal or a redaction, never a stored raw value.
* The kill test of the proof for every write function that has more than one step: kill at each step and check the
  invariants (the counted total is at least the true spend; no row is half written; the integrity check passes).
* A two-version test: a file at the previous schema version opens and migrates; a file at a newer version is refused.
* The repository test for SQL outside `db.py` and `dbq/`, seen to fail first.
* A test in a bare environment where the editable install cannot hide the module path, as the global rules require.

Security. The database never stores a credential: the key is read at call time and is not an input to any write function.
Stored text passes the same redaction (`telegraph.redact` and the home and host removal) as the ledger. The files are 0600
in a 0700 directory, outside the repository, and `.gitignore` is not needed because the default path is not inside a checkout
(the code refuses a `[database] dir` inside the repository). Backups keep the same permissions.

## 13. What I could not determine

* The poll cost of `list_runs` and `_compare` at many runs. Only 14 run ledgers exist on this machine and I did not benchmark.
* Whether `~/.cache/tt-orchard/` is on a local disk on every machine that runs the harness. WAL needs one.
* Speed of `synchronous = FULL` on a spinning or network-backed disk.
* Whether a per-run telegraph ledger should live in the run directory (travels with the run) or in the shared database under
  a scope. The note assumes a scope and keeps the choice open.
* How PR 115 changes before it merges. The index design depends on its scanner returning per-file totals.
* How often a full chunk check of the offset index should run. The note says once a day per file; I measured nothing about log
  sizes or the time to hash them.
* Whether the check "no supervisor has an open run that uses this directory" before a migration can be made reliable. The proof
  tests the sentinel, which holds even when that check is missed.
* The count of review findings by class is my reading of the inline comments; no tool produced them. Counts are 14 of 42 (PR 106),
  about 8 of 23 (PR 111), about 11 of 18 (PR 115).

## 14. Implementation status (the telegraph migration)

Built: `orchard/db.py` (the access layer), `orchard/telegraph_store.py` (the telegraph operations), `orchard/telegraph_migrate.py` (the
importer and the cutover), `orchard/db_cmd.py` (`tt-orchard db status | migrate telegraph | export telegraph | backup | restore |
confirm-restore`). `telegraph.py`, `telegraph_cmd.py`, `telegraph_view.py` and `spend_cmd.py` read and write the ledger through the
store. The code for the jsonl format that is left is the importer and the reader of an archived file for `db export --from-archive`;
a frozen copy of the old `Telegraph` lives in `tests/legacy_telegraph.py` so the tests can run the real old writer. A repository
test (`tests/test_no_sql_outside_the_database_layer.py`) fails on SQL, or an import of `sqlite3`, in any other module.

Where the implementation differs from the text above, and why:

* One database per telegraph state directory, `<dir>/project.db`. Section 3 proposed one file in `~/.cache/tt-orchard/db/` with the directory as a scope.
  The state directory is the one `[telegraph] dir`, `$ORCHARD_TELEGRAPH_DIR` or `--dir` names (issue 133), so the markers, the sentinel,
  `open.lock`, `db.users`, the backups and the reply files all live beside the ledger they protect, and a per-run ledger keeps travelling
  with its run directory (the open question in section 13). `scope` is still the first column of every key and every lookup (I1). The
  `[database] dir` setting and `ORCHARD_DB_DIR` of section 3 are not built. `index.db` and the other tables of section 1 are not built.
* No `telegraph_calls` projection and no `telegraph_effective` view. The accounting is `telegraph.accounting` run over the records of
  one scope, read in `seq` order, so every total, every refusal sentence and every floating-point sum is the function the jsonl ledger used,
  adding in ledger order (issue 153, second point: the sums are not taken in SQL, and a test compares them exactly with the old code's
  sums). There is no `telegraph_torn` table: a torn note is an event of kind `torn`.
* Reads for a page are one read transaction (`telegraph_store.read_snapshot`), and the page never creates, recovers or migrates a file.
  The page's cache key is the newest `seq`, the record count, the `meta` rows and the guard names (file times are not used).
* A schema version above the program's is rejected twice: before the migration, and again after the write lock is taken (issue 153, first
  point). A file at the current version that is missing a table is refused and not moved aside (the integrity check cannot see it).
* Recovery opens the newest backup that passes its own check, staged and marked inside the staging file, as section 8 says; with no usable
  backup it starts empty and unconfirmed. A `-wal` file is kept under a second name (a hard link) while a possibly damaged file is opened,
  because SQLite deletes it when the connection that failed to open the file closes.
* The migration does the malformed-line and totals checks before it writes `MIGRATING-TO-SQLITE`, and removes the marker again when the
  start is refused, so a refused start leaves no marker. It refuses a ledger larger than 16 MiB (the page refuses that size as well).
* `confirm-restore` opens the database with its full recovery, refuses when a migration did not finish or the integrity check fails, charges
  every reply file that has no record at the worst case, then confirms. `--note` is required.
* Not built: the check that no supervisor has an open run that uses the directory (the note called it a requirement "if cheap"). The old
  `telegraph.lock` is held for the whole command and the sentinel refuses old writers, which is the protection the note relies on.
* Not built: the second equality check against the provider's usage page (it cannot be automated); a repository refusal of a state
  directory inside the checkout.
