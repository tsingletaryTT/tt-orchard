# A project database for tt-orchard: what moves, what gets indexed, what stays a file

Status: design note with a proof of the riskiest claim. No production module is changed. The proof is
`orchard/_proofs/telegraph_sqlite.py` with `tests/test_proof_telegraph_sqlite.py`; the product does not import it.
Date: 2026-10-28. Version of the repository when written: 0.9.35.

The operator's question: "if a little sqlite DB inside this project helps everything run smoothly THEN BY ALL MEANS let's do it."

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
skipped_helper, skipped_other)` lets a poll read only the bytes after `offset`, up to the last newline. A file whose inode
changed, whose size is smaller than `offset`, or that has been rewritten (same size, newer mtime and a changed hash of its
first 4 KiB) restarts from zero. The walk over directories stays, so the symlink and depth findings stay (section 11).

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
* The file is created with `os.open(..., 0o600)` and `fchmod`, so an older file with loose permissions is tightened. SQLite gives
  `-wal` and `-shm` the permissions of the main file; the proof test checks all three are 0600. The directory is 0700.
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
* `telegraph_cache(scope, identity_sha256 PRIMARY KEY, call_id, reply_file)`.
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

## 8. Corruption, backups, and what can be rebuilt

Open sequence: create the directory and file private, set pragmas, run `PRAGMA integrity_check`, run migrations. A failed
check, or any `sqlite3.DatabaseError`, moves the file and its `-wal` and `-shm` to
`<name>.corrupt-<timestamp>-<pid>` and starts a new file. For `index.db` that is the whole recovery. For `project.db` the
open then restores the newest backup that passes the integrity check (if there is none, it starts empty and telegraph
refuses to send until `tt-orchard db recover` has been run, because an empty ledger would let the caps be passed). The proof
tests the move-aside with a file whose first page is intact and whose other pages are overwritten; only the integrity check
finds that damage, and the mutation that skips the check goes red.

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
accounting rules of `Telegraph._accounting` and `_over_budget`. It is not used by the product. 71 tests in
`tests/test_proof_telegraph_sqlite.py` pass in about 3 seconds. Results:

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
* Mutations. 23 guards removed one at a time, each by a script that restores the file in a `finally`; all 23 turned the test
  file red: deferred `BEGIN`; check and insert in separate transactions; unknown usage counted at the settlement; each of the
  six caps removed; unknown cost taken as zero; torn records not counted; `busy_timeout` zero; the reservation not written
  before the send; `synchronous = NORMAL`; no `chmod` of an existing file; the directory created 0755; the integrity check skipped;
  a newer schema accepted; the settlement always marking usage known; the settled problem ignored; a bool taken as a number; imported
  lines not kept; a legacy settlement ignored. The first run left the skipped-integrity-check mutation green, because the
  damaged file used in the test also failed on open; the test now keeps page 1 intact.
* Migration check on the real data. `python -m orchard._proofs.telegraph_sqlite import-check <telegraph dir> --db <file>`
  read the backfilled ledger without changing it: 77 records imported; 40 calls, 1,101,006 tokens in, 99,570 tokens out,
  $1.2106117, per-problem numbers (17 problems) equal before and after; exit status 0.

What the proof does not show. The timing: this machine's temporary directory may be memory backed, so the speed of
`synchronous = FULL` was not measured on a disk. A power cut was not tested (SIGKILL does not drop the operating system's
write cache). A network file system was not tested. Threads in one process, which the web server will add, were not tested.

## 10. Migration plan for existing data

Step 0, before any code: copy the telegraph directory aside (read only, with `sha256` recorded).

Step 1, migration command: `tt-orchard db migrate-telegraph [--dir <telegraph dir>]`.
1. Take an exclusive `flock` on the old `telegraph.lock` for the whole import, so no old-style writer runs meanwhile.
2. Parse `telegraph.jsonl` as today (a malformed line is skipped and counted, and the count is printed).
3. Insert every record into `telegraph_events` in order, verbatim, and project calls, torn notes and the cache index
   (`cache/*.json`) in the same transaction. Sidecar `telegraph.jsonl.torn-*` files without a note get one, as `_quarantine_torn_tail` would.
4. Run the equality check (below). If it fails, roll back, leave the old files alone, exit non-zero.
5. Write `MIGRATED-TO-SQLITE` (a text file with the date, the sha256 of the ledger, the row counts) in the old directory and
   rename `telegraph.jsonl` to `telegraph.jsonl.archived`. The reply files stay where they are. Nothing is deleted.

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
* The count of review findings by class is my reading of the inline comments; no tool produced them. Counts are 14 of 42 (PR 106),
  about 8 of 23 (PR 111), about 11 of 18 (PR 115).
