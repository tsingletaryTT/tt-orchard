# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PROOF ONLY. THE PRODUCT DOES NOT USE IT. A SQLite reservation ledger with the accounting rules of `orchard.telegraph`.

Nothing in the product imports this module. It exists to test the claims of
docs/superpowers/specs/2026-10-28-project-database-design.md: telegraph's budget ledger can live in one SQLite file
and keep every rule it has today, with less code for locking and recovery, and the invariants I1 to I6 of that note
hold (tests/test_proof_telegraph_sqlite_invariants.py names one test for each).

The rules, copied from `orchard.telegraph._accounting` and `_over_budget`:

  * a call is reserved at its worst case before any request is sent;
  * a settlement replaces the reservation with the numbers the provider reported;
  * a settlement with `usage_known` false (a timeout, an answer with no usage) counts at the reservation;
  * a settlement with no reservation (a record from before reservations existed) counts by its own numbers;
  * a cost that is not a number is counted apart as unknown and is never taken as zero;
  * a torn record counts as one call of unknown cost, in the global totals only;
  * the caps are per problem and global: calls, tokens, and dollars (which need a price). They are counted per scope
    (a scope is what a ledger directory is today).

What SQLite changes: the check and the reservation are one `BEGIN IMMEDIATE` transaction, so two processes cannot
both spend the same allowance, and a process killed at any point leaves the reservation complete or absent.
The flock, the fsync calls, the torn-tail quarantine, and the sidecar notes of `telegraph.py` have no counterpart here.

What this module adds because the review of PR 132 required it:
  I1  `scope` is part of every key, so a cached answer is never served to another scope;
  I2  a restored backup, a moved-aside file, or a new file beside backups starts "unconfirmed": `reserve` refuses until a
      person confirms;
  I3  `migrate_directory` leaves a refusing sentinel where `telegraph.jsonl` was, and `refuse_if_legacy_present` makes new
      code refuse a directory that holds both the marker and a live jsonl;
  I4  the state directory is made private or refused (link, other owner), and creating it is safe when two processes race;
  I5  only confirmed corruption moves a file aside; every other SQLite error is `LedgerUnavailable` (retry later);
  I6  `LogIndex` checks the bytes it has consumed (a tail-chunk digest on every poll, every chunk on a full check) before
      it reuses a stored total.

Run `python -m orchard._proofs.telegraph_sqlite import-check <telegraph dir> --db <file>` to import a jsonl ledger and
compare its totals with the jsonl accounting (read only for the source).
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 30_000
DEFAULT_SCOPE = "default"
MARKER = "MIGRATED-TO-SQLITE"
ARCHIVE = "telegraph.jsonl.archived"
LEDGER_NAME = "telegraph.jsonl"
LOCK_NAME = "telegraph.lock"


class DatabaseTooNew(Exception):
    """The file was written by a newer version of the program."""


class LedgerUnavailable(Exception):
    """The database cannot be used right now (busy, locked, full, I/O error, no permission). The file is untouched.
    Retry later; telegraph refuses the call, as it does today when the ledger cannot be written."""


class UnsafeDirectory(Exception):
    """The state directory or file is a link, belongs to someone else, or is not what it must be."""


class LegacyLedgerPresent(Exception):
    """A directory holds the migration marker and a live telegraph.jsonl: an old writer is still running."""


class MigrationMismatch(Exception):
    """The totals after an import differ from the totals before it. Nothing was changed."""


class AlreadyMigrated(Exception):
    """This directory was migrated before."""


class _Corrupt(Exception):
    """Internal: SQLite said the file is damaged (or the integrity check failed)."""


@dataclass
class Limits:
    """The caps `_over_budget` reads from `telegraph.Config`. Any object with these attributes works."""
    max_calls: int = 10 ** 9
    max_total_tokens: int = 10 ** 12
    max_cost_usd: float | None = None
    max_calls_per_problem: int = 10 ** 9
    max_tokens_per_problem: int | None = None
    max_cost_per_problem_usd: float | None = None


# ------------------------------------------------------------------------------------------------ schema and migrations

def _migrate_1(conn: sqlite3.Connection) -> None:
    """One row per call and scope. The reservation columns and the settlement columns are filled at different times."""
    conn.execute("""
        CREATE TABLE calls (
            scope           TEXT NOT NULL DEFAULT 'default',
            call_id         TEXT NOT NULL,
            has_reserve     INTEGER NOT NULL DEFAULT 0 CHECK (has_reserve IN (0, 1)),
            rproblem        TEXT,
            rmodel          TEXT,
            reserved_tokens INTEGER,
            reserved_cost   REAL,
            settled         INTEGER NOT NULL DEFAULT 0 CHECK (settled IN (0, 1)),
            sproblem        TEXT,
            smodel          TEXT,
            outcome         TEXT,
            tokens_in       INTEGER NOT NULL DEFAULT 0,
            tokens_out      INTEGER NOT NULL DEFAULT 0,
            cost            REAL,
            usage_known     INTEGER,
            PRIMARY KEY (scope, call_id),
            CHECK (has_reserve = 1 OR settled = 1)
        )""")
    conn.execute("CREATE TABLE torn (scope TEXT NOT NULL DEFAULT 'default', name TEXT NOT NULL, tokens INTEGER NOT NULL, "
                 "PRIMARY KEY (scope, name))")
    # Every imported line, verbatim, in order: reports, refusals and cache hits live only here, and `export` reads it.
    conn.execute("CREATE TABLE records (seq INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL DEFAULT 'default', "
                 "raw TEXT NOT NULL)")
    # I1: the key has the scope in it, and so does every lookup.
    conn.execute("CREATE TABLE cache (scope TEXT NOT NULL, identity TEXT NOT NULL, call_id TEXT NOT NULL, "
                 "reply_file TEXT NOT NULL, PRIMARY KEY (scope, identity))")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    # I6: what an offset index has consumed of each log.
    conn.execute("CREATE TABLE log_files (path TEXT PRIMARY KEY, dev INTEGER NOT NULL, inode INTEGER NOT NULL, "
                 "offset INTEGER NOT NULL, total INTEGER NOT NULL, tail_digest TEXT NOT NULL, digests TEXT NOT NULL)")
    # What a call counts for: its settlement unless that has unknown usage, otherwise its reservation (the worst case).
    conn.execute("""
        CREATE VIEW effective AS
        SELECT scope, call_id,
               CASE WHEN settled = 1 THEN sproblem ELSE rproblem END AS problem,
               CASE WHEN settled = 1 AND (usage_known IS NOT 0 OR has_reserve = 0)
                    THEN tokens_in + tokens_out ELSE COALESCE(reserved_tokens, 0) END AS tokens,
               CASE WHEN settled = 1 AND (usage_known IS NOT 0 OR has_reserve = 0)
                    THEN cost ELSE reserved_cost END AS cost
        FROM calls""")


MIGRATIONS = [_migrate_1]            # index i takes user_version i to i + 1; append only, never edit one that has shipped


# ------------------------------------------------------------------------------------------------ value rules

def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _cost(value):
    """A number, or None. A bool is not a number, as in telegraph."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _flag(value):
    return 0 if value is False else 1 if value is True else None


def _sent(rec: dict) -> bool:
    return "call_id" in rec and not rec.get("cached") and rec.get("outcome") in ("ok", "error")


# ------------------------------------------------------------------------------------------------ I5: what is corruption

def _is_corruption(exc: sqlite3.Error) -> bool:
    """True only when SQLite says the file is damaged: SQLITE_CORRUPT (11) or SQLITE_NOTADB (26), or the message that
    goes with them. A busy or locked file, a full disk, an I/O error, a read-only file and a file that cannot be opened are
    not corruption, whatever exception class carries them (OperationalError is a kind of DatabaseError)."""
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None and (code & 0xFF) in (11, 26):
        return True
    text = str(exc).lower()
    return "malformed" in text or "not a database" in text


# ------------------------------------------------------------------------------------------------ I4: the directory

def _private_directory(directory: Path) -> None:
    """The state directory exists, is a real directory owned by this user, and is mode 0700. A missing directory is
    created (two processes may do this at once; losing the race is not an error). A link, a directory of another user, or
    a file in its place is refused and left alone. A loose mode on our own directory is tightened."""
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
    except OSError as exc:
        raise LedgerUnavailable(f"cannot create {directory}: {exc.strerror or type(exc).__name__}") from exc
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)      # a link or a file fails here
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UnsafeDirectory(f"{directory} is a link or not a directory") from exc
        raise LedgerUnavailable(f"cannot open {directory}: {exc.strerror or type(exc).__name__}") from exc
    try:
        st = os.fstat(fd)
        if st.st_uid != os.geteuid():
            raise UnsafeDirectory(f"{directory} belongs to another user")
        if stat.S_IMODE(st.st_mode) & 0o077:
            os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


# ------------------------------------------------------------------------------------------------ the ledger

class Ledger:
    """One SQLite file. Opening it makes the directory private, the file private, sets the pragmas, checks the file and
    migrates the schema. `moved_aside` is the path a damaged file was moved to, or None."""

    def __init__(self, path, *, busy_timeout_ms: int = BUSY_TIMEOUT_MS):
        self.path = Path(path)
        self.moved_aside: Path | None = None
        self._busy = busy_timeout_ms
        _private_directory(self.path.parent)
        existed = os.path.lexists(self.path)
        had_backups = any((self.path.parent / "backups").glob("*.db")) if (self.path.parent / "backups").is_dir() else False
        try:
            self._conn = self._open_checked()
        except _Corrupt:
            self._move_aside()
            self._conn = self._open_checked()
        self._migrate()
        if self.moved_aside is not None:
            self._mark_unreconciled(f"the database was damaged and moved to {self.moved_aside.name}")
        elif not existed and had_backups:
            self._mark_unreconciled("the database file was missing but backups exist")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- opening

    def _open_checked(self) -> sqlite3.Connection:
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)   # created private; -wal and -shm follow its mode
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise UnsafeDirectory(f"{self.path} is a link") from exc
            raise LedgerUnavailable(f"cannot open {self.path}: {exc.strerror or type(exc).__name__}") from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
                raise UnsafeDirectory(f"{self.path} is not a regular file of this user")
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        try:
            conn = sqlite3.connect(self.path, isolation_level=None, timeout=self._busy / 1000)
        except sqlite3.Error as exc:
            raise self._classified(exc) from exc
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(self._busy)}")
            # Switching a new file to WAL needs an exclusive lock and can answer "locked" at once, without waiting for
            # the busy timeout, when another process is doing the same (found by the concurrent-first-open test).
            # The switch is retried until the busy timeout has passed.
            self._retry_while_busy(lambda: conn.execute("PRAGMA journal_mode = WAL"))
            conn.execute("PRAGMA synchronous = FULL")            # a reservation must survive a power cut, as fsync does today
            conn.execute("PRAGMA foreign_keys = ON")
            verdict = self._retry_while_busy(lambda: conn.execute("PRAGMA integrity_check").fetchall())
        except sqlite3.Error as exc:
            conn.close()
            raise self._classified(exc) from exc
        if verdict != [("ok",)]:
            conn.close()
            raise _Corrupt("integrity_check failed")
        return conn

    def _retry_while_busy(self, action):
        deadline = time.monotonic() + self._busy / 1000
        while True:
            try:
                return action()
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    raise
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    @staticmethod
    def _classified(exc: sqlite3.Error) -> Exception:
        if _is_corruption(exc):
            return _Corrupt(str(exc))
        return LedgerUnavailable(f"{type(exc).__name__}: {exc}")

    def _move_aside(self) -> None:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        target = self.path.with_name(f"{self.path.name}.corrupt-{stamp}-{os.getpid()}")
        os.replace(self.path, target)
        for suffix in ("-wal", "-shm"):
            with contextlib.suppress(OSError):
                os.replace(str(self.path) + suffix, str(target) + suffix)
        self.moved_aside = target

    def _migrate(self) -> None:
        have = self.user_version()
        if have > len(MIGRATIONS):
            self._conn.close()
            raise DatabaseTooNew(f"{self.path} has schema version {have}; this program knows {len(MIGRATIONS)}")
        while have < len(MIGRATIONS):
            with self._write():
                have = self.user_version()               # another process may have migrated while this one waited
                if have >= len(MIGRATIONS):
                    break
                MIGRATIONS[have](self._conn)
                self._conn.execute(f"PRAGMA user_version = {have + 1}")

    @contextlib.contextmanager
    def _write(self):
        """A writer: BEGIN IMMEDIATE takes the write lock at once, so a read and the write that depends on it cannot be
        split by another writer. Commits on success, rolls back on any exception. A busy or locked database is a
        `LedgerUnavailable` and changes nothing."""
        try:
            self._conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise LedgerUnavailable(f"{type(exc).__name__}: {exc}") from exc
        try:
            yield
        except BaseException as exc:
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute("ROLLBACK")
            # A disk-full, I/O or read-only error from a statement inside the body is retryable as well (I5). Corruption
            # keeps its own type so that it can be told apart.
            if isinstance(exc, sqlite3.OperationalError) and not _is_corruption(exc):
                raise LedgerUnavailable(f"{type(exc).__name__}: {exc}") from exc
            raise
        try:
            self._conn.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute("ROLLBACK")
            raise LedgerUnavailable(f"{type(exc).__name__}: {exc}") from exc

    # -- I2: unconfirmed state, backups, restore

    def _mark_unreconciled(self, reason: str) -> None:
        with self._write():
            self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('unreconciled', ?)", (reason,))

    def unreconciled(self):
        row = self._conn.execute("SELECT value FROM meta WHERE key = 'unreconciled'").fetchone()
        return row[0] if row else None

    def confirm_reconciled(self, note: str) -> None:
        """A person says the history is accounted for. Only this clears the state."""
        with self._write():
            self._conn.execute("DELETE FROM meta WHERE key = 'unreconciled'")
            self._conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('confirmed', ?)",
                               (f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {note}"[:300],))

    def backup(self, directory) -> Path:
        """An online backup (the SQLite backup API) into `directory`, private, with a name that sorts by time."""
        directory = Path(directory)
        _private_directory(directory)
        dest = directory / f"project-{time.strftime('%Y%m%dT%H%M%S')}-{time.monotonic_ns()}.db"
        target = sqlite3.connect(dest)
        try:
            self._conn.backup(target)
        finally:
            target.close()
        os.chmod(dest, 0o600)
        return dest

    @classmethod
    def restore(cls, path, backup, **kw) -> "Ledger":
        """Put a backup in place of the database. The history since the backup is unknown, so the result is unconfirmed."""
        path = Path(path)
        _private_directory(path.parent)
        if os.path.lexists(path):
            stamp = time.strftime("%Y%m%dT%H%M%S")
            replaced = path.with_name(f"{path.name}.replaced-{stamp}-{os.getpid()}")
            os.replace(path, replaced)
            for suffix in ("-wal", "-shm"):
                with contextlib.suppress(OSError):
                    os.replace(str(path) + suffix, str(replaced) + suffix)
        shutil.copyfile(backup, path)
        os.chmod(path, 0o600)
        ledger = cls(path, **kw)
        ledger._mark_unreconciled(f"restored from backup {Path(backup).name}; calls made since are not in it")
        return ledger

    def reconcile_replies(self, reply_dir, worst_tokens: int, worst_cost, scope: str = DEFAULT_SCOPE) -> int:
        """For every reply file with no call row, add an unsettled reservation at the worst case. Calls that failed or
        timed out left no reply file and cannot be found this way, so this never confirms."""
        added = 0
        with self._write():
            for f in sorted(Path(reply_dir).glob("*.txt")):
                row = self._conn.execute("SELECT 1 FROM calls WHERE scope = ? AND call_id = ?", (scope, f.stem)).fetchone()
                if row is None:
                    self._conn.execute(
                        "INSERT INTO calls (scope, call_id, has_reserve, rproblem, rmodel, reserved_tokens, reserved_cost) "
                        "VALUES (?, ?, 1, '(recovered)', NULL, ?, ?)", (scope, f.stem, _int(worst_tokens), _cost(worst_cost)))
                    added += 1
        return added

    # -- facts about the file

    def user_version(self) -> int:
        return self._conn.execute("PRAGMA user_version").fetchone()[0]

    def pragmas(self) -> dict:
        return {"journal_mode": self._conn.execute("PRAGMA journal_mode").fetchone()[0],
                "synchronous": self._conn.execute("PRAGMA synchronous").fetchone()[0],
                "busy_timeout": self._conn.execute("PRAGMA busy_timeout").fetchone()[0],
                "foreign_keys": self._conn.execute("PRAGMA foreign_keys").fetchone()[0]}

    def integrity_ok(self) -> bool:
        return self._conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]

    def count_rows(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    def rows(self) -> list[dict]:
        cur = self._conn.execute("SELECT * FROM calls ORDER BY scope, call_id")
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]

    def unsettled(self, scope: str = DEFAULT_SCOPE) -> list[str]:
        return [r[0] for r in self._conn.execute(
            "SELECT call_id FROM calls WHERE settled = 0 AND scope = ? ORDER BY call_id", (scope,))]

    # -- accounting (the same numbers as telegraph.accounting)

    def accounting(self, scope: str = DEFAULT_SCOPE) -> dict:
        total_calls, total_tokens, total_cost, total_unknown = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens), 0), COALESCE(SUM(COALESCE(cost, 0)), 0), "
            "COALESCE(SUM(cost IS NULL), 0) FROM effective WHERE scope = ?", (scope,)).fetchone()
        torn_calls, torn_tokens = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM torn WHERE scope = ?", (scope,)).fetchone()
        problems = {}
        for name, calls, tokens, cost, unknown in self._conn.execute(
                "SELECT problem, COUNT(*), SUM(tokens), SUM(COALESCE(cost, 0)), SUM(cost IS NULL) "
                "FROM effective WHERE scope = ? GROUP BY problem", (scope,)):
            problems[name] = {"calls": calls, "tokens": tokens, "cost": cost, "unknown_cost": unknown}
        return {"calls": total_calls + torn_calls, "tokens": total_tokens + torn_tokens, "cost": total_cost,
                "unknown_cost": total_unknown + torn_calls, "problems": problems}

    def totals(self, scope: str = DEFAULT_SCOPE) -> dict:
        """What `telegraph.summarise_records` reports for the calls that were sent."""
        calls, tin, tout, cost = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0), "
            "COALESCE(SUM(COALESCE(cost, 0)), 0) FROM calls WHERE settled = 1 AND scope = ?", (scope,)).fetchone()
        return {"calls": calls, "tokens_in": tin, "tokens_out": tout, "cost_usd": cost}

    # -- the budget

    def _refusal(self, cfg, model: str, tokens: int, cost, problem: str, acct: dict):
        """The reason this call may not be sent, or None. The same rules and the same words as `Telegraph._over_budget`."""
        worst = {"tokens": tokens, "cost": cost}
        here = acct["problems"].get(problem, {"calls": 0, "tokens": 0, "cost": 0.0, "unknown_cost": 0})
        name = f"problem {problem}"
        if here["calls"] >= cfg.max_calls_per_problem:
            return f"the call cap for {name} is reached ({here['calls']} of {cfg.max_calls_per_problem})"
        if cfg.max_tokens_per_problem is not None and here["tokens"] >= cfg.max_tokens_per_problem:
            return f"the token cap for {name} is reached ({here['tokens']} of {cfg.max_tokens_per_problem})"
        if cfg.max_cost_per_problem_usd is not None:
            if here["unknown_cost"]:
                return (f"{here['unknown_cost']} earlier calls for {name} have no known cost (they were made without a price), "
                        "so the cost cap cannot be checked; price them or use a new ledger directory")
            if worst["cost"] is None:
                return f"the cost cap for {name} needs a price, and there is no price for {model}"
            if here["cost"] + worst["cost"] > cfg.max_cost_per_problem_usd:
                return (f"the cost cap for {name} would be passed (spent {here['cost']:.6f}, this call could cost up "
                        f"to {worst['cost']:.6f}, cap {cfg.max_cost_per_problem_usd})")
        if acct["calls"] >= cfg.max_calls:
            return f"the call cap is reached ({acct['calls']} of {cfg.max_calls})"
        if acct["tokens"] >= cfg.max_total_tokens:
            return f"the token cap is reached ({acct['tokens']} of {cfg.max_total_tokens})"
        if cfg.max_cost_usd is not None:
            if acct["unknown_cost"]:
                return (f"{acct['unknown_cost']} earlier calls have no known cost (they were made without a price), so the "
                        "cost cap cannot be checked; price them or use a new ledger directory")
            if worst["cost"] is None:
                return f"the cost cap needs a price, and there is no price for {model}"
            if acct["cost"] + worst["cost"] > cfg.max_cost_usd:
                return (f"the cost cap would be passed (spent {acct['cost']:.6f}, this call could cost up to "
                        f"{worst['cost']:.6f}, cap {cfg.max_cost_usd})")
        return None

    def _unconfirmed_refusal(self):
        reason = self.unreconciled()
        if reason:
            return (f"the ledger's history may be incomplete ({reason}), so no outside call is allowed until a person "
                    "confirms it with `tt-orchard db confirm-restore`")
        return None

    def would_refuse(self, cfg, model: str, tokens: int, cost, problem: str, scope: str = DEFAULT_SCOPE):
        """The decision for a call without reserving anything (a read, no lock)."""
        return self._unconfirmed_refusal() or self._refusal(cfg, model, tokens, cost, problem, self.accounting(scope))

    def reserve(self, call_id: str, problem: str, model: str, tokens: int, cost, cfg, *, scope: str = DEFAULT_SCOPE,
                after_insert=None):
        """Check the caps and write the reservation as ONE transaction. Returns the refusal reason, or None when the
        reservation was written. `after_insert` is a test seam: it runs inside the transaction, before the commit."""
        with self._write():
            why = self._unconfirmed_refusal() or self._refusal(cfg, model, tokens, cost, problem, self.accounting(scope))
            if why:
                return why
            self._conn.execute(
                "INSERT INTO calls (scope, call_id, has_reserve, rproblem, rmodel, reserved_tokens, reserved_cost) "
                "VALUES (?, ?, 1, ?, ?, ?, ?)", (scope, call_id, problem, model, _int(tokens), _cost(cost)))
            if after_insert is not None:
                after_insert()
        return None

    def settle(self, call_id: str, *, problem, model, outcome: str, tokens_in, tokens_out, cost_usd, usage_known,
               scope: str = DEFAULT_SCOPE, cache=None) -> None:
        """Replace a reservation's worst case with what the provider reported. With `cache` = (identity, reply_file) the
        cache row is written in the same transaction, so an answer is never cached without its call (I1 keeps it per scope)."""
        with self._write():
            self._conn.execute(_SETTLE, (scope, call_id, problem, model, outcome, _int(tokens_in), _int(tokens_out),
                                         _cost(cost_usd), _flag(usage_known)))
            if cache is not None:
                self._conn.execute("INSERT OR REPLACE INTO cache (scope, identity, call_id, reply_file) VALUES (?, ?, ?, ?)",
                                   (scope, cache[0], call_id, cache[1]))

    def cache_get(self, scope: str, identity: str):
        row = self._conn.execute("SELECT call_id, reply_file FROM cache WHERE scope = ? AND identity = ?",
                                 (scope, identity)).fetchone()
        return {"call_id": row[0], "reply_file": row[1]} if row else None

    def record_torn(self, name: str, tokens: int, scope: str = DEFAULT_SCOPE) -> None:
        with self._write():
            self._conn.execute("INSERT OR IGNORE INTO torn (scope, name, tokens) VALUES (?, ?, ?)", (scope, name, _int(tokens)))

    # -- import and export

    def _import(self, recs, raws, scope: str) -> None:
        """Import jsonl records in order. The caller holds the write transaction."""
        for rec, raw in zip(recs, raws):
            self._conn.execute("INSERT INTO records (scope, raw) VALUES (?, ?)", (scope, raw))
            if "reserve" in rec:
                self._conn.execute(_RESERVE_UPSERT, (scope, str(rec["reserve"]), rec.get("problem"), rec.get("model"),
                                                     _int(rec.get("tokens")), _cost(rec.get("cost_usd"))))
            elif _sent(rec):
                self._conn.execute(_SETTLE, (scope, rec["call_id"], rec.get("problem"), rec.get("model"), rec["outcome"],
                                             _int(rec.get("tokens_in")), _int(rec.get("tokens_out")),
                                             _cost(rec.get("cost_usd")), _flag(rec.get("usage_known"))))
            elif "torn" in rec:
                self._conn.execute("INSERT OR IGNORE INTO torn (scope, name, tokens) VALUES (?, ?, ?)",
                                   (scope, rec["torn"], _int(rec.get("tokens"))))

    def import_records(self, recs, raws=None, scope: str = DEFAULT_SCOPE) -> int:
        """Import jsonl records in order, in one transaction. Every record is also kept verbatim in `records`."""
        raws = raws or [json.dumps(r, sort_keys=True) for r in recs]
        with self._write():
            self._import(recs, raws, scope)
        return len(recs)

    def export_records(self, scope: str = DEFAULT_SCOPE):
        for (raw,) in self._conn.execute("SELECT raw FROM records WHERE scope = ? ORDER BY seq", (scope,)):
            yield json.loads(raw)


_SETTLE = """
INSERT INTO calls (scope, call_id, settled, sproblem, smodel, outcome, tokens_in, tokens_out, cost, usage_known)
VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (scope, call_id) DO UPDATE SET settled = 1, sproblem = excluded.sproblem, smodel = excluded.smodel,
    outcome = excluded.outcome, tokens_in = excluded.tokens_in, tokens_out = excluded.tokens_out,
    cost = excluded.cost, usage_known = excluded.usage_known"""

_RESERVE_UPSERT = """
INSERT INTO calls (scope, call_id, has_reserve, rproblem, rmodel, reserved_tokens, reserved_cost) VALUES (?, ?, 1, ?, ?, ?, ?)
ON CONFLICT (scope, call_id) DO UPDATE SET has_reserve = 1, rproblem = excluded.rproblem, rmodel = excluded.rmodel,
    reserved_tokens = excluded.reserved_tokens, reserved_cost = excluded.reserved_cost"""


# ------------------------------------------------------------------------------------------------ the migration check

def _shape(summary: dict, acct: dict) -> dict:
    """The numbers that must be equal before and after a migration, with dollars rounded to a billionth."""
    return {"calls": summary["calls"], "tokens_in": summary["tokens_in"], "tokens_out": summary["tokens_out"],
            "cost_usd": round(summary["cost_usd"], 9), "accounted_calls": acct["calls"],
            "accounted_tokens": acct["tokens"], "accounted_cost": round(acct["cost"], 9),
            "unknown_cost": acct["unknown_cost"],
            "by_problem": {str(k): [v["calls"], v["tokens"], round(v["cost"], 9), v["unknown_cost"]]
                           for k, v in sorted(acct["problems"].items(), key=lambda kv: str(kv[0]))}}


def compare(directory, db: Ledger, scope: str = DEFAULT_SCOPE) -> dict:
    """The jsonl ledger's totals and the database's totals, and whether they are equal."""
    from orchard import telegraph as tg
    recs = _read_jsonl(directory)[0]                      # the old jsonl ledger, read as the old code read it
    want = _shape(tg.summarise_records(recs), tg.accounting(recs))
    got = _shape(db.totals(scope), db.accounting(scope))
    return {"equal": want == got, "jsonl": want, "sqlite": got}


def _read_jsonl(directory):
    path = Path(directory) / LEDGER_NAME
    recs, raws = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            recs.append(item)
            raws.append(line)
    return recs, raws


def import_and_compare(directory, db: Ledger, scope: str = DEFAULT_SCOPE) -> dict:
    """Import `telegraph.jsonl` from `directory` (read only) and check that the totals are the same afterwards."""
    recs, raws = _read_jsonl(directory)
    count = db.import_records(recs, raws, scope)
    return {**compare(directory, db, scope), "records_imported": count}


# ------------------------------------------------------------------------------------------------ I3: the cutover

def refuse_if_legacy_present(directory) -> None:
    """New code calls this before it uses a directory. The marker says the directory was migrated; a regular file (or a
    link) named telegraph.jsonl beside it means an old writer is running and building a second, incomplete history."""
    d = Path(directory)
    if os.path.lexists(d / MARKER):
        led = d / LEDGER_NAME
        if led.is_symlink() or led.is_file():
            raise LegacyLedgerPresent(f"{d} was migrated, but a live {LEDGER_NAME} is there again: an old writer is running; "
                                      "stop it, then reconcile before any call")


def migrate_directory(directory, db: Ledger, scope: str = DEFAULT_SCOPE) -> dict:
    """Import a telegraph directory into `db` and cut the old files over, so no old writer can go on.

    1. take the old lock (`telegraph.lock`, the lock the old code takes), so no old reservation is half written;
    2. import every record and compare the totals inside one transaction; a difference rolls everything back;
    3. rename `telegraph.jsonl` to `telegraph.jsonl.archived` (atomic) and make a DIRECTORY named `telegraph.jsonl`: an old
       process that opens it for appending gets EISDIR and refuses the call (the old code's "ledger cannot be written or
       read" path), so a request that was in flight cannot recreate the file with its settlement;
    4. write `MIGRATED-TO-SQLITE`. Reply files are left where they are. Nothing is deleted.
    The reservation of a request in flight at the cutover is in the import and counts at its worst case."""
    d = Path(directory)
    if os.path.lexists(d / MARKER):
        raise AlreadyMigrated(str(d))
    lock = os.open(d / LOCK_NAME, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with db._write():
            if db._conn.execute("SELECT 1 FROM meta WHERE key = ?", (f"migrated:{scope}",)).fetchone():
                raise AlreadyMigrated(f"scope {scope} was imported before")
            recs, raws = _read_jsonl(d)
            db._import(recs, raws, scope)
            result = compare(d, db, scope)
            if not result["equal"]:
                raise MigrationMismatch(json.dumps(result["jsonl"]) + " != " + json.dumps(result["sqlite"]))
            db._conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (f"migrated:{scope}", str(len(recs))))
        led = d / LEDGER_NAME
        sha = hashlib.sha256(led.read_bytes()).hexdigest()
        os.rename(led, d / ARCHIVE)
        os.mkdir(led, 0o700)
        (led / "README.txt").write_text("This ledger moved to the project database. An old writer that finds this directory "
                                        "refuses to run. See MIGRATED-TO-SQLITE.\n")
        (d / MARKER).write_text(json.dumps({"migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "archive": ARCHIVE,
                                            "sha256": sha, "records": len(recs), "scope": scope}) + "\n")
        return {**result, "records_imported": len(recs)}
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock, fcntl.LOCK_UN)
        os.close(lock)


# ------------------------------------------------------------------------------------------------ I6: an offset index

class LogIndex:
    """Running totals per append-only log (here: the sum of the integer `n` of each JSON line), with the offset reached.

    The contract. A stored total is reused only while the bytes it was built from are unchanged. On every poll:
      * a new inode, or a file smaller than the offset, restarts from zero;
      * the tail chunk (the bytes from the start of the chunk that holds the offset, up to the offset) is hashed and must
        equal its stored digest, so a truncate-and-regrow or an edit in the last chunk is caught, at a cost of one chunk;
      * `full=True` also hashes every complete chunk before it. That reads the whole consumed prefix, so it runs on a slow
        schedule (and on request) and does not run on every poll.
    What an ordinary poll cannot see is an edit of the same size in an earlier chunk. A full check sees it. The index is
    therefore labelled with when it was last fully checked, and nothing that decides a budget reads it."""

    def __init__(self, db: Ledger, chunk: int = 1 << 20):
        self.db, self.chunk = db, chunk

    def _tail_start(self, offset: int) -> int:
        """Where the tail chunk starts. The tail is never empty once something was consumed: when `offset` is an exact
        multiple of the chunk size, the tail is the whole last chunk, so an ordinary poll still validates it."""
        return ((offset - 1) // self.chunk) * self.chunk if offset > 0 else 0

    @staticmethod
    def _digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()[:32]

    def update(self, path, full: bool = False) -> dict:
        C = self.chunk
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            st = os.fstat(fd)
            row = self.db._conn.execute(
                "SELECT dev, inode, offset, total, tail_digest, digests FROM log_files WHERE path = ?", (str(path),)).fetchone()
            read = 0
            rescanned = False
            offset, total, digests, tail = 0, 0, [], b""
            if row is not None:
                dev, inode, offset, total, tail_digest, digests_json = row
                digests = json.loads(digests_json)
                tail_start = self._tail_start(offset)
                ok = (dev, inode) == (st.st_dev, st.st_ino) and st.st_size >= offset
                if ok:
                    tail = os.pread(fd, offset - tail_start, tail_start)
                    read += len(tail)
                    ok = self._digest(tail) == tail_digest
                if ok and full:
                    for i, want in enumerate(digests):
                        block = os.pread(fd, C, i * C)
                        read += len(block)
                        if self._digest(block) != want:
                            ok = False
                            break
                if not ok:
                    rescanned = True
                    offset, total, digests, tail = 0, 0, [], b""
            tail_start = self._tail_start(offset)
            new = os.pread(fd, max(st.st_size - offset, 0), offset)
            read += len(new)
            end = new.rfind(b"\n") + 1                            # a half-written last line is left for the next poll
            for raw_line in new[:end].splitlines():
                try:
                    total += int(json.loads(raw_line)["n"])
                except (ValueError, KeyError, TypeError):
                    pass
            offset += end
            body = tail + new[:end]                               # the bytes from tail_start to the new offset
            new_tail_start = self._tail_start(offset)
            complete = (new_tail_start - tail_start) // C         # whole chunks that now lie before the new tail
            digests = digests + [self._digest(body[k * C:(k + 1) * C]) for k in range(complete)]
            tail_digest = self._digest(body[complete * C:])
            with self.db._write():
                self.db._conn.execute(
                    "INSERT OR REPLACE INTO log_files (path, dev, inode, offset, total, tail_digest, digests) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", (str(path), st.st_dev, st.st_ino, offset, total, tail_digest,
                                                     json.dumps(digests)))
            return {"total": total, "offset": offset, "bytes_read": read, "rescanned": rescanned}
        finally:
            os.close(fd)


# ------------------------------------------------------------------------------------------------ test helpers (processes)

def worker(path, cfg_table, base, index, count, problems, tokens, cost, settle=False):
    """One process of the multi-process test: `count` reservations, optionally settled. Returns [(call_id, granted)]."""
    from orchard import telegraph as tg
    cfg = tg.load(cfg_table, base=Path(base))
    out = []
    with Ledger(path) as db:
        for k in range(count):
            call_id, problem = f"w{index}-{k}", problems[(index + k) % len(problems)]
            why = db.reserve(call_id, problem, "small", tokens, cost, cfg)
            out.append((call_id, why is None))
            if settle and why is None:
                db.settle(call_id, problem=problem, model="small", outcome="ok", tokens_in=60, tokens_out=40,
                          cost_usd=0.0, usage_known=True)
    return out


def _main(argv) -> int:
    cmd = argv[0]
    if cmd == "import-check":
        directory, db_path = Path(argv[1]), Path(argv[argv.index("--db") + 1])
        with Ledger(db_path) as db:
            result = import_and_compare(directory, db)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["equal"] else 1
    path = argv[1]
    with Ledger(path) as db:
        if cmd in ("hold-after-reserve", "hold-in-transaction"):
            call_id, problem, tokens, cost = argv[2], argv[3], int(argv[4]), float(argv[5])
            if cmd == "hold-after-reserve":
                db.reserve(call_id, problem, "small", tokens, cost, Limits())
                print("RESERVED", flush=True)
            else:
                def stall():
                    print("INSIDE", flush=True)
                    time.sleep(120)
                db.reserve(call_id, problem, "small", tokens, cost, Limits(), after_insert=stall)
            time.sleep(120)
        elif cmd == "storm":                                       # reserve and settle until killed; report each commit
            prefix, every = argv[2], int(argv[3])
            i = 0
            while True:
                call_id = f"{prefix}-{i}"
                db.reserve(call_id, "p", "small", 1000 + i % 7, 0.25, Limits())
                print("R", call_id, flush=True)
                if i % every:
                    db.settle(call_id, problem="p", model="small", outcome="ok", tokens_in=10, tokens_out=5,
                              cost_usd=0.0625, usage_known=bool(i % 2))
                    print("S", call_id, flush=True)
                i += 1
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
