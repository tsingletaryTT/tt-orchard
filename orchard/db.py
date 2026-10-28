# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The project database: one SQLite file in a state directory, and the only module (with orchard/telegraph_store.py) that
holds SQL.

The design is docs/superpowers/specs/2026-10-28-project-database-design.md. Its invariants are requirements here:

  I2  after any event that can lose history (a restore, a file moved aside, a new file beside backups) the ledger is
      unconfirmed. `reservation_block()` says so, and only `confirm_reconciled` clears it.
  I4  the directory is a real directory owned by this user and mode 0700; the file is a regular file of this user that
      is not a link. The checks run on descriptors.
  I5  only a failed integrity check or SQLite's own "malformed" / "not a database" moves a file aside. Every other error is a
      retryable `LedgerUnavailable` and leaves every file as it was.
  I7  recovery state is on disk before a replacement database exists: `RECOVERY-REQUIRED` is written (and synced) first, an
      exclusive lock on `open.lock` covers the whole open sequence, and every reservation is refused while the marker exists.
  I8  a restore is staged, marked and installed under exclusion (`db.users` is held shared by every open connection and
      exclusively by a restore).

Layout of a state directory (all names are fixed):
  project.db (+ -wal, -shm)   the database
  open.lock                   exclusive for the whole open sequence of any process
  db.users                    shared while a Database object is open; exclusive during a restore
  RECOVERY-REQUIRED           the recovery marker (JSON: reason, time, moved_to)
  backups/project-*.db        online backups, newest 14 kept
  project.db.corrupt-*        a damaged file moved aside (never deleted)
  project.db.replaced-*       a live file replaced by a restore (never deleted)

The telegraph directory is the state directory for the telegraph ledger. The path always comes from the caller (the
configuration or an environment variable), never from `Path.home()`.

Credentials never go in the database: no function here takes a key, and the telegraph store only receives text that has
been through `telegraph.redact`.
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
import threading
import time
from pathlib import Path
from typing import Callable

from orchard import safeio

DB_NAME = "project.db"
BUSY_TIMEOUT_MS = 30_000
MIN_SQLITE = (3, 37, 0)                  # STRICT tables
BACKUP_KEEP = 14
BACKUP_INTERVAL_S = 3600
MARKER_NAME = "RECOVERY-REQUIRED"
INIT_NAME = "INITIALIZED"             # written at the first open: proof that a ledger existed, whatever happens to the database
MARKER_MAX_BYTES = 64 * 1024
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_SIBLINGS = ("-wal", "-shm", "-journal")


class LedgerUnavailable(Exception):
    """The database cannot be used right now (busy, locked, full, I/O error, no permission, a restore in progress). Every file
    is as it was. Retry later; telegraph refuses the call and sends nothing, as it does when a ledger cannot be written."""


class LedgerRefused(Exception):
    """The database or its directory must not be used as it is (a link, another user's directory, a newer schema, a recovery
    that a person has to confirm, a legacy ledger). A retry does not help; the message says what to do."""


class UnsafeDirectory(LedgerRefused):
    """The state directory or a file in it is a link, belongs to someone else, or is not the kind of file it must be."""


class DatabaseTooNew(LedgerRefused):
    """The file was written by a newer version of the program."""


class RecoveryRequired(LedgerRefused):
    """`RECOVERY-REQUIRED` is present: no reservation is allowed until a person confirms the history."""


class _Corrupt(Exception):
    """Internal: SQLite said the file is damaged, or the integrity check failed."""


# ----------------------------------------------------------------------------------------------------------- schema

def _migrate_1(conn: sqlite3.Connection) -> None:
    """Telegraph, version 1. `scope` is the first column of every key (I1). The event table is the source of truth; the record
    is kept as the jsonl line had it, so `db export telegraph` writes the old format."""
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) STRICT")
    conn.execute("""
        CREATE TABLE telegraph_events (
            seq      INTEGER PRIMARY KEY AUTOINCREMENT,
            scope    TEXT NOT NULL,
            ts       REAL,
            kind     TEXT NOT NULL CHECK (kind IN ('reserve', 'settle', 'refuse', 'cache_hit', 'report', 'torn', 'other')),
            call_id  TEXT,
            raw_json TEXT NOT NULL CHECK (length(raw_json) <= 4000000)
        ) STRICT""")
    conn.execute("CREATE INDEX telegraph_events_scope ON telegraph_events (scope, seq)")
    conn.execute("""
        CREATE TABLE telegraph_cache (
            scope           TEXT NOT NULL,
            identity_sha256 TEXT NOT NULL,
            call_id         TEXT NOT NULL,
            reply_file      TEXT NOT NULL,
            text            TEXT NOT NULL,
            PRIMARY KEY (scope, identity_sha256)
        ) STRICT""")


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [_migrate_1]   # entry i: version i -> i + 1. Never edit one that shipped.


# ---------------------------------------------------------------------------------------------------- error classes

def is_corruption(exc: BaseException) -> bool:
    """True only when SQLite says the file is damaged: SQLITE_CORRUPT (11) or SQLITE_NOTADB (26), or the message that goes with
    them. A busy or locked file, a full disk, an I/O error, a read-only file and a file that cannot be opened are not
    corruption, whatever exception class carries them (OperationalError is a kind of DatabaseError)."""
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None and (code & 0xFF) in (11, 26):
        return True
    text = str(exc).lower()
    return "malformed" in text or "not a database" in text


def _unavailable(exc: BaseException, what: str = "") -> LedgerUnavailable:
    text = f"{type(exc).__name__}: {exc}"
    return LedgerUnavailable((what + ": " if what else "") + text)


def _fsync_dir_fd(fd: int) -> None:
    os.fsync(fd)


# ------------------------------------------------------------------------------------------------ I4: the directory

def private_directory(directory) -> int:
    """Make `directory` a private state directory and return a descriptor for it (the caller closes it).

    The directory is created when missing (two processes may do this at once; losing the race is not an error). It is opened
    with O_DIRECTORY | O_NOFOLLOW, so a link or a file in its place is refused (`UnsafeDirectory`), and every check after that
    is on the descriptor: the owner must be this user (another user's directory is refused and left as it is), and a mode with
    any group or other bit is tightened to 0700 (a directory others can write could let them swap the files in it)."""
    directory = os.path.abspath(os.fspath(directory))
    try:
        os.makedirs(directory, mode=0o700, exist_ok=True)
    except FileExistsError:
        pass                                                      # a file or link is in the way: the open below says which
    except OSError as exc:
        raise LedgerUnavailable(f"cannot create {directory}: {exc.strerror or type(exc).__name__}") from exc
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _O_CLOEXEC)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UnsafeDirectory(f"{directory} is a link or not a directory") from exc
        raise LedgerUnavailable(f"cannot open {directory}: {exc.strerror or type(exc).__name__}") from exc
    try:
        st = os.fstat(fd)
        if st.st_uid != os.geteuid():
            raise UnsafeDirectory(f"{directory} belongs to another user")
        if stat.S_IMODE(st.st_mode) & 0o077:
            try:
                os.fchmod(fd, 0o700)
            except OSError as exc:
                raise LedgerUnavailable(f"cannot tighten the mode of {directory}: {exc.strerror}") from exc
    except BaseException:
        os.close(fd)
        raise
    return fd


def _check_sibling(dir_fd: int, name: str) -> None:
    """A -wal, -shm or -journal file that exists must be a regular file of this user and not a link (SQLite would follow it)."""
    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise LedgerUnavailable(f"cannot look at {name}: {exc.strerror}") from exc
    if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
        raise UnsafeDirectory(f"{name} is not a regular file of this user")


def _kind_of_name(dir_fd: int, name: str) -> str | None:
    if dir_fd < 0:
        return None
    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LedgerUnavailable(f"cannot look at {name}: {exc.strerror}") from exc
    return ("file" if stat.S_ISREG(st.st_mode) else "dir" if stat.S_ISDIR(st.st_mode)
            else "link" if stat.S_ISLNK(st.st_mode) else "other")


def _exists(dir_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise LedgerUnavailable(f"cannot look at {name}: {exc.strerror}") from exc


def _lock_file(dir_fd: int, name: str, how: int, timeout_s: float, *, create: bool) -> int:
    """Open a lock file under a directory descriptor and take `how` (a flock) without blocking for longer than `timeout_s`. With
    `create` False a missing lock file gives -1 (a reader never creates one). Returns the descriptor; closing it releases the lock."""
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | _O_CLOEXEC | (os.O_CREAT if create else 0)
    try:
        fd = os.open(name, flags, 0o600, dir_fd=dir_fd)
    except FileNotFoundError:
        if not create:
            return -1
        raise LedgerUnavailable(f"cannot open {name}: the directory is gone") from None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EISDIR):
            raise UnsafeDirectory(f"{name} is a link or a directory") from exc
        raise LedgerUnavailable(f"cannot open {name}: {exc.strerror or type(exc).__name__}") from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
            raise UnsafeDirectory(f"{name} is not a regular file of this user")
        if create:
            os.fchmod(fd, 0o600)
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, how | fcntl.LOCK_NB)
                return fd
            except (BlockingIOError, PermissionError):
                if time.monotonic() >= deadline:
                    raise LedgerUnavailable(f"{name} is held by another process") from None
                time.sleep(0.01)
            except OSError as exc:
                raise LedgerUnavailable(f"cannot lock {name}: {exc.strerror}") from exc
    except BaseException:
        os.close(fd)
        raise


class Database:
    """One SQLite file in a state directory. Opening it runs the whole sequence: private directory, `open.lock`, the marker, the
    file checks, pragmas, the integrity check, recovery when needed, and migrations. One connection is made per thread.

    `hook(step)` is a test seam: it is called after each recovery and migration step (names like "recover:2") and may raise to
    simulate a crash at that point."""

    def __init__(self, directory, name: str = DB_NAME, *, busy_timeout_ms: int = BUSY_TIMEOUT_MS, hook=None,
                 backup_interval_s: float | None = BACKUP_INTERVAL_S, memory: bool = False):
        if sqlite3.sqlite_version_info < MIN_SQLITE:
            raise LedgerRefused(f"SQLite {sqlite3.sqlite_version} is too old; STRICT tables need "
                                f"{'.'.join(map(str, MIN_SQLITE))} or newer")
        self.name = name
        self._busy = int(busy_timeout_ms)
        self._hook = hook or (lambda step: None)
        self._backup_interval = backup_interval_s
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._closed = False
        self.moved_aside: str | None = None
        self._dirfd = -1
        self._open_lock = -1
        self._users = -1
        self._file_id: tuple | None = None
        if memory:
            self.directory = ""
            self._memory = True
            conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
            conn.execute("PRAGMA foreign_keys = ON")
            self._local.conn = conn
            self._conns.append(conn)
            self._migrate()
            return
        self._memory = False
        self.directory = os.path.abspath(os.fspath(directory))
        try:
            self._dirfd = private_directory(self.directory)
            self._open_sequence()
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------------------------------------- open sequence

    def _open_sequence(self) -> None:
        self._open_lock = self._flock("open.lock", fcntl.LOCK_EX)
        try:
            self._users = self._flock("db.users", fcntl.LOCK_SH)
            marker = self._marker()                                # I7: the marker is read first
            exists = _exists(self._dirfd, self.name)
            for suffix in _SIBLINGS:
                _check_sibling(self._dirfd, self.name + suffix)
            has_history = bool(self.list_backups()) or _exists(self._dirfd, INIT_NAME)
            if not exists and (marker is not None or has_history):
                reason = (marker or {}).get("reason") or "the database file was missing but this directory had a ledger before (backups or INITIALIZED)"
                self._recover(reason, move=False)
            else:
                keep = self._keep_wal()
                try:
                    self._conn_main = self._open_checked()
                    self._local.conn = self._conn_main
                    self._register(self._conn_main)
                except _Corrupt as exc:
                    self._recover(f"the database was damaged ({exc})", move=True, keep_wal=keep)
                    keep = None
                finally:
                    self._drop_keep(keep)
            self._migrate()
            self._check_schema()
            if self._marker() is not None and self.unreconciled() is None:
                self._write_unreconciled("recovery marker present (" + str((self._marker() or {}).get("reason", ""))[:200] + ")")
            self._maybe_backup()
            if not _exists(self._dirfd, INIT_NAME):                 # durable evidence that must exist before the first reservation
                try:
                    safeio.atomic_write(os.path.join(self.directory, INIT_NAME), json.dumps(
                        {"initialized": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n", mode=0o600)
                except OSError as exc:
                    raise _unavailable(exc, "cannot record that the ledger exists") from exc
        finally:
            fcntl.flock(self._open_lock, fcntl.LOCK_UN)
            os.close(self._open_lock)
            self._open_lock = -1

    def _flock(self, name: str, how: int, timeout_s: float | None = None) -> int:
        """Open a lock file under the directory descriptor and take `how` without blocking forever: it waits for the busy timeout."""
        return _lock_file(self._dirfd, name, how, self._busy / 1000 if timeout_s is None else timeout_s, create=True)

    def _open_checked(self, name: str | None = None) -> sqlite3.Connection:
        """Open (and create) the file without following a link, check it on its descriptor, set the pragmas, run the integrity
        check. `_Corrupt` means damaged; every other failure is LedgerUnavailable or LedgerRefused."""
        name = name or self.name
        try:
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | _O_CLOEXEC, 0o600, dir_fd=self._dirfd)
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EISDIR):
                raise UnsafeDirectory(f"{name} is a link or a directory and cannot be the database file") from exc
            raise LedgerUnavailable(f"cannot open {name}: {exc.strerror or type(exc).__name__}") from exc
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
                raise UnsafeDirectory(f"{name} is not a regular file of this user")
            os.fchmod(fd, 0o600)                                   # an older file with a loose mode is tightened
            file_id = (st.st_dev, st.st_ino)
        finally:
            os.close(fd)
        path = os.path.join(self.directory, name)
        try:
            conn = sqlite3.connect(path, isolation_level=None, timeout=self._busy / 1000, check_same_thread=False)
        except sqlite3.Error as exc:
            raise self._classified(exc) from exc
        try:
            self._configure(conn)
            verdict = self._retry_while_busy(lambda: conn.execute("PRAGMA integrity_check").fetchall())
        except sqlite3.Error as exc:
            conn.close()
            raise self._classified(exc) from exc
        except BaseException:
            conn.close()
            raise
        if verdict != [("ok",)]:
            conn.close()
            raise _Corrupt("integrity_check failed")
        try:                                                       # the path still names the file that was checked
            now = os.stat(name, dir_fd=self._dirfd, follow_symlinks=False)
            if (now.st_dev, now.st_ino) != file_id or not stat.S_ISREG(now.st_mode):
                raise LedgerUnavailable(f"{name} changed while it was being opened")
        except OSError as exc:
            conn.close()
            raise LedgerUnavailable(f"{name} disappeared while it was being opened") from exc
        except LedgerUnavailable:
            conn.close()
            raise
        if name == self.name:
            self._file_id = file_id
        return conn

    def _configure(self, conn: sqlite3.Connection) -> None:
        conn.execute(f"PRAGMA busy_timeout = {int(self._busy)}")
        # Switching a new file to WAL needs an exclusive lock and can answer "locked" at once, without waiting for the busy
        # timeout, when another process does the same. The switch is retried until the busy timeout has passed.
        mode = self._retry_while_busy(lambda: conn.execute("PRAGMA journal_mode = WAL").fetchone())
        if not mode or str(mode[0]).lower() != "wal":
            raise LedgerUnavailable("the database cannot use WAL (a network file system?); it must be on a local disk")
        conn.execute("PRAGMA synchronous = FULL")                  # a reservation survives a power cut, as fsync did
        conn.execute("PRAGMA foreign_keys = ON")

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
        if is_corruption(exc):
            return _Corrupt(str(exc))
        return _unavailable(exc)

    def _register(self, conn: sqlite3.Connection) -> None:
        with self._conns_lock:
            self._conns.append(conn)

    def conn(self) -> sqlite3.Connection:
        """This thread's connection (made on first use with the same pragmas)."""
        if self._closed:
            raise LedgerUnavailable("the database is closed")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            try:
                conn = sqlite3.connect(os.path.join(self.directory, self.name), isolation_level=None,
                                       timeout=self._busy / 1000, check_same_thread=False)
                self._configure(conn)
            except sqlite3.Error as exc:
                raise _unavailable(exc) from exc
            self._local.conn = conn
            self._register(conn)
        return conn

    def close(self) -> None:
        self._closed = True
        with self._conns_lock:
            conns, self._conns = self._conns, []
        for conn in conns:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
        self._local = threading.local()
        for attr in ("_users", "_open_lock"):
            fd = getattr(self, attr, -1)
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)                                  # closing a descriptor releases its flock
                setattr(self, attr, -1)
        if self._dirfd >= 0:
            with contextlib.suppress(OSError):
                os.close(self._dirfd)
            self._dirfd = -1

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------------------------------------- transactions

    @contextlib.contextmanager
    def write(self):
        """A writer. `BEGIN IMMEDIATE` takes the write lock at once, so a read and the write that depends on it cannot be
        split by another writer. Commits on success and rolls back on any exception. A busy, locked, full or failing database
        is a retryable `LedgerUnavailable` and changes nothing."""
        conn = self.conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        try:
            yield conn
        except BaseException as exc:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            if isinstance(exc, sqlite3.Error):                      # disk full, I/O, read-only, a constraint: refuse, retry later
                raise _unavailable(exc) from exc
            raise
        try:
            conn.execute("COMMIT")
        except sqlite3.Error as exc:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise _unavailable(exc) from exc

    @contextlib.contextmanager
    def read(self):
        """One read transaction: every query inside sees the same snapshot (the summary, the accounting and the rows describe one
        moment)."""
        conn = self.conn()
        try:
            conn.execute("BEGIN")
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        try:
            yield conn
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        finally:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")

    # --------------------------------------------------------------------------------------------------- migrations

    def user_version(self) -> int:
        return self.conn().execute("PRAGMA user_version").fetchone()[0]

    def _migrate(self) -> None:
        have = self.user_version()
        if have > len(MIGRATIONS):
            raise DatabaseTooNew(f"{self.name} has schema version {have}; this program knows {len(MIGRATIONS)}")
        backed_up = False
        while have < len(MIGRATIONS):
            if have > 0 and not backed_up and not self._memory:
                self._backup_quietly()                              # before a migration runs, the file is backed up
                backed_up = True
            with self.write() as conn:
                have = conn.execute("PRAGMA user_version").fetchone()[0]     # another process may have migrated meanwhile
                if have > len(MIGRATIONS):                          # ... or a newer program did: checked after the lock is held
                    raise DatabaseTooNew(f"{self.name} has schema version {have}; this program knows {len(MIGRATIONS)}")
                if have >= len(MIGRATIONS):
                    break
                MIGRATIONS[have](conn)
                conn.execute(f"PRAGMA user_version = {have + 1}")
            have = self.user_version()
        self._hook("migrate:done")

    REQUIRED_TABLES = ("meta", "telegraph_events", "telegraph_cache")

    def _check_schema(self) -> None:
        """A file at the current schema version must hold its tables. A table that is missing is not corruption (the integrity check
        passes), so nothing is moved aside; the file is refused and a person restores a backup."""
        have = {r[0] for r in self.conn().execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        missing = [t for t in self.REQUIRED_TABLES if t not in have]
        if missing:
            raise LedgerRefused(f"{self.name} is missing {', '.join(missing)} although its schema version is current; restore a "
                                "backup with `tt-orchard db restore`")

    # ------------------------------------------------------------------------------------------------------- meta

    def meta_get(self, key: str, conn: sqlite3.Connection | None = None) -> str | None:
        row = (conn or self.conn()).execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def meta_set(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                     (key, value))

    def meta_delete(self, conn: sqlite3.Connection, key: str) -> None:
        conn.execute("DELETE FROM meta WHERE key = ?", (key,))

    # ------------------------------------------------------------------------------------- I7: the recovery marker

    def _marker(self) -> dict | None:
        """The recovery marker's content, or None when there is none. A marker that cannot be read is still a marker (a dict that
        says so): no reservation is allowed while it exists."""
        if not _exists(self._dirfd, MARKER_NAME):
            return None
        try:
            data = safeio.read_json(MARKER_NAME, max_bytes=MARKER_MAX_BYTES, anchor=self._dirfd, expect=dict)
        except safeio.SafeReadError:
            return {"reason": "the recovery marker cannot be read", "time": None, "moved_to": None}
        return data

    def recovery_marker(self) -> dict | None:
        return self._marker() if not self._memory else None

    def exists_name(self, name: str) -> str | None:
        """'file', 'dir', 'link' or 'other' for a name in the state directory (never followed), None when absent."""
        return _kind_of_name(self._dirfd, name)

    def _write_marker(self, reason: str, moved_to: str | None = None) -> None:
        body = json.dumps({"reason": reason[:400], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "moved_to": moved_to,
                           "pid": os.getpid()}, sort_keys=True) + "\n"
        try:
            safeio.atomic_write(os.path.join(self.directory, MARKER_NAME), body, mode=0o600)     # syncs the file and the directory
        except OSError as exc:
            raise _unavailable(exc, "cannot write the recovery marker") from exc

    def _keep_wal(self) -> str | None:
        """A second name (a hard link) for an existing `-wal` file. SQLite deletes a `-wal` when the connection that failed to open
        a damaged file closes, and that file can hold commits the damaged main file lacks; the link keeps its content for a person."""
        if _kind_of_name(self._dirfd, self.name + "-wal") != "file":
            return None
        keep = f"{self.name}-wal.keep-{os.getpid()}"
        try:
            os.link(self.name + "-wal", keep, src_dir_fd=self._dirfd, dst_dir_fd=self._dirfd, follow_symlinks=False)
        except OSError:
            return None
        return keep

    def _drop_keep(self, keep: str | None) -> None:
        if keep is not None:
            with contextlib.suppress(OSError):
                os.unlink(keep, dir_fd=self._dirfd)

    def _recover(self, reason: str, *, move: bool, keep_wal: str | None = None) -> None:
        """I7. Steps: 2 marker (synced), 3 move the damaged file and its -wal and -shm aside, 4 create the new file (from the newest
        backup that passes its check, else empty) and migrate it, 5 write the `unreconciled` row. A crash after any step is
        safe: the marker stays and refuses every reservation, whatever the file holds."""
        self._hook("recover:1")                                    # nothing has changed yet
        existing = self._marker()
        if existing is None:
            self._write_marker(reason)
        self._hook("recover:2")
        moved = None
        if move and _exists(self._dirfd, self.name):
            stamp = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
            moved = f"{self.name}.corrupt-{stamp}"
            n = 0
            while _exists(self._dirfd, moved):
                n += 1
                moved = f"{self.name}.corrupt-{stamp}-{n}"
            for suffix in ("",) + _SIBLINGS:                        # the main file first: the group is one unit to a person
                with contextlib.suppress(FileNotFoundError):
                    os.rename(self.name + suffix, moved + suffix, src_dir_fd=self._dirfd, dst_dir_fd=self._dirfd)
            if keep_wal is not None:
                if _exists(self._dirfd, moved + "-wal"):
                    self._drop_keep(keep_wal)                      # the live -wal was still there and moved with the rest
                else:                                              # SQLite deleted it: the link is the only copy left
                    with contextlib.suppress(OSError):
                        os.rename(keep_wal, moved + "-wal", src_dir_fd=self._dirfd, dst_dir_fd=self._dirfd)
            _fsync_dir_fd(self._dirfd)
            self.moved_aside = moved
            if existing is None:
                self._write_marker(reason, moved_to=moved)         # same content plus the name, written before anything is created
        self._hook("recover:3")
        restored_from = self._install_newest_backup(reason)
        if restored_from is None:
            self._conn_main = self._open_checked()
            self._local.conn = self._conn_main
            self._register(self._conn_main)
            self._migrate()
        self._hook("recover:4")
        note = reason + (f"; restored from backup {restored_from}, calls made since are not in it" if restored_from else
                         "; started empty" + (f" (the old file is {moved})" if moved else ""))
        self._write_unreconciled(note)
        self._hook("recover:5")

    def _install_newest_backup(self, reason: str) -> str | None:
        """Stage the newest backup that passes its integrity check, mark it unconfirmed inside the staged file, and publish it with one
        rename. Returns the backup's name, or None when there is no usable backup (nothing was changed then)."""
        for backup in reversed(self.list_backups()):
            staged = self._stage_backup(os.path.join(self.directory, "backups", backup), reason)
            if staged is None:
                continue
            try:
                os.rename(staged, self.name, src_dir_fd=self._dirfd, dst_dir_fd=self._dirfd)
                _fsync_dir_fd(self._dirfd)
            except OSError as exc:
                with contextlib.suppress(OSError):
                    os.unlink(staged, dir_fd=self._dirfd)
                raise _unavailable(exc, "cannot install the restored file") from exc
            self._conn_main = self._open_checked()
            self._local.conn = self._conn_main
            self._register(self._conn_main)
            return backup
        return None

    def _stage_backup(self, backup_path: str, reason: str, staged_name: str | None = None) -> str | None:
        """Copy `backup_path` (with the online backup API) to a private staging file in the directory, check it, mark it
        unconfirmed, and return the staged name. None when the backup does not pass (a newer schema included)."""
        staged_name = staged_name or f"{self.name}.restore-{os.getpid()}-{time.monotonic_ns()}"
        partial = staged_name + ".partial"
        for leftover in (partial,):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(leftover, dir_fd=self._dirfd)
        src = dst = None
        try:
            fd = os.open(partial, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _O_CLOEXEC, 0o600, dir_fd=self._dirfd)
            os.close(fd)
            src = sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True, isolation_level=None, timeout=self._busy / 1000)
            if src.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise _Corrupt("the backup fails its integrity check")
            if src.execute("PRAGMA user_version").fetchone()[0] > len(MIGRATIONS):
                raise DatabaseTooNew("the backup has a newer schema")
            dst = sqlite3.connect(os.path.join(self.directory, partial), isolation_level=None, timeout=self._busy / 1000)
            src.backup(dst)
            dst.execute("PRAGMA journal_mode = DELETE")             # the staged file is one file, with no -wal beside it
            if dst.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise _Corrupt("the staged copy fails its integrity check")
            dst.execute("BEGIN IMMEDIATE")
            self.meta_set(dst, "unreconciled", reason[:400])
            dst.execute("COMMIT")
            dst.close()
            dst = None
            sfd = os.open(partial, os.O_RDWR | os.O_NOFOLLOW | _O_CLOEXEC, dir_fd=self._dirfd)
            try:
                os.fchmod(sfd, 0o600)
                os.fsync(sfd)
            finally:
                os.close(sfd)
            os.rename(partial, staged_name, src_dir_fd=self._dirfd, dst_dir_fd=self._dirfd)
            _fsync_dir_fd(self._dirfd)
            return staged_name
        except (sqlite3.Error, _Corrupt, DatabaseTooNew, OSError):
            with contextlib.suppress(FileNotFoundError, OSError):
                os.unlink(partial, dir_fd=self._dirfd)
            return None
        finally:
            for c in (src, dst):
                if c is not None:
                    with contextlib.suppress(sqlite3.Error):
                        c.close()

    # ------------------------------------------------------------------------------------- I2: the unconfirmed state

    def _write_unreconciled(self, reason: str) -> None:
        with self.write() as conn:
            self.meta_set(conn, "unreconciled", reason[:400])

    def unreconciled(self) -> str | None:
        return self.meta_get("unreconciled")

    def reservation_block(self, conn: sqlite3.Connection | None = None) -> str | None:
        """Why no reservation may be made, or None. Two sources: the `RECOVERY-REQUIRED` file (the authority: it exists even when
        the database is missing or fresh) and the `unreconciled` row (its mirror)."""
        marker = None if self._memory else self._marker()
        row = self.meta_get("unreconciled", conn)
        if marker is None and row is None:
            return None
        reason = row or (marker or {}).get("reason") or "recovery is required"
        return (f"the ledger's history may be incomplete ({reason}), so no outside call is allowed until a person confirms it "
                "with `tt-orchard db confirm-restore`")

    def confirm_reconciled(self, note: str) -> bool:
        """A person says the history is accounted for. Only this clears the state. The row is deleted first and the marker second;
        a crash between the two leaves the marker, which keeps refusing, and confirming again finishes. True when there was
        something to clear."""
        had = self.unreconciled() is not None or (not self._memory and self._marker() is not None)
        with self.write() as conn:
            self.meta_delete(conn, "unreconciled")
            self.meta_set(conn, "confirmed", f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {note}"[:300])
        self._hook("confirm:row")
        if not self._memory:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(MARKER_NAME, dir_fd=self._dirfd)
            _fsync_dir_fd(self._dirfd)
        self._hook("confirm:marker")
        return had

    # ------------------------------------------------------------------------------------------------------ backups

    def _backup_dir(self) -> str:
        return os.path.join(self.directory, "backups")

    def list_backups(self) -> list[str]:
        """Backup file names, oldest first (the names sort by time)."""
        try:
            names = os.listdir(self._backup_dir())
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise LedgerUnavailable(f"cannot list the backups: {exc.strerror}") from exc
        return sorted(n for n in names if n.startswith("project-") and n.endswith(".db"))

    def backup(self) -> str:
        """An online backup (the SQLite backup API) into `backups/`, private, with a name that sorts by time. Returns its path.
        The newest BACKUP_KEEP are kept."""
        bdir_fd = private_directory(self._backup_dir())
        try:
            name = f"project-{time.strftime('%Y%m%dT%H%M%S')}-{time.monotonic_ns()}.db"
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _O_CLOEXEC, 0o600, dir_fd=bdir_fd)
            os.close(fd)
            target = sqlite3.connect(os.path.join(self._backup_dir(), name), isolation_level=None)
            try:
                self.conn().backup(target)
            except sqlite3.Error as exc:
                target.close()
                with contextlib.suppress(OSError):
                    os.unlink(name, dir_fd=bdir_fd)
                raise _unavailable(exc, "backup failed") from exc
            target.close()
            sfd = os.open(name, os.O_RDWR | os.O_NOFOLLOW | _O_CLOEXEC, dir_fd=bdir_fd)
            try:
                os.fchmod(sfd, 0o600)
                os.fsync(sfd)
            finally:
                os.close(sfd)
            os.fsync(bdir_fd)
            for old in self.list_backups()[:-BACKUP_KEEP]:
                with contextlib.suppress(OSError):
                    os.unlink(old, dir_fd=bdir_fd)
            return os.path.join(self._backup_dir(), name)
        finally:
            os.close(bdir_fd)

    def _backup_quietly(self) -> None:
        try:
            self.backup()
        except (LedgerUnavailable, LedgerRefused, OSError):
            pass                                                   # a backup that fails never blocks the work it protects

    def _maybe_backup(self) -> None:
        """Take a backup when the newest is older than the interval and the database holds something. Nothing is backed up while
        the file is unconfirmed (a restored file is not a new fact)."""
        if self._backup_interval is None or self._memory or self.unreconciled() is not None:
            return
        names = self.list_backups()
        if names:
            try:
                age = time.time() - os.stat(os.path.join(self._backup_dir(), names[-1])).st_mtime
            except OSError:
                age = self._backup_interval + 1
            if age < self._backup_interval:
                return
        empty = self.conn().execute("SELECT NOT EXISTS (SELECT 1 FROM telegraph_events)").fetchone()[0]
        if not empty:
            self._backup_quietly()

    # ---------------------------------------------------------------------------------------------- facts about it

    def integrity_ok(self) -> bool:
        return self.conn().execute("PRAGMA integrity_check").fetchall() == [("ok",)]

    def pragmas(self) -> dict:
        c = self.conn()
        return {"journal_mode": c.execute("PRAGMA journal_mode").fetchone()[0],
                "synchronous": c.execute("PRAGMA synchronous").fetchone()[0],
                "busy_timeout": c.execute("PRAGMA busy_timeout").fetchone()[0],
                "foreign_keys": c.execute("PRAGMA foreign_keys").fetchone()[0]}

    def status(self) -> dict:
        c = self.conn()
        return {"path": os.path.join(self.directory, self.name), "schema_version": self.user_version(),
                "events": c.execute("SELECT COUNT(*) FROM telegraph_events").fetchone()[0],
                "unconfirmed": self.unreconciled(), "recovery_marker": self.recovery_marker(),
                "backups": self.list_backups(), "pragmas": self.pragmas()}


# --------------------------------------------------------------------------------------------------- read only access

class ReadOnlyDatabase:
    """A read-only view for pages and reports: it never creates, recovers, migrates or moves a file. `exists` says whether the file
    is there. Any problem is a LedgerUnavailable or LedgerRefused that the caller shows as "the spend is unknown"."""

    def __init__(self, directory, name: str = DB_NAME, *, busy_timeout_ms: int = 5000):
        self.directory = os.path.abspath(os.fspath(directory))
        self.name = name
        self._busy = busy_timeout_ms
        self._conn: sqlite3.Connection | None = None
        self._users = -1
        try:
            self._dirfd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _O_CLOEXEC)
        except FileNotFoundError:
            self._dirfd = -1
            self.exists = False
            return
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                raise UnsafeDirectory(f"{self.directory} is a link or not a directory") from exc
            raise LedgerUnavailable(f"cannot open {self.directory}: {exc.strerror}") from exc
        try:
            if os.fstat(self._dirfd).st_uid != os.geteuid():
                raise UnsafeDirectory(f"{self.directory} belongs to another user")
            try:
                st = os.stat(name, dir_fd=self._dirfd, follow_symlinks=False)
            except FileNotFoundError:
                self.exists = False
                return
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
                raise UnsafeDirectory(f"{name} is not a regular file of this user")
            for suffix in _SIBLINGS:
                _check_sibling(self._dirfd, name + suffix)
            self.exists = True
            # Coordinate with a recovery or restore: wait for open.lock, then hold db.users shared until this reader closes, so a
            # restore cannot move the files under an open snapshot. A reader creates no lock file.
            gate = _lock_file(self._dirfd, "open.lock", fcntl.LOCK_EX, busy_timeout_ms / 1000, create=False)
            try:
                self._users = _lock_file(self._dirfd, "db.users", fcntl.LOCK_SH, busy_timeout_ms / 1000, create=False)
            finally:
                if gate >= 0:
                    os.close(gate)
        except BaseException:
            self.close()
            raise

    def marker(self) -> dict | None:
        if self._dirfd < 0 or not _exists(self._dirfd, MARKER_NAME):
            return None
        try:
            return safeio.read_json(MARKER_NAME, max_bytes=MARKER_MAX_BYTES, anchor=self._dirfd, expect=dict)
        except safeio.SafeReadError:
            return {"reason": "the recovery marker cannot be read"}

    def exists_name(self, name: str) -> str | None:
        return _kind_of_name(self._dirfd, name)

    def file_stats(self) -> tuple:
        """(size, mtime_ns) of the database and its -wal: a cheap key for "has it changed"."""
        out = []
        for suffix in ("", "-wal"):
            try:
                st = os.stat(self.name + suffix, dir_fd=self._dirfd, follow_symlinks=False)
                out.append((st.st_ino, st.st_size, st.st_mtime_ns))
            except OSError:
                out.append(None)
        return tuple(out)

    @contextlib.contextmanager
    def read(self):
        """One read transaction over the file (a snapshot). Raises LedgerUnavailable when the file cannot be read."""
        if not self.exists:
            raise LedgerUnavailable("there is no database file")
        if self._conn is None:
            try:
                self._conn = sqlite3.connect(f"file:{os.path.join(self.directory, self.name)}?mode=ro", uri=True,
                                             isolation_level=None, timeout=self._busy / 1000)
                self._conn.execute(f"PRAGMA busy_timeout = {int(self._busy)}")
            except sqlite3.Error as exc:
                raise _unavailable(exc) from exc
        conn = self._conn
        try:
            conn.execute("BEGIN")
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        try:
            if conn.execute("PRAGMA user_version").fetchone()[0] > len(MIGRATIONS):
                raise DatabaseTooNew(f"{self.name} has a newer schema than this program knows ({len(MIGRATIONS)})")
            yield conn
        except sqlite3.Error as exc:
            raise _unavailable(exc) from exc
        finally:
            with contextlib.suppress(sqlite3.Error):
                conn.execute("ROLLBACK")

    def close(self) -> None:
        if self._conn is not None:
            with contextlib.suppress(sqlite3.Error):
                self._conn.close()
            self._conn = None
        if getattr(self, "_users", -1) >= 0:                       # released after SQLite is closed
            with contextlib.suppress(OSError):
                os.close(self._users)
            self._users = -1
        if getattr(self, "_dirfd", -1) >= 0:
            with contextlib.suppress(OSError):
                os.close(self._dirfd)
            self._dirfd = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------------------------------- I8: a restore

def restore_backup(directory, backup_path, name: str = DB_NAME, *, hook=None, busy_timeout_ms: int = 5000) -> str:
    """Put a backup in place of the database. Needs no other process to have the database open: it takes `db.users` exclusively
    without waiting and refuses ("close the other processes") when it cannot. Returns the name of the file it replaced, or ''.

    Steps (a crash after any of them leaves every reservation refused, because the marker is written before anything else):
      1 take `open.lock`, then `db.users` exclusively (no waiting)
      2 write `RECOVERY-REQUIRED` (reason: restore), synced
      3 stage: copy the backup to `<name>.restore-<id>` (0600, synced), check it, put the `unreconciled` row inside it
      4 move `-wal`, `-shm` and then the live file to `.replaced-*` as one group (the old file is never deleted)
      5 rename the staged file to the live name and sync the directory
      6 release the locks"""
    hook = hook or (lambda step: None)
    backup_path = os.path.abspath(os.fspath(backup_path))
    directory = os.path.abspath(os.fspath(directory))
    try:
        bst = os.stat(backup_path)
    except OSError as exc:
        raise LedgerRefused(f"cannot read the backup {backup_path}: {exc.strerror}") from exc
    if not stat.S_ISREG(bst.st_mode):
        raise LedgerRefused(f"{backup_path} is not a regular file")
    # A backup that is damaged or too new is refused before anything is touched.
    probe = None
    try:
        probe = sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True, isolation_level=None)
        if probe.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise LedgerRefused(f"{backup_path} fails its integrity check")
        if probe.execute("PRAGMA user_version").fetchone()[0] > len(MIGRATIONS):
            raise DatabaseTooNew(f"{backup_path} has a newer schema than this program knows")
    except sqlite3.Error as exc:
        raise LedgerRefused(f"{backup_path} is not a usable backup ({type(exc).__name__})") from exc
    finally:
        if probe is not None:
            probe.close()
    db = Database.__new__(Database)                                # the staging helpers need the directory state and no connection
    db.name, db._busy, db._hook = name, busy_timeout_ms, hook
    db.directory = directory
    db._memory, db._closed, db.moved_aside = False, False, None
    db._local, db._conns, db._conns_lock = threading.local(), [], threading.Lock()
    db._dirfd, db._open_lock, db._users, db._file_id = -1, -1, -1, None
    try:
        db._dirfd = private_directory(directory)
        db._open_lock = db._flock("open.lock", fcntl.LOCK_EX)
        try:
            db._users = db._flock("db.users", fcntl.LOCK_EX, timeout_s=0)
        except LedgerUnavailable:
            raise LedgerRefused("another process has the database open; close the other processes (the supervisor, the web page, "
                                "any telegraph command) and run the restore again") from None
        hook("restore:1")
        reason = f"restored from backup {os.path.basename(backup_path)}; calls made since are not in it"
        if db._marker() is None:
            db._write_marker(reason, moved_to=None)
        hook("restore:2")
        ident = hashlib.sha256(f"{backup_path}|{bst.st_size}|{bst.st_mtime_ns}".encode()).hexdigest()[:12]
        staged = f"{name}.restore-{ident}"
        reuse = False
        if _exists(db._dirfd, staged):
            check = None
            try:
                check = sqlite3.connect(f"file:{os.path.join(directory, staged)}?mode=ro", uri=True, isolation_level=None)
                reuse = (check.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
                         and check.execute("SELECT 1 FROM meta WHERE key = 'unreconciled'").fetchone() is not None)
            except sqlite3.Error:
                reuse = False
            finally:
                if check is not None:
                    check.close()
            if not reuse:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(staged, dir_fd=db._dirfd)
        if not reuse:
            if db._stage_backup(backup_path, reason, staged_name=staged) is None:
                raise LedgerRefused(f"the backup {backup_path} could not be staged")
        hook("restore:3")
        replaced = ""
        if _exists(db._dirfd, name):
            stamp = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
            replaced = f"{name}.replaced-{stamp}"
            n = 0
            while _exists(db._dirfd, replaced):
                n += 1
                replaced = f"{name}.replaced-{stamp}-{n}"
            for suffix in ("-wal", "-shm") + ("",):
                with contextlib.suppress(FileNotFoundError):
                    os.rename(name + suffix, replaced + suffix, src_dir_fd=db._dirfd, dst_dir_fd=db._dirfd)
            _fsync_dir_fd(db._dirfd)
        hook("restore:4")
        os.rename(staged, name, src_dir_fd=db._dirfd, dst_dir_fd=db._dirfd)
        _fsync_dir_fd(db._dirfd)
        hook("restore:5")
        return replaced
    finally:
        db.close()
