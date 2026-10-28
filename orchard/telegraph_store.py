# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The telegraph ledger on the project database (orchard/db.py): reservations, settlements, refusals, cache hits, verdicts,
torn notes and the answer cache. With db.py this is the only module that holds SQL.

What is stored. `telegraph_events` has one row per ledger record, in order (`seq`), with the record exactly as the jsonl line
had it (`raw_json`, `json.dumps(sort_keys=True)`). It is the source of truth: `db export telegraph` writes the old lines back, and the
accounting is computed from it by `orchard.telegraph.accounting`, the same function the jsonl ledger used, adding in ledger
order. That keeps every total, every refusal sentence and every floating-point sum equal to what the file gave (issue 153).
`telegraph_cache` holds one row per answer, keyed by (scope, identity) (I1). `scope` is the first column of every key and every
lookup; a ledger directory uses the scope "default" unless the caller names another.

Transactions. A reservation is ONE `BEGIN IMMEDIATE` transaction: check the guards, read the records, apply the caps through
`decide`, insert the reservation (or the refusal). The write lock is never held across a network call. A settlement is one
transaction that inserts the settlement and the cache row together, so an answer is never cached without its call.

Guards (I3, I10, I2). A write or a read refuses while the directory is in a state that would split the history: a live
`telegraph.jsonl` file (an old writer, or a ledger that was never migrated), a cutover that has not reached `complete`, or
`MIGRATING-TO-SQLITE` on disk. A reservation is also refused while the ledger is unconfirmed (db.py).

Nothing here reads a credential or a prompt. The caller passes records that have already been through `telegraph.redact`.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Callable

from orchard import db as dbm
from orchard.db import (Database, DatabaseTooNew, LedgerRefused, LedgerUnavailable, ReadOnlyDatabase)

DEFAULT_SCOPE = "default"
LEDGER = "telegraph.jsonl"
ARCHIVE = "telegraph.jsonl.archived"
LOCK = "telegraph.lock"
MARKER_MIGRATING = "MIGRATING-TO-SQLITE"
MARKER_MIGRATED = "MIGRATED-TO-SQLITE"
KINDS = ("reserve", "settle", "refuse", "cache_hit", "report", "torn", "other")


class LegacyLedgerPresent(LedgerRefused):
    """A live `telegraph.jsonl` is in a directory that is, or is being, migrated: an old writer is running."""


class NotMigrated(LedgerRefused):
    """The directory still holds a jsonl ledger. Run `tt-orchard db migrate telegraph --dir <directory>`."""


class MigrationIncomplete(LedgerRefused):
    """A migration of this directory started and did not finish. Run `tt-orchard db migrate telegraph --dir <directory>` again."""


def cutover_key(scope: str) -> str:
    return f"cutover:{scope}"


def kind_of(rec: dict) -> str:
    """The event kind of a record (the jsonl line's shape): reserve, settle (a request that was sent), refuse, cache_hit,
    report, torn; anything else is kept verbatim as `other`."""
    if "reserve" in rec:
        return "reserve"
    if "torn" in rec:
        return "torn"
    if "report" in rec:
        return "report"
    if "call_id" in rec:
        if rec.get("cached"):
            return "cache_hit"
        if rec.get("outcome") in ("ok", "error"):
            return "settle"
        return "refuse"
    return "other"


def _key_of(rec: dict, kind: str):
    value = rec.get({"reserve": "reserve", "torn": "torn", "report": "report"}.get(kind, "call_id"))
    return value if isinstance(value, str) else (None if value is None else str(value))


def _ts_of(rec: dict):
    ts = rec.get("ts")
    return float(ts) if isinstance(ts, (int, float)) and not isinstance(ts, bool) and math.isfinite(ts) else None


def encode(rec: dict) -> str:
    """The text stored for a record. A value that JSON cannot carry (NaN, Infinity) is refused: it would not survive an export."""
    try:
        return json.dumps(rec, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise LedgerUnavailable(f"the record holds a value that cannot be stored ({type(exc).__name__})") from exc


@dataclass
class Snapshot:
    """What one read transaction saw."""
    records: list = field(default_factory=list)
    blocked: str | None = None             # why reservations are refused (unconfirmed), shown by the page
    present: bool = True                   # False: there is no database yet (an empty history)


def _guard_state(exists_name: Callable[[str], str | None], meta_get: Callable[[str], str | None], scope: str) -> None:
    """Raise unless the directory is in a state where the ledger can be used (I3, I10). `exists_name` looks at a name in the
    directory without following links; `meta_get` reads the database's `meta`."""
    phase = meta_get(cutover_key(scope))
    live = exists_name(LEDGER)
    migrating = exists_name(MARKER_MIGRATING) is not None
    migrated = exists_name(MARKER_MIGRATED) is not None
    if live in ("file", "link", "other"):
        if migrating or migrated or phase is not None:
            raise LegacyLedgerPresent(f"this directory was (or is being) migrated to the database, but a live {LEDGER} is there "
                                      "again: an old writer is running. Stop it, then reconcile before any call")
        raise NotMigrated(f"this directory still holds a jsonl ledger ({LEDGER}); run `tt-orchard db migrate telegraph` first")
    if migrating or (phase is not None and phase != "complete") or (migrated and phase is None):
        if migrated and phase is None:
            raise LedgerRefused("this directory was migrated, but the database does not record it (the database file was lost or "
                                "replaced); restore a backup with `tt-orchard db restore` before any call")
        raise MigrationIncomplete("a migration of this ledger started and did not finish; run `tt-orchard db migrate telegraph` "
                                  "again to finish it")


class TelegraphStore:
    """The ledger of one scope in one state directory. Every method is one transaction."""

    def __init__(self, db: Database, scope: str = DEFAULT_SCOPE, *, own_db: bool = False, guard: bool = True):
        self.db, self.scope, self._own, self._guard_on = db, scope, own_db, guard

    @classmethod
    def open(cls, directory, scope: str = DEFAULT_SCOPE, **kw) -> "TelegraphStore":
        return cls(Database(directory, **kw), scope, own_db=True)

    def close(self) -> None:
        if self._own:
            self.db.close()

    def __enter__(self) -> "TelegraphStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----------------------------------------------------------------------------------------------------- guards

    def _guard(self, conn) -> None:
        if self._guard_on and not self.db._memory:
            _guard_state(self.db.exists_name, lambda k: self.db.meta_get(k, conn), self.scope)

    # ------------------------------------------------------------------------------------------------------ reads

    def _records(self, conn) -> list[dict]:
        out = []
        for (raw,) in conn.execute("SELECT raw_json FROM telegraph_events WHERE scope = ? ORDER BY seq", (self.scope,)):
            try:
                item = json.loads(raw)
            except (ValueError, RecursionError) as exc:
                raise LedgerUnavailable("a stored record cannot be read") from exc
            if not isinstance(item, dict):
                raise LedgerUnavailable("a stored record is not an object")
            out.append(item)
        return out

    def records(self) -> list[dict]:
        """Every record of this scope, in order, from one snapshot. Raises when the directory is in a guarded state."""
        with self.db.read() as conn:
            self._guard(conn)
            return self._records(conn)

    def snapshot(self) -> Snapshot:
        with self.db.read() as conn:
            self._guard(conn)
            return Snapshot(self._records(conn), self.db.reservation_block(conn))

    def cache_get(self, identity: str) -> dict | None:
        with self.db.read() as conn:
            row = conn.execute("SELECT call_id, reply_file, text FROM telegraph_cache WHERE scope = ? AND identity_sha256 = ?",
                               (self.scope, identity)).fetchone()
        return {"call_id": row[0], "reply_file": row[1], "text": row[2]} if row else None

    def cache_count(self) -> int:
        with self.db.read() as conn:
            return conn.execute("SELECT COUNT(*) FROM telegraph_cache WHERE scope = ?", (self.scope,)).fetchone()[0]

    def export_lines(self) -> list[str]:
        """The raw line of every record of this scope, in order: what the jsonl ledger held."""
        with self.db.read() as conn:
            return [raw for (raw,) in conn.execute("SELECT raw_json FROM telegraph_events WHERE scope = ? ORDER BY seq",
                                                   (self.scope,))]

    def torn_names(self) -> list[str]:
        with self.db.read() as conn:
            return [n for (n,) in conn.execute(
                "SELECT call_id FROM telegraph_events WHERE scope = ? AND kind = 'torn' ORDER BY seq", (self.scope,))]

    # ----------------------------------------------------------------------------------------------------- writes

    def _insert(self, conn, rec: dict) -> None:
        kind = kind_of(rec)
        conn.execute("INSERT INTO telegraph_events (scope, ts, kind, call_id, raw_json) VALUES (?, ?, ?, ?, ?)",
                     (self.scope, _ts_of(rec), kind, _key_of(rec, kind), encode(rec)))

    def append(self, rec: dict) -> None:
        """One record in one transaction (a refusal, a cache hit, a settlement with no cache row). Raises on any failure."""
        with self.db.write() as conn:
            self._guard(conn)
            self._insert(conn, rec)

    def reserve(self, reservation: dict, decide: Callable[[list], str | None], refusal: Callable[[str], dict]) -> str | None:
        """The check and the reservation as ONE transaction, so two processes cannot both spend the same allowance and a process
        killed at any point leaves the reservation complete or absent. `decide(records)` returns the reason this call may not be
        sent, or None (the caps, from `Telegraph._over_budget`). A refusal is recorded in the same transaction. Returns the
        refusal reason, or None when the reservation was written."""
        with self.db.write() as conn:
            self._guard(conn)
            why = self.db.reservation_block(conn)
            if why is None:
                why = decide(self._records(conn))
            if why:
                self._insert(conn, refusal(why))
                return why
            self._insert(conn, reservation)
        return None

    def settle(self, rec: dict, cache: tuple | None = None) -> None:
        """Record the result of a request and, with `cache` = (identity, reply_file, text, call_id), its cache row, in one
        transaction: the answer is visible to the cache only together with the call that made it."""
        with self.db.write() as conn:
            self._guard(conn)
            self._insert(conn, rec)
            if cache is not None:
                identity, reply_file, text, call_id = cache
                conn.execute("INSERT INTO telegraph_cache (scope, identity_sha256, call_id, reply_file, text) VALUES (?, ?, ?, ?, ?) "
                             "ON CONFLICT (scope, identity_sha256) DO UPDATE SET call_id = excluded.call_id, "
                             "reply_file = excluded.reply_file, text = excluded.text",
                             (self.scope, identity, call_id, reply_file, text))

    def report(self, call_id: str, make: Callable[[str], dict]) -> None:
        """Record a verdict. A cache hit is reported against the call that made the answer. Raises KeyError for an unknown call."""
        with self.db.write() as conn:
            self._guard(conn)
            by_id = {r["call_id"]: r for r in self._records(conn) if "call_id" in r}
            rec = by_id.get(call_id)
            if rec is not None and rec.get("cached"):
                call_id = rec.get("origin") or ""
                rec = by_id.get(call_id)
            if rec is None:
                raise KeyError(call_id)
            self._insert(conn, make(call_id))

    def add_recovered_reservations(self, ids: list[str], tokens: int, cost) -> int:
        """For every id with no record of its own, a reservation at the worst case (the counted spend only goes up). Used after a
        restore; it never confirms the history. Returns how many were added."""
        added = 0
        with self.db.write() as conn:
            known = set()
            for r in self._records(conn):
                for key in ("reserve", "call_id"):
                    if key in r:
                        known.add(str(r[key]))
            for cid in ids:
                if cid not in known:
                    self._insert(conn, {"reserve": cid, "problem": "(recovered)", "model": None, "tokens": int(tokens),
                                        "cost_usd": cost, "recovered": True})
                    added += 1
        return added

    # ---------------------------------------------------------------------------------------- import (the migration)

    def import_records(self, conn, items: list[tuple[dict, str]]) -> None:
        """Insert records with the raw text they had, in order. The caller holds the write transaction `conn`."""
        for rec, raw in items:
            kind = kind_of(rec)
            conn.execute("INSERT INTO telegraph_events (scope, ts, kind, call_id, raw_json) VALUES (?, ?, ?, ?, ?)",
                         (self.scope, _ts_of(rec), kind, _key_of(rec, kind), raw))

    def import_cache(self, conn, items: list[tuple[str, dict]]) -> int:
        n = 0
        for identity, data in items:
            conn.execute("INSERT OR IGNORE INTO telegraph_cache (scope, identity_sha256, call_id, reply_file, text) "
                         "VALUES (?, ?, ?, ?, ?)", (self.scope, identity, str(data.get("call_id") or ""),
                                                    str(data["reply_file"]), data["text"]))
            n += 1
        return n

    def count_events(self, conn=None) -> int:
        c = conn or self.db.conn()
        return c.execute("SELECT COUNT(*) FROM telegraph_events WHERE scope = ?", (self.scope,)).fetchone()[0]

    def records_in(self, conn) -> list[dict]:
        """The records read inside the caller's transaction (the migration compares them with the file)."""
        return self._records(conn)

    def lines_in(self, conn) -> list[str]:
        return [raw for (raw,) in conn.execute("SELECT raw_json FROM telegraph_events WHERE scope = ? ORDER BY seq", (self.scope,))]


# ------------------------------------------------------------------------------------------------ read only (pages)

def read_snapshot(directory, scope: str = DEFAULT_SCOPE) -> Snapshot:
    """The records of a directory's ledger for a page or a report, in one read transaction, without creating, migrating or
    moving anything. A directory with no database and no legacy ledger is an empty history. Raises LedgerUnavailable or
    LedgerRefused (NotMigrated, LegacyLedgerPresent, MigrationIncomplete, UnsafeDirectory, DatabaseTooNew) for every other case."""
    ro = ReadOnlyDatabase(directory)
    try:
        if not ro.exists:
            live = ro.exists_name(LEDGER)
            if live in ("file", "link", "other"):
                raise NotMigrated(f"this directory still holds a jsonl ledger ({LEDGER}); run `tt-orchard db migrate telegraph`")
            if ro.exists_name(MARKER_MIGRATED) is not None or ro.exists_name(MARKER_MIGRATING) is not None:
                raise LedgerRefused("this directory was migrated, but the database file is missing; restore a backup")
            if ro.marker() is not None:
                raise LedgerRefused("the database file is missing and a recovery is pending")
            return Snapshot([], None, present=False)
        with ro.read() as conn:
            def meta(key):
                row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
                return row[0] if row else None
            _guard_state(ro.exists_name, meta, scope)
            records = []
            for (raw,) in conn.execute("SELECT raw_json FROM telegraph_events WHERE scope = ? ORDER BY seq", (scope,)):
                try:
                    item = json.loads(raw)
                except (ValueError, RecursionError) as exc:
                    raise LedgerUnavailable("a stored record cannot be read") from exc
                if not isinstance(item, dict):
                    raise LedgerUnavailable("a stored record is not an object")
                records.append(item)
            block = meta("unreconciled")
            marker = ro.marker()
            reason = block or (marker or {}).get("reason")
            blocked = (f"the ledger's history may be incomplete ({reason}); no outside call is allowed until a person confirms it"
                       if (block is not None or marker is not None) else None)
            return Snapshot(records, blocked)
    finally:
        ro.close()


def fingerprint(directory, scope: str = DEFAULT_SCOPE) -> tuple | None:
    """(key, size) for a page's cache, or None when there is no database file. `size` is the size of the database file. `key` is a
    cheap change key: the events table only grows, so the newest `seq`, the count, the `meta` rows (the unconfirmed state and the
    migration phase) and the names that guard the directory identify the content. `key` is None when the file cannot be read as a
    database (nothing is cached then)."""
    try:
        ro = ReadOnlyDatabase(directory)
    except (LedgerUnavailable, LedgerRefused):
        return None
    try:
        if not ro.exists:
            return None
        size = ro.file_stats()[0][1] if ro.file_stats()[0] else 0
        try:
            with ro.read() as conn:
                seq, count = conn.execute("SELECT COALESCE(MAX(seq), 0), COUNT(*) FROM telegraph_events WHERE scope = ?",
                                          (scope,)).fetchone()
                metas = tuple(conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall())
            names = tuple(ro.exists_name(n) for n in (LEDGER, MARKER_MIGRATING, MARKER_MIGRATED, dbm.MARKER_NAME))
            return (seq, count, metas, names), size
        except (LedgerUnavailable, LedgerRefused):
            return None, size
    finally:
        ro.close()


def read_export_lines(directory, scope: str = DEFAULT_SCOPE) -> list[str]:
    """The raw record lines of one scope, in order, for `db export telegraph`. No guard: a person can always get the data out.
    A directory with no database has none."""
    ro = ReadOnlyDatabase(directory)
    try:
        if not ro.exists:
            return []
        with ro.read() as conn:
            return [raw for (raw,) in conn.execute("SELECT raw_json FROM telegraph_events WHERE scope = ? ORDER BY seq", (scope,))]
    finally:
        ro.close()


def status_report(directory, scope: str = DEFAULT_SCOPE) -> dict:
    """Facts for `tt-orchard db status`, read only."""
    ro = ReadOnlyDatabase(directory)
    try:
        out: dict = {"directory": ro.directory, "database": ro.exists, "recovery_marker": ro.marker(),
                     "legacy_ledger": ro.exists_name(LEDGER), "archived_ledger": ro.exists_name(ARCHIVE) is not None,
                     "migrating_marker": ro.exists_name(MARKER_MIGRATING) is not None,
                     "migrated_marker": ro.exists_name(MARKER_MIGRATED) is not None}
        if ro.exists:
            with ro.read() as conn:
                def meta(key):
                    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
                    return row[0] if row else None
                out.update({"schema_version": conn.execute("PRAGMA user_version").fetchone()[0],
                            "events": conn.execute("SELECT COUNT(*) FROM telegraph_events WHERE scope = ?", (scope,)).fetchone()[0],
                            "cache_rows": conn.execute("SELECT COUNT(*) FROM telegraph_cache WHERE scope = ?", (scope,)).fetchone()[0],
                            "unconfirmed": meta("unreconciled"), "cutover": meta(cutover_key(scope))})
        try:
            names = os.listdir(os.path.join(ro.directory, "backups"))
        except OSError:
            names = []
        out["backups"] = sorted(n for n in names if n.startswith("project-") and n.endswith(".db"))
        return out
    finally:
        ro.close()
