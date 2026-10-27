# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PROOF ONLY. THE PRODUCT DOES NOT USE IT. A SQLite reservation ledger with the accounting rules of `orchard.telegraph`.

Nothing in the product imports this module. It exists to test one claim of
docs/superpowers/specs/2026-10-28-project-database-design.md: telegraph's budget ledger can live in one SQLite file
and keep every rule it has today, with less code for locking and recovery.

The rules, copied from `orchard.telegraph._accounting` and `_over_budget`:

  * a call is reserved at its worst case before any request is sent;
  * a settlement replaces the reservation with the numbers the provider reported;
  * a settlement with `usage_known` false (a timeout, an answer with no usage) counts at the reservation;
  * a settlement with no reservation (a record from before reservations existed) counts by its own numbers;
  * a cost that is not a number is counted apart as unknown and is never taken as zero;
  * a torn record counts as one call of unknown cost, in the global totals only;
  * the caps are per problem and global: calls, tokens, and dollars (which need a price).

What SQLite changes: the check and the reservation are one `BEGIN IMMEDIATE` transaction, so two processes cannot
both spend the same allowance, and a process killed at any point leaves the reservation complete or absent.
The flock, the fsync calls, the torn-tail quarantine, and the sidecar notes of `telegraph.py` have no counterpart here.

Run `python -m orchard._proofs.telegraph_sqlite import-check <telegraph dir> --db <file>` to import a jsonl ledger and
compare its totals with the jsonl accounting (read only for the source).
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 30_000


class DatabaseTooNew(Exception):
    """The file was written by a newer version of the program."""


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
    """One row per call. The reservation columns and the settlement columns are filled at different times."""
    conn.execute("""
        CREATE TABLE calls (
            call_id         TEXT PRIMARY KEY,
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
            CHECK (has_reserve = 1 OR settled = 1)
        )""")
    conn.execute("CREATE TABLE torn (name TEXT PRIMARY KEY, tokens INTEGER NOT NULL)")
    # Every imported line, verbatim, in order: reports, refusals and cache hits live only here, and `export` reads it.
    conn.execute("CREATE TABLE records (seq INTEGER PRIMARY KEY AUTOINCREMENT, raw TEXT NOT NULL)")
    # What a call counts for: its settlement unless that has unknown usage, otherwise its reservation (the worst case).
    conn.execute("""
        CREATE VIEW effective AS
        SELECT call_id,
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


# ------------------------------------------------------------------------------------------------ the ledger

class Ledger:
    """One SQLite file. Opening it makes the directory private, the file private, sets the pragmas, checks the file and
    migrates the schema. `moved_aside` is the path a damaged file was moved to, or None."""

    def __init__(self, path, *, busy_timeout_ms: int = BUSY_TIMEOUT_MS):
        self.path = Path(path)
        self.moved_aside: Path | None = None
        self._busy = busy_timeout_ms
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, mode=0o700)
        try:
            self._conn = self._open_checked()
        except sqlite3.DatabaseError:
            self._move_aside()
            self._conn = self._open_checked()
        self._migrate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._conn.close()

    # -- opening

    def _open_checked(self) -> sqlite3.Connection:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)       # create it private; SQLite gives -wal and -shm its mode
        try:
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=self._busy / 1000)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")            # a reservation must survive a power cut, as fsync does today
            conn.execute(f"PRAGMA busy_timeout = {int(self._busy)}")
            conn.execute("PRAGMA foreign_keys = ON")
            verdict = conn.execute("PRAGMA integrity_check").fetchall()
            if verdict != [("ok",)]:
                raise sqlite3.DatabaseError("integrity_check failed")
        except sqlite3.DatabaseError:
            conn.close()
            raise
        return conn

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
        split by another writer. Commits on success, rolls back on any exception."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            with contextlib.suppress(sqlite3.Error):
                self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

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
        cur = self._conn.execute("SELECT * FROM calls ORDER BY call_id")
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]

    def unsettled(self) -> list[str]:
        return [r[0] for r in self._conn.execute("SELECT call_id FROM calls WHERE settled = 0 ORDER BY call_id")]

    # -- accounting (the same numbers as telegraph.accounting)

    def accounting(self) -> dict:
        total_calls, total_tokens, total_cost, total_unknown = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens), 0), COALESCE(SUM(COALESCE(cost, 0)), 0), "
            "COALESCE(SUM(cost IS NULL), 0) FROM effective").fetchone()
        torn_calls, torn_tokens = self._conn.execute("SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM torn").fetchone()
        problems = {}
        for name, calls, tokens, cost, unknown in self._conn.execute(
                "SELECT problem, COUNT(*), SUM(tokens), SUM(COALESCE(cost, 0)), SUM(cost IS NULL) "
                "FROM effective GROUP BY problem"):
            problems[name] = {"calls": calls, "tokens": tokens, "cost": cost, "unknown_cost": unknown}
        return {"calls": total_calls + torn_calls, "tokens": total_tokens + torn_tokens, "cost": total_cost,
                "unknown_cost": total_unknown + torn_calls, "problems": problems}

    def totals(self) -> dict:
        """What `telegraph.summarise_records` reports for the calls that were sent."""
        calls, tin, tout, cost = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(tokens_in), 0), COALESCE(SUM(tokens_out), 0), "
            "COALESCE(SUM(COALESCE(cost, 0)), 0) FROM calls WHERE settled = 1").fetchone()
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

    def would_refuse(self, cfg, model: str, tokens: int, cost, problem: str):
        """The decision for a call without reserving anything (a read, no lock)."""
        return self._refusal(cfg, model, tokens, cost, problem, self.accounting())

    def reserve(self, call_id: str, problem: str, model: str, tokens: int, cost, cfg, *, after_insert=None):
        """Check the caps and write the reservation as ONE transaction. Returns the refusal reason, or None when the
        reservation was written. `after_insert` is a test seam: it runs inside the transaction, before the commit."""
        with self._write():
            why = self._refusal(cfg, model, tokens, cost, problem, self.accounting())
            if why:
                return why
            self._conn.execute(
                "INSERT INTO calls (call_id, has_reserve, rproblem, rmodel, reserved_tokens, reserved_cost) "
                "VALUES (?, 1, ?, ?, ?, ?)", (call_id, problem, model, _int(tokens), _cost(cost)))
            if after_insert is not None:
                after_insert()
        return None

    def settle(self, call_id: str, *, problem, model, outcome: str, tokens_in, tokens_out, cost_usd, usage_known) -> None:
        """Replace a reservation's worst case with what the provider reported (one statement, so it is one transaction)."""
        self._conn.execute(_SETTLE, (call_id, problem, model, outcome, _int(tokens_in), _int(tokens_out), _cost(cost_usd),
                                     _flag(usage_known)))

    def record_torn(self, name: str, tokens: int) -> None:
        self._conn.execute("INSERT OR IGNORE INTO torn (name, tokens) VALUES (?, ?)", (name, _int(tokens)))

    # -- import and export

    def import_records(self, recs, raws=None) -> int:
        """Import jsonl records in order, in one transaction. Every record is also kept verbatim in `records`."""
        raws = raws or [json.dumps(r, sort_keys=True) for r in recs]
        with self._write():
            for rec, raw in zip(recs, raws):
                self._conn.execute("INSERT INTO records (raw) VALUES (?)", (raw,))
                if "reserve" in rec:
                    self._conn.execute(_RESERVE_UPSERT, (str(rec["reserve"]), rec.get("problem"), rec.get("model"),
                                                         _int(rec.get("tokens")), _cost(rec.get("cost_usd"))))
                elif _sent(rec):
                    self._conn.execute(_SETTLE, (rec["call_id"], rec.get("problem"), rec.get("model"), rec["outcome"],
                                                 _int(rec.get("tokens_in")), _int(rec.get("tokens_out")),
                                                 _cost(rec.get("cost_usd")), _flag(rec.get("usage_known"))))
                elif "torn" in rec:
                    self._conn.execute("INSERT OR IGNORE INTO torn (name, tokens) VALUES (?, ?)",
                                       (rec["torn"], _int(rec.get("tokens"))))
        return len(recs)

    def export_records(self):
        for (raw,) in self._conn.execute("SELECT raw FROM records ORDER BY seq"):
            yield json.loads(raw)


_SETTLE = """
INSERT INTO calls (call_id, settled, sproblem, smodel, outcome, tokens_in, tokens_out, cost, usage_known)
VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (call_id) DO UPDATE SET settled = 1, sproblem = excluded.sproblem, smodel = excluded.smodel,
    outcome = excluded.outcome, tokens_in = excluded.tokens_in, tokens_out = excluded.tokens_out,
    cost = excluded.cost, usage_known = excluded.usage_known"""

_RESERVE_UPSERT = """
INSERT INTO calls (call_id, has_reserve, rproblem, rmodel, reserved_tokens, reserved_cost) VALUES (?, 1, ?, ?, ?, ?)
ON CONFLICT (call_id) DO UPDATE SET has_reserve = 1, rproblem = excluded.rproblem, rmodel = excluded.rmodel,
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


def compare(directory, db: Ledger) -> dict:
    """The jsonl ledger's totals and the database's totals, and whether they are equal."""
    from orchard import telegraph as tg
    recs = tg.read_ledger(Path(directory))
    want = _shape(tg.summarise_records(recs), tg.accounting(recs))
    got = _shape(db.totals(), db.accounting())
    return {"equal": want == got, "jsonl": want, "sqlite": got}


def import_and_compare(directory, db: Ledger) -> dict:
    """Import `telegraph.jsonl` from `directory` (read only) and check that the totals are the same afterwards."""
    path = Path(directory) / "telegraph.jsonl"
    recs, raws = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            recs.append(item)
            raws.append(line)
    count = db.import_records(recs, raws)
    return {**compare(directory, db), "records_imported": count}


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
