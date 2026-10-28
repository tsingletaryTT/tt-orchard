"""Test helpers for the telegraph ledger, which lives in the project database (orchard/db.py).

The tests used to read and write `telegraph.jsonl` directly. They now go through these helpers, which use the same public functions
the product uses (the export and the importer), so a test sees the ledger as a person would after `tt-orchard db export telegraph`.
"""
import json
import sqlite3
from pathlib import Path

from orchard import db as dbm
from orchard import telegraph_store as ts


def ledger_lines(directory, scope: str = ts.DEFAULT_SCOPE) -> list[str]:
    """The ledger's record lines in order (what `tt-orchard db export telegraph` prints). A directory with no database has none."""
    return ts.read_export_lines(Path(directory), scope)


def ledger_records(directory, scope: str = ts.DEFAULT_SCOPE) -> list[dict]:
    return [json.loads(line) for line in ledger_lines(directory, scope)]


def ledger_text(directory, scope: str = ts.DEFAULT_SCOPE) -> str:
    """The ledger as the old file would read: one line per record."""
    return "".join(line + "\n" for line in ledger_lines(directory, scope))


def database_bytes(directory) -> bytes:
    """Every byte of the database and the files beside it (-wal, -shm), for a test that greps for a credential."""
    out = b""
    for name in ("project.db", "project.db-wal", "project.db-shm"):
        path = Path(directory) / name
        if path.is_file():
            out += path.read_bytes()
    return out


def tree_text(directory) -> str:
    """Every file under `directory` as text (the database and its -wal included, decoded with replacement), for a test that
    greps for a credential. A key stored as text in the database shows up here."""
    return "".join(p.read_bytes().decode("utf-8", errors="replace") for p in sorted(Path(directory).rglob("*")) if p.is_file())


def cache_rows(directory, scope: str = ts.DEFAULT_SCOPE) -> list[dict]:
    """The answer cache of a directory (read only, with plain sqlite3)."""
    path = Path(directory) / "project.db"
    if not path.is_file():
        return []
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        cur = conn.execute("SELECT identity_sha256, call_id, reply_file, text FROM telegraph_cache WHERE scope = ?", (scope,))
        return [dict(zip(("identity", "call_id", "reply_file", "text"), row)) for row in cur.fetchall()]
    finally:
        conn.close()


def set_cache_text(directory, text: str, scope: str = ts.DEFAULT_SCOPE) -> None:
    """Rewrite the text of every cached answer (a test of an entry an earlier version wrote)."""
    conn = sqlite3.connect(Path(directory) / "project.db", isolation_level=None)
    try:
        conn.execute("UPDATE telegraph_cache SET text = ? WHERE scope = ?", (text, scope))
    finally:
        conn.close()


def clear_ledger(directory, scope: str = ts.DEFAULT_SCOPE) -> None:
    """Remove every record of a scope (a test of a page that recovers when a bad ledger is replaced by an empty one)."""
    if not (Path(directory) / "project.db").is_file():
        return
    conn = sqlite3.connect(Path(directory) / "project.db", isolation_level=None)
    try:
        conn.execute("DELETE FROM telegraph_events WHERE scope = ?", (scope,))
    finally:
        conn.close()


def corrupt_stored_record(directory, scope: str = ts.DEFAULT_SCOPE) -> None:
    """Make the newest stored record text that is not JSON, as a hand edit of the database file could."""
    conn = sqlite3.connect(Path(directory) / "project.db", isolation_level=None)
    try:
        conn.execute("UPDATE telegraph_events SET raw_json = '{not json' WHERE seq = "
                     "(SELECT MAX(seq) FROM telegraph_events WHERE scope = ?)", (scope,))
    finally:
        conn.close()


def seed_lines(directory, lines, scope: str = ts.DEFAULT_SCOPE) -> None:
    """Put raw ledger lines in the ledger of `directory` as they are written, including numbers a JSON writer would refuse (NaN,
    Infinity, a negative count): a ledger from an older or damaged writer. Each line must parse as a JSON object."""
    items = []
    for line in lines:
        rec = json.loads(line)
        assert isinstance(rec, dict), line
        items.append((rec, line))
    with dbm.Database(directory) as db:
        store = ts.TelegraphStore(db, scope, guard=False)
        with db.write() as conn:
            store.import_records(conn, items)


def seed_ledger(directory, records, scope: str = ts.DEFAULT_SCOPE) -> None:
    """Put `records` (dicts, in order) in the ledger of `directory`, the way earlier writers left them. They are imported with the
    raw text `json.dumps(sort_keys=True)` gives, as the old writer wrote them. Creates the database."""
    with dbm.Database(directory) as db:
        store = ts.TelegraphStore(db, scope, guard=False)
        with db.write() as conn:
            store.import_records(conn, [(r, json.dumps(r, sort_keys=True)) for r in records])


def replace_ledger(directory, records, scope: str = ts.DEFAULT_SCOPE) -> None:
    """Make the ledger of a scope hold exactly `records`."""
    clear_ledger(directory, scope)
    seed_ledger(directory, records, scope)
