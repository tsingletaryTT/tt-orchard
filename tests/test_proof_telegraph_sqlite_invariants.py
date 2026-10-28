"""Proof only (the product does not use it): one named test for each invariant of the project database design note.

The invariants come from the review of PR 132. Each test name starts with the invariant it checks (I1 to I6); the note lists
them at its top, and a future migration must keep every one of them.

  I1  a cached answer is never served across scopes
  I2  after a restore (or a move-aside, or a new file beside backups) sending is refused until a person confirms
  I3  after migration an old writer cannot recreate telegraph.jsonl, and new code refuses a directory that holds both
  I4  the state directory is private, owned by the user and not a link; creating it is safe when two processes race
  I5  only confirmed corruption moves a database aside; busy, locked, full, I/O and permission errors are retryable
  I6  an offset index validates what it consumed before it reuses totals
"""
import hashlib
import json
import multiprocessing
import os
import shutil
import sqlite3
import stat
from pathlib import Path

import pytest

import legacy_telegraph as tg                 # the jsonl implementation, frozen: the proof compares against it
from orchard._proofs import telegraph_sqlite as proof
from test_proof_telegraph_sqlite import KEY, Wire, table, telegraph  # noqa: F401 - shared helpers

LIM = proof.Limits()


def tables(tmp_path):
    return tg.load(table(), base=tmp_path)


# =============================================================================================== I1: scope

def test_i1_a_cached_answer_is_served_only_to_its_own_scope(tmp_path):
    with proof.Ledger(tmp_path / "db" / "project.db") as db:
        db.reserve("a1", "p", "small", 100, 0.1, LIM, scope="run-a")
        db.settle("a1", problem="p", model="small", outcome="ok", tokens_in=1, tokens_out=1, cost_usd=0.0,
                  usage_known=True, scope="run-a", cache=("identity-1", "replies/a1.txt"))
        assert db.cache_get("run-a", "identity-1")["call_id"] == "a1"
        assert db.cache_get("run-b", "identity-1") is None
        assert db.cache_get("default", "identity-1") is None


def test_i1_the_same_request_in_two_scopes_keeps_two_cache_rows(tmp_path):
    with proof.Ledger(tmp_path / "db" / "project.db") as db:
        for scope, cid in (("run-a", "a1"), ("run-b", "b1")):
            db.reserve(cid, "p", "small", 100, 0.1, LIM, scope=scope)
            db.settle(cid, problem="p", model="small", outcome="ok", tokens_in=1, tokens_out=1, cost_usd=0.0,
                      usage_known=True, scope=scope, cache=("same-identity", f"replies/{cid}.txt"))
        assert db.cache_get("run-a", "same-identity")["call_id"] == "a1"
        assert db.cache_get("run-b", "same-identity")["call_id"] == "b1"


def test_i1_caps_and_totals_are_counted_per_scope(tmp_path):
    cfg = proof.Limits(max_calls=1)
    with proof.Ledger(tmp_path / "db" / "project.db") as db:
        assert db.reserve("a1", "p", "small", 100, 0.1, cfg, scope="run-a") is None
        assert "call cap" in db.reserve("a2", "p", "small", 100, 0.1, cfg, scope="run-a")
        assert db.reserve("b1", "p", "small", 100, 0.1, cfg, scope="run-b") is None
        assert db.accounting("run-a")["calls"] == 1 and db.accounting("run-b")["calls"] == 1
        assert db.accounting("run-c")["calls"] == 0


# =============================================================================================== I2: restore

def make_history(path, n=5):
    with proof.Ledger(path) as db:
        for i in range(n):
            assert db.reserve(f"c{i}", "p", "small", 1000, 0.25, LIM) is None
    return path


def test_i2_after_a_restore_from_an_older_backup_sending_is_refused_until_a_person_confirms(tmp_path):
    path = tmp_path / "db" / "project.db"
    with proof.Ledger(path) as db:
        db.reserve("c0", "p", "small", 1000, 0.25, LIM)
        backup = db.backup(tmp_path / "db" / "backups")
        for i in range(1, 6):                                     # reservations made after the backup
            db.reserve(f"c{i}", "p", "small", 1000, 0.25, LIM)
    cap = proof.Limits(max_calls=3)
    with proof.Ledger.restore(path, backup) as db:
        assert db.accounting()["calls"] == 1                      # the restored file lost five reservations
        why = db.reserve("late", "p", "small", 1, 0.0, cap)
        assert why and "confirm" in why                           # so the cap of 3 does not make room again
        assert db.accounting()["calls"] == 1
        db.confirm_reconciled("operator checked the provider's usage page")
        assert db.reserve("late", "p", "small", 1, 0.0, cap) is None


def test_i2_a_moved_aside_database_starts_unconfirmed(tmp_path):
    path = tmp_path / "db" / "project.db"
    make_history(path)
    path.write_bytes(b"this is not a database file at all" * 200)
    with proof.Ledger(path) as db:
        assert db.moved_aside is not None
        assert "confirm" in db.reserve("x", "p", "small", 1, 0.0, LIM)


def test_i2_a_new_file_beside_existing_backups_starts_unconfirmed(tmp_path):
    path = tmp_path / "db" / "project.db"
    with proof.Ledger(path) as db:
        db.reserve("c0", "p", "small", 1000, 0.25, LIM)
        db.backup(tmp_path / "db" / "backups")
    path.unlink()                                                  # the file is gone, the backups are not
    with proof.Ledger(path) as db:
        assert "confirm" in db.reserve("x", "p", "small", 1, 0.0, LIM)


def test_i2_a_first_ever_open_is_not_unconfirmed(tmp_path):
    with proof.Ledger(tmp_path / "db" / "project.db") as db:
        assert db.reserve("x", "p", "small", 1, 0.0, LIM) is None


def test_i2_the_unconfirmed_state_survives_reopening_and_is_cleared_only_by_confirming(tmp_path):
    path = tmp_path / "db" / "project.db"
    path.parent.mkdir()
    path.write_bytes(b"garbage" * 1000)
    with proof.Ledger(path):
        pass
    with proof.Ledger(path) as db:
        assert db.unreconciled() is not None
        db.confirm_reconciled("checked")
    with proof.Ledger(path) as db:
        assert db.unreconciled() is None


def test_i2_replies_without_a_row_are_counted_at_their_worst_case_and_do_not_confirm(tmp_path):
    replies = tmp_path / "replies"
    replies.mkdir()
    for name in ("aaa111", "bbb222"):
        (replies / f"{name}.txt").write_text("an answer")
    path = tmp_path / "db" / "project.db"
    path.parent.mkdir()
    path.write_bytes(b"garbage" * 1000)
    with proof.Ledger(path) as db:
        added = db.reconcile_replies(replies, worst_tokens=1500, worst_cost=0.5)
        assert added == 2 and db.reconcile_replies(replies, 1500, 0.5) == 0       # a second pass adds nothing
        acct = db.accounting()
        assert (acct["calls"], acct["tokens"], acct["cost"]) == (2, 3000, 1.0)
        assert db.unreconciled() is not None                      # failed and timed-out calls left no reply file


# =============================================================================================== I3: legacy writers

def old_directory(tmp_path):
    """A telegraph directory as the jsonl implementation leaves it, made by the real Telegraph."""
    tel, _ = telegraph(tmp_path)
    tel.ask("p", "t", "f1")
    tel.ask("p", "t", "f2")
    return tmp_path / "tg"


def test_i3_after_migration_an_old_telegraph_refuses_every_call_and_sends_nothing(tmp_path):
    old = old_directory(tmp_path)
    original = (old / "telegraph.jsonl").read_bytes()
    with proof.Ledger(tmp_path / "db" / "project.db") as db:
        proof.migrate_directory(old, db)
    wire = Wire()
    cfg = tg.load(table(), base=tmp_path)
    tel = tg.Telegraph(cfg, old, transport=wire, environ={"TG_KEY": KEY})      # the old code, as a stale process runs it
    got = tel.ask("p", "t", "a new question")
    assert not got.ok and "ledger cannot be written or read" in got.reason and wire.requests == []
    assert not (old / "telegraph.jsonl").is_file()                  # nothing recreated the ledger
    assert (old / "telegraph.jsonl.archived").read_bytes() == original   # the archive is the old ledger, byte for byte
    assert (old / "MIGRATED-TO-SQLITE").is_file()


def test_i3_the_settlement_of_a_request_in_flight_during_the_cutover_cannot_recreate_the_ledger(tmp_path):
    old = old_directory(tmp_path)
    db = proof.Ledger(tmp_path / "db" / "project.db")
    done = {}

    def cutover_during_request(url, headers, body, timeout):
        done["migrated"] = proof.migrate_directory(old, db)          # the cutover happens while the request is out
        return 200, json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 10}}).encode()
    cfg = tg.load(table(max_calls=50), base=tmp_path)
    tel = tg.Telegraph(cfg, old, transport=cutover_during_request, environ={"TG_KEY": KEY})
    got = tel.ask("p", "t", "in flight")
    assert done["migrated"]["equal"]
    assert not (old / "telegraph.jsonl").is_file()                  # the late settlement could not recreate it
    assert "could not record" in got.reason                          # and the old code said so
    acct = db.accounting()
    assert acct["calls"] == 3                                        # two settled before, the reservation of the third
    assert db.unsettled() and len(db.unsettled()) == 1               # counted at its worst case, never lost
    db.close()


def test_i3_new_code_refuses_a_directory_that_holds_the_marker_and_a_live_jsonl(tmp_path):
    old = old_directory(tmp_path)
    with proof.Ledger(tmp_path / "db" / "project.db") as db:
        proof.migrate_directory(old, db)
    proof.refuse_if_legacy_present(old)                               # the migrated directory is fine
    shutil.rmtree(old / "telegraph.jsonl")                            # a legacy writer got a regular file back
    (old / "telegraph.jsonl").write_text('{"call_id": "late"}\n')
    with pytest.raises(proof.LegacyLedgerPresent):
        proof.refuse_if_legacy_present(old)


def test_i3_a_migration_whose_totals_differ_changes_nothing(tmp_path, monkeypatch):
    old = old_directory(tmp_path)
    before = {p.name: p.read_bytes() for p in old.iterdir() if p.is_file()}
    real = proof.compare
    monkeypatch.setattr(proof, "compare", lambda *a, **k: {**real(*a, **k), "equal": False})
    with proof.Ledger(tmp_path / "db" / "project.db") as db:
        with pytest.raises(proof.MigrationMismatch):
            proof.migrate_directory(old, db)
        assert db.count_rows() == 0 and list(db.export_records()) == []     # rolled back
    assert {p.name: p.read_bytes() for p in old.iterdir() if p.is_file()} == before
    assert (old / "telegraph.jsonl").is_file() and not (old / "MIGRATED-TO-SQLITE").exists()


def test_i3_migrating_twice_is_refused(tmp_path):
    old = old_directory(tmp_path)
    with proof.Ledger(tmp_path / "db" / "project.db") as db:
        proof.migrate_directory(old, db)
        with pytest.raises(proof.AlreadyMigrated):
            proof.migrate_directory(old, db)
        assert db.totals()["calls"] == 2                              # no row was imported twice


# =============================================================================================== I4: directory

def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def test_i4_an_existing_directory_with_loose_permissions_is_tightened(tmp_path):
    d = tmp_path / "db"
    d.mkdir()
    d.chmod(0o777)
    with proof.Ledger(d / "project.db"):
        assert mode(d) == 0o700


def test_i4_a_directory_that_is_a_link_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "db").symlink_to(real)
    with pytest.raises(proof.UnsafeDirectory):
        proof.Ledger(tmp_path / "db" / "project.db")
    assert not (real / "project.db").exists()


def test_i4_a_directory_owned_by_someone_else_is_refused_and_left_alone(tmp_path, monkeypatch):
    d = tmp_path / "db"
    d.mkdir()
    d.chmod(0o777)
    other = os.geteuid() + 1
    monkeypatch.setattr(proof.os, "geteuid", lambda: other)
    with pytest.raises(proof.UnsafeDirectory):
        proof.Ledger(d / "project.db")
    assert mode(d) == 0o777                                           # it was not ours to change
    assert not (d / "project.db").exists()


def test_i4_a_database_file_that_is_a_link_is_refused(tmp_path):
    d = tmp_path / "db"
    d.mkdir(mode=0o700)
    target = tmp_path / "elsewhere.db"
    target.write_bytes(b"")
    (d / "project.db").symlink_to(target)
    with pytest.raises(proof.UnsafeDirectory):
        proof.Ledger(d / "project.db")


def test_i4_losing_the_race_to_create_the_directory_is_not_an_error(tmp_path, monkeypatch):
    real_mkdir = os.mkdir

    def other_process_wins(path, *a, **k):
        real_mkdir(path, *a, **k)                                     # the other process created it first
        raise FileExistsError(path)
    monkeypatch.setattr(proof.os, "mkdir", other_process_wins)
    with proof.Ledger(tmp_path / "a" / "b" / "project.db") as db:
        assert db.count_rows() == 0
    assert mode(tmp_path / "a" / "b") == 0o700


def open_once(path):
    with proof.Ledger(path) as db:
        return db.count_rows()


def test_i4_four_processes_opening_a_missing_directory_at_once_all_succeed(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    for round_ in range(5):
        path = str(tmp_path / f"r{round_}" / "deep" / "project.db")
        with ctx.Pool(4) as pool:
            assert pool.map(open_once, [path] * 4) == [0, 0, 0, 0]
        assert mode(Path(path).parent) == 0o700


# =============================================================================================== I5: what moves a file aside

class Flaky:
    """A connection that raises `error` when a statement contains `needle`, and otherwise passes everything through."""

    def __init__(self, conn, needle, error):
        self._c, self._needle, self._error = conn, needle, error

    def execute(self, sql, *a):
        if self._needle in sql:
            raise self._error
        return self._c.execute(sql, *a)

    def __getattr__(self, name):
        return getattr(self._c, name)


def inject(monkeypatch, needle, error):
    real = sqlite3.connect
    if needle is None:
        def connect(*a, **k):
            raise error
    else:
        def connect(*a, **k):
            return Flaky(real(*a, **k), needle, error)
    monkeypatch.setattr(proof.sqlite3, "connect", connect)


RETRYABLE = ["database is locked", "disk I/O error", "database or disk is full", "unable to open database file",
             "attempt to write a readonly database", "database table is locked"]
CORRUPT = ["database disk image is malformed", "file is not a database", "malformed database schema (calls)"]


def snapshot(d):
    return sorted((p.name, p.read_bytes()) for p in d.iterdir() if p.is_file())


@pytest.mark.parametrize("message", RETRYABLE)
@pytest.mark.parametrize("where", [None, "integrity_check", "journal_mode"], ids=["at-connect", "at-check", "at-pragma"])
def test_i5_a_lock_or_io_error_is_retryable_and_leaves_the_file_untouched(tmp_path, monkeypatch, message, where):
    path = tmp_path / "db" / "project.db"
    make_history(path)
    before = snapshot(path.parent)
    inject(monkeypatch, where, sqlite3.OperationalError(message))
    with pytest.raises(proof.LedgerUnavailable):
        proof.Ledger(path, busy_timeout_ms=100)
    assert snapshot(path.parent) == before                            # the file is not moved or changed, and there is no corrupt-* sibling
    assert not list(path.parent.glob("*.corrupt-*"))


@pytest.mark.parametrize("message", CORRUPT)
@pytest.mark.parametrize("where", ["integrity_check", "journal_mode"])
def test_i5_confirmed_corruption_moves_the_file_aside_and_starts_unconfirmed(tmp_path, monkeypatch, message, where):
    path = tmp_path / "db" / "project.db"
    make_history(path)
    calls = {"n": 0}
    real = sqlite3.connect

    def connect(*a, **k):                                             # only the first connection sees the damage
        calls["n"] += 1
        conn = real(*a, **k)
        return Flaky(conn, where, sqlite3.DatabaseError(message)) if calls["n"] == 1 else conn
    monkeypatch.setattr(proof.sqlite3, "connect", connect)
    with proof.Ledger(path) as db:
        assert db.moved_aside is not None and db.moved_aside.exists()
        assert db.unreconciled() is not None


def test_i5_an_integrity_check_that_fails_moves_the_file_aside(tmp_path):
    """Real damage, found only by the check (see also the damaged-file test of the first proof file)."""
    path = tmp_path / "db" / "project.db"
    with proof.Ledger(path) as db:
        for i in range(400):
            db.reserve(f"c{i}", "p", "small", 100, 0.1, LIM)
    data = bytearray(path.read_bytes())
    for i in range(4096, len(data), 4096):
        data[i + 20:i + 400] = os.urandom(380)
    path.write_bytes(bytes(data))
    with proof.Ledger(path) as db:
        assert db.moved_aside is not None


def test_i5_a_write_that_finds_the_database_locked_raises_a_retryable_error_and_changes_nothing(tmp_path):
    path = tmp_path / "db" / "project.db"
    make_history(path, 1)
    holder = sqlite3.connect(path, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with proof.Ledger(path, busy_timeout_ms=150) as db:
            with pytest.raises(proof.LedgerUnavailable):
                db.reserve("blocked", "p", "small", 1, 0.0, LIM)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    with proof.Ledger(path) as db:
        assert db.accounting()["calls"] == 1 and db.moved_aside is None


# =============================================================================================== I6: offset index

CHUNK = 64


def line(n):
    return json.dumps({"n": n}) + "\n"


def write_lines(path, values):
    path.write_text("".join(line(v) for v in values))


def index(tmp_path):
    db = proof.Ledger(tmp_path / "db" / "index.db")
    return db, proof.LogIndex(db, chunk=CHUNK)


def test_i6_an_appended_log_is_read_from_its_offset_and_the_total_is_kept(tmp_path):
    log = tmp_path / "agent.jsonl"
    write_lines(log, range(1, 101))
    db, ix = index(tmp_path)
    first = ix.update(log)
    assert first["total"] == 5050 and first["bytes_read"] == log.stat().st_size
    with open(log, "a") as f:
        f.write(line(1000))
    second = ix.update(log)
    assert second["total"] == 6050 and not second["rescanned"]
    assert second["bytes_read"] <= CHUNK + len(line(1000))            # one chunk to check, plus the new bytes
    assert ix.update(log)["bytes_read"] <= CHUNK                      # nothing new: at most the check
    db.close()


def test_i6_a_half_written_last_line_is_not_consumed_until_it_is_whole(tmp_path):
    log = tmp_path / "agent.jsonl"
    write_lines(log, [1, 2])
    with open(log, "a") as f:
        f.write('{"n": 5')
    db, ix = index(tmp_path)
    assert ix.update(log)["total"] == 3
    with open(log, "a") as f:
        f.write("0}\n")
    assert ix.update(log)["total"] == 53
    db.close()


def test_i6_a_truncate_and_regrow_that_keeps_the_start_is_detected(tmp_path):
    log = tmp_path / "agent.jsonl"
    write_lines(log, range(1, 101))
    db, ix = index(tmp_path)
    ix.update(log)
    keep = log.read_bytes()[:CHUNK * 3]                               # same start, then different content past the old offset
    log.write_bytes(keep.rsplit(b"\n", 1)[0] + b"\n" + b"".join(line(7).encode() for _ in range(200)))
    got = ix.update(log)
    expected = sum(json.loads(x)["n"] for x in log.read_text().splitlines())
    assert got["rescanned"] and got["total"] == expected
    db.close()


def test_i6_a_same_size_edit_in_the_last_chunk_is_detected_by_an_ordinary_poll(tmp_path):
    log = tmp_path / "agent.jsonl"
    write_lines(log, range(10, 110))
    db, ix = index(tmp_path)
    ix.update(log)
    raw = bytearray(log.read_bytes())
    raw[-5:-2] = b"999"                                               # same length, inside the last chunk
    log.write_bytes(bytes(raw))
    got = ix.update(log)
    assert got["rescanned"] and got["total"] == sum(json.loads(x)["n"] for x in log.read_text().splitlines())
    db.close()


def test_i6_a_same_size_edit_in_an_early_chunk_is_missed_by_an_ordinary_poll_and_caught_by_a_full_check(tmp_path):
    """The residual risk, stated as a test: the cheap poll cannot see an edit far behind the offset."""
    log = tmp_path / "agent.jsonl"
    write_lines(log, range(10, 110))
    db, ix = index(tmp_path)
    stale = ix.update(log)["total"]
    raw = bytearray(log.read_bytes())
    raw[6:8] = b"99"
    log.write_bytes(bytes(raw))
    assert ix.update(log)["total"] == stale                           # the index still shows the old total
    got = ix.update(log, full=True)
    assert got["rescanned"] and got["total"] == sum(json.loads(x)["n"] for x in log.read_text().splitlines())
    db.close()


def test_i6_a_smaller_file_restarts_from_zero(tmp_path):
    log = tmp_path / "agent.jsonl"
    write_lines(log, range(1, 51))
    db, ix = index(tmp_path)
    ix.update(log)
    write_lines(log, [5, 6])
    got = ix.update(log)
    assert got["rescanned"] and got["total"] == 11
    db.close()


def test_i6_a_new_file_under_the_same_name_restarts_from_zero_even_when_it_is_larger(tmp_path):
    log = tmp_path / "agent.jsonl"
    write_lines(log, range(1, 51))
    db, ix = index(tmp_path)
    ix.update(log)
    fresh = tmp_path / "fresh.jsonl"
    write_lines(fresh, [3] * 400)                                     # larger, so only the inode gives it away
    os.replace(fresh, log)
    got = ix.update(log)
    assert got["rescanned"] and got["total"] == 1200
    db.close()


def test_i4_a_locked_answer_to_the_wal_switch_is_retried_until_the_timeout(tmp_path, monkeypatch):
    """Two first opens at once: SQLite answers the switch to WAL with "locked" at once. The open waits and tries again."""
    state = {"left": 3}
    real = sqlite3.connect

    class LockedThrice(Flaky):
        def execute(self, sql, *a):
            if "journal_mode" in sql and state["left"] > 0:
                state["left"] -= 1
                raise sqlite3.OperationalError("database is locked")
            return self._c.execute(sql, *a)

    monkeypatch.setattr(proof.sqlite3, "connect", lambda *a, **k: LockedThrice(real(*a, **k), None, None))
    with proof.Ledger(tmp_path / "db" / "project.db") as db:
        assert db.count_rows() == 0
    assert state["left"] == 0


# --- review round 2: two medium findings, fixed in the proof ---------------------------------------------------------

@pytest.mark.parametrize("message", ["database or disk is full", "disk I/O error", "attempt to write a readonly database"])
@pytest.mark.parametrize("statement", ["INSERT INTO calls", "INSERT OR REPLACE INTO cache"])
def test_i5_an_error_inside_the_transaction_body_is_retryable_and_leaves_no_open_transaction(tmp_path, message, statement):
    path = tmp_path / "db" / "project.db"
    with proof.Ledger(path) as db:
        real = db._conn
        db._conn = Flaky(real, statement, sqlite3.OperationalError(message))     # BEGIN succeeds, the statement after it fails
        if statement.startswith("INSERT INTO calls"):
            with pytest.raises(proof.LedgerUnavailable):
                db.reserve("c1", "p", "small", 10, 0.0, LIM)
        else:
            real.execute("INSERT INTO calls (scope, call_id, has_reserve, reserved_tokens) VALUES ('default', 'c1', 1, 10)")
            with pytest.raises(proof.LedgerUnavailable):
                db.settle("c1", problem="p", model="small", outcome="ok", tokens_in=1, tokens_out=1, cost_usd=0.0,
                          usage_known=True, cache=("id", "replies/c1.txt"))
        db._conn = real
        assert not real.in_transaction                                            # the failed write was rolled back
        assert db.cache_get("default", "id") is None
        assert db.reserve("c2", "p", "small", 10, 0.0, LIM) is None               # and the next writer is not blocked
        assert db.rows()[0]["settled"] == 0                                       # the settle did not half-apply


def test_i6_an_edit_in_the_last_chunk_is_detected_when_the_offset_is_an_exact_multiple_of_the_chunk(tmp_path):
    log = tmp_path / "agent.jsonl"
    log.write_bytes(b'{"n":1}\n' * 8)                                 # 8 lines of 8 bytes: the offset is 64, one whole chunk
    db, ix = index(tmp_path)
    assert ix.update(log)["total"] == 8
    raw = bytearray(log.read_bytes())
    raw[-3:-2] = b"5"                                                 # the last line becomes {"n":5}, same size, inside the chunk
    log.write_bytes(bytes(raw))
    got = ix.update(log)
    assert got["rescanned"] and got["total"] == 12
    db.close()


def test_i6_the_totals_stay_right_across_offsets_that_are_and_are_not_multiples_of_the_chunk(tmp_path):
    log = tmp_path / "agent.jsonl"
    db, ix = index(tmp_path)
    expected = 0
    with open(log, "wb") as f:
        for step in range(40):                                        # lines of 8 and 9 bytes, so offsets cross chunk edges
            body = (b'{"n":%d}\n' % (step % 10)) if step % 3 else (b'{"n": %d}\n' % (step % 10))
            f.write(body)
            f.flush()
            expected += step % 10
            got = ix.update(log, full=(step % 4 == 0))
            assert got["total"] == expected and not got["rescanned"], step
    db.close()


# --- requirements I7 to I10 (Copilot review round 2): to be written with the migration -----------------------------------
# Each placeholder is a strict xfail: it fails today because nothing implements the requirement. When the migration PR
# writes the real test under the same name, the marker has to come off, because a passing strict xfail is an error.

MIGRATION = pytest.mark.xfail(strict=True, reason="requirement of the design note, to be written with the migration")


@MIGRATION
def test_I7_recovery_state_is_on_disk_before_a_replacement_database_exists_and_every_opener_honours_it():
    pytest.fail("to be written with the migration: kill at each step of the replacement sequence (design note, I7)")


@MIGRATION
def test_I8_a_staged_restore_is_marked_before_installation_and_a_crash_at_any_step_leaves_reservations_refused():
    pytest.fail("to be written with the migration: kill at each step of the restore sequence (design note, I8)")


@MIGRATION
def test_I9_orphaned_sidecars_and_malformed_fragments_are_recovered_or_block_the_migration():
    pytest.fail("to be written with the migration: sidecars and a torn tail counted, or the migration refuses (design note, I9)")


@MIGRATION
def test_I10_a_crash_at_any_step_of_the_cutover_resumes_and_no_phase_lets_a_writer_send_into_a_second_history():
    pytest.fail("to be written with the migration: kill at each step of the cutover, then resume (design note, I10)")
