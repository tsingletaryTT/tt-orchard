"""The project database access layer (orchard/db.py): directory and file checks, the error classes, recovery, restore, backups, migrations.

Each test names the invariant of docs/superpowers/specs/2026-10-28-project-database-design.md that it checks (I2, I4, I5, I7, I8).
"""
import multiprocessing
import os
import sqlite3
import stat
import threading
import time
from pathlib import Path

import pytest

from dbtools import KILLED, blocked, make_history, reserve, run_child, snapshot
from orchard import db as dbm
from orchard import telegraph_store as ts


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def count(directory):
    with ts.TelegraphStore.open(directory) as store:
        return len(store.records())


# =============================================================================================== I4: the directory and the file

def test_i4_an_existing_directory_with_loose_permissions_is_tightened(tmp_path):
    d = tmp_path / "state"
    d.mkdir()
    d.chmod(0o777)
    with dbm.Database(d):
        assert mode(d) == 0o700


def test_i4_a_directory_that_is_a_link_is_refused_and_nothing_is_created_in_the_target(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "state").symlink_to(real)
    with pytest.raises(dbm.UnsafeDirectory):
        dbm.Database(tmp_path / "state")
    assert list(real.iterdir()) == []


def test_i4_a_directory_owned_by_someone_else_is_refused_and_left_alone(tmp_path, monkeypatch):
    d = tmp_path / "state"
    d.mkdir()
    d.chmod(0o777)
    other = os.geteuid() + 1
    monkeypatch.setattr(dbm.os, "geteuid", lambda: other)
    with pytest.raises(dbm.UnsafeDirectory):
        dbm.Database(d)
    monkeypatch.undo()
    assert mode(d) == 0o777 and list(d.iterdir()) == []              # it was not ours to change, and nothing was made in it


def test_i4_a_file_in_place_of_the_directory_is_refused(tmp_path):
    (tmp_path / "state").write_text("not a directory")
    with pytest.raises(dbm.LedgerRefused):
        dbm.Database(tmp_path / "state")


@pytest.mark.parametrize("kind", ["link", "directory", "fifo"])
def test_i4_a_database_file_that_is_not_a_regular_file_is_refused_and_left_alone(tmp_path, kind):
    d = tmp_path / "state"
    d.mkdir(mode=0o700)
    target = d / "project.db"
    if kind == "link":
        outside = tmp_path / "elsewhere.db"
        outside.write_bytes(b"")
        target.symlink_to(outside)
    elif kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)
    result = {}

    def attempt():
        try:
            dbm.Database(d)
        except dbm.LedgerRefused as exc:
            result["error"] = exc

    t = threading.Thread(target=attempt, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive(), "opening a special file blocked"
    assert isinstance(result.get("error"), dbm.UnsafeDirectory)
    assert target.is_symlink() == (kind == "link") and not list(d.glob("project.db.corrupt-*")) and not (d / "RECOVERY-REQUIRED").exists()


@pytest.mark.parametrize("suffix", ["-wal", "-shm"])
def test_i4_a_wal_or_shm_file_that_is_a_link_is_refused(tmp_path, suffix):
    make_history(tmp_path / "state")
    d = tmp_path / "state"
    outside = tmp_path / "outside"
    outside.write_bytes(b"")
    (d / ("project.db" + suffix)).symlink_to(outside)
    with pytest.raises(dbm.UnsafeDirectory):
        dbm.Database(d)
    assert outside.read_bytes() == b""


def test_i4_losing_the_race_to_create_the_directory_is_not_an_error(tmp_path, monkeypatch):
    real_mkdir = os.mkdir

    def other_process_wins(path, *a, **k):
        real_mkdir(path, *a, **k)                                     # the other process created it first
        raise FileExistsError(path)
    monkeypatch.setattr(dbm.os, "mkdir", other_process_wins)
    with dbm.Database(tmp_path / "a" / "b"):
        pass
    assert mode(tmp_path / "a" / "b") == 0o700


def open_once(path):
    with dbm.Database(path) as db:
        return db.user_version()


def test_i4_four_processes_opening_a_missing_directory_at_once_all_succeed(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    for round_ in range(5):
        path = str(tmp_path / f"r{round_}" / "deep")
        with ctx.Pool(4) as pool:
            assert pool.map(open_once, [path] * 4) == [1, 1, 1, 1]
        assert mode(Path(path)) == 0o700


class Flaky:
    """A connection that raises `error` when a statement contains `needle`, and otherwise passes everything through."""

    def __init__(self, conn, needle, error):
        self._c, self._needle, self._error = conn, needle, error

    def execute(self, sql, *a):
        if self._needle is not None and self._needle in sql:
            raise self._error
        return self._c.execute(sql, *a)

    def __getattr__(self, name):
        return getattr(self._c, name)


def inject(monkeypatch, needle, error, only_first=False):
    real = sqlite3.connect
    state = {"n": 0}
    if needle is None:
        def connect(*a, **k):
            raise error
    else:
        def connect(*a, **k):
            state["n"] += 1
            conn = real(*a, **k)
            return Flaky(conn, needle, error) if (not only_first or state["n"] == 1) else conn
    monkeypatch.setattr(dbm.sqlite3, "connect", connect)


def test_i4_a_locked_answer_to_the_wal_switch_is_retried_until_the_timeout(tmp_path, monkeypatch):
    state = {"left": 3}
    real = sqlite3.connect

    class LockedThrice(Flaky):
        def execute(self, sql, *a):
            if "journal_mode" in sql and state["left"] > 0:
                state["left"] -= 1
                raise sqlite3.OperationalError("database is locked")
            return self._c.execute(sql, *a)

    monkeypatch.setattr(dbm.sqlite3, "connect", lambda *a, **k: LockedThrice(real(*a, **k), None, None))
    with dbm.Database(tmp_path / "state") as db:
        assert db.user_version() == 1
    assert state["left"] == 0


def test_i4_the_database_and_its_wal_files_are_private_while_it_is_open(tmp_path):
    with ts.TelegraphStore.open(tmp_path / "state") as store:
        reserve(store, "c1")
        names = sorted(p.name for p in (tmp_path / "state").iterdir())
        assert "project.db-wal" in names                              # the files beside the database exist now
        for p in (tmp_path / "state").iterdir():
            if p.is_file():
                assert mode(p) == 0o600, p
        assert mode(tmp_path / "state") == 0o700


def test_i4_an_existing_database_with_a_loose_mode_is_tightened(tmp_path):
    make_history(tmp_path / "state")
    (tmp_path / "state" / "project.db").chmod(0o644)
    with dbm.Database(tmp_path / "state"):
        assert mode(tmp_path / "state" / "project.db") == 0o600


def test_the_pragmas_are_read_back(tmp_path):
    with dbm.Database(tmp_path / "state") as db:
        p = db.pragmas()
    assert p == {"journal_mode": "wal", "synchronous": 2, "busy_timeout": dbm.BUSY_TIMEOUT_MS, "foreign_keys": 1}


def test_a_database_that_cannot_use_wal_is_refused(tmp_path, monkeypatch):
    real = sqlite3.connect

    class NoWal(Flaky):
        def execute(self, sql, *a):
            if "journal_mode" in sql:
                return self._c.execute("PRAGMA journal_mode = DELETE")
            return self._c.execute(sql, *a)
    monkeypatch.setattr(dbm.sqlite3, "connect", lambda *a, **k: NoWal(real(*a, **k), None, None))
    with pytest.raises(dbm.LedgerUnavailable, match="WAL"):
        dbm.Database(tmp_path / "state")


# =============================================================================================== I5: what moves a file aside

RETRYABLE = ["database is locked", "disk I/O error", "database or disk is full", "unable to open database file",
             "attempt to write a readonly database", "database table is locked"]
CORRUPT = ["database disk image is malformed", "file is not a database", "malformed database schema (meta)"]


@pytest.mark.parametrize("message", RETRYABLE)
@pytest.mark.parametrize("where", [None, "integrity_check", "journal_mode"], ids=["at-connect", "at-check", "at-pragma"])
def test_i5_a_lock_or_io_error_is_retryable_and_leaves_every_file_untouched(tmp_path, monkeypatch, message, where):
    d = make_history(tmp_path / "state")
    before = snapshot(d)
    inject(monkeypatch, where, sqlite3.OperationalError(message))
    with pytest.raises(dbm.LedgerUnavailable):
        dbm.Database(d, busy_timeout_ms=100)
    assert snapshot(d) == before
    assert not list(d.glob("*.corrupt-*")) and not (d / "RECOVERY-REQUIRED").exists()


@pytest.mark.parametrize("message", CORRUPT)
@pytest.mark.parametrize("where", ["integrity_check", "journal_mode"])
def test_i5_confirmed_corruption_moves_the_file_aside_and_starts_unconfirmed(tmp_path, monkeypatch, message, where):
    d = make_history(tmp_path / "state")
    inject(monkeypatch, where, sqlite3.DatabaseError(message), only_first=True)   # only the first connection sees the damage
    with dbm.Database(d) as db:
        assert db.moved_aside and (d / db.moved_aside).exists()
        assert db.unreconciled() is not None and (d / "RECOVERY-REQUIRED").exists()


def test_i5_an_integrity_check_that_fails_on_a_really_damaged_file_moves_it_aside(tmp_path):
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        for i in range(400):
            reserve(store, f"c{i}", tokens=100)
    data = bytearray((d / "project.db").read_bytes())
    for i in range(4096, len(data), 4096):                            # page 1 stays whole, so only the check can tell
        data[i + 20:i + 400] = os.urandom(380)
    (d / "project.db").write_bytes(bytes(data))
    with dbm.Database(d) as db:
        assert db.moved_aside is not None
        assert (d / db.moved_aside).stat().st_size == len(data)       # the damaged file is kept, not deleted


def test_i5_a_file_that_is_not_a_database_is_moved_aside_with_its_siblings(tmp_path):
    d = tmp_path / "state"
    d.mkdir(mode=0o700)
    (d / "project.db").write_bytes(b"this is not a database file at all" * 200)
    (d / "project.db-wal").write_bytes(b"x")
    with dbm.Database(d) as db:
        moved = db.moved_aside
    assert (d / moved).read_bytes().startswith(b"this is not a database")
    assert (d / (moved + "-wal")).read_bytes() == b"x"


def test_i5_a_write_that_finds_the_database_locked_raises_a_retryable_error_before_the_body_runs(tmp_path):
    d = make_history(tmp_path / "state", 1)
    holder = sqlite3.connect(d / "project.db", isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    entered = []
    try:
        with dbm.Database(d, busy_timeout_ms=150) as db:
            with pytest.raises(dbm.LedgerUnavailable):
                with db.write():
                    entered.append(True)                              # a deferred BEGIN would get here and fail later
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    assert entered == []
    assert count(d) == 1 and not list(d.glob("*.corrupt-*"))


@pytest.mark.parametrize("message", ["database or disk is full", "disk I/O error", "attempt to write a readonly database"])
def test_i5_an_error_inside_the_transaction_body_is_retryable_and_leaves_no_open_transaction(tmp_path, message):
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        real = store.db.conn()
        flaky = Flaky(real, "INSERT INTO telegraph_events", sqlite3.OperationalError(message))
        store.db._local.conn = flaky                                  # BEGIN succeeds, the INSERT after it fails
        with pytest.raises(dbm.LedgerUnavailable):
            reserve(store, "c1")
        store.db._local.conn = real
        assert not real.in_transaction                                # the failed write was rolled back
        assert store.records() == []
        assert reserve(store, "c2") is None                           # and the next writer is not blocked


def test_i5_a_commit_that_fails_is_retryable_and_rolled_back(tmp_path):
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        real = store.db.conn()
        store.db._local.conn = Flaky(real, "COMMIT", sqlite3.OperationalError("disk I/O error"))
        with pytest.raises(dbm.LedgerUnavailable):
            reserve(store, "c1")
        store.db._local.conn = real
        assert not real.in_transaction and store.records() == []


def test_i5_a_corruption_found_during_a_read_is_not_a_reason_to_move_the_file(tmp_path):
    d = make_history(tmp_path / "state")
    with ts.TelegraphStore.open(d) as store:
        store.db._local.conn = Flaky(store.db.conn(), "SELECT raw_json", sqlite3.DatabaseError("database disk image is malformed"))
        with pytest.raises(dbm.LedgerUnavailable):
            store.records()
    assert not list(d.glob("*.corrupt-*"))                            # only the open sequence moves a file, after its own check


def test_one_read_transaction_sees_one_snapshot(tmp_path):
    d = make_history(tmp_path / "state", 2)
    with dbm.Database(d) as db:
        with db.read() as conn:
            first = conn.execute("SELECT COUNT(*) FROM telegraph_events").fetchone()[0]
            with ts.TelegraphStore.open(d) as other:                  # another writer commits in the middle
                reserve(other, "late")
            second = conn.execute("SELECT COUNT(*) FROM telegraph_events").fetchone()[0]
        assert first == second == 2
        with db.read() as conn:
            assert conn.execute("SELECT COUNT(*) FROM telegraph_events").fetchone()[0] == 3


# =============================================================================================== I2: the unconfirmed state

def test_i2_a_first_ever_open_is_not_unconfirmed(tmp_path):
    with ts.TelegraphStore.open(tmp_path / "state") as store:
        assert reserve(store, "x") is None
        assert store.db.unreconciled() is None and not (tmp_path / "state" / "RECOVERY-REQUIRED").exists()


def test_i2_after_a_restore_from_an_older_backup_sending_is_refused_until_a_person_confirms(tmp_path):
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        reserve(store, "c0")
        backup = store.db.backup()
        for i in range(1, 6):                                         # reservations made after the backup
            reserve(store, f"c{i}")
    dbm.restore_backup(d, backup)
    with ts.TelegraphStore.open(d) as store:
        assert len(store.records()) == 1                              # the restored file lost five reservations
        why = reserve(store, "late", decide=lambda records: None)
        assert why and "confirm" in why
        assert len(store.records()) == 2                              # only the refusal was recorded
        store.db.confirm_reconciled("operator checked the provider's usage page")
        assert reserve(store, "late2") is None
    assert not (d / "RECOVERY-REQUIRED").exists()


def test_i2_a_moved_aside_database_starts_unconfirmed(tmp_path):
    d = make_history(tmp_path / "state")
    (d / "project.db").write_bytes(b"garbage" * 1000)
    assert "confirm" in blocked(d)


def test_i2_a_new_file_beside_existing_backups_starts_unconfirmed(tmp_path):
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        reserve(store, "c0")
        store.db.backup()
    (d / "project.db").unlink()                                       # the file is gone, the backups are not
    for suffix in ("-wal", "-shm"):
        (d / ("project.db" + suffix)).unlink(missing_ok=True)
    why = blocked(d)
    assert why and "confirm" in why


def test_i2_the_unconfirmed_state_survives_reopening_and_is_cleared_only_by_confirming(tmp_path):
    d = tmp_path / "state"
    d.mkdir(mode=0o700)
    (d / "project.db").write_bytes(b"garbage" * 1000)
    with dbm.Database(d):
        pass
    with dbm.Database(d) as db:
        assert db.unreconciled() is not None
        assert db.confirm_reconciled("checked") is True
    with dbm.Database(d) as db:
        assert db.unreconciled() is None and db.recovery_marker() is None
        assert db.confirm_reconciled("again") is False                # nothing left to clear


def test_i2_a_recovered_ledger_comes_back_from_the_newest_good_backup(tmp_path):
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        for i in range(3):
            reserve(store, f"c{i}")
        store.db.backup()
        reserve(store, "after-the-backup")
    (d / "project.db").write_bytes(b"x" * 5000)
    for suffix in ("-wal", "-shm"):
        (d / ("project.db" + suffix)).unlink(missing_ok=True)
    with ts.TelegraphStore.open(d) as store:
        assert [r["reserve"] for r in store.records()] == ["c0", "c1", "c2"]    # history after the backup is not in it
        assert "confirm" in reserve(store, "x")


def test_i2_an_unreadable_backup_is_skipped_for_the_next_one(tmp_path):
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        reserve(store, "c0")
        good = store.db.backup()
        time.sleep(0.01)
        reserve(store, "c1")
        bad = store.db.backup()
    Path(bad).write_bytes(b"not a database" * 100)
    (d / "project.db").write_bytes(b"x" * 5000)
    for suffix in ("-wal", "-shm"):
        (d / ("project.db" + suffix)).unlink(missing_ok=True)
    with ts.TelegraphStore.open(d) as store:
        assert [r["reserve"] for r in store.records()] == ["c0"]
    assert Path(good).exists()


def test_i2_a_marker_written_by_another_process_while_this_one_is_open_refuses_the_next_reservation(tmp_path):
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        assert reserve(store, "c1") is None
        with dbm.Database(d) as other:                                # another process starts a recovery
            other._write_marker("recovery started elsewhere")
        why = reserve(store, "c2")
        assert why and "confirm" in why


def test_i2_an_unconfirmed_row_written_by_another_process_while_this_one_is_open_refuses_the_next_reservation(tmp_path):
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        assert reserve(store, "c1") is None
        with dbm.Database(d) as other:
            other._write_unreconciled("marked elsewhere")
        why = reserve(store, "c2")
        assert why and "confirm" in why


# =============================================================================================== I7: recovery state before a replacement

I7_STEPS = ["recover:1", "recover:2", "recover:3", "recover:4", "recover:5"]


@pytest.mark.parametrize("step", I7_STEPS)
def test_i7_killed_at_each_step_of_the_recovery_a_second_opener_is_never_granted_a_reservation(tmp_path, step):
    d = make_history(tmp_path / "state", 3)
    (d / "project.db").write_bytes(b"garbage" * 1000)                 # step 1 is "decide recovery is needed"
    for suffix in ("-wal", "-shm"):
        (d / ("project.db" + suffix)).unlink(missing_ok=True)
    done = run_child("recover", d, step)
    assert done.returncode == KILLED, done.stderr
    if step == "recover:1":
        assert not (d / "RECOVERY-REQUIRED").exists()                 # nothing had changed: the next opener repeats from 1
    else:
        assert (d / "RECOVERY-REQUIRED").exists()                     # from step 2 on the marker is the authority
    why = blocked(d)                                                  # a second process opens and asks
    assert why and "confirm" in why
    assert (d / "RECOVERY-REQUIRED").exists()
    with dbm.Database(d) as db:
        assert db.unreconciled() is not None                          # the row mirrors the marker once an opener has finished
        assert db.confirm_reconciled("checked")
    assert blocked(d) is None and not (d / "RECOVERY-REQUIRED").exists()


@pytest.mark.parametrize("step", ["recover:2", "recover:3"])
def test_i7_with_the_marker_written_and_no_database_the_next_opener_creates_no_unmarked_file(tmp_path, step):
    d = make_history(tmp_path / "state", 2)
    (d / "project.db").write_bytes(b"garbage" * 1000)
    for suffix in ("-wal", "-shm"):
        (d / ("project.db" + suffix)).unlink(missing_ok=True)
    assert run_child("recover", d, step).returncode == KILLED
    if step == "recover:3":
        assert not (d / "project.db").exists()                        # moved aside, nothing created yet
    reader = dbm.ReadOnlyDatabase(d)
    try:
        assert reader.marker() is not None                            # a reader of the directory can see the state without a database
    finally:
        reader.close()
    assert "confirm" in blocked(d)                                    # and the opener that follows is refused all the same


def test_i7_the_marker_is_synced_before_anything_is_moved(tmp_path, monkeypatch):
    d = make_history(tmp_path / "state")
    (d / "project.db").write_bytes(b"garbage" * 1000)
    for suffix in ("-wal", "-shm"):
        (d / ("project.db" + suffix)).unlink(missing_ok=True)
    seen = {}
    real_rename = os.rename

    def spy(src, dst, *a, **k):
        if str(src) == "project.db":
            seen["marker_exists_at_first_rename"] = (d / "RECOVERY-REQUIRED").is_file()
            seen["marker_is_whole_json"] = (d / "RECOVERY-REQUIRED").read_text().rstrip().endswith("}")
        return real_rename(src, dst, *a, **k)
    monkeypatch.setattr(dbm.os, "rename", spy)
    with dbm.Database(d):
        pass
    assert seen == {"marker_exists_at_first_rename": True, "marker_is_whole_json": True}


def test_i7_every_open_holds_open_lock_for_the_whole_sequence(tmp_path):
    d = make_history(tmp_path / "state")
    import fcntl
    fd = os.open(d / "open.lock", os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        started = time.monotonic()
        with pytest.raises(dbm.LedgerUnavailable, match="open.lock"):
            dbm.Database(d, busy_timeout_ms=200)
        assert time.monotonic() - started >= 0.15                     # it waited for the lock, it did not skip it
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_i7_a_crash_between_clearing_the_row_and_removing_the_marker_keeps_refusing_until_confirmed_again(tmp_path):
    d = make_history(tmp_path / "state")
    (d / "project.db").write_bytes(b"garbage" * 1000)
    for suffix in ("-wal", "-shm"):
        (d / ("project.db" + suffix)).unlink(missing_ok=True)
    assert "confirm" in blocked(d)
    assert run_child("confirm", d, "confirm:row").returncode == KILLED
    assert (d / "RECOVERY-REQUIRED").exists()                         # the row is gone, the marker is not
    assert "confirm" in blocked(d)                                    # ... so the refusal continues
    with dbm.Database(d) as db:
        db.confirm_reconciled("checked again")
    assert blocked(d) is None


# =============================================================================================== I8: a staged restore

I8_STEPS = ["restore:1", "restore:2", "restore:3", "restore:4", "restore:5"]


def history_with_backup(d):
    with ts.TelegraphStore.open(d) as store:
        reserve(store, "c0")
        backup = store.db.backup()
        for i in range(1, 4):
            reserve(store, f"c{i}")
    return backup


@pytest.mark.parametrize("step", I8_STEPS)
def test_i8_killed_at_each_step_of_a_restore_reservations_stay_refused_and_a_rerun_finishes(tmp_path, step):
    d = tmp_path / "state"
    backup = history_with_backup(d)
    assert run_child("restore", d, step, backup).returncode == KILLED
    if step == "restore:1":
        assert not (d / "RECOVERY-REQUIRED").exists()                 # nothing was changed: the live file is still trusted
        assert blocked(d) is None
    else:
        assert (d / "RECOVERY-REQUIRED").exists()                     # at no later step is a live file neither marked nor covered
        why = blocked(d)
        assert why and "confirm" in why
    dbm.restore_backup(d, backup)                                     # run it again from the start
    with ts.TelegraphStore.open(d) as store:
        assert [r["reserve"] for r in store.records() if "reserve" in r] == ["c0"]
        assert "confirm" in reserve(store, "late")
        store.db.confirm_reconciled("checked")
    assert blocked(d) is None


def test_i8_the_old_live_file_and_its_wal_are_kept_and_not_deleted(tmp_path):
    d = tmp_path / "state"
    backup = history_with_backup(d)
    before = count(d)
    replaced = dbm.restore_backup(d, backup)
    assert replaced and (d / replaced).is_file()
    old = sqlite3.connect(d / replaced)
    try:
        assert old.execute("SELECT COUNT(*) FROM telegraph_events").fetchone()[0] == before
    finally:
        old.close()


def test_i8_the_staged_file_is_marked_before_it_is_installed(tmp_path):
    d = tmp_path / "state"
    backup = history_with_backup(d)
    assert run_child("restore", d, "restore:3", backup).returncode == KILLED      # staged, not yet installed
    staged = [p for p in d.iterdir() if p.name.startswith("project.db.restore-") and not p.name.endswith(".partial")]
    assert len(staged) == 1
    conn = sqlite3.connect(staged[0])
    try:
        assert conn.execute("SELECT value FROM meta WHERE key = 'unreconciled'").fetchone() is not None
    finally:
        conn.close()
    assert stat.S_IMODE(staged[0].stat().st_mode) == 0o600


def test_i8_a_restore_refuses_while_another_process_has_the_database_open(tmp_path):
    d = tmp_path / "state"
    backup = history_with_backup(d)
    with dbm.Database(d):                                             # a supervisor, the web page or a telegraph command
        with pytest.raises(dbm.LedgerRefused, match="close the other processes"):
            dbm.restore_backup(d, backup)
    assert not (d / "RECOVERY-REQUIRED").exists()                     # refused before anything was written


def test_i8_a_damaged_or_too_new_backup_is_refused_before_anything_is_touched(tmp_path):
    d = tmp_path / "state"
    history_with_backup(d)
    before = snapshot(d)
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"not a database" * 100)
    with pytest.raises(dbm.LedgerRefused):
        dbm.restore_backup(d, bad)
    newer = tmp_path / "newer.db"
    conn = sqlite3.connect(newer)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(dbm.DatabaseTooNew):
        dbm.restore_backup(d, newer)
    assert snapshot(d) == before and not (d / "RECOVERY-REQUIRED").exists()


# =============================================================================================== schema versions

def test_a_file_at_the_previous_schema_version_opens_and_migrates_with_a_backup_first(tmp_path, monkeypatch):
    d = make_history(tmp_path / "state")
    extra = lambda conn: conn.execute("CREATE TABLE extra_table (x INTEGER) STRICT")        # noqa: E731
    monkeypatch.setattr(dbm, "MIGRATIONS", [*dbm.MIGRATIONS, extra])
    before = len(list((d / "backups").glob("*.db"))) if (d / "backups").exists() else 0
    with dbm.Database(d, backup_interval_s=None) as db:
        assert db.user_version() == 2
        assert db.conn().execute("SELECT name FROM sqlite_master WHERE name = 'extra_table'").fetchone()
    backups = sorted((d / "backups").glob("*.db"))
    assert len(backups) == before + 1                                 # taken before the migration ran
    old = sqlite3.connect(backups[-1])
    try:
        assert old.execute("PRAGMA user_version").fetchone()[0] == 1
    finally:
        old.close()


def test_a_database_with_a_newer_schema_is_refused_by_name(tmp_path):
    d = make_history(tmp_path / "state")
    conn = sqlite3.connect(d / "project.db")
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(dbm.DatabaseTooNew, match="99"):
        dbm.Database(d)
    assert not list(d.glob("*.corrupt-*"))                            # a newer file is not damaged


def test_a_schema_version_that_turns_newer_after_the_write_lock_is_taken_is_refused(tmp_path):
    """Issue 153: the version read before the lock said 0, and another process migrated the file in between."""
    d = tmp_path / "state"
    d.mkdir(mode=0o700)
    conn = sqlite3.connect(d / "project.db")
    conn.execute("PRAGMA user_version = 99")
    conn.close()

    class StaleVersion(dbm.Database):
        def user_version(self):                                       # the read made before the lock
            return 0
    with pytest.raises(dbm.DatabaseTooNew):
        StaleVersion(d)


def test_two_processes_opening_a_file_at_an_older_version_migrate_it_once(tmp_path, monkeypatch):
    d = tmp_path / "state"
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(4) as pool:
        assert pool.map(open_once, [str(d)] * 4) == [1, 1, 1, 1]
    with dbm.Database(d) as db:
        tables = [r[0] for r in db.conn().execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    assert tables.count("telegraph_events") == 1


# =============================================================================================== backups

def test_backups_are_private_whole_and_the_newest_fourteen_are_kept(tmp_path):
    d = make_history(tmp_path / "state", 2)
    with dbm.Database(d, backup_interval_s=None) as db:
        paths = [db.backup() for _ in range(17)]
        listed = db.list_backups()
    assert len(listed) == dbm.BACKUP_KEEP == 14
    assert Path(paths[-1]).exists() and not Path(paths[0]).exists()
    assert mode(d / "backups") == 0o700 and mode(paths[-1]) == 0o600
    conn = sqlite3.connect(paths[-1])
    try:
        assert conn.execute("SELECT COUNT(*) FROM telegraph_events").fetchone()[0] == 2
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        conn.close()


def test_a_backup_is_taken_at_open_when_the_newest_is_older_than_the_interval_and_there_is_data(tmp_path):
    d = tmp_path / "state"
    with dbm.Database(d):
        pass
    assert not (d / "backups").exists()                               # an empty database is not worth a backup
    make_history(d, 1)
    with dbm.Database(d):
        pass
    first = sorted((d / "backups").glob("*.db"))
    assert len(first) == 1
    with dbm.Database(d):
        pass
    assert sorted((d / "backups").glob("*.db")) == first              # the newest is younger than an hour: no new one
    old = time.time() - 7200
    os.utime(first[0], (old, old))
    with dbm.Database(d):
        pass
    assert len(list((d / "backups").glob("*.db"))) == 2


def test_a_backup_dir_that_is_a_link_is_refused(tmp_path):
    d = make_history(tmp_path / "state", 1)
    outside = tmp_path / "outside"
    outside.mkdir()
    (d / "backups").symlink_to(outside)
    with dbm.Database(d, backup_interval_s=None) as db:
        with pytest.raises(dbm.UnsafeDirectory):
            db.backup()
    assert list(outside.iterdir()) == []
