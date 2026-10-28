"""Third review of PR 155 (head 4d18756). Two classes: every directory scan separates absent from unreadable, and everything created or
opened under the state directory goes through a directory descriptor that refuses a link before anything is created."""
import io
import os
import stat

import pytest

from dbtools import make_history, reserve
from ledger_io import ledger_records
from orchard import cli
from orchard import db as dbm
from orchard import db_cmd
from orchard import telegraph as tg
from orchard import telegraph_migrate as mig
from orchard import telegraph_store as ts
from test_telegraph_migrate import legacy_directory
from test_telegraph_store import KEY, Wire, table

root_user = os.geteuid() == 0


def run_cli(argv, tmp_path, env=None):
    out = io.StringIO()
    code = cli.main(argv, env={"HOME": str(tmp_path / "home"), **(env or {})}, stdout=out)
    return code, out.getvalue()


def recovering_directory(tmp_path):
    d = make_history(tmp_path / "state", 1)
    with dbm.Database(d) as db:
        db._write_marker("test")
        db._write_unreconciled("test")
    cfg = tmp_path / "t.toml"
    cfg.write_text('[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["small"]\n')
    return d, cfg


# ---- a reply scan that cannot finish does not let the recovery be cleared ---------------------------------------------------------

@pytest.mark.skipif(root_user, reason="root can enter a folder with no permissions")
def test_confirm_restore_refuses_when_a_scope_folder_of_replies_cannot_be_read(tmp_path):
    d, cfg = recovering_directory(tmp_path)
    (d / "replies" / "s~run").mkdir(parents=True)
    (d / "replies" / "s~run" / "abc.txt").write_text("an answer")
    (d / "replies" / "s~run").chmod(0)
    try:
        code, text = run_cli(["db", "confirm-restore", "--dir", str(d), "--note", "x", "--telegraph-config", str(cfg)], tmp_path)
        assert code == 2 and "could not be read completely" in text and (d / "RECOVERY-REQUIRED").exists()
        code, text = run_cli(["db", "confirm-restore", "--dir", str(d), "--note", "x", "--no-replies", "--telegraph-config", str(cfg)], tmp_path)
        assert code == 0 and not (d / "RECOVERY-REQUIRED").exists()        # a person who skips the replies says so
    finally:
        (d / "replies" / "s~run").chmod(0o700)


def test_confirm_restore_refuses_when_replies_is_not_a_folder(tmp_path):
    d, cfg = recovering_directory(tmp_path)
    (d / "replies").write_text("a file where the folder should be")
    code, text = run_cli(["db", "confirm-restore", "--dir", str(d), "--note", "x", "--telegraph-config", str(cfg)], tmp_path)
    assert code == 2 and (d / "RECOVERY-REQUIRED").exists()


def test_an_absent_replies_folder_and_a_linked_entry_are_not_failures(tmp_path):
    d, cfg = recovering_directory(tmp_path)
    ids, incomplete = db_cmd.reply_scan(d)
    assert ids == {} and incomplete == []                                   # nothing was ever answered: complete
    (d / "replies").mkdir()
    (d / "replies" / "link.txt").symlink_to(tmp_path)
    (d / "replies" / "s~linked").symlink_to(tmp_path)
    ids, incomplete = db_cmd.reply_scan(d)
    assert ids == {} and incomplete == []                                   # links are skipped on purpose, never followed
    code, text = run_cli(["db", "confirm-restore", "--dir", str(d), "--note", "x", "--telegraph-config", str(cfg)], tmp_path)
    assert code == 0


# ---- a link anywhere under the state directory is refused before anything is created ------------------------------------------------

@pytest.mark.parametrize("scope", ["default", "run-b"])
def test_a_symlinked_replies_folder_never_lets_a_call_write_outside_the_state_directory(tmp_path, scope):
    d = tmp_path / "state"
    d.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    (d / "replies").symlink_to(outside)
    cfg = tg.load(table(max_calls=5, max_calls_per_problem=9), base=tmp_path)
    got = tg.Telegraph(cfg, d, transport=Wire(), environ={"TG_KEY": KEY}, scope=scope).ask("p", "t", "f")
    assert got.ok                                                           # the answer is still returned and recorded
    assert list(outside.rglob("*")) == []                                   # nothing was created through the link


def test_a_symlinked_scope_folder_and_a_symlinked_reply_file_are_not_written_through(tmp_path):
    d = tmp_path / "state"
    (d / "replies").mkdir(parents=True, mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    (d / "replies" / "s~run-b").symlink_to(outside)
    cfg = tg.load(table(max_calls=5, max_calls_per_problem=9), base=tmp_path)
    tg.Telegraph(cfg, d, transport=Wire(), environ={"TG_KEY": KEY}, scope="run-b").ask("p", "t", "f")
    assert list(outside.rglob("*")) == []
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    t = tg.Telegraph(cfg, d, transport=Wire(), environ={"TG_KEY": KEY})
    (d / "replies" / "fixedid.txt").symlink_to(victim)
    t._write_reply("replies/fixedid.txt", "overwritten?")
    assert victim.read_text() == "keep"


def test_a_symlinked_backups_folder_is_refused_when_listing_and_when_taking_a_backup(tmp_path):
    d = make_history(tmp_path / "state", 1)
    outside = tmp_path / "outside"
    outside.mkdir()
    (d / "backups").symlink_to(outside)
    with pytest.raises(dbm.UnsafeDirectory):
        dbm.Database(d)
    assert list(outside.iterdir()) == []


@pytest.mark.skipif(root_user, reason="root can read a folder with no permissions")
def test_a_cache_folder_that_cannot_be_read_stops_the_migration_and_an_absent_one_does_not(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    (d / "cache").chmod(0)
    try:
        with pytest.raises(mig.MigrationRefused):
            mig.migrate(d)
        assert not (d / "MIGRATING-TO-SQLITE").exists()
    finally:
        (d / "cache").chmod(0o700)
    import shutil
    shutil.rmtree(d / "cache")
    assert mig.migrate(d)["phase"] == "complete"                           # absent is genuinely absent


def test_the_status_command_says_when_the_backups_cannot_be_listed(tmp_path):
    d = make_history(tmp_path / "state", 1)
    outside = tmp_path / "outside"
    outside.mkdir()
    (d / "backups").symlink_to(outside)
    st = ts.status_report(d)
    assert st["backups"] is None and st["backups_error"]
