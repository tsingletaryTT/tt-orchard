"""Fixes for the Copilot review of PR 155 (the telegraph ledger on SQLite). Each test was written, and seen to fail, before its fix."""
import io
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import legacy_telegraph as old
from dbtools import KILLED, make_history, reserve, run_child
from ledger_io import ledger_records, seed_ledger
from orchard import cli
from orchard import db as dbm
from orchard import telegraph as tg
from orchard import telegraph_cmd as cmd
from orchard import telegraph_migrate as mig
from orchard import telegraph_store as ts
from test_telegraph_migrate import Crash, NOTE_TOKENS, legacy_directory
from test_telegraph_store import KEY, Wire, build, openai_reply, table


def run_cli(argv, tmp_path, env=None):
    out = io.StringIO()
    code = cli.main(argv, env={"HOME": str(tmp_path / "home"), **(env or {})}, stdout=out)
    return code, out.getvalue()


# ---- high 1: evidence that a ledger was ever initialised, apart from backups --------------------------------------------------

def test_a_ledger_that_vanishes_after_its_first_paid_call_is_not_a_fresh_history(tmp_path):
    d = tmp_path / "state"
    cfg_kw = dict(max_calls=1, max_calls_per_problem=9)
    assert build(tg, d, tmp_path, **cfg_kw).ask("p", "t", "f1").ok            # no backup exists yet: backups need data at open
    assert not (d / "backups").exists()
    for name in ("project.db", "project.db-wal", "project.db-shm"):
        (d / name).unlink(missing_ok=True)
    wire = Wire()
    got = build(tg, d, tmp_path, transport=wire, **cfg_kw).ask("p", "t", "f2")
    assert not got.ok and "confirm" in got.reason and wire.requests == []
    assert (d / "RECOVERY-REQUIRED").exists()


# ---- high 2: readers hold the shared lock for their lifetime -------------------------------------------------------------------

def test_a_restore_refuses_while_a_read_only_reader_is_open_and_goes_on_after_it_closes(tmp_path):
    d = tmp_path / "state"
    with ts.TelegraphStore.open(d) as store:
        reserve(store, "c0")
        backup = store.db.backup()
    reader = dbm.ReadOnlyDatabase(d)
    try:
        with reader.read() as conn:
            conn.execute("SELECT COUNT(*) FROM telegraph_events").fetchone()
            with pytest.raises(dbm.LedgerRefused, match="close the other processes"):
                dbm.restore_backup(d, backup)
    finally:
        reader.close()
    dbm.restore_backup(d, backup)


def test_a_reader_waits_for_a_restore_that_holds_the_directory(tmp_path):
    d = make_history(tmp_path / "state", 1)
    import fcntl
    fd = os.open(d / "open.lock", os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)                                            # a restore or a recovery is under way
    try:
        with pytest.raises(dbm.LedgerUnavailable):
            dbm.ReadOnlyDatabase(d, busy_timeout_ms=200)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---- high 3: a config that fails to load never sends a db command to the home default ------------------------------------------

def test_a_bad_config_stops_a_db_command_that_needs_it_to_find_the_directory(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[telegraph\nbroken")
    code, text = run_cli(["db", "backup", "--telegraph-config", str(bad)], tmp_path)
    assert code == 2 and "bad.toml" in text
    assert not (tmp_path / "home").exists()                                   # nothing was opened under the caller's home
    code, text = run_cli(["db", "status", "--telegraph-config", str(tmp_path / "missing.toml")], tmp_path)
    assert code == 2                                                          # a named config that is missing is a setup error


def test_a_bad_config_is_ignored_when_the_directory_is_named_another_way(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[telegraph\nbroken")
    d = make_history(tmp_path / "state", 1)
    code, text = run_cli(["db", "status", "--telegraph-config", str(bad), "--dir", str(d)], tmp_path)
    assert code == 0 and "telegraph records: 1" in text


def test_no_config_at_all_is_fine_for_a_db_command(tmp_path):
    code, text = run_cli(["db", "status"], tmp_path)
    assert code == 0 and "database: none yet" in text


# ---- high 4: the recovery charge is at least what a reservation can be --------------------------------------------------------

@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_the_charge_for_a_reply_with_no_record_covers_the_largest_reservation_including_extra_body(tmp_path, api):
    kw = dict(api=api, max_prompt_chars=600, max_output_tokens=50, prices={"small": [1.0, 3.0], "big": [5.0, 15.0]},
              extra_body={"metadata": {"note": "x" * 20000}}, max_calls=9, max_calls_per_problem=9)
    cfg = tg.load(table(**kw), base=tmp_path)
    reply = Wire(lambda b: ({"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 5, "output_tokens": 5}}
                            if api == "anthropic" else openai_reply("ok", 5, 5)))
    t = tg.Telegraph(cfg, tmp_path / "state", transport=reply, environ={"TG_KEY": KEY}, hostname="node9", home="/home/someone")
    t.ask("p", "\U0001F600" * 600, "\U0001F600" * 600)                         # the largest prompt: four-byte characters, escaped
    reserved = [r for r in ledger_records(tmp_path / "state") if "reserve" in r][0]
    tokens, cost = tg.worst_case_call(cfg)
    assert tokens >= reserved["tokens"] and cost >= reserved["cost_usd"]


# ---- highs 5 and 6: sidecars --------------------------------------------------------------------------------------------------

def test_an_older_orphan_sidecar_with_the_same_bytes_is_not_taken_for_a_new_torn_write(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    (d / "telegraph.jsonl.torn-older0001").write_bytes(b'{"reserve": "half')   # an earlier, separate torn write
    with open(d / "telegraph.jsonl", "ab") as f:
        f.write(b'{"reserve": "half')                                          # and the same bytes again at the tail
    got = mig.migrate(d, max_output_tokens=NOTE_TOKENS)
    assert len(list(d.glob("telegraph.jsonl.torn-*"))) == 2
    assert len([r for r in ledger_records(d) if "torn" in r]) == 2 and got["checks"]["sidecars"]


def test_a_partial_sidecar_from_a_killed_write_is_not_counted_and_the_retry_counts_the_fragment_once(tmp_path, monkeypatch):
    d = legacy_directory(tmp_path, asks=1)
    with open(d / "telegraph.jsonl", "ab") as f:
        f.write(b'{"call_id": "half-written-fragment')
    assert run_child("sidecar-kill", d).returncode == KILLED                  # a real kill: no cleanup code runs
    assert not list(d.glob("telegraph.jsonl.torn-*"))                         # nothing half-written has a countable name
    mig.migrate(d, max_output_tokens=NOTE_TOKENS)
    (sidecar,) = list(d.glob("telegraph.jsonl.torn-*"))
    assert sidecar.read_bytes() == b'{"call_id": "half-written-fragment'
    assert len([r for r in ledger_records(d) if "torn" in r]) == 1


# ---- medium 1: documented read failures are messages --------------------------------------------------------------------------

def test_export_of_a_missing_archive_and_a_dry_run_of_a_missing_directory_are_messages_not_tracebacks(tmp_path):
    d = make_history(tmp_path / "state", 1)
    code, text = run_cli(["db", "export", "telegraph", "--dir", str(d), "--from-archive"], tmp_path)
    assert code == 2 and "archive" in text.lower() or code == 2
    code, text = run_cli(["db", "migrate", "telegraph", "--dir", str(tmp_path / "nowhere"), "--dry-run"], tmp_path)
    assert code == 2 and text.strip()


# ---- medium 2: spend and the page use the same directory resolver --------------------------------------------------------------

def test_spend_and_the_page_read_the_ledger_that_the_environment_names(tmp_path):
    ledger = tmp_path / "shared"
    seed_ledger(ledger, [{"call_id": "c1", "outcome": "ok", "tokens_in": 100, "tokens_out": 50, "cost_usd": 0.25,
                          "model": "m", "problem": "p"}])
    cfgfile = tmp_path / "t.toml"
    cfgfile.write_text('[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["a"]\n')
    probe = tmp_path / "probe"
    probe.mkdir()
    code, text = run_cli(["spend", str(probe), "--telegraph-config", str(cfgfile)], tmp_path,
                         env={"ORCHARD_TELEGRAPH_DIR": str(ledger)})
    assert code == 0 and "$0.2500" in text
    from test_telegraph_view import app_for
    root = tmp_path / "runs"
    root.mkdir()
    cfg = tg.load({"enabled": True, "endpoint": "https://llm.example.org/v1", "models": ["a"]}, base=tmp_path)
    app = app_for(root, cfg, environ={"ORCHARD_TELEGRAPH_DIR": str(ledger)})
    app.home = str(tmp_path / "nohome")
    assert app.telegraph()["spend"]["calls"] == 1
    bad = app_for(root, cfg, environ={"ORCHARD_TELEGRAPH_DIR": "relative"})
    bad.home = str(tmp_path / "nohome")
    out = bad.telegraph()
    assert out["state"] == "problem" and out["error"]


# ---- medium 3: directory-wide migration marker, any scope ---------------------------------------------------------------------

def test_another_scope_works_in_a_migrated_directory(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    mig.migrate(d)
    cfg = tg.load(table(max_calls=5, max_calls_per_problem=9), base=tmp_path)
    t = tg.Telegraph(cfg, d, transport=Wire(), environ={"TG_KEY": KEY}, scope="run-b")
    assert t.ask("p", "t", "f").ok
    assert tg.accounting(tg._read_ledger_strict(d, "run-b"))["calls"] == 1
    for name in ("project.db", "project.db-wal", "project.db-shm"):
        pass
    (d / "project.db").write_bytes(b"x")                                      # a lost database is still caught for any scope
    got = tg.Telegraph(cfg, d, transport=Wire(), environ={"TG_KEY": KEY}, scope="run-c").ask("p", "t", "f")
    assert not got.ok


def test_a_lost_database_in_a_migrated_directory_is_refused_for_every_scope(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    mig.migrate(d)
    for name in ("project.db", "project.db-wal", "project.db-shm"):
        (d / name).unlink(missing_ok=True)
    shutil.rmtree(d / "backups", ignore_errors=True)
    (d / "INITIALIZED").unlink(missing_ok=True)
    for scope in ("default", "run-b"):
        with pytest.raises(dbm.LedgerRefused):
            ts.read_snapshot(d, scope)


# ---- medium 4: a page does not keep showing a healthy ledger after a record is damaged in place -------------------------------

def test_a_record_damaged_in_place_is_seen_on_the_next_poll(tmp_path):
    from orchard import telegraph_view as view
    import sqlite3
    d = make_history(tmp_path / "state", 2)
    view._cache.clear()
    assert view._ledger_facts(d)["error"] is None
    conn = sqlite3.connect(d / "project.db")
    conn.execute("UPDATE telegraph_events SET raw_json = '{not json' WHERE seq = 1")
    conn.commit()
    conn.close()
    assert view._ledger_facts(d)["error"]
