"""Fixes for the second Copilot review of PR 155 (head d940e5d). Tests first. Each fix is for a class: every path put in a SQLite URI,
every place that picks a scope, every bound on how much redaction can enlarge a request."""
import io
import json
import os
import sqlite3
from pathlib import Path

import pytest

from dbtools import KILLED, make_history, reserve, run_child
from ledger_io import cache_rows, ledger_records, seed_ledger
from orchard import cli
from orchard import db as dbm
from orchard import telegraph as tg
from orchard import telegraph_migrate as mig
from orchard import telegraph_store as ts
from test_telegraph_migrate import Crash, crash_at, legacy_directory
from test_telegraph_store import KEY, Wire, build, openai_reply, table


def run_cli(argv, tmp_path, env=None):
    out = io.StringIO()
    code = cli.main(argv, env={"HOME": str(tmp_path / "home"), **(env or {})}, stdout=out)
    return code, out.getvalue()


# ---- every path that goes into a SQLite URI is encoded ---------------------------------------------------------------------------

AWKWARD = "odd#dir?x=1%20y"


def test_a_directory_with_uri_characters_is_read_backed_up_and_restored_without_touching_another_file(tmp_path):
    d = tmp_path / AWKWARD / "state"
    with ts.TelegraphStore.open(d) as store:
        reserve(store, "c0")
        backup = store.db.backup()
    assert len(ts.read_snapshot(d).records) == 1                           # the reader opened the right file
    assert ts.fingerprint(d)[0] is not None and ts.read_export_lines(d)
    stray = [p for p in tmp_path.rglob("*") if p.name in ("odd", "project.db?mode=ro")]
    assert stray == []                                                     # no file was created by a URI read as a fragment
    dbm.restore_backup(d, backup)                                          # the backup probe and the staged check use URIs too
    with dbm.Database(d) as db:
        assert db.unreconciled() is not None


def test_a_backup_path_with_uri_characters_is_checked_as_that_file(tmp_path):
    d = make_history(tmp_path / "state", 1)
    odd = tmp_path / "b#ackup?.db"
    with dbm.Database(d) as db:
        src = db.backup()
    import shutil
    shutil.copy(src, odd)
    dbm.restore_backup(d, odd)
    assert not (tmp_path / "b").exists()


# ---- a database truncated to nothing is a lost history ------------------------------------------------------------------------------

def test_a_project_db_truncated_to_zero_bytes_beside_initialized_goes_to_marked_recovery_and_keeps_the_wal(tmp_path):
    d = tmp_path / "state"
    t = build(tg, d, tmp_path, max_calls=1, max_calls_per_problem=9)
    assert t.ask("p", "t", "f1").ok
    (d / "project.db").write_bytes(b"")
    (d / "project.db-wal").write_bytes(b"wal evidence")
    wire = Wire()
    got = build(tg, d, tmp_path, transport=wire, max_calls=1, max_calls_per_problem=9).ask("p", "t", "f2")
    assert not got.ok and "confirm" in got.reason and wire.requests == []
    assert (d / "RECOVERY-REQUIRED").exists()
    moved = [p for p in d.glob("project.db.corrupt-*-wal")]
    assert len(moved) == 1 and moved[0].read_bytes() == b"wal evidence"


def test_a_valid_but_empty_database_in_place_of_the_ledger_is_a_lost_history_too(tmp_path):
    d = tmp_path / "state"
    make_history(d, 2)
    for name in ("project.db", "project.db-wal", "project.db-shm"):
        (d / name).unlink(missing_ok=True)
    sqlite3.connect(d / "project.db").close()                               # a fresh valid file with no schema
    with dbm.Database(d) as db:
        assert db.unreconciled() is not None


def test_a_zero_byte_file_with_no_history_is_just_a_new_database(tmp_path):
    d = tmp_path / "state"
    d.mkdir(mode=0o700)
    (d / "project.db").write_bytes(b"")
    with dbm.Database(d) as db:
        assert db.unreconciled() is None


# ---- reply files keep their scope --------------------------------------------------------------------------------------------------

def test_a_reply_is_written_under_its_scope_and_confirm_restore_charges_each_scope_its_own(tmp_path):
    d = tmp_path / "state"
    cfg = tg.load(table(max_calls=9, max_calls_per_problem=9, max_output_tokens=100, max_prompt_chars=500), base=tmp_path)
    for scope in ("default", "run a/b"):
        got = tg.Telegraph(cfg, d, transport=Wire(), environ={"TG_KEY": KEY}, scope=scope).ask("p", "t", f"f {scope}")
        assert got.ok
    assert len(list((d / "replies").glob("*.txt"))) == 1                    # the default scope keeps the old layout
    scoped = [p for p in (d / "replies").iterdir() if p.is_dir()]
    assert len(scoped) == 1 and scoped[0].name.startswith("s~") and len(list(scoped[0].glob("*.txt"))) == 1
    for scope in ("default", "run a/b"):                                    # the records are lost, the replies are not
        with ts.TelegraphStore.open(d, scope=scope) as store:
            with store.db.write() as conn:
                conn.execute("DELETE FROM telegraph_events WHERE scope = ?", (scope,))
    with dbm.Database(d) as db:
        db._write_marker("test")
        db._write_unreconciled("test")
    cfgfile = tmp_path / "t.toml"
    cfgfile.write_text('[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["small"]\n')
    code, text = run_cli(["db", "confirm-restore", "--dir", str(d), "--note", "checked", "--telegraph-config", str(cfgfile)], tmp_path)
    assert code == 0, text
    for scope in ("default", "run a/b"):
        recovered = [r for r in ledger_records(d, scope) if r.get("recovered")]
        assert len(recovered) == 1, (scope, text)                           # each scope has exactly its own reply
    assert not (d / "RECOVERY-REQUIRED").exists()


# ---- redaction can enlarge a request, and the recovery charge covers it -----------------------------------------------------------

@pytest.mark.parametrize("api", ["openai", "anthropic"])
def test_the_recovery_charge_covers_a_one_character_key_that_expands_extra_body_ten_times(tmp_path, api):
    kw = dict(api=api, max_prompt_chars=600, max_output_tokens=50, prices={"small": [1.0, 3.0], "big": [5.0, 15.0]},
              extra_body={"metadata": {"note": "x" * 20000}}, max_calls=9, max_calls_per_problem=9)
    cfg = tg.load(table(**kw), base=tmp_path)
    reply = Wire(lambda b: ({"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 5, "output_tokens": 5}}
                            if api == "anthropic" else openai_reply("ok", 5, 5)))
    t = tg.Telegraph(cfg, tmp_path / "state", transport=reply, environ={"TG_KEY": "x"}, hostname="a", home="/home/someone")
    t.ask("p", "x" * 600, "x" * 600)
    reserved = [r for r in ledger_records(tmp_path / "state") if "reserve" in r][0]
    assert reserved["tokens"] > 200000                                      # the body really did grow to about 200,000 characters
    tokens, cost = tg.worst_case_call(cfg)
    assert tokens >= reserved["tokens"] and cost >= reserved["cost_usd"]


# ---- a resumed migration keeps its scope ---------------------------------------------------------------------------------------------

def test_a_migration_started_for_one_scope_cannot_be_resumed_into_another(tmp_path):
    d = legacy_directory(tmp_path, asks=2)
    with pytest.raises(Crash):
        mig.migrate(d, scope="A", hook=crash_at("after:3"))
    with pytest.raises(mig.MigrationRefused, match="scope A"):
        mig.migrate(d)                                                      # the default scope would import the records again
    assert ledger_records(d, "default") == []
    assert mig.migrate(d, scope="A")["phase"] == "complete"
    assert len(ledger_records(d, "A")) > 0


def test_a_marker_that_cannot_be_read_stops_a_resume(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    with pytest.raises(Crash):
        mig.migrate(d, hook=crash_at("after:2"))
    (d / "MIGRATING-TO-SQLITE").write_text("not json")
    with pytest.raises(mig.MigrationRefused):
        mig.migrate(d)


# ---- spend --dir wins over a bad environment ------------------------------------------------------------------------------------------

def test_spend_with_an_explicit_dir_ignores_a_bad_directory_variable_and_an_agent_shell_mark(tmp_path):
    ledger = tmp_path / "shared"
    seed_ledger(ledger, [{"call_id": "c1", "outcome": "ok", "tokens_in": 1, "tokens_out": 1, "cost_usd": 0.25, "model": "m", "problem": "p"}])
    cfgfile = tmp_path / "t.toml"
    cfgfile.write_text('[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["a"]\n')
    probe = tmp_path / "probe"
    probe.mkdir()
    for env in ({"ORCHARD_TELEGRAPH_DIR": "relative"}, {"ORCHARD_RUN_DIR": str(tmp_path / "run")}):
        code, text = run_cli(["spend", str(probe), "--dir", str(ledger), "--telegraph-config", str(cfgfile)], tmp_path, env=env)
        assert code == 0 and "$0.2500" in text, (env, text)
    code, text = run_cli(["spend", str(probe), "--dir", str(ledger)], tmp_path, env={"ORCHARD_TELEGRAPH_DIR": "relative"})
    assert code == 0 and "$0.2500" in text                                  # the branch with no tables file too


# ---- a missing database beside INITIALIZED is an error for readers --------------------------------------------------------------------

def test_every_reader_reports_a_missing_database_when_the_directory_had_a_ledger(tmp_path):
    d = make_history(tmp_path / "state", 2)
    for name in ("project.db", "project.db-wal", "project.db-shm"):
        (d / name).unlink(missing_ok=True)
    import shutil
    shutil.rmtree(d / "backups", ignore_errors=True)
    assert (d / "INITIALIZED").exists()
    with pytest.raises(dbm.LedgerRefused, match="missing"):
        ts.read_snapshot(d)
    with pytest.raises(dbm.LedgerRefused, match="missing"):
        ts.read_export_lines(d)
    assert tg._read_ledger(d) == []                                         # the display reader never shows a guess
    cfgfile = tmp_path / "t.toml"
    cfgfile.write_text('[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["a"]\n')
    code, text = run_cli(["telegraph", "usage", "--dir", str(d), "--telegraph-config", str(cfgfile)], tmp_path)
    assert code == 2 and "spend is unknown" in text
    from orchard import telegraph_view as view
    view._cache.clear()
    assert view._ledger_facts(d)["error"]
