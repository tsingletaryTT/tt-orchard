"""`tt-orchard db migrate telegraph` (orchard/telegraph_migrate.py): the import is equal to the ledger it came from, the cutover stops old
writers, a crash at any step resumes, and torn fragments and sidecars are recovered or block the migration (I3, I9, I10).

The "old writer" in these tests is the frozen jsonl implementation (tests/legacy_telegraph.py), run against the same directory.
"""
import hashlib
import io
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

import legacy_telegraph as old
from dbtools import KILLED, run_child, snapshot
from ledger_io import cache_rows, ledger_lines, ledger_records
from orchard import cli, db_cmd
from orchard import db as dbm
from orchard import telegraph as tg
from orchard import telegraph_migrate as mig
from orchard import telegraph_store as ts
from test_telegraph_store import KEY, PRICES, SCENARIOS, Wire, build, normalise, table

NOTE_TOKENS = 777


class Crash(Exception):
    """Raised by the step hook to stop the command at a named step, as a kill would."""


def crash_at(step):
    def hook(name):
        if name == step:
            raise Crash(name)
    return hook


def legacy_directory(tmp_path, asks=2, **kw):
    """A telegraph directory as the old writer leaves it: ledger, reply files, cache entries."""
    d = tmp_path / "tg"
    t = build(old, d, tmp_path, **kw)
    for i in range(asks):
        t.ask("p", "t", f"f{i}")
    return d


def old_writer(d, tmp_path, wire=None, **kw):
    """One ask by a process that still runs the old code. Returns (the answer, the number of requests it sent)."""
    wire = wire or Wire()
    kw.setdefault("max_calls", 3)
    kw.setdefault("max_calls_per_problem", 9)
    return build(old, d, tmp_path, transport=wire, **kw).ask("p", "t", "late question"), len(wire.requests)


def new_writer(d, tmp_path, **kw):
    wire = Wire()
    kw.setdefault("max_calls", 3)
    kw.setdefault("max_calls_per_problem", 9)
    return build(tg, d, tmp_path, transport=wire, **kw).ask("p", "t", "another late question"), len(wire.requests)


def files_of(d):
    return {p.name: p.read_bytes() for p in Path(d).iterdir() if p.is_file()}


def legacy_records(d):
    return [json.loads(line) for line in Path(d, "telegraph.jsonl").read_text().splitlines() if line.strip()]


# =============================================================================================== equality on the legacy scenarios

@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.__name__)
def test_a_migrated_directory_has_the_totals_of_the_ledger_it_came_from_and_keeps_deciding_the_same_way(tmp_path, scenario):
    d = tmp_path / "tg"
    scenario(old, d, tmp_path)
    original = (d / "telegraph.jsonl").read_bytes()
    records = legacy_records(d)
    sidecars_before = sorted(p.name for p in d.glob("telegraph.jsonl.torn-*"))
    got = mig.migrate(d, max_output_tokens=NOTE_TOKENS)
    assert got["phase"] == "complete" and got["checks"] == {"lines": True, "totals": True, "sidecars": True, "cache": True}
    assert got["export_matches_archive"] is True
    assert sidecars_before == []
    assert (d / "telegraph.jsonl.archived").read_bytes() == original            # the archive is the old ledger, byte for byte
    assert ledger_records(d) == records                                         # the export is the ledger, record for record
    assert tg.accounting(tg._read_ledger_strict(d)) == old.accounting(records)
    assert tg.summarise_records(tg._read_ledger_strict(d)) == old.summarise_records(records)
    assert len(cache_rows(d)) == len(list((d / "cache").glob("*.json")) if (d / "cache").exists() else [])


def test_after_the_migration_the_product_decides_like_the_old_code_would_have(tmp_path):
    """Continue the same history in two copies: one stays a jsonl ledger run by the old code, one is migrated and run by the product."""
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        legacy_directory(root, asks=2, max_calls=3, max_calls_per_problem=9)
    mig.migrate(b / "tg")
    got_old = build(old, a / "tg", a, max_calls=3, max_calls_per_problem=9)
    got_new = build(tg, b / "tg", b, max_calls=3, max_calls_per_problem=9)
    for i in range(3):
        x, y = got_old.ask("p", "t", f"next{i}"), got_new.ask("p", "t", f"next{i}")
        assert (x.ok, x.reason, x.cached, x.tokens_in, x.cost_usd) == (y.ok, y.reason, y.cached, y.tokens_in, y.cost_usd)
    assert not y.ok and "call cap" in y.reason                                  # and the cap of 3 held across the cutover


def test_the_cache_entries_of_the_old_directory_are_imported_and_still_answer_a_repeated_question(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    assert len(list((d / "cache").glob("*.json"))) == 1
    got = mig.migrate(d)
    assert got["cache_imported"] == 1 and len(cache_rows(d)) == 1
    wire = Wire()
    again = build(tg, d, tmp_path, transport=wire).ask("p", "t", "f0")
    assert again.cached and wire.requests == []
    assert (d / "cache").is_dir() and len(list((d / "cache").glob("*.json"))) == 1       # nothing was deleted


def test_reply_files_and_unrelated_files_are_left_alone(tmp_path):
    d = legacy_directory(tmp_path, asks=2)
    (d / "decisions.md").write_text("notes a person wrote\n")
    (d / "backfill-sources.json").write_text("{}")
    before = {str(p.relative_to(d)): p.read_bytes() for p in d.rglob("*") if p.is_file() and p.parts[-2] in ("replies", d.name, "cache")
              and p.name not in ("telegraph.jsonl", "telegraph.lock")}
    mig.migrate(d)
    after = {str(p.relative_to(d)): p.read_bytes() for p in d.rglob("*") if p.is_file()}
    for name, data in before.items():
        assert after[name] == data
    assert (d / "decisions.md").read_text() == "notes a person wrote\n"


def test_a_directory_with_no_ledger_is_still_cut_over_so_an_old_writer_cannot_start_one(tmp_path):
    d = tmp_path / "tg"
    d.mkdir()
    got = mig.migrate(d)
    assert got["phase"] == "complete" and (d / "telegraph.jsonl").is_dir() and not (d / "telegraph.jsonl.archived").exists()
    answer, sent = old_writer(d, tmp_path)
    assert not answer.ok and sent == 0


# =============================================================================================== a dry run

def test_a_dry_run_reports_equal_totals_and_changes_nothing(tmp_path):
    d = legacy_directory(tmp_path, asks=3)
    before = (sorted(p.name for p in d.rglob("*")), files_of(d))
    got = mig.migrate(d, dry_run=True)
    assert got["status"] == "dry-run" and got["equal"] is True and got["totals"]["calls"] == 3
    assert (sorted(p.name for p in d.rglob("*")), files_of(d)) == before        # no lock file, no marker, no database
    assert not (d / "project.db").exists() and not (d / "MIGRATING-TO-SQLITE").exists()


def test_a_dry_run_counts_a_fragment_and_a_sidecar_the_way_the_real_run_will(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    with open(d / "telegraph.jsonl", "ab") as f:
        f.write(b'{"call_id": "half')
    (d / "telegraph.jsonl.torn-abc12345").write_bytes(b"{")
    before = files_of(d)
    got = mig.migrate(d, dry_run=True, max_output_tokens=NOTE_TOKENS)
    assert got["fragment_to_move"] is True and got["sidecars"] == 1 and got["equal"] and files_of(d) == before
    real = mig.migrate(d, max_output_tokens=NOTE_TOKENS)
    assert real["totals"]["accounted_calls"] == got["totals"]["accounted_calls"] == 1 + 2


def test_a_dry_run_of_a_migrated_directory_says_so(tmp_path):
    d = legacy_directory(tmp_path)
    mig.migrate(d)
    with pytest.raises(mig.AlreadyMigrated):
        mig.migrate(d, dry_run=True)


# =============================================================================================== I3: the cutover stops old writers

def test_i3_after_migration_an_old_telegraph_refuses_every_call_and_sends_nothing(tmp_path):
    d = legacy_directory(tmp_path)
    original = (d / "telegraph.jsonl").read_bytes()
    mig.migrate(d)
    answer, sent = old_writer(d, tmp_path)
    assert not answer.ok and "ledger cannot be written or read" in answer.reason and sent == 0
    assert (d / "telegraph.jsonl").is_dir()                                     # nothing recreated the ledger as a file
    assert (d / "telegraph.jsonl.archived").read_bytes() == original
    assert (d / "MIGRATED-TO-SQLITE").is_file() and not (d / "MIGRATING-TO-SQLITE").exists()
    marker = json.loads((d / "MIGRATED-TO-SQLITE").read_text())
    assert marker["sha256"] == hashlib.sha256(original).hexdigest() and marker["records"] == len(legacy_records_of(original))


def legacy_records_of(data):
    return [line for line in data.decode().splitlines() if line.strip()]


def test_i3_the_settlement_of_a_request_in_flight_during_the_cutover_cannot_recreate_the_ledger(tmp_path):
    d = legacy_directory(tmp_path)
    done = {}

    def cutover_during_request(url, headers, body, timeout):
        done["migrated"] = mig.migrate(d)                                       # the cutover happens while the request is out
        return 200, json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 10}}).encode()
    cfg = old.load(table(max_calls=50), base=tmp_path)
    got = old.Telegraph(cfg, d, transport=cutover_during_request, environ={"TG_KEY": KEY}).ask("p", "t", "in flight")
    assert done["migrated"]["phase"] == "complete"
    assert (d / "telegraph.jsonl").is_dir()                                     # the late settlement could not recreate it
    assert "could not record" in got.reason                                     # and the old code said so
    acct = tg.accounting(tg._read_ledger_strict(d))
    assert acct["calls"] == 3                                                   # two settled before, the reservation of the third
    assert tg.summarise_records(tg._read_ledger_strict(d))["unsettled"] == 1    # counted at its worst case, never lost


def test_i3_new_code_refuses_a_directory_that_holds_the_marker_and_a_live_jsonl(tmp_path):
    d = legacy_directory(tmp_path)
    mig.migrate(d)
    assert tg._read_ledger_strict(d)                                            # the migrated directory is fine
    shutil.rmtree(d / "telegraph.jsonl")
    (d / "telegraph.jsonl").write_text('{"call_id": "late"}\n')                 # a legacy writer got a regular file back
    with pytest.raises(ts.LegacyLedgerPresent):
        tg._read_ledger_strict(d)
    answer, sent = new_writer(d, tmp_path)
    assert not answer.ok and sent == 0


def test_i3_a_migration_whose_totals_differ_changes_nothing(tmp_path, monkeypatch):
    d = legacy_directory(tmp_path)
    before = files_of(d)
    real = mig.compare
    monkeypatch.setattr(mig, "compare", lambda *a, **k: {**real(*a, **k), "equal": False})
    with pytest.raises(mig.MigrationMismatch):
        mig.migrate(d)
    assert ledger_records(d) == []                                              # rolled back: the database holds nothing
    with dbm.Database(d) as db:
        assert db.meta_get("cutover:default") is None
    after = files_of(d)
    assert after["telegraph.jsonl"] == before["telegraph.jsonl"]
    assert not (d / "MIGRATED-TO-SQLITE").exists() and not (d / "MIGRATING-TO-SQLITE").exists()
    assert (d / "telegraph.jsonl").is_file()
    answer, sent = old_writer(d, tmp_path)                                      # the old writer is still free to work
    assert sent == 1


def test_i3_an_import_that_loses_or_changes_a_record_is_caught_by_the_equality_check_and_rolled_back(tmp_path, monkeypatch):
    """The check can fail on a real difference, not only when it is told to: the import here drops the last record, then changes one."""
    d = legacy_directory(tmp_path, asks=3)
    real = ts.TelegraphStore.import_records
    for damage in ("drops the last record", "changes a cost"):
        def broken(self, conn, items, damage=damage):
            if damage.startswith("drops"):
                items = items[:-1]
            else:
                items = [(dict(r, cost_usd=(r.get("cost_usd") or 0) + 1.0) if "cost_usd" in r and "call_id" in r else r, raw)
                         if i == 1 else (r, raw) for i, (r, raw) in enumerate(items)]
                items = [(r, json.dumps(r, sort_keys=True) if i == 1 else raw) for i, (r, raw) in enumerate(items)]
            return real(self, conn, items)
        monkeypatch.setattr(ts.TelegraphStore, "import_records", broken)
        with pytest.raises(mig.MigrationMismatch):
            mig.migrate(d)
        assert ledger_records(d) == [] and (d / "telegraph.jsonl").is_file() and not (d / "MIGRATING-TO-SQLITE").exists()
        monkeypatch.setattr(ts.TelegraphStore, "import_records", real)
    assert mig.migrate(d)["phase"] == "complete"                                # and the unbroken import goes through


def test_i3_migrating_twice_is_refused_and_imports_nothing_twice(tmp_path):
    d = legacy_directory(tmp_path)
    first = mig.migrate(d)
    with pytest.raises(mig.AlreadyMigrated):
        mig.migrate(d)
    assert len(ledger_records(d)) == first["totals"]["records"]


def test_a_directory_whose_database_has_rows_but_no_migration_record_is_not_imported_into(tmp_path):
    d = legacy_directory(tmp_path)
    with ts.TelegraphStore.open(d, scope="default") as store:
        with store.db.write() as conn:
            store.import_records(conn, [({"report": "x", "helped": True}, '{"helped": true, "report": "x"}')])
    with pytest.raises(mig.MigrationRefused, match="second history"):
        mig.migrate(d)
    assert not (d / "MIGRATING-TO-SQLITE").exists()


# =============================================================================================== I9: sidecars and torn fragments

def test_i9_a_torn_last_line_is_moved_to_a_private_sidecar_and_counted_once(tmp_path):
    d = legacy_directory(tmp_path, asks=2)
    with open(d / "telegraph.jsonl", "ab") as f:
        f.write(b'{"call_id": "half')
    got = mig.migrate(d, max_output_tokens=NOTE_TOKENS)
    (sidecar,) = list(d.glob("telegraph.jsonl.torn-*"))
    assert sidecar.read_bytes() == b'{"call_id": "half' and stat.S_IMODE(sidecar.stat().st_mode) == 0o600
    assert got["recovery"] == {"fragments_moved": 1, "notes_added": 1}
    notes = [r for r in ledger_records(d) if "torn" in r]
    assert notes == [{"torn": sidecar.name, "tokens": NOTE_TOKENS}]
    acct = tg.accounting(tg._read_ledger_strict(d))
    assert acct["calls"] == 2 + 1 and acct["unknown_cost"] == 3 and acct["tokens"] >= NOTE_TOKENS    # no prices here: every cost is unknown
    assert (d / "telegraph.jsonl.archived").read_bytes().endswith(b"\n")        # the archive is whole lines again


def test_i9_an_orphan_sidecar_with_no_note_gets_one_and_a_sidecar_with_a_note_does_not(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    (d / "telegraph.jsonl.torn-orphan001").write_bytes(b'{"reserve": "half')
    (d / "telegraph.jsonl.torn-noted0002").write_bytes(b"x")
    with open(d / "telegraph.jsonl", "a") as f:
        f.write(json.dumps({"torn": "telegraph.jsonl.torn-noted0002", "tokens": 10}) + "\n")
    got = mig.migrate(d, max_output_tokens=NOTE_TOKENS)
    notes = sorted((r["torn"], r["tokens"]) for r in ledger_records(d) if "torn" in r)
    assert notes == [("telegraph.jsonl.torn-noted0002", 10), ("telegraph.jsonl.torn-orphan001", NOTE_TOKENS)]
    assert got["recovery"]["notes_added"] == 1 and got["checks"]["sidecars"] is True


def test_i9_a_missing_ledger_with_a_sidecar_is_recovered_too(tmp_path):
    d = tmp_path / "tg"
    d.mkdir()
    (d / "telegraph.jsonl.torn-orphan001").write_bytes(b"frag")
    got = mig.migrate(d, max_output_tokens=NOTE_TOKENS)
    assert [r["torn"] for r in ledger_records(d)] == ["telegraph.jsonl.torn-orphan001"] and got["phase"] == "complete"


def test_i9_a_malformed_line_that_is_not_the_last_blocks_the_migration_and_names_the_line(tmp_path):
    d = legacy_directory(tmp_path, asks=2)
    lines = (d / "telegraph.jsonl").read_text().splitlines()
    lines.insert(1, "{this is not json")
    lines.insert(3, "[1, 2]")
    (d / "telegraph.jsonl").write_text("\n".join(lines) + "\n")
    before = files_of(d)
    with pytest.raises(mig.MalformedLedger) as info:
        mig.migrate(d)
    assert info.value.lines == [2, 4] and "2, 4" in str(info.value) and "--accept-malformed 2" in str(info.value)
    assert {k: v for k, v in files_of(d).items() if k in before} == before      # the old files are as they were
    assert not (d / "MIGRATING-TO-SQLITE").exists() and ledger_records(d) == []     # no marker, nothing imported


def test_i9_the_accepted_count_must_be_the_count_a_person_inspected(tmp_path):
    d = legacy_directory(tmp_path, asks=2)
    lines = (d / "telegraph.jsonl").read_text().splitlines()
    lines.insert(1, "{this is not json")
    (d / "telegraph.jsonl").write_text("\n".join(lines) + "\n")
    for wrong in (0, 2, None):
        with pytest.raises(mig.MalformedLedger):
            mig.migrate(d, accept_malformed=wrong)
    got = mig.migrate(d, accept_malformed=1)
    assert got["malformed_accepted"] == 1 and got["phase"] == "complete"
    assert "{this is not json" in (d / "telegraph.jsonl.archived").read_text()    # kept in the archive, verbatim
    assert "{this is not json" not in "\n".join(ledger_lines(d))                  # and not a record


def test_i9_blank_lines_are_not_malformed(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    with open(d / "telegraph.jsonl", "a") as f:
        f.write("\n   \n")
    assert mig.migrate(d)["phase"] == "complete"


def test_i9_a_note_for_a_sidecar_that_no_longer_exists_blocks_the_migration(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    with open(d / "telegraph.jsonl", "a") as f:
        f.write(json.dumps({"torn": "telegraph.jsonl.torn-gone0001", "tokens": 10}) + "\n")
    with pytest.raises(mig.MigrationMismatch, match="sidecars"):
        mig.migrate(d)
    assert not (d / "MIGRATING-TO-SQLITE").exists() and (d / "telegraph.jsonl").is_file()


def test_i9_recovery_after_a_crash_between_the_cut_and_the_note_still_counts_the_fragment_once(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    with open(d / "telegraph.jsonl", "ab") as f:
        f.write(b'{"call_id": "half')
    real = mig.safeio.append_line

    def die_before_the_note(*a, **k):
        raise Crash("before the note")
    mig.safeio.append_line = die_before_the_note
    try:
        with pytest.raises(Crash):
            mig.migrate(d, max_output_tokens=NOTE_TOKENS)
    finally:
        mig.safeio.append_line = real
    assert len(list(d.glob("telegraph.jsonl.torn-*"))) == 1                     # the fragment is safe in its sidecar, cut from the ledger
    mig.migrate(d, max_output_tokens=NOTE_TOKENS)                               # a rerun writes the missing note, once
    assert len([r for r in ledger_records(d) if "torn" in r]) == 1


# =============================================================================================== I10: a crash at any step, then a restart

STEPS = ["after:1", "after:2", "inside:3", "after:3", "inside:4", "after:4", "inside:5", "after:5", "after:6a", "after:6b", "after:6c"]


def check_directory_after_a_crash(d, tmp_path, step):
    """What an old writer and the product do against a directory left at `step`, and what the resumed command makes of it."""
    before = len(legacy_records(d)) if (d / "telegraph.jsonl").is_file() else None
    answer_old, sent_old = old_writer(d, tmp_path)
    answer_new, sent_new = new_writer(d, tmp_path)
    assert sent_old + sent_new <= 1, (step, "two writers both sent: a second history")        # the cap was 3 and two calls were made
    return answer_old, sent_old, answer_new, sent_new, before


@pytest.mark.parametrize("step", STEPS)
def test_i10_a_crash_at_each_step_resumes_and_no_phase_lets_a_writer_send_into_a_second_history(tmp_path, step):
    d = legacy_directory(tmp_path, asks=2, max_calls=3, max_calls_per_problem=9)
    originals = legacy_records(d)
    with pytest.raises(Crash):
        mig.migrate(d, hook=crash_at(step))
    answer_old, sent_old, answer_new, sent_new, _ = check_directory_after_a_crash(d, tmp_path, step)
    late_records = 0
    if step in ("after:4", "inside:4"):
        # The only window the old lock does not cover: no file of that name exists, and the old writer recreates it.
        recreated = (d / "telegraph.jsonl").is_file()
        if recreated:
            late_records = len(legacy_records(d))
            with pytest.raises(mig.MigrationRefused):
                mig.migrate(d)                                                  # the resume refuses: a person reconciles the late records
            assert not answer_new.ok and sent_new == 0                          # and the product refuses while the file is back
            return
    if step == "after:6c":                                                      # the command had finished when it was killed
        with pytest.raises(mig.AlreadyMigrated):
            mig.migrate(d)
        got = {"phase": "complete"}
    else:
        got = mig.migrate(d, max_output_tokens=NOTE_TOKENS)                     # the restart
    assert got["phase"] == "complete" and (d / "telegraph.jsonl").is_dir()
    assert (d / "MIGRATED-TO-SQLITE").is_file() and not (d / "MIGRATING-TO-SQLITE").exists()
    records = ledger_records(d)
    sent = sent_old + sent_new
    assert [r for r in records if "call_id" in r and r.get("outcome") == "ok"][:2] == [r for r in originals if r.get("outcome") == "ok"]
    assert tg.accounting(records)["calls"] == 2 + sent                          # every call that was sent is in the one history
    archive = (d / "telegraph.jsonl.archived").read_text().splitlines()
    assert [json.loads(line) for line in archive] == records[:len(archive)]     # the archive is the start of the database's story
    assert len(records) - len(archive) <= (2 if step == "after:6c" else 0)     # (only a call made after the command had finished is later)
    answer, new_sent = new_writer(d, tmp_path)
    assert (new_sent == 1) == (2 + sent < 3)                                    # and the cap of 3 still holds after the cutover


@pytest.mark.parametrize("step", STEPS)
def test_i10_killed_with_sigkill_at_each_step_the_command_resumes(tmp_path, step):
    d = legacy_directory(tmp_path, asks=2, max_calls=3, max_calls_per_problem=9)
    original = (d / "telegraph.jsonl").read_bytes()
    done = run_child("migrate", d, step)
    assert done.returncode == KILLED, (step, done.stderr)
    old_answer, sent_old, new_answer, sent_new, _ = check_directory_after_a_crash(d, tmp_path, step)
    if (d / "telegraph.jsonl").is_file() and (d / "telegraph.jsonl.archived").exists():
        with pytest.raises(mig.MigrationRefused):
            mig.migrate(d)
        return
    if step == "after:6c":
        with pytest.raises(mig.AlreadyMigrated):
            mig.migrate(d)
    else:
        assert mig.migrate(d)["phase"] == "complete"
    records = ledger_records(d)
    assert tg.accounting(records)["calls"] == 2 + sent_old + sent_new
    assert (d / "telegraph.jsonl.archived").exists() and original.startswith(b"") and (d / "telegraph.jsonl.archived").read_bytes()[:len(original)] == original


def test_i10_after_a_crash_before_the_archive_a_late_record_of_an_old_writer_is_imported_by_the_resume(tmp_path):
    d = legacy_directory(tmp_path, asks=2, max_calls=5, max_calls_per_problem=9)
    with pytest.raises(Crash):
        mig.migrate(d, hook=crash_at("after:3"))                                # imported, the old ledger is still live
    answer, sent = old_writer(d, tmp_path, max_calls=5)                         # an old writer that was not stopped appends to it
    assert sent == 1
    got = mig.migrate(d)
    assert got["phase"] == "complete" and len(ledger_records(d)) == len(legacy_records_of((d / "telegraph.jsonl.archived").read_bytes()))
    assert tg.accounting(tg._read_ledger_strict(d))["calls"] == 3


def test_i10_a_ledger_whose_imported_part_was_edited_is_refused_not_guessed(tmp_path):
    d = legacy_directory(tmp_path, asks=2)
    with pytest.raises(Crash):
        mig.migrate(d, hook=crash_at("after:3"))
    text = (d / "telegraph.jsonl").read_text().replace('"tokens_in": 120', '"tokens_in": 999', 1)
    (d / "telegraph.jsonl").write_text(text)
    with pytest.raises(mig.MigrationRefused, match="changed since it was imported"):
        mig.migrate(d)
    assert (d / "telegraph.jsonl").is_file()                                    # nothing was archived on a guess


def test_i10_an_archive_that_already_exists_is_never_overwritten_by_the_rename(tmp_path):
    d = legacy_directory(tmp_path, asks=2)
    with pytest.raises(Crash):
        mig.migrate(d, hook=crash_at("after:3"))
    shutil.copy(d / "telegraph.jsonl", d / "telegraph.jsonl.archived")          # an archive is already there
    before = (d / "telegraph.jsonl.archived").read_bytes()
    with pytest.raises(mig.MigrationRefused, match="old writer got through"):
        mig.migrate(d)
    assert (d / "telegraph.jsonl.archived").read_bytes() == before and (d / "telegraph.jsonl").is_file()


def test_i10_a_ledger_that_changes_between_the_import_and_the_archive_is_not_archived(tmp_path):
    d = legacy_directory(tmp_path, asks=2)

    def hook(step):
        if step == "after:3":                                                   # a writer that ignored the old lock appends now
            with open(d / "telegraph.jsonl", "a") as f:
                f.write(json.dumps({"call_id": "sneaked", "outcome": "ok", "tokens_in": 1, "tokens_out": 1}) + "\n")
    with pytest.raises(mig.MigrationRefused, match="changed after it was imported"):
        mig.migrate(d, hook=hook)
    assert (d / "telegraph.jsonl").is_file() and not (d / "telegraph.jsonl.archived").exists()
    assert mig.migrate(d)["phase"] == "complete"                                # a rerun imports the late record and finishes
    assert "sneaked" in "\n".join(ledger_lines(d))


def test_i10_a_resume_after_the_archive_when_a_regular_file_is_back_refuses_and_leaves_both_files(tmp_path):
    d = legacy_directory(tmp_path, asks=2)
    with pytest.raises(Crash):
        mig.migrate(d, hook=crash_at("after:4"))
    (d / "telegraph.jsonl").write_text('{"call_id": "late", "outcome": "ok"}\n')
    with pytest.raises(mig.MigrationRefused, match="old writer got through"):
        mig.migrate(d)
    assert (d / "telegraph.jsonl").is_file() and (d / "telegraph.jsonl.archived").is_file()
    answer, sent = new_writer(d, tmp_path)
    assert not answer.ok and sent == 0                                          # new code refuses the directory in this state


def test_i10_the_old_lock_is_held_for_the_whole_command(tmp_path):
    import fcntl
    d = legacy_directory(tmp_path)
    seen = []

    def hook(step):
        fd = os.open(d / "telegraph.lock", os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen.append((step, "free"))
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BlockingIOError:
                seen.append((step, "held"))
        finally:
            os.close(fd)
    mig.migrate(d, hook=hook)
    assert seen and all(state == "held" for _step, state in seen), seen
    fd = os.open(d / "telegraph.lock", os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)                              # released at step 7
    os.close(fd)


def test_the_markers_are_written_and_synced_before_the_ledger_is_touched(tmp_path, monkeypatch):
    d = legacy_directory(tmp_path)
    seen = {}
    real_rename = os.rename

    def spy(src, dst, *a, **k):
        if str(src) == "telegraph.jsonl":
            seen["marker_before_archive"] = (d / "MIGRATING-TO-SQLITE").is_file()
            body = json.loads((d / "MIGRATING-TO-SQLITE").read_text())
            seen["sha_recorded"] = body["ledger_sha256"] == hashlib.sha256((d / "telegraph.jsonl").read_bytes()).hexdigest()
        return real_rename(src, dst, *a, **k)
    monkeypatch.setattr(mig.os, "rename", spy)
    mig.migrate(d)
    assert seen == {"marker_before_archive": True, "sha_recorded": True}


# =============================================================================================== hostile cache entries and sidecars

def test_hostile_files_in_the_cache_directory_are_skipped_and_never_hang_the_import(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    cache = d / "cache"
    os.mkfifo(cache / ("a" * 64 + ".json"))
    (cache / ("b" * 64 + ".json")).write_text("[1, 2]")
    (cache / ("c" * 64 + ".json")).write_text('{"text": 5, "reply_file": "x"}')
    (cache / ("d" * 64 + ".json")).symlink_to("/dev/zero")
    (cache / ("e" * 64 + ".json")).mkdir()
    (cache / "not-a-hash.json").write_text("{}")
    (cache / ("f" * 64 + ".json")).write_bytes(b"x" * (mig.MAX_CACHE_BYTES + 10))
    got = mig.migrate(d)
    assert got["cache_imported"] == 1 and got["cache_skipped"] == 7 and len(cache_rows(d)) == 1


def test_a_cache_directory_that_is_a_link_is_not_followed(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    shutil.move(str(d / "cache"), str(tmp_path / "elsewhere"))
    (d / "cache").symlink_to(tmp_path / "elsewhere")
    got = mig.migrate(d)
    assert got["cache_imported"] == 0 and cache_rows(d) == []


def test_a_ledger_that_is_a_link_or_a_fifo_is_refused_before_anything_is_written(tmp_path):
    for kind in ("link", "fifo"):
        d = tmp_path / kind
        d.mkdir()
        if kind == "link":
            (tmp_path / "other.jsonl").write_text("")
            (d / "telegraph.jsonl").symlink_to(tmp_path / "other.jsonl")
        else:
            os.mkfifo(d / "telegraph.jsonl")
        with pytest.raises(mig.MigrationRefused):
            mig.migrate(d)
        assert not (d / "MIGRATING-TO-SQLITE").exists()


def test_a_ledger_larger_than_the_cap_is_refused_and_not_read(tmp_path):
    d = tmp_path / "tg"
    d.mkdir()
    with open(d / "telegraph.jsonl", "wb") as f:
        f.truncate(mig.MAX_LEDGER_BYTES + 1)
    with pytest.raises(mig.MigrationRefused):
        mig.migrate(d)


def test_a_directory_that_does_not_exist_is_refused_and_not_created(tmp_path):
    with pytest.raises(mig.MigrationRefused):
        mig.migrate(tmp_path / "nowhere")
    assert not (tmp_path / "nowhere").exists()


# =============================================================================================== the command line

def run_cli(argv, tmp_path):
    out = io.StringIO()
    code = cli.main(argv, env={"HOME": str(tmp_path / "home")}, stdout=out)
    return code, out.getvalue()


def test_the_commands_migrate_export_and_status_work_end_to_end(tmp_path):
    d = legacy_directory(tmp_path, asks=2)
    original = (d / "telegraph.jsonl").read_text().splitlines()
    code, text = run_cli(["db", "migrate", "telegraph", "--dir", str(d), "--dry-run"], tmp_path)
    assert code == 0 and "dry run: nothing was changed" in text and "totals equal" in text
    code, text = run_cli(["db", "migrate", "telegraph", "--dir", str(d)], tmp_path)
    assert code == 0 and "the export equals the archived ledger line by line: yes" in text
    code, text = run_cli(["db", "export", "telegraph", "--dir", str(d)], tmp_path)
    assert code == 0 and text.splitlines() == original
    code, text = run_cli(["db", "export", "telegraph", "--dir", str(d), "--from-archive"], tmp_path)
    assert code == 0 and text.splitlines() == original
    code, text = run_cli(["db", "status", "--dir", str(d)], tmp_path)
    assert code == 0 and "migration: complete" in text and "telegraph records: 4" in text and "MIGRATED-TO-SQLITE" in text
    code, text = run_cli(["db", "migrate", "telegraph", "--dir", str(d)], tmp_path)
    assert code == 2 and "migrated already" in text


def test_the_command_says_what_to_do_about_a_malformed_line(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    with open(d / "telegraph.jsonl", "a") as f:
        f.write("not json\n")
    code, text = run_cli(["db", "migrate", "telegraph", "--dir", str(d)], tmp_path)
    assert code == 2 and "--accept-malformed 1" in text
    code, text = run_cli(["db", "migrate", "telegraph", "--dir", str(d), "--accept-malformed", "1"], tmp_path)
    assert code == 0


def test_the_command_resolves_the_directory_from_the_config_then_the_environment(tmp_path):
    d = legacy_directory(tmp_path, asks=1)
    cfg = tmp_path / "t.toml"
    cfg.write_text(f'[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["a"]\ndir = "{d}"\n')
    out = io.StringIO()
    code = cli.main(["db", "status", "--telegraph-config", str(cfg)], env={"HOME": str(tmp_path / "nohome")}, stdout=out)
    assert code == 0 and str(d) in out.getvalue()
    out = io.StringIO()
    code = cli.main(["db", "status", "--telegraph-config", str(tmp_path / "missing.toml")],
                    env={"HOME": str(tmp_path / "nohome"), "ORCHARD_TELEGRAPH_DIR": str(d)}, stdout=out)
    assert code == 0 and str(d) in out.getvalue()
