"""`tt-orchard db migrate telegraph --dry-run-first`: the dry run, then one confirmation, then the real migration.

It never migrates without a yes, never when a check is different, never while another process holds `telegraph.lock`, and never in a
directory that is not an unmigrated one. Every test uses a temporary directory; no test reads or writes a real state directory."""
import fcntl
import io
import json
import os
import stat

import pytest

from ledger_io import ledger_records
from orchard import cli, db_cmd
from orchard import telegraph_migrate as mig
from orchard import telegraph_notice as tn
from orchard import telegraph as tg

RECORD = {"call_id": "a1", "problem": "p", "model": "small", "outcome": "ok", "tokens_in": 10, "tokens_out": 5,
          "cost_usd": 0.5, "usage_known": True, "ts": 1.0}


def make(tmp_path, lines=3, extra=""):
    d = tmp_path / "state"
    d.mkdir(exist_ok=True)
    (d / "telegraph.jsonl").write_text("".join(json.dumps({**RECORD, "call_id": f"c{i}"}) + "\n" for i in range(lines)) + extra)
    return d


def snapshot(directory):
    out = {}
    for p in sorted(directory.rglob("*")):
        st = os.lstat(p)
        out[str(p.relative_to(directory))] = (stat.S_IFMT(st.st_mode), st.st_size, st.st_mtime_ns,
                                              p.read_bytes() if p.is_file() else None)
    return out


class Asker:
    """A confirmation prompt that records the question and answers as told. `during` runs inside the prompt (a race)."""

    def __init__(self, answer=True, during=None):
        self.answer, self.during, self.questions = answer, during, []

    def __call__(self, question):
        self.questions.append(question)
        if self.during:
            self.during()
        return self.answer


def guided(d, tmp_path, *extra, ask="none"):
    out = io.StringIO()
    args = cli._parser().parse_args(["db", "migrate", "telegraph", "--dir", str(d), "--dry-run-first", *extra])
    kw = {} if ask == "none" else {"ask": ask}
    code = db_cmd.main(args, out=out, environ={}, home=tmp_path, **kw)
    return code, out.getvalue()


@pytest.fixture
def spy(monkeypatch):
    """Which calls reached the migration: (dry_run flag) for each."""
    calls = []
    real = mig.migrate

    def wrapper(directory, **kw):
        calls.append(bool(kw.get("dry_run")))
        return real(directory, **kw)
    monkeypatch.setattr(mig, "migrate", wrapper)
    return calls


def test_a_yes_after_a_passing_dry_run_migrates_and_the_ledger_is_the_same(tmp_path, spy):
    d = make(tmp_path)
    ask = Asker(True)
    code, text = guided(d, tmp_path, ask=ask)
    assert code == 0 and spy == [True, False]                     # the dry run first, then the real run
    assert "dry run: nothing was changed" in text and "equal" in text and "DIFFERENT" not in text
    assert "migration migrated" in text
    assert len(ask.questions) == 1 and "Nothing is deleted" in ask.questions[0]
    assert [r["call_id"] for r in ledger_records(d)] == ["c0", "c1", "c2"]
    assert (d / "telegraph.jsonl.archived").is_file() and tn.state_of(d) is None


def test_a_no_changes_nothing(tmp_path, spy):
    d = make(tmp_path)
    before = snapshot(d)
    code, text = guided(d, tmp_path, ask=Asker(False))
    assert code == 2 and spy == [True] and "nothing was changed" in text.lower()
    assert snapshot(d) == before


def test_without_a_terminal_and_without_an_answer_nothing_is_migrated(tmp_path, spy):
    d = make(tmp_path)
    before = snapshot(d)
    code, text = guided(d, tmp_path)                              # no ask given: the default needs a terminal, and pytest has none
    assert code == 2 and False not in spy and "terminal" in text and "db migrate telegraph --dry-run" in text
    assert snapshot(d) == before


def test_a_different_check_stops_before_the_question(tmp_path, spy, monkeypatch):
    d = make(tmp_path)
    real = mig.migrate

    def different(directory, **kw):
        out = real(directory, **kw)
        if kw.get("dry_run"):
            out = {**out, "equal": False, "checks": {**out["checks"], "totals": False}}
        return out
    monkeypatch.setattr(mig, "migrate", different)
    ask, before = Asker(True), snapshot(d)
    code, text = guided(d, tmp_path, ask=ask)
    assert code == 2 and ask.questions == [] and "DIFFERENT" in text and snapshot(d) == before


def test_a_ledger_that_cannot_be_checked_stops_before_the_question(tmp_path, spy):
    d = make(tmp_path, extra="not a record\n" + json.dumps({**RECORD, "call_id": "z"}) + "\n")
    ask, before = Asker(True), snapshot(d)
    code, text = guided(d, tmp_path, ask=ask)
    assert code == 2 and ask.questions == [] and "not records" in text and snapshot(d) == before


def test_accept_malformed_goes_through_both_steps(tmp_path, spy):
    d = make(tmp_path, extra="not a record\n" + json.dumps({**RECORD, "call_id": "z"}) + "\n")
    code, text = guided(d, tmp_path, "--accept-malformed", "1", ask=Asker(True))
    assert code == 0 and spy == [True, False] and tn.state_of(d) is None


def hold_lock(d):
    fd = os.open(d / "telegraph.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def test_a_lock_held_by_another_process_stops_before_the_question(tmp_path, spy):
    d = make(tmp_path)
    fd = hold_lock(d)
    try:
        before, ask = snapshot(d), Asker(True)
        code, text = guided(d, tmp_path, ask=ask)
        assert code == 2 and ask.questions == [] and False not in spy
        assert "telegraph.lock" in text and "another" in text and snapshot(d) == before
    finally:
        os.close(fd)


def test_a_lock_taken_while_the_question_is_open_stops_the_migration(tmp_path, spy):
    d = make(tmp_path)
    held = []
    ask = Asker(True, during=lambda: held.append(hold_lock(d)))
    try:
        code, text = guided(d, tmp_path, ask=ask)
        assert code == 2 and False not in spy and "telegraph.lock" in text
        assert (d / "telegraph.jsonl").is_file() and not (d / "telegraph.jsonl.archived").exists()
    finally:
        for fd in held:
            os.close(fd)


def test_an_absent_lock_file_is_not_created_by_the_check(tmp_path):
    d = make(tmp_path)
    assert mig.lock_is_held(d) is False and not (d / "telegraph.lock").exists()


@pytest.mark.parametrize("state", ["half-finished", "migrated", "empty"])
def test_a_directory_that_is_not_an_unmigrated_one_is_refused_without_a_question(tmp_path, spy, state):
    d = make(tmp_path)
    if state == "half-finished":
        (d / "MIGRATING-TO-SQLITE").write_text("{}")
    elif state == "migrated":
        mig.migrate(d)
        spy.clear()
    else:
        (d / "telegraph.jsonl").unlink()
    ask, before = Asker(True), snapshot(d)
    code, text = guided(d, tmp_path, ask=ask)
    assert code == 2 and ask.questions == [] and spy == [] and snapshot(d) == before
    assert text.strip()


def test_both_dry_run_flags_together_are_refused(tmp_path, spy):
    d = make(tmp_path)
    out = io.StringIO()
    args = cli._parser().parse_args(["db", "migrate", "telegraph", "--dir", str(d), "--dry-run", "--dry-run-first"])
    before = snapshot(d)
    assert db_cmd.main(args, out=out, environ={}, home=tmp_path, ask=Asker(True)) == 2 and spy == [] and snapshot(d) == before


def test_the_existing_commands_behave_as_before(tmp_path):
    d = make(tmp_path)
    out = io.StringIO()
    code = db_cmd.main(cli._parser().parse_args(["db", "migrate", "telegraph", "--dir", str(d), "--dry-run"]), out=out, environ={}, home=tmp_path)
    assert code == 0 and "dry run: nothing was changed" in out.getvalue() and (d / "telegraph.jsonl").is_file()
    out = io.StringIO()
    code = db_cmd.main(cli._parser().parse_args(["db", "migrate", "telegraph", "--dir", str(d)]), out=out, environ={}, home=tmp_path)
    assert code == 0 and "migration migrated" in out.getvalue()
