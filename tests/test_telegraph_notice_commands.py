"""`telegraph ask | report | usage` on a directory that still holds an old ledger: each refuses before it opens anything, says what to
do on its first line, sends nothing and spends nothing. Temporary directories only; the transport is a fake that records calls."""
import json
import os
import stat

import pytest

from ledger_io import ledger_records
from orchard import telegraph as tg
from orchard import telegraph_migrate as mig
from test_telegraph_cmd import files, ok_transport, run as run_tg, write

RECORD = {"call_id": "a1", "problem": "p", "model": "small", "outcome": "ok", "tokens_in": 10, "tokens_out": 5,
          "cost_usd": 0.5, "usage_known": True, "ts": 1.0}


def snapshot(directory):
    """Every name under a directory with its mode, size, content and modification time: a change of any kind shows."""
    out = {}
    for p in sorted(directory.rglob("*")):
        st = os.lstat(p)
        out[str(p.relative_to(directory))] = (stat.S_IFMT(st.st_mode), st.st_size, st.st_mtime_ns,
                                              p.read_bytes() if p.is_file() else None)
    return out


def unready(tmp_path, kind, lines=1):
    d = tmp_path / "state"
    d.mkdir(parents=True, exist_ok=True)
    (d / "telegraph.jsonl").write_text("".join(json.dumps({**RECORD, "call_id": f"c{i}"}) + "\n" for i in range(lines)))
    if kind == "half-finished":
        (d / "MIGRATING-TO-SQLITE").write_text("{}")
    return d


@pytest.mark.parametrize("kind", ["unmigrated", "half-finished"])
def test_the_telegraph_object_refuses_before_it_opens_anything_and_sends_nothing(tmp_path, kind):
    """The check is at the layer that fails: `Telegraph.ask` is what the supervisor and the agent shells call."""
    d = unready(tmp_path, kind)
    before, t = snapshot(d), ok_transport()
    cfg = tg.load({"enabled": True, "endpoint": "https://llm.example.org/v1", "models": ["small"], "key_env": "TG_KEY"}, base=tmp_path)
    got = tg.Telegraph(cfg, d, transport=t, environ={"TG_KEY": "k3y-VALUE-0123456789"}, home=str(tmp_path)).ask("p", "task", "failure")
    assert not got.ok and t.seen == []
    assert got.reason.startswith("Run `tt-orchard db migrate telegraph")
    assert "No outside call was made" in got.reason and str(d) not in got.reason
    assert snapshot(d) == before                       # no project.db, no INITIALIZED, no lock file: nothing was created


def test_the_command_ask_prints_what_to_do_on_the_first_line_and_exits_3(tmp_path):
    d = unready(tmp_path, "unmigrated")
    before, t = snapshot(d), ok_transport()
    code, text = run_tg(["ask", "--problem", "p", *files(tmp_path), "--dir", str(d), "--telegraph-config", str(write(tmp_path))],
                        tmp_path, transport=t)
    assert code == 3 and text.splitlines()[0].startswith("telegraph: Run `tt-orchard db migrate telegraph --dry-run --dir ")
    assert "no outside call was made" in text.lower() and "Nothing is deleted" in text
    assert t.seen == [] and snapshot(d) == before


def test_ask_with_telegraph_off_still_says_it_is_off_and_touches_nothing(tmp_path):
    d = unready(tmp_path, "unmigrated")
    before = snapshot(d)
    code, text = run_tg(["ask", "--problem", "p", *files(tmp_path), "--dir", str(d), "--telegraph-config", str(write(tmp_path, "x = 1\n"))],
                        tmp_path, transport=ok_transport())
    assert code == 3 and "not enabled" in text and snapshot(d) == before


def test_report_on_an_unmigrated_directory_exits_2_with_the_command_and_creates_nothing(tmp_path):
    d = unready(tmp_path, "unmigrated")
    before = snapshot(d)
    code, text = run_tg(["report", "abc", "--helped", "--dir", str(d), "--telegraph-config", str(write(tmp_path))], tmp_path)
    assert code == 2 and snapshot(d) == before
    assert text.splitlines()[0].startswith("telegraph: Run `tt-orchard db migrate telegraph --dry-run --dir ")


@pytest.mark.parametrize("kind", ["unmigrated", "half-finished"])
def test_usage_puts_the_command_on_the_first_line_and_still_says_the_spend_is_unknown(tmp_path, kind):
    d = unready(tmp_path, kind)
    before = snapshot(d)
    code, text = run_tg(["usage", "--dir", str(d), "--telegraph-config", str(write(tmp_path))], tmp_path)
    lines = text.splitlines()
    assert code == 2 and lines[0].startswith("telegraph: Run `tt-orchard db migrate telegraph")
    assert "spend is unknown" in text and "cannot be read" not in lines[0]
    assert snapshot(d) == before


def test_usage_of_a_migrated_directory_is_unchanged(tmp_path):
    d = unready(tmp_path, "unmigrated", lines=2)
    mig.migrate(d)
    code, text = run_tg(["usage", "--dir", str(d), "--telegraph-config", str(write(tmp_path))], tmp_path)
    assert code == 0 and "calls: 2" in text and "Run `" not in text


def test_ask_after_the_migration_goes_out_and_is_recorded(tmp_path):
    d = unready(tmp_path, "unmigrated")
    mig.migrate(d)
    t = ok_transport()
    code, text = run_tg(["ask", "--problem", "p", *files(tmp_path), "--dir", str(d), "--telegraph-config", str(write(tmp_path))],
                        tmp_path, transport=t)
    assert code == 0 and len(t.seen) == 1 and any(r.get("outcome") == "ok" and r.get("call_id") != "c0" for r in ledger_records(d))
