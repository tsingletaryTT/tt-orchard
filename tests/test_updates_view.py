# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The updates panel of the web page (issue 231): GET /api/updates and orchard/updates_view.py. Read only, from the cache file."""
import json
import subprocess
import time
import urllib.request

from orchard import updates, updates_view
from test_webui import Served, make_app


def write_cache(path, rows, checked_at):
    path.write_text(json.dumps({"report": {"rows": rows, "checked_at": checked_at}}))


ROW = {"key": "tt-metal", "item": "tt-metal", "installed": "v0.60", "installed_date": "2026-10-01", "upstream": "v0.61",
       "upstream_date": "2026-10-20", "status": "behind", "why": "", "proposal": ["pull it"],
       "moves_with": ["tt-lang (rule R1)"], "ready": "no"}


def test_rows_come_from_the_cache(tmp_path):
    p = tmp_path / "u.json"
    write_cache(p, [ROW], time.time() - 60)
    d = updates_view.build(p)
    assert d["read_only"] is True and d["state"] == "ok"
    r = d["rows"][0]
    assert (r["item"], r["status"], r["tone"], r["moves_with"], r["ready"]) == ("tt-metal", "behind", "warn", ["tt-lang (rule R1)"], "no")
    assert d["run_command"] == "tt-orchard updates --refresh"


def test_a_missing_cache_is_a_state_not_an_error(tmp_path):
    d = updates_view.build(tmp_path / "none.json")
    assert d["state"] == "none" and d["rows"] == []


def test_an_old_check_is_marked_stale_and_still_shown(tmp_path):
    p = tmp_path / "u.json"
    write_cache(p, [ROW], time.time() - 7 * 3600)
    d = updates_view.build(p)
    assert d["state"] == "stale" and len(d["rows"]) == 1 and "older" in d["note"]


def test_hostile_values_are_cut_cleaned_and_redacted(tmp_path):
    p = tmp_path / "u.json"
    bad = dict(ROW, item="x" * 5000, why="boom\x1b[31m", status="bogus", proposal="not a list", moves_with=[1, "ok\x00"], ready=7)
    write_cache(p, [bad, "junk", 5] + [ROW] * 100, time.time())
    d = updates_view.build(p)
    assert len(d["rows"]) <= updates_view.MAX_ROWS
    r = d["rows"][0]
    assert len(r["item"]) <= updates.TEXT_MAX and r["status"] == "unknown" and "\x1b" not in r["why"]
    assert r["proposal"] == [] and r["moves_with"] == ["ok?"] and r["ready"] == "-"


def test_redaction_is_applied_to_every_text_field(tmp_path):
    p = tmp_path / "u.json"
    red = lambda t: t.replace("SECRET", "<token>")
    write_cache(p, [dict(ROW, item="has SECRET", why="why SECRET", status="unknown", proposal=["p SECRET"], moves_with=["m SECRET"],
                         ready="r SECRET", installed="i SECRET", upstream="u SECRET")], time.time())
    assert "SECRET" not in json.dumps(updates_view.build(p, redact=red))


def test_the_build_makes_no_call_and_no_write(tmp_path, monkeypatch):
    p = tmp_path / "u.json"
    write_cache(p, [ROW], time.time())
    before = p.read_bytes()
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("process")))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    updates_view.build(p)
    assert p.read_bytes() == before and list(tmp_path.iterdir()) == [p]


def test_the_route_serves_the_view_and_returns_no_path(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "tt-orchard").mkdir()
    write_cache(tmp_path / "tt-orchard" / "updates.json", [ROW], time.time())
    root = tmp_path / "runs"
    root.mkdir()
    with Served(make_app(root)) as s:
        code, _, body = s.request("GET", "/api/updates")
        code2, _, _ = s.request("POST", "/api/updates")
    text = body.decode() if isinstance(body, bytes) else body
    assert code == 200 and json.loads(text)["rows"][0]["item"] == "tt-metal"
    assert str(tmp_path) not in text
    assert code2 >= 400


def test_a_status_that_is_not_a_string_is_an_unknown_row_not_a_500(tmp_path):
    p = tmp_path / "u.json"
    write_cache(p, [dict(ROW, status=[]), dict(ROW, status={}), dict(ROW, status=None), dict(ROW, status=3)], time.time())
    d = updates_view.build(p)
    assert [r["status"] for r in d["rows"]] == ["unknown"] * 4
