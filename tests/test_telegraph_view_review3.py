# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixes for the third Copilot review of PR 111 (API side). Each test was written, and seen to fail, before its fix."""
import json
import math

import pytest

from ledger_io import seed_lines
from orchard import telegraph_view as view
from test_telegraph_view import app_for, setup
from test_telegraph_view_review import ok_record, reserve_record, write_records
from test_webui import Served


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    return r


# ---- 1. a number that is not finite is a damaged ledger -------------------------------------------------------------------

def write_raw(d, lines):
    seed_lines(d, [l for l in lines if l.strip()])


@pytest.mark.parametrize("field", ["cost_usd", "tokens_in", "tokens_out", "latency_s", "ts"])
@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_number_makes_the_ledger_a_problem_with_unknown_spend(root, tmp_path, field, literal):
    cfg, d, _ = setup(tmp_path)
    good = json.dumps(ok_record("a1", "p", 10, 5))
    bad = json.dumps(ok_record("a2", "p", 10, 5)).replace(f'"{field}": ' + json.dumps(ok_record("a2", "p", 10, 5)[field]),
                                                          f'"{field}": {literal}')
    assert literal in bad
    write_raw(d, [good + "\n", bad + "\n"])
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "problem" and out["meter"]["text"] == "Helper: ledger unreadable · spend unknown"
    json.dumps(out, allow_nan=False)                          # the answer itself holds no NaN or Infinity


@pytest.mark.parametrize("record", [
    {"reserve": "r1", "problem": "p", "model": "small", "tokens": float("inf"), "cost_usd": 0.01},
    {"reserve": "r1", "problem": "p", "model": "small", "tokens": 100, "cost_usd": float("nan")},
    {"torn": "x", "tokens": float("inf")},
])
def test_non_finite_numbers_in_reservations_and_torn_notes_are_refused_too(root, tmp_path, record):
    cfg, d, _ = setup(tmp_path)
    write_raw(d, [json.dumps(ok_record("a1", "p", 10, 5)) + "\n", json.dumps(record) + "\n"])
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "problem"
    json.dumps(out, allow_nan=False)


def test_a_non_finite_number_nested_in_a_record_is_refused(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    write_raw(d, ['{"call_id": "a1", "outcome": "ok", "extra": {"deep": [1, NaN]}}\n'])
    assert app_for(root, cfg).telegraph()["state"] == "problem"


def test_finite_numbers_still_work(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [ok_record("a1", "p", 10, 5, cost=0.5)])
    out = app_for(root, cfg).telegraph()
    assert out["state"] != "problem" and math.isfinite(out["spend"]["cost_usd"])


def test_the_route_never_sends_nan_even_for_a_damaged_ledger(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    write_raw(d, ['{"call_id": "a1", "outcome": "ok", "cost_usd": NaN, "tokens_in": 1, "tokens_out": 1, "problem": "p"}\n'])
    with Served(app_for(root, cfg)) as s:
        code, _, body = s.request("GET", "/api/telegraph")
    assert code == 200
    json.loads(body, parse_constant=lambda c: (_ for _ in ()).throw(AssertionError(f"{c} in the answer")))


# ---- 2. only the newest rows are built ---------------------------------------------------------------------------------------

def test_only_the_newest_fifty_calls_are_turned_into_rows(root, tmp_path, monkeypatch):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [ok_record(f"c{i:05d}", f"p{i % 7}", 1, 1) for i in range(3000)])
    built = []
    real = view._call_row
    monkeypatch.setattr(view, "_call_row", lambda *a, **k: (built.append(1), real(*a, **k))[1])
    out = app_for(root, cfg).telegraph()
    assert len(out["recent"]) == 50 and out["recent"][0]["call_id"] == "c02999" and out["recent"][-1]["call_id"] == "c02950"
    assert len(built) <= 100                                    # 50 for the dry run at read time and 50 for the answer


def test_unsettled_reservations_older_than_the_window_do_not_hide_newer_calls(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [reserve_record("old", "p", "small", 10, 0.01)] + [ok_record(f"c{i:03d}", "p", 1, 1) for i in range(60)])
    rows = app_for(root, cfg).telegraph()["recent"]
    assert len(rows) == 50 and rows[0]["call_id"] == "c059" and all(r["outcome"] == "ok" for r in rows)


def test_a_settled_reply_with_no_usage_still_finds_its_reservation_when_the_reservation_is_old(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    settled = {**ok_record("u1", "p", 10, 5, cost=0.0001), "usage_known": False}
    write_records(d, [reserve_record("u1", "p", "small", 2000, 0.02)] + [ok_record(f"c{i:03d}", "p", 1, 1) for i in range(5)] + [settled])
    rows = app_for(root, cfg).telegraph()["recent"]
    assert rows[0]["call_id"] == "u1" and rows[0]["cost_basis"] == "worst case" and rows[0]["cost_usd"] == pytest.approx(0.02)
