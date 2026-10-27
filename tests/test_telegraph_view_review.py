# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixes for the first Copilot review of PR 111 (API side). Each test was written, and seen to fail, before its fix."""
import json
import os

import pytest

from orchard import telegraph as tg
from orchard import telegraph_view as view
from test_telegraph_view import KEY, app_for, fill, setup
from test_telegraph_view_ledger import crash_after_reserve, no_usage


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    return r


def ok_record(call_id, problem, tin, tout, model="small", cost=0.001):
    return {"call_id": call_id, "problem": problem, "model": model, "ts": 1.0, "outcome": "ok", "cached": False,
            "tokens_in": tin, "tokens_out": tout, "cost_usd": cost, "latency_s": 0.1, "usage_known": True}


def write_records(d, records):
    d.mkdir(parents=True, exist_ok=True)
    (d / "telegraph.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


# ---- 3. problems beyond the table's limit still count for the state ----------------------------------------------------

def sixty_problems(tmp_path, hot_tokens=850):
    cfg, d, _ = setup(tmp_path, max_calls=1000, max_total_tokens=10 ** 9, max_calls_per_problem=9, max_tokens_per_problem=1000)
    records = [ok_record(f"c{i:02d}{j}", f"p{i:02d}", 5, 5) for i in range(59) for j in range(2)]
    records.append(ok_record("hot1", "zzz", hot_tokens - 50, 50))
    write_records(d, records)
    return cfg


def test_a_problem_at_eighty_percent_of_its_cap_turns_the_meter_amber_among_sixty_problems(root, tmp_path):
    out = app_for(root, sixty_problems(tmp_path)).telegraph()
    assert out["state"] == "near" and out["tone"] == "warn"
    assert "problem zzz at 85% of its cap" in out["meter"]["text"]


def test_the_table_shows_the_flagged_problems_first_and_says_how_many_are_not_shown(root, tmp_path):
    out = app_for(root, sixty_problems(tmp_path)).telegraph()
    assert len(out["problems"]) == view.PROBLEMS_MAX == 50
    assert out["problems"][0]["name"] == "zzz" and out["problems"][0]["near"] is True
    assert out["problems_hidden"] == 10
    small = app_for(root, sixty_problems(tmp_path / "again")).telegraph()
    assert small["problems_hidden"] == 10


def test_a_problem_at_its_cap_among_sixty_makes_no_difference_to_a_green_ledger_below_eighty(root, tmp_path):
    out = app_for(root, sixty_problems(tmp_path, hot_tokens=700)).telegraph()
    assert out["state"] == "ok" and out["problems_hidden"] == 10


def test_few_problems_hide_none(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    fill(tel, 2)
    assert app_for(root, cfg).telegraph()["problems_hidden"] == 0


# ---- 1. a small positive cost is never shown as zero -------------------------------------------------------------------

def test_a_positive_cost_below_a_hundredth_of_a_cent_is_never_shown_as_zero(root, tmp_path):
    cfg, d, tel = setup(tmp_path, prices={"small": [0.01, 0.01], "big": [0.01, 0.01]})
    tel.ask("p", "task " * 80, "failed")                         # about $0.000015
    out = app_for(root, cfg).telegraph()
    assert 0 < out["spend"]["cost_usd"] < 0.0001
    assert out["meter"]["text"].startswith("Helper: <$0.0001 · 1 call")
    assert "$0.0000" not in json.dumps(out)


def test_a_small_positive_cost_cap_is_not_shown_as_a_zero_cap(root, tmp_path):
    cfg, d, tel = setup(tmp_path, prices={"small": [0.01, 0.01], "big": [0.01, 0.01]})
    import dataclasses
    tel.ask("p", "task " * 80, "failed")
    capped = dataclasses.replace(cfg, max_cost_usd=0.00003)                  # the call cost half of this cap
    out = app_for(root, capped).telegraph()
    assert "cost <$0.0001 of <$0.0001" in out["meter"]["text"] and "$0.0000" not in json.dumps(out)


# ---- 2. an unreadable ledger is not an empty one ---------------------------------------------------------------------

def test_a_ledger_that_is_a_directory_is_a_problem_and_spend_is_unknown(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    d.mkdir()
    (d / "telegraph.jsonl").mkdir()
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "problem" and out["tone"] == "warn" and out["error"]
    assert out["meter"]["text"] == "Helper: ledger unreadable · spend unknown"
    assert "$0.00" not in out["meter"]["text"] and "0 calls" not in out["meter"]["text"]
    assert out["caps"] == []                                    # no bars at 0 percent for a ledger that was not read


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a file with no permissions")
def test_a_ledger_with_no_read_permission_is_a_problem_not_a_green_zero(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    fill(tel, 1)
    os.chmod(d / "telegraph.jsonl", 0)
    try:
        out = app_for(root, cfg).telegraph()
    finally:
        os.chmod(d / "telegraph.jsonl", 0o600)
    assert out["state"] == "problem" and "spend unknown" in out["meter"]["text"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can enter a directory with no permissions")
def test_a_ledger_folder_that_cannot_be_entered_is_a_problem(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    fill(tel, 1)
    os.chmod(d, 0)
    try:
        out = app_for(root, cfg).telegraph()
    finally:
        os.chmod(d, 0o700)
    assert out["state"] == "problem" and "spend unknown" in out["meter"]["text"]


def test_a_missing_ledger_is_still_an_empty_history(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "ok" and out["error"] is None and out["meter"]["text"].startswith("Helper: $0.00 · 0 calls")


def test_every_error_state_says_spend_unknown_and_recovers_when_reading_works(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    d.mkdir()
    (d / "telegraph.jsonl").write_text(json.dumps({"call_id": "x", "outcome": "ok", "tokens_in": "lots", "model": "m"}) + "\n")
    app = app_for(root, cfg)
    out = app.telegraph()
    assert out["state"] == "problem" and out["meter"]["text"].endswith("spend unknown")
    (d / "telegraph.jsonl").write_text("")
    again = app.telegraph()
    assert again["state"] == "ok" and again["error"] is None


def test_an_off_telegraph_with_an_unreadable_ledger_stays_off(root, tmp_path):
    d = tmp_path / "ledger"
    d.mkdir()
    (d / "telegraph.jsonl").mkdir()
    cfg = tg.load({"enabled": False, "dir": str(d)}, base=tmp_path)
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "off" and out["meter"]["text"] == "Helper: off"
