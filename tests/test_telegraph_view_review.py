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
