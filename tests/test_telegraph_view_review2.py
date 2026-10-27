# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixes for the second Copilot review of PR 111 (API side). Each test was written, and seen to fail, before its fix."""
import json

import pytest

from orchard import telegraph as tg
from orchard import telegraph_view as view
from test_telegraph_view import app_for, setup
from test_telegraph_view_review import ok_record, reserve_record, write_records
from test_webui import Served


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    return r


# ---- 1. a damaged record never becomes an HTTP 500 ---------------------------------------------------------------------

BAD = [[], {}, 5, None, "x", [1, 2], {"a": 1}, 1.5, True]


def damaged(key, value, kind):
    if kind == "refusal":
        rec = {"call_id": "r1", "problem": "p", "model": "small", "outcome": "budget", "cached": False, "reason": "cap",
               "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    elif kind == "sent":
        rec = ok_record("s1", "p", 10, 5)
    elif kind == "cached":
        rec = {**ok_record("k1", "p", 0, 0, cost=0.0), "cached": True, "origin": "s1"}
    else:
        rec = reserve_record("v1", "p", "small", 100, 0.01)
    rec[{"call_id": "reserve"}.get(key, key) if kind == "reserve" and key == "call_id" else key] = value
    return rec


@pytest.mark.parametrize("kind", ["refusal", "sent", "cached", "reserve"])
@pytest.mark.parametrize("key", ["call_id", "origin", "problem", "model"])
@pytest.mark.parametrize("value", BAD, ids=lambda v: repr(v))
def test_a_damaged_record_gives_an_answer_and_never_an_error(root, tmp_path, kind, key, value):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [ok_record("good", "p", 10, 5), damaged(key, value, kind)])
    app = app_for(root, cfg)
    out = app.telegraph()                                      # must not raise
    assert isinstance(out["meter"]["text"], str)
    json.dumps(out)
    if isinstance(value, (list, dict)):                        # a value that cannot be a key or a name is a damaged ledger
        assert out["state"] == "problem" and out["meter"]["text"] == "Helper: ledger unreadable · spend unknown"


def test_a_list_for_call_id_in_a_refusal_is_http_200_with_a_problem_state(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [{"call_id": [], "outcome": "budget"}])
    with Served(app_for(root, cfg)) as s:
        code, _, body = s.request("GET", "/api/telegraph")
    assert code == 200 and json.loads(body)["state"] == "problem"


# ---- 2. one read, one snapshot -------------------------------------------------------------------------------------------

def test_the_summary_comes_from_the_same_records_as_the_accounting_and_the_rows(root, tmp_path, monkeypatch):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [reserve_record("c1", "p", "small", 500, 0.01)])
    unsettled = [reserve_record("c1", "p", "small", 500, 0.01)]
    settled = unsettled + [ok_record("c1", "p", 100, 50)]
    reads = []

    def read(directory):
        reads.append(1)
        return unsettled if len(reads) == 1 else settled          # the call settles after the first read
    monkeypatch.setattr(tg, "_read_ledger_strict", read)

    def forbidden(directory):
        raise AssertionError("the summary must not read the ledger again")
    monkeypatch.setattr(tg, "summarise", forbidden)
    out = app_for(root, cfg).telegraph()
    assert len(reads) == 1
    assert out["summary"]["unsettled"] == 1 and out["summary"]["calls"] == 0
    assert out["spend"]["unsettled"] == 1
    assert [r["outcome"] for r in out["recent"]] == ["unsettled"]


def test_the_snapshot_is_cached_until_the_ledger_changes(root, tmp_path, monkeypatch):
    cfg, d, tel = setup(tmp_path)
    write_records(d, [ok_record("a1", "p", 10, 5)])
    real, reads = tg._read_ledger_strict, []
    monkeypatch.setattr(tg, "_read_ledger_strict", lambda directory: (reads.append(1), real(directory))[1])
    app = app_for(root, cfg)
    app.telegraph()
    app.telegraph()
    assert len(reads) == 1
    write_records(d, [ok_record("a1", "p", 10, 5), ok_record("a2", "p", 10, 5)])
    assert app.telegraph()["summary"]["calls"] == 2 and len(reads) == 2


def test_summarise_records_gives_what_summarise_gives(tmp_path):
    write_records(tmp_path, [ok_record("a1", "p", 10, 5), reserve_record("b2", "p", "small", 9, 0.1)])
    assert tg.summarise_records(tg.read_ledger(tmp_path)) == tg.summarise(tmp_path)


# ---- 3 and 4. the meter names the cap that is closest ----------------------------------------------------------------------

def test_a_problem_blocked_by_an_unknown_cost_counts_as_at_its_cap_for_the_meter(root, tmp_path):
    cfg, d, _ = setup(tmp_path, max_calls=1000, max_total_tokens=10 ** 9, max_calls_per_problem=9, max_tokens_per_problem=1000,
                      max_cost_per_problem_usd=100.0, prices={"small": [1.0, 1.0]})
    write_records(d, [ok_record("a1", "aaa", 10, 5, cost=10.0), {**ok_record("a2", "aaa", 10, 5), "cost_usd": None},
                      ok_record("b1", "bbb", 500, 0, cost=1.0)])
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "near"
    assert "problem aaa at 100% of its cap" in out["meter"]["text"]
    blocked = next(p for p in out["problems"] if p["name"] == "aaa")
    assert blocked["percent"] == 100 and blocked["at_cap"] is True


def test_a_global_cap_at_eighty_does_not_hide_a_problem_at_ninety(root, tmp_path):
    cfg, d, _ = setup(tmp_path, max_calls=10, max_total_tokens=10 ** 9, max_calls_per_problem=9, max_tokens_per_problem=1000)
    records = [ok_record(f"c{i}", f"p{i}", 5, 5) for i in range(7)] + [ok_record("h1", "hot", 850, 50)]    # 8 of 10 calls
    write_records(d, records)
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "near"
    assert "problem hot at 90% of its cap" in out["meter"]["text"] and "calls 8 of 10" not in out["meter"]["text"]


def test_a_global_cap_that_is_closer_is_still_named(root, tmp_path):
    cfg, d, _ = setup(tmp_path, max_calls=10, max_total_tokens=10 ** 9, max_calls_per_problem=9, max_tokens_per_problem=1000)
    records = [ok_record(f"c{i}", f"p{i}", 5, 5) for i in range(8)] + [ok_record("h1", "hot", 800, 0)]       # 9 of 10 calls, 80 percent
    write_records(d, records)
    assert "calls 9 of 10" in app_for(root, cfg).telegraph()["meter"]["text"]


def test_below_the_amber_threshold_the_closest_cap_is_named_whether_global_or_per_problem(root, tmp_path):
    cfg, d, _ = setup(tmp_path, max_calls=10, max_total_tokens=10 ** 9, max_calls_per_problem=9, max_tokens_per_problem=1000)
    write_records(d, [ok_record("a1", "p", 250, 250), ok_record("a2", "q", 5, 5)])           # problem p at 50 percent; calls 2 of 10
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "ok" and "problem p at 50% of its cap" in out["meter"]["text"]


def test_a_tie_names_the_global_cap(root, tmp_path):
    cfg, d, _ = setup(tmp_path, max_calls=10, max_total_tokens=10 ** 9, max_calls_per_problem=9, max_tokens_per_problem=1000)
    write_records(d, [ok_record(f"c{i}", f"p{i}", 5, 5) for i in range(5)] + [ok_record("h", "hot", 250, 250)])   # 6 of 10 vs 50
    assert "calls 6 of 10" in app_for(root, cfg).telegraph()["meter"]["text"]
    write_records(d, [ok_record(f"c{i}", f"p{i}", 5, 5) for i in range(4)] + [ok_record("h", "hot", 250, 250)])   # 5 of 10 vs 50
    assert "calls 5 of 10" in app_for(root, cfg).telegraph()["meter"]["text"]
