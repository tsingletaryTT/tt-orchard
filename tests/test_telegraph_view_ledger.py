# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The web view against the reservation ledger of PR 106: `reserve` and `torn` lines, unsettled reservations, calls with no known
cost and a cost cap that refuses new calls while any exist. The ledgers are written by the real `telegraph.Telegraph`."""
import dataclasses
import json

import pytest

from ledger_io import ledger_records, seed_ledger
from orchard import telegraph as tg
from test_telegraph_view import KEY, app_for, fill, reply, setup
from test_webui import Served


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    return r


def die(url, headers, body, timeout):
    raise KeyboardInterrupt                                   # the process stops after the reservation, before any result


def crash_after_reserve(tel, problem="stuck"):
    real = tel.transport
    tel.transport = die
    with pytest.raises(KeyboardInterrupt):
        tel.ask(problem, "a task that never finished " * 40, "failed")
    tel.transport = real


def no_usage(url, headers, body, timeout):
    return 200, json.dumps({"choices": [{"message": {"content": "try fill"}, "finish_reason": "stop"}]}).encode()


# ---- reserve and torn lines are not calls --------------------------------------------------------------------------------

def test_a_settled_call_is_one_row_and_its_reserve_line_is_not_a_second(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    tel.ask("p", "task " * 80, "failed")
    lines = ledger_records(d)
    assert any("reserve" in r for r in lines)                 # the producer really wrote one
    out = app_for(root, cfg).telegraph()
    assert len(out["recent"]) == 1 and out["recent"][0]["outcome"] == "ok"
    assert out["summary"]["calls"] == 1 and out["spend"]["calls"] == 1 and out["spend"]["unsettled"] == 0


def test_an_unsettled_reservation_is_shown_plainly_and_counts_at_its_worst_case(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=5)
    crash_after_reserve(tel)
    out = app_for(root, cfg).telegraph()
    assert out["summary"]["calls"] == 0                       # nothing was sent and settled
    assert out["spend"]["calls"] == 1 and out["spend"]["unsettled"] == 1
    (row,) = out["recent"]
    assert row["outcome"] == "unsettled" and row["held"] is True and row["cost_usd"] > 0
    assert "worst case" in row["reason"]
    assert "problem stuck at 33% of its cap" in out["meter"]["text"] and out["meter"]["text"].startswith("Helper: $0.")
    assert "· 1 call ·" in out["meter"]["text"]                # the meter counts what the caps count
    cap = next(c for c in out["caps"] if c["id"] == "calls")
    assert cap["used"] == 1


def test_an_unsettled_reservation_counts_toward_a_cap_like_the_producer_counts_it(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=1)
    crash_after_reserve(tel)
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "capped" and "call cap is reached (1 of 1)" in out["capped"]["reason"]
    assert not tel.ask("again", "another task " * 80, "failed").ok          # the producer agrees


def test_a_torn_record_is_one_call_of_unknown_cost_and_never_a_row(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=5)
    # A torn record is a note that an earlier writer left when its last line was cut short (the migration writes it).
    seed_ledger(d, [{"torn": "telegraph.jsonl.torn-ab12cd34", "tokens": 1500}])
    tel.ask("p2", "task " * 80, "failed")
    assert any("torn" in r for r in ledger_records(d))
    out = app_for(root, cfg).telegraph()
    assert out["summary"]["torn"] == 1
    assert out["spend"]["unknown_cost"] == 1 and out["spend"]["calls"] == 2
    assert all(r["outcome"] != "torn" and "torn" not in r["call_id"] for r in out["recent"])
    assert len(out["recent"]) == 1
    assert "+ 1 unknown" in out["meter"]["text"]


# ---- unknown cost is never zero ---------------------------------------------------------------------------------------

def test_calls_with_no_price_are_counted_as_unknown_and_never_shown_as_zero_dollars(root, tmp_path):
    cfg, d, tel = setup(tmp_path, prices={})
    tel.ask("p", "task " * 80, "failed")
    out = app_for(root, cfg).telegraph()
    assert out["spend"]["unknown_cost"] == 1
    assert out["meter"]["text"].startswith("Helper: unknown cost · 1 call")
    assert "$0.00" not in out["meter"]["text"]


def test_a_cost_cap_refuses_new_calls_while_a_cost_is_unknown_and_the_page_says_so(root, tmp_path):
    cfg, d, tel = setup(tmp_path, prices={"small": [1.0, 5.0]}, max_calls=9, max_calls_per_problem=9)
    first = tel.ask("p", "task " * 80, "failed")
    tel.report(first.call_id, helped=False)
    tel.ask("p", "task two " * 80, "failed")                   # the second model has no price: its cost is unknown
    capped = dataclasses.replace(cfg, max_cost_usd=100.0)       # a cost cap now exists
    out = app_for(root, capped).telegraph()
    assert out["state"] == "capped" and out["tone"] == "bad"
    reason = out["capped"]["reason"]
    assert "no known cost" in reason and "cost cap cannot be checked" in reason and "refused" in reason
    cost = next(c for c in out["caps"] if c["id"] == "cost")
    assert cost["blocked"] is True and cost["tone"] == "bad"
    assert "CAPPED" in out["meter"]["text"] and "no known cost" in out["meter"]["text"]
    assert out["meter"]["text"].endswith("new calls are refused")      # the whole reason fits in the meter
    # the producer refuses for the same reason
    refused = tg.Telegraph(capped, d, transport=reply(), environ={"TG_KEY": KEY}).ask("p3", "task three " * 80, "failed")
    assert not refused.ok and "no known cost" in refused.reason


def test_without_a_cost_cap_an_unknown_cost_does_not_stop_the_helper(root, tmp_path):
    cfg, d, tel = setup(tmp_path, prices={"small": [1.0, 5.0]}, max_calls=9, max_calls_per_problem=9)
    first = tel.ask("p", "task " * 80, "failed")
    tel.report(first.call_id, helped=False)
    tel.ask("p", "task two " * 80, "failed")
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "ok" and out["spend"]["unknown_cost"] == 1
    assert all(c["blocked"] is False for c in out["caps"])


def test_a_per_problem_cost_cap_is_full_for_a_problem_with_an_unknown_cost(root, tmp_path):
    cfg, d, tel = setup(tmp_path, prices={"small": [1.0, 5.0]}, max_calls=9, max_calls_per_problem=9)
    first = tel.ask("p", "task " * 80, "failed")
    tel.report(first.call_id, helped=False)
    tel.ask("p", "task two " * 80, "failed")
    capped = dataclasses.replace(cfg, max_cost_per_problem_usd=100.0)
    out = app_for(root, capped).telegraph()
    (p,) = out["problems"]
    assert p["at_cap"] is True and p["cost_unknown_calls"] == 1 and out["state"] == "near"


def test_a_reply_with_no_usage_is_marked_and_counted_at_its_reserved_worst_case(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    tel = tg.Telegraph(cfg, d, transport=no_usage, environ={"TG_KEY": KEY})
    tel.ask("p", "task " * 80, "failed")
    out = app_for(root, cfg).telegraph()
    (row,) = out["recent"]
    assert row["usage_known"] is False and "no usage" in row["reason"]
    assert out["summary"]["unknown_usage_calls"] == 1
    assert out["spend"]["cost_usd"] > 0.0035                  # the reserved worst case, above any estimate


# ---- refusals, and nothing secret ------------------------------------------------------------------------------------

def test_a_refused_call_shows_its_reason_and_is_not_counted_as_spend(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=1, max_calls_per_problem=9)
    fill(tel, 1)
    refused = tel.ask("later", "another task " * 80, "failed")
    assert not refused.ok
    out = app_for(root, cfg).telegraph()
    assert out["recent"][0]["outcome"] == "budget" and "call cap is reached" in out["recent"][0]["reason"]
    assert out["spend"]["calls"] == 1 and out["summary"]["refused"] == 1


def test_reservation_and_torn_lines_never_leak_strings_that_should_be_redacted(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    lines = [{"reserve": "r1", "problem": "/home/someone/proj sk-abcdefghijklmnopqrstuvwx", "model": "quietbox-model",
              "tokens": 100, "cost_usd": None},
             {"torn": "telegraph.jsonl.torn-1", "tokens": 10}]
    seed_ledger(d, lines)
    app = app_for(root, cfg)
    app.hostname, app.home = "quietbox", "/home/someone"
    blob = json.dumps(app.telegraph())
    assert "/home/someone" not in blob and "sk-abcdefghij" not in blob and "quietbox" not in blob


def test_replies_prompts_and_the_cache_are_still_never_returned_with_the_new_records(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    tel.ask("p", "SECRET-TASK-TEXT " * 40, "failed")
    crash_after_reserve(tel, "q")
    blob = json.dumps(app_for(root, cfg).telegraph())
    assert "SECRET-TASK-TEXT" not in blob and "use fill for constants" not in blob
    assert '"reply_file"' not in blob and "prompt_sha256" not in blob and KEY not in blob


def test_a_ledger_directory_that_cannot_be_read_still_gives_an_answer(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    d.mkdir()
    (d / "project.db").mkdir()                                # the database file is a directory: a read is refused
    out = app_for(root, cfg).telegraph()
    assert out["summary"]["calls"] == 0 and out["spend"]["calls"] == 0 and out["state"] in ("ok", "problem")


def test_the_route_returns_the_new_fields_as_json(root, tmp_path):
    cfg, d, tel = setup(tmp_path)
    crash_after_reserve(tel)
    with Served(app_for(root, cfg)) as s:
        code, _, body = s.request("GET", "/api/telegraph")
    data = json.loads(body)
    assert code == 200 and data["spend"]["unsettled"] == 1


def test_the_by_model_table_counts_calls_with_no_known_cost(root, tmp_path):
    cfg, d, tel = setup(tmp_path, prices={"small": [1.0, 5.0]}, max_calls=9, max_calls_per_problem=9)
    first = tel.ask("p", "task " * 80, "failed")
    tel.report(first.call_id, helped=False)
    tel.ask("p", "task two " * 80, "failed")
    models = {m["model"]: m for m in app_for(root, cfg).telegraph()["by_model"]}
    assert models["small"]["cost_unknown_calls"] == 0 and models["big"]["cost_unknown_calls"] == 1
