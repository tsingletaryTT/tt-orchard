# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixes for the fourth Copilot review of PR 111 (API side). Each test was written, and seen to fail, before its fix."""
import dataclasses
import json
import math

import pytest

from orchard import telegraph as tg
from orchard import telegraph_view as view
from test_telegraph_view import app_for, setup
from test_telegraph_view_review import ok_record, reserve_record, write_records
from test_telegraph_view_review3 import write_raw
from test_webui import Served, write_ledger


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    return r


def strict_json(out):
    return json.dumps(out, allow_nan=False)


# ---- 1. derived numbers: a sum of finite values can overflow, and a string can read as NaN ---------------------------------

def test_a_sum_of_finite_costs_that_overflows_is_a_damaged_ledger(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [ok_record("a1", "p", 1, 1, cost=1.7e308), ok_record("a2", "p", 1, 1, cost=1.7e308)])
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "problem" and out["meter"]["text"] == "Helper: ledger unreadable · spend unknown"
    strict_json(out)


def test_an_overflowing_sum_does_not_break_a_configured_cost_cap(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [ok_record("a1", "p", 1, 1, cost=1.7e308), ok_record("a2", "p", 1, 1, cost=1.7e308)])
    out = app_for(root, dataclasses.replace(cfg, max_cost_usd=10.0)).telegraph()          # must not raise
    assert out["state"] == "problem"
    strict_json(out)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "nan", "inf"])
def test_a_latency_written_as_a_string_that_reads_as_not_finite_is_refused(root, tmp_path, value):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [{**ok_record("a1", "p", 1, 1), "latency_s": value}])
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "problem"
    strict_json(out)


def test_a_huge_but_finite_amount_does_not_raise_in_the_percent(root, tmp_path):
    cfg, d, _ = setup(tmp_path, max_calls=5, max_total_tokens=10, max_calls_per_problem=9, max_tokens_per_problem=10)
    write_records(d, [ok_record("a1", "p", 10 ** 300, 10 ** 300, cost=1.0)])
    out = app_for(root, cfg).telegraph()                       # int(inf) would raise OverflowError
    strict_json(out)
    assert out["state"] in ("capped", "problem")


def test_a_ratio_that_overflows_is_capped_and_never_raises(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [ok_record("a1", "p", 1, 1, cost=1e300)])
    tiny = dataclasses.replace(cfg, max_cost_usd=1e-300, max_cost_per_problem_usd=1e-300)    # 1e300 * 100 / 1e-300 is infinite
    out = app_for(root, tiny).telegraph()                      # must not raise OverflowError
    strict_json(out)
    cost_cap = next(c for c in out["caps"] if c["id"] == "cost")
    assert cost_cap["percent"] == view.PERCENT_MAX and out["state"] == "capped"
    assert out["problems"][0]["percent"] == view.PERCENT_MAX


def test_an_integer_too_large_for_a_float_is_a_damaged_ledger(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [ok_record("a1", "p", 10 ** 400, 1)])
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "problem"
    strict_json(out)


@pytest.mark.parametrize("field,value", [("cost_usd", -1.0), ("tokens_in", -5), ("tokens_out", -5), ("latency_s", -1.0)])
def test_a_negative_amount_is_a_damaged_ledger(root, tmp_path, field, value):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [ok_record("a1", "p", 10, 5), {**ok_record("a2", "p", 10, 5), field: value}])
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "problem" and out["meter"]["text"].endswith("spend unknown")


def test_a_negative_reservation_or_torn_amount_is_refused_too(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    write_records(d, [reserve_record("r1", "p", "small", -100, 0.01)])
    assert app_for(root, cfg).telegraph()["state"] == "problem"
    write_records(d, [{"torn": "x", "tokens": -3}])
    assert app_for(root, cfg).telegraph()["state"] == "problem"


def test_nothing_non_finite_or_negative_is_cached_and_the_next_good_ledger_is_read(root, tmp_path):
    cfg, d, _ = setup(tmp_path)
    app = app_for(root, cfg)
    write_records(d, [ok_record("a1", "p", 1, 1, cost=1.7e308), ok_record("a2", "p", 1, 1, cost=1.7e308)])
    assert app.telegraph()["state"] == "problem"
    write_records(d, [ok_record("a1", "p", 1, 1, cost=0.5)])
    out = app.telegraph()
    assert out["state"] != "problem" and math.isfinite(out["spend"]["cost_usd"])


def test_every_number_in_a_good_answer_is_finite_and_not_negative(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=9, max_calls_per_problem=9)
    tel.ask("a", "task one " * 80, "failed")
    write_records_extra = write_records                         # a reservation and a torn note beside real calls
    out = app_for(root, cfg).telegraph()

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            assert math.isfinite(v) and v >= 0 or v is None, v
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk({k: v for k, v in out.items() if k not in ("now",)})


# ---- 2. a run's own ledger that cannot be read is an explicit unknown --------------------------------------------------------

def run_with_ledger(root, tmp_path, content=None, make=None):
    run = write_ledger(root / "org--a", ("stage_start", 0, {}))
    if content is not None:
        (run / "telegraph.jsonl").write_text(content)
    if make:
        make(run / "telegraph.jsonl")
    cfg, d, _ = setup(tmp_path)
    return app_for(root, cfg), run


def test_a_run_ledger_that_is_a_directory_gives_an_error_state_and_not_none(root, tmp_path):
    app, run = run_with_ledger(root, tmp_path, make=lambda p: p.mkdir())
    assert app.run_detail("org--a")["telegraph"] == {"error": "the run's ledger cannot be read"}


def test_a_run_ledger_with_a_damaged_record_gives_an_error_state(root, tmp_path):
    app, run = run_with_ledger(root, tmp_path, content=json.dumps({"call_id": [], "outcome": "ok"}) + "\n")
    assert app.run_detail("org--a")["telegraph"]["error"]


def test_a_run_ledger_that_is_too_large_gives_an_error_state(root, tmp_path):
    def big(p):
        with p.open("wb") as fh:
            fh.truncate(view.LEDGER_MAX_BYTES + 1)
    app, run = run_with_ledger(root, tmp_path, make=big)
    assert "too large" in app.run_detail("org--a")["telegraph"]["error"]


def test_a_run_with_no_ledger_or_a_link_still_has_no_spend_line(root, tmp_path):
    app, run = run_with_ledger(root, tmp_path)
    assert app.run_detail("org--a")["telegraph"] is None
    outside = tmp_path / "outside.jsonl"
    outside.write_text("")
    (run / "telegraph.jsonl").symlink_to(outside)
    assert app.run_detail("org--a")["telegraph"] is None


def test_a_good_run_ledger_still_gives_its_numbers(root, tmp_path):
    app, run = run_with_ledger(root, tmp_path, content=json.dumps(ok_record("a1", "p", 10, 5)) + "\n")
    got = app.run_detail("org--a")["telegraph"]
    assert got["calls"] == 1 and "error" not in got


# ---- 3. the meter title says what the meter says -----------------------------------------------------------------------------

def test_the_meter_title_starts_with_the_meters_own_text(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=5, max_calls_per_problem=9)
    for i in range(4):
        tel.ask(f"some-quite-long-problem-name-{i}", f"task {i} " * 80, "failed")
    out = app_for(root, cfg).telegraph()
    assert out["state"] == "near"
    assert out["meter"]["title"].startswith(out["meter"]["text"])
