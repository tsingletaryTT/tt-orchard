# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 461, third batch: an integer too large for a float (`10**400`) must be refused or shown as a bound, never raise OverflowError.

Modules: telegraph_view (the ledger page), speedwatch (the baseline file), text_stages (the perplexity reference of stage 1). Each test feeds
`10**400`, `-10**400`, `inf` and `NaN` to the entry point that reads the value. Where the safe reader would refuse a 400-digit integer before the
module sees it, the test calls the module below the reader, the way a caller with a different parser would."""
import json
import math

import pytest

from orchard import speedwatch as sw, telegraph_view as tv

HOSTILE = [10 ** 400, -(10 ** 400), float("inf"), float("-inf"), float("nan")]


def _id(v):
    """A short test id: a 400-digit integer would fill the report."""
    return "huge-int" if isinstance(v, int) and abs(v) > 10 ** 100 else None


# ---- telegraph_view ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", HOSTILE, ids=_id)
def test_a_ledger_amount_a_float_cannot_hold_is_refused_as_a_damaged_amount(value):
    with pytest.raises(ValueError):                       # ValueError is the refusal; OverflowError is the crash
        tv._non_negative("cost_usd", value)


@pytest.mark.parametrize("value", [10 ** 400, float("inf"), float("nan")], ids=_id)
def test_a_derived_amount_a_float_cannot_hold_is_refused(value):
    with pytest.raises(ValueError):
        tv._check_derived({"spent": value})


def test_a_large_but_ordinary_amount_is_still_accepted():
    tv._non_negative("tokens_in", 10 ** 30)               # an int a float holds
    tv._non_negative("cost_usd", 12.5)
    tv._check_derived({"spent": 3, "rows": [1.5, 2]})


@pytest.mark.parametrize("used,limit", [(10 ** 400, 5), (10 ** 400, 10 ** 399), (5, 10 ** 400), (10 ** 400, 10 ** 400)], ids=["big-used", "both-big", "big-limit", "equal-big"])
def test_a_percent_of_a_huge_integer_never_raises_and_stays_in_range(used, limit):
    assert 0 <= tv._percent(used, limit) <= tv.PERCENT_MAX


def test_a_percent_over_a_huge_limit_is_zero_and_over_a_huge_use_is_the_cap():
    assert tv._percent(5, 10 ** 400) == 0
    assert tv._percent(10 ** 400, 5) == tv.PERCENT_MAX


@pytest.mark.parametrize("value", [10 ** 400, -(10 ** 400)], ids=_id)
def test_a_time_stamp_a_float_cannot_hold_is_not_a_time(value):
    assert tv._time(value) is None


@pytest.mark.parametrize("value", [10 ** 400, -(10 ** 400)], ids=_id)
def test_dollars_of_a_huge_integer_do_not_raise(value):
    assert isinstance(tv.usd(value), str)


def test_a_ledger_record_with_a_huge_amount_is_a_damaged_ledger_in_check_records():
    for key in tv.NUMBER_KEYS:
        with pytest.raises(ValueError):
            tv._check_records([{"call_id": "c1", key: 10 ** 400}])


# ---- speedwatch -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tok_s", [10 ** 400], ids=_id)
def test_a_baseline_file_with_a_speed_a_float_cannot_hold_is_ignored(tmp_path, monkeypatch, tok_s):
    path = tmp_path / "baselines.json"
    path.write_text(json.dumps({"http://h:1 m": {"tok_s": tok_s}, "http://h:1 ok": {"tok_s": 12.5}}))
    # A caller with a reader that has no digit cap: the safe reader is replaced by plain json for this test.
    monkeypatch.setattr(sw.safeio, "read_json", lambda p, **kw: json.loads(p.read_text()))
    b = sw.Baselines(path)
    assert b.get("http://h:1", "m") is None               # the huge entry is dropped, not converted
    assert b.get("http://h:1", "ok") == 12.5


def test_update_keeps_working_when_the_file_holds_a_huge_speed(tmp_path, monkeypatch):
    path = tmp_path / "baselines.json"
    path.write_text(json.dumps({"http://h:1 m": {"tok_s": 10 ** 400}}))
    monkeypatch.setattr(sw.safeio, "read_json", lambda p, **kw: json.loads(p.read_text()))
    assert sw.Baselines(path).update("http://h:1", "m", 20.0) == 20.0


def test_a_reservation_row_with_a_huge_amount_shows_no_amount_and_does_not_raise():
    row = tv._reservation_row({"reserve": "c1", "cost_usd": 10 ** 400}, lambda t: t)
    assert row["cost_usd"] is None


def test_a_call_row_with_a_huge_amount_or_latency_does_not_raise():
    row = tv._call_row({"call_id": "c1", "cost_usd": 10 ** 400, "latency_s": 10 ** 400, "tokens_in": 10 ** 400, "outcome": "ok"}, {}, {}, lambda t: t)
    assert row["cost_usd"] is None and row["latency_s"] == 0.0 and row["tokens_in"] == 0


def test_a_ledger_holding_a_huge_amount_answers_with_a_damaged_ledger_error_and_no_traceback():
    """The whole read, through `_check_records`, as the page does it."""
    records = [{"call_id": "c1", "problem": "p", "model": "m", "cost_usd": 10 ** 400, "tokens_in": 1, "tokens_out": 1}]
    with pytest.raises(ValueError):
        tv._check_records(records)


@pytest.mark.parametrize("baseline", [10 ** 400, -(10 ** 400), float("inf"), float("nan"), 0, -1.0], ids=lambda v: "huge-int" if isinstance(v, int) and abs(v) > 10 ** 100 else None)
def test_judge_refuses_a_baseline_no_float_can_hold_with_its_own_error(baseline):
    with pytest.raises(ValueError, match="baseline must be a speed greater than 0"):
        sw.judge([10.0, 11.0, 12.0], baseline)


def test_a_percent_whose_scaled_ratio_a_float_cannot_hold_is_the_cap():
    """10**307 fits a float, but `10**307 * 100 / 1` is a quotient no float holds: whole numbers are compared as whole numbers."""
    assert tv._percent(10 ** 307, 1) == tv.PERCENT_MAX
    assert tv._percent(10 ** 307, 3) == tv.PERCENT_MAX
    assert tv._percent(10, 40) == 25                         # ordinary whole numbers still divide


def test_a_percent_with_one_float_and_one_huge_integer_does_not_raise():
    assert tv._percent(10 ** 400, 2.5) == tv.PERCENT_MAX
    assert tv._percent(2.5, 10 ** 400) == 0
