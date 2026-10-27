# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The virtual local spend for the web page (orchard/virtualspend_view.py, GET /api/virtual-spend).

Fake agent logs under a fake runs folder. The route is read only, answers on loopback names only, returns no path, keeps
the virtual figure apart from the helper spend, and gives an empty answer (never a 500) for a folder it cannot read."""
import json
from types import SimpleNamespace

import pytest

from orchard import telegraph as tg
from orchard import virtualspend as vs
from orchard import virtualspend_view as view
from test_virtualspend import turn, write_log
from test_webui import Served, make_app, write_ledger


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    return r


def app_with(root, vcfg=None, clock=None):
    app = make_app(root, clock=clock)
    app.cfg = SimpleNamespace(runs_root=root, coder=SimpleNamespace(port=8001, target="org/coder", chips=2),
                              telegraph=tg.Config(), virtual_spend=vcfg or vs.Config())
    return app


def get(app, path="/api/virtual-spend"):
    with Served(app) as s:
        code, _, body = s.request("GET", path)
    return code, json.loads(body)


def test_the_answer_lists_runs_and_probe_outputs_and_prices_each(root):
    write_log(root, "org--a/stages/1/log/run-1.jsonl", [turn(1, 1_000_000, 100_000)])
    write_log(root, "kprobe/rms5/rmsnorm/attempt-1/log/probe-1.jsonl", [turn(1, 2_000_000, 0)])
    (root / "kprobe" / "rms5" / "rmsnorm" / "attempt-1" / "attempt.json").write_text(json.dumps({"agent_wall_s": 3600}))
    code, data = get(app_with(root))
    assert code == 200 and data["virtual"] is True and data["scope"] == "all"
    names = [r["name"] for r in data["rows"]]
    assert sorted(names) == ["kprobe/rms5", "org--a"]
    assert names[0] == "kprobe/rms5"                                    # the larger virtual cost first ($0.20 against $0.15)
    row = data["rows"][0]
    assert (row["turns"], row["prompt_tokens"], row["agent_wall_s"]) == (1, 2_000_000, 3600)
    assert row["virtual"][0] == {"name": "claude-haiku-5-5", "cost_usd": pytest.approx(0.2)}
    assert data["total"]["turns"] == 2 and data["total"]["virtual"][0]["cost_usd"] == pytest.approx(0.35)
    assert data["meter"]["text"] == "Local ≈ $0.35 (virtual, at claude-haiku-5-5)"
    assert "not part of the helper spend" in data["meter"]["title"]


def test_the_kernel_probe_folder_itself_is_not_a_row(root):
    write_log(root, "kprobe/rms5/op/attempt-1/log/p.jsonl", [turn()])
    names = [r["name"] for r in get(app_with(root))[1]["rows"]]
    assert names == ["kprobe/rms5"]


def test_a_selected_run_scopes_the_total_and_the_meter_to_that_run(root):
    write_log(root, "org--a/stages/1/log/run.jsonl", [turn(1, 1_000_000, 0)])
    write_log(root, "org--b/stages/1/log/run.jsonl", [turn(1, 3_000_000, 0)])
    code, data = get(app_with(root), "/api/virtual-spend?run=org--a")
    assert data["scope"] == "run" and data["run"] == "org--a"
    assert data["total"]["prompt_tokens"] == 1_000_000 and data["meter"]["text"] == "Local ≈ $0.10 (virtual, at claude-haiku-5-5)"
    assert len(data["rows"]) == 2, "the table still lists every run"


def test_a_run_with_no_local_logs_has_no_meter_and_a_zero_total(root):
    write_log(root, "org--a/stages/1/log/run.jsonl", [turn(1, 1_000_000, 0)])
    write_ledger(root / "org--idle", ("stage_start", 0, {}))
    code, data = get(app_with(root), "/api/virtual-spend?run=org--idle")
    assert code == 200 and data["meter"] is None and data["total"]["turns"] == 0
    assert [r["name"] for r in data["rows"]] == ["org--a"] and data["total"]["virtual"][0]["cost_usd"] == 0


def test_no_logs_at_all_gives_no_meter_and_no_rows(root):
    code, data = get(app_with(root))
    assert code == 200 and data["meter"] is None and data["rows"] == [] and data["more_rows"] == 0


def test_a_missing_runs_folder_is_an_empty_answer(tmp_path):
    code, data = get(app_with(tmp_path / "gone"))
    assert code == 200 and data["rows"] == [] and data["meter"] is None


def test_an_unknown_or_odd_run_name_is_never_turned_into_a_path(root, tmp_path):
    outside = tmp_path / "outside"
    write_log(outside, "log/run.jsonl", [turn(1, 5_000_000, 0)])
    write_log(root, "org--a/log/run.jsonl", [turn(1, 1_000_000, 0)])
    app = app_with(root)
    for name in ("../outside", "/" + str(outside), "org--a/../../outside", "x" * 500):
        code, data = get(app, "/api/virtual-spend?run=" + name.replace("/", "%2F"))
        assert code == 200 and data["total"]["turns"] == 0 and "5000000" not in json.dumps(data)
    code, data = get(app, "/api/virtual-spend?run=")                    # a blank value is no run at all
    assert code == 200 and data["scope"] == "all" and data["total"]["prompt_tokens"] == 1_000_000


def test_no_response_holds_a_path_or_the_home_directory(root):
    write_log(root, "org--a/stages/1/log/run.jsonl", ["{bad"])
    app = app_with(root)
    code, data = get(app)
    text = json.dumps(data)
    assert str(root) not in text and "/tmp/" not in text and "orchard-runs" not in text


def test_a_row_name_is_redacted_and_cut(root):
    name = "org--tok-hf_" + "a" * 30
    write_log(root, f"{name}/log/run.jsonl", [turn()])
    app = app_with(root)
    data = get(app)[1]
    assert "hf_" + "a" * 30 not in json.dumps(data) and "<token>" in data["rows"][0]["name"]


def test_a_malformed_log_folder_gives_a_row_with_what_was_left_out_not_an_error(root):
    write_log(root, "org--a/log/run.jsonl", ["{bad", turn(1, 10, 1)])
    write_log(root, "org--b/log/run.jsonl", ["{bad"])
    code, data = get(app_with(root))
    assert code == 200
    row = [r for r in data["rows"] if r["name"] == "org--a"][0]
    assert row["turns"] == 1 and "1 lines that are not JSON" in row["left_out"]
    assert "org--b" not in [r["name"] for r in data["rows"]]


def test_a_scan_that_raises_leaves_that_folder_out_and_keeps_the_rest(root, monkeypatch):
    write_log(root, "org--a/log/run.jsonl", [turn(1, 10, 1)])
    write_log(root, "org--b/log/run.jsonl", [turn(1, 20, 2)])
    real = vs.scan

    def flaky(p):
        if p.name == "org--a":
            raise RuntimeError("boom")
        return real(p)
    monkeypatch.setattr(vs, "scan", flaky)
    code, data = get(app_with(root))
    assert code == 200 and [r["name"] for r in data["rows"]] == ["org--b"]


def test_the_rows_are_capped_and_the_rest_counted(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_ROWS", 2)
    for i in range(4):
        write_log(root, f"org--{i}/log/run.jsonl", [turn(1, 100 * (i + 1), 0)])
    code, data = get(app_with(root))
    assert len(data["rows"]) == 2 and data["more_rows"] == 2
    assert data["total"]["prompt_tokens"] == 1000                       # the total still counts every run


def test_the_directories_looked_at_are_capped(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_CANDIDATES", 3)
    for i in range(6):
        write_log(root, f"org--{i}/log/run.jsonl", [turn()])
    assert len(get(app_with(root))[1]["rows"]) == 3


def test_links_under_the_runs_folder_are_not_followed(root, tmp_path):
    outside = tmp_path / "outside"
    write_log(outside, "log/run.jsonl", [turn(1, 5_000_000, 0)])
    (root / "linked").symlink_to(outside)
    (root / "kprobe").mkdir()
    (root / "kprobe" / "linked").symlink_to(outside)
    assert get(app_with(root))[1]["rows"] == []


def test_the_references_come_from_the_config_in_order(root):
    write_log(root, "org--a/log/run.jsonl", [turn(1, 1_000_000, 100_000)])
    cfg = vs.load({"reference": "tiny", "price_in": 1, "price_out": 2, "extra": [{"name": "big", "price_in": 2, "price_out": 10}]})
    data = get(app_with(root, cfg))[1]
    assert [r["name"] for r in data["references"]] == ["tiny", "big"]
    assert data["meter"]["text"] == "Local ≈ $1.20 (virtual, at tiny)"
    assert [v["name"] for v in data["rows"][0]["virtual"]] == ["tiny", "big"]
    assert data["rows"][0]["virtual"][1]["cost_usd"] == pytest.approx(3.0)


def test_a_config_without_the_table_uses_the_defaults(root):
    write_log(root, "org--a/log/run.jsonl", [turn(1, 1_000_000, 0)])
    app = make_app(root)                                                # its cfg has no virtual_spend attribute
    assert get(app)[1]["references"][0]["name"] == "claude-haiku-5-5"


def test_scans_are_cached_for_a_short_time_and_then_read_again(root):
    t = [1000.0]
    app = app_with(root, clock=lambda: t[0])
    write_log(root, "org--a/log/run.jsonl", [turn(1, 100, 0)])
    assert app.virtual_spend()["total"]["prompt_tokens"] == 100
    write_log(root, "org--a/log/run.jsonl", [turn(1, 100, 0), turn(2, 50, 0)])
    t[0] += view.CACHE_S - 1
    assert app.virtual_spend()["total"]["prompt_tokens"] == 100         # still cached
    t[0] += 2
    assert app.virtual_spend()["total"]["prompt_tokens"] == 150


def test_the_route_answers_loopback_hosts_only_and_takes_no_post(root):
    app = app_with(root)
    with Served(app) as s:
        assert s.request("GET", "/api/virtual-spend", host="evil.example.com")[0] == 403
        code, _ = s.post("/api/virtual-spend", {})[:2]
        assert code in (404, 405)


def test_the_route_spends_nothing_and_writes_nothing(root, monkeypatch):
    write_log(root, "org--a/log/run.jsonl", [turn()])
    before = sorted(p.as_posix() for p in root.rglob("*"))
    calls = []
    monkeypatch.setattr(tg, "http_post", lambda *a, **k: calls.append(a))
    get(app_with(root))
    assert calls == [] and sorted(p.as_posix() for p in root.rglob("*")) == before


def test_the_helper_meter_facts_are_unchanged_by_local_logs(root):
    write_log(root, "org--a/log/run.jsonl", [turn(1, 9_000_000, 900_000)])
    app = app_with(root)
    t = app.telegraph()
    assert t["state"] == "off" and t["meter"]["text"] == "Helper: off" and t["summary"]["cost_usd"] == 0


def test_the_figure_is_labelled_virtual_in_every_place_it_appears(root):
    write_log(root, "org--a/log/run.jsonl", [turn()])
    data = get(app_with(root))[1]
    assert "virtual" in data["label"] and "virtual" in data["meter"]["text"] and "never added" in data["label"]



# ---- cases added after the first mutation table ----------------------------------------------------------------------

def test_candidates_leave_out_links_before_any_scan(root, tmp_path):
    (tmp_path / "real").mkdir()
    (root / "linked").symlink_to(tmp_path / "real")
    (root / "plain").mkdir()
    assert [n for n, _ in view.candidates(root)] == ["plain"]


def test_a_run_name_in_the_answer_is_redacted(root):
    write_log(root, "org--a/log/run.jsonl", [turn()])
    code, data = get(app_with(root), "/api/virtual-spend?run=hf_" + "b" * 30)
    assert code == 200 and "b" * 30 not in json.dumps(data) and data["run"] == "<token>"


def test_a_small_cost_keeps_four_digits_in_the_meter_text(root):
    write_log(root, "org--a/log/run.jsonl", [turn(1, 1000, 10)])
    data = get(app_with(root))[1]
    assert data["meter"]["text"] == "Local ≈ $0.0001 (virtual, at claude-haiku-5-5)"
    assert view.usd(0) == "$0.00" and view.usd(0.5) == "$0.50"



# ---- review fix 7: one refresh at a time, and the cache is dated when the scan finishes ---------------------------------

def test_overlapping_requests_share_one_scan(root, monkeypatch):
    import threading
    import time as _time
    for i in range(3):
        write_log(root, f"org--{i}/log/run.jsonl", [turn(1, 100, 1)])
    real = vs.scan
    calls, guard = [], threading.Lock()

    def slow(path):
        with guard:
            calls.append(path.name)
        _time.sleep(0.15)
        return real(path)
    monkeypatch.setattr(vs, "scan", slow)
    app = app_with(root)
    results = []
    threads = [threading.Thread(target=lambda: results.append(app.virtual_spend()["total"]["turns"])) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [3] * 8
    assert sorted(calls) == ["org--0", "org--1", "org--2"], calls            # each folder scanned once, not once per request


def test_the_cache_is_dated_when_the_scan_ends_so_a_slow_scan_is_still_reused(root, monkeypatch):
    write_log(root, "org--a/log/run.jsonl", [turn(1, 100, 1)])
    t = [1000.0]
    real = vs.scan
    calls = []

    def slow(path):
        calls.append(path.name)
        t[0] += view.CACHE_S + 5                                             # the scan takes longer than the cache lifetime
        return real(path)
    monkeypatch.setattr(vs, "scan", slow)
    app = app_with(root, clock=lambda: t[0])
    app.virtual_spend()
    app.virtual_spend()
    assert calls == ["org--a"], "the second request must reuse the result for the full cache lifetime"
