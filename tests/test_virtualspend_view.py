# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The virtual local spend for the web page (orchard/virtualspend_view.py, GET /api/virtual-spend).

Fake agent logs under a fake runs folder. The route is read only, answers on loopback names only, returns no path, keeps
the virtual figure apart from the helper spend, and gives an empty answer (never a 500) for a folder it cannot read."""
import json
import os
import shutil
from pathlib import Path
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

    def flaky(p, **kw):
        if p.name == "org--a":
            raise RuntimeError("boom")
        return real(p, **kw)
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

    def slow(path, **kw):
        with guard:
            calls.append(path.name)
        _time.sleep(0.15)
        return real(path, **kw)
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

    def slow(path, **kw):
        calls.append(path.name)
        t[0] += view.CACHE_S + 5                                             # the scan takes longer than the cache lifetime
        return real(path, **kw)
    monkeypatch.setattr(vs, "scan", slow)
    app = app_with(root, clock=lambda: t[0])
    app.virtual_spend()
    app.virtual_spend()
    assert calls == ["org--a"], "the second request must reuse the result for the full cache lifetime"



# ---- review fix 3 (third round): a selected run is always scanned, and directories left unscanned are reported ---------------

def _many(root, n=6):
    for i in range(n):
        write_log(root, f"org--{i:02d}/log/run.jsonl", [turn(1, 1_000_000 * (i + 1), 0)])


def test_a_selected_run_past_the_scan_cap_is_still_scanned(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_CANDIDATES", 3)
    _many(root)
    code, data = get(app_with(root), "/api/virtual-spend?run=org--05")
    assert code == 200 and data["scope"] == "run"
    assert data["total"]["prompt_tokens"] == 6_000_000
    assert data["meter"]["text"] == "Local ≈ $0.60 (virtual, at claude-haiku-5-5)"
    assert "org--05" in [r["name"] for r in data["rows"]], "its row is in the table too"


def test_an_unlisted_run_name_is_never_scanned(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_CANDIDATES", 3)
    _many(root)
    scanned = []
    real = vs.scan
    monkeypatch.setattr(vs, "scan", lambda p, **kw: (scanned.append(p.name), real(p, **kw))[1])
    code, data = get(app_with(root), "/api/virtual-spend?run=../org--05")
    assert code == 200 and data["total"]["turns"] == 0 and data["meter"] is None
    assert sorted(scanned) == ["org--00", "org--01", "org--02"]


def test_directories_left_unscanned_are_reported(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_CANDIDATES", 3)
    _many(root)
    data = get(app_with(root))[1]
    assert data["not_scanned"] == 3 and len(data["rows"]) == 3
    assert "3 runs were not scanned" in data["meter"]["title"]


def test_a_scanned_extra_run_is_not_counted_as_unscanned(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_CANDIDATES", 3)
    _many(root)
    data = get(app_with(root), "/api/virtual-spend?run=org--05")[1]
    assert data["not_scanned"] == 2


def test_nothing_unscanned_when_every_directory_fits(root):
    _many(root, 3)
    data = get(app_with(root))[1]
    assert data["not_scanned"] == 0 and "not scanned" not in data["meter"]["title"]


def test_the_extra_run_is_cached_like_the_others(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_CANDIDATES", 3)
    _many(root)
    t = [1000.0]
    app = app_with(root, clock=lambda: t[0])
    scanned = []
    real = vs.scan
    monkeypatch.setattr(vs, "scan", lambda p, **kw: (scanned.append(p.name), real(p, **kw))[1])
    app.virtual_spend("org--05")
    app.virtual_spend("org--05")
    assert scanned.count("org--05") == 1
    t[0] += view.CACHE_S + 1
    app.virtual_spend("org--05")
    assert scanned.count("org--05") == 2


def test_the_hard_cap_on_enumerated_directories_applies(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_NAMES", 4)
    _many(root, 8)
    assert len(view.candidates(root)) == 4


# ---- review fix 3 (fourth round): discovery is bounded while it streams, and a cut-short listing is reported -----------------

class _CountingScandir:
    """Counts the entries a listing takes from os.scandir."""

    def __init__(self, monkeypatch):
        self.taken = 0
        real = os.scandir
        spy = self

        def scandir(path):
            inner = real(path)

            class Wrapped:
                def __enter__(self_inner):
                    inner.__enter__()
                    return self_inner

                def __exit__(self_inner, *a):
                    return inner.__exit__(*a)

                def __iter__(self_inner):
                    return self_inner

                def __next__(self_inner):
                    entry = next(inner)
                    spy.taken += 1
                    return entry

                def close(self_inner):
                    inner.close()
            return Wrapped()
        monkeypatch.setattr(view.os, "scandir", scandir)


def test_discovery_stops_taking_entries_at_the_budget_and_says_it_was_cut_short(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_NAMES", 5)
    for i in range(40):
        (root / f"d{i:02d}").mkdir()
        (root / f"f{i:02d}.txt").write_text("x")
    spy = _CountingScandir(monkeypatch)
    found, truncated = view.discover(root)
    assert truncated is True and len(found) <= 5
    assert spy.taken <= 5 + 1, f"{spy.taken} entries were taken from a listing with a budget of 5"


def test_files_count_against_the_entry_budget(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_NAMES", 3)
    for i in range(10):
        (root / f"f{i}.txt").write_text("x")
    (root / "zz-dir").mkdir()
    found, truncated = view.discover(root)
    assert truncated is True


def test_the_probe_folder_shares_the_same_budget(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_NAMES", 6)
    for i in range(5):
        (root / f"d{i}").mkdir()
    (root / "kprobe").mkdir()
    for i in range(30):
        (root / "kprobe" / f"p{i:02d}").mkdir()
    spy = _CountingScandir(monkeypatch)
    found, truncated = view.discover(root)
    assert truncated is True and spy.taken <= 6 + 2


def test_a_listing_that_fits_is_not_truncated(root):
    for i in range(4):
        (root / f"d{i}").mkdir()
    found, truncated = view.discover(root)
    assert truncated is False and len(found) == 4


def test_a_listing_exactly_at_the_budget_is_not_truncated(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_NAMES", 4)
    for i in range(4):
        (root / f"d{i}").mkdir()
    found, truncated = view.discover(root)
    assert truncated is False and len(found) == 4


def test_the_answer_and_the_meter_say_when_discovery_was_cut_short(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_NAMES", 3)
    for i in range(8):
        write_log(root, f"org--{i}/log/run.jsonl", [turn(1, 1_000_000, 0)])
    data = get(app_with(root))[1]
    assert data["discovery_truncated"] is True
    assert "folder listing was cut short" in data["meter"]["title"]


def test_the_answer_says_discovery_was_complete_when_it_was(root):
    write_log(root, "org--a/log/run.jsonl", [turn()])
    data = get(app_with(root))[1]
    assert data["discovery_truncated"] is False and "cut short" not in data["meter"]["title"]


def test_discovery_leaves_out_links_and_survives_an_unreadable_folder(root, tmp_path):
    (tmp_path / "real").mkdir()
    (root / "linked").symlink_to(tmp_path / "real")
    (root / "plain").mkdir()
    assert [n for n, _ in view.discover(root)[0]] == ["plain"]
    assert view.discover(tmp_path / "missing") == ([], False)


def test_a_cached_answer_still_says_the_listing_was_cut_short(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_NAMES", 3)
    for i in range(8):
        write_log(root, f"org--{i}/log/run.jsonl", [turn(1, 1_000_000, 0)])
    app = app_with(root)
    first, second = app.virtual_spend(), app.virtual_spend()                 # the second answer comes from the cache
    assert first["discovery_truncated"] is True and second["discovery_truncated"] is True


# ---- review (fifth round): the runs folder is pinned too, so no ancestor of a scanned folder is followed ----------------------

def test_the_probe_folder_swapped_for_a_link_after_discovery_is_not_read(root, tmp_path, monkeypatch):
    write_log(root, "kprobe/rms5/op/attempt-1/log/p.jsonl", [turn(1, 5, 1)])
    outside = tmp_path / "outside"
    write_log(outside, "rms5/op/attempt-1/log/p.jsonl", [turn(1, 999, 9)])
    real = view.discover

    def discover_then_swap(r, *a):
        found = real(r, *a)
        shutil.rmtree(root / "kprobe")
        (root / "kprobe").symlink_to(outside, target_is_directory=True)      # the ancestor of a listed folder is swapped
        return found
    monkeypatch.setattr(view, "discover", discover_then_swap)
    data = get(app_with(root))[1]
    assert data["total"]["turns"] == 0 and "999" not in json.dumps(data)


def test_a_run_folder_swapped_for_a_link_after_discovery_is_not_read(root, tmp_path, monkeypatch):
    write_log(root, "org--a/stages/1/log/run.jsonl", [turn(1, 5, 1)])
    outside = tmp_path / "outside-run"
    write_log(outside, "stages/1/log/run.jsonl", [turn(1, 999, 9)])
    real = view.discover

    def discover_then_swap(r, *a):
        found = real(r, *a)
        shutil.rmtree(root / "org--a")
        (root / "org--a").symlink_to(outside, target_is_directory=True)
        return found
    monkeypatch.setattr(view, "discover", discover_then_swap)
    data = get(app_with(root))[1]
    assert data["total"]["turns"] == 0 and "999" not in json.dumps(data)


def test_the_runs_folder_itself_swapped_after_the_scan_began_is_not_followed(root, tmp_path, monkeypatch):
    write_log(root, "org--a/stages/1/log/run.jsonl", [turn(1, 5, 1)])
    outside = tmp_path / "outside-all"
    write_log(outside, "org--a/stages/1/log/run.jsonl", [turn(1, 999, 9)])
    real = view.discover

    def discover_then_swap_root(r, *a):
        found = real(r, *a)
        moved = r.parent / "moved-runs"
        r.rename(moved)
        r.symlink_to(outside, target_is_directory=True)
        return found
    monkeypatch.setattr(view, "discover", discover_then_swap_root)
    data = get(app_with(root))[1]
    assert "999" not in json.dumps(data)
    assert data["total"]["prompt_tokens"] == 5, "the pinned original is what was read"


def test_a_selected_run_past_the_cap_is_also_read_beneath_the_pinned_runs_folder(root, tmp_path, monkeypatch):
    monkeypatch.setattr(view, "MAX_CANDIDATES", 1)
    write_log(root, "org--a/log/run.jsonl", [turn(1, 5, 1)])
    write_log(root, "org--b/log/run.jsonl", [turn(1, 6, 1)])
    outside = tmp_path / "outside-b"
    write_log(outside, "log/run.jsonl", [turn(1, 999, 9)])
    app = app_with(root)
    app.virtual_spend()                                                         # the base scan lists both
    shutil.rmtree(root / "org--b")
    (root / "org--b").symlink_to(outside, target_is_directory=True)
    data = app.virtual_spend("org--b")
    assert "999" not in json.dumps(data) and data["total"]["turns"] == 0


def test_scan_beneath_a_pin_refuses_a_swapped_component_and_reads_an_intact_one(root, tmp_path):
    write_log(root, "a/b/log/run.jsonl", [turn(1, 5, 1)])
    pin = vs.open_pin(root)
    try:
        assert vs.scan(root / "a" / "b", beneath=pin)["turns"] == 1
        shutil.rmtree(root / "a")
        outside = tmp_path / "out"
        write_log(outside, "b/log/run.jsonl", [turn(1, 999, 9)])
        (root / "a").symlink_to(outside, target_is_directory=True)
        s = vs.scan(root / "a" / "b", beneath=pin)
        assert s["turns"] == 0 and s["complete"] is False and s["skipped"]["links"] == 1
    finally:
        pin.close()


def test_a_folder_that_is_not_under_the_pin_cannot_be_scanned_beneath_it(root, tmp_path):
    (tmp_path / "elsewhere").mkdir()
    pin = vs.open_pin(root)
    try:
        s = vs.scan(tmp_path / "elsewhere", beneath=pin)
        assert s["turns"] == 0 and s["complete"] is False
    finally:
        pin.close()


def test_no_descriptor_is_left_open_after_a_pinned_refresh(root):
    write_log(root, "kprobe/p/op/attempt-1/log/p.jsonl", [turn()])
    write_log(root, "org--a/log/run.jsonl", [turn()])
    app = app_with(root)
    before = set(os.listdir("/proc/self/fd"))
    app.virtual_spend()
    app.virtual_spend("org--a")
    assert set(os.listdir("/proc/self/fd")) == before


def test_a_selected_run_past_the_cap_leaves_no_descriptor_open(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_CANDIDATES", 1)
    write_log(root, "org--a/log/run.jsonl", [turn()])
    write_log(root, "org--b/log/run.jsonl", [turn()])
    app = app_with(root)
    app.virtual_spend()
    before = set(os.listdir("/proc/self/fd"))
    assert app.virtual_spend("org--b")["total"]["turns"] == 1
    assert set(os.listdir("/proc/self/fd")) == before


def test_nothing_is_scanned_when_the_runs_folder_cannot_be_pinned(root, monkeypatch):
    write_log(root, "org--a/log/run.jsonl", [turn()])
    monkeypatch.setattr(vs, "open_pin", lambda path: None)
    data = get(app_with(root))[1]
    assert data["total"]["turns"] == 0 and data["rows"] == [] and data["meter"] is None


def test_discovery_after_pinning_lists_the_original_folder_not_a_replacement_with_other_names(root, tmp_path, monkeypatch):
    write_log(root, "org--a/log/run.jsonl", [turn(1, 5, 1)])
    other = tmp_path / "other-runs"
    write_log(other, "org--z/log/run.jsonl", [turn(1, 999, 9)])
    real_pin = vs.open_pin

    def pin_then_swap(path):
        pin = real_pin(path)
        moved = Path(path).parent / "moved-runs"
        Path(path).rename(moved)
        Path(path).symlink_to(other, target_is_directory=True)           # right after the pin
        return pin
    monkeypatch.setattr(vs, "open_pin", pin_then_swap)
    data = get(app_with(root))[1]
    assert [r["name"] for r in data["rows"]] == ["org--a"] and "999" not in json.dumps(data)
    assert data["total"]["prompt_tokens"] == 5


def test_discover_through_a_pin_lists_the_probe_outputs_too(root):
    write_log(root, "kprobe/p1/op/attempt-1/log/p.jsonl", [turn()])
    write_log(root, "org--a/log/run.jsonl", [turn()])
    pin = vs.open_pin(root)
    try:
        found, cut = view.discover(root, pin)
    finally:
        pin.close()
    assert sorted(n for n, _ in found) == ["kprobe/p1", "org--a"] and cut is False


def test_discover_through_a_pin_takes_entries_from_the_descriptor(root, monkeypatch):
    for i in range(6):
        (root / f"d{i}").mkdir()
    seen = []
    real = os.scandir
    monkeypatch.setattr(view.os, "scandir", lambda target: (seen.append(type(target).__name__), real(target))[1])
    pin = vs.open_pin(root)
    try:
        view.discover(root, pin)
    finally:
        pin.close()
    assert seen and set(seen) == {"int"}, "every listing is of a descriptor, never of a path"


def test_the_probe_folder_is_not_opened_when_the_budget_is_already_spent(root, monkeypatch):
    for i in range(3):
        (root / f"d{i}").mkdir()
    (root / "kprobe").mkdir()
    for i in range(5):
        (root / "kprobe" / f"p{i}").mkdir()
    monkeypatch.setattr(view, "MAX_NAMES", 4)                                  # the root listing takes the whole budget
    opened = []
    real = os.scandir
    monkeypatch.setattr(view.os, "scandir", lambda target: (opened.append(target), real(target))[1])
    pin = vs.open_pin(root)
    try:
        found, cut = view.discover(root, pin)
    finally:
        pin.close()
    assert len(opened) == 1 and cut is True
    assert all(not n.startswith("kprobe/") for n, _ in found)


# ---- sweep: a limit that leaves logs unread, or a folder that cannot be read, is reported, never shown as zero -----------------

def test_a_run_whose_scan_was_incomplete_is_counted_and_named_in_the_answer(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_DEPTH", 2)
    write_log(root, "org--a/log/run.jsonl", [turn(1, 5, 1)])
    write_log(root, "org--b/x/y/log/run.jsonl", [turn(1, 7, 1)])             # below the depth limit: org--b is read in part
    write_log(root, "org--b/log/run.jsonl", [turn(1, 7, 1)])
    data = get(app_with(root))[1]
    assert data["not_fully_read"] == 1
    assert "1 runs were not fully read" in data["meter"]["title"] or "1 run was not fully read" in data["meter"]["title"]
    row = [r for r in data["rows"] if r["name"] == "org--b"][0]
    assert row["complete"] is False and "folders below the depth limit" in row["left_out"]
    assert [r for r in data["rows"] if r["name"] == "org--a"][0]["complete"] is True


def test_every_run_fully_read_reports_none_not_fully_read(root):
    write_log(root, "org--a/log/run.jsonl", [turn()])
    data = get(app_with(root))[1]
    assert data["not_fully_read"] == 0 and "not fully read" not in data["meter"]["title"]


def test_a_folder_whose_scan_fails_is_counted_not_dropped_silently(root, monkeypatch):
    write_log(root, "org--a/log/run.jsonl", [turn(1, 10, 1)])
    write_log(root, "org--b/log/run.jsonl", [turn(1, 20, 2)])
    real = vs.scan

    def flaky(p, **kw):
        if p.name == "org--a":
            raise RuntimeError("boom")
        return real(p, **kw)
    monkeypatch.setattr(vs, "scan", flaky)
    data = get(app_with(root))[1]
    assert [r["name"] for r in data["rows"]] == ["org--b"] and data["scan_failures"] == 1
    assert "1 run could not be read" in data["meter"]["title"]


def test_a_runs_folder_that_cannot_be_pinned_is_an_error_in_the_answer_not_an_empty_list(root, monkeypatch):
    write_log(root, "org--a/log/run.jsonl", [turn()])
    monkeypatch.setattr(vs, "open_pin", lambda path: None)
    data = get(app_with(root))[1]
    assert data["runs_folder_unreadable"] is True and data["rows"] == []


def test_a_readable_runs_folder_reports_no_error(root):
    write_log(root, "org--a/log/run.jsonl", [turn()])
    data = get(app_with(root))[1]
    assert data["runs_folder_unreadable"] is False and data["scan_failures"] == 0


def test_a_missing_runs_folder_is_unreadable_not_empty(tmp_path):
    data = get(app_with(tmp_path / "gone"))[1]
    assert data["runs_folder_unreadable"] is True


def test_the_probe_folder_swapped_for_a_link_after_the_pin_is_not_listed(root, tmp_path, monkeypatch):
    write_log(root, "kprobe/p1/op/attempt-1/log/p.jsonl", [turn(1, 5, 1)])
    outside = tmp_path / "outside"
    write_log(outside, "zz/op/attempt-1/log/p.jsonl", [turn(1, 999, 9)])
    real_pin = vs.open_pin

    def pin_then_swap(path):
        pin = real_pin(path)
        os.rename(Path(path) / "kprobe", Path(path) / "moved-kprobe")
        (Path(path) / "kprobe").symlink_to(outside, target_is_directory=True)
        return pin
    monkeypatch.setattr(vs, "open_pin", pin_then_swap)
    data = get(app_with(root))[1]
    assert "999" not in json.dumps(data)


def test_a_zero_turn_run_read_in_part_is_in_the_rows(root, monkeypatch):
    monkeypatch.setattr(vs, "MAX_DEPTH", 2)
    write_log(root, "org--deep/x/y/log/run.jsonl", [turn(1, 5, 1)])
    data = get(app_with(root))[1]
    assert [r["name"] for r in data["rows"]] == ["org--deep"] and data["rows"][0]["complete"] is False and data["not_fully_read"] == 1


def test_a_failed_scan_of_the_selected_run_past_the_cap_is_counted(root, monkeypatch):
    monkeypatch.setattr(view, "MAX_CANDIDATES", 1)
    write_log(root, "org--a/log/run.jsonl", [turn()])
    write_log(root, "org--b/log/run.jsonl", [turn()])
    real = vs.scan

    def flaky(p, **kw):
        if p.name == "org--b":
            raise RuntimeError("boom")
        return real(p, **kw)
    monkeypatch.setattr(vs, "scan", flaky)
    data = get(app_with(root), "/api/virtual-spend?run=org--b")[1]
    assert data["scan_failures"] == 1
