# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The virtual local-spend figure and table in a real browser (installed Chrome through Playwright).

Fake agent logs under a fake runs folder, the real server on a loopback port. They skip when `playwright` or Chrome is missing
(see tests/test_webui_browser.py for how to run them). One test records every request the page makes: only GETs to the page's
own server, so the page never spends anything."""
import json
import os
import shutil
from types import SimpleNamespace

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

from orchard import telegraph as tg
from orchard import virtualspend as vs
from test_telegraph_view import fill, setup
from test_virtualspend import turn, write_log
from test_webui import Served, make_app, write_ledger

CHROME = os.environ.get("ORCHARD_CHROME") or shutil.which("google-chrome") or shutil.which("chromium")
pytestmark = pytest.mark.skipif(not CHROME, reason="no Chrome or Chromium on this machine")

SHOWN = "Local ≈ $0.35 (virtual, at claude-haiku-5-5)"
FIRST = "Local ≈ $0.15 (virtual, at claude-haiku-5-5)"


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    write_ledger(r / "org--a", ("stage_start", 0, {}), model="org/a-model")
    write_ledger(r / "org--idle", ("stage_start", 0, {}), model="org/idle-model")
    write_log(r, "org--a/stages/1/log/run-1.jsonl", [turn(1, 1_000_000, 100_000)])
    write_log(r, "kprobe/rms5/rmsnorm/attempt-1/log/probe-1.jsonl", [turn(1, 2_000_000, 0)])
    return r


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox"])
        yield b
        b.close()


def new_page(browser, width=1400):
    ctx = browser.new_context(viewport={"width": width, "height": 900})
    pg = ctx.new_page()
    pg.set_default_timeout(15000)
    return ctx, pg


def app_for(root, telegraph=None, vcfg=None):
    app = make_app(root)
    app.cfg = SimpleNamespace(runs_root=root, coder=SimpleNamespace(port=8001, target="org/coder", chips=2),
                              telegraph=telegraph or tg.Config(), virtual_spend=vcfg or vs.Config())
    app.environ = {"TG_KEY": "k"}
    return app


def open_page(pg, served, mode="operator", select=None):
    pg.goto(f"http://127.0.0.1:{served.port}/?mode={mode}" + (f"#run/{select}" if select else ""))
    pg.wait_for_selector("[data-run-row]:visible" if mode == "operator" else ".run-item:visible")


def local(pg):
    return pg.locator("#local-meter")


@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_the_figure_shows_in_both_styles_for_the_run_the_page_selects_first(root, browser, mode):
    ctx, pg = new_page(browser)
    with Served(app_for(root)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode={mode}")
        expect(local(pg)).to_have_text(FIRST)                           # the page selects the first listed run, org--a
        expect(local(pg)).to_be_visible()
    ctx.close()


def test_with_no_run_listed_the_figure_covers_the_probe_outputs_that_have_logs(tmp_path, browser):
    r = tmp_path / "orchard-runs"
    write_log(r, "kprobe/rms5/rmsnorm/attempt-1/log/probe-1.jsonl", [turn(1, 2_000_000, 0)])
    write_log(r, "kprobe/rms6/rmsnorm/attempt-1/log/probe-1.jsonl", [turn(1, 1_000_000, 100_000)])
    ctx, pg = new_page(browser)
    with Served(app_for(r)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")
        expect(local(pg)).to_have_text(SHOWN)                           # $0.20 + $0.15 over both outputs
    ctx.close()


def test_a_selected_run_scopes_the_figure_and_a_run_with_no_logs_hides_it(root, browser):
    ctx, pg = new_page(browser)
    with Served(app_for(root)) as s:
        open_page(pg, s, select="org--a")
        expect(local(pg)).to_have_text("Local ≈ $0.15 (virtual, at claude-haiku-5-5)")
        pg.locator('[data-run-name="org--idle"]:visible').click()
        expect(local(pg)).to_be_hidden()
        pg.locator('[data-run-name="org--a"]:visible').click()
        expect(local(pg)).to_have_text("Local ≈ $0.15 (virtual, at claude-haiku-5-5)")
    ctx.close()


def test_the_figure_is_hidden_when_there_are_no_local_logs(tmp_path, browser):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    write_ledger(r / "org--idle", ("stage_start", 0, {}), model="org/idle-model")
    ctx, pg = new_page(browser)
    with Served(app_for(r)) as s:
        open_page(pg, s)
        pg.wait_for_timeout(1500)
        expect(local(pg)).to_be_hidden()
        pg.click("#view-telegraph-btn")
        expect(pg.locator("#tg-local-empty")).to_be_visible()
    ctx.close()


@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_the_helper_meter_keeps_its_own_state_and_the_local_figure_has_none(root, tmp_path, browser, mode):
    cfg, d, tel = setup(tmp_path, max_calls=2, max_calls_per_problem=9)
    fill(tel, 2)                                                       # the call cap is used up: the helper meter is red
    ctx, pg = new_page(browser)
    with Served(app_for(root, telegraph=cfg)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode={mode}")
        expect(local(pg)).to_have_text(FIRST)
        helper = pg.locator("#spend-meter")
        expect(helper).to_contain_text("CAPPED")
        assert helper.get_attribute("data-tone") == "bad" and helper.get_attribute("data-state") == "capped"
        assert local(pg).get_attribute("data-tone") is None and local(pg).get_attribute("data-state") is None
        assert "CAPPED" not in local(pg).inner_text() and "$0.0070" not in local(pg).inner_text()      # no real figure in it
        helper_style = helper.evaluate("e => getComputedStyle(e).backgroundColor")
        local_style = local(pg).evaluate("e => getComputedStyle(e).backgroundColor")
        assert helper_style != local_style
        assert local(pg).evaluate("e => getComputedStyle(e).borderTopStyle") == "dashed"
    ctx.close()


def test_the_helper_meter_text_is_the_same_with_and_without_local_logs(root, tmp_path, browser):
    cfg, d, tel = setup(tmp_path, max_calls=10, max_calls_per_problem=9)
    fill(tel, 1)
    empty = tmp_path / "empty-runs"
    empty.mkdir()
    texts = []
    for runs in (root, empty):
        ctx, pg = new_page(browser)
        with Served(app_for(runs, telegraph=cfg)) as s:
            pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")
            expect(pg.locator("#spend-meter")).to_contain_text("$0.0035 · 1 call")
            texts.append(pg.inner_text("#spend-meter"))
        ctx.close()
    assert texts[0] == texts[1]


def test_the_view_has_a_table_with_a_column_per_reference_and_the_rows(root, browser):
    cfg = vs.load({"extra": [{"name": "claude-sonnet-5-5", "price_in": 2.0, "price_out": 10.0}]})
    ctx, pg = new_page(browser)
    with Served(app_for(root, vcfg=cfg)) as s:
        open_page(pg, s)
        local(pg).click()
        expect(pg.locator("#telegraph-view")).to_be_visible()
        head = pg.locator("#tg-local-head th")
        expect(head).to_have_text(["Run or probe directory", "Turns", "Prompt tokens", "Completion tokens", "Agent time",
                                   "Virtual $ at claude-haiku-5-5", "Virtual $ at claude-sonnet-5-5"])
        rows = pg.locator("#tg-local-body tr")
        expect(rows).to_have_count(2)
        expect(rows.nth(0)).to_contain_text("kprobe/rms5")
        expect(rows.nth(0)).to_contain_text("$0.20")
        expect(rows.nth(0)).to_contain_text("$4.00")                    # 2,000,000 prompt tokens at $2 per million
        expect(rows.nth(1)).to_contain_text("org--a")
        expect(rows.nth(1)).to_contain_text("not recorded")
        expect(pg.locator("#tg-local-note")).to_contain_text("never added to the real helper spend")
        expect(pg.locator("#tg-local-note")).to_contain_text("prompt caching")
        expect(pg.locator("#tg-local-h")).to_contain_text("never added to the helper spend above")
        text = pg.inner_text("#telegraph-view")
        assert str(root) not in text and "orchard-runs" not in text
    ctx.close()


def test_the_page_sends_nothing_but_gets_and_reads_the_local_endpoint(root, browser):
    ctx, pg = new_page(browser)
    seen = []
    pg.on("request", lambda r: seen.append((r.method, r.url)))
    with Served(app_for(root)) as s:
        open_page(pg, s)
        local(pg).click()
        pg.wait_for_timeout(1500)
        base = f"http://127.0.0.1:{s.port}/"
        assert all(url.startswith(base) for _, url in seen), seen
        assert all(m == "GET" for m, _ in seen), seen
        assert any("/api/virtual-spend" in url for _, url in seen)
    ctx.close()


def test_the_figure_is_hidden_during_an_outage_and_comes_back(root, browser):
    ctx, pg = new_page(browser)
    with Served(app_for(root)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")
        expect(local(pg)).to_have_text(FIRST)
        pg.route("**/api/virtual-spend*", lambda route: route.abort())
        expect(local(pg)).to_be_hidden(timeout=40000)
        pg.unroute("**/api/virtual-spend*")
        expect(local(pg)).to_have_text(FIRST, timeout=40000)
    ctx.close()


def test_a_failing_local_endpoint_does_not_touch_the_helper_meter(root, tmp_path, browser):
    cfg, d, tel = setup(tmp_path, max_calls=10, max_calls_per_problem=9)
    fill(tel, 1)
    ctx, pg = new_page(browser)
    with Served(app_for(root, telegraph=cfg)) as s:
        pg.route("**/api/virtual-spend*", lambda route: route.fulfill(status=500, body="{}", content_type="application/json"))
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")
        expect(pg.locator("#spend-meter")).to_contain_text("$0.0035 · 1 call")
        pg.wait_for_timeout(2500)
        expect(local(pg)).to_be_hidden()
        expect(pg.locator("#spend-meter")).to_contain_text("$0.0035 · 1 call")
        assert pg.locator("#spend-meter").get_attribute("data-state") == "ok"
    ctx.close()


def test_the_figure_follows_new_logs_on_a_later_poll(root, browser):
    ctx, pg = new_page(browser)
    with Served(app_for(root)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")
        expect(local(pg)).to_have_text(FIRST)
        write_log(root, "org--a/stages/2/log/run-2.jsonl", [turn(1, 10_000_000, 0)])
        expect(local(pg)).to_have_text("Local ≈ $1.15 (virtual, at claude-haiku-5-5)", timeout=60000)      # after the 30 s cache
    ctx.close()


@pytest.mark.parametrize("mode", ["operator", "playful"])
@pytest.mark.parametrize("width", [1440, 820, 400])
def test_no_horizontal_page_scroll_with_the_figure_and_a_capped_helper_meter(root, tmp_path, browser, mode, width):
    cfg, d, tel = setup(tmp_path, max_calls=2, max_calls_per_problem=9)
    fill(tel, 2)
    ctx, pg = new_page(browser, width)
    with Served(app_for(root, telegraph=cfg)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode={mode}")
        expect(local(pg)).to_be_visible()
        assert pg.evaluate("document.documentElement.scrollWidth") <= width
        local(pg).click()
        expect(pg.locator("#tg-local-body tr").first).to_be_visible()
        assert pg.evaluate("document.documentElement.scrollWidth") <= width
        expect(local(pg)).to_be_in_viewport()
        expect(pg.locator("#spend-meter")).to_be_in_viewport()
    ctx.close()



def test_the_hover_text_says_the_figure_is_virtual_and_not_part_of_the_helper_spend(root, browser):
    ctx, pg = new_page(browser)
    with Served(app_for(root)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")
        expect(local(pg)).to_have_text(FIRST)
        title = local(pg).get_attribute("title")
        assert "Virtual spend" in title and "Nobody was charged" in title and "not part of the helper spend" in title
    ctx.close()



@pytest.mark.parametrize("width", [1400, 820, 400])
@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_both_meters_stay_on_screen_at_the_bottom_of_a_tall_page(tmp_path, browser, mode, width):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    for i in range(45):
        write_log(r, f"kprobe/probe-{i:02d}/op/attempt-1/log/p.jsonl", [turn(1, 1_000_000 + i, 10)])
    ctx, pg = new_page(browser, width)
    with Served(app_for(r)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode={mode}#telegraph")
        expect(pg.locator("#tg-local-body tr")).to_have_count(45)
        for _ in range(50):                                              # the page grows as its parts render: wait until it is tall
            if pg.evaluate("document.body.scrollHeight") > 1500:
                break
            pg.wait_for_timeout(100)
        pg.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'instant'})")
        assert pg.evaluate("window.scrollY") > 500, "the page must be taller than the window for this test to mean anything"
        expect(local(pg)).to_be_in_viewport()
        expect(pg.locator("#spend-meter")).to_be_in_viewport()
        a, b = local(pg).bounding_box(), pg.locator("#spend-meter").bounding_box()
        apart_y = a["y"] + a["height"] <= b["y"] + 1 or b["y"] + b["height"] <= a["y"] + 1
        apart_x = a["x"] + a["width"] <= b["x"] + 1 or b["x"] + b["width"] <= a["x"] + 1
        assert apart_x or apart_y, "the two meters must not overlap"
        if width <= 900:
            assert b["y"] + b["height"] <= a["y"] + 1, "pinned along the bottom, the helper meter sits above the local figure"
        else:
            assert abs(a["y"] - b["y"]) < 40, "in the top bar the two figures sit side by side"
        if width <= 900:                                                 # pinned along the bottom edge: nothing may hide under them
            top_of_meters = min(a["y"], b["y"])
            last = pg.evaluate("(() => { const r = document.querySelector('#tg-local-body tr:last-child').getBoundingClientRect(); return r.bottom; })()")
            assert last <= top_of_meters + 1, f"the last table row ends at {last}, under the meters at {top_of_meters}"
    ctx.close()


def test_the_page_names_the_runs_that_were_not_scanned(tmp_path, browser, monkeypatch):
    from orchard import virtualspend_view as view
    monkeypatch.setattr(view, "MAX_CANDIDATES", 2)
    r = tmp_path / "orchard-runs"
    for i in range(5):
        write_log(r, f"kprobe/p{i}/op/attempt-1/log/p.jsonl", [turn(1, 1_000_000 * (i + 1), 0)])
    ctx, pg = new_page(browser)
    with Served(app_for(r)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator#telegraph")
        expect(pg.locator("#tg-local-note")).to_contain_text("3 runs not scanned")
        expect(local(pg)).to_have_attribute("title", __import__("re").compile("3 runs were not scanned"))
    ctx.close()



def test_the_figure_expires_even_when_a_request_never_answers(root, browser):
    ctx, pg = new_page(browser)
    with Served(app_for(root)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")
        expect(local(pg)).to_have_text(FIRST)
        pg.route("**/api/virtual-spend*", lambda route: None)           # every later request stays pending: no response, no failure
        expect(local(pg)).to_be_hidden(timeout=30000)
        pg.unroute("**/api/virtual-spend*")
    ctx.close()


def test_a_fresh_answer_keeps_the_figure_beyond_the_expiry_time(root, browser):
    ctx, pg = new_page(browser)
    with Served(app_for(root)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")
        expect(local(pg)).to_have_text(FIRST)
        pg.wait_for_timeout(20000)                                       # longer than the 15 s expiry; polls keep renewing it
        expect(local(pg)).to_have_text(FIRST)
    ctx.close()


def test_the_page_says_the_folder_listing_was_cut_short(tmp_path, browser, monkeypatch):
    from orchard import virtualspend_view as view
    monkeypatch.setattr(view, "MAX_NAMES", 3)
    r = tmp_path / "orchard-runs"
    for i in range(8):
        write_log(r, f"org--{i}/log/run.jsonl", [turn(1, 1_000_000, 0)])
    ctx, pg = new_page(browser)
    with Served(app_for(r)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator#telegraph")
        expect(pg.locator("#tg-local-note")).to_contain_text("The folder listing was cut short")
    ctx.close()


def test_the_page_says_how_many_runs_the_table_leaves_out(tmp_path, browser, monkeypatch):
    from orchard import virtualspend_view as view
    monkeypatch.setattr(view, "MAX_ROWS", 2)
    r = tmp_path / "orchard-runs"
    for i in range(5):
        write_log(r, f"kprobe/p{i}/op/attempt-1/log/p.jsonl", [turn(1, 1_000_000 * (i + 1), 0)])
    ctx, pg = new_page(browser)
    with Served(app_for(r)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator#telegraph")
        expect(pg.locator("#tg-local-body tr")).to_have_count(2)
        expect(pg.locator("#tg-local-note")).to_contain_text("3 more runs not shown in this table")
        expect(pg.locator("#tg-local-note")).not_to_contain_text("not scanned")      # every directory was scanned
        expect(local(pg)).to_have_text("Local ≈ $1.50 (virtual, at claude-haiku-5-5)")   # the meter still counts all five
    ctx.close()


def test_the_note_follows_a_change_in_the_hidden_rows(tmp_path, browser, monkeypatch):
    from orchard import virtualspend_view as view
    monkeypatch.setattr(view, "MAX_ROWS", 2)
    monkeypatch.setattr(view, "CACHE_S", 1.0)
    r = tmp_path / "orchard-runs"
    for i in range(3):
        write_log(r, f"kprobe/p{i}/op/attempt-1/log/p.jsonl", [turn(1, 1_000_000 * (i + 1), 0)])
    ctx, pg = new_page(browser)
    with Served(app_for(r)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator#telegraph")
        expect(pg.locator("#tg-local-note")).to_contain_text("1 more run not shown in this table")
        write_log(r, "kprobe/p9/op/attempt-1/log/p.jsonl", [turn(1, 100, 0)])        # ranks below the shown rows: only the note changes
        expect(pg.locator("#tg-local-note")).to_contain_text("2 more runs not shown in this table", timeout=30000)
    ctx.close()


def test_the_hidden_rows_note_does_not_claim_the_meter_total_for_a_selected_run(tmp_path, browser, monkeypatch):
    from orchard import virtualspend_view as view
    monkeypatch.setattr(view, "MAX_ROWS", 2)
    r = tmp_path / "orchard-runs"
    write_ledger(r / "org--sel", ("stage_start", 0, {}), model="org/sel")
    write_log(r, "org--sel/log/run.jsonl", [turn(1, 1_000_000, 0)])
    for i in range(4):
        write_log(r, f"kprobe/p{i}/op/attempt-1/log/p.jsonl", [turn(1, 2_000_000 + i, 0)])
    ctx, pg = new_page(browser)
    with Served(app_for(r)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")                       # the page selects the first run
        expect(local(pg)).to_have_text("Local ≈ $0.10 (virtual, at claude-haiku-5-5)")      # the selected run only
        pg.click("#view-telegraph-btn")
        expect(pg.locator("#tg-local-note")).to_contain_text("more runs not shown in this table")
        expect(pg.locator("#tg-local-note")).not_to_contain_text("in the meter's total")
        expect(pg.locator("#tg-local-note")).to_contain_text("the meter shows the selected run only")
    ctx.close()


def test_the_hidden_rows_note_claims_the_meter_total_when_the_meter_covers_every_run(tmp_path, browser, monkeypatch):
    from orchard import virtualspend_view as view
    monkeypatch.setattr(view, "MAX_ROWS", 2)
    r = tmp_path / "orchard-runs"
    for i in range(4):
        write_log(r, f"kprobe/p{i}/op/attempt-1/log/p.jsonl", [turn(1, 1_000_000, 0)])
    ctx, pg = new_page(browser)
    with Served(app_for(r)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator#telegraph")
        expect(pg.locator("#tg-local-note")).to_contain_text("2 more runs not shown in this table; their tokens are in the meter's total")
        expect(local(pg)).to_have_text("Local ≈ $0.40 (virtual, at claude-haiku-5-5)")
    ctx.close()


def test_the_page_marks_a_run_that_was_not_fully_read_and_says_so(tmp_path, browser, monkeypatch):
    from orchard import virtualspend as virtual
    monkeypatch.setattr(virtual, "MAX_DEPTH", 2)
    r = tmp_path / "orchard-runs"
    write_log(r, "kprobe/p0/log/run.jsonl", [turn(1, 1_000_000, 0)])
    write_log(r, "kprobe/p1/x/y/log/run.jsonl", [turn(1, 5, 0)])             # below the depth limit
    write_log(r, "kprobe/p1/log/run.jsonl", [turn(1, 1_000_000, 0)])
    ctx, pg = new_page(browser)
    with Served(app_for(r)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator#telegraph")
        expect(pg.locator("#tg-local-note")).to_contain_text("1 run not fully read")
        expect(pg.locator("#tg-local-body tr", has_text="kprobe/p1")).to_contain_text("(not fully read)")
        expect(pg.locator("#tg-local-body tr", has_text="kprobe/p0")).not_to_contain_text("(not fully read)")
        expect(local(pg)).to_have_attribute("title", __import__("re").compile("was not fully read"))
    ctx.close()


def test_the_page_says_the_runs_folder_cannot_be_read_instead_of_listing_nothing(tmp_path, browser):
    ctx, pg = new_page(browser)
    with Served(app_for(tmp_path / "gone")) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator#telegraph")
        expect(pg.locator("#tg-local-note")).to_contain_text("The runs folder cannot be read")
        expect(pg.locator("#tg-local-empty")).to_be_hidden()
    ctx.close()


def test_the_page_says_a_run_failed_when_a_price_overflows_its_cost(tmp_path, browser):
    r = tmp_path / "orchard-runs"
    write_log(r, "kprobe/big/op/attempt-1/log/p.jsonl", [turn(1, 1_000_000, 0)])
    write_log(r, "kprobe/small/op/attempt-1/log/p.jsonl", [turn(1, 10, 0)])
    ctx, pg = new_page(browser)
    with Served(app_for(r, vcfg=vs.load({"price_in": 1e307}))) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator#telegraph")
        expect(pg.locator("#tg-local-note")).to_contain_text("1 run could not be read")
        expect(pg.locator("#tg-local-body tr")).to_have_count(1)
        text = pg.inner_text("#telegraph-view")
        assert "Infinity" not in text and "NaN" not in text
    ctx.close()
