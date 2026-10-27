# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The spend meter and the Telegraph view in a real browser (installed Chrome through Playwright).

Ledgers are written by the real `telegraph.Telegraph` with a fake transport, and the server is the real one on a loopback
port, so these tests show what a person sees. They skip when `playwright` or Chrome is missing (see
tests/test_webui_browser.py for how to run them). No request ever leaves the machine: one test records every request the page
makes and checks that only GET /api/... calls to the page's own server were sent."""
import os
import shutil

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

from orchard import telegraph as tg
from test_telegraph_view import KEY, app_for, fill, setup
from test_webui import Served, write_ledger, with_supervisor

CHROME = os.environ.get("ORCHARD_CHROME") or shutil.which("google-chrome") or shutil.which("chromium")
pytestmark = pytest.mark.skipif(not CHROME, reason="no Chrome or Chromium on this machine")


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    write_ledger(r / "org--live", ("stage_start", 0, {}), model="org/live-model")
    return r


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox"])
        yield b
        b.close()


def new_page(browser, width=1400, mode="operator"):
    ctx = browser.new_context(viewport={"width": width, "height": 900})
    pg = ctx.new_page()
    pg.set_default_timeout(15000)
    return ctx, pg


def open_page(pg, served, mode="operator"):
    pg.goto(f"http://127.0.0.1:{served.port}/?mode={mode}")
    pg.wait_for_selector("[data-run-row]:visible")


def meter(pg):
    return pg.locator("#spend-meter")


def app_in_state(root, tmp_path, state):
    if state == "off":
        return app_for(root, tg.Config())
    kw = {"ok": dict(max_calls=10), "near": dict(max_calls=5), "capped": dict(max_calls=2)}[state]
    cfg, d, tel = setup(tmp_path, max_calls_per_problem=9, **kw)
    fill(tel, {"ok": 1, "near": 4, "capped": 2}[state])
    return app_for(root, cfg)


@pytest.mark.parametrize("state,tone,words", [
    ("off", "dim", "Helper: off"),
    ("ok", "good", "$0.0035 · 1 call"),
    ("near", "warn", "near cap: calls 4 of 5"),
    ("capped", "bad", "CAPPED: the call cap is reached (2 of 2)"),
])
@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_the_meter_shows_spend_and_state_in_both_styles(root, tmp_path, browser, state, tone, words, mode):
    ctx, pg = new_page(browser)
    with Served(app_in_state(root, tmp_path, state)) as s:
        open_page(pg, s, mode)
        expect(meter(pg)).to_contain_text(words)
        assert meter(pg).get_attribute("data-tone") == tone and meter(pg).get_attribute("data-state") == state
    ctx.close()


def test_the_meter_is_on_screen_in_every_view_and_after_scrolling(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    with Served(app_in_state(root, tmp_path, "near")) as s:
        open_page(pg, s)
        for view in ("view-hardware-btn", "view-settings-btn", "view-telegraph-btn", "view-orchard-btn"):
            pg.click(f"#{view}")
            expect(meter(pg)).to_be_in_viewport()
            expect(meter(pg)).to_contain_text("$")
        pg.click("#view-telegraph-btn")
        pg.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        expect(meter(pg)).to_be_in_viewport()                                # the top bar sticks
    ctx.close()


@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_the_meter_opens_the_view_with_the_numbers_the_meter_shows(root, tmp_path, browser, mode):
    ctx, pg = new_page(browser)
    with Served(app_in_state(root, tmp_path, "capped")) as s:
        open_page(pg, s, mode)
        meter(pg).click()
        view = pg.locator("#telegraph-view")
        expect(view).to_be_visible()
        expect(pg.locator("#tg-banner")).to_contain_text("a cap has stopped it")
        expect(pg.locator("#tg-banner")).to_contain_text("the call cap is reached (2 of 2)")
        expect(pg.locator("#tg-recent tr")).to_have_count(2)
        expect(pg.locator("#tg-config")).to_contain_text("llm.example.org")
        text = pg.inner_text("#telegraph-view")
        assert KEY not in text and "TG_KEY" not in text
        assert pg.evaluate("location.hash") == "#telegraph"
    ctx.close()


def test_a_link_to_the_view_opens_it(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    with Served(app_in_state(root, tmp_path, "ok")) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator#telegraph")
        expect(pg.locator("#telegraph-view")).to_be_visible()
    ctx.close()


def test_a_refused_call_shows_its_reason_in_the_table(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    cfg, d, tel = setup(tmp_path, max_calls=1, max_calls_per_problem=9)
    fill(tel, 1)
    tel.ask("later", "another task " * 80, "failed")
    with Served(app_for(root, cfg)) as s:
        open_page(pg, s)
        meter(pg).click()
        row = pg.locator("#tg-recent tr").first
        expect(row).to_contain_text("budget")
        expect(row).to_contain_text("the call cap is reached (1 of 1)")
    ctx.close()


def test_a_telegraph_that_is_off_explains_how_to_turn_it_on(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    with Served(app_in_state(root, tmp_path, "off")) as s:
        open_page(pg, s)
        meter(pg).click()
        expect(pg.locator("#tg-banner")).to_contain_text("Telegraph is off")
        expect(pg.locator(".tg-how")).to_contain_text("5.9g")
        expect(pg.locator("#tg-recent-empty")).to_be_visible()
    ctx.close()


def test_the_meter_says_unknown_during_an_outage_and_recovers(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    with Served(app_in_state(root, tmp_path, "ok")) as s:
        open_page(pg, s)
        expect(meter(pg)).to_contain_text("$0.0035")
        pg.route("**/api/**", lambda route: route.abort())
        expect(meter(pg)).to_contain_text("spend unknown", timeout=40000)
        assert meter(pg).get_attribute("data-tone") == "bad"
        pg.unroute("**/api/**")
        expect(meter(pg)).to_contain_text("$0.0035", timeout=40000)
    ctx.close()


def test_one_failing_telegraph_answer_does_not_leave_an_old_amount_on_screen(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    with Served(app_in_state(root, tmp_path, "ok")) as s:
        open_page(pg, s)
        pg.route("**/api/telegraph", lambda route: route.fulfill(status=500, body="{}", content_type="application/json"))
        expect(meter(pg)).to_contain_text("spend unknown", timeout=40000)
    ctx.close()


def test_the_meter_follows_the_ledger_on_the_next_poll(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    cfg, d, tel = setup(tmp_path, max_calls=10, max_calls_per_problem=9)
    with Served(app_for(root, cfg)) as s:
        open_page(pg, s)
        expect(meter(pg)).to_contain_text("$0.00 · 0 calls")
        fill(tel, 2)
        expect(meter(pg)).to_contain_text("$0.0070 · 2 calls", timeout=15000)
    ctx.close()


def test_the_page_sends_nothing_but_gets_to_its_own_server(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    seen = []
    pg.on("request", lambda r: seen.append((r.method, r.url)))
    with Served(app_in_state(root, tmp_path, "near")) as s:
        open_page(pg, s)
        meter(pg).click()
        pg.wait_for_timeout(1500)
        base = f"http://127.0.0.1:{s.port}/"
        assert all(url.startswith(base) for _, url in seen), seen
        assert all(m == "GET" for m, _ in seen), seen
        assert any(url.endswith("/api/telegraph") for _, url in seen)
    ctx.close()


@pytest.mark.parametrize("mode", ["operator", "playful"])
@pytest.mark.parametrize("width", [1440, 820, 400])
@pytest.mark.parametrize("state", ["off", "capped"])
def test_no_horizontal_page_scroll_on_the_view_at_three_widths(root, tmp_path, browser, mode, width, state):
    ctx, pg = new_page(browser, width)
    with Served(app_in_state(root, tmp_path, state)) as s:
        open_page(pg, s, mode)
        meter(pg).click()
        expect(pg.locator("#telegraph-view")).to_be_visible()
        assert pg.evaluate("document.documentElement.scrollWidth") <= width
        expect(meter(pg)).to_be_visible()
    ctx.close()


def test_a_run_with_its_own_ledger_shows_its_spend_in_the_header(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    run = root / "org--live"
    cfg, d, _ = setup(tmp_path)
    from test_telegraph_view import Clock, reply
    tg.Telegraph(cfg, run, transport=reply(), environ={"TG_KEY": KEY}, clock=Clock()).ask("p", "task " * 80, "failed")
    with Served(app_for(root, cfg)) as s:
        open_page(pg, s)
        expect(pg.locator("#run-spend")).to_be_visible()
        expect(pg.locator("#run-spend")).to_contain_text("Helper spend for this run: $0.0035 · 1 call")
    ctx.close()
