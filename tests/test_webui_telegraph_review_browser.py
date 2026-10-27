# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixes for the first Copilot review of PR 111 that need a browser: the meter stays on screen while the page scrolls at phone and
tablet widths, and a poll that never answers cannot leave an old dollar amount on screen."""
import pytest

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

from test_telegraph_view import app_for, fill, setup
from test_webui_telegraph_browser import CHROME, browser, meter, new_page, open_page, root  # noqa: F401
from test_webui import Served

pytestmark = pytest.mark.skipif(not CHROME, reason="no Chrome or Chromium on this machine")


def capped_app(root, tmp_path):
    cfg, d, tel = setup(tmp_path, max_calls=2, max_calls_per_problem=9)
    fill(tel, 2)
    return app_for(root, cfg)


def on_screen(pg):
    box = meter(pg).bounding_box()
    vh = pg.evaluate("window.innerHeight")
    return box is not None and box["y"] >= -1 and box["y"] + box["height"] <= vh + 1


@pytest.mark.parametrize("mode", ["operator", "playful"])
@pytest.mark.parametrize("width", [400, 820])
def test_the_meter_stays_on_screen_while_scrolling_the_telegraph_view_on_a_phone_and_a_tablet(root, tmp_path, browser, mode, width):
    ctx, pg = new_page(browser, width)
    with Served(capped_app(root, tmp_path)) as s:
        open_page(pg, s, mode)
        meter(pg).click()
        expect(pg.locator("#telegraph-view")).to_be_visible()
        height = pg.evaluate("document.documentElement.scrollHeight")
        assert height > 2 * pg.evaluate("window.innerHeight"), "the page must be long enough to scroll"
        for y in (0, height // 3, 2 * height // 3, height):
            pg.evaluate(f"window.scrollTo(0, {y})")
            pg.wait_for_timeout(150)
            assert on_screen(pg), (width, mode, y, meter(pg).bounding_box())
        assert pg.evaluate("document.documentElement.scrollWidth") <= width
    ctx.close()


@pytest.mark.parametrize("mode", ["operator", "playful"])
@pytest.mark.parametrize("width", [400, 820])
def test_the_pinned_meter_does_not_cover_the_end_of_the_page(root, tmp_path, browser, mode, width):
    ctx, pg = new_page(browser, width)
    with Served(capped_app(root, tmp_path)) as s:
        open_page(pg, s, mode)
        meter(pg).click()
        pg.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        pg.wait_for_timeout(200)
        last = pg.locator("body").evaluate("b => { const els = [...b.querySelectorAll('.machine, .machine *')].filter(e => e.offsetParent); "
                                           "return Math.max(...els.map(e => e.getBoundingClientRect().bottom)); }")
        assert last <= meter(pg).bounding_box()["y"] + 1, (last, meter(pg).bounding_box())
    ctx.close()


@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_the_meter_is_still_in_the_top_bar_on_a_wide_screen_and_sticks_there(root, tmp_path, browser, mode):
    ctx, pg = new_page(browser, 1440)
    with Served(capped_app(root, tmp_path)) as s:
        open_page(pg, s, mode)
        meter(pg).click()
        pg.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        pg.wait_for_timeout(150)
        box = meter(pg).bounding_box()
        assert box["y"] >= 0 and box["y"] < 140
    ctx.close()


def test_a_poll_that_never_answers_marks_the_amount_unknown_without_waiting_for_it_to_fail(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    with Served(capped_app(root, tmp_path)) as s:
        open_page(pg, s)
        expect(meter(pg)).to_contain_text("$0.0070")
        pg.route("**/api/telegraph", lambda route: None)           # the request stays pending: no answer and no failure
        expect(meter(pg)).to_contain_text("spend unknown", timeout=23000)
        assert meter(pg).get_attribute("data-tone") == "bad"
    ctx.close()


def test_a_poll_that_hung_does_not_block_later_polls_once_the_server_answers_again(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    with Served(capped_app(root, tmp_path)) as s:
        open_page(pg, s)
        pg.route("**/api/telegraph", lambda route: None)
        expect(meter(pg)).to_contain_text("spend unknown", timeout=23000)
        pg.unroute("**/api/telegraph")
        expect(meter(pg)).to_contain_text("$0.0070", timeout=45000)
    ctx.close()
