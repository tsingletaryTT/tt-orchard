# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixes for the third Copilot review of PR 111 that need a browser (installed Chrome through Playwright)."""
import dataclasses

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

from test_telegraph_view import app_for, setup
from test_webui_telegraph_browser import CHROME, browser, meter, new_page, open_page, root  # noqa: F401
from test_webui_telegraph_review_browser import capped_app
from test_webui import Served

pytestmark = pytest.mark.skipif(not CHROME, reason="no Chrome or Chromium on this machine")


def unknown_cost_cfg(tmp_path, **replace):
    cfg, d, tel = setup(tmp_path, prices={"small": [1.0, 5.0]}, max_calls=9, max_calls_per_problem=9)
    first = tel.ask("p", "task " * 80, "failed")
    tel.report(first.call_id, helped=False)
    tel.ask("p", "task two " * 80, "failed")                      # the second model has no price
    return dataclasses.replace(cfg, **replace)


# ---- 3. the last amount stays visible, marked as old, after an outage ---------------------------------------------------------

@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_after_an_outage_the_meter_says_unknown_and_shows_the_last_known_amount(root, tmp_path, browser, mode):
    ctx, pg = new_page(browser)
    with Served(capped_app(root, tmp_path)) as s:
        open_page(pg, s, mode)
        expect(meter(pg)).to_contain_text("$0.0070")
        pg.route("**/api/**", lambda route: route.abort())
        expect(meter(pg)).to_contain_text("spend unknown", timeout=40000)
        expect(meter(pg)).to_contain_text("last known $0.0070 · 2 calls")
        expect(meter(pg)).to_contain_text("s ago")
        assert meter(pg).get_attribute("data-tone") == "bad"
        pg.unroute("**/api/**")
        expect(meter(pg)).not_to_contain_text("spend unknown", timeout=40000)
    ctx.close()


@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_the_telegraph_view_says_its_numbers_are_old_during_an_outage_and_clears_the_note_after(root, tmp_path, browser, mode):
    ctx, pg = new_page(browser)
    with Served(capped_app(root, tmp_path)) as s:
        open_page(pg, s, mode)
        meter(pg).click()
        expect(pg.locator("#tg-numbers")).to_contain_text("$0.0070")
        assert pg.locator("#tg-stale").is_hidden()
        pg.route("**/api/telegraph", lambda route: route.abort())
        expect(pg.locator("#tg-stale")).to_be_visible(timeout=40000)
        expect(pg.locator("#tg-stale")).to_contain_text("cannot read the helper's spend right now")
        expect(pg.locator("#tg-stale")).to_contain_text("s ago and may be out of date")
        assert pg.locator("#telegraph-view").get_attribute("data-stale") == "true"
        pg.unroute("**/api/telegraph")
        expect(pg.locator("#tg-stale")).to_be_hidden(timeout=40000)
        assert pg.locator("#telegraph-view").get_attribute("data-stale") is None
    ctx.close()


def test_with_no_amount_ever_read_the_meter_does_not_invent_a_last_known_one(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    with Served(capped_app(root, tmp_path)) as s:
        pg.route("**/api/telegraph", lambda route: route.abort())
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")
        expect(meter(pg)).to_contain_text("spend unknown", timeout=40000)
        assert "last known" not in meter(pg).inner_text()
    ctx.close()


# ---- 4. a per-problem cost cap is named in the unknown-cost messages -----------------------------------------------------------

def banner_text(browser, root, cfg):
    ctx, pg = new_page(browser)
    with Served(app_for(root, cfg)) as s:
        open_page(pg, s)
        meter(pg).click()
        expect(pg.locator("#tg-banner")).to_contain_text("has no known cost")
        out = pg.inner_text("#tg-banner") + " | " + pg.inner_text("#tg-numbers")
    ctx.close()
    return out


def test_only_a_per_problem_cost_cap_is_named_as_the_thing_that_refuses_calls(root, tmp_path, browser):
    text = banner_text(browser, root, unknown_cost_cfg(tmp_path, max_cost_per_problem_usd=100.0))
    assert "per-problem cost cap refuses new calls for a problem while it has any" in text
    assert "No cost cap is set" not in text and "no cost cap is set" not in text


def test_a_global_cost_cap_is_still_named(root, tmp_path, browser):
    text = banner_text(browser, root, unknown_cost_cfg(tmp_path, max_cost_usd=100.0))
    assert "A cost cap refuses new calls while any exist." in text


def test_with_no_cost_cap_the_page_says_calls_still_go_out(root, tmp_path, browser):
    text = banner_text(browser, root, unknown_cost_cfg(tmp_path))
    assert "No cost cap is set, so new calls still go out." in text and "no cost cap is set, so calls still go out" in text


# ---- 5. polling does not wait for the metadata request -------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_the_meter_is_polled_even_when_the_metadata_request_never_answers(root, tmp_path, browser, mode):
    ctx, pg = new_page(browser)
    with Served(capped_app(root, tmp_path)) as s:
        pg.route("**/api/meta", lambda route: None)
        pg.goto(f"http://127.0.0.1:{s.port}/?mode={mode}")
        expect(meter(pg)).to_contain_text("$0.0070")
        pg.route("**/api/telegraph", lambda route: None)
        expect(meter(pg)).to_contain_text("spend unknown", timeout=23000)
    ctx.close()
