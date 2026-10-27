# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What a person sees for the reservation ledger: unsettled reservations, calls with no known cost, a cost cap that refuses new
calls while a cost is unknown, and a refused call's reason. Real Telegraph ledgers, real server, installed Chrome."""
import dataclasses
import re

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

from test_telegraph_view import app_for, setup
from test_telegraph_view_ledger import crash_after_reserve
from test_webui_telegraph_browser import CHROME, browser, meter, new_page, open_page, root  # noqa: F401
from test_webui import Served

pytestmark = pytest.mark.skipif(not CHROME, reason="no Chrome or Chromium on this machine")


def unknown_cost_app(root, tmp_path, cost_cap):
    cfg, d, tel = setup(tmp_path, prices={"small": [1.0, 5.0]}, max_calls=9, max_calls_per_problem=9)
    first = tel.ask("p", "task " * 80, "failed")
    tel.report(first.call_id, helped=False)
    tel.ask("p", "task two " * 80, "failed")                       # unpriced second model
    return app_for(root, dataclasses.replace(cfg, max_cost_usd=100.0) if cost_cap else cfg)


@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_an_unknown_cost_is_counted_and_the_cost_cap_block_is_explained(root, tmp_path, browser, mode):
    ctx, pg = new_page(browser)
    with Served(unknown_cost_app(root, tmp_path, True)) as s:
        open_page(pg, s, mode)
        expect(meter(pg)).to_contain_text("$0.0035 + 1 unknown")
        expect(meter(pg)).to_contain_text("CAPPED")
        assert meter(pg).get_attribute("data-tone") == "bad"
        meter(pg).click()
        banner = pg.locator("#tg-banner")
        expect(banner).to_contain_text("1 call has no known cost. A cost cap refuses new calls while any exist.")
        expect(pg.locator("#tg-caps")).to_contain_text("blocked, a call has no known cost")
        expect(pg.locator("#tg-numbers")).to_contain_text("Unknown cost")
        text = pg.inner_text("#tg-by-model")
        assert "unknown cost" in text and "$0.00 " not in text.replace("$0.0035", "")
    ctx.close()


def test_without_a_cost_cap_the_page_says_calls_still_go_out(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    with Served(unknown_cost_app(root, tmp_path, False)) as s:
        open_page(pg, s)
        meter(pg).click()
        expect(pg.locator("#tg-banner")).to_contain_text("No cost cap is set, so new calls still go out.")
        assert meter(pg).get_attribute("data-tone") == "good"
    ctx.close()


def test_a_call_with_no_price_is_never_shown_as_zero_dollars_anywhere(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    cfg, d, tel = setup(tmp_path, prices={})
    tel.ask("p", "task " * 80, "failed")
    with Served(app_for(root, cfg)) as s:
        open_page(pg, s)
        expect(meter(pg)).to_contain_text("Helper: unknown cost · 1 call")
        meter(pg).click()
        expect(pg.locator("#tg-recent tr")).to_have_count(1)
        view = pg.inner_text("#telegraph-view")
        assert not re.search(r"\$0\.00(?!\d)", view), view
        expect(pg.locator("#tg-recent")).to_contain_text("unknown")
    ctx.close()


@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_an_unsettled_reservation_is_a_row_and_a_note_and_counts_toward_the_caps(root, tmp_path, browser, mode):
    ctx, pg = new_page(browser)
    cfg, d, tel = setup(tmp_path, max_calls=5)
    crash_after_reserve(tel)
    with Served(app_for(root, cfg)) as s:
        open_page(pg, s, mode)
        expect(meter(pg)).to_contain_text("1 call")
        meter(pg).click()
        expect(pg.locator("#tg-recent tr")).to_have_count(1)
        row = pg.locator("#tg-recent tr").first
        expect(row).to_contain_text("unsettled")
        expect(row).to_contain_text("up to $")
        expect(pg.locator("#tg-banner")).to_contain_text("1 reservation has no recorded result")
        expect(pg.locator("#tg-numbers")).to_contain_text("Unsettled")
    ctx.close()


def test_a_refused_call_shows_its_reason_and_a_torn_record_is_not_a_row(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    cfg, d, tel = setup(tmp_path, max_calls=2, max_calls_per_problem=9)
    d.mkdir(parents=True)
    (d / "telegraph.jsonl").write_bytes(b'{"call_id": "ab12", "problem": "p", "outcome": "ok"')
    tel.ask("p2", "task " * 80, "failed")
    refused = tel.ask("p3", "task three " * 80, "failed")
    assert not refused.ok
    with Served(app_for(root, cfg)) as s:
        open_page(pg, s)
        meter(pg).click()
        rows = pg.locator("#tg-recent tr")
        expect(rows).to_have_count(2)                              # the refusal and the one real call
        expect(rows.first).to_contain_text("the call cap is reached (2 of 2)")
        expect(pg.locator("#tg-numbers")).to_contain_text("1 torn record")
    ctx.close()


@pytest.mark.parametrize("mode", ["operator", "playful"])
@pytest.mark.parametrize("width", [1440, 820, 400])
def test_no_page_scroll_with_unknown_costs_and_unsettled_rows(root, tmp_path, browser, mode, width):
    ctx, pg = new_page(browser, width)
    cfg, d, tel = setup(tmp_path, prices={"small": [1.0, 5.0]}, max_calls=9, max_calls_per_problem=9)
    first = tel.ask("p", "task " * 80, "failed")
    tel.report(first.call_id, helped=False)
    tel.ask("p", "task two " * 80, "failed")
    crash_after_reserve(tel, "z")
    app = app_for(root, dataclasses.replace(cfg, max_cost_usd=100.0))
    with Served(app) as s:
        open_page(pg, s, mode)
        meter(pg).click()
        expect(pg.locator("#telegraph-view")).to_be_visible()
        assert pg.evaluate("document.documentElement.scrollWidth") <= width
    ctx.close()
