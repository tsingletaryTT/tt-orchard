# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixes for the fourth Copilot review of PR 111 that need a browser (installed Chrome through Playwright)."""
import json

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

from test_telegraph_view import app_for, setup
from test_telegraph_view_review import ok_record, write_records
from test_webui_telegraph_browser import CHROME, browser, meter, new_page, open_page, root  # noqa: F401
from test_webui import Served, write_ledger

pytestmark = pytest.mark.skipif(not CHROME, reason="no Chrome or Chromium on this machine")

LONG = "a" * 70                                                   # a problem name with no break in it


def near_app(root, tmp_path):
    cfg, d, _ = setup(tmp_path, max_calls=1000, max_total_tokens=10 ** 9, max_calls_per_problem=9, max_tokens_per_problem=1000)
    write_records(d, [ok_record("h1", LONG, 850, 50)])          # one problem at 90 percent of its token cap
    return app_for(root, cfg)


def clipped(pg):
    return pg.evaluate("""() => { const el = document.getElementById('spend-text'); const cs = getComputedStyle(el);
        return { overflow: el.scrollWidth > Math.ceil(el.getBoundingClientRect().width) + 1, ellipsis: cs.textOverflow === 'ellipsis',
                 nowrap: cs.whiteSpace === 'nowrap', text: el.textContent }; }""")


@pytest.mark.parametrize("mode", ["operator", "playful"])
@pytest.mark.parametrize("width", [320, 400, 820, 1440])
def test_the_meter_wraps_so_the_closest_cap_percentage_is_always_visible(root, tmp_path, browser, mode, width):
    ctx, pg = new_page(browser, width)
    with Served(near_app(root, tmp_path)) as s:
        open_page(pg, s, mode)
        expect(meter(pg)).to_contain_text("at 90% of its cap")
        info = clipped(pg)
        assert "90% of its cap" in info["text"]
        assert not info["overflow"] and not info["ellipsis"] and not info["nowrap"], info
        box = meter(pg).bounding_box()
        assert box["x"] >= -1 and box["x"] + box["width"] <= width + 1, box          # inside the screen
        assert pg.evaluate("document.documentElement.scrollWidth") <= width
    ctx.close()


def test_the_meter_title_repeats_the_whole_text(root, tmp_path, browser):
    ctx, pg = new_page(browser, 400)
    with Served(near_app(root, tmp_path)) as s:
        open_page(pg, s)
        expect(meter(pg)).to_contain_text("at 90% of its cap")
        assert "at 90% of its cap" in meter(pg).get_attribute("title")
    ctx.close()


@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_a_run_whose_own_ledger_cannot_be_read_says_its_spend_is_unknown(root, tmp_path, browser, mode):
    ctx, pg = new_page(browser)
    run = write_ledger(root / "org--live", ("stage_start", 0, {}), model="org/live-model")
    (run / "telegraph.jsonl").mkdir()
    with Served(app_for(root, setup(tmp_path)[0])) as s:
        open_page(pg, s, mode)
        expect(pg.locator("#run-spend")).to_be_visible()
        expect(pg.locator("#run-spend")).to_contain_text("Helper spend for this run: unknown")
        expect(pg.locator("#run-spend")).to_contain_text("the run's ledger cannot be read")
        assert "$0.00" not in pg.locator("#run-spend").inner_text()
    ctx.close()


def test_a_run_with_no_ledger_shows_no_spend_line(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    write_ledger(root / "org--live", ("stage_start", 0, {}), model="org/live-model")
    with Served(app_for(root, setup(tmp_path)[0])) as s:
        open_page(pg, s)
        expect(pg.locator("#run-spend")).to_be_hidden()
    ctx.close()
