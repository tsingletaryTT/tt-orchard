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


def _mk(base, name):
    (base / name).mkdir()
    return base / name


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


def test_the_view_shows_small_costs_unreadable_ledgers_hidden_problems_and_cost_bases(root, tmp_path, browser):
    import json
    from test_telegraph_view_review import ok_record, reserve_record, sixty_problems, write_records
    ctx, pg = new_page(browser)
    # a cost under $0.0001 and a reply with no usage
    cfg, d, _ = setup(_mk(tmp_path, "a"))
    write_records(d, [ok_record("a1", "p", 10, 5, cost=0.000004),
                      reserve_record("c3", "q", "small", 2000, 0.02), {**ok_record("c3", "q", 10, 5, cost=0.0001), "usage_known": False}])
    with Served(app_for(root, cfg)) as s:
        open_page(pg, s)
        meter(pg).click()
        expect(pg.locator("#tg-recent")).to_contain_text("<$0.0001")
        expect(pg.locator("#tg-recent")).to_contain_text("$0.02 (worst case)")
        assert "$0.0000" not in pg.inner_text("#telegraph-view")
    # an unreadable ledger
    cfg2, d2, _ = setup(_mk(tmp_path, "b"))
    d2.mkdir(parents=True)
    (d2 / "telegraph.jsonl").mkdir()
    with Served(app_for(root, cfg2)) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")
        pg.wait_for_selector("[data-run-row]:visible")
        expect(pg.locator("#spend-text")).to_have_text("Helper: ledger unreadable · spend unknown")
        meter(pg).click()
        expect(pg.locator("#tg-numbers")).to_contain_text("unknown")
        assert pg.locator("#tg-caps .tg-cap").count() == 0
    # sixty problems
    with Served(app_for(root, sixty_problems(_mk(tmp_path, "c")))) as s:
        pg.goto(f"http://127.0.0.1:{s.port}/?mode=operator")
        pg.wait_for_selector("[data-run-row]:visible")
        expect(meter(pg)).to_contain_text("near cap: problem zzz at 85% of its cap")
        meter(pg).click()
        expect(pg.locator("#tg-by-problem")).to_contain_text("10 more problems not shown")
        expect(pg.locator("#tg-by-problem tr").first).to_contain_text("zzz")
    ctx.close()


@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_telegraph_polling_does_not_wait_for_the_first_run_list_to_answer(root, tmp_path, browser, mode):
    import re
    ctx, pg = new_page(browser)
    with Served(capped_app(root, tmp_path)) as s:
        pg.route(re.compile(r".*/api/(runs|machine|launches)(\?.*)?$"), lambda route: None)     # the first refresh never answers
        pg.goto(f"http://127.0.0.1:{s.port}/?mode={mode}")
        expect(meter(pg)).to_contain_text("$0.0070")               # the first Telegraph poll does not depend on it
        pg.route("**/api/telegraph", lambda route: None)           # and when Telegraph stops answering...
        expect(meter(pg)).to_contain_text("spend unknown", timeout=23000)       # ...its own timers still mark the amount unknown
        assert meter(pg).get_attribute("data-tone") == "bad"
    ctx.close()


def test_telegraph_is_polled_once_per_tick_not_once_for_each_timer(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    seen = []
    pg.on("request", lambda r: seen.append(r.url) if r.url.endswith("/api/telegraph") else None)
    with Served(capped_app(root, tmp_path)) as s:
        open_page(pg, s)
        seen.clear()
        pg.wait_for_timeout(11000)                                 # two ticks of five seconds
        assert 1 <= len(seen) <= 3, seen
    ctx.close()


def test_a_pending_poll_is_not_joined_by_more_requests(root, tmp_path, browser):
    ctx, pg = new_page(browser)
    seen = []
    with Served(capped_app(root, tmp_path)) as s:
        open_page(pg, s)
        pg.route("**/api/telegraph", lambda route: seen.append(route.request.url))
        pg.wait_for_timeout(13000)                                 # two or three poll ticks pass
        assert len(seen) == 1, seen
    ctx.close()


def test_a_poll_that_hung_does_not_block_later_polls_once_the_server_answers_again(root, tmp_path, browser):
    """The server itself hangs on one request (it never answers it) and answers every later one. Only the page's own time limit
    frees the poll, so without it the amount would stay unknown for ever."""
    import threading
    ctx, pg = new_page(browser)
    app = capped_app(root, tmp_path)
    real, calls, never = app.telegraph, [], threading.Event()

    def telegraph():
        calls.append(1)
        if len(calls) == 2:                                        # the first call is the page's first poll
            never.wait(120)
        return real()
    app.telegraph = telegraph
    with Served(app) as s:
        open_page(pg, s)
        expect(meter(pg)).to_contain_text("$0.0070")
        expect(meter(pg)).to_contain_text("spend unknown", timeout=23000)
        # The stale text keeps the old amount, so first wait for "spend unknown" to go: only then has a poll really answered.
        expect(meter(pg)).not_to_contain_text("spend unknown", timeout=30000)      # the hung request was dropped and the next poll answered
        expect(meter(pg)).to_contain_text("$0.0070")
        assert len(calls) >= 3                                       # the first poll, the hung one and at least one later poll
        never.set()
    ctx.close()
