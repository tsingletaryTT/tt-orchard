# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Cans view in a real browser (installed Chrome through Playwright).

Fake specs and bundles in a temporary folder, the real server on a loopback port. They skip when `playwright` or Chrome is missing (see
tests/test_webui_browser.py for how to run them). One test records every request the page makes: only GETs to the page's own server, so
the view cannot change anything and reaches no Hub."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

from cans_fixture import add_spec, make_checkout, passed_boot, record, stage_bundle
from orchard import can
from test_webui import Served, make_app

CHROME = os.environ.get("ORCHARD_CHROME") or shutil.which("google-chrome") or shutil.which("chromium")
pytestmark = pytest.mark.skipif(not CHROME, reason="no Chrome or Chromium on this machine")


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox"])
        yield b
        b.close()


@pytest.fixture
def served(tmp_path):
    """Two cans: `clean` (staged, sums match, gate and boot recorded as passed) and `broken` (staged, a file changed, a gate recorded as failed)."""
    runs = tmp_path / "orchard-runs"
    runs.mkdir()
    app = make_app(runs)
    checkout = make_checkout(tmp_path)
    app.cans_specs_dir, app.cans_base = checkout / "config", checkout
    app.environ = {}
    clean = add_spec(checkout, "clean", repo="someone/clean-model")
    bad = add_spec(checkout, "broken", repo="someone/broken-model")
    cans = runs / "cans"
    b1 = stage_bundle(cans, clean)
    record(cans, clean, b1, gate=can.gate_record([]), boot=passed_boot())
    b2 = stage_bundle(cans, bad)
    record(cans, bad, b2, gate=can.gate_record(["scrub: a token-looking string in run.sh", "card: no Limitations section"]))
    (b2 / "run.sh").write_text("#!/bin/sh\necho changed\n")
    with Served(app) as s:
        s.app, s.cans_dir = app, cans
        yield s


def open_cans(browser, served, mode="operator", width=1400):
    ctx = browser.new_context(viewport={"width": width, "height": 900})
    pg = ctx.new_page()
    pg.set_default_timeout(15000)
    pg.goto(f"http://127.0.0.1:{served.port}/?mode={mode}#cans")
    pg.wait_for_selector("#cans-body tr.cans-row")
    return ctx, pg


@pytest.mark.parametrize("mode", ["operator", "playful"])
def test_the_view_lists_both_cans_with_words_for_their_status_in_both_styles(browser, served, mode):
    ctx, pg = open_cans(browser, served, mode)
    expect(pg.locator("#cans-view")).to_be_visible()
    expect(pg.locator("#view-cans-btn")).to_have_attribute("aria-pressed", "true")
    rows = pg.locator("#cans-body tr.cans-row")
    expect(rows).to_have_count(2)
    expect(rows.nth(0)).to_contain_text("failing")
    expect(rows.nth(0)).to_contain_text("differs: 1 problem")
    expect(rows.nth(0)).to_contain_text("failed with 2 problems")
    expect(rows.nth(1)).to_contain_text("booted")
    expect(rows.nth(1)).to_contain_text("decode 31.4 tok/s")
    expect(rows.nth(1)).to_contain_text("cache empty")
    ctx.close()


def test_a_row_opens_with_the_keyboard_and_the_details_name_the_card_rows_and_the_hub_state(browser, served):
    ctx, pg = open_cans(browser, served)
    toggle = pg.locator("#can-toggle-clean")
    detail = pg.locator("#can-detail-clean")
    expect(detail).to_be_hidden()
    toggle.focus()
    pg.keyboard.press("Enter")
    expect(toggle).to_have_attribute("aria-expanded", "true")
    expect(toggle).to_be_focused()
    expect(detail).to_be_visible()
    expect(detail).to_contain_text("Decode")
    expect(detail).to_contain_text("31.4 tok/s")
    expect(detail).to_contain_text("Greedy decoding only.")
    expect(detail).to_contain_text("not checked here")
    link = detail.locator("a")
    expect(link).to_have_attribute("href", "https://huggingface.co/someone/clean-model")
    expect(link).to_have_attribute("rel", "noopener noreferrer")
    pg.keyboard.press("Space")
    expect(detail).to_be_hidden()
    ctx.close()


def test_an_open_row_stays_open_when_the_page_polls_again(browser, served):
    ctx, pg = open_cans(browser, served)
    pg.locator("#can-toggle-broken").click()
    expect(pg.locator("#can-detail-broken")).to_be_visible()
    pg.evaluate("pollCans()")
    pg.wait_for_timeout(300)
    expect(pg.locator("#can-detail-broken")).to_be_visible()
    expect(pg.locator("#can-detail-broken")).to_contain_text("run.sh changed after SHA256SUMS was written")
    ctx.close()


def test_the_page_makes_only_get_requests_to_its_own_server_and_none_to_the_hub(browser, served):
    ctx = browser.new_context(viewport={"width": 1400, "height": 900})
    pg = ctx.new_page()
    seen = []
    pg.on("request", lambda r: seen.append((r.method, r.url)))
    pg.goto(f"http://127.0.0.1:{served.port}/?mode=operator#cans")
    pg.wait_for_selector("#cans-body tr.cans-row")
    pg.locator("#can-toggle-clean").click()
    pg.wait_for_timeout(300)
    assert seen and all(m == "GET" and u.startswith(f"http://127.0.0.1:{served.port}/") for m, u in seen), seen
    assert any(u.endswith("/api/cans") for _, u in seen)
    ctx.close()


def test_a_server_that_stops_answering_leaves_the_rows_and_says_they_may_be_old(browser, served):
    ctx, pg = open_cans(browser, served)
    pg.route("**/api/cans", lambda route: route.abort())
    pg.evaluate("pollCans()")
    expect(pg.locator("#cans-stale")).to_be_visible()
    expect(pg.locator("#cans-stale")).to_contain_text("may be out of date")
    expect(pg.locator("#cans-body tr.cans-row")).to_have_count(2)
    ctx.close()


def test_the_pane_does_not_scroll_sideways_at_phone_width(browser, served):
    ctx, pg = open_cans(browser, served, "operator", width=390)
    assert pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
    ctx.close()


def test_leaving_the_view_stops_its_polling(browser, served):
    ctx, pg = open_cans(browser, served)
    assert pg.evaluate("S.cansTimer !== null")
    pg.locator("#view-hardware-btn").click()
    assert pg.evaluate("S.cansTimer === null")
    ctx.close()
