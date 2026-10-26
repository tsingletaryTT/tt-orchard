# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The web page's own JavaScript, driven in a real browser (issue 31).

The server tests cannot see a client bug: the page once sent a blank steps box as 0 and only a browser showed it. These tests
start the real server on a loopback port with fake runs, gozer and preflight (test_webui.make_app), open it in the installed
Chrome through Playwright, and check what a person sees and what the page sends: which run is selected first, the style
switch and what the browser remembers, the confirmation before Pause and Abort (a cancel sends nothing; Abort needs two
clicks), and the machine chip after an outage.

They skip when the `playwright` package or Chrome is missing. To run them here without installing anything into the main
interpreter, point PYTHONPATH at a virtualenv that has `playwright` (the package drives the installed Chrome; it downloads
no browser). Set ORCHARD_CHROME to a different browser binary.
"""
import os
import shutil
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect

from orchard.supervisor import Control
from test_webui import Served, make_app, with_supervisor, write_ledger

CHROME = os.environ.get("ORCHARD_CHROME") or shutil.which("google-chrome") or shutil.which("chromium")
pytestmark = pytest.mark.skipif(not CHROME, reason="no Chrome or Chromium on this machine")

LIVE = 4242


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    return r


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, headless=True, args=["--no-sandbox"])
        yield b
        b.close()


@pytest.fixture
def page(browser):
    ctx = browser.new_context(viewport={"width": 1400, "height": 900})
    pg = ctx.new_page()
    pg.set_default_timeout(15000)
    yield pg
    ctx.close()


def two_runs(root):
    """A finished run that is older, and a live run: the page must put the live one first."""
    old = write_ledger(root / "org--old", ("stage_start", 0, {}), model="org/old-model")
    live = write_ledger(root / "org--live", ("stage_start", 0, {}), model="org/live-model")
    with_supervisor(live, LIVE)
    return old, live


def open_page(page, served, query=""):
    page.goto(f"http://127.0.0.1:{served.port}/{query}")
    page.wait_for_selector("#runs li")


def mode_of(page):
    return page.evaluate("document.documentElement.dataset.mode")


def test_the_live_run_is_first_in_the_list_and_selected(root, page):
    two_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        rows = page.locator("#runs li")
        assert rows.count() == 2
        assert "live-model" in rows.nth(0).inner_text()
        assert "old-model" in rows.nth(1).inner_text()
        expect(page.locator("#run")).to_contain_text("live-model")


def test_choosing_another_run_shows_that_run(root, page):
    two_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        page.locator("#runs li").nth(1).locator("button, a").first.click()
        expect(page.locator("#run")).to_contain_text("old-model")
        expect(page.locator("#run")).not_to_contain_text("live-model")


def test_operator_is_the_default_and_the_style_switch_changes_the_page_and_is_remembered(root, page):
    two_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        assert mode_of(page) == "operator"            # the default since 2026-10-26
        page.click("#mode-playful-btn")
        assert mode_of(page) == "playful"
        assert page.get_attribute("#mode-playful-btn", "aria-pressed") == "true"
        assert page.get_attribute("#mode-operator-btn", "aria-pressed") == "false"
        page.reload()
        page.wait_for_selector("#runs li")
        assert mode_of(page) == "playful"             # the browser remembered it, and the remembered choice beats the default


def test_a_link_can_pick_a_style_and_nothing_else(root, page):
    two_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s, "?mode=playful")
        assert mode_of(page) == "playful"
        open_page(page, s, "?mode=%22%3E%3Cscript%3E")
        assert mode_of(page) == "playful"                        # the stored choice stays; the link's value is ignored


def test_operator_style_shows_small_pictures_beside_the_real_words_and_playful_shows_none_of_them(root, page):
    two_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        band = page.locator("#run .run-head .badge")
        expect(band).to_contain_text("running")                  # the real state name is always on the page
        pic = band.locator(".op-pic")
        expect(pic).to_be_visible()
        assert pic.get_attribute("aria-hidden") == "true" and pic.inner_text().strip() == "🏃"
        pause = page.get_by_role("button", name="Pause", exact=True)   # the picture is aria-hidden, so the name is still just "Pause"
        expect(pause.locator(".op-pic")).to_be_visible()
        assert page.locator("#stepper .step .stat .op-pic").first.is_visible()
        page.click("#mode-playful-btn")
        assert page.locator(".op-pic:visible").count() == 0


def test_operator_pictures_do_not_animate_when_the_browser_asks_for_reduced_motion(root, browser):
    two_runs(root)
    ctx = browser.new_context(viewport={"width": 1400, "height": 900}, reduced_motion="reduce")
    pg = ctx.new_page()
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(pg, s)
        names = pg.evaluate("[...document.querySelectorAll('#run *')].map(e => getComputedStyle(e).animationName).filter(n => n && n !== 'none')")
        assert names == [], names
    ctx.close()


def test_cancelling_pause_sends_nothing(root, page):
    _, live = two_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        page.get_by_role("button", name="Pause", exact=True).click()
        assert page.locator("#confirm").evaluate("d => d.open")
        assert "Pause this run?" in page.locator("#confirm-title").inner_text()
        page.click("#confirm-cancel")
        expect(page.locator("#confirm")).not_to_have_attribute("open", "")
        page.wait_for_timeout(500)
        assert not (live / "control").exists()


def test_confirming_pause_writes_the_control_word(root, page):
    _, live = two_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        page.get_by_role("button", name="Pause", exact=True).click()
        page.click("#confirm-ok")
        expect(page.locator("#confirm")).not_to_have_attribute("open", "")
        for _ in range(50):
            if (live / "control").exists():
                break
            page.wait_for_timeout(100)
        assert (live / "control").read_text().strip() == "pause"


def test_abort_needs_a_second_click(root, page):
    _, live = two_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        page.get_by_role("button", name="Abort", exact=True).click()
        ok = page.locator("#confirm-ok")
        assert ok.inner_text() == "Abort run"
        ok.click()                                         # the first click only arms the button
        assert page.locator("#confirm").evaluate("d => d.open")
        assert "again" in ok.inner_text().lower()
        page.wait_for_timeout(500)
        assert not (live / "control").exists()
        ok.click()
        for _ in range(50):
            if (live / "control").exists():
                break
            page.wait_for_timeout(100)
        assert (live / "control").read_text().strip() == "abort"


def test_the_machine_chip_recovers_after_an_outage(root, page):
    two_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        chip = page.locator("#machine-chip")
        expect(chip).not_to_contain_text("unknown")
        expect(chip).to_contain_text("chips")
        page.route("**/api/**", lambda route: route.abort())
        expect(chip).to_contain_text("unknown", timeout=40000)
        page.unroute("**/api/**")
        expect(chip).not_to_contain_text("unknown", timeout=40000)
