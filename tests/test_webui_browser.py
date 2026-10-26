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
import re
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
    page.wait_for_selector("[data-run-row]:visible")


def mode_of(page):
    return page.evaluate("document.documentElement.dataset.mode")


def test_the_live_run_is_first_in_the_list_and_selected(root, page):
    two_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        rows = page.locator("[data-run-row]:visible")
        assert rows.count() == 2
        assert "live-model" in rows.nth(0).inner_text()
        assert "old-model" in rows.nth(1).inner_text()
        expect(page.locator("#run")).to_contain_text("live-model")


def test_choosing_another_run_shows_that_run(root, page):
    two_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        page.locator("[data-run-row]:visible").nth(1).click()
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
        page.wait_for_selector("[data-run-row]:visible")
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


# ---- the run table in Operator style (2026-10-26) ----------------------------------------------------------------------------

LONG_ID = "some-organisation-with-a-long-name/an-extremely-long-model-identifier-that-would-never-fit-in-a-narrow-column-v2-final"


def table_runs(root):
    """A live run, a blocked one, a finished one and one with a very long id, with different stages."""
    write_ledger(root / "org--old-blocked", ("stage_start", 1, {}),
                 ("decision", 1, {"decision": "blocked", "code": "stage-failed", "reason": "x"}), model="org/blocked-model")
    write_ledger(root / "org--ready", ("stage_start", 0, {}), ("stage_end", 0, {"result": "pass"}),
                 ("decision", None, {"decision": "ready for operator review", "bundle": "stages/8/bundle"}), model="org/ready-model")
    write_ledger(root / "org--long", ("stage_start", 2, {}), model=LONG_ID)
    live = write_ledger(root / "org--live", ("stage_start", 3, {}), model="org/live-model")
    with_supervisor(live, LIVE)
    return live


def row_names(page):
    return [r.get_attribute("data-run-name") for r in page.locator("#runs-body tr[data-run-row]").all()]


def test_operator_style_lists_runs_in_a_table_with_a_header_and_playful_keeps_its_list(root, page):
    table_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        heads = [h.text_content().strip() for h in page.locator("#runs-table thead th").all()]
        assert heads == ["State", "Model", "Stage", "Age"]
        assert page.locator("#runs-table").is_visible() and not page.locator("#runs").is_visible()
        first = page.locator("#runs-body tr").first
        assert "live-model" in first.inner_text() and "now" in first.inner_text()
        assert "RUNNING" in first.inner_text().upper()                      # the real state word beside the picture
        assert first.locator(".c-stage").inner_text().strip() == "3"
        assert page.locator("#runs-body tr").nth(1).get_attribute("data-state") in ("ready-for-operator-review", "stopped-or-crashed", "blocked", "running")
        page.click("#mode-playful-btn")
        assert page.locator("#runs").is_visible() and not page.locator("#runs-table").is_visible()


def test_table_rows_keep_the_order_of_the_list_live_first_then_newest(root, page):
    table_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        in_table = row_names(page)
        page.click("#mode-playful-btn")
        in_list = [li.get_attribute("data-run-name") for li in page.locator("#runs li[data-run-row]").all()]
        assert in_table == in_list and in_table[0] == "org--live" and len(in_table) == 4


def test_the_selected_row_is_aria_selected_and_the_table_is_a_grid_with_one_tab_stop(root, page):
    table_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        assert page.get_attribute("#runs-table", "role") == "grid"
        sel = page.locator('#runs-body tr[aria-selected="true"]')
        assert sel.count() == 1 and sel.get_attribute("data-run-name") == "org--live"
        assert page.locator('#runs-body tr[tabindex="0"]').count() == 1
        assert page.locator('#runs-body tr[tabindex="-1"]').count() == 3


def test_arrow_keys_move_the_selection_and_the_run_page_follows_and_enter_keeps_it(root, page):
    table_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        page.locator("#runs-body tr").first.focus()
        page.keyboard.press("ArrowDown")
        second = row_names(page)[1]
        assert page.locator('#runs-body tr[aria-selected="true"]').get_attribute("data-run-name") == second
        assert page.evaluate("document.activeElement.dataset.runName") == second     # focus moved with the selection
        expect(page.locator("#run")).not_to_contain_text("live-model")
        page.keyboard.press("ArrowUp")
        assert page.locator('#runs-body tr[aria-selected="true"]').get_attribute("data-run-name") == "org--live"
        page.keyboard.press("ArrowUp")                                                 # the first row is the edge
        assert page.locator('#runs-body tr[aria-selected="true"]').get_attribute("data-run-name") == "org--live"
        page.keyboard.press("ArrowDown")
        page.keyboard.press("Enter")
        assert page.locator('#runs-body tr[aria-selected="true"]').get_attribute("data-run-name") == second


def test_a_very_long_model_id_is_clipped_in_the_cell_keeps_its_full_text_and_title(root, page):
    table_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        cell = page.locator('#runs-body tr[data-run-name="org--long"] .c-model')
        assert cell.get_attribute("title") == LONG_ID
        assert cell.locator(".c-model-text").text_content() == LONG_ID              # selectable and copyable in full
        box = cell.bounding_box()
        assert box["width"] < 260 and box["height"] < 70
        assert not page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
        assert page.evaluate("getComputedStyle(document.querySelector('#runs-body td.c-model')).userSelect") != "none"


def test_clicking_a_row_selects_it_and_dragging_to_copy_text_does_not(root, page):
    table_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        page.locator('#runs-body tr[data-run-name="org--ready"]').click()
        assert page.locator('#runs-body tr[aria-selected="true"]').get_attribute("data-run-name") == "org--ready"
        page.wait_for_timeout(1000)                              # let the page finish drawing the chosen run
        cell = page.locator('#runs-body tr[data-run-name="org--live"] .c-model-text')
        box = cell.bounding_box()
        page.mouse.move(box["x"] + 1, box["y"] + 6)
        page.mouse.down()
        page.mouse.move(box["x"] + 60, box["y"] + 6, steps=5)
        page.mouse.up()                                          # a drag across the text: a selection inside the table, then the click that ends it
        assert page.evaluate("window.getSelection().toString().length") > 0
        assert page.locator('#runs-body tr[aria-selected="true"]').get_attribute("data-run-name") == "org--ready"


def test_the_tooltips_give_the_stage_name_and_the_exact_time(root, page):
    table_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        row = page.locator('#runs-body tr[data-run-name="org--live"]')
        assert row.locator(".c-stage").get_attribute("title").startswith("stage 3")
        assert re.search(r"\d", row.locator(".c-age").get_attribute("title") or "")


def test_a_narrow_screen_drops_the_age_column_and_never_scrolls_sideways(root, browser):
    table_runs(root)
    ctx = browser.new_context(viewport={"width": 320, "height": 800})
    pg = ctx.new_page()
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(pg, s)
        assert not pg.locator("#runs-table th.c-age").is_visible()
        assert pg.locator("#runs-table th.c-model").is_visible()
        assert not pg.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")
    ctx.close()
    ctx = browser.new_context(viewport={"width": 1400, "height": 800})
    pg = ctx.new_page()
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(pg, s)
        assert pg.locator("#runs-table th.c-age").is_visible()               # the 360 px rail has room for all four
        assert 340 <= pg.locator(".rail").bounding_box()["width"] <= 380
    ctx.close()


def test_no_runs_says_so_in_the_table_with_a_way_to_start(root, page):
    with Served(make_app(root)) as s:
        page.goto(f"http://127.0.0.1:{s.port}/")
        expect(page.locator("#runs-body")).to_contain_text("No runs here yet")
        assert page.locator("#runs-body button").is_visible()


def test_many_runs_stay_in_one_scrolling_page_table(root, page):
    for i in range(40):
        write_ledger(root / f"org--m{i:02d}", ("stage_start", i % 9, {}), model=f"org/model-{i}")
    with Served(make_app(root)) as s:
        page.goto(f"http://127.0.0.1:{s.port}/")
        page.wait_for_selector("[data-run-row]:visible")
        assert page.locator("#runs-body tr[data-run-row]").count() == 40
        assert not page.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth")


def test_enter_opens_the_focused_row_even_when_it_is_not_the_selected_one(root, page):
    table_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        names = row_names(page)
        page.locator("#runs-body tr").nth(2).focus()                      # focus without selecting, as a Tab key would
        assert page.locator('#runs-body tr[aria-selected="true"]').get_attribute("data-run-name") == "org--live"
        page.keyboard.press("Enter")
        assert page.locator('#runs-body tr[aria-selected="true"]').get_attribute("data-run-name") == names[2]


def test_a_redraw_of_the_rows_keeps_the_keyboard_focus_on_the_same_run(root, page):
    table_runs(root)
    with Served(make_app(root, alive={LIVE})) as s:
        open_page(page, s)
        target = row_names(page)[1]
        page.locator("#runs-body tr").nth(1).focus()
        write_ledger(root / "org--long", ("stage_end", 2, {"result": "pass"}), ("stage_start", 4, {}))      # changes a row, so the next poll redraws
        expect(page.locator('#runs-body tr[data-run-name="org--long"] .c-stage')).to_contain_text("4", timeout=15000)
        assert page.evaluate("document.activeElement.dataset.runName") == target
