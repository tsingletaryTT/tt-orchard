# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The run list in Operator style is a compact table (2026-10-26). Playful keeps its list.

Static checks of the markup, script and stylesheet. The behaviour (order, selection, keyboard, narrow widths) is checked in a
real browser in tests/test_webui_browser.py.
"""
import re
from pathlib import Path

from test_webui_operator_fun import _tokens, ratio

WEB = Path(__file__).resolve().parent.parent / "orchard" / "web"
JS = lambda: (WEB / "app.js").read_text(encoding="utf-8")
CSS = lambda: (WEB / "app.css").read_text(encoding="utf-8")
HTML = lambda: (WEB / "index.html").read_text(encoding="utf-8")


def fn_body(js, header):
    start = js.index(header)
    return js[start:js.index("\n}\n", start)]


def table_html():
    html = HTML()
    start = html.index('id="runs-table"')
    return html[html.rindex("<table", 0, start):html.index("</table>", start)]


def test_the_table_has_real_grid_semantics_and_a_header_row_with_four_named_columns():
    t = table_html()
    assert 'role="grid"' in t and 'aria-label="Runs on this machine"' in t
    heads = re.findall(r'<th[^>]*scope="col"[^>]*>(.*?)</th>', t, re.S)
    assert [re.sub(r"<[^>]+>", "", h).strip() for h in heads] == ["State", "Model", "Stage", "Age"]
    assert "<thead>" in t and '<tbody id="runs-body"' in t


def test_the_old_list_stays_for_playful():
    assert '<ul id="runs" class="runs"' in HTML()


def test_each_row_is_a_gridrow_with_aria_selected_a_roving_tabindex_and_both_lists_are_marked_as_rows():
    body = fn_body(JS(), "function renderRuns")
    assert 'role: "row"' in body and '"aria-selected"' in body
    assert "tabindex: String(rovingStop(r, i) ? 0 : -1)" in body
    assert body.count('"data-run-row"') == 2                               # the list item and the table row both carry the marker


def test_cells_carry_the_full_model_id_the_stage_name_and_the_exact_time_as_tooltips():
    body = fn_body(JS(), "function renderRuns")
    assert 'class: "c-model", title: tip' in body and "const tip = full === r.name ? full" in body
    assert "stage.name" in body                                       # the stage column's tooltip
    assert "toLocaleString" in body                                    # the age column's tooltip


def test_keyboard_up_down_move_the_selection_and_enter_opens():
    js = JS()
    assert 'addEventListener("keydown"' in js and '"ArrowDown"' in js and '"ArrowUp"' in js
    block = js[js.index('$("runs-body").addEventListener("keydown"'):]
    block = block[:block.index("\n  });")]
    assert '"Enter"' in block and "select(" in block and ".focus()" in block


def test_a_drag_to_copy_text_does_not_count_as_a_click_on_the_row():
    body = fn_body(JS(), "function renderRuns")
    assert "getSelection" in body


def test_the_table_shows_only_in_operator_style_and_the_list_only_outside_it():
    css = CSS()
    assert re.search(r'^\.runs-table-wrap \{ display: none; \}', css, re.M)
    assert re.search(r'^html\[data-mode="operator"\] \.runs-table-wrap \{ display: block;', css, re.M)
    assert re.search(r'^html\[data-mode="operator"\] #runs \{ display: none; \}', css, re.M)


def test_narrow_widths_drop_the_age_column_instead_of_scrolling_sideways():
    css = CSS()
    assert re.search(r'\.runs-table-wrap \{[^}]*container-type: inline-size', css)
    m = re.search(r'@container \(max-width: (\d+)px\) \{([^}]*\{[^}]*\}[^}]*)\}', css)
    assert m and ".c-age" in m.group(2) and "display: none" in m.group(2)
    assert re.search(r'#runs-table \{[^}]*table-layout: fixed', css)


def test_the_selected_row_is_marked_by_more_than_colour_and_text_stays_selectable():
    css = CSS()
    sel = re.search(r'#runs-table tr\[aria-selected="true"\][^{]*\{([^}]*)\}', css)
    assert sel and "box-shadow" in sel.group(1) and "background" in sel.group(1)
    assert "user-select: none" not in css[css.index("#runs-table"):]


def test_the_operator_rail_is_wide_enough_for_the_table():
    assert re.search(r'@media \(min-width: 901px\) \{\s*html\[data-mode="operator"\] body:not\(\.wide\) \.sheet \{ grid-template-columns: 360px', CSS())


def _pairs(scheme):
    t = _tokens(scheme)
    return [(t["--ink"], t["--paper"]), (t["--ink"], t["--paper-2"]), (t["--muted"], t["--paper"]), (t["--muted"], t["--paper-2"]), (t["--ink"], t["--paper-3"]), (t["--muted"], t["--paper-3"])]


def test_table_text_is_aa_on_the_row_and_the_selected_row_in_light_and_dark():
    import pytest
    for scheme in ("light", "dark"):
        for fg, bg in _pairs(scheme):
            assert ratio(fg, bg) >= 4.5, (scheme, fg, bg)
