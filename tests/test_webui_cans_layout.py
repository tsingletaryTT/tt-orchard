# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where the Cans view sits in the page and how it is drawn (static checks on the markup, the script and the styles).

The view only reads: its script makes one GET to /api/cans and builds every string with text nodes. Behaviour that needs a browser is in
tests/test_webui_cans_browser.py."""
import re

from test_webui_layout import CSS, JS, by_id, fn_body, has_id, order, page
from test_webui_operator_fun import _tokens as operator_tokens, ratio
from test_webui_telegraph_layout import _playful_tokens


def cans_block(js):
    start = js.index("// ---- cans:")
    return js[start:js.index("// ---- machine health:", start)]


def test_the_tab_is_a_view_button_after_telegraph_in_the_same_nav():
    t = page()
    tab, tg = by_id(t, "view-cans-btn"), by_id(t, "view-telegraph-btn")
    assert tab.tag == "button" and tab.attrs.get("type") == "button" and "view-btn" in tab.classes
    assert tab.parent is tg.parent and tab.parent.kids.index(tab) == tab.parent.kids.index(tg) + 1
    assert tab.attrs.get("aria-pressed") == "false" and tab.all_text().replace("🥫", "").strip() == "Cans"
    assert tab.parent.tag == "nav" and tab.parent.attrs.get("aria-label") == "Views"
    emoji = [k for k in tab.kids if "emoji" in k.classes]
    assert len(emoji) == 1 and emoji[0].attrs.get("aria-hidden") == "true", "the picture is decoration; the word carries the meaning"


def test_the_pane_is_a_hidden_labelled_section_with_one_heading_and_says_it_only_reads():
    t = page()
    view = by_id(t, "cans-view")
    assert view.tag == "section" and "hidden" in view.attrs and view.attrs.get("aria-labelledby") == "cans-h"
    assert by_id(t, "cans-h").tag == "h1" and by_id(t, "cans-h").inside(view)
    assert [n for n in view.walk() if n.tag == "h1"] == [by_id(t, "cans-h")]
    assert "only reads" in view.all_text() and "Hub" in view.all_text()
    assert not [n for n in view.walk() if n.tag in ("form", "input", "select", "textarea")]
    assert view.parent is by_id(t, "telegraph-view").parent, "a sibling of the other views"


def test_the_pane_has_a_status_note_a_stale_alert_and_a_table_with_header_cells_that_have_scope():
    t = page()
    view = by_id(t, "cans-view")
    assert by_id(t, "cans-stale").attrs.get("role") == "alert" and "hidden" in by_id(t, "cans-stale").attrs
    assert by_id(t, "cans-folder").attrs.get("role") == "status"
    table = by_id(t, "cans-table")
    assert table.inside(view) and by_id(t, "cans-body").inside(table)
    heads = [n for n in table.walk() if n.tag == "th"]
    assert [h.all_text() for h in heads] == ["Status", "Can", "Kind", "Mesh", "Weights", "Bundle", "SHA256SUMS", "Gate", "Boot"]
    assert all(h.attrs.get("scope") == "col" for h in heads)
    assert any(n.tag == "caption" for n in table.walk())
    assert "hidden" in by_id(t, "cans-empty").attrs
    assert order(t, "cans-stale", "cans-folder", "cans-table") == ["cans-stale", "cans-folder", "cans-table"]


def test_the_hub_note_is_in_the_page_and_filled_from_the_answer():
    t = page()
    assert by_id(t, "cans-hub").inside(by_id(t, "cans-view"))
    assert "Hub" in by_id(t, "cans-hub-h").all_text()


def test_the_view_is_one_of_the_views_setview_switches_and_has_a_hash():
    js = JS()
    assert re.search(r'\$\("view-cans-btn"\)\.addEventListener\("click",\s*\(\)\s*=>\s*setView\("cans"\)\)', js)
    body = fn_body(js, "function setView")
    assert '$("cans-view").hidden = view !== "cans"' in body
    assert '$("view-cans-btn").setAttribute("aria-pressed", String(view === "cans"))' in body
    assert 'history.replaceState(null, "", "#cans")' in body
    assert 'hash === "#cans"' in js
    assert "stopCans()" in body, "polling stops when another view is shown"


def test_the_cans_script_reads_one_endpoint_sends_nothing_and_builds_text_with_text_nodes():
    block = cans_block(JS())
    assert block.count('api("') == 1 and '"/api/cans"' in block
    assert "body:" not in block and "POST" not in block and "method" not in block
    for forbidden in ("innerHTML", "insertAdjacentHTML", "eval(", "document.write", "outerHTML"):
        assert forbidden not in block, forbidden
    assert not re.search("[\U0001F300-\U0001FAFF☀-➿]", block), "no emoji in the script; the markup has the one decoration"
    assert "localStorage" not in block


def test_the_link_to_the_hub_is_made_only_from_the_servers_url_and_opens_safely():
    block = cans_block(JS())
    assert "hub_url" in block and 'rel: "noopener noreferrer"' in block and 'target: "_blank"' in block
    assert block.count('"a"') == 1
    assert 'startsWith("https://huggingface.co/")' in block, "the page checks the address before it makes a link"


def test_rows_open_their_details_with_a_real_button_that_says_whether_it_is_open():
    block = cans_block(JS())
    assert 'h("button"' in block and '"aria-expanded"' in block and '"aria-controls"' in block
    assert 'type: "button"' in block
    fn = fn_body(JS(), "function toggleCan")
    assert "S.cansOpen" in fn and "renderCans" in fn


def test_a_status_is_a_word_and_a_mark_never_colour_alone():
    block = cans_block(JS())
    assert "c.summary" in block and 'class: "mark"' in block and '"aria-hidden": "true"' in block
    css = CSS()
    assert re.search(r'\.cans-status \.mark', css)


def test_open_rows_survive_a_redraw_and_a_redraw_is_skipped_when_nothing_changed():
    js = JS()
    fn = fn_body(js, "function renderCans")
    assert "changed($(\"cans-view\")" in fn and "S.cansOpen" in fn
    assert "S.cansOpen = new Set" in js or "cansOpen: new Set" in js


def test_polling_is_slow_stops_with_the_view_and_marks_old_data_as_old():
    js = JS()
    block = cans_block(js)
    assert re.search(r"POLL_CANS_MS\s*=\s*15000", block)
    stop = fn_body(js, "function stopCans")
    assert "clearInterval" in stop
    poll = fn_body(js, "async function pollCans")
    assert "S.cansPending" in poll and "cans-stale" in block and "S.cansOk" in poll
    assert 'document.hidden' in block


def test_the_cans_rules_work_in_both_styles_and_every_tone_is_readable():
    css = CSS()
    for sel in (r"\.cans \{", r"\.cans-table", r"\.cans-detail", r"\.cans-toggle", r"\.cans-status"):
        assert re.search(sel, css), sel
    assert re.search(r'\.cans-table tr\[data-tone="bad"\]', css) and re.search(r'\.cans-table tr\[data-tone="warn"\]', css)
    assert ":focus-visible" in css and re.search(r"\.cans-toggle:focus-visible", css)
    # Tone text sits on the page paper only (a row has no tint, so a good cell never lands on a warn or bad background). Checked in the
    # playful and the operator palette, light and dark. The warn cell uses --warn-ink where a palette defines it, as the other panes do.
    assert not re.search(r"\.cans-table tr\[data-tone=\"(good|warn|bad)\"\] \{[^}]*background", css), "no row tint"
    for scheme in ("light", "dark"):
        for name, tokens in (("playful", _playful_tokens(scheme)), ("operator", operator_tokens(scheme))):
            for tone in ("good", "bad"):
                assert ratio(tokens[f"--{tone}"], tokens["--paper"]) >= 4.5, (name, scheme, tone)
            assert ratio(tokens.get("--warn-ink", tokens["--warn"]), tokens["--paper"]) >= 4.5, (name, scheme, "warn")
            assert ratio(tokens["--muted"], tokens["--paper"]) >= 4.5, (name, scheme, "dim")


def test_the_table_scrolls_inside_its_wrapper_on_a_narrow_screen_and_the_pane_does_not_overflow():
    t = page()
    table = by_id(t, "cans-table")
    assert "table-wrap" in table.parent.classes
    css = CSS()
    assert re.search(r"\.cans \{[^}]*min-width: 0", css)
    assert re.search(r"\.cans-detail[^{]*\{[^}]*overflow-wrap: anywhere", css) or "overflow-wrap: anywhere" in css


def test_a_redraw_puts_keyboard_focus_back_on_the_button_that_had_it():
    fn = fn_body(JS(), "function renderCans")
    assert "document.activeElement" in fn and 'startsWith("can-toggle-")' in fn
    assert fn.index("document.activeElement") < fn.index('clear($("cans-body"))') < fn.rindex(".focus()")


def test_the_format_helper_knows_the_hash():
    from test_webui_layout import WEB
    text = (WEB / "format.js").read_text(encoding="utf-8")
    assert 'hash === "#cans"' in text and 'view: "cans"' in text
