# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where the virtual-spend figure sits in the page and how it is drawn (static checks on the markup, script and styles).

The local meter is a second figure beside the helper's spend meter. It has no tone: the helper meter's amber and red mean
real money only. Behaviour that needs a browser is in tests/test_webui_virtual_browser.py."""
import re

from test_webui_layout import CSS, JS, by_id, fn_body, page
from test_webui_operator_fun import ratio
from test_webui_telegraph_layout import _playful_tokens


def test_the_local_meter_is_a_hidden_button_in_the_top_bar_right_after_the_helper_meter():
    t = page()
    local, helper = by_id(t, "local-meter"), by_id(t, "spend-meter")
    assert local.tag == "button" and local.attrs.get("type") == "button" and "local-meter" in local.classes
    assert "hidden" in local.attrs, "no figure until the first answer says there are local logs"
    right = helper.parent
    assert local.parent is right and right.kids.index(local) == right.kids.index(helper) + 1
    assert not local.inside(by_id(t, "run-view")), "outside every view, so no view can hide it"
    assert by_id(t, "local-text").inside(local)
    assert "data-tone" not in local.attrs and "data-state" not in local.attrs


def test_the_local_meter_has_no_emoji_and_says_virtual_in_its_title():
    local = by_id(page(), "local-meter")
    assert "virtual" in local.attrs["title"].lower()
    assert not re.search("[\U0001F300-\U0001FAFF☀-➿]", local.all_text() + local.attrs["title"])


def test_the_local_meter_opens_the_telegraph_view():
    js = JS()
    assert re.search(r'\$\("local-meter"\)\.addEventListener\("click",\s*\(\)\s*=>\s*setView\("telegraph"\)\)', js)


def test_the_view_has_a_local_virtual_table_before_the_recent_calls_and_says_it_is_not_added():
    t = page()
    view = by_id(t, "telegraph-view")
    section = by_id(t, "tg-local")
    assert section.inside(view) and by_id(t, "tg-local-head").inside(section) and by_id(t, "tg-local-body").inside(section)
    text = section.all_text()
    assert "Local (virtual)" in text and "never added to the helper spend" in text
    nodes = [n for n in view.walk()]
    assert nodes.index(section) < nodes.index(by_id(t, "tg-recent-h")), "the real calls follow; the two are different blocks"
    assert "hidden" in by_id(t, "tg-local-empty").attrs
    assert not section.inside(by_id(t, "tg-spend-h").parent), "its own section, outside the real spend block"


def test_the_local_script_reads_one_endpoint_sends_nothing_and_builds_text_with_text_nodes():
    js = JS()
    start = js.index("// ---- virtual spend:")
    block = js[start:js.index("// ---- telegraph: the spend meter")]
    assert block.count('api("') == 1 and '"/api/virtual-spend"' in block
    assert "body:" not in block and "method" not in block and "POST" not in block
    assert "innerHTML" not in block and "insertAdjacentHTML" not in block and "eval(" not in block
    assert not re.search("[\U0001F300-\U0001FAFF☀-➿]", block)


def test_the_local_script_never_touches_the_helper_meter():
    js = JS()
    block = js[js.index("// ---- virtual spend:"):js.index("// ---- telegraph: the spend meter")]
    for forbidden in ('"spend-meter"', '"spend-text"', "renderSpendMeter", "S.telegraph", "dataset.tone", "dataset.state"):
        assert forbidden not in block, forbidden


def test_an_answer_for_a_run_that_is_no_longer_selected_is_dropped():
    js = JS()
    poll = fn_body(js, "async function pollVirtual")
    assert "const asked = S.selected" in poll and "S.selected !== asked" in poll
    assert poll.index("S.selected !== asked") < poll.index("renderLocalMeter(S.virtual)")
    assert "encodeURIComponent(asked)" in poll


def test_the_figure_is_hidden_after_an_outage_and_when_there_is_nothing_to_show():
    js = JS()
    poll = fn_body(js, "async function pollVirtual")
    assert "renderLocalMeter(null)" in poll and "STALE_MS" in poll and "S.vsOk" in poll
    meter = fn_body(js, "function renderLocalMeter")
    assert "el.hidden = !meter" in meter and "textContent" in meter


def test_polling_asks_with_every_refresh_and_when_a_run_is_selected():
    js = JS()
    runs = fn_body(js, "async function refreshRuns")
    assert "pollVirtual()" in runs and runs.index("pollTelegraph()") < runs.index("pollVirtual()") < runs.index("Promise.all")
    sel = fn_body(js, "function select")
    assert sel.index("S.selected = name") < sel.index("renderLocalMeter(null)") < sel.index("pollVirtual()"), "the old figure goes first"


def test_the_table_has_a_column_for_every_reference_and_marks_them_virtual():
    js = JS()
    body = fn_body(js, "function renderLocalTable")
    assert "Virtual $ at" in body and "for (const r of v.references)" in body and "for (const c of r.virtual)" in body
    assert "v.label" in body and "prompt caching" in body
    assert "not recorded" in body, "a run with no agent time says so and does not show 0"


def test_the_local_meter_has_its_own_rules_in_both_styles_and_no_tone_rules():
    css = CSS()
    assert re.search(r"^\.local-meter \{[^}]*border: 2px dashed", css, re.M)
    assert re.search(r"^\.local-meter\[hidden\] \{ display: none; \}", css, re.M)
    assert re.search(r'^html\[data-mode="operator"\] \.local-meter \{[^}]*dashed', css, re.M)
    assert "local-meter[data-tone" not in css and "local-meter[data-state" not in css
    for rule in re.findall(r"^[^\n{]*\.local-meter[^\n{]*\{[^}]*\}", css, re.M):
        assert "--warn" not in rule and "--bad" not in rule and "#f2c14e" not in rule and "#b3290a" not in rule, rule


def test_the_local_meter_wraps_on_a_narrow_screen():
    css = CSS()
    assert re.search(r"@media \(max-width: 900px\) \{[^}]*\.local-meter \{ max-width: 100%; \}[^}]*\.local-text \{ white-space: normal; \}", css)


# ---- colour: both styles, light and dark, 4.5:1 text and 3:1 outline ---------------------------------------------------------

def test_playful_local_meter_text_is_aa_in_light_and_dark():
    css = CSS()
    rule = re.search(r"^\.local-meter \{[^}]*\}", css, re.M).group(0)
    assert "color: var(--muted)" in rule and "background: var(--paper)" in rule and "dashed var(--muted)" in rule
    for scheme in ("light", "dark"):
        t = _playful_tokens(scheme)
        assert ratio(t["--muted"], t["--paper"]) >= 4.5, scheme
        assert ratio(t["--muted"], t["--paper"]) >= 3.0, (scheme, "the dashed outline")


def test_operator_local_meter_is_aa_on_the_dark_top_bar():
    css = CSS()
    bar = re.search(r'^html\[data-mode="operator"\] \.topbar \{[^}]*background: (#[0-9a-fA-F]{6})', css, re.M).group(1)
    rule = re.search(r'^html\[data-mode="operator"\] \.local-meter \{[^}]*\}', css, re.M).group(0)
    color = re.search(r"color: (#[0-9a-fA-F]{6})", rule).group(1)
    border = re.search(r"border: 1px dashed (#[0-9a-fA-F]{6})", rule).group(1)
    assert "background: transparent" in rule
    assert ratio(color, bar) >= 4.5 and ratio(border, bar) >= 3.0



def test_the_local_table_is_redrawn_only_when_its_data_changes():
    body = fn_body(JS(), "function renderLocalTable")
    assert 'changed($("tg-local"), v.rows.concat([v.references]))' in body
