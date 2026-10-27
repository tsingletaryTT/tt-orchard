# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where the spend meter and the Telegraph view sit in the page, and that their colours are readable (static checks).

The meter is in the top bar, so it is on screen in every view and both styles. The view only reads. Behaviour that needs a
browser is in tests/test_webui_telegraph_browser.py."""
import re

import pytest

from test_webui_layout import CSS, JS, by_id, fn_body, page
from test_webui_operator_fun import _tokens as operator_tokens, ratio


def test_the_spend_meter_is_a_button_in_the_top_bar_before_the_machine_chip():
    t = page()
    meter = by_id(t, "spend-meter")
    assert meter.tag == "button" and meter.attrs.get("type") == "button" and "spend-meter" in meter.classes
    right = meter.parent
    assert "topbar-right" in right.classes and right.kids[0] is meter
    assert by_id(t, "machine-chip") in right.kids
    assert not meter.inside(by_id(t, "run-view")), "the meter is outside every view, so no view can hide it"
    text = by_id(t, "spend-text")
    assert text.inside(meter) and text.attrs.get("class") == "spend-text"


def test_the_meter_opens_the_telegraph_view_and_there_is_a_tab_for_it():
    js = JS()
    assert re.search(r'\$\("spend-meter"\)\.addEventListener\("click",\s*\(\)\s*=>\s*setView\("telegraph"\)\)', js)
    assert re.search(r'\$\("view-telegraph-btn"\)\.addEventListener\("click",\s*\(\)\s*=>\s*setView\("telegraph"\)\)', js)
    t = page()
    assert by_id(t, "view-telegraph-btn").parent is by_id(t, "view-orchard-btn").parent
    assert 'location.hash === "#telegraph"' in js and '"#telegraph"' in fn_body(js, "function setView")
    assert '$("telegraph-view").hidden = view !== "telegraph"' in fn_body(js, "function setView")


def test_the_view_has_every_section_and_a_readme_pointer():
    t = page()
    view = by_id(t, "telegraph-view")
    assert "hidden" in view.attrs
    for ident in ("tg-banner", "tg-numbers", "tg-caps", "tg-recent", "tg-by-model", "tg-by-problem", "tg-config", "tg-models"):
        assert by_id(t, ident).inside(view), ident
    assert "5.9g" in view.all_text()
    assert "never sends a request to the helper" in view.all_text()
    for th in [n for n in view.walk() if n.tag == "th"]:
        assert th.attrs.get("scope") == "col"


def test_the_meter_is_redrawn_on_every_poll_and_says_unknown_after_an_outage():
    js = JS()
    poll = fn_body(js, "async function pollTelegraph")
    assert poll.index("renderSpendMeter(S.telegraph)") < poll.index("renderTelegraph()")     # outside the view's change guard
    stale = fn_body(js, "function markTelegraphStale")
    assert "renderSpendMeter(null)" in stale and "STALE_MS" in stale and "S.tgOk" in stale and "S.tgStart" in stale
    # staleness is judged before the request is awaited, so a request that never answers cannot keep an old amount on screen
    assert poll.index("markTelegraphStale()") < poll.index("await api(")
    assert "timeout: TG_TIMEOUT_MS" in poll and "if (S.tgPending) return;" in poll
    assert "(${r.cost_basis})" in js, "a row with no reported usage says whether its cost is a worst case or an estimate"
    assert "AbortController" in fn_body(js, "async function api") and "ctl.abort()" in fn_body(js, "async function api")
    meter = fn_body(js, "function renderSpendMeter")
    assert "spend unknown" in meter and 'dataset.tone = "bad"' in meter
    runs = fn_body(js, "async function refreshRuns")
    assert "pollTelegraph" not in runs, "polling must not depend on, or be repeated by, the run list's refresh"
    main = fn_body(js, "async function main")
    assert main.index("startTelegraphPolling()") < main.index("await refreshRuns()"), "timers first, so a hung request cannot stop them"
    start = fn_body(js, "function startTelegraphPolling")
    assert "setInterval(" in start and "markTelegraphStale" in start and "POLL_TELEGRAPH_MS" in start and "2000" in start
    assert start.strip().splitlines()[1].strip() == "pollTelegraph();", "the first poll is immediate, not one tick later"
    assert "pollTelegraph" not in fn_body(js, "function renderTelegraph")


def test_the_view_never_sends_anything_and_builds_text_with_text_nodes():
    js = JS()
    start = js.index("// ---- telegraph: the spend meter")
    block = js[start:js.index("// ---- machine health")]
    assert "body:" not in block and "method" not in block and "POST" not in block
    assert "innerHTML" not in block and "insertAdjacentHTML" not in block and "eval(" not in block
    assert block.count('api("') == 1 and 'api("/api/telegraph", { timeout: TG_TIMEOUT_MS })' in block


def test_every_emoji_the_telegraph_script_adds_is_none():
    js = JS()
    block = js[js.index("// ---- telegraph: the spend meter"):js.index("// ---- machine health")]
    assert not re.search("[\U0001F300-\U0001FAFF☀-➿]", block)


def test_each_tone_has_a_rule_in_both_styles_and_the_view_has_rules():
    css = CSS()
    for tone in ("good", "warn", "bad", "dim"):
        assert re.search(rf'^\.spend-meter\[data-tone="{tone}"\]', css, re.M), tone
    for tone in ("dim", "warn", "bad"):
        assert re.search(rf'^html\[data-mode="operator"\] \.spend-meter\[data-tone="{tone}"\]', css, re.M), tone
    assert re.search(r'^html\[data-mode="operator"\] \.spend-meter \{', css, re.M)
    for sel in (".telegraph", ".tg-banner", ".tg-cap", ".tg-bar", ".tg-table", ".tg-num", ".run-spend"):
        assert re.search(rf"^{re.escape(sel)}[ ,\{{\[]", css, re.M), sel
    assert re.search(r"^\.views \{[^}]*flex-wrap: wrap", css, re.M), "four tabs must wrap on a phone"


def test_a_cap_in_a_cost_unit_matches_the_servers_dollar_rule():
    js = JS()
    body = fn_body(js, "function usd")
    assert "v === 0" in body and "v < 0.01" in body and "toFixed(4)" in body and "toFixed(2)" in body
    assert 'if (v < 0.0001) return "<$0.0001";' in body, "a real cost under 0.0001 must not show as $0.0000"
    assert body.index("v < 0.0001") < body.index("v < 0.01")


# ---- colour: both styles, light and dark, 4.5:1 -------------------------------------------------------------------------------

def _playful_tokens(scheme):
    css = CSS()
    light = css[css.index(":root {"):]
    light = light[:light.index("\n}\n")]
    out = dict(re.findall(r"(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", light))
    if scheme == "dark":
        dark = css[css.index("@media (prefers-color-scheme: dark) {\n  :root {"):]
        dark = dark[:dark.index("\n  }\n")]
        out.update(dict(re.findall(r"(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", dark)))
    return out


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_playful_meter_and_banner_pairs_are_aa(scheme):
    t = _playful_tokens(scheme)
    assert ratio(t["--ink"], t["--paper"]) >= 4.5
    assert ratio(t["--muted"], t["--paper"]) >= 4.5                       # the off state
    assert ratio(t["--warn-ink"], t["--warn-bg"]) >= 4.5, (scheme, "warn meter")
    assert ratio(t["--bad"], t["--bad-bg"]) >= 4.5, (scheme, "bad meter")
    css = CSS()
    assert re.search(r'^\.spend-meter\[data-tone="warn"\] \{[^}]*color: var\(--warn-ink\)', css, re.M)
    assert re.search(r'^\.spend-meter\[data-tone="bad"\] \{[^}]*color: var\(--bad\)', css, re.M)
    for name in ("good", "warn", "bad", "dim"):
        assert ratio(t["--ink"], t[f"--{name}-bg"]) >= 4.5, (scheme, name, "banner")
    assert ratio(t["--ink"], t["--paper-2"]) >= 4.5 and ratio(t["--muted"], t["--paper-2"]) >= 4.5      # the number cards


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_operator_banner_cap_text_and_number_cards_are_aa(scheme):
    t = operator_tokens(scheme)
    for name in ("good", "warn", "bad", "dim"):
        assert ratio(t["--ink"], t[f"--{name}-bg"]) >= 4.5, (scheme, name, "banner")
    for name in ("warn", "bad"):
        assert ratio(t[f"--{name}"], t["--paper"]) >= 4.5, (scheme, name, "cap text on paper")
    assert ratio(t["--ink"], t["--paper-2"]) >= 4.5 and ratio(t["--muted"], t["--paper-2"]) >= 4.5


def test_the_operator_meter_pairs_on_the_dark_top_bar_are_aa():
    css = CSS()
    bar = "#092221"
    def colours(tone):
        m = re.search(rf'^html\[data-mode="operator"\] \.spend-meter\[data-tone="{tone}"\] \{{([^}}]*)\}}', css, re.M)
        body = m.group(1)
        return (re.search(r"(?<![-\w])background: (#[0-9a-fA-F]{6})", body).group(1) if "background" in body else bar,
                re.search(r"(?<![-\w])color: (#[0-9a-fA-F]{6})", body).group(1))
    base = re.search(r'^html\[data-mode="operator"\] \.spend-meter \{([^}]*)\}', css, re.M).group(1)
    assert ratio(re.search(r"(?<![-\w])color: (#[0-9a-fA-F]{6})", base).group(1), bar) >= 4.5       # the ok state
    for tone in ("dim", "warn", "bad"):
        bg, fg = colours(tone)
        assert ratio(fg, bg) >= 4.5, (tone, bg, fg, ratio(fg, bg))


def test_up_to_900_px_the_meter_is_pinned_to_the_bottom_edge_with_a_solid_background_in_both_styles():
    css = CSS()
    media = re.search(r"@media \(max-width: 900px\) \{\n  \.spend-meter \{([^}]*)\}", css)
    assert media and "position: fixed" in media.group(1) and "bottom: 8px" in media.group(1)
    assert re.search(r"@media \(max-width: 900px\) \{[^}]*\n  html\[data-mode=\"operator\"\] \.spend-meter \{ background: #092221; \}", css)
    assert re.search(r"body \.rail \{ padding-bottom: 6\.5rem; \}", css), "room under the pinned meter"
    # the operator tone rules come first in the file order of specificity: warn and bad keep their own backgrounds
    assert css.index('.spend-meter[data-tone="bad"] { background: #b3290a') < css.index("html[data-mode=\"operator\"] .spend-meter { background: #092221; }")


def test_the_run_spend_line_is_white_on_the_operator_state_band_and_the_band_is_aa_for_white():
    css = CSS()
    assert re.search(r'^html\[data-mode="operator"\] \.run-head \.run-spend \{ color: #fff; \}', css, re.M)
    for scheme in ("light", "dark"):
        t = operator_tokens(scheme)
        for name in ("running", "good", "warn", "bad", "dim"):
            for suffix in ("", "-2"):
                assert ratio("#ffffff", t[f"--band-{name}{suffix}"]) >= 4.5
