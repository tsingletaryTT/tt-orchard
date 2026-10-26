# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Operator style is the default, and it carries a small set of deliberate emoji and stronger colour (2026-10-26).

Before this change Operator hid every `.emoji` span. Now it keeps one picture for each run state, stage mark, verdict tone and
primary button, in spans of class `op-pic` that Playful never shows. The pictures are decoration: aria-hidden, with the real
word printed beside each. The colour tokens are checked for AA contrast (4.5:1) in the light and the dark scheme.
"""
import re
from pathlib import Path

from orchard import lexicon, status

WEB = Path(__file__).resolve().parent.parent / "orchard" / "web"
JS = lambda: (WEB / "app.js").read_text(encoding="utf-8")
CSS = lambda: (WEB / "app.css").read_text(encoding="utf-8")
MODE = lambda: (WEB / "mode.js").read_text(encoding="utf-8")
HTML = lambda: (WEB / "index.html").read_text(encoding="utf-8")


def fn_body(js, header):
    start = js.index(header)
    return js[start:js.index("\n}\n", start)]


# ---- the lexicon tables ----------------------------------------------------------------------------------------------

def test_every_state_has_one_operator_emoji_and_they_differ():
    assert set(lexicon.OPERATOR_STATES) == set(status.STATES)
    assert len(set(lexicon.OPERATOR_STATES.values())) == len(lexicon.OPERATOR_STATES)


def test_every_stage_mark_has_an_operator_emoji_including_pending_and_stopped():
    assert set(lexicon.OPERATOR_MARKS) == set(lexicon.STATUS_MARKS) | {"pending", "stopped"}
    assert len(set(lexicon.OPERATOR_MARKS.values())) == len(lexicon.OPERATOR_MARKS)


def test_verdict_tones_and_primary_buttons_have_emoji():
    from orchard import verdict
    assert set(lexicon.OPERATOR_TONES) == set(verdict.TONES)
    assert set(lexicon.OPERATOR_BUTTONS) == {"pause", "resume", "retry", "abort"}


def test_the_page_meta_carries_the_operator_tables(tmp_path):
    from test_webui import make_app
    root = tmp_path / "runs"
    root.mkdir()
    op = make_app(root).meta()["op"]
    assert op["states"] == lexicon.OPERATOR_STATES and op["marks"] == lexicon.OPERATOR_MARKS
    assert op["tones"] == lexicon.OPERATOR_TONES and op["buttons"] == lexicon.OPERATOR_BUTTONS


# ---- the default style ------------------------------------------------------------------------------------------------

def test_operator_is_the_default_style_and_a_link_or_stored_choice_still_wins():
    js = MODE()
    assert re.search(r'fromLink\s*\|\|\s*valid\(stored\(\)\)\s*\|\|\s*"operator"', js)
    assert '|| "playful"' not in js


# ---- the pictures -------------------------------------------------------------------------------------------------------

def test_op_pic_is_hidden_by_default_and_shown_only_in_operator_style():
    css = CSS()
    assert re.search(r'^\.op-pic \{ display: none; \}', css, re.M)
    assert re.search(r'^html\[data-mode="operator"\] \.op-pic \{ display: inline;', css, re.M)
    # the orchard emoji spans stay hidden in operator style, as before
    assert re.search(r'^html\[data-mode="operator"\] \.emoji \{ display: none; \}', css, re.M)


def test_the_state_badge_adds_an_aria_hidden_operator_emoji_beside_the_real_name():
    body = fn_body(JS(), "function stateBadge")
    assert re.search(r'h\("span", \{ class: "op-pic", "aria-hidden": "true" \}, [^)]*op\.states\[state\]', body)
    assert 'class: "name"' in body                                   # the real state word is still printed


def test_the_stepper_adds_an_operator_mark_for_every_status_including_pending_and_stopped():
    body = fn_body(JS(), "function renderStepper")
    assert re.search(r'class: "op-pic", "aria-hidden": "true" \}, `\$\{S\.meta\.op\.marks\[status\]', body)
    assert '[opMark, "not yet"]' in body                              # a pending stage has a picture too
    assert "status}" in body or "`${status}" in body                  # the status word stays


def test_verdict_lines_get_a_tone_emoji_before_the_headline_and_the_headline_stays_text():
    body = fn_body(JS(), "function renderVerdict")
    assert re.search(r'op-pic[^\n]*op\.tones\[i\.tone\]', body)
    assert 'h("strong", { class: "verdict-headline" }, i.headline)' in body


def test_pause_resume_retry_and_abort_buttons_get_an_emoji_and_keep_their_label():
    body = fn_body(JS(), "function renderActions")
    assert re.search(r'const add = \(label, cls, fn, title, pic\)', body)
    assert 'h("span", { class: "op-pic", "aria-hidden": "true" }, `${S.meta.op.buttons[pic]} `)' in body
    for key, label in (("pause", "Pause"), ("resume", "Resume"), ("abort", "Abort")):
        assert re.search(rf'add\("{label}"[^\n]*"{key}"\)', body), key
    assert body.count('"retry")') >= 1


def test_empty_state_is_friendlier_in_operator_style_and_unchanged_in_playful():
    html = HTML()
    empty = html[html.index('id="run-empty"'):html.index('id="run"')]
    assert "Pick a run" in empty
    assert 'class="op-pic"' in empty and 'aria-hidden="true"' in empty
    assert "lbl-operator" in empty


# ---- motion ---------------------------------------------------------------------------------------------------------------

def test_the_stage_pop_is_inside_a_no_preference_motion_query_and_does_not_loop():
    css = CSS()
    m = re.search(r'@media \(prefers-reduced-motion: no-preference\) \{(.*?)\n\}\n', css[css.index('html[data-mode="operator"] .run-head .mode-chip'):], re.S)
    assert m, "the operator motion block exists"
    block = m.group(1)
    assert "orchard-pop" in block and "data-fresh" in block
    pop = re.search(r'animation:\s*orchard-pop[^;]*;', block).group(0)
    assert "infinite" not in pop
    # no animation anywhere else outside a query
    outside = css.replace(block, "")
    assert "orchard-pop" not in outside.replace("@keyframes orchard-pop", "")


def test_only_a_stage_that_just_passed_is_marked_fresh():
    body = fn_body(JS(), "function renderStepper")
    assert 'fresh' in body and "S.stepSeen" in body
    assert re.search(r'prev\s*&&\s*prev\[s\.number\]\s*!==\s*"pass"\s*&&\s*status\s*===\s*"pass"', body)


# ---- colour: stronger, accessible ------------------------------------------------------------------------------------

def _lum(hex_):
    h = hex_.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _tokens(scheme):
    css = CSS()
    light = css[css.index('html[data-mode="operator"] {'):]
    light = light[:light.index("\n}\n")]
    dark = css[css.index("@media (prefers-color-scheme: dark) {\n  html[data-mode=\"operator\"] {"):]
    dark = dark[:dark.index("\n  }\n")]
    block = light if scheme == "light" else light + dark          # dark overrides light token by token
    out = {}
    for name, val in re.findall(r"(--[a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", block):
        out[name] = val
    return out


import pytest


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_state_band_colours_give_white_text_at_least_aa_at_both_ends(scheme):
    t = _tokens(scheme)
    for name in ("running", "good", "warn", "bad", "dim"):
        for suffix in ("", "-2"):
            c = t[f"--band-{name}{suffix}"]
            assert ratio("#ffffff", c) >= 4.5, (scheme, name, suffix, c, ratio("#ffffff", c))


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_verdict_tones_give_ink_and_muted_text_at_least_aa(scheme):
    t = _tokens(scheme)
    for name in ("good", "warn", "bad", "info"):
        bg = t[f"--v-{name}-bg"]
        assert ratio(t["--ink"], bg) >= 4.5, (scheme, name, "ink")
        assert ratio(t["--muted"], bg) >= 4.5, (scheme, name, "muted")
        assert ratio(t[f"--v-{name}"], bg) >= 4.5, (scheme, name, "edge text")


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_badge_and_stage_tones_are_aa_on_their_backgrounds(scheme):
    t = _tokens(scheme)
    for name in ("good", "warn", "bad", "alarm", "dim"):
        assert ratio(t[f"--{name}"], t[f"--{name}-bg"]) >= 4.5, (scheme, name)
        assert ratio(t[f"--{name}"], t["--paper"]) >= 4.5, (scheme, name, "on paper")


def test_the_state_band_has_a_soft_gradient_accent_and_verdicts_use_the_stronger_tokens():
    css = CSS()
    assert re.search(r'#run\[data-state="running"\] \.run-head \{[^}]*linear-gradient\([^}]*--band-running', css)
    assert re.search(r'#verdict \.verdict-item\[data-tone="good"\] \{[^}]*var\(--v-good-bg\)', css)
    assert re.search(r'#verdict \.verdict-item\[data-tone="info"\] \{[^}]*var\(--v-info-bg\)', css)


def _composite(fg_rgba, bg_hex):
    r, g, b, a = fg_rgba
    h = bg_hex.lstrip("#")
    bg = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(c * a + k * (1 - a)):02x}" for c, k in zip((r, g, b), bg))


def _band_text_colours():
    """Every text colour the operator CSS puts on the state band, with its alpha: the real colour, not opaque white."""
    css = CSS()
    out = {}
    for sel in (".run-head .run-sub", ".run-head h1", ".run-head .badge", ".run-head .mode-chip"):
        m = re.search(r'html\[data-mode="operator"\] ' + re.escape(sel) + r' \{[^}]*?(?<![-\w])color:\s*([^;]+);', css)
        assert m, sel
        v = m.group(1).strip()
        if v.startswith("rgba"):
            r, g, b, a = [float(x) for x in re.findall(r"[\d.]+", v)]
            out[sel] = (r, g, b, a)
        else:
            assert re.fullmatch(r"#[0-9a-fA-F]{6}|#fff", v), (sel, v)
            h = v.lstrip("#") * (2 if len(v) == 4 else 1)
            out[sel] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 1.0)
    return out


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_text_on_the_state_band_is_aa_with_its_alpha_over_the_lightest_stop(scheme):
    t = _tokens(scheme)
    for sel, rgba in _band_text_colours().items():
        for name in ("running", "good", "warn", "bad", "dim"):
            for suffix in ("", "-2"):
                bg = t[f"--band-{name}{suffix}"]
                seen = _composite(rgba, bg)
                assert ratio(seen, bg) >= 4.5, (scheme, sel, name, suffix, rgba, round(ratio(seen, bg), 2))
