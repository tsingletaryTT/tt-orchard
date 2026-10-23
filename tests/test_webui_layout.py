# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where each control of the browser page sits (orchard/web/index.html, app.js, mode.js, app.css).

The page's JavaScript has no automated tests (issue 31), so these tests read the markup as a tree and read the script and
the stylesheet as text. They pin the placement rules from the 2026-10-23 design review: a control's scope matches its
container (the whole page in the top bar, one view's controls in that view, one run's controls in that run's header, one
pane's controls in that pane), and the two page styles (playful, operator) both follow them. Behaviour that needs a browser
is checked by hand with screenshots, and the log entry says so.
"""
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "orchard" / "web"
VOID = {"meta", "link", "br", "img", "input", "hr"}


class Node:
    def __init__(self, tag, attrs, parent):
        self.tag, self.attrs, self.parent, self.kids, self.text = tag, dict(attrs), parent, [], ""

    @property
    def classes(self):
        return set((self.attrs.get("class") or "").split())

    def walk(self):
        yield self
        for k in self.kids:
            yield from k.walk()

    def all_text(self):
        return " ".join((self.text + " " + " ".join(k.all_text() for k in self.kids)).split())

    def inside(self, node):
        p = self.parent
        while p is not None:
            if p is node:
                return True
            p = p.parent
        return False


class Tree(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", [], None)
        self.cur = self.root
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        n = Node(tag, attrs, self.cur)
        self.cur.kids.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.text += data


def page():
    return Tree((WEB / "index.html").read_text(encoding="utf-8"))


def by_id(tree, ident):
    for n in tree.root.walk():
        if n.attrs.get("id") == ident:
            return n
    raise AssertionError(f"no element with id {ident!r} in index.html")


def has_id(tree, ident):
    return any(n.attrs.get("id") == ident for n in tree.root.walk())


def order(tree, *ids):
    """The ids in document order, so a test can say which comes first."""
    seen = [n.attrs["id"] for n in tree.root.walk() if n.attrs.get("id") in ids]
    return seen


JS = lambda: (WEB / "app.js").read_text(encoding="utf-8")
CSS = lambda: (WEB / "app.css").read_text(encoding="utf-8")


# ---- the page style switch is a setting of the page, and does not look like navigation ---------------------------------------

def test_the_style_switch_does_not_use_the_view_button_class():
    t = page()
    sw = by_id(t, "mode-switch")
    buttons = [n for n in sw.walk() if n.tag == "button"]
    assert [b.attrs["data-mode"] for b in buttons] == ["playful", "operator"]
    for b in buttons:
        assert "view-btn" not in b.classes, "the style switch must not be styled like the view tabs"
        assert "style-btn" in b.classes
    assert "style-switch" in sw.classes and "view-btn" not in sw.classes


def test_the_view_tabs_are_the_only_view_buttons():
    t = page()
    views = by_id(t, "view-orchard-btn").parent
    tabs = [n for n in t.root.walk() if "view-btn" in n.classes]
    assert {n.attrs["id"] for n in tabs} == {"view-orchard-btn", "view-hardware-btn", "view-settings-btn"}
    assert all(n.inside(views) for n in tabs)


def test_the_style_switch_has_a_visible_label_and_stays_a_labelled_group_with_pressed_buttons():
    t = page()
    sw = by_id(t, "mode-switch")
    assert sw.attrs.get("role") == "group" and sw.attrs.get("aria-label")
    label = [n for n in sw.walk() if "style-label" in n.classes]
    assert len(label) == 1 and label[0].all_text() == "Style"
    for b in (n for n in sw.walk() if n.tag == "button"):
        assert b.attrs.get("aria-pressed") in ("true", "false") and b.attrs.get("type") == "button"
    assert "aria-hidden" not in label[0].attrs, "the label is visible to everyone"


def test_the_top_bar_order_is_chip_clock_style_then_the_primary_action():
    t = page()
    right = by_id(t, "machine-chip").parent
    assert "topbar-right" in right.classes
    kids = [k for k in right.kids]
    ids_or_classes = [k.attrs.get("id") or sorted(k.classes)[0] for k in kids]
    assert ids_or_classes == ["machine-chip", "clock", "mode-switch", "new-run-btn"]


def test_the_style_switch_has_its_own_rules_in_both_styles():
    css = CSS()
    assert re.search(r"^\.style-switch \{", css, re.M) and re.search(r"^\.style-btn \{", css, re.M)
    assert re.search(r'^\.style-btn\[aria-pressed="true"\]', css, re.M)
    assert re.search(r'^html\[data-mode="operator"\] \.style-btn \{', css, re.M)
    assert re.search(r'^html\[data-mode="operator"\] \.style-btn\[aria-pressed="true"\] \{', css, re.M)


def test_the_style_script_still_finds_its_buttons():
    js = (WEB / "mode.js").read_text(encoding="utf-8")
    assert "#mode-switch button[data-mode]" in js


# ---- the machine's state is visible from the top bar, and the rail panel no longer yanks you to another view --------------------

def test_the_top_bar_has_a_machine_chip_that_opens_the_machine_view():
    t = page()
    chip = by_id(t, "machine-chip")
    assert chip.tag == "button" and "machine-chip" in chip.classes and chip.attrs.get("type") == "button"
    assert chip.attrs.get("data-tone") in ("good", "warn", "bad", "unknown")
    assert re.search(r'\$\("machine-chip"\)\.addEventListener\("click",\s*\(\)\s*=>\s*setView\("hardware"\)\)', JS())


def test_the_chip_text_is_built_from_the_same_data_as_the_rail_panel():
    js = JS()
    body = js[js.index("function renderMachineChip"):]
    body = body[:body.index("\n}\n")]
    assert "chips free" in body and "coder" in body
    for tone in ('"good"', '"warn"', '"bad"'):
        assert tone in body
    assert 'dataset.tone' in body or 'data-tone' in body
    rm = js[js.index("function renderMachine()"):]
    rm = rm[:rm.index("\n}\n")]
    assert "renderMachineChip(m, all, free)" in rm, "renderMachine must feed the chip"


def test_the_chip_has_tone_rules_in_both_styles():
    css = CSS()
    for tone in ("good", "warn", "bad", "unknown"):
        assert re.search(rf'\.machine-chip\[data-tone="{tone}"\]', css), tone
    assert re.search(r'^html\[data-mode="operator"\] \.machine-chip \{', css, re.M)


def test_the_rail_machine_panel_has_a_details_link_and_no_check_button():
    t = page()
    assert not has_id(t, "machine-check-btn")
    link = by_id(t, "machine-details-link")
    assert link.all_text() == "Details ›" and link.tag == "a"
    head = link.parent
    assert "rail-head" in head.classes and by_id(t, "machine-h").parent is head
    assert "machine-check-btn" not in JS()
    assert re.search(r'\$\("machine-details-link"\)\.addEventListener\("click"[^\n]*setView\("hardware"\)', JS())


def test_opening_the_machine_view_from_the_rail_does_not_start_the_checks():
    js = JS()
    handler = js[js.index('$("machine-details-link")'):]
    handler = handler[:handler.index("\n")]
    assert "startChecks" not in handler and "preventDefault" in handler
