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


# ---- views: Orchard/Runs, Machine and Models; the diagnostics live with the machine, and nothing follows Save ----------------------

def label_spans(node):
    return {c: n.all_text() for n in node.walk() for c in ("lbl-playful", "lbl-operator") if c in n.classes}


def test_the_machine_and_models_tabs_are_named_for_what_they_hold():
    t = page()
    machine, models, first = by_id(t, "view-hardware-btn"), by_id(t, "view-settings-btn"), by_id(t, "view-orchard-btn")
    assert "Machine" in machine.all_text() and "Hardware" not in machine.all_text()
    assert "Models" in models.all_text() and "Settings" not in models.all_text()
    assert label_spans(first) == {"lbl-playful": "Orchard", "lbl-operator": "Runs"}, "playful says Orchard, operator says Runs"
    assert by_id(t, "hw-h").all_text() == "Machine" and by_id(t, "settings-h").all_text() == "Models"


def test_the_first_tab_label_follows_the_page_style_in_the_stylesheet():
    css = CSS()
    assert re.search(r'^html\[data-mode="operator"\] \.lbl-playful \{ display: none; \}', css, re.M)
    assert re.search(r'^html:not\(\[data-mode="operator"\]\) \.lbl-operator \{ display: none; \}', css, re.M)


def test_the_checks_and_the_caches_are_in_the_machine_view_and_not_in_models():
    t = page()
    machine, models = by_id(t, "hardware-view"), by_id(t, "settings-view")
    inside_models = {n.attrs["id"] for n in models.walk() if n.attrs.get("id")}
    for ident in ("checks-h", "checks-run", "checks-boxes", "caches-h", "caches-run", "caches-rows", "caches-lab"):
        assert by_id(t, ident).inside(machine), ident
        assert ident not in inside_models, f"{ident} is still in the Models view"


def test_every_id_in_the_page_is_unique():
    ids = [n.attrs["id"] for n in page().root.walk() if n.attrs.get("id")]
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})


def test_nothing_follows_the_save_button_in_the_models_view():
    t = page()
    models = by_id(t, "settings-view")
    form = by_id(t, "settings-form")
    direct = [k for k in models.kids if k.tag in ("section", "form", "div", "p")]
    assert direct[-1] is form, "the form is the last block of the Models view"
    ids_in_form = [n.attrs["id"] for n in form.walk() if n.attrs.get("id")]
    assert ids_in_form.index("settings-save") > ids_in_form.index("settings-roles")
    after_save = ids_in_form[ids_in_form.index("settings-save") + 1:]
    assert set(after_save) <= {"settings-files", "settings-error"}, after_save


def test_the_view_hashes_are_machine_and_models_and_the_old_ones_still_open_them():
    js = JS()
    assert 'replaceState(null, "", "#machine")' in js and 'replaceState(null, "", "#models")' in js
    boot = js[js.index('location.hash === "#machine"'):] if 'location.hash === "#machine"' in js else ""
    assert boot, "the start-up code reads #machine"
    for old in ('"#hardware"', '"#settings"', '"#models"'):
        assert old in js, old
    assert 'href="#machine"' in (WEB / "index.html").read_text(encoding="utf-8")


def test_opening_the_machine_view_reads_the_last_check_and_models_does_not():
    js = JS()
    sv = js[js.index("function setView"):]
    sv = sv[:sv.index("\n}\n")]
    hw = sv[sv.index('if (view === "hardware")'):sv.index("stopToplike();\n  document.body")]
    assert "pollChecks()" in hw
    ls = js[js.index("async function loadSettings"):]
    ls = ls[:ls.index("\n}\n")]
    assert "pollChecks" not in ls


# ---- the New bring-up dialog: the button you can press is the main one; the rarely changed options are folded -----------------

def fn_body(js, header):
    start = js.index(header)
    return js[start:js.index("\n}\n", start)]


def test_run_checks_is_the_primary_button_before_the_checks_pass_and_start_is_not():
    t = page()
    assert "btn-primary" in by_id(t, "nr-check").classes
    start = by_id(t, "nr-start")
    assert "btn-primary" not in start.classes and "disabled" in start.attrs


def test_one_function_moves_the_primary_style_and_the_disabled_state_together():
    js = JS()
    body = fn_body(js, "function setStartReady")
    assert '$("nr-start").disabled = !ready' in body
    assert '$("nr-start").classList.toggle("btn-primary", ready)' in body
    assert '$("nr-check").classList.toggle("btn-primary", !ready)' in body
    for header in ("function openNewRun", "async function runChecks", "function renderPreflight"):
        assert "setStartReady(" in fn_body(js, header), header
    for header in ("function openNewRun", "async function runChecks", "function renderPreflight"):
        assert '$("nr-start").disabled' not in fn_body(js, header).replace('$("nr-start").disabled = !ready', ""), header
    wire = fn_body(js, "function wire")
    assert 'setStartReady(false)' in wire[wire.index('$("nr-mode").addEventListener'):wire.index('$("nr-mode").addEventListener') + 300]


def test_base_where_and_research_sit_in_a_closed_options_disclosure_under_the_model():
    t = page()
    opts = by_id(t, "nr-options")
    assert opts.tag == "details" and "open" not in opts.attrs
    for ident in ("nr-base", "nr-mode", "nr-research"):
        assert by_id(t, ident).inside(opts), ident
    assert not by_id(t, "nr-model").inside(opts)
    summary = by_id(t, "nr-options-summary")
    assert summary.tag == "summary" and summary.all_text().startswith("Options:")
    assert order(t, "nr-model", "nr-options", "nr-error", "nr-check") == ["nr-model", "nr-options", "nr-error", "nr-check"]


def test_the_options_summary_states_the_current_values_and_follows_every_option():
    js = JS()
    body = fn_body(js, "function updateOptionsSummary")
    for word in ("this box", "the lab box", "full research", "no research", "research without GitHub"):
        assert word in body, word
    assert 'value === "lab" ? "the lab box" : "this box"' in body
    assert '${base ? `, base ${base}` : ""}' in body
    assert "updateOptionsSummary()" in fn_body(js, "function openNewRun")
    wire = fn_body(js, "function wire")
    for ident in ("nr-mode", "nr-research"):
        assert re.search(rf'\$\("{ident}"\)\.addEventListener\("change"[^\n]*updateOptionsSummary\(\)', wire), ident
    assert re.search(r'for \(const id of \["nr-model", "nr-base"\]\) \$\(id\)\.addEventListener\("input"[^\n]*updateOptionsSummary\(\)', wire)


def test_a_bad_base_opens_the_options_before_it_takes_focus():
    body = fn_body(JS(), "async function runChecks")
    i = body.index('$("nr-base").focus()')
    assert '$("nr-options").open = true' in body[:i]


def test_the_where_tests_run_label_says_it_applies_to_this_run_and_where_the_default_is_set():
    t = page()
    labels = [n for n in t.root.walk() if n.tag == "label" and n.attrs.get("for") == "nr-mode"]
    assert len(labels) == 1
    text = labels[0].all_text()
    assert "this run only" in text and "Models" in text and text.startswith("Where the hardware tests run")


# ---- a run's actions: where the button is, what it is called, and that the page and the server agree on which runs can act ------

def test_the_blocked_panel_ends_its_unblock_list_with_a_retry_button():
    t = page()
    panel = by_id(t, "block-panel")
    btn = by_id(t, "block-retry")
    assert btn.inside(panel) and btn.tag == "button" and btn.all_text() == "Retry run" and "btn-primary" in btn.classes
    assert btn.parent is by_id(t, "block-unblock").parent, "the button sits in the same column as the unblock steps"
    assert order(t, "block-unblock", "block-retry") == ["block-unblock", "block-retry"]


def test_the_page_and_the_server_name_the_same_states_as_retryable():
    from orchard import webui
    m = re.search(r"const RETRY_STATES = \[([^\]]*)\];", JS())
    assert m, "app.js has a RETRY_STATES list"
    assert set(re.findall(r'"([a-z-]+)"', m.group(1))) == set(webui.RETRYABLE)


def test_the_retry_button_is_shown_only_for_a_retryable_run_and_runs_the_same_handler_as_the_header_one():
    js = JS()
    assert '$("block-retry").addEventListener("click", doRetry)' in js
    block = fn_body(js, "function renderBlock")
    assert '$("block-retry").hidden = !RETRY_STATES.includes(d.state)' in block
    actions = fn_body(js, "function renderActions")
    assert "RETRY_STATES.includes(d.state)" in actions


def test_an_aborted_run_is_resumed_with_a_button_that_says_so_and_a_confirmation_that_says_what_it_does():
    js = JS()
    actions = fn_body(js, "function renderActions")
    assert '"aborted" ? "Resume run" : "Retry run"' in actions
    retry = fn_body(js, "async function doRetry")
    for words in ("tt-orchard bringup", "same model", "Resume this run?"):
        assert words in retry, words
    assert "resumes from the ledger, where the abort left it" in retry
    assert 'S.detail && S.detail.state === "aborted"' in retry


def test_both_buttons_that_open_the_results_file_have_one_name_and_one_handler():
    js = JS()
    assert "Open the bundle" not in js and "Read RESULTS.md" not in js
    assert js.count('"Open RESULTS.md"') == 2
    assert "function openResults()" in js
    assert js.count("openResults") >= 3          # the definition and the two buttons


def test_abort_sits_apart_from_the_routine_actions():
    js = JS()
    actions = fn_body(js, "function renderActions")
    assert actions.count('"btn-danger btn-apart"') == 2           # the running and the paused row
    assert re.search(r"^\.actions \.btn-apart \{[^}]*margin-left", CSS(), re.M)


def test_the_next_hints_use_the_names_on_the_buttons():
    from orchard import run_view
    assert "Resume run" in run_view.NEXT["aborted"] and "ledger" in run_view.NEXT["aborted"]
    assert "Start a new bring-up to try again" not in run_view.NEXT["aborted"]
    assert "Open RESULTS.md" in run_view.NEXT["ready-for-operator-review"]
    assert "open the bundle" not in run_view.NEXT["ready-for-operator-review"].lower()
    assert "Retry" in run_view.NEXT["blocked"] and "Retry" in run_view.NEXT["stopped-or-crashed"]


# ---- Operator style is plain text: every emoji the page can show sits in a span that style hides ----------------------------------

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⏩-⏿⬀-⯿✅✨]")
TYPOGRAPHIC = set("✓✕✗•")          # the check and cross marks next to the words pass and fail are plain glyphs


def pictures(text):
    return [c for c in EMOJI.findall(text) if c not in TYPOGRAPHIC]


def test_every_emoji_in_the_markup_is_inside_a_span_the_operator_style_hides():
    t = page()
    for n in t.root.walk():
        if pictures(n.text):
            hidden = "emoji" in n.classes or "brand-mark" in n.classes or any("emoji" in a.classes for a in _ancestors(n))
            assert hidden, f"<{n.tag} id={n.attrs.get('id')}> shows {pictures(n.text)} outside an .emoji span"


def _ancestors(n):
    p = n.parent
    while p is not None:
        yield p
        p = p.parent


def test_every_emoji_the_script_adds_is_in_an_emoji_span_or_in_the_scene_caption_that_operator_style_removes():
    lines = JS().split("\n")
    def span(start_marker, end_marker):
        a = next(i for i, l in enumerate(lines) if l.startswith(start_marker))
        b = next(i for i, l in enumerate(lines) if i > a and l.startswith(end_marker))
        return range(a, b + 1)
    scene = {i for i, l in enumerate(lines) if l.startswith("const WEATHER_WORD")} | set(span("const WEATHER = {", "function renderDetails"))
    assert len(scene) > 3, "the scene caption region was found"
    bad = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("//") or not pictures(line) or i in scene:
            continue
        if 'class: "emoji"' not in line:
            bad.append(f"app.js:{i + 1}: {line.strip()[:100]}")
    assert not bad, "\n".join(bad)


def test_the_mode_chip_the_stepper_marks_and_the_unreadable_badge_put_their_pictures_in_emoji_spans():
    js = JS()
    mode = js[js.index('const mode = $("run-mode");'):]
    mode = mode[:mode.index("\n  S.activity")]
    assert mode.count('class: "emoji"') >= 2 and "replaceChildren" in mode
    step = fn_body(js, "function renderStepper")
    assert 'const mark = h("span", { class: "emoji", "aria-hidden": "true" }, `${stopped ? "⏹" : S.meta.marks[status] || ""} `);' in step
    assert re.search(r'\.\.\.\(r \? \[mark,\s*`\$\{status\}', step)
    assert 'h("span", { class: "emoji", "aria-hidden": "true" }, "⚠ "), "unreadable"' in fn_body(js, "function stateBadge")


def test_operator_style_hides_every_emoji_span():
    css = CSS()
    assert re.search(r'^html\[data-mode="operator"\] \.emoji \{ display: none; \}', css, re.M)
