"""The pretty status view (orchard/orchard_view.py) and the --style flag of the status command.

The plain text view is what the operator runbook and a small local model read, so it must not change.
"""
import io
import json
import re

import pytest

from orchard import lexicon, orchard_view, status, supervisor, ui
from orchard.ledger import Ledger
from orchard.stages import STAGES

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RIGHT_EDGE = "║│┃╗╣┐┤┘╝╯╮"


def facts(tmp_path, **override):
    run = tmp_path / "run"
    run.mkdir()
    with Ledger(run / "ledger.jsonl") as led:
        led.append("run_start", None, model="org/some-model", versions={}, inputs={})
        led.append("stage_start", 0)
        led.append("stage_end", 0, result="pass")
        led.append("stage_start", 1)
    f = status.collect(run, now=1_800_000_000, pid_alive=lambda p: True, lock_holder=lambda p: None,
                       disk_free_gb=lambda p: 200.0, gozer_status=lambda: "")
    f.update(override)
    return f, run


STYLES = {
    "plain": ui.Style("none", False),
    "emoji": ui.Style("none", True),
    "colour": ui.Style("truecolor", True),
}


@pytest.mark.parametrize("name", STYLES)
@pytest.mark.parametrize("state", status.STATES)
def test_no_line_has_a_right_hand_border_and_none_is_wider_than_80_columns(tmp_path, name, state):
    f, _ = facts(tmp_path, state=state)
    for line in orchard_view.render_pretty(f, STYLES[name]).splitlines():
        assert ui.visible_width(line) <= 80, line
        assert not ANSI.sub("", line).rstrip().endswith(tuple(RIGHT_EDGE)) or line.strip() in ("║", "╔", "╚"), line


@pytest.mark.parametrize("state", status.STATES)
def test_the_real_state_word_is_always_on_the_page(tmp_path, state):
    f, _ = facts(tmp_path, state=state)
    assert state in orchard_view.render_pretty(f, STYLES["emoji"])


def test_the_real_stage_names_are_on_the_page(tmp_path):
    f, _ = facts(tmp_path)
    page = orchard_view.render_pretty(f, STYLES["emoji"])
    for row in f["stages"]:
        assert row["name"] in page


def test_without_colour_there_are_no_escape_codes(tmp_path):
    f, _ = facts(tmp_path)
    assert "\x1b" not in orchard_view.render_pretty(f, STYLES["emoji"])
    assert "\x1b" not in orchard_view.render_pretty(f, STYLES["plain"])


def test_with_colour_there_are_escape_codes_and_stripping_them_gives_the_uncoloured_page(tmp_path):
    f, _ = facts(tmp_path)
    coloured = orchard_view.render_pretty(f, STYLES["colour"])
    assert "\x1b[38;2;" in coloured
    assert ANSI.sub("", coloured) == orchard_view.render_pretty(f, STYLES["emoji"])


def test_a_plain_style_page_has_no_emoji(tmp_path):
    f, _ = facts(tmp_path)
    page = orchard_view.render_pretty(f, STYLES["plain"])
    assert all(ord(c) < 0x2000 or c in "║╔╚═─" for c in page)


def squash(text):
    """Content without layout: wrapping may move a word to the next line, never change the words."""
    return " ".join(text.split())


def test_the_page_carries_the_next_step_and_the_model(tmp_path):
    f, _ = facts(tmp_path)
    page = orchard_view.render_pretty(f, STYLES["emoji"])
    assert squash(f["hint"]) in squash(page) and "org/some-model" in page


def test_a_hyphenated_name_is_never_split_across_lines(tmp_path):
    f, _ = facts(tmp_path, state="ready-for-operator-review", run_dir="/r/" + "x" * 50)
    for line in orchard_view.render_pretty(f, STYLES["emoji"]).splitlines():
        assert not line.rstrip().endswith("-"), line


def test_a_paused_run_shows_why(tmp_path):
    f, _ = facts(tmp_path, state="paused",
                 pause={"reason": "watchdog: no file written", "kind": "watchdog",
                        "detector": "no_file_written", "stage": 1})
    page = orchard_view.render_pretty(f, STYLES["emoji"])
    assert "no_file_written" in page and "stage 1" in page


def test_a_finished_run_ends_with_the_closing_line(tmp_path):
    f, run = facts(tmp_path, state="ready-for-operator-review")
    page = orchard_view.render_pretty(f, STYLES["emoji"])
    closing = lexicon.closing_line("ready-for-operator-review", STYLES["emoji"], where=f["run_dir"])
    assert squash(closing) in squash(page)


def test_the_gate_section_shows_each_lease_line(tmp_path):
    f, _ = facts(tmp_path, leases=["board 0000046131924062: 2 chips FREE"])
    assert "board 0000046131924062: 2 chips FREE" in orchard_view.render_pretty(f, STYLES["emoji"])


# ---- the command ------------------------------------------------------------------------------

def run_status(run, capsys, *args, env=None):
    rc = status.main(["--run-dir", str(run), *args])
    return rc, capsys.readouterr().out


def test_piped_output_is_byte_identical_to_the_plain_render(tmp_path, capsys, monkeypatch):
    f, run = facts(tmp_path)
    monkeypatch.setattr(status, "collect", lambda *a, **k: f)
    rc, out = run_status(run, capsys)                       # capsys stdout is not a terminal
    assert rc == 0 and out == status.render(f) + "\n"


def test_style_plain_equals_the_plain_render_even_with_a_terminal(tmp_path, capsys, monkeypatch):
    f, run = facts(tmp_path)
    monkeypatch.setattr(status, "collect", lambda *a, **k: f)
    monkeypatch.setattr(ui, "stream_is_tty", lambda s: True, raising=False)
    rc, out = run_status(run, capsys, "--style", "plain")
    assert out == status.render(f) + "\n"


def test_style_pretty_prints_the_pretty_page(tmp_path, capsys, monkeypatch):
    f, run = facts(tmp_path)
    monkeypatch.setattr(status, "collect", lambda *a, **k: f)
    monkeypatch.setenv("NO_COLOR", "1")
    rc, out = run_status(run, capsys, "--style", "pretty")
    assert rc == 0 and out.startswith("╔") and "org/some-model" in out


def test_json_is_never_styled(tmp_path, capsys, monkeypatch):
    f, run = facts(tmp_path)
    monkeypatch.setattr(status, "collect", lambda *a, **k: f)
    rc, out = run_status(run, capsys, "--json", "--style", "pretty")
    assert json.loads(out) == json.loads(json.dumps(f, sort_keys=True))
    assert "\x1b" not in out


def test_the_supervisor_status_subcommand_forwards_the_style_flag(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(status, "main", lambda argv: seen.append(argv) or 0)
    supervisor.main(["status", "--run-dir", str(tmp_path), "--style", "pretty"])
    assert seen == [["--run-dir", str(tmp_path), "--style", "pretty"]]


def test_the_supervisor_status_subcommand_omits_style_when_not_given(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(status, "main", lambda argv: seen.append(argv) or 0)
    supervisor.main(["status", "--run-dir", str(tmp_path)])
    assert seen == [["--run-dir", str(tmp_path)]]


def test_a_coloured_line_never_starts_with_an_escape_code(tmp_path):
    """A continuation row has no label to colour; painting its empty label left a stray escape in front
    of the left bar."""
    f, _ = facts(tmp_path, leases=["board a: 2 chips FREE", "board b: 2 chips FREE"])
    for line in orchard_view.render_pretty(f, STYLES["colour"]).splitlines():
        assert not line.startswith("\x1b"), repr(line)


def test_every_escape_code_is_closed_on_the_same_line(tmp_path):
    f, _ = facts(tmp_path, leases=["board a: 2 chips FREE", "board b: 2 chips FREE"])
    for line in orchard_view.render_pretty(f, STYLES["colour"]).splitlines():
        assert line.count("\x1b[0m") == len(re.findall(r"\x1b\[(?!0m)[0-9;]*m", line)), repr(line)


def test_a_run_with_no_model_yet_does_not_print_none(tmp_path):
    f, _ = facts(tmp_path, model=None, state="not-started")
    assert "None" not in orchard_view.render_pretty(f, STYLES["emoji"])
