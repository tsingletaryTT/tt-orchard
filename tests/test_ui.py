# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Terminal capability and styling (orchard/ui.py).

The rule under test: colour and emoji appear only when a person is looking at a terminal that can show
them. Anything else (a pipe, a file, a small model reading status through a tool) gets plain text.
"""
import io

import pytest

from orchard import ui


class Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


class Pipe(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return False


TRUECOLOR = {"TERM": "xterm-256color", "COLORTERM": "truecolor"}


def test_a_pipe_gets_plain_text_even_when_the_environment_is_colourful():
    s = ui.detect(Pipe(), TRUECOLOR, "auto")
    assert (s.color, s.emoji) == ("none", False)


def test_a_terminal_gets_colour_and_emoji():
    s = ui.detect(Tty(), TRUECOLOR, "auto")
    assert (s.color, s.emoji) == ("truecolor", True)


def test_no_color_turns_colour_off_and_keeps_emoji():
    s = ui.detect(Tty(), {**TRUECOLOR, "NO_COLOR": "1"}, "auto")
    assert (s.color, s.emoji) == ("none", True)


def test_orchard_plain_turns_both_off():
    s = ui.detect(Tty(), {**TRUECOLOR, "ORCHARD_PLAIN": "1"}, "auto")
    assert (s.color, s.emoji) == ("none", False)


def test_a_dumb_terminal_gets_plain_text():
    s = ui.detect(Tty(), {"TERM": "dumb"}, "auto")
    assert (s.color, s.emoji) == ("none", False)


def test_a_terminal_that_is_not_utf8_gets_no_emoji():
    t = Tty()
    t.encoding = "ascii"
    assert ui.detect(t, TRUECOLOR, "auto").emoji is False


def test_style_plain_wins_over_a_terminal():
    s = ui.detect(Tty(), TRUECOLOR, "plain")
    assert (s.color, s.emoji) == ("none", False)


def test_style_pretty_forces_emoji_on_a_pipe_but_still_honours_no_color():
    s = ui.detect(Pipe(), {**TRUECOLOR, "NO_COLOR": "1"}, "pretty")
    assert (s.color, s.emoji) == ("none", True)
    assert ui.detect(Pipe(), TRUECOLOR, "pretty").color == "truecolor"


def test_colour_depth_follows_the_environment():
    assert ui.detect(Tty(), {"TERM": "xterm-256color"}, "auto").color == "256"
    assert ui.detect(Tty(), {"TERM": "xterm"}, "auto").color == "none"


def test_truecolor_paint_uses_the_brand_green():
    s = ui.Style(color="truecolor", emoji=True)
    assert s.paint("ok", "good") == "\x1b[38;2;111;171;160mok\x1b[0m"      # #6FABA0


def test_256_colour_paint_uses_a_256_colour_escape():
    out = ui.Style(color="256", emoji=True).paint("ok", "good")
    assert out.startswith("\x1b[38;5;") and out.endswith("ok\x1b[0m")


def test_paint_without_colour_returns_the_text_unchanged():
    assert ui.Style(color="none", emoji=True).paint("ok", "good") == "ok"


def test_an_unknown_role_is_an_error_not_a_silent_default():
    with pytest.raises(KeyError):
        ui.Style(color="truecolor", emoji=True).paint("x", "no-such-role")


def test_icon_is_empty_without_emoji_and_padded_with_it():
    assert ui.Style(color="none", emoji=False).icon("🍎") == ""
    assert ui.Style(color="none", emoji=True).icon("🍎") == "🍎 "


def test_visible_width_ignores_escapes_and_counts_emoji_as_two_columns():
    assert ui.visible_width("\x1b[38;2;1;2;3mab\x1b[0m") == 2
    assert ui.visible_width("🍎a") == 3
