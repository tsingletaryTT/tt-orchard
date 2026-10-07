"""Terminal styling for people: colour and emoji, only where a person is looking at a terminal.

The rule: a pipe, a file, a dumb terminal or `--style plain` gets plain text. A small local model that
reads `status` through a tool is in that group, so its input never changes. Nothing here is used by
`--json`.

Colours are the Tenstorrent docs-site palette (CLAUDE.md, "Tenstorrent brand colors"): the primary accent,
the teal, green, yellow and red tints and the orange accent. Depth is detected, not assumed: 24-bit when
COLORTERM says so, 256 colours when TERM says so, otherwise none. `NO_COLOR` (https://no-color.org) turns
colour off and leaves emoji alone; `ORCHARD_PLAIN=1` turns both off for `auto`.

Boxes drawn by the views use a left bar and a bottom bar only. A right-hand border breaks when the
terminal is narrower than the text, so nothing here draws one.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# role -> hex, from the docs-site theme. `dim` is a lighter tint of the secondary text colour (#3A5452) so
# it stays readable on a dark terminal.
ROLES = {
    "title": "#74C5DF",     # teal
    "accent": "#1B8EB1",    # primary accent
    "good": "#6FABA0",      # green
    "warn": "#F6BC42",      # yellow
    "bad": "#FF9E8A",       # red
    "alarm": "#FA512E",     # orange / red accent
    "dim": "#7F9A98",
}

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STYLES = ("auto", "pretty", "plain")


def stream_is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _xterm256(r: int, g: int, b: int) -> int:
    """The nearest colour in the 6x6x6 cube of the 256-colour palette."""
    return 16 + 36 * round(r / 255 * 5) + 6 * round(g / 255 * 5) + round(b / 255 * 5)


@dataclass(frozen=True)
class Style:
    color: str = "none"      # "truecolor", "256" or "none"
    emoji: bool = False
    pretty: bool = True      # False when detection decided the reader gets the plain view

    def paint(self, text: str, role: str) -> str:
        hex_color = ROLES[role]                   # an unknown role is a bug, not a default colour
        if self.color == "none":
            return text
        r, g, b = _rgb(hex_color)
        if self.color == "truecolor":
            return f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m"
        return f"\x1b[38;5;{_xterm256(r, g, b)}m{text}\x1b[0m"

    def icon(self, emoji: str) -> str:
        return emoji + " " if self.emoji else ""


PLAIN = Style("none", False, pretty=False)


def _utf8(stream) -> bool:
    return "utf" in (getattr(stream, "encoding", None) or "").lower()


def _depth(env) -> str:
    if env.get("NO_COLOR"):
        return "none"
    if env.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    if "256color" in env.get("TERM", ""):
        return "256"
    return "none"


def detect(stream, env, style: str = "auto") -> Style:
    """The style for `stream`. `plain` never decorates. `auto` decorates only on a capable terminal.
    `pretty` decorates whatever the stream is, but still honours NO_COLOR and a stream that is not UTF-8."""
    if style == "plain":
        return PLAIN
    if style == "auto" and (not stream_is_tty(stream) or env.get("ORCHARD_PLAIN")
                            or env.get("TERM", "") == "dumb"):
        return PLAIN
    return Style(_depth(env), _utf8(stream))


def visible_width(text: str) -> int:
    """Terminal columns the text takes: escapes count 0, wide characters and emoji count 2, and a
    variation selector or a joiner does not add a column of its own."""
    width, last, joined = 0, 0, False
    for c in ANSI.sub("", text):
        if c == "‍":
            joined = True
            continue
        if c == "️":
            width += 2 - last if last == 1 else 0
            last = 2
            continue
        if unicodedata.combining(c):
            continue
        w = 2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
        if joined:
            joined, last = False, 0      # the joined character shares the previous cell
            continue
        width += w
        last = w
    return width
