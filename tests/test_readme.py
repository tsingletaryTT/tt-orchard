"""The README is checked against the code it describes.

A reader with no other source follows README.md command by command, so these tests check what can
drift without anyone noticing:

(a) every `--flag` the README shows for `orchard.supervisor run` or `control` exists in the real
    argparse parser. The parser is built by calling the module's own `parse` with
    `ArgumentParser.parse_args` replaced, so nothing runs and no hardware is touched. Any other
    `--flag` in the README must belong to another command line in this repository (hardware
    check, park check, sizing; read the same way) or be listed in
    OTHER_TOOL_FLAGS with the external tool it belongs to, so a misspelt supervisor flag cannot
    hide in prose;
(b) every relative link points to a file or directory that exists, and every in-page anchor
    matches a heading;
(c) the README holds no absolute path under /home/ or /mnt/ and no email address, because it is
    written for people on other machines;
(d) the README names every stage 0 to 8, by the name the stage table in orchard/stages.py uses,
    and has a table row for each stage number.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from orchard import hardware_check, park_check, sizing, supervisor
from orchard.stages import STAGES

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"

# Flags of external tools the README shows, with the tool each belongs to. A flag here is not
# checked against any parser; keep the list short and name the tool.
OTHER_TOOL_FLAGS = {
    "--cache-dir": "hf download",
    "--delete": "rsync (named as something the command runner does not stop)",
    "--force": "gozer (the README says never to pass it)",
    "--help": "gozer reset --help",
    "--out": "tt-model package-thin",
    "--version": "tt-model --version",
}

FLAG = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")


def _text() -> str:
    return README.read_text(encoding="utf-8")


class _Captured(Exception):
    """Raised in place of parsing, so the caller stops before it acts on any arguments."""


def _captured_parser(module, call) -> argparse.ArgumentParser:
    """The ArgumentParser on which `call` invokes parse_args. Nothing is parsed: parse_args is
    replaced by a function that records the parser and raises, so code after it never runs."""
    captured: list[argparse.ArgumentParser] = []

    def fake_parse_args(self, args=None, namespace=None):
        captured.append(self)
        raise _Captured

    mp = pytest.MonkeyPatch()
    mp.setattr(argparse.ArgumentParser, "parse_args", fake_parse_args)
    try:
        call(module)
    except _Captured:
        pass
    finally:
        mp.undo()
    assert captured, f"{module.__name__} built no parser"
    return captured[0]


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {s for a in parser._actions for s in a.option_strings}


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def supervisor_flags() -> set[str]:
    """The flags of `python3 -m orchard.supervisor run` and `control`."""
    top = _captured_parser(supervisor, lambda m: m.parse([]))
    subs = _subparsers(top)
    assert set(subs) == {"run", "control"}, sorted(subs)
    return _option_strings(subs["run"]) | _option_strings(subs["control"])


def other_repo_flags() -> set[str]:
    """The flags of the repository's other command lines (hardware check, park check, sizing)."""
    flags: set[str] = set()
    flags |= _option_strings(_captured_parser(hardware_check, lambda m: m.parse_args([])))
    flags |= _option_strings(_captured_parser(park_check, lambda m: m.parse_args([])))
    flags |= _option_strings(_captured_parser(sizing, lambda m: m.main([])))
    return flags


def supervisor_shown_flags(text: str) -> set[str]:
    """Flags the README shows for the supervisor: every flag in a fenced block that runs
    orchard.supervisor, and the first column of each table row that starts with a flag."""
    shown: set[str] = set()
    for block in re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S):
        if "orchard.supervisor" in block:
            shown |= set(FLAG.findall(block))
    for row in re.findall(r"^\|\s*`(--[a-z][a-z0-9-]*)`\s*\|", text, flags=re.M):
        shown.add(row)
    return shown


# ---- (a) flags ------------------------------------------------------------------------------------

def test_the_readme_shows_supervisor_flags_at_all():
    # Without this, a README that dropped the run command would pass (a) with nothing to check.
    shown = supervisor_shown_flags(_text())
    assert {"--model", "--run-dir", "--tiers", "--coder-target"} <= shown


def test_every_supervisor_flag_the_readme_shows_exists():
    real = supervisor_flags()
    missing = sorted(supervisor_shown_flags(_text()) - real)
    assert not missing, f"README shows flags that orchard.supervisor does not have: {missing}"


def test_every_other_flag_in_the_readme_is_accounted_for():
    known = supervisor_flags() | other_repo_flags() | set(OTHER_TOOL_FLAGS)
    unknown = sorted(set(FLAG.findall(_text())) - known)
    assert not unknown, (f"README names flags that no parser here has and OTHER_TOOL_FLAGS does "
                         f"not list: {unknown}")


# ---- (b) links ------------------------------------------------------------------------------------

def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens."""
    s = heading.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def test_every_relative_link_points_to_something_that_exists():
    broken = []
    for target in re.findall(r"\]\(([^)\s]+)\)", _text()):
        if re.match(r"[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            continue                       # a URL, mailto, or an in-page anchor (checked below)
        path = target.split("#", 1)[0]
        if not (REPO / path).exists():
            broken.append(target)
    assert not broken, f"README links to paths that do not exist: {broken}"


def test_every_in_page_anchor_matches_a_heading():
    text = _text()
    anchors = {_slug(h) for h in re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.M)}
    broken = [a for a in re.findall(r"\]\(#([^)]+)\)", text) if a not in anchors]
    assert not broken, f"README anchors with no matching heading: {broken}"


# ---- (c) nothing from one machine or one person -----------------------------------------------------

def test_no_absolute_home_or_mnt_path():
    hits = re.findall(r"/(?:home|mnt)/\S*", _text())
    assert not hits, f"README holds absolute paths from one machine: {hits}"


def test_no_email_address():
    hits = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", _text())
    assert not hits, f"README holds email addresses: {hits}"


# ---- (d) stages -------------------------------------------------------------------------------------

def test_the_readme_names_every_stage():
    text = _text()
    lower = text.lower()
    assert [s.number for s in STAGES] == list(range(9))
    missing_names = [f"{s.number}: {s.name}" for s in STAGES if s.name.lower() not in lower]
    assert not missing_names, f"README does not name these stages as orchard/stages.py does: {missing_names}"
    missing_rows = [n for n in range(9) if not re.search(rf"^\|\s*{n}\s*\|", text, flags=re.M)]
    assert not missing_rows, f"README has no table row for stages {missing_rows}"
