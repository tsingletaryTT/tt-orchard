# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The operator runbook skill (orchard/skills/operator-runbook.md) cannot drift from the CLI.

Every orchard command line the runbook quotes is parsed with the real argparse parsers, every
state the status command can print has a row, and the NEVER list is present."""
import re
from pathlib import Path

import pytest

from orchard import operator_checks, status, supervisor
from test_readme import _Captured, _captured_parser

REPO = Path(__file__).resolve().parent.parent
SKILL = REPO / "orchard" / "skills" / "operator-runbook.md"
# The values a human hands the operator. Nothing else in double braces is allowed.
VALUES = {"ORCHARD_DIR": "/o", "RUN_DIR": "/r", "RUN_SCRIPT": "/s.sh", "OPERATOR_LOG": "/l.txt"}


def text() -> str:
    return SKILL.read_text(encoding="utf-8")


def orchard_commands() -> list[str]:
    """Each `python3 -m orchard.<module> ...` line in the skill, placeholders filled in."""
    out = []
    for line in text().splitlines():
        m = re.match(r"\s+(python3 -m orchard\.\w+ .*)$", line)
        if m:
            out.append(re.sub(r"\{\{(\w+)\}\}", lambda g: VALUES[g.group(1)], m.group(1)))
    return out


def test_front_matter_has_name_description_and_status():
    front = text().split("---")[1]
    assert "\nname: operator-runbook\n" in front
    assert "\ndescription: " in front and "\nstatus: draft." in front


def test_the_skill_is_short_enough_for_a_small_context():
    assert len(text().splitlines()) <= 150


def test_the_skill_names_only_the_documented_placeholders():
    assert set(re.findall(r"\{\{(.*?)\}\}", text())) <= set(VALUES)


def test_the_skill_holds_no_machine_paths_or_email():
    t = text()
    assert "/home/" not in t and "/mnt/" not in t and "@" not in t


def test_it_quotes_the_commands_it_depends_on():
    cmds = orchard_commands()
    assert any(" status " in c for c in cmds)
    assert any(" control " in c and c.endswith(" resume") for c in cmds)
    assert any("operator_checks" in c for c in cmds)


@pytest.mark.parametrize("cmd", orchard_commands())
def test_every_quoted_orchard_command_parses_with_the_real_argparse(cmd):
    words = cmd.split()
    module, args = words[2], words[3:]
    if module == "orchard.supervisor":
        parsed = supervisor.parse(args)
        assert parsed.cmd in ("status", "control") and parsed.run_dir == "/r"
    else:
        assert module == "orchard.operator_checks"
        parser = _captured_parser(operator_checks, lambda m: m.main([]))
        ns = parser.parse_args(args)
        assert ns.run_dir == "/r"


def test_the_runbook_never_quotes_a_run_or_abort_command():
    for cmd in orchard_commands():
        assert " run " not in cmd and not cmd.endswith(" abort")


def test_every_state_status_can_print_has_a_row_in_the_decision_table():
    rows = [l for l in text().splitlines() if l.startswith("|")]
    for state in status.STATES:
        assert any(l.startswith(f"| {state} ") for l in rows), state


def test_the_table_and_the_hint_rules_use_the_same_words():
    # If a hint changes its wording, the runbook row that quotes it must change too.
    hints = " ".join(r[2] for r in status.HINT_RULES)
    t = text()
    for phrase in ("resume once", "the same stage has paused", "needs a decision", "control word is waiting",
                   "5 hours"):
        assert phrase in hints and phrase in t, phrase


def test_the_never_list_covers_each_forbidden_act():
    never = text().split("## NEVER")[1].split("##")[0]
    for word in ("publish", "push", "upload", "--public", "--publish", "delete", "edit", "flags",
                 "control abort", "credential", "tt-smi -r"):
        assert word in never, word


def test_the_stop_and_ask_triggers_are_listed():
    ask = text().split("## Stop and ask a human")[1]
    for trigger in ("pauses twice", "refused", "40 GB", "not cover"):
        assert trigger in ask, trigger


def test_the_loop_has_five_steps_and_waits_in_its_own_command():
    loop = text().split("## The loop")[1].split("##")[0]
    assert len(re.findall(r"^\d\. ", loop, re.M)) == 5
    assert re.search(r"^\s+sleep 300$", loop, re.M)


def test_every_status_command_in_the_runbook_pins_the_plain_view():
    """A small model may read through a pseudo-terminal, where `auto` would decorate the output. The
    runbook and its tests are written against the plain block, so the command says so."""
    status_cmds = [c for c in orchard_commands() if " status " in c]
    assert status_cmds
    for c in status_cmds:
        assert "--style plain" in c, c
