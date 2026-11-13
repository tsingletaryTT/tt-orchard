# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The block-family authoring skill (issue 707) cannot drift from the code it tells a small coder to use.

The skill is `orchard/skills/block-family-authoring/SKILL.md`. These tests check that it exists, has the front matter of the stage skills, names
no machine path, and that every name it relies on exists: each inline code name that starts with an underscore, is all capitals, or is snake_case
must be an attribute of `orchard.port_template`, a function or constant of the generated tiny Qwen4 port, a name in the skill's `new-names` block
(names the reader creates), a name in its `other-names` block (which must appear in port_template.py, the fake ttnn or the generated port), a name
in its `quoted-names` block (quoted from a failure line of a run), or the name of a test function that exists. Each
`path::test_name` the skill cites must exist. `problems(text)` returns the list of faults so that the mutation tests can show each check is live."""
from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest

from orchard import port_template as pt
from test_port_template import tiny_qwen4

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "orchard" / "skills" / "block-family-authoring" / "SKILL.md"
MACHINE_PATH = re.compile(r"/home/|/mnt/|/Users/|/root/")
INLINE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
NAMEISH = re.compile(r"^(?:_[a-z][a-z0-9_]*|[A-Z][A-Z0-9_]{3,}|[a-z][a-z0-9]*(?:_[a-z0-9]+)+)$")
BLOCK = re.compile(r"```(new-names|other-names|quoted-names)\n(.*?)```", re.S)
TEST_ID = re.compile(r"(tests/[\w./-]+\.py)::(\w+)")


def _generated_names() -> set:
    text, _plan = tiny_qwen4()
    names = set()
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _sources() -> str:
    text, _plan = tiny_qwen4()
    return "\n".join([(ROOT / "orchard" / "port_template.py").read_text(), (ROOT / "orchard" / "skills" / "full-port-templates" / "cpu_ttnn.py").read_text(), text])


def _fields() -> set:
    return {f.name for cls in (pt.BlockFamily, pt.Site) for f in dataclasses.fields(cls)}


def _is_test(token: str) -> bool:
    """A cited test function name: it must be defined in some file under tests/."""
    return token.startswith("test_") and any(f"def {token}(" in p.read_text() for p in (ROOT / "tests").glob("test_*.py"))


def _blocks(text: str) -> dict:
    return {kind: [line.strip() for line in body.splitlines() if line.strip()] for kind, body in BLOCK.findall(text)}


def problems(text: str) -> list:
    """Every fault of a skill text: front matter, machine paths, a name that does not exist, an unused listed name, a cited test that is missing."""
    out = []
    parts = text.split("---")
    front = parts[1] if len(parts) > 2 and text.startswith("---") else ""
    if "\nname: block-family-authoring\n" not in "\n" + front.lstrip("\n") + "\n" or "\nstatus: draft." not in "\n" + front.lstrip("\n"):
        out.append("front matter: name and a status that starts with 'draft.' are required")
    out += [f"machine path on line {i}" for i, line in enumerate(text.splitlines(), 1) if MACHINE_PATH.search(line)]
    blocks = _blocks(text)
    new, other, quoted = set(blocks.get("new-names", [])), blocks.get("other-names", []), blocks.get("quoted-names", [])
    generated, sources, fields = _generated_names(), _sources(), _fields()
    prose = BLOCK.sub("", text)
    for token in sorted({m for m in INLINE.findall(prose) if NAMEISH.match(m)}):
        if token in new or token in other or token in quoted or token in fields or token in generated or hasattr(pt, token) or _is_test(token):
            continue
        out.append(f"unknown name: {token}")
    for name in other:
        if name not in sources:
            out.append(f"other-names entry not found in the code: {name}")
        if f"`{name}`" not in prose:
            out.append(f"other-names entry never used in the text: {name}")
    for name in quoted:
        if f"`{name}`" not in prose:
            out.append(f"quoted-names entry never used in the text: {name}")
    for name in new:
        if f"`{name}`" not in prose and name not in text.replace("```new-names", ""):
            out.append(f"new-names entry never used: {name}")
        if hasattr(pt, name):
            out.append(f"new-names entry already exists in port_template: {name}")
    for path, name in sorted(set(TEST_ID.findall(prose + "\n" + "\n".join(re.findall(r"```python\n(.*?)```", text, re.S))))):
        file = ROOT / path
        if not file.is_file() or not any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(ast.parse(file.read_text()))):
            out.append(f"cited test does not exist: {path}::{name}")
    return out


def test_the_skill_exists_and_has_the_front_matter_of_a_draft_skill():
    assert SKILL.is_file()
    text = SKILL.read_text()
    assert text.startswith("---\nname: block-family-authoring\nstatus: draft.")


def test_the_skill_has_no_machine_path_and_no_unknown_name_and_cites_only_tests_that_exist():
    assert problems(SKILL.read_text()) == []


def test_the_skill_is_found_by_the_stage_skill_resolver():
    from orchard.stages import resolve_skill
    assert resolve_skill("block-family-authoring", [ROOT / "orchard" / "skills"]) == SKILL


def test_the_skill_names_the_steps_the_issue_asks_for():
    text = SKILL.read_text()
    for needle in ("### 1.", "### 2.", "### 3.", "### 4.", "### 5.", "### 6.", "### 7.", "### 8.", "## The four families so far", "## Mistakes seen from the small coder"):
        assert needle in text, needle
    mistakes = text.split("## Mistakes seen from the small coder")[1].split("Also seen:")[0]
    assert len(re.findall(r"^\d+\. ", mistakes, re.M)) == 10


# ---- each check turns red under a mutation of a copy of the text -----------------------------------------------------------------------------

def _mutations():
    text = SKILL.read_text()
    return {
        "a function is renamed": text.replace("`_compose_layer`", "`_compose_layers`", 1),
        "a constant is renamed": text.replace("`FIXED_HELPERS`", "`FIXED_HELPER_LIST`", 1),
        "a registry function is renamed": text.replace("`register_block_family`", "`register_family`", 1),
        "a helper that the generated port holds is renamed": text.replace("`_apply_rotary`", "`_apply_rope`", 1),
        "a machine path appears": text + "\nSee /home/someone/notes.\n",
        "the status is not a draft": text.replace("status: draft.", "status: ready.", 1),
        "a cited test is renamed": text.replace("test_the_reference_comparison_fails_when_a_stream_weight_is_dropped", "test_no_such_test", 1),
        "an other-names entry is removed from the code": text.replace("neg_exp\nconv_taps", "neg_exp_gone\nconv_taps", 1),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_a_mutated_copy_of_the_skill_is_reported(name):
    mutated = _mutations()[name]
    assert mutated != SKILL.read_text(), "the mutation does not apply: " + name
    assert problems(mutated), name
