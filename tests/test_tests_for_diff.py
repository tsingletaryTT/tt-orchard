# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/tests_for_diff.py picks the tests a diff needs (docs/test-tiers.md).

Every case builds a small repository in a temporary directory (an `orchard/` package, a `tests/` directory, a few other files) and
asks the script which tests a given set of changes needs. The script reads the Python files as syntax trees, so the repository under
test never has to import. The rules under test:

* a test is selected when its own imports, or the imports of any file it imports, reach a changed module;
* a test is selected when a string in it names a changed file (the way tests load scripts and templates by path) or names a changed
  module (`-m orchard.a`, `import orchard.a` inside a string of code);
* the tests listed in `tests/tier_lists.py` as `WHOLE_REPO` are always selected;
* a change the script cannot map to tests selects the full suite and says why. It never selects fewer tests when it is unsure.
"""
from __future__ import annotations

import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("tests_for_diff", REPO / "scripts" / "tests_for_diff.py")
tfd = importlib.util.module_from_spec(_spec)
sys.modules["tests_for_diff"] = tfd
_spec.loader.exec_module(tfd)

TIERS = "WHOLE_REPO = {'tests/test_guard.py': (), 'tests/test_big.py': ('test_registry',)}\nSLOW_FILES = frozenset()\n"


def make_repo(root: Path, files: dict[str, str]) -> Path:
    """Write `files` under `root`, with an empty `orchard/__init__.py`, the guard tests and a tiers registry unless given."""
    base = {
        "orchard/__init__.py": "",
        "tests/tier_lists.py": TIERS,
        "tests/test_guard.py": "def test_guard():\n    pass\n",
        "tests/test_big.py": "def test_registry():\n    pass\n\n\ndef test_other():\n    pass\n",
        "tests/test_unrelated.py": "def test_x():\n    pass\n",
    }
    base.update(files)
    for rel, text in base.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def pick(root, changes, **kw):
    return tfd.select(root, [c if isinstance(c, tuple) else ("M", c) for c in changes], **kw)


GUARDS = {"tests/test_guard.py"}
GUARD_NODES = {"tests/test_big.py::test_registry"}


def only_guards(sel) -> bool:
    return not sel.full and set(sel.files) == GUARDS and set(sel.nodes) == GUARD_NODES


# ---- import edges ---------------------------------------------------------------------------------

def test_a_changed_module_selects_the_tests_that_import_it(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": "X = 1\n", "tests/test_a.py": "from orchard import a\n\n\ndef test_a():\n    assert a.X\n"})
    sel = pick(tmp_path, ["orchard/a.py"])
    assert not sel.full
    assert set(sel.files) == GUARDS | {"tests/test_a.py"}
    assert "tests/test_unrelated.py" not in sel.files


def test_a_change_reaches_a_test_through_a_chain_of_imports(tmp_path):
    make_repo(tmp_path, {
        "orchard/a.py": "X = 1\n",
        "orchard/b.py": "from orchard import a\n",
        "orchard/c.py": "from orchard import b\n",
        "tests/test_c.py": "from orchard import c\n\n\ndef test_c():\n    pass\n",
    })
    sel = pick(tmp_path, ["orchard/a.py"])
    assert "tests/test_c.py" in sel.files
    why = " ".join(sel.why["tests/test_c.py"])
    assert "orchard/c.py" in why and "orchard/b.py" in why and "orchard/a.py" in why   # the chain is shown


def test_a_change_reaches_a_test_through_a_test_helper(tmp_path):
    make_repo(tmp_path, {
        "orchard/a.py": "X = 1\n",
        "tests/fakes.py": "from orchard import a\n",
        "tests/test_f.py": "from fakes import a\n\n\ndef test_f():\n    pass\n",
    })
    assert "tests/test_f.py" in pick(tmp_path, ["orchard/a.py"]).files


@pytest.mark.parametrize("line", [
    "import orchard.a",
    "import orchard.a as z",
    "from orchard import a",
    "from orchard import a as z",
    "from orchard import x, a",
    "from orchard.a import X",
    "def f():\n    from orchard import a",
    "def f():\n    import orchard.a",
    "try:\n    from orchard import a\nexcept ImportError:\n    a = None",
    "if False:\n    from orchard import a",
    "class K:\n    def m(self):\n        from orchard.a import X",
])
def test_every_form_of_import_links_a_test_to_the_module(tmp_path, line):
    make_repo(tmp_path, {"orchard/a.py": "X = 1\n", "orchard/x.py": "", "tests/test_form.py": line + "\n\n\ndef test_t():\n    pass\n"})
    assert "tests/test_form.py" in pick(tmp_path, ["orchard/a.py"]).files


@pytest.mark.parametrize("line", ["from . import a", "from .a import X", "from orchard import a"])
def test_a_relative_import_inside_the_package_is_an_edge(tmp_path, line):
    make_repo(tmp_path, {
        "orchard/a.py": "X = 1\n",
        "orchard/user.py": line + "\n",
        "tests/test_user.py": "from orchard import user\n\n\ndef test_u():\n    pass\n",
    })
    assert "tests/test_user.py" in pick(tmp_path, ["orchard/a.py"]).files


def test_a_subpackage_module_is_reached_by_its_dotted_name(tmp_path):
    make_repo(tmp_path, {
        "orchard/sub/__init__.py": "",
        "orchard/sub/deep.py": "X = 1\n",
        "tests/test_deep.py": "from orchard.sub import deep\n\n\ndef test_d():\n    pass\n",
        "tests/test_deep2.py": "import orchard.sub.deep\n\n\ndef test_d2():\n    pass\n",
    })
    sel = pick(tmp_path, ["orchard/sub/deep.py"])
    assert {"tests/test_deep.py", "tests/test_deep2.py"} <= set(sel.files)


def test_a_changed_package_init_selects_every_test_that_imports_anything_from_the_package(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": "", "tests/test_a.py": "from orchard import a\n\n\ndef test_a():\n    pass\n"})
    assert "tests/test_a.py" in pick(tmp_path, ["orchard/__init__.py"]).files


def test_a_test_that_imports_nothing_from_the_changed_module_is_left_out(tmp_path):
    make_repo(tmp_path, {
        "orchard/a.py": "", "orchard/b.py": "", "tests/test_b.py": "from orchard import b\n\n\ndef test_b():\n    pass\n"})
    assert only_guards(pick(tmp_path, ["orchard/a.py"]))


def test_direct_mode_keeps_only_the_tests_that_import_the_changed_module_themselves(tmp_path):
    make_repo(tmp_path, {
        "orchard/a.py": "X = 1\n",
        "orchard/b.py": "from orchard import a\n",
        "tests/test_a.py": "from orchard import a\n\n\ndef test_a():\n    pass\n",
        "tests/test_b.py": "from orchard import b\n\n\ndef test_b():\n    pass\n",
    })
    far = pick(tmp_path, ["orchard/a.py"])
    near = pick(tmp_path, ["orchard/a.py"], direct=True)
    assert {"tests/test_a.py", "tests/test_b.py"} <= set(far.files)
    assert "tests/test_a.py" in near.files and "tests/test_b.py" not in near.files
    assert set(near.files) >= GUARDS                      # guards stay in direct mode


def test_eager_mode_leaves_out_the_imports_inside_functions_and_the_default_keeps_them(tmp_path):
    """`orchard/cli.py` imports a command's module inside the function that runs the command. A test that calls the command reaches the module
    through that call, so the default follows the lazy import and `--eager` does not."""
    make_repo(tmp_path, {
        "orchard/a.py": "X = 1\n",
        "orchard/cli.py": "def main():\n    from orchard import a\n    return a.X\n",
        "orchard/top.py": "from orchard import a\n",
        "tests/test_cli.py": "from orchard import cli\n\n\ndef test_c():\n    pass\n",
        "tests/test_top.py": "from orchard import top\n\n\ndef test_t():\n    pass\n",
    })
    default = pick(tmp_path, ["orchard/a.py"])
    eager = pick(tmp_path, ["orchard/a.py"], eager=True)
    assert {"tests/test_cli.py", "tests/test_top.py"} <= set(default.files)
    assert "tests/test_top.py" in eager.files and "tests/test_cli.py" not in eager.files


def test_the_eager_and_direct_flags_are_accepted_by_the_command_and_named_in_the_explanation(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": "X = 1\n"})
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "orchard" / "a.py").write_text("X = 2\n")
    r = run_script(tmp_path, "--base", "HEAD", "--eager")
    assert r.returncode == 0 and "imports that run at load only" in r.stderr


def test_a_file_in_scripts_is_an_importable_node_too(tmp_path):
    make_repo(tmp_path, {
        "scripts/tool.py": "X = 1\n",
        "scripts/other.py": "import tool\n",
        "tests/test_other.py": "import sys\nsys.path.insert(0, 'scripts')\nimport other\n\n\ndef test_o():\n    pass\n",
    })
    assert "tests/test_other.py" in pick(tmp_path, ["scripts/tool.py"]).files


# ---- string references ----------------------------------------------------------------------------

@pytest.mark.parametrize("body", [
    'from pathlib import Path\nP = Path(".") / "scripts" / "tool.py"\n',
    'CMD = "python3 scripts/tool.py --flag"\n',
    'import importlib.util\nS = importlib.util.spec_from_file_location("t", "scripts/tool.py")\n',
])
def test_a_test_that_names_a_script_by_path_is_selected_when_the_script_changes(tmp_path, body):
    make_repo(tmp_path, {"scripts/tool.py": "X = 1\n", "tests/test_named.py": body + "\n\ndef test_n():\n    pass\n"})
    assert "tests/test_named.py" in pick(tmp_path, ["scripts/tool.py"]).files


def test_a_docstring_that_mentions_a_script_does_not_select_a_test(tmp_path):
    """A docstring describes a file. A test that only describes `scripts/tool.py` in prose does not depend on it."""
    make_repo(tmp_path, {"scripts/tool.py": "X = 1\n",
                         "tests/test_d.py": '"""Checks the output of scripts/tool.py and `orchard.a`."""\n\n\ndef test_d():\n    """Runs tool.py."""\n',
                         "orchard/a.py": ""})
    assert only_guards(pick(tmp_path, ["scripts/tool.py"]))
    assert only_guards(pick(tmp_path, ["orchard/a.py"]))


def test_a_program_name_in_a_message_of_a_module_is_not_an_edge(tmp_path):
    """orchard/__init__.py and the CLI print `tt-orchard` in messages. Only a test that names the program runs it."""
    make_repo(tmp_path, {
        "bin/tt-orchard": "#!/usr/bin/env python3\nfrom orchard import cli\n",
        "orchard/cli.py": "from orchard import a\n",
        "orchard/a.py": "X = 1\n",
        "orchard/msg.py": 'HINT = "run tt-orchard --help"\n',
        "tests/test_msg.py": "from orchard import msg\n\n\ndef test_m():\n    pass\n",
    })
    assert "tests/test_msg.py" not in pick(tmp_path, ["orchard/a.py"]).files


def test_a_comment_that_mentions_a_script_does_not_select_a_test(tmp_path):
    make_repo(tmp_path, {"scripts/tool.py": "X = 1\n", "tests/test_c.py": "# scripts/tool.py is not used here\n\n\ndef test_c():\n    pass\n"})
    assert only_guards(pick(tmp_path, ["scripts/tool.py"]))


@pytest.mark.parametrize("body", [
    'import subprocess, sys\nsubprocess.run([sys.executable, "-m", "orchard.a"])\n',
    'CODE = "import orchard.a; orchard.a.run()"\n',
    'CODE = "from orchard import a, b"\n',
])
def test_a_module_named_inside_a_string_of_code_is_an_edge(tmp_path, body):
    make_repo(tmp_path, {"orchard/a.py": "", "orchard/b.py": "", "tests/test_s.py": body + "\n\ndef test_s():\n    pass\n"})
    assert "tests/test_s.py" in pick(tmp_path, ["orchard/a.py"]).files


def test_the_command_line_script_is_a_node_that_imports_the_cli(tmp_path):
    make_repo(tmp_path, {
        "bin/tt-orchard": "#!/usr/bin/env python3\nfrom orchard import cli\ncli.main()\n",
        "orchard/cli.py": "def main():\n    pass\n",
        "tests/test_exe.py": 'import subprocess\nsubprocess.run(["bin/tt-orchard", "--version"])\n\n\ndef test_e():\n    pass\n',
    })
    assert "tests/test_exe.py" in pick(tmp_path, ["orchard/cli.py"]).files


def test_a_string_reference_is_followed_through_a_helper(tmp_path):
    make_repo(tmp_path, {
        "scripts/tool.py": "X = 1\n",
        "tests/runner.py": 'SCRIPT = "tool.py"\n',
        "tests/test_r.py": "import runner\n\n\ndef test_r():\n    pass\n",
    })
    assert "tests/test_r.py" in pick(tmp_path, ["scripts/tool.py"]).files


def test_a_test_with_a_computed_import_runs_for_every_python_change(tmp_path):
    make_repo(tmp_path, {
        "orchard/a.py": "",
        "tests/test_dyn.py": "import importlib\n\n\ndef load(name):\n    return importlib.import_module(name)\n\n\ndef test_d():\n    pass\n",
    })
    assert "tests/test_dyn.py" in pick(tmp_path, ["orchard/a.py"]).files
    assert "tests/test_dyn.py" not in pick(tmp_path, ["docs/x.md"]).files      # a docs change does not need it


def test_a_test_file_that_does_not_parse_runs_for_every_python_change(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": "", "tests/test_broken.py": "def test_x(:\n"})
    sel = pick(tmp_path, ["orchard/a.py"])
    assert "tests/test_broken.py" in sel.files
    assert any("test_broken" in n for n in sel.notes)


def test_a_changed_module_that_does_not_parse_still_gives_a_selection(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": "def f(:\n", "tests/test_a.py": "from orchard import a\n\n\ndef test_a():\n    pass\n"})
    sel = pick(tmp_path, ["orchard/a.py"])
    assert not sel.full and "tests/test_a.py" in sel.files


# ---- tests, docs, data ------------------------------------------------------------------------------

def test_a_changed_test_file_selects_itself(tmp_path):
    make_repo(tmp_path, {"tests/test_new.py": "def test_n():\n    pass\n"})
    assert "tests/test_new.py" in pick(tmp_path, [("A", "tests/test_new.py")]).files


def test_a_deleted_test_file_selects_nothing_and_is_not_an_error(tmp_path):
    make_repo(tmp_path, {})
    sel = pick(tmp_path, [("D", "tests/test_gone.py")])
    assert only_guards(sel)


def test_a_change_fragment_is_ignored(tmp_path):
    make_repo(tmp_path, {"changes/x.toml": "target = '1.0.0'\n"})
    assert only_guards(pick(tmp_path, [("A", "changes/x.toml")]))


def test_a_docs_change_selects_the_guards_and_the_tests_that_name_the_file(tmp_path):
    make_repo(tmp_path, {
        "docs/a.md": "x\n", "docs/b.md": "y\n",
        "tests/test_docs.py": 'DOC = "a.md"\n\n\ndef test_d():\n    pass\n',
    })
    assert set(pick(tmp_path, ["docs/a.md"]).files) == GUARDS | {"tests/test_docs.py"}
    assert only_guards(pick(tmp_path, ["docs/b.md"]))


def test_the_readme_and_claude_md_are_documents(tmp_path):
    make_repo(tmp_path, {"README.md": "x\n", "CLAUDE.md": "y\n", "tests/test_readme.py": 'R = "README.md"\n\n\ndef test_r():\n    pass\n'})
    assert "tests/test_readme.py" in pick(tmp_path, ["README.md"]).files
    assert only_guards(pick(tmp_path, ["CLAUDE.md"]))


def test_a_data_file_maps_to_the_tests_that_name_it_or_its_folder(tmp_path):
    make_repo(tmp_path, {
        "tests/fixtures/tier-plan/q.config.json": "{}\n",
        "tests/test_plan.py": 'F = "tier-plan"\n\n\ndef test_p():\n    pass\n',
        "tests/fixtures/solo.json": "{}\n",
        "tests/test_solo.py": 'F = "solo.json"\n\n\ndef test_s():\n    pass\n',
    })
    assert "tests/test_plan.py" in pick(tmp_path, ["tests/fixtures/tier-plan/q.config.json"]).files
    assert "tests/test_solo.py" in pick(tmp_path, ["tests/fixtures/solo.json"]).files


def test_a_data_file_that_no_test_names_selects_the_full_suite(tmp_path):
    make_repo(tmp_path, {"tests/fixtures/orphan.json": "{}\n"})
    sel = pick(tmp_path, ["tests/fixtures/orphan.json"])
    assert sel.full and any("tests/fixtures/orphan.json" in r for r in sel.reasons)


def test_a_new_module_nobody_imports_selects_only_the_guards_and_its_own_tests(tmp_path):
    make_repo(tmp_path, {"orchard/new.py": "X = 1\n"})
    assert only_guards(pick(tmp_path, [("A", "orchard/new.py")]))
    make_repo(tmp_path, {"tests/test_new.py": "from orchard import new\n\n\ndef test_n():\n    pass\n"})
    assert "tests/test_new.py" in pick(tmp_path, [("A", "orchard/new.py"), ("A", "tests/test_new.py")]).files


# ---- the guards -------------------------------------------------------------------------------------

def test_the_guards_are_selected_for_every_change_that_is_not_the_full_suite(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": ""})
    for change in ["orchard/a.py", "docs/anything.md", "changes/z.toml"]:
        sel = pick(tmp_path, [change])
        assert not sel.full
        assert GUARDS <= set(sel.files) and GUARD_NODES <= set(sel.nodes), change


def test_a_guard_that_lives_in_a_big_file_is_selected_by_name_and_the_file_stays_out(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": ""})
    sel = pick(tmp_path, ["orchard/a.py"])
    assert "tests/test_big.py" not in sel.files
    assert "tests/test_big.py::test_registry" in sel.nodes


def test_a_guard_node_is_dropped_when_its_whole_file_is_already_selected(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": "", "tests/test_big.py": "from orchard import a\n\n\ndef test_registry():\n    pass\n"})
    sel = pick(tmp_path, ["orchard/a.py"])
    assert "tests/test_big.py" in sel.files and not sel.nodes


def test_a_registry_naming_a_missing_guard_is_reported(tmp_path):
    make_repo(tmp_path, {"tests/tier_lists.py": "WHOLE_REPO = {'tests/test_nowhere.py': ()}\nSLOW_FILES = frozenset()\n", "orchard/a.py": ""})
    sel = pick(tmp_path, ["orchard/a.py"])
    assert sel.full and any("test_nowhere" in r for r in sel.reasons)


def test_a_repository_without_a_registry_gets_the_full_suite(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": ""})
    (tmp_path / "tests" / "tier_lists.py").unlink()
    sel = pick(tmp_path, ["orchard/a.py"])
    assert sel.full and any("tier_lists.py" in r for r in sel.reasons)


# ---- the fallbacks ----------------------------------------------------------------------------------

FULL_CASES = [
    (("M", "tests/conftest.py"), "conftest.py"),
    (("M", "tests/helper_fakes.py"), "helper_fakes.py"),
    (("M", "tests/tier_lists.py"), "tier_lists.py"),
    (("M", "pyproject.toml"), "pyproject.toml"),
    (("M", "requirements.txt"), "requirements.txt"),
    (("M", "requirements-dev.txt"), "requirements-dev.txt"),
    (("D", "orchard/a.py"), "orchard/a.py"),
    (("R", "orchard/a.py", "orchard/c.py"), "orchard/a.py"),
    (("D", "scripts/tool.py"), "scripts/tool.py"),
    (("M", "orchard/skills/s/SKILL.md"), "orchard/skills/s/SKILL.md"),
    (("M", "orchard/skills/s/template.py"), "orchard/skills/s/template.py"),
    (("A", "orchard/package_templates/p.py"), "orchard/package_templates/p.py"),
    (("M", "Makefile"), "Makefile"),
    (("M", ".github/workflows/ci.yml"), ".github/workflows/ci.yml"),
    (("M", "scripts/setup.sh"), "scripts/setup.sh"),
    (("M", "somewhere/new/place.txt"), "somewhere/new/place.txt"),
]


@pytest.mark.parametrize("change,named", FULL_CASES, ids=[c[1] for c in FULL_CASES])
def test_a_change_the_script_cannot_map_selects_the_full_suite_and_names_the_file(tmp_path, change, named):
    make_repo(tmp_path, {"orchard/a.py": "", "tests/helper_fakes.py": "", "tests/conftest.py": "",
                         "scripts/tool.py": "", "scripts/setup.sh": "", "orchard/skills/s/SKILL.md": ""})
    sel = pick(tmp_path, [change])
    assert sel.full
    assert any(named in r for r in sel.reasons), sel.reasons
    assert sel.args() == ["tests"]


@pytest.mark.parametrize("changed", ["tests/helper_fakes.py", "pyproject.toml", "requirements.txt", "tests/tier_lists.py", "tests/conftest.py"])
def test_a_test_that_names_a_shared_file_does_not_make_a_change_to_it_a_small_selection(tmp_path, changed):
    """The full-suite rule applies to these files whether or not a test names them: pyproject.toml is read by the tier tests, and a helper
    is named by every test that imports it."""
    name = changed.rsplit("/", 1)[-1]
    content = TIERS if changed == "tests/tier_lists.py" else "x\n"         # a registry that loads, so only the rule decides
    make_repo(tmp_path, {changed: content, "tests/test_names_it.py": f'F = "{name}"\n\n\ndef test_n():\n    pass\n'})
    sel = pick(tmp_path, [changed])
    assert sel.full and any(changed in r for r in sel.reasons), sel.reasons


def test_one_unmappable_change_among_mappable_ones_selects_the_full_suite(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": "", "tests/test_a.py": "from orchard import a\n\n\ndef test_a():\n    pass\n"})
    assert pick(tmp_path, ["orchard/a.py", "pyproject.toml"]).full


def test_a_rename_of_a_test_file_selects_the_new_name(tmp_path):
    make_repo(tmp_path, {"tests/test_new_name.py": "def test_n():\n    pass\n"})
    sel = pick(tmp_path, [("R", "tests/test_old_name.py", "tests/test_new_name.py")])
    assert not sel.full and "tests/test_new_name.py" in sel.files


# ---- the arguments ----------------------------------------------------------------------------------

def test_the_arguments_name_files_and_nodes_in_a_stable_order(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": "", "tests/test_b.py": "from orchard import a\n\n\ndef test_b():\n    pass\n",
                         "tests/test_a.py": "from orchard import a\n\n\ndef test_a():\n    pass\n"})
    args = pick(tmp_path, ["orchard/a.py"]).args()
    assert args == ["tests/test_a.py", "tests/test_b.py", "tests/test_guard.py", "tests/test_big.py::test_registry"]


def test_the_fast_option_leaves_out_slow_tests(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": ""})
    args = pick(tmp_path, ["orchard/a.py"]).args(fast=True)
    assert args[-2:] == ["-m", "not slow"]


# ---- the git layer ----------------------------------------------------------------------------------

def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                   check=True, capture_output=True)


def run_script(root, *args):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "tests_for_diff.py"), "--root", str(root), *args],
                          capture_output=True, text=True)


def test_the_command_reads_changes_from_git_including_new_and_renamed_files(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": "X = 1\n", "orchard/old.py": "Y = 1\n",
                         "tests/test_a.py": "from orchard import a\n\n\ndef test_a():\n    pass\n"})
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "orchard" / "a.py").write_text("X = 2\n")
    (tmp_path / "tests" / "test_new.py").write_text("def test_n():\n    pass\n")
    r = run_script(tmp_path, "--base", "HEAD")
    assert r.returncode == 0, r.stderr
    args = shlex.split(r.stdout)
    assert "tests/test_a.py" in args and "tests/test_new.py" in args and "tests/test_guard.py" in args
    assert "tests/test_unrelated.py" not in args
    git(tmp_path, "mv", "orchard/old.py", "orchard/new_name.py")
    r = run_script(tmp_path, "--base", "HEAD")
    assert shlex.split(r.stdout) == ["tests"]                          # a renamed module falls back to the full suite
    assert "orchard/old.py" in r.stderr and "full suite" in r.stderr


def test_the_command_says_how_it_chose_on_standard_error(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": "X = 1\n", "tests/test_a.py": "from orchard import a\n\n\ndef test_a():\n    pass\n"})
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "orchard" / "a.py").write_text("X = 2\n")
    r = run_script(tmp_path, "--base", "HEAD")
    assert "changed: orchard/a.py" in r.stderr
    assert "tests/test_a.py  (tests/test_a.py -> orchard/a.py)" in r.stderr and "selected" in r.stderr
    assert r.stdout.strip() and "\n" not in r.stdout.strip()           # one line of arguments on standard output


def test_a_base_that_does_not_exist_selects_the_full_suite_and_says_so(tmp_path):
    make_repo(tmp_path, {})
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    r = run_script(tmp_path, "--base", "no-such-ref")
    assert r.returncode == 0 and shlex.split(r.stdout) == ["tests"]
    assert "no-such-ref" in r.stderr and "does not exist" in r.stderr and "full suite" in r.stderr


def test_a_directory_that_is_not_a_repository_selects_the_full_suite(tmp_path):
    make_repo(tmp_path, {})
    r = run_script(tmp_path)
    assert r.returncode == 0 and shlex.split(r.stdout) == ["tests"]
    assert "not a git repository" in r.stderr and "full suite" in r.stderr


def test_nothing_changed_still_selects_the_guards(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": ""})
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    r = run_script(tmp_path, "--base", "HEAD")
    assert shlex.split(r.stdout) == ["tests/test_guard.py", "tests/test_big.py::test_registry"]
    assert "nothing changed" in r.stderr


def test_the_fast_and_direct_flags_are_accepted_by_the_command(tmp_path):
    make_repo(tmp_path, {"orchard/a.py": "X = 1\n"})
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    (tmp_path / "orchard" / "a.py").write_text("X = 2\n")
    r = run_script(tmp_path, "--base", "HEAD", "--fast", "--direct")
    assert r.returncode == 0 and shlex.split(r.stdout)[-2:] == ["-m", "not slow"]
