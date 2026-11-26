# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The test tiers (docs/test-tiers.md): which tests are guards, which are slow, which need a chip, and what runs by default.

`tests/tier_lists.py` lists the files that are `slow` (measured above a threshold, written there) and the tests that are `whole_repo`
guards (they scan the repository, so any new module, flag or file can fail them). `tests/conftest.py` turns the lists into markers. A test
file in neither list is in the default tier and runs under `-m "not slow"`. The markers are registered in `pyproject.toml` with strict
marker checking, so a misspelt marker fails collection.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import tier_lists as tiers

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
OPTIONS = PYPROJECT["tool"]["pytest"]["ini_options"]


def defined_tests(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")}


# ---- the lists --------------------------------------------------------------------------------------

def test_every_guard_names_a_test_file_and_test_functions_that_exist():
    for rel, names in tiers.WHOLE_REPO.items():
        path = REPO / rel
        assert path.is_file(), f"{rel} is listed in WHOLE_REPO and does not exist"
        missing = set(names) - defined_tests(path)
        assert not missing, f"{rel} has no test named {sorted(missing)}"


def test_every_slow_file_exists():
    gone = [rel for rel in tiers.SLOW_FILES if not (REPO / rel).is_file()]
    assert not gone, f"SLOW_FILES names files that do not exist: {gone}"


def test_a_file_cannot_be_both_a_whole_file_guard_and_slow():
    both = [rel for rel, names in tiers.WHOLE_REPO.items() if not names and rel in tiers.SLOW_FILES]
    assert not both, f"a guard that always runs cannot be slow: {both}"


def test_the_lists_hold_test_files_only():
    for rel in [*tiers.WHOLE_REPO, *tiers.SLOW_FILES, *tiers.HARDWARE_FILES, *tiers.NOT_GUARDS]:
        assert re.fullmatch(r"tests/test_[A-Za-z0-9_]+\.py", rel), rel


def test_the_slow_tier_is_a_minority_of_the_files_and_the_guards_are_few():
    files = list(TESTS.glob("test_*.py"))
    assert 0 < len(tiers.SLOW_FILES) < len(files) / 4
    assert 0 < len(tiers.WHOLE_REPO) < 40


# ---- which markers a test gets ----------------------------------------------------------------------

def test_a_test_file_in_neither_list_is_in_the_default_tier():
    assert tiers.markers_for("tests/test_brand_new_file.py", "test_anything") == ()


def test_a_whole_file_guard_gets_the_whole_repo_marker_for_every_test():
    rel = next(rel for rel, names in tiers.WHOLE_REPO.items() if not names)
    assert tiers.markers_for(rel, "test_whatever") == ("whole_repo",)


def test_a_guard_named_in_a_slow_file_gets_whole_repo_and_the_rest_of_the_file_gets_slow():
    rel, names = next((r, n) for r, n in tiers.WHOLE_REPO.items() if n and r in tiers.SLOW_FILES)
    assert tiers.markers_for(rel, names[0]) == ("whole_repo",)
    assert tiers.markers_for(rel, "test_some_other_test_in_the_file") == ("slow",)


def test_a_slow_file_that_holds_no_guard_gets_slow():
    rel = next(r for r in sorted(tiers.SLOW_FILES) if r not in tiers.WHOLE_REPO)
    assert tiers.markers_for(rel, "test_x") == ("slow",)


def test_a_hardware_file_gets_the_hardware_marker():
    got = tiers.markers_for("tests/test_on_a_chip.py", "test_x", hardware_files=frozenset({"tests/test_on_a_chip.py"}))
    assert got == ("hardware",)


def test_a_hardware_test_is_skipped_unless_the_operator_opts_in_and_says_why():
    assert "ORCHARD_RUN_HARDWARE" in tiers.hardware_skip_reason({})
    assert "ORCHARD_RUN_HARDWARE" in tiers.hardware_skip_reason({"ORCHARD_RUN_HARDWARE": "0"})
    assert tiers.hardware_skip_reason({"ORCHARD_RUN_HARDWARE": "1"}) is None


def test_no_test_file_opens_a_device_without_being_a_hardware_file():
    """Importing ttnn or ttml opens the device (CLAUDE.md), so a test module that imports either at its top level has to be a hardware file."""
    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            if any(n.split(".")[0] in ("ttnn", "ttml") for n in names) and f"tests/{path.name}" not in tiers.HARDWARE_FILES:
                offenders.append(path.name)
    assert not offenders, f"these import ttnn or ttml at the top level and are not in HARDWARE_FILES: {offenders}"


# ---- the repository scanners ------------------------------------------------------------------------

WALKERS = ("rglob", "glob", "iterdir", "walk", "listdir", "scandir")
ROOT_NAMES = re.compile(r"\b(ORCHARD|ORCHARD_DIR|PACKAGE|SCRIPTS|SRC|ROOT|REPO|REPO_ROOT)\b")
ROOT_LITERALS = re.compile(r"""['"](orchard|scripts|changes|docs|README\.md|CLAUDE\.md)['"]""")


def walks_the_repository(tree: ast.AST) -> bool:
    """True when the tree calls a directory walker (glob, rglob, iterdir, os.walk, listdir, scandir) on an expression that names the repository
    root or one of its source directories."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name in WALKERS:
                text = ast.unparse(node)
                if ROOT_NAMES.search(text) or ROOT_LITERALS.search(text):
                    return True
    return False


def scans_the_source_tree(path: Path, scripts_dir: Path | None = None) -> bool:
    """True when a test file reads many files of the repository: it walks the repository itself, or it loads a script by file name and that
    script does. Such a test fails when a new module breaks the rule it checks (the ratchets are of the second kind)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if walks_the_repository(tree):
        return True
    scripts = scripts_dir if scripts_dir is not None else REPO / "scripts"
    names = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return any(walks_the_repository(ast.parse(s.read_text(encoding="utf-8"))) for s in scripts.glob("*.py") if s.name in names)


def test_every_test_file_that_scans_the_source_tree_is_a_guard():
    found = [f"tests/{p.name}" for p in sorted(TESTS.glob("test_*.py")) if scans_the_source_tree(p)]
    unlisted = [rel for rel in found if rel not in tiers.WHOLE_REPO and rel not in tiers.NOT_GUARDS]
    assert not unlisted, (
        f"these test files walk the source tree, so a new module can fail them: {unlisted}. Add each to WHOLE_REPO in tests/tier_lists.py "
        f"(a test name in the tuple picks one function), or to NOT_GUARDS with the reason.")


def test_the_scanner_check_sees_a_scan_when_one_is_present(tmp_path):
    scan = tmp_path / "test_scan.py"
    scan.write_text("ORCHARD = None\n\n\ndef test_x():\n    for p in ORCHARD.rglob('*.py'):\n        pass\n", encoding="utf-8")
    plain = tmp_path / "test_plain.py"
    plain.write_text("def test_x(tmp_path):\n    list(tmp_path.glob('*.json'))\n", encoding="utf-8")
    assert scans_the_source_tree(scan) and not scans_the_source_tree(plain)


def test_the_scanner_check_sees_the_other_ways_to_walk_the_repository(tmp_path):
    for i, body in enumerate([
        "import glob\nROOT = None\n\n\ndef test_x():\n    glob.glob(str(ROOT / 'orchard/**/*.py'), recursive=True)\n",
        "import os\nREPO = None\n\n\ndef test_x():\n    list(os.walk(REPO))\n",
        "from pathlib import Path\n\n\ndef test_x():\n    list((Path('.') / 'changes').iterdir())\n",
    ]):
        f = tmp_path / f"test_walk{i}.py"
        f.write_text(body, encoding="utf-8")
        assert scans_the_source_tree(f), body


def test_the_scanner_check_follows_a_script_the_test_loads_by_name(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "counter.py").write_text("def count(root):\n    return list((root / 'orchard').rglob('*.py'))\n", encoding="utf-8")
    (scripts / "plain.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    uses = tmp_path / "test_uses.py"
    uses.write_text("NAME = 'counter.py'\n\n\ndef test_x():\n    pass\n", encoding="utf-8")
    other = tmp_path / "test_other.py"
    other.write_text("NAME = 'plain.py'\n\n\ndef test_x():\n    pass\n", encoding="utf-8")
    assert scans_the_source_tree(uses, scripts_dir=scripts) and not scans_the_source_tree(other, scripts_dir=scripts)


def test_every_not_a_guard_entry_gives_a_reason_and_names_an_existing_file():
    for rel, why in tiers.NOT_GUARDS.items():
        assert (REPO / rel).is_file() and len(why.split()) >= 4, rel


# ---- the markers in pytest --------------------------------------------------------------------------

def test_the_markers_are_registered_in_pyproject_with_strict_checking():
    registered = {line.split(":")[0].strip() for line in OPTIONS["markers"]}
    assert {"slow", "whole_repo", "hardware", "real_bundle"} <= registered
    assert "--strict-markers" in OPTIONS["addopts"]


def pytest_run(*args, cwd=REPO):
    return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args], cwd=cwd, capture_output=True, text=True, timeout=300)


def test_a_misspelt_marker_fails_collection(tmp_path):
    (tmp_path / "test_bad.py").write_text("import pytest\n\n\n@pytest.mark.sloww\ndef test_x():\n    pass\n", encoding="utf-8")
    r = pytest_run("-c", str(REPO / "pyproject.toml"), "--rootdir", str(tmp_path), "--collect-only", "-q", str(tmp_path / "test_bad.py"))
    assert r.returncode != 0 and "sloww" in r.stdout + r.stderr


def collected(*args):
    r = pytest_run("--collect-only", "-q", "-p", "no:xdist", *args)
    assert r.returncode == 0, r.stdout + r.stderr
    return [line for line in r.stdout.splitlines() if "::" in line]


def test_collection_applies_the_markers_from_the_lists():
    guard = next(rel for rel, names in tiers.WHOLE_REPO.items() if not names and rel not in tiers.SLOW_FILES)
    slow = next(rel for rel in sorted(tiers.SLOW_FILES) if rel not in tiers.WHOLE_REPO)
    plain = "tests/test_tier_lists.py"
    files = [guard, slow, plain]
    everything = collected(*files)
    not_slow = collected("-m", "not slow", *files)
    only_guards = collected("-m", "whole_repo", *files)
    assert any(line.startswith(slow) for line in everything) and not any(line.startswith(slow) for line in not_slow)
    assert any(line.startswith(plain) for line in not_slow)                # an unlisted file runs by default
    assert any(line.startswith(guard) for line in not_slow)
    assert only_guards and all(line.startswith(guard) for line in only_guards)


# ---- the evidence behind the guard list -------------------------------------------------------------

# The failures of the fault run of 2026-11-23 (see tests/tier_lists.py): one scratch copy of main with a module that has raw file access,
# an unchecked float, direct sqlite use and an unregistered reader, a new flag the README does not name, a home path in a skill file and in a
# package template, and a malformed change fragment, then one full suite. Each of these tests failed, and each must be a guard.
FAULT_RUN_FAILURES = [
    ("tests/test_finite_ratchet.py", "test_no_file_gained_a_float_check_that_a_huge_integer_can_crash"),
    ("tests/test_hostile_readers.py", "test_the_registry_covers_these_modules_and_the_rest_are_listed"),
    ("tests/test_no_sql_outside_the_database_layer.py", "test_no_module_but_the_database_layer_holds_sql_or_imports_sqlite3"),
    ("tests/test_hardening_ratchet.py", "test_no_file_gained_raw_file_access"),
    ("tests/test_readme.py", "test_every_flag_of_the_tt_orchard_command_is_in_the_readme"),
    ("tests/test_skills.py", "test_no_agent_facing_file_names_a_machine_path"),
    ("tests/test_version.py", "test_every_fragment_in_the_repository_passes_check"),
    ("tests/test_version.py", "test_no_fragment_in_the_repository_is_at_or_below_the_release"),
    ("tests/test_version_cli.py", "test_the_script_check_passes_on_this_repository"),
]


def test_every_test_that_failed_in_the_fault_run_is_a_guard():
    for rel, name in FAULT_RUN_FAILURES:
        assert "whole_repo" in tiers.markers_for(rel, name), f"{rel}::{name} failed in the fault run and is not marked whole_repo"
        assert defined_tests(REPO / rel) >= {name}, f"{rel} no longer has {name}"


def test_a_guard_in_a_slow_file_is_never_marked_slow():
    for rel, name in FAULT_RUN_FAILURES:
        assert "slow" not in tiers.markers_for(rel, name), f"{rel}::{name} would be left out by -m 'not slow'"


# ---- the hook in conftest.py ------------------------------------------------------------------------

class StubItem:
    def __init__(self, rel, name):
        self.path = REPO / rel
        self.name = name
        self.originalname = name.split("[")[0]
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker.name if hasattr(marker, "name") else str(marker))

    def get_closest_marker(self, name):
        return next((m for m in self.markers if m == name), None)


def test_the_conftest_hook_marks_a_parametrized_test_by_its_function_name(monkeypatch):
    import conftest
    rel, name = FAULT_RUN_FAILURES[1]                                       # a guard picked by function name, in a slow file
    item = StubItem(rel, name + "[skills/x.md]")
    conftest.pytest_collection_modifyitems(None, [item])
    assert "whole_repo" in item.markers and "slow" not in item.markers


def test_the_conftest_hook_skips_a_hardware_test_unless_the_operator_opts_in(monkeypatch):
    import conftest
    monkeypatch.setattr(tiers, "HARDWARE_FILES", frozenset({"tests/test_on_a_chip.py"}))
    monkeypatch.delenv("ORCHARD_RUN_HARDWARE", raising=False)
    item = StubItem("tests/test_on_a_chip.py", "test_x")
    conftest.pytest_collection_modifyitems(None, [item])
    assert item.markers.count("hardware") == 1 and "skip" in item.markers
    monkeypatch.setenv("ORCHARD_RUN_HARDWARE", "1")
    opted = StubItem("tests/test_on_a_chip.py", "test_x")
    conftest.pytest_collection_modifyitems(None, [opted])
    assert "hardware" in opted.markers and "skip" not in opted.markers


def test_a_copy_of_conftest_run_away_from_the_tests_directory_still_works(tmp_path):
    """tests/test_conftest_guard.py runs a copy of conftest.py in an empty directory. The tier hook must not fail collection there."""
    import shutil
    shutil.copy(TESTS / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_inner.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)], cwd=tmp_path,
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "1 passed" in r.stdout, r.stdout + r.stderr

