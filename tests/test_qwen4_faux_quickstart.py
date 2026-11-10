"""The Qwen4-faux quick start (examples/qwen4_faux_quickstart.py) and its page (docs/qwen4-faux.md).

The script is not run here: it needs transformers 5.19.0, which the test interpreter may not have. These tests check that it parses, that
it keeps the functions the page names, that the shim it loads is where it says, and that the page's links and cited names exist.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "examples" / "qwen4_faux_quickstart.py"
PAGE = ROOT / "docs" / "qwen4-faux.md"


def test_the_quickstart_parses_and_keeps_the_functions_the_page_names():
    tree = ast.parse(SCRIPT.read_text())
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert {"load_shim", "load_model", "greedy", "main"} <= names
    page = PAGE.read_text()
    for name in ("load_model", "examples/qwen4_faux_quickstart.py"):
        assert name in page


def test_the_shim_the_quickstart_loads_exists_and_has_the_two_entry_points():
    text = SCRIPT.read_text()
    assert 'orchard" / "skills" / "reference-gate-templates" / "qwen4_dense.py' in text
    shim = ROOT / "orchard" / "skills" / "reference-gate-templates" / "qwen4_dense.py"
    source = shim.read_text()
    assert "def prepare_config" in source and "def dense_layers" in source


def test_the_page_cites_files_that_exist():
    page = PAGE.read_text()
    for ref in sorted(set(re.findall(r"`((?:docs|scripts|orchard|examples)/[A-Za-z0-9_./-]+\.(?:md|py|json))`", page))):
        assert (ROOT / ref).exists(), ref


def test_the_page_names_the_evidence_directory_and_it_holds_the_proofs_it_cites():
    page = PAGE.read_text()
    directory = ROOT / "docs" / "run-logs" / "evidence" / "2026-11-08-qwen4-franken"
    assert "docs/run-logs/evidence/2026-11-08-qwen4-franken/" in page
    for name in ("proofs-k4-f32.json", "proofs-k20-bf16.json"):
        assert name in page and (directory / name).is_file()


def test_the_page_holds_no_machine_path_and_marks_its_status_labels():
    page = PAGE.read_text()
    assert "/home/" not in page and "/mnt/" not in page
    for label in ("Verified", "Assumed", "Unknown"):
        assert label in page
