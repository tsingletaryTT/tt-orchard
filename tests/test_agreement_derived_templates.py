# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Every place that copies serve_and_compare.py next to a config must also copy agreement_core.py and agreement_prompts.json when the config
sets `agreement_set` (Copilot review of PR 394): the copy of serve_and_compare.py imports agreement_core when the flag is on, and fails before
it starts the server when the file is missing."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import suite_control

TEMPLATES = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "weights-swap-templates"
BASE_CFG = {"run_dir": "RUN", "nearest_model_id": "a/b", "base_snapshot": "x", "new_snapshot": "y", "new_model_id": "c/d", "tt_cache": "t",
            "hf_home": "h", "port": 1}


def load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", TEMPLATES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lists():
    """(name, function from a config to the file names it copies) for each derived template."""
    return [("graft_trial", load("graft_trial").shipped), ("sampling_trial", load("sampling_trial").shipped),
            ("suite_control", suite_control.files_for)]


def import_check(directory: Path, cfg: dict) -> subprocess.CompletedProcess:
    """Run serve_and_compare's agreement preflight from `directory` as the stage would. With the helper present it exits 2 about the missing
    reference; with it missing it dies on an ImportError."""
    (directory / "cfg-for-check.json").write_text(json.dumps({**cfg, "run_dir": str(directory)}))
    code = ("import json, sys; sys.path.insert(0, '.'); import serve_and_compare as s; "
            "s.agreement_preflight(json.load(open('cfg-for-check.json')))")
    return subprocess.run([sys.executable, "-c", code], cwd=directory, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("name,files", lists(), ids=[n for n, _ in lists()])
def test_the_copy_list_holds_the_agreement_files_exactly_when_the_flag_is_on(name, files):
    on = set(files({**BASE_CFG, "agreement_set": True}))
    off = set(files(BASE_CFG))
    assert {"agreement_core.py", "agreement_prompts.json"} <= on
    assert not ({"agreement_core.py", "agreement_prompts.json"} & off)


@pytest.mark.parametrize("name,files", lists(), ids=[n for n, _ in lists()])
def test_a_directory_built_from_the_copy_list_can_import_the_agreement_helper(name, files, tmp_path):
    cfg = {**BASE_CFG, "agreement_set": True}
    for n in files(cfg):
        shutil.copy(TEMPLATES / n, tmp_path / n)
    r = import_check(tmp_path, cfg)
    assert r.returncode == 2 and "agreement reference" in r.stderr and "ModuleNotFoundError" not in r.stderr, r.stderr


def test_the_check_itself_fails_when_the_helper_is_left_out(tmp_path):
    cfg = {**BASE_CFG, "agreement_set": True}
    for n in ("serve_and_compare.py",):
        shutil.copy(TEMPLATES / n, tmp_path / n)
    r = import_check(tmp_path, cfg)
    assert r.returncode != 2 and "agreement_core" in r.stderr


def test_the_supervisors_template_list_names_the_same_files():
    from orchard import supervisor
    assert set(supervisor.AGREEMENT_TEMPLATES) == {"agreement_core.py", "agreement_prompts.json"}
    assert all((TEMPLATES / n).is_file() for n in supervisor.AGREEMENT_TEMPLATES)
