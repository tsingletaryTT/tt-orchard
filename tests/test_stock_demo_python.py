# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""demo_python in the stock-runtime stage (issue 92): the interpreter every child of stock_check.py runs under, and the facts that
carry it from bringup.toml into stock_config.json."""
import sys

import pytest

from test_stock_check_template import Rig
from test_stock_draft import PARAMS, draft


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def wrapper(rig):
    """A stand-in for the demo venv's python: it logs its arguments and runs the test interpreter."""
    log = rig.tmp / "demo-python.log"
    script = rig.tmp / "demo-python"
    script.write_text(f'#!/bin/sh\necho "$@" >> {log}\nexec {sys.executable} "$@"\n')
    script.chmod(0o755)
    rig.cfg["demo_python"] = str(script)
    rig.write_cfg()
    return log


def test_prepare_runs_the_import_check_and_the_reference_under_demo_python(rig):
    log = wrapper(rig)
    rig.cfg["import_check"] = ["json"]
    rig.write_cfg()
    done = rig.go("prepare")
    assert done.returncode == 0, done.stderr
    calls = log.read_text().splitlines()
    assert "-c import json" in calls
    assert any("generate_reference_hf.py" in c for c in calls)


def test_the_import_check_judges_demo_python_and_not_the_interpreter_running_the_template(rig):
    """The template itself is stdlib only; what has to import numpy and cv2 is the interpreter that runs the demo."""
    script = rig.tmp / "broken-python"
    script.write_text('#!/bin/sh\necho "ImportError: numpy is broken in the demo venv" >&2\nexit 1\n')
    script.chmod(0o755)
    rig.cfg["demo_python"] = str(script)
    rig.cfg["import_check"] = ["json"]
    rig.write_cfg()
    done = rig.go("prepare")
    assert done.returncode == 2 and "numpy is broken in the demo venv" in done.stderr
    assert not (rig.run_dir / "tt-metal-scratch").exists()


def test_run_starts_the_demo_with_demo_python(rig):
    log = wrapper(rig)
    rig.go("prepare")
    rig.go("run")
    assert any(c.startswith("-m pytest models/tt_transformers/demo/simple_text_demo.py") for c in log.read_text().splitlines())


def test_without_demo_python_the_children_run_under_the_template_interpreter(rig):
    log = rig.tmp / "demo-python.log"
    rig.go("prepare")
    rig.go("run")
    assert not log.exists()


def test_the_stage_skill_tells_the_agent_not_to_install_packages_or_set_the_path_itself():
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "orchard" / "skills" / "stock-runtime-check.md").read_text()
    assert "demo_python" in text and "Do not install packages" in text


@pytest.mark.parametrize("value,message", [("relative/python", "absolute"), ("/nonexistent/demo/bin/python", "demo_python"),
                                           (7, "demo_python")])
def test_a_bad_demo_python_is_exit_2_before_anything_else(rig, value, message):
    rig.cfg["demo_python"] = value
    rig.write_cfg()
    done = rig.go("prepare")
    assert done.returncode == 2 and message in done.stderr
    assert not (rig.run_dir / "tt-metal-scratch").exists()


# ---- the drafted config ---------------------------------------------------------------------------------------------

def test_a_demo_python_input_goes_into_the_config(tmp_path):
    cfg, problems = draft(tmp_path, inputs={"model": "/hf/snap", "stock_params": PARAMS, "demo_python": "/v/demo/bin/python"})
    assert problems == [] and cfg["demo_python"] == "/v/demo/bin/python"


def test_no_demo_python_input_leaves_the_key_out(tmp_path):
    cfg, _ = draft(tmp_path)
    assert "demo_python" not in cfg


@pytest.mark.parametrize("bad", ["relative/python", "", 5])
def test_a_demo_python_that_is_not_an_absolute_path_is_a_problem(tmp_path, bad):
    cfg, problems = draft(tmp_path, inputs={"model": "/hf/snap", "stock_params": PARAMS, "demo_python": bad})
    assert cfg is None and any("demo_python" in p for p in problems)
