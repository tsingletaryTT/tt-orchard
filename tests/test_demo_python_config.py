# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`demo_python` in bringup.toml (issue 92): the interpreter that runs tt-metal demos, handed to the stock-runtime stage the way
`reference_python` is handed to stage 1."""
from pathlib import Path

import pytest

from orchard import bringup_config as bc
from orchard import supervisor
from test_bringup_config import MINIMAL, load


def test_a_demo_python_is_read_and_resolved_against_the_config_directory(tmp_path):
    cfg = load(tmp_path, 'demo_python = "venvs/demo/bin/python"\n' + MINIMAL)
    assert cfg.demo_python == tmp_path / "venvs" / "demo" / "bin" / "python"
    assert load(tmp_path, 'demo_python = "/abs/py"\n' + MINIMAL).demo_python == Path("/abs/py")
    assert load(tmp_path).demo_python is None


def test_the_demo_python_is_handed_to_the_run_as_an_input(tmp_path):
    argv = bc.supervisor_argv(load(tmp_path, 'demo_python = "/v/demo/bin/python"\n' + MINIMAL), "Cloudflare/clef",
                              inputs={"model": "/hf/snap"})
    inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"]
    assert inputs == ["model=/hf/snap", "demo_python=/v/demo/bin/python"]
    assert supervisor.parse(argv).input == inputs


def test_both_interpreters_are_handed_over_and_stay_separate(tmp_path):
    cfg = load(tmp_path, 'reference_python = "/v/ref/bin/python"\ndemo_python = "/v/demo/bin/python"\n' + MINIMAL)
    argv = bc.supervisor_argv(cfg, "Cloudflare/clef")
    inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"]
    assert "reference_python=/v/ref/bin/python" in inputs and "demo_python=/v/demo/bin/python" in inputs


def test_without_a_demo_python_no_such_input_is_passed(tmp_path):
    argv = bc.supervisor_argv(load(tmp_path), "Cloudflare/clef", inputs={"model": "/hf/snap"})
    assert not any(a.startswith("demo_python=") for a in argv)


def test_an_empty_demo_python_is_refused(tmp_path):
    with pytest.raises(bc.BringupConfigError, match="demo_python"):
        load(tmp_path, 'demo_python = ""\n' + MINIMAL)


def test_the_example_config_lists_the_key_and_it_is_a_known_key():
    assert "demo_python" in bc.TOP_KEYS
    text = (Path(__file__).resolve().parent.parent / "config" / "bringup.example.toml").read_text()
    assert "# demo_python =" in text
