# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""demo_python in lab mode (review of PR 101): the stock-runtime route and the demo-env check are off for a lab run, and
`supervisor.lab_problems` wants every absolute input under the lab root, so a home-directory demo_python must not be passed."""
from types import SimpleNamespace

from orchard import bringup_config as bc
from orchard import supervisor
from test_bringup_config import LAB, MINIMAL, load

PATHS = SimpleNamespace(cache_root="/srv/orchard/cache", hf_home="/srv/orchard/hf", tt_model_root="/srv/orchard/tt-model/models")


def problems(cfg):
    args = supervisor.parse(bc.supervisor_argv(cfg, "Altworld/Hemmingway-1"))
    return supervisor.lab_problems(args, "/srv/orchard/runs/x", PATHS, supervisor.pairs(args.input, "input"))


def test_a_lab_run_with_a_home_directory_demo_python_passes_the_lab_root_check(tmp_path):
    cfg = load(tmp_path, 'demo_python = "/home/op/orchard-venvs/tt-demo/bin/python"\n' + LAB)
    assert problems(cfg) == []


def test_a_lab_run_passes_no_demo_python_input(tmp_path):
    cfg = load(tmp_path, 'demo_python = "/home/op/orchard-venvs/tt-demo/bin/python"\n' + LAB)
    argv = bc.supervisor_argv(cfg, "Altworld/Hemmingway-1")
    assert not any(a.startswith("demo_python=") for a in argv)


def test_the_unrelated_inputs_of_a_lab_run_are_still_checked(tmp_path):
    cfg = load(tmp_path, 'demo_python = "/home/op/x/bin/python"\n' + LAB)
    argv = bc.supervisor_argv(cfg, "Altworld/Hemmingway-1", inputs={"model": "/home/op/snap"})
    args = supervisor.parse(argv)
    out = supervisor.lab_problems(args, "/srv/orchard/runs/x", PATHS, supervisor.pairs(args.input, "input"))
    assert len(out) == 1 and "--input model" in out[0]


def test_a_local_run_still_passes_the_demo_python_input(tmp_path):
    argv = bc.supervisor_argv(load(tmp_path, 'demo_python = "/v/demo/bin/python"\n' + MINIMAL), "Cloudflare/clef")
    assert "demo_python=/v/demo/bin/python" in argv


def test_a_reference_python_is_still_passed_in_a_lab_run(tmp_path):
    """Only the unused local interpreter is left out; the reference interpreter's rules are not changed here."""
    cfg = load(tmp_path, 'reference_python = "/srv/orchard/venvs/reference/bin/python"\n' + LAB)
    argv = bc.supervisor_argv(cfg, "Altworld/Hemmingway-1")
    assert "reference_python=/srv/orchard/venvs/reference/bin/python" in argv
