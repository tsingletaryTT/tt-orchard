# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Copilot's review of PR 80 on orchard/suite_control.py. Each test group names the finding it covers."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_suite_control import CHECKOUT, _prep, build, ok_run, prep, stage_cfg  # noqa: E402,F401

from orchard import suite_control as sc  # noqa: E402


def make_container_stage(prep, **extra):
    c = stage_cfg(prep)
    c.pop("bundle_dir")
    c.update(package="org/pkg", profile="p", chips=4, operator_home=str(prep["stage"].parent), **extra)
    (prep["stage"] / "swap_config.json").write_text(json.dumps(c))
    return c


# ---- 1: the container template imports serve_and_compare.py ------------------------------------------------------------

def test_a_container_control_copies_serve_and_compare_too(prep):
    make_container_stage(prep)
    out = sc.prepare(prep["stage"], orchard_dir=CHECKOUT, run=ok_run)
    assert (Path(out["control_dir"]) / "serve_and_compare.py").is_file()
    assert "serve_and_compare.py" in sc.files_for({"package": "p"})
    assert "serve_and_compare.py" in sc.files_for({"bundle_dir": "b"})


def test_a_checkout_missing_serve_and_compare_is_named_for_a_container(prep, tmp_path):
    make_container_stage(prep)
    fake = tmp_path / "fake"
    t = fake / "orchard" / "skills" / "weights-swap-templates"
    t.mkdir(parents=True)
    for n in ("prepare_swap.py", "serve_and_compare_container.py", "prompt_suite.json"):
        (t / n).write_text("# x\n")
    with pytest.raises(sc.ControlError, match="serve_and_compare.py"):
        sc.prepare(prep["stage"], orchard_dir=fake, run=ok_run)
