# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The optional `layer_outputs(model, ids)` hook of a port module (issue 505, orchard/skills/full-port-templates/port_check.py).

When a check ends below the gate's agreement bar and the port module defines the hook, the worker (inside the same hardware test and the same
lease) saves the per-layer hidden states to evidence/layer-port.npz, and the parent prints one `PORT_CHECK_LAYERS <sha256> <n>` line that names the
file. The supervisor's capture of the output holds that line, so `layer-diff` can tell the file the test wrote from a file written afterwards. The
record, the gate and the exit code do not change. The port runs here as the real worker subprocess on the CPU with a fake `ttnn`."""
import hashlib
import json
import re

import numpy as np
import pytest

from orchard import full_port
from port_check_fakes import HONEST, WRONG, make_run, run_template

LINE = re.compile(r"^PORT_CHECK_LAYERS ([0-9a-f]{64}) (\d+)$", re.M)

HOOK = '''
import numpy as np
def layer_outputs(model, ids):
    return [np.full((len(ids), 4), float(k), dtype=np.float32) for k in range(3)]
'''


def layers_file(stage):
    return stage / "evidence" / "layer-port.npz"


def test_a_failing_port_with_the_hook_leaves_the_layers_and_one_line_that_names_them(tmp_path):
    run, stage, _ = make_run(tmp_path, WRONG + HOOK)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout + done.stderr
    found = LINE.findall(done.stdout)
    assert len(found) == 1 and found[0][1] == "3"
    assert hashlib.sha256(layers_file(stage).read_bytes()).hexdigest() == found[0][0]
    with np.load(layers_file(stage)) as z:
        assert sorted(z.files) == ["layer_000", "layer_001", "layer_002", "prompt_ids"]
        assert z["layer_002"].shape == (3, 4) and z["prompt_ids"].tolist() == [1, 2, 3]


def test_a_passing_port_does_not_run_the_hook(tmp_path):
    run, stage, _ = make_run(tmp_path, HONEST + HOOK)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0 and not LINE.findall(done.stdout) and not layers_file(stage).exists()


def test_the_bar_is_the_gates(tmp_path):
    src = full_port.TEMPLATE.read_text()
    assert f"LAYER_DIAG_BELOW = {full_port.TOP1_MIN}" in src


def test_a_failing_port_without_the_hook_says_so_and_leaves_no_file(tmp_path):
    run, stage, _ = make_run(tmp_path, WRONG)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0 and not LINE.findall(done.stdout) and not layers_file(stage).exists()
    assert "defines no layer_outputs" in done.stdout


def test_a_hook_that_raises_does_not_change_the_result(tmp_path):
    run, stage, _ = make_run(tmp_path, WRONG + "\ndef layer_outputs(model, ids):\n    raise ValueError('boom')\n")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0 and not LINE.findall(done.stdout) and not layers_file(stage).exists()
    assert "layer outputs not written" in done.stdout and "boom" in done.stdout
    assert json.loads((stage / "evidence" / "port-check.json").read_text())["top1_agreement"] < 0.1


@pytest.mark.parametrize("returned", ["None", "[]", "3", "'abc'", "[object()]", "[np.zeros(3, dtype=object)]", "[np.full(2, np.nan)] * 600"])
def test_a_hook_that_returns_the_wrong_thing_is_refused_with_a_reason(tmp_path, returned):
    run, stage, _ = make_run(tmp_path, WRONG + f"\nimport numpy as np\ndef layer_outputs(model, ids):\n    return {returned}\n")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0 and not layers_file(stage).exists() and not LINE.findall(done.stdout)
    assert "layer outputs not written" in done.stdout


def test_a_result_with_more_elements_than_the_cap_is_refused_before_it_is_built(tmp_path):
    big = ("\nimport numpy as np\ndef layer_outputs(model, ids):\n"
           "    return [np.broadcast_to(np.zeros(1, dtype=np.float32), (2 ** 15, 2 ** 14))]\n")      # 2 ** 29 elements, no memory behind it
    run, stage, _ = make_run(tmp_path, WRONG + big)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0 and not layers_file(stage).exists() and "more than" in done.stdout


def test_a_hook_that_kills_the_worker_fails_the_run_like_any_request_that_gets_no_answer(tmp_path):
    run, stage, _ = make_run(tmp_path, WRONG + "\nimport os\ndef layer_outputs(model, ids):\n    os._exit(3)\n")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "did not answer 'layers'" in done.stdout
    assert 'PORT_CHECK_FAILURE {"kind": "port", "op": "layers"}' in done.stdout


def test_a_file_left_from_an_earlier_run_is_removed_before_the_port_runs(tmp_path):
    run, stage, _ = make_run(tmp_path, WRONG)
    layers_file(stage).parent.mkdir(parents=True, exist_ok=True)
    layers_file(stage).write_bytes(b"stale")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0 and not layers_file(stage).exists()


def test_a_port_that_prints_the_layers_line_itself_fails_the_run(tmp_path):
    forged = WRONG + HOOK + "\nprint('PORT_CHECK_LAYERS ' + '0' * 64 + ' 3')\n"
    run, stage, _ = make_run(tmp_path, forged)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "PORT_CHECK" in done.stdout


def test_the_gate_and_the_record_are_unchanged_by_the_hook(tmp_path):
    from orchard.stages import gate_full_port
    from port_check_fakes import write_result
    run, stage, _ = make_run(tmp_path, WRONG + HOOK)
    run_template(tmp_path, run, stage)
    write_result(stage)
    reasons = gate_full_port(stage, run).reasons
    assert any("below the minimum" in r or "agreement" in r for r in reasons) and not any("PORT_CHECK_LAYERS" in r for r in reasons)
