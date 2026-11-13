# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The CPU run of a stage 2 port before any board check (issue 697).

`port_check.py --cpu N` runs the port's load, one prefill of the stage 1 prompt and one decode step in a worker that imports the torch
stand-in of ttnn (orchard/skills/full-port-templates/cpu_ttnn.py) behind the verified cheat sheet's signatures. Exit 0 is a pass, exit 2 a
fault of the port (the same FAILED and PORT_CHECK_FAILURE lines a board failure prints), exit 3 no verdict. The supervisor runs it after the
static device-ops check, before it leases a board or parks the coder, and a fault becomes the attempt's test result, so the repair rounds get
it without a board.

The template tests run the shipped template in a subprocess with this interpreter (torch is needed by the stand-in). The real ttnn is never
imported: a decoy package named ttnn on PYTHONPATH records an import, and the tests check that it stays silent.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import full_port
from orchard.supervisor import EXIT_BLOCKED

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "orchard" / "skills" / "full-port-templates"
PROMPT = [3, 7, 11]
FIRST = 5

# A port that runs on the stand-in: one upload, one projection, a reshape, a read-back. Every call matches the sheet.
GOOD = '''import torch
import ttnn


def load(model_dir, mesh_device):
    w = ttnn.from_torch(torch.ones(32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    return {"w": w, "dev": mesh_device}


def _forward(model, n):
    x = ttnn.from_torch(torch.ones(1, 1, 32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=model["dev"])
    y = ttnn.linear(x, model["w"])
    y = ttnn.reshape(y, (1, 1, 32, 32))
    return ttnn.to_torch(y)[0, 0, -1]


def prefill(model, ids):
    return _forward(model, len(ids))


def decode_step(model, token_id, position):
    return _forward(model, 1)


def close(model):
    pass
'''


def stage(tmp_path: Path, port_text: str, *, sheet: bool = True, stand_in: bool = True) -> Path:
    """A run directory shaped like the supervisor's: stages/2 holds the template, the stand-in and the sheet; port/ holds the port; stage 1
    holds the prompt and reference ids; the config names a small snapshot directory."""
    run = tmp_path / "run"
    d = run / "stages" / "2"
    (d / "port").mkdir(parents=True)
    shutil.copy(TEMPLATES / "port_check.py", d / "port_check.py")
    if stand_in:
        shutil.copy(ROOT / full_port.CPU_TTNN_SOURCE, d / full_port.CPU_TTNN_NAME)
        shutil.copy(ROOT / full_port.CPU_GDN_SOURCE, d / full_port.CPU_GDN_NAME)
    if sheet:
        shutil.copy(full_port.CHEAT_SOURCE, d / full_port.CHEAT_NAME)
    (d / "port" / "port_model.py").write_text(port_text)
    ref = run / "stages" / "1" / "evidence" / "reference"
    ref.mkdir(parents=True)
    (ref / "prompt-ids.json").write_text(json.dumps({"prompt_ids": PROMPT}))
    (ref / "generated-ids.json").write_text(json.dumps({"generated_ids": [FIRST, 6, 7]}))
    snap = tmp_path / "snapshot"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    (d / full_port.CONFIG_NAME).write_text(json.dumps({"model_snapshot": str(snap)}))
    decoy = tmp_path / "decoy" / "ttnn"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text(f"open({str(tmp_path / 'REAL-TTNN-IMPORTED')!r}, 'w').write('yes')\n")
    return d


def run_cpu(d: Path, chips: str = "1"):
    tmp = d.parents[2]
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(tmp / "decoy")}          # no TT_VISIBLE_DEVICES: this mode needs no chip
    r = subprocess.run([sys.executable, "-P", "-s", str(d / "port_check.py"), "--cpu", chips], capture_output=True, text=True, env=env,
                       cwd=str(d.parent.parent), timeout=300)
    assert not (tmp / "REAL-TTNN-IMPORTED").exists(), "the CPU run imported the real ttnn"
    return r


def failed_line(r) -> str:
    """The FAILED line, with the escaped quotes of the worker's message read back as plain quotes."""
    return next(ln for ln in r.stdout.splitlines() if ln.startswith("FAILED: ")).replace("\\'", "'")


def no_verdict_line(r) -> str:
    return next(ln for ln in r.stdout.splitlines() if ln.startswith("CPU run gave no verdict: "))


# ---- the template: what passes, what is a fault of the port, what gives no verdict ---------------------------------------------------

def test_a_port_that_runs_on_the_stand_in_passes_and_writes_no_record(tmp_path):
    d = stage(tmp_path, GOOD)
    r = run_cpu(d)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CPU run passed" in r.stdout and "No board was used" in r.stdout and "FAILED" not in r.stdout
    assert not (d / "evidence" / "port-check.json").exists()


def test_a_generated_port_passes_the_cpu_run(tmp_path):
    """The generator's own output for the small standard decoder runs on the stand-in behind the sheet, on one and on two stand-in chips."""
    from test_port_template import std_generate, std_weights, write_checkpoint
    weights = std_weights()
    text, plan = std_generate(weights=weights)
    assert plan.stubs == []
    d = stage(tmp_path, text)
    (tmp_path / "gen").mkdir()
    snap = write_checkpoint(tmp_path / "gen", weights)
    (d / full_port.CONFIG_NAME).write_text(json.dumps({"model_snapshot": str(snap)}))
    for chips in ("1", "2"):
        r = run_cpu(d, chips)
        assert r.returncode == 0, r.stdout + r.stderr
        assert f"on {chips} stand-in chip(s)" in r.stdout


@pytest.mark.parametrize("old, new, op, says", [
    # wrong argument counts to ttnn ops, from the sheet (the commonest board failures of arms C to E8)
    ("ttnn.reshape(y, (1, 1, 32, 32))", "ttnn.reshape(y, 1, 1, 32, 32)", "prefill", "ttnn.reshape takes 3 positional argument(s) in the verified sheet"),
    ("ttnn.linear(x, model[\"w\"])", "ttnn.linear(x, model[\"w\"], bogus=1)", "prefill", "ttnn.linear passes 'bogus'"),
    ("ttnn.linear(x, model[\"w\"])", "ttnn.linear(x)", "prefill", "ttnn.linear is missing the argument(s) input_tensor_b"),
    # a missing dict key, a missing name, an index past the end, a bad import, a wrong call to the port's own function
    ("return _forward(model, len(ids))", "return _forward(model, model['absent'])", "prefill", "KeyError"),
    ("    return {\"w\": w, \"dev\": mesh_device}", "    return {\"w\": w, \"dev\": undefined_name}", "load", "NameError"),
    ("return ttnn.to_torch(y)[0, 0, -1]", "return ttnn.to_torch(y)[0, 0, -1][[0]] if [][0] else None", "prefill", "IndexError"),
    ("import torch\n", "import torch\nimport no_such_module_for_the_port\n", "init", "ModuleNotFoundError"),
    ("return _forward(model, 1)", "return _forward(model)", "decode", "missing 1 required positional argument"),
])
def test_a_fault_of_the_port_fails_the_cpu_run_with_the_line_and_the_repair_shape(tmp_path, old, new, op, says):
    assert old in GOOD
    d = stage(tmp_path, GOOD.replace(old, new, 1))
    r = run_cpu(d)
    assert r.returncode == 2, r.stdout + r.stderr
    line = failed_line(r)
    assert line.startswith(f"FAILED: the port failed at '{op}'") and says in line, line            # the board check's own form, which repair parses
    assert "CPU run found a fault of the port" in r.stdout and "No board was used" in r.stdout
    if op != "init":
        assert "[port_model.py:" in line                                       # the port's own line is named
    assert f'PORT_CHECK_FAILURE {{"kind": "port", "op": "{op}"}}' in r.stdout
    assert str(tmp_path) not in line


@pytest.mark.parametrize("old, new, why", [
    # a ttnn name the stand-in lacks (it may exist on the device): the stand-in's gap
    ("y = ttnn.linear(x, model[\"w\"])", "y = ttnn.not_an_op_of_the_stand_in(x)", "is not in the CPU stand-in of ttnn"),
    ("import ttnn\n", "import ttnn\nimport ttnn.not_a_namespace_here\n", "a ttnn module the CPU stand-in does not provide"),
    # an error raised inside the stand-in (here a width mismatch): the device may not raise it the same way
    ("torch.ones(1, 1, 32, 32", "torch.ones(1, 1, 32, 64", "raised outside the port's files"),
    # an AttributeError at a port line can come from the stand-in's tensors
    ("return ttnn.to_torch(y)[0, 0, -1]", "return y.padded_shape_of_the_device", "can come from the stand-in's tensors"),
    # the sheet allows the call and the stand-in's own form does not take it
    ("ttnn.reshape(y, (1, 1, 32, 32))", "ttnn.reshape(y, (1, 1, 32, 32), memory_config=None, pad_value=0)", None),
])
def test_what_the_port_may_not_be_at_fault_for_gives_no_verdict(tmp_path, old, new, why):
    assert old in GOOD
    d = stage(tmp_path, GOOD.replace(old, new, 1))
    r = run_cpu(d)
    if why is None:          # reshape's sheet signature is full and does not list pad_value: the sheet refuses it, which is a fault
        assert r.returncode == 2 and "passes 'pad_value'" in failed_line(r)
        return
    assert r.returncode == 3, r.stdout + r.stderr
    assert why in no_verdict_line(r) and "FAILED" not in r.stdout and "PORT_CHECK_FAILURE" not in r.stdout


def test_the_sheet_decides_the_argument_check(tmp_path):
    """Without the sheet in the stage directory the reshape call with four loose sizes still fails on the stand-in's own form, which is a gap
    of the stand-in and gives no verdict. With the sheet it is a fault. So the verdict comes from the sheet's signature."""
    bad = GOOD.replace("ttnn.reshape(y, (1, 1, 32, 32))", "ttnn.reshape(y, 1, 1, 32, 32)")
    assert run_cpu(stage(tmp_path / "with", bad)).returncode == 2
    r = run_cpu(stage(tmp_path / "without", bad, sheet=False))
    assert r.returncode == 3 and "does not take this call" in no_verdict_line(r)
    assert no_verdict_line(r).endswith("(the CPU stand-in does not cover this call)")


def test_a_sheet_signature_that_lists_keywords_refuses_an_unlisted_keyword(tmp_path):
    """`ttnn.multiply` lists its keyword `input_tensor_a_activations` since issue 759, so the check treats the signature as complete: a keyword the sheet
    does not list is a fault of the port (exit 2), and so is a third input. (Keywords are checked only where the sheet lists them; no entry of the
    sheet is a short form any more for multiply and add.)"""
    kw = GOOD.replace("y = ttnn.reshape(y, (1, 1, 32, 32))", "y = ttnn.multiply(y, y, memory_config=None)\n    y = ttnn.reshape(y, (1, 1, 32, 32))")
    r = run_cpu(stage(tmp_path / "kw", kw))
    assert r.returncode == 2 and "input_tensor_a_activations" in failed_line(r)                  # the sheet lists one keyword and memory_config is not it
    three = GOOD.replace("y = ttnn.reshape(y, (1, 1, 32, 32))", "y = ttnn.add(y, y, y)\n    y = ttnn.reshape(y, (1, 1, 32, 32))")
    r = run_cpu(stage(tmp_path / "three", three))
    assert r.returncode == 2 and "ttnn.add takes 2 positional argument(s)" in failed_line(r)


def test_no_stand_in_gives_no_verdict(tmp_path):
    r = run_cpu(stage(tmp_path, GOOD, stand_in=False))
    assert r.returncode == 3 and "CPU run gave no verdict" in r.stdout


def test_a_checkpoint_over_the_limit_is_not_loaded(tmp_path):
    marker = tmp_path / "LOAD-RAN"
    d = stage(tmp_path, GOOD.replace("def load(model_dir, mesh_device):\n", f"def load(model_dir, mesh_device):\n    open({str(marker)!r}, 'w').close()\n"))
    with open(tmp_path / "snapshot" / "model.safetensors", "wb") as fh:          # sparse: no disk space is used
        fh.truncate((4 << 30) + 1)
    r = run_cpu(d)
    assert r.returncode == 3 and "above the CPU run's limit" in no_verdict_line(r) and not marker.exists()


def test_missing_stage_1_ids_give_no_verdict(tmp_path):
    d = stage(tmp_path, GOOD)
    (d.parents[1] / "stages" / "1" / "evidence" / "reference" / "prompt-ids.json").unlink()
    r = run_cpu(d)
    assert r.returncode == 3 and "its inputs do not read" in no_verdict_line(r)


def test_a_port_that_prints_the_mark_fails(tmp_path):
    d = stage(tmp_path, GOOD.replace("    pass\n", "    print('PORT_CHECK_SHA256 ' + '0' * 64)\n"))
    r = run_cpu(d)
    assert r.returncode == 2 and "only this file prints that line" in failed_line(r)


def test_the_cpu_flag_needs_a_chip_count(tmp_path):
    d = stage(tmp_path, GOOD)
    for flags in ([], ["x"], ["0"], ["1", "2"]):
        r = subprocess.run([sys.executable, "-P", "-s", str(d / "port_check.py"), "--cpu", *flags], capture_output=True, text=True,
                           env={"PATH": os.environ.get("PATH", "")}, cwd=str(d.parent.parent), timeout=60)
        assert r.returncode == 2 and "usage: port_check.py --cpu N" in r.stdout


# ---- the supervisor: a fault found on the CPU goes to repair with no board -----------------------------------------------------------

@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_cpu_fault_is_the_attempts_test_result_before_any_lease_or_park(tmp_path):
    from test_supervisor_full_port import HONEST, coder_writes, failure, stage_2, with_snapshot
    from test_supervisor import Rig
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        faulty = HONEST.replace("    return _peak(100)", "    return _peak(model['absent'])")
        coder_writes(r, {}, port=faulty)
        assert r.run() == EXIT_BLOCKED
        decisions = [e["data"] for e in r.entries() if e["event"] == "decision" and e["stage"] == 2]
        names = [d.get("decision") for d in decisions]
        assert "CPU run failed before the board" in names, names
        assert not {"hardware phase", "hardware test started", "test lease taken", "park"} & set(names)       # no lease, no park, no test
        note = next(d for d in decisions if d.get("decision") == "CPU run failed before the board")
        assert "KeyError" in note["message"] and str(tmp_path) not in json.dumps(note)
        marker = json.loads((stage_2(r) / "test-result.json").read_text())
        assert marker["returncode"] == 2 and marker["timed_out"] is False and marker["chips"] == []
        out = (stage_2(r) / "evidence" / "hw-test-output.txt").read_text()
        from orchard import thin_repair
        assert thin_repair.parse_failed(out).op == "prefill"          # the thin repair reads the line as it reads a board failure
        assert "CPU run found a fault of the port" in out and "FAILED: the port failed at 'prefill'" in out and 'PORT_CHECK_FAILURE {"kind": "port", "op": "prefill"}' in out
        assert "exited with code 2" in failure(r)["reason"]
        from orchard import port_repair
        assert port_repair.read_failure(r.run_dir, stage_2(r))["op"] == "prefill"         # the repair rounds read it as a board failure
    finally:
        r.close()


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_port_the_cpu_run_cannot_judge_still_goes_to_the_board(tmp_path):
    from test_supervisor_full_port import HONEST, coder_writes, stage_2, with_snapshot
    from test_supervisor import Rig
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        coder_writes(r, {}, port=HONEST)            # ttnn.linear(None, None): the stand-in raises inside itself, which is no verdict
        r.run()
        decisions = [e["data"] for e in r.entries() if e["event"] == "decision" and e["stage"] == 2]
        names = [d.get("decision") for d in decisions]
        assert "CPU run gave no verdict" in names and "hardware test started" in names
        assert names.index("CPU run gave no verdict") < names.index("hardware test started")
        nv = next(d for d in decisions if d.get("decision") == "CPU run gave no verdict")
        assert nv["reason"].startswith("CPU run gave no verdict:") and str(tmp_path) not in nv["reason"]
        assert json.loads((stage_2(r) / "test-result.json").read_text())["returncode"] == 0
        for name in (full_port.CPU_TTNN_NAME, full_port.CPU_GDN_NAME):
            assert (stage_2(r) / name).read_bytes() == (ROOT / (full_port.CPU_TTNN_SOURCE if name == full_port.CPU_TTNN_NAME
                                                                 else full_port.CPU_GDN_SOURCE)).read_bytes()
    finally:
        r.close()


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_the_cpu_run_runs_again_after_each_repair_and_a_repaired_port_reaches_the_board(tmp_path):
    """A port that fails at load (a missing checkpoint key) is caught on the CPU; the repair round writes a port the CPU run cannot judge; the
    rerun goes through the CPU run again and then to the board, where it passes. Two CPU runs, one board test, no board test for the fault."""
    from test_supervisor_full_port import HONEST, with_snapshot
    from test_supervisor import Rig
    from test_supervisor_port_repair import BROKEN, exit_codes, rounds, start
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        start(r, port=BROKEN, per_round=[{"port/port_model.py": HONEST}], unattended=False)
        r.run()
        names = [e["data"].get("decision") for e in r.entries() if e["event"] == "decision" and e["stage"] == 2]
        cpu = [n for n in names if n and n.startswith("CPU run")]
        assert cpu == ["CPU run failed before the board", "CPU run gave no verdict"], cpu
        assert names.index("CPU run failed before the board") < names.index("port repair round") < names.index("CPU run gave no verdict") \
            < names.index("hardware test started")
        assert names.count("hardware test started") == 1 and len(rounds(r)) == 1
        assert exit_codes(r) == [2, 0]                     # the CPU fault and the board pass, in the shape the repair rounds read
        assert [d["result"] for d in r.ends(2)] == ["pass"]
    finally:
        r.close()
