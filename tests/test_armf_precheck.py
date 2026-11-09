"""Arm F of the stage 2 experiment (issues 677 and 561): the coder reaches on-device compute sooner.

Four changes, each tested at the layer that decides it:
- the device-ops message puts the instruction to run each projection with ttnn.linear or ttnn.matmul first (orchard/ttnn_calls.py, copied into
  the port check template);
- the TTNN cheat sheet starts with its Linear and matmul section;
- `port_check.py --static` runs only the static device-ops analysis (no chip, no worker), and the supervisor runs it before it leases a board or
  parks the coder (`Supervisor._device_ops_precheck`), leaving the same test record and output a failed board test leaves;
- `stage_tools.py gate-check` runs the same static analysis for the coder, so a host-only port is caught in the step that wrote it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import op_cheats, ttnn_calls
from orchard.supervisor import EXIT_BLOCKED

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "orchard" / "skills" / "full-port-templates"

HOST_ONLY = "import ttnn\nimport torch\n\ndef load(model_dir, mesh_device):\n    w = torch.zeros(4, 4)\n    ttnn.from_torch(w)\n    return {}\n"
WITH_LINEAR = HOST_ONLY.replace("    return {}", "    return {'w': ttnn.linear(w, w)}")


# ---- the message ---------------------------------------------------------------------------------------------------------------

def test_the_instruction_to_run_projections_on_the_device_comes_before_the_upload_form():
    text = ttnn_calls.WHAT_TO_DO
    assert text.startswith("Run each projection")
    assert text.index("ttnn.linear") < text.index("ttnn.from_torch")
    # every fact the message carried before the reorder is still there
    for fact in ("ttnn.matmul", "ttnn.as_tensor", "layout=ttnn.TILE_LAYOUT", "stages/2/cheat_sheet.md", "A host torch port with `import ttnn`",
                 "computes on the CPU", "An import, a comment, a string and code that is not reached make no calls"):
        assert fact in text, fact


def test_the_device_ops_reason_orders_the_instruction_before_the_upload_and_the_calls_found():
    reason = ttnn_calls.device_ops_reason({"ttnn.from_torch": 3, "ttnn.to_torch": 1})
    assert reason.startswith("device-ops: no source file under stages/2/port/ makes a reachable ttnn matmul-family call")
    assert reason.index("Run each projection") < reason.index("ttnn.from_torch(w") < reason.index("ttnn calls found:")
    assert reason.rstrip().endswith("ttnn.to_torch x1, ttnn.from_torch x3.") or "ttnn.from_torch x3" in reason.split("ttnn calls found:")[1]
    assert ttnn_calls.device_ops_reason({}).count("none") == 1                     # no call at all is named


def test_the_template_copy_of_the_shared_block_carries_the_message():
    template = (TEMPLATES / "port_check.py").read_text()
    assert "def device_ops_reason" in template and "WHAT_TO_DO" in template


# ---- port_check.py --static ----------------------------------------------------------------------------------------------------

def stage(tmp_path: Path, port_text: str | None, *, tools: bool = False) -> Path:
    """A run directory shaped like the real one: stages/2 holds the template (and the stage tools), port/ holds the coder's file."""
    d = tmp_path / "run" / "stages" / "2"
    (d / "port").mkdir(parents=True)
    shutil.copy(TEMPLATES / "port_check.py", d / "port_check.py")
    if tools:
        shutil.copy(TEMPLATES / "stage_tools.py", d / "stage_tools.py")
    if port_text is not None:
        (d / "port" / "port_model.py").write_text(port_text)
    return d


def run_static(d: Path, *flags: str, env_extra: dict | None = None):
    env = {"PATH": os.environ.get("PATH", ""), **(env_extra or {})}          # no TT_VISIBLE_DEVICES: this mode needs no chip
    return subprocess.run([sys.executable, "-P", "-s", str(d / "port_check.py"), "--static", *flags], capture_output=True, text=True,
                          env=env, cwd=str(d.parent.parent), timeout=120)


def test_static_mode_refuses_a_host_only_port_with_the_gates_own_message(tmp_path):
    d = stage(tmp_path, HOST_ONLY)
    r = run_static(d)
    assert r.returncode == 2, r.stdout + r.stderr
    failed = [ln for ln in r.stdout.splitlines() if ln.startswith("FAILED: ")]
    assert len(failed) == 1 and failed[0].startswith("FAILED: device-ops: no source file under stages/2/port/")
    assert failed[0].index("Run each projection") < failed[0].index("ttnn calls found:") and "ttnn.from_torch x1" in failed[0]
    assert 'PORT_CHECK_FAILURE {"kind": "port", "op": "device-ops"}' in r.stdout
    assert str(tmp_path) not in r.stdout                                          # a relative path only


def test_static_mode_passes_a_port_with_a_reachable_matmul_family_call(tmp_path):
    r = run_static(stage(tmp_path, WITH_LINEAR))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "static device-ops check passed" in r.stdout and "FAILED" not in r.stdout


def test_static_mode_reports_a_syntax_error_with_its_line(tmp_path):
    r = run_static(stage(tmp_path, "import ttnn\n\ndef load(:\n    pass\n"))
    assert r.returncode == 2
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("FAILED: "))
    assert "the port failed at 'load'" in line and "port_model.py:3" in line and "SyntaxError" in line
    assert 'PORT_CHECK_FAILURE {"kind": "port", "op": "load"}' in r.stdout


def test_static_mode_names_a_missing_port_module(tmp_path):
    r = run_static(stage(tmp_path, None))
    assert r.returncode == 2 and "port_model.py does not exist" in r.stdout


def test_static_mode_refuses_a_link_in_the_port_directory(tmp_path):
    d = stage(tmp_path, WITH_LINEAR)
    (d / "port" / "other.py").symlink_to(d / "port" / "port_model.py")
    r = run_static(d)
    assert r.returncode == 2 and "is a link" in r.stdout


def test_static_mode_does_not_touch_the_record_unless_asked(tmp_path):
    d = stage(tmp_path, WITH_LINEAR)
    (d / "evidence").mkdir()
    record = d / "evidence" / "port-check.json"
    record.write_text("{}")
    assert run_static(d).returncode == 0 and record.is_file()                      # the coder's gate-check must not erase what last-check reads
    assert run_static(d, "--fresh").returncode == 0 and not record.exists()         # the supervisor's pre-check clears a stale record as a real run does


def test_static_mode_scan_is_bounded_in_files(tmp_path):
    d = stage(tmp_path, HOST_ONLY)
    for i in range(ttnn_calls.STATIC_FILES_MAX + 20):
        (d / "port" / f"m{i:04d}.py").write_text("x = 1\n")
    r = run_static(d)
    assert r.returncode == 2
    line = next(ln for ln in r.stdout.splitlines() if ln.startswith("FAILED: "))
    assert "scan stopped" in line and str(ttnn_calls.STATIC_FILES_MAX) in line


def test_static_mode_scan_is_bounded_in_bytes(tmp_path):
    d = stage(tmp_path, HOST_ONLY)
    chunk = ("x = '" + "a" * 1000 + "'\n") * 1000                                  # about 1 MB each
    for i in range(ttnn_calls.STATIC_BYTES_MAX // len(chunk) + 3):
        (d / "port" / f"big{i}.py").write_text(chunk)
    r = run_static(d)
    assert r.returncode == 2 and "scan stopped" in r.stdout


# ---- the cheat sheet -----------------------------------------------------------------------------------------------------------

def test_the_cheat_sheet_starts_with_linear_and_matmul():
    sheet = op_cheats.load_sheet(op_cheats.TTNN_SHEET)
    titles = [s["title"] for s in sheet["sections"]]
    assert titles[:4] == ["Linear and matmul", "RMS norm and softmax", "Embedding", "Tensors, layout and dtype"], titles
    assert sorted(titles) == sorted(set(titles)) and len(titles) == 8             # nothing dropped or duplicated
    text = op_cheats.render(sheet)
    assert text.index("ttnn.linear") < text.index("ttnn.from_torch")              # the rendered file shows a projection before an upload
    assert len(text.splitlines()) < op_cheats.RENDER_LINE_BUDGET


# ---- the gate-check stage tool -------------------------------------------------------------------------------------------------

def run_tool(d: Path, *argv: str):
    return subprocess.run([sys.executable, "-P", "-s", str(d / "stage_tools.py"), *argv], capture_output=True, text=True,
                          env={"PATH": os.environ.get("PATH", "")}, cwd=str(d.parent.parent), timeout=120)


def test_gate_check_prints_the_gates_message_for_a_host_only_port(tmp_path):
    d = stage(tmp_path, HOST_ONLY, tools=True)
    r = run_tool(d, "gate-check")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "device-ops: no source file under stages/2/port/" in r.stdout and "Run each projection" in r.stdout
    assert "PORT_CHECK_FAILURE" not in r.stdout                                    # the tool prints advice, not the supervisor's line


def test_gate_check_passes_a_port_with_a_projection_and_runs_no_chip(tmp_path):
    d = stage(tmp_path, WITH_LINEAR, tools=True)
    r = run_tool(d, "gate-check")
    assert r.returncode == 0 and "static device-ops check passed" in r.stdout
    assert "no board" in r.stdout.lower()                                          # it says it did not run the check on the chip


def test_gate_check_is_listed_by_the_tool_and_by_the_task_text():
    assert "gate-check" in (TEMPLATES / "stage_tools.py").read_text().split("COMMANDS = {", 1)[1].split("}", 1)[0]
    from orchard import port_facts
    text = port_facts.tools_text(2)
    assert "stage_tools.py gate-check" in text and "before you reply" in text


# ---- the supervisor: no board for a host-only port -----------------------------------------------------------------------------

@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_host_only_port_is_refused_before_a_board_is_leased_or_the_coder_parked(tmp_path):
    from test_supervisor_full_port import HONEST, coder_writes, failure, stage_2, with_snapshot            # noqa: F401
    from test_supervisor import Rig
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        host_only = HONEST.replace("ttnn.linear(None, None)", "pass")
        coder_writes(r, {}, port=host_only)
        assert r.run() == EXIT_BLOCKED
        decisions = [e["data"] for e in r.entries() if e["event"] == "decision" and e["stage"] == 2]
        names = [d.get("decision") for d in decisions]
        assert "device-ops pre-check failed before the board" in names
        assert not {"hardware phase", "hardware test started", "test lease taken", "park"} & set(names)       # no lease, no park, no test
        note = next(d for d in decisions if d.get("decision") == "device-ops pre-check failed before the board")
        assert note["message"].startswith("device-ops: no source file") and str(tmp_path) not in json.dumps(note)
        # the failure reaches the repair path in the shape a failed board test leaves
        marker = json.loads((stage_2(r) / "test-result.json").read_text())
        assert marker["returncode"] == 2 and marker["timed_out"] is False
        out = (stage_2(r) / "evidence" / "hw-test-output.txt").read_text()
        assert "FAILED: device-ops:" in out and 'PORT_CHECK_FAILURE {"kind": "port", "op": "device-ops"}' in out
        assert "exited with code 2" in failure(r)["reason"]
        assert not (stage_2(r) / "evidence" / "port-check.json").exists()
    finally:
        r.close()


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_port_with_a_projection_still_goes_to_the_board(tmp_path):
    from test_supervisor_full_port import HONEST, coder_writes, stage_2, with_snapshot
    from test_supervisor import Rig
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        coder_writes(r, {}, port=HONEST)
        r.run()
        names = [e["data"].get("decision") for e in r.entries() if e["event"] == "decision" and e["stage"] == 2]
        assert "hardware test started" in names and "device-ops pre-check failed before the board" not in names
        assert json.loads((stage_2(r) / "test-result.json").read_text())["returncode"] == 0
    finally:
        r.close()


def test_the_pre_check_runs_only_for_the_stage_2_port_check_command(tmp_path):
    from test_supervisor_full_port import with_snapshot
    from test_supervisor import Rig
    from orchard.ledger import Ledger
    from orchard.stages import FULL_PORT_STAGE_2, FULL_PORT_STAGE_3
    from orchard.supervisor import build
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        stage_dir = r.run_dir / "stages" / "2"
        (stage_dir / "port").mkdir(parents=True)
        (stage_dir / "port" / "port_model.py").write_text(HOST_ONLY)          # a port the pre-check would refuse
        spec2, spec3 = FULL_PORT_STAGE_2, FULL_PORT_STAGE_3
        assert spec2.draft == "full_port" and spec2.number == 2          # the spec the pre-check is for: only the command (or the stage) keeps it out
        with Ledger(r.run_dir / "ledger.jsonl") as led:
            sup = build(r.args, led, adapter=r.adapter_cls(r.m, owner_pid=100), coder=r.coder_cls(r.m), home=r.home, containers=r.containers)
            other = {"command": "python3 -P -s stages/2/graft_trial.py", "deadline_s": 60}          # another test of the stage (the graft trial)
            assert sup._device_ops_precheck(spec2, stage_dir, other) is False
            assert sup._device_ops_precheck(spec3, stage_dir, {"command": "python3 -P -s stages/3/port_check.py", "deadline_s": 60}) is False
            assert not (stage_dir / "evidence").exists() and not (stage_dir / "test-result.json").exists()
    finally:
        r.close()
