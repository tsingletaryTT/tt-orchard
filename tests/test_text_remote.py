# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The hop from the dependency-free supervisor to the interpreter that has numpy (orchard/text_remote.py, orchard/text_cli.py), the stdlib-only import of the
text route, and the chip orchestrator template."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import text_cli, text_remote, text_runtime as rt

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ENV = {"PATH": os.environ.get("PATH", "")}


def test_the_stage_table_and_the_adapter_import_without_numpy():
    code = ("import sys; sys.modules['numpy'] = None; import orchard.stages, orchard.supervisor, orchard.text_runtime, orchard.text_remote; "
            "assert 'orchard.text_stages' not in sys.modules; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-400:]


def test_a_text_gate_with_no_selection_fails_without_importing_the_numeric_module():
    code = ("import sys; import orchard.stages as s; res = s.gate_text_reference('/nonexistent/stage', '/nonexistent/run'); assert not res.ok; "
            "assert 'ledger holds no text runtime selection' in res.reasons[0]; assert 'orchard.text_stages' not in sys.modules and 'numpy' not in sys.modules; print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-400:]


def test_a_gate_runs_in_another_interpreter_and_gives_the_same_answer_as_in_process(tmp_path):
    args = {"stage_dir": str(tmp_path / "s"), "run_dir": str(tmp_path)}
    hop = text_remote.call(PY, "gate_reference", args, timeout_s=120, env=ENV)
    assert hop == text_cli.dispatch("gate_reference", args)["result"] and hop and "text_config.json" in hop[0]


def test_an_unknown_verb_and_a_malformed_request_are_errors_not_passes():
    assert "unknown verb" in text_cli.dispatch("nope", {})["error"]
    assert "bad request" in text_cli.dispatch("gate_chip", {})["error"]
    assert "no text runtime" in text_cli.dispatch("references", {"runtime": "nope"})["error"]
    with pytest.raises(rt.TextStageError, match="unknown verb"):
        text_remote.call(PY, "nope", {}, timeout_s=60, env=ENV)


def test_a_verb_that_dies_without_an_answer_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(text_remote, "CLI", tmp_path / "dead.py")
    (tmp_path / "dead.py").write_text("import sys; sys.stderr.write('boom'); sys.exit(3)\n")
    with pytest.raises(rt.TextStageError, match="no answer"):
        text_remote.call(PY, "gate_reference", {}, timeout_s=60, env=ENV)


def test_a_verb_past_its_timeout_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(text_remote, "CLI", tmp_path / "slow.py")
    (tmp_path / "slow.py").write_text("import time; time.sleep(60)\n")
    with pytest.raises(rt.TextStageError, match="ran past"):
        text_remote.call(PY, "gate_reference", {}, timeout_s=1, env=ENV)


def test_the_interpreter_is_started_isolated_so_a_module_in_the_environment_cannot_shadow_numpy(tmp_path):
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "numpy.py").write_text("raise RuntimeError('shadowed numpy')\n")
    out = text_remote.call(PY, "gate_reference", {"stage_dir": str(tmp_path / "s"), "run_dir": str(tmp_path)}, timeout_s=120, env={**ENV, "PYTHONPATH": str(poison)})
    assert out and "text_config.json" in out[0]


# ---- the chip orchestrator ------------------------------------------------------------------------------------------------------

def run_orchestrator(stage, commands):
    cfg = stage / rt.CHIP_CONFIG_FILE
    cfg.write_text(json.dumps({"commands": commands}))
    p = subprocess.run([PY, str(rt.CHIP_TEST), "--config", str(cfg)], capture_output=True, text=True)
    return p, json.loads((stage / "text_chip_run.json").read_text())


def cmd(name, code, outputs=()):
    return {"name": name, "argv": [PY, "-c", code], "outputs": list(outputs)}


def test_the_orchestrator_runs_the_commands_in_order_and_records_each(tmp_path):
    p, rec = run_orchestrator(tmp_path, [cmd("a", "open('a.txt','w').write('1')", ["a.txt"]), cmd("b", "open('b.txt','w').write('1')", ["b.txt"])])
    assert p.returncode == 0 and rec["ok"] and [c["name"] for c in rec["commands"]] == ["a", "b"] and (tmp_path / "log" / "a.log").is_file()


def test_the_orchestrator_stops_at_the_first_failed_command(tmp_path):
    p, rec = run_orchestrator(tmp_path, [cmd("a", "import sys; sys.exit(4)"), cmd("b", "open('b.txt','w').write('1')", ["b.txt"])])
    assert p.returncode == 1 and not rec["ok"] and [c["name"] for c in rec["commands"]] == ["a"] and rec["commands"][0]["exit"] == 4
    assert not (tmp_path / "b.txt").exists()


def test_the_orchestrator_fails_a_command_that_left_out_a_promised_file(tmp_path):
    p, rec = run_orchestrator(tmp_path, [cmd("a", "pass", ["never.json"])])
    assert p.returncode == 1 and rec["commands"][0]["missing_outputs"] == ["never.json"]


def test_the_lease_name_can_be_set_by_the_environment_and_defaults_to_the_supervisors():
    def who(env):
        r = subprocess.run([sys.executable, "-c", "import orchard.supervisor as s; print(s.WHO)"], cwd=ROOT, capture_output=True, text=True,
                           env={**os.environ, **env})
        return r.stdout.strip()
    assert who({"ORCHARD_LEASE_WHO": "claude:example"}) == "claude:example"
    assert who({"ORCHARD_LEASE_WHO": ""}) == "orchard:supervisor"
