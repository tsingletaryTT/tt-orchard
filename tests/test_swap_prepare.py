# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2's prepare step run by the supervisor (orchard/swap_prepare.py).

When the supervisor has drafted a swap_config.json, it runs prepare_swap.py itself and writes hw_test.json and handoff.json from the
result. The agent is called only when the script exits nonzero, and then it is given the script's message. The prepare phase also
may not leave a result file behind: result.json comes from the finish step, which reads the real test output."""
import json
import os
from pathlib import Path

import pytest

from fake_model import call, final, turn
from orchard import swap_prepare
from orchard.handoff import NOTE_KEYS
from orchard.supervisor import EXIT_READY, Supervisor
from run_fakes import FILES, bringup, where
from test_supervisor import Rig

pytestmark = pytest.mark.usefixtures("stub_tools")


# ---- the module -------------------------------------------------------------------------------------------------------------

def stage(tmp_path, *, script: str | None, config: dict | None = None):
    """A run directory with stages/2 holding a swap_config.json and a stand-in prepare_swap.py (a few lines of Python)."""
    run = tmp_path / "run"
    sd = run / "stages" / "2"
    (sd / "evidence").mkdir(parents=True)
    if config is not None:
        (sd / "swap_config.json").write_text(json.dumps(config))
    if script is not None:
        (sd / "prepare_swap.py").write_text(script)
    return run, sd


BUNDLE_CONFIG = {"bundle_dir": "/b", "nearest_model_id": "Qwen/Qwen3.8-27B", "base_snapshot": "/x", "new_snapshot": "/y"}
BUILDS = ("import pathlib\nh = pathlib.Path(__file__).resolve().parent\n(h / 'model-dir').mkdir()\n(h / 'run.sh').write_text('#!/bin/bash')\n"
          "print('model-dir built')\n")


def prepare(run, sd):
    return swap_prepare.prepare(run_dir=run, stage_dir=sd, env=dict(os.environ))


def test_a_script_that_builds_its_files_leaves_a_test_and_a_handoff_and_no_agent_note(tmp_path):
    run, sd = stage(tmp_path, script=BUILDS, config=BUNDLE_CONFIG)
    out = prepare(run, sd)
    assert out.done and out.note is None and out.returncode == 0
    assert json.loads((sd / "hw_test.json").read_text()) == {"command": "python3 stages/2/serve_and_compare.py", "deadline_s": 3600}
    note = json.loads((sd / "handoff.json").read_text())
    assert all(note.get(k) not in (None, "") for k in NOTE_KEYS) and note["stage"] == 2
    assert "model-dir built" in (sd / "evidence" / "prepare-swap-output.txt").read_text()


def test_a_container_config_runs_the_container_script(tmp_path):
    run, sd = stage(tmp_path, config={**BUNDLE_CONFIG, "bundle_dir": None, "package": "p/q"},
                    script="import pathlib\n(pathlib.Path(__file__).resolve().parent / 'model-dir').mkdir()\n")
    out = prepare(run, sd)           # a container package builds no run.sh
    assert out.done
    assert json.loads((sd / "hw_test.json").read_text())["command"] == "python3 stages/2/serve_and_compare_container.py"


def test_exit_2_hands_the_scripts_message_to_the_agent_and_writes_no_test(tmp_path):
    run, sd = stage(tmp_path, config=BUNDLE_CONFIG, script="import sys\nprint('prepare_swap: port 8100 is in use', file=sys.stderr)\nsys.exit(2)\n")
    out = prepare(run, sd)
    assert not out.done and out.returncode == 2
    assert "port 8100 is in use" in out.note and "swap_config.json" in out.note
    assert not (sd / "hw_test.json").exists() and not (sd / "handoff.json").exists()


def test_another_nonzero_exit_goes_to_the_agent_with_the_exit_code(tmp_path):
    run, sd = stage(tmp_path, config=BUNDLE_CONFIG, script="raise RuntimeError('boom')\n")
    out = prepare(run, sd)
    assert not out.done and out.returncode == 1 and "exited 1" in out.note and "boom" in out.note
    assert not (sd / "hw_test.json").exists()


def test_a_script_that_exits_0_without_building_the_model_directory_goes_to_the_agent(tmp_path):
    run, sd = stage(tmp_path, config=BUNDLE_CONFIG, script="print('nothing built')\n")
    out = prepare(run, sd)
    assert not out.done and "model-dir" in out.note
    assert not (sd / "hw_test.json").exists()


def test_a_bundle_config_needs_a_run_script_too(tmp_path):
    run, sd = stage(tmp_path, config=BUNDLE_CONFIG,
                    script="import pathlib\n(pathlib.Path(__file__).resolve().parent / 'model-dir').mkdir()\n")
    out = prepare(run, sd)
    assert not out.done and "run.sh" in out.note


def test_a_script_that_runs_too_long_goes_to_the_agent(tmp_path, monkeypatch):
    monkeypatch.setattr(swap_prepare, "PREPARE_TIMEOUT_S", 1)
    run, sd = stage(tmp_path, config=BUNDLE_CONFIG, script="import time\ntime.sleep(30)\n")
    out = prepare(run, sd)
    assert not out.done and "did not finish" in out.note


@pytest.mark.parametrize("config,script", [(None, BUILDS), (BUNDLE_CONFIG, None)])
def test_without_a_config_or_a_script_nothing_runs_and_the_agent_works_as_before(tmp_path, config, script):
    run, sd = stage(tmp_path, config=config, script=script)
    out = prepare(run, sd)
    assert not out.done and out.note is None and out.returncode is None
    assert not (sd / "hw_test.json").exists()


def test_an_old_test_file_is_replaced_by_the_new_one(tmp_path):
    run, sd = stage(tmp_path, config=BUNDLE_CONFIG, script=BUILDS)
    (sd / "hw_test.json").write_text(json.dumps({"command": "rm -rf /", "deadline_s": 1}))
    assert prepare(run, sd).done
    assert json.loads((sd / "hw_test.json").read_text())["command"] == "python3 stages/2/serve_and_compare.py"


def test_results_written_before_the_hardware_test_are_moved_aside(tmp_path):
    run, sd = stage(tmp_path, script=None)
    for name in ("result.json", "test-result.json", "evidence/swap-check.json", "evidence/hw-test-output.txt"):
        (sd / name).write_text("{}")
    (sd / "hw_test.json").write_text("{}")
    moved = swap_prepare.set_aside_results(sd)
    assert sorted(moved) == ["evidence/hw-test-output.txt", "evidence/swap-check.json", "result.json", "test-result.json"]
    for name in moved:
        assert not (sd / name).exists()
        assert (sd / f"{name}{swap_prepare.REJECTED_SUFFIX}").is_file()
    assert (sd / "hw_test.json").is_file()
    assert swap_prepare.set_aside_results(sd) == []


# ---- the supervisor ---------------------------------------------------------------------------------------------------------

SERVE_STUB = """import json, pathlib
here = pathlib.Path(__file__).resolve().parent
(here / "evidence").mkdir(exist_ok=True)
(here / "evidence" / "devices.txt").write_text("ran")
(here / "evidence" / "swap-check.json").write_text(json.dumps({"result_draft": {"top1_agreement": 0.94}}))
"""


def drafting(monkeypatch, prepare_script: str):
    """Stand in for the drafting step: write a config and the stand-in scripts into stage 2's directory, as _draft does."""
    def draft(self, spec, stage_dir):
        if spec.number == 2 and spec.draft == "swap_config":
            (stage_dir / "swap_config.json").write_text(json.dumps(BUNDLE_CONFIG))
            (stage_dir / "prepare_swap.py").write_text(prepare_script)
            (stage_dir / "serve_and_compare.py").write_text(SERVE_STUB)
    monkeypatch.setattr(Supervisor, "_draft", draft)


class Counting:
    """A scripted model that records the phases it was asked for and the user message of each."""
    def __init__(self, override=None):
        self.asked, self.override = [], override or {}

    def __call__(self, request):
        if "tools" in request:
            key = where(request)
            if turn(request) == 0:
                self.asked.append((key, request["messages"][1]["content"]))
        return bringup(request, overrides=self.override)


def stage2_phases(counting):
    return [key for key, _ in counting.asked if key[0] == 2]


def test_the_supervisor_runs_prepare_itself_and_never_asks_the_agent_for_it(tmp_path, monkeypatch):
    drafting(monkeypatch, BUILDS)
    r = Rig(tmp_path)
    c = Counting()
    r.script = c
    try:
        assert r.run() == EXIT_READY
        assert (2, "prepare") not in stage2_phases(c) and (2, "finish") in stage2_phases(c)
        sd = r.run_dir / "stages" / "2"
        assert json.loads((sd / "hw_test.json").read_text())["command"] == "python3 stages/2/serve_and_compare.py"
        assert (sd / "evidence" / "devices.txt").read_text() == "ran"          # the hardware test ran what the supervisor wrote
        assert any(d.get("decision") == "prepare run by the supervisor" for d in r.decisions())
    finally:
        r.close()


def test_exit_2_asks_the_agent_with_the_scripts_message(tmp_path, monkeypatch):
    drafting(monkeypatch, "import sys\nprint('prepare_swap: tt_cache is not empty', file=sys.stderr)\nsys.exit(2)\n")
    r = Rig(tmp_path)
    c = Counting()
    r.script = c
    try:
        r.run()
        [(_, user)] = [a for a in c.asked if a[0] == (2, "prepare")][:1]
        assert "tt_cache is not empty" in user and "exited 2" in user
        assert any(d.get("decision") == "prepare run by the supervisor" and d.get("returncode") == 2 for d in r.decisions())
    finally:
        r.close()


def test_without_a_config_the_agent_prepares_as_before(tmp_path, monkeypatch):
    monkeypatch.setattr(Supervisor, "_draft", lambda self, spec, stage_dir: None)
    r = Rig(tmp_path)
    c = Counting()
    r.script = c
    try:
        assert r.run() == EXIT_READY
        assert (2, "prepare") in stage2_phases(c)
        user = next(u for k, u in c.asked if k == (2, "prepare"))
        assert "exited" not in user
        assert not any(d.get("decision") == "prepare run by the supervisor" for d in r.decisions())
    finally:
        r.close()


FAKE_RESULT = {"serves": True, "server_ready_s": 280.5, "coherent": True, "free_run_text": "first 200 characters of the chip's greedy text",
               "top1_agreement": 0.94, "n_tokens": 32, "cache_dir": "x", "evidence": ["stages/2/evidence/swap-check.json"]}


def test_a_result_the_prepare_agent_wrote_is_not_what_the_stage_ends_with(tmp_path, monkeypatch):
    """The 2026-11 run: the prepare agent wrote result.json with the skill's example numbers. A finish step that wrote nothing would
    have left it in place, and the gate reads that file. The finish step here ends without writing."""
    monkeypatch.setattr(Supervisor, "_draft", lambda self, spec, stage_dir: None)
    r = Rig(tmp_path)

    def script(request):
        if "tools" in request and where(request) == (2, "prepare"):
            files = {**FILES[(2, "prepare")], "result.json": FAKE_RESULT, "test-result.json": {"returncode": 0, "timed_out": False}}
            return bringup(request, overrides={(2, "prepare"): files})
        if "tools" in request and where(request) == (2, "finish"):
            return final("stage 2 finish done")                 # writes nothing
        return bringup(request)
    r.script = script
    r.args.unattended = True
    try:
        r.run()
        sd = r.run_dir / "stages" / "2"
        assert any(d.get("decision") == "results written before the test were set aside" for d in r.decisions())
        done = r.ends(2)
        assert done and done[0]["result"] != "pass"
        assert not (sd / "result.json").exists() or json.loads((sd / "result.json").read_text()) != FAKE_RESULT
        partial = sorted((r.run_dir / "stages").glob("2*"))
        assert any((p / f"result.json{swap_prepare.REJECTED_SUFFIX}").is_file() for p in partial)
    finally:
        r.close()
