# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A run started with stock parameters takes the stock-runtime path through the real supervisor and the fake machine: stage 0 is
supervisor code, stage 1 is the normal agent stage, stage 2 serves through tt-transformers on one board (a scripted test writes the
template's evidence), stages 3 to 7 are skipped with their reasons, and stage 8 builds the bundle. No chip, no network."""
import json
import os
import sys
from pathlib import Path

import pytest

from fake_model import FakeModel
from orchard.ledger import Ledger
from orchard.stages import SHIPPED_TEMPLATES
from orchard.supervisor import EXIT_BLOCKED, EXIT_READY, build, parse
from run_fakes import (FILES, TEMPLATE_SHA, TEMPLATE_TEXT, FakeContainers, Machine, MachineAdapter, MachineCoder, argv, bringup,
                       clock, done, note, plenty, write_tiers)
from test_stock_runtime_gates import CHECK
from test_stock_runtime_triage import LLAMA, params, snapshot, tensors_for

pytestmark = pytest.mark.usefixtures("stub_tools")

CHECK_RUN = {**CHECK, "model_id": "Altworld/Hemmingway-1", "template_sha256": TEMPLATE_SHA["stock_check.py"]}
STOCK_FILES = {
    (2, "prepare"): {"evidence/notes.txt": "plan", "stock_check.py": TEMPLATE_TEXT["stock_check.py"],
                     "fake_stock_run.py": ("import json, pathlib\nhere = pathlib.Path(__file__).resolve().parent\n"
                                           "(here / 'evidence').mkdir(exist_ok=True)\n"
                                           f"(here / 'evidence' / 'stock-check.json').write_text({json.dumps(json.dumps(CHECK_RUN))})\n"
                                           "(here / 'evidence' / 'scratch.patch').write_text('')\n"),
                     "hw_test.json": {"command": "python3 stages/2/fake_stock_run.py", "deadline_s": 60},
                     "handoff.json": note(2)},
    (2, "finish"): {"evidence/finish-notes.txt": "the evidence is the template's"},
}


class Rig:
    def __init__(self, tmp, *, cfg_edit=None, files=None, chips=2):
        self.tmp, self.m = tmp, Machine()
        self.run_dir = tmp / "run"
        cfg = LLAMA if cfg_edit is None else {**LLAMA, **cfg_edit}
        snap = snapshot(tmp, cfg=cfg, tensors=tensors_for(LLAMA))
        self.params = params(tmp)
        self.files = {**FILES, **STOCK_FILES, **(files or {})}
        self.chip_server = FakeModel(lambda r: bringup(r, overrides=self.files), models=["qwen-27b"])
        self.cpu_server = FakeModel(bringup, models=["cpu-model"])
        self.chip_server.__enter__()
        self.cpu_server.__enter__()
        tiers = write_tiers(tmp / "tiers.toml", self.chip_server.endpoint, self.cpu_server.endpoint)
        self.argv = argv(tmp, tiers, self.chip_server.endpoint, chips=chips) + [
            "--input", f"model={snap}", "--input", f"stock_params={self.params}", "--unattended"]
        self.clock = clock()

    def run(self, pid=100):
        path = self.run_dir / "ledger.jsonl"
        self.m.owner_pid = pid
        with Ledger(path) as led:
            self.led = led
            return build(parse(self.argv), led, adapter=MachineAdapter(self.m, owner_pid=pid), coder=MachineCoder(self.m),
                         versions={"tt_model": "test"}, clock=self.clock, sleep=self.clock.sleep, disk_usage=plenty,
                         home=self.tmp / "operator-home", containers=FakeContainers()).run()

    def entries(self):
        with Ledger(self.run_dir / "ledger.jsonl") as led:
            return led.read()

    def close(self):
        self.chip_server.__exit__()
        self.cpu_server.__exit__()


@pytest.fixture
def rig(tmp_path):
    made = []

    def make(**kw):
        r = Rig(tmp_path, **kw)
        made.append(r)
        return r
    yield make
    for r in made:
        r.close()


def ends(entries):
    return {e["stage"]: e["data"] for e in entries if e["event"] == "stage_end"}


def test_a_stock_run_goes_from_triage_to_the_bundle_with_the_stock_stages(rig):
    r = rig()
    assert r.run() == EXIT_READY
    es = r.entries()
    e = ends(es)
    assert e[0]["result"] == "pass" and e[0]["path"] == "stock-runtime" and e[0]["class"] == "stock-runtime"
    assert e[1]["result"] == "pass" and e[2]["result"] == "pass"
    for n in (3, 4, 5, 6, 7):
        assert e[n]["result"] == "skipped" and e[n]["reason"].startswith("stock-runtime path")
    assert e[8]["result"] == "pass"
    start = next(x for x in es if x["event"] == "run_start")
    assert start["data"]["inputs"]["stock_params"] == str(r.params)


def test_stage_zero_ran_no_agent_and_wrote_the_triage_files(rig):
    r = rig()
    r.run()
    es = r.entries()
    assert not [x for x in es if x["stage"] == 0 and x["event"] in ("agent_turn", "agent_step")]
    d = json.loads((r.run_dir / "stages/0/delta.json").read_text())
    assert d["class"] == "stock-runtime" and d["stock_runtime"]["name"] == "Llama-Test-1B-Instruct"
    assert (r.run_dir / "stages/0/evidence/stock-layout.json").is_file()
    assert any(x["event"] == "evidence" and x["stage"] == 0 and x["data"].get("what") == "stock triage" for x in es)


def test_stage_two_names_the_stock_skill_and_runs_one_hardware_test_under_a_lease(rig):
    r = rig()
    r.run()
    es = r.entries()
    started = [x for x in es if x["stage"] == 2 and x["data"].get("decision") == "hardware test started"]
    assert len(started) == 1
    assert (r.run_dir / "stages/2/test-result.json").is_file()
    assert not r.m.leases and not r.m.coder_running


def test_a_config_that_differs_stops_the_run_as_a_full_port_before_stage_two(rig):
    r = rig(cfg_edit={"num_hidden_layers": 3})
    assert r.run() == EXIT_BLOCKED
    e = ends(r.entries())
    assert e[0]["result"] == "pass" and e[0]["path"] == "full-port" and e[0]["class"] == "full-port"
    assert 2 not in e and not any(n in e for n in (3, 4, 5, 6, 7, 8))
    assert "path_reasons" in json.loads((r.run_dir / "stages/0/delta.json").read_text())


def test_a_stage_two_that_misses_the_bar_fails_and_does_not_pass(rig):
    bad = {**CHECK_RUN, "teacher_forced": {"tokens": 511, "top1": 0.5, "top5": 0.9}}
    files = {(2, "prepare"): {**STOCK_FILES[(2, "prepare")], "fake_stock_run.py":
             STOCK_FILES[(2, "prepare")]["fake_stock_run.py"].replace(json.dumps(json.dumps(CHECK_RUN)), json.dumps(json.dumps(bad)))}}
    r = rig(files=files)
    code = r.run()
    assert code == EXIT_BLOCKED
    assert ends(r.entries())[2]["result"] != "pass"
    assert 8 not in ends(r.entries())


def test_an_agent_in_stage_two_can_write_into_the_scratch_directory_and_nowhere_else_outside_its_stage(rig):
    files = {(2, "prepare"): {**STOCK_FILES[(2, "prepare")], "tt-metal-scratch/note.txt": "patched"}}
    r = rig(files=files)
    r.run()
    assert (r.run_dir / "tt-metal-scratch" / "note.txt").read_text() == "patched"
    assert not (r.run_dir / "stages/2/tt-metal-scratch").exists()


def test_an_agent_in_stage_one_has_no_scratch_directory(rig):
    files = {(1, "run"): {**FILES[(1, "run")], "tt-metal-scratch/note.txt": "patched"}}
    r = rig(files=files)
    r.run()
    assert not (r.run_dir / "tt-metal-scratch" / "note.txt").exists()
    assert (r.run_dir / "stages/1/tt-metal-scratch/note.txt").is_file()        # a plain stage-relative path


def _supervisor(r):
    path = r.run_dir / "ledger.jsonl"
    led = Ledger(path)
    led.__enter__()
    r.m.owner_pid = 100
    sup = build(parse(r.argv), led, adapter=MachineAdapter(r.m, owner_pid=100), coder=MachineCoder(r.m), versions={"tt_model": "test"},
                clock=r.clock, sleep=r.clock.sleep, disk_usage=plenty, home=r.tmp / "operator-home", containers=FakeContainers())
    return sup, led


def _stage_with_delta(r):
    from orchard.stages import spec_for
    (r.run_dir / "stages/0").mkdir(parents=True, exist_ok=True)
    from test_stock_runtime_gates import DELTA
    (r.run_dir / "stages/0/delta.json").write_text(json.dumps({**DELTA, "stock_runtime": {**DELTA["stock_runtime"], "name": "Llama-Test-1B-Instruct"}}))
    stage = r.run_dir / "stages/2"
    stage.mkdir(parents=True, exist_ok=True)
    return spec_for(2, "stock-runtime"), stage


def test_the_draft_copies_the_template_itself(rig):
    r = rig()
    sup, led = _supervisor(r)
    try:
        spec, stage = _stage_with_delta(r)
        sup._draft_stock(spec, stage)
        assert (stage / "stock_check.py").read_text() == TEMPLATE_TEXT["stock_check.py"]
        assert (stage / "stock_config.json").is_file()
    finally:
        led.__exit__(None, None, None)


def test_a_config_already_in_the_stage_is_kept_on_a_resume(rig):
    r = rig()
    sup, led = _supervisor(r)
    try:
        spec, stage = _stage_with_delta(r)
        (stage / "stock_config.json").write_text('{"kept": true}')
        sup._draft_stock(spec, stage)
        assert json.loads((stage / "stock_config.json").read_text()) == {"kept": True}
        assert not (stage / "stock_check.py").exists()
    finally:
        led.__exit__(None, None, None)


def test_a_draft_with_a_missing_fact_writes_nothing_and_says_why(rig):
    r = rig()
    sup, led = _supervisor(r)
    try:
        spec, stage = _stage_with_delta(r)
        (r.run_dir / "stages/0/delta.json").write_text("{}")
        sup._draft_stock(spec, stage)
        assert not (stage / "stock_config.json").exists()
        assert any(e["data"].get("decision") == "stock config not drafted" for e in led.read())
    finally:
        led.__exit__(None, None, None)


def test_the_supervisor_drafted_the_stock_config_and_copied_the_template(rig):
    r = rig()
    r.run()
    cfg = json.loads((r.run_dir / "stages/2/stock_config.json").read_text())
    assert cfg["mesh_device"] == "P300" and cfg["model_name"] == "Llama-Test-1B-Instruct"
    assert cfg["scratch_dir"] == str(r.run_dir / "tt-metal-scratch")
    assert (r.run_dir / "stages/2/stock_check.py").read_text() == TEMPLATE_TEXT["stock_check.py"]
    assert any(x["data"].get("decision") == "stock config drafted" for x in r.entries() if x["stage"] == 2)
