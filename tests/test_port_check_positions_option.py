# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`--port-check-positions` (issue 731): the supervisor option that sets the stage 2 check's position cap. The wiring is tested end to end: the
supervisor writes the cap into `port_check_config.json`, the check template (which runs for real on the CPU with a fake `ttnn`) reads it and
records it, the gate recomputes it, the ledger's run_start records it only when it is not the default, and a resume keeps what the run
recorded. The chips, the coder and the model are fakes."""
import json

import pytest

from orchard import full_port
from orchard.supervisor import Control, parse
from run_fakes import argv
from test_supervisor import Rig
from test_supervisor_full_port import coder_writes, stage_2, with_snapshot

pytestmark = pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    yield r
    r.close()


def run_to_the_end_of_stage_2(rig, **kw):
    """A full-port stage 2 whose check passes; the run is aborted at its first wait, which is after stage 2 ended."""
    coder_writes(rig, {}, unattended=False)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run(**kw)


def record(rig):
    return json.loads((stage_2(rig) / "evidence" / "port-check.json").read_text())


def run_start(rig):
    return next(e for e in rig.entries() if e["event"] == "run_start")["data"]


def config(rig):
    return json.loads((stage_2(rig) / full_port.CONFIG_NAME).read_text())


# ---- the option ---------------------------------------------------------------------------------------------------------------

def test_parse_has_the_option_default_none_and_checks_the_range(tmp_path):
    a = argv(tmp_path, tmp_path / "t.toml", "http://127.0.0.1:8000/v1")
    assert parse(a).port_check_positions is None
    assert parse(a + ["--port-check-positions", "16"]).port_check_positions == 16
    assert parse(a + ["--port-check-positions", "8192"]).port_check_positions == 8192
    for bad in ("15", "8193", "-1", "abc", "6.5", ""):
        with pytest.raises(SystemExit):
            parse(a + ["--port-check-positions", bad])


# ---- the wiring -------------------------------------------------------------------------------------------------------------

def test_a_run_with_no_option_is_what_it_was(rig):
    run_to_the_end_of_stage_2(rig)
    assert [d["result"] for d in rig.ends(2)] == ["pass"]
    assert record(rig)["positions"] == {"cap": 64, "scored": 32, "reference_length": 32}
    assert "port_check_positions" not in run_start(rig) and "max_positions" not in config(rig)       # nothing new in a default run's files


def test_the_cap_reaches_the_template_the_record_the_gate_and_the_ledger(rig):
    rig.args.port_check_positions = 16
    run_to_the_end_of_stage_2(rig)
    assert config(rig)["max_positions"] == 16
    assert record(rig)["positions"] == {"cap": 16, "scored": 16, "reference_length": 32} and len(record(rig)["chip_ids"]) == 16
    assert [d["result"] for d in rig.ends(2)] == ["pass"]                     # the gate recomputed the block and accepted it
    assert run_start(rig)["port_check_positions"] == 16


def test_naming_the_default_changes_nothing_in_the_files(rig):
    rig.args.port_check_positions = full_port.DEFAULT_POSITIONS
    run_to_the_end_of_stage_2(rig)
    assert "port_check_positions" not in run_start(rig) and "max_positions" not in config(rig)
    assert record(rig)["positions"]["cap"] == 64


def test_a_coder_that_edits_the_config_during_prepare_gets_the_supervisors_cap_back(rig):
    rig.args.port_check_positions = 16
    coder_writes(rig, {full_port.CONFIG_NAME: json.dumps({"model_snapshot": "x", "max_positions": 8000})}, unattended=False)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert config(rig)["max_positions"] == 16 and record(rig)["positions"]["cap"] == 16


# ---- resume -----------------------------------------------------------------------------------------------------------------

@pytest.fixture
def built(monkeypatch):
    """The supervisors `Rig.run` builds, in order, so a test can read what a resumed supervisor believes (the files of a finished stage are not rewritten)."""
    import test_supervisor
    from orchard.supervisor import build
    made = []
    monkeypatch.setattr(test_supervisor, "build", lambda *a, **k: made.append(build(*a, **k)) or made[-1])
    return made


def test_a_resume_with_another_cap_is_refused(rig, built):
    rig.args.port_check_positions = 16
    run_to_the_end_of_stage_2(rig)
    rig.args.port_check_positions = 32
    with pytest.raises(ValueError, match="port check positions"):
        rig.run(pid=200)


def test_a_resume_without_the_option_keeps_the_recorded_cap(rig, built):
    rig.args.port_check_positions = 16
    run_to_the_end_of_stage_2(rig)
    assert built[-1].port_check_positions == 16
    rig.args.port_check_positions = None
    rig.run(pid=200)                                          # no option: nothing is refused
    assert built[-1].port_check_positions == 16 and run_start(rig)["port_check_positions"] == 16


def test_a_resume_that_names_the_recorded_cap_is_accepted(rig, built):
    rig.args.port_check_positions = 16
    run_to_the_end_of_stage_2(rig)
    rig.args.port_check_positions = 16
    rig.run(pid=200)
    assert built[-1].port_check_positions == 16


def test_a_run_that_started_with_the_default_refuses_a_resume_that_names_another_cap(rig, built):
    run_to_the_end_of_stage_2(rig)
    assert built[-1].port_check_positions == 64
    rig.args.port_check_positions = 16
    with pytest.raises(ValueError, match="port check positions"):
        rig.run(pid=200)
    rig.args.port_check_positions = 64                         # the same as the run's own default: accepted
    rig.run(pid=200)
    assert built[-1].port_check_positions == 64


def test_a_resume_without_the_option_of_a_run_that_started_with_the_default_keeps_the_default(rig, built):
    run_to_the_end_of_stage_2(rig)
    rig.args.port_check_positions = None
    rig.run(pid=200)
    assert built[-1].port_check_positions == 64 and "port_check_positions" not in run_start(rig)
