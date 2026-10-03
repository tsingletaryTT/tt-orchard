"""The supervisor's parts: control file, actuator, external stand-in, coder tier, versions, and the
reading of a hardware test's record."""
import json

import pytest

from fake_model import FakeModel, final
from fakes import FakeRun
from orchard.server import ServerSpec
from orchard.supervisor import (Control, ExternalStandIn, RunActuator, coder_tier, hardware_test_failure,
                                resolve_versions)
from orchard.tiers import load
from run_fakes import write_tiers


def test_a_control_command_acts_once_and_is_kept(tmp_path):
    c = Control(tmp_path)
    assert c.take() is None
    c.write("resume")
    assert c.take() == "resume" and c.take() is None
    assert (tmp_path / "control.done-1").read_text() == "resume\n"
    (tmp_path / "control").write_text("explode\n")
    assert c.take() is None and (tmp_path / "control.done-2").exists()
    with pytest.raises(ValueError):
        c.write("explode")


def test_the_actuator_queues_nudges_and_reads_operator_commands(tmp_path):
    a = RunActuator(Control(tmp_path))
    a.nudge("stage-agent", "do something else")
    assert a.take_nudges("stage-agent") == ["do something else"] and a.take_nudges("stage-agent") == []
    assert a.stop_reason() is None
    Control(tmp_path).write("pause")
    assert a.stop_reason() == "operator-pause"
    a.clear()
    a.escalate("stage-agent", 2)
    Control(tmp_path).write("abort")
    assert a.stop_reason() == "abort"          # abort wins over everything


def test_the_external_stand_in_asks_the_cpu_tier_and_never_stops_it():
    with FakeModel(lambda r: final("4"), models=["cpu-model"]) as fm:
        s = ExternalStandIn(fm.endpoint, "cpu-model")
        s.spawn()
        s.wait_ready()
        assert s.ask("What is 2 + 2?") == "4"
        s.stop()
        assert s.confirm_stopped().stopped and s.record()["kind"] == "external"
    assert fm.requests[0]["model"] == "cpu-model" and "tools" not in fm.requests[0]


def test_the_coder_tier_is_the_chip_tier_on_the_coder_port(tmp_path):
    cfg = load(write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1"))
    assert coder_tier(cfg, 8000) == "large"
    with pytest.raises(ValueError, match="exactly one chip tier"):
        coder_tier(cfg, 11434)                  # the CPU tier is never the coder


def test_versions_record_tt_model_and_firmware_and_mark_the_rest_todo():
    devices = {"device_info": [{"firmwares": {"fw_bundle_version": "19.15.0.0"}}] * 4}
    run = FakeRun({"tt-model": [(0, "0.1.0\n", "")], "tt-smi": [(0, devices, "")]}, key=lambda a: a[0])
    v = resolve_versions(ServerSpec("org/coder", "container", 8000, "m"), run=run)
    assert v["tt_model"] == "0.1.0" and v["firmware"] == ["19.15.0.0"]
    assert v["tt_metal"].startswith("TODO") and v["vllm"].startswith("TODO")


def test_the_stage_watchdog_runs_the_writeless_turn_detector(tmp_path):
    from types import SimpleNamespace

    from orchard.ledger import Ledger
    from orchard.stages import spec_for
    from orchard.supervisor import Supervisor
    from orchard.watchdog import NoFileWritten
    with Ledger(tmp_path / "ledger.jsonl") as led:
        fake = SimpleNamespace(actuator=None, ledger=led, clock=lambda: 0.0)
        wd = Supervisor._watchdog(fake, spec_for(2, "weights-only"))
    assert any(isinstance(d, NoFileWritten) for d in wd.detectors)


def test_the_stage_watchdog_runs_the_turn_repeat_detector(tmp_path):
    from types import SimpleNamespace

    from orchard.ledger import Ledger
    from orchard.stages import spec_for
    from orchard.supervisor import Supervisor
    from orchard.watchdog import TurnRepeat
    with Ledger(tmp_path / "ledger.jsonl") as led:
        fake = SimpleNamespace(actuator=None, ledger=led, clock=lambda: 0.0)
        wd = Supervisor._watchdog(fake, spec_for(2, "weights-only"))
    assert any(isinstance(d, TurnRepeat) for d in wd.detectors)


# ---- did the hardware test succeed? ---------------------------------------------------------------
# Only an exit code of exactly 0 with no timeout counts as success. Anything else, including a
# record that is missing or cannot be read, is a failure, so a failed test gets no gate feedback.

def record(tmp_path, data):
    (tmp_path / "test-result.json").write_text(data if isinstance(data, str) else json.dumps(data))
    return tmp_path


def test_a_test_that_exited_0_in_time_succeeded(tmp_path):
    assert hardware_test_failure(record(tmp_path, {"returncode": 0, "timed_out": False})) is None


@pytest.mark.parametrize("data,want", [
    ({"returncode": 4, "timed_out": False}, (4, False)),
    ({"returncode": None, "timed_out": True}, (None, True)),
    ({"returncode": 0, "timed_out": True}, (0, True)),
    # No exit code and no timeout: the command was refused when it was started.
    ({"returncode": None, "timed_out": False}, (None, False)),
    ({"returncode": False, "timed_out": False}, (False, False)),     # a bool is not an exit code
    ({"timed_out": False}, (None, False)),
])
def test_a_test_that_did_not_exit_0_in_time_failed(tmp_path, data, want):
    got = hardware_test_failure(record(tmp_path, data))
    assert (got["returncode"], got["timed_out"]) == want


@pytest.mark.parametrize("data", ["{not json", "[1, 2]"])
def test_an_unreadable_record_counts_as_a_failure(tmp_path, data):
    got = hardware_test_failure(record(tmp_path, data))
    assert got is not None and "test-result.json" in got["problem"]


def test_a_missing_record_counts_as_a_failure(tmp_path):
    got = hardware_test_failure(tmp_path)
    assert got is not None and "test-result.json" in got["problem"]
