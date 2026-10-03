"""The supervisor's parts: control file, actuator, external stand-in, coder tier, versions."""
import pytest

from fake_model import FakeModel, final
from fakes import FakeRun
from orchard.server import ServerSpec
from orchard.supervisor import Control, ExternalStandIn, RunActuator, coder_tier, resolve_versions
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
