"""The `--thin-port` option (issue 677, arm E): its place in the run (parse, build, run_start, resume), its hand-over to the existing stage 2
flow, and its pass-through from `tt-orchard bringup`. The step itself is in tests/test_thin_port.py."""
from __future__ import annotations

import json

import pytest

from orchard import stages, supervisor, thin_port
from orchard.supervisor import NOTE_KEYS
from orchard.stages import thin_port as thin_port_chosen, thin_port_options_problem


@pytest.fixture
def rig(tmp_path, stub_tools, unbuilt_stages_5_6):
    from test_supervisor import Rig
    r = Rig(tmp_path)
    yield r
    r.close()


def thin_decisions(rig):
    return [d for d in rig.decisions() if str(d["decision"]).startswith("thin port")]


# ---- the option ---------------------------------------------------------------------------------------------------------------------------

def test_parse_has_the_option_and_it_defaults_off(tmp_path):
    from run_fakes import argv
    a = argv(tmp_path, tmp_path / "t.toml", "http://127.0.0.1:8000/v1")
    assert supervisor.parse(a).thin_port is False
    assert supervisor.parse(a + ["--thin-port"]).thin_port is True


@pytest.mark.parametrize("full_port,thin,refused", [(True, True, False), (False, True, True), (True, False, False), (False, False, False)])
def test_the_option_needs_a_full_port(full_port, thin, refused):
    why = thin_port_options_problem(full_port=full_port, thin_port=thin)
    assert (why is not None) is refused
    if refused:
        assert "--thin-port" in why and "--full-port" in why


def test_the_supervisor_refuses_the_option_without_a_full_port(rig):
    rig.args.thin_port = True
    with pytest.raises(ValueError, match="--full-port"):
        rig.run()


def test_the_option_cannot_be_added_on_a_resume(rig):
    assert rig.run() == supervisor.EXIT_READY                      # started (and finished) without it
    rig.args.thin_port = True
    with pytest.raises(ValueError, match="only when the run starts"):
        rig.run(pid=200)


def test_only_run_start_turns_the_option_on():
    started = [{"event": "run_start", "stage": None, "data": {"thin_port": True}}]
    assert thin_port_chosen(started) is True
    assert thin_port_chosen([{"event": "run_start", "stage": None, "data": {}}]) is False
    assert thin_port_chosen([{"event": "run_start", "stage": None, "data": {"thin_port": 1}}]) is False          # exactly true
    assert thin_port_chosen([{"event": "decision", "stage": 2, "data": {"thin_port": True}}]) is False          # no later entry counts


# ---- in a run: the default path, the fall-back and the hand-over ---------------------------------------------------------------------------

@pytest.mark.usefixtures("stub_tools")
def test_without_the_option_nothing_of_the_thin_path_runs(tmp_path):
    from test_supervisor_full_port import HONEST, coder_writes, with_snapshot
    from test_supervisor import Rig
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        coder_writes(r, {}, port=HONEST)
        r.run()
        assert thin_decisions(r) == []
        start = next(e for e in r.entries() if e["event"] == "run_start")
        assert "thin_port" not in start["data"]
        assert any(d["decision"] == "agent step" and d.get("phase") == "prepare" for d in r.decisions())
    finally:
        r.close()


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_with_the_option_run_start_records_it_and_an_unusable_input_falls_back_to_the_agent_step(tmp_path, monkeypatch):
    from test_supervisor_full_port import HONEST, coder_writes, with_snapshot
    from test_supervisor import Rig
    monkeypatch.setattr(thin_port, "run", lambda **kw: thin_port.Outcome("skipped", "no reference interpreter"))
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        r.args.thin_port = True
        coder_writes(r, {}, port=HONEST)
        r.run()
        start = next(e for e in r.entries() if e["event"] == "run_start")
        assert start["data"]["thin_port"] is True
        skipped = [d for d in thin_decisions(r) if d["decision"] == "thin port not used"]
        assert skipped and skipped[0]["reason"] == "no reference interpreter"
        assert any(d["decision"] == "agent step" and d.get("phase") == "prepare" for d in r.decisions())          # the ordinary prepare step ran
    finally:
        r.close()


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_a_done_step_replaces_the_agent_prepare_step_and_the_port_goes_to_the_existing_flow(tmp_path, monkeypatch):
    from test_supervisor_full_port import HONEST, coder_writes, with_snapshot
    from test_supervisor import Rig
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return thin_port.Outcome("done", stubs_left=["_B_ple"], requests=3)
    monkeypatch.setattr(thin_port, "run", fake)
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        r.args.thin_port = True
        coder_writes(r, {}, port=HONEST)
        r.run()
        assert not any(d["decision"] == "agent step" and d.get("phase") == "prepare" and e_stage == 2
                       for d, e_stage in [(e["data"], e["stage"]) for e in r.entries() if e["event"] == "decision"])
        assert seen["stage"] == 2 and seen["model"] and seen["endpoint"] and "temperature" in seen["sampling"]
        handoff = json.loads((r.run_dir / "stages" / "2" / "handoff.json").read_text())
        assert all(handoff.get(k) not in (None, "") for k in NOTE_KEYS)                # the note the hardware phase reads, written by the supervisor
        # the port is the skeleton here, so the existing flow's CPU pre-check refuses it before any board: the hand-over reached that code
        assert any(d["decision"] == "device-ops pre-check failed before the board" for d in r.decisions())
    finally:
        r.close()


@pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")
def test_an_error_step_ends_the_stage_the_way_a_failed_agent_prepare_step_does(tmp_path, monkeypatch):
    from test_supervisor_full_port import coder_writes, with_snapshot
    from test_supervisor import Rig
    monkeypatch.setattr(thin_port, "run", lambda **kw: thin_port.Outcome("error", "the model request failed and was retried once (OSError)"))
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        r.args.thin_port = True
        coder_writes(r, {})
        assert r.run() == supervisor.EXIT_BLOCKED
        text = " ".join(str(d.get("reason", "")) for d in r.decisions())
        assert "the prepare step ended: error the model request failed and was retried once" in text
    finally:
        r.close()


def test_the_option_is_for_stage_2_of_a_full_port_only(tmp_path, monkeypatch):
    """Stage 3's prepare (the port copied from stage 2) and the weights-only stages never call the thin step."""
    calls = []
    monkeypatch.setattr(thin_port, "run", lambda **kw: calls.append(kw) or thin_port.Outcome("skipped", "x"))
    import inspect
    src = inspect.getsource(supervisor.Supervisor._stage_body)
    assert "_thin_port_prepare" in src
    helper = inspect.getsource(supervisor.Supervisor._thin_port_prepare)
    assert "n == 2" in helper and 'spec.draft == "full_port"' in helper
