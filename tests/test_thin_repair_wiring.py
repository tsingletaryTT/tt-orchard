"""The thin repair inside the supervisor's port repair rounds (issue 677, arm E2): with `--thin-port` a failed board test is repaired by one chat
request for the failing function, and the agent loop stays the fallback. The chips, the coder and the model are fakes; the supervisor, the
stage machine, the gate and the check template (which runs for real on the CPU with a fake `ttnn`) are the real ones."""
from __future__ import annotations

import json
import re

import pytest

from orchard import port_repair, thin_port, thin_repair
from orchard.supervisor import EXIT_ABORTED
from port_check_fakes import HONEST
from run_fakes import bringup, where
from test_supervisor import Rig
from test_supervisor_full_port import PREPARE, make, never_wait, started, with_snapshot
from test_supervisor_port_repair import BROKEN, exit_codes, rounds

pytestmark = pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")

LOAD_FIXED = "def load(model_dir, mesh_device):\n    return {\"dir\": model_dir, \"device\": mesh_device}\n"


@pytest.fixture
def rig(tmp_path, monkeypatch):
    # The thin PREPARE step has no reference interpreter in the rig: it reports "skipped" and the agent's prepare step writes the broken port.
    monkeypatch.setattr(thin_port, "run", lambda **kw: thin_port.Outcome("skipped", "no reference interpreter"))
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.thin_port = True
    yield r
    r.close()


def scripted(rig, thin_reply, *, agent_repairs=None):
    """Chat requests without `tools` are thin requests: answer them with `thin_reply(n, request)` (a message dict, or an int for an HTTP error).
    An agent repair step writes `agent_repairs` (default: the honest port, so a test sees whether the agent loop ran at all)."""
    thin = []
    base = rig.script

    def script(request):
        if "tools" not in request:
            thin.append(request)
            return thin_reply(len(thin), request)
        if where(request) == (2, "repair"):
            return bringup(request, overrides={(2, "repair"): agent_repairs or {"port/port_model.py": HONEST}})
        return base(request)
    rig.script = script
    return thin


def fenced(code):
    return {"role": "assistant", "content": "```python\n" + code + "```"}


def setup(rig, port=BROKEN):
    make(rig, over={(2, "prepare"): {**PREPARE, "port/port_model.py": port}})
    rig.args.unattended = False
    from orchard.supervisor import Control
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")


def thin_rounds(rig):
    return [d for d in rig.decisions() if d.get("decision") == "thin repair round"]


def agent_repair_steps(rig):
    return [d for d in rig.decisions() if d.get("decision") == "agent step" and d.get("phase") == "repair"]


def test_a_failed_check_is_repaired_by_one_request_for_the_failing_function_and_no_agent_repair_step_runs(rig):
    setup(rig)
    thin = scripted(rig, lambda n, r: fenced(LOAD_FIXED))
    assert rig.run() == EXIT_ABORTED
    assert exit_codes(rig) == [2, 0] and [d["result"] for d in rig.ends(2)] == ["pass"]
    assert len(thin) == 1 and agent_repair_steps(rig) == []
    user = thin[0]["messages"][-1]["content"]
    assert "## The function to fix: load" in user and "FAILED: the port failed at 'load'" in user
    d = thin_rounds(rig)
    assert len(d) == 1 and d[0]["round"] == 1 and d[0]["function"] == "load" and d[0]["error_type"] == "KeyError"
    assert d[0]["outcome"] == "spliced" and d[0]["requests"] == 1 and isinstance(d[0]["completion_tokens"], int)
    assert [(x["round"], x["op"]) for x in rounds(rig)] == [(1, "load")]                     # the round is counted as before


def test_the_ledger_of_a_thin_round_holds_no_reply_text_and_no_path(rig):
    setup(rig)
    scripted(rig, lambda n, r: fenced(LOAD_FIXED + "# SENTINEL_77aa\n"))
    rig.run()
    text = json.dumps(rig.entries())
    assert "SENTINEL_77aa" not in text and str(rig.run_dir) not in json.dumps(thin_rounds(rig))


def test_two_refused_replies_hand_the_round_to_the_agent_loop(rig):
    setup(rig)
    thin = scripted(rig, lambda n, r: {"role": "assistant", "content": "I cannot do that"})
    assert rig.run() == EXIT_ABORTED
    assert len(thin) == thin_repair.REQUESTS_PER_ROUND
    handed = [d for d in rig.decisions() if d.get("decision") == "thin repair handed to the coder"]
    assert handed and handed[0]["round"] == 1 and handed[0]["outcome"] == "refused" and handed[0]["requests"] == thin_repair.REQUESTS_PER_ROUND
    assert len(agent_repair_steps(rig)) == 1 and exit_codes(rig) == [2, 0]                  # the agent loop repaired it, as before


def test_a_failure_the_thin_repair_cannot_place_goes_to_the_agent_loop_without_a_request(rig):
    setup(rig, port=HONEST.replace("def prefill(model, ids):\n    return _peak(100)", "X = 1\ndef prefill(model, ids):\n    return _peak(100)\nraise KeyError('k')"))
    thin = scripted(rig, lambda n, r: fenced(LOAD_FIXED))
    rig.run()
    assert thin == []
    handed = [d for d in rig.decisions() if d.get("decision") == "thin repair handed to the coder"]
    assert handed and "top-level function" in handed[0]["reason"] or "Python error" in handed[0]["reason"]


def test_an_endpoint_that_fails_hands_the_round_to_the_agent_loop(rig):
    setup(rig)
    scripted(rig, lambda n, r: 500)
    rig.run()
    handed = [d for d in rig.decisions() if d.get("decision") == "thin repair handed to the coder"]
    assert handed and len(agent_repair_steps(rig)) == 1


def test_without_the_option_no_thin_repair_runs(tmp_path, monkeypatch):
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        setup(r)
        thin = scripted(r, lambda n, req: fenced(LOAD_FIXED))
        r.run()
        assert thin == [] and thin_rounds(r) == [] and len(agent_repair_steps(r)) == 1
    finally:
        r.close()


def test_rounds_past_the_thin_cap_are_the_agent_loops(rig, monkeypatch):
    monkeypatch.setattr(thin_repair, "MAX_ROUNDS", 0)
    setup(rig)
    thin = scripted(rig, lambda n, r: fenced(LOAD_FIXED))
    rig.run()
    assert thin == [] and len(agent_repair_steps(rig)) == 1
    handed = [d for d in rig.decisions() if d.get("decision") == "thin repair handed to the coder"]
    assert handed and "past the" in handed[0]["reason"]


def test_a_failure_that_is_not_the_ports_goes_to_the_agent_loop_without_a_request(rig):
    """The thin repair asks about Python errors of the port. A numerics failure (the check ran to the end) is the agent loop's."""
    from orchard.ledger import Ledger
    from orchard.stages import FULL_PORT_STAGE_2
    from orchard.supervisor import build
    asked = []
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        led.append("run_start", None, thin_port=True)
        sup = build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=100), coder=rig.coder_cls(rig.m), home=rig.home, containers=rig.containers)
        sup.http = lambda *a, **k: asked.append(a)
        sup._driven_stop = lambda spec: None              # no watchdog or actuator exists outside a run
        done, stop = sup._thin_repair_round(FULL_PORT_STAGE_2, rig.run_dir / "stages" / "2", 1, {"kind": "numerics", "op": "numerics"}, "raw", False)
        assert (done, stop) == (False, None) and asked == []
        handed = [e["data"] for e in led.read() if e["data"].get("decision") == "thin repair handed to the coder"]
        assert handed and "not a failure of the port itself" in handed[0]["reason"]
