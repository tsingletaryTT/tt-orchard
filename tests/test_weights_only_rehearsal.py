# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The weights-only rehearsal option (`--weights-only-rehearsal`, Track C item C5 of the Qwen4 drop-in plan).

A rehearsal "release" on the weights-only route (the perturbed Qwen4 frankenmodel served through the Qwen4 bundle) gives text that is not
readable, so `coherent` is false wherever the swap check judges the free-run text: stage 2's result, each stage 4 configuration and the stage 7
boot check. The option waives that one field in those three gates, only for a measured `false`, and nothing else. The operator gives it when the
run starts, run_start records it, and nothing an agent writes in a stage directory turns it on."""
import functools
import json
import types

import pytest

from orchard import stages, supervisor
from orchard.stages import (PACKAGE_STAGE_7, SIDECAR_STAGE_2, WEIGHTS_ONLY_STAGE_2, WEIGHTS_ONLY_STAGE_4, GateResult, gate_mesh_swap,
                            gate_package, gate_weights_swap, gate_weights_swap_sidecar, rehearsal_options_problem, weights_only_rehearsal)
from test_gate_package import stage7  # noqa: F401  (the fixture: a staged, boot-checked and finished stage 7)
from test_stages import SWAP, mesh_stage, stage, swap_config


# ---- stage 2 -------------------------------------------------------------------------------------------------------------------------

def swap2(tmp_path, rehearsal=None, **change):
    data = {k: v for k, v in {**SWAP, **change}.items() if v is not ...}
    sd, run = stage(tmp_path, 2, "result.json", data)
    return gate_weights_swap(sd, run) if rehearsal is None else gate_weights_swap(sd, run, rehearsal=rehearsal)


def test_the_stage_2_gate_waives_a_measured_false_coherent_in_a_rehearsal(tmp_path):
    g = swap2(tmp_path, rehearsal=True, coherent=False)
    assert g.ok, g.reasons
    assert g.waived == ("coherent",)


def test_the_stage_2_gate_is_unchanged_without_the_option(tmp_path):
    for kw in ({}, {"rehearsal": False}):
        g = swap2(tmp_path / str(len(kw)), coherent=False, **kw)
        assert not g.ok and len(g.reasons) == 1 and "coherent must be true" in g.reasons[0], g.reasons
        assert g.waived == ()


def test_a_readable_rehearsal_waives_nothing(tmp_path):
    g = swap2(tmp_path, rehearsal=True)
    assert g.ok and g.waived == ()


@pytest.mark.parametrize("value", [..., None, "false", 0, [], {}])
def test_the_waiver_takes_only_a_measured_false(tmp_path, value):
    # A missing or malformed field is no measurement. The option accepts unreadable text; an absent check still fails.
    g = swap2(tmp_path, rehearsal=True, coherent=value)
    assert not g.ok and any("coherent" in r for r in g.reasons), g.reasons
    assert g.waived == ()


@pytest.mark.parametrize("change", [{"top1_agreement": 0.5}, {"n_tokens": 4}, {"serves": False}, {"server_ready_s": 0},
                                    {"evidence": ["stages/2/evidence/missing.txt"]}])
def test_a_rehearsal_still_fails_every_other_stage_2_field(tmp_path, change):
    g = swap2(tmp_path, rehearsal=True, coherent=False, **change)
    assert not g.ok
    assert not any("coherent" in r for r in g.reasons), g.reasons


def test_a_field_in_result_json_waives_nothing(tmp_path):
    data = {**SWAP, "coherent": False, "weights_only_rehearsal": True, "rehearsal": True, "waived": ["coherent"]}
    g = gate_weights_swap(*stage(tmp_path, 2, "result.json", data))
    assert not g.ok and any("coherent must be true" in r for r in g.reasons) and g.waived == ()


def test_the_sidecar_class_stage_2_gate_takes_the_same_waiver(tmp_path):
    sd, run = stage(tmp_path, 2, "result.json", {**SWAP, "coherent": False})
    off, on = gate_weights_swap_sidecar(sd, run), gate_weights_swap_sidecar(sd, run, rehearsal=True)
    assert any("coherent must be true" in r for r in off.reasons) and off.waived == ()
    assert not any("coherent must be true" in r for r in on.reasons) and on.waived == ("coherent",)   # fails for its sidecar parity only


# ---- stage 4 -------------------------------------------------------------------------------------------------------------------------

def test_the_stage_4_gate_waives_coherent_in_each_passing_configuration(tmp_path):
    configs = [swap_config(2, coherent=False), swap_config(4, coherent=False)]
    on = gate_mesh_swap(*mesh_stage(tmp_path / "on", configs), required=(2, 4), rehearsal=True)
    assert on.ok, on.reasons
    assert on.waived == ("coherent",)
    off = gate_mesh_swap(*mesh_stage(tmp_path / "off", configs), required=(2, 4))
    assert not off.ok and off.waived == ()
    assert sorted(off.reasons) == ["the 2-chip configuration: coherent must be true (the free-run text is readable), got False",
                                   "the 4-chip configuration: coherent must be true (the free-run text is readable), got False"]


def test_a_stage_4_rehearsal_still_holds_each_configuration_to_the_other_fields(tmp_path):
    g = gate_mesh_swap(*mesh_stage(tmp_path, [swap_config(4, coherent=False, top1_agreement=25 / 32)]), required=(4,), rehearsal=True)
    assert g.reasons == ("the 4-chip configuration: top1_agreement 0.78125 is below the minimum of 0.85",)


def test_a_stage_4_configuration_with_no_coherent_field_is_not_waived(tmp_path):
    entry = {k: v for k, v in swap_config(4).items() if k != "coherent"}
    g = gate_mesh_swap(*mesh_stage(tmp_path, [entry]), required=(4,), rehearsal=True)
    assert not g.ok and g.waived == ()


# ---- stage 7 -------------------------------------------------------------------------------------------------------------------------



def _boot_coherent(s, value):
    v = s["stage"] / "verify/evidence/verify.json"
    data = json.loads(v.read_text())
    if value is ...:
        data.pop("coherent")
    else:
        data["coherent"] = value
    v.write_text(json.dumps(data))


def test_the_stage_7_gate_waives_the_boot_checks_coherent_in_a_rehearsal(stage7):
    _boot_coherent(stage7, False)
    off = gate_package(stage7["stage"], stage7["run"])
    assert any("the boot check's free-run text is not coherent" in r for r in off.reasons) and off.waived == ()
    on = gate_package(stage7["stage"], stage7["run"], rehearsal=True)
    assert on.ok, on.reasons
    assert on.waived == ("coherent",)


@pytest.mark.parametrize("value", [..., None, "false"])
def test_the_stage_7_waiver_takes_only_a_measured_false(stage7, value):
    _boot_coherent(stage7, value)
    g = gate_package(stage7["stage"], stage7["run"], rehearsal=True)
    assert any("not coherent" in r for r in g.reasons) and g.waived == ()


def test_a_stage_7_rehearsal_still_fails_a_boot_check_below_the_bar(stage7):
    v = stage7["stage"] / "verify/evidence/verify.json"
    v.write_text(json.dumps({**json.loads(v.read_text()), "coherent": False, "top1_agreement": 0.5}))
    g = gate_package(stage7["stage"], stage7["run"], rehearsal=True)
    assert not g.ok and any("below" in r for r in g.reasons)


# ---- the ledger answer and the options -------------------------------------------------------------------------------------------------

def test_only_run_start_turns_the_option_on():
    start = {"event": "run_start", "data": {"model": "m"}}
    claims = [{"event": "decision", "data": {"decision": "weights-only rehearsal", "weights_only_rehearsal": True}},
              {"event": "evidence", "data": {"weights_only_rehearsal": True}},
              {"event": "run_start", "data": {"weights_only_rehearsal": "yes"}}]
    assert weights_only_rehearsal([start, *claims]) is False
    assert weights_only_rehearsal([{**start, "data": {"weights_only_rehearsal": True}}]) is True
    # The two rehearsal options are separate answers.
    assert weights_only_rehearsal([{**start, "data": {"full_port_rehearsal": True}}]) is False
    assert stages.full_port_rehearsal([{**start, "data": {"weights_only_rehearsal": True}}]) is False


@pytest.mark.parametrize("full_port,fp_rehearsal,wo_rehearsal,refused", [
    (False, False, True, False),
    (True, True, True, False),          # one rehearsal model, whichever route stage 0 measures
    (True, False, True, True),          # the full-port route would still ask for readable text
    (False, False, False, False),
    (True, True, False, False),
])
def test_the_option_combinations(full_port, fp_rehearsal, wo_rehearsal, refused):
    why = rehearsal_options_problem(full_port=full_port, full_port_rehearsal=fp_rehearsal, weights_only_rehearsal=wo_rehearsal)
    assert (why is not None) is refused
    if refused:
        assert "--weights-only-rehearsal" in why and "--full-port-rehearsal" in why


# ---- the supervisor's choice of gate -------------------------------------------------------------------------------------------------

class _Ledger:
    def __init__(self, entries=()):
        self.entries, self.appended = list(entries), []

    def read(self):
        return list(self.entries)

    def append(self, event, stage, **data):
        self.appended.append((event, stage, data))
        self.entries.append({"event": event, "stage": stage, "data": data})


def _sup(entries, required=None):
    return types.SimpleNamespace(ledger=_Ledger(entries), required_chips=required)


ON = [{"event": "run_start", "stage": None, "data": {"weights_only_rehearsal": True}}]
OFF = [{"event": "run_start", "stage": None, "data": {}}]
FULL = [{"event": "run_start", "stage": None, "data": {"full_port": True, "full_port_rehearsal": True}}]


@pytest.mark.parametrize("spec", [WEIGHTS_ONLY_STAGE_2, SIDECAR_STAGE_2, PACKAGE_STAGE_7])
def test_the_supervisor_gives_the_waiver_to_the_weights_only_gates_from_run_start(spec):
    g = supervisor.Supervisor._gate(_sup(ON), spec)
    assert isinstance(g, functools.partial) and g.func is spec.gate and g.keywords == {"rehearsal": True}
    for entries in (OFF, FULL):
        assert supervisor.Supervisor._gate(_sup(entries), spec) is spec.gate


def test_stage_4_gets_the_waiver_and_the_required_counts():
    g = supervisor.Supervisor._gate(_sup(ON, required=(2, 4)), WEIGHTS_ONLY_STAGE_4)
    assert g.func is gate_mesh_swap and g.keywords == {"rehearsal": True, "required": (2, 4)}
    g = supervisor.Supervisor._gate(_sup(ON), WEIGHTS_ONLY_STAGE_4)
    assert g.func is gate_mesh_swap and g.keywords == {"rehearsal": True}
    g = supervisor.Supervisor._gate(_sup(OFF, required=(2, 4)), WEIGHTS_ONLY_STAGE_4)
    assert g.func is gate_mesh_swap and g.keywords == {"required": (2, 4)}


def test_no_other_gate_gets_the_waiver():
    for spec in stages.STAGES:
        if spec.gate is None or spec.number == 0:
            continue
        g = supervisor.Supervisor._gate(_sup(ON, required=(2, 4)), spec)
        assert "rehearsal" not in getattr(g, "keywords", {}), spec.name


def _checked(spec, gate_result, tmp_path):
    sup = types.SimpleNamespace(ledger=_Ledger(ON), run_dir=tmp_path, required_chips=None,
                                _run_gate=lambda spec, sd: gate_result, _note_unverified=lambda n, g: None)
    out = supervisor.Supervisor._check_gate(sup, spec, tmp_path / "stages" / str(spec.number))
    return out, [d for e, _, d in sup.ledger.appended if e == "decision"]


def test_a_pass_that_used_the_waiver_leaves_one_ledger_decision(tmp_path):
    out, decisions = _checked(PACKAGE_STAGE_7, GateResult(True, waived=("coherent",)), tmp_path)
    assert out.ok
    assert decisions == [{"decision": supervisor.WEIGHTS_REHEARSAL_WAIVED, "field": "coherent", "by": "--weights-only-rehearsal",
                          "note": decisions[0]["note"]}]
    assert "every other stage 7 check still applied" in decisions[0]["note"]


def test_no_decision_without_a_waiver_or_without_a_pass(tmp_path):
    assert _checked(PACKAGE_STAGE_7, GateResult(True), tmp_path)[1] == []
    assert _checked(PACKAGE_STAGE_7, GateResult(False, ("top1",), waived=("coherent",)), tmp_path)[1] == []


# ---- the whole run (the fake two-board machine) --------------------------------------------------------------------------------------

def _incoherent_run(rig):
    from run_fakes import FILES, SWAP_SCRIPTS, bringup, swap_entry
    two = {"result.json": {**FILES[(2, "finish")]["result.json"], "coherent": False, "free_run_text": "xq9Zk xq9Zk"}}
    four = {"result.json": {"configs": [{**swap_entry(n), "coherent": False} for n in SWAP_SCRIPTS]}}
    rig.script = lambda r: bringup(r, overrides={(2, "finish"): two, (4, "finish"): four})


@pytest.fixture
def rig(tmp_path, stub_tools):
    from test_supervisor import Rig
    r = Rig(tmp_path)
    yield r
    r.close()


def test_parse_has_the_option_and_it_defaults_off(tmp_path):
    from run_fakes import argv
    a = argv(tmp_path, tmp_path / "t.toml", "http://127.0.0.1:8000/v1")
    assert supervisor.parse(a).weights_only_rehearsal is False
    assert supervisor.parse(a + ["--weights-only-rehearsal"]).weights_only_rehearsal is True


def test_a_rehearsal_run_reaches_ready_and_the_ledger_names_each_waiver(rig):
    _incoherent_run(rig)
    rig.args.weights_only_rehearsal = True
    assert rig.run() == supervisor.EXIT_READY
    start = next(e for e in rig.entries() if e["event"] == "run_start")
    assert start["data"]["weights_only_rehearsal"] is True
    waived = [e for e in rig.entries() if e["event"] == "decision" and e["data"].get("decision") == supervisor.WEIGHTS_REHEARSAL_WAIVED]
    assert [e["stage"] for e in waived] == [2, 4]
    assert all(e["data"]["field"] == "coherent" and e["data"]["by"] == "--weights-only-rehearsal" for e in waived)
    assert not any(d.get("decision") == supervisor.REHEARSAL_WAIVED for d in rig.decisions())


def test_without_the_option_the_same_run_blocks_at_stage_2(rig):
    _incoherent_run(rig)
    rig.args.unattended = True
    assert rig.run() == supervisor.EXIT_BLOCKED
    assert "weights_only_rehearsal" not in next(e for e in rig.entries() if e["event"] == "run_start")["data"]
    blocked = [d for d in rig.decisions() if d["decision"] == "blocked"][0]
    assert "coherent" in blocked["reason"]
    assert not any(d.get("decision") == supervisor.WEIGHTS_REHEARSAL_WAIVED for d in rig.decisions())


def test_the_option_cannot_be_added_on_a_resume(rig):
    assert rig.run() == supervisor.EXIT_READY                  # started (and finished) without it
    rig.args.weights_only_rehearsal = True
    with pytest.raises(ValueError, match="only when the run starts"):
        rig.run(pid=200)


def test_a_resume_without_the_flag_keeps_the_recorded_waiver(rig):
    from fakes import Crash
    _incoherent_run(rig)
    rig.args.weights_only_rehearsal = True
    with pytest.raises(Crash):
        rig.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 2)
    rig.args.weights_only_rehearsal = False                    # the resume leaves the flag out; stage 4 still gets the waiver
    assert rig.run(pid=200) == supervisor.EXIT_READY
    assert weights_only_rehearsal(rig.entries()) is True
    assert [e["stage"] for e in rig.entries() if e["event"] == "decision"
            and e["data"].get("decision") == supervisor.WEIGHTS_REHEARSAL_WAIVED] == [2, 4]


def test_the_supervisor_refuses_a_half_rehearsal(rig):
    rig.args.weights_only_rehearsal = True
    rig.args.full_port = True
    with pytest.raises(ValueError, match="--full-port-rehearsal"):
        rig.run()
    rig.args.full_port_rehearsal = True
    assert rig.run() == supervisor.EXIT_READY                  # both rehearsal options together start (stage 0 measures weights-only here)
    assert weights_only_rehearsal(rig.entries()) and stages.full_port_rehearsal(rig.entries())
