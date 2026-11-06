# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 6 of the full-port route in the supervisor (issue 329). The chip, the coder and the model are fakes; the supervisor, the stage machine, the gate,
the park and restore path and the hardware-test runner are the real ones, and the supervisor's quality template runs for real on the CPU against a fake
`ttnn`. Stages 4 and 5 for a full port are other slices; here they are the generic stages the scripted coder passes, reached by the attended resume that
goes on past the pause before stage 4."""
import dataclasses
import json

import pytest

from orchard import full_port_quality as fq
from orchard import supervisor
from orchard.stages import FULL_PORT_STAGE_6, spec_for
from orchard.supervisor import EXIT_READY, REHEARSAL_WAIVED, Control
from port_check_fakes import HONEST
from quality_fakes import LOOPS, TINY_CONFIG
from run_fakes import where, bringup
from test_supervisor import Rig, step_skills
from test_supervisor_full_port import FULL_PORT, PREPARE, STAGE_1, started, with_snapshot, never_wait

pytestmark = pytest.mark.usefixtures("stub_tools", "generic_stage_5")


@pytest.fixture
def generic_stage_5(monkeypatch):
    """These tests are about stage 6 alone. Stage 5 of a full port is its own slice (tests/test_supervisor_full_port_serve.py), so here stage 5 is the
    generic serving-check stage the scripted coder passes, as it was before stage 5 was built. The chain of all the stages is tested in
    tests/test_full_port_chain.py."""
    from orchard import stages
    monkeypatch.setattr(stages, "FULL_PORT_STAGE_5", stages.STAGES[5])

STAGE_1_Q = STAGE_1          # test_supervisor_full_port.py's stage 1 holds the top-five file


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    (tmp_path / "snapshot" / "config.json").write_text(json.dumps(TINY_CONFIG))
    r.args.full_port = True
    yield r
    r.close()


def make_q(rig, *, port=HONEST, chips=None):
    """The scripted coder: the port in stage 2 (and in stage 3 when the run configures chips), the stage 1 files with the top five, and the usual finish
    steps. Stage 6 has no scripted step: no agent is asked for it."""
    files = {**FULL_PORT, (1, "run"): STAGE_1_Q, (2, "prepare"): {**PREPARE, "port/port_model.py": port}}
    if chips:
        from test_supervisor_full_port_stage3 import PREPARE_3
        files[(3, "prepare")] = {**PREPARE_3, "port/port_model.py": port}

    def script(request):
        f = dict(files)
        if "tools" in request:
            n = where(request)[0]
            assert n != 6, "stage 6 asked the coder for a step"
        from test_supervisor_full_port_stage3 import from_record
        for n in (2, 3):
            record = rig.run_dir / "stages" / str(n) / "evidence" / "port-check.json"
            if record.is_file() and "load_s" in record.read_text():
                f[(n, "finish")] = from_record(rig, n)
        return bringup(request, overrides=f)
    rig.script = script
    pauses = []

    def on_sleep():
        # an attended operator goes on past the pause before stage 4, once; a later pause (a failed stage) ends the run
        Control(rig.run_dir).write("abort" if pauses else "resume")
        pauses.append(1)
    rig.on_sleep = on_sleep


def stage_6(rig):
    return rig.run_dir / "stages" / "6"


def decisions_of(rig, n, name):
    return [e["data"] for e in rig.entries() if e["stage"] == n and e["event"] == "decision" and e["data"].get("decision") == name]


def test_the_spec_is_the_quality_stage_only_for_a_full_port_with_the_flag():
    assert spec_for(6, "full-port", None, "full-port", full_port=True) is FULL_PORT_STAGE_6
    assert spec_for(6, "full-port", None, "full-port", full_port=False) is not FULL_PORT_STAGE_6
    assert spec_for(6, "weights-only", None, "weights-only", full_port=True) is not FULL_PORT_STAGE_6
    s = FULL_PORT_STAGE_6
    assert s.skill == "full-port-quality" and s.draft == "full_port_quality" and s.needs_coder is False and s.gate_file == "result.json" and s.boards == 1


def test_stage_6_runs_the_port_of_stage_2_with_no_agent_and_passes(rig):
    make_q(rig)
    assert rig.run() == EXIT_READY
    assert [d["result"] for d in rig.ends(6)] == ["pass"]
    assert step_skills(rig, 6) == []
    assert not [e for e in rig.entries() if e["stage"] == 6 and e["data"].get("decision") == "agent step"]
    result = json.loads((stage_6(rig) / "result.json").read_text())
    assert result["derived_by"] == "supervisor" and result["source_stage"] == 2 and result["benchmark"]["repeats"] >= 3
    (copied,) = decisions_of(rig, 6, "stage 2 port copied")
    assert [f["path"] for f in copied["files"]] == ["stages/6/port/port_model.py"]
    (d,) = decisions_of(rig, 6, "result derived by the supervisor")
    assert d["benchmark_label"] == "benchmark" and d["leading_match"] == 32
    marker = json.loads((stage_6(rig) / "test-result.json").read_text())
    assert marker["command"] == fq.COMMAND and marker["returncode"] == 0


def test_the_hardware_test_of_stage_6_runs_under_a_lease_the_supervisor_took(rig):
    make_q(rig)
    rig.run()
    ds = [e["data"] for e in rig.entries() if e["stage"] == 6 and e["event"] == "decision"]
    assert any(d.get("decision") == "hardware phase" for d in ds) and any(d.get("decision") == "hardware test started" for d in ds)
    marker = json.loads((stage_6(rig) / "test-result.json").read_text())
    rec = json.loads((stage_6(rig) / "evidence" / "port-quality.json").read_text())
    assert marker["chips"] == rec["chips"] != []
    assert rig.m.leases == {} and rig.m.coder_running is False          # everything is given back at the end of the run


def test_with_chips_configured_stage_6_measures_stage_3s_port_on_all_four(rig, monkeypatch):
    # Stage 4 of a full port is another slice, and the generic stage 4 needs hardware test records for a 4-chip container; it is skipped here.
    real = supervisor.spec_for
    monkeypatch.setattr(supervisor, "spec_for", lambda n, *a, **k: dataclasses.replace(real(n, *a, **k), skip="stage 4 is another slice") if n == 4 else real(n, *a, **k))
    rig.args.required_chips = (2, 4)
    make_q(rig, chips=True)
    assert rig.run() == EXIT_READY
    (copied,) = decisions_of(rig, 6, "stage 3 port copied")
    assert copied["files"][0]["path"] == "stages/6/port/port_model.py"
    rec = json.loads((stage_6(rig) / "evidence" / "port-quality.json").read_text())
    assert len(rec["chips"]) == 4 and json.loads((stage_6(rig) / "result.json").read_text())["source_stage"] == 3
    assert [d["result"] for d in rig.ends(6)] == ["pass"]


def test_the_template_and_the_fixed_command_are_the_supervisors(rig):
    make_q(rig)
    rig.run()
    assert (stage_6(rig) / fq.TEMPLATE_NAME).read_bytes() == fq.TEMPLATE.read_bytes()
    assert json.loads((stage_6(rig) / "hw_test.json").read_text())["command"] == fq.COMMAND
    assert decisions_of(rig, 6, "port quality template copied")


def test_the_rehearsal_option_waives_coherent_in_stage_6_and_says_so(rig):
    rig.args.full_port_rehearsal = True
    make_q(rig, port=LOOPS)
    assert rig.run() == EXIT_READY
    result = json.loads((stage_6(rig) / "result.json").read_text())
    assert result["coherent"] is False
    waived = decisions_of(rig, 6, REHEARSAL_WAIVED)
    assert len(waived) == 1 and waived[0]["field"] == "coherent"


def test_without_the_rehearsal_option_a_looping_port_fails_stage_6_with_the_reason(rig):
    make_q(rig, port=LOOPS)
    code = rig.run()
    assert code != EXIT_READY
    ends = rig.ends(6)
    assert ends and ends[-1]["result"] != "pass" and "readable floor" in " ".join(ends[-1].get("reasons", []))


def test_a_port_that_rewrites_its_own_source_during_the_stage_6_test_is_not_passed(rig):
    """The same rewrite in stage 2 would be refused by stage 2's own gate, so the port is honest there and dishonest only in stage 6's copy."""
    make_q(rig)
    # stage 6's copy is made from stage 2's passed port; the rewrite is planted in the copy's source by a stage-2 port that carries it
    # and only acts when it runs under stages/6
    sneaky = HONEST.replace("def load(model_dir, mesh_device):\n",
                            "def load(model_dir, mesh_device):\n"
                            "    import pathlib\n"
                            "    p = pathlib.Path(__file__)\n"
                            "    if 'stages/6/' in str(p):\n"
                            "        p.write_text(p.read_text() + '\\n# changed while the test ran\\n')\n", 1)
    make_q(rig, port=sneaky)
    code = rig.run()
    assert code != EXIT_READY
    ends = rig.ends(6)
    assert ends[-1]["result"] != "pass" and "supervisor copied" in " ".join(ends[-1].get("reasons", []))


def bundle(rig, name="RESULTS.md"):
    return (rig.run_dir / "stages" / "8" / "bundle" / name).read_text()


def test_the_operator_bundle_reports_stage_6_with_labels_and_lists_the_todo_figures(rig):
    make_q(rig)
    assert rig.run() == EXIT_READY
    text = bundle(rig)
    assert "## Full port: quality against the reference and benchmark" in text
    section = text.split("## Full port: quality against the reference and benchmark")[1].split("\n## ")[0]
    assert "### Benchmark (benchmark)" in section and "Share of the decode floor" in section and "(measured)" in section
    assert "| Decode speed (tok/s) |" in section and "Top-5 logit gap difference from the reference" in section
    stage6 = text.split("## Stage 6:")[1].split("\n## ")[0]
    assert "| share of the estimated decode floor, median speed |" in stage6 and "| measured |" in stage6 and "| TODO |" in stage6
    risks = bundle(rig, "RISKS.md")
    assert "quality on prompts other than the stage 1 prompt: TODO" in risks and "decode floor assumes every weight is bf16" in risks


def test_the_bundle_reports_stage_5_numbers_with_their_labels(rig):
    """Stage 5 is built by another slice, which writes stages/5/result.json. Whatever `numbers` it holds are printed with their labels."""
    make_q(rig)
    base = rig.script
    from run_fakes import FILES, done
    stage5 = {"result.json": {"checks": {k: {"pass": True, "evidence": done(5)} for k in ("boots", "passkey", "canary")},
                              "numbers": [{"name": "boot to ready", "value": 41.5, "unit": "s", "label": "measured", "evidence": done(5)},
                                          {"name": "replay of the acceptance corpus", "value": None, "unit": "", "label": "TODO"}]}}
    files = {**FULL_PORT, (1, "run"): STAGE_1_Q, (5, "finish"): stage5}
    rig.script = lambda request: bringup(request, overrides=files) if where(request) == (5, "finish") or "tools" not in request else base(request)
    assert rig.run() == EXIT_READY
    text = bundle(rig)
    stage5_text = text.split("## Stage 5:")[1].split("\n## ")[0]
    assert "| boot to ready | 41.5 s | measured |" in stage5_text and "| replay of the acceptance corpus | TODO | TODO | - |" in stage5_text
    assert "stage 5 replay of the acceptance corpus: TODO" in bundle(rig, "RISKS.md")


def test_a_top_five_file_replaced_after_stage_1_passed_stops_stage_6(rig):
    """Review of PR 491: the replacement changes the reference gaps, and the record's own hash would match it. The ledger's hash from stage 1's pass
    is the one that counts, before the test runs and again at the gate."""
    make_q(rig)
    inner = rig.script
    top5 = rig.run_dir / "stages" / "1" / "evidence" / "reference" / "top5-logits.json"
    tampered = []

    def tamper(request):
        if "tools" in request and where(request)[0] == 2 and top5.is_file() and not tampered:       # the coder's first step after stage 1 passed
            data = json.loads(top5.read_text())
            data["steps"][0]["top5"][0]["logit"] += 6.0
            top5.write_text(json.dumps(data))
            tampered.append(1)
        return inner(request)
    rig.script = tamper
    assert rig.run() != EXIT_READY
    assert tampered
    ends = rig.ends(6)
    assert ends and ends[-1]["result"] != "pass"
    assert "not the file stage 1's gate accepted" in " ".join(ends[-1].get("reasons", []))
