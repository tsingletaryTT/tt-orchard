# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 143: stage 0 of an audio run runs as supervisor code, so the local coder never boots for a run that does not use it.

The preflight knows a model is audio before stage 0 (orchard/audio_facts.py) and the command records two run_start inputs, `audio_model=1` and `audio_base=<the
base repository id>`. With them the stage table gives stage 0 a harness body (`AUDIO_STAGE_0`, `needs_coder` False): the supervisor copies the shipped
`delta_triage.py` into the stage, writes its `triage_config.json` from the run's inputs and runs it. The template stays the judge of whether the model is audio:
when it reports a text model the stage fails. A run started without the inputs (every run from before this change) keeps the generic stage 0 and the coder.

The runs below use the supervisor as it runs, the real template on small audio snapshots (tests/test_delta_triage_audio.py), and the fakes of
test_supervisor_audio_runtime.py for everything after stage 0."""
import copy
import json
import os
from pathlib import Path

import pytest

from orchard import audio_runtime, audio_stage0, stages, supervisor
from orchard.stages import AUDIO_STAGE_0, STAGES, spec_for
from orchard.supervisor import EXIT_READY, Control
from run_fakes import Crash
from test_audio_chip_config import REAL_CONFIG
from test_delta_triage_audio import CONFIG, TENSORS, TEXT_ENCODER_CONFIG, TEXT_TENSORS, write_safetensors
from test_cli import conf  # noqa: F401  (a fixture)
from test_supervisor import Stop
from test_supervisor_audio_runtime import BASE, MODEL, arig, decisions, make_environment, started  # noqa: F401

pytestmark = pytest.mark.usefixtures("stub_tools")

BASE_ID = BASE.split("@")[0]
SUPPORTED = copy.deepcopy(CONFIG)
SUPPORTED["model"]["diffusion"] = copy.deepcopy(REAL_CONFIG["model"]["diffusion"])


def make_audio_snapshot(snap: Path, *, seed: float) -> None:
    """Add what stage 0's template reads (weights with a header, the text encoder directory, the card) to a snapshot that holds a config."""
    (snap / "model_config.json").write_text(json.dumps(SUPPORTED))
    write_safetensors(snap / "model.safetensors", TENSORS, seed)
    (snap / "t5gemma-b-b-ul2").mkdir(exist_ok=True)
    (snap / "t5gemma-b-b-ul2" / "config.json").write_text(json.dumps(TEXT_ENCODER_CONFIG))
    write_safetensors(snap / "t5gemma-b-b-ul2" / "model.safetensors", TEXT_TENSORS, 3.0)
    (snap / "README.md").write_text("---\nlicense: other\nlibrary_name: stable-audio-3\npipeline_tag: text-to-audio\n---\n\n# Audio\n")


def snapshots(rig) -> dict:
    return {k: Path(v.split("=", 1)[1]) for v in rig.args.input for k in ("model", "base") if v.startswith(k + "=")}


@pytest.fixture
def hrig(arig):
    """The audio rig with real audio snapshots and the two run_start inputs the command records. Stage 0 is then the harness body."""
    snaps = snapshots(arig)
    make_audio_snapshot(snaps["model"], seed=2.0)
    make_audio_snapshot(snaps["base"], seed=1.0)
    arig.args.input += [f"{audio_stage0.MODEL_INPUT}=1", f"{audio_stage0.BASE_INPUT}={BASE_ID}"]
    return arig


# ---- the stage table and the inputs ----------------------------------------------------------------------------------

def test_the_flag_gives_stage_0_a_harness_body_that_needs_no_coder():
    assert spec_for(0, None, audio_model=True) is AUDIO_STAGE_0
    assert AUDIO_STAGE_0.harness and AUDIO_STAGE_0.needs_coder is False and AUDIO_STAGE_0.number == 0
    assert AUDIO_STAGE_0.gate is STAGES[0].gate and AUDIO_STAGE_0.gate_file == "delta.json"


def test_without_the_flag_stage_0_is_the_agent_stage():
    assert spec_for(0, None) is STAGES[0] and STAGES[0].needs_coder and not STAGES[0].harness


def test_the_flag_changes_no_other_stage():
    for n in range(1, 9):
        assert spec_for(n, "weights-only", None, "weights-only", audio_model=True) is spec_for(n, "weights-only", None, "weights-only")


def test_the_stock_runtime_stage_0_keeps_its_place_before_the_audio_stage():
    assert spec_for(0, None, stock=True, audio_model=True) is stages.STOCK_STAGE_0


def start(inputs):
    return {"event": "run_start", "stage": None, "data": {"inputs": inputs}}


@pytest.mark.parametrize("inputs,expected", [({"audio_model": "1"}, True), ({}, False), ({"audio_model": "0"}, False), ({"audio_model": 1}, False),
                                             ({"audio_model": "true"}, False), ({"audio_model": ["1"]}, False)])
def test_only_the_exact_input_selects_the_stage(inputs, expected):
    assert audio_stage0.selected([start(inputs)]) is expected


def test_the_first_run_start_decides_and_a_later_entry_cannot_change_it():
    later = {"event": "run_start", "stage": None, "data": {"inputs": {"audio_model": "1"}}}
    assert audio_stage0.selected([start({}), later]) is False
    assert audio_stage0.selected([start({"audio_model": "1"}), start({})]) is True


@pytest.mark.parametrize("entries", [None, "x", [None], [{"event": "run_start"}], [{"event": "run_start", "data": None}],
                                     [{"event": "run_start", "data": {"inputs": "x"}}], [{"event": "decision", "data": {"inputs": {"audio_model": "1"}}}]])
def test_a_ledger_that_does_not_say_so_selects_nothing(entries):
    assert audio_stage0.selected(entries) is False


# ---- the run -----------------------------------------------------------------------------------------------------------

def test_stage_0_of_an_audio_run_boots_no_coder(hrig):
    """The wanted behaviour of issue 143 (the strict xfail of tests/test_audio_stage0_coder.py): zero coder starts through stage 0."""
    with pytest.raises(Stop):
        hrig.run()
    assert hrig.m.coder_starts == 0
    end = hrig.ends(0)[0]
    assert end["result"] == "pass" and end["facts"]["modality"] == "audio" and end["class"] == "weights-only"
    assert decisions(hrig, "pause") and not started(hrig, 1)


def test_the_template_in_the_stage_is_the_shipped_one_and_its_config_comes_from_the_inputs(hrig):
    with pytest.raises(Stop):
        hrig.run()
    stage = hrig.run_dir / "stages" / "0"
    assert (stage / "delta_triage.py").read_bytes() == Path(stages.SHIPPED_TEMPLATES["delta_triage.py"]).read_bytes()
    cfg = json.loads((stage / "triage_config.json").read_text())
    snaps = snapshots(hrig)
    assert cfg["model_snapshot"] == str(snaps["model"]) and cfg["nearest_snapshot"] == str(snaps["base"])
    assert cfg["nearest_model_id"] == BASE_ID and cfg["model_id"].startswith("stabilityai/stable-audio-3-small-music")
    assert Path(cfg["run_dir"]) == hrig.run_dir.resolve() or Path(cfg["run_dir"]) == hrig.run_dir


def test_the_ledger_records_the_triage_evidence_and_the_gate_judged_what_the_template_wrote(hrig):
    with pytest.raises(Stop):
        hrig.run()
    evidence = [e for e in hrig.entries() if e["event"] == "evidence" and e["stage"] == 0 and e["data"].get("what") == "audio triage"]
    assert len(evidence) == 1 and evidence[0]["data"]["path"].endswith("stages/0/delta.json") and len(evidence[0]["data"]["sha256"]) == 64
    assert not any(e["event"] == "agent_step" or e["event"] == "step_start" for e in hrig.entries() if e["stage"] == 0)


def test_a_complete_audio_run_never_starts_the_coder(hrig):
    make_environment(hrig)
    assert hrig.run() == EXIT_READY
    assert hrig.m.coder_starts == 0
    assert [hrig.ends(n)[0]["result"] for n in (0, 1, 2, 8)] == ["pass"] * 4


def test_a_resume_after_the_pause_starts_no_coder_either(hrig):
    with pytest.raises(Stop):
        hrig.run()
    assert hrig.m.coder_starts == 0
    make_environment(hrig)
    seen = []

    def resume():
        seen.append(1)
        if len(seen) == 1:
            Control(hrig.run_dir).write("resume")
        else:
            raise Stop("a second wait")
    hrig.on_sleep = resume
    assert hrig.run() == EXIT_READY
    assert hrig.m.coder_starts == 0 and len(hrig.ends(0)) == 1


def test_a_run_killed_while_stage_0_ran_resumes_with_no_coder_to_give_back(hrig):
    """Recovery has no coder lease to give back or adopt: no coder ever started."""
    make_environment(hrig)
    with pytest.raises(Crash):
        hrig.run(crash_if=lambda e: e["event"] == "stage_start" and e["stage"] == 0)
    assert hrig.m.coder_starts == 0 and not hrig.m.coder_running
    assert hrig.run(pid=101) == EXIT_READY
    assert hrig.m.coder_starts == 0 and hrig.ends(0)[-1]["result"] == "pass"


def test_an_operator_abort_in_the_pause_after_stage_0_leaves_no_coder_to_stop(hrig):
    hrig.on_sleep = lambda: Control(hrig.run_dir).write("abort")
    assert hrig.run() != EXIT_READY
    assert hrig.m.coder_starts == 0 and not hrig.m.coder_running and hrig.ends(0)[0]["result"] == "pass"


def test_a_run_started_without_the_inputs_still_boots_the_coder_for_stage_0(arig):
    """Every run from before this change, and a text run, keeps the agent stage 0 and its coder."""
    with pytest.raises(Stop):
        arig.run()
    assert arig.m.coder_starts == 1


# ---- the template is the judge --------------------------------------------------------------------------------------

def test_a_model_the_template_calls_a_text_model_fails_the_stage_and_goes_no_further(hrig):
    (snapshots(hrig)["model"] / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"], "hidden_size": 8}))
    with pytest.raises(Stop):
        hrig.run()
    end = hrig.ends(0)[0]
    assert end["result"] == "fail" and any("text model" in r for r in end["reasons"]) and not started(hrig, 1)
    assert hrig.m.coder_starts == 0 and "not escalated" in decisions(hrig, "pause")[-1]["reason"]


def test_a_missing_input_is_a_named_reason_and_runs_nothing(hrig):
    hrig.args.input = [v for v in hrig.args.input if not v.startswith(audio_stage0.BASE_INPUT + "=")]
    with pytest.raises(Stop):
        hrig.run()
    end = hrig.ends(0)[0]
    assert end["result"] == "fail" and any(audio_stage0.BASE_INPUT in r for r in end["reasons"])
    assert not (hrig.run_dir / "stages" / "0" / "delta_triage.py").exists() and hrig.m.coder_starts == 0


def test_a_template_that_exits_with_an_error_fails_the_stage_with_its_last_line(hrig):
    gone = snapshots(hrig)["base"].parent / "not-there"
    hrig.args.input = [f"base={gone}" if v.startswith("base=") else v for v in hrig.args.input]
    with pytest.raises(Stop):
        hrig.run()
    end = hrig.ends(0)[0]
    assert end["result"] == "fail" and any("delta_triage.py exited 2" in r and "not a directory" in r for r in end["reasons"])
    assert hrig.m.coder_starts == 0


def test_a_template_that_does_not_finish_in_time_fails_the_stage(hrig, monkeypatch):
    monkeypatch.setattr(audio_stage0, "TIMEOUT_S", 0.0001)
    with pytest.raises(Stop):
        hrig.run()
    assert any("did not finish" in r for r in hrig.ends(0)[0]["reasons"])


def test_the_gate_still_refuses_an_edited_template_copy(hrig, monkeypatch):
    """The gate compares the stage's copy with the shipped file. The body copies it fresh, so an edit between the copy and the gate is what is simulated here."""
    real = supervisor.Supervisor._check_gate

    def tamper(self, spec, stage_dir):
        if spec.number == 0:
            (stage_dir / "delta_triage.py").write_text("# edited\n")
        return real(self, spec, stage_dir)
    monkeypatch.setattr(supervisor.Supervisor, "_check_gate", tamper)
    with pytest.raises(Stop):
        hrig.run()
    assert hrig.ends(0)[0]["result"] == "fail" and any("differs from the shipped template" in r for r in hrig.ends(0)[0]["reasons"])


def test_the_template_runs_with_the_agent_environment_and_no_token(hrig, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secretsecretsecret")
    seen = []
    real = audio_stage0.run_template

    def spy(*a, **kw):
        seen.append(dict(kw.get("env") or {}))
        return real(*a, **kw)
    monkeypatch.setattr(audio_stage0, "run_template", spy)
    with pytest.raises(Stop):
        hrig.run()
    assert seen and all("HF_TOKEN" not in env and env.get("HOME", "").startswith(str(hrig.run_dir.resolve())) for env in seen)


# ---- the command records the inputs ------------------------------------------------------------------------------------

def test_the_real_audio_run_records_the_two_inputs(tmp_path, conf):
    from orchard import cli  # noqa: F401
    from test_audio_preflight import BASE as BASE_REPO, MODEL as MODEL_REPO, full_snaps, run, world
    snaps = full_snaps(tmp_path)
    code, out, rec = run(conf, ["bringup", MODEL_REPO], sig=world(local=snaps))
    assert code == 0 and f"{audio_stage0.MODEL_INPUT}=1" in rec.argv and f"{audio_stage0.BASE_INPUT}={BASE_REPO}" in rec.argv
    assert f"base={snaps[BASE_REPO]}" in rec.argv


def test_a_text_run_records_neither_input(conf):
    from test_audio_preflight import plain_signals, run
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=plain_signals(local_snapshot=lambda m: Path("/s")))
    assert not any(a.startswith(("audio_model=", "audio_base=")) for a in (rec.argv or []))


def inputs_for(base_data, base_snapshot, recorded=None, resuming=False):
    from orchard import cli
    return cli._audio_stage0_inputs(base_data, base_snapshot, recorded=recorded, resuming=resuming)


@pytest.mark.parametrize("base_data,snapshot", [({}, "/b"), ({"audio": False, "base": "o/b"}, "/b"), ({"audio": True}, "/b"), ({"audio": True, "base": ""}, "/b"),
                                                ({"audio": True, "base": "o/b"}, None), ({"audio": True, "base": 5}, "/b"), ({"audio": "yes", "base": "o/b"}, "/b")])
def test_nothing_is_recorded_unless_the_row_says_audio_and_the_base_is_known(base_data, snapshot):
    assert inputs_for(base_data, snapshot) == {}


def test_an_audio_row_with_its_base_records_both_inputs():
    assert inputs_for({"audio": True, "base": "o/b"}, "/b") == {"audio_model": "1", "audio_base": "o/b"}


def test_a_resume_keeps_what_the_run_started_with_whatever_the_preflight_says_now():
    row = {"audio": True, "base": "o/b"}
    assert inputs_for(row, "/b", recorded={}, resuming=True) == {}                          # started before this change: the agent stage 0 and its coder
    assert inputs_for({}, None, recorded={"audio_model": "1", "audio_base": "o/old"}, resuming=True) == {"audio_model": "1", "audio_base": "o/old"}
    assert inputs_for(row, "/b", recorded=None, resuming=False) == {"audio_model": "1", "audio_base": "o/b"}


@pytest.mark.parametrize("recorded", [{"audio_model": "0", "audio_base": "o/b"}, {"audio_model": 1, "audio_base": "o/b"}, {"audio_model": "1"},
                                      {"audio_model": "1", "audio_base": ""}, {"audio_model": "1", "audio_base": 7}, {"audio_base": "o/b"}])
def test_a_resume_keeps_only_a_well_formed_record(recorded):
    assert inputs_for({"audio": True, "base": "o/new"}, "/b", recorded=recorded, resuming=True) == {}
