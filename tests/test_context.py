"""The fresh context each agent step starts from."""
import json

from orchard.context import build_messages, facts_from
from orchard.stages import STAGES


def setup(tmp_path, skill="Compare the configs."):
    run = tmp_path / "run"
    (run / "stages" / "0").mkdir(parents=True)
    skill_path = tmp_path / "delta-triage.md"
    skill_path.write_text(skill)
    return run, skill_path


def msgs(run, skill_path, n=0, phase="run", entries=(), resumed=False, refs=None):
    return build_messages(spec=STAGES[n], phase=phase, run_dir=run, stage_dir=run / "stages" / str(n),
                          skill_path=skill_path, refs=refs or {}, entries=list(entries),
                          facts={"model": "Altworld/Hemmingway-1"}, resumed=resumed)


def test_a_first_step_holds_the_skill_the_rules_and_the_task(tmp_path):
    run, sp = setup(tmp_path)
    system, user = msgs(run, sp, refs={"model-bringup": None})
    assert "stage 0 (intake and delta triage)" in system and "Compare the configs." in system
    assert "Never publish" in system and "model-bringup: not installed" in system
    assert "Write delta.json in your stage directory" in user
    assert "- model: Altworld/Hemmingway-1" in user and "nothing yet" in user


def test_earlier_results_are_included_and_later_ones_are_not(tmp_path):
    run, sp = setup(tmp_path)
    (run / "stages" / "0" / "delta.json").write_text(json.dumps({"path": "weights-only"}))
    (run / "stages" / "2").mkdir(parents=True)
    (run / "stages" / "2" / "result.json").write_text('{"pcc": 0.999}')
    _, user = msgs(run, sp, n=1)
    assert "### stages/0/delta.json (sha256 " in user and '"weights-only"' in user
    assert "0.999" not in user


def test_the_finish_step_sees_the_hardware_test_record(tmp_path):
    run, sp = setup(tmp_path)
    sd = run / "stages" / "2"
    sd.mkdir(parents=True)
    (sd / "test-result.json").write_text('{"returncode": 3}')
    _, user = msgs(run, sp, n=2, phase="finish")
    assert "## stages/2/test-result.json" in user and '"returncode": 3' in user
    assert "Write result.json from that evidence" in user


def test_resume_notes_are_given_only_to_a_resumed_step(tmp_path):
    run, sp = setup(tmp_path)
    (run / "stages" / "0" / "RESUME.md").write_text("tensors compared; tokenizer next")
    assert "tokenizer next" in msgs(run, sp, resumed=True)[1]
    assert "tokenizer next" not in msgs(run, sp, resumed=False)[1]


def test_the_ledger_excerpt_says_why_an_attempt_failed(tmp_path):
    run, sp = setup(tmp_path)
    entries = [{"event": "stage_end", "stage": 0, "data": {"result": "escalate",
                                                         "reasons": ["delta.json is missing"]}}]
    assert "- stage 0: escalate (delta.json is missing)" in msgs(run, sp, entries=entries)[1]


def test_a_long_skill_is_cut_and_says_so(tmp_path):
    run, sp = setup(tmp_path, skill="x" * 50000)
    assert "[cut at 40000 characters]" in msgs(run, sp)[0]


def test_facts_list_the_model_and_the_inputs(tmp_path):
    facts = facts_from({"model": "m", "inputs": {"base": "/mnt/base"}}, tmp_path)
    assert facts == {"model": "m", "run directory": str(tmp_path), "input base": "/mnt/base"}


def test_facts_list_the_required_chip_configurations_when_the_run_names_them(tmp_path):
    facts = facts_from({"model": "m", "required_chips": [2, 4]}, tmp_path)
    assert facts["required chip configurations"] == "2, 4"
    assert "required chip configurations" not in facts_from({"model": "m"}, tmp_path)


def test_stage_4_is_told_which_configurations_are_required_and_to_record_optional_failures(tmp_path):
    run, sp = setup(tmp_path)
    facts = {"model": "m", "required chip configurations": "2, 4"}
    _, user = build_messages(spec=STAGES[4], phase="run", run_dir=run, stage_dir=run / "stages" / "4",
                             skill_path=sp, refs={}, entries=[], facts=facts, resumed=False)
    assert "Required chip counts for this run: 2, 4." in user
    assert "pass false and a reason" in user
    _, other = build_messages(spec=STAGES[3], phase="run", run_dir=run, stage_dir=run / "stages" / "3",
                              skill_path=sp, refs={}, entries=[], facts=facts, resumed=False)
    assert "Required chip counts" not in other
    _, plain = build_messages(spec=STAGES[4], phase="run", run_dir=run, stage_dir=run / "stages" / "4",
                              skill_path=sp, refs={}, entries=[], facts={"model": "m"}, resumed=False)
    assert "Required chip counts" not in plain
