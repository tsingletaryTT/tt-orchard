"""RunPaths: the machine paths a skill may name, and the rendering of their placeholders."""
from pathlib import Path

import pytest

import orchard
from orchard.context import build_messages
from orchard.paths import PLACEHOLDERS, RunPaths, UnknownPlaceholder, render
from orchard.stages import STAGES

CHECKOUT = Path(orchard.__file__).resolve().parent.parent


def paths(tmp_path) -> RunPaths:
    return RunPaths(orchard_dir="/o", hf_home="/hf", operator_home="/op", tt_model_root="/ttm",
                    cache_root="/cache")


def test_defaults_come_from_the_code_location_the_home_and_the_run_directory(tmp_path):
    p = RunPaths.resolve(tmp_path / "runs" / "r1", home=tmp_path / "op", environ={})
    assert p.orchard_dir == str(CHECKOUT)
    assert p.operator_home == str(tmp_path / "op")
    assert p.hf_home == str(tmp_path / "op" / ".cache" / "huggingface")
    assert p.tt_model_root == str(tmp_path / "op" / ".cache" / "tt-model" / "models")
    assert p.cache_root == str(tmp_path / "runs" / "cache")


def test_hf_home_defaults_to_the_environment_variable(tmp_path):
    p = RunPaths.resolve(tmp_path / "r", home=tmp_path, environ={"HF_HOME": str(tmp_path / "hf")})
    assert p.hf_home == str(tmp_path / "hf")


def test_explicit_values_win_and_are_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = RunPaths.resolve(tmp_path / "r", home=tmp_path, environ={"HF_HOME": "/env-hf"},
                         cache_root="caches", hf_home="hf", operator_home="me")
    assert p.cache_root == str(tmp_path / "caches")
    assert p.hf_home == str(tmp_path / "hf")
    assert p.operator_home == str(tmp_path / "me")
    # The tt-model store belongs to the operator, so it follows --operator-home.
    assert p.tt_model_root == str(tmp_path / "me" / ".cache" / "tt-model" / "models")


def test_a_record_round_trips(tmp_path):
    p = paths(tmp_path)
    assert RunPaths.from_record(p.record()) == p
    assert set(p.placeholders()) == set(PLACEHOLDERS)


def test_render_substitutes_every_known_placeholder(tmp_path):
    text = ("cp {{ORCHARD_DIR}}/orchard/x.py; {{HF_HOME}} {{OPERATOR_HOME}} {{TT_MODEL_ROOT}} "
            "{{CACHE_ROOT}}/<slug>/tt_cache")
    out = render(text, paths(tmp_path))
    assert out == "cp /o/orchard/x.py; /hf /op /ttm /cache/<slug>/tt_cache"


def test_render_refuses_an_unknown_placeholder_and_names_it(tmp_path):
    with pytest.raises(UnknownPlaceholder, match=r"\{\{ORCHARD_HOME\}\}.*weights-swap-check.md"):
        render("cp {{ORCHARD_HOME}}/x", paths(tmp_path), source="weights-swap-check.md")


def test_render_refuses_a_lower_case_or_spaced_placeholder(tmp_path):
    for text in ("{{orchard_dir}}", "{{ ORCHARD_DIR }}", "{{}}"):
        with pytest.raises(UnknownPlaceholder):
            render(text, paths(tmp_path))


def test_render_without_paths_refuses_any_placeholder_and_passes_plain_text():
    assert render("no placeholders here", None) == "no placeholders here"
    with pytest.raises(UnknownPlaceholder, match="no run paths"):
        render("{{HF_HOME}}", None)


def test_the_context_hands_the_agent_the_rendered_skill(tmp_path):
    run = tmp_path / "run"
    (run / "stages" / "2").mkdir(parents=True)
    skill = tmp_path / "weights-swap-check.md"
    skill.write_text("cp {{ORCHARD_DIR}}/orchard/skills/weights-swap-templates/prepare_swap.py\n"
                     "tt_cache under {{CACHE_ROOT}}/<slug>/tt_cache\n")
    system, _ = build_messages(spec=STAGES[2], phase="prepare", run_dir=run, stage_dir=run / "stages" / "2",
                               skill_path=skill, refs={}, facts={}, entries=[], resumed=False,
                               paths=paths(tmp_path))
    assert "cp /o/orchard/skills/weights-swap-templates/prepare_swap.py" in system
    assert "tt_cache under /cache/<slug>/tt_cache" in system
    assert "{{" not in system


def test_the_context_refuses_a_skill_with_an_unknown_placeholder(tmp_path):
    run = tmp_path / "run"
    (run / "stages" / "2").mkdir(parents=True)
    skill = tmp_path / "weights-swap-check.md"
    skill.write_text("cp {{ORCHARD}}/x\n")
    with pytest.raises(UnknownPlaceholder, match=r"\{\{ORCHARD\}\}"):
        build_messages(spec=STAGES[2], phase="prepare", run_dir=run, stage_dir=run / "stages" / "2",
                       skill_path=skill, refs={}, facts={}, entries=[], resumed=False,
                       paths=paths(tmp_path))
