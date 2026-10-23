# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 7 writes the use sections of the card from the model's own files, and the gate reads the same files again."""
import json
import os
import sys
from pathlib import Path

import pytest

import package_fakes
from card_use_fixtures import README_WITH_USE, TEMPLATE
from package_fakes import SOURCE_ENV, SOURCE_EXTRA, fake_bin, fake_boot_result, make_run, make_source, write
from orchard.package import finish, stage_all
from orchard.stages import gate_package

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")
FLAGS = "--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning_parser qwen3"


def build(tmp_path, monkeypatch, *, readme=README_WITH_USE, template=TEMPLATE, flags=FLAGS, skip_vision=True,
          tool_calls=None):
    monkeypatch.setattr(package_fakes, "SOURCE_EXTRA", f"{SOURCE_EXTRA} {flags}".strip())
    models = tmp_path / "models"
    env = {**SOURCE_ENV, **({"QWEN36_SKIP_VISION": "1"} if skip_vision else {})}
    r = make_run(tmp_path, make_source(models, env=env))
    snap = r["snapshot"]
    for name, text in (("README.md", readme), ("chat_template.jinja", template)):
        if text is not None:
            if os.path.lexists(snap / name):
                (snap / name).unlink()
            (snap / name).write_text(text)
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    run_env = dict(os.environ, FAKE_TT_MODEL_PYTHON=sys.executable, FAKE_SWAP_SERVER=str(FAKE_SERVER),
                   FAKE_SWAP_CONFIG=str(tmp_path / "server.json"))
    stage = r["run"] / "stages/7"
    stage.mkdir(parents=True)
    if tool_calls is not None:
        write(stage / "benchmark/speed.json", {"prompts": {"tool-call": {"runs": [{"tool_calls": tool_calls}]}}})
    stage_all(r["run"], stage, namespace="episod", models_root=models, env=run_env)
    fake_boot_result(stage)
    finish(r["run"], stage)
    return {**r, "stage": stage, "pkg": stage / "package/hemmingway-1-p300"}


def reasons(s):
    return list(gate_package(s["stage"], s["run"]).reasons)


@pytest.fixture
def staged(tmp_path, monkeypatch, stub_tools):
    return build(tmp_path, monkeypatch)


def test_the_staged_card_says_what_the_model_is_for_and_how_to_call_it(staged):
    card = (staged["pkg"] / "README.md").read_text()
    assert "## How to call it" in card and "Text generation with" not in card
    assert "> AREX-like models are intended for long-horizon agents and tool-augmented research." in card
    assert "Thinking is on by default" in card and "`reasoning_content`" in card and "`tool_calls`" in card
    assert "Text only" in card and "`xhigh`, `medium`, `low`" in card
    assert gate_package(staged["stage"], staged["run"]).ok


def test_a_package_whose_sources_give_no_facts_keeps_the_generic_card(tmp_path, monkeypatch, stub_tools):
    s = build(tmp_path, monkeypatch, readme=None, template="{{ messages }}", flags="", skip_vision=False)
    card = (s["pkg"] / "README.md").read_text()
    assert "Text generation with" in card and "## How to call it" not in card
    assert gate_package(s["stage"], s["run"]).ok


def test_a_tool_call_in_the_runs_speed_file_is_cited_and_shipped_with_the_package(tmp_path, monkeypatch, stub_tools):
    s = build(tmp_path, monkeypatch, tool_calls=[{"name": "run_tests", "arguments": "{}"}])
    card = (s["pkg"] / "README.md").read_text()
    assert "the tool-use prompts returned tool calls (`run_tests`)" in card
    shipped = s["pkg"] / "evidence/stages/7/benchmark/speed.json"
    assert shipped.is_file() and gate_package(s["stage"], s["run"]).ok
    shipped.unlink()
    assert any("missing at evidence/stages/7/benchmark/speed.json" in r for r in reasons(s))


def test_the_gate_refuses_a_card_edited_to_claim_images(staged):
    card = staged["pkg"] / "README.md"
    card.write_text(card.read_text().replace("## How to call it", "The package accepts images.\n\n## How to call it"))
    assert any("claims image or video input" in r for r in reasons(staged))


def test_the_gate_reads_run_sh_again_and_refuses_text_only_when_the_vision_flag_is_gone(staged):
    run_sh = staged["pkg"] / "run.sh"
    run_sh.write_text(run_sh.read_text().replace('export QWEN36_SKIP_VISION="1"\n', ""))
    assert any("says the package is text only" in r for r in reasons(staged))


def test_the_gate_reads_the_template_again_and_refuses_a_wrong_thinking_default(staged):
    t = staged["snapshot"] / "chat_template.jinja"
    t.write_text("{%- if enable_thinking is defined and enable_thinking is true %}x{%- endif %}")
    assert any("says thinking is on by default" in r for r in reasons(staged))


def test_the_gate_refuses_a_card_that_lost_its_how_to_call_it_section(staged):
    card = staged["pkg"] / "README.md"
    text = card.read_text()
    head, rest = text.split("## How to call it", 1)
    card.write_text(head + "## Expected performance" + rest.split("## Expected performance", 1)[1])
    assert any("How to call it" in r for r in reasons(staged))
