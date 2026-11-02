# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 381, Copilot's first review: each finding as a test. The fakes are those of tests/test_stack_pins.py."""
from __future__ import annotations

import importlib.machinery
import json
import sys
import types

import pytest

from orchard import skillsfresh, stack_pins as sp
from test_stack_pins import GOOD, QWEN4, build, config, full, names, probe_says, py, repo  # noqa: F401  (py is a fixture)


def test_credentials_in_an_origin_or_an_error_line_never_reach_the_record(tmp_path, py):
    bad = repo(tmp_path, "tt-metal", origin="https://user:s3cretpass@evil.example/chips/tt-metal.git")
    down = repo(tmp_path, "vllm", down=True)
    src, fake = full(tmp_path, py, metal=bad, vllm=down)

    def leaky(argv, t, e=None):
        if "ls-remote" in argv and argv[2] == down[0]:
            return 128, "", "fatal: unable to access 'https://bob:hunter2tok@github.com/x/y.git/': hf_abcdefgh12345678"
        return fake(argv, t, e)
    text = json.dumps(build(src, leaky, GOOD()))
    for secret in ("s3cretpass", "evil.example", "hunter2tok", "hf_abcdefgh12345678", "user:"):
        assert secret not in text
    rec = json.loads(text)
    assert rec["parts"]["tt_metal"]["state"] == skillsfresh.UNKNOWN and "not on github.com" in rec["parts"]["tt_metal"]["detail"]


@pytest.mark.parametrize("installed,minimum,below", [
    ("5.19.0.dev0", "5.19.0rc1", True), ("5.19.0a1", "5.19.0b1", True), ("5.19.0b2", "5.19.0rc1", True), ("5.19.0rc1", "5.19.0rc2", True),
    ("5.19.0rc2", "5.19.0rc1", False), ("5.19.0", "5.19.0rc1", False), ("5.19.0.post1", "5.19.0", False), ("5.19.0", "5.19.0.post1", True),
    ("5.19.0.post2", "5.19.0.post1", False), ("5.19.0.dev1", "5.19.0.dev0", False), ("5.19.0.dev0", "5.19.0", True), ("5.19.0a1", "5.19.0.dev9", False)])
def test_suffix_kinds_and_numbers_sort_in_pep_440_order(installed, minimum, below):
    assert sp._below(sp.parse_version(installed), sp.parse_version(minimum)) is below


@pytest.mark.parametrize("text", ["5.19.0-custom", "5.19.0.dev1.post2", "5", "x.y", "5.19.0.1.2", "", None, 5])
def test_a_version_form_the_check_does_not_know_is_not_compared(text):
    assert sp.parse_version(text) is None


def test_an_unreadable_version_does_not_hide_a_missing_class(tmp_path, py):
    src, fake = full(tmp_path, py, model=config(tmp_path, **QWEN4))
    rec = build(src, fake, probe_says("5.19.0-custom", {"Qwen4ExpForCausalLM": False}))
    assert names(rec["hazards"]) == [sp.H_CLASS] and "transformers_unchecked" in names(rec["advisories"]) and rec["severity"] == "critical"


def test_a_skills_clone_whose_origin_is_not_tenstorrent_skills_is_not_compared(tmp_path, py):
    fork = repo(tmp_path, "skills", behind=65, days=42, origin="https://github.com/someone/skills-fork.git")
    other = repo(tmp_path, "skills-two", behind=65, days=42, origin="https://github.com/tenstorrent/tt-metal.git")
    for given in (fork, other):
        src, fake = full(tmp_path, py, skills=given)
        rec = build(src, fake, GOOD())
        assert rec["hazards"] == [] and names(rec["advisories"]) == [sp.H_SKILLS + "_unchecked"]
        assert "tenstorrent/skills" in rec["parts"]["skills"][0]["detail"]


def test_the_real_skills_upstream_still_blocks(tmp_path, py):
    src, fake = full(tmp_path, py, skills=repo(tmp_path, "skills", behind=65, days=42))
    assert names(build(src, fake, GOOD())["hazards"]) == [sp.H_SKILLS]


def test_the_vllm_hint_names_the_branch_it_was_compared_with(tmp_path, py):
    src, fake = full(tmp_path, py, vllm=repo(tmp_path, "vllm", behind=65, days=42, branch="feature-x"))
    src = sp.Sources(**{**src.__dict__, "vllm_branch": "dev"})
    reason = build(src, fake, GOOD())["hazards"][0]["reason"]
    assert "switch dev" in reason and "pull --ff-only origin dev" in reason and "origin main" not in reason


def test_the_vllm_hint_on_the_compared_branch_is_a_fast_forward_of_it(tmp_path, py):
    src, fake = full(tmp_path, py, vllm=repo(tmp_path, "vllm", behind=65, days=42, branch="dev"))
    src = sp.Sources(**{**src.__dict__, "vllm_branch": "dev"})
    reason = build(src, fake, GOOD())["hazards"][0]["reason"]
    assert "pull --ff-only origin dev" in reason and "switch" not in reason


def test_the_class_lookup_includes_the_model_module_the_reference_loader_searches(monkeypatch, capsys):
    top = types.ModuleType("transformers")
    top.__spec__ = importlib.machinery.ModuleSpec("transformers", None)
    top.__version__ = "5.0.0"
    mod = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
    mod.Qwen3_5Model = object
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5.modeling_qwen3_5", mod)
    monkeypatch.setitem(sys.modules, "transformers", top)
    monkeypatch.setattr(sys, "argv", ["-c", "qwen3_5", "Qwen3_5Model", "Missing"])
    exec(compile(sp.PROBE_SCRIPT, "<probe>", "exec"), {})
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["classes"] == {"Qwen3_5Model": True, "Missing": False}


def test_the_probe_passes_the_model_type_and_only_a_plain_one(py):
    seen = []
    answer = (0, json.dumps({"importable": True, "version": "5.0.0", "classes": {}}), "")
    run = lambda argv, t, e=None: (seen.append(argv), answer)[1]   # noqa: E731
    for mt in ("qwen3_5", "a b", "../x", None, 5, "x" * 300):
        sp.probe_interpreter(py, [], run=run, model_type=mt)
    assert [a[3] for a in seen] == ["qwen3_5", "-", "-", "-", "-", "-"]


def test_the_probe_runs_in_the_reference_import_environment(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/some/site")
    monkeypatch.setenv("VIRTUAL_ENV", "/some/venv")
    env = sp._probe_env()
    assert env["PYTHONPATH"] == "/some/site" and env["VIRTUAL_ENV"] == "/some/venv"
    seen = []
    sp.probe_interpreter(sys.executable, [], run=lambda argv, t, e=None: (seen.append(argv), (1, "", ""))[1])
    assert "-I" not in seen[0] and "-E" not in seen[0]


def test_the_probe_script_takes_the_current_directory_off_the_path(monkeypatch, capsys):
    monkeypatch.setattr(sys, "path", ["", ".", "/keep"])
    monkeypatch.setattr(sys, "argv", ["-c", "-"])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    exec(compile(sp.PROBE_SCRIPT, "<probe>", "exec"), {})
    assert sys.path == ["/keep"]
