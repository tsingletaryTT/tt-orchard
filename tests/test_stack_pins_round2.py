# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 381, Copilot's second review round, and the coordinator's leak check: each finding as a test. Fakes are those of tests/test_stack_pins.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import json
import os
import sys

import pytest

from orchard import skillsfresh, stack_pins as sp
from test_stack_pins import GOOD, QWEN4, build, config, full, names, probe_says, py, repo  # noqa: F401  (py is a fixture)


def real_runner():
    """The module's original process runner, which the suite's guard replaces in `skillsfresh`."""
    spec = importlib.util.spec_from_file_location("skillsfresh_real2", skillsfresh.__file__)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["skillsfresh_real2"] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("skillsfresh_real2", None)
    return mod._raw_run


# ---- (1) a file in the invoking directory cannot answer for transformers ---------------------------------------------

def test_a_transformers_py_in_the_invoking_directory_is_not_run_with_pythonpath_dot(tmp_path, monkeypatch):
    (tmp_path / "transformers.py").write_text('__version__ = "99.0.0"\nclass PlantedForCausalLM: pass\nopen("ran-planted-code", "w").write("x")\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", ".")
    monkeypatch.setattr(skillsfresh, "_raw_run", real_runner())
    out = sp.probe_interpreter(sys.executable, ["PlantedForCausalLM"], env=dict(os.environ))
    assert not (tmp_path / "ran-planted-code").exists()
    assert out["version"] != "99.0.0" and out["classes"].get("PlantedForCausalLM", False) is False


def test_the_script_itself_drops_the_current_directory_even_when_the_runner_and_the_environment_do_not(tmp_path):
    """The layers are independent: here the child starts in the planted directory with PYTHONPATH=. and nothing else filtered."""
    (tmp_path / "transformers.py").write_text('__version__ = "99.0.0"\nopen("ran-planted-code", "w").write("x")\n')
    raw = real_runner()
    out = sp.probe_interpreter(sys.executable, [], run=lambda argv, t, e=None: raw(argv, t, {**e, "PYTHONPATH": "."}, cwd=str(tmp_path)))
    assert not (tmp_path / "ran-planted-code").exists() and out["version"] != "99.0.0"


def test_the_process_runner_starts_the_child_in_the_directory_it_is_given(tmp_path):
    raw = real_runner()
    rc, out, _ = raw([sys.executable, "-c", "import os; print(os.getcwd())"], 30, dict(os.environ), cwd=str(tmp_path))
    assert rc == 0 and os.path.realpath(out.strip()) == os.path.realpath(tmp_path)


def test_the_probe_child_starts_in_an_empty_temporary_directory(monkeypatch):
    seen = {}

    def fake_raw(argv, timeout, env=None, cwd=None, max_bytes=None):
        seen["cwd"], seen["listing"] = cwd, os.listdir(cwd)
        return 1, "", ""
    monkeypatch.setattr(skillsfresh, "_raw_run", fake_raw)
    sp.real_probe_run([sys.executable, "-c", "pass"], 5, {})
    assert seen["listing"] == [] and seen["cwd"] and not os.path.exists(seen["cwd"])         # empty while it ran, removed after


def test_pythonpath_entries_that_are_relative_the_current_directory_or_under_the_run_directory_are_dropped(tmp_path, monkeypatch):
    run_dir, ok = tmp_path / "run", tmp_path / "site"
    (run_dir / "lib").mkdir(parents=True)
    ok.mkdir()
    monkeypatch.chdir(tmp_path)
    base = {"PATH": "/usr/bin", "PYTHONPATH": os.pathsep.join(["", ".", "rel", str(tmp_path), str(run_dir / "lib"), str(ok)])}
    assert sp._probe_env(base, exclude=(str(run_dir),))["PYTHONPATH"] == str(ok)
    assert "PYTHONPATH" not in sp._probe_env({"PYTHONPATH": "."})


# ---- (2) every home prefix is redacted in the record, the hint and the block reason ------------------------------------

def test_the_operator_home_another_users_home_and_the_invoking_home_are_redacted_everywhere(tmp_path, py, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "invoker"))
    ops = tmp_path / "ops-home"
    other = "/home/someoneelse/tt-metal"
    mine = str(tmp_path / "invoker" / "vllm")
    ops.mkdir()
    metal = repo(tmp_path, "ops-home/tt-metal", behind=65, days=42)
    vllm = repo(tmp_path, "vllm", behind=65, days=42)
    src, fake = full(tmp_path, py, metal=metal, vllm=vllm, skills=repo(tmp_path, "skills", behind=65, days=42))
    src = sp.Sources(**{**src.__dict__, "homes": (str(ops),)})
    rec = sp.build_record(src, new_run=True, run=fake, probe_run=GOOD(), now=1.0e9,
                                                                 cache=skillsfresh.Cache(tmp_path / "none.json", write=False))
    text = json.dumps(rec)
    assert str(ops) not in text and "<home>" in text
    assert sp.redact(f"git -C {other} pull; also {mine}", (str(ops),)).count("<home>") >= 1 and "someoneelse" not in sp.redact(other)
    assert "/root" not in sp.redact("cd /root/x") and sp.redact("cd /root/x") == "cd <home>/x"
    block_text = "; ".join(f"{h['name']}: {h['reason']}" for h in rec["hazards"])
    assert str(ops) not in block_text and "<home>/tt-metal" in block_text           # the hint still names the checkout, under the placeholder


# ---- (4) GitHub names are case-insensitive ----------------------------------------------------------------------------

@pytest.mark.parametrize("origin", ["https://github.com/Tenstorrent/Skills.git", "git@github.com:TENSTORRENT/skills", "https://github.com/tenstorrent/SKILLS"])
def test_a_differently_cased_skills_origin_is_the_skills_upstream(tmp_path, py, origin):
    src, fake = full(tmp_path, py, skills=repo(tmp_path, "skills", behind=65, days=42, origin=origin))
    assert names(build(src, fake, GOOD())["hazards"]) == [sp.H_SKILLS]


# ---- (5) a stated version the check cannot read is a finding, and a table minimum does not mask it ---------------------

@pytest.mark.parametrize("stated", ["5.19.0rc1.dev0", "5.19.0-custom", "five"])
def test_an_unreadable_stated_version_is_an_advisory_and_the_table_minimum_still_applies(tmp_path, py, stated):
    model = config(tmp_path, model_type="qwen4_exp", architectures=["Qwen4ExpForCausalLM"], transformers_version=stated)
    src, fake = full(tmp_path, py, model=model)
    low = build(src, fake, probe_says("5.0.0", {"Qwen4ExpForCausalLM": True}))
    assert "transformers_minimum_unreadable" in names(low["advisories"]) and names(low["hazards"]) == [sp.H_BELOW]
    good = build(src, fake, probe_says("5.19.0", {"Qwen4ExpForCausalLM": True}))
    assert good["severity"] == "advisory" and "transformers_minimum_unreadable" in names(good["advisories"])      # not ok: the stated minimum was not checked


def test_an_unreadable_stated_version_with_no_table_entry_is_not_ok_even_when_the_class_is_present(tmp_path, py):
    model = config(tmp_path, model_type="llama", architectures=["LlamaForCausalLM"], transformers_version="5.19.0rc1.dev0")
    src, fake = full(tmp_path, py, model=model)
    rec = build(src, fake, probe_says("4.57.6", {"LlamaForCausalLM": True}))
    assert rec["severity"] == "advisory" and "transformers_minimum_unreadable" in names(rec["advisories"])


# ---- (6) update commands quote the path ------------------------------------------------------------------------------

def test_update_commands_quote_a_path_with_a_space_and_a_shell_character(tmp_path, py):
    import shlex
    awkward = "TT builds; echo x $(id)"
    metal = repo(tmp_path, awkward + "-metal", behind=65, days=42, origin="https://github.com/tenstorrent/tt-metal.git")
    vllm = repo(tmp_path, awkward + "-vllm", behind=65, days=42, origin="https://github.com/tenstorrent/vllm.git")
    skills = repo(tmp_path, "skills", behind=65, days=42)
    src, fake = full(tmp_path, py, metal=metal, vllm=vllm, skills=skills)
    rec = build(src, fake, GOOD())
    reasons = {h["name"]: h["reason"] for h in rec["hazards"]}
    for name in (sp.H_TT_METAL, sp.H_VLLM):
        piece = reasons[name].split("git -C ", 1)[1].split(" pull", 1)[0]
        assert shlex.split(piece) == [str(metal[0] if name == sp.H_TT_METAL else vllm[0])]       # one word: the whole path, however awkward
    assert shlex.split(skillsfresh.skills_update("/a b/c;d", "main"))[2] == "/a b/c;d"


# ---- git errors with credentials ---------------------------------------------------------------------------------------

LEAKS = ("realtoken123", "hf_abcdefgh12345678", "ghp_abcdefgh12345678", "s3cretpass")


@pytest.mark.parametrize("where", ["ls_remote", "compare", "rev_parse", "remote_url"])
def test_a_git_error_with_a_credential_never_reaches_the_record_a_reason_or_a_hint(tmp_path, py, where):
    m = repo(tmp_path, "tt-metal", behind=65, days=42)
    src, fake = full(tmp_path, py, metal=m)
    err = ("fatal: unable to access 'https://user:realtoken123@github.com/tenstorrent/tt-metal.git/': "
           "hf_abcdefgh12345678 ghp_abcdefgh12345678 and https://bob:s3cretpass@example.org/x")

    def leaky(argv, t, e=None):
        hit = {"ls_remote": "ls-remote" in argv, "compare": argv[0] == "gh" and "/compare/" in argv[2], "rev_parse": argv[3:5] == ["rev-parse", "HEAD"],
               "remote_url": argv[3:5] == ["remote", "get-url"]}[where]
        if hit and argv[2:3] != ["x"]:
            if where == "remote_url":
                return 0, "https://user:realtoken123@evil.example/x.git\n", ""
            return 128, "", err
        return fake(argv, t, e)
    rec = build(src, leaky, GOOD())
    text = json.dumps(rec)
    for secret in LEAKS:
        assert secret not in text
    block_text = "; ".join(f"{h['name']}: {h['reason']}" for h in rec["hazards"] + rec["advisories"])
    assert not any(s in block_text for s in LEAKS)


def test_an_exception_message_with_a_credential_is_scrubbed_before_it_reaches_a_ledger_decision():
    text = sp.redact("OSError: cannot write https://u:realtoken123@h.example/x hf_abcdefgh12345678")
    assert "realtoken123" not in text and "hf_abcdefgh12345678" not in text
