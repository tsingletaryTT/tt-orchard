# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 381, Copilot's third review: redact before cutting, no user-site startup files in the probe, repository identity for every checkout,
home words before quoting, every architecture checked, and blocking as an opt-in. Fakes are those of tests/test_stack_pins.py."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
import sys

import pytest

from orchard import narrate, skillsfresh, stack_pins as sp
from test_stack_pins import GOOD, QWEN4, build, config, full, names, probe_says, py, repo  # noqa: F401  (py is a fixture)
from test_stack_pins_round2 import real_runner


# ---- (1) redact the whole text, then cut ------------------------------------------------------------------------------

def _cut_at_the_at_sign(prefix: str, secret: str, length: int = 200) -> str:
    """A line in which the `@` after `https://user:<secret>` is character number `length` (counting from 0): a cut at `length` ends between the
    password and the `@`, which is where the old order (cut, then redact) left the password in view."""
    head = "https://user:" + secret
    filler = "x" * (length - len(prefix) - len(head))
    assert filler.isidentifier() or filler == "" or filler.isalnum()
    return prefix + filler + head + "@github.com/tenstorrent/tt-metal.git"


def test_a_git_error_cut_between_a_password_and_its_at_sign_leaves_no_password(tmp_path, py):
    err = _cut_at_the_at_sign("fatal: ", "PASSWORDSECRET99")
    assert err.index("@") == 200
    m = repo(tmp_path, "tt-metal", down=True)
    src, fake = full(tmp_path, py, metal=m)
    fake_err = lambda argv, t, e=None: (128, "", err) if "ls-remote" in argv and argv[2] == m[0] else fake(argv, t, e)       # noqa: E731
    text = json.dumps(build(src, fake_err, GOOD()))
    assert "PASSWORDSECRET99" not in text and "user:" not in text
    assert "tt-metal" in text                                       # the finding is still there, the secret is not


def _ledger_text_and_reason_of_a_failed_write(rig, message: str) -> tuple[str, str]:
    """Run the stack-pins step with a write that fails with `OSError(message)`. Returns the whole ledger text and the reason the ledger
    gave for `stack pins not written` (the rest of that line)."""
    from test_supervisor_stack_pins import run
    from orchard.supervisor import Control

    def broken(stage_dir, new_run):
        raise OSError(message)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    run(rig, broken)
    text = (rig.run_dir / "ledger.jsonl").read_text()
    assert "stack pins not written" in text
    return text, text.split("stack pins not written")[1].split("\n")[0]


def test_the_ledger_reason_of_a_failed_write_is_redacted_before_it_is_cut(rig_for_round3):
    """Issue 426: the `@` has to be character 200 of the text the supervisor builds, so that a cut before the redaction ends between the
    password and the `@`. That position is a property of this string. It used to be built from the run's temp directory, so a long
    `--basetemp` made the padding negative and the `@` land beyond 200. The prefix is a fixed path of its own now."""
    err = _cut_at_the_at_sign("OSError: /srv/pins/cache ", "LEDGERPASSWORD7")
    assert err.index("@") == 200
    text, _reason = _ledger_text_and_reason_of_a_failed_write(rig_for_round3, err[len("OSError: "):])
    assert "LEDGERPASSWORD7" not in text


def test_the_ledger_reason_of_a_failed_write_hides_the_operator_home(rig_for_round3):
    """The message starts with the run's operator home, so the first 200 characters of an unredacted text are the home path itself
    however long the temp directory is. The check names the start of the path and its last name, and does not depend on a cut
    position. The part after the path has to survive: the reason is redacted, not dropped."""
    rig = rig_for_round3
    home = str(rig.home)
    _text, reason = _ledger_text_and_reason_of_a_failed_write(rig, f"{home}/cache cannot be written")
    assert home[:40] not in reason and rig.home.name not in reason
    assert "cannot be written" in reason


@pytest.fixture
def rig_for_round3(tmp_path):
    from test_supervisor import Rig
    r = Rig(tmp_path)
    yield r
    r.close()


# ---- (2) no user-site startup file runs with the probe -----------------------------------------------------------------

def test_the_probe_environment_turns_the_user_site_off():
    assert sp._probe_env({"PATH": "/usr/bin", "PYTHONNOUSERSITE": ""})["PYTHONNOUSERSITE"] == "1"
    assert sp._probe_env()["PYTHONNOUSERSITE"] == "1"


def test_a_pth_file_in_the_user_site_of_an_agent_writable_home_does_not_run_in_the_probe(tmp_path, monkeypatch):
    base = getattr(sys, "_base_executable", sys.executable)
    home = tmp_path / "run-home"
    home.mkdir()
    raw = real_runner()
    where = subprocess.run([base, "-c", "import site; print(site.getusersitepackages())"], capture_output=True, text=True,
                           env={"HOME": str(home), "PATH": os.environ.get("PATH", "")}).stdout.strip()
    probe = subprocess.run([base, "-c", "import site; print(site.ENABLE_USER_SITE)"], capture_output=True, text=True,
                           env={"HOME": str(home), "PATH": os.environ.get("PATH", "")}).stdout.strip()
    if not where or probe != "True":
        pytest.skip("this interpreter does not enable a user site for a fresh HOME, so a planted .pth file cannot be shown to run")
    os.makedirs(where)
    marker = tmp_path / "pth-ran"
    with open(os.path.join(where, "evil.pth"), "w") as f:         # hardening: ok a test file planted for the probe to ignore
        f.write(f"import os; open({str(marker)!r}, 'w').write('x')\n")
    monkeypatch.setattr(skillsfresh, "_raw_run", raw)
    out = sp.probe_interpreter(base, [], env={"PATH": os.environ.get("PATH", ""), "HOME": str(home)})
    assert out["state"] == "answered" and not marker.exists()
    # the planted file is live: the same interpreter without the setting runs it
    subprocess.run([base, "-c", "pass"], env={"HOME": str(home), "PATH": os.environ.get("PATH", "")}, capture_output=True)
    assert marker.exists()


def test_every_record_lists_what_the_probe_cannot_match():
    rec = sp.build_record(sp.Sources(), new_run=True, run=lambda *a, **k: (1, "", ""), now=1.0)
    assert rec["probe_limits"] == list(sp.PROBE_LIMITS) and len(rec["probe_limits"]) >= 4
    assert any("PYTHONNOUSERSITE" in s for s in rec["probe_limits"]) and any("reference_python" in s for s in rec["probe_limits"])
    assert sp.unchecked_record("x", new_run=True)["probe_limits"] == list(sp.PROBE_LIMITS)


# ---- (3) every checkout must be the project it is compared with -------------------------------------------------------

@pytest.mark.parametrize("origin", ["https://github.com/someone/tt-metal.git", "https://github.com/tenstorrent/vllm.git", "git@github.com:other/x"])
def test_a_tt_metal_checkout_with_another_origin_is_not_compared(tmp_path, py, origin):
    src, fake = full(tmp_path, py, metal=repo(tmp_path, "tt-metal", behind=65, days=42, origin=origin))
    rec = build(src, fake, GOOD())
    assert sp.H_TT_METAL not in names(rec["hazards"]) and names(rec["advisories"]) == [sp.H_TT_METAL + "_unchecked"]
    assert rec["parts"]["tt_metal"]["state"] == skillsfresh.UNKNOWN and "tenstorrent/tt-metal" in rec["parts"]["tt_metal"]["detail"]


def test_the_expected_names_are_case_insensitive_and_the_plugin_default_is_named(tmp_path, py):
    src, fake = full(tmp_path, py, metal=repo(tmp_path, "tt-metal", behind=65, days=42, origin="https://github.com/Tenstorrent/TT-Metal.git"))
    assert names(build(src, fake, GOOD())["hazards"]) == [sp.H_TT_METAL]
    assert sp.Sources().metal_repo == "tenstorrent/tt-metal" and sp.Sources().vllm_repo == "tenstorrent/vllm-tt-plugin"


def test_the_plugin_origin_is_checked_against_its_configured_repository(tmp_path, py):
    vllm = repo(tmp_path, "vllm", behind=65, days=42, origin="https://github.com/tenstorrent/vllm-tt-plugin.git")
    src, fake = full(tmp_path, py, vllm=vllm)
    assert "tt_vllm_plugin_stale" not in names(build(src, fake, GOOD())["hazards"])             # full() expects tenstorrent/vllm
    src = sp.Sources(**{**src.__dict__, "vllm_repo": "Tenstorrent/VLLM-TT-Plugin"})
    assert names(build(src, fake, GOOD())["hazards"]) == [sp.H_VLLM]


# ---- (4) a home prefix is replaced before the path is quoted ------------------------------------------------------------

def test_the_hint_for_a_path_under_the_invoking_home_with_a_space_is_a_valid_command(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "me"))
    hint = sp._branch_update("main")(str(tmp_path / "me" / "TT builds; x"), "main")
    word = hint.split("git -C ", 1)[1].split(" pull", 1)[0]
    assert word == "~/'TT builds; x'" and shlex.split(word) == ["~/TT builds; x"]            # the shell expands the unquoted ~


def test_the_hint_for_another_home_says_what_to_replace(tmp_path):
    hint = sp._branch_update("main", homes=("/srv/ops",))("/srv/ops/TT builds", "main")
    assert "git -C <home>/'TT builds' pull --ff-only origin main" in hint and "replace <home> with that user's home directory" in hint
    other = sp._branch_update("main")("/home/zed/x y", "main")
    assert "zed" not in other and "<home>/'x y'" in other and "replace <home>" in other
    assert sp._path_word("/work/TT builds") == ("'/work/TT builds'", "")


def test_the_final_redaction_leaves_a_finished_hint_alone(tmp_path, py, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "me"))
    (tmp_path / "me" / "TT builds").mkdir(parents=True)
    m = repo(tmp_path, "me/TT builds/tt-metal", behind=65, days=42, origin="https://github.com/tenstorrent/tt-metal.git")
    src, fake = full(tmp_path, py, metal=m)
    reason = build(src, fake, GOOD())["hazards"][0]["reason"]
    assert "git -C ~/'TT builds/tt-metal' pull --ff-only origin main" in reason and str(tmp_path) not in reason.replace(str(tmp_path / "me"), "")


# ---- (5) every architecture is checked; a rejected or surplus entry is never ok ------------------------------------------

def _archs(tmp_path, entries):
    return config(tmp_path, model_type="m", architectures=entries, transformers_version="4.0.0")


def test_a_missing_ninth_class_is_found(tmp_path, py):
    entries = [f"A{i}ForCausalLM" for i in range(9)]
    src, fake = full(tmp_path, py, model=_archs(tmp_path, entries))
    rec = build(src, fake, probe_says("5.0.0", {e: e != "A8ForCausalLM" for e in entries}))
    assert names(rec["hazards"]) == [sp.H_CLASS] and "A8ForCausalLM" in rec["hazards"][0]["reason"]


@pytest.mark.parametrize("entries,why", [
    ([f"A{i}" for i in range(65)], "only the first 64"),
    (["Good", None, 5, "a b"], "3 of the 4"),
    ("LlamaForCausalLM", "not a list"),
])
def test_an_architecture_entry_that_was_not_checked_is_an_advisory_and_never_ok(tmp_path, py, entries, why):
    src, fake = full(tmp_path, py, model=_archs(tmp_path, entries))
    rec = build(src, fake, probe_says("5.0.0", {f"A{i}": True for i in range(65)} | {"Good": True}))
    assert rec["severity"] == "advisory" and "transformers_architectures_unchecked" in names(rec["advisories"])
    assert why in next(a["reason"] for a in rec["advisories"] if a["name"] == "transformers_architectures_unchecked")
    assert rec["parts"]["transformers"]["state"] != skillsfresh.CURRENT


def test_the_class_names_at_the_bound_are_all_sent_to_the_interpreter(tmp_path, py):
    entries = [f"A{i}" for i in range(sp.MAX_CLASSES)]
    probe = probe_says("5.0.0", {e: True for e in entries})
    src, fake = full(tmp_path, py, model=_archs(tmp_path, entries))
    rec = build(src, fake, probe)
    assert len(probe.calls[0][4:]) == sp.MAX_CLASSES and rec["severity"] == "ok"


# ---- relative interpreter path -------------------------------------------------------------------------------------------

def test_a_relative_interpreter_path_is_refused_not_looked_up_from_another_directory(tmp_path, monkeypatch):
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    exe = tmp_path / ".venv" / "bin" / "python"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    out = sp.probe_interpreter(".venv/bin/python", [], run=lambda *a, **k: pytest.fail("must not run"))
    assert out["state"] == "unknown" and "relative" in out["why"]


# ---- blocking is an opt-in ---------------------------------------------------------------------------------------------

def test_without_the_opt_in_a_critical_finding_is_a_would_block_advisory_with_its_named_reason(tmp_path, py):
    src, fake = full(tmp_path, py, model=config(tmp_path, **QWEN4))
    rec = sp.build_record(src, new_run=True, run=fake, probe_run=probe_says("4.57.6", {"Qwen4ExpForCausalLM": False}), now=1.0, block=False)
    assert rec["hazards"] == [] and rec["severity"] == "advisory" and rec["blocking_requested"] is False
    flagged = [a for a in rec["advisories"] if a.get("would_block")]
    assert [a["name"] for a in flagged] == [sp.H_BELOW, sp.H_CLASS] and "5.19.0" in flagged[0]["reason"]


def test_with_the_opt_in_the_same_findings_are_critical(tmp_path, py):
    src, fake = full(tmp_path, py, model=config(tmp_path, **QWEN4))
    rec = sp.build_record(src, new_run=True, run=fake, probe_run=probe_says("4.57.6", {"Qwen4ExpForCausalLM": False}), now=1.0, block=True)
    assert names(rec["hazards"]) == [sp.H_BELOW, sp.H_CLASS] and rec["blocking_requested"] is True


def test_the_check_the_supervisor_holds_reads_the_environment_variable(tmp_path, py, monkeypatch):
    src, fake = full(tmp_path, py, model=config(tmp_path, **QWEN4))
    stage = tmp_path / "s"
    stage.mkdir()
    probe = probe_says("4.57.6", {"Qwen4ExpForCausalLM": False})
    monkeypatch.delenv(sp.BLOCK_ENV, raising=False)
    off, _ = sp.make_check(src, run=fake, probe_run=probe, offline=False)(stage, True)
    monkeypatch.setenv(sp.BLOCK_ENV, "1")
    on, _ = sp.make_check(src, run=fake, probe_run=probe, offline=False)(stage, True)
    assert off["hazards"] == [] and names(on["hazards"]) == [sp.H_BELOW, sp.H_CLASS]


def test_a_would_block_record_does_not_stop_stage_0_and_the_ledger_and_the_narration_name_it(rig_for_round3):
    rig = rig_for_round3
    from test_supervisor_stack_pins import run, evidence
    from orchard.supervisor import Control

    def advisory_only(stage_dir, new_run):
        rec = sp.unchecked_record("fake", new_run=new_run)
        rec["advisories"] = [dict(sp._entry(sp.H_BELOW, "transformers 4.57.6 is installed and the checkpoint needs 5.19.0 or later"), would_block=True)]
        return rec, sp.write_record(stage_dir, rec)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    run(rig, advisory_only)
    (end,) = [e["data"] for e in rig.entries() if e["event"] == "stage_end" and e["stage"] == 0]
    assert end["result"] == "pass"
    (entry,) = evidence(rig)
    assert entry["would_block"] == [sp.H_BELOW] and entry["blocking"] is False
    line = next(e for e in rig.entries() if e["event"] == "evidence" and e["data"].get("what") == "stack pins")
    text = " ".join(l.text for l in narrate.lines_for(line))
    assert sp.H_BELOW in text and "ORCHARD_STACK_PINS_BLOCK=1" in text


def test_the_real_check_is_told_whether_the_operator_opted_in(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from orchard import supervisor
    seen = []
    monkeypatch.setattr(sp, "make_check", lambda sources, **kw: seen.append(kw))
    sup = SimpleNamespace(paths=SimpleNamespace(operator_home=str(tmp_path)), home=tmp_path, inputs={}, run_dir=tmp_path / "run", extra_env={})
    args = SimpleNamespace(skills_dir=[])
    supervisor.stack_check_for(sup, args, environ={})
    supervisor.stack_check_for(sup, args, environ={"ORCHARD_STACK_PINS_BLOCK": "1"})
    supervisor.stack_check_for(sup, args, environ={"ORCHARD_STACK_PINS_BLOCK": "yes"})
    assert [kw["block"] for kw in seen] == [False, True, False]
