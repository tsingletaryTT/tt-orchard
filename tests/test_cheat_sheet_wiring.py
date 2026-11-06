# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Where the cheat sheets reach the coder (issue 523): the stage 2 task text, `stage_tools.py cheat`, the files the supervisor installs next to the
facts file, and the kernel probe's attempt directory. The sheets themselves are checked in tests/test_op_cheats.py."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import full_port, kernelprobe, op_cheats, port_facts, supervisor
from test_stage_tools_last_check import TOOL


def stage_dir(tmp_path, *, ttnn=True, ttlang=False):
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    stage.mkdir(parents=True)
    if ttnn:
        (stage / "cheat_sheet.json").write_bytes(op_cheats.TTNN_SHEET.read_bytes())
    if ttlang:
        (stage / "ttlang_cheat_sheet.json").write_bytes(op_cheats.TTLANG_SHEET.read_bytes())
    import shutil
    shutil.copy2(TOOL, stage / "stage_tools.py")
    return stage


def cli(stage, *args):
    return subprocess.run([sys.executable, "-I", str(stage / "stage_tools.py"), *args], cwd=stage.parent.parent, env={"PATH": os.environ.get("PATH", "")},
                          capture_output=True, text=True, timeout=30)


# ---- the task text ----------------------------------------------------------------------------------------------------------------------

def test_the_stage_2_task_names_the_ttnn_sheet_in_one_line_and_the_cheat_tool():
    note = port_facts.prepare_note(facts=True, skeleton=True, excerpts=True)
    lines = [l for l in note.replace("\n\n", "\n").split("\n") if full_port.CHEAT_MD_NAME in l]
    assert len(lines) == 1 and "TTNN" in lines[0]
    assert "cheat NAME" in note


def test_without_a_copied_sheet_the_task_names_neither_the_sheet_nor_the_tool():
    note = port_facts.prepare_note(facts=True, skeleton=True, cheat=False)
    assert full_port.CHEAT_MD_NAME not in note and "cheat" not in note


def test_the_ttlang_sheet_is_not_named_in_the_port_task():
    assert "ttlang" not in port_facts.prepare_note(facts=True, skeleton=True).lower().replace("tt-lang", "")


# ---- stage_tools cheat ------------------------------------------------------------------------------------------------------------------

def test_cheat_prints_one_entry_by_full_name_or_by_its_last_part(tmp_path):
    stage = stage_dir(tmp_path)
    full = cli(stage, "cheat", "ttnn.linear")
    short = cli(stage, "cheat", "linear")
    assert full.returncode == 0 and full.stdout == short.stdout
    assert "ttnn.linear(input_tensor_a, input_tensor_b" in full.stdout and "seen:" in full.stdout
    assert len(full.stdout.splitlines()) <= 4


def test_cheat_with_no_name_lists_the_names(tmp_path):
    out = cli(stage_dir(tmp_path), "cheat").stdout
    assert "ttnn.embedding" in out and "ttnn.rms_norm" in out and "entries in the ttnn sheet" in out


def test_cheat_can_print_a_ttlang_entry_with_its_status(tmp_path):
    done = cli(stage_dir(tmp_path, ttlang=True), "cheat", "--sheet", "ttlang", "sim.reduce_sum")
    assert done.returncode == 0 and "[simulator only]" in done.stdout


def test_cheat_refuses_an_unknown_name_a_missing_sheet_and_a_hostile_name(tmp_path):
    stage = stage_dir(tmp_path)
    assert cli(stage, "cheat", "ttnn.does_not_exist").returncode == 1
    assert cli(stage, "cheat", "x" * 200).returncode == 2
    assert cli(stage, "cheat", "--sheet", "ttlang").returncode == 2          # no ttlang sheet in this directory
    gone = stage_dir(tmp_path / "other", ttnn=False)
    done = cli(gone, "cheat", "ttnn.linear")
    assert done.returncode == 2 and "cheat_sheet.json" in done.stdout


def test_cheat_does_not_follow_a_link_where_the_sheet_should_be(tmp_path):
    stage = stage_dir(tmp_path, ttnn=False)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"sections": [{"entries": [{"name": "ttnn.planted", "signature": "x", "usage": "x", "notes": "x", "seen_in": "x"}]}]}))
    (stage / "cheat_sheet.json").symlink_to(outside)
    done = cli(stage, "cheat", "ttnn.planted")
    assert done.returncode == 2 and "planted(" not in done.stdout


def test_cheat_output_has_no_control_characters(tmp_path):
    stage = stage_dir(tmp_path, ttnn=False)
    (stage / "cheat_sheet.json").write_text(json.dumps({"sections": [{"entries": [
        {"name": "ttnn.x", "signature": "ttnn.x(a)\x1b[31m", "usage": "u\x00", "notes": "n", "seen_in": "s"}]}]}))
    out = cli(stage, "cheat", "ttnn.x").stdout
    assert "\x1b" not in out and "\x00" not in out


def test_the_cheat_tool_prints_the_same_text_as_the_library_for_every_entry_of_each_sheet(tmp_path):
    stage = stage_dir(tmp_path, ttlang=True)
    for which, path in (("ttnn", op_cheats.TTNN_SHEET), ("ttlang", op_cheats.TTLANG_SHEET)):
        sheet = op_cheats.load_sheet(path)
        for e in op_cheats.entries(sheet):
            got = cli(stage, "cheat", "--sheet", which, e["name"]).stdout
            assert e["signature"][:60] in got, e["name"]
            assert all(ln[:60] in got for ln in e["usage"].split("\n")), e["name"]


# ---- what the supervisor installs -------------------------------------------------------------------------------------------------------

def pin(tmp_path, *, orchard_dir=None):
    run = tmp_path / "run"
    sd = run / "stages" / "2"
    sd.mkdir(parents=True)
    entries = []
    stub = SimpleNamespace(inputs={"model": str(run)}, run_dir=run,
                           paths=SimpleNamespace(orchard_dir=str(orchard_dir or Path(supervisor.__file__).resolve().parent.parent)),
                           ledger=SimpleNamespace(append=lambda *a, **k: entries.append((a, k))))
    stub._run_json = lambda *a: supervisor.Supervisor._run_json(stub, *a)
    problems = supervisor.Supervisor._pin_full_port(stub, SimpleNamespace(number=2), sd)
    return problems, sd, entries


def test_the_supervisor_installs_the_sheet_data_and_the_rendered_sheet_and_says_so_in_the_ledger(tmp_path):
    problems, sd, entries = pin(tmp_path)
    assert problems == []
    assert (sd / full_port.CHEAT_NAME).read_bytes() == op_cheats.TTNN_SHEET.read_bytes()
    assert (sd / full_port.CHEAT_MD_NAME).read_text() == op_cheats.render(op_cheats.load_sheet(op_cheats.TTNN_SHEET))
    assert {full_port.CHEAT_NAME, full_port.CHEAT_MD_NAME} <= set(entries[-1][1]["copied"])


def test_an_edit_the_coder_made_to_the_sheet_is_undone_on_the_next_pin(tmp_path):
    _, sd, _ = pin(tmp_path)
    (sd / full_port.CHEAT_NAME).write_text("{}")
    (sd / full_port.CHEAT_MD_NAME).write_text("planted")
    supervisor.Supervisor._pin_full_port(SimpleNamespace(inputs={"model": str(tmp_path)}, run_dir=sd.parent.parent, paths=SimpleNamespace(
        orchard_dir=str(Path(supervisor.__file__).resolve().parent.parent)), ledger=SimpleNamespace(append=lambda *a, **k: None),
        _run_json=lambda *a: None), SimpleNamespace(number=2), sd)
    assert (sd / full_port.CHEAT_NAME).read_bytes() == op_cheats.TTNN_SHEET.read_bytes() and "planted" not in (sd / full_port.CHEAT_MD_NAME).read_text()


def test_a_sheet_that_cannot_be_read_is_skipped_with_a_ledger_decision_and_the_stage_still_pins(tmp_path):
    fake = tmp_path / "orchard-copy"
    templates = fake / "orchard" / "skills" / "full-port-templates"
    templates.mkdir(parents=True)
    for name in (full_port.TEMPLATE_NAME, full_port.TOOLS_NAME):
        (templates / name).write_bytes((full_port.TEMPLATE.parent / name).read_bytes())
    problems, sd, entries = pin(tmp_path, orchard_dir=fake)               # no ttnn_cheat_sheet.json there
    assert problems == [] and not (sd / full_port.CHEAT_MD_NAME).exists()
    assert any(k.get("decision") == "cheat sheet not copied" for _, k in entries)
    assert full_port.CHEAT_NAME not in entries[-1][1]["copied"]


# ---- the kernel probe -------------------------------------------------------------------------------------------------------------------

def test_the_probe_writes_the_ttlang_sheet_into_the_attempt_directory_and_the_task_names_it(tmp_path):
    adir = tmp_path / "attempt"
    adir.mkdir()
    assert kernelprobe.cheat_line(adir) == ""                              # nothing written: no sentence
    assert kernelprobe.write_cheat_sheet(adir) is True
    text = (adir / kernelprobe.CHEAT_NAME).read_text()
    assert text == op_cheats.render(op_cheats.load_sheet(op_cheats.TTLANG_SHEET)) and "simulator only" in text
    line = kernelprobe.cheat_line(adir)
    assert kernelprobe.CHEAT_NAME in line and "compile" in line


def test_the_sheet_is_not_a_template_file_so_the_authoritative_check_does_not_hash_it():
    assert kernelprobe.CHEAT_NAME not in kernelprobe.TEMPLATE_FILES


from test_kernelprobe import FACTS as PROBE_FACTS, fake_kenv  # noqa: E402,F401  (the probe's fake tt-lang environment)


def test_prepare_attempt_puts_the_sheet_next_to_the_spec_and_the_whole_kernel_task_names_it(tmp_path, fake_kenv):
    adir = tmp_path / "attempts" / "add" / "attempt-1"
    adir.parent.mkdir(parents=True)
    kernelprobe.prepare_attempt(adir, op="add", env=fake_kenv, tt_lang=PROBE_FACTS)
    assert (adir / kernelprobe.CHEAT_NAME).is_file()
    assert kernelprobe.CHEAT_NAME in kernelprobe.attempt_user_message("add", adir)
    (adir / kernelprobe.CHEAT_NAME).unlink()
    assert kernelprobe.CHEAT_NAME not in kernelprobe.attempt_user_message("add", adir)


def test_the_staged_kernel_task_names_the_sheet_when_the_directory_has_it(tmp_path):
    from orchard import kernelstaged
    adir = tmp_path / "a"
    adir.mkdir()
    stage = SimpleNamespace(name="copy", instruction="Copy x to out.")
    assert kernelprobe.CHEAT_NAME not in kernelstaged._user_message(stage, 1, 3, "row_sum", adir)
    assert kernelprobe.write_cheat_sheet(adir)
    assert kernelprobe.CHEAT_NAME in kernelstaged._user_message(stage, 2, 3, "row_sum", adir)


@pytest.mark.parametrize("broken", [{"sections": None}, {"sections": [{"entries": None}]}, {"sections": "x"}])
def test_cheat_reports_a_sheet_with_null_containers_as_exit_2_and_not_a_traceback(tmp_path, broken):
    stage = stage_dir(tmp_path)
    (stage / "cheat_sheet.json").write_text(json.dumps(broken))
    for args in (("cheat",), ("cheat", "linear")):
        done = cli(stage, *args)
        assert done.returncode == 2 and "malformed" in done.stdout and "Traceback" not in done.stderr
