# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The decoder idioms file (helper-vs-coder): its content rules, its one line in the stage 2 task, its installation in the stage directory,
and the two cheat-sheet mistakes taken from the run 20 and run 22 coder ports. The wiring tests assert on the supervisor's pin and on the
task text the coder gets, because those are the layers that fail when the file never reaches the coder."""
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

from orchard import full_port, op_cheats, port_facts, supervisor

IDIOMS = full_port.TEMPLATE.parent / "example_decoder_idioms.md"


def pin(tmp_path, orchard_dir=None):
    run = tmp_path / "run"
    sd = run / "stages" / "2"
    sd.mkdir(parents=True)
    entries = []
    stub = SimpleNamespace(inputs={"model": str(run)}, run_dir=run,
                           paths=SimpleNamespace(orchard_dir=str(orchard_dir or Path(supervisor.__file__).resolve().parent.parent)),
                           ledger=SimpleNamespace(append=lambda *a, **k: entries.append((a, k))))
    stub._run_json = lambda *a: supervisor.Supervisor._run_json(stub, *a)
    return supervisor.Supervisor._pin_full_port(stub, SimpleNamespace(number=2), sd), sd, entries


# ---- the file ---------------------------------------------------------------------------------------------------------------------------

def test_the_file_is_short_labelled_as_idioms_and_names_the_four_entry_points():
    text = IDIOMS.read_text()
    assert len(text.splitlines()) < 150
    assert "IDIOMS TO ADAPT, NOT CODE TO COPY" in text
    for name in ("load", "prefill", "decode_step", "close"):
        assert f"`{name}(" in text or f"def {name}(" in text, name


def test_the_file_has_no_model_numbers_machine_paths_or_model_names():
    text = IDIOMS.read_text()
    assert not re.search(r"/home/|/mnt/|HF_TOKEN", text)
    assert not re.search(r"qwen|llama|minicpm", text, re.I)
    code = "\n".join(l for l in text.splitlines() if l.startswith((" ", "\t")) or "=" in l)
    assert not re.search(r"(?<![\w.\[])\d{2,}(?![\w])", code), "a two-digit or longer number in a code line"


def test_every_ttnn_name_the_file_uses_is_in_the_verified_cheat_sheet():
    sheet_text = op_cheats.TTNN_SHEET.read_text()
    names = set(re.findall(r"ttnn\.[A-Za-z_][A-Za-z0-9_.]*", IDIOMS.read_text()))
    assert names, "the file shows no ttnn call"
    missing = sorted(n for n in names if n not in sheet_text)
    assert missing == [], missing


# ---- the task text ----------------------------------------------------------------------------------------------------------------------

def test_the_stage_2_task_names_the_idioms_file_in_one_line():
    note = port_facts.prepare_note(facts=True, skeleton=True, excerpts=True, idioms=True)
    lines = [l for l in note.replace("\n\n", "\n").split("\n") if full_port.IDIOMS_NAME in l]
    assert len(lines) == 1 and "adapt" in lines[0]


def test_without_an_installed_file_the_task_does_not_name_it():
    assert full_port.IDIOMS_NAME not in port_facts.prepare_note(facts=True, skeleton=True, excerpts=True)


# ---- the installed stage files ----------------------------------------------------------------------------------------------------------

def test_the_supervisor_installs_the_idioms_and_says_so_in_the_ledger(tmp_path):
    problems, sd, entries = pin(tmp_path)
    assert problems == []
    assert (sd / full_port.IDIOMS_NAME).read_bytes() == IDIOMS.read_bytes()
    assert full_port.IDIOMS_NAME in entries[-1][1]["copied"]


def test_an_edit_the_coder_made_to_the_idioms_is_undone_on_the_next_pin(tmp_path):
    _, sd, _ = pin(tmp_path)
    (sd / full_port.IDIOMS_NAME).write_text("planted")
    run = sd.parent.parent
    stub = SimpleNamespace(inputs={"model": str(run)}, run_dir=run, paths=SimpleNamespace(orchard_dir=str(Path(supervisor.__file__).resolve().parent.parent)),
                           ledger=SimpleNamespace(append=lambda *a, **k: None))
    stub._run_json = lambda *a: supervisor.Supervisor._run_json(stub, *a)
    supervisor.Supervisor._pin_full_port(stub, SimpleNamespace(number=2), sd)
    assert (sd / full_port.IDIOMS_NAME).read_bytes() == IDIOMS.read_bytes()


def test_a_missing_idioms_file_is_skipped_with_a_decision_and_never_stops_the_stage(tmp_path):
    fake = tmp_path / "orch"
    shutil.copytree(full_port.TEMPLATE.parent, fake / "orchard" / "skills" / "full-port-templates")
    (fake / "orchard" / "skills" / "full-port-templates" / "example_decoder_idioms.md").unlink()
    problems, sd, entries = pin(tmp_path / "t", orchard_dir=fake)
    assert problems == [] and not (sd / full_port.IDIOMS_NAME).exists()
    assert any(k.get("decision") == "idioms file not copied" for _a, k in entries)
    assert full_port.IDIOMS_NAME not in entries[-1][1]["copied"]


# ---- the cheat-sheet mistakes -----------------------------------------------------------------------------------------------------------

LOAD_GATE = "weight elements as host torch or numpy tensors"


def mistake(needle):
    sheet = op_cheats.load_sheet(op_cheats.TTNN_SHEET)
    hits = [m for m in sheet["mistakes"] if needle in m["symptom"]]
    assert len(hits) == 1, needle
    return sheet, hits[0]


def test_the_load_gate_mistake_names_the_cause_the_run_and_a_check():
    sheet, m = mistake(LOAD_GATE)
    assert op_cheats.shape_problems(sheet) == []
    assert "run 22" in m["seen_in"] and m["checks"]
    assert "reachable" in m["fix"] and "shape" in m["fix"]


def test_the_weights_from_a_config_built_model_mistake_names_the_cause_the_run_and_a_check():
    sheet, m = mistake("module.weight.data")
    assert "run 20" in m["seen_in"] and m["checks"] and "read_tensors" in m["fix"]


def test_the_new_mistakes_reach_the_rendered_sheet_the_coder_reads():
    rendered = op_cheats.render(op_cheats.load_sheet(op_cheats.TTNN_SHEET))
    assert LOAD_GATE in rendered and "module.weight.data" in rendered
    assert len(rendered.splitlines()) < op_cheats.RENDER_LINE_BUDGET


def test_the_supervisors_prepare_step_names_the_idioms_file_when_it_is_installed(tmp_path):
    """The wiring layer: `_prepare_in_code` builds the note the coder gets, from what stands in the stage directory."""
    sd = tmp_path / "stages" / "2"
    sd.mkdir(parents=True)
    stub = SimpleNamespace(_port_seed={}, telegraph=None)
    spec = SimpleNamespace(draft="full_port", number=2, skill="full-port-check")
    without = supervisor.Supervisor._prepare_in_code(stub, spec, sd).note
    assert full_port.IDIOMS_NAME not in without
    (sd / full_port.IDIOMS_NAME).write_text("x")
    with_file = supervisor.Supervisor._prepare_in_code(stub, spec, sd).note
    assert full_port.IDIOMS_NAME in with_file


def test_a_directory_where_the_idioms_file_goes_is_a_decision_and_not_a_traceback(tmp_path):
    run = tmp_path / "run"
    sd = run / "stages" / "2"
    (sd / full_port.IDIOMS_NAME).mkdir(parents=True)
    entries = []
    stub = SimpleNamespace(inputs={"model": str(run)}, run_dir=run, paths=SimpleNamespace(orchard_dir=str(Path(supervisor.__file__).resolve().parent.parent)),
                           ledger=SimpleNamespace(append=lambda *a, **k: entries.append((a, k))))
    stub._run_json = lambda *a: supervisor.Supervisor._run_json(stub, *a)
    assert supervisor.Supervisor._pin_full_port(stub, SimpleNamespace(number=2), sd) == []
    assert any(k.get("decision") == "idioms file not copied" for _a, k in entries)
