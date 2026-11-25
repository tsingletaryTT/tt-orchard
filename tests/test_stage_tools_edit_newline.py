# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`edit` and a line break typed as two characters (backslash and n).

In the Hy4-flash trial of stage 2 (2026-11-25) the third repair round passed `--new 'a\\nb'` to `stage_tools.py edit`. The tool wrote the two characters
into port_model.py, outside any string, and the file stopped parsing (a SyntaxError at line 37). The check blamed the port and the round was spent.

The rule these tests hold: for a Python file that parses before the edit, an edit whose result does not parse is not written as given. When the new text
holds backslash-n pairs and turning them into line breaks makes the file parse, that is what is written, and the message says so. When nothing makes it
parse, the call is refused with the file untouched. A backslash-n inside a string literal, which parses as given, is left alone. Each guard has a mutation in
the PR body that turns its test red."""
import ast
import os
import shutil
import subprocess
import sys

from test_stage_tools_last_check import TOOL

BODY = "def f(x):\n    y = x + 1\n    return y\n\n\ndef g(x):\n    return x\n"


def stage_with(tmp_path, body=BODY, name="port_model.py"):
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / name).write_text(body)
    return stage


def cli(stage, *args):
    shutil.copy2(TOOL, stage / "stage_tools.py")
    return subprocess.run([sys.executable, "-I", str(stage / "stage_tools.py"), "edit", *args], cwd=stage.parent.parent,
                          env={"PATH": os.environ.get("PATH", "")}, capture_output=True, text=True, timeout=30)


def port(stage, name="port_model.py"):
    return (stage / "port" / name).read_text()


def test_backslash_n_between_statements_becomes_a_line_break_and_the_message_says_so(tmp_path):
    stage = stage_with(tmp_path)
    done = cli(stage, "stages/2/port/port_model.py", "--old", "    y = x + 1\n", "--new", "    y = x + 1\\n    z = y * 2\n")
    assert done.returncode == 0, done.stdout
    ast.parse(port(stage))
    assert "    y = x + 1\n    z = y * 2\n    return y\n" in port(stage)
    assert "\\n" not in port(stage)
    assert "1 backslash-n pair" in done.stdout and "line break" in done.stdout, done.stdout


def test_the_message_names_how_many_pairs_were_turned_into_line_breaks(tmp_path):
    stage = stage_with(tmp_path)
    done = cli(stage, "stages/2/port/port_model.py", "--old", "    y = x + 1\n", "--new", "    a = 1\\n    b = 2\\n    y = x + a + b\n")
    assert done.returncode == 0, done.stdout
    assert "2 backslash-n pairs" in done.stdout, done.stdout
    ast.parse(port(stage))


def test_a_backslash_n_inside_a_string_literal_is_written_as_given(tmp_path):
    stage = stage_with(tmp_path)
    done = cli(stage, "stages/2/port/port_model.py", "--old", "    return y\n", "--new", "    print('a\\nb')\n    return y\n")
    assert done.returncode == 0, done.stdout
    assert "print('a\\nb')" in port(stage)
    assert "backslash-n" not in done.stdout


def test_a_result_that_cannot_be_made_to_parse_is_refused_with_the_file_untouched(tmp_path):
    stage = stage_with(tmp_path)
    before = port(stage)
    done = cli(stage, "stages/2/port/port_model.py", "--old", "    y = x + 1\n", "--new", "    y = (x + 1\\n")
    assert done.returncode == 2, done.stdout
    assert port(stage) == before
    assert "Nothing was changed" in done.stdout and "--new-file" in done.stdout


def test_a_mixed_new_text_with_a_real_string_escape_and_a_typed_break_is_refused_not_guessed(tmp_path):
    stage = stage_with(tmp_path)
    before = port(stage)
    done = cli(stage, "stages/2/port/port_model.py", "--old", "    return y\n", "--new", "    print('a\\nb')\\n    return y\n")
    assert done.returncode == 2, done.stdout
    assert port(stage) == before


def test_a_file_that_did_not_parse_before_is_edited_as_given(tmp_path):
    stage = stage_with(tmp_path, body="def f(:\n    pass\n")
    done = cli(stage, "stages/2/port/port_model.py", "--old", "def f(:\n", "--new", "def f(x):\\n    x = 1\n")
    assert done.returncode == 0, done.stdout
    assert "def f(x):\\n    x = 1" in port(stage)


def test_a_file_that_is_not_python_is_edited_as_given(tmp_path):
    stage = stage_with(tmp_path, body="a=1\n", name="notes.txt")
    done = cli(stage, "stages/2/port/notes.txt", "--old", "a=1\n", "--new", "a=1\\nb=2\n")
    assert done.returncode == 0, done.stdout
    assert port(stage, "notes.txt") == "a=1\\nb=2\n"


def test_a_new_file_with_the_same_mistake_gets_the_same_treatment(tmp_path):
    stage = stage_with(tmp_path)
    (stage.parent.parent / "new.txt").write_text("    y = x + 1\\n    z = y * 2\n")
    done = cli(stage, "stages/2/port/port_model.py", "--old", "    y = x + 1\n", "--new-file", "new.txt")
    assert done.returncode == 0, done.stdout
    ast.parse(port(stage))
    assert "\\n" not in port(stage)


def test_an_old_text_with_typed_breaks_that_matches_nothing_gets_a_hint_to_use_old_file(tmp_path):
    stage = stage_with(tmp_path)
    before = port(stage)
    done = cli(stage, "stages/2/port/port_model.py", "--old", "    y = x + 1\\n    return y\n", "--new", "    return x\n")
    assert done.returncode == 1, done.stdout
    assert port(stage) == before
    assert "--old-file" in done.stdout and "backslash" in done.stdout


def test_a_plain_edit_with_no_backslash_pair_is_unchanged(tmp_path):
    stage = stage_with(tmp_path)
    done = cli(stage, "stages/2/port/port_model.py", "--old", "    y = x + 1\n", "--new", "    y = x + 2\n")
    assert done.returncode == 0, done.stdout
    assert "backslash" not in done.stdout and "line break" not in done.stdout
