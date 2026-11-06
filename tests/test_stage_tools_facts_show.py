# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`facts` and `show` (orchard/skills/full-port-templates/stage_tools.py, issue 505): one call in place of `cat port_model.py | head -200`,
`grep -E "ple|PLE" port_facts.json` and `python3 -c` over the facts file, the reads that run 18 of the tiny Qwen4 rehearsal repeated."""
import json
import os
import subprocess
import sys

import pytest

from test_stage_tools_last_check import TOOL, load_tool

FACTS = {"schema": "orchard-port-facts/1", "model_type": "qwen4_exp_text",
         "config": {"hidden_size": 64, "hc_count": 4, "ple_layer_ids": [2], "head_dim": 32},
         "layer_schedule": {"linear_attention": "0-2", "full_attention": "3"},
         "tensors": ["lm_head.weight  BF16 [384, 64]  x1", "model.layers.{i}.ple.weight  BF16 [8, 64]  x1  i=[1]", "model.embed_tokens.weight  BF16 [384, 64]  x1"]}


def stage_with(tmp_path, facts=FACTS):
    run = tmp_path / "run"
    stage = run / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    if facts is not None:
        (stage / "port_facts.json").write_text(json.dumps(facts))
    (stage / "port" / "port_model.py").write_text("".join(f"line {i}\n" for i in range(1, 501)))
    return stage


def cli(stage, *args):
    import shutil
    shutil.copy2(TOOL, stage / "stage_tools.py")
    return subprocess.run([sys.executable, "-I", str(stage / "stage_tools.py"), *args], cwd=stage.parent.parent, env={"PATH": os.environ.get("PATH", "")},
                          capture_output=True, text=True, timeout=30)


# ---- facts ----------------------------------------------------------------------------------------------

def test_facts_with_no_key_lists_the_top_level_keys(tmp_path):
    done = cli(stage_with(tmp_path), "facts")
    assert done.returncode == 0 and "config" in done.stdout and "layer_schedule" in done.stdout and "tensors" in done.stdout


def test_facts_prints_one_value_by_a_dotted_path(tmp_path):
    stage = stage_with(tmp_path)
    done = cli(stage, "facts", "config.ple_layer_ids")
    assert done.returncode == 0 and done.stdout.strip() == "[2]"
    done = cli(stage, "facts", "layer_schedule.full_attention")
    assert done.stdout.strip() == '"3"'


def test_facts_takes_a_list_index(tmp_path):
    assert "lm_head.weight" in cli(stage_with(tmp_path), "facts", "tensors.0").stdout


def test_a_key_that_is_not_there_names_the_keys_that_are(tmp_path):
    done = cli(stage_with(tmp_path), "facts", "config.ple")
    assert done.returncode == 1 and "no key 'ple'" in done.stdout and "ple_layer_ids" in done.stdout


def test_find_searches_keys_and_text_case_insensitively_and_names_each_path(tmp_path):
    done = cli(stage_with(tmp_path), "facts", "--find", "PLE")
    assert done.returncode == 0
    assert "config.ple_layer_ids" in done.stdout and "tensors.1" in done.stdout and "ple.weight" in done.stdout


def test_find_with_no_hit_says_so(tmp_path):
    assert "no key or text contains" in cli(stage_with(tmp_path), "facts", "--find", "zzzz").stdout


def test_a_long_value_is_cut_and_says_how_much_was_left_out(tmp_path):
    facts = {"tensors": [f"t{i}" * 20 for i in range(2000)]}
    done = cli(stage_with(tmp_path, facts), "facts", "tensors")
    assert len(done.stdout) < 4500 and "more characters not shown" in done.stdout


def test_find_stops_at_a_bound(tmp_path):
    facts = {"tensors": [f"ple {i}" for i in range(500)]}
    done = cli(stage_with(tmp_path, facts), "facts", "--find", "ple")
    assert len(done.stdout.splitlines()) <= 31 and "more matches not shown" in done.stdout


def test_no_facts_file_is_a_reason(tmp_path):
    done = cli(stage_with(tmp_path, facts=None), "facts")
    assert done.returncode == 2 and "port_facts.json missing" in done.stdout


@pytest.mark.parametrize("bad", ["[1, 2]", '"text"', "null", "{", ""])
def test_a_facts_file_that_is_not_an_object_is_a_reason(tmp_path, bad):
    stage = stage_with(tmp_path)
    (stage / "port_facts.json").write_text(bad)
    done = cli(stage, "facts")
    assert done.returncode == 2 and "Traceback" not in done.stderr and "port_facts.json" in done.stdout


@pytest.mark.parametrize("key", ["", "a..b", ".", "config.", "x" * 500, "tensors.99999999999999999999", "tensors.-1"])
def test_odd_keys_give_a_reason_and_no_traceback(tmp_path, key):
    done = cli(stage_with(tmp_path), "facts", key)
    assert done.returncode in (1, 2) and "Traceback" not in done.stderr


def test_a_deeply_nested_facts_file_does_not_crash_find(tmp_path):
    nest = "[" * 5000 + "]" * 5000
    stage = stage_with(tmp_path)
    (stage / "port_facts.json").write_text('{"a": ' + nest + "}")
    done = cli(stage, "facts", "--find", "x")
    assert "Traceback" not in done.stderr


# ---- show -----------------------------------------------------------------------------------------------

def test_show_prints_a_range_with_line_numbers_and_a_header(tmp_path):
    done = cli(stage_with(tmp_path), "show", "port/port_model.py", "--lines", "10:12")
    assert done.returncode == 0
    lines = done.stdout.splitlines()
    assert "500 lines" in lines[0] and "10-12" in lines[0]
    assert lines[1:] == ["    10  line 10", "    11  line 11", "    12  line 12"]


def test_show_accepts_a_path_from_the_run_directory(tmp_path):
    done = cli(stage_with(tmp_path), "show", "stages/2/port/port_model.py", "--lines", "1:2")
    assert done.returncode == 0 and "line 2" in done.stdout


def test_the_default_range_is_the_first_two_hundred_lines_and_the_maximum_is_four_hundred(tmp_path):
    stage = stage_with(tmp_path)
    assert len(cli(stage, "show", "port/port_model.py").stdout.splitlines()) == 201
    done = cli(stage, "show", "port/port_model.py", "--lines", "1:450")
    assert done.returncode == 2 and "at most 400" in done.stdout


@pytest.mark.parametrize("rng", ["0:5", "5:2", "a:b", "5", "-1:3", "1:2:3", "1:", ":3", "99999999999999999999:99999999999999999999"])
def test_a_bad_range_is_refused(tmp_path, rng):
    done = cli(stage_with(tmp_path), "show", "port/port_model.py", f"--lines={rng}")
    assert done.returncode == 2 and "--lines" in done.stdout and "Traceback" not in done.stderr


def test_a_range_past_the_end_says_how_long_the_file_is(tmp_path):
    done = cli(stage_with(tmp_path), "show", "port/port_model.py", "--lines", "600:610")
    assert done.returncode == 2 and "500 lines" in done.stdout


def test_a_path_outside_the_run_directory_is_refused(tmp_path):
    stage = stage_with(tmp_path)
    (tmp_path / "secret.txt").write_text("hunter2\n")
    for rel in ("../../../secret.txt", str(tmp_path / "secret.txt"), "/etc/hostname"):
        done = cli(stage, "show", rel)
        assert done.returncode == 2 and "outside the run directory" in done.stdout and "hunter2" not in done.stdout, rel


def test_a_link_inside_the_run_that_leads_outside_is_refused(tmp_path):
    stage = stage_with(tmp_path)
    (tmp_path / "secret.txt").write_text("hunter2\n")
    (stage / "port" / "link.txt").symlink_to(tmp_path / "secret.txt")
    (stage / "port" / "dirlink").symlink_to(tmp_path)
    for rel in ("port/link.txt", "port/dirlink/secret.txt"):
        done = cli(stage, "show", rel)
        assert done.returncode == 2 and "hunter2" not in done.stdout, rel


def test_a_fifo_a_directory_and_a_missing_file_are_reasons(tmp_path):
    stage = stage_with(tmp_path)
    os.mkfifo(stage / "port" / "fifo")
    for rel, word in (("port/fifo", "not a regular file"), ("port", "not a regular file"), ("port/none.py", "missing")):
        done = cli(stage, "show", rel)
        assert done.returncode == 2 and word in done.stdout, rel


def test_a_binary_file_is_refused_and_an_oversize_file_is_refused(tmp_path):
    stage = stage_with(tmp_path)
    (stage / "port" / "blob.bin").write_bytes(b"abc\0def")
    assert "binary" in cli(stage, "show", "port/blob.bin").stdout
    (stage / "port" / "big.txt").write_text("x\n" * 1_100_000)
    assert "larger than" in cli(stage, "show", "port/big.txt").stdout


def test_a_long_line_is_cut(tmp_path):
    stage = stage_with(tmp_path)
    (stage / "port" / "wide.txt").write_text("y" * 5000 + "\n")
    out = cli(stage, "show", "port/wide.txt").stdout.splitlines()
    assert len(out[1]) < 400 and out[1].endswith("...")


def test_control_characters_in_a_file_do_not_reach_the_terminal(tmp_path):
    stage = stage_with(tmp_path)
    (stage / "port" / "esc.txt").write_text("a\x1b[31mred\x1b[0m\n")
    assert "\x1b" not in cli(stage, "show", "port/esc.txt").stdout


def test_the_usage_line_names_every_tool(tmp_path):
    out = cli(stage_with(tmp_path)).stdout
    for name in ("last-check", "layer-diff", "facts", "show"):
        assert name in out


@pytest.mark.parametrize("needle", ["", "x" * 101])
def test_find_refuses_an_empty_or_an_overlong_text(tmp_path, needle):
    done = cli(stage_with(tmp_path), "facts", "--find", needle)
    assert done.returncode == 2 and "1 to 100 characters" in done.stdout
