# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`last-check` (orchard/skills/full-port-templates/stage_tools.py, issue 505): one call that says which check failed, where, with which numbers
and which file holds the evidence. The fixtures copy the shape of run 18 of the tiny Qwen4 rehearsal: a numerics failure (exit 0, agreement
0.5625, first mismatch at position 0) and a failure at `prefill`."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import full_port

TOOL = Path(full_port.TEMPLATE).parent / "stage_tools.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("stage_tools_under_test", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REF = list(range(100, 132))
# 14 of 32 differ, the first at position 0 (run 18's numbers: 18 of 32 match, 0.5625)
CHIP = list(REF)
for k in (0, 1, 2, 3, 5, 8, 9, 12, 15, 17, 20, 24, 28, 30):
    CHIP[k] = 7


def make_stage(tmp_path: Path, *, output: str, returncode: int, record: dict | None = None, honest_hash: bool = True) -> Path:
    stage = tmp_path / "run" / "stages" / "2"
    (stage / "evidence").mkdir(parents=True)
    out = stage / "evidence" / "hw-test-output.txt"
    out.write_text(output)
    digest = hashlib.sha256(out.read_bytes()).hexdigest() if honest_hash else "0" * 64
    (stage / "test-result.json").write_text(json.dumps({"command": "python3 -P -s stages/2/port_check.py", "returncode": returncode, "timed_out": False,
                                                        "output": {"path": "stages/2/evidence/hw-test-output.txt", "sha256": digest}}))
    if record is not None:
        (stage / "evidence" / "port-check.json").write_text(json.dumps(record))
    return stage


NUMERICS_OUT = "loaded in 0.01s\ntop-1 agreement 0.5625 over 32 positions\nPORT_CHECK_SHA256 " + "a" * 64 + "\n"
PREFILL_OUT = ("port| some log\nFAILED: the port failed at 'prefill': 'RuntimeError: The size of tensor a (32) must match the size of tensor b (8) at "
               "non-singleton dimension 3'\nPORT_CHECK_FAILURE {\"kind\": \"port\", \"op\": \"prefill\"}\n")


def numerics_record(**over) -> dict:
    return {"schema": "orchard-port-check/1", "reference_ids": REF, "chip_ids": CHIP, "top1_agreement": 0.5625, **over}


def text(tool, stage) -> str:
    return "\n".join(tool.last_check(stage))


def test_a_numerics_failure_names_the_agreement_the_first_mismatch_and_the_evidence(tmp_path):
    stage = make_stage(tmp_path, output=NUMERICS_OUT, returncode=0, record=numerics_record())
    out = text(load_tool(), stage)
    assert "0.5625" in out and "18 of 32" in out and "14 of 32 ids differ" in out
    assert "0.85" in out
    assert "first mismatch at position 0" in out and "prefill" in out
    assert "| 100 | 7" in out                                  # a row: position, reference id, port id
    assert "stages/2/evidence/port-check.json" in out
    assert "names no layer" in out or "layer-diff" in out      # the output points at the next tool


def test_a_failure_at_prefill_names_the_op_the_kind_and_the_message(tmp_path):
    stage = make_stage(tmp_path, output=PREFILL_OUT, returncode=2)
    out = text(load_tool(), stage)
    assert "prefill" in out and "kind port" in out and "RuntimeError: The size of tensor a (32)" in out
    assert "stages/2/evidence/hw-test-output.txt" in out


def test_the_summary_is_under_forty_lines_even_for_a_long_failure(tmp_path):
    long = "FAILED: " + "x" * 5000 + "\n" + 'PORT_CHECK_FAILURE {"kind": "port", "op": "decode"}\n'
    stage = make_stage(tmp_path, output=long, returncode=2)
    lines = load_tool().last_check(stage)
    assert len(lines) < 40 and all(len(l) <= 400 for l in lines)


def test_a_setup_failure_says_the_port_is_not_to_blame(tmp_path):
    out_text = 'FAILED: TT_VISIBLE_DEVICES names no chip\nPORT_CHECK_FAILURE {"kind": "setup", "op": null}\n'
    out = text(load_tool(), make_stage(tmp_path, output=out_text, returncode=2))
    assert "kind setup" in out and "An edit to the port cannot fix it" in out


def test_a_passing_agreement_is_reported_as_a_pass_with_its_number(tmp_path):
    rec = numerics_record(chip_ids=list(REF), top1_agreement=1.0)
    out = text(load_tool(), make_stage(tmp_path, output=NUMERICS_OUT, returncode=0, record=rec))
    assert "1.0000" in out and "meets" in out and "first mismatch" not in out


def test_the_agreement_is_recomputed_from_the_ids_and_a_disagreeing_record_field_is_called_out(tmp_path):
    rec = numerics_record(top1_agreement=0.99)             # the record's own field says 0.99; the ids say 0.5625
    out = text(load_tool(), make_stage(tmp_path, output=NUMERICS_OUT, returncode=0, record=rec))
    assert "0.5625" in out and "differs from the ids" in out


def test_no_check_yet_is_said_plainly(tmp_path):
    stage = tmp_path / "run" / "stages" / "2"
    stage.mkdir(parents=True)
    assert "no check has run" in text(load_tool(), stage)


def test_an_output_file_that_differs_from_the_supervisors_record_is_flagged(tmp_path):
    stage = make_stage(tmp_path, output=PREFILL_OUT, returncode=2, honest_hash=False)
    assert "does not match" in text(load_tool(), stage)


def test_a_missing_record_after_exit_zero_is_not_checked_rather_than_a_pass(tmp_path):
    out = text(load_tool(), make_stage(tmp_path, output=NUMERICS_OUT, returncode=0, record=None))
    assert "not checked" in out and "meets" not in out


@pytest.mark.parametrize("bad", [None, [], {}, 3, "", "a\0b", 10**20, [1, "x"], {"reference_ids": 1}])
def test_a_wrongly_typed_record_gives_a_reason_and_no_traceback(tmp_path, bad):
    rec = numerics_record(reference_ids=bad, chip_ids=bad)
    out = text(load_tool(), make_stage(tmp_path, output=NUMERICS_OUT, returncode=0, record=rec))
    assert "not checked" in out


@pytest.mark.parametrize("bad", [None, [], 3, "x", {"returncode": "0"}, {"returncode": True}, {"output": 5}])
def test_a_wrongly_typed_test_record_gives_a_reason(tmp_path, bad):
    stage = make_stage(tmp_path, output=PREFILL_OUT, returncode=2)
    (stage / "test-result.json").write_text(json.dumps(bad))
    out = text(load_tool(), stage)
    assert "not checked" in out or "no check has run" in out


def test_a_fifo_or_a_link_in_place_of_the_output_is_refused_without_blocking(tmp_path):
    stage = make_stage(tmp_path, output=PREFILL_OUT, returncode=2)
    out_file = stage / "evidence" / "hw-test-output.txt"
    out_file.unlink()
    os.mkfifo(out_file)
    assert "not checked" in text(load_tool(), stage)
    out_file.unlink()
    out_file.symlink_to("/dev/zero")
    assert "not checked" in text(load_tool(), stage)


def test_a_link_to_a_regular_file_is_refused_too(tmp_path):
    stage = make_stage(tmp_path, output=PREFILL_OUT, returncode=2)
    real = tmp_path / "elsewhere.txt"
    real.write_text(PREFILL_OUT)
    out_file = stage / "evidence" / "hw-test-output.txt"
    out_file.unlink()
    out_file.symlink_to(real)
    assert "not checked" in text(load_tool(), stage)


def test_an_oversize_output_is_refused(tmp_path):
    stage = make_stage(tmp_path, output="x" * 10, returncode=2)
    tool = load_tool()
    (stage / "evidence" / "hw-test-output.txt").write_text("x" * (tool.OUTPUT_MAX + 1))
    assert "not checked" in text(tool, stage)


def test_the_agreement_bar_is_the_gates(tmp_path):
    assert load_tool().TOP1_MIN == full_port.TOP1_MIN


def test_the_command_line_runs_from_the_stage_directory_with_no_orchard_import(tmp_path):
    stage = make_stage(tmp_path, output=NUMERICS_OUT, returncode=0, record=numerics_record())
    import shutil
    shutil.copy2(TOOL, stage / "stage_tools.py")
    env = {"PATH": os.environ.get("PATH", "")}
    done = subprocess.run([sys.executable, "-I", str(stage / "stage_tools.py"), "last-check"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert done.returncode == 0 and "0.5625" in done.stdout
    import ast
    imported = {n.module if isinstance(n, ast.ImportFrom) else a.name for n in ast.walk(ast.parse(TOOL.read_text())) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in (getattr(n, "names", []) or [None])}
    assert not [m for m in imported if m and m.split(".")[0] == "orchard"], "the tool runs from a stage directory and may not import orchard"


# ---- run 20 (the next command, and the stale-output notice) --------------------------------------------------------------------------
# In run 20 the coder called `last-check` 23 times, mostly right after a write, and each time it printed the old failure: the check runs again
# only when the step ends. It also never ran `edit` or `layer-diff`: no output named a line to edit, and no output gave the command to run next.

FRAME_OUT = ("port| some log\nFAILED: the port failed at 'prefill': \"[port_model.py:4: logits = ttnn.to_torch(logits)] RuntimeError: TT_FATAL "
             "Can't convert a tensor distributed on MeshShape([1, 2]) mesh\"\nPORT_CHECK_FAILURE {\"kind\": \"port\", \"op\": \"prefill\"}\n")
PORT_SRC = "import ttnn\n\ndef prefill(model, ids):\n    logits = ttnn.to_torch(logits)\n    return logits\n"


def with_port(stage: Path, src: str = PORT_SRC) -> Path:
    (stage / "port").mkdir(exist_ok=True)
    (stage / "port" / "port_model.py").write_text(src)
    old = stage / "test-result.json"
    os.utime(stage / "port" / "port_model.py", ns=(old.stat().st_mtime_ns - 10**9, old.stat().st_mtime_ns - 10**9))   # written before the check
    return stage


def next_lines(out: str) -> list[str]:
    return [l for l in out.splitlines() if l.startswith("next: ")]


def test_every_failure_says_the_output_is_from_before_this_step_and_that_a_write_does_not_rerun_it(tmp_path):
    out = text(load_tool(), make_stage(tmp_path, output=PREFILL_OUT, returncode=2))
    assert "runs again" in out and "no tool call" in out and "writing a file does not run it" in out


def test_a_port_edited_after_the_check_is_called_out(tmp_path):
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2))
    os.utime(stage / "port" / "port_model.py", None)               # now: after the check
    os.utime(stage / "test-result.json", ns=(1, 1))
    assert "changed after this check" in text(load_tool(), stage)
    stage2 = with_port(make_stage(tmp_path / "b", output=FRAME_OUT, returncode=2))
    assert "changed after this check" not in text(load_tool(), stage2)


def test_a_failure_at_a_port_line_prints_one_edit_command_with_that_line_as_the_old_text(tmp_path):
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2))
    out = text(load_tool(), stage)
    nxt = next_lines(out)
    assert len(nxt) == 1
    assert nxt[0].startswith("next: python3 stages/2/stage_tools.py edit stages/2/port/port_model.py --old 'logits = ttnn.to_torch(logits)' --new ")
    assert "port_model.py line 4" in out                            # the location is read back to the coder as a sentence


def test_the_edit_command_takes_its_old_text_from_the_file_and_is_quoted_for_the_shell(tmp_path):
    src = "import ttnn\n\ndef prefill(model, ids):\n    x = f(\"it's\")  # $HOME `id`\n    return x\n"
    out_text = "FAILED: the port failed at 'prefill': '[port_model.py:4: x = f(\"it\\'s\")] ValueError: boom'\nPORT_CHECK_FAILURE {\"kind\": \"port\", \"op\": \"prefill\"}\n"
    stage = with_port(make_stage(tmp_path, output=out_text, returncode=2), src)
    nxt = next_lines(text(load_tool(), stage))
    import shlex
    words = shlex.split(nxt[0][len("next: "):])                     # one shell command; the old text is one word
    assert words[words.index("--old") + 1] == "x = f(\"it's\")  # $HOME `id`"


def test_a_frame_in_a_port_that_changed_after_the_check_gets_no_edit_command(tmp_path):
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2))
    os.utime(stage / "port" / "port_model.py", None)
    os.utime(stage / "test-result.json", ns=(1, 1))
    assert not next_lines(text(load_tool(), stage))


@pytest.mark.parametrize("src", ["", "   \n", "x" * 400 + "\n", "x" * 151 + "\n"])
def test_a_line_that_is_missing_empty_or_too_long_gets_no_edit_command(tmp_path, src):
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2), "\n\n\n" + src)         # the frame names line 4
    assert not next_lines(text(load_tool(), stage))


def test_a_message_that_only_looks_like_a_frame_is_not_taken_as_one(tmp_path):
    out_text = "FAILED: the port failed at 'prefill': 'ValueError: see [port_model.py:4: logits = 1] in the docs'\nPORT_CHECK_FAILURE {\"kind\": \"port\", \"op\": \"prefill\"}\n"
    stage = with_port(make_stage(tmp_path, output=out_text, returncode=2))
    assert not next_lines(text(load_tool(), stage))


def test_a_frame_naming_a_file_outside_the_port_directory_gets_no_edit_command(tmp_path):
    out_text = FRAME_OUT.replace("port_model.py:4", "../port_check.py:4")
    stage = with_port(make_stage(tmp_path, output=out_text, returncode=2))
    assert not next_lines(text(load_tool(), stage))


def test_a_device_ops_gate_failure_points_at_the_cheat_sheet_for_the_projection_call(tmp_path):
    out_text = "FAILED: device-ops: no source file makes a reachable ttnn matmul-family call\nPORT_CHECK_FAILURE {\"kind\": \"port\", \"op\": \"device-ops\"}\n"
    nxt = next_lines(text(load_tool(), make_stage(tmp_path, output=out_text, returncode=2)))
    assert nxt == ["next: python3 stages/2/stage_tools.py cheat ttnn.linear"]


def test_a_setup_failure_has_no_next_command(tmp_path):
    out_text = 'FAILED: TT_VISIBLE_DEVICES names no chip\nPORT_CHECK_FAILURE {"kind": "setup", "op": null}\n'
    assert not next_lines(text(load_tool(), make_stage(tmp_path, output=out_text, returncode=2)))


def test_a_numerics_failure_with_a_layer_file_names_layer_diff_as_the_next_command(tmp_path):
    from test_stage_tools_layer_diff import build
    stage = build(tmp_path)
    (stage / "evidence" / "port-check.json").write_text(json.dumps(numerics_record()))
    assert next_lines(text(load_tool(), stage)) == ["next: python3 stages/2/stage_tools.py layer-diff"]


def test_a_numerics_failure_without_a_layer_file_says_to_add_the_hook_and_names_no_command(tmp_path):
    stage = make_stage(tmp_path, output=NUMERICS_OUT, returncode=0, record=numerics_record())
    nxt = next_lines(text(load_tool(), stage))
    assert len(nxt) == 1 and "layer_outputs(model, ids)" in nxt[0] and "python3" not in nxt[0]


def test_a_passing_check_has_no_next_command(tmp_path):
    rec = numerics_record(chip_ids=list(REF), top1_agreement=1.0)
    assert not next_lines(text(load_tool(), make_stage(tmp_path, output=NUMERICS_OUT, returncode=0, record=rec)))


def test_a_line_of_exactly_the_limit_still_gets_the_command(tmp_path):
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT, returncode=2), "\n\n\n" + "x" * 150 + "\n")
    assert len(next_lines(text(load_tool(), stage))) == 1


def test_a_frame_whose_name_climbs_out_of_the_port_directory_gets_no_edit_command(tmp_path):
    stage = with_port(make_stage(tmp_path, output=FRAME_OUT.replace("port_model.py:4", "a/../../port_check.py:4"), returncode=2))
    (stage / "port_check.py").write_text("\n\n\nsecret = 1\n")
    out = text(load_tool(), stage)
    assert not next_lines(out) and "secret" not in out


def test_a_setup_failure_with_a_frame_shaped_message_gets_no_edit_command(tmp_path):
    out_text = FRAME_OUT.replace('"op": "prefill"', '"op": null').replace('"kind": "port"', '"kind": "setup"')
    stage = with_port(make_stage(tmp_path, output=out_text, returncode=2))
    assert not next_lines(text(load_tool(), stage))
