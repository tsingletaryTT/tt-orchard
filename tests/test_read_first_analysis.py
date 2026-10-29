# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The analysis script of docs/run-logs/2026-10-28-rmsnorm-read-first.md (issue 164): tool-argument forms, calls that have no result
in the transcript, and shell output that only looks like a reading of the specification."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "docs" / "run-logs" / "2026-10-28-rmsnorm-read-first-analysis.py"
SPEC_TEXT = "\n".join(["# Spec", "## Dataflow buffer", "a", "## Block", "b", "## Block states"]
                      + [f"state line {i}" for i in range(8)] + ["## Other", "z"]) + "\n"


def mod():
    return runpy.run_path(str(SCRIPT), run_name="analysis")


def attempt(tmp_path, turns):
    """An attempt directory. `turns` is a list of (calls, results): calls are (id, name, args) and results {id: text}; a result is
    stored on the NEXT turn's `sent`, as the real transcript does."""
    d = tmp_path / "rmsnorm" / "attempt-1"
    (d / "reference").mkdir(parents=True)
    (d / "log").mkdir()
    (d / "reference" / "TTLangSpecification.md").write_text(SPEC_TEXT)
    (d / "attempt.json").write_text(json.dumps({"attempt": 1, "passed_sim": False, "category": "numeric_mismatch", "first_line": "x"}))
    rows, pending = [], {}
    for n, (calls, results) in enumerate(turns, start=1):
        rows.append({"turn": n, "sent": [{"role": "tool", "tool_call_id": i, "content": t} for i, t in pending.items()],
                     "received": {"tool_calls": [{"id": i, "function": {"name": nm, "arguments": a}} for i, nm, a in calls]}})
        pending = results
    (d / "log" / "probe-1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return d


def test_tool_arguments_as_a_string_an_object_a_list_or_null_all_give_a_dict():
    calls_and_results = mod()["calls_and_results"]
    for raw, want in [('{"path": "a"}', {"path": "a"}), ({"path": "a"}, {"path": "a"}), ("[1]", {}), ([1], {}), (None, {}),
                      ("not json", {}), ("null", {}), (7, {})]:
        rows = [{"turn": 1, "sent": [], "received": {"tool_calls": [{"id": "1", "function": {"name": "read_file", "arguments": raw}}]}}]
        calls, _ = calls_and_results(rows)
        assert calls[0]["args"] == want, raw


def test_a_final_call_without_a_result_makes_the_measurements_a_lower_bound(tmp_path):
    check = "kernel_check.py ran after this write:\nPASS"
    complete = attempt(tmp_path / "a", [([("1", "write_file", '{"path": "kernel.py"}')], {"1": check}), ([], {})])
    cut = attempt(tmp_path / "b", [([("1", "write_file", '{"path": "kernel.py"}')], {"1": check})])      # no later turn holds the result
    m = mod()["analyse"]
    assert m(complete)["measurements_complete"] is True and m(complete)["first_check_turn"] == 1
    got = m(cut)
    assert got["measurements_complete"] is False and got["calls_without_result"] == 1 and got["first_check_turn"] is None


def spec_numbers():
    return "\n".join(f"{n}:state line {n}" for n in range(7, 14))


def shell_read(tmp_path, command, output):
    d = attempt(tmp_path, [([("1", "shell", json.dumps({"command": command}))], {"1": output}), ([], {})])
    return mod()["analyse"](d)["block_states_read"]


def test_numbered_output_of_a_single_command_on_the_specification_is_credited(tmp_path):
    assert shell_read(tmp_path, "grep -n state TTLangSpecification.md", spec_numbers()) == 1
    assert shell_read(tmp_path / "b", "sed -n 7,13p reference/TTLangSpecification.md", spec_numbers()) == 1


def test_numbered_output_that_may_come_from_another_file_is_not_credited(tmp_path):
    out = spec_numbers()
    assert shell_read(tmp_path / "a", "grep -n state TTLangSpecification.md notes.txt", out) == 0     # two files: bare numbers
    assert shell_read(tmp_path / "b", "grep -n x TTLangSpecification.md; grep -n state other.py", out) == 0
    assert shell_read(tmp_path / "c", "cat other.md && echo TTLangSpecification.md", out) == 0
    assert shell_read(tmp_path / "e", "grep -n x TTLangSpecification.md; seq 7 13 | nl -s:", out) == 0   # a second command, no second file
    names = "Must be Written\nMust be Read\nRead-Write\n"
    assert shell_read(tmp_path / "d", "cat TTLangSpecification.md other.md", names) == 0


def test_lines_prefixed_with_the_specification_name_are_credited_whatever_else_ran(tmp_path):
    out = "\n".join(f"TTLangSpecification.md:{n}:state line" for n in range(7, 14))
    assert shell_read(tmp_path, "grep -n state TTLangSpecification.md notes.txt", out) == 1
