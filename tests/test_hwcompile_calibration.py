# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The committed comparison of the device-less chip compile with the chip (scripts/calibrate_hwcompile.py).

docs/run-logs/evidence/hwcompile-calibration/calibration.json says the check agreed with the chip on every kernel of the pair evidence.
These tests do not trust that file's own `agree` flags: they read the chip's result files again and judge every row from its fields.
They also pin the script's rule for agreement and for reading the chip files."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CALIBRATION = REPO / "docs" / "run-logs" / "evidence" / "hwcompile-calibration" / "calibration.json"


def _script():
    spec = importlib.util.spec_from_file_location("calibrate_hwcompile_under_test", REPO / "scripts" / "calibrate_hwcompile.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows():
    return json.loads(CALIBRATION.read_text(encoding="utf-8"))


def test_every_chip_result_of_the_evidence_has_a_row_and_every_row_agrees_when_judged_again():
    script, data = _script(), _rows()
    by_prefix = {"1.1.6": [r for r in data["rows"] if r["version"].startswith("1.1.6")],
                 "1.1.10": [r for r in data["rows"] if r["version"].startswith("1.1.10")]}
    for prefix, rows in by_prefix.items():
        truth = script.chip_results(script.EVIDENCE / "results" / script.CHIP_FILES[prefix])
        assert {(r["kernel"], tuple(r["shape"])) for r in rows} == set(truth), prefix          # no chip result was left out
        for row in rows:
            chip = truth[(row["kernel"], tuple(row["shape"]))]
            assert row["chip"] == chip, row["kernel"]                                           # the row quotes the chip file
            assert script.agrees(chip, row["here"]) is True and row["agree"] is True, (prefix, row["kernel"], row["here"])
    assert (len(by_prefix["1.1.6"]), len(by_prefix["1.1.10"])) == (9, 32)


def test_the_comparison_covers_chip_passes_and_three_different_chip_refusals():
    """A comparison in which the chip only ever passed, or only ever failed, would not show the check tells the two apart."""
    rows = _rows()["rows"]
    refused = {r["chip"]["first_line"] for r in rows if not r["chip"]["ok"]}
    assert sum(r["chip"]["ok"] for r in rows) == 30 and sum(not r["chip"]["ok"] for r in rows) == 11
    assert len(refused) == 3
    assert any("unexpected keyword argument 'shape'" in line for line in refused)
    assert any("Unknown function: ttl.grid_size" in line for line in refused)
    assert any("UnaryMinusOp" in line for line in refused)
    for package in ("1.1.6", "1.1.10"):
        mine = [r for r in rows if r["version"].startswith(package)]
        assert any(r["chip"]["ok"] for r in mine) and any(not r["chip"]["ok"] for r in mine), package


@pytest.mark.parametrize("chip,here,want", [
    ({"ok": True, "first_line": ""}, {"category": "compiled", "first_line": ""}, True),
    ({"ok": True, "first_line": ""}, {"category": "compile_error", "first_line": "error: x"}, False),
    ({"ok": True, "first_line": ""}, {"category": "timeout", "first_line": ""}, False),
    ({"ok": False, "first_line": "error: x"}, {"category": "compile_error", "first_line": "error: x"}, True),
    ({"ok": False, "first_line": "error: x"}, {"category": "compile_error", "first_line": "error: y"}, False),      # refused for another reason
    ({"ok": False, "first_line": "error: x"}, {"category": "compiled", "first_line": ""}, False),
    ({"ok": False, "first_line": "error: x"}, {"category": "guard_failed", "first_line": "error: x"}, False),       # a check that did not run agrees with nothing
    ({"ok": False, "first_line": "error: x"}, {"category": "no_kernel_written", "first_line": "error: x"}, False),
])
def test_the_rule_for_agreement(chip, here, want):
    assert _script().agrees(chip, here) is want


def test_the_chip_files_are_read_as_one_result_per_kernel_and_shape(tmp_path):
    script = _script()
    path = tmp_path / "chip.jsonl"
    path.write_text("noise\nRESULT " + json.dumps({"name": "k", "results": [
        {"shape": [64, 128], "ok": True, "pcc": 1.0},
        {"shape": [128, 128], "ok": False, "error": "Err: first line\n  --> file:3\n"}]}) + "\n", encoding="utf-8")
    assert script.chip_results(path) == {("k", (64, 128)): {"ok": True, "first_line": ""},
                                         ("k", (128, 128)): {"ok": False, "first_line": "Err: first line"}}


def test_the_tensor_count_is_read_from_the_operation():
    script = _script()
    assert script.tensor_count("import ttl\n@ttl.operation()\ndef k(a, b, out): pass\n", "k") == 3
    with pytest.raises(SystemExit):
        script.tensor_count("x = 1\n", "k")


def test_calibrate_judges_each_row_with_the_check_it_is_given(monkeypatch):
    """With a check that compiles everything, the rows where the chip refused the kernel must disagree."""
    script = _script()
    monkeypatch.setattr(script.ttlang_hwcompile, "package_identity",
                        lambda python: {"package": "tt-lang", "version": "1.1.6", "compiler_sha256": "0" * 64, "files": []})

    def always(python, kernel, shapes, **kw):
        return {"category": "compiled", "first_line": "", "kernel_sha256": "1" * 64, "kernel_sources": [{}]}
    result = script.calibrate(["/venv/bin/python"], always)
    assert len(result["rows"]) == 9 and sum(r["agree"] for r in result["rows"]) == 1          # only the kernel the chip ran
