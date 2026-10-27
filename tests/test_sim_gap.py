"""A kernel that uses a ttl function the specification documents and the simulator lacks (`ttl.raw_element_read` and
`ttl.raw_element_write`) is judged on hardware only.

The check names the case `sim_gap` and says so; the probe runs such a kernel on hardware under its own lease and the
report marks the result `hardware only`. A hardware-only pass is never counted as a simulator pass."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import kernelprobe as kp
from test_kernel_loop_templates import CHECK, OUT, SIM
from test_kernelprobe import (FACTS, GOOD_HW_SET, KERNELS, canned, fake_kenv, hw_writer,  # noqa: F401
                                    _passed_attempt)

SRC = Path(os.environ.get("ORCHARD_TTLANG_SRC", Path.home() / "orchard-src" / "tt-lang"))
REPO = Path(__file__).resolve().parent.parent


def sim_gap_text() -> str:
    return (OUT / "sim_gap.txt").read_text()


def sim_gap_result() -> dict:
    return json.loads((OUT / "sim_gap.result.json").read_text())


def classify(text, result, phase="sim"):
    return CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=result,
                          bar=0.999, phase=phase)


# ---- the check ------------------------------------------------------------------------------------------------


def test_the_recorded_simulator_output_for_raw_element_read_is_a_sim_gap():
    category, line = classify(sim_gap_text(), sim_gap_result())
    assert category == "sim_gap"
    assert "ttl.raw_element_read" in line
    assert "simulator" in line and "hardware" in line


def test_the_sim_gap_line_says_the_kernel_is_judged_on_hardware_only_and_does_not_blame_the_kernel():
    _, line = classify(sim_gap_text(), sim_gap_result())
    assert "only on hardware" in line
    assert "documents" in line                                  # the specification documents the function
    assert len(line) <= 300


@pytest.mark.parametrize("name", ["raw_element_read", "raw_element_write"])
def test_both_documented_functions_are_sim_gaps(name):
    text = f"AttributeError: '_TTLNamespace' object has no attribute '{name}'"
    result = {"phase": "sim", "sets": [], "error": {"phase": "run", "type": "RuntimeError",
                                                     "text": f"node0-read: {text}"}}
    category, line = classify(text, result)
    assert category == "sim_gap" and f"ttl.{name}" in line


def test_a_misspelled_name_is_still_a_tt_lang_error():
    text = "AttributeError: '_TTLNamespace' object has no attribute 'raw_elemnt_read'"
    result = {"phase": "sim", "sets": [], "error": {"phase": "run", "type": "RuntimeError", "text": f"node0-read: {text}"}}
    assert classify(text, result)[0] == "tt_lang_error"


def test_a_name_the_specification_does_not_document_is_still_a_tt_lang_error():
    text = "AttributeError: '_TTLNamespace' object has no attribute 'frobnicate'"
    result = {"phase": "sim", "sets": [], "error": {"phase": "run", "type": "RuntimeError", "text": f"node0-read: {text}"}}
    assert classify(text, result)[0] == "tt_lang_error"


def test_the_same_words_about_another_object_are_not_a_sim_gap():
    text = "AttributeError: 'Block' object has no attribute 'raw_element_read'"
    result = {"phase": "sim", "sets": [], "error": {"phase": "run", "type": "RuntimeError", "text": f"node0-read: {text}"}}
    assert classify(text, result)[0] == "tt_lang_error"


def test_a_kernel_cannot_claim_a_sim_gap_by_printing_the_words():
    """Only the error the simulator raised counts; text the kernel printed does not."""
    printed = "AttributeError: '_TTLNamespace' object has no attribute 'raw_element_read'\n"
    result = {"phase": "sim", "sets": [], "error": {"phase": "run", "type": "RuntimeError",
                                                     "text": "Deadlock detected: all generators blocked"}}
    assert classify(printed + "Deadlock detected: all generators blocked", result)[0] == "tt_lang_error"


def test_on_hardware_the_same_words_are_a_compile_error_and_never_a_sim_gap():
    assert classify(sim_gap_text(), sim_gap_result(), phase="hw")[0] == "compile_error"


def test_every_function_the_check_treats_as_a_sim_gap_is_documented_in_the_specification():
    spec = SRC / "docs" / "sphinx" / "specs" / "TTLangSpecification.md"
    if not spec.is_file():
        pytest.skip("SKIPPED: no tt-lang checkout (run `tt-orchard kernel-env`)")
    text = spec.read_text(encoding="utf-8")
    for name in CHECK.SIM_GAP_NAMES:
        assert f"ttl.{name}(" in text, name


def test_the_check_prints_the_gap_on_its_fail_line(tmp_path):
    """The agent reads the FAIL line of its own check: it must say the simulator lacks the function."""
    if not SIM.exists():
        pytest.skip("SKIPPED: no tt-lang-sim in the ttlang venv (run `tt-orchard kernel-env`)")
    tdir = REPO / "orchard" / "skills" / "kernel-loop-templates"
    adir = tmp_path / "attempt"
    adir.mkdir()
    (adir / "kernel.py").write_text((KERNELS / "linear_scan_raw_element.py").read_text())
    (adir / "kernel_spec.json").write_text((tdir / "ops" / "linear_scan.json").read_text())
    done = subprocess.run([sys.executable, str(tdir / "kernel_check.py"), "--attempt-dir", str(adir), "--sim", str(SIM),
                           "--timeout-s", "120"], capture_output=True, text=True, timeout=300)
    first = done.stdout.splitlines()[0]
    assert first.startswith("FAIL: sim_gap:") and "ttl.raw_element_read" in first and "hardware" in first, done.stdout
    assert json.loads((adir / "check.json").read_text())["category"] == "sim_gap"


# ---- the probe ------------------------------------------------------------------------------------------------


def sim_gap_attempt(tmp_path, env, n=1):
    """An attempt the simulator could not run because of a sim gap: no simulator pass, the checked copy in place."""
    adir, attempt = _passed_attempt(tmp_path, env, n=n)
    attempt.update(passed_sim=False, category="sim_gap", first_line="the simulator has no ttl.raw_element_read")
    (adir / "attempt.json").write_text(json.dumps(attempt))
    return adir, attempt


def test_sim_gap_is_a_probe_category():
    assert "sim_gap" in kp.CATEGORIES


def test_a_sim_gap_attempt_runs_on_hardware_and_is_marked_hardware_only(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    calls = []
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], calls))
    assert len([c for c in calls if "--out" in c]) == 1                    # the kernel ran on hardware
    assert attempt["hw"]["passed"] is True and attempt["hw"]["hardware_only"] is True
    assert attempt["passed_sim"] is False                                  # never recorded as a simulator pass
    assert json.loads((adir / "attempt.json").read_text())["hw"]["hardware_only"] is True


def test_the_hardware_run_of_a_sim_gap_attempt_uses_the_probes_own_lease_and_the_checked_copy(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    calls = []
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], calls))
    argv = next(c for c in calls if "--out" in c)
    assert argv[:2] == ["gozer", "run"] and "--chips" in argv
    kernel = Path(argv[argv.index("--kernel") + 1])
    assert kernel == kp.verdict_dir(tmp_path, "add", 1) / kp.CLEAN_DIR / "kernel.py"


def test_a_sim_gap_attempt_whose_checked_copy_changed_is_not_run(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    (kp.verdict_dir(tmp_path, "add", 1) / kp.CLEAN_DIR / "kernel.py").write_text("# changed\n")
    calls = []
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], calls))
    assert [c for c in calls if "--out" in c] == []
    assert attempt["hw"]["category"] in ("harness_error", "forbidden_code", "no_kernel")


def test_other_simulator_failures_still_get_no_hardware_run(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    attempt["category"] = "tt_lang_error"
    calls = []
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], calls))
    assert [c for c in calls if "--out" in c] == [] and attempt["hw"] is None


def test_a_simulator_pass_is_not_marked_hardware_only(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], []))
    assert attempt["hw"]["passed"] is True and not attempt["hw"].get("hardware_only")


def test_a_failed_hardware_run_of_a_sim_gap_attempt_is_marked_hardware_only_too(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    bad = dict(GOOD_HW_SET, pcc=0.2)
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([bad], []))
    assert attempt["hw"]["passed"] is False and attempt["hw"]["hardware_only"] is True


# ---- the report -----------------------------------------------------------------------------------------------


def report_for(tmp_path, fake_kenv, attempts):
    header = kp.report_header(env=fake_kenv, coder_model="m", probe_id="p", ops=["add"], tt_lang=FACTS,
                              hw={"available": True, "smoke_tail": ["PASSED!"]})
    kp.write_report(tmp_path / "out", attempts, header)
    return (tmp_path / "out" / "kernel-probe-report.md").read_text()


def test_the_report_says_a_hardware_only_pass_was_not_checked_by_the_simulator(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], []))
    md = report_for(tmp_path, fake_kenv, [attempt])
    row = next(l for l in md.splitlines() if l.startswith("| 1 |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1] == "no"                                   # simulator column
    assert cells[2] == "yes (hardware only)"                  # hardware column
    assert "none of 1 attempts passed in the simulator" in md  # not counted as a simulator pass
    assert "1 of 1 attempts that ran on hardware passed" in md
    assert "hardware only: 1 attempt (1 passed on hardware)" in md and "the simulator did not check" in md
    assert "attempts to first pass in the simulator: 1" not in md


def test_the_report_marks_a_failed_hardware_only_attempt_and_the_category(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([dict(GOOD_HW_SET, pcc=0.2)], []))
    md = report_for(tmp_path, fake_kenv, [attempt])
    row = next(l for l in md.splitlines() if l.startswith("| 1 |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == "no (hardware only)"
    assert "hardware only: 1 attempt (0 passed on hardware)" in md


def test_a_report_with_no_sim_gap_attempt_has_no_hardware_only_text(tmp_path, fake_kenv):
    adir, attempt = _passed_attempt(tmp_path, fake_kenv)
    kp.run_hardware([attempt], probe_dir=tmp_path, env=fake_kenv, shared_python="/p",
                    run_watched=hw_writer([GOOD_HW_SET], []))
    md = report_for(tmp_path, fake_kenv, [attempt])
    assert "hardware only" not in md and "simulator did not check" not in md


def test_a_sim_gap_attempt_that_never_reached_hardware_is_listed_as_not_judged(tmp_path, fake_kenv):
    adir, attempt = sim_gap_attempt(tmp_path, fake_kenv)
    md = report_for(tmp_path, fake_kenv, [attempt])
    assert "simulator lacks a function" in md and "not run on hardware" in md


def test_a_record_that_says_both_a_simulator_pass_and_a_sim_gap_is_not_hardware_only():
    assert kp.hardware_only({"passed_sim": True, "category": "sim_gap"}) is False
    assert kp.hardware_only({"passed_sim": False, "category": "sim_gap"}) is True
    assert kp.hardware_only({"passed_sim": False, "category": "tt_lang_error"}) is False


def test_the_attempt_line_names_the_sim_gap():
    line = kp._attempt_line(dict(canned("linear_scan", 2), category="sim_gap",
                                 first_line="the simulator has no ttl.raw_element_read"))
    assert line.startswith("linear_scan 2: sim_gap (the simulator has no ttl.raw_element_read)")


def test_the_skill_says_what_the_hardware_measurement_showed_about_the_two_functions():
    flat = " ".join(kp.SKILL.read_text().split())
    assert "ttl.raw_element_read(blk, i, j)" in flat and "ttl.raw_element_write(blk, i, j, value)" in flat
    assert "the simulator lacks them" in flat and "`FAIL: sim_gap`" in flat and "judged on hardware only" in flat
    assert "copied elements, wrote constants and shifted rows" in flat
    assert "refused `+`, `*`, `if` expressions and lists of values" in flat
    assert "`wait()` block" in flat
