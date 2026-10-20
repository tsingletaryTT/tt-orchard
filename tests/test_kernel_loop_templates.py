# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The kernel loop's templates: references, harness and check (orchard/skills/kernel-loop-templates).

Two kinds of test live here. The offline ones import kernel_check.py by path (it is stdlib only) and feed
`classify` text and result files that the simulator and the harness really wrote (tests/fixtures/kernel-loop/
outputs, recorded with tt-lang-sim 1.1.10; machine paths replaced by <ATTEMPT_DIR>, <TEMPLATES> and <TTLANG_VENV>).
The `needs_sim` ones copy the templates and a fixture kernel into tmp_path and run kernel_check.py against the
real simulator in the ttlang venv. They skip, and say so, when that venv is absent.
"""
import importlib.util
import json
import re
import os
import shutil
import signal as _signal_mod
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "kernel-loop-templates"
FIX = Path(__file__).parent / "fixtures" / "kernel-loop"
KERNELS = FIX / "kernels"
OUT = FIX / "outputs"

VENV = Path(os.environ.get("ORCHARD_TTLANG_VENV", Path.home() / "orchard-venvs" / "ttlang"))
SIM = VENV / "bin" / "tt-lang-sim"
needs_sim = pytest.mark.skipif(not SIM.exists(), reason="SKIPPED: no tt-lang-sim in the ttlang venv (run `tt-orchard kernel-env`); "
                                                        "the kernel loop was not run on the simulator")


def _load_check():
    spec = importlib.util.spec_from_file_location(
        "kernel_check_under_test", REPO / "orchard" / "skills" / "kernel-loop-templates" / "kernel_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CHECK = _load_check()


@pytest.mark.parametrize("name", ["numeric_mismatch", "shape_mismatch", "tt_lang_error", "import_error"])
def test_classify_reads_real_recorded_simulator_output(name):
    """Offline: the text and the result file are what the simulator and the harness really wrote (step 4)."""
    text = (OUT / f"{name}.txt").read_text()
    rp = OUT / f"{name}.result.json"
    result = json.loads(rp.read_text()) if rp.exists() else None
    got, line = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=result, bar=0.999)
    assert got == name and line


def test_without_the_diagnostics_file_the_check_reports_the_environment():
    """A tt-lang-sim wheel with no diagnostics.py ends every diagnosed kernel error as FileNotFoundError. That is a fault
    in the environment, so it must not be blamed on the kernel as a tt_lang_error."""
    text = ("Traceback (most recent call last):\n"
            "  File \"<TTLANG_VENV>/lib/python3.12/site-packages/ttl/sim/greenlet_scheduler.py\", line 338, in run\n"
            "FileNotFoundError: [Errno 2] No such file or directory: "
            "'<TTLANG_VENV>/lib/python3.12/site-packages/ttl/ttl/diagnostics.py'\n")
    result = {"phase": "sim", "sets": [],
              "error": {"phase": "run", "type": "FileNotFoundError",
                        "text": "[Errno 2] No such file or directory: '.../ttl/ttl/diagnostics.py'"}}
    got, line = CHECK.classify(kernel_present=True, timed_out=False, returncode=1, text=text, result=result, bar=0.999)
    assert got == "harness_error"
    assert "diagnostics.py" in line
    got, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=1, text=text, result=None, bar=0.999)
    assert got == "harness_error"


def test_classify_phase_hw_uses_the_hardware_names():
    """On hardware a runtime fault is compile_error and a low PCC is hw_numeric_mismatch (Task 9 calls it that way)."""
    text = (OUT / "tt_lang_error.txt").read_text()
    result = json.loads((OUT / "tt_lang_error.result.json").read_text())
    assert CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=result, bar=0.99,
                          phase="hw")[0] == "compile_error"
    result = json.loads((OUT / "numeric_mismatch.result.json").read_text())
    assert CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text="", result=result, bar=0.99,
                          phase="hw")[0] == "hw_numeric_mismatch"


# --- the simulator tests --------------------------------------------------------------------------------------------

def _attempt(tmp_path, kernel_name, op="add", spec_edit=None):
    """A copy of the templates and one fixture kernel, laid out the way an attempt directory is."""
    tdir = tmp_path / "templates"
    shutil.copytree(TEMPLATES, tdir, ignore=shutil.ignore_patterns("__pycache__"))
    adir = tmp_path / "attempt"
    adir.mkdir()
    if kernel_name is not None:
        shutil.copy(KERNELS / f"{kernel_name}.py", adir / "kernel.py")
    spec = json.loads((tdir / "ops" / f"{op}.json").read_text())
    if spec_edit:
        spec_edit(spec)
    (adir / "kernel_spec.json").write_text(json.dumps(spec))
    return tdir, adir


def _run_check(tdir, adir, *extra, timeout=360):
    done = subprocess.run([sys.executable, str(tdir / "kernel_check.py"), "--attempt-dir", str(adir), "--sim", str(SIM),
                           *extra], capture_output=True, text=True, timeout=timeout)
    check = json.loads((adir / "check.json").read_text())
    return done, check


@needs_sim
def test_add_good_passes(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_good")
    done, check = _run_check(tdir, adir)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == "PASS"
    assert check["passed"] is True and check["category"] is None
    assert all(s["pcc"] >= 0.999 for s in check["sets"])


@pytest.mark.parametrize("kernel,category,extra,op", [
    ("add_wrong_op", "numeric_mismatch", (), "add"),
    ("add_shape_error", "shape_mismatch", (), "add"),
    ("add_deadlock", "tt_lang_error", (), "add"),
    ("add_syntax_error", "import_error", (), "add"),
    ("add_undecorated_named", "no_kernel", (), "add"),
    ("add_no_def", "no_kernel", (), "add"),
    ("add_two_misnamed", "forbidden_code", (), "add"),
    ("add_spin", "timeout", ("--timeout-s", "5"), "add"),
    ("add_nan", "numeric_mismatch", (), "add"),
    # correlated with the right answer, so PCC alone passes them
    ("add_scaled", "numeric_mismatch", (), "add"),
    ("rmsnorm_no_norm", "numeric_mismatch", (), "rmsnorm"),
    ("rmsnorm_wrong_n", "numeric_mismatch", (), "rmsnorm"),
    # the static gate, one fixture per rule
    ("forbidden_import", "forbidden_code", (), "add"),
    ("forbidden_name", "forbidden_code", (), "add"),
    ("forbidden_dunder", "forbidden_code", (), "add"),
    ("forbidden_readback", "forbidden_code", (), "add"),
    ("forbidden_undecorated", "forbidden_code", (), "add"),
    # prints the missing-diagnostics text itself; diagnostics.py is present, so the kernel is blamed
    ("add_spoof_diagnostics", "tt_lang_error", (), "add"),
    ("forbidden_in_main", "forbidden_code", (), "add"),
    ("forbidden_readback_import", "forbidden_code", (), "add"),
    ("forbidden_stacked_decorator", "forbidden_code", (), "add"),
    ("forbidden_for_binding", "forbidden_code", (), "add"),
    ("forbidden_walrus", "forbidden_code", (), "add"),
    ("forbidden_typing_sys", "forbidden_code", (), "add"),
    ("forbidden_attr_os", "forbidden_code", (), "add"),
])
@needs_sim
def test_each_broken_kernel_gets_its_category(tmp_path, kernel, category, extra, op):
    tdir, adir = _attempt(tmp_path, kernel, op=op)
    done, check = _run_check(tdir, adir, *extra)
    assert done.returncode == 1, done.stdout + done.stderr
    assert done.stdout.startswith(f"FAIL: {category}: ")
    assert check["category"] == category and check["passed"] is False
    assert check["first_line"]


@needs_sim
def test_a_missing_kernel_is_no_kernel(tmp_path):
    tdir, adir = _attempt(tmp_path, None)
    done, check = _run_check(tdir, adir)
    assert done.returncode == 1
    assert check["category"] == "no_kernel" and check["passed"] is False


@needs_sim
def test_a_nan_output_is_recorded_as_not_finite(tmp_path):
    """The harness's `finite` flag must say False when the kernel writes NaN; a PCC of None alone is not enough
    for whoever reads check.json to tell a NaN from a constant output."""
    tdir, adir = _attempt(tmp_path, "add_nan")
    _, check = _run_check(tdir, adir)
    assert check["category"] == "numeric_mismatch"
    assert check["sets"] and all(s["finite"] is False for s in check["sets"])


@needs_sim
def test_the_heavy_tail_sets_run(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_good")
    _, check = _run_check(tdir, adir)
    names = [s["name"] for s in check["sets"]]
    assert "seed0" in names and "seed0-tail" in names
    assert len(names) == 6


@needs_sim
def test_a_spec_cannot_name_other_code(tmp_path):
    """The reference is a key into references.REFERENCES. A spec that names `os.system` must reach no code."""
    def edit(spec):
        spec["reference"] = "os.system"
    tdir, adir = _attempt(tmp_path, "add_good", spec_edit=edit)
    done, check = _run_check(tdir, adir)
    assert done.returncode == 1
    assert check["category"] == "harness_error", check
    assert check["sets"] == []


@needs_sim
def test_check_json_hashes_the_files_that_took_part(tmp_path):
    import hashlib
    tdir, adir = _attempt(tmp_path, "add_good")
    _, check = _run_check(tdir, adir)
    hashes = check["sha256"]
    assert set(hashes) == {"kernel.py", "kernel_harness.py", "references.py", "kernel_check.py", "spec"}
    assert all(isinstance(v, str) and len(v) == 64 for v in hashes.values())
    assert hashes["kernel.py"] == hashlib.sha256((adir / "kernel.py").read_bytes()).hexdigest()
    assert hashes["kernel_harness.py"] == hashlib.sha256((tdir / "kernel_harness.py").read_bytes()).hexdigest()


@needs_sim
def test_references_match_independent_formulas():
    """The references against hand-written loops on a 4x8 input, in the venv's python (this interpreter may lack torch).
    Importing torch there opens no device; the script never imports ttnn or ttl."""
    script = r"""
import math, sys
sys.path.insert(0, sys.argv[1])
import torch
from references import REFERENCES
g = torch.Generator().manual_seed(7)
A = torch.randn(4, 8, generator=g); B = torch.randn(4, 8, generator=g)
S = torch.sigmoid(torch.randn(4, 8, generator=g))
def close(x, y): return abs(x - y) <= 1e-5 * max(1.0, abs(y))
got = REFERENCES["silu_mul"]({"gate": A, "up": B})
for i in range(4):
    for j in range(8):
        g_ = float(A[i, j]); assert close(float(got[i, j]), g_ / (1 + math.exp(-g_)) * float(B[i, j])), ("silu_mul", i, j)
got = REFERENCES["rmsnorm"]({"x": A, "w": B})
for i in range(4):
    ms = sum(float(A[i, j]) ** 2 for j in range(8)) / 8
    for j in range(8):
        assert close(float(got[i, j]), float(A[i, j]) / math.sqrt(ms + 1e-6) * float(B[i, j])), ("rmsnorm", i, j)
got = REFERENCES["linear_scan"]({"a": S, "b": B})
for j in range(8):
    h = 0.0
    for t in range(4):
        h = float(S[t, j]) * h + float(B[t, j])
        assert close(float(got[t, j]), h), ("linear_scan", t, j)
got = REFERENCES["add"]({"a": A, "b": B})
assert torch.equal(got, A + B)
print("REFERENCES OK")
"""
    done = subprocess.run([str(VENV / "bin" / "python"), "-c", script, str(TEMPLATES)], capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0 and "REFERENCES OK" in done.stdout, done.stdout + done.stderr


@pytest.mark.parametrize("op", ["silu_mul", "rmsnorm"])
@needs_sim
def test_the_good_kernels_pass_on_the_simulator(tmp_path, op):
    tdir, adir = _attempt(tmp_path, f"{op}_good", op=op)
    done, check = _run_check(tdir, adir)
    assert done.returncode == 0, done.stdout + done.stderr
    assert check["passed"] is True and check["op"] == op
    assert len(check["sets"]) == 6 and all(s["pcc"] >= 0.999 and s["finite"] for s in check["sets"])


def test_the_op_specs_name_only_known_references():
    """Every spec under ops/ names a reference in the table and lists its inputs in the order the kernel takes them."""
    import ast
    tree = ast.parse((TEMPLATES / "references.py").read_text())
    table = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "REFERENCES")
    known = {k.value for k in table.value.keys}
    specs = sorted((TEMPLATES / "ops").glob("*.json"))
    assert {p.stem for p in specs} == {"add", "silu_mul", "rmsnorm", "linear_scan"}
    for p in specs:
        spec = json.loads(p.read_text())
        assert spec["op"] == p.stem and spec["reference"] in known
        assert spec["pcc_bar_sim"] == 0.999 and spec["seeds"] == [0, 1, 2] and spec["heavy_tail"] is True


# --- fix round 1: error bars beyond PCC, the static gate, the setup phase -------------------------------------------

SIM_BARS = {"pcc": 0.999, "rel_l2": 1e-2, "slope": [0.98, 1.02], "max_abs_atol": 1e-2, "max_abs_rtol": 1e-2}
HW_BARS = {"pcc": 0.99, "rel_l2": 5e-2, "slope": [0.95, 1.05], "max_abs_atol": 1e-1, "max_abs_rtol": 5e-2}


def _good_set(**over):
    s = {"name": "seed0", "seed": 0, "heavy_tail": False, "pcc": 0.99999, "max_abs_err": 0.001, "finite": True,
         "rel_l2": 0.001, "slope": 1.0, "exp_abs_max": 4.0}
    s.update(over)
    return s


def _classify_sets(sets, phase="sim", bars=None, **kw):
    args = dict(kernel_present=True, timed_out=False, returncode=0, text="", result={"phase": phase, "sets": sets,
                "error": None}, bar=(bars or (SIM_BARS if phase == "sim" else HW_BARS))["pcc"], phase=phase)
    args.update(kw)
    if bars is not None:
        args["bars"] = bars
    return CHECK.classify(**args)


def test_a_set_inside_every_bar_passes():
    assert _classify_sets([_good_set()], bars=SIM_BARS) == (None, "")


@pytest.mark.parametrize("over,needle", [
    ({"rel_l2": 0.05}, "rel_l2 0.05 above bar 0.01"),
    ({"slope": 2.0}, "slope 2.0 outside [0.98, 1.02]"),
    ({"slope": 0.5}, "slope 0.5 outside [0.98, 1.02]"),
    ({"max_abs_err": 0.2, "exp_abs_max": 4.0}, "max_abs_err 0.2 above"),
])
def test_each_new_bar_fails_a_set_alone(over, needle):
    """PCC passes in every case: only the named field is out of its bar, and the FAIL line names it."""
    category, line = _classify_sets([_good_set(**over)], bars=SIM_BARS)
    assert category == "numeric_mismatch"
    assert needle in line and line.startswith("set seed0: ")


@pytest.mark.parametrize("field", ["rel_l2", "slope", "max_abs_err", "exp_abs_max", "pcc"])
@pytest.mark.parametrize("how", ["missing", "null"])
def test_a_missing_or_null_field_fails(field, how):
    s = _good_set()
    if how == "missing":
        del s[field]
    else:
        s[field] = None
    category, line = _classify_sets([s], bars=SIM_BARS)
    assert category == "numeric_mismatch"
    assert field in line


def test_the_hw_bars_are_wider_and_use_the_hw_name():
    s = _good_set(slope=1.03, rel_l2=0.03)
    assert _classify_sets([s], phase="sim", bars=SIM_BARS)[0] == "numeric_mismatch"
    assert _classify_sets([s], phase="hw", bars=HW_BARS) == (None, "")
    assert _classify_sets([_good_set(slope=1.2)], phase="hw", bars=HW_BARS)[0] == "hw_numeric_mismatch"


def test_classify_without_bars_uses_the_shipped_defaults():
    """Task 9 may call classify with only `bar`: the new bars still apply, from the phase's defaults."""
    assert _classify_sets([_good_set(slope=2.0)])[0] == "numeric_mismatch"
    assert _classify_sets([_good_set()]) == (None, "")
    assert _classify_sets([_good_set(slope=1.03)], phase="hw") == (None, "")


def test_bars_from_spec_reads_each_phase():
    spec = json.loads((TEMPLATES / "ops" / "add.json").read_text())
    assert CHECK.bars_from_spec(spec, "sim") == SIM_BARS
    assert CHECK.bars_from_spec(spec, "hw") == HW_BARS


def test_every_op_spec_has_the_provisional_bars():
    for p in sorted((TEMPLATES / "ops").glob("*.json")):
        spec = json.loads(p.read_text())
        assert spec["provisional"] is True
        assert spec["rel_l2_bar_sim"] == 1e-2 and spec["rel_l2_bar_hw"] == 5e-2, p.name
        assert spec["slope_band_sim"] == [0.98, 1.02] and spec["slope_band_hw"] == [0.95, 1.05], p.name
        assert spec["max_abs_atol_sim"] == 1e-2 and spec["max_abs_rtol_sim"] == 1e-2, p.name
        assert spec["max_abs_atol_hw"] == 1e-1 and spec["max_abs_rtol_hw"] == 5e-2, p.name


def test_classify_timed_out_offline():
    got = CHECK.classify(kernel_present=True, timed_out=True, returncode=None, text="PROGRESS import\n", result=None,
                         bar=0.999)
    assert got[0] == "timeout"


@pytest.mark.parametrize("phase", ["sim", "hw"])
def test_a_setup_phase_error_is_the_harness(phase):
    """The device would not open: not the kernel's fault, in either phase."""
    result = {"phase": phase, "sets": [], "error": {"phase": "setup", "type": "RuntimeError",
                                                    "text": "Failed to open device 0: chip is busy"}}
    text = "PROGRESS import\nPROGRESS open_device\nRuntimeError: Failed to open device 0: chip is busy\n"
    category, line = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=result,
                                    bar=0.999, phase=phase)
    assert category == "harness_error" and "chip is busy" in line


def test_a_nan_set_fails_offline():
    s = _good_set(pcc=None, finite=False, rel_l2=None, slope=None, max_abs_err=float("nan"))
    category, line = _classify_sets([s], bars=SIM_BARS)
    assert category == "numeric_mismatch" and "finite False" in line


def test_a_missing_result_after_the_kernel_started_is_the_kernel():
    text = "PROGRESS import\nPROGRESS open_device\nPROGRESS set seed0\nPROGRESS run seed0\n"
    category, line = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=None,
                                    bar=0.999)
    assert category == "tt_lang_error"
    assert line == "the kernel ended the process without a result"
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=None,
                                 bar=0.99, phase="hw")
    assert category == "compile_error"


def test_a_missing_result_before_the_kernel_loaded_is_the_harness():
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=1, text="Traceback ...\n",
                                 result=None, bar=0.999)
    assert category == "harness_error"


SPOOF = ("FileNotFoundError: [Errno 2] No such file or directory: 'ttl/sim/diagnostics.py'\n"
         "PROGRESS import\n")


def test_spoofed_diagnostics_text_is_the_kernel_when_the_file_is_present():
    result = {"phase": "sim", "sets": [], "error": {"phase": "run", "type": "ValueError", "text": "boom"}}
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=SPOOF, result=result,
                                 bar=0.999, diagnostics_present=True)
    assert category == "tt_lang_error"
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=SPOOF, result=result,
                                 bar=0.999, diagnostics_present=False)
    assert category == "harness_error"


def test_diagnostics_file_lookup(tmp_path):
    """The file the wheel fails to ship is ttl/ttl/diagnostics.py. The wheel's loader, ttl/sim/diagnostics.py, is always
    there and reads the other one (ttl/sim/diagnostics.py lines 35-36; ttlang_env.diagnostics_target)."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    sim = venv / "bin" / "tt-lang-sim"
    sim.write_text("")
    assert CHECK.diagnostics_present(sim) is None                 # no ttl package found: unknown
    ttl = venv / "lib" / "python3.12" / "site-packages" / "ttl"
    (ttl / "ttl").mkdir(parents=True)
    assert CHECK.diagnostics_present(sim) is False
    (ttl / "ttl" / "diagnostics.py").write_text("")
    assert CHECK.diagnostics_present(sim) is True


def test_the_loader_alone_does_not_count_as_the_diagnostics_file(tmp_path):
    """I1. The loader ttl/sim/diagnostics.py is present in every wheel; with the target ttl/ttl/diagnostics.py absent,
    the file is missing."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    sim = venv / "bin" / "tt-lang-sim"
    sim.write_text("")
    ttl = venv / "lib" / "python3.12" / "site-packages" / "ttl"
    (ttl / "sim").mkdir(parents=True)
    (ttl / "sim" / "diagnostics.py").write_text("# the loader\n")
    assert CHECK.diagnostics_present(sim) is False


# the static gate

@pytest.mark.parametrize("kernel,rules", [
    ("forbidden_import", ["import"]),
    ("forbidden_name", ["name"]),
    ("forbidden_dunder", ["dunder"]),
    ("forbidden_readback", ["readback"]),
    ("forbidden_undecorated", ["operation"]),
    ("kernel_exit", ["import", "name"]),                   # `import os`, then `os._exit(0)`
    ("forbidden_in_main", ["name"]),
    ("forbidden_readback_import", ["readback", "readback"]),  # the import, then the bare use
    ("forbidden_stacked_decorator", ["operation"]),
    ("forbidden_for_binding", ["operation"]),
    ("forbidden_walrus", ["operation"]),
    # I4: `from typing import sys as s`, then `s.modules["builtins"].open(...)`
    ("forbidden_typing_sys", ["import", "name", "name"]),
    ("forbidden_attr_os", ["name"]),                       # ttnn.os
])
def test_the_gate_refuses_each_rule(kernel, rules):
    ok, reason = CHECK.check_kernel_source(KERNELS / f"{kernel}.py")
    assert ok is False
    n = len(rules)
    assert reason.startswith(f"{n} violation{'s' if n > 1 else ''}: {rules[0]} at line "), reason
    found = CHECK.kernel_source_violations(KERNELS / f"{kernel}.py")
    assert [v["rule"] for v in found] == rules
    assert all(isinstance(v["line"], int) and v["line"] > 0 and v["remedy"] for v in found)


@pytest.mark.parametrize("kernel", ["add_good", "silu_mul_good", "rmsnorm_good", "add_wrong_op", "add_nan", "add_scaled",
                                    "add_shape_error", "add_deadlock", "add_spin", "rmsnorm_no_norm", "rmsnorm_wrong_n",
                                    "add_spoof_diagnostics", "add_syntax_error", "add_alias_module", "add_alias_ttnn",
                                    "add_alias_function", "add_alias_function_as", "add_boilerplate"])
def test_the_gate_lets_plain_kernels_through(kernel):
    """add_syntax_error does not parse; the gate leaves it to the harness, which reports import_error."""
    assert CHECK.check_kernel_source(KERNELS / f"{kernel}.py") == (True, "")


def test_the_gate_reports_no_operation():
    ok, reason = CHECK.check_kernel_source(KERNELS / "add_no_def.py")
    assert ok is False and reason.startswith("no-operation")


NO_OPERATION = "no-operation: kernel.py defines no @ttl.operation function"


# ---- Task 14d: the operation is the one @ttl.operation function, whatever its name ------------------------------
# Three live runs (L4 linear_scan, L4c rmsnorm twice) named the function after the op although the skill said the name
# had to be OPERATION, so the harness stopped requiring the name.

def test_a_decorated_function_under_any_name_is_the_operation():
    assert CHECK.check_kernel_source(KERNELS / "add_misnamed.py") == (True, "")
    assert CHECK.operation_name(KERNELS / "add_misnamed.py") == "eltwise_add"


def test_a_function_named_operation_works_as_before():
    assert CHECK.check_kernel_source(KERNELS / "add_good.py") == (True, "")
    assert CHECK.operation_name(KERNELS / "add_good.py") == "OPERATION"


def test_the_operation_is_found_through_each_accepted_spelling_of_the_decorator(tmp_path):
    for header, deco in (("import ttl as tl", "@tl.operation(grid='full')"),
                         ("from ttl import operation", "@operation(grid='full')"),
                         ("from ttl import operation as op", "@op(grid='full')"),
                         ("import ttl", "@ttl.operation")):
        k = tmp_path / "kernel.py"
        k.write_text(f"{header}\nimport ttnn\n\n\n{deco}\ndef rmsnorm(x_in, w_in, out):\n    pass\n")
        assert CHECK.check_kernel_source(k) == (True, ""), header
        assert CHECK.operation_name(k) == "rmsnorm", header


@pytest.mark.parametrize("kernel", ["add_no_def", "add_undecorated_named"])
def test_without_a_decorated_function_the_kernel_is_no_kernel(kernel):
    ok, reason = CHECK.check_kernel_source(KERNELS / f"{kernel}.py")
    assert ok is False and reason == NO_OPERATION
    assert CHECK.operation_name(KERNELS / f"{kernel}.py") is None


def test_a_function_with_another_decorator_is_not_the_operation(tmp_path):
    k = tmp_path / "kernel.py"
    k.write_text("import ttl\nimport ttnn\n\n\n@ttl.compute()\ndef rmsnorm(x_in, w_in, out):\n    pass\n")
    assert CHECK.check_kernel_source(k) == (False, NO_OPERATION)


def test_two_decorated_functions_are_refused_by_name():
    found = CHECK.kernel_source_violations(KERNELS / "add_two_misnamed.py")
    ops = [v for v in found if v["rule"] == "operation"]
    assert len(ops) == 1 and "eltwise_add" in ops[0]["detail"] and "eltwise_add_again" in ops[0]["detail"]
    assert ops[0]["remedy"] == "define exactly one @ttl.operation function"
    assert CHECK.operation_name(KERNELS / "add_two_misnamed.py") is None


OP_HEADER = 'import ttl\nimport ttnn\n\n\n'


@pytest.mark.parametrize("name", ["open", "exec", "os", "sys", "references", "getattr", "to_torch", "from_torch",
                                  "input", "__init__", "__dunder__"])
def test_the_operations_name_may_not_be_one_the_gate_forbids(tmp_path, name):
    k = tmp_path / "kernel.py"
    k.write_text(OP_HEADER + f'@ttl.operation(grid="full")\ndef {name}(a, b, out):\n    pass\n')
    ok, reason = CHECK.check_kernel_source(k)
    assert ok is False and not reason.startswith("no-operation"), (name, reason)
    assert CHECK.operation_name(k) is None


@pytest.mark.parametrize("tail", ["\nkernel = None\n", "\nfor kernel in []:\n    pass\n", "\n\ndef kernel():\n    pass\n",
                                  "\nimport math as kernel\n"])
def test_the_operations_name_is_bound_only_by_its_def(tmp_path, tail):
    k = tmp_path / "kernel.py"
    k.write_text(OP_HEADER + '@ttl.operation(grid="full")\ndef kernel(a, b, out):\n    pass\n' + tail)
    ok, reason = CHECK.check_kernel_source(k)
    assert ok is False and "operation" in reason, reason


def test_the_operation_has_one_decorator_only(tmp_path):
    k = tmp_path / "kernel.py"
    k.write_text(OP_HEADER + 'def wrap(f):\n    return f\n\n\n@wrap\n@ttl.operation(grid="full")\n'
                 'def kernel(a, b, out):\n    pass\n')
    ok, reason = CHECK.check_kernel_source(k)
    assert ok is False and "decorated with ttl.operation and nothing else" in reason, reason


def test_a_file_that_does_not_parse_still_passes_the_gate():
    assert CHECK.check_kernel_source(KERNELS / "add_syntax_error.py") == (True, "")



GATE_HEADER = 'import ttl\nimport ttnn\n\n\n@ttl.operation(grid="full")\ndef OPERATION(a, b, out):\n'


@pytest.mark.parametrize("line,rule", [
    ("    import sys", "import"), ("    import os.path", "import"), ("    import atexit", "import"),
    ("    import builtins", "import"), ("    import subprocess", "import"), ("    import importlib", "import"),
    ("    import references", "import"), ("    import torch", "import"), ("    from . import x", "import"),
    ("    from references import REFERENCES", "import"),
    ("    open('f')", "name"), ("    exec('1')", "name"), ("    eval('1')", "name"), ("    compile('1', 'f', 'exec')", "name"),
    ("    x = __import__", "name"), ("    globals()", "name"), ("    locals()", "name"), ("    setattr(a, 'b', 1)", "name"),
    ("    getattr(a, 'b')", "name"), ("    breakpoint()", "name"), ("    input()", "name"), ("    x = sys", "name"),
    ("    x = os", "name"), ("    x = __builtins__", "name"), ("    x = references", "name"),
    ("    x = a.__dict__", "dunder"), ("    x = ttl.__spec__", "dunder"),
    ("    x = ttnn.to_torch", "readback"), ("    x = ttnn.from_torch", "readback"),
    # I4: a forbidden name imported under another name, or bound by an import as, or reached as an attribute
    ("    from ttnn import os as o", "name"), ("    from typing import Any as sys", "name"),
    ("    from ttl import Thing as builtins", "name"), ("    x = ttl.something.os", "name"),
    ("    x = a.subprocess", "name"), ("    x = a.open", "name"),
    # I4: from typing, only the type names a kernel plausibly uses (typing.re is a module in some versions)
    ("    from typing import re", "import"), ("    from typing import get_type_hints", "import"),
    ("    from __future__ import braces", "import"),
])
def test_the_gate_rules_in_detail(tmp_path, line, rule):
    p = tmp_path / "kernel.py"
    p.write_text(GATE_HEADER + line + "\n")
    ok, reason = CHECK.check_kernel_source(p)
    assert ok is False and reason.startswith(f"1 violation: {rule} at line 7: "), reason


def test_the_gate_allows_common_typing_names(tmp_path):
    """I4. The typing names a kernel plausibly uses still pass, with or without an alias that is not forbidden."""
    p = tmp_path / "kernel.py"
    p.write_text("from __future__ import annotations\nfrom typing import Any, Optional, Tuple as T, Callable\n"
                 + GATE_HEADER + "    x: Optional[Any] = None\n")
    assert CHECK.check_kernel_source(p) == (True, "")


def test_every_allowed_typing_name_is_not_a_module():
    """I4. The allow-list of typing names holds no module in the interpreter running the tests."""
    import inspect
    import typing
    for name in CHECK.TYPING_NAMES:
        assert not inspect.ismodule(getattr(typing, name, None)), name


@needs_sim
@pytest.mark.parametrize("kernel", ["forbidden_typing_sys", "forbidden_attr_os"])
def test_a_module_reached_through_typing_or_an_attribute_is_forbidden_code(tmp_path, kernel):
    """I4. The check ends forbidden_code and names the rule in its first line; nothing reaches the simulator."""
    tdir, adir = _attempt(tmp_path, kernel)
    done, check = _run_check(tdir, adir)
    assert done.stdout.startswith("FAIL: forbidden_code: "), done.stdout
    first = check["first_line"]
    assert re.match(r"\d+ violations?: (import|name) at line \d+: ", first), first
    assert "name at line" in first
    assert check["sets"] == [] and not (adir / "harness-result.json").exists()


def test_the_gate_allows_math(tmp_path):
    p = tmp_path / "kernel.py"
    p.write_text("import math\n" + GATE_HEADER + "    x = math.sqrt(2.0)\n")
    assert CHECK.check_kernel_source(p) == (True, "")


@pytest.mark.parametrize("src", [
    "import ttl\n\n\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\n@ttl.compute()\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\ndef f(a, b, out):\n    pass\n\n\nOPERATION = ttl.operation(grid='full')(f)\n",
    "import ttl\n\n\ndef outer():\n    @ttl.operation(grid='full')\n    def OPERATION(a, b, out):\n        pass\n\n\nOPERATION = 1\n",
])
def test_operation_must_be_a_decorated_top_level_def(tmp_path, src):
    p = tmp_path / "kernel.py"
    p.write_text(src)
    ok, reason = CHECK.check_kernel_source(p)
    assert ok is False and "operation at line " in reason, reason
    assert {v["rule"] for v in CHECK.kernel_source_violations(p)} == {"operation"}


def test_a_bare_operation_decorator_is_allowed(tmp_path):
    p = tmp_path / "kernel.py"
    p.write_text("import ttl\n\n\n@ttl.operation\ndef OPERATION(a, b, out):\n    pass\n")
    assert CHECK.check_kernel_source(p) == (True, "")


@needs_sim
def test_a_forbidden_kernel_never_reaches_the_simulator(tmp_path):
    tdir, adir = _attempt(tmp_path, "forbidden_import")
    done, check = _run_check(tdir, adir)
    assert done.stdout.startswith("FAIL: forbidden_code: 1 violation: import at line ")
    assert [(v["rule"], v["line"]) for v in check["violations"]] == [("import", 7)]
    assert check["sets"] == [] and "PROGRESS" not in (adir / "harness-output.txt").read_text()
    assert not (adir / "harness-result.json").exists()


@needs_sim
def test_the_new_fixtures_fail_on_the_named_fields(tmp_path):
    """Each correlated-but-wrong kernel is failed by a bar other than PCC, and its PCC is above the PCC bar."""
    for kernel, op in (("add_scaled", "add"), ("rmsnorm_wrong_n", "rmsnorm")):
        tdir, adir = _attempt(tmp_path / kernel, kernel, op=op)
        _, check = _run_check(tdir, adir)
        assert check["category"] == "numeric_mismatch"
        assert all(s["pcc"] >= 0.999 for s in check["sets"]), check["sets"]
        assert any(f in check["first_line"] for f in ("rel_l2", "slope", "max_abs_err")), check["first_line"]
    tdir, adir = _attempt(tmp_path / "no_norm", "rmsnorm_no_norm", op="rmsnorm")
    _, check = _run_check(tdir, adir)
    plain = next(s for s in check["sets"] if s["name"] == "seed0")
    assert plain["pcc"] >= 0.999 and plain["rel_l2"] > 1e-2


@needs_sim
def test_check_json_records_the_bars_and_the_new_fields(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_good")
    _, check = _run_check(tdir, adir)
    assert check["bars"] == SIM_BARS
    for s in check["sets"]:
        assert s["rel_l2"] is not None and s["rel_l2"] <= 1e-2
        assert 0.98 <= s["slope"] <= 1.02
        assert s["exp_abs_max"] > 0


@needs_sim
def test_a_zero_timeout_is_a_timeout(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_good")
    _, check = _run_check(tdir, adir, "--timeout-s", "0")
    assert check["category"] == "timeout"


def _fake_stack(tmp_path):
    """A stand-in ttnn whose open_device raises, and a ttl whose decorators do nothing. No real ttnn is imported."""
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "ttnn.py").write_text(
        "TILE_LAYOUT = 'tile'\nbfloat16 = 'bf16'\nTensor = object\n"
        "def open_device(device_id=0):\n    raise RuntimeError('fake: device 0 is busy')\n"
        "def close_device(device):\n    pass\n")
    (fake / "ttl.py").write_text("def operation(*a, **k):\n    return lambda f: f\n")
    return fake


@needs_sim
def test_a_device_that_will_not_open_is_the_harness(tmp_path):
    fake = _fake_stack(tmp_path)
    out = tmp_path / "result.json"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = str(fake)
    done = subprocess.run([str(VENV / "bin" / "python"), str(TEMPLATES / "kernel_harness.py"), "--spec",
                           str(TEMPLATES / "ops" / "add.json"), "--kernel", str(KERNELS / "add_good.py"), "--out",
                           str(out), "--phase", "hw"], capture_output=True, text=True, timeout=120, env=env)
    result = json.loads(out.read_text())
    assert result["error"]["phase"] == "setup", result
    assert "device 0 is busy" in result["error"]["text"]
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=done.returncode,
                                 text=done.stdout + done.stderr, result=result, bar=0.99, phase="hw")
    assert category == "harness_error"


@needs_sim
def test_a_kernel_that_ends_the_process_is_blamed(tmp_path):
    """kernel_exit.py calls os._exit(0) inside OPERATION. The gate refuses it, so this runs the harness directly."""
    out = tmp_path / "result.json"
    done = subprocess.run([str(SIM), str(TEMPLATES / "kernel_harness.py"), "--", "--spec", str(TEMPLATES / "ops" / "add.json"),
                           "--kernel", str(KERNELS / "kernel_exit.py"), "--out", str(out), "--phase", "sim"],
                          capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert not out.exists()
    text = done.stdout + done.stderr
    assert "PROGRESS import" in text
    markers = [l for l in text.splitlines() if l.startswith("PROGRESS ")]
    assert markers[-1] == "PROGRESS run seed0", markers
    category, line = CHECK.classify(kernel_present=True, timed_out=False, returncode=done.returncode, text=text,
                                    result=None, bar=0.999)
    assert (category, line) == ("tt_lang_error", "the kernel ended the process without a result")


@needs_sim
def test_an_input_error_after_a_kernel_ran_is_still_the_harness(tmp_path):
    """The second seed cannot seed a generator. The kernel has already run once, and the fault is still not its own."""
    def edit(spec):
        spec["seeds"] = [0, "not-a-seed"]
    tdir, adir = _attempt(tmp_path, "add_good", spec_edit=edit)
    _, check = _run_check(tdir, adir)
    assert check["category"] == "harness_error", check
    assert [s["name"] for s in check["sets"]] == ["seed0", "seed0-tail"]



# --- fix round 2: every violation at once, harmless boilerplate, aliases, markers ------------------------------------

UPSTREAM = KERNELS / "upstream_eltwise_add.py"     # tt-lang examples/eltwise_add.py, copied byte for byte


def test_the_upstream_example_is_flagged_only_for_torch_and_readback():
    """The example the agent is told to read: its kernel is fine, so only `import torch` and the host code's
    from_torch/to_torch calls may be flagged, each with a remedy. `__name__` and the main block are not flagged."""
    found = CHECK.kernel_source_violations(UPSTREAM)
    assert [v["rule"] for v in found] == ["import", "readback", "readback", "readback", "readback"], found
    assert found[0]["line"] == 4 and "torch" in found[0]["detail"]
    assert all(v["remedy"] for v in found)
    assert all(v["remedy"] == "the harness reads the result back; remove the host code and keep only the "
               "@ttl.operation function"
               for v in found if v["rule"] == "readback")
    ok, reason = CHECK.check_kernel_source(UPSTREAM)
    assert ok is False and reason.startswith("5 violations: import at line 4: ")


def _src(tmp_path, text):
    p = tmp_path / "kernel.py"
    p.write_text(text)
    return p


def test_the_gate_reports_every_violation(tmp_path):
    p = _src(tmp_path, "import os\nimport sys\n" + GATE_HEADER + "    open('f')\n    x = a.__dict__\n")
    found = CHECK.kernel_source_violations(p)
    assert [(v["rule"], v["line"]) for v in found] == [("import", 1), ("import", 2), ("name", 9), ("dunder", 10)]
    assert all(set(v) >= {"rule", "line", "remedy"} for v in found)
    ok, reason = CHECK.check_kernel_source(p)
    assert reason.startswith("4 violations: import at line 1: ")
    assert reason.count(" at line ") == 4


def test_a_long_list_is_capped_at_eight(tmp_path):
    body = "".join(f"    open('f{i}')\n" for i in range(11))
    ok, reason = CHECK.check_kernel_source(_src(tmp_path, GATE_HEADER + body))
    assert reason.startswith("11 violations: ")
    assert reason.count(" at line ") == 8 and reason.endswith("and 3 more")


def test_every_rule_has_a_remedy():
    assert set(CHECK.REMEDIES) == {"import", "name", "dunder", "readback", "operation"}
    assert all(isinstance(r, str) and r for r in CHECK.REMEDIES.values())


@pytest.mark.parametrize("src", [
    "from __future__ import annotations\n" + GATE_HEADER + "    pass\n",
    "from typing import Any\n" + GATE_HEADER + "    pass\n",
    GATE_HEADER + "    pass\n\n\n__all__ = ['OPERATION']\n",
    GATE_HEADER + "    pass\n\n\nif __name__ == '__main__':\n    print('host code')\n",
])
def test_harmless_boilerplate_passes_the_gate(tmp_path, src):
    assert CHECK.check_kernel_source(_src(tmp_path, src)) == (True, "")


@pytest.mark.parametrize("line", ["    input = a", "    y = a\n    input_blk = y", "    x = input if False else a"])
def test_input_is_a_legal_variable_name(tmp_path, line):
    assert CHECK.check_kernel_source(_src(tmp_path, GATE_HEADER + line + "\n")) == (True, "")


def test_input_as_a_parameter_name_is_legal(tmp_path):
    src = 'import ttl\n\n\n@ttl.operation(grid="full")\ndef OPERATION(input, b, out):\n    x = input\n'
    assert CHECK.check_kernel_source(_src(tmp_path, src)) == (True, "")


@pytest.mark.parametrize("src", [
    "import ttl as tl\n\n\n@tl.operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl as tl\n\n\n@tl.operation\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttl import operation\n\n\n@operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttl import operation as op\n\n\n@op(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttl import operation as op\n\n\n@op\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttnn as tn\nimport ttl\n\n\n@ttl.operation(grid='full')\ndef OPERATION(a: tn.Tensor, b, out):\n    pass\n",
])
def test_the_legal_aliases_pass_the_gate(tmp_path, src):
    assert CHECK.check_kernel_source(_src(tmp_path, src)) == (True, "")


@pytest.mark.parametrize("src", [
    "import ttnn\n\n\n@ttnn.operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttnn as tl\n\n\n@tl.operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\n@ttl.operation_like(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\n@ttl.operations\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\n@my_operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttnn import operation\n\n\n@operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttl import operation\n\n\ndef operation(**k):\n    return lambda f: f\n\n\n@operation(grid='full')\n"
    "def OPERATION(a, b, out):\n    pass\n",
    "import ttl as tl\n\ntl = None\n\n\n@tl.operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "from ttl import operation as op\nop = 1\n\n\n@op(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
    "import ttl\n\n\n@ttl.compute.operation(grid='full')\ndef OPERATION(a, b, out):\n    pass\n",
])
def test_a_decorator_that_is_not_ttl_operation_is_refused(tmp_path, src):
    found = CHECK.kernel_source_violations(_src(tmp_path, src))
    assert "operation" in {v["rule"] for v in found}, found


@pytest.mark.parametrize("src", [
    "from ttnn import to_torch\n" + GATE_HEADER + "    pass\n",
    "from ttnn import from_torch as ft\n" + GATE_HEADER + "    pass\n",
    "from ttl import from_torch\n" + GATE_HEADER + "    pass\n",
    GATE_HEADER + "    to_torch(a)\n",
])
def test_readback_names_are_refused_however_they_arrive(tmp_path, src):
    assert {v["rule"] for v in CHECK.kernel_source_violations(_src(tmp_path, src))} == {"readback"}


@pytest.mark.parametrize("src", ["from ttnn import *\n" + GATE_HEADER + "    pass\n",
                                 "from ttl import *\n" + GATE_HEADER + "    pass\n"])
def test_star_imports_are_refused(tmp_path, src):
    assert {v["rule"] for v in CHECK.kernel_source_violations(_src(tmp_path, src))} == {"import"}


@pytest.mark.parametrize("tail", [
    "for OPERATION in [OPERATION]:\n    pass\n",
    "with open_ctx() as OPERATION:\n    pass\n",
    "if (OPERATION := OPERATION):\n    pass\n",
    "x = [(OPERATION := y) for y in [1]]\n",
    "if True:\n    OPERATION = 1\n",
    "del OPERATION\n",
    "try:\n    pass\nexcept Exception as OPERATION:\n    pass\n",
    "if True:\n    @ttl.operation(grid='full')\n    def OPERATION(a, b, out):\n        pass\n",
    "import ttl as OPERATION\n",
    "class OPERATION:\n    pass\n",
    "def f():\n    global OPERATION\n    OPERATION = 1\n",
])
def test_operation_bound_any_other_way_is_refused(tmp_path, tail):
    found = CHECK.kernel_source_violations(_src(tmp_path, GATE_HEADER + "    pass\n\n\n" + tail))
    assert [v["rule"] for v in found] == ["operation"], found


@pytest.mark.parametrize("decorators", ["@staticmethod\n@ttl.operation(grid='full')\n",
                                        "@ttl.operation(grid='full')\n@staticmethod\n",
                                        "@ttl.operation(grid='full')\n@ttl.operation(grid='full')\n"])
def test_a_stacked_decorator_is_refused(tmp_path, decorators):
    """Another decorator above or below ttl.operation, or ttl.operation twice: OPERATION would not be the
    operation ttl built."""
    src = "import ttl\n\n\n" + decorators + "def OPERATION(a, b, out):\n    pass\n"
    found = CHECK.kernel_source_violations(_src(tmp_path, src))
    assert [v["rule"] for v in found] == ["operation"]


def test_a_local_name_operation_inside_a_function_is_fine(tmp_path):
    src = GATE_HEADER + "    OPERATION = 1\n"          # a local inside OPERATION's body binds nothing at module scope
    assert CHECK.check_kernel_source(_src(tmp_path, src)) == (True, "")


@pytest.mark.parametrize("text,blamed", [
    ("PROGRESS import\n", True),
    ("PROGRESS import\nPROGRESS open_device\n", False),
    ("PROGRESS import\nPROGRESS open_device\nPROGRESS set seed0\n", False),
    ("PROGRESS import\nPROGRESS open_device\nPROGRESS set seed0\nPROGRESS run seed0\n", True),
    ("PROGRESS import\nPROGRESS open_device\nPROGRESS set seed0\nPROGRESS run seed0\nsome kernel print\n", True),
    ("PROGRESS import\nPROGRESS run seed0\nPROGRESS set seed0-tail\n", False),
    ("", False),
])
def test_the_last_marker_decides_who_ended_the_process(text, blamed):
    category, _ = CHECK.classify(kernel_present=True, timed_out=False, returncode=0, text=text, result=None, bar=0.999)
    assert category == ("tt_lang_error" if blamed else "harness_error")


def test_a_nan_pcc_fails():
    category, line = _classify_sets([_good_set(pcc=float("nan"))], bars=SIM_BARS)
    assert category == "numeric_mismatch" and "pcc nan" in line


def _venv_with_diagnostics(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    sim = venv / "bin" / "tt-lang-sim"
    sim.write_text("#!/bin/sh\n")
    sim.chmod(0o755)
    pkg = venv / "lib" / "python3.12" / "site-packages" / "ttl" / "ttl"
    pkg.mkdir(parents=True)
    (pkg / "diagnostics.py").write_text("")
    return sim


def test_diagnostics_lookup_follows_a_symlink(tmp_path):
    sim = _venv_with_diagnostics(tmp_path)
    links = tmp_path / "links"
    links.mkdir()
    (links / "tt-lang-sim").symlink_to(sim)
    assert CHECK.diagnostics_present(links / "tt-lang-sim") is True


def test_diagnostics_lookup_finds_a_bare_command_on_path(tmp_path, monkeypatch):
    sim = _venv_with_diagnostics(tmp_path)
    monkeypatch.setenv("PATH", str(sim.parent))
    assert CHECK.diagnostics_present("tt-lang-sim") is True
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert CHECK.diagnostics_present("tt-lang-sim") is None


@pytest.mark.parametrize("kernel", ["add_alias_module", "add_alias_ttnn", "add_alias_function", "add_alias_function_as",
                                    "add_boilerplate"])
@needs_sim
def test_the_alias_and_boilerplate_kernels_pass_on_the_simulator(tmp_path, kernel):
    """add_boilerplate's main block raises SystemExit; it passes only because the harness never runs that block."""
    tdir, adir = _attempt(tmp_path, kernel)
    done, check = _run_check(tdir, adir)
    assert done.returncode == 0, done.stdout + done.stderr
    assert check["passed"] is True and check["violations"] == []


@needs_sim
def test_check_json_lists_every_violation(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_good")
    shutil.copy(UPSTREAM, adir / "kernel.py")
    done, check = _run_check(tdir, adir)
    assert check["category"] == "forbidden_code"
    assert check["first_line"].startswith("5 violations: ")
    assert [v["rule"] for v in check["violations"]] == ["import"] + ["readback"] * 4


# m2: a simulator that times out is stopped with everything it started.

def _dead_or_zombie(pid):
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return True
    return text[text.rfind(")") + 2:].split()[0] == "Z"


def test_a_simulator_timeout_stops_the_grandchild_too(tmp_path):
    """A stand-in for tt-lang-sim starts a grandchild in its own process group and sleeps. The check's timeout must
    stop both. Every process here is one this test started; a leftover is killed in the finally only while it is
    alive and still carries this test's marker."""
    tdir, adir = _attempt(tmp_path, "add_good")
    marks = tmp_path / "pids"
    marks.mkdir()
    sim = tmp_path / "fake-sim"
    sim.write_text("#!" + sys.executable + "\n"
                   "import os, subprocess, sys, time\n"
                   f"marks = {str(marks)!r}\n"
                   "open(os.path.join(marks, 'sim'), 'w').write(str(os.getpid()))\n"
                   "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                   "open(os.path.join(marks, 'grandchild'), 'w').write(str(g.pid))\n"
                   "time.sleep(60)\n")
    sim.chmod(0o755)
    pids = []
    marker = f"orchard-m2-{os.getpid()}-{time.monotonic_ns()}"
    try:
        done = subprocess.run([sys.executable, str(tdir / "kernel_check.py"), "--attempt-dir", str(adir), "--sim",
                               str(sim), "--timeout-s", "3"], capture_output=True, text=True, timeout=60,
                              env=dict(os.environ, ORCHARD_TEST_MARKER=marker))
        pids = [int((marks / n).read_text()) for n in ("sim", "grandchild") if (marks / n).is_file()]
        assert len(pids) == 2, done.stdout + done.stderr
        assert done.stdout.startswith("FAIL: timeout"), done.stdout + done.stderr
        end = time.monotonic() + 10
        while time.monotonic() < end and not all(_dead_or_zombie(p) for p in pids):
            time.sleep(0.05)
        assert all(_dead_or_zombie(p) for p in pids), [p for p in pids if not _dead_or_zombie(p)]
    finally:
        for p in pids:
            _kill_if_ours(p, marker)


@pytest.mark.parametrize("pid", ["own group", 1, 0, -3, None])
def test_the_simulator_kill_never_signals_a_group_that_is_not_the_childs_own(monkeypatch, pid):
    """os.killpg is replaced by a recorder, so no real signal can be sent whatever the guard decides."""
    sent = []
    monkeypatch.setattr(os, "killpg", lambda *a: sent.append(a))
    killed = []
    proc = type("P", (), {"pid": os.getpgrp() if pid == "own group" else pid, "kill": lambda self: killed.append(1)})()
    CHECK._kill_own_group(proc)
    assert sent == [] and killed == [1]


# R1: when kernel_check itself is stopped, the simulator (in its own session since m2) must not be orphaned.

def _ours_alive(pid, marker):
    """True only when `pid` is alive, not a zombie, and its environment still carries this test's marker, so a pid the
    kernel has reused for another process is never treated as ours."""
    if _dead_or_zombie(pid):
        return False
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            return marker.encode() in f.read()
    except OSError:
        return False


def _kill_if_ours(pid, marker):
    if _ours_alive(pid, marker):
        try:
            os.kill(pid, 9)
        except (ProcessLookupError, PermissionError):
            pass


def _start_check_with_fake_sim(tmp_path, marker, preexec_fn=None):
    """kernel_check.py as this test's own child (its own session), running a stand-in simulator that records its pid,
    starts a grandchild and sleeps. Returns (check process, marks directory)."""
    tdir, adir = _attempt(tmp_path, "add_good")
    marks = tmp_path / "pids"
    marks.mkdir()
    sim = tmp_path / "fake-sim"
    sim.write_text("#!" + sys.executable + "\n"
                   "import os, subprocess, sys, time\n"
                   f"marks = {str(marks)!r}\n"
                   "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                   "open(os.path.join(marks, 'grandchild'), 'w').write(str(g.pid))\n"
                   "open(os.path.join(marks, 'sim'), 'w').write(str(os.getpid()))\n"
                   "time.sleep(60)\n")
    sim.chmod(0o755)
    env = dict(os.environ, ORCHARD_TEST_MARKER=marker)
    proc = subprocess.Popen([sys.executable, str(tdir / "kernel_check.py"), "--attempt-dir", str(adir), "--sim",
                             str(sim), "--timeout-s", "50"], env=env, start_new_session=True,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=preexec_fn)
    end = time.monotonic() + 30
    while time.monotonic() < end and not (marks / "sim").is_file():
        time.sleep(0.05)
    assert (marks / "sim").is_file(), "the stand-in simulator never started"
    return proc, marks


def _pids(marks):
    return [int((marks / n).read_text()) for n in ("sim", "grandchild") if (marks / n).is_file()]


def _all_gone(pids, marker, wait_s=10.0):
    end = time.monotonic() + wait_s
    while time.monotonic() < end and any(_ours_alive(p, marker) for p in pids):
        time.sleep(0.05)
    return [p for p in pids if _ours_alive(p, marker)]


def test_sigterm_to_kernel_check_stops_the_simulator_and_its_grandchild(tmp_path):
    marker = f"orchard-r1-{os.getpid()}-{time.monotonic_ns()}"
    proc, marks = None, None
    try:
        proc, marks = _start_check_with_fake_sim(tmp_path, marker)
        proc.send_signal(_signal_mod.SIGTERM)                 # our own child, by its Popen handle
        proc.wait(timeout=20)
        assert _all_gone(_pids(marks), marker) == []
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)
        for p in (_pids(marks) if marks else []):
            _kill_if_ours(p, marker)


def test_sigkill_of_kernel_check_still_stops_the_simulator(tmp_path):
    """No handler can run after SIGKILL; the simulator dies through its parent-death signal."""
    marker = f"orchard-r1k-{os.getpid()}-{time.monotonic_ns()}"
    proc, marks = None, None
    try:
        proc, marks = _start_check_with_fake_sim(tmp_path, marker)
        proc.kill()
        proc.wait(timeout=20)
        sim_pid = int((marks / "sim").read_text())
        assert _all_gone([sim_pid], marker) == []
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)
        for p in (_pids(marks) if marks else []):
            _kill_if_ours(p, marker)


def test_the_cleanup_helper_leaves_a_process_without_the_marker_alone(tmp_path):
    """S2. A pid the kernel reused for another process must never be killed: the helper checks the marker first."""
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True,
                             env=dict(os.environ, ORCHARD_TEST_MARKER="someone-else"))
    try:
        end = time.monotonic() + 10
        while time.monotonic() < end and not _ours_alive(other.pid, "someone-else"):
            time.sleep(0.02)
        _kill_if_ours(other.pid, f"orchard-s2-{os.getpid()}")
        time.sleep(0.2)
        assert other.poll() is None
        _kill_if_ours(other.pid, "someone-else")
        assert other.wait(timeout=5) == -9
    finally:
        other.kill()
        other.wait(timeout=5)



# M2: smaller guards around the simulator.

def test_a_simulator_that_was_already_reaped_is_never_signalled(monkeypatch):
    """After communicate() has reaped the simulator its pid may be reused; nothing may be looked up or signalled."""
    calls = []
    monkeypatch.setattr(os, "killpg", lambda *a: calls.append(("killpg", a)))
    monkeypatch.setattr(os, "getpgid", lambda *a: calls.append(("getpgid", a)) or a[0])
    proc = type("P", (), {"pid": 424242, "returncode": 0, "kill": lambda self: calls.append(("kill",))})()
    CHECK._kill_own_group(proc)
    assert calls == []


def test_a_simulator_whose_parent_is_already_gone_exits_before_it_runs(tmp_path):
    """The parent-death signal only covers a parent that dies after prctl. The child compares its parent with the
    pid recorded before the fork and exits at once when they differ. A fake parent pid stands in for a dead parent."""
    marker = f"orchard-m2p-{os.getpid()}-{time.monotonic_ns()}"
    env = dict(os.environ, ORCHARD_TEST_MARKER=marker)
    orphan = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env,
                              preexec_fn=CHECK._die_with_parent_fn(parent=os.getpid() + 999999))
    normal = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env,
                              preexec_fn=CHECK._die_with_parent_fn())
    try:
        assert orphan.wait(timeout=10) == 1
        time.sleep(0.3)
        assert normal.poll() is None
    finally:
        for p in (orphan, normal):
            p.kill()
            p.wait(timeout=5)


def test_a_sighup_that_starts_ignored_stays_ignored(tmp_path):
    """kernel_check started with SIGHUP ignored (as under nohup) keeps ignoring it; SIGTERM still stops it."""
    marker = f"orchard-m2h-{os.getpid()}-{time.monotonic_ns()}"
    proc, marks = None, None

    def ignore_hup():
        _signal_mod.signal(_signal_mod.SIGHUP, _signal_mod.SIG_IGN)
    try:
        proc, marks = _start_check_with_fake_sim(tmp_path, marker, preexec_fn=ignore_hup)
        proc.send_signal(_signal_mod.SIGHUP)
        time.sleep(0.5)
        assert proc.poll() is None
        proc.send_signal(_signal_mod.SIGTERM)
        proc.wait(timeout=20)
        assert _all_gone(_pids(marks), marker) == []
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)
        for p in (_pids(marks) if marks else []):
            _kill_if_ours(p, marker)


def test_the_check_docstring_says_what_an_interrupt_does():
    doc = " ".join(CHECK.__doc__.split())
    assert "interrupt" in doc and "no check.json" in doc


@pytest.mark.parametrize("sig", [_signal_mod.SIGTERM, _signal_mod.SIGHUP], ids=["SIGTERM", "SIGHUP"])
def test_an_interrupted_check_exits_with_the_signal_that_stopped_it(tmp_path, sig):
    """A1. A watcher reading the exit status must see the signal that arrived (-15 for SIGTERM, -1 for SIGHUP), not
    the SIGINT exit a bare KeyboardInterrupt gives; the simulator and its grandchild are gone, and no check.json."""
    marker = f"orchard-a1-{os.getpid()}-{time.monotonic_ns()}"
    proc, marks = None, None

    def default_dispositions():
        # W9: the child starts with SIGHUP and SIGINT at their defaults even when pytest itself runs under nohup;
        # test_a_sighup_that_starts_ignored_stays_ignored covers an inherited SIG_IGN.
        for s in (_signal_mod.SIGHUP, _signal_mod.SIGINT):
            _signal_mod.signal(s, _signal_mod.SIG_DFL)
    try:
        proc, marks = _start_check_with_fake_sim(tmp_path, marker, preexec_fn=default_dispositions)
        proc.send_signal(sig)
        assert proc.wait(timeout=20) == -sig
        assert _all_gone(_pids(marks), marker) == []
        assert not (tmp_path / "attempt" / "check.json").exists()
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)
        for p in (_pids(marks) if marks else []):
            _kill_if_ours(p, marker)


@needs_sim
def test_a_slice_counted_in_elements_is_refused_by_the_simulator_with_the_tiles_message(tmp_path):
    """Task 14c: tensor slices count tiles. A slice counted in elements reaches past the tensor, and the simulator
    says so in tiles; the kernel-probe skill quotes this message."""
    tdir, adir = _attempt(tmp_path, "add_element_slices")
    done, check = _run_check(tdir, adir)
    assert check["category"] == "tt_lang_error"
    assert re.search(r"IndexError: row slice \d+:\d+ is outside the tensor, which has \d+ tile\(s\) along it",
                     check["first_line"]), check["first_line"]


# ---- Task 14d: the harness calls the function the gate found -----------------------------------------------------

def _harness():
    spec = importlib.util.spec_from_file_location("kernel_harness_under_test", TEMPLATES / "kernel_harness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_harness_takes_the_operation_the_gate_names():
    from types import SimpleNamespace
    harness = _harness()
    module = SimpleNamespace(eltwise_add="the operation", OPERATION=None, helper="not it")
    assert harness.operation_of(module, KERNELS / "add_misnamed.py") == "the operation"
    module = SimpleNamespace(OPERATION="the old name")
    assert harness.operation_of(module, KERNELS / "add_good.py") == "the old name"
    for kernel in ("add_undecorated_named.py", "add_two_misnamed.py", "add_no_def.py"):
        assert harness.operation_of(SimpleNamespace(eltwise_add="x", OPERATION="y"), KERNELS / kernel) is None, kernel


@needs_sim
def test_a_kernel_with_a_freely_named_operation_passes_the_agents_check(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_misnamed")
    done, check = _run_check(tdir, adir)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == "PASS" and check["passed"] is True


# --- where: the source line the simulator blamed (pilot 2, 2026-10-17) ---------------------------------------------
# The coder's own check printed only `FAIL: <category>: <one line>`. The simulator's pointer to the failing line sat
# in harness-output.txt, which the coder never reads, so it fixed errors without knowing which line raised them.

def _fixture_text(name):
    return (OUT / name).read_text(encoding="utf-8")


def test_where_names_the_kernel_line_the_simulator_blamed():
    # Recorded in pilot 2: a compute-thread error in a kernel.py.
    assert CHECK.source_pointer(_fixture_text("binary-op-shape-with-pointer.txt")) == \
        "kernel.py line 87: out_blk.store(x_v * bcast * w_v)"
    assert CHECK.source_pointer(_fixture_text("shape_mismatch.txt")) == "kernel.py line 46: tx_a = ttl.copy(a[r, c], a_block)"


def test_where_takes_the_first_diagnostic_when_several_threads_blocked():
    assert CHECK.source_pointer(_fixture_text("tt_lang_error.txt")) == "kernel.py line 44: a_dfb.wait() as a_blk,"


@pytest.mark.parametrize("text", [
    "",
    "PROGRESS import\nsome kernel print\n",
    # a pointer into the simulator's own code is not the coder's line
    "error: ValueError: x\n  --> /venv/site-packages/ttl/sim/dfb.py:891:9\n   |\n891 |     raise ValueError(\n   |\n",
    # a pointer with no excerpt line
    "error: ValueError: x\n  --> /w/kernel.py:7:1\n   |\n",
])
def test_where_is_empty_without_a_kernel_line(text):
    assert CHECK.source_pointer(text) == ""


def test_where_is_one_bounded_line_without_the_machine_path():
    long = "x = " + "a" * 500
    out = CHECK.source_pointer(f"error: E\n  --> /secret/home/dir/kernel.py:3:1\n   |\n3 |     {long}\n   |\n")
    assert out.startswith("kernel.py line 3: x = aaa") and "\n" not in out and "/secret" not in out and len(out) <= 200


@needs_sim
def test_a_failing_check_prints_where_under_the_unchanged_fail_line(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_shape_error", op="add")
    done, check = _run_check(tdir, adir)
    lines = done.stdout.splitlines()
    assert lines[0].startswith("FAIL: shape_mismatch: ")          # the first line is still the one the probe reads
    assert len(lines) == 2 and re.fullmatch(r"where: kernel\.py line \d+: .+", lines[1]), done.stdout
    assert check["first_line"] == lines[0].split(": ", 2)[2]       # check.json is unchanged by the extra line


def test_where_follows_only_the_first_pointer_even_when_a_later_one_is_in_kernel_py():
    text = ("error: E1\n  --> /venv/ttl/sim/dfb.py:9:1\n   |\n9 |     raise\n   |\n"
            "error: E2\n  --> /w/kernel.py:7:1\n   |\n7 |     y = 1\n   |\n")
    assert CHECK.source_pointer(text) == ""


# --- hint: tensor slices count tiles (pilot 3, 2026-10-17) -------------------------------------------------------
# With the failing line in front of it, the coder still passed element coordinates (`row slice 0:32`) to a tensor of
# 2 tiles and did not fix it in 20 turns. The message says "tile(s)" and never says that slices count tiles.

REAL_INDEX_ERROR = "error: IndexError: row slice 0:32 is outside the tensor, which has 2 tile(s) along it"


def test_unit_hint_says_slices_count_tiles_and_names_the_extent():
    hint = CHECK.unit_hint(REAL_INDEX_ERROR)
    assert hint.startswith("tensor slices count tiles, not elements")
    assert "32x32" in hint and "2 tile(s) along the row axis" in hint and "within 0:2" in hint and "0:32" in hint


def test_unit_hint_names_the_axis_the_error_names():
    assert "along the col axis" in CHECK.unit_hint("error: IndexError: col slice 8:16 is outside the tensor, "
                                                    "which has 8 tile(s) along it")


@pytest.mark.parametrize("text", [
    "",
    "error: ValueError: Shape mismatch in binary operation: left shape (2, 1) does not match right shape (2, 8).",
    # a different unit is not the element-for-tile mistake
    "error: IndexError: row slice 0:3 is outside the tensor, which has 2 element(s) along it",
])
def test_unit_hint_is_empty_for_other_errors(text):
    assert CHECK.unit_hint(text) == ""


@needs_sim
def test_a_failing_check_prints_the_unit_hint_after_where(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_element_slices")
    done, check = _run_check(tdir, adir)
    lines = done.stdout.splitlines()
    assert lines[0].startswith("FAIL: tt_lang_error: ") and lines[1].startswith("where: kernel.py line ")
    assert len(lines) == 3 and lines[2].startswith("hint: tensor slices count tiles, not elements"), done.stdout


# --- numbers: what the numeric mismatch looked like (L4e, 2026-10-17) -------------------------------------------------
# Two sampled attempts at different seeds wrote the same wrong rmsnorm (the mean divided by 8 tiles, not 256 elements).
# Both got `FAIL: numeric_mismatch: set seed0: pcc 0.9796, finite True, bar 0.999` and never saw that their output was
# about 964 times too large, a number the check had in its own result. The check now prints it, with the worst
# expected-and-got pairs, on a line under the FAIL line.

def _torch():
    return pytest.importorskip("torch")


def test_metrics_lists_the_worst_positions_with_expected_and_got():
    torch = _torch()
    exp = torch.zeros(4, 5)
    got = torch.zeros(4, 5)
    got[2, 3] = 50.0
    got[0, 1] = -7.0
    got[3, 4] = 2.0
    worst = _harness().metrics(got, exp, torch)["worst"]
    assert [w["at"] for w in worst] == [[2, 3], [0, 1], [3, 4]]
    assert [w["got"] for w in worst] == [50.0, -7.0, 2.0] and all(w["expected"] == 0.0 for w in worst)


def test_metrics_lists_at_most_three_and_puts_non_finite_values_first():
    torch = _torch()
    exp = torch.ones(2, 6)
    got = torch.ones(2, 6) * 2
    got[1, 5] = float("nan")
    got[0, 0] = float("inf")
    worst = _harness().metrics(got, exp, torch)["worst"]
    assert len(worst) == 3 and {tuple(w["at"]) for w in worst[:2]} == {(1, 5), (0, 0)}
    assert all(w["got"] is None for w in worst[:2])                    # nan and inf are not JSON numbers: recorded as null
    json.dumps(worst, allow_nan=False)


def test_metrics_worst_follows_the_shape_of_a_one_dimensional_output():
    torch = _torch()
    got = torch.zeros(10)
    got[7] = 3.0
    assert _harness().metrics(got, torch.zeros(10), torch)["worst"][0]["at"] == [7]


SET = {"name": "seed0", "seed": 0, "heavy_tail": False, "pcc": 0.9796, "max_abs_err": 9094.4, "finite": True,
       "rel_l2": 983.1, "slope": 964.0, "exp_abs_max": 6.4,
       "worst": [{"at": [0, 12], "expected": 0.12, "got": 9094.5}, {"at": [3, 4], "expected": -1.5, "got": 800.25},
                 {"at": [9, 9], "expected": 0.5, "got": 700.0}]}
BARS = {"pcc": 0.999, "rel_l2": 0.01, "slope": [0.98, 1.02], "max_abs_atol": 0.01, "max_abs_rtol": 0.01}


def detail(sets, bars=BARS, bar=0.999):
    return CHECK.numeric_detail(sets, bar, bars)


def test_numbers_says_how_many_times_too_big_and_names_the_worst_places():
    line = detail([SET])
    assert line.startswith("seed0: ")
    assert "about 964x expected" in line and "pcc 0.9796" in line and "rel_l2 983" in line
    assert "max_abs_err 9094" in line and "expected values reach 6.4" in line
    assert "[0, 12]: expected 0.12, got 9094.5" in line and "[3, 4]: expected -1.5, got 800.25" in line
    assert "[9, 9]" not in line                                           # two places are enough on one line
    assert "\n" not in line and len(line) < 420


def test_numbers_says_too_small_opposite_sign_and_right_scale_differently():
    assert "about 0.0058x expected" in detail([dict(SET, slope=0.0058)]).replace("0.00580", "0.0058")
    assert "opposite sign" in detail([dict(SET, slope=-0.9)])
    right = detail([dict(SET, slope=1.0, pcc=0.4)])
    assert "scale is about right" in right and "pcc 0.4" in right


def test_numbers_says_when_the_output_is_not_finite():
    line = detail([dict(SET, finite=False, pcc=None, slope=None, rel_l2=None, worst=[{"at": [1, 1], "expected": 1.0, "got": None}])])
    assert "not finite" in line and "[1, 1]: expected 1, got nan or inf" in line


def test_numbers_names_the_first_failing_set_and_counts_the_rest():
    good = dict(SET, name="seed1", pcc=1.0, slope=1.0, rel_l2=0.0, max_abs_err=0.0)
    line = detail([good, SET, dict(SET, name="seed2")])
    assert line.split(":")[0] == "seed0"                                   # the first set that fails, not the first set
    assert "2 of 3 input sets fail" in line
    assert "1 of 2 input sets fail" in detail([good, SET])                 # the count appears when there is more than one set
    assert "input sets fail" not in detail([SET])                          # and not for a single set


def test_numbers_works_for_a_result_with_no_worst_list():
    old = {k: v for k, v in SET.items() if k != "worst"}
    line = detail([old])
    assert "about 964x expected" in line and "worst" not in line


def test_numbers_is_empty_when_no_set_fails_or_there_are_none():
    good = dict(SET, pcc=1.0, slope=1.0, rel_l2=0.0, max_abs_err=0.0)
    assert detail([good]) == "" and detail([]) == ""


def test_numbers_reads_the_recorded_numeric_fixture():
    result = json.loads((OUT / "numeric_mismatch.result.json").read_text())
    line = detail(result["sets"], bar=0.999)
    assert "slope" in line.lower() or "x expected" in line


@needs_sim
def test_a_numeric_mismatch_prints_numbers_under_the_unchanged_fail_line(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_wrong_op", op="add")
    done, check = _run_check(tdir, adir)
    lines = done.stdout.splitlines()
    assert lines[0].startswith("FAIL: numeric_mismatch: set ") and check["first_line"] == lines[0].split(": ", 2)[2]
    assert len(lines) == 2 and lines[1].startswith("numbers: seed0"), done.stdout
    assert re.search(r"x expected|scale is about right|opposite sign", lines[1]), lines[1]
    assert "worst" in lines[1] and "expected" in lines[1] and "got" in lines[1]


@needs_sim
def test_a_non_numeric_failure_prints_no_numbers_line(tmp_path):
    tdir, adir = _attempt(tmp_path, "add_shape_error", op="add")
    done, _ = _run_check(tdir, adir)
    assert not any(l.startswith("numbers:") for l in done.stdout.splitlines())


@pytest.mark.parametrize("slope,expect", [(1.3, "about 1.3x expected"), (0.6, "about 0.6x expected"), (1.2, "scale is about right"),
                                          (0.85, "scale is about right"), (3.0, "about 3x expected")])
def test_numbers_calls_a_scale_error_from_a_quarter_over_or_a_fifth_under(slope, expect):
    assert expect in detail([dict(SET, slope=slope)])


def test_numbers_names_the_worst_first_and_keeps_the_order_the_harness_gave():
    line = detail([SET])
    assert line.index("[0, 12]") < line.index("[3, 4]")


def test_numbers_prints_nothing_for_a_non_finite_set_with_no_numbers_beyond_what_it_has():
    line = detail([{"name": "seed0", "finite": False, "pcc": None}])
    assert line.startswith("seed0: got is not finite") and "pcc" not in line and "rel_l2" not in line
