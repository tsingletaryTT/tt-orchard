# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The kernel probe's capability ladder (issue 175): the rungs, the failure classes, the wall budget rule, the report
table and the loop over the rungs. The tests of `kernel-probe --ladder` through kernelprobe.main are in
test_kernelprobe.py (they need that file's fixtures). The simulator tests skip, and say so, without the ttlang venv."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import defaults, kernelladder as kl

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "kernel-loop-templates"
KERNELS = Path(__file__).parent / "fixtures" / "kernel-loop" / "kernels"
VENV = Path(os.environ.get("ORCHARD_TTLANG_VENV", Path.home() / "orchard-venvs" / "ttlang"))
SIM = VENV / "bin" / "tt-lang-sim"
needs_sim = pytest.mark.skipif(not SIM.exists(), reason="SKIPPED: no tt-lang-sim in the ttlang venv (run `tt-orchard kernel-env`)")


# ---- the rungs ----------------------------------------------------------------------------------------------------

def test_the_ladder_runs_from_easy_to_hard_and_every_rung_is_a_probe_op_with_a_spec():
    assert defaults.KERNEL_PROBE_LADDER == ("add", "mul", "scale_shift", "add_row", "rsqrt", "row_sum", "center",
                                           "rmsnorm", "linear_scan")
    for op in defaults.KERNEL_PROBE_LADDER:
        assert op in defaults.KERNEL_PROBE_OPS
        spec = json.loads((TEMPLATES / "ops" / f"{op}.json").read_text())
        assert spec["op"] == op and spec["reference"] == op


def test_every_rung_has_the_same_simulator_bars_and_input_seeds():
    """One bar for every rung, so a pass means the same thing on each (the rungs differ in the op, not in the strictness)."""
    keys = ("pcc_bar_sim", "rel_l2_bar_sim", "slope_band_sim", "max_abs_atol_sim", "max_abs_rtol_sim", "seeds",
            "heavy_tail", "dtype")
    specs = {op: json.loads((TEMPLATES / "ops" / f"{op}.json").read_text()) for op in defaults.KERNEL_PROBE_LADDER}
    first = specs["add"]
    for op, spec in specs.items():
        assert {k: spec[k] for k in keys} == {k: first[k] for k in keys}, op


def test_the_new_references_compute_what_their_specs_say():
    script = """
import sys
sys.path.insert(0, sys.argv[1])
import torch
from references import REFERENCES
g = torch.Generator().manual_seed(3)
A = torch.randn(64, 256, generator=g); B = torch.randn(64, 256, generator=g); R = torch.randn(32, 256, generator=g)
assert torch.equal(REFERENCES["mul"]({"a": A, "b": B}), A * B)
assert torch.allclose(REFERENCES["scale_shift"]({"x": A}), A * 2.5 + 0.5)
got = REFERENCES["add_row"]({"x": A, "r": R})
for i in (0, 31, 32, 63):
    for j in (0, 5, 255):
        assert abs(float(got[i, j]) - (float(A[i, j]) + float(R[i % 32, j]))) < 1e-6, ("add_row", i, j)
P = torch.sigmoid(A)
assert torch.allclose(REFERENCES["rsqrt"]({"x": P}), 1 / torch.sqrt(P))
got = REFERENCES["row_sum"]({"x": A})
assert got.shape == (64, 32) and torch.count_nonzero(got[:, 1:]) == 0
for i in (0, 17, 63):
    assert abs(float(got[i, 0]) - sum(float(A[i, j]) for j in range(256))) < 1e-3, ("row_sum", i)
print("LADDER REFERENCES OK")
"""
    py = VENV / "bin" / "python"
    if not py.exists():
        pytest.skip("SKIPPED: no ttlang venv with torch")
    done = subprocess.run([str(py), "-c", script, str(TEMPLATES)], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and "LADDER REFERENCES OK" in done.stdout, done.stdout + done.stderr


# Wrong kernels: each changes one thing in a good kernel, and the unedited check must refuse it. A bar that passes
# these would count a wrong kernel as a pass.
WRONG = [
    ("mul", "a_blk * b_blk", "a_blk + b_blk"),
    ("scale_shift", "x_blk * scale + shift", "x_blk * scale"),
    ("scale_shift", "ttl.block.fill(2.5, shape=(1, col_tiles))", "ttl.block.fill(2.0, shape=(1, col_tiles))"),
    ("add_row", "x_blk + r_blk", "x_blk"),
    ("rsqrt", "ttl.math.rsqrt(x_blk)", "ttl.math.sqrt(x_blk)"),
    ("row_sum", "ttl.math.reduce_sum(x_blk, dims=[-1], shape=(1, 1))",
     "ttl.math.reduce_sum(x_blk * x_blk, dims=[-1], shape=(1, 1))"),
]


@pytest.mark.parametrize("op, old, new", WRONG, ids=[f"{w[0]}-{i}" for i, w in enumerate(WRONG)])
@needs_sim
def test_the_check_refuses_a_wrong_kernel_for_each_new_rung(tmp_path, op, old, new):
    import shutil
    tdir = tmp_path / "templates"
    shutil.copytree(TEMPLATES, tdir, ignore=shutil.ignore_patterns("__pycache__"))
    adir = tmp_path / "attempt"
    adir.mkdir()
    text = (KERNELS / f"{op}_good.py").read_text()
    assert old in text
    (adir / "kernel.py").write_text(text.replace(old, new))
    shutil.copy(tdir / "ops" / f"{op}.json", adir / "kernel_spec.json")
    done = subprocess.run([sys.executable, str(tdir / "kernel_check.py"), "--attempt-dir", str(adir), "--sim", str(SIM)],
                          capture_output=True, text=True, timeout=360)
    check = json.loads((adir / "check.json").read_text())
    assert check["passed"] is False and check["category"], (done.stdout, check)


# ---- failure classes ----------------------------------------------------------------------------------------------

def attempt(passed=False, category=None, first="", status="done", detail="", wall=100.0, n=1, seed=None, temp=0.7):
    return {"op": "add", "attempt": n, "passed_sim": passed, "category": category, "first_line": first,
            "status": status, "status_detail": detail, "agent_wall_s": wall, "turns": 5,
            "sampling": {"temperature": temp, "top_p": None, "seed": seed}}


@pytest.mark.parametrize("rec, expected", [
    (attempt(True), None),
    (attempt(category="tt_lang_error", first="Deadlock detected: all generators blocked"), "deadlock"),
    (attempt(category="tt_lang_error", first="Cannot perform copy (as destination): not a valid next dataflow step"),
     "dataflow_step"),
    (attempt(category="tt_lang_error", first="pop(): not a valid next dataflow step"), "dataflow_step"),
    (attempt(category="shape_mismatch", first="Shape mismatch in store()"), "shape_mismatch"),
    (attempt(category="tt_lang_error", first="broadcast shape mismatch at dimension 1"), "shape_mismatch"),
    (attempt(category="numeric_mismatch", first="pcc 0.9903, bar 0.999"), "numeric"),
    (attempt(category="tt_lang_error", first="TypeError: unsupported operand type(s)"), "tt_lang_error"),
    (attempt(category="import_error", first="ModuleNotFoundError: ttl.dfb_reset"), "import_error"),
    (attempt(category="forbidden_code", first="FAIL: forbidden_code: import os"), "forbidden_code"),
    (attempt(category="no_kernel", first="kernel.py does not exist", status="done"), "no_kernel"),
    (attempt(category="no_kernel", first="kernel.py does not exist", status="wall"), "wall_clock"),
    (attempt(category=None, first="", status="wall"), "wall_clock"),
    (attempt(category="numeric_mismatch", first="pcc 0.9, bar 0.999", status="wall"), "numeric"),
    (attempt(category="tt_lang_error", first="TypeError: unsupported operand", status="wall"), "tt_lang_error"),
    (attempt(category="timeout", first="the simulator timed out"), "other"),
])
def test_failure_class(rec, expected):
    assert kl.failure_class(rec) == expected


def test_every_class_failure_class_returns_is_one_the_table_lists():
    recs = [attempt(category=c, first=f) for c, f in [("tt_lang_error", "Deadlock"), ("timeout", "x"),
                                                      ("numeric_mismatch", "p"), ("import_error", "m")]]
    assert {kl.failure_class(r) for r in recs} <= set(kl.FAILURE_CLASSES)


# ---- the wall budget ----------------------------------------------------------------------------------------------

def test_the_wall_budget_is_four_times_the_slowest_passing_pilot_between_300_and_1800():
    assert kl.size_wall_budget([200.0]) == 800.0
    assert kl.size_wall_budget([200.0, 350.0]) == 1400.0           # the slowest pass sets it
    assert kl.size_wall_budget([30.0]) == 300.0                    # the floor
    assert kl.size_wall_budget([700.0]) == 1800.0                  # the cap
    assert kl.size_wall_budget([]) == 1800.0                       # no pass to scale from: the cap
    assert kl.size_wall_budget([None, 0, -5]) == 1800.0            # records without a usable wall time count as none


def test_the_projection_adds_the_overhead_of_each_attempt():
    plan = [{"attempts": 10, "expected_s": 100.0}, {"attempts": 2, "expected_s": 500.0}]
    assert kl.projected_s(plan) == 10 * (100.0 + kl.ATTEMPT_OVERHEAD_S) + 2 * (500.0 + kl.ATTEMPT_OVERHEAD_S)
    assert kl.projected_s([]) == 0.0


# ---- the options --------------------------------------------------------------------------------------------------

def test_parse_rungs():
    assert kl.parse_rungs(None) == list(defaults.KERNEL_PROBE_LADDER)
    assert kl.parse_rungs("rsqrt, add") == ["rsqrt", "add"]
    with pytest.raises(kl.LadderRefused, match="not a rung"):
        kl.parse_rungs("add,conv2d")
    with pytest.raises(kl.LadderRefused, match="twice"):
        kl.parse_rungs("add,add")


def test_parse_seconds_map():
    assert kl.parse_seconds_map("add=300, mul=450.5", "--rung-wall") == {"add": 300.0, "mul": 450.5}
    assert kl.parse_seconds_map(None, "--rung-wall") == {}
    for bad in ("add", "add=", "add=x", "conv=5", "add=0", "add=-1", "add=nan", "add=9000", "add=5,add=6"):
        with pytest.raises(kl.LadderRefused):
            kl.parse_seconds_map(bad, "--rung-wall")


# ---- the table ----------------------------------------------------------------------------------------------------

def write_rung(out: Path, op: str, recs: list[dict], *, wall=1800.0, library="none"):
    d = out / op
    d.mkdir(parents=True)
    header = {"probe_id": f"p-{op}", "coder_model": "coder-x", "budget": {"wall_s": wall, "max_turns": 40},
              "sampling": {"temperatures": [0.7], "seed_base": 77, "seed_base_from": "--seed"},
              "library": {"mode": library}, "tt_lang": {"sim_version": "1.1.10", "source_sha": "a" * 40}}
    for r in recs:
        r["op"] = op
    (d / "kernel-probe-report.json").write_text(json.dumps({"header": header, "attempts": recs}))


def test_the_report_has_a_row_per_rung_with_passes_classes_and_settings(tmp_path):
    write_rung(tmp_path, "add", [attempt(True, n=i, seed=77 + i, wall=60.0 + i) for i in (1, 2, 3)], wall=300.0)
    write_rung(tmp_path, "center", [attempt(True, n=1, seed=78), attempt(category="numeric_mismatch", first="pcc 0.99", n=2, seed=79),
                                    attempt(category="tt_lang_error", first="Deadlock detected", n=3, seed=80)])
    write_rung(tmp_path, "rmsnorm", [attempt(category="tt_lang_error", first="x is not a valid next dataflow step", n=1, seed=78),
                                     attempt(category="no_kernel", first="", status="wall", n=2, seed=79)])
    data = kl.write_ladder_report(tmp_path, list(defaults.KERNEL_PROBE_LADDER))
    rows = {r["op"]: r for r in data["rungs"]}
    assert rows["add"]["attempts"] == 3 and rows["add"]["passed"] == 3 and rows["add"]["classes"] == {}
    assert rows["add"]["median_pass_wall_s"] == 62.0 and rows["add"]["wall_budget_s"] == 300.0
    assert rows["center"]["classes"] == {"deadlock": 1, "numeric": 1}
    assert rows["rmsnorm"]["classes"] == {"dataflow_step": 1, "wall_clock": 1} and rows["rmsnorm"]["passed"] == 0
    assert data["first_zero_rung"] == "rmsnorm"
    assert data["not_run"] == ["mul", "scale_shift", "add_row", "rsqrt", "row_sum", "linear_scan"]
    text = (tmp_path / "ladder-report.md").read_text()
    assert "| 1 | add | 3 | 3 | none | 62 | 300 | 78 to 80 |" in text
    assert "| 7 | center | 3 | 1 | deadlock 1, numeric 1 |" in text
    assert "The first rung where passes fall to zero: `rmsnorm` (rung 8 of 9)." in text
    assert "Rungs not run: mul, scale_shift, add_row, rsqrt, row_sum, linear_scan." in text
    assert "- coder model: coder-x" in text and "- seed base: 77" in text and "- temperatures: [0.7]" in text
    assert "- library: ['none']" in text and "- sim version: 1.1.10" in text
    assert "rmsnorm 1" in text and "wall_clock" in text                # the wall-clock note names the rung


def test_a_ladder_with_a_pass_on_every_rung_has_no_zero_rung(tmp_path):
    write_rung(tmp_path, "add", [attempt(True)])
    data = kl.write_ladder_report(tmp_path, ["add"])
    assert data["first_zero_rung"] is None
    assert "No rung that ran had zero passes." in (tmp_path / "ladder-report.md").read_text()


def test_the_first_zero_rung_is_the_first_in_ladder_order_not_the_order_the_rungs_ran(tmp_path):
    write_rung(tmp_path, "rmsnorm", [attempt(category="numeric_mismatch", first="p")])
    write_rung(tmp_path, "mul", [attempt(category="numeric_mismatch", first="p")])
    assert kl.write_ladder_report(tmp_path, ["rmsnorm", "mul"])["first_zero_rung"] == "mul"


def test_a_rung_with_a_pass_after_a_zero_rung_does_not_hide_the_zero(tmp_path):
    write_rung(tmp_path, "add", [attempt(True)])
    write_rung(tmp_path, "mul", [attempt(category="numeric_mismatch", first="p")])
    write_rung(tmp_path, "rsqrt", [attempt(True)])
    assert kl.write_ladder_report(tmp_path, ["add", "mul", "rsqrt"])["first_zero_rung"] == "mul"


# ---- the loop -----------------------------------------------------------------------------------------------------

class Args:
    def __init__(self, **kw):
        self.op = None
        self.out = None
        self.attempts = 10
        self.seed = "100"
        self.temperature = "0.7"
        self.temperatures = None
        self.wall_s = None
        self.ladder = True
        self.rungs = None
        self.rung_wall = None
        self.rung_expect_s = None
        self.max_hours = None
        self.report_only = False
        self.__dict__.update(kw)


def fake_main(calls, code_for=None, *, passes=True):
    def main(sub):
        calls.append({"op": sub.op, "out": sub.out, "wall_s": sub.wall_s, "ladder": sub.ladder, "seed": sub.seed,
                      "attempts": sub.attempts, "temperature": sub.temperature})
        write_rung(Path(sub.out).parent, sub.op, [attempt(passes, category=None if passes else "numeric_mismatch", first="" if passes else "p",
                                                         n=i, seed=100 + i) for i in (1, 2)], wall=sub.wall_s)
        return (code_for or {}).get(sub.op, 0)
    return main


def test_run_calls_main_once_per_rung_in_order_with_the_same_sampling_and_the_rungs_own_wall_budget(tmp_path):
    calls = []
    out = []
    code = kl.run(Args(out=str(tmp_path), rungs="add,rsqrt,rmsnorm", rung_wall="add=300,rsqrt=500", wall_s=1200.0),
                  fake_main(calls), default_wall_s=1800.0, say=out.append)
    assert code == 0
    assert [c["op"] for c in calls] == ["add", "rsqrt", "rmsnorm"]
    assert [c["wall_s"] for c in calls] == [300.0, 500.0, 1200.0]
    assert [Path(c["out"]).name for c in calls] == ["add", "rsqrt", "rmsnorm"]
    assert all(Path(c["out"]).parent == tmp_path and c["ladder"] is False for c in calls)
    assert {(c["seed"], c["attempts"], c["temperature"]) for c in calls} == {("100", 10, "0.7")}   # one sampling for all
    assert (tmp_path / "ladder-report.md").is_file() and any("ladder report" in l for l in out)


def test_run_uses_the_probe_default_wall_budget_when_none_is_given(tmp_path):
    calls = []
    kl.run(Args(out=str(tmp_path), rungs="add"), fake_main(calls), default_wall_s=1800.0, say=lambda l: None)
    assert calls[0]["wall_s"] == 1800.0


def test_run_stops_at_a_rung_that_returns_a_code_and_still_writes_the_report(tmp_path):
    calls = []
    code = kl.run(Args(out=str(tmp_path), rungs="add,mul,rsqrt"), fake_main(calls, {"mul": 3}), default_wall_s=1800.0,
                  say=lambda l: None)
    assert code == 3 and [c["op"] for c in calls] == ["add", "mul"]
    data = json.loads((tmp_path / "ladder-report.json").read_text())
    assert "rung mul ended with exit code 3" in data["stopped"]
    assert "The ladder stopped: rung mul ended" in (tmp_path / "ladder-report.md").read_text()


def test_run_stops_when_the_projection_is_above_max_hours_and_says_which_rungs_did_not_run(tmp_path):
    calls = []
    clock = iter([0.0, 0.0, 600.0, 600.0, 600.0])
    code = kl.run(Args(out=str(tmp_path), rungs="add,mul,rmsnorm", max_hours=1.0, rung_wall="add=300,mul=300,rmsnorm=1800"),
                  fake_main(calls), default_wall_s=1800.0, say=lambda l: None, clock=lambda: next(clock))
    # after add: 600 s elapsed + mul 10 x 360 + rmsnorm 10 x 1860 = 22,800 s = 6.3 hours, above 1
    assert code == 0 and [c["op"] for c in calls] == ["add"]
    data = json.loads((tmp_path / "ladder-report.json").read_text())
    assert "above --max-hours 1" in data["stopped"] and "not run: mul, rmsnorm" in data["stopped"]


def test_run_keeps_going_when_the_projection_is_inside_max_hours(tmp_path):
    calls = []
    kl.run(Args(out=str(tmp_path), rungs="add,mul", max_hours=6.0, rung_wall="add=300,mul=300", attempts=2),
           fake_main(calls), default_wall_s=1800.0, say=lambda l: None)
    assert [c["op"] for c in calls] == ["add", "mul"]


def test_run_uses_expected_seconds_for_the_projection_when_given(tmp_path):
    calls = []
    clock = iter([0.0, 0.0, 10.0, 10.0, 10.0, 10.0])
    kl.run(Args(out=str(tmp_path), rungs="add,rmsnorm", max_hours=1.0, rung_wall="add=300,rmsnorm=1800",
                rung_expect_s="rmsnorm=60"), fake_main(calls), default_wall_s=1800.0, say=lambda l: None,
           clock=lambda: next(clock))
    assert [c["op"] for c in calls] == ["add", "rmsnorm"]          # 10 x (60 + 60) s is 0.3 hours


@pytest.mark.parametrize("kw, words", [
    ({"op": "add"}, "do not give --op"),
    ({"rungs": "conv2d"}, "not a rung"),
    ({"rung_wall": "add=x"}, "not a number"),
    ({"rung_wall": "mul=300", "rungs": "add"}, "does not run"),
    ({"rung_expect_s": "mul=300", "rungs": "add"}, "does not run"),
    ({"max_hours": 0}, "above 0"),
    ({"max_hours": "soon"}, "not a number"),
])
def test_run_refuses_bad_options_before_running_anything(tmp_path, capsys, kw, words):
    calls = []
    code = kl.run(Args(out=str(tmp_path), **kw), fake_main(calls), default_wall_s=1800.0, say=lambda l: None)
    assert code == 2 and calls == [] and words in capsys.readouterr().err
    assert not (tmp_path / "ladder-report.md").exists()


def test_report_only_writes_the_table_from_the_rung_directories_and_runs_nothing(tmp_path):
    write_rung(tmp_path, "add", [attempt(True)])
    calls = []
    code = kl.run(Args(out=str(tmp_path), report_only=True, rungs="add"), fake_main(calls), default_wall_s=1800.0,
                  say=lambda l: None)
    assert code == 0 and calls == [] and "| 1 | add | 1 | 1 |" in (tmp_path / "ladder-report.md").read_text()
