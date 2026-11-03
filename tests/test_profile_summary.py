# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard profile` (issue 113, step 1): the bounded summary of a profiler run, tested on fake profiler output with no device.

The fake files use the real column names of tt-metal's ops CSV (tools/tracy/process_ops_logs.py) and of `tt-perf-report --csv`. The tests state:
the sums and groups come out right, a missing column or cell is `null` and is named (never 0), the summary stays under its size cap whatever the
input holds, and a run with no profiler, no free board, no CSV or a failed command gives a summary with `could_not_measure` and the exit code that
says so. The hostile-input corpus runs over the readers in tests/test_hostile_readers.py (registry in tests/hostile_inputs.py).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path

import pytest

from orchard import bench_control, cli, profile_summary as ps, safeio

RAW_HEADER = ["OP CODE", "OP TYPE", "GLOBAL CALL COUNT", "DEVICE ID", "DEVICE ARCH", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]",
              "DRAM BW UTIL PER CTRL (%)", "NPE CONG IMPACT (%)"]


def raw_csv(rows, header=RAW_HEADER) -> str:
    """Rows are dicts keyed by header name; a key that is not in `header` is dropped."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def op(code, dur, gap, dev=0, **kw):
    return {"OP CODE": code, "OP TYPE": "tt_dnn_device", "DEVICE ID": dev, "DEVICE ARCH": "blackhole", "DEVICE KERNEL DURATION [ns]": dur,
            "OP TO OP LATENCY [ns]": gap, **kw}


def sp(name):
    return {"OP CODE": name, "OP TYPE": "signpost"}


def layered_rows():
    """Two layers. Layer 0: boundary 1 op, mixer 2 ops, mlp 1 op. Layer 1: boundary 1 op, mixer 1 op, moe 2 ops. Then 2 ops outside."""
    return [op("Embedding", 1000, 0),                                  # before any signpost: outside the layers
            sp("layer0/boundary"), op("AllGather", 4000, 500),
            sp("layer0/mixer"), op("Matmul", 10000, 300), op("Softmax", 2000, 200),
            sp("layer_0.mlp"), op("Matmul", 20000, 400),
            sp("layer1:boundary"), op("AllGather", 5000, 600),
            sp("layer1/mixer"), op("Matmul", 11000, 100),
            sp("something else"), op("Add", 500, 100),                 # an unrelated signpost does not change the group
            sp("layer1/moe"), op("SparseMatmul", 30000, 700), op("Matmul", 6000, 50),
            sp("outside_layers"), op("LayerNorm", 700, 20), op("Matmul", 9000, 30)]


# ----------------------------------------------------------------------------------------------------------------- the sums

def test_totals_and_groups_add_up_and_the_rule_of_thumb_is_labelled():
    got = ps.analyze_raw(raw_csv(layered_rows()))
    assert got["total"]["ops"] == 12
    assert got["total"]["time_in_ops_us"] == pytest.approx(1 + 4 + 10 + 2 + 20 + 5 + 11 + 0.5 + 30 + 6 + 0.7 + 9)
    g = got["groups"]
    assert (g["layer_boundary"]["ops"], g["layer_boundary"]["time_in_ops_us"], g["layer_boundary"]["gap_us"]) == (2, 9.0, 1.1)
    assert (g["mixer"]["ops"], g["mixer"]["time_in_ops_us"], g["mixer"]["gap_us"]) == (4, 23.5, 0.7)    # the "something else" op stays in layer 1's mixer
    assert (g["moe_or_mlp"]["ops"], g["moe_or_mlp"]["time_in_ops_us"]) == (3, 56.0)
    assert (g["outside_layers"]["ops"], g["outside_layers"]["time_in_ops_us"]) == (3, 10.7)
    assert g["mixer"]["rule_of_thumb_floor_us"] == [12.0, 20.0]                                        # 4 ops at 3 to 5 microseconds
    assert got["layers_seen"] == 2
    l0, l1 = got["per_layer"]
    assert l0["layer"] == 0 and l0["mixer"]["ops"] == 2 and l0["moe_or_mlp"]["ops"] == 1
    assert l1["layer"] == 1 and l1["moe_or_mlp"]["ops"] == 2 and "outside_layers" not in l1


def test_the_device_with_most_rows_is_summarised_and_the_others_are_named():
    rows = [op("Matmul", 1000, 10, dev=0), op("Matmul", 1000, 10, dev=1), op("Add", 1000, 10, dev=1)]
    got = ps.analyze_raw(raw_csv(rows))
    assert got["device"] == 1 and got["devices_seen"] == 2 and got["total"]["ops"] == 2
    assert any("other devices" in c for c in got["could_not_measure"])


@pytest.mark.parametrize("signposts", [[], [sp("layer0")], [sp("layerX/mixer")], [sp("layer0/attention")], [sp("my_layer0/mixer")]])
def test_no_matching_signpost_means_no_groups_and_a_stated_reason(signposts):
    got = ps.analyze_raw(raw_csv(signposts + [op("Matmul", 1000, 10)]))
    assert got["groups"] is None and got["per_layer"] == [] and got["total"]["ops"] == 1
    assert any(c.startswith("op groups:") for c in got["could_not_measure"])


def test_a_missing_column_or_cell_is_none_and_named_never_zero():
    got = ps.analyze_raw(raw_csv([op("Matmul", 1000, 10), op("Add", 1000, 10)], header=["OP CODE", "OP TYPE", "DEVICE ID", "OP TO OP LATENCY [ns]"]))
    assert got["total"]["time_in_ops_us"] is None and got["total"]["gap_us"] == 0.02
    assert any("DEVICE KERNEL DURATION [ns]" in c for c in got["could_not_measure"])
    # one bad cell among good ones: counted as an op, adds no time, and is reported
    got = ps.analyze_raw(raw_csv([op("A", "NaN", 10), op("B", "1e999", 10), op("C", "", 10), op("D", "x", 10), op("E", 2000, 10)]))
    assert got["total"]["ops"] == 5 and got["total"]["ops_without_duration"] == 4 and got["total"]["time_in_ops_us"] == 2.0
    assert any("device time of 4 op(s)" in c for c in got["could_not_measure"])
    assert got["dram_per_bank"] is None and got["noc_congestion"] is None
    assert any("per-bank DRAM" in c for c in got["could_not_measure"]) and any("network congestion" in c for c in got["could_not_measure"])


def test_counter_columns_are_read_when_present_and_labelled_as_percent_of_peak():
    rows = [op("Matmul", 1000, 10, **{"DRAM BW UTIL PER CTRL (%)": "[10.0, 80.5, 20.0]", "NPE CONG IMPACT (%)": "12.5"}),
            op("Add", 1000, 10, **{"DRAM BW UTIL PER CTRL (%)": "5.0 6.0 7.0", "NPE CONG IMPACT (%)": "2"})]
    got = ps.analyze_raw(raw_csv(rows))
    assert got["dram_per_bank"]["controllers"] == 3 and got["dram_per_bank"]["max_pct"] == 80.5
    assert got["dram_per_bank"]["busiest_op_per_controller_pct"] == [10.0, 80.5, 20.0] and "percent of peak" in got["dram_per_bank"]["unit"]
    assert got["noc_congestion"]["max_pct"] == 12.5 and got["noc_congestion"]["ops_at_or_over_10_pct"] == 1


@pytest.mark.parametrize("cell", ["[1, 2, x]", "[nan]", "[inf]", "[-5]", "[2000]", "[" + ", ".join(["1"] * 40) + "]", "\x00", "[]"])
def test_a_bad_bank_cell_is_not_measured(cell):
    got = ps.analyze_raw(raw_csv([op("Matmul", 1000, 10, **{"DRAM BW UTIL PER CTRL (%)": cell})]))
    assert got["dram_per_bank"] is None


def test_the_header_must_name_the_op_code():
    with pytest.raises(ps.ProfileError):
        ps.analyze_raw("A,B\n1,2\n")
    with pytest.raises(ps.ProfileError):
        ps.analyze_raw("")


# --------------------------------------------------------------------------------------------------------- the report CSV

REPORT_HEADER = ["ID", "Total %", "Bound", "OP Code", "Device", "Device Time", "Op-to-Op Gap", "Cores", "DRAM", "DRAM %", "FLOPs", "FLOPs %", "Advice"]


def report_csv(rows, header=REPORT_HEADER) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=header, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


def test_top_ops_are_ranked_by_device_time_with_utilization_and_the_reports_advice():
    rows = [{"ID": 1, "OP Code": "Add", "Device Time": 5.0, "Op-to-Op Gap": 1.0, "Total %": 2.0},
            {"ID": 2, "OP Code": "Matmul", "Device Time": 90.5, "Op-to-Op Gap": 3.0, "Total %": 60.2, "Bound": "DRAM", "Cores": 64, "DRAM": 410.0, "DRAM %": 82.0,
             "FLOPs %": 11.0, "Advice": "Increase grid size (currently using 8) • in0_block_w=1 is small, try in0_block_w=2 or above"},
            {"ID": 3, "OP Code": "Bad", "Device Time": "NaN"}]
    top, could = ps.read_report(report_csv(rows), 2)
    assert [t["op_code"] for t in top] == ["Matmul", "Add"]
    m = top[0]
    assert (m["device_time_us"], m["dram_pct_of_peak"], m["flops_pct_of_peak"], m["bound"], m["cores"]) == (90.5, 82.0, 11.0, "DRAM", 64)
    assert m["advice"] == ["Increase grid size (currently using 8)", "in0_block_w=1 is small, try in0_block_w=2 or above"]
    assert top[1]["dram_pct_of_peak"] is None and top[1]["advice"] == []                          # a blank cell is not measured, not 0
    assert could == []


def test_a_report_without_the_ranking_columns_is_refused_and_other_missing_columns_are_named():
    with pytest.raises(ps.ProfileError):
        ps.read_report("OP Code,Cores\nA,1\n", 3)
    top, could = ps.read_report("OP Code,Device Time\nA,1.5\n", 3)
    assert top[0]["op_code"] == "A" and top[0]["flops_pct_of_peak"] is None
    assert any("FLOPs %" in c for c in could)


@pytest.mark.parametrize("cell", ["NaN", "inf", "-1", "1e999", "1" * 5000, "\x00", "[1]", "True", " ", "0x10", "1_000"])
def test_a_report_cell_that_is_not_a_plain_number_is_none(cell):
    top, _ = ps.read_report(report_csv([{"OP Code": "A", "Device Time": 1.0, "Cores": cell, "DRAM": cell, "FLOPs %": cell}]), 3)
    assert top[0]["cores"] is None and top[0]["dram_gb_s"] is None and top[0]["flops_pct_of_peak"] is None


# ------------------------------------------------------------------------------------------------------- the size bound

def test_the_summary_stays_under_its_cap_whatever_the_input_holds():
    """Thousands of layers, thousands of op codes with long names, and advice cells at the limit. The file is written from this input."""
    rows = []
    for layer in range(3000):
        rows += [sp(f"layer{layer}/mixer"), op(f"Op{layer}" + "x" * 500, 1000 + layer, 10)]
    raw = ps.analyze_raw(raw_csv(rows))
    report = report_csv([{"ID": i, "OP Code": f"Matmul{i}" + "y" * 1000, "Device Time": 100 + i, "Advice": " • ".join(["z" * 2000] * 50)} for i in range(200)])
    top, could = ps.read_report(report, ps.MAX_TOP)
    assert len(top) == ps.MAX_TOP and all(len(t["advice"]) <= ps.MAX_ADVICE_ITEMS and all(len(a) <= ps.MAX_ADVICE_CHARS for a in t["advice"]) for t in top)
    s = ps.build_summary(raw=raw, top_ops=top, top_source="tt-perf-report", extra_could=["x" * 10_000] * 400, run=None, command_sha="0" * 64,
                         status_reason=None, top_n=ps.MAX_TOP)
    text = ps._dump(s)
    assert len(text.encode()) <= ps.SUMMARY_MAX_BYTES
    assert json.loads(text)["limits"]["cut"]                       # something was cut, and the summary says what


def test_a_summary_cut_to_fit_keeps_valid_json_and_names_every_list_it_cut():
    big = {"schema": ps.SCHEMA, "per_layer": [{"layer": i, "mixer": {"ops": 1}} for i in range(5000)], "top_ops": [{"op_code": "a" * 80, "advice": ["b" * 240] * 5}] * 50,
           "by_op_code": [{"op_code": "c" * 80}] * 15, "could_not_measure": ["d" * 300] * 200, "groups": None, "dram_per_bank": None, "noc_congestion": None}
    out = ps.bound(big)
    assert len(ps._dump(out).encode()) <= ps.SUMMARY_MAX_BYTES
    assert {"per_layer", "could_not_measure"} <= set(out["limits"]["cut"])


def test_bound_never_emits_nan():
    with pytest.raises(ValueError):
        ps.bound({"x": float("nan")})                              # allow_nan=False: a NaN is refused, not written


def test_a_small_summary_is_not_cut():
    raw = ps.analyze_raw(raw_csv(layered_rows()))
    s = ps.build_summary(raw=raw, top_ops=None, top_source=None, extra_could=[], run=None, command_sha=None, status_reason=None, top_n=5)
    assert s["limits"]["cut"] == [] and len(s["per_layer"]) == 2


# ------------------------------------------------------------------------------------------------- the command, fake hardware

class Out(io.StringIO):
    pass


def args_for(tmp_path, *words, **kw):
    ns = ps.parser().parse_args(["--out", str(tmp_path / "out"), *[a for k, v in kw.items() for a in (f"--{k.replace('_', '-')}", str(v))], "--", *words])
    return ns


PLAN = {"board": "B1", "chips": [2, 3], "exact": 2}


def fake_run_factory(rows=None, report=None, exit_code=0, timed_out=False, calls=None):
    """A stand-in for ps.run_group. The profiler run writes the raw CSV where the profiler would; tt-perf-report writes its CSV."""
    def run(argv, *, timeout_s, log_path, env=None, term_grace_s=150.0):
        if calls is not None:
            calls.append(list(argv))
        if "tracy" in argv:
            out_dir = Path(argv[argv.index("-o") + 1]) / "reports" / "2026_11_03_10_00_00"
            out_dir.mkdir(parents=True)
            if rows is not None:
                (out_dir / "ops_perf_results_2026_11_03_10_00_00.csv").write_text(raw_csv(rows))
            return exit_code, timed_out
        csv_out = Path(argv[argv.index("--csv") + 1])
        if report is None:
            return 1, False
        csv_out.write_text(report)
        return 0, False
    return run


REPORT = report_csv([{"ID": 1, "OP Code": "Matmul", "Device Time": 90.5, "Op-to-Op Gap": 3.0, "Total %": 60.0, "Advice": "Increase grid size (currently using 8)"}])


def test_a_run_leases_a_free_board_wraps_the_command_in_the_profiler_and_writes_the_summary(tmp_path):
    calls: list = []
    out = Out()
    code = ps.main(args_for(tmp_path, "python3", "-m", "pytest", "t.py", perf_report="tt-perf-report"), out=out, run=fake_run_factory(layered_rows(), REPORT, calls=calls),
                   probe=lambda py: None, tools=lambda h: None, choose=lambda wants, **kw: [PLAN])
    assert code == 0
    gozer = calls[0]
    assert gozer[:8] == ["gozer", "run", "--chips", "1", "--exact", "2", "--who", "tt-orchard:profile"]
    tail = gozer[gozer.index("--") + 1:]
    assert tail[:6] == ["python3", "-m", "tracy", "-r", "--no-web-server", "-o"] and tail[7:] == ["-m", "pytest", "t.py"]
    assert calls[1][0] == "tt-perf-report" and "--ignore-signposts" in calls[1]
    s = json.loads((tmp_path / "out" / ps.SUMMARY_FILE).read_text())
    assert s["status"] == "measured" and s["top_ops_source"] == "tt-perf-report" and s["top_ops"][0]["op_code"] == "Matmul"
    assert s["run"]["exit_code"] == 0 and s["groups"]["mixer"]["ops"] == 4
    assert "t.py" not in json.dumps(s) and str(tmp_path) not in json.dumps(s)                  # the command text and local paths stay out


def test_no_profiler_in_the_python_given_is_a_stated_gap_and_exit_3_with_nothing_run(tmp_path):
    calls: list = []
    out = Out()
    code = ps.main(args_for(tmp_path, "python3", "x.py"), out=out, run=fake_run_factory(layered_rows(), REPORT, calls=calls),
                   probe=lambda py: "the Python given has no `tracy` module", choose=lambda wants, **kw: [PLAN])
    assert code == 3 and calls == []
    s = json.loads((tmp_path / "out" / ps.SUMMARY_FILE).read_text())
    assert s["status"] == "not_measured" and s["total"] is None and s["top_ops"] == []
    assert any("tracy" in c for c in s["could_not_measure"])


def test_the_real_probe_reports_a_python_without_tracy(tmp_path):
    """Not a stand-in: has_profiler runs a real interpreter (the base Python of this one, with no virtual environment) and find_spec('tracy') is None there."""
    import sys
    base = Path(sys.base_prefix) / "bin" / "python3"
    import subprocess
    has_tracy = subprocess.run([str(base), "-I", "-c", "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('tracy') else 3)"]).returncode == 0
    if has_tracy:
        pytest.skip("the base Python can import a module named tracy")
    problem = ps.has_profiler(str(base))
    assert problem is not None and "tracy" in problem
    # and the same probe says nothing is wrong for a Python that has the module: a stand-in interpreter that exits 0
    ok = tmp_path / "py-ok"
    ok.write_text("#!/bin/sh\nexit 0\n")
    ok.chmod(0o755)
    assert ps.has_profiler(str(ok)) is None
    assert "cannot be run" in ps.has_profiler(str(tmp_path / "no-such-python"))


def test_no_free_board_is_a_stated_gap_and_exit_3(tmp_path):
    calls: list = []

    def choose(wants, **kw):
        raise bench_control.NoFreeBoard("1 session asked for 1 board and 0 free (board 0: BUSY)")
    code = ps.main(args_for(tmp_path, "python3", "x.py"), out=Out(), run=fake_run_factory(layered_rows(), REPORT, calls=calls), probe=lambda py: None, tools=lambda h: None, choose=choose)
    assert code == 3 and calls == []
    s = json.loads((tmp_path / "out" / ps.SUMMARY_FILE).read_text())
    assert s["status_reason"] == "no free board" and any("no board was free" in c for c in s["could_not_measure"])


def test_a_run_that_exports_no_csv_is_exit_3_and_says_so(tmp_path):
    code = ps.main(args_for(tmp_path, "python3", "x.py"), out=Out(), run=fake_run_factory(None, REPORT), probe=lambda py: None, tools=lambda h: None, choose=lambda w, **kw: [PLAN])
    assert code == 3
    s = json.loads((tmp_path / "out" / ps.SUMMARY_FILE).read_text())
    assert s["status"] == "not_measured" and any("no ops_perf_results CSV" in c for c in s["could_not_measure"])


def test_a_failed_command_still_summarises_what_it_exported_and_exits_1(tmp_path):
    code = ps.main(args_for(tmp_path, "python3", "x.py", perf_report="tt-perf-report"), out=Out(), run=fake_run_factory(layered_rows(), REPORT, exit_code=7), probe=lambda py: None, tools=lambda h: None,
                   choose=lambda w, **kw: [PLAN])
    assert code == 1
    s = json.loads((tmp_path / "out" / ps.SUMMARY_FILE).read_text())
    assert s["run"]["exit_code"] == 7 and "exited with code 7" in s["status_reason"] and s["total"]["ops"] == 12


def test_a_timeout_is_recorded(tmp_path):
    code = ps.main(args_for(tmp_path, "python3", "x.py", timeout=5, perf_report="tt-perf-report"),
                   out=Out(), run=fake_run_factory(layered_rows(), REPORT, exit_code=None, timed_out=True), probe=lambda py: None, tools=lambda h: None, choose=lambda w, **kw: [PLAN])
    assert code == 1
    s = json.loads((tmp_path / "out" / ps.SUMMARY_FILE).read_text())
    assert s["run"]["timed_out"] is True and "stopped after 5 s" in s["status_reason"]


def test_a_queued_lease_is_stated(tmp_path):
    code = ps.main(args_for(tmp_path, "python3", "x.py"), out=Out(), run=fake_run_factory(None, None, exit_code=10), probe=lambda py: None, tools=lambda h: None, choose=lambda w, **kw: [PLAN])
    assert code == 3
    s = json.loads((tmp_path / "out" / ps.SUMMARY_FILE).read_text())
    assert "queued" in s["status_reason"] or "no ops_perf_results" in s["status_reason"]


def test_when_tt_perf_report_fails_the_top_rows_come_from_the_raw_csv_and_the_gap_is_named(tmp_path):
    code = ps.main(args_for(tmp_path, "python3", "x.py", perf_report="tt-perf-report"), out=Out(), run=fake_run_factory(layered_rows(), None), probe=lambda py: None, tools=lambda h: None,
                   choose=lambda w, **kw: [PLAN])
    assert code == 0
    s = json.loads((tmp_path / "out" / ps.SUMMARY_FILE).read_text())
    assert s["status"] == "partial" and s["top_ops_source"].startswith("raw ops csv")
    assert any("utilization and advice" in c and "exited with code 1" in c for c in s["could_not_measure"])
    assert all(t["advice"] == [] and "dram_pct_of_peak" not in t for t in s["top_ops"])


def test_from_reads_an_existing_profile_with_no_lease_and_no_command(tmp_path):
    src = tmp_path / "prof" / "deeper"
    src.mkdir(parents=True)
    (src / "ops_perf_results_2026.csv").write_text(raw_csv(layered_rows()))
    (src / "ops_perf_results_2025.csv").write_text(raw_csv([op("Old", 1, 1)]))
    ns = ps.parser().parse_args(["--out", str(tmp_path / "out"), "--from", str(tmp_path / "prof"), "--perf-report", str(tmp_path / "none")])
    code = ps.main(ns, out=Out(), run=fake_run_factory(None, REPORT), probe=lambda py: pytest.fail("no profiler probe for --from"),
                   choose=lambda *a, **k: pytest.fail("no lease for --from"))
    assert code == 0
    s = json.loads((tmp_path / "out" / ps.SUMMARY_FILE).read_text())
    assert s["total"]["ops"] == 12                                                            # the newest file by name


@pytest.mark.parametrize("words,why", [([], "must start with a Python"), (["ls"], "must start with a Python"), (["python3"], "needs a script"),
                                       (["/usr/bin/env", "python3", "x.py"], "must start with a Python")])
def test_the_command_must_be_a_python_with_a_script_or_module(tmp_path, words, why):
    out = Out()
    code = ps.main(args_for(tmp_path, *words), out=out, run=fake_run_factory(), probe=lambda py: None, tools=lambda h: None, choose=lambda w, **kw: [PLAN])
    assert code == 2 and why in out.getvalue()


@pytest.mark.parametrize("top", [0, -1, 51, 10**20])
def test_top_is_bounded(tmp_path, top):
    out = Out()
    ns = ps.parser().parse_args(["--out", str(tmp_path / "o"), "--top", str(top), "--", "python3", "x.py"])
    assert ps.main(ns, out=out, run=fake_run_factory(), probe=lambda py: None, tools=lambda h: None, choose=lambda w, **k: [PLAN]) == 2


def test_from_with_a_command_or_a_file_is_refused(tmp_path):
    f = tmp_path / "f"
    f.write_text("x")
    for extra in (["--from", str(tmp_path), "--", "python3", "x.py"], ["--from", str(f)]):
        ns = ps.parser().parse_args(["--out", str(tmp_path / "o"), *extra])
        assert ps.main(ns, out=Out(), run=fake_run_factory(), probe=lambda py: None, tools=lambda h: None, choose=lambda w, **k: [PLAN]) == 2


# ------------------------------------------------------------------------------------- files that something else controls

def test_find_ops_csv_never_follows_a_link_or_takes_a_directory(tmp_path):
    (tmp_path / "real").mkdir()
    (tmp_path / "elsewhere.csv").write_text("OP CODE\n")
    (tmp_path / "real" / "ops_perf_results_1.csv").symlink_to(tmp_path / "elsewhere.csv")
    (tmp_path / "ops_perf_results_2.csv").mkdir()
    assert ps.find_ops_csv(tmp_path / "real") is None
    assert ps.find_ops_csv(tmp_path) is None
    assert ps.find_ops_csv(tmp_path / "missing") is None


def test_a_raw_csv_over_the_cap_is_refused_not_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "RAW_MAX_BYTES", 100)
    p = tmp_path / "ops_perf_results_x.csv"
    p.write_text(raw_csv(layered_rows()))
    with pytest.raises(safeio.TooLarge):
        ps.analyze_raw_file(p)
    s, code = ps.summarise_dir(tmp_path, tmp_path, top_n=3, perf_report=str(tmp_path / "none"), report_timeout_s=1.0)
    assert code == 3 and s["status"] == "not_measured" and any("could not be read" in c for c in s["could_not_measure"])


def test_a_csv_with_a_huge_cell_is_refused_not_crashed():
    with pytest.raises(ps.ProfileError):
        ps.analyze_raw("OP CODE,OP TYPE\n" + '"' + "a" * 200_000 + '",x\n')


def test_the_summary_is_written_atomically_and_replaces_a_link(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("keep")
    (out / ps.SUMMARY_FILE).symlink_to(victim)
    ps._write_summary(out, {"schema": ps.SCHEMA})
    assert victim.read_text() == "keep" and not (out / ps.SUMMARY_FILE).is_symlink()


# ---------------------------------------------------------------------------------------------------------- the wiring

def test_tt_orchard_profile_is_a_command_and_from_runs_through_main(tmp_path, capsys):
    src = tmp_path / "prof"
    src.mkdir()
    (src / "ops_perf_results_1.csv").write_text(raw_csv(layered_rows()))
    rc = cli.main(["profile", "--from", str(src), "--out", str(tmp_path / "out"), "--perf-report", str(tmp_path / "none")])
    assert rc == 0 and json.loads((tmp_path / "out" / ps.SUMMARY_FILE).read_text())["total"]["ops"] == 12


def test_the_command_digest_does_not_hold_the_command_text():
    d = ps.command_digest(["python3", "/secret/path/x.py"])
    assert len(d) == 64 and "secret" not in d
    assert ps.command_digest(["a", "b"]) != ps.command_digest(["ab"])


def test_run_group_kills_the_whole_process_group_on_timeout(tmp_path):
    """A child that starts its own child and sleeps: after the timeout neither is left."""
    import sys
    marker = tmp_path / "pids"
    script = ("import os, subprocess, sys, time\n"
              f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
              f"open({str(marker)!r}, 'w').write(str(p.pid))\n"
              "time.sleep(60)\n")
    code, timed_out = ps.run_group([sys.executable, "-c", script], timeout_s=2.0, log_path=tmp_path / "log", term_grace_s=5.0)
    assert timed_out is True
    grandchild = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        for _ in range(50):                      # the group was signalled; wait for init to reap the orphan
            os.kill(grandchild, 0)
            import time
            time.sleep(0.1)
        os.kill(grandchild, 0)


# ------------------------------------------------------------------------------- the Tracy tools and TT_METAL_HOME

def make_tools(root: Path, which=ps.TRACY_TOOLS) -> Path:
    d = root / "build" / "tools" / "profiler" / "bin"
    d.mkdir(parents=True)
    for t in which:
        (d / t).write_text("#!/bin/sh\n")
    return root


def test_has_tracy_tools_names_what_is_missing(tmp_path):
    assert "TT_METAL_HOME is not set" in ps.has_tracy_tools(None)
    assert "TT_METAL_HOME is not set" in ps.has_tracy_tools("")
    assert "tracy-capture and tracy-csvexport not found" in ps.has_tracy_tools(str(tmp_path))
    home = make_tools(tmp_path / "half", which=("tracy-capture",))
    assert ps.has_tracy_tools(str(home)).startswith("tracy-csvexport not found")
    assert ps.has_tracy_tools(str(make_tools(tmp_path / "full"))) is None
    assert ps.has_tracy_tools("a\0b") is not None and ps.has_tracy_tools(123) is not None


def test_a_missing_tool_directory_stops_the_run_before_any_lease(tmp_path):
    calls: list = []
    chose: list = []
    ns = args_for(tmp_path, "python3", "x.py", tt_metal_home=tmp_path / "empty")
    code = ps.main(ns, out=Out(), run=fake_run_factory(layered_rows(), REPORT, calls=calls), probe=lambda py: None,
                   choose=lambda w, **kw: chose.append(1) or [PLAN], environ={})
    assert code == 3 and calls == [] and chose == []
    s = json.loads((tmp_path / "out" / ps.SUMMARY_FILE).read_text())
    assert s["status_reason"] == "no profiler" and any("TT_METAL_HOME" in c or "tracy-capture" in c for c in s["could_not_measure"])
    assert str(tmp_path) not in json.dumps(s)


# ------------------------------------------------------- files cut from the first real run (see docs/run-logs/2026-11-03-profile-first-run.md)

FIXTURES = Path(__file__).parent / "fixtures"


def test_the_real_exported_csv_columns_are_read_and_the_empty_counter_columns_are_not_measured():
    got = ps.analyze_raw(FIXTURES.joinpath("profile_ops_perf_results_sample.csv").read_text())
    assert got["devices_seen"] == 1 and got["arch"] == "blackhole" and got["total"]["ops"] > 20
    assert got["columns_missing"] == []                                   # all eight columns the summary reads are in the real header
    assert got["groups"]["mixer"]["ops"] > 0 and got["groups"]["moe_or_mlp"]["ops"] > 0 and got["layers_seen"] >= 2
    assert got["dram_per_bank"] is None and got["noc_congestion"] is None  # present but empty: NoC event capture was off in that run
    assert any("NoC event capture was not on" in c for c in got["could_not_measure"])
    assert all(r["time_in_ops_us"] > 0 for r in got["top_by_code"])


def test_the_real_perf_report_csv_gives_utilization_and_advice():
    top, could = ps.read_report(FIXTURES.joinpath("profile_perf_report_sample.csv").read_text(), 3)
    assert could == [] and len(top) == 3 and top[0]["op_code"].startswith("MatmulDeviceOperation")
    assert top[0]["device_time_us"] >= top[1]["device_time_us"] >= top[2]["device_time_us"]
    assert top[0]["dram_pct_of_peak"] is not None and top[0]["flops_pct_of_peak"] is not None and top[0]["advice"]
    assert top[0]["op_to_op_gap_us"] is None or top[0]["op_to_op_gap_us"] >= 0


def test_a_gap_over_a_millisecond_is_host_idle_time_and_is_kept_out_of_the_gap_sum():
    got = ps.analyze_raw(raw_csv([op("A", 1000, 50_000), op("B", 1000, 3_000_000_000), op("C", 1000, 1_000_000), op("D", 1000, 1_000_001)]))
    t = got["total"]
    assert t["ops"] == 4 and t["gap_us"] == 1050.0 and t["long_gap_ops"] == 2 and t["long_gap_us"] == 3_001_000.001


def test_counter_cells_that_are_present_but_unreadable_are_reported_as_unreadable_not_as_absent():
    got = ps.analyze_raw(raw_csv([op("A", 1000, 10, **{"DRAM BW UTIL PER CTRL (%)": "banks: 1;2", "NPE CONG IMPACT (%)": "high"})]))
    assert got["dram_per_bank"] is None and got["noc_congestion"] is None
    text = " ".join(got["could_not_measure"])
    assert "1 op row(s) have a value in DRAM BW UTIL PER CTRL (%) that is not a list of percentages" in text
    assert "1 op row(s) have a value in NPE CONG IMPACT (%) that is not a number" in text


def test_noc_traces_adds_the_profilers_own_flag_and_nothing_else(tmp_path):
    base = ps.tracy_argv(["python3", "x.py"], tmp_path)
    more = ps.tracy_argv(["python3", "x.py"], tmp_path, noc_traces=True)
    assert "--collect-noc-traces" not in base and [a for a in more if a not in base] == ["--collect-noc-traces"]
    assert more[more.index("-o") + 2:] == ["x.py"]                       # the workload's own words come last, untouched


def test_the_run_gets_tt_metal_home_from_the_flag_or_the_environment(tmp_path):
    seen: list = []
    base = fake_run_factory(layered_rows(), REPORT)

    def run(argv, **kw):
        seen.append(kw.get("env"))
        return base(argv, **kw)
    home = make_tools(tmp_path / "metal")
    ps.main(args_for(tmp_path, "python3", "x.py", tt_metal_home=home, perf_report="tt-perf-report"), out=Out(), run=run, probe=lambda py: None,
            choose=lambda w, **kw: [PLAN], environ={"PATH": "/bin", "TT_METAL_HOME": "/ignored"})
    assert seen[0]["TT_METAL_HOME"] == str(home) and seen[0]["PATH"] == "/bin"
    seen.clear()
    ps.main(args_for(tmp_path, "python3", "x.py", perf_report="tt-perf-report"), out=Out(), run=run, probe=lambda py: None,
            choose=lambda w, **kw: [PLAN], environ={"TT_METAL_HOME": str(home)})
    assert seen[0]["TT_METAL_HOME"] == str(home)
