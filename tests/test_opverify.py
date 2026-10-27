# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard inventory verify-candidates` (orchard/opverify.py) and its simulator script
(orchard/skills/ops-inventory-templates/verify_in_sim.py).

The host side is tested with a fake container runner. The script's references are tested with a fake ttnn made from
torch, so a correct op passes, a wrong op fails and a pair with no reference is said to be untested. A run against the
real simulator is opt-in (ORCHARD_LIVE_SIM=1); the suite never starts a container otherwise."""
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from orchard import opinventory as oi
from orchard import opverify as ov

SCRIPT = Path(ov.SCRIPT)


def load_script():
    spec = importlib.util.spec_from_file_location("verify_in_sim_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- shapes and case selection -------------------------------------------------------------------------------------

@pytest.mark.parametrize("shape,expected", [
    ([1, 8, 32], (32, 32)), ([1, 1, 8, 8], (32, 32)), ([1, 8, 2, 16], (32, 32)), ([4096, 4096], (64, 64)),
    ([100, 33], (64, 64)), ([33, 31], (64, 32)), ([64, 64], (64, 64)), ([8], (32, 32)), ([200], (32, 64)),
    ([], (32, 64)), (None, (32, 64)), ([0, 5], (32, 32)), ([True, 5], (32, 32)), (["a", 70], (32, 64)),
    ([1, 1, 1, 1024], (32, 64)),
])
def test_small_shape_takes_the_last_two_dimensions_capped_and_tile_aligned(shape, expected):
    assert ov.small_shape(shape) == expected


def candidate_row(op, cands, shape=(1, 8, 32), dtype="torch.float32", status="candidate"):
    return {"op": op, "status": status, "candidates": [{"ttnn": t, "why": "synonym"} for t in cands],
            "sample": {"output_shapes": [list(shape)], "dtype": dtype}}


def test_select_cases_makes_one_case_per_candidate_pair_and_skips_other_statuses():
    inv = {"ops": [candidate_row("aten.a", ["x", "y"]), candidate_row("aten.b", ["z"], status="mapped"),
                   candidate_row("aten.c", [], status="candidate")],
           "patterns": []}
    cases, left = ov.select_cases(inv)
    assert [c["id"] for c in cases] == ["aten.a->x", "aten.a->y"] and left == 0
    assert cases[0]["rows"] == 32 and cases[0]["cols"] == 32 and cases[0]["dtype"] == "torch.float32"
    assert cases[0]["kind"] == "op" and cases[0]["why"] == "synonym"


def test_select_cases_takes_recognised_groups_but_not_verified_ones_and_carries_eps_and_shape():
    inv = {"ops": [], "patterns": [
        {"pattern": "rms_norm", "ttnn": "rms_norm", "status": "pattern", "shape": [1, 8, 2048], "eps": 1e-6},
        {"pattern": "rotary", "ttnn": "r", "status": "pattern-verified", "shape": [1, 2], "eps": None},
        {"pattern": "silu_mul", "ttnn": "swiglu", "status": "pattern", "shape": None, "eps": None}]}
    cases, _ = ov.select_cases(inv)
    assert [c["id"] for c in cases] == ["pattern:rms_norm", "pattern:silu_mul"]
    assert cases[0]["rows"] == 32 and cases[0]["cols"] == 64 and cases[0]["const"] == [1e-6]
    assert cases[1]["const"] == [] and cases[1]["kind"] == "pattern"


def test_select_cases_caps_the_count_and_says_how_many_were_left_out():
    inv = {"ops": [candidate_row(f"aten.o{i}", ["x"]) for i in range(7)], "patterns": []}
    cases, left = ov.select_cases(inv, 5)
    assert len(cases) == 5 and left == 2
    assert [c["id"] for c in cases] == [f"aten.o{i}->x" for i in range(5)]


def test_select_cases_tolerates_malformed_rows():
    inv = {"ops": ["x", {"op": "aten.a", "status": "candidate", "candidates": [{"ttnn": "t"}], "sample": None},
                   {"op": "aten.b", "status": "candidate", "sample": {"output_shapes": []}}],
           "patterns": [None, "y"]}
    cases, _ = ov.select_cases(inv)
    assert [c["id"] for c in cases] == ["aten.a->t"] and cases[0]["rows"] == 32 and cases[0]["cols"] == 64


# --- the container command -----------------------------------------------------------------------------------------

def test_container_command_is_isolated_and_names_the_simulator():
    argv = ov.container_argv("img:1", "/w", "nm")
    assert argv[:4] == ["docker", "run", "--rm", "--name"] and argv[4] == "nm"
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--entrypoint") + 1] == "/opt/venv/bin/python3"
    assert "/w:/work:ro" in argv and "--privileged" not in argv and "--device" not in argv
    envs = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert "TT_METAL_SIMULATOR=/opt/sim/bh/libttsim_bh.so" in envs and "TT_METAL_ARCH_NAME=blackhole" in envs
    assert "TT_METAL_SLOW_DISPATCH_MODE=1" in envs and "TT_METAL_DISABLE_SFPLOADMACRO=1" in envs
    assert argv[-3:] == ["img:1", "/work/verify_in_sim.py", "/work/cases.json"]


# --- reading the result --------------------------------------------------------------------------------------------

def result_line(results, **kw):
    return "log line\n" + ov.MARK + json.dumps({"ttnn_version": "w", "torch_version": "t", "results": results, **kw}) + "\ntail"


def res(i, outcome="pass", **kw):
    return {"id": i, "outcome": outcome, "reason": kw.pop("reason", ""), "pcc": 1.0, "rel_l2": 0.0, "slope": 1.0,
            "seconds": 0.1, **kw}


def good_controls():
    return [res("control:right-op"), res("control:wrong-op", "fail"), res("control:wrong-integer-op", "fail")]


def test_parse_takes_the_last_marked_line_and_refuses_text_without_one():
    a = ov.MARK + json.dumps({"results": []})
    b = ov.MARK + json.dumps({"results": [{"id": "x"}]})
    assert ov.parse(a + "\n" + b)["results"] == [{"id": "x"}]
    assert ov.parse("noise" + ov.MARK + json.dumps({"results": []}))["results"] == []
    for bad in ("no marker", ov.MARK + "{", ov.MARK + "[]", ov.MARK + json.dumps({"results": 3})):
        with pytest.raises(ValueError):
            ov.parse(bad)


def test_controls_hold_only_when_the_right_op_passes_and_both_wrong_ops_fail():
    assert ov.check_controls(good_controls()) == []
    swapped = [res("control:right-op", "fail"), res("control:wrong-op", "fail"), res("control:wrong-integer-op", "fail")]
    assert "control:right-op: expected pass, got fail" in ov.check_controls(swapped)
    for wrong in ("control:wrong-op", "control:wrong-integer-op"):
        results = [res("control:right-op"), res("control:wrong-op", "fail"), res("control:wrong-integer-op", "fail")]
        for r in results:
            if r["id"] == wrong:
                r["outcome"] = "pass"
        assert [m for m in ov.check_controls(results) if wrong in m]
    assert len(ov.check_controls([])) == 3
    errored = [res("control:right-op"), res("control:wrong-op", "error"), res("control:wrong-integer-op", "fail")]
    assert ov.check_controls(errored), "an error is not the failure a control expects"


def test_the_controls_are_in_the_case_list_the_container_gets():
    got = []

    def run(argv, timeout, name):
        work = argv[argv.index("-v") + 1].split(":")[0]
        got.append(json.loads((Path(work) / "cases.json").read_text())["cases"])
        assert (Path(work) / "verify_in_sim.py").is_file()
        return 0, result_line([])

    inv = {"ops": [candidate_row("aten.a", ["x"])], "patterns": []}
    import tempfile
    out = Path(tempfile.mkdtemp())
    ov.main(inv_path(inv, out), out_dir=out / "o", run=run, say=lambda *_: None)
    ids = [c["id"] for c in got[0]]
    assert ids[0] == "aten.a->x" and set(ids[1:]) == {c["id"] for c in ov.CONTROLS}


def inv_path(inv, d):
    p = Path(d) / "ops-inventory.json"
    p.write_text(json.dumps(inv))
    return p


def test_the_controls_have_the_expected_shape():
    assert [c["expect"] for c in ov.CONTROLS] == ["pass", "fail", "fail"]
    wrong = [c for c in ov.CONTROLS if c["expect"] == "fail"]
    assert all("as" in c for c in wrong)                      # a wrong op is looked up as the right pair


@pytest.mark.parametrize("outcome,status", [("pass", "verified"), ("fail", "candidate-failed"),
                                            ("error", "candidate-failed"), ("no_reference", "not-tested"),
                                            ("weird", "candidate-failed"), (None, "candidate-failed")])
def test_classify(outcome, status):
    assert ov.classify({"outcome": outcome}) == status


# --- entries and the report ----------------------------------------------------------------------------------------

CASES = [{"id": "aten.a->x", "kind": "op", "aten": "aten.a", "ttnn": "x", "rows": 32, "cols": 32, "why": "synonym",
          "dtype": "torch.bool"},
         {"id": "aten.a->y", "kind": "op", "aten": "aten.a", "ttnn": "y", "rows": 32, "cols": 32, "why": "synonym"},
         {"id": "aten.b->z", "kind": "op", "aten": "aten.b", "ttnn": "z", "rows": 32, "cols": 64},
         {"id": "aten.c->w", "kind": "op", "aten": "aten.c", "ttnn": "w", "rows": 32, "cols": 64},
         {"id": "pattern:rms_norm", "kind": "pattern", "pattern": "rms_norm", "ttnn": "rms_norm", "rows": 32,
          "cols": 64, "const": [1e-6]}]


def run_result():
    return {"ttnn_version": "wheel1", "torch_version": "t2", "results": [
        res("aten.a->x", compared="exact"), res("aten.a->y"), res("aten.b->z", "fail", reason="pcc 0.5 is below 0.99"),
        res("aten.c->w", "no_reference", reason="no reference"), res("pattern:rms_norm", compared="float")]}


def test_entries_keep_the_first_passing_candidate_and_list_the_others():
    e = ov.entries_from(CASES, run_result(), image="img", date="2026-10-27")
    assert set(e) == {"aten.a", "pattern:rms_norm"}
    a = e["aten.a"]
    assert a["ttnn"] == "x" and a["also_passed"] == ["y"] and a["compared"] == "exact"
    assert a["shape"] == [32, 32] and a["dtype"] == "torch.bool" and a["image"] == "img"
    assert a["ttnn_wheel"] == "wheel1" and a["torch"] == "t2" and a["date"] == "2026-10-27"
    assert "not run on hardware" in a["label"]
    assert e["pattern:rms_norm"]["ttnn"] == "rms_norm"


def test_a_failed_untested_or_missing_case_makes_no_entry():
    r = run_result()
    r["results"] = [x for x in r["results"] if x["id"] != "aten.a->x"]
    e = ov.entries_from(CASES, r, image="i")
    assert e["aten.a"]["ttnn"] == "y" and e["aten.a"]["also_passed"] == []
    assert "aten.b" not in e and "aten.c" not in e


def test_summary_classifies_each_case_and_counts():
    s = ov.summarize(CASES, run_result(), [], 3)
    assert [r["status"] for r in s["rows"]] == ["verified", "verified", "candidate-failed", "not-tested", "verified"]
    assert s["counts"] == {"verified": 3, "candidate-failed": 1, "not-tested": 1} and s["left_out"] == 3
    assert s["rows"][2]["reason"] == "pcc 0.5 is below 0.99"


def test_a_failed_control_turns_every_row_suspect():
    s = ov.summarize(CASES, run_result(), ["control:right-op: expected pass, got fail"], 0)
    assert {r["status"] for r in s["rows"]} == {"suspect"}


def test_a_case_with_no_result_is_an_error_row():
    s = ov.summarize(CASES, {"results": []}, [], 0)
    assert {r["status"] for r in s["rows"]} == {"candidate-failed"} and "no result" in s["rows"][0]["reason"]


def test_markdown_states_the_label_the_bars_and_the_paste_rows():
    run = run_result()
    s = ov.summarize(CASES, run, [], 2)
    e = ov.entries_from(CASES, run, image="img")
    md = ov.render_md(s, e, run, "img")
    assert "not run on hardware" in md and "PCC at least 0.99" in md and "wheel1" in md
    assert "Controls: the right op passed and both wrong ops failed." in md
    assert '  "aten.a": {"ttnn": "x"},' in md and "aten_to_ttnn.json" in md
    assert "2 candidate pairs were over the cap" in md
    assert "rms_norm -> rms_norm" in md and "pcc 0.5 is below 0.99" in md


def test_markdown_withholds_the_paste_rows_when_a_control_failed():
    run = run_result()
    s = ov.summarize(CASES, run, ["control:x: expected pass, got fail"], 0)
    md = ov.render_md(s, ov.entries_from(CASES, run, image="i"), run, "i")
    assert "No result of this run is believed" in md and "aten_to_ttnn.json" not in md and "Groups that matched" not in md


def test_a_table_cell_cannot_break_out_of_its_row():
    run = run_result()
    run["results"][2]["reason"] = "a | b"
    md = ov.render_md(ov.summarize(CASES, run, [], 0), {}, run, "i")
    assert "a / b" in md


def test_files_are_written_and_a_failed_control_leaves_no_entries(tmp_path):
    run = run_result()
    e = ov.entries_from(CASES, run, image="i")
    ov.write_outputs(tmp_path / "ok", CASES, run, ov.summarize(CASES, run, [], 0), e, "i")
    v = json.loads((tmp_path / "ok" / "verified-mappings.json").read_text())
    assert set(v["entries"]) == {"aten.a", "pattern:rms_norm"} and v["schema"] == 1 and "label" in v
    ov.write_outputs(tmp_path / "bad", CASES, run, ov.summarize(CASES, run, ["x"], 0), e, "i")
    assert json.loads((tmp_path / "bad" / "verified-mappings.json").read_text())["entries"] == {}
    full = json.loads((tmp_path / "ok" / "verify-candidates.json").read_text())
    assert full["image"] == "i" and full["summary"]["counts"]["verified"] == 3 and "label" in full


def test_inventory_reads_what_the_verified_file_says_after_a_run(tmp_path):
    """The file the command writes is the file the inventory reads: an op verified by a run is `verified` next time."""
    run = run_result()
    e = ov.entries_from(CASES, run, image="i")
    ov.write_outputs(tmp_path, CASES, run, ov.summarize(CASES, run, [], 0), e, "i")
    loaded = oi.load_verified(tmp_path / "verified-mappings.json")
    assert loaded["aten.a"]["ttnn"] == "x"
    assert oi.status_of("aten.a", {}, {}, {"x"}, verified=loaded)[0] == "verified"


def test_merge_record_adds_to_the_package_file_and_replaces_equal_keys(tmp_path):
    p = tmp_path / "verified_mappings.json"
    ov.merge_record({"aten.a": {"ttnn": "x"}}, p)
    ov.merge_record({"aten.b": {"ttnn": "y"}, "aten.a": {"ttnn": "x2"}}, p)
    assert json.loads(p.read_text())["entries"] == {"aten.a": {"ttnn": "x2"}, "aten.b": {"ttnn": "y"}}
    assert not list(tmp_path.glob("*.tmp"))
    p.write_text("junk")
    ov.merge_record({"aten.c": {"ttnn": "z"}}, p)
    assert json.loads(p.read_text())["entries"] == {"aten.c": {"ttnn": "z"}}


def test_the_package_ships_a_verified_file_that_the_inventory_reads_and_that_holds_no_unverified_claim():
    d = json.loads((oi.DATA_DIR / "verified_mappings.json").read_text())
    assert d["schema"] == 1 and "not run on hardware" in d["label"]
    for k, v in d["entries"].items():
        assert isinstance(v["ttnn"], str) and v["image"] and v["ttnn_wheel"] and "not run on hardware" in v["label"], k


# --- main ----------------------------------------------------------------------------------------------------------

def inv_with(cands=("x",)):
    return {"ops": [candidate_row("aten.a", list(cands))], "patterns": []}


def main_with(tmp_path, inv, results, **kw):
    inv_path(inv, tmp_path)
    say = []

    def run(argv, timeout, name):
        return 0, result_line(results)

    rc = ov.main(tmp_path, out_dir=tmp_path / "out", run=kw.pop("run", run), say=say.append, **kw)
    return rc, say


def test_main_exits_0_when_the_controls_hold_even_if_a_candidate_failed(tmp_path):
    rc, say = main_with(tmp_path, inv_with(), [res("aten.a->x", "fail", reason="r")] + good_controls())
    assert rc == 0 and any("candidate-failed 1" in s for s in say)
    assert (tmp_path / "out" / "verified-mappings.json").is_file()


def test_main_exits_2_and_believes_nothing_when_a_control_fails(tmp_path):
    bad = [res("control:right-op"), res("control:wrong-op", "pass"), res("control:wrong-integer-op", "fail")]
    rc, say = main_with(tmp_path, inv_with(), [res("aten.a->x")] + bad)
    assert rc == 2 and any("no result is believed" in s for s in say)
    assert json.loads((tmp_path / "out" / "verified-mappings.json").read_text())["entries"] == {}


def test_main_exits_3_when_there_is_nothing_to_verify(tmp_path):
    rc, say = main_with(tmp_path, {"ops": [], "patterns": []}, [])
    assert rc == 3 and any("nothing to verify" in s for s in say) and not (tmp_path / "out").exists()


def test_main_exits_2_for_an_unreadable_inventory(tmp_path):
    (tmp_path / "ops-inventory.json").write_text("{")
    assert ov.main(tmp_path, out_dir=tmp_path / "o", run=lambda *a: (0, ""), say=lambda *_: None) == 2
    (tmp_path / "ops-inventory.json").write_text(json.dumps({"ops": "x"}))
    assert ov.main(tmp_path, out_dir=tmp_path / "o", run=lambda *a: (0, ""), say=lambda *_: None) == 2
    assert ov.main(tmp_path / "nope", out_dir=tmp_path / "o", run=lambda *a: (0, ""), say=lambda *_: None) == 2


def test_main_exits_2_when_the_container_fails_or_prints_no_result(tmp_path):
    inv_path(inv_with(), tmp_path)
    say = []
    assert ov.main(tmp_path, out_dir=tmp_path / "o", run=lambda *a: (125, "docker: no such image"), say=say.append) == 2
    assert "exit 125" in say[-1] and "no such image" in say[-1]
    assert ov.main(tmp_path, out_dir=tmp_path / "o", run=lambda *a: (0, "nothing"), say=say.append) == 2
    assert "ORCHARD_VERIFY" in say[-1]


def test_main_with_record_merges_into_the_given_file_and_without_it_writes_nothing_there(tmp_path):
    rec = tmp_path / "rec.json"
    results = [res("aten.a->x")] + good_controls()
    rc, say = main_with(tmp_path, inv_with(), results, record=True, record_path=rec)
    assert rc == 0 and json.loads(rec.read_text())["entries"]["aten.a"]["ttnn"] == "x"
    rec.unlink()
    main_with(tmp_path, inv_with(), results, record=False, record_path=rec)
    assert not rec.exists()


def test_main_does_not_record_when_a_control_failed(tmp_path):
    rec = tmp_path / "rec.json"
    bad = [res("control:right-op", "fail"), res("control:wrong-op", "fail"), res("control:wrong-integer-op", "fail")]
    rc, _ = main_with(tmp_path, inv_with(), [res("aten.a->x")] + bad, record=True, record_path=rec)
    assert rc == 2 and not rec.exists()


def test_the_scratch_directory_is_removed_after_the_run(tmp_path):
    seen = []

    def run(argv, timeout, name):
        seen.append(argv[argv.index("-v") + 1].split(":")[0])
        return 0, result_line(good_controls())

    ov.run_simulator([], run=run)
    assert seen and not Path(seen[0]).exists()

    def boom(argv, timeout, name):
        seen.append(argv[argv.index("-v") + 1].split(":")[0])
        return 1, "bad"

    with pytest.raises(RuntimeError):
        ov.run_simulator([], run=boom)
    assert not Path(seen[1]).exists()


def test_real_run_stops_a_command_that_outlives_its_timeout():
    t0 = time.time()
    rc, out = ov._real_run(["sleep", "30"], 1)
    assert rc == 124 and "timed out" in out and time.time() - t0 < 15


def test_real_run_returns_output_and_exit_code_and_a_missing_command():
    assert ov._real_run([sys.executable, "-c", "print('hi')"], 20) == (0, "hi\n")
    rc, out = ov._real_run(["/nonexistent-binary-xyz"], 5)
    assert rc == 127 and "FileNotFoundError" in out


# --- the script, with a fake ttnn made from torch ------------------------------------------------------------------

torch = pytest.importorskip("torch")


class FakeCtx:
    """What the script's adapters need from ttnn, built from torch. `wrong` names ttnn ops that should misbehave."""

    def __init__(self, ops=None):
        self.torch = torch
        self.ttnn = types.SimpleNamespace(bfloat16="bf16", int32="i32", float32="f32", uint32="u32", TILE_LAYOUT="tile",
                                          ROW_MAJOR_LAYOUT="rm")
        self.dev = "dev"
        self.ops = ops or {}

    def put(self, t, dtype=None, layout=None):
        return t.clone()

    def get(self, t):
        return t.float()

    def fn(self, name):
        return self.ops[name]


def case(aten, ttnn, **kw):
    return {"id": f"{aten}->{ttnn}", "kind": "op", "aten": aten, "ttnn": ttnn, "rows": 32, "cols": 64, **kw}


def test_the_script_passes_a_correct_op_and_fails_a_wrong_one():
    m = load_script()
    good = FakeCtx({"exp2": lambda x: torch.exp2(x.float())})
    assert m.run_case(good, case("aten.exp2.default", "exp2"))["outcome"] == "pass"
    wrong = FakeCtx({"exp2": lambda x: torch.exp(x.float())})
    r = m.run_case(wrong, case("aten.exp2.default", "exp2"))
    assert r["outcome"] == "fail" and r["reason"]


def test_a_scaled_output_fails_on_slope_even_when_pcc_is_perfect():
    m = load_script()
    c = FakeCtx({"exp2": lambda x: 3 * torch.exp2(x.float())})
    r = m.run_case(c, case("aten.exp2.default", "exp2"))
    assert r["outcome"] == "fail" and r["pcc"] > 0.999 and ("slope" in r["reason"] or "relative L2" in r["reason"])


def test_a_nan_output_fails():
    m = load_script()
    c = FakeCtx({"exp2": lambda x: torch.full_like(x.float(), float("nan"))})
    r = m.run_case(c, case("aten.exp2.default", "exp2"))
    assert r["outcome"] == "fail" and "NaN" in r["reason"]


def test_a_raising_op_is_an_error_with_its_text_and_a_pair_with_no_reference_is_untested():
    m = load_script()

    def boom(x):
        raise RuntimeError("TT_THROW: no kernel")

    r = m.run_case(FakeCtx({"exp2": boom}), case("aten.exp2.default", "exp2"))
    assert r["outcome"] == "error" and "RuntimeError: TT_THROW: no kernel" == r["reason"]
    r = m.run_case(FakeCtx(), case("aten.exp2.default", "totally_unknown"))
    assert r["outcome"] == "no_reference" and "no torch reference" in r["reason"]


def test_integer_ops_must_be_equal_and_two_operands_are_never_the_same_tensor():
    m = load_script()
    ops = {"bitwise_and": torch.bitwise_and, "bitwise_or": torch.bitwise_or, "logical_and": lambda a, b: (a != 0) & (b != 0)}
    c = FakeCtx(ops)
    ints = {"dtype": "torch.int32"}
    assert m.run_case(c, case("aten.__and__.Tensor", "bitwise_and", **ints))["outcome"] == "pass"
    assert m.run_case(c, case("aten.__and__.Tensor", "bitwise_or", **ints, **{"as": ["__and__", "bitwise_and"]}))["outcome"] == "fail"
    assert m.run_case(c, case("aten.__and__.Tensor", "logical_and", **ints))["outcome"] == "fail"
    # on masks (bool) the two agree
    assert m.run_case(c, case("aten.__and__.Tensor", "logical_and", dtype="torch.bool"))["outcome"] == "pass"
    m._DRAWS[0] = 0
    a, b = m.seeded((4, 4), "int", 0, 1 << 12), m.seeded((4, 4), "int", 0, 1 << 12)
    assert not torch.equal(a, b)


def test_seeds_are_reproducible_per_case():
    m = load_script()
    m._DRAWS[0] = 0
    first = m.seeded((3, 3))
    m._DRAWS[0] = 0
    assert torch.equal(first, m.seeded((3, 3)))


def test_a_pattern_pair_is_tested_only_under_its_own_ttnn_name():
    m = load_script()
    assert m.adapter_for({"kind": "pattern", "pattern": "rotary", "ttnn": "experimental.rotary_embedding_hf"})
    assert m.adapter_for({"kind": "pattern", "pattern": "rotary", "ttnn": "experimental.rotary_embedding_llama"}) is None
    assert m.adapter_for({"kind": "pattern", "pattern": "rms_norm", "ttnn": "layer_norm"}) is None
    assert m.adapter_for({"kind": "pattern", "pattern": "silu_mul", "ttnn": "swiglu"})
    assert m.adapter_for({"kind": "pattern", "pattern": "nothing", "ttnn": "x"}) is None


def test_the_as_field_picks_the_adapter_of_another_pair():
    m = load_script()
    assert m.adapter_for({"kind": "op", "aten": "aten.exp2.default", "ttnn": "exp"}) is None
    assert m.adapter_for({"kind": "op", "aten": "aten.exp2.default", "ttnn": "exp", "as": ["exp2", "exp2"]}) is m.a_unary


@pytest.mark.parametrize("aten,ttnn", [
    ("aten.__and__.Tensor", "bitwise_and"), ("aten.__and__.Tensor", "logical_and"), ("aten.index.Tensor", "embedding"),
    ("aten.to.dtype_layout", "typecast"), ("aten.new_ones.default", "ones"), ("aten.new_ones.default", "full"),
    ("aten.to.device", "to_device"), ("aten.copy_.default", "copy"), ("aten.copy_.default", "assign"),
    ("aten.copy_.default", "clone"), ("aten.div_.Tensor", "div_"), ("aten.div_.Tensor", "div"),
    ("aten.bitwise_not.default", "bitwise_not"), ("aten.masked_fill.Scalar", "where"),
    ("aten.split_with_sizes.default", "split"), ("aten.split_with_sizes.default", "chunk"),
    ("aten.type_as.default", "typecast"), ("aten.exp2.default", "exp2"), ("aten.mm.default", "matmul"),
    ("aten.clip.default", "clamp"), ("aten._softmax.default", "softmax"), ("aten.t.default", "transpose"),
    ("aten.rsqrt.default", "rsqrt"), ("aten.erf.default", "erf"), ("aten.__or__.Tensor", "bitwise_or"),
    ("aten.new_zeros.default", "zeros"), ("aten.div_.Tensor", "divide_"), ("aten.index_select.default", "embedding"),
])
def test_every_pair_the_variant_search_can_propose_for_the_recorded_traces_has_a_reference(aten, ttnn):
    m = load_script()
    assert m.adapter_for(case(aten, ttnn)) is not None


def test_the_pairs_the_recorded_traces_propose_are_covered_or_declared_untested():
    """Every candidate the shipped search offers for the recorded traces has a reference in the script, except the pairs
    named here. A new proposal with no reference shows up as a difference."""
    m = load_script()
    D = oi.DATA_DIR
    missing = set()
    for fx in ("qwen3-tiny", "rwkv", "mamba2", "moshi-lm"):
        raw = json.loads((Path(__file__).parent / "fixtures" / "ops-inventory" / f"raw-{fx}.json").read_text())
        inv = oi.build(raw, mapping=oi.load_table(D / "aten_to_ttnn.json"), decomps=oi.load_table(D / "decompositions.json"),
                       dump=oi.load_dump("0.80.0"), installed_ttnn="0.80.0", ignored=oi.load_ignored(D / "ignored_ops.json"))
        for r in inv["ops"]:
            for c in r["candidates"]:
                if m.adapter_for({"kind": "op", "aten": r["op"], "ttnn": c["ttnn"]}) is None:
                    missing.add((r["op"], c["ttnn"]))
    # gather is not a stand-in for advanced indexing without a dimension and an index of the same rank
    assert missing == {("aten.index.Tensor", "gather")}


def test_stats_and_judge_use_the_stated_bars():
    m = load_script()
    ref = torch.linspace(-1, 1, 64 * 32).reshape(32, 64)
    ok = m.stats(ref, ref * 1.0)
    assert m.judge(ok, False) == ("pass", "") and m.judge(ok, True) == ("pass", "")
    assert m.BARS["float"] == {"pcc": 0.99, "rel_l2": 5e-2, "slope": [0.95, 1.05]}
    scaled = m.stats(ref, ref * 1.5)
    out, why = m.judge(scaled, False)
    assert out == "fail" and ("relative L2" in why or "slope" in why)
    # a slope outside the bar with every other number inside it
    s = {"finite": True, "pcc": 0.9999, "rel_l2": 0.01, "slope": 1.2, "max_abs": 0.1, "exact": False}
    assert m.judge(s, False)[0] == "fail" and "slope" in m.judge(s, False)[1]
    s["slope"] = 1.05
    assert m.judge(s, False)[0] == "pass"
    s["slope"] = 0.949
    assert m.judge(s, False)[0] == "fail"
    s.update(slope=1.0, pcc=0.98)
    assert "pcc" in m.judge(s, False)[1]
    s.update(pcc=0.99, rel_l2=0.0501)
    assert "relative L2" in m.judge(s, False)[1]
    assert m.judge({**ok, "exact": False}, True)[0] == "fail"
    with pytest.raises(ValueError):
        m.stats(torch.zeros(3), torch.zeros(4))
    inf = m.stats(ref, torch.full_like(ref, float("inf")))
    assert m.judge(inf, False)[0] == "fail"
    const = m.stats(torch.ones(8), torch.ones(8))
    assert const["pcc"] == 1.0 and const["exact"]


def test_the_script_prints_one_marked_line_with_versions_and_bars(monkeypatch, capsys):
    m = load_script()

    class Dev:
        pass

    fake_ttnn = types.SimpleNamespace(open_device=lambda device_id: Dev(), close_device=lambda d: None,
                                      bfloat16="bf16", int32="i32", float32="f32", uint32="u32", TILE_LAYOUT="t",
                                      ROW_MAJOR_LAYOUT="r")
    monkeypatch.setitem(sys.modules, "ttnn", fake_ttnn)
    import tempfile
    p = Path(tempfile.mkdtemp()) / "c.json"
    p.write_text(json.dumps({"cases": [case("aten.exp2.default", "totally_unknown")]}))
    assert m.main([str(p)]) == 0
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith(m.MARK)][0]
    d = json.loads(line[len(m.MARK):])
    assert d["results"][0]["outcome"] == "no_reference" and d["bars"] == m.BARS and d["torch_version"]


# --- the real simulator, opt in ------------------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("ORCHARD_LIVE_SIM") != "1",
                    reason="opt-in: set ORCHARD_LIVE_SIM=1 to run the ttsim image (about 10 s, no chip)")
def test_live_simulator_run_holds_its_controls(tmp_path):
    inv = {"ops": [candidate_row("aten.__and__.Tensor", ["bitwise_and", "logical_and"], dtype="torch.int32"),
                   candidate_row("aten.exp2.default", ["exp2", "exp"])], "patterns": []}
    inv_path(inv, tmp_path)
    say = []
    rc = ov.main(tmp_path, out_dir=tmp_path / "out", say=say.append)
    summary = json.loads((tmp_path / "out" / "verify-candidates.json").read_text())["summary"]
    by = {r["id"]: r["status"] for r in summary["rows"]}
    assert rc == 0, say
    assert by["aten.__and__.Tensor->bitwise_and"] == "verified"
    assert by["aten.__and__.Tensor->logical_and"] == "candidate-failed"      # integers: logical and is not bitwise and
    assert by["aten.exp2.default->exp2"] == "verified" and by["aten.exp2.default->exp"] == "not-tested"      # no reference for that pair: not a pass
