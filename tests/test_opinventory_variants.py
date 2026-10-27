# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `candidate`, `verified` and `pattern` statuses in the op inventory (orchard/opinventory.py).

A candidate is a guess about names and a pattern is a recognised group of ops; neither is ever `mapped`, and every
summary and gate that counts mapped ops must keep that."""
import json
from pathlib import Path

import pytest

from orchard import opinventory as oi

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ops-inventory"
DUMP_080 = oi.load_dump("0.80.0")
MAPPING = oi.load_table(oi.DATA_DIR / "aten_to_ttnn.json")
DECOMPS = oi.load_table(oi.DATA_DIR / "decompositions.json")
IGNORED = oi.load_ignored(oi.DATA_DIR / "ignored_ops.json")
NAMES = {"matmul", "add", "typecast", "gather", "embedding", "bitwise_and", "logical_and", "rms_norm", "swiglu",
         "experimental.rotary_embedding_hf"}


def op(name, count=1, inl=1):
    return {"op": name, "count": count, "count_in_layers": inl, "sample": None, "modules": []}


def raw(*ops, **kw):
    d = {"schema": 1, "model_dir": "<m>", "architectures": [], "layers_in_config": 2, "layers_traced": 2,
         "note": "", "ops": list(ops), "failures": [], "traced_class": "X"}
    d.update(kw)
    return d


def build(r, names=NAMES, **kw):
    dump = {"ttnn_version": "t1", "names": sorted(names), "sha256": "x"}
    return oi.build(r, mapping=MAPPING, decomps=DECOMPS, dump=dump, installed_ttnn="t1", ignored=IGNORED, **kw)


def rows(inv):
    return {r["op"]: r for r in inv["ops"]}


def test_an_op_with_no_table_row_and_a_close_name_is_a_candidate_and_not_mapped():
    inv = build(raw(op("aten.__and__.Tensor"), op("aten.mm.default")))
    r = rows(inv)["aten.__and__.Tensor"]
    assert r["status"] == "candidate" and r["ttnn"] is None
    assert [(c["ttnn"], c["why"]) for c in r["candidates"]] == [("bitwise_and", "synonym"), ("logical_and", "synonym")]
    assert inv["summary"]["distinct"]["mapped"] == 0
    assert inv["summary"]["distinct"]["candidate"] == 2          # `mm` is a candidate for matmul too
    assert inv["summary"]["scaled_total"]["mapped"] == 0


def test_an_op_nothing_resembles_stays_unmapped():
    r = rows(build(raw(op("aten.einsum.default"))))["aten.einsum.default"]
    assert r["status"] == "unmapped" and r["candidates"] == []


def test_a_table_row_beats_a_candidate_search_and_a_mapped_op_has_no_candidates():
    r = rows(build(raw(op("aten.add.Tensor"))))["aten.add.Tensor"]
    assert r["status"] == "mapped" and r["candidates"] == []


def test_a_table_row_whose_name_is_missing_from_the_dump_is_a_candidate_search_not_a_mapping():
    # aten.sub.Tensor maps to `subtract`, which this dump lacks; the search may offer names, `mapped` it is not
    r = rows(build(raw(op("aten.sub.Tensor"))))["aten.sub.Tensor"]
    assert r["status"] == "unmapped"


def test_the_search_can_be_turned_off():
    inv = build(raw(op("aten.__and__.Tensor")), variants=False)
    assert rows(inv)["aten.__and__.Tensor"]["status"] == "unmapped"
    assert inv["summary"]["distinct"]["candidate"] == 0


def test_without_a_dump_nothing_is_a_candidate_and_nothing_is_claimed():
    inv = oi.build(raw(op("aten.__and__.Tensor")), mapping=MAPPING, decomps=DECOMPS, dump=None, ignored=IGNORED)
    assert rows(inv)["aten.__and__.Tensor"]["status"] == "unchecked"
    assert inv["patterns"] == [] and inv["summary"]["patterns"] == 0
    # a trace that holds groups still claims none without a dump
    inv = oi.build(fixture("qwen3-tiny-seq"), mapping=MAPPING, decomps=DECOMPS, dump=None, ignored=IGNORED)
    assert inv["patterns"] == [] and inv["patterns_unavailable"] == [] and inv["summary"]["patterns"] == 0


def test_a_dump_for_another_ttnn_version_gives_no_candidates_either():
    dump = {"ttnn_version": "t1", "names": sorted(NAMES)}
    inv = oi.build(raw(op("aten.__and__.Tensor")), mapping=MAPPING, decomps=DECOMPS, dump=dump, installed_ttnn="t2",
                   ignored=IGNORED)
    assert rows(inv)["aten.__and__.Tensor"]["status"] == "unchecked"


def test_the_summary_has_a_slot_for_every_status():
    inv = build(raw(op("aten.mm.default")))
    assert set(inv["summary"]["distinct"]) == set(oi.STATUSES) and set(inv["summary"]["scaled_total"]) == set(oi.STATUSES)
    assert "candidate" in oi.STATUSES and "verified" in oi.STATUSES


def test_scaled_counts_of_candidates_use_the_same_scaling_as_other_ops():
    inv = build(raw(op("aten.__and__.Tensor", count=3, inl=2), layers_in_config=10, layers_traced=2))
    assert rows(inv)["aten.__and__.Tensor"]["scaled_count"] == 2 * 5 + 1


# --- verified ------------------------------------------------------------------------------------------------------

VERIFIED = {"aten.__and__.Tensor": {"ttnn": "bitwise_and", "pcc": 1.0, "simulator": "ttsim"}}


def test_a_verified_entry_makes_the_op_verified_and_not_mapped():
    inv = build(raw(op("aten.__and__.Tensor")), verified=VERIFIED)
    r = rows(inv)["aten.__and__.Tensor"]
    assert r["status"] == "verified" and r["ttnn"] == "bitwise_and"
    assert r["verified"] == {"pcc": 1.0, "simulator": "ttsim"}
    assert inv["summary"]["distinct"]["verified"] == 1 and inv["summary"]["distinct"]["mapped"] == 0


def test_a_verified_entry_counts_only_while_its_ttnn_name_is_in_the_dump():
    inv = build(raw(op("aten.__and__.Tensor")), names=NAMES - {"bitwise_and"}, verified=VERIFIED)
    assert rows(inv)["aten.__and__.Tensor"]["status"] != "verified"


def test_a_table_row_still_beats_a_verified_entry():
    v = {"aten.add.Tensor": {"ttnn": "matmul"}}
    r = rows(build(raw(op("aten.add.Tensor")), verified=v))["aten.add.Tensor"]
    assert r["status"] == "mapped" and r["ttnn"] == "add"


def test_a_decomposition_still_beats_a_verified_entry():
    mapping = {"aten.mm.default": {"ttnn": "matmul"}, "aten.add.Tensor": {"ttnn": "add"}}
    decomps = {"aten.addmm.default": ["aten.mm.default", "aten.add.Tensor"]}
    v = {"aten.addmm.default": {"ttnn": "matmul"}}
    assert oi.status_of("aten.addmm.default", mapping, decomps, {"matmul", "add"}, verified=v)[0] == "composable"
    # with a leaf missing the decomposition fails and the verified entry is used
    assert oi.status_of("aten.addmm.default", mapping, decomps, {"matmul"}, verified=v)[0] == "verified"


def test_ignored_ops_are_never_verified():
    v = {"aten.alias.default": {"ttnn": "add"}}
    assert oi.status_of("aten.alias.default", {}, {}, {"add"}, ignored=frozenset({"aten.alias.default"}),
                        verified=v)[0] == "ignored"


def test_load_verified_reads_entries_and_ignores_junk(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps({"entries": {"aten.a": {"ttnn": "x", "pcc": 1}, "aten.b": {"pcc": 1}, "aten.c": "x"}}))
    assert oi.load_verified(p) == {"aten.a": {"ttnn": "x", "pcc": 1}}
    assert oi.load_verified(tmp_path / "missing.json") == {}
    p.write_text("not json")
    assert oi.load_verified(p) == {}
    p.write_text(json.dumps([1, 2]))
    assert oi.load_verified(p) == {}
    p.write_text(json.dumps({"entries": [1]}))
    assert oi.load_verified(p) == {}


# --- patterns ------------------------------------------------------------------------------------------------------

def fixture(name):
    return json.loads((FIXTURES / f"raw-{name}.json").read_text())


def real_build(name, **kw):
    return oi.build(fixture(name), mapping=MAPPING, decomps=DECOMPS, dump=DUMP_080, installed_ttnn="0.80.0",
                    ignored=IGNORED, **kw)


def test_a_real_trace_reports_its_rmsnorm_swiglu_and_rotary_groups():
    inv = real_build("qwen3-tiny-seq")
    got = {p["pattern"]: (p["ttnn"], p["occurrences"], p["scaled_count"], p["status"]) for p in inv["patterns"]}
    assert got == {"rms_norm": ("rms_norm", 5, 17, "pattern"), "silu_mul": ("swiglu", 1, 4, "pattern"),
                   "rotary": ("experimental.rotary_embedding_hf", 2, 8, "pattern")}
    assert inv["summary"]["patterns"] == 3


def test_a_pattern_does_not_change_the_status_of_the_ops_inside_it():
    inv = real_build("qwen3-tiny-seq")
    r = rows(inv)
    assert r["aten.rsqrt.default"]["status"] == "mapped" and r["aten.silu.default"]["status"] == "mapped"
    assert inv["summary"]["distinct"]["mapped"] == 26


def test_a_trace_without_a_sequence_has_no_patterns():
    inv = real_build("qwen3-tiny")
    assert inv["patterns"] == [] and inv["summary"]["patterns"] == 0


def test_a_pattern_verified_by_the_simulator_says_so():
    v = {"pattern:rms_norm": {"ttnn": "rms_norm", "pcc": 0.99999}}
    inv = real_build("qwen3-tiny-seq", verified=v)
    p = {p["pattern"]: p for p in inv["patterns"]}
    assert p["rms_norm"]["status"] == "pattern-verified" and p["rms_norm"]["verified"] == {"pcc": 0.99999}
    assert p["silu_mul"]["status"] == "pattern"
    # a verification of another ttnn op does not count
    other = real_build("qwen3-tiny-seq", verified={"pattern:rms_norm": {"ttnn": "layer_norm"}})
    assert {p["pattern"]: p["status"] for p in other["patterns"]}["rms_norm"] == "pattern"


def test_a_group_whose_ttnn_op_is_not_in_the_dump_is_listed_as_unavailable():
    r = fixture("qwen3-tiny-seq")
    dump = {"ttnn_version": "t1", "names": ["add", "mul"]}
    inv = oi.build(r, mapping=MAPPING, decomps=DECOMPS, dump=dump, installed_ttnn="t1", ignored=IGNORED)
    assert inv["patterns"] == [] and {u["pattern"] for u in inv["patterns_unavailable"]} == {"rms_norm", "silu_mul", "rotary"}


# --- counts on the real fixtures, before and after -----------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("qwen3-tiny", {"mapped": 26, "candidate": 5, "unmapped": 1}),
    ("rwkv", {"mapped": 17, "composable": 1, "candidate": 2, "unmapped": 0}),
    ("mamba2", {"mapped": 30, "composable": 1, "candidate": 3, "unmapped": 1}),
    ("moshi-lm", {"mapped": 26, "composable": 1, "candidate": 6, "unmapped": 1}),
])
def test_the_recorded_traces_on_the_0_80_dump(name, expected):
    d = real_build(name)["summary"]["distinct"]
    assert {k: d[k] for k in expected} == expected and d["unchecked"] == 0 and d["verified"] == 0


@pytest.mark.parametrize("name", ["qwen3-tiny", "rwkv", "mamba2", "moshi-lm"])
def test_candidates_never_change_the_mapped_count(name):
    on = real_build(name)["summary"]["distinct"]
    off = real_build(name, variants=False)["summary"]["distinct"]
    assert on["mapped"] == off["mapped"] and on["composable"] == off["composable"]
    assert on["candidate"] + on["unmapped"] == off["unmapped"]


def test_on_this_tree_the_old_dump_alone_leaves_every_op_unchecked():
    """The 0.65 dump is not used for ttnn 0.80.0, and no dump means no claim and no candidate."""
    inv = oi.build(fixture("qwen3-tiny"), mapping=MAPPING, decomps=DECOMPS,
                   dump=oi.load_dump("0.65.1rc17.dev6200"), installed_ttnn="0.80.0", ignored=IGNORED)
    assert inv["summary"]["distinct"]["unchecked"] == 32 and inv["summary"]["distinct"]["candidate"] == 0


def test_the_shipped_0_80_dump_is_loadable_for_exactly_that_version():
    assert DUMP_080 is not None and DUMP_080["ttnn_version"] == "0.80.0" and len(DUMP_080["names"]) > 800


# --- the report ----------------------------------------------------------------------------------------------------

def test_markdown_shows_candidates_with_their_reason_and_says_they_are_not_mapped():
    md = oi.render_md(build(raw(op("aten.__and__.Tensor"))))
    assert "| candidate | bitwise_and (synonym), logical_and (synonym) |" in md
    assert "Neither is `mapped`" in md and "| candidate | 1 |" in md


def test_markdown_lists_pattern_groups_with_their_status_and_the_unavailable_ones():
    inv = real_build("qwen3-tiny-seq")
    md = oi.render_md(inv)
    assert "| rms_norm | rms_norm | pattern | 6 | 5 | 17 |" in md
    dump = {"ttnn_version": "t1", "names": ["add"]}
    r = fixture("qwen3-tiny-seq")
    md2 = oi.render_md(oi.build(r, mapping=MAPPING, decomps=DECOMPS, dump=dump, installed_ttnn="t1", ignored=IGNORED))
    assert "none of rms_norm is in the dump | not claimed" in md2


def test_markdown_sorts_candidates_after_unmapped_and_before_mapped():
    inv = build(raw(op("aten.__and__.Tensor"), op("aten.einsum.default"), op("aten.add.Tensor")))
    md = oi.render_md(inv)
    assert md.index("aten.einsum.default") < md.index("aten.__and__.Tensor") < md.index("aten.add.Tensor")


def test_record_in_delta_carries_candidates_patterns_and_never_lists_a_candidate_as_unmapped(tmp_path):
    inv = real_build("qwen3-tiny-seq")
    sd = tmp_path / "s"
    sd.mkdir()
    (sd / "ops-inventory.json").write_text(json.dumps(inv))
    (sd / "delta.json").write_text("{}")
    assert oi.record_in_delta(sd)
    rec = json.loads((sd / "delta.json").read_text())["inventory"]
    assert rec["unmapped_ops"] == ["aten.diff.default"]
    assert {c["op"] for c in rec["candidate_ops"]} == {"aten.__and__.Tensor", "aten.index.Tensor",
                                                       "aten.to.dtype_layout", "aten.new_ones.default",
                                                       "aten.to.device"}
    assert rec["candidate_ops"][0]["candidates"]
    assert {p["pattern"] for p in rec["patterns"]} == {"rms_norm", "silu_mul", "rotary"}
    assert rec["distinct"]["candidate"] == 5 and rec["distinct"]["mapped"] == 26


def test_run_inventory_reads_the_verified_file_next_to_the_tables(tmp_path):
    # a fake runner: the trace is the fixture, the interpreter answers the ttnn version
    raw_text = (FIXTURES / "raw-qwen3-tiny.json").read_text()

    def run(argv, timeout):
        if "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text(raw_text)
            return 0, "", ""
        return 0, "0.80.0\n", ""

    say = []
    rc = oi.run_inventory(tmp_path / "m", python="py", ttnn_python="py", out_dir=tmp_path / "out", run=run,
                          say=say.append, verified_file=tmp_path / "v.json")
    assert rc == 0
    inv = json.loads((tmp_path / "out" / "ops-inventory.json").read_text())
    assert inv["summary"]["distinct"]["candidate"] == 5
    (tmp_path / "v.json").write_text(json.dumps({"entries": {"aten.__and__.Tensor": {"ttnn": "bitwise_and"}}}))
    oi.run_inventory(tmp_path / "m", python="py", ttnn_python="py", out_dir=tmp_path / "out2", run=run, say=say.append,
                     verified_file=tmp_path / "v.json")
    inv2 = json.loads((tmp_path / "out2" / "ops-inventory.json").read_text())
    assert inv2["summary"]["distinct"]["verified"] == 1 and inv2["summary"]["distinct"]["candidate"] == 4
    assert any("candidate 5" in line for line in say)


def test_run_inventory_names_the_groups_it_found_and_says_candidates_are_not_mapped(tmp_path):
    raw_text = (FIXTURES / "raw-qwen3-tiny-seq.json").read_text()

    def run(argv, timeout):
        if "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text(raw_text)
            return 0, "", ""
        return 0, "0.80.0\n", ""

    say = []
    oi.run_inventory(tmp_path / "m", python="py", ttnn_python="py", out_dir=tmp_path / "out", run=run, say=say.append,
                     verified_file=tmp_path / "none.json")
    text = "\n".join(say)
    assert "op groups recognised: rms_norm -> rms_norm (17), silu_mul -> swiglu (4)" in text
    assert "candidates are guesses and are not mapped" in text and "verify-candidates" in text


def test_the_shipped_verified_file_is_used_by_default_and_turns_recorded_candidates_into_verified(tmp_path):
    raw_text = (FIXTURES / "raw-qwen3-tiny.json").read_text()

    def run(argv, timeout):
        if "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text(raw_text)
            return 0, "", ""
        return 0, "0.80.0\n", ""

    oi.run_inventory(tmp_path / "m", python="py", ttnn_python="py", out_dir=tmp_path / "out", run=run,
                     say=lambda *_: None)
    d = json.loads((tmp_path / "out" / "ops-inventory.json").read_text())["summary"]["distinct"]
    assert d["verified"] == 5 and d["candidate"] == 0 and d["mapped"] == 26 and d["unmapped"] == 1
