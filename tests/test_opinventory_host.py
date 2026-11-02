# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `host` status, the op notes table and the routed-expert group (orchard/opinventory.py, orchard/oppatterns.py).

The Qwen4 trace leaves seven ops unmapped. These tests pin how each one is now classified, how the loader refuses a
malformed notes table, and how the expert group is recognised one condition at a time. The real trace fixture is the
4-layer Flash-Next trace (meta device, `--stub-data-dependent`)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import opinventory as oi
from orchard import oppatterns as op

FIXTURES = Path(__file__).parent / "fixtures" / "ops-inventory"
NOTES_FILE = oi.DATA_DIR / "op_notes.json"
NOTE = {"reason": "built on the host", "evidence": "probe 4", "host": True}


def mini(ops=("aten.cummax.default",)):
    return {"schema": 1, "model_dir": "/m", "architectures": ["X"], "traced_class": "X", "layers_in_config": 4,
            "layers_traced": 4, "seq_len": 8, "torch": "2", "transformers": "5", "failures": [],
            "ops": [{"op": o, "count": 2, "count_in_layers": 0, "sample": None, "modules": ["model"]} for o in ops]}


def rows(inv):
    return {r["op"]: r for r in inv["ops"]}


# ---- status_of -------------------------------------------------------------------------------------------------------

def test_a_host_op_is_host_with_or_without_a_dump():
    host = {"aten.cummax.default": NOTE}
    assert oi.status_of("aten.cummax.default", {}, {}, {"add"}, host=host) == ("host", None, [])
    assert oi.status_of("aten.cummax.default", {}, {}, None, host=host) == ("host", None, [])


def test_without_a_host_entry_the_same_op_is_unmapped():
    assert oi.status_of("aten.cummax.default", {}, {}, {"add"}, host={})[0] == "unmapped"


def test_an_ignored_op_stays_ignored_even_if_a_notes_table_marks_it_host():
    got = oi.status_of("aten.alias.default", {}, {}, {"add"}, ignored=frozenset({"aten.alias.default"}),
                       host={"aten.alias.default": NOTE})
    assert got[0] == "ignored"


def test_a_table_row_wins_over_a_host_entry():
    got = oi.status_of("aten.add.Tensor", {"aten.add.Tensor": {"ttnn": "add"}}, {}, {"add"},
                       host={"aten.add.Tensor": NOTE})
    assert got[0] == "mapped"


def test_a_host_leaf_makes_a_decomposition_composable_only_for_mapped_leaves():
    """A host leaf is not built from ttnn ops, so a decomposition that needs one is not composable (leaves are looked up without the notes, so a host leaf is simply unmapped)."""
    got = oi.status_of("aten.x.default", {"aten.add.Tensor": {"ttnn": "add"}},
                       {"aten.x.default": ["aten.add.Tensor", "aten.cummax.default"]}, {"add"},
                       host={"aten.cummax.default": NOTE})
    assert got[0] == "unmapped"


# ---- build and the report --------------------------------------------------------------------------------------------

def test_build_counts_host_ops_notes_them_and_the_markdown_explains_host():
    notes = {"aten.cummax.default": NOTE}
    inv = oi.build(mini(), mapping={}, decomps={}, dump={"ttnn_version": "1", "names": ["add"]}, installed_ttnn="1",
                   notes=notes)
    assert rows(inv)["aten.cummax.default"]["status"] == "host"
    assert rows(inv)["aten.cummax.default"]["note"] == {"reason": "built on the host", "evidence": "probe 4"}
    assert inv["summary"]["distinct"]["host"] == 1 and inv["summary"]["scaled_total"]["host"] == 2
    md = oi.render_md(inv)
    assert "| host | 1 |" in md and "run on the host by design" in md
    assert "built on the host" in md and "probe 4" in md


def test_a_trace_without_notes_shows_no_notes_section_and_a_zero_host_count():
    inv = oi.build(mini(), mapping={}, decomps={}, dump=None)
    assert inv["summary"]["distinct"]["host"] == 0 and "note" not in rows(inv)["aten.cummax.default"]
    assert "Notes on ops" not in oi.render_md(inv)


def test_a_note_on_a_non_host_op_is_shown_and_does_not_change_its_status():
    notes = {"aten.cummax.default": {"reason": "why", "evidence": "what"}}
    inv = oi.build(mini(), mapping={}, decomps={}, dump={"ttnn_version": "1", "names": ["add"]}, installed_ttnn="1",
                   notes=notes)
    assert rows(inv)["aten.cummax.default"]["status"] == "unmapped"
    assert "Notes on ops" in oi.render_md(inv)


# ---- load_notes: the table format ------------------------------------------------------------------------------------

def write(tmp_path, data):
    p = tmp_path / "n.json"
    p.write_text(json.dumps(data))
    return p


def test_load_notes_reads_a_good_table(tmp_path):
    got = oi.load_notes(write(tmp_path, {"aten.a.default": {"reason": "r", "evidence": "e"},
                                         "aten.b.default": {"reason": "r", "evidence": "e", "host": True}}))
    assert got["aten.a.default"]["host"] is False and got["aten.b.default"]["host"] is True


@pytest.mark.parametrize("bad", [
    [], "x", 3,                                                  # not an object
    {"aten.a": "text"}, {"aten.a": []}, {"aten.a": None},        # entry not an object
    {"aten.a": {"evidence": "e"}}, {"aten.a": {"reason": "r"}},  # a missing field
    {"aten.a": {"reason": "", "evidence": "e"}}, {"aten.a": {"reason": "r", "evidence": "  "}},
    {"aten.a": {"reason": 1, "evidence": "e"}}, {"aten.a": {"reason": "r", "evidence": ["e"]}},
    {"aten.a": {"reason": "r", "evidence": "e", "host": "yes"}},      # host must be a boolean
    {"aten.a": {"reason": "r", "evidence": "e", "ttnn": "add"}},      # unknown key
    {"": {"reason": "r", "evidence": "e"}},                           # empty op name
])
def test_load_notes_refuses_a_malformed_table(tmp_path, bad):
    with pytest.raises(ValueError):
        oi.load_notes(write(tmp_path, bad))


def test_load_notes_refuses_a_missing_file_and_non_json(tmp_path):
    with pytest.raises(ValueError):
        oi.load_notes(tmp_path / "absent.json")
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(ValueError):
        oi.load_notes(p)


# ---- the shipped table on the real Flash-Next trace ------------------------------------------------------------------

def real_inventory():
    raw = json.loads((FIXTURES / "raw-qwen4-flash-next.json").read_text())
    dump = oi.load_dump("0.80.0")
    assert dump is not None
    return oi.build(raw, mapping=oi.load_table(oi.DATA_DIR / "aten_to_ttnn.json"),
                    decomps=oi.load_table(oi.DATA_DIR / "decompositions.json"), dump=dump, installed_ttnn="0.80.0",
                    ignored=oi.load_ignored(oi.DATA_DIR / "ignored_ops.json"), notes=oi.load_notes(NOTES_FILE),
                    verified=oi.load_verified())


def test_unflatten_is_a_decomposition_to_reshape_and_cites_the_probe():
    r = rows(real_inventory())["aten.unflatten.int"]
    assert r["status"] == "composable" and r["leaves"] == ["aten.view.default"]
    assert "reshape" in r["note"]["reason"] and "0.99996" in r["note"]["evidence"]


@pytest.mark.parametrize("name", ["aten.eye.default", "aten.cummax.default", "aten.diff.default"])
def test_eye_cummax_and_diff_are_host(name):
    r = rows(real_inventory())[name]
    assert r["status"] == "host" and r["note"]["reason"] and r["note"]["evidence"]


def test_eye_names_the_ttnn_op_that_takes_it_as_a_constant():
    assert "chunk_gated_delta_rule" in rows(real_inventory())["aten.eye.default"]["note"]["reason"]


def test_diff_names_its_source_and_says_whether_it_matters_on_the_device():
    note = rows(real_inventory())["aten.diff.default"]["note"]
    assert "find_packed_sequence_indices" in note["evidence"] and "masking_utils" in note["evidence"]
    assert "device" in note["reason"]


def test_the_expert_loop_ops_stay_unmapped_and_say_where_they_are_handled():
    inv = real_inventory()
    for name in ("aten.histc.default", "aten.index_put_.default", "transformers.grouped_mm_fallback.default"):
        r = rows(inv)[name]
        assert r["status"] == "unmapped" and "experimental.moe_compute" in r["note"]["reason"], name


def test_after_the_notes_only_the_three_expert_ops_are_unmapped_and_the_summary_counts_host():
    inv = real_inventory()
    left = sorted(r["op"] for r in inv["ops"] if r["status"] == "unmapped")
    assert left == ["aten.histc.default", "aten.index_put_.default", "transformers.grouped_mm_fallback.default"]
    assert inv["summary"]["distinct"]["host"] == 3


def test_every_note_names_an_op_in_a_fixture_and_host_notes_do_not_overlap_the_other_tables():
    seen = set()
    for f in FIXTURES.glob("raw-*.json"):
        seen |= {o["op"] for o in json.loads(f.read_text()).get("ops", [])}
    notes = oi.load_notes(NOTES_FILE)
    assert notes and not sorted(set(notes) - seen)
    host = {k for k, v in notes.items() if v["host"]}
    assert host == {"aten.eye.default", "aten.cummax.default", "aten.diff.default"}
    assert not host & set(oi.load_table(oi.DATA_DIR / "aten_to_ttnn.json"))
    assert not host & oi.load_ignored(oi.DATA_DIR / "ignored_ops.json")
    assert not host & set(oi.load_table(oi.DATA_DIR / "decompositions.json"))


def test_the_summary_table_includes_host():
    inv = real_inventory()
    assert "| host | 3 |" in oi.render_md(inv)
    assert "host" in oi.STATUSES


# ---- the routed-expert group -----------------------------------------------------------------------------------------

MOE = "model.layers.0.mlp.experts"
NAMES = {"experimental.moe_compute"}


def rec(ident, opname, module=MOE, ins=()):
    return {"id": ident, "op": opname, "in": list(ins), "module": module, "shape": [512], "const": []}


def experts(module=MOE, extra=()):
    return [rec("h", "aten.histc.default", module), rec("p", "aten.index_put_.default", module),
            rec("g", "transformers.grouped_mm_fallback.default", module), *extra]


def test_the_three_expert_ops_in_one_module_are_one_pattern_with_the_ttnn_op():
    res = op.recognise(experts(), NAMES, 48)
    [f] = res["found"]
    assert f["pattern"] == "moe_experts" and f["ttnn"] == "experimental.moe_compute" and f["occurrences"] == 1
    assert f["modules"] == [MOE] and f["scaled_count"] == 48 and f["nodes"] == 3


def test_the_pattern_carries_a_note_that_says_it_was_not_run_and_what_a_check_needs():
    [f] = op.recognise(experts(), NAMES, 48)["found"]
    note = f["note"]
    assert "not run" in note and "4 dies" in note and "512 experts" in note and "top 10" in note


def test_a_child_module_of_the_experts_module_joins_the_group():
    seq = experts(extra=[rec("a", "aten.silu.default", MOE + ".act_fn")])
    [f] = op.recognise(seq, NAMES, 4)["found"]
    assert f["nodes"] == 4


@pytest.mark.parametrize("drop", ["h", "p", "g"])
def test_a_module_missing_one_of_the_three_ops_is_not_a_group(drop):
    assert op.recognise([n for n in experts() if n["id"] != drop], NAMES, 4)["found"] == []


def test_the_three_ops_in_different_modules_are_not_a_group():
    seq = [rec("h", "aten.histc.default", "model.layers.0.mlp.experts"),
           rec("p", "aten.index_put_.default", "model.layers.1.mlp.experts"),
           rec("g", "transformers.grouped_mm_fallback.default", "model.layers.2.mlp.experts")]
    assert op.recognise(seq, NAMES, 4)["found"] == []


@pytest.mark.parametrize("module", ["model.layers.0.mlp", "model.layers.0.mlp.shared_expert", "model", "experts",
                                    "model.layers.0.mlp.experts_extra", "x.mlp.experts.y"])
def test_the_ops_outside_an_mlp_experts_module_are_not_a_group(module):
    assert op.recognise(experts(module), NAMES, 4)["found"] == []


def test_two_expert_modules_are_two_occurrences_and_the_modules_are_listed():
    seq = experts("model.layers.0.mlp.experts") + [dict(n, id=n["id"] + "2", module="model.layers.1.mlp.experts")
                                                    for n in experts()]
    [f] = op.recognise(seq, NAMES, 4)["found"]
    assert f["occurrences"] == 2 and f["modules"] == ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts"]


def test_without_the_ttnn_op_in_the_dump_nothing_is_claimed():
    res = op.recognise(experts(), {"add"}, 4)
    assert res["found"] == [] and res["unavailable"][0]["pattern"] == "moe_experts"


def test_build_lists_the_group_as_pattern_never_pattern_verified_and_the_markdown_shows_the_note():
    inv = real_inventory()
    [f] = [f for f in inv["patterns"] if f["pattern"] == "moe_experts"]
    assert f["status"] == "pattern" and f["ttnn"] == "experimental.moe_compute"
    md = oi.render_md(inv)
    assert "moe_experts" in md and "4 dies" in md


def test_the_member_ops_keep_their_own_status_when_the_group_is_found():
    inv = real_inventory()
    assert rows(inv)["aten.histc.default"]["status"] == "unmapped"
