# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import opinventory as oi

FIXTURES = Path(__file__).parent / "fixtures" / "ops-inventory"
MAPPING = {"aten.mm.default": {"ttnn": "matmul"}, "aten.embedding.default": {"ttnn": "embedding"},
           "aten.add.Tensor": {"ttnn": "add"}}
DECOMPS = {"aten.addmm.default": ["aten.mm.default", "aten.add.Tensor"],
           "aten.bad.default": ["aten.mm.default", "aten.weird_scan.default"]}
DUMP = {"ttnn_version": "0.65.1", "names": ["matmul", "embedding", "add"]}


def op(name, count, in_layers, modules=("model.layers.0.x",)):
    return {"op": name, "count": count, "count_in_layers": in_layers, "sample": None, "modules": list(modules)}


def raw(**over):
    base = {"schema": 1, "model_dir": "/m", "architectures": ["XForCausalLM"], "traced_class": "XForCausalLM",
            "layers_in_config": 24, "layers_traced": 2, "seq_len": 8, "torch": "2", "transformers": "5", "failures": [],
            "ops": [op("aten.mm.default", 10, 8), op("aten.embedding.default", 1, 0, ("model.embed",)),
                    op("aten.weird_scan.default", 3, 3), op("aten.addmm.default", 2, 2), op("aten.bad.default", 1, 1)]}
    base.update(over)
    return base


def rows(inv):
    return {r["op"]: r for r in inv["ops"]}


def test_a_name_in_the_matching_dump_is_mapped():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.mm.default"]["status"] == "mapped" and rows(inv)["aten.mm.default"]["ttnn"] == "matmul"


NOTE_NO_DUMP = "no recorded dump for the installed ttnn; nothing is claimed as mapped"
NOTE_OTHER = "the recorded dump is for ttnn 0.65.1 and ttnn 0.70.0 is installed; not used"
NOTE_UNKNOWN = "the installed ttnn version is unknown, so the recorded dump is not used"


@pytest.mark.parametrize("dump,installed,note", [(None, "0.65.1", NOTE_NO_DUMP), (DUMP, "0.70.0", NOTE_OTHER),
                                                 (DUMP, None, NOTE_UNKNOWN)],
                         ids=["no dump", "dump for another version", "installed version unknown"])
def test_nothing_is_claimed_without_a_dump_for_the_installed_version(dump, installed, note):
    """Review focus 2. Each case has its own exact note."""
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=dump, installed_ttnn=installed)
    assert {r["status"] for r in inv["ops"]} == {"unchecked"}
    assert inv["ttnn_dump"] is None and inv["ttnn_dump_note"] == note


def test_an_op_with_no_mapping_and_no_decomposition_is_unmapped():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.weird_scan.default"]["status"] == "unmapped"


def test_a_decomposition_with_every_leaf_mapped_is_composable():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    r = rows(inv)["aten.addmm.default"]
    assert r["status"] == "composable" and r["leaves"] == ["aten.mm.default", "aten.add.Tensor"]


def test_a_decomposition_with_an_unmapped_leaf_is_unmapped():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.bad.default"]["status"] == "unmapped"


def test_a_decomposition_cycle_does_not_loop():
    cyc = {"aten.a.default": ["aten.b.default"], "aten.b.default": ["aten.a.default"]}
    assert oi.status_of("aten.a.default", {}, cyc, {"matmul"})[0] == "unmapped"


def test_counts_in_layers_are_scaled_to_the_real_depth():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert rows(inv)["aten.mm.default"]["scaled_count"] == 8 * 12 + 2      # 8 in layers x (24 / 2) + 2 outside
    assert rows(inv)["aten.embedding.default"]["scaled_count"] == 1


def test_without_a_layer_count_nothing_is_scaled_and_the_report_says_so():
    """Review focus 5."""
    inv = oi.build(raw(layers_in_config=None, layers_traced=None), mapping=MAPPING, decomps=DECOMPS, dump=DUMP,
                   installed_ttnn="0.65.1")
    assert rows(inv)["aten.mm.default"]["scaled_count"] == 10 and "not scaled" in inv["scaling"]


def test_a_skipped_trace_gives_an_inventory_with_the_reason_and_no_ops():
    inv = oi.build({"schema": 1, "model_dir": "/m", "architectures": [], "skipped": "needs remote code"},
                   mapping=MAPPING, decomps=DECOMPS, dump=DUMP)
    assert inv["skipped"] == "needs remote code" and inv["ops"] == []


def test_against_marks_ops_the_base_model_also_uses():
    base = oi.build(raw(ops=[op("aten.mm.default", 1, 1)]), mapping=MAPPING, decomps=DECOMPS, dump=DUMP,
                    installed_ttnn="0.65.1")
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1", against=base)
    assert rows(inv)["aten.mm.default"]["in_base"] is True and rows(inv)["aten.weird_scan.default"]["in_base"] is False


def test_the_summary_counts_ops_by_status():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert inv["summary"]["distinct"] == {"mapped": 2, "composable": 1, "unmapped": 2, "unchecked": 0, "ignored": 0}


def test_a_nested_decomposition_is_composable_and_lists_the_final_leaves():
    nested = {"aten.outer.default": ["aten.addmm.default", "aten.embedding.default"], **DECOMPS}
    s, _, leaves = oi.status_of("aten.outer.default", MAPPING, nested, {"matmul", "embedding", "add"})
    assert s == "composable" and leaves == ["aten.mm.default", "aten.add.Tensor", "aten.embedding.default"]


def test_an_ignored_op_is_ignored_with_or_without_a_dump():
    ig = frozenset({"aten.alias.default"})
    assert oi.status_of("aten.alias.default", {}, {}, {"add"}, ignored=ig)[0] == "ignored"
    assert oi.status_of("aten.alias.default", {}, {}, None, ignored=ig)[0] == "ignored"
    assert oi.status_of("aten.alias.default", {}, {}, {"add"})[0] == "unmapped"


def test_a_metadata_only_leaf_needs_nothing_so_the_decomposition_is_still_composable():
    d = {"aten.x.default": ["aten.mm.default", "aten.alias.default"]}
    got = oi.status_of("aten.x.default", MAPPING, d, {"matmul"}, ignored=frozenset({"aten.alias.default"}))
    assert got == ("composable", None, ["aten.mm.default"])
    assert oi.status_of("aten.x.default", MAPPING, d, {"matmul"})[0] == "unmapped"


def test_build_counts_ignored_ops_and_markdown_says_they_are_metadata_only():
    r = raw(ops=[op("aten.alias.default", 4, 4), op("aten.mm.default", 1, 1)])
    inv = oi.build(r, mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1",
                   ignored=frozenset({"aten.alias.default"}))
    assert rows(inv)["aten.alias.default"]["status"] == "ignored" and inv["summary"]["distinct"]["ignored"] == 1
    assert inv["summary"]["scaled_total"]["ignored"] == 4 * 12
    md = oi.render_md(inv)
    assert "| ignored | 1 |" in md and "Ignored ops are metadata only" in md
    nodump = oi.build(r, mapping=MAPPING, decomps=DECOMPS, dump=None, ignored=frozenset({"aten.alias.default"}))
    assert rows(nodump)["aten.alias.default"]["status"] == "ignored"
    assert rows(nodump)["aten.mm.default"]["status"] == "unchecked"


def test_shipped_tables_make_composable_and_mapped_reachable_on_real_fixtures():
    """Proven on the packaged tables and the real traces, with a synthetic dump of every mapped ttnn name."""
    mapping = oi.load_table(oi.DATA_DIR / "aten_to_ttnn.json")
    decomps = oi.load_table(oi.DATA_DIR / "decompositions.json")
    ignored = oi.load_ignored(oi.DATA_DIR / "ignored_ops.json")
    dump = {"ttnn_version": "9.9.9", "names": sorted({e["ttnn"] for e in mapping.values()})}
    total = {"mapped": 0, "composable": 0}
    for f in sorted(FIXTURES.glob("raw-*.json")):
        inv = oi.build(json.loads(f.read_text()), mapping=mapping, decomps=decomps, dump=dump, installed_ttnn="9.9.9",
                       ignored=ignored)
        for k in total:
            total[k] += inv["summary"]["distinct"][k]
        for r in inv["ops"]:
            if r["op"] == "aten.select.int":
                assert r["status"] == "composable" and r["leaves"] == ["aten.slice.Tensor", "aten.squeeze.dim"]
        assert rows(inv).get("aten.alias.default", {"status": "ignored"})["status"] == "ignored"
    assert total["composable"] >= 3 and total["mapped"] >= 80, total


def test_every_ignored_op_and_decomposition_names_an_op_in_a_fixture():
    seen = set()
    for f in FIXTURES.glob("raw-*.json"):
        seen |= {o["op"] for o in json.loads(f.read_text()).get("ops", [])}
    ignored = oi.load_ignored(oi.DATA_DIR / "ignored_ops.json")
    decomps = oi.load_table(oi.DATA_DIR / "decompositions.json")
    assert ignored and not sorted(ignored - seen) and not sorted(set(decomps) - seen)
    assert not ignored & {"aten.clone.default", "aten.copy_.default"}


def test_load_ignored_refuses_anything_but_a_list_of_names(tmp_path):
    p = tmp_path / "i.json"
    p.write_text(json.dumps({"aten.x": 1}))
    with pytest.raises(ValueError):
        oi.load_ignored(p)


def test_an_outside_term_below_zero_is_clamped():
    """count_in_layers larger than count would otherwise give a negative outside term."""
    inv = oi.build(raw(ops=[op("aten.mm.default", 4, 6)]), mapping=MAPPING, decomps=DECOMPS, dump=DUMP,
                   installed_ttnn="0.65.1")
    assert rows(inv)["aten.mm.default"]["scaled_count"] == 6 * 12      # the outside term 4 - 6 is clamped to 0 (no -2)


def test_all_layers_traced_needs_no_scaling_and_says_so():
    inv = oi.build(raw(layers_in_config=2, layers_traced=2), mapping=MAPPING, decomps=DECOMPS, dump=DUMP,
                   installed_ttnn="0.65.1")
    assert inv["scaling"] == "all layers traced: no scaling needed"
    assert rows(inv)["aten.mm.default"]["scaled_count"] == 10


def test_render_md_survives_a_failure_record_with_missing_keys():
    inv = oi.build(raw(failures=[{"stage": "export"}, {"error": "boom"}, {}]), mapping=MAPPING, decomps=DECOMPS,
                   dump=DUMP, installed_ttnn="0.65.1")
    md = oi.render_md(inv)
    assert "export" in md and "boom" in md


def test_a_dump_version_that_is_a_path_is_refused(tmp_path):
    assert oi.load_dump("../x", data_dir=tmp_path) is None and oi.load_dump("", data_dir=tmp_path) is None


def test_a_version_cannot_reach_a_file_outside_the_data_directory(tmp_path):
    """The path resolves to a real, well-formed dump outside data_dir; only the version check stops it."""
    data = tmp_path / "data"
    (data / "ttnn_ops-x").mkdir(parents=True)
    evil = "x/../../evil"
    (tmp_path / "evil.json").write_text(json.dumps({"ttnn_version": evil, "names": ["add"]}))
    assert oi.load_dump(evil, data_dir=data) is None


def test_load_dump_needs_the_version_inside_the_file_to_match(tmp_path):
    (tmp_path / "ttnn_ops-0.65.1.json").write_text(json.dumps({"ttnn_version": "0.9", "names": ["add"]}))
    assert oi.load_dump("0.65.1", data_dir=tmp_path) is None


def test_a_table_entry_without_a_ttnn_name_is_refused(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"aten.x.default": {"nope": 1}}))
    with pytest.raises(ValueError):
        oi.load_table(p)


def test_every_table_entry_names_an_op_that_a_real_trace_contains():
    """A typo in a table would never match and would quietly leave an op unmapped."""
    seen = set()
    for f in FIXTURES.glob("raw-*.json"):
        seen |= {o["op"] for o in json.loads(f.read_text()).get("ops", [])}
    table = oi.load_table(oi.DATA_DIR / "aten_to_ttnn.json")
    unknown = sorted(k for k in table if k not in seen)
    assert not unknown, f"not in any fixture trace: {unknown}"


def test_the_markdown_says_what_mapped_means_and_names_the_dump():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    md = oi.render_md(inv)
    assert "name exists in the dump of ttnn 0.65.1" in md and "does not say" in md


# ---- run_inventory, with a fake command runner ----

def fake_run(template_rc, version="0.65.1", fixture="raw-qwen3-tiny.json"):
    calls = []

    def run(argv, timeout):
        calls.append(argv)
        if "--out" in argv:                       # the template call
            if template_rc in (0, 3):
                Path(argv[argv.index("--out") + 1]).write_text((FIXTURES / fixture).read_text())
            return template_rc, "", "boom" if template_rc not in (0, 3) else ""
        return 0, version + "\n", ""              # the metadata call
    run.calls = calls
    return run


def test_run_inventory_writes_both_files(tmp_path):
    said = []
    rc = oi.run_inventory("/m", python="py", ttnn_python="py", out_dir=tmp_path / "o", run=fake_run(0),
                          dump_dir=tmp_path / "dumps", say=said.append)
    assert rc == 0
    assert (tmp_path / "o" / "ops-inventory.json").is_file() and (tmp_path / "o" / "ops-inventory.md").is_file()
    inv = json.loads((tmp_path / "o" / "ops-inventory.json").read_text())
    assert {r["status"] for r in inv["ops"]} <= {"unchecked", "ignored"} and "unchecked" in {
        r["status"] for r in inv["ops"]}                                 # no dump in dump_dir
    assert any("distinct ops by status" in s for s in said)


def test_run_inventory_returns_2_when_the_template_exits_4(tmp_path):
    said = []
    rc = oi.run_inventory("/m", python="py", ttnn_python="py", out_dir=tmp_path / "o", run=fake_run(4),
                          dump_dir=tmp_path / "dumps", say=said.append)
    assert rc == 2 and not (tmp_path / "o" / "ops-inventory.json").exists()
    assert any("did not run" in s and "boom" in s for s in said)


def test_run_inventory_passes_the_class_and_sizes_to_the_template(tmp_path):
    run = fake_run(0)
    oi.run_inventory("/m", python="py", ttnn_python="py", out_dir=tmp_path, layers=3, seq=16, cls="XCls", run=run,
                     dump_dir=tmp_path / "dumps", say=lambda s: None)
    argv = run.calls[0]
    assert argv[argv.index("--class") + 1] == "XCls" and argv[argv.index("--layers") + 1] == "3" \
        and argv[argv.index("--seq") + 1] == "16"


def test_a_dump_directory_with_a_dump_and_no_tables_works(tmp_path):
    """The tables always come from the package; dump_dir only says where to look for the dump."""
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    mapping = oi.load_table(oi.DATA_DIR / "aten_to_ttnn.json")
    (dumps / "ttnn_ops-0.65.1.json").write_text(
        json.dumps({"ttnn_version": "0.65.1", "names": sorted({e["ttnn"] for e in mapping.values()})}))
    rc = oi.run_inventory("/m", python="py", ttnn_python="py", out_dir=tmp_path / "o", run=fake_run(0),
                          dump_dir=dumps, say=lambda s: None)
    inv = json.loads((tmp_path / "o" / "ops-inventory.json").read_text())
    assert rc == 0 and inv["ttnn_dump"]["ttnn_version"] == "0.65.1" and "mapped" in {r["status"] for r in inv["ops"]}


def test_with_an_unknown_installed_version_the_live_note_is_the_no_dump_note(tmp_path):
    """run_inventory passes dump=load_dump(None)=None, so build() never sees the unknown-version case from here."""
    run = fake_run(0)
    base = run

    def run2(argv, timeout):
        return (1, "", "no ttnn") if "importlib.metadata" in " ".join(argv) else base(argv, timeout)
    oi.run_inventory("/m", python="py", ttnn_python="py", out_dir=tmp_path / "o", run=run2,
                     dump_dir=tmp_path / "dumps", say=lambda s: None)
    inv = json.loads((tmp_path / "o" / "ops-inventory.json").read_text())
    assert inv["ttnn_dump_note"] == NOTE_NO_DUMP


@pytest.mark.parametrize("content", [None, "not json", "[1, 2]", '{"ops": "x"}', '{"nope": []}'],
                         ids=["missing", "malformed", "not an object", "ops not a list", "no ops"])
def test_a_bad_against_file_returns_2_before_the_trace_runs(tmp_path, content):
    against = tmp_path / "base.json"
    if content is not None:
        against.write_text(content)
    run = fake_run(0)
    said = []
    rc = oi.run_inventory("/m", python="py", ttnn_python="py", out_dir=tmp_path / "o", against=against, run=run,
                          dump_dir=tmp_path / "dumps", say=said.append)
    assert rc == 2 and run.calls == [] and any("against" in m for m in said)


def test_a_corrupt_raw_ops_file_returns_2_with_a_message(tmp_path):
    def run(argv, timeout):
        if "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text("{broken")
            return 0, "", ""
        return 0, "0.65.1\n", ""
    said = []
    rc = oi.run_inventory("/m", python="py", ttnn_python="py", out_dir=tmp_path / "o", run=run,
                          dump_dir=tmp_path / "dumps", say=said.append)
    assert rc == 2 and any("raw-ops.json" in m for m in said)


def test_template_exit_3_still_builds_an_inventory_and_the_markdown_shows_the_failures(tmp_path):
    fx = tmp_path / "raw.json"
    d = json.loads((FIXTURES / "raw-qwen3-tiny.json").read_text())
    d["failures"] = [{"stage": "export", "error": "kaboom"}]
    fx.write_text(json.dumps(d))

    def run(argv, timeout):
        if "--out" in argv:
            Path(argv[argv.index("--out") + 1]).write_text(fx.read_text())
            return 3, "", ""
        return 0, "0.65.1\n", ""
    rc = oi.run_inventory("/m", python="py", ttnn_python="py", out_dir=tmp_path / "o", run=run,
                          dump_dir=tmp_path / "dumps", say=lambda s: None)
    md = (tmp_path / "o" / "ops-inventory.md").read_text()
    assert rc == 0 and "Trace failures:" in md and "export: kaboom" in md


# ---- Task 4b: python-level nodes and higher-order bodies are carried through ------------------------------------------

def test_python_nodes_and_higher_order_nodes_are_carried_into_traced_and_the_markdown():
    r = raw()
    r["python_nodes"] = {"<built-in function getitem>": 7, "<built-in function add>": 1}
    r["higher_order_nodes"] = {"wrap_with_set_grad_enabled": 2}
    inv = oi.build(r, mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert inv["traced"]["python_nodes"] == {"<built-in function getitem>": 7, "<built-in function add>": 1}
    assert inv["traced"]["higher_order_nodes"] == {"wrap_with_set_grad_enabled": 2}
    md = oi.render_md(inv)
    assert "8 python-level nodes (getitem and similar) are not ops and are not counted" in md
    assert "bodies of 2 higher-order operations were traversed" in md


def test_an_older_raw_trace_without_the_new_keys_still_builds_and_prints_neither_line():
    inv = oi.build(raw(), mapping=MAPPING, decomps=DECOMPS, dump=DUMP, installed_ttnn="0.65.1")
    assert inv["traced"].get("python_nodes") in (None, {}) and inv["traced"].get("higher_order_nodes") in (None, {})
    md = oi.render_md(inv)
    assert "python-level nodes" not in md and "higher-order operations were traversed" not in md


# ---- Task 4b: the committed dump of ttnn 0.65.1rc17.dev6200 ------------------------------------------------------------

REAL_DUMP_VERSION = "0.65.1rc17.dev6200"
REAL_DUMP = oi.load_dump(REAL_DUMP_VERSION)

# Taken from the dump of ttnn 0.65.1rc17.dev6200 on 2026-10-16 with the fixtures regenerated the same day (bodies of
# higher-order operations traversed, python-level nodes left out of `ops`, rsub, conv1d, cos and sin mapped). They must be
# regenerated whenever the dump, a mapping table or a fixture changes.
REAL_DUMP_COUNTS = {
    "qwen3-tiny": {"mapped": 26, "composable": 0, "unmapped": 6, "ignored": 3},
    "rwkv": {"mapped": 17, "composable": 1, "unmapped": 2, "ignored": 2},
    "mamba2": {"mapped": 30, "composable": 1, "unmapped": 4, "ignored": 3},
    "moshi-lm": {"mapped": 26, "composable": 1, "unmapped": 7, "ignored": 3},
}


@pytest.mark.skipif(REAL_DUMP is None, reason=(
    f"SKIPPED: orchard/data/ttnn_ops-{REAL_DUMP_VERSION}.json is absent, so the fixtures were not checked against a "
    "real ttnn dump"))
@pytest.mark.parametrize("name", sorted(REAL_DUMP_COUNTS))
def test_fixtures_against_the_real_dump_have_no_python_nodes_or_wrappers_and_pinned_counts(name):
    inv = oi.build(json.loads((FIXTURES / f"raw-{name}.json").read_text()),
                   mapping=oi.load_table(oi.DATA_DIR / "aten_to_ttnn.json"),
                   decomps=oi.load_table(oi.DATA_DIR / "decompositions.json"), dump=REAL_DUMP,
                   installed_ttnn=REAL_DUMP_VERSION, ignored=oi.load_ignored(oi.DATA_DIR / "ignored_ops.json"))
    ops = [r["op"] for r in inv["ops"]]
    assert "<built-in function getitem>" not in ops
    assert not any("wrap_with_set_grad_enabled" in o or o.startswith("<built-in") for o in ops), ops
    assert not any(o.startswith("higher_order") for o in ops)
    d = inv["summary"]["distinct"]
    assert {k: d[k] for k in REAL_DUMP_COUNTS[name]} == REAL_DUMP_COUNTS[name]
    assert d["unchecked"] == 0


def test_rsub_scalar_and_conv1d_are_mapped_to_names_in_the_real_dump():
    mapping = oi.load_table(oi.DATA_DIR / "aten_to_ttnn.json")
    assert mapping["aten.rsub.Scalar"] == {"ttnn": "rsub"}
    assert mapping["aten.conv1d.default"] == {"ttnn": "conv1d"}
    # cos and sin appear in the rotary-embedding bodies once higher-order bodies are traversed; both are plain
    # elementwise functions with the same meaning in ttnn.
    assert mapping["aten.cos.default"] == {"ttnn": "cos"} and mapping["aten.sin.default"] == {"ttnn": "sin"}
    if REAL_DUMP is not None:
        assert {"rsub", "conv1d", "cos", "sin"} <= set(REAL_DUMP["names"])
