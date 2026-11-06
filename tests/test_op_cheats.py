# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The verified cheat sheets (issue 523, orchard/op_cheats.py). The logic is tested on a small fake source tree, so the tests run anywhere;
the shipped sheets are checked against the real trees when the environment names them (the pinned tt-metal tree, the installed tt-lang
package, the tt-lang checkout), and each such test says which variable it needs. Nothing here imports ttnn, ttl or ttml."""
import copy
import json
import os
import re
from pathlib import Path

import pytest

from orchard import op_cheats as oc

TTNN_TREE = os.environ.get("ORCHARD_TTNN_TREE")                    # a directory that holds ttnn/ (the tt-metal tree of the coder image)
TTL_HARDWARE = os.environ.get("ORCHARD_TTLANG_HARDWARE")           # the installed tt-lang package directory (ttl/)
TTL_TREE = os.environ.get("ORCHARD_TTLANG_TREE")                   # the checkout's python/ttl
TTL_SIM = os.environ.get("ORCHARD_TTLANG_SIM")                     # the checkout's python/sim


# ---- a fake tree ----------------------------------------------------------------------------------------------------------------------

NANOBIND = '''
void bind(nb::module_& mod) {
    ttnn::bind_function<"foo">(
        mod, doc, &ttnn::foo,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("epsilon") = 1e-12,
        nb::arg("weight") = nb::none());
    ttnn::bind_function<"bar">(mod, doc, &ttnn::bar, nb::arg("x"), nb::arg("y"));
    ttnn::bind_function<"bar">(mod, doc, &ttnn::bar2, nb::arg("x"), nb::arg("z"));
    ttnn::bind_function<"rot">(mod, doc, &ttnn::rot, nb::arg("q"));
    detail::bind_unary<"relu">(mod, doc);
}
'''
PYTHON = '''
def helper(tensor, *, dtype=None, layout=None):
    return tensor

class Box:
    def __init__(self, src, dst):
        pass
    def push(self, item):
        pass

LAYOUT = 1
'''


def make_tree(root: Path, *, extra_py: str = "") -> Path:
    (root / "ttnn" / "cpp" / "ttnn" / "operations" / "experimental" / "rot").mkdir(parents=True)
    (root / "ttnn" / "cpp" / "ttnn" / "operations" / "norm").mkdir(parents=True)
    (root / "ttnn" / "cpp" / "ttnn" / "operations" / "experimental" / "rot" / "rot_nanobind.cpp").write_text('ttnn::bind_function<"rot">(mod, d, &f, nb::arg("q"));')
    (root / "ttnn" / "cpp" / "ttnn" / "operations" / "norm" / "norm_nanobind.cpp").write_text(NANOBIND.replace('    ttnn::bind_function<"rot">(mod, doc, &ttnn::rot, nb::arg("q"));\n', ""))
    (root / "ttnn" / "ttnn").mkdir(parents=True)
    (root / "ttnn" / "ttnn" / "core.py").write_text(PYTHON + extra_py)
    return root


def entry(name, signature, usage, **kw):
    base = {"name": name, "kind": "op", "signature": signature, "usage": usage, "notes": "a note", "seen_in": "somewhere", "in": ["tt_metal"]}
    base.update(kw)
    return base


def sheet_of(*entries, kind="ttnn", roots=None, mistakes=()):
    return {"schema": oc.SCHEMA, "sheet": kind, "title": "t", "roots": roots or {"tt_metal": "the tree"},
            "sections": [{"title": "s", "entries": list(entries)}], "mistakes": list(mistakes)}


GOOD = entry("ttnn.foo", "ttnn.foo(input_tensor, *, epsilon=1e-6, weight=None)", "y = ttnn.foo(x, epsilon=1e-6, weight=w)")


@pytest.fixture
def index(tmp_path):
    return {"tt_metal": oc.build_index(make_tree(tmp_path / "tree"))}


# ---- finding a name and its arguments -------------------------------------------------------------------------------------------------

def test_a_binding_gives_the_argument_names_inside_its_own_call(index):
    found, args, where = oc.resolve(index["tt_metal"], "ttnn.foo")
    assert found and args == {"input_tensor", "epsilon", "weight"} and where[0].endswith("norm_nanobind.cpp")


def test_a_python_definition_gives_its_parameters_and_a_class_gives_its_constructor_arguments(index):
    assert oc.resolve(index["tt_metal"], "ttnn.helper")[1] == {"tensor", "dtype", "layout"}
    assert oc.resolve(index["tt_metal"], "ttnn.Box")[1] == {"src", "dst"}
    assert oc.resolve(index["tt_metal"], "Box.push")[1] == {"item"}


def test_a_module_level_constant_exists_with_no_arguments_and_a_template_registration_exists_by_name(index):
    assert oc.resolve(index["tt_metal"], "ttnn.LAYOUT")[:2] == (True, set())
    assert oc.resolve(index["tt_metal"], "ttnn.relu") == (True, None, [])


def test_an_absent_name_is_not_found(index):
    assert oc.resolve(index["tt_metal"], "ttnn.does_not_exist")[0] is False


def test_the_namespace_prefix_is_checked_against_the_path_of_the_binding(index):
    assert oc.resolve(index["tt_metal"], "ttnn.experimental.rot")[0] is True
    assert oc.resolve(index["tt_metal"], "ttnn.experimental.foo")[0] is False       # foo is bound under norm/, not experimental/


# ---- verify ---------------------------------------------------------------------------------------------------------------------------

def test_a_correct_entry_has_no_problem(index):
    assert oc.verify(sheet_of(GOOD), index) == []


def test_mutation_a_fake_op_turns_the_check_red(index):
    fake = entry("ttnn.fake_op", "ttnn.fake_op(x)", "y = ttnn.fake_op(x)")
    problems = oc.verify(sheet_of(GOOD, fake), index)
    assert problems and "ttnn.fake_op" in problems[0] and "not found" in problems[0]


def test_mutation_an_invented_argument_in_the_signature_turns_the_check_red(index):
    bad = entry("ttnn.foo", "ttnn.foo(input_tensor, eps=1e-6)", "y = ttnn.foo(x)")
    assert any("'eps'" in p for p in oc.verify(sheet_of(bad), index))


def test_mutation_an_invented_keyword_in_the_usage_line_turns_the_check_red(index):
    bad = entry("ttnn.foo", "ttnn.foo(input_tensor, *, epsilon=1e-6)", "y = ttnn.foo(x, gamma=w)")
    assert any("'gamma'" in p for p in oc.verify(sheet_of(bad), index))


def test_the_arguments_of_one_overload_must_fit_one_definition(index):
    both = entry("ttnn.bar", "ttnn.bar(x, y, z)", "y = ttnn.bar(x, y)")             # y is in one binding and z in the other: no call takes both
    assert any("no single definition" in p for p in oc.verify(sheet_of(both), index))
    ok = entry("ttnn.bar", "ttnn.bar(x, z)", "y = ttnn.bar(x, z=1)")
    assert oc.verify(sheet_of(ok), index) == []


def test_a_registered_name_with_no_readable_arguments_may_not_be_used_with_a_keyword(index):
    assert oc.verify(sheet_of(entry("ttnn.relu", "ttnn.relu(x)", "y = ttnn.relu(x)")), index) == []
    assert any("keyword" in p for p in oc.verify(sheet_of(entry("ttnn.relu", "ttnn.relu(x)", "y = ttnn.relu(x, inplace=True)")), index))


def test_a_usage_that_does_not_call_the_name_or_does_not_parse_is_a_problem(index):
    assert oc.verify(sheet_of(entry("ttnn.foo", "ttnn.foo(input_tensor)", "y = ttnn.bar(x)")), index)
    assert oc.verify(sheet_of(entry("ttnn.foo", "ttnn.foo(input_tensor)", "y = ttnn.foo(x")), index)


def test_a_claimed_root_that_was_not_read_is_a_problem_and_not_a_pass(tmp_path):
    empty = oc.build_index(tmp_path / "nothing-here")
    assert empty.files == 0 and empty.incomplete
    assert any("was not read" in p for p in oc.verify(sheet_of(GOOD), {"tt_metal": empty}))
    twice = oc.verify(sheet_of(GOOD, entry("ttnn.bar", "ttnn.bar(x, y)", "y = ttnn.bar(x, y)")), {"tt_metal": empty})
    assert len([p for p in twice if "root 'tt_metal' was not read" in p and not p.startswith("mistake")]) == 1
    assert any("was not read" in p for p in oc.verify(sheet_of(GOOD), {}))


def test_a_mistake_check_that_names_an_absent_argument_is_verified_both_ways(index):
    ok = {"symptom": "s", "fix": "f", "seen_in": "x", "checks": [{"root": "tt_metal", "name": "ttnn.foo", "absent_args": ["gamma"], "present_args": ["weight"]}]}
    assert oc.verify(sheet_of(GOOD, mistakes=[ok]), index) == []
    wrong = copy.deepcopy(ok)
    wrong["checks"][0]["absent_args"] = ["weight"]
    assert any("claimed absent" in p for p in oc.verify(sheet_of(GOOD, mistakes=[wrong]), index))
    wrong["checks"][0] = {"root": "tt_metal", "name": "ttnn.foo", "present_args": ["gamma"]}
    assert any("claimed present" in p for p in oc.verify(sheet_of(GOOD, mistakes=[wrong]), index))
    wrong["checks"][0] = {"root": "tt_metal", "name": "ttnn.foo", "absent": True}
    assert any("claimed absent" in p for p in oc.verify(sheet_of(GOOD, mistakes=[wrong]), index))


# ---- the TT-Lang statuses -------------------------------------------------------------------------------------------------------------

@pytest.fixture
def lang(tmp_path):
    hw, sim = tmp_path / "hw", tmp_path / "sim"
    hw.mkdir()
    sim.mkdir()
    (hw / "ops.py").write_text("def reduce_sum(input, *, dims):\n    pass\n\ndef copy(src, dst):\n    pass\n")
    (hw / "sim").mkdir()
    (hw / "sim" / "math.py").write_text("def reduce_sum(block, dims, shape):\n    pass\n")      # the wheel's own simulator: it must not count
    (sim / "math.py").write_text("def reduce_sum(block, dims, shape):\n    pass\n\ndef sim_fill(value, shape):\n    pass\n")
    return {"hardware": oc.build_index(hw), "sim": oc.build_index(sim)}


def lang_sheet(*entries, mistakes=()):
    return sheet_of(*entries, kind="ttlang", roots={"hardware": "h", "sim": "s"}, mistakes=mistakes)


def lang_entry(name, signature, usage, status, roots, **kw):
    return entry(name, signature, usage, status=status, **{"in": roots}, **kw)


def test_the_simulator_inside_the_installed_package_is_not_part_of_the_hardware_surface(lang):
    assert oc.resolve(lang["hardware"], "ttl.math.reduce_sum")[1] == {"input", "dims"}


def test_a_compiles_on_hardware_entry_needs_evidence_and_a_hardware_signature(lang):
    e = lang_entry("ttl.copy", "ttl.copy(src, dst)", "tx = ttl.copy(a, b)", "compiles on hardware", ["hardware"])
    assert any("evidence" in p for p in oc.verify(lang_sheet(e), lang))
    e["evidence"] = "ran on the chip"
    assert oc.verify(lang_sheet(e), lang) == []
    e["signature"] = "ttl.copy(src, dst, shape)"
    assert oc.verify(lang_sheet(e), lang)


def test_a_simulator_only_entry_lists_what_the_hardware_package_lacks_and_the_claim_is_checked(lang):
    e = lang_entry("sim.reduce_sum", "ttl.math.reduce_sum(block, dims, shape)", "s = ttl.math.reduce_sum(x, dims=[-1], shape=(1, 1))", "simulator only", ["sim"],
                   hardware_absent=["shape"])
    assert oc.verify(lang_sheet(e), lang) == []
    e["hardware_absent"] = ["dims"]                                   # the hardware package has dims
    assert any("is there" in p for p in oc.verify(lang_sheet(e), lang))
    e["hardware_absent"] = []
    assert any("hardware_absent" in p for p in oc.verify(lang_sheet(e), lang))


def test_a_simulator_only_entry_that_claims_the_hardware_root_is_refused(lang):
    e = lang_entry("sim.reduce_sum", "ttl.math.reduce_sum(block, dims, shape)", "s = ttl.math.reduce_sum(x, dims=[1], shape=(1, 1))", "simulator only",
                   ["sim", "hardware"], hardware_absent=["shape"])
    assert oc.verify(lang_sheet(e), lang)


def test_a_hardware_absent_name_that_the_hardware_package_has_is_refused(lang):
    e = lang_entry("ttl.copy", "ttl.copy(src, dst)", "tx = ttl.copy(a, b)", "simulator only", ["sim"], hardware_absent=["copy"])
    assert any("is listed as absent" in p for p in oc.verify(lang_sheet(e), lang))


def test_a_whole_name_the_hardware_package_lacks_can_be_listed_and_is_checked(lang):
    e = lang_entry("ttl.sim_fill", "ttl.sim_fill(value, shape)", "x = ttl.sim_fill(1, (1, 1))", "simulator only", ["sim"], hardware_absent=["sim_fill"])
    assert oc.verify(lang_sheet(e), lang) == []
    e["hardware_absent"] = ["reduce_sum"]                              # not this entry's name, so it is an argument claim and the name is absent
    assert oc.verify(lang_sheet(e), lang)


# ---- the shape of a sheet -------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["name", "signature", "usage", "notes", "seen_in"])
@pytest.mark.parametrize("bad", [None, [], {}, 5, "", "  ", "a\0b", "x" * 601])
def test_a_bad_text_field_is_a_problem_and_never_an_exception(field, bad):
    e = copy.deepcopy(GOOD)
    e[field] = bad
    assert oc.shape_problems(sheet_of(e))


@pytest.mark.parametrize("bad", [None, [], "tt_metal", ["nope"], [1], {}])
def test_the_in_field_must_list_named_roots(bad):
    e = copy.deepcopy(GOOD)
    e["in"] = bad
    assert any("`in`" in p for p in oc.shape_problems(sheet_of(e)))


@pytest.mark.parametrize("sheet", [None, [], "s", 3, {}, {"schema": oc.SCHEMA}, {"schema": oc.SCHEMA, "sheet": "ttnn", "title": "t", "roots": {"a": "b"}, "sections": {}}])
def test_a_sheet_of_the_wrong_shape_is_refused_without_an_exception(sheet):
    assert oc.shape_problems(sheet)


def test_a_duplicate_name_a_bad_kind_and_a_dotted_garbage_name_are_refused():
    assert any("twice" in p for p in oc.shape_problems(sheet_of(GOOD, GOOD)))
    assert oc.shape_problems(sheet_of(dict(GOOD, kind="thing")))
    assert oc.shape_problems(sheet_of(dict(GOOD, name="ttnn..foo")))
    assert oc.shape_problems(sheet_of(dict(GOOD, name="ttnn.foo; rm -rf")))


def test_a_sheet_file_that_is_a_link_or_a_fifo_or_too_large_is_refused_with_one_error(tmp_path):
    real = tmp_path / "real.json"
    real.write_text(json.dumps(sheet_of(GOOD)))
    assert oc.load_sheet(real)["sheet"] == "ttnn"
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(oc.SheetError):
        oc.load_sheet(link)
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(oc.SheetError):
        oc.load_sheet(fifo)
    big = tmp_path / "big.json"
    big.write_text(" " * (oc.SHEET_MAX_BYTES + 1))
    with pytest.raises(oc.SheetError):
        oc.load_sheet(big)
    with pytest.raises(oc.SheetError):
        oc.load_sheet(tmp_path / "missing.json")


# ---- a hostile tree -------------------------------------------------------------------------------------------------------------------

def test_a_link_or_fifo_in_the_tree_is_not_read_and_a_huge_file_is_skipped_and_noted(tmp_path):
    root = make_tree(tmp_path / "tree")
    outside = tmp_path / "outside.py"
    outside.write_text("def planted(a):\n    pass\n")
    (root / "ttnn" / "ttnn" / "linked.py").symlink_to(outside)
    os.mkfifo(root / "ttnn" / "ttnn" / "fifo.py")
    (root / "ttnn" / "ttnn" / "huge.py").write_text("def planted_huge(a):\n    pass\n" + "#" * 4_000_010)
    idx = oc.build_index(root)
    assert oc.resolve(idx, "ttnn.planted")[0] is False
    assert oc.resolve(idx, "ttnn.planted_huge")[0] is False         # over the byte cap: not read
    assert idx.incomplete                                           # the huge file was skipped and the index says so
    assert oc.resolve(idx, "ttnn.foo")[0] is True


def test_a_python_file_that_does_not_parse_is_skipped_without_an_exception(tmp_path):
    root = make_tree(tmp_path / "tree", extra_py="\ndef broken(:\n")
    assert oc.build_index(root).files > 0


# ---- the shipped sheets ---------------------------------------------------------------------------------------------------------------

SHEETS = {"ttnn": oc.TTNN_SHEET, "ttlang": oc.TTLANG_SHEET}


@pytest.mark.parametrize("which", sorted(SHEETS))
def test_each_shipped_sheet_has_the_right_shape_and_names_no_machine_path_or_token(which):
    path = SHEETS[which]
    sheet = oc.load_sheet(path)
    assert sheet["sheet"] == which
    text = path.read_text(encoding="utf-8")
    assert not re.search(r"/home/|/mnt/|hf_[A-Za-z0-9]{10}|~/", text), "a machine path or token is in the sheet"


@pytest.mark.parametrize("which", sorted(SHEETS))
def test_each_shipped_sheet_renders_deterministically_within_the_line_budget(which):
    sheet = oc.load_sheet(SHEETS[which])
    text = oc.render(sheet)
    assert text == oc.render(copy.deepcopy(sheet))
    assert len(text.splitlines()) < oc.RENDER_LINE_BUDGET, len(text.splitlines())
    for e in oc.entries(sheet):
        assert e["signature"] in text and all(ln in text for ln in e["usage"].split("\n"))


def test_the_ttnn_sheet_covers_every_family_the_issue_names():
    names = {e["name"] for e in oc.entries(oc.load_sheet(oc.TTNN_SHEET))}
    for needed in ("ttnn.embedding", "ttnn.linear", "ttnn.matmul", "ttnn.rms_norm", "ttnn.experimental.rotary_embedding_llama",
                   "ttnn.transformer.scaled_dot_product_attention", "ttnn.transformer.scaled_dot_product_attention_decode",
                   "ttnn.experimental.paged_update_cache", "ttnn.argmax", "ttnn.topk", "ttnn.sampling", "ttnn.to_layout", "ttnn.typecast",
                   "ttnn.all_gather", "ttnn.reduce_scatter", "ttnn.all_reduce", "ttnn.MatmulMultiCoreReuseMultiCastProgramConfig"):
        assert needed in names, needed


def test_the_ttnn_sheet_has_a_mistake_for_each_failure_in_runs_16_to_18():
    text = " ".join(m["symptom"] for m in oc.load_sheet(oc.TTNN_SHEET)["mistakes"])
    for needle in ("prefill not implemented yet", "[6, 32] but got: [6, 6]", "unterminated string literal", "ttnn.conv1d"):
        assert needle in text, needle


def test_the_ttlang_sheet_marks_the_reduce_sum_shape_pitfall_as_simulator_only():
    sheet = oc.load_sheet(oc.TTLANG_SHEET)
    by = {e["name"]: e for e in oc.entries(sheet)}
    assert by["sim.reduce_sum"]["status"] == "simulator only" and "shape" in by["sim.reduce_sum"]["hardware_absent"]
    assert "shape" not in by["ttl.math.reduce_sum"]["signature"]
    assert any("issue 205" in m["seen_in"] for m in sheet["mistakes"])


def need(var, value):
    if not value:
        pytest.skip(f"set {var} to a source tree to check the shipped sheet against it (CPU only; nothing is imported)")
    return value


def test_the_shipped_ttnn_sheet_matches_the_tt_metal_tree():
    tree = Path(need("ORCHARD_TTNN_TREE", TTNN_TREE))
    problems = oc.verify(oc.load_sheet(oc.TTNN_SHEET), {"tt_metal": oc.build_index(tree)})
    assert problems == []


def test_the_shipped_ttlang_sheet_matches_the_hardware_package_and_the_checkout():
    roots = {"hardware": need("ORCHARD_TTLANG_HARDWARE", TTL_HARDWARE), "tree": need("ORCHARD_TTLANG_TREE", TTL_TREE), "sim": need("ORCHARD_TTLANG_SIM", TTL_SIM)}
    problems = oc.verify(oc.load_sheet(oc.TTLANG_SHEET), {k: oc.build_index(Path(v)) for k, v in roots.items()})
    assert problems == []


def test_mutation_a_fake_op_added_to_the_shipped_ttnn_sheet_is_found_on_the_real_tree():
    tree = Path(need("ORCHARD_TTNN_TREE", TTNN_TREE))
    sheet = oc.load_sheet(oc.TTNN_SHEET)
    sheet["sections"][0]["entries"].append(entry("ttnn.fake_op", "ttnn.fake_op(x)", "y = ttnn.fake_op(x)"))
    problems = oc.verify(sheet, {"tt_metal": oc.build_index(tree)})
    assert problems and all("fake_op" in p for p in problems)
