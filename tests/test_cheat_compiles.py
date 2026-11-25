# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A cheat-sheet claim about a chip call needs a compile result from the chip package's own compiler (issue 801, need 3).

Before: `op_cheats.verify` judged `[compiles on hardware]` and `[hardware signature only]` by finding the name and the argument names in the
chip package's Python files. `ttl.math.fill` and `ttl.math.broadcast` are defined in those files and the compiler refuses them inside a kernel
(`ttl.math.fill is not available; use ttl.block.fill`), so the check passed while the sheet taught a form the chip would not compile.

After: `orchard/cheat_compiles.py` reads recorded compiles (`ttlang_cheat_compiles.json`: small probe kernels, each compiled with each chip
package's compiler and no device) and `verify` takes them. A `compiles on hardware` claim needs a compiled record for the hardware package; a
`hardware signature only` claim must not be contradicted by a refusal; a form that a probe records as refused may not be an entry at all.
These tests use made-up records for the logic and, where the packages are installed, the real compilers for the data.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from orchard import cheat_compiles as cc
from orchard import op_cheats
from orchard import ttlang_hwcompile as hc
from test_ttlang_hwcompile import PAIR, SHARED, on_116, on_pair

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "kernel-loop-templates"

PAIR_ID = {"package": "tt-lang", "version": "1.1.10.dev20261026+light", "compiler_sha256": "a" * 64, "files": ["_src/ttl_ast.py"]}
OLD_ID = {"package": "tt-lang", "version": "1.1.6", "compiler_sha256": "b" * 64, "files": ["_src/ttl_ast.py"]}
SAME_VERSION_OTHER_BYTES = dict(PAIR_ID, compiler_sha256="c" * 64)


def sheet() -> dict:
    return copy.deepcopy(op_cheats.load_sheet(op_cheats.TTLANG_SHEET))


def set_status(sh: dict, name: str, status: str) -> None:
    op_cheats.find_entry(sh, name)["status"] = status


def kernel_lines(tag: str) -> list:
    return ["import ttl", f"# probe {tag}", "x = 1"]


def sha_of(lines: list) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def record(category: str, target: dict, lines: list, first_line: str = "") -> dict:
    """A record shaped as orchard/ttlang_hwcompile.py writes it."""
    ok = category == "compiled"
    rec = {"schema": hc.SCHEMA, "category": category, "compiled": ok, "target": target, "kernel_sha256": sha_of(lines), "operation": "probe",
           "shapes": [[64, 256]], "first_line": first_line, "kernel_sources": [{"name": "k.cpp", "bytes": 10, "sha256": "d" * 64}] if ok else [],
           "stand_in_used": [], "grid": None, "arch": None, "guard": [{"state": "refused", "path": "/dev/tenstorrent"}] if ok else []}
    return rec


def probe(expect: str, names: list, results: dict, tag: str = "p") -> dict:
    """results lists (target, category, first_line): one compile record per chip package."""
    lines = kernel_lines(tag)
    body = {"expect": expect, "shapes": [[64, 256]], "kernel": lines, "kernel_sha256": sha_of(lines),
            "compile": [{"date": "2026-11-24", "record": record(cat, tgt, lines, first)} for tgt, cat, first in results]}
    body["covers" if expect == "compiled" else "refuses"] = names
    return body


def data(probes: dict, targets=(PAIR_ID, OLD_ID)) -> dict:
    return {"schema": cc.SCHEMA, "date": "2026-11-24", "note": "test data", "targets": list(targets), "probes": probes}


def claim_names(sh: dict, status: str) -> list:
    return [e["name"] for e in op_cheats.entries(sh) if e.get("status") == status]


def full_data(sh: dict, *, skip=(), refuse=()) -> dict:
    """Records that satisfy every claim of the sheet: one compiled probe per `compiles on hardware` entry."""
    probes = {}
    for name in claim_names(sh, "compiles on hardware"):
        if name in skip:
            continue
        cat = ("compile_error", "error: the compiler refused it") if name in refuse else ("compiled", "")
        probes["p-" + name] = probe("compiled", [name], [(PAIR_ID, *cat)], tag=name)
    return data(probes)


# ---- the claims ------------------------------------------------------------------------------------------------------------------------

def test_a_sheet_whose_claims_all_have_a_compiled_record_has_no_problem():
    sh = sheet()
    assert cc.claim_problems(sh, full_data(sh), PAIR_ID) == []


def test_a_compiles_on_hardware_claim_with_no_record_is_a_problem_that_names_the_entry():
    sh = sheet()
    problems = cc.claim_problems(sh, full_data(sh, skip=("ttl.copy",)), PAIR_ID)
    assert any(p.startswith("ttl.copy:") and "no compile" in p for p in problems), problems


def test_a_compiles_on_hardware_claim_whose_only_record_is_a_refusal_is_a_problem_with_the_compilers_line():
    sh = sheet()
    problems = cc.claim_problems(sh, full_data(sh, refuse=("ttl.copy",)), PAIR_ID)
    assert any(p.startswith("ttl.copy:") and "refused" in p and "the compiler refused it" in p for p in problems), problems


def test_one_compiled_record_among_refusals_is_enough_for_the_claim():
    sh = sheet()
    d = full_data(sh, skip=("ttl.copy",))
    d["probes"]["bad"] = probe("compiled", ["ttl.copy"], [(PAIR_ID, *("compile_error", "x"))], tag="bad")
    d["probes"]["good"] = probe("compiled", ["ttl.copy"], [(PAIR_ID, *("compiled", ""))], tag="good")
    assert cc.claim_problems(sh, d, PAIR_ID) == []


@pytest.mark.parametrize("category", ["guard_failed", "timeout", "not_run", "bad_result"])
def test_a_check_that_could_not_be_made_is_not_a_compile(category):
    """A broken instrument is not a kernel that compiles, and it is not a refusal either."""
    sh = sheet()
    d = full_data(sh, skip=("ttl.copy",))
    d["probes"]["broken"] = probe("compiled", ["ttl.copy"], [(PAIR_ID, *(category, "the check did not run"))], tag="broken")
    problems = cc.claim_problems(sh, d, PAIR_ID)
    assert any(p.startswith("ttl.copy:") and "no compile" in p for p in problems), problems


def test_a_record_made_with_another_compiler_does_not_count():
    sh = sheet()
    d = full_data(sh, skip=("ttl.copy",))
    d["probes"]["old"] = probe("compiled", ["ttl.copy"], [(OLD_ID, *("compiled", ""))], tag="old")
    problems = cc.claim_problems(sh, d, PAIR_ID)
    assert any(p.startswith("ttl.copy:") and "no compile" in p for p in problems), problems


def test_the_same_version_with_other_compiler_bytes_is_another_compiler():
    sh = sheet()
    d = full_data(sh)
    problems = cc.claim_problems(sh, d, SAME_VERSION_OTHER_BYTES)
    assert any("were not made with this hardware package" in p for p in problems), problems


def test_a_hardware_package_the_data_does_not_name_is_one_problem_for_the_sheet_not_one_per_entry():
    sh = sheet()
    problems = cc.claim_problems(sh, full_data(sh), OLD_ID | {"version": "9.9.9"})
    assert sum("were not made with this hardware package" in p for p in problems) == 1, problems


# ---- the weaker tag --------------------------------------------------------------------------------------------------------------------

def test_a_hardware_signature_only_claim_may_have_no_record_because_the_tag_says_so():
    sh = sheet()
    names = claim_names(sh, "hardware signature only")
    assert names, "the sheet has hardware-signature-only entries"
    assert cc.claim_problems(sh, full_data(sh), PAIR_ID) == []


def test_a_hardware_signature_only_claim_the_compiler_refuses_is_a_problem():
    """The case of issue 801: the name is in the package's files and the compiler refuses the call."""
    sh = sheet()
    name = claim_names(sh, "hardware signature only")[0]
    d = full_data(sh)
    d["probes"]["refused"] = probe("compiled", [name], [(PAIR_ID, *("compile_error", "error: x is not available; use y"))], tag="refused")
    problems = cc.claim_problems(sh, d, PAIR_ID)
    assert any(p.startswith(f"{name}:") and "x is not available; use y" in p for p in problems), problems


def test_a_hardware_signature_only_claim_with_a_compiled_record_is_not_a_problem_but_is_listed_as_understated():
    sh = sheet()
    name = claim_names(sh, "hardware signature only")[0]
    d = full_data(sh)
    d["probes"]["ok"] = probe("compiled", [name], [(PAIR_ID, *("compiled", ""))], tag="ok")
    assert cc.claim_problems(sh, d, PAIR_ID) == []
    assert cc.understated(sh, d, PAIR_ID) == [name]


def test_nothing_is_understated_when_no_weak_claim_has_a_compiled_record():
    sh = sheet()
    assert cc.understated(sh, full_data(sh), PAIR_ID) == []


# ---- a form the compiler refuses may not be an entry -----------------------------------------------------------------------------------

def refused_form_data(sh: dict, form: str) -> dict:
    d = full_data(sh)
    d["probes"]["no-" + form] = probe("refused", [form], [(PAIR_ID, *("compile_error", f"error: {form} is not available; use another"))], tag=form)
    return d


@pytest.mark.parametrize("status", ["compiles on hardware", "hardware signature only"])
def test_an_entry_for_a_form_the_hardware_compiler_refuses_is_a_problem_whatever_its_tag(status):
    sh = sheet()
    entry = copy.deepcopy(op_cheats.find_entry(sh, "ttl.block.fill"))
    entry.update(name="ttl.math.fill", status=status, signature="ttl.math.fill(value, *, shape)", usage="half = ttl.math.fill(0.5, shape=(1, 1))")
    sh["sections"][2]["entries"].append(entry)
    problems = cc.claim_problems(sh, refused_form_data(sh, "ttl.math.fill"), PAIR_ID)
    assert any(p.startswith("ttl.math.fill:") and "refuses" in p and "is not available" in p for p in problems), problems


def test_a_refused_form_that_is_not_an_entry_is_fine():
    sh = sheet()
    assert cc.claim_problems(sh, refused_form_data(sh, "ttl.math.fill"), PAIR_ID) == []


def test_a_probe_expected_to_be_refused_that_compiled_is_a_problem_in_the_data():
    sh = sheet()
    d = full_data(sh)
    d["probes"]["no-fill"] = probe("refused", ["ttl.math.fill"], [(PAIR_ID, *("compiled", ""))], tag="fill")
    problems = cc.claim_problems(sh, d, PAIR_ID)
    assert any("no-fill" in p and "and it compiled" in p for p in problems), problems
    assert not any("no verdict" in p for p in problems), problems          # a compile is a verdict: the other message is for a check that gave none


def test_a_refusal_that_does_not_name_the_form_is_not_a_refusal_of_it():
    """A probe can fail for a reason of its own (a shape mismatch). Only the compiler's refusal of the named form counts, and its line names it."""
    sh = sheet()
    d = full_data(sh)
    d["probes"]["no-fill"] = probe("refused", ["ttl.math.fill"], [(PAIR_ID, "compile_error", "error: shape mismatch between two tensors")], tag="fill")
    problems = cc.claim_problems(sh, d, PAIR_ID)
    assert any("no-fill" in p and "does not name" in p for p in problems), problems
    assert cc.refused_forms(d, PAIR_ID) == {}


def test_a_refused_probe_whose_hardware_record_is_a_broken_check_is_not_taken_as_a_refusal():
    sh = sheet()
    entry = copy.deepcopy(op_cheats.find_entry(sh, "ttl.block.fill"))
    entry.update(name="ttl.math.fill", signature="ttl.math.fill(value, *, shape)", usage="half = ttl.math.fill(0.5, shape=(1, 1))")
    sh["sections"][2]["entries"].append(entry)
    d = full_data(sh)
    d["probes"]["no-fill"] = probe("refused", ["ttl.math.fill"], [(PAIR_ID, *("timeout", "the compile took too long"))], tag="fill")
    problems = cc.claim_problems(sh, d, PAIR_ID)
    assert any("no-fill" in p and "no verdict" in p for p in problems), problems


# ---- the data must be what it says -----------------------------------------------------------------------------------------------------

def test_a_record_for_other_kernel_bytes_than_the_probe_text_is_a_problem():
    sh = sheet()
    d = full_data(sh)
    next(iter(d["probes"].values()))["kernel"].append("# edited after the compile")
    problems = cc.claim_problems(sh, d, PAIR_ID)
    assert any("kernel_sha256" in p for p in problems), problems


def test_a_record_that_names_another_kernel_than_its_probe_is_a_problem():
    sh = sheet()
    d = full_data(sh)
    first = next(iter(d["probes"].values()))
    first["compile"][0]["record"]["kernel_sha256"] = "e" * 64
    problems = cc.claim_problems(sh, d, PAIR_ID)
    assert any("the record is for another kernel" in p for p in problems), problems


def test_a_probe_that_covers_a_name_the_sheet_does_not_have_is_a_problem():
    sh = sheet()
    d = full_data(sh)
    d["probes"]["orphan"] = probe("compiled", ["ttl.no_such_call"], [(PAIR_ID, *("compiled", ""))], tag="orphan")
    problems = cc.claim_problems(sh, d, PAIR_ID)
    assert any("orphan" in p and "ttl.no_such_call" in p for p in problems), problems


def malformed_probe() -> dict:
    bad = probe("compiled", ["ttl.copy"], [(PAIR_ID, "compiled", "")], tag="bad")
    bad["compile"][0]["record"]["guard"] = []          # a compiled record must carry the proof that the device was shut out
    return bad


def test_a_malformed_record_is_refused_with_the_data_and_nothing_else_is_judged():
    sh = sheet()
    d = full_data(sh, skip=("ttl.copy",))
    d["probes"]["bad"] = malformed_probe()
    problems = cc.claim_problems(sh, d, PAIR_ID)
    assert len(problems) == 1 and "bad" in problems[0] and "shut out" in problems[0], problems


def test_a_malformed_record_never_counts_as_a_compile():
    """The data's shape check names it; `usable_records` is the second guard, for a caller that reads records without the shape check."""
    assert cc.usable_records(malformed_probe(), PAIR_ID) == []
    assert len(cc.usable_records(probe("compiled", ["ttl.copy"], [(PAIR_ID, "compiled", "")], tag="ok"), PAIR_ID)) == 1


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d.update(schema="orchard-cheat-compiles/0"), "schema"),
    (lambda d: d.update(targets="nope"), "targets"),
    (lambda d: d.update(probes=[]), "probes"),
    (lambda d: d["probes"]["p-ttl.copy"].update(expect="maybe"), "expect"),
    (lambda d: d["probes"]["p-ttl.copy"].update(covers=[]), "covers"),
    (lambda d: d["probes"]["p-ttl.copy"].update(kernel="import ttl"), "kernel"),
    (lambda d: d["probes"]["p-ttl.copy"].update(kernel_sha256="zz"), "kernel_sha256"),
    (lambda d: d["probes"]["p-ttl.copy"].update(compile={}), "compile"),
])
def test_a_data_file_of_the_wrong_shape_is_refused_with_a_reason(mutate, expect):
    sh = sheet()
    d = full_data(sh)
    mutate(d)
    problems = cc.claim_problems(sh, d, PAIR_ID)
    assert problems and any(expect in p for p in problems), problems


def test_a_hardware_target_that_is_not_an_identity_is_a_problem():
    sh = sheet()
    problems = cc.claim_problems(sh, full_data(sh), {"version": "1"})
    assert problems and any("hardware package" in p for p in problems), problems


# ---- verify ----------------------------------------------------------------------------------------------------------------------------

def test_verify_says_so_when_no_compile_results_are_given():
    problems = op_cheats.verify(sheet(), {})
    assert any("no compile results were given" in p for p in problems), problems


def test_verify_says_so_when_the_hardware_package_is_not_identified():
    sh = sheet()
    problems = op_cheats.verify(sh, {}, compiles=full_data(sh))
    assert any("the hardware package was not identified" in p for p in problems), problems


def test_verify_adds_the_claim_problems_to_its_own():
    sh = sheet()
    problems = op_cheats.verify(sh, {}, compiles=full_data(sh, skip=("ttl.copy",)), hardware_target=PAIR_ID)
    assert any(p.startswith("ttl.copy:") and "no compile" in p for p in problems), problems
    assert not any("no compile results were given" in p for p in problems)


def test_the_ttnn_sheet_needs_no_compile_results():
    ttnn = op_cheats.load_sheet(op_cheats.TTNN_SHEET)
    problems = op_cheats.verify(ttnn, {})
    assert not any("compile" in p for p in problems), problems


# ---- the shipped data ------------------------------------------------------------------------------------------------------------------

def shipped() -> dict:
    return cc.load()


def pair_identity() -> dict:
    return hc.package_identity(PAIR)


def test_the_shipped_data_is_well_formed():
    assert cc.shape_problems(shipped()) == []


def test_the_shipped_data_gives_every_compiles_on_hardware_entry_a_compiled_record_for_the_pair():
    """Judged from the data alone, with the identity the data names for the pair (no compiler runs)."""
    d = shipped()
    pair = next(t for t in d["targets"] if t["version"].endswith("+light"))
    assert cc.claim_problems(op_cheats.load_sheet(op_cheats.TTLANG_SHEET), d, pair) == []


def test_the_shipped_data_records_that_the_pair_refuses_math_fill_and_math_broadcast():
    d = shipped()
    pair = next(t for t in d["targets"] if t["version"].endswith("+light"))
    refused = cc.refused_forms(d, pair)
    assert set(refused) >= {"ttl.math.fill", "ttl.math.broadcast"}, refused
    assert all("is not available" in line for line in refused.values()), refused


def test_the_shipped_sheet_has_no_entry_for_a_refused_form():
    d = shipped()
    names = {e["name"] for e in op_cheats.entries(op_cheats.load_sheet(op_cheats.TTLANG_SHEET))}
    for t in d["targets"]:
        assert not names & set(cc.refused_forms(d, t)), t["version"]


def test_the_probe_kernels_use_the_forms_they_say_they_cover():
    """A probe that does not contain the call it covers would pass without testing it."""
    d = shipped()
    for name, p in d["probes"].items():
        text = "\n".join(p["kernel"])
        for covered in p.get("covers", []) + p.get("refuses", []):
            leaf = covered.rsplit(".", 1)[-1]
            assert re.search(rf"\b{re.escape(leaf)}\b", text), (name, covered)


# ---- the probes ------------------------------------------------------------------------------------------------------------------------

def test_a_probe_change_that_does_not_apply_exactly_once_is_an_error():
    from orchard import cheat_probes as cp
    with pytest.raises(cp.ProbeError, match="occurs 0 times"):
        cp._change("abc", "x", "y")
    with pytest.raises(cp.ProbeError, match="occurs 2 times"):
        cp._change("xx", "x", "y")
    assert cp._change("axb", "x", "y") == "ayb"


def test_every_probe_differs_from_its_base_kernel_and_from_every_other_probe():
    """A probe equal to its base would pass without exercising the call it claims to cover."""
    from orchard import cheat_probes as cp
    defs = cp.definitions()
    texts = [d["text"] for d in defs]
    assert len(set(texts)) == len(texts)
    for d in defs:
        if d["name"] not in ("elementwise", "reduce-sum-broadcast-fill"):
            assert d["text"] not in (cp.ELEMENTWISE, cp.REDUCTION), d["name"]


def test_the_shipped_data_holds_the_kernels_the_definitions_write():
    """The data names the text that was compiled; a change to a definition that was not followed by scripts/verify_cheat_compiles.py is caught here."""
    from orchard import cheat_probes as cp
    d = shipped()
    defs = {p["name"]: p for p in cp.definitions()}
    assert set(d["probes"]) == set(defs)
    for name, body in d["probes"].items():
        assert body["kernel"] == defs[name]["text"].rstrip("\n").split("\n"), name
        assert body["expect"] == defs[name]["expect"], name
        assert body.get("covers", body.get("refuses")) == defs[name].get("covers", defs[name].get("refuses")), name


def test_every_entry_the_sheet_tags_compiles_on_hardware_is_covered_by_a_probe():
    d = shipped()
    covered = {n for p in d["probes"].values() for n in p.get("covers", [])}
    claimed = set(claim_names(sheet(), "compiles on hardware"))
    assert claimed <= covered, claimed - covered


# ---- the data against the real compilers (no device) ----------------------------------------------------------------------------------

def recompile(python, probe_body: dict, tmp_path: Path) -> dict:
    path = tmp_path / "kernel.py"
    path.write_text(cc.probe_text(probe_body), encoding="utf-8")
    return hc.compile_check(python, path, [tuple(s) for s in probe_body["shapes"]])


@on_pair
def test_every_recorded_verdict_for_the_pair_is_what_the_pair_compiler_says_now(tmp_path):
    ident = pair_identity()
    for name, body in shipped()["probes"].items():
        recorded = [c["record"] for c in body["compile"] if hc.same_target(c["record"]["target"], ident)]
        assert recorded, f"{name}: no record for the installed pair"
        got = recompile(PAIR, body, tmp_path)
        assert got["category"] == recorded[0]["category"], (name, got["first_line"], recorded[0]["first_line"])
        assert got["kernel_sha256"] == recorded[0]["kernel_sha256"] == cc.probe_sha256(body), name


@on_116
def test_every_recorded_verdict_for_1_1_6_is_what_that_compiler_says_now(tmp_path):
    ident = hc.package_identity(SHARED)
    for name, body in shipped()["probes"].items():
        recorded = [c["record"] for c in body["compile"] if hc.same_target(c["record"]["target"], ident)]
        assert recorded, f"{name}: no record for the installed 1.1.6"
        assert recompile(SHARED, body, tmp_path)["category"] == recorded[0]["category"], name


@on_pair
def test_the_static_check_alone_passes_math_fill_and_the_compile_records_catch_it(tmp_path):
    """The failure of issue 801 replayed: the name is a Python function in the pair's files, so the old check found nothing wrong."""
    ttl_dir = hc.package_dir(PAIR)
    sh = sheet()
    entry = copy.deepcopy(op_cheats.find_entry(sh, "ttl.block.fill"))
    entry.update(name="ttl.math.fill", status="hardware signature only", signature="ttl.math.fill(value, *, shape)",
                 usage="half = ttl.math.fill(0.5, shape=(1, 1))", **{"in": ["hardware"]})
    entry.pop("evidence", None)
    sh["sections"][2]["entries"].append(entry)
    hardware = op_cheats.build_index(ttl_dir)
    static = [p for p in op_cheats.verify(sh, {"hardware": hardware, "tree": hardware, "sim": hardware, "shared": hardware}, compiles=shipped(),
                                           hardware_target=pair_identity()) if p.startswith("ttl.math.fill:")]
    assert any("refuses" in p for p in static), static
    assert not any("not found" in p for p in static), static            # the static half has nothing to say: the name is there


@on_pair
def test_the_identity_of_a_package_directory_is_the_identity_of_its_interpreter():
    """scripts/verify_cheat_sheet.py takes the `ttl/` directory of the hardware package and has no interpreter argument."""
    assert hc.identity_of(hc.package_dir(PAIR)) == hc.package_identity(PAIR)


def test_the_identity_of_a_directory_that_is_not_a_compiler_package_is_an_error(tmp_path):
    (tmp_path / "ttl").mkdir()
    with pytest.raises(hc.HwCompileError):
        hc.identity_of(tmp_path / "ttl")


# ---- the commands ----------------------------------------------------------------------------------------------------------------------

def _script(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_cheat_sheet_names_a_hardware_root_that_is_not_a_compiler_package(tmp_path, capsys):
    """The identity of the hardware package comes from its directory; a directory without the compiler is an error."""
    empty = tmp_path / "ttl"
    empty.mkdir()
    argv = ["ttlang"] + [a for name in ("hardware", "shared", "tree", "sim") for a in ("--root", f"{name}={empty}")]
    assert _script("verify_cheat_sheet").main(argv) == 2
    assert "could not be read" in capsys.readouterr().out


def test_verify_cheat_sheet_names_a_compile_file_that_is_not_there(tmp_path, capsys):
    empty = tmp_path / "ttl"
    empty.mkdir()
    argv = ["ttlang", "--compiles", str(tmp_path / "missing.json")] + [a for name in ("hardware", "shared", "tree", "sim") for a in ("--root", f"{name}={empty}")]
    assert _script("verify_cheat_sheet").main(argv) == 2
    assert "compile records" in capsys.readouterr().out


CHECKOUT = Path.home() / "orchard-src" / "tt-lang" / "python"


@on_pair
def test_verify_cheat_sheet_passes_the_shipped_sheet_against_the_real_trees_with_the_shipped_compiles(capsys):
    shared = hc.package_dir(SHARED) if SHARED.exists() else None
    if not (shared and (CHECKOUT / "ttl").is_dir() and (CHECKOUT / "sim").is_dir()):
        pytest.skip("the shared venv or the tt-lang checkout is not on this machine")
    argv = ["ttlang", "--root", f"hardware={hc.package_dir(PAIR)}", "--root", f"shared={shared}",
            "--root", f"tree={CHECKOUT / 'ttl'}", "--root", f"sim={CHECKOUT / 'sim'}"]
    assert _script("verify_cheat_sheet").main(argv) == 0
    out = capsys.readouterr().out
    assert "all found" in out and "no compile results" not in out
    for name in ("ttl.matmul", "ttl.math.exp"):                     # raised tags are suggested, never forced
        assert f"note: {name} is tagged 'hardware signature only'" in out


def fake_compile(python, kernel, shapes, **kw):
    """A compile_check that returns the record the data holds for the kernel's text: what the script writes, with no compiler."""
    text = Path(kernel).read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    refused = "ttl.math.fill" in text or "ttl.math.broadcast" in text
    ident = kw["identity"]
    lines = text.rstrip("\n").split("\n")
    rec = record("compile_error" if refused else "compiled", ident, lines, "error: ttl.math.fill is not available; use ttl.block.fill" if refused else "")
    assert rec["kernel_sha256"] == sha
    return rec


def test_the_generator_writes_data_that_passes_its_own_shape_check_and_names_each_probes_kernel(monkeypatch):
    gen = _script("verify_cheat_compiles")
    monkeypatch.setattr(gen.ttlang_hwcompile, "package_identity", lambda python: PAIR_ID if "pair" in str(python) else OLD_ID)
    data = gen.build([Path("/x/pair/bin/python"), Path("/x/old/bin/python")], "2026-11-24", fake_compile, say=lambda *_: None)
    assert cc.shape_problems(data) == []
    assert [t["version"] for t in data["targets"]] == [PAIR_ID["version"], OLD_ID["version"]]
    assert gen.expectation_failures(data) == []
    assert all(len(p["compile"]) == 2 for p in data["probes"].values())


def test_the_generator_notices_a_probe_that_was_expected_to_be_refused_and_compiled():
    gen = _script("verify_cheat_compiles")
    sh = sheet()
    d = full_data(sh)
    d["probes"]["no-fill"] = probe("refused", ["ttl.math.fill"], [(PAIR_ID, "compiled", "")], tag="fill")
    failures = gen.expectation_failures(d)
    assert any("no-fill" in f and "expected a refusal" in f for f in failures), failures


def test_the_generator_notices_a_hardware_package_that_refused_a_probe_expected_to_compile():
    gen = _script("verify_cheat_compiles")
    d = full_data(sheet(), refuse=("ttl.copy",))
    failures = gen.expectation_failures(d)
    assert any("hardware package refused" in f for f in failures), failures


def test_the_generator_does_not_mind_another_package_refusing_a_probe_that_the_hardware_package_takes():
    """tt-lang 1.1.6 has no ttl.make_dfb: a record of that is expected."""
    gen = _script("verify_cheat_compiles")
    d = full_data(sheet())
    next(iter(d["probes"].values()))["compile"].append({"date": "2026-11-24", "record": record("compile_error", OLD_ID, ["x"], "no attribute")})
    assert gen.expectation_failures(d) == []


def test_a_check_that_gave_no_verdict_is_a_failure_in_the_generator():
    gen = _script("verify_cheat_compiles")
    d = full_data(sheet())
    next(iter(d["probes"].values()))["compile"][0]["record"] = record("timeout", PAIR_ID, ["x"], "took too long")
    assert any("did not give a verdict" in f for f in gen.expectation_failures(d))


def test_the_difference_report_names_a_changed_verdict_and_a_changed_kernel():
    gen = _script("verify_cheat_compiles")
    old = full_data(sheet())
    new = copy.deepcopy(old)
    name = next(iter(new["probes"]))
    new["probes"][name]["compile"][0]["record"]["category"] = "compile_error"
    new["probes"][name]["kernel_sha256"] = "f" * 64
    out = gen.differences(old, new)
    assert any("kernel changed" in d for d in out) and any("compiled" in d and "compile_error" in d for d in out), out
    assert gen.differences(old, old) == []
