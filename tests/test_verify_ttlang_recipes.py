# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The recipe verifier (scripts/verify_ttlang_recipes.py): the overlay it builds for the simulator, the chip-compile entry it writes for
each chip package, and the rule for reusing a simulator record. The simulator and the chip compilers are not run here: their results
are the records that tests/test_ttlang_recipes.py and tests/test_recipes_dialect_gate.py check, and tests/test_ttlang_hwcompile.py runs
the compile check itself."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from orchard import op_cheats

REPO = Path(__file__).resolve().parent.parent


def _script():
    spec = importlib.util.spec_from_file_location("verify_ttlang_recipes_under_test", REPO / "scripts" / "verify_ttlang_recipes.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recipe(name="row-reduce-spread"):
    return copy.deepcopy(next(r for r in op_cheats.load_recipes()["recipes"] if r["name"] == name))


def test_the_overlay_adds_the_recipe_ops_and_leaves_the_shipped_templates_alone(tmp_path):
    script = _script()
    recipes = op_cheats.load_recipes()
    before = (REPO / "orchard/skills/kernel-loop-templates/references.py").read_text()
    overlay = script.build_overlay(tmp_path, recipes)
    for recipe in recipes["recipes"]:
        assert json.loads((overlay / "ops" / f"{recipe['spec']['op']}.json").read_text())["op"] == recipe["spec"]["op"]
    assert "recipe_references" in (overlay / "references.py").read_text()
    assert not (overlay / "library").exists() and not list(overlay.glob("ttlang_recipes*.json"))
    assert (REPO / "orchard/skills/kernel-loop-templates/references.py").read_text() == before
    assert not (REPO / "orchard/skills/kernel-loop-templates/ops/recipe_l2norm.json").exists()


def test_every_recipe_has_a_reference_the_overlay_can_find():
    spec = importlib.util.spec_from_file_location("recipe_references_under_test", REPO / "orchard/skills/kernel-loop-templates/recipe_references.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for recipe in op_cheats.load_recipes()["recipes"]:
        assert recipe["spec"]["reference"] in module.REFERENCES, recipe["name"]


TARGET = {"package": "tt-lang", "version": "7.7.7", "compiler_sha256": "a" * 64, "files": ["_src/ttl_ast.py"]}


def _fake_compile(verdicts):
    """A compile_check that answers from the kernel's text: `verdicts` maps a fragment of the kernel to a category."""
    calls = []

    def compile_check(python, kernel, shapes, *, identity=None, **kw):
        text = Path(kernel).read_text(encoding="utf-8")
        category = next((c for fragment, c in verdicts.items() if fragment in text), "compiled")
        calls.append({"python": python, "text": text, "shapes": shapes, "identity": identity})
        compiled = category == "compiled"
        return {"schema": "orchard-hwcompile/1", "category": category, "compiled": compiled, "target": identity,
                "kernel_sha256": hashlib.sha256(text.encode()).hexdigest(), "operation": "op", "shapes": [list(s) for s in shapes],
                "first_line": "" if compiled else f"error: {category}",
                "kernel_sources": [{"name": "a.cpp", "bytes": 1, "sha256": "0" * 64}] if compiled else [],
                "stand_in_used": [], "grid": [13, 10], "arch": "blackhole", "guard": [{"path": "/dev/tenstorrent", "state": "refused"}]}
    return compile_check, calls


def test_a_compile_entry_compiles_the_skeleton_and_every_wrong_form_with_one_identity(tmp_path, monkeypatch):
    script = _script()
    monkeypatch.setattr(script.ttlang_hwcompile, "package_identity", lambda python: dict(TARGET))
    recipe = _recipe()
    check, calls = _fake_compile({"ttl.math.fill(": "compile_error", "rsqrt(total + eps)": "compile_error", "ttl.math.broadcast(": "compile_error"})
    entry = script.compile_entry(recipe, "/venv/bin/python", tmp_path, "2026-11-22", check)
    assert entry["date"] == "2026-11-22" and entry["record"]["compiled"] is True
    assert entry["record"]["kernel_sha256"] == op_cheats.skeleton_sha256(recipe)            # the bytes the sheet prints are the bytes compiled
    assert calls[0]["text"] == op_cheats.skeleton_text(recipe)
    assert calls[0]["shapes"] == [(64, 256), (64, 256)]                                      # the spec's input, then its output
    assert [n["name"] for n in entry["negatives"]] == [n["name"] for n in recipe["compile_negatives"]]
    assert all(n["failed"] is True and n["category"] == "compile_error" for n in entry["negatives"])
    assert len(calls) == 1 + len(recipe["compile_negatives"]) and all(c["identity"] == TARGET for c in calls)
    assert op_cheats.compile_status(recipe, {"compile": [entry]}, TARGET)["level"] == "compile"


@pytest.mark.parametrize("category", ["compiled", "timeout", "guard_failed", "bad_result", "not_run", "no_kernel_written"])
def test_a_wrong_form_counts_as_refused_only_when_the_compiler_refused_it(tmp_path, monkeypatch, category):
    """A wrong form that timed out, or that the check could not run, was not refused by the compiler: it must not vouch for the check."""
    script = _script()
    monkeypatch.setattr(script.ttlang_hwcompile, "package_identity", lambda python: dict(TARGET))
    recipe = _recipe("running-total-down-rows")
    check, _ = _fake_compile({"ttl.math.fill(": category})
    entry = script.compile_entry(recipe, "/venv/bin/python", tmp_path, "2026-11-22", check)
    assert entry["negatives"][0]["failed"] is False and entry["negatives"][0]["category"] == category
    status = op_cheats.compile_status(recipe, {"compile": [entry]}, TARGET)
    assert status["claims"] == [] and status["negatives_open"] == ["fill under ttl.math"]


def test_a_simulator_record_is_reused_only_for_the_same_skeleton():
    script = _script()
    recipe = _recipe()
    old = op_cheats.load_verified()["records"][recipe["name"]]
    kept = script.reusable_simulator_record(recipe, old)
    assert kept["skeleton_sha256"] == op_cheats.skeleton_sha256(recipe) and kept["passed"] is True and "compile" not in kept
    recipe["skeleton"].append("# edited")
    assert script.reusable_simulator_record(recipe, old) is None
    for bad in (None, [], "x", {}):
        assert script.reusable_simulator_record(_recipe(), bad) is None


def test_verify_recipe_keeps_a_chip_entry_only_for_the_same_skeleton(tmp_path, monkeypatch):
    script = _script()
    monkeypatch.setattr(script.ttlang_hwcompile, "package_identity", lambda python: dict(TARGET))
    recipe = _recipe()
    old = copy.deepcopy(op_cheats.load_verified()["records"][recipe["name"]])
    old["chip"] = [{"target": TARGET, "kernel_sha256": old["skeleton_sha256"], "date": "2026-12-01", "evidence": "docs/run-logs/x.md",
                    "pcc_min": 0.9999, "passed": True}]
    check, _ = _fake_compile({})
    new = script.verify_recipe(recipe, None, tmp_path, None, ["/venv/bin/python"], None, "2026-11-22", old, check)
    assert new["chip"] == old["chip"] and len(new["compile"]) == 1
    recipe["skeleton"].append("# edited")
    with pytest.raises(SystemExit, match="exactly this skeleton"):
        script.verify_recipe(recipe, None, tmp_path, None, ["/venv/bin/python"], None, "2026-11-22", old, check)


def test_the_default_chip_pythons_are_only_venvs_that_hold_a_compiler(tmp_path, monkeypatch):
    script = _script()
    pair = tmp_path / "pair"
    monkeypatch.setattr(script.ttlang_pair, "default_venv", lambda: pair)
    monkeypatch.setattr(script.Path, "home", classmethod(lambda cls: tmp_path))
    assert script.default_chip_pythons() == []
    site = pair / "lib" / "python3.12" / "site-packages"
    (site / "ttl" / "_src").mkdir(parents=True)
    (site / "ttl" / "_src" / "ttl_ast.py").write_text("x\n")
    (site / "tt_lang-1.2.3.dist-info").mkdir()
    (site / "tt_lang-1.2.3.dist-info" / "METADATA").write_text("Version: 1.2.3\n")
    assert script.default_chip_pythons() == [pair / "bin" / "python"]


def test_a_new_simulator_run_of_an_edited_skeleton_drops_the_old_chip_entry(tmp_path, monkeypatch):
    """A chip entry belongs to the bytes that ran on the chip. When the skeleton is edited and verified again, it is not carried over."""
    script = _script()
    monkeypatch.setattr(script.ttlang_hwcompile, "package_identity", lambda python: dict(TARGET))
    recipe = _recipe()
    old = copy.deepcopy(op_cheats.load_verified()["records"][recipe["name"]])
    old["chip"] = [{"target": TARGET, "kernel_sha256": old["skeleton_sha256"], "date": "2026-12-01", "evidence": "docs/run-logs/x.md",
                    "pcc_min": 0.9999, "passed": True}]
    recipe["skeleton"].append("# edited")
    monkeypatch.setattr(script, "simulator_record", lambda recipe, overlay, work, sim, windows: {
        "skeleton_sha256": op_cheats.skeleton_sha256(recipe), "passed": True, "negatives": []})
    check, _ = _fake_compile({})
    new = script.verify_recipe(recipe, tmp_path, tmp_path, "sim", ["/venv/bin/python"], None, "2026-11-22", old, check)
    assert new["chip"] == [] and new["skeleton_sha256"] != old["skeleton_sha256"]
