# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The recipe verifier's static look at the chip form (scripts/verify_ttlang_recipes.py): `ttl.*` calls are checked against what an installed
package exposes, by namespace and by keyword, with `ast` and without importing anything. The simulator runs are not tested here; their
results are the records that tests/test_ttlang_recipes.py checks."""
from __future__ import annotations

import copy
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


@pytest.fixture
def chip_package(tmp_path):
    """A made-up chip package shaped like tt-lang 1.1.6: elementwise names in one file, `ttl.math` assembled from it and from operators,
    and no `block` namespace."""
    pkg = tmp_path / "site-packages" / "ttl"
    pkg.mkdir(parents=True)
    (pkg / "_generated_elementwise.py").write_text('__all__ = ["rsqrt", "sqrt"]\n\ndef rsqrt(input): ...\n\ndef sqrt(input): ...\n')
    (pkg / "operators.py").write_text("def broadcast(input, *, dims, shape): ...\n\ndef reduce_sum(input, *, dims): ...\n\n"
                                      "def fill(value, *, shape, dtype=None): ...\n")
    (pkg / "ttl_math.py").write_text("from ._generated_elementwise import *\nfrom .operators import broadcast, reduce_sum, fill\n")
    (pkg / "ttl.py").write_text("def operation(grid=None, options=None): ...\n\ndef compute(): ...\n\ndef datamovement(): ...\n\n"
                                "def make_dataflow_buffer_like(tensor, shape, block_count=2): ...\n\ndef copy(src, dst): ...\n")
    (pkg / "__init__.py").write_text("from ttl.ttl import (operation, compute, datamovement, make_dataflow_buffer_like, copy, math)\n"
                                     "from ttl._generated_elementwise import *\n")
    return pkg


def _recipe(name="row-reduce-spread"):
    return copy.deepcopy(next(r for r in op_cheats.load_recipes()["recipes"] if r["name"] == name))


def test_the_shipped_chip_forms_are_found_in_a_package_shaped_like_the_chip_package(chip_package):
    script = _script()
    namespaces = script.chip_namespaces(chip_package)
    for recipe in op_cheats.load_recipes()["recipes"]:
        result = script.static_hardware_check(recipe, namespaces)
        assert result["ok"] is True and result["problems"] == [], (recipe["name"], result)
        assert result["checked"], recipe["name"]                       # an empty list of calls would pass for the wrong reason


def test_the_namespaces_are_read_from_the_package_without_importing_it(chip_package):
    namespaces = _script().chip_namespaces(chip_package)
    assert namespaces["math"] == {"rsqrt", "sqrt", "broadcast", "reduce_sum", "fill"}
    assert {"operation", "copy", "math", "rsqrt"} <= namespaces["top"] and "block" not in namespaces["top"]
    assert namespaces["params"]["reduce_sum"] == {"input", "dims"}


@pytest.mark.parametrize("what,old,new,fragment", [
    ("the simulator's namespace", "ttl.math.broadcast(", "ttl.block.broadcast(", "namespace"),
    ("a name the package does not export", "ttl.math.rsqrt(", "ttl.math.reciprocal_sqrt(", "not exported"),
    ("a keyword the function does not have", "ttl.math.reduce_sum(x_blk * x_blk, dims=[-1])",
     "ttl.math.reduce_sum(x_blk * x_blk, dims=[-1], shape=(1, 1))", "keyword 'shape'"),
    ("a top-level name that does not exist", "ttl.copy(", "ttl.move(", "not exported"),
])
def test_a_chip_form_that_uses_what_the_package_lacks_is_a_named_problem(chip_package, what, old, new, fragment):
    script = _script()
    recipe = _recipe()
    recipe["hardware_replace"] = [[a, b.replace(old, new)] for a, b in recipe["hardware_replace"]]
    if not any(old in b for _, b in op_cheats.load_recipes()["recipes"][0]["hardware_replace"]):
        # the thing to break is in the unchanged part of the skeleton: edit the skeleton itself
        recipe["skeleton"] = [line.replace(old, new) for line in recipe["skeleton"]]
        recipe["hardware_replace"] = op_cheats.load_recipes()["recipes"][0]["hardware_replace"]
    result = script.static_hardware_check(recipe, script.chip_namespaces(chip_package))
    assert result["ok"] is False and any(fragment in p for p in result["problems"]), (what, result)


def test_a_chip_form_with_no_calls_is_not_ok(chip_package):
    script = _script()
    recipe = _recipe()
    recipe["skeleton"], recipe["hardware_replace"] = ["x = 1"], []
    assert script.static_hardware_check(recipe, script.chip_namespaces(chip_package))["ok"] is False


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
