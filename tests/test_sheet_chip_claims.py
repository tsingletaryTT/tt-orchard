# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The plain TT-Lang cheat sheet no longer teaches a chip form that the chip compilers refuse (issue 801).

Before: the sheet called `ttl.math.fill` and `ttl.math.broadcast` the hardware names, called `ttl.block.fill` simulator only, and said the
hardware package has no `shape` argument on `reduce_sum`. The chip compiler the kernel probe uses (the pair, tt-lang 1.1.10.dev20261026+light)
takes `ttl.block.*` and `shape=`; both chip compilers refuse `ttl.math.fill` and `ttl.math.broadcast` inside a kernel. These tests pin the
sheet's text to those facts. The tests with the real compilers use the chip packages of this machine and skip elsewhere; they run the
device-less compile check (orchard/ttlang_hwcompile.py) and open no device.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from orchard import op_cheats
from orchard import ttlang_hwcompile as hc
from test_ttlang_hwcompile import PAIR, SHARED, on_116, on_pair

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "kernel-loop-templates"
OLD_SHEET_LINES = 141                       # the plain sheet before this change (PR 796's default, pinned at 142 lines with a trailing newline)
MAX_LINES = round(OLD_SHEET_LINES * 1.05)   # the operator's "no worse than before": at most about 5% longer


def sheet() -> dict:
    return op_cheats.load_sheet(op_cheats.TTLANG_SHEET)


def rendered() -> str:
    return op_cheats.render(sheet())


def entry(name: str) -> dict:
    found = op_cheats.find_entry(sheet(), name)
    assert found is not None, f"the sheet has no entry {name}"
    return found


# ---- the text ----------------------------------------------------------------------------------------------------------------------

def test_no_line_offers_math_fill_or_math_broadcast_as_a_working_form():
    """Every line that names the refused forms says they are refused. A usage line or a signature with them is the bug."""
    for line in rendered().splitlines():
        if re.search(r"ttl\.math\.(fill|broadcast)\b", line):
            assert "not available" in line or "refuse" in line, line
    for e in op_cheats.entries(sheet()):
        assert not re.search(r"ttl\.math\.(fill|broadcast)\b", e["usage"] + e["signature"]), e["name"]


def test_fill_and_broadcast_are_taught_in_the_block_namespace_with_a_compile_status():
    for name in ("ttl.block.fill", "ttl.block.broadcast"):
        e = entry(name)
        assert e["status"] == "compiles on hardware" and e.get("evidence"), name
        assert name in e["usage"], name


@pytest.mark.parametrize("name", ["ttl.math.reduce_sum", "ttl.math.reduce_max"])
def test_the_reductions_pass_shape_because_the_simulator_requires_it_and_the_pair_takes_it(name):
    e = entry(name)
    assert "shape" in e["signature"] and "shape=" in e["usage"], e
    assert "simulator requires" in e["notes"] and "1.1.6" in e["notes"], "the entry says who requires shape and who refuses it"
    assert e["status"] == "compiles on hardware" and e.get("evidence")


def test_nothing_says_the_hardware_package_lacks_shape_or_the_block_namespace():
    text = rendered()
    assert "no shape argument in the hardware package" not in text
    assert "is not in the hardware package" not in text
    assert "the hardware names live under ttl.math" not in text
    assert "Write the hardware form above" not in text


def test_the_refusal_messages_are_in_the_sheet_as_a_mistake():
    text = " ".join(m["symptom"] + " " + m["fix"] for m in sheet()["mistakes"])
    for message in ("ttl.math.fill is not available; use ttl.block.fill", "ttl.math.broadcast is not available; use ttl.block.broadcast",
                    "reduce_sum() got an unexpected keyword argument 'shape'"):
        assert message in text, message
    shape_fix = next(m for m in sheet()["mistakes"] if "unexpected keyword argument 'shape'" in m["symptom"])["fix"]
    assert "tt-lang 1.1.6" in shape_fix and "simulator" in shape_fix, "the 1.1.6 fallback is labelled, with the simulator's side of it"


def test_the_roots_name_the_pair_as_the_hardware_package_and_1_1_6_as_the_fallback():
    roots = sheet()["roots"]
    assert "pair" in roots["hardware"] and "1.1.10" in roots["hardware"]
    assert "1.1.6" in roots["shared"] and "fallback" in roots["shared"]


def test_the_sheet_is_at_most_five_percent_longer_than_the_old_one():
    assert len(rendered().splitlines()) <= MAX_LINES, (len(rendered().splitlines()), MAX_LINES)


def test_what_the_simulator_judges_is_unchanged():
    """The probe's judge is the simulator: every example it relies on keeps its form (shape= on the reductions, ttl.block for fill and
    broadcast, make_dataflow_buffer_like with block_count)."""
    text = rendered()
    assert "reduce_sum(x_blk * x_blk, dims=[-1], shape=(1, 1))" in text
    assert "ttl.block.fill(0.5, shape=(1, 1))" in text
    assert "ttl.block.broadcast(inv, dims=[-1], shape=(1, col_tiles))" in text
    assert "make_dataflow_buffer_like" in text and "block_count" in text


# ---- the facts behind the text, with the real chip compilers -----------------------------------------------------------------------

SKELETON = "\n".join(json.loads((TEMPLATES / "ttlang_recipes.json").read_text())["recipes"][0]["skeleton"]) + "\n"
SHAPES = [(64, 256), (64, 256)]
NO_SHAPE = SKELETON.replace("dims=[-1], shape=(1, 1))", "dims=[-1])")


def compile_text(python, tmp_path, text):
    path = tmp_path / "kernel.py"
    path.write_text(text, encoding="utf-8")
    return hc.compile_check(python, path, SHAPES)


def test_the_skeleton_has_the_forms_the_variants_below_replace():
    """The variants are built by text replacement; a skeleton that changed would turn every variant into the same kernel."""
    assert SKELETON.count("ttl.block.fill") == 2 and SKELETON.count("ttl.block.broadcast") == 1 and "shape=(1, 1))" in SKELETON
    assert NO_SHAPE != SKELETON and "reduce_sum(x_blk * x_blk, dims=[-1])" in NO_SHAPE


@pytest.mark.parametrize("python", [pytest.param(SHARED, marks=on_116, id="1.1.6"), pytest.param(PAIR, marks=on_pair, id="pair")])
def test_both_compilers_take_the_block_forms_and_refuse_the_math_forms(python, tmp_path):
    good = compile_text(python, tmp_path, NO_SHAPE)
    assert good["category"] == "compiled", good["first_line"]
    for old, new, message in (("ttl.block.fill", "ttl.math.fill", "ttl.math.fill is not available; use ttl.block.fill"),
                              ("ttl.block.broadcast", "ttl.math.broadcast", "ttl.math.broadcast is not available; use ttl.block.broadcast")):
        bad = compile_text(python, tmp_path, NO_SHAPE.replace(old, new))
        assert bad["category"] == "compile_error" and message in bad["first_line"], (new, bad["first_line"])


@on_pair
def test_the_pair_takes_shape_on_the_reduction():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        assert compile_text(PAIR, Path(tmp), SKELETON)["category"] == "compiled"


@on_116
def test_1_1_6_refuses_shape_on_the_reduction(tmp_path):
    bad = compile_text(SHARED, tmp_path, SKELETON)
    assert bad["category"] == "compile_error" and "reduce_sum() got an unexpected keyword argument 'shape'" in bad["first_line"]


@on_pair
def test_make_dfb_compiles_with_the_pair(tmp_path):
    """The entry for ttl.make_dfb used to say the hardware package lacks it: 1.1.6 does, the pair does not."""
    text = SKELETON.replace("ttl.make_dataflow_buffer_like(x, shape=(1, col_tiles), block_count=2)", "ttl.make_dfb('bf16', shape=(1, col_tiles), block_count=2)")
    assert text != SKELETON
    assert compile_text(PAIR, tmp_path, text)["category"] == "compiled"


# ---- the sheet against the chip packages' source trees (the verifier of issue 523) ------------------------------------------------

CHECKOUT = Path.home() / "orchard-src" / "tt-lang" / "python"


def _site_ttl(python: Path) -> Path | None:
    try:
        return hc.package_dir(python)
    except hc.HwCompileError:
        return None


def test_the_sheet_verifies_against_the_pair_and_1_1_6_trees():
    pair, shared = _site_ttl(PAIR), _site_ttl(SHARED)
    if not (pair and shared and (CHECKOUT / "ttl").is_dir() and (CHECKOUT / "sim").is_dir()):
        pytest.skip("the chip packages or the tt-lang checkout are not on this machine")
    roots = {"hardware": pair, "shared": shared, "tree": CHECKOUT / "ttl", "sim": CHECKOUT / "sim"}
    indexes = {name: op_cheats.build_index(path) for name, path in roots.items()}
    assert op_cheats.verify(sheet(), indexes) == []
