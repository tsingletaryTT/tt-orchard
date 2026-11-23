# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Recipes by kind of op in the TT-Lang cheat sheet (orchard/op_cheats.py, orchard/skills/kernel-loop-templates/ttlang_recipes.json).

What these tests keep true:
- The default sheet is byte for byte what it was before recipes existed (a sha256 pin), so a run that does not ask for recipes
  has the same attempt directory as before.
- A recipe reaches the coder tagged `[passes the simulator]` only when a verification record for exactly that skeleton says the
  unedited check passed and every wrong variant failed. What a recipe says about the chip is in tests/test_recipes_dialect_gate.py.
- `--cheat-sheet-recipes on` travels from the command line to the attempt directory and into the report; off changes nothing.

Nothing here runs the simulator. The records are produced by scripts/verify_ttlang_recipes.py, which does.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from fake_model import FakeModel, final
from orchard import kernelprobe as kp
from orchard import op_cheats
# The fixtures of the kernel probe's own tests (stub tools, skill directory, freshness, consistency, a made-up tt-lang checkout).
from test_kernelprobe import (FACTS, consistency_ok, env_ready, fake_kenv, gozer_tripwire, no_run, probe_args,  # noqa: F401
                              run_attempt, skills_current, ttlang_skill)
from test_cli import _fake_probe_main
from orchard import cli

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "kernel-loop-templates"
# sha256 of the sheet `op_cheats.render` writes for the TT-Lang sheet with recipes off (133 lines). If this changes on purpose, a run
# with recipes off is no longer comparable with the runs recorded before it: say so in the log. It changed on 2026-11-22 (issue 801: the
# chip claims were corrected); the sheet before, 142 lines, had sha256 f425cee86e425cb61197ad6cfa3836e5651f720307f5bbb75e8adde573b0d96b,
# and the A/B of 2026-11-21 ran with it.
DEFAULT_SHEET_SHA256 = "bc9e27bc801f1a9d99728f0387d2b066acc1e42badcd24c554bffd77c9968c53"
FAMILIES = ["row-reduce-spread", "column-reduce-accumulate", "elementwise-with-operands", "running-total-down-rows"]
RECIPES_HEADING = "## Recipes by kind of op"


def _sheet():
    return op_cheats.load_sheet(op_cheats.TTLANG_SHEET)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _kernel_check_module():
    spec = importlib.util.spec_from_file_location("kernel_check_under_test", TEMPLATES / "kernel_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- the default output is unchanged -------------------------------------------------------------------------------


def test_the_default_sheet_is_byte_for_byte_the_one_before_recipes():
    assert _sha(op_cheats.render(_sheet())) == DEFAULT_SHEET_SHA256


def test_write_cheat_sheet_without_recipes_writes_the_default_bytes(tmp_path):
    assert kp.write_cheat_sheet(tmp_path) is True
    assert _sha((tmp_path / kp.CHEAT_NAME).read_text(encoding="utf-8")) == DEFAULT_SHEET_SHA256


# ---- the data file ----------------------------------------------------------------------------------------------------


def test_the_shipped_recipes_file_loads_and_names_the_four_families():
    data = op_cheats.load_recipes()
    assert [r["name"] for r in data["recipes"]] == FAMILIES
    assert op_cheats.recipes_shape_problems(data) == []


def _bad(mutate):
    data = copy.deepcopy(op_cheats.load_recipes())
    mutate(data)
    return op_cheats.recipes_shape_problems(data)


@pytest.mark.parametrize("what,mutate,fragment", [
    ("wrong schema", lambda d: d.update(schema="x"), "schema"),
    ("no intro", lambda d: d.update(intro=[]), "intro"),
    ("not a list of recipes", lambda d: d.update(recipes="x"), "recipes"),
    ("an empty list of recipes", lambda d: d.update(recipes=[]), "recipes"),
    ("a duplicate name", lambda d: d["recipes"][1].update(name=d["recipes"][0]["name"]), "twice"),
    ("a name with a space", lambda d: d["recipes"][0].update(name="a b"), "name"),
    ("no skeleton", lambda d: d["recipes"][0].update(skeleton=[]), "skeleton"),
    ("a skeleton line that is not text", lambda d: d["recipes"][0]["skeleton"].append(3), "skeleton"),
    ("a NUL in a skeleton line", lambda d: d["recipes"][0]["skeleton"].append("a\0b"), "NUL"),
    ("a very long skeleton line", lambda d: d["recipes"][0]["skeleton"].append("x" * 400), "long"),
    ("too many skeleton lines", lambda d: d["recipes"][0].update(skeleton=["x = 1"] * 200), "lines"),
    ("a plan that is not a list", lambda d: d["recipes"][0].update(plan="x"), "plan"),
    ("the old chip line pairs", lambda d: d["recipes"][0].update(hardware_replace=[["a", "b"]]), "hardware_replace"),
    ("no compile negative", lambda d: d["recipes"][0].update(compile_negatives=[]), "compile_negatives"),
    ("compile negatives that are not a list", lambda d: d["recipes"][0].update(compile_negatives={"a": 1}), "compile_negatives"),
    ("too many compile negatives", lambda d: d["recipes"][0].update(compile_negatives=d["recipes"][0]["compile_negatives"] * 7), "compile_negatives"),
    ("a compile negative with no name", lambda d: d["recipes"][0]["compile_negatives"][0].pop("name"), "compile negative"),
    ("two compile negatives with one name", lambda d: d["recipes"][0]["compile_negatives"].append(dict(d["recipes"][0]["compile_negatives"][0])), "name of its own"),
    ("a compile negative with no claim", lambda d: d["recipes"][0]["compile_negatives"][0].update(claim=""), "claim"),
    ("a compile negative with a very long claim", lambda d: d["recipes"][0]["compile_negatives"][0].update(claim="x" * 400), "claim"),
    ("a compile negative pair of the wrong size", lambda d: d["recipes"][0]["compile_negatives"][0].update(replace=[["only one"]]), "replace"),
    ("a negative with no replacement", lambda d: d["recipes"][0].update(negatives=[{"name": "n", "claim": "c"}]), "negative"),
    ("a spec that is not an object", lambda d: d["recipes"][0].update(spec=[]), "spec"),
    ("a spec for another op", lambda d: d["recipes"][0]["spec"].update(op="other"), "spec"),
])
def test_a_malformed_recipes_file_is_refused_with_a_reason_that_names_the_problem(what, mutate, fragment):
    problems = _bad(mutate)
    assert problems and any(fragment in p for p in problems), (what, problems)


def test_load_recipes_refuses_a_file_that_is_json_but_malformed(tmp_path):
    """The loader checks the shape itself; the renderer's own check would hide a loader that did not."""
    data = copy.deepcopy(op_cheats.load_recipes())
    data["schema"] = "orchard-recipes/0"
    path = tmp_path / "r.json"
    path.write_text(json.dumps(data))
    with pytest.raises(op_cheats.RecipeError, match="schema"):
        op_cheats.load_recipes(path)


def test_a_recipes_file_that_is_not_json_or_too_big_is_a_named_error(tmp_path):
    bad = tmp_path / "r.json"
    bad.write_text("{not json")
    with pytest.raises(op_cheats.RecipeError, match="r.json"):
        op_cheats.load_recipes(bad)
    bad.write_text(json.dumps({"x": "y" * (op_cheats.RECIPES_MAX_BYTES + 1)}))
    with pytest.raises(op_cheats.RecipeError, match="r.json"):
        op_cheats.load_recipes(bad)


# ---- every skeleton is a kernel the check accepts, and its hardware form is checkable ------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_each_skeleton_passes_the_checks_source_rules(tmp_path, name):
    """The unedited check reads kernel.py before it runs it (imports, forbidden names, one operation). A recipe that broke those
    rules would teach the coder a kernel the check refuses."""
    recipe = next(r for r in op_cheats.load_recipes()["recipes"] if r["name"] == name)
    kernel = tmp_path / "kernel.py"
    kernel.write_text("\n".join(recipe["skeleton"]) + "\n")
    ok, reason = _kernel_check_module().check_kernel_source(kernel)
    assert ok, reason
    assert _kernel_check_module().kernel_source_violations(kernel) == []


@pytest.mark.parametrize("name", FAMILIES)
def test_each_compile_negative_changes_lines_that_exist_and_is_still_python(name):
    recipe = next(r for r in op_cheats.load_recipes()["recipes"] if r["name"] == name)
    for neg in recipe["compile_negatives"]:
        wrong = op_cheats.apply_replacements(recipe, neg["replace"], "compile negative")
        compile(wrong, f"{name}-wrong", "exec")
        assert wrong != "\n".join(recipe["skeleton"])


def test_no_recipe_carries_hand_written_chip_lines():
    """Until 2026-11-22 each recipe said `for the chip, write ttl.math.fill`. Both chip compilers refuse that form, so no recipe may
    carry hand-written chip lines again: what the sheet says about the chip comes from compile records only."""
    for recipe in op_cheats.load_recipes()["recipes"]:
        assert "hardware_replace" not in recipe
        assert "ttl.math.fill" not in "\n".join(recipe["skeleton"]) and "ttl.math.broadcast" not in "\n".join(recipe["skeleton"])


def test_a_replacement_that_does_not_occur_once_is_an_error():
    recipe = copy.deepcopy(op_cheats.load_recipes()["recipes"][0])
    with pytest.raises(op_cheats.RecipeError, match="once"):
        op_cheats.apply_replacements(recipe, [["a line the skeleton does not have", "x"]], "a compile negative")
    with pytest.raises(op_cheats.RecipeError, match="once"):
        op_cheats.apply_replacements(recipe, [["ttl", "ttl2"]], "a compile negative")              # occurs many times


# ---- the verification records ------------------------------------------------------------------------------------------


def test_the_shipped_records_match_the_shipped_skeletons():
    assert op_cheats.verified_problems(op_cheats.load_recipes(), op_cheats.load_verified()) == []


def _records():
    return copy.deepcopy(op_cheats.load_verified())


@pytest.mark.parametrize("what,mutate,fragment", [
    ("a missing record", lambda v: v["records"].pop(FAMILIES[0]), FAMILIES[0]),
    ("an edited skeleton", lambda v: v["records"][FAMILIES[1]].update(skeleton_sha256="0" * 64), "skeleton"),
    ("a failed check", lambda v: v["records"][FAMILIES[2]].update(passed=False), "passed"),
    ("a pass on fewer than six input sets", lambda v: v["records"][FAMILIES[0]].update(sets=3), "sets"),
    ("a wrong variant that passed", lambda v: v["records"][FAMILIES[0]]["negatives"][0].update(failed=False), "wrong variant"),
    ("a wrong variant never run", lambda v: v["records"][FAMILIES[2]]["negatives"].pop(), "wrong variant"),
    ("compile entries that are not a list", lambda v: v["records"][FAMILIES[3]].update(compile="x"), "compile"),
    ("a compile entry that is not an object", lambda v: v["records"][FAMILIES[3]]["compile"].append("x"), "compile entry"),
    ("a compile entry with no record", lambda v: v["records"][FAMILIES[3]]["compile"][0].pop("record"), "compile record"),
    ("a compile record that says compiled with no kernel source", lambda v: v["records"][FAMILIES[3]]["compile"][0]["record"].update(kernel_sources=[]), "kernel sources"),
    ("a compile record that says compiled with no guard proof", lambda v: v["records"][FAMILIES[3]]["compile"][0]["record"].update(guard=[]), "shut out"),
    ("a compile record whose flag and category disagree", lambda v: v["records"][FAMILIES[3]]["compile"][0]["record"].update(category="compile_error"), "compiled"),
    ("a compile entry with no date", lambda v: v["records"][FAMILIES[3]]["compile"][0].pop("date"), "date"),
    ("compile negatives of the wrong type", lambda v: v["records"][FAMILIES[3]]["compile"][0].update(negatives=[{"name": "n", "failed": "yes"}]), "negatives"),
    ("a chip entry with no evidence", lambda v: v["records"][FAMILIES[3]].update(chip=[{"passed": True}]), "chip entry"),
    ("too many compile entries", lambda v: v["records"][FAMILIES[3]].update(compile=v["records"][FAMILIES[3]]["compile"] * 5), "at most"),
    ("a record for a recipe that is not there", lambda v: v["records"].update(ghost=v["records"][FAMILIES[0]]), "ghost"),
    ("another schema", lambda v: v.update(schema="x"), "schema"),
])
def test_a_stale_or_failed_record_is_a_reason_not_a_tag(what, mutate, fragment):
    verified = _records()
    mutate(verified)
    problems = op_cheats.verified_problems(op_cheats.load_recipes(), verified)
    assert problems and any(fragment in p for p in problems), (what, problems)


def test_recipes_with_a_stale_record_are_refused_rather_than_rendered_untagged():
    verified = _records()
    verified["records"][FAMILIES[0]]["skeleton_sha256"] = "0" * 64
    with pytest.raises(op_cheats.RecipeError, match="skeleton"):
        op_cheats.render_recipes(op_cheats.load_recipes(), verified)


# ---- the rendered section --------------------------------------------------------------------------------------------


def test_the_sheet_with_recipes_starts_with_the_default_sheet_and_then_the_recipes():
    default = op_cheats.render(_sheet())
    full = op_cheats.render_with_recipes(_sheet())
    assert full.startswith(default) and full[len(default):].lstrip("\n").startswith(RECIPES_HEADING)


def test_the_recipes_section_with_no_chip_package_tags_every_recipe_as_not_compiled():
    section = op_cheats.render_recipes(op_cheats.load_recipes(), op_cheats.load_verified())
    headings = [line for line in section.splitlines() if line.startswith("### ")]
    assert len(headings) == len(FAMILIES)
    assert all(h.endswith("[passes the simulator] [not compiled with a chip package]") for h in headings)
    assert "compiles with" not in section and "The chip compiler refuses" not in section.split("###", 1)[1]
    assert "the unedited check" in section
    for recipe in op_cheats.load_recipes()["recipes"]:
        assert recipe["title"] in section and recipe["skeleton"][0] in section


def test_the_recipes_headings_do_not_name_the_probes_target_ops():
    """The A/B measures whether a recipe for a kind of op helps on ops it was not written from. A heading that names those ops
    would hand the coder the answer's address."""
    data = op_cheats.load_recipes()
    heads = " ".join(r["title"] + " " + r["applies_to"] for r in data["recipes"]).lower()
    for held_out in ("center", "rmsnorm", "softmax", "row_sum", "linear_scan"):
        assert held_out not in heads, held_out


def test_the_recipes_section_stays_under_its_line_budget():
    section = op_cheats.render_recipes(op_cheats.load_recipes(), op_cheats.load_verified())
    assert section.count("\n") + 1 <= op_cheats.RECIPES_LINE_BUDGET


# ---- the switch reaches the attempt directory -------------------------------------------------------------------------


def test_write_cheat_sheet_with_recipes_adds_the_section_after_the_default_bytes(tmp_path):
    assert kp.write_cheat_sheet(tmp_path, recipes=True) is True
    text = (tmp_path / kp.CHEAT_NAME).read_text(encoding="utf-8")
    assert text.startswith(op_cheats.render(_sheet())) and RECIPES_HEADING in text


def test_an_attempt_without_the_switch_has_the_default_sheet_and_no_recipes_field(tmp_path, fake_kenv):
    with FakeModel(lambda request: final("done")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, run_watched=lambda argv, **kw: pytest.fail(f"unexpected {argv}"))
    adir = tmp_path / "probe" / "add" / "attempt-1"
    assert _sha((adir / kp.CHEAT_NAME).read_text(encoding="utf-8")) == DEFAULT_SHEET_SHA256
    assert "cheat_recipes" not in attempt["provenance"]


def test_an_attempt_with_the_switch_has_the_recipes_and_says_so_in_its_provenance(tmp_path, fake_kenv):
    with FakeModel(lambda request: final("done")) as server:
        attempt = run_attempt(tmp_path, fake_kenv, server, cheat_recipes=True,
                              run_watched=lambda argv, **kw: pytest.fail(f"unexpected {argv}"))
    adir = tmp_path / "probe" / "add" / "attempt-1"
    text = (adir / kp.CHEAT_NAME).read_text(encoding="utf-8")
    assert RECIPES_HEADING in text
    record = attempt["provenance"]["cheat_recipes"]
    assert record["on"] is True and record["recipes"] == FAMILIES
    assert record["data_sha256"] == hashlib.sha256((TEMPLATES / "ttlang_recipes.json").read_bytes()).hexdigest()


# ---- main: the report, the refusals, the command line -------------------------------------------------------------------------


def _run_main(tmp_path, fake_kenv, **kw):
    with FakeModel(lambda r: final("I will not write a kernel")) as server:
        code = kp.main(probe_args(tmp_path, attempts=2, endpoint=server.endpoint, **kw), env=fake_kenv, out=io.StringIO(),
                       probe=lambda e, m: True, run_watched_fn=no_run)
    return code, json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text())


def test_main_with_recipes_on_gives_every_attempt_the_recipes_and_records_it_in_the_report(tmp_path, fake_kenv, env_ready):
    code, data = _run_main(tmp_path, fake_kenv, cheat_sheet_recipes="on")
    assert code == 0
    assert data["header"]["cheat_recipes"]["on"] is True and data["header"]["cheat_recipes"]["recipes"] == FAMILIES
    assert all(a["provenance"]["cheat_recipes"]["on"] is True for a in data["attempts"])
    for n in (1, 2):
        assert RECIPES_HEADING in (tmp_path / "out" / "add" / f"attempt-{n}" / kp.CHEAT_NAME).read_text(encoding="utf-8")
    assert "Cheat sheet recipes: on" in (tmp_path / "out" / "kernel-probe-report.md").read_text()


@pytest.mark.parametrize("kw", [{}, {"cheat_sheet_recipes": "off"}], ids=["no attribute at all", "off"])
def test_main_with_recipes_off_or_unnamed_changes_nothing(tmp_path, fake_kenv, env_ready, kw):
    code, data = _run_main(tmp_path, fake_kenv, **kw)
    assert code == 0 and "cheat_recipes" not in data["header"]
    assert all("cheat_recipes" not in a["provenance"] for a in data["attempts"])
    for n in (1, 2):
        sheet = (tmp_path / "out" / "add" / f"attempt-{n}" / kp.CHEAT_NAME).read_text(encoding="utf-8")
        assert _sha(sheet) == DEFAULT_SHEET_SHA256
    assert "Cheat sheet recipes" not in (tmp_path / "out" / "kernel-probe-report.md").read_text()


def test_main_refuses_recipes_with_the_staged_probe_and_runs_nothing(tmp_path, fake_kenv, env_ready, capsys):
    code = kp.main(probe_args(tmp_path, op="center", staged=True, phase="sim", cheat_sheet_recipes="on", agent_hw=False),
                   env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 2 and not (tmp_path / "out").exists()
    err = capsys.readouterr().err
    assert "--cheat-sheet-recipes" in err and "--staged" in err


def test_main_refuses_recipes_whose_record_is_stale_before_any_attempt_exists(tmp_path, fake_kenv, env_ready, monkeypatch, capsys):
    stale = copy.deepcopy(op_cheats.load_verified())
    stale["records"][FAMILIES[0]]["skeleton_sha256"] = "0" * 64
    path = tmp_path / "verified.json"
    path.write_text(json.dumps(stale))
    monkeypatch.setattr(op_cheats, "RECIPES_VERIFIED", path)
    code = kp.main(probe_args(tmp_path, cheat_sheet_recipes="on"), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 2 and not (tmp_path / "out").exists()
    assert "skeleton" in capsys.readouterr().err


def test_main_refuses_a_switch_value_that_is_neither_on_nor_off(tmp_path, fake_kenv, env_ready, capsys):
    code = kp.main(probe_args(tmp_path, cheat_sheet_recipes="maybe"), env=fake_kenv, out=io.StringIO(),
                   probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 2 and not (tmp_path / "out").exists()
    assert "--cheat-sheet-recipes" in capsys.readouterr().err


def test_the_command_line_takes_the_switch_and_defaults_it_to_off(monkeypatch, tmp_path):
    calls = _fake_probe_main(monkeypatch)
    base = ["kernel-probe", "--op", "add", "--endpoint", "http://127.0.0.1:9/v1", "--model", "m"]
    cli.main(base, env={"HOME": str(tmp_path)}, stdout=io.StringIO())
    cli.main(base + ["--cheat-sheet-recipes", "on"], env={"HOME": str(tmp_path)}, stdout=io.StringIO())
    assert [c[0].cheat_sheet_recipes for c in calls] == ["off", "on"]
    with pytest.raises(SystemExit):
        cli.main(base + ["--cheat-sheet-recipes", "maybe"], env={"HOME": str(tmp_path)}, stdout=io.StringIO())


def test_an_attempt_that_asked_for_recipes_it_cannot_have_stops_and_writes_no_sheet(tmp_path, monkeypatch):
    """Running on without them would measure the other condition. The default still tolerates an unreadable sheet (existing behaviour)."""
    stale = copy.deepcopy(op_cheats.load_verified())
    stale["records"][FAMILIES[0]]["skeleton_sha256"] = "0" * 64
    path = tmp_path / "verified.json"
    path.write_text(json.dumps(stale))
    monkeypatch.setattr(op_cheats, "RECIPES_VERIFIED", path)
    with pytest.raises(kp.ProbeBlocked, match="skeleton"):
        kp.write_cheat_sheet(tmp_path / "a", recipes=True)
    assert not (tmp_path / "a" / kp.CHEAT_NAME).exists()
    adir = tmp_path / "b"
    adir.mkdir()
    assert kp.write_cheat_sheet(adir) is True                      # recipes off: the stale records are never read


def test_the_budget_test_can_fail(monkeypatch):
    """The line budget is a ceiling on the rendered section, so a section above it must be seen as above it."""
    section = op_cheats.render_recipes(op_cheats.load_recipes(), op_cheats.load_verified())
    assert section.count("\n") + 1 > 50                           # the section is long enough that a budget of 50 would fail
