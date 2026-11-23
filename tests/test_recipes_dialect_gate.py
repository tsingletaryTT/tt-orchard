# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What the cheat sheet's recipes say about the chip, and the gate that may turn them on.

The rules:
- A recipe's chip tag comes from a compile record for exactly the run's chip package (version and compiler bytes) and exactly the
  skeleton the sheet prints. Any other record gives `not compiled`. No tag says a recipe ran on a chip without a chip entry.
- `--cheat-sheet-recipes auto` is on only when every recipe compiled with that package's compiler and the compiler refused every wrong
  form. Otherwise it is off and the report says why.
- `--cheat-sheet-recipes control` gives a section of the same form and length with no recipe content.

The records were made by scripts/verify_ttlang_recipes.py with the real compilers and no device. The last tests compile the shipped
skeletons again when this machine has those compilers, and skip elsewhere."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from fake_model import FakeModel, final
from orchard import kernelprobe as kp
from orchard import op_cheats
from orchard import ttlang_hwcompile as hc
from test_kernelprobe import (FACTS, consistency_ok, env_ready, fake_kenv, gozer_tripwire, no_run, probe_args,  # noqa: F401
                              run_attempt, skills_current, ttlang_skill)
from test_cli import _fake_probe_main
from orchard import cli
from test_ttlang_hwcompile import PAIR, SHARED, landlock_available

DEFAULT_SHEET_SHA256 = "f425cee86e425cb61197ad6cfa3836e5651f720307f5bbb75e8adde573b0d96b"
FAMILIES = ["row-reduce-spread", "column-reduce-accumulate", "elementwise-with-operands", "running-total-down-rows"]
REDUCTIONS = FAMILIES[:2]
OTHER = {"package": "tt-lang", "version": "5.5.5", "compiler_sha256": "5" * 64}


def _data():
    return op_cheats.load_recipes()


def _verified():
    return copy.deepcopy(op_cheats.load_verified())


def _target(prefix: str) -> dict:
    """The chip package in the shipped records whose version starts with `prefix` (`1.1.10` is the pair, `1.1.6` the old shared venv)."""
    found = [t for t in op_cheats.load_verified()["targets"] if t["version"].startswith(prefix)]
    assert len(found) == 1, f"the shipped records have {len(found)} chip packages for {prefix}"
    return found[0]


def _headings(section: str) -> list[str]:
    return [line for line in section.splitlines() if line.startswith("### ")]


# ---- what the shipped records show ------------------------------------------------------------------------------------------------

def test_the_shipped_records_cover_the_pair_and_the_old_package_with_well_formed_entries():
    verified = op_cheats.load_verified()
    assert op_cheats.verified_problems(_data(), verified) == []
    assert sorted(t["version"][:6] for t in verified["targets"]) == ["1.1.10", "1.1.6"]
    for name in FAMILIES:
        entries = verified["records"][name]["compile"]
        assert len(entries) == 2 and all(hc.record_problems(e["record"]) == [] for e in entries)
        assert verified["records"][name]["chip"] == []                 # no recipe has run on a chip


def test_every_recipe_compiles_with_the_pair_and_the_reductions_do_not_with_116():
    data, verified = _data(), op_cheats.load_verified()
    levels = {t: {r["name"]: op_cheats.compile_status(r, verified["records"][r["name"]], _target(t))["level"] for r in data["recipes"]}
              for t in ("1.1.10", "1.1.6")}
    assert levels["1.1.10"] == {name: "compile" for name in FAMILIES}
    assert levels["1.1.6"] == {name: ("refused" if name in REDUCTIONS else "compile") for name in FAMILIES}
    for recipe in data["recipes"]:
        if recipe["name"] in REDUCTIONS:
            status = op_cheats.compile_status(recipe, verified["records"][recipe["name"]], _target("1.1.6"))
            assert "unexpected keyword argument 'shape'" in status["first_line"]


def test_both_compilers_refused_fill_under_ttl_math_wherever_the_skeleton_compiled():
    """The claim the sheet had the wrong way round until 2026-11-22: it told the coder to write `ttl.math.fill` for the chip."""
    data, verified = _data(), op_cheats.load_verified()
    seen = 0
    for prefix in ("1.1.10", "1.1.6"):
        for recipe in data["recipes"]:
            status = op_cheats.compile_status(recipe, verified["records"][recipe["name"]], _target(prefix))
            if status["level"] != "compile":
                continue
            lines = [c["first_line"] for c in status["claims"] if "ttl.math.fill" in c["claim"]]
            assert lines and "ttl.math.fill is not available; use ttl.block.fill" in lines[0], (prefix, recipe["name"])
            assert status["negatives_open"] == []
            seen += 1
    assert seen == 6                                                   # four recipes on the pair, two on 1.1.6


def test_the_pair_refused_a_reduction_used_directly_and_the_recipe_says_how_to_write_it():
    recipe = _data()["recipes"][0]
    status = op_cheats.compile_status(recipe, op_cheats.load_verified()["records"][recipe["name"]], _target("1.1.10"))
    direct = [c for c in status["claims"] if "used directly" in c["claim"]]
    assert direct and "cannot lower tensor store" in direct[0]["first_line"]
    skeleton = "\n".join(recipe["skeleton"])
    assert "rsqrt(total * one + eps)" in skeleton and "rsqrt(total + eps)" not in skeleton
    assert any("Multiply the reduction's result" in step for step in recipe["plan"])


def test_no_record_or_data_file_holds_a_machine_path():
    for path in (op_cheats.RECIPES, op_cheats.RECIPES_VERIFIED, op_cheats.RECIPES_CONTROL, op_cheats.RECIPES_CONTROL_VERIFIED):
        text = path.read_text(encoding="utf-8")                        # hardening: ok our own repository file
        assert "/home/" not in text and "/tmp/" not in text and "/mnt/" not in text, path.name


# ---- compile_status: which record counts -------------------------------------------------------------------------------------------

def _status(mutate=None, target=None, name=FAMILIES[0]):
    verified = _verified()
    recipe = next(r for r in _data()["recipes"] if r["name"] == name)
    rec = verified["records"][name]
    if mutate:
        mutate(rec, recipe)
    return op_cheats.compile_status(recipe, rec, target if target is not None else _target("1.1.10"))


def test_a_record_counts_only_for_the_same_package_and_the_same_skeleton():
    assert _status()["level"] == "compile"
    pair = _target("1.1.10")
    assert _status(target=OTHER)["level"] == "none"
    assert _status(target={**pair, "compiler_sha256": "0" * 64})["level"] == "none"          # same version, other compiler bytes
    assert _status(target={**pair, "version": pair["version"] + ".1"})["level"] == "none"

    def other_bytes(rec, recipe):
        for entry in rec["compile"]:
            entry["record"]["kernel_sha256"] = "0" * 64
    assert _status(other_bytes)["level"] == "none"

    def edited_skeleton(rec, recipe):
        recipe["skeleton"].append("# edited after it was compiled")
    assert _status(edited_skeleton)["level"] == "none"


@pytest.mark.parametrize("target", [None, {}, "tt-lang 1.1.6", {"package": "tt-lang", "version": 3, "compiler_sha256": "0" * 64}])
def test_no_or_a_malformed_chip_package_gives_no_result_and_no_label(target):
    recipe = _data()["recipes"][0]
    status = op_cheats.compile_status(recipe, op_cheats.load_verified()["records"][recipe["name"]], target)
    assert status["level"] == "none" and status["label"] is None and status["claims"] == []


@pytest.mark.parametrize("category", ["timeout", "guard_failed", "bad_result", "not_run", "no_kernel_written"])
def test_a_check_that_did_not_finish_is_no_result_and_never_a_refusal(category):
    def unfinished(rec, recipe):
        record = rec["compile"][0]["record"]
        record.update(category=category, compiled=False, kernel_sources=[], first_line="no verdict")
    status = _status(unfinished)
    assert status["level"] == "none" and status["first_line"] == ""


def test_a_chip_entry_counts_only_on_top_of_a_compile_for_the_same_package_and_skeleton():
    pair = _target("1.1.10")

    def chip(target, sha=None):
        def mutate(rec, recipe):
            rec["chip"] = [{"target": target, "kernel_sha256": sha or op_cheats.skeleton_sha256(recipe), "date": "2026-12-01",
                            "evidence": "docs/run-logs/2026-12-01-recipes-chip.md", "pcc_min": 0.99991, "passed": True}]
        return mutate
    assert _status(chip(pair))["level"] == "chip"
    assert _status(chip(OTHER))["level"] == "compile"
    assert _status(chip(pair, "0" * 64))["level"] == "compile"
    old = _target("1.1.6")
    assert _status(chip(old), target=old)["level"] == "refused"                               # a chip entry does not lift a refusal


# ---- the rendered section ----------------------------------------------------------------------------------------------------------

def test_the_section_for_the_pair_tags_every_recipe_as_compiled_with_no_chip():
    pair = _target("1.1.10")
    section = op_cheats.render_recipes(_data(), op_cheats.load_verified(), pair)
    label = f"tt-lang {pair['version']}"
    assert all(h.endswith(f"[passes the simulator] [compiles with {label}, no chip]") for h in _headings(section)) and len(_headings(section)) == 4
    assert section.count("Not run on a chip.") == 4
    assert section.count("The chip compiler refuses `ttl.math.fill`: write `ttl.block.fill`.") == 4
    assert "ran on a chip" not in section.lower().replace("not run on a chip", "")
    for gone in ("For the chip, change these lines", "chip:      ", "has no `ttl.block` namespace", "0 of 7"):
        assert gone not in section, gone


def test_the_section_for_116_says_which_recipes_its_compiler_refused_and_why():
    section = op_cheats.render_recipes(_data(), op_cheats.load_verified(), _target("1.1.6"))
    tags = [h.rsplit("] [", 1)[1].rstrip("]") for h in _headings(section)]
    assert tags == ["refused by the compiler of tt-lang 1.1.6"] * 2 + ["compiles with tt-lang 1.1.6, no chip"] * 2
    assert section.count("refused the skeleton with `") == 2 and section.count("unexpected keyword argument 'shape'") == 2
    assert section.count("The chip compiler refuses") - 1 == 2          # one mention in the intro, then one claim per compiled recipe


def test_the_section_for_an_unknown_package_claims_nothing_about_it():
    section = op_cheats.render_recipes(_data(), op_cheats.load_verified(), OTHER)
    assert all(h.endswith("[passes the simulator] [not compiled with tt-lang 5.5.5]") for h in _headings(section))
    assert "accepted the skeleton and wrote" not in section and "The chip compiler refuses `" not in section


def test_a_chip_entry_changes_the_tag_and_names_its_evidence():
    verified = _verified()
    pair = _target("1.1.10")
    recipe = _data()["recipes"][2]
    verified["records"][recipe["name"]]["chip"] = [{"target": pair, "kernel_sha256": op_cheats.skeleton_sha256(recipe), "date": "2026-12-01",
                                                    "evidence": "docs/run-logs/2026-12-01-recipes-chip.md", "pcc_min": 0.99991, "passed": True}]
    section = op_cheats.render_recipes(_data(), verified, pair)
    tags = [h.rsplit("] [", 1)[1].rstrip("]") for h in _headings(section)]
    assert tags.count(f"ran on a chip with tt-lang {pair['version']}") == 1 and tags[2].startswith("ran on a chip")
    assert "Ran on a chip on 2026-12-01: worst PCC 0.9999 (docs/run-logs/2026-12-01-recipes-chip.md)." in section


@pytest.mark.parametrize("prefix", ["1.1.10", "1.1.6", None])
@pytest.mark.parametrize("files", ["recipes", "control"])
def test_each_section_stays_under_the_line_budget_for_every_chip_package(prefix, files):
    data_path, verified_path = kp._section_files("on" if files == "recipes" else "control")
    section = op_cheats.render_recipes(op_cheats.load_recipes(data_path), op_cheats.load_verified(verified_path),
                                       _target(prefix) if prefix else None)
    assert section.count("\n") + 1 <= op_cheats.RECIPES_LINE_BUDGET


# ---- the gate ----------------------------------------------------------------------------------------------------------------------

def test_the_gate_holds_for_the_pair_and_not_for_116():
    data, verified = _data(), op_cheats.load_verified()
    assert op_cheats.recipes_gate(data, verified, _target("1.1.10")) == []
    reasons = op_cheats.recipes_gate(data, verified, _target("1.1.6"))
    assert len(reasons) == 2 and all(name in " ".join(reasons) for name in REDUCTIONS) and all("refused the skeleton" in r for r in reasons)


def test_the_gate_does_not_hold_without_a_chip_package_or_for_one_with_no_records():
    data, verified = _data(), op_cheats.load_verified()
    assert op_cheats.recipes_gate(data, verified, None) == ["no chip package was found for this run, so no recipe is known to compile for it"]
    reasons = op_cheats.recipes_gate(data, verified, OTHER)
    assert len(reasons) == 4 and all("no compile result for tt-lang 5.5.5" in r for r in reasons)


@pytest.mark.parametrize("what,mutate,fragment", [
    ("one recipe compiled for other bytes", lambda v: v["records"][FAMILIES[3]]["compile"][0]["record"].update(kernel_sha256="0" * 64), "no compile result"),
    ("one recipe has no compile entry", lambda v: v["records"][FAMILIES[2]].update(compile=[]), "no compile result"),
    ("a wrong form was not refused", lambda v: v["records"][FAMILIES[1]]["compile"][0]["negatives"][0].update(failed=False, category="compiled"), "did not refuse"),
    ("a wrong form only timed out", lambda v: v["records"][FAMILIES[1]]["compile"][0]["negatives"][0].update(category="timeout"), "did not refuse"),
    ("a wrong form was never compiled", lambda v: v["records"][FAMILIES[0]]["compile"][0]["negatives"].pop(), "did not refuse"),
    ("the simulator record is stale", lambda v: v["records"][FAMILIES[0]].update(skeleton_sha256="0" * 64), "skeleton"),
    ("the simulator check failed", lambda v: v["records"][FAMILIES[0]].update(passed=False), "passed"),
    ("a compile entry is malformed", lambda v: v["records"][FAMILIES[0]]["compile"][0].update(record="x"), "compile record"),
])
def test_one_recipe_short_of_the_bar_closes_the_gate_with_a_reason(what, mutate, fragment):
    verified = _verified()
    assert verified["records"][FAMILIES[0]]["compile"][0]["record"]["target"]["version"].startswith("1.1.10")
    mutate(verified)
    reasons = op_cheats.recipes_gate(_data(), verified, _target("1.1.10"))
    assert reasons and any(fragment in r for r in reasons), (what, reasons)


# ---- the control section -----------------------------------------------------------------------------------------------------------

def _control():
    return op_cheats.load_recipes(op_cheats.RECIPES_CONTROL), op_cheats.load_verified(op_cheats.RECIPES_CONTROL_VERIFIED)


def test_the_control_section_is_verified_like_the_recipes_and_its_gate_holds_for_both_packages():
    data, verified = _control()
    assert op_cheats.verified_problems(data, verified) == []
    for prefix in ("1.1.10", "1.1.6"):
        target = next(t for t in verified["targets"] if t["version"].startswith(prefix))
        assert op_cheats.recipes_gate(data, verified, target) == []


def test_the_control_section_has_none_of_the_recipes_content():
    """The control arm measures what more sheet text of the same form does. It may not teach what the recipes teach."""
    data, _ = _control()
    for recipe in data["recipes"]:
        text = "\n".join(recipe["skeleton"] + recipe["plan"] + [recipe["title"], recipe["applies_to"], recipe["worked_op"], recipe["units"]])
        for taught in ("reduce_sum", "reduce_max", "broadcast", "ttl.block", "fill(", "+=", "rsqrt", "total"):
            assert taught not in text, (recipe["name"], taught)
        words = set(__import__("re").findall(r"[a-z_]+", text.lower()))
        for held_out in ("center", "rmsnorm", "softmax", "row_sum", "linear_scan", "norm", "mean", "normalise", "average"):
            assert held_out not in words, (recipe["name"], held_out)


def test_the_control_section_has_the_recipes_framing_and_length():
    data, verified = _control()
    recipes, recipes_verified = _data(), op_cheats.load_verified()
    assert data["title"] == recipes["title"] and data["intro"] == recipes["intro"] and len(data["recipes"]) == len(recipes["recipes"])
    pair = _target("1.1.10")
    control = op_cheats.render_recipes(data, verified, pair)
    treatment = op_cheats.render_recipes(recipes, recipes_verified, pair)
    lines = (control.count("\n"), treatment.count("\n"))
    assert abs(lines[0] - lines[1]) <= 0.03 * lines[1], lines
    assert abs(len(control) - len(treatment)) <= 0.03 * len(treatment), (len(control), len(treatment))
    assert [h.split(" [", 1)[1] for h in _headings(control)] == [h.split(" [", 1)[1] for h in _headings(treatment)]       # the same tags


# ---- the probe: modes, the chip package, the report --------------------------------------------------------------------------------

def test_auto_is_on_only_when_the_gate_holds_for_the_runs_chip_package():
    assert kp.resolve_recipes_mode("auto", _target("1.1.10")) == ("on", [])
    mode, reasons = kp.resolve_recipes_mode("auto", _target("1.1.6"))
    assert mode == "off" and len(reasons) == 2
    assert kp.resolve_recipes_mode("auto", None)[0] == "off"
    assert kp.resolve_recipes_mode("auto", OTHER)[0] == "off"
    for mode in ("on", "off", "control"):
        assert kp.resolve_recipes_mode(mode, None) == (mode, [])         # the explicit modes do not consult the gate


def test_auto_is_off_when_the_records_cannot_be_read(monkeypatch, tmp_path):
    pair = _target("1.1.10")
    monkeypatch.setattr(op_cheats, "RECIPES_VERIFIED", tmp_path / "absent.json")
    mode, reasons = kp.resolve_recipes_mode("auto", pair)
    assert mode == "off" and reasons


def test_the_chip_package_of_the_run_is_read_from_files_or_reported_as_missing(tmp_path):
    target, problem = kp.recipes_target(tmp_path / "no-venv" / "bin" / "python")
    assert target is None and "no ttl package" in problem


def test_write_cheat_sheet_renders_the_tags_for_the_package_it_is_given(tmp_path):
    for name, target, tag in (("pair", _target("1.1.10"), "no chip]"), ("old", _target("1.1.6"), "[refused by the compiler of tt-lang 1.1.6]"),
                              ("none", None, "[not compiled with a chip package]")):
        adir = tmp_path / name
        adir.mkdir()
        assert kp.write_cheat_sheet(adir, recipes="on", target=target) is True
        text = (adir / kp.CHEAT_NAME).read_text(encoding="utf-8")
        assert tag in text and text.startswith(op_cheats.render(op_cheats.load_sheet(op_cheats.TTLANG_SHEET)))
    adir = tmp_path / "control"
    adir.mkdir()
    assert kp.write_cheat_sheet(adir, recipes="control", target=_target("1.1.10")) is True
    text = (adir / kp.CHEAT_NAME).read_text(encoding="utf-8")
    assert "Add two tensors, one row of tiles per block" in text and "Reduce each row" not in text
    with pytest.raises(ValueError):
        kp.write_cheat_sheet(adir, recipes="maybe")


def _run_main(tmp_path, fake_kenv, monkeypatch, target, **kw):
    monkeypatch.setattr(kp, "recipes_target", lambda python: (target, None if target else "no ttl package beneath the venv"))
    out = io.StringIO()
    with FakeModel(lambda r: final("I will not write a kernel")) as server:
        code = kp.main(probe_args(tmp_path, attempts=1, endpoint=server.endpoint, **kw), env=fake_kenv, out=out,
                       probe=lambda e, m: True, run_watched_fn=no_run)
    return code, json.loads((tmp_path / "out" / "kernel-probe-report.json").read_text()), out.getvalue()


def _sheet_of(tmp_path) -> str:
    return (tmp_path / "out" / "add" / "attempt-1" / kp.CHEAT_NAME).read_text(encoding="utf-8")


def test_main_with_auto_and_the_pair_turns_the_recipes_on_and_records_package_and_levels(tmp_path, fake_kenv, env_ready, monkeypatch):
    pair = _target("1.1.10")
    code, data, said = _run_main(tmp_path, fake_kenv, monkeypatch, pair, cheat_sheet_recipes="auto")
    assert code == 0 and "--cheat-sheet-recipes auto: on" in said
    record = data["header"]["cheat_recipes"]
    assert record["on"] is True and record["mode"] == "auto" and record["section"] == "on"
    assert record["target"] == {"version": pair["version"], "compiler_sha256": pair["compiler_sha256"]}
    assert record["levels"] == {name: "compile" for name in FAMILIES}
    assert data["attempts"][0]["provenance"]["cheat_recipes"]["levels"] == record["levels"]
    assert f"[compiles with tt-lang {pair['version']}, no chip]" in _sheet_of(tmp_path)
    report = (tmp_path / "out" / "kernel-probe-report.md").read_text()
    assert "Cheat sheet recipes: on (asked for: auto)" in report and "No level is a chip run" in report


@pytest.mark.parametrize("prefix", ["1.1.6", None])
def test_main_with_auto_and_a_package_short_of_the_gate_runs_with_the_default_sheet_and_says_why(tmp_path, fake_kenv, env_ready, monkeypatch, prefix):
    code, data, said = _run_main(tmp_path, fake_kenv, monkeypatch, _target(prefix) if prefix else None, cheat_sheet_recipes="auto")
    assert code == 0 and "--cheat-sheet-recipes auto: off (" in said
    record = data["header"]["cheat_recipes"]
    assert record["on"] is False and record["mode"] == "auto" and record["gate"]
    assert "cheat_recipes" not in data["attempts"][0]["provenance"]
    assert hashlib.sha256(_sheet_of(tmp_path).encode()).hexdigest() == DEFAULT_SHEET_SHA256
    assert "Cheat sheet recipes: off (asked for: auto). The gate did not hold" in (tmp_path / "out" / "kernel-probe-report.md").read_text()


def test_main_with_on_shows_the_recipes_for_116_with_the_refusals_tagged(tmp_path, fake_kenv, env_ready, monkeypatch):
    code, data, _ = _run_main(tmp_path, fake_kenv, monkeypatch, _target("1.1.6"), cheat_sheet_recipes="on")
    assert code == 0
    assert data["header"]["cheat_recipes"]["levels"] == {n: ("refused" if n in REDUCTIONS else "compile") for n in FAMILIES}
    assert _sheet_of(tmp_path).count("[refused by the compiler of tt-lang 1.1.6]") == 2


def test_main_with_on_and_no_chip_package_says_so_and_tags_nothing_as_compiled(tmp_path, fake_kenv, env_ready, monkeypatch):
    code, data, said = _run_main(tmp_path, fake_kenv, monkeypatch, None, cheat_sheet_recipes="on")
    assert code == 0 and "no chip package found" in said
    assert data["header"]["cheat_recipes"]["target"] is None
    assert set(data["header"]["cheat_recipes"]["levels"].values()) == {"none"}
    assert _sheet_of(tmp_path).count("[not compiled with a chip package]") == 4


def test_main_with_control_gives_the_control_section_and_records_it(tmp_path, fake_kenv, env_ready, monkeypatch):
    code, data, _ = _run_main(tmp_path, fake_kenv, monkeypatch, _target("1.1.10"), cheat_sheet_recipes="control")
    assert code == 0
    record = data["header"]["cheat_recipes"]
    assert record["section"] == "control" and record["mode"] == "control" and record["recipes"][0] == "ctl-add-rows"
    assert record["data_sha256"] == hashlib.sha256(op_cheats.RECIPES_CONTROL.read_bytes()).hexdigest()       # hardening: ok our own file
    sheet = _sheet_of(tmp_path)
    assert "Add two tensors, one row of tiles per block" in sheet and "Reduce each row" not in sheet
    assert "Cheat sheet recipes: control (asked for: control)" in (tmp_path / "out" / "kernel-probe-report.md").read_text()


@pytest.mark.parametrize("mode", ["auto", "control"])
def test_main_refuses_the_new_modes_with_the_staged_probe(tmp_path, fake_kenv, env_ready, capsys, mode):
    code = kp.main(probe_args(tmp_path, op="center", staged=True, phase="sim", cheat_sheet_recipes=mode, agent_hw=False),
                   env=fake_kenv, out=io.StringIO(), probe=lambda e, m: True, run_watched_fn=no_run)
    assert code == 2 and not (tmp_path / "out").exists()
    err = capsys.readouterr().err
    assert f"--cheat-sheet-recipes {mode} is not offered with --staged" in err


def test_the_command_line_takes_the_four_modes_and_still_defaults_to_off(monkeypatch, tmp_path):
    calls = _fake_probe_main(monkeypatch)
    base = ["kernel-probe", "--op", "add", "--endpoint", "http://127.0.0.1:9/v1", "--model", "m"]
    for mode in (None, "on", "off", "auto", "control"):
        cli.main(base + (["--cheat-sheet-recipes", mode] if mode else []), env={"HOME": str(tmp_path)}, stdout=io.StringIO())
    assert [c[0].cheat_sheet_recipes for c in calls] == ["off", "on", "off", "auto", "control"]


# ---- the shipped records against the compilers of this machine ---------------------------------------------------------------------

def _installed(python: Path):
    try:
        return hc.package_identity(python)
    except hc.HwCompileError:
        return None


@pytest.mark.parametrize("python", [PAIR, SHARED], ids=["pair", "shared venv"])
@pytest.mark.parametrize("files", ["recipes", "control"])
def test_the_shipped_compile_records_are_what_the_installed_compiler_says_today(python, files, tmp_path):
    """Compile every shipped skeleton again with the installed chip package and compare with its record. Skips when this machine does
    not have a chip package the records name (by version and compiler bytes), or has no Landlock."""
    installed = _installed(python)
    data_path, verified_path = kp._section_files("on" if files == "recipes" else "control")
    data, verified = op_cheats.load_recipes(data_path), op_cheats.load_verified(verified_path)
    if installed is None or not landlock_available() or not any(hc.same_target(installed, t) for t in verified["targets"]):
        pytest.skip("this machine has no chip package that the shipped records name")
    for recipe in data["recipes"]:
        kernel = tmp_path / f"{recipe['name']}.py"
        kernel.write_text(op_cheats.skeleton_text(recipe), encoding="utf-8")
        now = hc.compile_check(python, kernel, hc.spec_shapes(recipe["spec"]), identity=installed)
        then = next(e["record"] for e in verified["records"][recipe["name"]]["compile"] if hc.same_target(e["record"]["target"], installed))
        assert (now["category"], now["first_line"], now["kernel_sha256"]) == (then["category"], then["first_line"], then["kernel_sha256"]), recipe["name"]
        assert [s["sha256"] for s in now["kernel_sources"]] == [s["sha256"] for s in then["kernel_sources"]], recipe["name"]
