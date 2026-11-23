#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Verify the recipes of the TT-Lang cheat sheet and write their verification records.

    python3 scripts/verify_ttlang_recipes.py [--recipes FILE] [--only NAME] [--work DIR] [--chip-python PY ...] [--reuse-simulator]
                                             [--sim PATH] [--windows FILE] [--no-write]

For each recipe in orchard/skills/kernel-loop-templates/ttlang_recipes.json (or `--recipes FILE`) this runs, one at a time and on the CPU:

1. the recipe's skeleton, as kernel.py, through the unedited kernel check (orchard/skills/kernel-loop-templates/kernel_check.py) on the
   real tt-lang simulator, with the recipe's own spec and CPU reference (recipe_references.py). The check runs all six input sets
   (three seeds, each with and without outliers). It must pass.
2. every wrong variant of the skeleton (the recipe's `negatives`: one expression changed, or a claim the recipe makes about a call).
   Each must FAIL the same check. A check that passes a wrong kernel cannot vouch for the right one.
3. for each chip package (`--chip-python`, default: the managed pair venv and the shared venv), the skeleton through that package's own
   compiler with no device (orchard/ttlang_hwcompile.py: the device is shut out with Landlock and `ttnn` is a stand-in), and then each
   of the recipe's `compile_negatives`, which that compiler must refuse. This shows the compiler accepts the skeleton and writes kernel
   source. It does not show the kernel runs on a chip, and the record says `compile`, never `chip`.

It then writes `<recipes>.verified.json` next to the recipes. A recipe whose skeleton changes needs a new run: every record carries the
skeleton's sha256, and the cheat sheet refuses a recipe whose simulator record does not match and shows no compile result for other bytes.
Each compile record names its chip package by version and by the sha256 of the compiler's files, so a result for one package is never
shown for another.

Everything runs on the CPU; no chip, no gozer and no device is opened. `--windows FILE` appends the UTC start and end of every simulator
run, for a caller that needs to know when the host was busy.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchard import op_cheats, safeio, ttlang_hwcompile, ttlang_pair  # noqa: E402

TEMPLATES = ROOT / "orchard" / "skills" / "kernel-loop-templates"
RUN_TIMEOUT_S = 400
CHECK_TIMEOUT_S = 300


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def build_overlay(work: Path, recipes: dict) -> Path:
    """A scratch copy of the shipped templates with the recipes' ops added: a spec per recipe under ops/ and the recipes' references
    added to references.py. The shipped templates are not changed."""
    overlay = work / "overlay"
    if overlay.exists():
        shutil.rmtree(overlay)
    shutil.copytree(TEMPLATES, overlay, ignore=shutil.ignore_patterns("library", "__pycache__", "ttlang_recipes*.json"))
    for recipe in recipes["recipes"]:
        (overlay / "ops" / f"{recipe['spec']['op']}.json").write_text(json.dumps(recipe["spec"]) + "\n", encoding="utf-8")
    with open(overlay / "references.py", "a", encoding="utf-8") as f:
        f.write("\n\nfrom recipe_references import REFERENCES as _RECIPE_REFERENCES  # noqa: E402\nREFERENCES.update(_RECIPE_REFERENCES)\n")
    return overlay


def run_check(overlay: Path, work: Path, label: str, spec: dict, kernel_text: str, sim: str, windows: Path | None) -> dict:
    """The unedited check on `kernel_text` in a fresh directory. Returns the check's own result (check.json)."""
    adir = work / "attempts" / label
    if adir.exists():
        shutil.rmtree(adir)
    adir.mkdir(parents=True)
    (adir / "kernel.py").write_text(kernel_text, encoding="utf-8")
    (adir / "check_config.json").write_text(json.dumps({"sim": sim, "timeout_s": CHECK_TIMEOUT_S, "tt_lang": {}}), encoding="utf-8")
    started = _now()
    try:
        subprocess.run([sys.executable, str(overlay / "kernel_check.py"), "--attempt-dir", str(adir),
                        "--spec", str(overlay / "ops" / f"{spec['op']}.json"), "--sim", sim, "--timeout-s", str(CHECK_TIMEOUT_S)],
                       capture_output=True, text=True, timeout=RUN_TIMEOUT_S, check=False)
    finally:
        if windows is not None:
            with open(windows, "a", encoding="utf-8") as f:
                f.write(f"simulator verification {label} {started:%H:%M:%SZ} {_now():%H:%M:%SZ}\n")
    return json.loads((adir / "check.json").read_text(encoding="utf-8"))


def _measure(sets: list, key: str, pick) -> float | None:
    values = [s.get(key) for s in sets if safeio.is_finite_number(s.get(key))]
    return pick(values) if values else None


# ---- the chip compilers ---------------------------------------------------------------------------------------------------------------


def default_chip_pythons() -> list:
    """The interpreters of the chip packages this machine has: the managed pair venv, then the shared venv. Only the ones that hold a
    tt-lang compiler are returned (read from files; nothing is imported)."""
    found = []
    for python in (ttlang_pair.default_venv() / "bin" / "python", Path.home() / ".tenstorrent-venv" / "bin" / "python"):
        try:
            ttlang_hwcompile.package_identity(python)
        except ttlang_hwcompile.HwCompileError:
            continue
        found.append(python)
    return found


def compile_entry(recipe: dict, python, work: Path, today: str, compile_check=ttlang_hwcompile.compile_check) -> dict:
    """One chip package's verdict on a recipe: the skeleton compiled with that package's own compiler and no device, then each of the
    recipe's compile negatives, which the same compiler must refuse. The identity of the package is read once, before the first compile,
    and every record of this entry carries that identity."""
    identity = ttlang_hwcompile.package_identity(python)
    label = ttlang_hwcompile.target_label(identity)
    shapes = ttlang_hwcompile.spec_shapes(recipe["spec"])
    tag = f"{recipe['name']}--{identity['version']}"
    kernel = work / "compile" / tag / "kernel.py"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text(op_cheats.skeleton_text(recipe), encoding="utf-8")
    print(f"{recipe['name']}: compile with {label}, no device ...", flush=True)
    record = compile_check(python, kernel, shapes, identity=identity)
    print(f"  -> {record['category']}{': ' + record['first_line'] if record['first_line'] else ''}", flush=True)
    negatives = []
    for neg in recipe["compile_negatives"]:
        wrong = kernel.parent / f"wrong-{len(negatives)}.py"
        wrong.write_text(op_cheats.apply_replacements(recipe, neg["replace"], f"compile negative {neg['name']!r}") + "\n", encoding="utf-8")
        got = compile_check(python, wrong, shapes, identity=identity)
        failed = got["category"] == "compile_error"
        negatives.append({"name": neg["name"], "failed": failed, "category": got["category"], "first_line": got["first_line"]})
        print(f"  wrong form {neg['name']!r} -> {got['category']}{': ' + got['first_line'] if got['first_line'] else ''}", flush=True)
    return {"date": today, "record": record, "negatives": negatives}


# ---- the run ------------------------------------------------------------------------------------------------------------


SIM_FIELDS = ("skeleton_sha256", "passed", "category", "first_line", "sets", "pcc_min", "rel_l2_max", "check_sha256", "negatives")


def simulator_record(recipe: dict, overlay: Path, work: Path, sim: str, windows: Path | None) -> dict:
    name, spec = recipe["name"], recipe["spec"]
    print(f"{name}: unedited check on the skeleton ...", flush=True)
    check = run_check(overlay, work, name, spec, op_cheats.skeleton_text(recipe), sim, windows)
    sets = check.get("sets") or []
    record = {"skeleton_sha256": op_cheats.skeleton_sha256(recipe), "passed": check.get("passed") is True, "category": check.get("category"),
              "first_line": check.get("first_line") or "", "sets": len(sets),
              "pcc_min": _measure(sets, "pcc", min), "rel_l2_max": _measure(sets, "rel_l2", max),
              "check_sha256": check.get("sha256") or {}, "negatives": []}
    print(f"  -> {'PASS' if record['passed'] else 'FAIL ' + str(record['category'])} on {record['sets']} sets", flush=True)
    for neg in recipe["negatives"]:
        wrong = op_cheats.apply_replacements(recipe, neg["replace"], f"negative {neg['name']!r}") + "\n"
        print(f"{name}: wrong variant {neg['name']!r} ...", flush=True)
        result = run_check(overlay, work, f"{name}--{neg['name'].replace(' ', '-')}", spec, wrong, sim, windows)
        failed = result.get("passed") is not True
        record["negatives"].append({"name": neg["name"], "failed": failed, "category": result.get("category"),
                                    "first_line": (result.get("first_line") or "")[:200]})
        print(f"  -> {'failed, as it must' if failed else 'PASSED, which it must not'} ({result.get('category')})", flush=True)
    return record


def reusable_simulator_record(recipe: dict, old) -> dict | None:
    """The simulator part of an earlier record, when it is for exactly this skeleton. A record for other bytes is never reused."""
    if not isinstance(old, dict) or old.get("skeleton_sha256") != op_cheats.skeleton_sha256(recipe):
        return None
    return {key: old.get(key) for key in SIM_FIELDS}


def verify_recipe(recipe: dict, overlay: Path | None, work: Path, sim: str | None, chip_pythons, windows: Path | None, today: str,
                  old=None, compile_check=ttlang_hwcompile.compile_check) -> dict:
    """One recipe's record: the simulator part (run now, or reused from `old` when `overlay` is None and `old` is for this skeleton),
    then one compile entry per chip package. Chip entries of `old` are kept only for the same skeleton."""
    record = reusable_simulator_record(recipe, old) if overlay is None else simulator_record(recipe, overlay, work, sim, windows)
    if record is None:
        raise SystemExit(f"{recipe['name']}: --reuse-simulator needs an earlier record for exactly this skeleton, and there is none")
    record["compile"] = [compile_entry(recipe, python, work, today, compile_check) for python in chip_pythons]
    kept = old.get("chip", []) if isinstance(old, dict) and old.get("skeleton_sha256") == record["skeleton_sha256"] else []
    record["chip"] = kept if isinstance(kept, list) else []
    return record


def _old_records(path: Path) -> dict:
    """The records of an earlier verification file of any schema, as plain data. Only `reusable_simulator_record` reads them."""
    try:
        old = safeio.read_json(path, max_bytes=op_cheats.RECIPES_MAX_BYTES, expect=dict)
    except safeio.SafeReadError:
        return {}
    records = old.get("records")
    return records if isinstance(records, dict) else {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recipes", help="the recipes file to verify (default: the shipped ttlang_recipes.json); the records are written "
                                      "next to it as <name>.verified.json")
    ap.add_argument("--only", help="verify one recipe by name (the others keep their existing record)")
    ap.add_argument("--work", help="scratch directory (default: a temporary one)")
    ap.add_argument("--chip-python", action="append",
                    help="the interpreter of a venv with a chip package; may be given more than once (default: the managed pair venv "
                         "and the shared venv, each when it holds a tt-lang compiler)")
    ap.add_argument("--reuse-simulator", action="store_true",
                    help="do not run the simulator: keep the simulator part of each existing record whose skeleton is unchanged and "
                         "only compile with the chip packages. Stops when a skeleton has no such record")
    ap.add_argument("--sim", help="the tt-lang-sim executable (default: the kernel probe's)")
    ap.add_argument("--windows", help="append the UTC window of every simulator run to this file")
    ap.add_argument("--no-write", action="store_true", help="print the result and do not write the records")
    args = ap.parse_args(argv)
    recipes_path = Path(args.recipes) if args.recipes else op_cheats.RECIPES
    verified_path = recipes_path.with_name(recipes_path.stem + ".verified.json")
    recipes = op_cheats.load_recipes(recipes_path)
    chip_pythons = [Path(p) for p in args.chip_python] if args.chip_python else default_chip_pythons()
    targets = [ttlang_hwcompile.package_identity(p) for p in chip_pythons]
    sim = args.sim
    if not sim and not args.reuse_simulator:
        from types import SimpleNamespace
        from orchard import kernelprobe
        sim = str(kernelprobe.default_env(SimpleNamespace(src_dir=None, venv_dir=None)).sim)
    work_ctx = tempfile.TemporaryDirectory(prefix="recipes-verify-") if not args.work else None
    work = Path(args.work) if args.work else Path(work_ctx.name)
    work.mkdir(parents=True, exist_ok=True)
    windows = Path(args.windows) if args.windows else None
    today = datetime.date.today().isoformat()
    try:
        overlay = None if args.reuse_simulator else build_overlay(work, recipes)
        old = _old_records(verified_path)
        records = {}
        for recipe in recipes["recipes"]:
            if args.only and recipe["name"] != args.only:
                if recipe["name"] in old:
                    records[recipe["name"]] = old[recipe["name"]]
                continue
            records[recipe["name"]] = verify_recipe(recipe, overlay, work, sim, chip_pythons, windows, today, old.get(recipe["name"]))
        sim_version = "not run (simulator records reused)"
        if sim:
            sim_version = subprocess.run([sim, "--version"], capture_output=True, text=True, timeout=60, check=False).stdout.strip() or "unknown"
        elif isinstance(_old_top(verified_path).get("simulator"), str):
            sim_version = _old_top(verified_path)["simulator"]
        out = {"schema": op_cheats.VERIFIED_SCHEMA, "date": today, "simulator": sim_version, "targets": targets,
               "note": "Simulator passes, and compiles with each chip package's own compiler made with no device. No recipe has been "
                       "run on a chip unless its record has a `chip` entry.", "records": records}
        problems = op_cheats.verified_problems(recipes, out)
        if problems:
            print("NOT WRITTEN: " + "; ".join(problems), file=sys.stderr)
            return 1
        for target in targets:
            reasons = op_cheats.recipes_gate(recipes, out, target)
            print(f"gate for {ttlang_hwcompile.target_label(target)}: {'holds' if not reasons else 'does not hold: ' + '; '.join(reasons)}")
        if not args.no_write:
            safeio.atomic_write(verified_path, json.dumps(out, indent=1) + "\n", mode=0o644)
            print(f"wrote {verified_path}")
        return 0
    finally:
        if work_ctx is not None:
            work_ctx.cleanup()


def _old_top(path: Path) -> dict:
    try:
        return safeio.read_json(path, max_bytes=op_cheats.RECIPES_MAX_BYTES, expect=dict)
    except safeio.SafeReadError:
        return {}


if __name__ == "__main__":
    sys.exit(main())
