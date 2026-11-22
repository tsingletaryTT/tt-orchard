#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Verify the recipes of the TT-Lang cheat sheet and write their verification records.

    python3 scripts/verify_ttlang_recipes.py [--only NAME] [--work DIR] [--chip-package DIR] [--sim PATH] [--windows FILE] [--no-write]

For each recipe in orchard/skills/kernel-loop-templates/ttlang_recipes.json this runs, one at a time and on the CPU:

1. the recipe's skeleton, as kernel.py, through the unedited kernel check (orchard/skills/kernel-loop-templates/kernel_check.py) on the
   real tt-lang simulator, with the recipe's own spec and CPU reference (recipe_references.py). The check runs all six input sets
   (three seeds, each with and without outliers). It must pass.
2. every wrong variant of the skeleton (the recipe's `negatives`: one expression changed, or a claim the recipe makes about a call).
   Each must FAIL the same check. A check that passes a wrong kernel cannot vouch for the right one.
3. a static look at the chip form (the skeleton with the recipe's `hardware_replace` pairs applied): every `ttl.*` call is found in the
   installed chip package under the same namespace, and every keyword is a parameter of a function of that name there. The package is read
   with `ast` and never imported (importing tt-lang can open a device). This shows the names and keywords exist. It does not show the
   kernel compiles for a chip, and no record says so.

It then writes `ttlang_recipes.verified.json` next to the recipes. A recipe whose skeleton changes needs a new run: the record carries the
skeleton's sha256, and the cheat sheet refuses a recipe whose record does not match.

The simulator runs use CPU only; no chip, no gozer and no device is opened. `--windows FILE` appends the UTC start and end of every
simulator run, for a caller that needs to know when the host was busy.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchard import op_cheats, safeio  # noqa: E402

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


# ---- the chip form, read statically ---------------------------------------------------------------------------------------


def _all_names(tree: ast.Module) -> set:
    """The names in a module's `__all__`, when it is a literal list or tuple (a starred `*other` is skipped by the caller)."""
    names: set = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            if isinstance(node.value, (ast.List, ast.Tuple)):
                names |= {e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
    return names


def _parse(path: Path) -> ast.Module:
    return ast.parse(safeio.read_bytes(path, max_bytes=4_000_000).decode("utf-8", errors="replace"))


def chip_namespaces(package: Path) -> dict:
    """What the installed chip package exposes: `ttl.<name>` names, `ttl.math.<name>` names, and the parameters of each function by name.

    `ttl.math` is the generated elementwise functions plus the names ttl_math.py imports from operators.py; `ttl` itself exports its
    `__init__` names plus the generated elementwise ones. There is no `ttl.block` namespace. Read with `ast`; nothing is imported."""
    generated = _all_names(_parse(package / "_generated_elementwise.py"))
    math_tree = _parse(package / "ttl_math.py")
    math_names = set(generated)
    for node in math_tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "operators":
            math_names |= {a.asname or a.name for a in node.names}
    init_tree = _parse(package / "__init__.py")
    top_names = set(generated)
    for node in ast.walk(init_tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("ttl.ttl", "ttl.operators"):
            top_names |= {a.asname or a.name for a in node.names}
    params: dict = {}
    for path in package.rglob("*.py"):
        if "sim" in path.relative_to(package).parts or "tutorials" in path.relative_to(package).parts:
            continue
        for node in ast.walk(_parse(path)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                params.setdefault(node.name, set()).update(x.arg for x in a.args + a.kwonlyargs + a.posonlyargs)
                if a.kwarg is not None:
                    params[node.name].add("**")
    return {"top": top_names, "math": math_names, "params": params}


def _call_chain(node: ast.AST) -> list | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return list(reversed(parts)) if isinstance(node, ast.Name) and node.id == "ttl" and parts else None


def static_hardware_check(recipe: dict, namespaces: dict) -> dict:
    """Every `ttl.*` call in the chip form exists under its namespace in the chip package, with only parameters it has."""
    text = "\n".join(op_cheats.hardware_skeleton(recipe))
    tree = ast.parse(text)
    checked, problems = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _call_chain(node.func) if isinstance(node.func, ast.Attribute) else None
        if chain is None:
            continue
        name = "ttl." + ".".join(chain)
        keywords = sorted(k.arg for k in node.keywords if k.arg)
        space, leaf = (namespaces["math"], chain[1]) if chain[0] == "math" and len(chain) == 2 else (namespaces["top"], chain[0])
        if len(chain) > 2 or (len(chain) == 2 and chain[0] != "math"):
            problems.append(f"{name}: a namespace the chip package does not have")
        elif leaf not in space:
            problems.append(f"{name}: not exported by the chip package")
        else:
            known = namespaces["params"].get(leaf)
            if known is not None and "**" not in known:
                missing = [k for k in keywords if k not in known]
                if missing:
                    problems.append(f"{name}: the keyword {missing[0]!r} is not a parameter of {leaf} in the chip package")
        checked.append({"name": name, "keywords": keywords})
    seen, unique = set(), []
    for c in checked:
        key = (c["name"], tuple(c["keywords"]))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return {"ok": not problems and bool(unique), "checked": unique, "problems": problems}


def package_version(package: Path) -> str:
    for meta in sorted(package.parent.glob("tt_lang-*.dist-info")):
        return meta.name.removesuffix(".dist-info")
    return "unknown"


def find_chip_package() -> Path:
    home = Path.home()
    for candidate in sorted((home / ".tenstorrent-venv" / "lib").glob("python3.*/site-packages/ttl")):
        return candidate
    raise SystemExit("no installed chip package found: pass --chip-package DIR (the ttl directory of the tt-lang package)")


# ---- the run ------------------------------------------------------------------------------------------------------------


def verify_recipe(recipe: dict, overlay: Path, work: Path, sim: str, namespaces: dict, windows: Path | None) -> dict:
    name, spec = recipe["name"], recipe["spec"]
    kernel_text = "\n".join(recipe["skeleton"]) + "\n"
    print(f"{name}: unedited check on the skeleton ...", flush=True)
    check = run_check(overlay, work, name, spec, kernel_text, sim, windows)
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
    record["hardware_static"] = static_hardware_check(recipe, namespaces)
    print(f"{name}: chip-form calls {'all found' if record['hardware_static']['ok'] else 'PROBLEMS: ' + '; '.join(record['hardware_static']['problems'])}", flush=True)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", help="verify one recipe by name (the others keep their existing record)")
    ap.add_argument("--work", help="scratch directory (default: a temporary one)")
    ap.add_argument("--chip-package", help="the ttl directory of the installed chip package (default: the shared venv's)")
    ap.add_argument("--sim", help="the tt-lang-sim executable (default: the kernel probe's)")
    ap.add_argument("--windows", help="append the UTC window of every simulator run to this file")
    ap.add_argument("--no-write", action="store_true", help="print the result and do not write ttlang_recipes.verified.json")
    args = ap.parse_args(argv)
    recipes = op_cheats.load_recipes()
    sim = args.sim
    if not sim:
        from types import SimpleNamespace
        from orchard import kernelprobe
        sim = str(kernelprobe.default_env(SimpleNamespace(src_dir=None, venv_dir=None)).sim)
    package = Path(args.chip_package) if args.chip_package else find_chip_package()
    namespaces = chip_namespaces(package)
    work_ctx = tempfile.TemporaryDirectory(prefix="recipes-verify-") if not args.work else None
    work = Path(args.work) if args.work else Path(work_ctx.name)
    work.mkdir(parents=True, exist_ok=True)
    windows = Path(args.windows) if args.windows else None
    try:
        overlay = build_overlay(work, recipes)
        existing = {}
        if args.only:
            existing = op_cheats.load_verified().get("records", {})
        records = dict(existing)
        for recipe in recipes["recipes"]:
            if args.only and recipe["name"] != args.only:
                continue
            records[recipe["name"]] = verify_recipe(recipe, overlay, work, sim, namespaces, windows)
        sim_version = subprocess.run([sim, "--version"], capture_output=True, text=True, timeout=60, check=False).stdout.strip()
        out = {"schema": op_cheats.VERIFIED_SCHEMA, "date": datetime.date.today().isoformat(), "simulator": sim_version or "unknown",
               "chip_package": package_version(package),
               "note": "Simulator passes and static name checks only. No recipe has been compiled on a chip.", "records": records}
        problems = op_cheats.verified_problems(recipes, out)
        if problems:
            print("NOT WRITTEN: " + "; ".join(problems), file=sys.stderr)
            return 1
        if not args.no_write:
            (TEMPLATES / "ttlang_recipes.verified.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {TEMPLATES / 'ttlang_recipes.verified.json'}")
        return 0
    finally:
        if work_ctx is not None:
            work_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())
