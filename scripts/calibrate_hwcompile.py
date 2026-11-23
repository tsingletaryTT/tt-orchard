#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Compare the device-less chip compile (orchard/ttlang_hwcompile.py) with what a chip said about the same kernels.

    python3 scripts/calibrate_hwcompile.py [--chip-python PY ...] [--out FILE] [--no-write]

docs/run-logs/evidence/ttlang-pair holds 21 kernels and, for two chip packages, what happened when each was compiled and run on a chip
under a lease on 2026-10-29 (`results/116-chip.jsonl` for tt-lang 1.1.6, `results/pair-chip.jsonl` for the pair). This script compiles the
same kernel files with the same packages and no device, and writes one row per kernel and shape: what the chip said, what the compile
check says, and whether they agree. They agree when the chip compiled and ran the kernel and the check compiles it, or when the chip
refused it and the check refuses it with the same first line.

The result shows how far the check can be trusted. Every chip failure in that evidence is a tt-lang compiler error, so the comparison
says nothing about a kernel that tt-lang compiles and tt-metal then fails to build, or about numbers on a chip.

Everything runs on the CPU. No chip, no gozer, no device: the check shuts /dev/tenstorrent out of its child before tt-lang is imported.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchard import safeio, ttlang_hwcompile, ttlang_pair  # noqa: E402

EVIDENCE = ROOT / "docs" / "run-logs" / "evidence" / "ttlang-pair"
OUT = ROOT / "docs" / "run-logs" / "evidence" / "hwcompile-calibration" / "calibration.json"
CHIP_FILES = {"1.1.6": "116-chip.jsonl", "1.1.10": "pair-chip.jsonl"}       # by the version prefix of the package that produced them
SCHEMA = "orchard-hwcompile-calibration/1"


def chip_results(path: Path) -> dict:
    """{(kernel name, (rows, cols)): {"ok": bool, "first_line": str}} from one results file of the chip run."""
    out = {}
    for line in safeio.read_bytes(path, max_bytes=4_000_000).decode("utf-8", errors="replace").splitlines():
        if not line.startswith("RESULT "):
            continue
        entry = json.loads(line[len("RESULT "):])
        for result in entry.get("results", []):
            first = "" if result.get("ok") else str(result.get("error", "")).split("\n")[0].strip()
            out[(entry["name"], tuple(result["shape"]))] = {"ok": result.get("ok") is True, "first_line": first[:300]}
    return out


def tensor_count(kernel_text: str, fn: str) -> int:
    """How many tensors the kernel's operation takes: the parameters of the function named `fn`."""
    for node in ast.parse(kernel_text).body:
        if isinstance(node, ast.FunctionDef) and node.name == fn:
            return len(node.args.args)
    raise SystemExit(f"no function {fn} in the kernel")


def agrees(chip: dict, here: dict) -> bool:
    """The chip compiled and ran it and the check compiles it, or both refuse it with the same first line."""
    if chip["ok"]:
        return here["category"] == "compiled"
    return here["category"] == "compile_error" and here["first_line"] == chip["first_line"]


def calibrate(pythons, compile_check=ttlang_hwcompile.compile_check) -> dict:
    manifest = safeio.read_json(EVIDENCE / "kernels" / "manifest.json", max_bytes=1_000_000, expect=list)
    rows, packages = [], []
    for python in pythons:
        identity = ttlang_hwcompile.package_identity(python)
        prefix = next((p for p in sorted(CHIP_FILES, key=len, reverse=True) if identity["version"].startswith(p)), None)
        if prefix is None:
            print(f"{python}: tt-lang {identity['version']} has no chip results in the evidence; skipped")
            continue
        truth = chip_results(EVIDENCE / "results" / CHIP_FILES[prefix])
        packages.append({"target": identity, "chip_results": f"docs/run-logs/evidence/ttlang-pair/results/{CHIP_FILES[prefix]}"})
        for entry in manifest:
            kernel = EVIDENCE / "kernels" / entry["file"]
            count = tensor_count(kernel.read_text(encoding="utf-8"), entry["fn"])        # hardening: ok our own repository file
            for shape in entry["shapes"]:
                chip = truth.get((entry["name"], tuple(shape)))
                if chip is None:
                    continue                # the chip run did not include this kernel for this package
                here = compile_check(python, kernel, [tuple(shape)] * count, fn=entry["fn"], identity=identity)
                row = {"version": identity["version"], "kernel": entry["name"], "shape": list(shape), "chip": chip,
                       "here": {"category": here["category"], "first_line": here["first_line"], "kernel_sha256": here["kernel_sha256"],
                                "kernel_sources": len(here["kernel_sources"])}}
                row["agree"] = agrees(chip, row["here"])
                rows.append(row)
                print(f"{'agree ' if row['agree'] else 'DIFFER'} tt-lang {identity['version']} {entry['name']} {shape}: chip "
                      f"{'ran it' if chip['ok'] else chip['first_line']} | here {here['category']} {here['first_line']}", flush=True)
    return {"schema": SCHEMA, "date": datetime.date.today().isoformat(), "packages": packages, "rows": rows,
            "note": "Every chip failure in this evidence is a tt-lang compiler error. The comparison does not cover a kernel that tt-lang "
                    "compiles and tt-metal fails to build, a kernel whose buffers do not fit the chip's free L1, or numbers on a chip."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chip-python", action="append", help="the interpreter of a venv with a chip package (default: the pair and the shared venv)")
    ap.add_argument("--out", help=f"where to write the rows (default: {OUT.relative_to(ROOT)})")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    pythons = [Path(p) for p in args.chip_python] if args.chip_python else [
        ttlang_pair.default_venv() / "bin" / "python", Path.home() / ".tenstorrent-venv" / "bin" / "python"]
    result = calibrate(pythons)
    differ = [r for r in result["rows"] if not r["agree"]]
    print(f"{len(result['rows']) - len(differ)} of {len(result['rows'])} rows agree with the chip")
    if not args.no_write:
        out = Path(args.out) if args.out else OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        safeio.atomic_write(out, json.dumps(result, indent=1) + "\n", mode=0o644)
        print(f"wrote {out}")
    return 1 if differ or not result["rows"] else 0


if __name__ == "__main__":
    sys.exit(main())
