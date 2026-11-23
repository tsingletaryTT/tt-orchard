#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Of the kernels that passed the simulator in finished kernel-probe runs, which compile with a chip package's own compiler?

    python3 scripts/hwcompile_probe_passes.py PROBE_DIR [PROBE_DIR ...] [--chip-python PY ...] [--out FILE]

Each PROBE_DIR is the output directory of one `tt-orchard kernel-probe` run (it holds kernel-probe-report.json), or a directory of such
directories. For every attempt whose record says `passed_sim`, the kernel the probe judged (`.verdict/<op>-<n>/authoritative/kernel.py`,
matched against the sha256 in the attempt's provenance) is compiled with each chip package and no device (orchard/ttlang_hwcompile.py).
The table says, per run, how many attempts passed the simulator and how many of those each chip compiler accepts.

A simulator pass that a chip compiler refuses cannot run on a chip with that package. A pass it accepts has cleared the first step only:
the check does not build the kernel source with tt-metal and does not run it. Everything here runs on the CPU and opens no device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchard import kernelprobe, safeio, ttlang_hwcompile, ttlang_pair  # noqa: E402

REPORT = "kernel-probe-report.json"
SCHEMA = "orchard-hwcompile-probe-passes/1"


def probe_dirs(paths) -> list[Path]:
    found = []
    for path in map(Path, paths):
        if (path / REPORT).is_file():
            found.append(path)
        elif path.is_dir():
            found += sorted(p for p in path.iterdir() if (p / REPORT).is_file())
    return found


def judged_kernel(probe_dir: Path, attempt: dict) -> Path | None:
    """The kernel file the probe's own check judged for this attempt, when its bytes still match the attempt's provenance."""
    op, n = attempt.get("op"), attempt.get("attempt")
    want = (attempt.get("provenance") or {}).get("kernel_sha256")
    if not isinstance(op, str) or type(n) is not int or not isinstance(want, str):
        return None
    path = kernelprobe.verdict_dir(probe_dir, op, n) / kernelprobe.CLEAN_DIR / "kernel.py"
    try:
        data = safeio.read_bytes(path, max_bytes=ttlang_hwcompile.MAX_KERNEL_BYTES)
    except safeio.SafeReadError:
        return None
    return path if hashlib.sha256(data).hexdigest() == want else None


def check_run(probe_dir: Path, targets: list, compile_check=ttlang_hwcompile.compile_check) -> dict:
    """One probe run: its recipes setting, its attempts, its simulator passes and, per chip package, the passes that compile."""
    report = safeio.read_json(probe_dir / REPORT, max_bytes=kernelprobe.RECORD_MAX_BYTES, expect=dict)
    attempts = [a for a in report.get("attempts", []) if isinstance(a, dict)]
    recipes = (report.get("header") or {}).get("cheat_recipes")
    row = {"run": probe_dir.name, "op": attempts[0].get("op") if attempts else None, "attempts": len(attempts),
           "recipes": (recipes.get("section", "on") if isinstance(recipes, dict) and recipes.get("on") is True else "off"),
           "simulator_passes": 0, "kernels": []}
    for attempt in attempts:
        if attempt.get("passed_sim") is not True:
            continue
        row["simulator_passes"] += 1
        kernel = judged_kernel(probe_dir, attempt)
        entry = {"attempt": attempt.get("attempt"), "kernel_sha256": (attempt.get("provenance") or {}).get("kernel_sha256"), "compile": []}
        for python, identity in targets:
            if kernel is None:
                entry["compile"].append({"version": identity["version"], "category": "not_run",
                                         "first_line": "the judged kernel is missing or its bytes do not match the attempt's record"})
                continue
            spec = kernelprobe.load_spec(attempt["op"])
            got = compile_check(python, kernel, ttlang_hwcompile.spec_shapes(spec), identity=identity)
            entry["compile"].append({"version": identity["version"], "category": got["category"], "first_line": got["first_line"]})
        row["kernels"].append(entry)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe_dir", nargs="+")
    ap.add_argument("--chip-python", action="append", help="the interpreter of a venv with a chip package (default: the pair and the shared venv)")
    ap.add_argument("--out", help="write the rows as JSON to this file")
    args = ap.parse_args(argv)
    pythons = [Path(p) for p in args.chip_python] if args.chip_python else [
        ttlang_pair.default_venv() / "bin" / "python", Path.home() / ".tenstorrent-venv" / "bin" / "python"]
    targets = []
    for python in pythons:
        try:
            targets.append((python, ttlang_hwcompile.package_identity(python)))
        except ttlang_hwcompile.HwCompileError as exc:
            print(f"skipped {python}: {exc}", file=sys.stderr)
    if not targets:
        print("no chip package found", file=sys.stderr)
        return 2
    rows = [check_run(d, targets) for d in probe_dirs(args.probe_dir)]
    versions = [identity["version"] for _, identity in targets]
    print("| run | op | recipes | attempts | simulator passes | " + " | ".join(f"compile with tt-lang {v}" for v in versions) + " |")
    print("|---|---|---|---|---|" + "---|" * len(versions))
    for row in rows:
        cells = []
        for v in versions:
            verdicts = [c for k in row["kernels"] for c in k["compile"] if c["version"] == v]
            good = sum(c["category"] == "compiled" for c in verdicts)
            why = "; ".join(sorted({c["first_line"] for c in verdicts if c["category"] != "compiled"}))
            cells.append(f"{good} of {len(verdicts)}" + (f" ({why})" if why else ""))
        print(f"| {row['run']} | {row['op']} | {row['recipes']} | {row['attempts']} | {row['simulator_passes']} | " + " | ".join(cells) + " |")
    if args.out:
        safeio.atomic_write(Path(args.out), json.dumps({"schema": SCHEMA, "targets": [i for _, i in targets], "rows": rows}, indent=1) + "\n", mode=0o644)
    return 0


if __name__ == "__main__":
    sys.exit(main())
