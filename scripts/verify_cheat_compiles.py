#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Compile the probe kernels of the TT-Lang cheat sheet with each chip package's own compiler and record the verdicts (issue 801).

    python3 scripts/verify_cheat_compiles.py              # compiles with the managed pair venv and the shared venv's tt-lang, writes the data
    python3 scripts/verify_cheat_compiles.py --check      # compiles, compares with the data on disk, writes nothing; exit 1 on a difference
    python3 scripts/verify_cheat_compiles.py --python PATH [--python PATH ...] [--out FILE]

The probes are in orchard/cheat_probes.py. Each one is compiled with `ttlang_hwcompile.compile_check`: the compiler of the chip package whose
interpreter is given reads the kernel with no device, and the child process shuts /dev/tenstorrent out of itself and proves it. No lease is
needed and nothing opens a chip. The data (`ttlang_cheat_compiles.json`) holds each probe's text and, per package, the record. Run this after
a chip package changes (the data names the compilers it was made with, and `verify` says so when the hardware package is another one), and
after a change to a probe. The hardware package, the managed pair, has to be among the interpreters: the claims are judged by it.

Exit 0 when the data was written (or, with --check, matches), 1 when an expectation failed or --check found a difference, 2 for a usage error.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orchard import cheat_compiles, cheat_probes, safeio, ttlang_hwcompile, ttlang_pair  # noqa: E402


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


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def compile_probe(name: str, probe: dict, python, identity: dict, work: Path, today: str, compile_check=ttlang_hwcompile.compile_check) -> dict:
    """One record of probe `name` made with the compiler of `python` (whose identity was read once, before the first compile)."""
    kernel = work / f"{name}--{identity['version']}" / "kernel.py"
    kernel.parent.mkdir(parents=True, exist_ok=True)
    kernel.write_text(cheat_compiles.probe_text(probe), encoding="utf-8")
    return {"date": today, "record": compile_check(python, kernel, [tuple(s) for s in probe["shapes"]], identity=identity)}


def build(pythons, today: str, compile_check=ttlang_hwcompile.compile_check, say=print) -> dict:
    """The whole data file: every probe compiled with every interpreter."""
    identities = [ttlang_hwcompile.package_identity(p) for p in pythons]
    probes = {}
    with tempfile.TemporaryDirectory(prefix="orchard-cheat-compiles-") as tmp:
        for d in cheat_probes.definitions():
            body = {"expect": d["expect"], ("covers" if d["expect"] == "compiled" else "refuses"): d["covers" if d["expect"] == "compiled" else "refuses"],
                    "shapes": d["shapes"], "kernel": d["text"].rstrip("\n").split("\n")}
            body["kernel_sha256"] = cheat_compiles.probe_sha256(body)
            body["compile"] = []
            for python, identity in zip(pythons, identities):
                item = compile_probe(d['name'], body, python, identity, Path(tmp), today, compile_check)
                body["compile"].append(item)
                rec = item["record"]
                say(f"{d['name']:28s} {ttlang_hwcompile.target_label(identity):38s} {rec['category']}"
                    f"{': ' + rec['first_line'] if rec['first_line'] else ''}")
            probes[d["name"]] = body
    return {"schema": cheat_compiles.SCHEMA, "date": today,
            "note": "Each probe kernel compiled with each chip package's own compiler, no device (orchard/ttlang_hwcompile.py). A `compile` record "
                    "means the compiler accepted the kernel and wrote kernel source; it does not show that tt-metal builds it, that the buffers fit, "
                    "or that the numbers are right on a chip. Written by scripts/verify_cheat_compiles.py.",
            "targets": identities, "probes": probes}


def expectation_failures(data: dict) -> list[str]:
    """Where a record disagrees with the probe's `expect`: a `refused` probe that compiled, a `compiled` probe the hardware package (the first
    target) refused, or a check that could not be made. A `compiled` probe that another package refuses is expected (1.1.6 lacks make_dfb)."""
    out = []
    hardware = data["targets"][0]
    for name, probe in data["probes"].items():
        for item in probe["compile"]:
            rec = item["record"]
            mine = ttlang_hwcompile.same_target(rec["target"], hardware)
            if rec["category"] not in ("compiled", "compile_error"):
                out.append(f"{name}: {ttlang_hwcompile.target_label(rec['target'])}: the check did not give a verdict ({rec['category']}: {rec['first_line']})")
            elif probe["expect"] == "refused" and rec["category"] == "compiled":
                out.append(f"{name}: {ttlang_hwcompile.target_label(rec['target'])}: expected a refusal and it compiled")
            elif probe["expect"] == "compiled" and mine and rec["category"] != "compiled":
                out.append(f"{name}: the hardware package refused a probe expected to compile ({rec['first_line']})")
    return out


def differences(old: dict, new: dict) -> list[str]:
    """What changed between two data files, by probe and package: the verdict (category and first line) and the kernel bytes."""
    out = []
    for name in sorted(set(old.get("probes", {})) | set(new.get("probes", {}))):
        a, b = old.get("probes", {}).get(name), new.get("probes", {}).get(name)
        if a is None or b is None:
            out.append(f"{name}: " + ("added" if a is None else "removed"))
            continue
        if a["kernel_sha256"] != b["kernel_sha256"]:
            out.append(f"{name}: the kernel changed")
        va = {(i["record"]["target"]["version"], i["record"]["target"]["compiler_sha256"]): (i["record"]["category"], i["record"]["first_line"]) for i in a["compile"]}
        vb = {(i["record"]["target"]["version"], i["record"]["target"]["compiler_sha256"]): (i["record"]["category"], i["record"]["first_line"]) for i in b["compile"]}
        for key in sorted(set(va) | set(vb)):
            if va.get(key) != vb.get(key):
                out.append(f"{name}: tt-lang {key[0]}: {va.get(key)} -> {vb.get(key)}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--python", action="append", default=[], metavar="PATH", help="interpreter of a venv with a chip package (first: the hardware package)")
    parser.add_argument("--out", metavar="FILE", help="where to write the data (default: the shipped file)")
    parser.add_argument("--check", action="store_true", help="compile and compare with the data on disk; write nothing")
    args = parser.parse_args(argv)
    pythons = [Path(p) for p in args.python] or default_chip_pythons()
    if not pythons:
        print("error: no venv with a tt-lang chip package was found: build the pair with `tt-orchard kernel-env --pair --apply`")
        return 2
    out = Path(args.out) if args.out else cheat_compiles.PATH
    try:
        data = build(pythons, _now().date().isoformat())
    except ttlang_hwcompile.HwCompileError as exc:
        print(f"error: {exc}")
        return 2
    failures = expectation_failures(data)
    for f in failures:
        print("expectation failed:", f)
    shape = cheat_compiles.shape_problems(data)
    for p in shape:
        print("problem:", p)
    if args.check:
        try:
            diff = differences(cheat_compiles.load(out), data)
        except cheat_compiles.CheatCompilesError as exc:
            print(f"error: {exc}")
            return 1
        for d in diff:
            print("differs:", d)
        print("the data on disk matches these compiles" if not diff else f"{len(diff)} differences from the data on disk")
        return 1 if diff or failures or shape else 0
    if failures or shape:
        print("not written: fix the probes or the expectations first")
        return 1
    safeio.atomic_write(out, json.dumps(data, indent=1) + "\n", mode=0o644)
    print(f"wrote {out.name}: {len(data['probes'])} probes, {len(data['targets'])} chip packages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
