#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Check a cheat sheet against source trees on the CPU (issue 523). Standard library plus orchard/op_cheats.py; it never imports ttnn, ttl or
ttml, because importing them opens the device. Run it after a change to a sheet, and against the tree a new coder image is built from:

    python3 scripts/verify_cheat_sheet.py ttnn --root tt_metal=DIR
    python3 scripts/verify_cheat_sheet.py ttlang --root hardware=DIR --root shared=DIR --root tree=DIR --root sim=DIR

`ttnn` and `ttlang` name the shipped sheets (orchard/skills/...); any other argument is the path of a sheet file. DIR for tt_metal is the
directory that holds `ttnn/`. For TT-Lang, hardware is the `ttl/` directory of the pair venv (the package the kernel probe compiles with), shared is the `ttl/` directory of the
shared venv (tt-lang 1.1.6, the fallback), tree is the checkout's `python/ttl` and sim is its
`python/sim`. A `--root` can also come from the variable ORCHARD_CHEAT_ROOT_<NAME> (upper case). Exit 0 when every entry and mistake check
passes, 1 with one line per problem, 2 for a usage error.

For TT-Lang the chip claims are also judged by recorded compiles with the hardware package's own compiler (issue 801): the identity of the
hardware package is read from the `ttl/` directory given for the root `hardware`, and the records are `ttlang_cheat_compiles.json` (or
`--compiles FILE`). A `compiles on hardware` entry needs a compiled record; a name found in the package's files is not enough. Entries tagged
`hardware signature only` that a compiled record covers are listed as notes. scripts/verify_cheat_compiles.py writes the records.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orchard import cheat_compiles, op_cheats, ttlang_hwcompile  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sheet", help="ttnn, ttlang or the path of a sheet file")
    parser.add_argument("--root", action="append", default=[], metavar="NAME=DIR", help="a source tree for the root NAME (repeat for each root)")
    parser.add_argument("--compiles", metavar="FILE", help="TT-Lang sheet: the recorded compiles (default: the shipped ttlang_cheat_compiles.json)")
    args = parser.parse_args(argv)
    path = {"ttnn": op_cheats.TTNN_SHEET, "ttlang": op_cheats.TTLANG_SHEET}.get(args.sheet, Path(args.sheet))
    try:
        sheet = op_cheats.load_sheet(path)
    except op_cheats.SheetError as exc:
        print(f"error: {exc}")
        return 2
    given = {}
    for item in args.root:
        name, _, directory = item.partition("=")
        if not name or not directory:
            print(f"error: --root wants NAME=DIR, got {item!r}")
            return 2
        given[name] = directory
    indexes = {}
    for name in sheet["roots"]:
        directory = given.get(name) or os.environ.get(f"ORCHARD_CHEAT_ROOT_{name.upper()}")
        if not directory:
            print(f"error: no tree for root {name!r}: pass --root {name}=DIR")
            return 2
        indexes[name] = op_cheats.build_index(directory)
        print(f"root {name}: {indexes[name].files} files read" + (" (incomplete)" if indexes[name].incomplete else ""))
    compiles = target = None
    if sheet["sheet"] == "ttlang":
        # The chip claims are judged by recorded compiles with the hardware package's compiler (issue 801): the identity of that package comes
        # from the tree given for the root `hardware`, which has to be the package's own `ttl/` directory.
        try:
            compiles = cheat_compiles.load(args.compiles)
            target = ttlang_hwcompile.identity_of(given.get("hardware") or os.environ["ORCHARD_CHEAT_ROOT_HARDWARE"])
        except (cheat_compiles.CheatCompilesError, ttlang_hwcompile.HwCompileError, KeyError) as exc:
            print(f"error: the compile records or the hardware package could not be read: {exc}")
            return 2
    problems = op_cheats.verify(sheet, indexes, compiles=compiles, hardware_target=target)
    count = sum(1 for _ in op_cheats.entries(sheet))
    for p in problems:
        print("problem:", p)
    if compiles is not None and not problems:
        for name in cheat_compiles.understated(sheet, compiles, target):
            print(f"note: {name} is tagged 'hardware signature only' and a kernel that uses it compiled with the hardware package: the tag could be raised")
    print(f"{count} entries, {len(sheet.get('mistakes', []))} mistakes checked: " + ("all found" if not problems else f"{len(problems)} problems"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
