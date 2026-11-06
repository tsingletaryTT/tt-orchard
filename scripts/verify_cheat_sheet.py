#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Check a cheat sheet against source trees on the CPU (issue 523). Standard library plus orchard/op_cheats.py; it never imports ttnn, ttl or
ttml, because importing them opens the device. Run it after a change to a sheet, and against the tree a new coder image is built from:

    python3 scripts/verify_cheat_sheet.py ttnn --root tt_metal=DIR
    python3 scripts/verify_cheat_sheet.py ttlang --root hardware=DIR --root tree=DIR --root sim=DIR

`ttnn` and `ttlang` name the shipped sheets (orchard/skills/...); any other argument is the path of a sheet file. DIR for tt_metal is the
directory that holds `ttnn/`. For TT-Lang, hardware is the installed package directory (`ttl/`), tree is the checkout's `python/ttl` and sim is its
`python/sim`. A `--root` can also come from the variable ORCHARD_CHEAT_ROOT_<NAME> (upper case). Exit 0 when every entry and mistake check
passes, 1 with one line per problem, 2 for a usage error.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orchard import op_cheats  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sheet", help="ttnn, ttlang or the path of a sheet file")
    parser.add_argument("--root", action="append", default=[], metavar="NAME=DIR", help="a source tree for the root NAME (repeat for each root)")
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
    problems = op_cheats.verify(sheet, indexes)
    count = sum(1 for _ in op_cheats.entries(sheet))
    for p in problems:
        print("problem:", p)
    print(f"{count} entries, {len(sheet.get('mistakes', []))} mistakes checked: " + ("all found" if not problems else f"{len(problems)} problems"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
