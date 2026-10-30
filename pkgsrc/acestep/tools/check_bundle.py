# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks that the two things this package does not ship match what it was tested with. Nothing here opens a device or the network.

    python tools/check_bundle.py --code BUNDLE_CODE_DIR --snapshot WEIGHTS_SNAPSHOT_DIR [--hash-weights]

BUNDLE_CODE_DIR is the `code/` directory of the community bundle (see pins.json for the repository and revision). Its hash is a hash over every file
(path and sha256 of its bytes, in sorted order; `__pycache__` and `.pyc` files are left out). WEIGHTS_SNAPSHOT_DIR is a snapshot of the model repository:
its directory name must be the pinned revision and the weight files must have the pinned sizes; `--hash-weights` also hashes them (about 5 GB).
The exit status is 0 when everything matches. The package's own files are checked by `sha256sum -c SHA256SUMS`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PINS = Path(__file__).resolve().parent.parent / "pins.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(code_dir) -> tuple[str, int]:
    """(sha256 over the sorted `path<TAB>sha256` lines of every file under `code_dir`, the number of files)."""
    root = Path(code_dir)
    lines = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_dir() or "__pycache__" in rel.parts or p.suffix == ".pyc":
            continue
        lines.append(f"{rel.as_posix()}\t{sha256_file(p)}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(), len(lines)


def problems(code_dir, snapshot, pins: dict, hash_weights: bool = False) -> list[str]:
    out = []
    code = Path(code_dir)
    if not code.is_dir():
        out.append(f"bundle code directory {code} does not exist")
    else:
        links = [p.relative_to(code).as_posix() for p in code.rglob("*") if p.is_symlink() and "__pycache__" not in p.parts]
        if links:
            out.append(f"bundle code holds {len(links)} link(s) (first: {links[0]}); ACE-Step finds its model code through its own resolved path and would silently run "
                       "other code. Use a real copy: hf download --local-dir")
        got, n = tree_hash(code)
        if got != pins["bundle_tree_sha256"] or n != pins["bundle_files"]:
            out.append(f"bundle code is not the pinned revision {pins['bundle_revision']}: {n} files, tree hash {got[:12]} (pinned {pins['bundle_files']} files, {pins['bundle_tree_sha256'][:12]})")
    snap = Path(snapshot)
    if not snap.is_dir():
        out.append(f"weights snapshot {snap} does not exist")
        return out
    if snap.resolve().name != pins["model_revision"] and snap.name != pins["model_revision"]:
        out.append(f"weights snapshot directory is named {snap.name}; the pinned revision is {pins['model_revision']}")
    for rel, want in pins["weights"].items():
        f = snap / rel
        if not f.is_file():
            out.append(f"missing weight file {rel}")
        elif f.stat().st_size != want["size"]:
            out.append(f"{rel}: size {f.stat().st_size}, pinned {want['size']}")
        elif hash_weights and sha256_file(f) != want["sha256"]:
            out.append(f"{rel}: sha256 differs from the pinned one")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--code", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--hash-weights", action="store_true")
    a = p.parse_args(argv)
    pins = json.loads(PINS.read_text())
    found = problems(a.code, a.snapshot, pins, a.hash_weights)
    for x in found:
        print("PROBLEM", x)
    print("check_bundle:", "FAIL" if found else "pass")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
