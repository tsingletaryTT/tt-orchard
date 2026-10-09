# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Keep a copied tt-model bundle's interpreter inside the bundle.

A bundle's uv venv is built "relocatable", but three things still name the directory tt-model first
installed it in, by absolute path: `venv/bin/python`, the `.python/cpython-3.12-*` alias of the
interpreter, and the `home` line of `venv/pyvenv.cfg`. A copy under the lab root works on the box it
was copied on only while the original is still there, and on another box not at all. `relink` points
each of them at the bundle's own `.python` (links relative, `home` at the bundle's path); it changes
nothing else and leaves a link it cannot place inside the bundle as it is, for `problems` to report.

Standard library only, and runnable as `python3 -c <this file's source> [--check] BUNDLE...`, so lab
setup can send it to a box that has no orchard checkout. Exit 1 when a problem remains.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MARK = "/.python/"


def _links(bundle: Path):
    for d in (bundle / ".python", bundle / "venv" / "bin"):     # the alias links first: the venv's link goes through one
        if d.is_dir():
            yield from sorted(p for p in d.iterdir() if p.is_symlink())


def _outside(bundle: Path, target: str) -> bool:
    return os.path.isabs(target) and not (target + "/").startswith(str(bundle) + "/")


def _inside(bundle: Path, target: str) -> Path | None:
    """The bundle's own copy of an absolute path that names some install's `.python/...`."""
    i = target.rfind(MARK)
    return bundle / ".python" / target[i + len(MARK):] if i >= 0 else None


def problems(bundle) -> list[str]:
    bundle = Path(os.path.abspath(bundle))
    if not (bundle / "venv").is_dir():
        return []
    out = []
    for link in _links(bundle):
        target = os.readlink(link)
        rel = link.relative_to(bundle)
        if _outside(bundle, target):
            out.append(f"{rel} links outside the bundle, to {target}")
        elif not link.exists():
            out.append(f"{rel} is a broken link (to {target})")
    cfg = bundle / "venv" / "pyvenv.cfg"
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "home" and _outside(bundle, value.strip()):
                out.append(f"venv/pyvenv.cfg home names {value.strip()}, outside the bundle")
    return out


def relink(bundle) -> list[str]:
    """Point the venv's interpreter, the `.python` aliases and `home` inside the bundle. Returns what
    changed, one line each."""
    bundle = Path(os.path.abspath(bundle))
    if not (bundle / "venv").is_dir():
        return []
    changed = []
    for link in _links(bundle):
        target = os.readlink(link)
        own = _inside(bundle, target) if _outside(bundle, target) else None
        if own is None or not os.path.exists(own):
            continue
        new = os.path.relpath(own, link.parent)
        tmp = link.with_name(link.name + ".orchard-relink")
        os.symlink(new, tmp)
        os.replace(tmp, link)
        changed.append(f"{link.relative_to(bundle)} -> {new}")
    cfg = bundle / "venv" / "pyvenv.cfg"
    if cfg.is_file():
        lines = cfg.read_text().splitlines(keepends=True)
        for i, line in enumerate(lines):
            key, _, value = line.partition("=")
            value = value.strip()
            if key.strip() != "home" or not _outside(bundle, value):
                continue
            own = _inside(bundle, value)
            if own is not None and os.path.isdir(own):
                lines[i] = f"home = {own}\n"
                changed.append(f"venv/pyvenv.cfg home = {own}")
        new_text = "".join(lines)
        if new_text != cfg.read_text():
            tmp = cfg.with_name("pyvenv.cfg.orchard-relink")
            tmp.write_text(new_text)
            os.replace(tmp, cfg)
    return changed


def main(argv) -> int:
    check = "--check" in argv
    bundles = [a for a in argv if a != "--check"]
    bad = False
    for b in bundles:
        if not check:
            for line in relink(b):
                print(f"{b}: {line}")
        for p in problems(b):
            print(f"{b}: {p}")
            bad = True
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
