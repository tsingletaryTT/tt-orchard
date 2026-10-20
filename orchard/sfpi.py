# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Is the box's SFPI compiler at least the version the chosen bundle pins? (preflight row `sfpi`)

On 2026-10-13 a system update downgraded SFPI from 7.83.0 to 7.67.0. Every host-served boot of the Qwen3.8 bundle then
failed in 13 s with `'vLut8si' is not a member of 'sfpi'`, and three boots were spent before dpkg.log showed why. The
bundle's own ttnn wheel pins the version it was built with: `ttnn/tt_metal/sfpi-version`, 7.73.0 for the Qwen3.8-27B
bundle. 7.83.0 and 7.87.0 booted it; 7.67.0 did not. The rule is installed >= pinned. Two systems and one failure make
that a rule fitted to three observations, so a row that blocks says so. Installing a system package is the operator's
call, so the row can only block.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from orchard.preflight import BLOCK, OK, WARN, Check

PIN_LINE = re.compile(r"""^\s*sfpi_version\s*=\s*['"]([^'"]+)['"]""", re.MULTILINE)
PIN_GLOB = "venv/lib/python3*/site-packages/ttnn/tt_metal/sfpi-version"
FIX = ("Install a newer release from https://github.com/tenstorrent/sfpi/releases (the .deb for this OS, "
       "`sudo dpkg -i`). tt-orchard does not install system packages")


def parse_version(text) -> tuple[int, ...] | None:
    """(7, 73, 0) from "7.73.0" or "7.87.0-1ubuntu2"; None when there is no dotted number."""
    m = re.match(r"\s*(\d+(?:\.\d+)*)", text) if isinstance(text, str) else None
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def pinned_version(models_root, bundle: str) -> str | None:
    """The SFPI version the installed bundle's ttnn wheel pins, or None when it names none."""
    for f in sorted((Path(models_root).expanduser() / bundle).glob(PIN_GLOB)):
        try:
            m = PIN_LINE.search(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if m and parse_version(m.group(1)):
            return m.group(1)
    return None


def _run(argv) -> tuple[int, str]:
    done = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    return done.returncode, done.stdout


def installed_version(run=_run) -> str | None:
    """The `sfpi` package version dpkg reports, or None when it cannot be read."""
    try:
        rc, out = run(["dpkg-query", "-W", "-f=${Version}", "sfpi"])
    except (OSError, subprocess.SubprocessError):
        return None
    return out.strip() if rc == 0 and parse_version(out) else None


def check_sfpi(installed: str | None, pinned: str | None, bundle: str, where: str | None = None) -> Check | None:
    """None when the bundle pins no version (nothing to compare). `where` names a box other than this one (the lab)."""
    if pinned is None:
        return None
    at = f" on {where}" if where else ""
    if installed is None:
        return Check("sfpi", WARN, f"the installed SFPI version{at} could not be read (`dpkg-query -W sfpi`); bundle "
                     f"{bundle} pins {pinned}")
    if parse_version(installed) < parse_version(pinned):
        return Check("sfpi", BLOCK, f"the installed SFPI compiler{at} is {installed} and bundle {bundle} pins {pinned}. "
                     "Kernel compiles fail with `'vLut8si' is not a member of 'sfpi'` (seen on 2026-10-13 with 7.67.0 "
                     f"against 7.73.0). {FIX}", "hardware-unhealthy")
    return Check("sfpi", OK, f"installed SFPI {installed}{at} is at least the {pinned} that bundle {bundle} pins")
