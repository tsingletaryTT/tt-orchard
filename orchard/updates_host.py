# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-stack rows of `tt-orchard updates`: SFPI, tt-kmd, tt-flash, tt-smi, the firmware bundle and the reference venv's packages.

`tt-orchard updates` knew the coder, tt-lang, tt-metal, tt-model-manager, tt-cli, gozer and the skills checkout, and called the host
current while the host SFPI (7.71.0) was below what its own tt-metal tree pins (7.89.0) and below what a bundle's install.sh demands
(7.85.0). This module adds the rest of the stack, read only, in the same row shape (`updates.Row`), so the table, the JSON, the cache,
the preflight hint and the web view need no change.

What it never does:

- It opens no chip and runs no tt-smi. tt-smi queued behind the kernel driver's mutex and hung while another process was starting a
  mesh (2026-11-22), and a hung call cannot be killed. The firmware bundle version is read from the driver's sysfs file, which does
  not open the device. That read runs in a daemon thread with a timeout, so a driver that does not answer gives `unknown (not read:
  ...)` and never blocks the command.
- It starts no tt-smi or tt-flash process either: their versions come from the `.dist-info` folder of the venv that holds them.
- The commands it runs are `dpkg-query -W -f=${Version}`, `apt-cache policy` and `modinfo -F version tenstorrent`, each for a fixed
  package or module name (`updates.Tools.cmd` refuses any other argument). None writes anything or reaches the network.
- Every file it reads goes through `safeio` with a size bound; every remote answer goes through `updates.clean`.

No vendor table pairs these versions for newer releases. tt-metal's INSTALLING.md names floors only and the firmware release notes name
no required tt-flash or tt-kmd version. Rule R5 in `updates.RULES` says so in the output; this module compares each item with the
newest upstream release and invents no pairing.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from orchard import safeio, updates
from orchard.updates import BEHIND, CURRENT, HOST_KEYS, UNKNOWN, Row, Tools, clean, vtuple  # noqa: F401

REFERENCE_PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "huggingface-hub")
FIRMWARE_FEED = "https://github.com/tenstorrent/tt-system-firmware/releases.atom"
FIRMWARE_RELEASES = "https://github.com/tenstorrent/tt-system-firmware/releases/tag/"
SYSFS_GLOB = "/sys/class/tenstorrent/*/tt_fw_bundle_ver"
SYSFS_TIMEOUT_S = 3.0           # choice: a sysfs attribute answers in microseconds; this is the wait before the driver is called silent
SMALL_FILE = 4096               # choice: a version file, a pin file or a shebang line is a few dozen bytes
BUNDLE_FILES_MAX = 40           # choice: most sfpi-version files read from installed bundles
DIST_ENTRIES_MAX = 5000         # choice: a venv's site-packages holds a few hundred entries
FEED_ENTRIES_MAX = 60
SFPI_LINE = re.compile(r"sfpi_version='(\d+(?:\.\d+)+)'")
POLICY_CANDIDATE = re.compile(r"^\s*Candidate:\s*(\S+)\s*$", re.M)
FIRMWARE_TAG = re.compile(r'href="' + re.escape(FIRMWARE_RELEASES) + r'(v\d+(?:\.\d+)*)"')
UPDATED = re.compile(r"<updated>([^<]{1,40})</updated>")

# Facts read from release notes, shown only for the exact upstream version they describe, so they cannot outlive it.
KNOWN_NOTES = {
    ("tt-kmd", "2.12.0"): "the ttkmd-2.12.0 release notes list a pinned-memory hang fix and an IOMMU leak fix (read 2026-11-22)",
}


@dataclass
class HostEnv:
    """What the host checks read besides `Tools`. Tests replace the fields."""
    tree: Path | None = None                     # the host tt-metal tree (its tt_metal/sfpi-version is the pin)
    bundle_root: Path | None = None              # tt-model's models folder: <org>/<model>/venv/lib/python*/site-packages/ttnn/...
    reference_python: Path | None = None         # the interpreter of the reference venv
    sysfs_glob: str = SYSFS_GLOB
    sysfs_timeout_s: float = SYSFS_TIMEOUT_S
    which_path: Callable = field(default=lambda name: shutil.which(name))


# ---- small readers -------------------------------------------------------------------------------------------------

def _cmp(a: tuple, b: tuple) -> int:
    """-1, 0 or 1 for two version tuples of different lengths (19.17.1 equals 19.17.1.0)."""
    n = max(len(a), len(b))
    a, b = a + (0,) * (n - len(a)), b + (0,) * (n - len(b))
    return (a > b) - (a < b)


def _dotted(s) -> str:
    """The first dotted version in a string, e.g. "2.14.1" from "2.14.1+cpu"; "" when there is none."""
    m = re.search(r"\d+(?:\.\d+)+", s) if isinstance(s, str) else None
    return m.group() if m else ""


def _read_small(path: Path) -> str | None:
    try:
        return safeio.read_bytes(path, max_bytes=SMALL_FILE).decode("utf-8", "replace")
    except (safeio.SafeReadError, OSError):
        return None


def sfpi_pin(path: Path) -> str | None:
    """`sfpi_version='7.89.0'` from a tt-metal tree's tt_metal/sfpi-version file (the same format ttnn wheels carry)."""
    text = _read_small(path)
    m = SFPI_LINE.search(text) if text else None
    return m.group(1) if m else None


def bundle_sfpi_floors(root: Path | None) -> list[tuple[str, str]]:
    """(bundle name, version) of the SFPI floor each installed tt-model bundle's ttnn declares. Bounded; a missing folder gives []."""
    out: list[tuple[str, str]] = []
    if root is None or not root.is_dir():
        return out
    for i, f in enumerate(sorted(root.glob("*/*/venv/lib/python3.*/site-packages/ttnn/tt_metal/sfpi-version"))):
        if i >= BUNDLE_FILES_MAX:
            break
        v = sfpi_pin(f)
        if v:
            rel = f.relative_to(root).parts            # <org>/<model>/venv/...: the model folder names the bundle
            out.append((clean(rel[1], 60), v))
    return out


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dist_version(venv_root: Path | None, name: str) -> str | None:
    """The version of a package in a venv, from the name of its `.dist-info` folder. No process is started and no code is imported."""
    if venv_root is None:
        return None
    sites = sorted(venv_root.glob("lib/python3.*/site-packages"))
    if not sites:
        return None
    want = _norm(name)
    try:
        with os.scandir(sites[-1]) as it:
            for n, e in enumerate(it):
                if n >= DIST_ENTRIES_MAX:
                    break
                if not e.name.endswith(".dist-info") or e.is_symlink():
                    continue
                stem = e.name[:-len(".dist-info")]
                pkg, _, ver = stem.partition("-")
                if _norm(pkg) == want and ver:
                    return clean(ver, 40)
    except OSError:
        return None
    return None


def venv_of_script(host: HostEnv, name: str) -> Path | None:
    """The venv that runs a console script: the interpreter named on its shebang line, two levels up from bin/."""
    p = host.which_path(name)
    if not p:
        return None
    try:
        # A console script is often a link on PATH (~/.local/bin/tt-smi -> the venv's script). Resolve the operator's own PATH entry
        # once, then read the file it names without following another link.
        head = safeio.read_prefix(Path(p).resolve(strict=True), max_bytes=200).decode("utf-8", "replace")
    except (safeio.SafeReadError, OSError, RuntimeError):
        return None
    first = head.splitlines()[0] if head else ""
    if not first.startswith("#!"):
        return None
    interp = Path(first[2:].strip().split()[0])
    return interp.parent.parent if interp.parent.name == "bin" and (interp.parent.parent / "pyvenv.cfg").exists() else None


def read_sysfs(path: str, timeout_s: float) -> tuple[str | None, str]:
    """(text, why). One sysfs attribute read in a daemon thread. A driver that does not answer in time gives (None, why) and the
    thread is left behind; the command never waits on the kernel."""
    box: dict = {}

    def work():
        try:
            box["v"] = safeio.read_bytes(Path(path), max_bytes=SMALL_FILE).decode("utf-8", "replace").strip()
        except (safeio.SafeReadError, OSError) as exc:
            box["e"] = type(exc).__name__
    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None, f"the driver did not answer in {timeout_s:g} s"
    return box.get("v"), box.get("e", "") if "v" not in box else ""


def _policy_candidate(tools: Tools, pkg: str) -> str | None:
    rc, out, _ = tools.cmd(["apt-cache", "policy", pkg])
    m = POLICY_CANDIDATE.search(out) if rc == 0 else None
    return clean(m.group(1), 40) if m and m.group(1) != "(none)" else None


def _dpkg_version(tools: Tools, pkg: str) -> str | None:
    rc, out, _ = tools.cmd(["dpkg-query", "-W", "-f=${Version}", pkg])
    v = clean(out.strip(), 40) if rc == 0 else ""
    return v if _dotted(v) else None


def _pypi_latest(tools: Tools, name: str):
    """(version, why) of the newest stable release on public PyPI."""
    doc, why = tools.json(f"https://pypi.org/pypi/{name}/json", max_bytes=updates.PYPI_BODY_MAX)
    info = doc.get("info") if isinstance(doc, dict) else None
    ver = clean(info.get("version"), 40) if isinstance(info, dict) else ""
    return (ver, "") if _dotted(ver) else (None, why or "no version in the answer")


# ---- the rows ------------------------------------------------------------------------------------------------------

def check_sfpi(tools: Tools, host: HostEnv) -> Row:
    row = Row("sfpi", "sfpi (device kernel compiler)")
    inst = _dpkg_version(tools, "sfpi")
    if inst is None:
        row.why = "the sfpi package is not installed (dpkg-query gave no version)"
        return row
    row.installed = inst
    cand = _policy_candidate(tools, "sfpi")
    pin = sfpi_pin(host.tree / "tt_metal" / "sfpi-version") if host.tree else None
    floors = bundle_sfpi_floors(host.bundle_root)
    top_floor = max(floors, key=lambda f: vtuple(f[1]) or (0,), default=None)
    row.upstream = cand or "-"
    local = vtuple(inst)
    notes = []
    if pin and _cmp(local, vtuple(pin)) < 0:
        notes.append(f"below the {pin} that the host tt-metal tree pins")
    if top_floor and _cmp(local, vtuple(top_floor[1])) < 0:
        notes.append(f"below the {top_floor[1]} that the ttnn in the {top_floor[0]} bundle declares and its install.sh enforces")
    behind_apt = bool(cand and _cmp(local, vtuple(cand)) < 0)
    if not cand and not pin and not floors:
        row.why = "no apt candidate, tree pin or bundle floor to compare with"
        return row
    if behind_apt or notes:
        row.status = BEHIND
        row.why = "; ".join(notes) or f"apt has {cand}"
        target = cand or pin or (top_floor[1] if top_floor else "")
        row.proposal = [f"sudo apt install sfpi={target}   # stop every lease holder first; kernels compile with the new compiler "
                        "on the next run, so open a mesh and run one port check afterwards"]
    else:
        row.status = CURRENT
        row.why = "meets the tree pin and every installed bundle floor" if (pin or floors) else ""
    return row


def check_kmd(tools: Tools, host: HostEnv) -> Row:
    row = Row("tt-kmd", "tt-kmd (kernel driver)")
    rc, out, _ = tools.cmd(["modinfo", "-F", "version", "tenstorrent"])
    inst = clean(out.strip(), 40) if rc == 0 else ""
    if not _dotted(inst):
        row.why = "modinfo gave no tenstorrent module version"
        return row
    row.installed = inst
    cand = _policy_candidate(tools, "tenstorrent-dkms")
    if not cand:
        row.why = "apt has no candidate for tenstorrent-dkms"
        return row
    row.upstream = cand
    if _cmp(vtuple(inst), vtuple(cand)) < 0:
        row.status = BEHIND
        row.why = KNOWN_NOTES.get(("tt-kmd", _dotted(cand)), "")
        row.proposal = [f"sudo apt install tenstorrent-dkms={cand}   # builds the module; it loads after the module is reloaded or "
                        "the machine reboots, so stop every lease holder and container first"]
    else:
        row.status = CURRENT
    return row


def _check_pypi_tool(tools: Tools, host: HostEnv, key: str, name: str, label: str) -> Row:
    row = Row(key, label)
    venv = venv_of_script(host, name)
    inst = dist_version(venv, name)
    if inst is None:
        row.why = f"{name} is not installed in a venv this check can read"
        return row
    row.installed = inst
    latest, why = _pypi_latest(tools, name)
    if latest is None:
        row.why = f"PyPI: {why}"
        return row
    row.upstream = latest
    if _cmp(vtuple(inst), vtuple(latest)) < 0:
        row.status = BEHIND
        row.proposal = [f"uv pip install -U {name}   # into the venv that holds it (`{name}` names it on its first line)"]
    else:
        row.status = CURRENT
    return row


def check_flash(tools: Tools, host: HostEnv) -> Row:
    return _check_pypi_tool(tools, host, "tt-flash", "tt-flash", "tt-flash (firmware flasher)")


def check_smi(tools: Tools, host: HostEnv) -> Row:
    return _check_pypi_tool(tools, host, "tt-smi", "tt-smi", "tt-smi")


def latest_firmware(tools: Tools):
    """(tag, date, why) of the newest non-candidate tt-system-firmware release in its public Atom feed. The older tt-firmware
    repository is archived; this is its successor."""
    status, text, why = tools.text(FIRMWARE_FEED)
    if why:
        return None, None, why
    best = None
    for entry in text.split("<entry>")[1:FEED_ENTRIES_MAX]:
        m = FIRMWARE_TAG.search(entry)
        if not m:
            continue
        key = vtuple(m.group(1))
        if key is None:
            continue
        d = UPDATED.search(entry)
        date = updates._date(d.group(1)) if d else None
        if best is None or _cmp(key, best[2]) > 0:
            best = (clean(m.group(1), 40), date, key)
    return (best[0], best[1], "") if best else (None, None, "the feed named no release tag")


def check_firmware(tools: Tools, host: HostEnv) -> Row:
    row = Row("firmware", "firmware bundle (on the chips)")
    import glob
    files = sorted(glob.glob(host.sysfs_glob))[:16]
    seen, why = [], ""
    for f in files:
        v, w = read_sysfs(f, host.sysfs_timeout_s)
        if v and _dotted(v):
            seen.append(clean(v, 40))
        elif w:
            why = w
            break
    if why or not seen:
        row.why = f"not read: {why or 'the driver exposes no firmware bundle file'}"
        return row
    low = min(seen, key=lambda s: vtuple(s) or (0,))
    row.installed = low + (f" (and {len(set(seen)) - 1} other)" if len(set(seen)) > 1 else "")
    tag, date, fwhy = latest_firmware(tools)
    if tag is None:
        row.why = f"tt-system-firmware releases: {fwhy}"
        return row
    row.upstream, row.upstream_date = tag, date
    if _cmp(vtuple(low), vtuple(tag)) < 0:
        row.status = BEHIND
        row.proposal = [f"# read the {tag} release notes and its doc/release migration guide: {FIRMWARE_RELEASES}{tag}",
                        f"tt-flash flash <fw_pack-{tag[1:]}.fwbundle from that release page>   # every device idle; a reset follows"]
    else:
        row.status = CURRENT
        row.why = "the bundle matches the newest tt-system-firmware release (tt-firmware, the older repository, is archived)"
    return row


def check_reference(tools: Tools, host: HostEnv) -> list[Row]:
    rows = []
    venv = host.reference_python.parent.parent if host.reference_python else None
    for name in REFERENCE_PACKAGES:
        row = Row(f"ref-{name}", f"reference venv: {name}")
        rows.append(row)
        inst = dist_version(venv, name)
        if inst is None:
            row.why = "no reference venv given (reference_python in the bringup config) or the package is not in it"
            continue
        row.installed = inst
        latest, why = _pypi_latest(tools, name)
        if latest is None:
            row.why = f"PyPI: {why}"
            continue
        row.upstream = latest
        li, lu = vtuple(inst), vtuple(latest)
        if _cmp(li, lu) >= 0:
            row.status = CURRENT
            continue
        row.status = BEHIND
        if lu[0] > li[0]:
            row.why = f"major: do not auto-upgrade ({li[0]}.x to {lu[0]}.x)"
            row.proposal = [f"# {name} {latest} is a major release: read its changelog and the pins of transformers and tokenizers first"]
        else:
            numerics = " A change moves reference numerics: pin a reference and compare on the copy first." \
                if name in ("torch", "transformers") else ""
            row.why = ("a newer release exists." + numerics).strip()
            row.proposal = [f"uv pip install --python <a COPY of the reference venv> -U {name}   # then compare against a pinned reference "
                            "before the venv in use changes"]
    return rows


def check_host(tools: Tools, host: HostEnv) -> list[Row]:
    """All host-stack rows. A check that raises anything but NotReadOnly becomes an unknown row, as in `updates.run_all`."""
    rows: list[Row] = []
    for key, label, fn in (("sfpi", "sfpi", check_sfpi), ("tt-kmd", "tt-kmd", check_kmd), ("tt-flash", "tt-flash", check_flash),
                           ("tt-smi", "tt-smi", check_smi), ("firmware", "firmware", check_firmware),
                           ("ref-torch", "reference venv", check_reference)):
        try:
            out = fn(tools, host)
            rows += out if isinstance(out, list) else [out]
        except updates.NotReadOnly:
            raise
        except Exception as exc:
            rows.append(Row(key, label, why=f"check failed: {type(exc).__name__}"))
    return rows
