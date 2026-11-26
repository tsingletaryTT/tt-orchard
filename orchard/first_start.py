# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The first-start check of a published package (issue 843).

Why. The first start of `raahemnabeel/hy4-flash` (a 220B mixture-of-experts model on four Blackhole chips) took a 107 min 43 s download of a 440 GB checkpoint,
then 24 min 41 s to ready while it built about 165 GB of caches, and it needed 17 GB of host RAM to start and 116 GB for full speed. The package card
states the disk, the RAM and the pinned-memory option, and the manifest sets the ready timeout (21600 s). A run that does not read the card finds
out at hour two. `check_disk` in orchard/preflight.py counts the download and a margin, and knows nothing of the caches a first start builds.

What it does. `tt-orchard first-start PACKAGE_DIR` (or `--card` and `--manifest`) reads the card text and the manifest and prints five rows in the style of the
preflight rows: disk (the checkpoint when it is not on disk, the caches, the margin, and what is left after the first start), host RAM, pinned memory,
download time and the ready timeout. It blocks only when a figure the card states is violated: the disk figures, the minimum RAM to start, and a ready
timeout shorter than a first-start time the card states. It warns when RAM is below the full-speed figure, when the card advises a pinned-memory option, and
when MemAvailable cannot be read. A figure the card does not state is a row that says `not stated`; it never blocks.

What each figure is. Every figure carries the card line it came from (`card line 11: ...`) or `not stated`. The card is free text: the readers look
for a few fixed wordings (see the patterns below) and miss others; a miss is `not stated`, never a guess. The link speed is an argument (`--link-mbps`): the
download time is `checkpoint size / rate`, and without a rate the row says so. Nothing is downloaded and no network is used.

The card and the manifest are outside files: they are read with `safeio` (regular files only, size caps, no links), the card's lines and the number
of lines read are bounded, and every figure is bounded (a figure above MAX_FIGURE is no figure). The command opens no device and takes no lease.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import NamedTuple

from orchard import safeio
from orchard.defaults import TEST_DISK_GB
from orchard.preflight import BLOCK, OK, WARN, Check

EXIT_OK, EXIT_REFUSED, EXIT_BLOCKED = 0, 2, 5       # the supervisor's own codes (orchard/cli.py)
NOT_STATED = "not stated"
CARD_MAX_BYTES = 1 << 20                            # a package card is a few KB; this bounds a hostile one
MANIFEST_MAX_BYTES = 4 << 20                        # a manifest is tens of KB
MAX_LINES, MAX_LINE = 20_000, 2_000                 # lines read, and the longest line read: a longer one is skipped
MAX_FIGURE = 100_000.0                              # GB or minutes: a figure above this is no figure (a hostile card cannot make a huge integer)
MAX_TIMEOUT_S = 10_000_000.0
QUOTE_CHARS = 160
MEMLOCK_DEFAULT = "8 MiB"                           # the Docker default the first start of the Hy4-flash container ran with (its mlock of the table failed)

_N = r"(\d{1,6}(?:\.\d{1,3})?)"                    # a bounded number: the patterns below cannot backtrack on a long run of digits
_RE_DISK = re.compile(rf"about\s{{1,3}}{_N}\s{{0,3}}GB\s{{1,3}}of\s{{1,3}}(?:free\s{{1,3}})?disk", re.I)
_RE_SPLIT = re.compile(rf"the\s{{1,3}}{_N}\s{{0,3}}GB\s{{1,3}}checkpoint\s{{1,3}}plus\s{{1,3}}about\s{{1,3}}{_N}\s{{0,3}}GB\s{{1,3}}of\s{{1,3}}caches", re.I)
_RE_RAM = re.compile(rf"about\s{{1,3}}{_N}\s{{0,3}}GB\s{{1,3}}of\s{{1,3}}free\s{{1,3}}host\s{{1,3}}RAM", re.I)
_RE_RAM_MIN = re.compile(rf"\({_N}\s{{0,3}}GB\s{{1,3}}minimum\)", re.I)
_RE_MEMLOCK = re.compile(r"--ulimit\s{1,3}memlock=", re.I)
_RE_LATER = re.compile(rf"later\s{{1,3}}starts?\s{{1,3}}(?:are\s{{1,3}}ready\s{{1,3}}in|take)\s{{1,3}}about\s{{1,3}}{_N}\s{{0,3}}(minutes?|hours?)", re.I)
_RE_FIRST = re.compile(rf"first\s{{1,3}}start\s{{1,3}}(?:takes|is\s{{1,3}}ready\s{{1,3}}in|needs)\s{{1,3}}(?:about\s{{1,3}})?{_N}\s{{0,3}}(minutes?|hours?)", re.I)
_HUB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
TIMEOUT_KEY = "VLLM_ENGINE_READY_TIMEOUT_S"
FIGURES = ("disk_total_gb", "checkpoint_gb", "cache_gb", "ram_full_gb", "ram_min_gb", "memlock_advised", "later_start_minutes", "first_start_minutes")


class Figure(NamedTuple):
    value: float | None
    source: str                                     # "card line N: <the line>", "manifest <path>", or NOT_STATED


def _absent() -> Figure:
    return Figure(None, NOT_STATED)


def _bounded(text: str, limit: float = MAX_FIGURE) -> float | None:
    try:
        v = float(text)  # finite: ok a string, never an int: a very long digit string becomes inf here and fits_float refuses it
    except (TypeError, ValueError, OverflowError):
        return None
    return v if safeio.fits_float(v) and 0 < v <= limit else None


def _minutes(number: str, unit: str) -> float | None:
    v = _bounded(number)
    if v is None:
        return None
    v = v * 60.0 if unit.lower().startswith("hour") else v
    return v if v <= MAX_FIGURE else None


def parse_card(text: str) -> dict[str, Figure]:
    """The figures of a package card, each with the line it came from. A figure the card does not state, or states in a wording these patterns do not read,
    or states above MAX_FIGURE, is `Figure(None, NOT_STATED)`. The first line that states a figure wins."""
    out = {k: _absent() for k in FIGURES}
    if not isinstance(text, str):
        return out
    for n, line in enumerate(text.splitlines()[:MAX_LINES], start=1):
        if len(line) > MAX_LINE:
            continue
        src = f"card line {n}: {line.strip()[:QUOTE_CHARS]}"
        m = _RE_DISK.search(line)
        if m and out["disk_total_gb"].value is None and (v := _bounded(m.group(1))) is not None:
            out["disk_total_gb"] = Figure(v, src)
            s = _RE_SPLIT.search(line)
            if s and (a := _bounded(s.group(1))) is not None and (b := _bounded(s.group(2))) is not None:
                out["checkpoint_gb"], out["cache_gb"] = Figure(a, src), Figure(b, src)
        m = _RE_RAM.search(line)
        if m and out["ram_full_gb"].value is None and (v := _bounded(m.group(1))) is not None:
            out["ram_full_gb"] = Figure(v, src)
            r = _RE_RAM_MIN.search(line)
            if r and (lo := _bounded(r.group(1))) is not None:
                out["ram_min_gb"] = Figure(lo, src)
        if out["memlock_advised"].value is None and _RE_MEMLOCK.search(line):
            out["memlock_advised"] = Figure(1.0, src)
        m = _RE_LATER.search(line)
        if m and out["later_start_minutes"].value is None and (v := _minutes(m.group(1), m.group(2))) is not None:
            out["later_start_minutes"] = Figure(v, src)
        m = _RE_FIRST.search(line)
        if m and out["first_start_minutes"].value is None and (v := _minutes(m.group(1), m.group(2))) is not None:
            out["first_start_minutes"] = Figure(v, src)
    return out


def _dig(node, path):
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def manifest_figures(manifest) -> dict[str, Figure]:
    """The figures of a package manifest: the ready timeout in seconds, from `container.serve.env`. Not a mapping, a missing key and a value that is not a
    finite number above zero are `Figure(None, NOT_STATED)`."""
    raw = _dig(manifest, ("container", "serve", "env", TIMEOUT_KEY))
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return {"ready_timeout_s": _absent()}
    v = _bounded(str(raw), MAX_TIMEOUT_S)
    return {"ready_timeout_s": Figure(v, f"manifest container.serve.env.{TIMEOUT_KEY}") if v is not None else _absent()}


def manifest_weights(manifest) -> tuple[str, str] | None:
    """(repo id, 40-character revision) of the weights the manifest pins, or None."""
    w = _dig(manifest, ("weights",))
    if not isinstance(w, dict):
        return None
    repo, rev = w.get("repo_id"), w.get("revision")
    if isinstance(repo, str) and isinstance(rev, str) and _HUB_ID.match(repo) and _REVISION.match(rev):
        return repo, rev
    return None


def snapshot_present(hf_home, repo: str, revision: str) -> bool:
    """Whether the hub cache under `hf_home` holds a non-empty snapshot of `repo` at `revision`. A repo id or revision that is not well formed is no."""
    if not (isinstance(repo, str) and isinstance(revision, str) and _HUB_ID.match(repo) and _REVISION.match(revision)):
        return False
    snap = Path(hf_home) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    try:
        with os.scandir(snap) as it:
            return next(it, None) is not None
    except OSError:
        return False


def _quote(fig: Figure) -> str:
    return fig.source if fig.value is not None else NOT_STATED


def _hours(seconds: float) -> str:
    return f"{seconds / 3600:g} h"


def _disk_row(card, *, free_gb, same_device, snapshot_local, min_free_gb) -> Check:
    ck, cache, total = card["checkpoint_gb"].value, card["cache_gb"].value, card["disk_total_gb"].value
    if ck is not None and cache is not None:
        download, caches = (0.0 if snapshot_local else ck), cache
        taken_text = (f"{download + caches:.0f} GB ({ck:.0f} GB checkpoint + {cache:.0f} GB caches)" if not snapshot_local
                      else f"{caches:.0f} GB of caches (the checkpoint is on disk)")
    elif total is not None:
        download, caches = max(total - (ck or 0.0), 0.0) if snapshot_local and ck else total, 0.0
        taken_text = f"{download:.0f} GB (the card's total; its split is {NOT_STATED})"
    else:
        return Check("first-start disk", OK, f"{NOT_STATED}: the card names no disk figure for the first start")
    if same_device:
        need = {"hf_home": download + caches + min_free_gb}
    else:
        need = {"hf_home": download + min_free_gb, "cache_root": caches + min_free_gb}
    short = [f"{label}: needs {need[label]:.0f} GB, {free_gb.get(label, 0.0):.0f} GB free" for label in need if free_gb.get(label, 0.0) < need[label]]
    stated = _quote(card["disk_total_gb"])
    if short:
        return Check("first-start disk", BLOCK, "; ".join(short) + f". The first start takes {taken_text} and the margin is {min_free_gb:.0f} GB ({stated})",
                     "disk-full", {"short_gb": {k: round(need[k] - free_gb.get(k, 0.0), 3) for k in need if free_gb.get(k, 0.0) < need[k]}, "same_device": same_device})
    free = min(free_gb.get(k, 0.0) for k in need) if not same_device else free_gb.get("hf_home", 0.0)
    after = free - (download + caches) if same_device else min(free_gb.get("hf_home", 0.0) - download, free_gb.get("cache_root", 0.0) - caches)
    return Check("first-start disk", OK, f"{free:.0f} GB free, needs {sum(need.values()) if same_device else max(need.values()):.0f} GB; the first start takes {taken_text} and "
                                         f"leaves {after:.0f} GB free after the first start ({stated})")


def _ram_row(card, mem_available_gb) -> Check:
    lo, full = card["ram_min_gb"], card["ram_full_gb"]
    if lo.value is None and full.value is None:
        return Check("first-start host ram", OK, f"{NOT_STATED}: the card names no host RAM figure")
    stated = _quote(full if full.value is not None else lo)
    if mem_available_gb is None:
        return Check("first-start host ram", WARN, f"MemAvailable could not be read; the card states {stated}")
    if lo.value is not None and mem_available_gb < lo.value:
        return Check("first-start host ram", BLOCK, f"needs {lo.value:.0f} GB to start, {mem_available_gb:.0f} GB available ({stated})", "model-does-not-fit")
    if full.value is not None and mem_available_gb < full.value:
        start = f"{lo.value:.0f} GB to start, " if lo.value is not None else ""
        return Check("first-start host ram", WARN, f"{mem_available_gb:.0f} GB available; the card states {start}{full.value:.0f} GB for full speed: with less the server starts "
                                                  f"but decodes slower ({stated})")
    return Check("first-start host ram", OK, f"{mem_available_gb:.0f} GB available ({stated})")


def _memlock_row(card) -> Check:
    f = card["memlock_advised"]
    if f.value is None:
        return Check("first-start memlock", OK, f"{NOT_STATED}: the card says nothing about a pinned-memory limit")
    return Check("first-start memlock", WARN, f"the card advises `--ulimit memlock=-1:-1` ({f.source}). Without it the container runs with the Docker default ({MEMLOCK_DEFAULT}): "
                                              "the pinned load of a large host table fails and the table can be swapped out under memory pressure")


def _download_row(card, snapshot_local, link_mb_s) -> Check:
    ck = card["checkpoint_gb"].value
    if snapshot_local:
        return Check("first-start download", OK, "the checkpoint is on disk: no download")
    if ck is None:
        return Check("first-start download", OK, f"checkpoint size {NOT_STATED}")
    rate = link_mb_s if safeio.fits_float(link_mb_s) and link_mb_s > 0 else None
    if rate is None:
        return Check("first-start download", OK, f"{ck:.0f} GB to download; link speed not given (--link-mbps), so no time is estimated")
    minutes = ck * 1000.0 / rate / 60.0
    return Check("first-start download", OK, f"{ck:.0f} GB at {rate:g} MB/s takes about {minutes:.0f} minutes")


def _timeout_row(card, manifest_figs, ready_timeout_s) -> Check:
    mf = manifest_figs.get("ready_timeout_s", _absent())
    eff, where = (ready_timeout_s, "given") if ready_timeout_s is not None else (mf.value, mf.source)
    first = card["first_start_minutes"]
    if eff is None:
        return Check("first-start ready timeout", OK, "not set in the manifest: the serve step's default applies" + (f"; the card states a first start of about {first.value:.0f} minutes" if first.value else ""))
    if first.value is not None and eff < first.value * 60.0:
        return Check("first-start ready timeout", BLOCK, f"the ready timeout is {eff:.0f} s ({_hours(eff)}) and the card states a first start of about {first.value:.0f} minutes ({first.source})", "blocked")
    tail = f"the card states a first start of about {first.value:.0f} minutes" if first.value is not None else "the card states no first-start time"
    return Check("first-start ready timeout", OK, f"{eff:.0f} s ({_hours(eff)}, {where}); {tail}")


def check_first_start(card: dict, manifest_figs: dict, *, free_gb: dict, same_device: bool, snapshot_local: bool, mem_available_gb, link_mb_s=None,
                      ready_timeout_s=None, min_free_gb: float = TEST_DISK_GB) -> list[Check]:
    """The five rows, in a fixed order. `free_gb` has the keys `hf_home` and `cache_root`. On one disk the checkpoint, the caches and one margin are needed;
    on two disks each needs its part and the margin. The caches are the card's figure; a snapshot already on disk is not downloaded again."""
    return [_disk_row(card, free_gb=free_gb, same_device=same_device, snapshot_local=snapshot_local, min_free_gb=min_free_gb),
            _ram_row(card, mem_available_gb), _memlock_row(card), _download_row(card, snapshot_local, link_mb_s),
            _timeout_row(card, manifest_figs, ready_timeout_s)]


# ---- the command --------------------------------------------------------------------------------------------------------------------

def parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=add_help, description="check a package's first start against the card and the manifest: disk, host RAM, pinned memory, download time and "
                                                                "the ready timeout (nothing is downloaded)")
    p.add_argument("package", nargs="?", help="the package directory (README.md and tt_kernel_manifest.json in it)")
    p.add_argument("--card", metavar="FILE", help="the package card (default PACKAGE/README.md)")
    p.add_argument("--manifest", metavar="FILE", help="the manifest (default PACKAGE/tt_kernel_manifest.json; a missing one is not an error)")
    p.add_argument("--hf-home", metavar="DIR", help="the Hugging Face home (default $HF_HOME or ~/.cache/huggingface): where the checkpoint is stored")
    p.add_argument("--cache-root", metavar="DIR", help="where the first start builds its caches (default ~/.cache/tt-model)")
    p.add_argument("--link-mbps", metavar="MB_S", help="the download rate in MB/s (a measured one): without it no download time is estimated")
    p.add_argument("--ready-timeout-s", metavar="S", help="the ready timeout the serve step will use (default: the manifest's)")
    p.add_argument("--min-free-gb", metavar="GB", help=f"the free-space margin kept after the first start (default {TEST_DISK_GB:g})")
    return p


def _refuse(out, message: str) -> int:
    print(f"tt-orchard first-start: {message}", file=out)
    return EXIT_REFUSED


def _number(text, flag, out, *, positive=True):
    if text is None:
        return None, None
    try:
        v = float(text)  # finite: ok a command-line string, never an int
    except ValueError:
        return None, _refuse(out, f"{flag} must be a number")
    if not safeio.fits_float(v) or (v <= 0 if positive else v < 0):
        return None, _refuse(out, f"{flag} must be a finite number above {'0' if positive else 'or equal to 0'}")
    return v, None


def default_facts(hf_home: Path, cache_root: Path, weights) -> dict:
    """The machine's figures for the five rows: free GB on each disk (the nearest existing parent of a path that is not made yet), whether the two paths share
    a disk, whether the pinned snapshot is in the hub cache, and MemAvailable in GB."""
    from orchard import residency
    free_h, mem = residency.machine_facts(hf_home)
    free_c, _ = residency.machine_facts(cache_root)

    def dev(path: Path):
        probe = path
        for _ in range(64):
            try:
                return os.stat(probe).st_dev
            except OSError:
                if probe.parent == probe:
                    return None
                probe = probe.parent
        return None
    d1, d2 = dev(hf_home), dev(cache_root)
    return {"free_gb": {"hf_home": (free_h or 0) / 1e9, "cache_root": (free_c or 0) / 1e9}, "same_device": d1 is not None and d1 == d2,
            "snapshot_local": bool(weights) and snapshot_present(hf_home, *weights), "mem_available_gb": None if mem is None else mem / 1e9}


def read_package(card_path, manifest_path):
    """(the card text, the manifest or None). The card must be there and readable; a manifest that is not there is None, and one that cannot be read safely is a
    refusal. Raises safeio.SafeReadError with a plain reason that holds no file content."""
    card = safeio.read_text_checked(card_path, max_bytes=CARD_MAX_BYTES, errors="replace")
    if card.missing:
        raise safeio.ReadFailed("the package card is not there")
    if card.refused:
        raise safeio.ReadFailed("the package card is not readable: " + card.reason)
    manifest = None
    if manifest_path is not None:
        m = safeio.read_json_checked(manifest_path, max_bytes=MANIFEST_MAX_BYTES, expect=dict)
        if m.refused:
            raise safeio.ReadFailed("the manifest is not readable: " + m.reason)
        manifest = None if m.missing else m.value
    return card.value, manifest


def main(args, *, out, facts=None) -> int:
    link, code = _number(args.link_mbps, "--link-mbps", out)
    if code is not None:
        return code
    timeout, code = _number(args.ready_timeout_s, "--ready-timeout-s", out)
    if code is not None:
        return code
    margin, code = _number(args.min_free_gb, "--min-free-gb", out, positive=False)
    if code is not None:
        return code
    pkg = Path(args.package) if args.package else None
    card_path = Path(args.card) if args.card else (pkg / "README.md" if pkg else None)
    if card_path is None:
        return _refuse(out, "give a package directory or --card")
    manifest_path = Path(args.manifest) if args.manifest else (pkg / "tt_kernel_manifest.json" if pkg else None)
    try:
        card_text, manifest = read_package(card_path, manifest_path)
    except safeio.SafeReadError as exc:
        return _refuse(out, exc.reason)
    hf_home = Path(os.path.expanduser(args.hf_home or os.environ.get("HF_HOME") or "~/.cache/huggingface"))
    cache_root = Path(os.path.expanduser(args.cache_root or "~/.cache/tt-model"))
    weights = manifest_weights(manifest)
    machine = (facts or (lambda h, c: default_facts(h, c, weights)))(hf_home, cache_root)
    rows = check_first_start(parse_card(card_text), manifest_figures(manifest), free_gb=machine["free_gb"], same_device=machine["same_device"],
                             snapshot_local=machine["snapshot_local"], mem_available_gb=machine["mem_available_gb"], link_mb_s=link, ready_timeout_s=timeout,
                             min_free_gb=TEST_DISK_GB if margin is None else margin)
    print(f"tt-orchard first-start: {pkg or card_path}", file=out)
    for c in rows:
        print(f"  [{c.status}] {c.name}: {c.detail}" + (f" ({c.reason})" if c.status == BLOCK and c.reason else ""), file=out)
    return EXIT_BLOCKED if any(c.status == BLOCK for c in rows) else EXIT_OK
