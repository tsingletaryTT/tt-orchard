# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard caches`: the tensor caches under the cache root, here and on the lab, and which can go.

A tensor cache is a directory with the `.orchard-model` marker the swap templates write (it names the model
whose weights it holds). One is "used" while a run that is not finished or aborted names it as a
`tt_cache` in a `swap_config.json`. `--prune --older-than DAYS` removes the caches that are not used and
have not been written for that long, after printing them and asking (`--yes` skips the question). The
shared kernel cache (`<cache_root>/kernels`) is listed and never pruned. Nothing outside the cache root is
ever removed.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from orchard.ledger import LedgerCorrupt, read_entries
from orchard.setup_machine import _run
from orchard.stages import run_progress

MARKER = ".orchard-model"
KERNELS = "kernels"
KERNEL_MODEL = "(shared kernel cache)"


@dataclass
class Cache:
    path: str
    model: str | None
    gb: float
    last_write: float
    box: str = "here"
    used_by: list[str] = field(default_factory=list)


def cache_root_of(cfg) -> Path:
    """The run's default when the config names none: <runs_root>/../cache, as orchard.paths resolves it."""
    return Path(cfg.cache_root) if cfg.cache_root else Path(cfg.runs_root) / "cache"


def used_caches(runs_root) -> dict[str, list[str]]:
    """realpath of a tensor cache -> the runs (not finished, not aborted) whose swap configs name it."""
    out: dict[str, list[str]] = {}
    for ledger in sorted(Path(runs_root).glob("*/ledger.jsonl")):
        try:
            p = run_progress(read_entries(ledger))
        except (OSError, ValueError, LedgerCorrupt):
            continue
        if p.finished or p.aborted:
            continue
        run = ledger.parent
        for sc in run.glob("stages/**/swap_config.json"):
            try:
                cache = json.loads(sc.read_text(encoding="utf-8")).get("tt_cache")
            except (OSError, ValueError, AttributeError):
                continue
            if isinstance(cache, str):
                names = out.setdefault(os.path.realpath(cache), [])
                if run.name not in names:
                    names.append(run.name)
    return out


def _measure(d: Path) -> tuple[float, float]:
    total, newest = 0, 0.0
    for top, _, files in os.walk(d):
        for f in files:
            try:
                st = os.lstat(os.path.join(top, f))
            except OSError:
                continue
            total += st.st_size
            newest = max(newest, st.st_mtime)
    return round(total / 1e9, 1), newest


def local_caches(cache_root) -> list[Cache]:
    root = Path(cache_root)
    out = []
    if not root.is_dir():
        return out
    for top, dirs, files in os.walk(root):
        here = Path(top)
        if here == root / KERNELS:
            dirs[:] = []
            gb, newest = _measure(here)
            out.append(Cache(str(here), KERNEL_MODEL, gb, newest))
            continue
        if MARKER in files:
            dirs[:] = []                                # a cache's own files are not searched for caches
            gb, newest = _measure(here)
            out.append(Cache(str(here), (here / MARKER).read_text(encoding="utf-8", errors="replace").strip(), gb, newest))
    return out


LAB_SCRIPT = r"""cd {root} 2>/dev/null || exit 0
for d in $(find . -maxdepth 5 -name {marker} -printf '%h\n') {kernels}; do
  [ -d "$d" ] || continue
  m=$(cat "$d/{marker}" 2>/dev/null || echo "{kernel_model}")
  k=$(du -sk "$d" | cut -f1)
  t=$(find "$d" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1)
  printf '%s\t%s\t%s\t%s\n' "$d" "$m" "$k" "${{t:-0}}"
done"""


def lab_caches(lab, cache_root, run=_run) -> list[Cache]:
    from orchard.lab_setup import remote
    script = LAB_SCRIPT.format(root=shlex.quote(str(cache_root)), marker=MARKER, kernels=KERNELS,
                               kernel_model=KERNEL_MODEL)
    rc, out, err = run(remote(lab, script), 600)
    if rc != 0:
        raise RuntimeError(f"listing the caches on {lab.host} failed: {(err or out).strip()[-200:]}")
    caches = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        rel, model, kb, t = parts
        path = os.path.normpath(os.path.join(str(cache_root), rel))
        caches.append(Cache(path, model, round(int(kb or 0) * 1024 / 1e9, 1), float(t or 0), box=lab.host))
    return caches


def prunable(caches: list[Cache], older_than_days: float, now: float) -> list[Cache]:
    return [c for c in caches if not c.used_by and c.model != KERNEL_MODEL
            and now - c.last_write > older_than_days * 86400]


def _under(path: str, root) -> bool:
    real, base = os.path.realpath(path), os.path.realpath(str(root))
    return real != base and os.path.commonpath([real, base]) == base


def render(caches: list[Cache], now: float) -> list[str]:
    lines = [f"{'box':<8} {'GB':>7} {'days':>5}  {'used by':<24} model / path"]
    for c in sorted(caches, key=lambda c: (c.box, -c.gb)):
        days = (now - c.last_write) / 86400 if c.last_write else float("nan")
        lines.append(f"{c.box:<8} {c.gb:>7.1f} {days:>5.0f}  {(', '.join(c.used_by) or '-'):<24} "
                     f"{c.model or '?'}\n{'':<48}{c.path}")
    lines.append(f"total {sum(c.gb for c in caches):.1f} GB in {len(caches)} cache(s)")
    return lines


def parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tt-orchard caches", description=__doc__.split("\n\n")[0], add_help=add_help)
    p.add_argument("--lab", action="store_true", help="also list the caches on the [lab] box")
    p.add_argument("--prune", action="store_true", help="remove the caches no unfinished run uses (asks first)")
    p.add_argument("--older-than", type=float, metavar="DAYS", help="with --prune: only caches not written for DAYS")
    p.add_argument("--yes", action="store_true", help="with --prune: do not ask")
    return p


def main(argv=None, *, cfg, run=_run, say=print, ask=None, now=time.time, args=None) -> int:
    args = args if args is not None else parser().parse_args(argv)
    if args.prune and args.older_than is None:
        say("refused: --prune needs --older-than DAYS")
        return 2
    if args.lab and cfg.lab is None:
        say("refused: --lab needs a [lab] table in the config")
        return 2
    t = now()
    root = cache_root_of(cfg)
    used = used_caches(cfg.runs_root)
    caches = local_caches(root)
    if args.lab:
        try:
            caches += lab_caches(cfg.lab, root, run=run)
        except RuntimeError as exc:
            say(f"refused: {exc}")
            return 2
    for c in caches:
        c.used_by = used.get(os.path.realpath(c.path), [])
    for line in render(caches, t):
        say(line)
    if not args.prune:
        return 0
    gone = [c for c in prunable(caches, args.older_than, t) if _under(c.path, root)]
    if not gone:
        say(f"nothing to prune: every cache is used or newer than {args.older_than:g} days")
        return 0
    say(f"would remove {len(gone)} cache(s), {sum(c.gb for c in gone):.1f} GB:")
    for c in gone:
        say(f"  {c.box}: {c.path}")
    ask = ask or (lambda q: sys.stdin.isatty() and input(f"{q} [y/N] ").strip().lower() in ("y", "yes"))
    if not (args.yes or ask("remove them?")):
        say("nothing removed")
        return 0
    for c in gone:
        if c.box == "here":
            shutil.rmtree(c.path)
        else:
            from orchard.lab_setup import remote
            rc, out, err = run(remote(cfg.lab, "rm -rf -- " + shlex.quote(c.path)), 600)
            if rc != 0:
                say(f"failed to remove {c.path} on {c.box}: {(err or out).strip()[-200:]}")
                return 1
        say(f"removed {c.box}: {c.path}")
    return 0
