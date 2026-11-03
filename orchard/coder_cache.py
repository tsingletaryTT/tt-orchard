# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Hugging Face cache the coder container mounts, and what the harness can say about it (issue 417).

`tt-model serve` starts the coder with the cache named by `HF_HOME` in the environment it runs in, mounted as `/hf` (seen on 2026-11-03: a
scratch `HF_HOME` appeared as the container's `/hf` mount). The harness's own `--hf-home` is not passed to it. When that cache lacks the
coder's base model, the container's vLLM boot downloads it into the cache, with no line in the container log while it works. A scratch
cache therefore made a coder boot write 94 GB and fill a root disk.

This module holds the parts that need no preflight state: which cache is mounted, how much partial download it holds, and a watch that a
live narrator can ask during a coder boot. The preflight rows are in `orchard/preflight.py` (`check_coder_cache`, `check_coder_container`).
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Callable, Mapping

# Qwen3-Coder-Next in bf16 is 149 GB on disk (2026-11-03). Used for the size of a missing coder model when the hub cannot be asked.
FALLBACK_BASE_GB = 160.0
# Free space wanted beyond the download itself: the container writes its own caches and logs as well.
HEADROOM_GB = 10.0
# A boot is reported as writing to the cache once partial downloads or the fall in free space pass this many decimal GB.
GROWTH_REPORT_GB = 2.0


def effective_cache(environ: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    """The cache `tt-model serve` mounts: `$HF_HOME`, else `~/.cache/huggingface` under `home` (default: the current user's home)."""
    env = os.environ if environ is None else environ
    raw = env.get("HF_HOME")
    if raw:
        return Path(raw)
    return (home if home is not None else Path.home()) / ".cache" / "huggingface"


_SHARD = re.compile(r"^(?P<stem>.+)-(?P<n>\d{5})-of-(?P<total>\d{5})\.safetensors$")


def shard_problem(snap: Path) -> str | None:
    """Why a snapshot directory's weights cannot be all there, or None. A snapshot that holds a config and some shards passes
    `preflight.snapshot_complete` when the weight index has not been downloaded yet, because there is no index to read. A sharded model
    (files named `<stem>-00001-of-00004.safetensors`) needs every shard of each stem and its `<stem>.safetensors.index.json`. This reads
    names only, in the snapshot's top directory. A single-file model has nothing to check here."""
    try:
        names = [e.name for e in os.scandir(snap)]
    except OSError as exc:
        return f"the snapshot directory cannot be read ({type(exc).__name__})"
    groups: dict[tuple[str, int], set[int]] = {}
    for name in names:
        m = _SHARD.match(name)
        if m:
            groups.setdefault((m["stem"], int(m["total"])), set()).add(int(m["n"]))
    for (stem, total), have in sorted(groups.items()):
        if f"{stem}.safetensors.index.json" not in names:
            return f"the weight index {stem}.safetensors.index.json is missing ({len(have)} of {total} shards are there)"
        if have != set(range(1, total + 1)):
            return f"{len(have)} of {total} {stem} shards are there"
    return None


def partial_download_bytes(cache: Path) -> int:
    """Bytes in the `.incomplete` files the Hub client keeps while it downloads, over every repository under `<cache>/hub`. A file that
    vanishes while this reads (the client renames a finished download) counts as zero, and so does a directory that cannot be read."""
    total = 0
    try:
        repos = list(os.scandir(Path(cache) / "hub"))
    except OSError:
        return 0
    for repo in repos:
        try:
            if repo.is_symlink() or not repo.is_dir(follow_symlinks=False):
                continue
            with os.scandir(Path(repo.path) / "blobs") as blobs:
                for f in blobs:
                    if f.name.endswith(".incomplete") and f.is_file(follow_symlinks=False):
                        try:
                            total += f.stat(follow_symlinks=False).st_size
                        except OSError:
                            pass
        except OSError:
            continue
    return total


def _free_bytes(path: Path) -> int | None:
    """Free bytes on the disk that holds `path`. A directory that does not exist yet (a cache the boot is about to create) is measured at
    its nearest existing parent, so the baseline is taken before the first write and a later fall is not read as zero."""
    probe = Path(path)
    for _ in range(64):
        try:
            return shutil.disk_usage(probe).free
        except FileNotFoundError:
            if probe.parent == probe:
                return None
            probe = probe.parent
        except (OSError, ValueError):
            return None
    return None


class BootWatch:
    """What a coder boot has written to the cache since it began. Made when the boot starts, asked once a minute by the narrator.

    Two numbers, because neither is complete: partial downloads (`.incomplete` files; a finished file is renamed and leaves this count) and
    the fall in free space on the cache's disk (it counts every writer on that disk, the coder among them). The report appears when either
    passes `GROWTH_REPORT_GB`."""

    def __init__(self, cache: Path, *, free: Callable[[Path], int | None] = _free_bytes,
                 partial: Callable[[Path], int] = partial_download_bytes):
        self.cache, self._free, self._partial = Path(cache), free, partial
        self._baseline = free(self.cache)

    def report(self) -> str | None:
        partial_gb = self._partial(self.cache) / 1e9
        now = self._free(self.cache)
        fell_gb = 0.0 if now is None or self._baseline is None else max(0.0, (self._baseline - now) / 1e9)
        if partial_gb < GROWTH_REPORT_GB and fell_gb < GROWTH_REPORT_GB:
            return None
        return (f"; the cache {self.cache} holds {partial_gb:.1f} GB of partial downloads and the free space on its disk has fallen "
                f"{fell_gb:.1f} GB since this boot began (other writers count too). A boot that writes this much and prints nothing is "
                "downloading a model into that cache")
