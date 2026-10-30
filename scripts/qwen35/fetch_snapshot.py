#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Download a pinned Hugging Face snapshot into the normal cache layout with parallel byte-range reads.

`hf download` of Qwen/Qwen3.5-35B-A3B ran at 4 to 6 MB/s on this machine; ranged reads of the same files ran at 35 to 50 MB/s
(docs/run-logs/2026-10-29-qwen35-package.md). The result is the same layout `hf download` writes: `blobs/<sha256>` files and
`snapshots/<revision>/<name>` links to them, so `snapshot_download(repo, revision=...)` and prepare_model_dir.py find it.
Each large file is checked against the sha256 the Hub lists for it.

    python3 scripts/qwen35/fetch_snapshot.py Qwen/Qwen3.5-35B-A3B 59d61f3ce65a6d9863b86d2e96597125219dc754

The cache is $HF_HUB_CACHE, or $HF_HOME/hub, or ~/.cache/huggingface/hub. A file already in the cache with the right size and hash is kept.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import sys
import time
from pathlib import Path

CHUNK = 64 * 1024 * 1024
WORKERS = 16
RETRIES = 6


def hub_dir() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_range(fs, remote: str, fd: int, lo: int, hi: int) -> int:
    for attempt in range(RETRIES):
        try:
            with fs.open(remote, "rb") as f:
                f.seek(lo)
                buf = f.read(hi - lo)
            if len(buf) != hi - lo:
                raise OSError(f"short read {len(buf)} of {hi - lo}")
            os.pwrite(fd, buf, lo)
            return hi - lo
        except Exception as exc:  # noqa: BLE001  network errors come in many types; retry them all, then fail loudly
            if attempt == RETRIES - 1:
                raise
            print(f"retry {attempt + 1} for {remote} [{lo}:{hi}]: {type(exc).__name__}", file=sys.stderr, flush=True)
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def main(argv) -> int:
    if len(argv) != 3 or len(argv[2]) != 40:
        print(__doc__)
        return 2
    repo, rev = argv[1], argv[2]
    from huggingface_hub import HfApi, HfFileSystem
    api, fs = HfApi(), HfFileSystem()
    org, name = repo.split("/", 1)
    root = hub_dir() / f"models--{org}--{name}"
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    snap = root / "snapshots" / rev
    snap.mkdir(parents=True, exist_ok=True)
    (root / "refs").mkdir(exist_ok=True)
    infos = api.model_info(repo, revision=rev, files_metadata=True).siblings
    total = sum(s.size or 0 for s in infos)
    done = 0
    t0 = time.time()
    for s in infos:
        remote = f"{repo}@{rev}/{s.rfilename}"
        oid = (s.lfs.sha256 if s.lfs else s.blob_id)
        blob = root / "blobs" / oid
        link = snap / s.rfilename
        link.parent.mkdir(parents=True, exist_ok=True)
        if not (blob.is_file() and blob.stat().st_size == s.size):
            part = root / "blobs" / f"{oid}.ranged"
            fd = os.open(part, os.O_RDWR | os.O_CREAT)
            try:
                os.ftruncate(fd, s.size)
                spans = [(lo, min(lo + CHUNK, s.size)) for lo in range(0, s.size, CHUNK)]
                with cf.ThreadPoolExecutor(WORKERS) as ex:
                    for n in ex.map(lambda sp: fetch_range(fs, remote, fd, *sp), spans):
                        done += n
                        print(f"{done / 1e9:.1f} / {total / 1e9:.1f} GB  {s.rfilename}  {done / 1e6 / max(time.time() - t0, 1):.0f} MB/s", flush=True)
            finally:
                os.close(fd)
            if s.lfs and file_sha256(part) != s.lfs.sha256:
                raise SystemExit(f"{s.rfilename}: sha256 differs from the Hub's {s.lfs.sha256}")
            os.replace(part, blob)
        else:
            done += s.size or 0
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(os.path.relpath(blob, link.parent))
    (root / "refs" / "main").write_text(rev)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
