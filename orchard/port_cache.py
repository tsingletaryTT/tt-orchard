# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The converted-weight cache declaration of a stage 2 port (issue 451).

A port's `load(model_dir, mesh_device)` may convert the checkpoint straight onto the chips and write no cache, or it may write a converted
copy under the cache root on the first boot. The disk check before stage 2 runs before the port exists, so it cannot tell which. The port
says so in `stages/2/port/converted_cache.json`:

    {"writes_converted_cache": true, "cache_dir": "<where, optional, 200 characters at most>"}

The supervisor reads it after the coder has written the port and before the hardware test (`Supervisor._check_port_cache`). A port that
declares a cache is held to the disk check for it (block code `disk-full`, with both figures). A port that declares none, or no file at all,
is not blocked on the cache, and the ledger says which of the two it was. The file is read once, through `safeio`, and the sha256 of the
bytes that were read goes into the ledger, so a later edit of the file cannot change what the check used. The declaration is the port's
own statement. It is not a measurement, and the check says nothing about whether the port keeps it.
"""
from __future__ import annotations

import hashlib

from orchard import safeio

FILE_NAME = "converted_cache.json"
MAX_BYTES = 4096                       # a declaration is a few dozen bytes
CACHE_DIR_MAX = 200
KEYS = ("writes_converted_cache", "cache_dir")


def read(port_dir) -> dict:
    """What the port declares: {"state": "writes" | "none" | "absent" | "unreadable", "reason": str | None, "cache_dir": str | None,
    "sha256": str | None}. `absent` is no file. `unreadable` is a file the bounded reader refuses or one that is not the shape above (the
    reason says which); it counts as a declaration of nothing. `sha256` is the hash of the bytes this call read, or None when none were."""
    out = {"state": "absent", "reason": None, "cache_dir": None, "sha256": None}
    path = port_dir / FILE_NAME
    try:
        raw = safeio.read_bytes(path, max_bytes=MAX_BYTES)
    except safeio.SafeReadError as exc:
        if isinstance(exc, safeio.ReadFailed) and "ENOENT" in exc.reason:
            return out
        return {**out, "state": "unreadable", "reason": f"{FILE_NAME} could not be read: {exc.reason}"}
    out["sha256"] = hashlib.sha256(raw).hexdigest()
    try:
        data = safeio.parse_json_bytes(raw, max_bytes=MAX_BYTES, expect=dict)
        extra = sorted(set(data) - set(KEYS))
        if extra:
            raise safeio.BadValue(f"unknown key {extra[0][:40]!r}")
        flag = data.get("writes_converted_cache")
        if not isinstance(flag, bool):
            raise safeio.BadValue("writes_converted_cache must be true or false")
        cache_dir = data.get("cache_dir")
        if cache_dir is not None:
            cache_dir = safeio.bounded_str(cache_dir, max_len=CACHE_DIR_MAX, what="cache_dir")
    except safeio.SafeReadError as exc:
        return {**out, "state": "unreadable", "reason": f"{FILE_NAME} is not a declaration: {exc.reason}"}
    return {**out, "state": "writes" if flag else "none", "cache_dir": cache_dir}
