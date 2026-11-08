# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Who has the Tenstorrent device nodes open right now (issue 560).

`tt-model serve` refuses with "chips already in use by another container" when it finds a claim on the chips it was asked for. It counts
host processes with the device open as well as containers, so the refusal can come from a short ttnn run of another session while
`docker ps` is empty. This module reads what can be seen after such a refusal: a scan of /proc/<pid>/fd for links to /dev/tenstorrent/*,
and `fuser -v` when the program exists. It reads only. It never opens a device node, signals a process or takes a lease.

What it cannot see: processes of other users when the supervisor is not root (their fd directories are unreadable; the count is
reported as `unreadable`). A holder that has already exited is gone, which is itself the finding for a short-lived host process.

A holder counts as a container process when its cgroup path names docker, containerd, libpod or kubepods. The supervisor retries a
refused start once only when no holder is a container process (orchard/supervisor.py).
"""
from __future__ import annotations

import os
import re
import shutil

from orchard.commands import run_command

DEVICE_PREFIX = "/dev/tenstorrent"
MAX_PROCESSES = 200_000        # a bound on the walk of /proc
MAX_HOLDERS = 50               # a bound on the report
MAX_NODES = 64                 # a bound on the device nodes listed and handed to fuser
READ_BYTES = 4096              # /proc/<pid>/comm and cgroup are short; this bounds a hostile one
CONTAINER_CGROUP = re.compile(r"docker|containerd|libpod|kubepods|crio", re.IGNORECASE)
FUSER_LIMIT = 2000


def _read_small(path: str) -> str:
    """Up to READ_BYTES of a /proc file, or "" when it cannot be read. /proc files report size 0, so safeio's size check does not fit them."""
    try:
        with open(path, "rb") as f:                     # hardening: ok a bounded read of a /proc file; safeio cannot size one
            return f.read(READ_BYTES).decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _holder(pid: int, proc_root: str, dev_prefix: str):
    """(devices held, unreadable) for one process: the device nodes its fd links point at."""
    fd_dir = os.path.join(proc_root, str(pid), "fd")
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return (), True                                  # gone, or not ours to read
    devices = set()
    for name in names:
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except OSError:
            continue
        if target == dev_prefix or target.startswith(dev_prefix + "/"):
            devices.add(target)
    return tuple(sorted(devices)), False


def list_nodes(dev_prefix: str, listdir) -> tuple[list[str], str | None]:
    """(device nodes, error): the numbered entries of the device directory as full paths, at most MAX_NODES, in numeric order. `fuser` does
    not look inside a directory, so the nodes are named one by one. A listing that fails gives no nodes and says why."""
    try:
        names = [n for n in listdir(dev_prefix) if isinstance(n, str) and n.isdigit()]
    except (OSError, ValueError) as exc:
        return [], f"cannot list {dev_prefix}: {type(exc).__name__}"
    return [f"{dev_prefix}/{n}" for n in sorted(names, key=int)[:MAX_NODES]], None


def scan(*, proc_root: str = "/proc", dev_prefix: str = DEVICE_PREFIX, which=shutil.which, run=run_command, timeout: float = 10.0,
         listdir=os.listdir) -> dict:
    """What holds the device nodes: {"holders": [{pid, comm, devices, container}], "unreadable": n, "fuser": {...} | None, "truncated": bool,
    "nodes": [...], "nodes_error": str | None}."""
    holders, unreadable, seen, truncated = [], 0, 0, False
    nodes, nodes_error = list_nodes(dev_prefix, listdir)
    try:
        entries = os.listdir(proc_root)
    except OSError as exc:
        return {"holders": [], "unreadable": 0, "fuser": None, "truncated": False, "nodes": nodes, "nodes_error": nodes_error,
                "error": f"cannot read {proc_root}: {type(exc).__name__}"}
    for entry in entries:
        if not entry.isdigit():
            continue
        seen += 1
        if seen > MAX_PROCESSES:
            truncated = True
            break
        devices, denied = _holder(int(entry), proc_root, dev_prefix)
        unreadable += denied
        if not devices:
            continue
        if len(holders) >= MAX_HOLDERS:
            truncated = True
            break
        cgroup = _read_small(os.path.join(proc_root, entry, "cgroup"))
        holders.append({"pid": int(entry), "comm": _read_small(os.path.join(proc_root, entry, "comm")).strip()[:64],
                        "devices": list(devices), "container": bool(CONTAINER_CGROUP.search(cgroup))})
    fuser = None
    exe = which("fuser")
    if exe and not nodes:
        # Never the directory: fuser would find nothing in it and the report would look like "no holder".
        fuser = {"skipped": nodes_error or f"no numbered device node under {dev_prefix}"}
    elif exe:
        # fuser -v prints its table on stderr and exits 1 when nothing holds the files; both are kept as they are.
        try:
            res = run([exe, "-v", *nodes], timeout)
            fuser = {"argv": list(res.argv), "returncode": res.returncode,
                     "output": f"{res.stdout}\n{res.stderr}".strip()[:FUSER_LIMIT]}
        except OSError as exc:
            fuser = {"error": f"{type(exc).__name__}"}
    return {"holders": sorted(holders, key=lambda h: h["pid"]), "unreadable": unreadable, "fuser": fuser, "truncated": truncated,
            "nodes": nodes, "nodes_error": nodes_error}


def describe(found: dict) -> str:
    """One bounded sentence for a block text."""
    holders = found.get("holders") or []
    if not holders:
        return ("no process visible to the supervisor had a device node open when it looked"
                + (f" ({found['unreadable']} processes could not be read)" if found.get("unreadable") else ""))
    parts = [f"pid {h['pid']} ({h['comm'] or '?'}, {'container' if h['container'] else 'host process'})" for h in holders[:8]]
    return "device nodes were open in " + ", ".join(parts) + (f" and {len(holders) - 8} more" if len(holders) > 8 else "")


def _node_index(device: str) -> int | None:
    tail = str(device).rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None


def any_container(found: dict, dev_indices=None) -> bool:
    """Whether a container process holds a device node. With `dev_indices` (the leased chips' device numbers) only holders of those chips
    count; a holder whose node is not a chip number counts for every lease, because nothing says it is elsewhere (issue 587)."""
    wanted = None if dev_indices is None else {int(i) for i in dev_indices}
    for h in found.get("holders") or []:
        if not h.get("container"):
            continue
        if wanted is None:
            return True
        indices = [_node_index(d) for d in h.get("devices") or []]
        if not indices or any(i is None or i in wanted for i in indices):
            return True
    return False
