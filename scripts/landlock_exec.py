#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run a command under a Landlock filesystem and network restriction (Linux 5.13 and later, no root, no user namespace).

    python3 landlock_exec.py SPEC.json

SPEC.json: {"ro": [paths], "rw": [paths], "rw_files": [paths], "connect_ports": [ports], "bind_ports": [ports], "env": {...},
"chdir": path, "argv": [...]}

After this process restricts itself and runs the command, the command and everything it starts can only
  * read, list and execute under the `ro` paths,
  * read, write, create and remove under the `rw` paths (no execute),
  * read and write the single files in `rw_files` (device files such as /dev/null),
  * open TCP connections to the ports in `connect_ports`, and bind only the TCP ports in `bind_ports`.
Every other path gives EACCES on open and on listing. Landlock does not hide names: `stat` on a path outside the rules still works.
The environment of the command is exactly `env`. Why this exists: bubblewrap cannot make a user namespace here (AppArmor restricts
unprivileged user namespaces), and Landlock needs neither.
"""
from __future__ import annotations

import ctypes
import json
import os
import struct
import sys

SYS_CREATE, SYS_ADD_RULE, SYS_RESTRICT = 444, 445, 446      # x86_64 syscall numbers (the same on aarch64: 444, 445, 446)
PR_SET_NO_NEW_PRIVS = 38
RULE_PATH_BENEATH, RULE_NET_PORT = 1, 2
# Filesystem rights (ABI 1 to 5)
EXECUTE, WRITE_FILE, READ_FILE, READ_DIR, REMOVE_DIR, REMOVE_FILE = 1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5
MAKE_CHAR, MAKE_DIR, MAKE_REG, MAKE_SOCK, MAKE_FIFO, MAKE_BLOCK, MAKE_SYM = 1 << 6, 1 << 7, 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12
REFER, TRUNCATE, IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
ALL_FS = (1 << 16) - 1
NET_BIND_TCP, NET_CONNECT_TCP = 1 << 0, 1 << 1
RO_DIR = EXECUTE | READ_FILE | READ_DIR
RO_FILE = EXECUTE | READ_FILE
RW_DIR = (WRITE_FILE | READ_FILE | READ_DIR | REMOVE_DIR | REMOVE_FILE | MAKE_DIR | MAKE_REG | MAKE_SYM | MAKE_FIFO | REFER | TRUNCATE)
RW_FILE = WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV
FILE_RIGHTS = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV      # the rights Landlock accepts on a path that is not a directory

libc = ctypes.CDLL(None, use_errno=True)


def _check(rc: int, what: str) -> int:
    if rc < 0:
        e = ctypes.get_errno()
        raise OSError(e, f"{what}: {os.strerror(e)}")
    return rc


def abi_version() -> int:
    return libc.syscall(SYS_CREATE, None, 0, 1)


def restrict(ro, rw, rw_files, connect_ports, bind_ports=()) -> int:
    """Apply the rules to this process. Returns the ABI version used. A missing path is skipped; a kernel without network rules
    (ABI below 4) refuses the call when ports are asked for, because a silent open network would make the claim false."""
    abi = abi_version()
    if abi < 1:
        raise OSError("Landlock is not available")
    if (connect_ports or bind_ports) and abi < 4:
        raise OSError("this kernel has no Landlock network rules (ABI 4); refusing to run with an open network")
    fs_mask = ALL_FS if abi >= 5 else ((1 << 15) - 1 if abi >= 3 else ((1 << 14) - 1 if abi >= 2 else (1 << 13) - 1))
    net_mask = (NET_BIND_TCP | NET_CONNECT_TCP) if abi >= 4 else 0
    attr = struct.pack("QQ", fs_mask, net_mask) if abi >= 4 else struct.pack("Q", fs_mask)
    buf = ctypes.create_string_buffer(attr, len(attr))
    fd = _check(libc.syscall(SYS_CREATE, buf, len(attr), 0), "landlock_create_ruleset")

    def allow(path, rights):
        try:
            pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return
        try:
            if not os.path.isdir(path):
                rights &= FILE_RIGHTS
            rights &= fs_mask
            rule = struct.pack("<Qi", rights, pfd)           # struct landlock_path_beneath_attr, packed
            rb = ctypes.create_string_buffer(rule, len(rule))
            _check(libc.syscall(SYS_ADD_RULE, fd, RULE_PATH_BENEATH, rb, 0), f"landlock_add_rule {path}")
        finally:
            os.close(pfd)

    for p in ro:
        allow(os.path.realpath(p), RO_DIR if os.path.isdir(p) else RO_FILE)
    for p in rw:
        allow(os.path.realpath(p), RW_DIR)
    for p in rw_files:
        allow(os.path.realpath(p), RW_FILE)
    for port in connect_ports:
        rule = struct.pack("<QQ", NET_CONNECT_TCP, int(port))   # struct landlock_net_port_attr
        rb = ctypes.create_string_buffer(rule, len(rule))
        _check(libc.syscall(SYS_ADD_RULE, fd, RULE_NET_PORT, rb, 0), f"landlock_add_rule port {port}")
    for port in bind_ports:
        rule = struct.pack("<QQ", NET_BIND_TCP, int(port))
        rb = ctypes.create_string_buffer(rule, len(rule))
        _check(libc.syscall(SYS_ADD_RULE, fd, RULE_NET_PORT, rb, 0), f"landlock_add_rule bind {port}")
    _check(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "prctl no_new_privs")
    _check(libc.syscall(SYS_RESTRICT, fd, 0), "landlock_restrict_self")
    os.close(fd)
    return abi


def main(argv: list) -> int:
    spec = json.load(open(argv[0], encoding="utf-8"))
    # Rules that name this process's own /proc entry are resolved here, before the command replaces it (the pid stays the same).
    pid = str(os.getpid())
    ro = [p.replace("{pid}", pid) for p in spec["ro"]]
    abi = restrict(ro, spec["rw"], spec.get("rw_files", []), spec.get("connect_ports", []), spec.get("bind_ports", []))
    if spec.get("chdir"):
        os.chdir(spec["chdir"])
    if os.environ.get("LANDLOCK_EXEC_VERBOSE"):
        print(f"landlock abi {abi} applied", file=sys.stderr)
    argv2 = spec["argv"]
    os.execvpe(argv2[0], argv2, spec["env"])
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
