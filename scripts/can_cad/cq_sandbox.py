#!/usr/bin/env python3
"""Run one model-generated CadQuery script with as much confinement as this machine allows without privileges, and export its `result`.

What it applies, in the child process and before any generated code runs (each layer reported in the returned `layers` dict):
- an empty environment (no tokens or other credentials: only PATH, HOME and TMPDIR, all pointing into the work directory or system paths);
- the work directory as the current directory and HOME;
- resource limits: CPU seconds, address space, file size, open files, and no new processes beyond a small number;
- a wall-clock timeout in the parent, with the child in its own session so the whole group is killed;
- Landlock (Linux 5.13+ for files, 6.7+ for TCP), when the kernel has it: the child may read the file system but write only inside the work
  directory, and TCP connect and bind are refused. Landlock does not cover UDP or Unix sockets.
Not applied: a network namespace (unprivileged ones are not available here), seccomp, or a separate user. A reply that is written to attack
this sandbox is not what it is meant to stop; it stops accidental damage (a script that deletes files, reads a token or fetches a URL).

The verdict about the geometry never comes from the generated code's own output. After the generated code has run, the trusted epilogue in the
child exports `result` to a STEP file in the work directory; the parent loads that file in a separate step (check_cq.py) and judges it.
"""
from __future__ import annotations

import ctypes
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile

# Landlock constants (include/uapi/linux/landlock.h). x86-64 and arm64 share these syscall numbers.
SYS_CREATE, SYS_ADD_RULE, SYS_RESTRICT = 444, 445, 446
FS_FLAGS = [1 << i for i in range(16)]          # EXECUTE .. IOCTL_DEV, in the header's order
FS_READ = (1 << 0) | (1 << 2) | (1 << 3)        # EXECUTE, READ_FILE, READ_DIR
NET_BIND, NET_CONNECT = 1 << 0, 1 << 1
ABI_FS_MASK = {1: (1 << 13) - 1, 2: (1 << 14) - 1, 3: (1 << 15) - 1}      # flags the ABI knows; ABI 4 and later add the network flags
ABI_FS_FULL = (1 << 16) - 1


def scrubbed_env(workdir: str) -> dict:
    """The only variables the generated code sees."""
    return {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}


def _landlock(workdir: str) -> dict:
    """Restrict this process. Returns what was applied; never raises (a kernel without Landlock reports it)."""
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(SYS_CREATE, None, 0, 1)
    if abi < 1:
        return {"landlock": False, "reason": "kernel has no Landlock"}
    fs_mask = ABI_FS_MASK.get(abi, ABI_FS_FULL) if abi < 5 else ABI_FS_FULL
    net_mask = (NET_BIND | NET_CONNECT) if abi >= 4 else 0

    class Attr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]

    attr = Attr(fs_mask, net_mask)
    size = ctypes.sizeof(attr) if abi >= 4 else 8
    fd = libc.syscall(SYS_CREATE, ctypes.byref(attr), size, 0)
    if fd < 0:
        return {"landlock": False, "reason": f"create failed errno {ctypes.get_errno()}"}

    class Rule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    def allow(path: str, access: int) -> None:
        pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = Rule(access & fs_mask, pfd)
            if libc.syscall(SYS_ADD_RULE, fd, 1, ctypes.byref(rule), 0) != 0:
                raise OSError(ctypes.get_errno(), f"landlock_add_rule {path}")
        finally:
            os.close(pfd)

    allow("/", FS_READ)
    allow(workdir, fs_mask)
    if libc.prctl(38, 1, 0, 0, 0) != 0:           # PR_SET_NO_NEW_PRIVS
        return {"landlock": False, "reason": "no_new_privs failed"}
    if libc.syscall(SYS_RESTRICT, fd, 0) != 0:
        return {"landlock": False, "reason": f"restrict_self failed errno {ctypes.get_errno()}"}
    os.close(fd)
    return {"landlock": True, "abi": abi, "files": "read anywhere, write only in the work directory", "tcp": bool(net_mask)}


def _child(workdir: str, code_path: str, epilogue_path: str, limits: dict) -> None:
    layers = {"env": "scrubbed", "limits": limits}
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_s"], limits["cpu_s"]))
    resource.setrlimit(resource.RLIMIT_AS, (limits["address_space"], limits["address_space"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits["file_bytes"], limits["file_bytes"]))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
    os.chdir(workdir)
    layers.update(_landlock(workdir))
    with open(os.path.join(workdir, "layers.json"), "w", encoding="utf-8") as f:     # before generated code; it can rewrite this file, which only hides the report
        json.dump(layers, f)
    ns = {"__name__": "__main__"}
    with open(code_path, encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, "generated.py", "exec"), ns)                                 # noqa: S102 - the point of this module
    with open(epilogue_path, encoding="utf-8") as f:
        exec(compile(f.read(), "epilogue.py", "exec"), ns)                          # noqa: S102


def run_code(code: str, epilogue: str = "", *, python: str = sys.executable, timeout_s: float = 60.0,
             cpu_s: int = 60, address_space: int = 4 << 30, file_bytes: int = 50 << 20) -> dict:
    """Run `code` in the sandbox. Returns returncode, timed_out, stderr tail, the layers the child reported, the files left in the
    work directory (name to size), and the work directory path (removed on return)."""
    limits = {"cpu_s": cpu_s, "address_space": address_space, "file_bytes": file_bytes}
    with tempfile.TemporaryDirectory(prefix="cqsandbox-") as work:
        code_path, epi_path = os.path.join(work, "generated_code.txt"), os.path.join(work, "epilogue_code.txt")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        with open(epi_path, "w", encoding="utf-8") as f:
            f.write(epilogue)
        proc = subprocess.Popen([python, os.path.abspath(__file__), "--child", work, code_path, epi_path, json.dumps(limits)],
                                env=scrubbed_env(work), cwd=work, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
        timed_out = False
        try:
            out, err = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
        layers = {}
        lp = os.path.join(work, "layers.json")
        if os.path.exists(lp):
            try:
                layers = json.load(open(lp, encoding="utf-8"))
            except ValueError:
                layers = {}
        files = {n: os.path.getsize(os.path.join(work, n)) for n in os.listdir(work) if n not in ("generated_code.txt", "epilogue_code.txt", "layers.json")}
        copied = {}
        for n in files:
            if n.endswith(".step"):
                with open(os.path.join(work, n), "rb") as src:
                    copied[n] = src.read()
        return {"returncode": proc.returncode, "timed_out": timed_out, "stderr": err.decode("utf-8", "replace")[-300:], "layers": layers,
                "files": files, "step_bytes": copied}


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--child":
        _child(sys.argv[2], sys.argv[3], sys.argv[4], json.loads(sys.argv[5]))
    else:
        sys.exit("internal: run through run_code()")
