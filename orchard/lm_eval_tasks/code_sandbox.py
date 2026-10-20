# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run a program a model wrote with no network and no access to the operator's files. Standard library only, no root.

HumanEval and MBPP score a reply by running it. Hugging Face's `reliability_guard` (what `evaluate`'s `code_eval` uses) is not a sandbox:
`open()` and sockets still work, and the program runs as the operator, who has a Hugging Face token file, SSH keys and a docker socket. This
module is what `code_chat_utils` runs instead. A child process restricts itself, then runs the program:

* Landlock (Linux 5.13 and later) limits the files the program can open to the Python installation, a few system directories and one
  scratch directory made for this program. The token file, `~/.ssh` and `/proc` are not on the list. On Landlock ABI 4 and later TCP bind
  and connect are denied too, and on ABI 6 and later the program cannot signal a process outside its own.
* seccomp makes `socket` fail with EPERM, so no connection of any kind can be opened, and does the same for `io_uring_setup` (it can open
  sockets without calling `socket`), `ptrace` and `process_vm_readv`/`process_vm_writev`.
* `PR_SET_NO_NEW_PRIVS` is set, so a setuid program run from inside cannot lift either restriction.

If any step fails, the child exits with SANDBOX_UNAVAILABLE and the caller refuses to score. It never runs the program unrestricted unless
the caller passes `unsandboxed=True` (the operator's explicit flag).

The restrictions are checked by `self_check()`, which runs a probe program inside the sandbox that tries to read a planted file and open a
socket. A kernel that accepts the calls and enforces nothing fails that check. This is a precaution for code a model produces from a
benchmark prompt, not a defence against an attacker with a kernel exploit.
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import resource
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

SANDBOX_UNAVAILABLE = 97          # the child's exit status when it could not restrict itself
PROGRAM_FAILED = 1
HERE = Path(__file__).resolve()

# ---- Landlock ------------------------------------------------------------------------------------------------------------------------------

SYS_LANDLOCK_CREATE_RULESET, SYS_LANDLOCK_ADD_RULE, SYS_LANDLOCK_RESTRICT_SELF = 444, 445, 446     # the same on x86_64 and aarch64
LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
LANDLOCK_RULE_PATH_BENEATH = 1
FS_EXECUTE, FS_WRITE_FILE, FS_READ_FILE, FS_READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
FS_REMOVE_DIR, FS_REMOVE_FILE, FS_MAKE_CHAR, FS_MAKE_DIR = 1 << 4, 1 << 5, 1 << 6, 1 << 7
FS_MAKE_REG, FS_MAKE_SOCK, FS_MAKE_FIFO, FS_MAKE_BLOCK, FS_MAKE_SYM = 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12
FS_REFER, FS_TRUNCATE, FS_IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
NET_BIND_TCP, NET_CONNECT_TCP = 1 << 0, 1 << 1
SCOPE_ABSTRACT_UNIX_SOCKET, SCOPE_SIGNAL = 1 << 0, 1 << 1
FILE_BITS = FS_EXECUTE | FS_WRITE_FILE | FS_READ_FILE | FS_TRUNCATE | FS_IOCTL_DEV      # the access bits a rule on a file (not a directory) may name

# What the program may read: the Python installation and the system's shared libraries. Nothing under /home, /etc, /proc, /sys or /run.
SYSTEM_READ = ("/usr", "/lib", "/lib64", "/bin", "/sbin")
DEVICES = ("/dev/null", "/dev/zero", "/dev/urandom", "/dev/random")

_libc = None


def _libc_():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.syscall.restype = ctypes.c_long
    return _libc


class SandboxUnavailable(RuntimeError):
    """The kernel or the process cannot give the restrictions this module promises."""


def landlock_abi() -> int:
    """The Landlock ABI version the kernel offers, or 0 when it has none (not built, not enabled or blocked)."""
    got = _libc_().syscall(SYS_LANDLOCK_CREATE_RULESET, None, 0, LANDLOCK_CREATE_RULESET_VERSION)
    return int(got) if got > 0 else 0


def _handled_fs(abi: int) -> int:
    handled = (1 << 13) - 1
    if abi >= 2:
        handled |= FS_REFER
    if abi >= 3:
        handled |= FS_TRUNCATE
    if abi >= 5:
        handled |= FS_IOCTL_DEV
    return handled


def _check(result: int, what: str) -> int:
    if result < 0:
        err = ctypes.get_errno()
        raise SandboxUnavailable(f"{what} failed: {os.strerror(err)} (errno {err})")
    return int(result)


def apply_landlock(read_paths, write_paths) -> int:
    """Restrict this process (and what it starts) to reading `read_paths` and reading and writing under `write_paths`. Returns the ABI used.
    Raises SandboxUnavailable when the kernel has no Landlock. A path that does not exist is skipped."""
    abi = landlock_abi()
    if abi < 1:
        raise SandboxUnavailable("this kernel has no Landlock (landlock_create_ruleset is unavailable)")
    handled = _handled_fs(abi)
    handled_net = (NET_BIND_TCP | NET_CONNECT_TCP) if abi >= 4 else 0
    scoped = (SCOPE_ABSTRACT_UNIX_SOCKET | SCOPE_SIGNAL) if abi >= 6 else 0
    if abi >= 6:
        attr = struct.pack("<QQQ", handled, handled_net, scoped)
    elif abi >= 4:
        attr = struct.pack("<QQ", handled, handled_net)
    else:
        attr = struct.pack("<Q", handled)
    buf = ctypes.create_string_buffer(attr, len(attr))
    ruleset = _check(_libc_().syscall(SYS_LANDLOCK_CREATE_RULESET, buf, len(attr), 0), "landlock_create_ruleset")
    try:
        read = (FS_EXECUTE | FS_READ_FILE | FS_READ_DIR) & handled
        write = handled & ~FS_EXECUTE & ~FS_REFER
        for paths, access in ((read_paths, read), (write_paths, read | write)):
            for p in paths:
                try:
                    fd = os.open(p, os.O_PATH | os.O_CLOEXEC)
                except OSError:
                    continue
                try:
                    allowed = access if os.path.isdir(p) else access & FILE_BITS
                    rule = ctypes.create_string_buffer(struct.pack("<Qi", allowed, fd), 12)
                    _check(_libc_().syscall(SYS_LANDLOCK_ADD_RULE, ruleset, LANDLOCK_RULE_PATH_BENEATH, rule, 0), f"landlock_add_rule({p})")
                finally:
                    os.close(fd)
        _check(_libc_().prctl(38, 1, 0, 0, 0), "prctl(PR_SET_NO_NEW_PRIVS)")
        _check(_libc_().syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset, 0), "landlock_restrict_self")
    finally:
        os.close(ruleset)
    return abi


# ---- seccomp -------------------------------------------------------------------------------------------------------------------------------

# machine -> (AUDIT_ARCH, {name: syscall number}). io_uring has the same numbers on both.
SECCOMP_ARCH = {
    "x86_64": (0xC000003E, {"socket": 41, "ptrace": 101, "process_vm_readv": 310, "process_vm_writev": 311}),
    "aarch64": (0xC00000B7, {"socket": 198, "ptrace": 117, "process_vm_readv": 270, "process_vm_writev": 271}),
}
IO_URING = {"io_uring_setup": 425, "io_uring_enter": 426, "io_uring_register": 427}
BPF_LD_W_ABS, BPF_JEQ_K, BPF_RET_K = 0x20, 0x15, 0x06
RET_ALLOW, RET_KILL_PROCESS, RET_EPERM = 0x7FFF0000, 0x80000000, 0x00050000 | 1


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def seccomp_program(machine: str | None = None) -> list[tuple[int, int, int, int]]:
    """The filter as (code, jt, jf, k) rows: kill the process on another architecture, fail each listed syscall with EPERM, allow the rest."""
    machine = machine or platform.machine()
    if machine not in SECCOMP_ARCH:
        raise SandboxUnavailable(f"no seccomp filter for the {machine} architecture")
    audit, names = SECCOMP_ARCH[machine]
    rows = [(BPF_LD_W_ABS, 0, 0, 4),                        # arch
            (BPF_JEQ_K, 1, 0, audit), (BPF_RET_K, 0, 0, RET_KILL_PROCESS),
            (BPF_LD_W_ABS, 0, 0, 0)]                        # syscall number
    for nr in sorted({*names.values(), *IO_URING.values()}):
        rows += [(BPF_JEQ_K, 0, 1, nr), (BPF_RET_K, 0, 0, RET_EPERM)]
    rows.append((BPF_RET_K, 0, 0, RET_ALLOW))
    return rows


def apply_seccomp() -> None:
    rows = seccomp_program()
    filters = (_SockFilter * len(rows))(*[_SockFilter(*r) for r in rows])
    prog = _SockFprog(len(rows), filters)
    _check(_libc_().prctl(38, 1, 0, 0, 0), "prctl(PR_SET_NO_NEW_PRIVS)")
    _check(_libc_().prctl(22, 2, ctypes.byref(prog), 0, 0), "prctl(PR_SET_SECCOMP)")


def python_prefixes() -> list[str]:
    """The directories the interpreter reads its standard library and extension modules from."""
    seen = []
    for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        if p and p not in seen:
            seen.append(p)
    return seen


def restrict(workdir) -> dict:
    """Apply every restriction to this process. Call it in the child, after reading the program and before running it."""
    abi = apply_landlock([*SYSTEM_READ, *python_prefixes(), *DEVICES], [str(workdir)])
    apply_seccomp()
    return {"landlock_abi": abi, "seccomp": True, "network_rules": abi >= 4}


# ---- the child -----------------------------------------------------------------------------------------------------------------------------

def _limits(cpu_s: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def child_main(argv: list[str]) -> int:
    """`python -I code_sandbox.py child WORKDIR CPU_SECONDS [--unsandboxed]`: read the program from stdin, restrict, run it."""
    workdir, cpu_s = argv[0], int(argv[1])
    unsandboxed = "--unsandboxed" in argv[2:]
    program = sys.stdin.read()
    os.chdir(workdir)
    _limits(cpu_s)
    if not unsandboxed:
        try:
            restrict(workdir)
        except SandboxUnavailable as exc:
            print(f"SANDBOX UNAVAILABLE: {exc}", file=sys.stderr)
            return SANDBOX_UNAVAILABLE
    scope = {"__name__": "__main__", "__builtins__": __builtins__}
    try:
        exec(compile(program, "<program>", "exec"), scope)
    except SystemExit as exc:
        return 0 if exc.code in (None, 0) else PROGRAM_FAILED
    except BaseException:
        import traceback
        traceback.print_exc(limit=3)
        return PROGRAM_FAILED
    return 0


# ---- the caller ----------------------------------------------------------------------------------------------------------------------------

# status is "passed", "failed", "timeout" or "sandbox_unavailable"; detail is the end of the child's stderr.
Outcome = namedtuple("Outcome", ["status", "detail"], defaults=[""])


def _child_env(workdir: str) -> dict:
    """Nothing is inherited: no token variable, no SSH_AUTH_SOCK, no operator HOME."""
    return {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}


def run_program(program: str, *, timeout_s: float = 10.0, unsandboxed: bool = False, python: str | None = None) -> Outcome:
    """Run `program` in a new session inside the sandbox, in a scratch directory that is removed afterwards. A program that runs past
    `timeout_s` is killed with its process group. The scratch directory is the only place it can write."""
    workdir = tempfile.mkdtemp(prefix="orchard-code-")
    argv = [python or sys.executable, "-I", str(HERE), "child", workdir, str(max(1, int(timeout_s))), *(["--unsandboxed"] if unsandboxed else [])]
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=_child_env(workdir),
                                start_new_session=True, close_fds=True)
        try:
            _, err = proc.communicate(program.encode("utf-8", errors="replace"), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.communicate()
            return Outcome("timeout", f"no result after {timeout_s:g} s")
        text = (err or b"").decode("utf-8", errors="replace")[-400:]
        if proc.returncode == SANDBOX_UNAVAILABLE:
            return Outcome("sandbox_unavailable", text.strip())
        return Outcome("passed" if proc.returncode == 0 else "failed", text.strip())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# The probe runs inside the sandbox and reports what it could do on its last line of stderr. Each attempt must fail with an OSError.
PROBE = r'''
import json, os, socket, sys
canary = {canary!r}
seen = {{}}
for name, action in (
        ("read_planted_file", lambda: open(canary).read()),
        ("read_proc_environ", lambda: open("/proc/self/environ").read()),
        ("list_home", lambda: os.listdir("/home")),
        ("tcp_socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)),
        ("udp_socket", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM)),
        ("unix_socket", lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)),
        ("write_beside_planted_file", lambda: open(canary + ".new", "w").write("x"))):
    try:
        action()
        seen[name] = "allowed"
    except OSError as exc:
        seen[name] = "blocked: " + type(exc).__name__
print(json.dumps(seen), file=sys.stderr)
sys.exit(0 if all(v.startswith("blocked") for v in seen.values()) else 1)
'''


def self_check(python: str | None = None) -> dict:
    """Run the probe in the sandbox with a file planted outside it. Returns {"ok", "detail", "seen"}. `ok` is True only when the sandbox
    started and every action the probe tried (reading the planted file and /proc, listing /home, opening a TCP, UDP and Unix socket, writing
    beside the planted file) was refused. A sandbox that failed to start is reported as such, and a probe that printed no report is not ok."""
    planted = tempfile.mkdtemp(prefix="orchard-code-canary-")
    try:
        secret = Path(planted) / "token"
        secret.write_text("hf_planted_by_self_check\n")
        outcome = run_program(PROBE.format(canary=str(secret)), timeout_s=20, python=python)
        if outcome.status == "sandbox_unavailable":
            return {"ok": False, "detail": f"the sandbox could not be set up: {outcome.detail}", "seen": None}
        try:
            seen = json.loads(outcome.detail.splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "detail": f"the probe gave no report ({outcome.status}): {outcome.detail}", "seen": None}
        allowed = [k for k, v in seen.items() if not v.startswith("blocked")]
        if outcome.status != "passed" or allowed:
            return {"ok": False, "detail": "the sandbox let the probe do: " + (", ".join(allowed) or outcome.status), "seen": seen}
        return {"ok": True, "detail": "every probe action was refused", "seen": seen}
    finally:
        shutil.rmtree(planted, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "child":
        sys.exit(child_main(sys.argv[2:]))
    print(json.dumps(self_check()))
    sys.exit(0)
