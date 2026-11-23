#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Compile one tt-lang kernel with a chip package's own compiler, with no device. The child of orchard/ttlang_hwcompile.py.

    <chip venv>/bin/python -I ttlang_compile_child.py --kernel FILE --fn NAME --shape RxC [--shape RxC ...]
                                                      --result FILE [--grid XxY] [--deny DIR ...]

It is run by the interpreter of the venv that holds the chip package (tt-lang 1.1.6 in the shared venv, or the pair's light wheel), so the
compiler that reads the kernel is the one the chip would use. It uses the standard library only and imports nothing from orchard.

What it does, in order:

1. Shuts the device out of this process, in the kernel. Landlock (an unprivileged Linux security module) is told to allow file access
   everywhere except beneath the denied directories (default: /dev/tenstorrent). The rule is inherited by every child process and cannot be
   lifted. The script then proves the rule is in force by asking for a listing of each denied directory, which must be refused, before any
   tt-lang code is imported. With no Landlock the script stops: it never runs unguarded.
2. Puts a stand-in module named `ttnn` in `sys.modules`. The real ttnn is never imported, so nothing in this process can ask the runtime
   for a device. The stand-in has a tensor class (shape, bfloat16, tile layout, DRAM), a device that only reports a compute grid, and
   permissive descriptor classes. Every other name asked of it is recorded in the result (`stand_in_used`).
3. Sets TTLANG_COMPILE_ONLY=1, loads the kernel file and calls its operation with stand-in tensors. tt-lang then parses the threads, builds
   its MLIR, runs its passes and writes one C++ source file per thread. It does not build that C++ (tt-metal does that for a chip at run
   time) and it runs nothing.
4. Writes the result file: `compiled` is true only when the call returned AND the compiler wrote at least one C++ file (their names, sizes
   and sha256 are listed). A call that returns without writing a kernel is reported as `no_kernel_written`, never as a pass.

What a pass shows: the chip package's compiler accepts every call, namespace and keyword in the kernel and can lower it to kernel source.
What it does not show: that tt-metal builds that source for the chip, that the buffers fit the chip's free L1 (the stand-in device makes the
compiler use its default budget), or that the kernel computes the right numbers on a chip.
"""
import argparse
import ctypes
import enum
import hashlib
import importlib.util
import json
import os
import stat
import sys
import traceback
import types

DEFAULT_DENY = ("/dev/tenstorrent",)
DEFAULT_GRID = (13, 10)          # an unharvested Blackhole's compute grid (tt-metal core descriptor blackhole_140_arch.yaml: x 0 to 12, y 0 to 9)
MAX_ERROR_CHARS = 4000
MAX_KERNEL_FILES = 64

# ---- 1. the device guard --------------------------------------------------------------------------------------------------------------

_SYS_CREATE_RULESET, _SYS_ADD_RULE, _SYS_RESTRICT_SELF = 444, 445, 446      # the same numbers on every Linux architecture
_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
# The file-system rights Landlock knows at each ABI version: 13 at version 1, refer at 2, truncate at 3, ioctl on devices at 5.
_RIGHTS_BY_ABI = {1: 13, 2: 14, 3: 15, 4: 15, 5: 16}
_MAX_RIGHTS = 16
# Rights that may be attached to a rule on a file that is not a directory: execute, write, read, truncate, ioctl on a device.
_FILE_RIGHTS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14) | (1 << 15)


class GuardError(Exception):
    """The device guard could not be put in force. The message says which step failed."""


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def deny_beneath(denied) -> list:
    """Refuse, for this process and every process it starts, any open or listing beneath each directory in `denied`.

    Landlock allows by rule, so this adds an allow rule for everything that is not a denied directory or on the way to one: every entry
    of `/` except the first part of a denied path, then every entry of that part except the next, and so on. Returns the resolved denied
    paths. Raises GuardError when Landlock is missing or a call fails; nothing is half-applied in that case, because the restriction is
    the last call."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:
        raise GuardError(f"the C library cannot be loaded: {exc}") from None
    targets = []
    for path in denied:
        real = os.path.realpath(path)
        if real == "/" or not os.path.isabs(real):
            raise GuardError(f"{path!r} cannot be denied: it resolves to {real!r}")
        targets.append(real)
    abi = libc.syscall(_SYS_CREATE_RULESET, None, ctypes.c_size_t(0), ctypes.c_uint32(1))       # flag 1 asks for the ABI version
    if abi < 1:
        raise GuardError(f"Landlock is not available on this kernel (errno {ctypes.get_errno()})")
    handled = (1 << _RIGHTS_BY_ABI.get(abi, _MAX_RIGHTS)) - 1
    attr = _RulesetAttr(handled)
    ruleset = libc.syscall(_SYS_CREATE_RULESET, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0))
    if ruleset < 0:
        raise GuardError(f"landlock_create_ruleset failed (errno {ctypes.get_errno()})")
    try:
        def on_the_way(path: str) -> bool:
            return any(t.startswith(path.rstrip("/") + "/") for t in targets)

        def allow(path: str) -> None:
            try:
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW)
            except OSError:
                return                      # gone, or not reachable by this user: nothing to allow
            try:
                mode = os.fstat(fd).st_mode
                if stat.S_ISLNK(mode):
                    return                  # a link grants nothing itself; its target is allowed where it lives
                rule = _PathBeneathAttr(handled if stat.S_ISDIR(mode) else handled & _FILE_RIGHTS, fd)
                if libc.syscall(_SYS_ADD_RULE, ruleset, _RULE_PATH_BENEATH, ctypes.byref(rule), ctypes.c_uint32(0)) != 0:
                    raise GuardError(f"landlock_add_rule failed for {path} (errno {ctypes.get_errno()})")
            finally:
                os.close(fd)

        def walk(directory: str) -> None:
            try:
                entries = sorted(e.path for e in os.scandir(directory))
            except OSError as exc:
                raise GuardError(f"{directory} cannot be listed: {exc}") from None
            for path in entries:
                if path in targets:
                    continue                # the denied directory itself gets no rule
                if on_the_way(path):
                    walk(path)
                else:
                    allow(path)

        walk("/")
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise GuardError(f"prctl(no_new_privs) failed (errno {ctypes.get_errno()})")
        if libc.syscall(_SYS_RESTRICT_SELF, ruleset, ctypes.c_uint32(0)) != 0:
            raise GuardError(f"landlock_restrict_self failed (errno {ctypes.get_errno()})")
    finally:
        os.close(ruleset)
    return targets


def prove_denied(targets) -> list:
    """One fact per denied directory, from a listing attempt made after the guard: `refused` (the guard is in force for it) or `absent`
    (there is nothing there to open). Listing a directory opens no device node. Raises GuardError when a listing succeeds: the guard is
    then not in force and the caller must stop."""
    facts = []
    for path in targets:
        try:
            os.listdir(path)
        except PermissionError:
            facts.append({"path": path, "state": "refused"})
        except FileNotFoundError:
            facts.append({"path": path, "state": "absent"})
        except OSError as exc:
            raise GuardError(f"{path}: the listing failed in a way that does not show the guard ({exc})") from None
        else:
            raise GuardError(f"{path} can still be listed: the guard is not in force")
    return facts


# ---- 2. the stand-in ttnn -------------------------------------------------------------------------------------------------------------

class DataType(enum.Enum):
    FLOAT32 = 1
    BFLOAT16 = 2
    BFLOAT8_B = 3
    BFLOAT4_B = 4
    INT32 = 5
    UINT32 = 6
    UINT16 = 7
    UINT8 = 8


class BufferType(enum.Enum):
    DRAM = 0
    L1 = 1
    SYSTEM_MEMORY = 2
    L1_SMALL = 3
    TRACE = 4


class StorageType(enum.Enum):
    HOST = 0
    DEVICE = 1


class Layout(enum.Enum):
    ROW_MAJOR = 0
    TILE = 1


class _Anything:
    """A descriptor stand-in: takes any arguments, any attribute can be set or read, and calling it gives another one."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _Anything()
        self.__dict__[name] = value
        return value

    def __call__(self, *args, **kwargs):
        return _Anything()

    def __iter__(self):
        return iter(())


class _Grid:
    def __init__(self, x: int, y: int):
        self.x, self.y = x, y


class StandInDevice:
    """Reports a compute grid and an architecture, and nothing else. It has no `shape`, so tt-lang treats it as one device and not a
    mesh."""

    def __init__(self, grid, arch: str = "blackhole"):
        self._grid = _Grid(*grid)
        self._arch = arch

    def arch(self):
        return self._arch

    def compute_with_storage_grid_size(self):
        return self._grid

    def id(self):
        return 0


class TensorMemoryLayout(enum.Enum):
    INTERLEAVED = 0
    HEIGHT_SHARDED = 2
    WIDTH_SHARDED = 3
    BLOCK_SHARDED = 4
    ND_SHARDED = 5


class _MemoryConfig:
    buffer_type = BufferType.DRAM
    memory_layout = TensorMemoryLayout.INTERLEAVED
    shard_spec = None


class StandInTensor:
    """What tt-lang reads from a tensor argument while it compiles: shape (in elements), dtype, layout, memory space and device."""

    def __init__(self, shape, device):
        self.shape = tuple(int(v) for v in shape)
        self.dtype = DataType.BFLOAT16
        self.layout = Layout.TILE
        self._device = device

    def memory_config(self):
        return _MemoryConfig()

    def storage_type(self):
        return StorageType.DEVICE

    def device(self):
        return self._device


class _DeviceInfo:
    # tt-lang asks the runtime how much L1 a device has left for buffers and passes the answer to its passes as an override. 0 is "no
    # override": each compiler then uses the budget built into it. So a kernel whose buffers do not fit the chip's free L1 is not caught.
    cb_limit = 0


def _device_info(device):
    return _DeviceInfo()


def _no_pages(device):
    return []


def _base_address(device, buffer_type):
    return 0


def install_stand_in() -> list:
    """Put the stand-in `ttnn` (and the submodules tt-lang imports by name) in sys.modules. Returns the list that collects every name
    asked of the stand-in beyond the ones it defines."""
    used: list = []

    def module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        mod.__path__ = []           # a package, so `import ttnn._ttnn.x` consults sys.modules for the parts placed there

        def fallback(attr, _name=name):
            if attr.startswith("__"):
                raise AttributeError(attr)
            used.append(f"{_name}.{attr}")
            return _Anything
        mod.__getattr__ = fallback
        sys.modules[name] = mod
        return mod

    ttnn = module("ttnn")
    inner = module("ttnn._ttnn")
    reports = module("ttnn._ttnn.reports")
    reports.get_device_info = _device_info
    reports.get_buffer_pages = _no_pages
    ttnn.get_allocator_base_address = _base_address
    inner.reports = reports
    ttnn._ttnn = inner
    ttnn.reports = reports
    ttnn.Tensor = StandInTensor
    ttnn.DataType = DataType
    ttnn.BufferType = BufferType
    ttnn.Layout = Layout
    ttnn.StorageType = StorageType
    ttnn.TensorMemoryLayout = TensorMemoryLayout
    ttnn.TILE_LAYOUT = Layout.TILE
    ttnn.ROW_MAJOR_LAYOUT = Layout.ROW_MAJOR
    ttnn.bfloat16 = DataType.BFLOAT16
    ttnn.float32 = DataType.FLOAT32
    return used


# ---- 3 and 4. the compile and its result ----------------------------------------------------------------------------------------------

def _shape(text: str) -> tuple:
    parts = text.lower().split("x")
    if len(parts) != 2 or not all(p.isdigit() and 0 < int(p) <= 1 << 20 for p in parts):
        raise argparse.ArgumentTypeError(f"{text!r} is not ROWSxCOLS in whole numbers")
    return int(parts[0]), int(parts[1])


def kernel_files(directory: str) -> list:
    """The C++ kernel sources tt-lang wrote into `directory`: name, bytes and sha256, in name order. Regular files only, by descriptor."""
    found = []
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return found
    for name in names[:MAX_KERNEL_FILES * 4]:
        if not name.endswith(".cpp") or name.startswith("."):
            continue
        try:
            fd = os.open(os.path.join(directory, name), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        except OSError:
            continue
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                continue
            digest, size = hashlib.sha256(), 0
            while True:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
                if size > 8 << 20:
                    break
            found.append({"name": name, "bytes": size, "sha256": digest.hexdigest()})
        finally:
            os.close(fd)
        if len(found) >= MAX_KERNEL_FILES:
            break
    return found


def _first_line(text: str) -> str:
    """The line that says what went wrong. tt-lang's pass errors start with a line that only names the package version
    (`RuntimeError: ttlang 1.1.10`) and put the message on the next line that starts with `error:`; both are kept then."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    if "error:" not in first:
        for line in lines[1:]:
            if line.startswith("error:"):
                first = f"{first} | {line}"
                break
    return first[:300]


def run(args) -> dict:
    result = {"schema": "orchard-hwcompile-child/1", "compiled": False, "stage": "guard", "guard": [], "grid": list(args.grid), "arch": args.arch,
              "kernels": [], "stand_in_used": [], "error_type": None, "error_first_line": "", "error": ""}
    # tt-lang writes its kernel sources to /tmp/$USER. A name of our own gives this run a directory nothing else writes to. It is made
    # before the guard, so it exists when the guard lists what to allow (a denied directory beneath /tmp would otherwise stop its creation).
    user = f"orchard-hwcompile-{os.getpid()}"
    out_dir = os.path.join("/tmp", user)
    try:
        os.mkdir(out_dir, 0o700)
    except OSError as exc:
        result["error_type"], result["error_first_line"] = "GuardError", f"the output directory cannot be made: {exc}"[:300]
        return result
    try:
        result["guard"] = prove_denied(deny_beneath(args.deny or DEFAULT_DENY))
    except GuardError as exc:
        result["error_type"], result["error_first_line"] = "GuardError", str(exc)[:300]
        try:
            os.rmdir(out_dir)
        except OSError:
            pass
        return result
    result["stage"] = "import"
    used = install_stand_in()
    result["stand_in_used"] = used
    os.environ["TTLANG_COMPILE_ONLY"] = "1"
    os.environ["USER"] = user
    try:
        import ttl                                                  # the chip package of the interpreter that runs this file
        result["tt_lang_version"] = str(getattr(ttl, "__version__", "unknown"))[:80]
        result["stage"] = "load"
        spec = importlib.util.spec_from_file_location("orchard_hwcompile_kernel", args.kernel)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, args.fn)
        result["stage"] = "compile"
        device = StandInDevice(args.grid, args.arch)
        fn(*[StandInTensor(shape, device) for shape in args.shape])
        result["kernels"] = kernel_files(out_dir)
        if result["kernels"]:
            result["compiled"] = True
            result["stage"] = "done"
        else:
            result["error_type"] = "no_kernel_written"
            result["error_first_line"] = "the call returned and the compiler wrote no kernel source"
    except BaseException as exc:                # a compiler may raise SystemExit; the result must still be written
        text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        result["error_type"] = type(exc).__name__
        result["error_first_line"] = _first_line(text)
        result["error"] = text[:MAX_ERROR_CHARS]
    finally:
        result["stand_in_used"] = sorted(set(used))[:200]
        for entry in kernel_files(out_dir) if not result["kernels"] else result["kernels"]:
            try:
                os.unlink(os.path.join(out_dir, entry["name"]))
            except OSError:
                pass
        try:
            os.rmdir(out_dir)
        except OSError:
            pass
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compile one tt-lang kernel with this interpreter's chip package, with no device.")
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--fn", required=True)
    ap.add_argument("--shape", action="append", type=_shape, required=True, help="ROWSxCOLS of one tensor argument, in call order")
    ap.add_argument("--result", required=True)
    ap.add_argument("--grid", type=_shape, default=DEFAULT_GRID, help="the stand-in device's compute grid as XxY")
    ap.add_argument("--arch", default="blackhole", choices=("blackhole", "wormhole_b0"), help="the stand-in device's architecture")
    ap.add_argument("--deny", action="append", help="a directory to shut out (default /dev/tenstorrent)")
    args = ap.parse_args(argv)
    result = run(args)
    data = (json.dumps(result, indent=1) + "\n").encode("utf-8")
    fd = os.open(args.result, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return 0 if result["compiled"] else 1


if __name__ == "__main__":
    sys.exit(main())
