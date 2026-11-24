# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Compile a tt-lang kernel with a chip package's own compiler, with no device and no lease.

Why. The kernel probe judges a kernel on the simulator. A simulator pass does not show the kernel compiles for a chip: issue 205 counted
0 of 7, and the two simulator passes for `center` in the recipes A/B of 2026-11-21 both fail in the chip package's compiler. Until now the
only way to find out was a leased chip. The first thing a chip run does with a kernel is compile it, and that step needs no chip.

What. `compile_check` runs orchard/ttlang_compile_child.py under the interpreter of a venv that holds a chip package (tt-lang 1.1.6 in the
shared venv, or the managed pair's light wheel). The child shuts /dev/tenstorrent out of itself with Landlock, proves it, replaces `ttnn`
with a stand-in, sets TTLANG_COMPILE_ONLY=1 and calls the kernel's operation with stand-in tensors. The package's compiler parses the
threads, builds its MLIR, runs its passes and writes the C++ kernel sources. Nothing imports ttnn and nothing can open a device.

How far a result can be trusted. `scripts/calibrate_hwcompile.py` replays the kernels of docs/run-logs/evidence/ttlang-pair, whose results
on a chip are recorded there for both packages, and compares. A pass here means: this package's compiler accepted the kernel and wrote
kernel source. It does not mean tt-metal built that source, that the buffers fit the chip's free L1, or that the numbers are right on a
chip. A record made here is named `compile` everywhere and never `chip`.

Nothing in this module imports ttl or ttnn. The chip package is identified by reading files (`package_identity`).
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from orchard import safeio

SCHEMA = "orchard-hwcompile/1"
CHILD = Path(__file__).resolve().with_name("ttlang_compile_child.py")
CHILD_SCHEMA = "orchard-hwcompile-child/1"
TIMEOUT_S = 300.0
MAX_KERNEL_BYTES = 256 * 1024             # the probe's own bound on a kernel file
MAX_RESULT_BYTES = 256 * 1024
MAX_IDENTITY_FILE = 2 * 1024 ** 3         # the compiler's shared library is tens of megabytes
MAX_FIRST_LINE = 300
MAX_TENSORS = 16
MAX_DIM = 1 << 20
# The files whose content decides what the compiler accepts. A package with the same version and other bytes is another compiler.
IDENTITY_FILES = ("_src/ttl_ast.py", "operators.py", "ttl_api.py", "_mlir_libs/libTTLangPythonCAPI.so")
CATEGORIES = ("compiled", "compile_error", "no_kernel_written", "guard_failed", "timeout", "not_run", "bad_result")
_VERSION = re.compile(r"[0-9A-Za-z.+!_-]{1,80}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ADDRESS = re.compile(r"\b0x[0-9a-fA-F]{6,}\b")


class HwCompileError(Exception):
    """The chip package cannot be identified or the check cannot be started. The message says why."""


# ---- which compiler ---------------------------------------------------------------------------------------------------------------------

def package_dir(python) -> Path:
    """The `ttl` directory of the venv whose interpreter is `python` (`<venv>/bin/python`). The path is taken as written: a venv's
    interpreter is a link to the system one, so resolving it would leave the venv."""
    venv = Path(os.path.abspath(os.fspath(python))).parent.parent
    try:
        found = sorted((venv / "lib").glob("python3.*/site-packages/ttl"))
    except OSError as exc:
        raise HwCompileError(f"the venv of {python} cannot be listed: {exc}") from None
    for candidate in found:
        if candidate.is_dir():
            return candidate
    raise HwCompileError(f"no ttl package beneath {venv}/lib: {python} is not the interpreter of a venv with tt-lang")


def _metadata_version(site: Path) -> str:
    """The version of the tt-lang distribution in `site`, from its METADATA file. The compiler wheel is `tt_lang`; the simulator wheel
    (`tt_lang_sim`) and the pair's helper (`tt_lang_light`) are other distributions and are skipped."""
    try:
        infos = sorted(p for p in site.glob("tt_lang-*.dist-info") if p.is_dir())
    except OSError as exc:
        raise HwCompileError(f"{site} cannot be listed: {exc}") from None
    for info in infos:
        try:
            text = safeio.read_bytes(info / "METADATA", max_bytes=1_000_000).decode("utf-8", errors="replace")
        except safeio.SafeReadError:
            continue
        for line in text.splitlines():
            if line.startswith("Version:"):
                version = line[len("Version:"):].strip()
                if _VERSION.fullmatch(version):
                    return version
            if not line.strip():
                break                   # the headers end at the first empty line
    raise HwCompileError(f"no tt_lang distribution with a readable version in {site}: this venv has no chip compiler")


def package_identity(python) -> dict:
    """What compiler `python` would use: {"package": "tt-lang", "version", "compiler_sha256", "files"}. Read from files; nothing is
    imported. `compiler_sha256` covers the files in IDENTITY_FILES that exist, each by name and sha256, so two installs with the same
    version and different compiler bytes get different identities. Raises HwCompileError."""
    return identity_of(package_dir(python))


def identity_of(ttl) -> dict:
    """The identity (as `package_identity` returns it) of the tt-lang compiler package in the directory `ttl`, the one that holds `_src/` and
    `operators.py`; its version comes from the distribution's METADATA next to it. For a caller that has the package's directory and no
    interpreter (scripts/verify_cheat_sheet.py). Raises HwCompileError."""
    ttl = Path(ttl)
    version = _metadata_version(ttl.parent)
    digest, seen = hashlib.sha256(), []
    for rel in IDENTITY_FILES:
        try:
            sha = safeio.sha256_file(ttl / rel, max_bytes=MAX_IDENTITY_FILE)
        except safeio.SafeReadError:
            continue
        digest.update(f"{rel}\0{sha}\n".encode("utf-8"))
        seen.append(rel)
    if "_src/ttl_ast.py" not in seen:
        raise HwCompileError(f"{ttl} has no readable _src/ttl_ast.py: it is not a tt-lang compiler package")
    return {"package": "tt-lang", "version": version, "compiler_sha256": digest.hexdigest(), "files": seen}


def identity_problems(identity) -> list[str]:
    """What is wrong with the shape of an identity read back from a record."""
    if not isinstance(identity, dict):
        return ["the target must be an object"]
    problems = []
    if identity.get("package") != "tt-lang":
        problems.append("target.package must be 'tt-lang'")
    if not (isinstance(identity.get("version"), str) and _VERSION.fullmatch(identity["version"])):
        problems.append("target.version must be a version string")
    if not (isinstance(identity.get("compiler_sha256"), str) and _HEX64.fullmatch(identity["compiler_sha256"])):
        problems.append("target.compiler_sha256 must be 64 hex digits")
    return problems


def same_target(a, b) -> bool:
    """Two identities name the same compiler: the same version and the same compiler bytes."""
    return (isinstance(a, dict) and isinstance(b, dict) and not identity_problems(a) and not identity_problems(b)
            and a["version"] == b["version"] and a["compiler_sha256"] == b["compiler_sha256"])


def target_label(identity: dict) -> str:
    return f"tt-lang {identity['version']}"


# ---- which function ---------------------------------------------------------------------------------------------------------------------

def _is_operation(decorator: ast.expr) -> bool:
    node = decorator.func if isinstance(decorator, ast.Call) else decorator
    return (isinstance(node, ast.Attribute) and node.attr == "operation" and isinstance(node.value, ast.Name) and node.value.id == "ttl")


def operation_name(kernel_text: str) -> str | None:
    """The name of the one top-level function decorated with `ttl.operation`, or None with none, several, or text that does not parse."""
    try:
        tree = ast.parse(kernel_text)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    names = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and any(_is_operation(d) for d in node.decorator_list)]
    return names[0] if len(names) == 1 else None


# ---- the check --------------------------------------------------------------------------------------------------------------------------

def _shapes(shapes) -> list[tuple[int, int]]:
    if not isinstance(shapes, (list, tuple)) or not 0 < len(shapes) <= MAX_TENSORS:
        raise HwCompileError(f"shapes must list 1 to {MAX_TENSORS} tensors")
    out = []
    for shape in shapes:
        ok = (isinstance(shape, (list, tuple)) and len(shape) == 2
              and all(type(v) is int and 0 < v <= MAX_DIM for v in shape))
        if not ok:
            raise HwCompileError(f"a shape must be two whole numbers from 1 to {MAX_DIM}, and {shape!r} is not")
        out.append((shape[0], shape[1]))
    return out


def spec_shapes(spec: dict) -> list[tuple[int, int]]:
    """The tensor arguments of a kernel for an op spec, in call order: each input, then the output (kernel_harness.py's order)."""
    if not isinstance(spec, dict) or not isinstance(spec.get("inputs"), dict):
        raise HwCompileError("the spec has no `inputs` object")
    return _shapes([*spec["inputs"].values(), spec.get("output")])


def _clean_line(text, work: str) -> str:
    """One line of compiler output for a record: the scratch directory's path and any memory address removed (so the same refusal
    gives the same line on every run and every machine), control characters dropped, bounded."""
    if not isinstance(text, str):
        return ""
    text = _ADDRESS.sub("0x...", text.replace(work, "<work>"))       # an object's address differs on every run
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    return text.strip()[:MAX_FIRST_LINE]


def _record(category: str, *, target: dict, kernel_sha256: str | None, fn: str | None, shapes, first_line: str = "", child: dict | None = None) -> dict:
    child = child or {}
    return {"schema": SCHEMA, "category": category, "compiled": category == "compiled", "target": target, "kernel_sha256": kernel_sha256,
            "operation": fn, "shapes": [list(s) for s in shapes], "first_line": first_line,
            "kernel_sources": child.get("kernels", []), "stand_in_used": child.get("stand_in_used", []),
            "grid": child.get("grid"), "arch": child.get("arch"), "guard": child.get("guard", [])}


def _child_problems(child) -> list[str]:
    """Type checks on the child's result file before any field is used."""
    if not isinstance(child, dict) or child.get("schema") != CHILD_SCHEMA:
        return ["the result is not an object of the child's schema"]
    problems = []
    if type(child.get("compiled")) is not bool:
        problems.append("`compiled` is not true or false")
    for key in ("kernels", "stand_in_used", "guard"):
        if not isinstance(child.get(key), list):
            problems.append(f"`{key}` is not a list")
    if not problems:
        for k in child["kernels"]:
            if not (isinstance(k, dict) and isinstance(k.get("name"), str) and len(k["name"]) <= 200 and type(k.get("bytes")) is int
                    and isinstance(k.get("sha256"), str) and _HEX64.fullmatch(k["sha256"])):
                problems.append("a kernel source entry is malformed")
                break
        if not all(isinstance(g, dict) and g.get("state") in ("refused", "absent") and isinstance(g.get("path"), str) for g in child["guard"]):
            problems.append("a guard entry is malformed")
        if not all(isinstance(s, str) and len(s) <= 200 for s in child["stand_in_used"]):
            problems.append("a stand-in entry is malformed")
    for key in ("error_type", "error_first_line"):
        if child.get(key) is not None and not isinstance(child.get(key), str):
            problems.append(f"`{key}` is not text")
    return problems


def compile_check(python, kernel, shapes, *, fn: str | None = None, work_dir=None, timeout_s: float = TIMEOUT_S, grid=None,
                  deny=None, run=subprocess.run, identity: dict | None = None) -> dict:
    """Compile the kernel file `kernel` with the chip package of the interpreter `python`. Returns one record (schema `orchard-hwcompile/1`).

    The package's identity and the kernel's sha256 are taken before the compiler runs, and the compiler reads a private copy of the kernel
    with exactly those bytes, so the record names what was compiled. `category` is `compiled` only when the child proved the device guard,
    the call returned and the compiler wrote at least one kernel source. Every other outcome has its own category and `compiled` false:
    a refused kernel is `compile_error` with the compiler's first line, and a check that could not be made is `guard_failed`, `timeout`,
    `not_run` or `bad_result`, so a broken instrument is never counted as a kernel that does not compile.

    Raises HwCompileError only for a caller's mistake (no such package, shapes of the wrong type)."""
    shapes = _shapes(shapes)
    target = identity if identity is not None else package_identity(python)
    try:
        data = safeio.read_bytes(kernel, max_bytes=MAX_KERNEL_BYTES)
    except safeio.SafeReadError as exc:
        return _record("not_run", target=target, kernel_sha256=None, fn=fn, shapes=shapes, first_line=f"the kernel file cannot be read: {exc}")
    sha = hashlib.sha256(data).hexdigest()
    text = data.decode("utf-8", errors="replace")
    name = fn or operation_name(text)
    if not name or not name.isidentifier():
        return _record("not_run", target=target, kernel_sha256=sha, fn=None, shapes=shapes,
                       first_line="the kernel file has no single top-level function decorated with ttl.operation")
    own = work_dir is None
    work = Path(tempfile.mkdtemp(prefix="orchard-hwcompile-")) if own else Path(work_dir)
    try:
        work.mkdir(parents=True, exist_ok=True)
        copy, result = work / "kernel.py", work / "result.json"
        for stale in (copy, result):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        safeio.atomic_write(copy, data, mode=0o600)
        argv = ["nice", "-n", "10", os.path.abspath(os.fspath(python)), "-I", str(CHILD), "--kernel", str(copy), "--fn", name, "--result", str(result)]
        for rows, cols in shapes:
            argv += ["--shape", f"{rows}x{cols}"]
        if grid is not None:
            argv += ["--grid", f"{int(grid[0])}x{int(grid[1])}"]
        for path in deny or ():
            argv += ["--deny", os.fspath(path)]
        # A small environment of our own: no PYTHONPATH, LD_LIBRARY_PATH or TT_METAL_HOME, so the host's ttnn is not even findable.
        env = {"PATH": "/usr/bin:/bin", "HOME": str(work), "LANG": "C.UTF-8", "TTLANG_COMPILE_ONLY": "1"}
        try:
            run(argv, cwd=str(work), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            return _record("timeout", target=target, kernel_sha256=sha, fn=name, shapes=shapes, first_line=f"no result after {timeout_s:g} s")
        except OSError as exc:
            return _record("not_run", target=target, kernel_sha256=sha, fn=name, shapes=shapes, first_line=f"the interpreter did not start: {exc}")
        try:
            child = safeio.read_json(result, max_bytes=MAX_RESULT_BYTES, expect=dict)
        except safeio.SafeReadError as exc:
            return _record("bad_result", target=target, kernel_sha256=sha, fn=name, shapes=shapes, first_line=f"the child left no usable result: {exc}")
        problems = _child_problems(child)
        if problems:
            return _record("bad_result", target=target, kernel_sha256=sha, fn=name, shapes=shapes, first_line=problems[0])
        first = _clean_line(child.get("error_first_line") or "", str(work))
        if child.get("error_type") == "GuardError" or not child["guard"]:
            return _record("guard_failed", target=target, kernel_sha256=sha, fn=name, shapes=shapes,
                           first_line=first or "the child did not prove the device guard", child=child)
        if child["compiled"] and child["kernels"]:
            return _record("compiled", target=target, kernel_sha256=sha, fn=name, shapes=shapes, child=child)
        if child["compiled"] or child.get("error_type") == "no_kernel_written":
            return _record("no_kernel_written", target=target, kernel_sha256=sha, fn=name, shapes=shapes,
                           first_line=first or "the call returned and the compiler wrote no kernel source", child=child)
        return _record("compile_error", target=target, kernel_sha256=sha, fn=name, shapes=shapes,
                       first_line=first or "the compiler raised an error with no message", child=child)
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)


def record_problems(record) -> list[str]:
    """What is wrong with the shape of a compile record read back from a file."""
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return [f"a compile record must be an object with schema {SCHEMA!r}"]
    problems = identity_problems(record.get("target"))
    if record.get("category") not in CATEGORIES:
        problems.append("category is not one this module writes")
    if type(record.get("compiled")) is not bool or record.get("compiled") != (record.get("category") == "compiled"):
        problems.append("`compiled` must be true exactly when the category is `compiled`")
    if not (isinstance(record.get("kernel_sha256"), str) and _HEX64.fullmatch(record["kernel_sha256"])) and record.get("compiled"):
        problems.append("a compiled record needs the kernel's sha256")
    if not isinstance(record.get("first_line"), str) or len(record["first_line"]) > MAX_FIRST_LINE:
        problems.append("first_line must be bounded text")
    sources = record.get("kernel_sources")
    if not isinstance(sources, list) or (record.get("compiled") and not sources):
        problems.append("a compiled record lists the kernel sources the compiler wrote")
    guard = record.get("guard")
    if record.get("compiled") and not (isinstance(guard, list) and guard and all(
            isinstance(g, dict) and g.get("state") in ("refused", "absent") for g in guard)):
        problems.append("a compiled record carries the proof that the device was shut out")
    return problems
