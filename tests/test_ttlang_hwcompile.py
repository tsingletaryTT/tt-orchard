# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The device-less chip compile check (orchard/ttlang_hwcompile.py and its child, orchard/ttlang_compile_child.py).

Most tests run the real child in a real, small venv whose `ttl` is a fake compiler written here: it does what the test's kernel tells it
(write a kernel source, raise, do nothing, read a denied file). So the guard, the stand-in and the result are tested as they run. The
tests at the end use the chip packages of this machine when they are installed, and skip elsewhere. No test imports ttl or ttnn in the
test process, and no test opens a device: the real-package tests go through the same guarded child."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from orchard import ttlang_hwcompile as hc

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ttlang_compile_child", REPO / "orchard" / "ttlang_compile_child.py")
child_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(child_mod)

FAKE_TTL = '''
import os
__version__ = "9.9.9"


class CompileError(Exception):
    pass


def emit(name, text="// kernel"):
    d = os.path.join("/tmp", os.environ["USER"])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "ttlang_kernel_%s_0.cpp" % name), "w") as f:
        f.write(text)


def operation(grid=None):
    def deco(fn):
        def call(*tensors):
            assert os.environ.get("TTLANG_COMPILE_ONLY") == "1", "not compile-only"
            for t in tensors:
                t.shape, t.dtype, t.layout, t.memory_config(), t.device().compute_with_storage_grid_size().x
            return fn(*tensors)
        return call
    return deco
'''


def landlock_available() -> bool:
    """Whether this kernel has Landlock, asked of the kernel directly (the ABI version query). It does not go through the child's own
    guard code: a skip decided by the code under test would turn a broken guard into skipped tests and a green run."""
    import ctypes
    try:
        return ctypes.CDLL(None, use_errno=True).syscall(444, None, ctypes.c_size_t(0), ctypes.c_uint32(1)) >= 1
    except OSError:
        return False


needs_landlock = pytest.mark.skipif(not landlock_available(), reason="this kernel has no Landlock, so the child refuses to run")


@pytest.fixture(scope="module")
def fake_venv(tmp_path_factory):
    """A real venv (no pip) with the fake compiler above as its `ttl` package, and a dist-info that names it tt-lang 9.9.9."""
    root = tmp_path_factory.mktemp("fakechip") / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root)], check=True, timeout=120)
    site = next((root / "lib").glob("python3.*/site-packages"))
    (site / "ttl" / "_src").mkdir(parents=True)
    (site / "ttl" / "__init__.py").write_text(FAKE_TTL, encoding="utf-8")
    (site / "ttl" / "_src" / "ttl_ast.py").write_text("# the fake compiler\n", encoding="utf-8")
    (site / "tt_lang-9.9.9.dist-info").mkdir()
    (site / "tt_lang-9.9.9.dist-info" / "METADATA").write_text("Metadata-Version: 2.4\nName: tt-lang\nVersion: 9.9.9\n\nbody\n", encoding="utf-8")
    return root


def py(venv: Path) -> str:
    return str(venv / "bin" / "python")


def kernel(tmp_path: Path, body: str, name: str = "kernel.py") -> Path:
    path = tmp_path / name
    path.write_text("import ttl\nimport ttnn\n\n\n@ttl.operation(grid=(1, 1))\ndef op(x, out):\n" + textwrap.indent(textwrap.dedent(body), "    "),
                    encoding="utf-8")
    return path


SHAPES = [(64, 256), (64, 256)]


# ---- which compiler -----------------------------------------------------------------------------------------------------------------

def test_identity_names_version_and_compiler_bytes(fake_venv):
    ident = hc.package_identity(py(fake_venv))
    assert ident["package"] == "tt-lang" and ident["version"] == "9.9.9"
    assert len(ident["compiler_sha256"]) == 64 and ident["files"] == ["_src/ttl_ast.py"]
    assert hc.identity_problems(ident) == []
    assert hc.target_label(ident) == "tt-lang 9.9.9"


def test_identity_changes_when_the_compiler_bytes_change(fake_venv):
    before = hc.package_identity(py(fake_venv))
    ast_file = next((fake_venv / "lib").glob("python3.*/site-packages/ttl/_src/ttl_ast.py"))
    original = ast_file.read_text(encoding="utf-8")
    try:
        ast_file.write_text(original + "# another compiler\n", encoding="utf-8")
        after = hc.package_identity(py(fake_venv))
    finally:
        ast_file.write_text(original, encoding="utf-8")
    assert after["version"] == before["version"] and after["compiler_sha256"] != before["compiler_sha256"]
    assert not hc.same_target(before, after)
    assert hc.same_target(before, hc.package_identity(py(fake_venv)))


def test_identity_refuses_a_venv_without_the_compiler(tmp_path):
    (tmp_path / "v" / "bin").mkdir(parents=True)
    with pytest.raises(hc.HwCompileError, match="no ttl package"):
        hc.package_identity(tmp_path / "v" / "bin" / "python")
    site = tmp_path / "v" / "lib" / "python3.12" / "site-packages"
    (site / "ttl" / "_src").mkdir(parents=True)
    (site / "ttl" / "_src" / "ttl_ast.py").write_text("x\n", encoding="utf-8")
    with pytest.raises(hc.HwCompileError, match="no tt_lang distribution"):
        hc.package_identity(tmp_path / "v" / "bin" / "python")          # a simulator wheel has no tt_lang dist-info
    (site / "tt_lang_sim-1.0.dist-info").mkdir()
    (site / "tt_lang_sim-1.0.dist-info" / "METADATA").write_text("Version: 1.0\n", encoding="utf-8")
    with pytest.raises(hc.HwCompileError, match="no tt_lang distribution"):
        hc.package_identity(tmp_path / "v" / "bin" / "python")


@pytest.mark.parametrize("metadata", ["", "Name: tt-lang\n", "Version: \n", "Version: a b\n", "\nVersion: 1.0\n", "Version: " + "9" * 200 + "\n"])
def test_identity_refuses_a_metadata_file_without_a_plain_version(tmp_path, metadata):
    site = tmp_path / "v" / "lib" / "python3.12" / "site-packages"
    (site / "ttl" / "_src").mkdir(parents=True)
    (site / "ttl" / "_src" / "ttl_ast.py").write_text("x\n", encoding="utf-8")
    (site / "tt_lang-1.0.dist-info").mkdir()
    (site / "tt_lang-1.0.dist-info" / "METADATA").write_text(metadata, encoding="utf-8")
    with pytest.raises(hc.HwCompileError):
        hc.package_identity(tmp_path / "v" / "bin" / "python")


@pytest.mark.parametrize("bad", [None, [], "x", 3, {}, {"package": "tt-lang", "version": 3, "compiler_sha256": "0" * 64},
                                 {"package": "ttnn", "version": "1", "compiler_sha256": "0" * 64},
                                 {"package": "tt-lang", "version": "1", "compiler_sha256": "0" * 63},
                                 {"package": "tt-lang", "version": "1", "compiler_sha256": None}])
def test_a_malformed_identity_is_a_problem_and_matches_nothing(bad):
    good = {"package": "tt-lang", "version": "1", "compiler_sha256": "0" * 64}
    assert hc.identity_problems(bad)
    assert not hc.same_target(bad, good) and not hc.same_target(good, bad) and not hc.same_target(bad, bad)


# ---- which function -----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    ("import ttl\n@ttl.operation(grid=(1, 1))\ndef a(x): pass\n", "a"),
    ("import ttl\n@ttl.operation\ndef a(x): pass\n", "a"),
    ("import ttl\ndef a(x): pass\n", None),
    ("import ttl\n@ttl.operation()\ndef a(x): pass\n@ttl.operation()\ndef b(x): pass\n", None),
    ("import ttl\n@other.operation()\ndef a(x): pass\n", None),
    ("def a(:\n", None),
    ("x = '\0'\n", None),
])
def test_operation_name(text, want):
    assert hc.operation_name(text) == want


# ---- the check, with the fake compiler ----------------------------------------------------------------------------------------------

@needs_landlock
def test_a_kernel_the_compiler_accepts_is_compiled_with_its_evidence(fake_venv, tmp_path):
    k = kernel(tmp_path, 'ttl.emit("compute")\nttl.emit("read")\n')
    rec = hc.compile_check(py(fake_venv), k, SHAPES)
    assert rec["category"] == "compiled" and rec["compiled"] is True
    assert rec["target"]["version"] == "9.9.9"
    assert rec["kernel_sha256"] == hashlib.sha256(k.read_bytes()).hexdigest()
    assert rec["operation"] == "op" and rec["shapes"] == [[64, 256], [64, 256]]
    assert [s["name"] for s in rec["kernel_sources"]] == ["ttlang_kernel_compute_0.cpp", "ttlang_kernel_read_0.cpp"]
    assert rec["kernel_sources"][0]["sha256"] == hashlib.sha256(b"// kernel").hexdigest()
    assert rec["guard"] == [{"path": "/dev/tenstorrent", "state": "refused" if os.path.isdir("/dev/tenstorrent") else "absent"}]
    assert hc.record_problems(rec) == []


@needs_landlock
def test_a_kernel_the_compiler_refuses_is_a_compile_error_with_its_first_line(fake_venv, tmp_path):
    k = kernel(tmp_path, 'raise ttl.CompileError("error: ttl.math.fill is not available; use ttl.block.fill")\n')
    rec = hc.compile_check(py(fake_venv), k, SHAPES)
    assert rec["category"] == "compile_error" and rec["compiled"] is False and rec["kernel_sources"] == []
    assert "ttl.math.fill is not available" in rec["first_line"]
    assert hc.record_problems(rec) == []


@needs_landlock
def test_a_call_that_returns_without_a_kernel_source_is_not_a_pass(fake_venv, tmp_path):
    rec = hc.compile_check(py(fake_venv), kernel(tmp_path, "pass\n"), SHAPES)
    assert rec["category"] == "no_kernel_written" and rec["compiled"] is False


@needs_landlock
def test_the_first_line_does_not_carry_the_scratch_path(fake_venv, tmp_path):
    work = tmp_path / "work"
    k = kernel(tmp_path, "raise ttl.CompileError('error at ' + __file__)\n")
    rec = hc.compile_check(py(fake_venv), k, SHAPES, work_dir=work)
    assert rec["category"] == "compile_error" and str(work) not in rec["first_line"] and "<work>/kernel.py" in rec["first_line"]


@needs_landlock
def test_the_first_line_is_the_same_on_every_run(fake_venv, tmp_path):
    k = kernel(tmp_path, "raise ttl.CompileError('error: %r not implemented' % object())\n")
    first, second = (hc.compile_check(py(fake_venv), k, SHAPES)["first_line"] for _ in range(2))
    assert first == second and first.endswith("error: <object object at 0x...> not implemented")


@needs_landlock
def test_the_denied_directory_cannot_be_read_or_listed_by_the_compiler(fake_venv, tmp_path):
    """The guard, on a directory of the test's own in place of /dev/tenstorrent: the compiler's process is refused the listing and the
    file, and so is a process it starts."""
    denied = tmp_path / "denied"
    denied.mkdir()
    (denied / "node").write_text("SENTINEL", encoding="utf-8")
    body = f'''
        import os, subprocess
        facts = []
        for attempt in (lambda: os.listdir({str(denied)!r}), lambda: open({str(denied / "node")!r}).read()):
            try:
                facts.append("READ " + str(attempt()))
            except PermissionError:
                facts.append("refused")
        facts.append("child-" + str(subprocess.run(["/bin/cat", {str(denied / "node")!r}], capture_output=True).returncode))
        raise ttl.CompileError("facts: " + ", ".join(facts))
    '''
    rec = hc.compile_check(py(fake_venv), kernel(tmp_path, body), SHAPES, deny=[denied])
    assert rec["first_line"].endswith("facts: refused, refused, child-1"), rec["first_line"]
    assert "SENTINEL" not in json.dumps(rec)
    assert rec["guard"] == [{"path": str(denied), "state": "refused"}]


@needs_landlock
def test_files_outside_the_denied_directory_stay_readable(fake_venv, tmp_path):
    (tmp_path / "plain.txt").write_text("ok", encoding="utf-8")
    denied = tmp_path / "denied"
    denied.mkdir()
    body = f'ttl.emit("compute", open({str(tmp_path / "plain.txt")!r}).read())\n'
    rec = hc.compile_check(py(fake_venv), kernel(tmp_path, body), SHAPES, deny=[denied])
    assert rec["compiled"] is True and rec["kernel_sources"][0]["sha256"] == hashlib.sha256(b"ok").hexdigest()


@needs_landlock
def test_ttnn_in_the_kernel_is_the_stand_in_and_what_is_asked_of_it_is_recorded(fake_venv, tmp_path):
    body = '''
        assert type(x).__name__ == "StandInTensor" and ttnn.Tensor is type(x), "not the stand-in"
        assert x.shape == (64, 256) and "BFLOAT16" in str(x.dtype)
        ttnn.open_device(device_id=0)
        ttl.emit("compute")
    '''
    rec = hc.compile_check(py(fake_venv), kernel(tmp_path, body), SHAPES)
    assert rec["compiled"] is True, rec["first_line"]
    assert rec["stand_in_used"] == ["ttnn.open_device"]


@needs_landlock
def test_the_grid_reaches_the_stand_in_device(fake_venv, tmp_path):
    body = 'g = x.device().compute_with_storage_grid_size()\nraise ttl.CompileError("grid %d %d" % (g.x, g.y))\n'
    assert hc.compile_check(py(fake_venv), kernel(tmp_path, body), SHAPES)["first_line"].endswith("grid 13 10")
    assert hc.compile_check(py(fake_venv), kernel(tmp_path, body), SHAPES, grid=(2, 3))["first_line"].endswith("grid 2 3")


def test_the_compiler_reads_a_private_copy_with_the_recorded_hash_in_a_small_environment(fake_venv, tmp_path):
    k = kernel(tmp_path, "pass\n")
    seen = {}

    def run(argv, **kw):
        copy = Path(argv[argv.index("--kernel") + 1])
        seen.update(argv=argv, env=kw["env"], copy=copy.read_bytes(), same_file=copy.samefile(k))
        k.write_text("# swapped while the compiler ran\n", encoding="utf-8")
        Path(argv[argv.index("--result") + 1]).write_text(json.dumps({
            "schema": hc.CHILD_SCHEMA, "compiled": True, "kernels": [{"name": "a.cpp", "bytes": 1, "sha256": "0" * 64}],
            "stand_in_used": [], "guard": [{"path": "/dev/tenstorrent", "state": "refused"}], "error_type": None, "error_first_line": ""}))

    original = k.read_bytes()
    rec = hc.compile_check(py(fake_venv), k, SHAPES, run=run)
    assert rec["compiled"] is True and rec["kernel_sha256"] == hashlib.sha256(original).hexdigest()
    assert seen["copy"] == original and not seen["same_file"]
    assert set(seen["env"]) == {"PATH", "HOME", "LANG", "TTLANG_COMPILE_ONLY"}
    assert seen["argv"][:5] == ["nice", "-n", "10", py(fake_venv), "-I"]


def _child_result(**over) -> dict:
    base = {"schema": hc.CHILD_SCHEMA, "compiled": True, "kernels": [{"name": "a.cpp", "bytes": 1, "sha256": "0" * 64}], "stand_in_used": [],
            "guard": [{"path": "/dev/tenstorrent", "state": "refused"}], "error_type": None, "error_first_line": ""}
    base.update(over)
    return base


def _run_writing(payload):
    def run(argv, **kw):
        if payload is not None:
            Path(argv[argv.index("--result") + 1]).write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return run


@pytest.mark.parametrize("payload,category", [
    (None, "bad_result"),
    ("not json", "bad_result"),
    ("[]", "bad_result"),
    ({"schema": "other"}, "bad_result"),
    (_child_result(compiled="yes"), "bad_result"),
    (_child_result(compiled=1), "bad_result"),
    (_child_result(kernels=None), "bad_result"),
    (_child_result(kernels=[{"name": "a.cpp"}]), "bad_result"),
    (_child_result(kernels=[{"name": "a.cpp", "bytes": 1, "sha256": "zz"}]), "bad_result"),
    (_child_result(guard="refused"), "bad_result"),
    (_child_result(guard=[{"path": "/dev/tenstorrent", "state": "open"}]), "bad_result"),
    (_child_result(stand_in_used=[3]), "bad_result"),
    (_child_result(error_first_line=["x"]), "bad_result"),
    (_child_result(guard=[]), "guard_failed"),
    (_child_result(compiled=False, kernels=[], guard=[], error_type="GuardError", error_first_line="Landlock is not available"), "guard_failed"),
    (_child_result(kernels=[]), "no_kernel_written"),
    (_child_result(compiled=False, kernels=[], error_type="no_kernel_written"), "no_kernel_written"),
    (_child_result(compiled=False, kernels=[], error_type="TypeError", error_first_line="TypeError: x"), "compile_error"),
    (_child_result(), "compiled"),
])
def test_only_a_proved_guard_and_a_written_kernel_make_a_pass(fake_venv, tmp_path, payload, category):
    rec = hc.compile_check(py(fake_venv), kernel(tmp_path, "pass\n"), SHAPES, run=_run_writing(payload))
    assert rec["category"] == category and rec["compiled"] is (category == "compiled")
    assert hc.record_problems(rec) == []


def test_a_timeout_and_an_interpreter_that_does_not_start_are_not_compile_errors(fake_venv, tmp_path):
    def slow(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    def missing(argv, **kw):
        raise FileNotFoundError("no such interpreter")
    k = kernel(tmp_path, "pass\n")
    assert hc.compile_check(py(fake_venv), k, SHAPES, run=slow, timeout_s=7)["category"] == "timeout"
    assert hc.compile_check(py(fake_venv), k, SHAPES, run=missing)["category"] == "not_run"


def test_a_stale_result_file_is_not_read_as_this_run(fake_venv, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "result.json").write_text(json.dumps(_child_result()), encoding="utf-8")
    rec = hc.compile_check(py(fake_venv), kernel(tmp_path, "pass\n"), SHAPES, work_dir=work, run=_run_writing(None))
    assert rec["category"] == "bad_result"


def test_a_kernel_file_that_cannot_be_used_is_not_run(fake_venv, tmp_path):
    def never(argv, **kw):
        raise AssertionError("the compiler was started")
    assert hc.compile_check(py(fake_venv), tmp_path / "absent.py", SHAPES, run=never)["category"] == "not_run"
    two = tmp_path / "two.py"
    two.write_text("import ttl\n@ttl.operation()\ndef a(x): pass\n@ttl.operation()\ndef b(x): pass\n", encoding="utf-8")
    assert hc.compile_check(py(fake_venv), two, SHAPES, run=never)["category"] == "not_run"
    big = tmp_path / "big.py"
    big.write_bytes(b"#" * (hc.MAX_KERNEL_BYTES + 1))
    assert hc.compile_check(py(fake_venv), big, SHAPES, run=never)["category"] == "not_run"
    os.mkfifo(tmp_path / "fifo.py")
    assert hc.compile_check(py(fake_venv), tmp_path / "fifo.py", SHAPES, run=never)["category"] == "not_run"


@pytest.mark.parametrize("shapes", [None, [], "64x64", [[64]], [[64, 0]], [[64, -1]], [[64, 1.5]], [[True, 64]], [[64, 2 ** 21]], [["64", "64"]],
                                    [[64, 64]] * 17])
def test_shapes_of_the_wrong_type_are_refused_before_anything_runs(fake_venv, tmp_path, shapes):
    with pytest.raises(hc.HwCompileError):
        hc.compile_check(py(fake_venv), kernel(tmp_path, "pass\n"), shapes, run=_run_writing(_child_result()))


def test_spec_shapes_are_the_inputs_then_the_output():
    assert hc.spec_shapes({"inputs": {"x": [64, 256], "w": [32, 256]}, "output": [64, 256]}) == [(64, 256), (32, 256), (64, 256)]
    for bad in (None, {}, {"inputs": []}, {"inputs": {"x": [64, 256]}}, {"inputs": {"x": "64"}, "output": [1, 1]}):
        with pytest.raises(hc.HwCompileError):
            hc.spec_shapes(bad)


@pytest.mark.parametrize("change", [
    {"schema": "x"}, {"category": "passed"}, {"compiled": "true"}, {"compiled": False}, {"kernel_sha256": None}, {"first_line": 3},
    {"first_line": "x" * 301}, {"kernel_sources": []}, {"kernel_sources": None}, {"guard": []}, {"guard": [{"state": "open"}]},
    {"target": None}, {"target": {"package": "tt-lang", "version": "1", "compiler_sha256": "short"}},
])
def test_a_compiled_record_with_a_missing_or_wrong_part_is_a_problem(change):
    good = {"schema": hc.SCHEMA, "category": "compiled", "compiled": True,
            "target": {"package": "tt-lang", "version": "1.0", "compiler_sha256": "0" * 64}, "kernel_sha256": "1" * 64, "operation": "op",
            "shapes": [[64, 64]], "first_line": "", "kernel_sources": [{"name": "a.cpp", "bytes": 1, "sha256": "2" * 64}],
            "stand_in_used": [], "grid": [13, 10], "arch": "blackhole", "guard": [{"path": "/dev/tenstorrent", "state": "refused"}]}
    assert hc.record_problems(good) == []
    assert hc.record_problems({**good, **change})


@pytest.mark.parametrize("bad", [None, [], "x", 1, {}])
def test_a_record_of_the_wrong_type_is_a_problem(bad):
    assert hc.record_problems(bad)


# ---- the child's guard, directly ----------------------------------------------------------------------------------------------------

def test_the_proof_fails_when_a_denied_directory_can_be_listed(tmp_path):
    """Without the guard a listing succeeds, and the proof must say so: this is what stops the child before tt-lang is imported."""
    with pytest.raises(child_mod.GuardError, match="can still be listed"):
        child_mod.prove_denied([str(tmp_path)])
    assert child_mod.prove_denied([str(tmp_path / "absent")]) == [{"path": str(tmp_path / "absent"), "state": "absent"}]


@needs_landlock
def test_the_child_itself_does_not_call_a_run_without_a_kernel_source_compiled(fake_venv, tmp_path):
    """The parent has the same rule (a pass needs a kernel source). This is the child's own, read from its result file."""
    results = {}
    for name, body in (("silent", "pass\n"), ("good", 'ttl.emit("compute")\n')):
        result = tmp_path / f"{name}.json"
        done = subprocess.run([py(fake_venv), "-I", str(REPO / "orchard" / "ttlang_compile_child.py"), "--kernel", str(kernel(tmp_path, body, f"{name}.py")),
                               "--fn", "op", "--shape", "64x256", "--shape", "64x256", "--result", str(result)],
                              capture_output=True, text=True, timeout=120, env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)})
        results[name] = (done.returncode, json.loads(result.read_text(encoding="utf-8")))
    assert results["silent"][0] == 1 and results["silent"][1]["compiled"] is False and results["silent"][1]["error_type"] == "no_kernel_written"
    assert results["good"][0] == 0 and results["good"][1]["compiled"] is True and len(results["good"][1]["kernels"]) == 1
    assert results["good"][1]["guard"] and results["good"][1]["stage"] == "done"


def test_the_root_cannot_be_the_denied_directory():
    code = "import sys; sys.path.insert(0, %r); import ttlang_compile_child as c\ntry:\n    c.deny_beneath(['/'])\nexcept c.GuardError as e:\n    print('refused', e)" % str(REPO / "orchard")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.startswith("refused"), out.stderr


def test_the_child_imports_nothing_from_orchard_and_names_no_real_ttnn_import():
    import ast
    tree = ast.parse((REPO / "orchard" / "ttlang_compile_child.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"argparse", "ctypes", "enum", "hashlib", "importlib", "json", "os", "stat", "sys", "traceback", "types", "ttl"}
    assert "ttnn" not in imported and "orchard" not in imported


@needs_landlock
def test_the_child_installs_the_guard_before_it_imports_the_compiler(fake_venv, tmp_path):
    """A compiler package that reads the denied directory at import time is refused too: the guard comes first."""
    site = next((fake_venv / "lib").glob("python3.*/site-packages"))
    denied = tmp_path / "denied"
    denied.mkdir()
    (denied / "node").write_text("SENTINEL", encoding="utf-8")
    init = site / "ttl" / "__init__.py"
    original = init.read_text(encoding="utf-8")
    try:
        init.write_text(original + f"\nSEEN = open({str(denied / 'node')!r}).read()\n", encoding="utf-8")
        rec = hc.compile_check(py(fake_venv), kernel(tmp_path, 'ttl.emit("compute")\n'), SHAPES, deny=[denied],
                               identity={"package": "tt-lang", "version": "9.9.9", "compiler_sha256": "0" * 64})
    finally:
        init.write_text(original, encoding="utf-8")
    assert rec["category"] == "compile_error" and "PermissionError" in rec["first_line"]


# ---- the chip packages of this machine, when they are installed ---------------------------------------------------------------------

SHARED = Path.home() / ".tenstorrent-venv" / "bin" / "python"
PAIR = Path(os.environ.get("ORCHARD_PAIR_VENV") or Path.home() / ".cache" / "tt-orchard" / "ttlang-pair" / "venv") / "bin" / "python"
EVIDENCE = REPO / "docs" / "run-logs" / "evidence" / "ttlang-pair" / "kernels"


def _has(python: Path, version_prefix: str) -> bool:
    try:
        return hc.package_identity(python)["version"].startswith(version_prefix)
    except hc.HwCompileError:
        return False


on_116 = pytest.mark.skipif(not (landlock_available() and _has(SHARED, "1.1.6")), reason="tt-lang 1.1.6 is not installed in the shared venv")
on_pair = pytest.mark.skipif(not (landlock_available() and _has(PAIR, "1.1.10")), reason="the tt-lang pair venv is not installed")

FILL_KERNEL = '''import ttl
import ttnn


@ttl.operation(grid=(1, 1))
def scale(x, out):
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.datamovement()
    def read():
        with x_dfb.reserve() as x_blk:
            tx = ttl.copy(x[0:1, 0:1], x_blk)
            tx.wait()

    @ttl.compute()
    def compute():
        with x_dfb.wait() as x_blk, out_dfb.reserve() as out_blk:
            out_blk.store(x_blk * ttl.NAMESPACE.fill(0.5, shape=(1, 1)))

    @ttl.datamovement()
    def write():
        with out_dfb.wait() as out_blk:
            tx = ttl.copy(out_blk, out[0:1, 0:1])
            tx.wait()
'''


@pytest.mark.parametrize("python", [pytest.param(SHARED, marks=on_116, id="1.1.6"), pytest.param(PAIR, marks=on_pair, id="pair")])
def test_both_chip_compilers_take_block_fill_and_refuse_math_fill(python, tmp_path):
    """The claim the cheat sheet had the wrong way round. Seen on this machine on 2026-11-22 with the real compilers."""
    block, math = tmp_path / "block.py", tmp_path / "math.py"
    block.write_text(FILL_KERNEL.replace("NAMESPACE", "block"), encoding="utf-8")
    math.write_text(FILL_KERNEL.replace("NAMESPACE", "math"), encoding="utf-8")
    good = hc.compile_check(python, block, [(32, 32), (32, 32)])
    assert good["category"] == "compiled" and len(good["kernel_sources"]) == 3, good["first_line"]
    assert good["guard"] and all(g["state"] in ("refused", "absent") for g in good["guard"])
    bad = hc.compile_check(python, math, [(32, 32), (32, 32)])
    assert bad["category"] == "compile_error" and "ttl.math.fill is not available; use ttl.block.fill" in bad["first_line"]


@on_116
def test_tt_lang_116_refuses_the_shape_keyword_as_the_chip_run_recorded():
    rec = hc.compile_check(SHARED, EVIDENCE / "library_rmsnorm.py", [(64, 128)] * 3)
    assert rec["category"] == "compile_error" and "reduce_sum() got an unexpected keyword argument 'shape'" in rec["first_line"]


@on_pair
def test_the_pair_compiles_the_kernel_that_ran_on_the_chip():
    rec = hc.compile_check(PAIR, EVIDENCE / "library_rmsnorm.py", [(64, 128)] * 3)
    assert rec["category"] == "compiled" and len(rec["kernel_sources"]) == 3, rec["first_line"]
