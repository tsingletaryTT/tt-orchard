# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The sandbox for model-written code (orchard/lm_eval_tasks/code_sandbox.py), checked with real child processes.

Each test that shows a refusal also shows the same program succeeding without the sandbox, so a program that only fails for another reason
cannot pass as a refusal. Every program runs in a child that limits itself (CPU, memory, file size) and has a timeout, and none of them
touches a chip or the operator's files: the "token" is a file the test plants in its own temporary directory."""
from __future__ import annotations

import importlib.util
import socket
import threading
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parent.parent / "orchard" / "lm_eval_tasks"
spec = importlib.util.spec_from_file_location("code_sandbox_under_test", DIR / "code_sandbox.py")
box = importlib.util.module_from_spec(spec)
spec.loader.exec_module(box)

needs_landlock = pytest.mark.skipif(not box.landlock_abi(), reason="SKIPPED: this kernel has no Landlock")


@pytest.fixture
def planted(tmp_path):
    f = tmp_path / "hf-token"
    f.write_text("hf_planted_token\n")
    return f


@pytest.fixture
def listener():
    """A loopback server in this process; `.connections` counts what reached it."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    srv.settimeout(0.2)
    state = {"connections": 0, "stop": False, "port": srv.getsockname()[1]}

    def serve():
        while not state["stop"]:
            try:
                c, _ = srv.accept()
                state["connections"] += 1
                c.close()
            except OSError:
                pass
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    yield state
    state["stop"] = True
    t.join(timeout=2)
    srv.close()


def passed(program, **kw):
    return box.run_program(program, timeout_s=15, **kw).status == "passed"


@needs_landlock
def test_the_self_check_passes_here_and_names_what_it_refused():
    got = box.self_check()
    assert got["ok"] is True, got
    assert set(got["seen"]) >= {"read_planted_file", "tcp_socket", "udp_socket", "unix_socket", "write_beside_planted_file", "read_proc_environ"}
    assert all(v.startswith("blocked") for v in got["seen"].values())


@needs_landlock
def test_code_in_the_sandbox_cannot_read_a_planted_token_file(planted):
    program = f"assert open({str(planted)!r}).read().startswith('hf_')\n"
    assert passed(program, unsandboxed=True), "the control must be able to read the file"
    out = box.run_program(program, timeout_s=15)
    assert out.status == "failed" and "PermissionError" in out.detail


@needs_landlock
def test_code_in_the_sandbox_cannot_open_a_socket_or_reach_a_loopback_server(listener):
    program = f"import socket\ns = socket.create_connection(('127.0.0.1', {listener['port']}), timeout=3)\ns.close()\n"
    assert passed(program, unsandboxed=True)
    assert listener["connections"] == 1
    out = box.run_program(program, timeout_s=15)
    assert out.status == "failed" and "PermissionError" in out.detail
    assert listener["connections"] == 1, "the sandboxed program reached the server"


@needs_landlock
@pytest.mark.parametrize("family,kind", [("AF_INET", "SOCK_DGRAM"), ("AF_INET6", "SOCK_STREAM"), ("AF_UNIX", "SOCK_STREAM")])
def test_no_kind_of_socket_can_be_created(family, kind):
    program = f"import socket\nsocket.socket(socket.{family}, socket.{kind}).close()\n"
    assert passed(program, unsandboxed=True)
    assert box.run_program(program, timeout_s=15).status == "failed"


@needs_landlock
def test_io_uring_which_can_open_sockets_without_socket_is_refused():
    program = ("import ctypes\nlibc = ctypes.CDLL(None, use_errno=True)\nr = libc.syscall(425, 4, 0)\n"
               "assert r == -1\nassert ctypes.get_errno() == 1, ctypes.get_errno()\n")      # EPERM; unrestricted it is EFAULT (a null params pointer)
    assert not passed(program, unsandboxed=True)
    assert passed(program)


@needs_landlock
def test_the_scratch_directory_is_writable_and_nothing_beside_the_token_is(planted):
    assert passed("open('scratch.txt', 'w').write('x')\nassert open('scratch.txt').read() == 'x'\n")
    outside = f"open({str(planted.with_name('new.txt'))!r}, 'w').write('x')\n"
    assert passed(outside, unsandboxed=True)
    planted.with_name("new.txt").unlink()
    assert box.run_program(outside, timeout_s=15).status == "failed"
    assert not planted.with_name("new.txt").exists()


@needs_landlock
def test_the_program_gets_no_credentials_and_no_operator_home(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_in_environment")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    program = ("import os\nassert 'HF_TOKEN' not in os.environ and 'SSH_AUTH_SOCK' not in os.environ\n"
               "assert os.environ['HOME'] == os.getcwd() and os.environ['HOME'].startswith('/tmp/orchard-code-')\n")
    assert passed(program)


def test_an_ordinary_program_passes_and_a_failing_one_does_not():
    assert passed("assert 1 + 1 == 2\n")
    assert box.run_program("assert False\n", timeout_s=15).status == "failed"
    assert box.run_program("import sys\nsys.exit(3)\n", timeout_s=15).status == "failed"
    assert passed("import sys\nsys.exit(0)\n")
    assert box.run_program("def (:\n", timeout_s=15).status == "failed"            # a syntax error fails the program


def test_a_program_that_runs_too_long_is_killed_and_reported_as_a_timeout():
    out = box.run_program("while True:\n    pass\n", timeout_s=1)
    assert out.status == "timeout"


def test_a_sleeping_program_is_killed_at_the_timeout_not_left_to_finish():
    """A busy loop is also stopped by the CPU limit, so only a program that uses no CPU shows that the timeout itself kills."""
    import time
    t0 = time.monotonic()
    out = box.run_program("import time\ntime.sleep(60)\n", timeout_s=1)
    assert out.status == "timeout" and time.monotonic() - t0 < 20


def test_a_child_that_cannot_restrict_itself_reports_sandbox_unavailable_and_runs_nothing(tmp_path):
    """A stand-in interpreter that exits as the real one does when Landlock is missing: the caller must not read that as a result."""
    marker = tmp_path / "ran"
    stub = tmp_path / "python"
    stub.write_text(f"#!/bin/sh\necho 'SANDBOX UNAVAILABLE: no landlock' >&2\nexit {box.SANDBOX_UNAVAILABLE}\n")
    stub.chmod(0o755)
    out = box.run_program(f"open({str(marker)!r}, 'w').write('x')\n", timeout_s=15, python=str(stub))
    assert out.status == "sandbox_unavailable" and "no landlock" in out.detail and not marker.exists()


def test_a_kernel_without_landlock_refuses_to_restrict(monkeypatch):
    monkeypatch.setattr(box, "landlock_abi", lambda: 0)
    with pytest.raises(box.SandboxUnavailable, match="no Landlock"):
        box.apply_landlock(["/usr"], [])


def test_the_real_child_exits_with_the_unavailable_status_and_runs_nothing_when_it_cannot_restrict(tmp_path):
    """The child itself, with Landlock made unavailable inside it: it must exit 97 and never reach the program."""
    import subprocess
    import sys
    marker = tmp_path / "ran"
    work = tmp_path / "work"
    work.mkdir()
    code = (f"import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('s', {str(DIR / 'code_sandbox.py')!r})\n"
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            f"m.landlock_abi = lambda: 0\n"
            f"sys.exit(m.child_main([{str(work)!r}, '5']))\n")
    got = subprocess.run([sys.executable, "-I", "-c", code], input=f"open({str(marker)!r}, 'w').write('x')\n".encode(),
                         capture_output=True, timeout=30)
    assert got.returncode == box.SANDBOX_UNAVAILABLE and b"no Landlock" in got.stderr and not marker.exists()


def test_the_self_check_fails_when_the_sandbox_is_unavailable(monkeypatch):
    monkeypatch.setattr(box, "run_program", lambda *a, **kw: box.Outcome("sandbox_unavailable", "no landlock"))
    got = box.self_check()
    assert got["ok"] is False and "could not be set up" in got["detail"]


def test_the_self_check_fails_when_the_probe_was_allowed_something(monkeypatch):
    report = '{"read_planted_file": "allowed", "tcp_socket": "blocked: PermissionError"}'
    monkeypatch.setattr(box, "run_program", lambda *a, **kw: box.Outcome("failed", report))
    got = box.self_check()
    assert got["ok"] is False and "read_planted_file" in got["detail"]


def test_the_self_check_fails_when_the_probe_reports_nothing(monkeypatch):
    monkeypatch.setattr(box, "run_program", lambda *a, **kw: box.Outcome("passed", ""))
    assert box.self_check()["ok"] is False


def test_the_seccomp_program_lists_socket_and_io_uring_and_kills_a_foreign_architecture():
    rows = box.seccomp_program("x86_64")
    ks = [r[3] for r in rows]
    assert 41 in ks and {425, 426, 427} <= set(ks)
    assert rows[0] == (box.BPF_LD_W_ABS, 0, 0, 4) and rows[2] == (box.BPF_RET_K, 0, 0, 0x80000000)      # SECCOMP_RET_KILL_PROCESS
    assert (box.BPF_RET_K, 0, 0, 0x00050001) in rows                                                 # SECCOMP_RET_ERRNO | EPERM
    assert rows[-1] == (box.BPF_RET_K, 0, 0, box.RET_ALLOW)
    assert 198 in [r[3] for r in box.seccomp_program("aarch64")]
    with pytest.raises(box.SandboxUnavailable, match="riscv64"):
        box.seccomp_program("riscv64")


@needs_landlock
def test_the_landlock_tcp_rules_alone_refuse_a_connection(listener):
    """Without the seccomp filter, so the network rules are tested on their own (they exist from Landlock ABI 4)."""
    if box.landlock_abi() < 4:
        pytest.skip("SKIPPED: Landlock ABI below 4 has no network rules")
    import subprocess
    import sys
    code = (f"import importlib.util, socket, sys\n"
            f"spec = importlib.util.spec_from_file_location('s', {str(DIR / 'code_sandbox.py')!r})\n"
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            f"m.apply_landlock(list(m.SYSTEM_READ) + m.python_prefixes(), [])\n"
            f"try:\n    socket.create_connection(('127.0.0.1', {listener['port']}), timeout=3)\n    sys.exit(0)\n"
            f"except OSError as e:\n    sys.exit(5 if isinstance(e, PermissionError) else 6)\n")
    got = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, timeout=30)
    assert got.returncode == 5, got.stderr[-300:]
    assert listener["connections"] == 0
