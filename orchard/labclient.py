# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The brain's side of a lab box: the connection to the lab helper (orchard/lab.py), a gozer adapter
that leases the lab's chips through it, and rsync for the run's files.

This is supervisor code. Agents never reach the lab: the command runner still refuses ssh, scp and
rsync in anything an agent runs (orchard/runner.py), and the hardware test command an agent writes is
checked on the brain before the supervisor hands it to the lab.

The lab root has the same absolute path on both boxes, so `Sync` copies a path to the same path.
"""
from __future__ import annotations

import base64
import itertools
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
from pathlib import Path

from orchard.adapters.gozer import GozerAdapter
from orchard.commands import CommandResult, run_command

SSH = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]
CALL_TIMEOUT_S = 120.0
SYNC_TIMEOUT_S = 4 * 3600.0          # the first copy of a model's weights can be tens of GB
CHECKOUT = Path(__file__).resolve().parent.parent


class LabError(Exception):
    """The lab helper refused a request, failed it, or went away."""


def ssh_launch_argv(*, host, root, gozer="gozer", path=(), python="python3") -> list[str]:
    """The ssh command that starts the helper from the checkout synced to <root>/orchard."""
    remote = (f"cd {shlex.quote(str(root) + '/orchard')} && exec {shlex.quote(python)} -m orchard.lab serve "
              f"--root {shlex.quote(str(root))} --gozer {shlex.quote(gozer)} --path {shlex.quote(':'.join(path))}")
    return [*SSH, host, remote]


class Lab:
    """One connection to a lab helper. Requests can overlap; each waits for its own reply."""

    def __init__(self, argv, *, env=None, cwd=None, host=None):
        self.host = host
        self.proc = subprocess.Popen(list(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env, cwd=cwd, start_new_session=True)
        self._ids = itertools.count(1)
        self._queues: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self.hello = self.call("hello")

    @classmethod
    def ssh(cls, *, host, root, gozer="gozer", path=(), python="python3") -> "Lab":
        return cls(ssh_launch_argv(host=host, root=root, gozer=gozer, path=path, python=python), host=host)

    @classmethod
    def local(cls, *, root, gozer="gozer", path=(), env=None) -> "Lab":
        """A helper on this machine, for tests and for a brain that is its own lab."""
        full = {**os.environ, **(env or {})}
        full["PYTHONPATH"] = os.pathsep.join(filter(None, [str(CHECKOUT), full.get("PYTHONPATH")]))
        return cls([sys.executable, "-m", "orchard.lab", "serve", "--root", str(root), "--gozer", str(gozer),
                    "--path", ":".join(path)], env=full)

    # ---- plumbing ----
    def _read(self) -> None:
        for raw in self.proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            q = self._queues.get(msg.get("id"))
            if q is not None:
                q.put(msg)
        self._closed = True
        for q in list(self._queues.values()):
            q.put({"ok": False, "error": "the lab helper went away"})

    def _send(self, op: str, **fields) -> tuple[int, queue.Queue]:
        rid = next(self._ids)
        q: queue.Queue = queue.Queue()
        self._queues[rid] = q
        line = (json.dumps({"id": rid, "op": op, **fields}) + "\n").encode()
        try:
            with self._lock:
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            self._queues.pop(rid, None)
            raise LabError(f"the lab helper went away: {exc}") from exc
        return rid, q

    def _final(self, rid: int, q: queue.Queue, timeout: float, on_chunk=None) -> dict:
        try:
            while True:
                msg = q.get(timeout=timeout)
                if "chunk" in msg:
                    if on_chunk is not None:
                        on_chunk(base64.b64decode(msg["chunk"]))
                    continue
                if not msg.get("ok"):
                    raise LabError(msg.get("error") or "the lab helper failed the request")
                return msg
        except queue.Empty as exc:
            raise LabError(f"no reply from the lab helper in {timeout:.0f} s") from exc
        finally:
            self._queues.pop(rid, None)

    def call(self, op: str, *, timeout: float = CALL_TIMEOUT_S, **fields) -> dict:
        rid, q = self._send(op, **fields)
        return self._final(rid, q, timeout)

    # ---- commands ----
    def run_argv(self, argv, timeout, *, env=None, kill_on_timeout=True) -> CommandResult:
        """`orchard.commands.run_command`, on the lab: what GozerAdapter and LabelledContainers call."""
        try:
            # The helper enforces `timeout`; the reply may take a little longer to arrive.
            r = self._final(*self._send("argv", argv=list(argv), timeout=float(timeout),
                                        kill_on_timeout=kill_on_timeout, env=env), float(timeout) + 30)
        except LabError as exc:
            return CommandResult(list(argv), None, "", str(exc))
        return CommandResult(list(argv), r["returncode"], r["stdout"], r["stderr"], timed_out=r["timed_out"],
                             left_running=r["left_running"])

    def start_shell(self, command, *, cwd, env, timeout, path=()) -> tuple[int, queue.Queue]:
        return self._send("shell", command=command, cwd=str(cwd), env=env, timeout=float(timeout), path=list(path))

    def shell(self, command, *, cwd, env, timeout, stdout, path=()) -> tuple[int | None, bool]:
        """Run `command` with bash on the lab, its output written to `stdout` as it comes. Returns
        (exit code or None, timed out), as agent.spawn_checked does. An interruption here (the
        supervisor's SIGINT or SIGTERM) kills it on the lab before going on."""
        rid, q = self.start_shell(command, cwd=cwd, env=env, timeout=timeout, path=path)

        def write(data: bytes):
            stdout.write(data)
            stdout.flush()
        try:
            r = self._final(rid, q, float(timeout) + 120, on_chunk=write)
        except BaseException:
            try:
                self.call("kill", target=rid, timeout=30)
            except LabError:
                pass
            raise
        return r["returncode"], r["timed_out"]

    def hold(self, lease_id: str) -> None:
        self.call("hold", lease_id=lease_id)

    def unhold(self, lease_id: str) -> None:
        self.call("unhold", lease_id=lease_id)

    # ---- files on the lab ----
    def disk_free_gb(self, path) -> float:
        return self.call("fs", action="disk_free", path=str(path))["free_gb"]

    def move_aside(self, path) -> str | None:
        return self.call("fs", action="move_aside", path=str(path))["aside"]

    def cache_audit(self, paths) -> dict:
        return self.call("fs", action="cache_audit", paths=[str(p) for p in paths])["audit"]

    # ---- ending ----
    def close(self) -> None:
        """Close stdin: the helper kills what it runs, releases what the brain held, and exits."""
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def drop(self) -> None:
        """What a broken connection looks like to the helper: stdin closes, nobody waits."""
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        if self.host is not None:
            self.proc.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class LabGozerAdapter(GozerAdapter):
    """gozer on the lab, through the helper. The helper's pid owns every lease, and the helper is
    told which leases the run holds, so it gives them back if the brain goes away."""

    def __init__(self, lab: Lab, *, gozer: str = "gozer", **kw):
        super().__init__(gozer=gozer, owner_pid=int(lab.hello["pid"]), run=lab.run_argv, **kw)
        self.lab = lab

    def acquire(self, *args, **kw):
        lease = super().acquire(*args, **kw)
        self.lab.hold(lease.lease_id)
        return lease

    def claim(self, *args, **kw):
        lease = super().claim(*args, **kw)
        self.lab.hold(lease.lease_id)
        return lease

    def release(self, lease):
        out = super().release(lease)
        self.lab.unhold(lease.lease_id)
        return out


class Sync:
    """rsync over ssh between a path on the brain and the same path on the lab. Never deletes."""

    def __init__(self, *, host, ssh=SSH, run=run_command, timeout: float = SYNC_TIMEOUT_S):
        self.host, self.ssh, self.run, self.timeout = host, list(ssh), run, timeout

    def _rsync(self, src: str, dst: str) -> None:
        argv = ["rsync", "-a", "--mkpath", "--partial", "-e", " ".join(self.ssh), src, dst]
        res = self.run(argv, self.timeout)
        if res.returncode != 0:
            raise LabError(f"rsync {src} -> {dst} failed ({res.returncode}): "
                           f"{(res.stderr.strip() or res.stdout.strip())[-400:]}")

    def up(self, path, *, is_dir: bool = True) -> None:
        p = str(path).rstrip("/") + ("/" if is_dir else "")
        self._rsync(p, f"{self.host}:{p}")

    def down(self, path, *, is_dir: bool = True) -> None:
        p = str(path).rstrip("/") + ("/" if is_dir else "")
        self._rsync(f"{self.host}:{p}", p)


class LabSide:
    """Everything the supervisor needs from a lab box, in one object (Supervisor(lab=...)): a lease
    adapter, a place to run a test, file operations there, and rsync to and from it."""

    def __init__(self, lab: Lab, sync: Sync, *, host: str, root: str, gozer: str = "gozer",
                 test_python: str | None = None):
        self.lab, self.sync = lab, sync
        self.adapter = LabGozerAdapter(lab, gozer=gozer)
        self.test_path = [str(Path(test_python).parent)] if test_python else []
        self.info = {"host": host, "root": root, "hostname": lab.hello.get("hostname"),
                     "pid": lab.hello.get("pid"), "python": lab.hello.get("python")}

    @classmethod
    def connect(cls, *, host, root, gozer="gozer", path=(), python="python3", test_python=None,
                run=run_command) -> "LabSide":
        """Copy this checkout to <root>/orchard on the lab (rsync, incremental), then start the helper
        from it. The lab runs the same orchard code as the brain."""
        res = run(["rsync", "-a", "--mkpath", "--delete", "--exclude", ".git", "--exclude", "__pycache__",
                   "--exclude", ".venv", "-e", " ".join(SSH), f"{CHECKOUT}/", f"{host}:{root}/orchard/"],
                  SYNC_TIMEOUT_S)
        if res.returncode != 0:
            raise LabError(f"copying tt-orchard to {host}:{root}/orchard failed: "
                           f"{(res.stderr.strip() or res.stdout.strip())[-400:]}")
        lab = Lab.ssh(host=host, root=root, gozer=gozer, path=path, python=python)
        return cls(lab, Sync(host=host, run=run), host=host, root=root, gozer=gozer, test_python=test_python)

    # what the supervisor calls
    def sync_up(self, path):
        self.sync.up(path)

    def sync_down(self, path):
        self.sync.down(path)

    def run_test(self, command, *, cwd, env, timeout, stdout):
        return self.lab.shell(command, cwd=cwd, env=env, timeout=timeout, stdout=stdout, path=self.test_path)

    def run_argv(self, argv, timeout, **kw):
        return self.lab.run_argv(argv, timeout, **kw)

    def disk_free_gb(self, path):
        return self.lab.disk_free_gb(path)

    def move_aside(self, path):
        return self.lab.move_aside(path)

    def cache_audit(self, paths):
        return self.lab.cache_audit(paths)

    def close(self):
        self.lab.close()
