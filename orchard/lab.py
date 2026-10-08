# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The lab helper: runs on the lab box and does the lab side of a run's hardware tests.

    python3 -m orchard.lab serve --root /srv/orchard [--gozer gozer] [--path DIR:DIR]

A run with a lab (`supervisor run --lab HOST`) keeps the supervisor and the agents' coder on the brain
box and runs every hardware test on the lab box, so the coder is never parked. The supervisor starts
this helper over ssh once per run (orchard/labclient.py) and talks to it in JSON lines on stdin and
stdout, one request per line, each with an id; replies carry the id, and a running command's output
comes back in chunks before its final reply.

Why a helper and not one ssh per command:
- gozer judges a lease by an owner process alive on the lab. The helper is that process: leases are
  taken with --owner-pid set to its pid, and the tests run as its children, so gozer counts them as
  the lease's own work (HELD, not HELD-FOREIGN).
- A test is killed at its deadline on the lab, as a whole process group. Killing an ssh client
  would leave the test running there.
- When the brain goes away (the supervisor exits or dies, or the connection drops), stdin closes.
  The helper then kills every test it is running and releases every lease the brain told it it
  holds (`hold`), so the lab's chips are never left leased by a run that is gone.

It runs only commands the supervisor sends: gozer, docker and tt-smi calls (`argv`) and the hardware
test command an agent wrote, which the supervisor has already passed through the command runner's
checks on the brain (`shell`). File operations (`fs`) stay under --root.

Paths are the same on both boxes (the lab root, such as /srv/orchard, exists on each), so a run's
files, caches, Hugging Face cache and installed bundles have one absolute path everywhere.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from orchard.commands import run_command
from orchard.defaults import RESET_TIMEOUT_S

QUEUE_TIMEOUT_S = 60.0          # gozer queue and cancel answer at once; this is only a backstop

CHUNK = 64 * 1024
MARKER = ".orchard-model"           # the tensor cache's owner marker (weights-swap templates)


class Helper:
    def __init__(self, *, root, gozer="gozer", out=None):
        self.root = os.path.realpath(root)
        self.gozer = gozer
        self.out = out if out is not None else sys.stdout.buffer
        self._write = threading.Lock()
        self._lock = threading.Lock()
        self.running: dict[str, subprocess.Popen] = {}     # request id -> a shell command's process
        self.held: set[str] = set()                        # lease ids the brain holds through us

    # ---- plumbing ----
    def send(self, msg: dict) -> None:
        line = (json.dumps(msg) + "\n").encode()
        with self._write:
            try:
                self.out.write(line)
                self.out.flush()
            except (BrokenPipeError, ValueError, OSError):
                pass

    def handle(self, req: dict) -> None:
        rid = req.get("id")
        try:
            op = req.get("op")
            fn = getattr(self, f"op_{op}", None) if isinstance(op, str) else None
            if fn is None:
                raise ValueError(f"unknown request {op!r}")
            reply = fn(rid, req)
            if reply is not None:
                self.send({"id": rid, "ok": True, **reply})
        except Exception as exc:                      # a request error is a reply, never a crash
            self.send({"id": rid, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def under_root(self, path) -> str:
        real = os.path.realpath(str(path))
        if real != self.root and not real.startswith(self.root + os.sep):
            raise ValueError(f"{path} is outside the lab root {self.root}")
        return real

    # ---- requests ----
    def op_hello(self, rid, req):
        return {"pid": os.getpid(), "hostname": socket.gethostname(), "root": self.root,
                "python": sys.version.split()[0], "platform": platform.platform()}

    def op_argv(self, rid, req):
        res = run_command(list(req["argv"]), float(req["timeout"]), env=self._env(req.get("env")),
                          kill_on_timeout=bool(req.get("kill_on_timeout", True)))
        return {"returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr,
                "timed_out": res.timed_out, "left_running": res.left_running}

    def op_shell(self, rid, req):
        env = self._env(req.get("env"), prepend=req.get("path") or ())
        proc = subprocess.Popen(["bash", "-c", req["command"]], cwd=req.get("cwd") or self.root, env=env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        with self._lock:
            self.running[rid] = proc
        deadline = time.monotonic() + float(req["timeout"])
        timed_out = False
        fd = proc.stdout.fileno()
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    timed_out = True
                    break
                ready, _, _ = select.select([fd], [], [], min(left, 1.0))
                if ready:
                    data = os.read(fd, CHUNK)
                    if not data:
                        break                          # every writer has closed the pipe
                    self.send({"id": rid, "chunk": base64.b64encode(data).decode()})
            if not timed_out:
                try:
                    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            if timed_out or proc.poll() is None:
                self._kill(proc)
            with self._lock:
                self.running.pop(rid, None)
            proc.stdout.close()
        return {"returncode": None if timed_out else proc.returncode, "timed_out": timed_out}

    def op_kill(self, rid, req):
        with self._lock:
            proc = self.running.get(req["target"])
        if proc is not None:
            self._kill(proc)
        return {"killed": proc is not None}

    def op_hold(self, rid, req):
        with self._lock:
            self.held.add(str(req["lease_id"]))
        return {}

    def op_unhold(self, rid, req):
        with self._lock:
            self.held.discard(str(req["lease_id"]))
        return {}

    def op_fs(self, rid, req):
        action = req.get("action")
        if action == "disk_free":
            return {"free_gb": round(shutil.disk_usage(self.under_root(req["path"])).free / 1e9, 1)}
        if action == "move_aside":
            path = Path(self.under_root(req["path"]))
            if not path.exists():
                return {"aside": None}
            k = 1
            while (aside := path.with_name(f"{path.name}.interrupted-{k}")).exists():
                k += 1
            os.rename(path, aside)
            return {"aside": str(aside)}
        if action == "cache_audit":
            out = {}
            for p in req["paths"]:
                real = Path(self.under_root(p))
                marker = real / MARKER
                out[p] = {"exists": real.is_dir(),
                          "marker": marker.read_text().strip() if marker.is_file() else None,
                          "files": sum(1 for f in real.iterdir() if f.name != MARKER) if real.is_dir() else 0}
            return {"audit": out}
        raise ValueError(f"unknown fs action {action!r}")

    # ---- helpers ----
    @staticmethod
    def _env(env, prepend=()):
        if env is None:
            return None
        out = {k: str(v) for k, v in env.items() if k != "PATH"}
        dirs = [os.path.expanduser(str(d)) for d in prepend]
        out["PATH"] = os.pathsep.join([*dirs, os.environ.get("PATH", "/usr/bin:/bin")])
        return out

    @staticmethod
    def _kill(proc) -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    # ---- the brain went away ----
    def shutdown(self) -> None:
        """Kill every running test, give back every lease the brain held through us, and cancel every
        queue ticket we own: a request still waiting when the brain went away (the waiting gozer process
        is killed with the rest, but its ticket would stay in the queue and hold freed chips back)."""
        with self._lock:
            procs, held = list(self.running.values()), sorted(self.held)
            self.held.clear()
        for proc in procs:
            self._kill(proc)
        for lease_id in held:
            run_command([self.gozer, "release", lease_id, "--json"], RESET_TIMEOUT_S, kill_on_timeout=False)
        for ticket in self._own_tickets():
            run_command([self.gozer, "cancel", ticket], QUEUE_TIMEOUT_S, kill_on_timeout=False)

    def _own_tickets(self) -> list[str]:
        res = run_command([self.gozer, "queue", "--json"], QUEUE_TIMEOUT_S)
        try:
            queue = json.loads(res.stdout or "{}").get("queue") or []
        except (ValueError, AttributeError):
            return []
        return [str(t["ticket"]) for t in queue if isinstance(t, dict) and t.get("ticket") and t.get("pid") == os.getpid()]


def serve(root, gozer="gozer", path=(), stdin=None, out=None) -> int:
    for d in reversed([os.path.expanduser(p) for p in path if p]):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    helper = Helper(root=root, gozer=gozer, out=out)
    stream = stdin if stdin is not None else sys.stdin.buffer
    threads = []
    try:
        for raw in stream:
            try:
                req = json.loads(raw)
            except ValueError:
                helper.send({"id": None, "ok": False, "error": "a request line is not JSON"})
                continue
            t = threading.Thread(target=helper.handle, args=(req,), daemon=True)
            t.start()
            threads.append(t)
    finally:
        helper.shutdown()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python3 -m orchard.lab", description="the lab side of a run's hardware tests")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve one brain on stdin and stdout until stdin closes")
    s.add_argument("--root", required=True, help="the lab root; file operations stay under it")
    s.add_argument("--gozer", default="gozer")
    s.add_argument("--path", default="", help="directories to put first on PATH, separated by ':'")
    args = p.parse_args(argv)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)       # a dropped ssh closes stdin; that is the signal we use
    return serve(args.root, gozer=args.gozer, path=[d for d in args.path.split(":") if d])


if __name__ == "__main__":
    sys.exit(main())
