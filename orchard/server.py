# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Start, stop, check and question the model server that holds a board.

This module owns the server side of a park (spec section 6, steps 3 and 5): starting the coder,
stopping it, confirming with the server's own tooling that it has stopped before any reset, and
asking it the canary. It never calls tt-smi. tt-model has no command that lists running servers,
so a stop is confirmed with docker, ps, pgrep, ss and curl, as the gozer-park skill describes
(checked on this box on 2026-10-02 against the Audio8 container).

Three kinds of server:
- "container": a v5.1 container package. `tt-model serve --detach` starts it, pinned to the leased
  chips with --device-id; docker shows it; it is never the supervisor's descendant, so gozer shows
  its chips HELD-FOREIGN while it runs.
- "bundle": a v5/v6 bundle. `tt-model serve` runs in the foreground, so it is started as a child in
  its own session. Its pid is then its process group id, and pgrep -g covers the vLLM workers it
  starts. It stays the supervisor's descendant, which gozer counts as the owner's own work.
- "process": any other command, started the same way and stopped with SIGTERM to its group. The
  CPU stand-in, the park check and the tests use it.

tt-model requires the --device-id count to match the profile's chip count. gozer grants whole
boards, so a 1-chip profile on a board lease fails to start; plan 4 picks profiles that use the
whole lease.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Protocol

from orchard.adapters import Lease
from orchard.canary import ask as canary_ask
from orchard.canary import post_json
from orchard.commands import run_command
from orchard.defaults import (CMD_TIMEOUT_S, READY_POLL_S, STANDIN_READY_S, START_TIMEOUT_S,
                              STOP_TIMEOUT_S)

KINDS = ("container", "bundle", "process")
CURL_COULD_NOT_CONNECT = 7     # curl's exit code when nothing accepts the connection
# `tt-model stop` says so when docker had to SIGKILL and it reset the mesh itself, from a
# throwaway container (`tt-model stop --help`). Its clean output has no such word.
MESH_RESET_TEXT = re.compile(r"\breset", re.IGNORECASE)


class ServerError(Exception):
    """The server could not be started, or exited."""


class ServerStarting(ServerError):
    """The start timed out, and the stop checks show something coming up. Start nothing else."""


class NotReady(ServerError):
    def __init__(self, waited_s: float, detail: str = ""):
        super().__init__(f"not ready after {waited_s} s {detail}".strip())
        self.waited_s = waited_s


@dataclass(frozen=True)
class ServerSpec:
    target: str                    # tt-model package or bundle id (org/name); a label for "process"
    kind: str
    port: int
    model: str                     # the model name the OpenAI API expects, for the canary
    profile: str = "default"
    image_id: str | None = None    # what `tt-model list` prints as `image <id>`
    argv: tuple[str, ...] = ()     # "process" only

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"server kind must be one of {KINDS}, got {self.kind!r}")
        if not 0 < self.port < 65536:
            raise ValueError(f"bad port {self.port}")
        if self.kind == "process" and not self.argv:
            raise ValueError("a process server needs argv")

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def container_name(self) -> str:
        # tt-model names the container tt-model-<name>-<profile> (gozer-park skill).
        return f"tt-model-{self.target.rsplit('/', 1)[-1]}-{self.profile}"


@dataclass(frozen=True)
class StopCheck:
    stopped: bool
    checks: dict[str, bool] = field(hash=False)        # True means "shows nothing running"
    evidence: dict[str, dict] = field(hash=False)


def proc_identity(pid: int, proc_root: str = "/proc") -> tuple[str | None, int] | None:
    """(boot id, start time) of a process, or None when it does not exist.

    The start time is field 22 of /proc/<pid>/stat, in clock ticks since boot. A pid can be reused
    by an unrelated process, but not with the same start time on the same boot, so the pair says
    whether a recorded pid is still the process orchard started.
    """
    try:
        with open(f"{proc_root}/{pid}/stat") as f:
            stat = f.read()
        # The command name is field 2, in parentheses, and may hold spaces and parentheses itself.
        # Count fields from the last ")": field 3 is the first one after it.
        start = int(stat[stat.rindex(")") + 1:].split()[19])
    except (OSError, ValueError, IndexError):
        return None
    try:
        with open(f"{proc_root}/sys/kernel/random/boot_id") as f:
            boot = f.read().strip() or None
    except OSError:
        boot = None
    return boot, start


def spawn_session(argv, env, log_path=None) -> subprocess.Popen:
    out = open(log_path, "ab") if log_path else subprocess.DEVNULL
    try:
        return subprocess.Popen(list(argv), env=env, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        if log_path:
            out.close()


class ServerControl:
    def __init__(self, spec: ServerSpec, *, run=run_command, spawn=spawn_session,
                 killpg=os.killpg, http=post_json, clock=time.monotonic, sleep=time.sleep,
                 tt_model: str = "tt-model", docker: str = "docker",
                 timeout: float = CMD_TIMEOUT_S, stop_timeout: float = STOP_TIMEOUT_S,
                 start_timeout: float = START_TIMEOUT_S, ready_poll_s: float = READY_POLL_S,
                 log_path: str | None = None, proc_root: str = "/proc", identity=None):
        self.spec = spec
        self.proc_root = proc_root
        # identity(pid) -> (boot id, start time) or None; tests replace it.
        self.identity = identity or (lambda pid: proc_identity(pid, self.proc_root))
        self.run, self.spawn, self.killpg, self.http = run, spawn, killpg, http
        self.clock, self.sleep = clock, sleep
        self.tt_model, self.docker = tt_model, docker
        self.timeout, self.stop_timeout, self.ready_poll_s = timeout, stop_timeout, ready_poll_s
        self.start_timeout = start_timeout
        self.log_path = log_path
        self.proc = None                                  # our child, when we started it
        self.pid: int | None = None
        self.pgid: int | None = None
        self.dev_indices: tuple[int, ...] | None = None   # None: unknown, so any device counts
        self.start_time: int | None = None                # of the process we started or adopted
        self.boot_id: str | None = None
        # What adopt() found out about a "process" server: "unknown" (we started it, or it is not
        # a process server), "match", "unverified" (the record has no start time) or "gone".
        self.adopted_identity = "unknown"

    # ---- ledger form --------------------------------------------------------------------------

    def record(self) -> dict:
        return {"kind": self.spec.kind, "target": self.spec.target, "port": self.spec.port,
                "pid": self.pid, "pgid": self.pgid,
                "start_time": self.start_time, "boot_id": self.boot_id,
                "dev_indices": None if self.dev_indices is None else list(self.dev_indices)}

    def adopt(self, record: dict) -> str | None:
        """Take over a server a previous supervisor started, from its ledger record.

        Returns a notice for the ledger when the recorded process cannot be confirmed as the one
        orchard started, and None otherwise. A "process" server is stopped with a signal to its
        group, so it is signalled only when the pid still has the recorded start time on the same
        boot. When it does not, the process counts as gone: the pid may belong to an unrelated
        process now. The port checks still run. A group whose leader is gone but whose members
        linger is not found by this; that is the price of never signalling a stranger.
        """
        self.proc = None
        self.pid, self.pgid = record.get("pid"), record.get("pgid")
        self.start_time, self.boot_id = record.get("start_time"), record.get("boot_id")
        devs = record.get("dev_indices")
        self.dev_indices = None if devs is None else tuple(int(i) for i in devs)
        self.adopted_identity = "unknown"
        if self.spec.kind != "process" or self.pid is None:
            return None
        if self.start_time is None:
            self.adopted_identity = "unverified"
            return (f"the recorded {self.spec.target} pid {self.pid} has no start time, so it is "
                    "not signalled")
        return None if self._identity_matches() else (
            f"the recorded {self.spec.target} pid {self.pid} is not the process orchard started "
            f"(start time {self.start_time}, now {self.identity(self.pid)}); treated as gone and "
            "not signalled")

    def _identity_matches(self) -> bool:
        """Does the recorded pid still have the recorded start time? Sets adopted_identity."""
        ok = self.identity(self.pid) == (self.boot_id, self.start_time)
        self.adopted_identity = "match" if ok else "gone"
        return ok

    # ---- start and stop -----------------------------------------------------------------------

    def _env(self, lease: Lease | None) -> dict:
        env = dict(os.environ)
        env.pop("TT_VISIBLE_DEVICES", None)      # a server without a lease sees no chip list
        if lease is not None:
            env.update(lease.env)
        return env

    def start(self, lease: Lease | None) -> None:
        s = self.spec
        if s.kind == "container":
            if lease is None:
                raise ServerError("a container server needs a lease")
            # --device-id pins the container to the leased chips. Without it tt-model picks free
            # chips itself, which may be on a board this lease does not hold.
            argv = [self.tt_model, "serve", s.target, "--detach", "--local-only",
                    "--no-update-check", "--port", str(s.port), "--profile", s.profile,
                    "--device-id", ",".join(str(i) for i in lease.dev_indices)]
            # The start has its own budget and is never killed: a killed `tt-model serve` can leave
            # docker bringing the container up while the caller believes it failed.
            self.dev_indices = tuple(lease.dev_indices)
            res = self.run(argv, self.start_timeout, env=self._env(lease), kill_on_timeout=False)
            if res.timed_out:
                check = self.confirm_stopped()
                if not check.stopped:
                    raise ServerStarting(f"tt-model serve still running after {self.start_timeout} s "
                                         f"and the server is coming up ({check.checks}); "
                                         "not starting another")
                raise ServerError(f"tt-model serve still running after {self.start_timeout} s and "
                                  "nothing came up")
            if res.returncode != 0:
                raise ServerError(f"tt-model serve exited {res.returncode}: "
                                  f"{(res.stderr.strip() or res.stdout.strip())[:500]}")
        else:
            argv = (list(s.argv) if s.kind == "process" else
                    [self.tt_model, "serve", s.target, "--local-only", "--no-update-check",
                     "--port", str(s.port)])
            self.proc = self.spawn(argv, self._env(lease), self.log_path)
            # start_new_session made the child a process group leader: its pid is its group id.
            self.pid = self.pgid = self.proc.pid
            ident = self.identity(self.pid)
            self.boot_id, self.start_time = ident if ident else (None, None)
        self.dev_indices = tuple(lease.dev_indices) if lease is not None else ()

    def stop(self) -> dict:
        s = self.spec
        if s.kind == "process":
            return self._signal_group()
        argv = [self.tt_model, "stop", s.target] + (["--profile", s.profile]
                                                     if s.kind == "container" else [])
        res = self.run(argv, self.stop_timeout)
        if self.proc is not None:
            self.proc.poll()          # reap a bundle's `tt-model serve`
        # A mesh reset by tt-model runs outside gozer, so the caller waits longer for quiet chips.
        out = {**res.record(), "mesh_reset": bool(MESH_RESET_TEXT.search(res.stdout + res.stderr))}
        if s.kind == "bundle" and res.returncode != 0:
            # tt-model 0.1.0 stops container packages only and refuses a bundle, which left the
            # bundle serving on its lease (2026-10-07). We started its `tt-model serve` as a group
            # leader, so stop the group the way a process server is stopped.
            out["signalled"] = self._signal_group()
        return out

    def _signal_group(self) -> dict:
        if self.pgid is None:
            return {"how": "not started"}
        if self.proc is None:
            # Adopted from the ledger: the pid may have been reused since. Check again now.
            if self.start_time is None or not self._identity_matches():
                return {"how": "not signalled: the recorded pid is not the process orchard started"}
        try:
            self.killpg(self.pgid, signal.SIGTERM)
        except ProcessLookupError:
            return {"how": "already gone"}
        if self.proc is None:
            return {"how": "SIGTERM (adopted; the stop checks decide)"}
        deadline = self.clock() + self.stop_timeout
        while self.clock() < deadline:
            if self.proc.poll() is not None:
                return {"how": "SIGTERM"}
            self.sleep(0.2)
        try:
            self.killpg(self.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        return {"how": "SIGKILL"}

    # ---- confirming the stop ------------------------------------------------------------------

    def confirm_stopped(self) -> StopCheck:
        checks: dict[str, bool] = {}
        ev: dict[str, dict] = {}
        if self.proc is not None:
            # An exited child stays a zombie until it is waited for, and ps lists zombies.
            self.proc.poll()
        if self.spec.kind == "container":
            self._check_docker(checks, ev)
        else:
            self._check_process(checks, ev)
        self._check_port(checks, ev)
        return StopCheck(all(checks.values()), checks, ev)

    def _is_mine(self, line: str) -> bool:
        parts = line.split(None, 3)
        if len(parts) < 3:
            return False
        image, name = parts[1], parts[2]
        ports = parts[3] if len(parts) > 3 else ""
        return (name == self.spec.container_name
                or (self.spec.image_id is not None and self.spec.image_id in image)
                or f":{self.spec.port}->" in ports)

    def _maps_our_device(self, paths: list[str], mine: bool) -> bool:
        # tt-model maps the whole directory in some cases. Another agent's container can do the
        # same (common for inference-server style runs). Such a container can open our chips, and
        # if it runs as root gozer cannot see it. The check accepts that risk: only our own
        # container's whole-directory mapping blocks. The notice that names the other container
        # (handoff.py, _wait_stopped) is the only signal an operator gets.
        if "/dev/tenstorrent" in paths:
            return mine
        if self.dev_indices is None:               # unknown chips: any device counts
            return any(p.startswith("/dev/tenstorrent/") for p in paths)
        return any(p == f"/dev/tenstorrent/{i}" for i in self.dev_indices for p in paths)

    def _check_docker(self, checks: dict, ev: dict) -> None:
        ps = self.run([self.docker, "ps", "--format", "{{.ID}} {{.Image}} {{.Names}} {{.Ports}}"],
                      self.timeout)
        mine = [ln for ln in ps.stdout.splitlines() if self._is_mine(ln)]
        mine_ids = [ln.split()[0] for ln in mine]
        checks["docker_ps"] = ps.returncode == 0 and not mine
        ev["docker_ps"] = {**ps.record(), "matching": mine}
        ids = self.run([self.docker, "ps", "-q"], self.timeout)
        id_list = ids.stdout.split()
        held, others, inspect_ok = [], [], True
        if ids.returncode == 0 and id_list:
            # A privileged container or one with a mounted /dev may list no device, so this check
            # runs in addition to the one above. Each line: full id, name, mapped device paths.
            ins = self.run([self.docker, "inspect", "--format",
                            "{{.Id}} {{.Name}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}",
                            *id_list], self.timeout)
            inspect_ok = ins.returncode == 0
            for ln in ins.stdout.splitlines():
                parts = ln.split()
                if len(parts) < 2:
                    continue
                is_mine = any(parts[0].startswith(i) for i in mine_ids)
                if self._maps_our_device(parts[2:], is_mine):
                    held.append(ln)
                elif "/dev/tenstorrent" in parts[2:]:
                    others.append(parts[1].lstrip("/"))
            ev["docker_inspect"] = {**ins.record(), "holding": held,
                                    "others_with_all_devices": others}
        checks["docker_devices"] = ids.returncode == 0 and inspect_ok and not held

    def _check_process(self, checks: dict, ev: dict) -> None:
        if self.pid is None or self.pgid is None:
            checks["process"] = False
            ev["process"] = {"error": "no pid recorded for this server"}
            return
        if self.spec.kind == "process" and self.adopted_identity == "gone":
            # The recorded process is not running any more (its pid is free or reused). ps and
            # pgrep would describe a stranger, so they are not asked.
            checks["process"] = checks["process_group"] = True
            ev["process"] = {"note": "the recorded process is gone; its pid has another start time"}
            return
        ps = self.run(["ps", "-p", str(self.pid), "-o", "pid="], self.timeout)
        checks["process"] = ps.returncode == 1 and not ps.stdout.strip()
        # pgrep -g matches process group ids, not command lines, so it cannot match itself.
        pg = self.run(["pgrep", "-g", str(self.pgid)], self.timeout)
        checks["process_group"] = pg.returncode == 1 and not pg.stdout.strip()
        ev["ps"], ev["pgrep"] = ps.record(), pg.record()

    def _check_port(self, checks: dict, ev: dict) -> None:
        ss = self.run(["ss", "-ltn", f"( sport = :{self.spec.port} )"], self.timeout)
        # ss exits 0 whether or not it finds a listener, so read the rows after the header.
        listening = [ln for ln in ss.stdout.splitlines()[1:] if ln.strip()]
        checks["port_closed"] = ss.returncode == 0 and not listening
        curl = self.run(["curl", "-sS", "--max-time", "3", f"{self.spec.endpoint}/health"],
                        self.timeout)
        # 0 means a server answered. 28 means something accepted and did not answer in time.
        # Only "could not connect" confirms that nothing listens.
        checks["health_refused"] = curl.returncode == CURL_COULD_NOT_CONNECT
        ev["ss"], ev["curl"] = ss.record(), curl.record()

    # ---- readiness and the canary -------------------------------------------------------------

    def wait_ready(self, budget_s: float) -> float:
        t0 = self.clock()
        while True:
            if self.proc is not None and self.proc.poll() is not None:
                raise ServerError(f"the server exited with code {self.proc.returncode} "
                                  "before it was ready")
            r = self.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "3",
                          f"{self.spec.endpoint}/health"], self.timeout)
            if r.returncode == 0 and r.stdout.strip() == "200":
                return round(self.clock() - t0, 3)
            waited = self.clock() - t0
            if waited >= budget_s:
                raise NotReady(round(waited, 3), r.stderr.strip()[:300])
            self.sleep(self.ready_poll_s)

    def ask(self, prompt: str) -> str:
        return canary_ask(self.spec.endpoint, self.spec.model, prompt, http=self.http)


class StandIn(Protocol):
    """Starting is two calls, so the caller can record the pid between them: `spawn` returns as
    soon as the process exists, and `wait_ready` can then take minutes (a CPU model load)."""

    def spawn(self) -> None: ...

    def wait_ready(self) -> None: ...

    def ask(self, prompt: str) -> str: ...

    def stop(self) -> None: ...

    def record(self) -> dict: ...

    def adopt(self, record: dict) -> str | None: ...

    def confirm_stopped(self) -> StopCheck: ...


class ServerStandIn:
    """The CPU stand-in: a local server process. It gets no lease and no chip list.

    The spec requires that the stand-in never touch the coder's weight or tensor caches (a cleared
    cache turns a 2-3 min warm restart into a 30 min cold boot). This class cannot check that; the
    stand-in's command line, set in plan 4, must point at its own cache.
    """

    def __init__(self, server: ServerControl, ready_budget_s: float = STANDIN_READY_S):
        if server.spec.kind == "container":
            raise ValueError("the CPU stand-in runs as a process or bundle, never a container")
        self.server, self.ready_budget_s = server, ready_budget_s

    def spawn(self) -> None:
        self.server.start(None)

    def wait_ready(self) -> None:
        self.server.wait_ready(self.ready_budget_s)

    def ask(self, prompt: str) -> str:
        return self.server.ask(prompt)

    def stop(self) -> None:
        self.server.stop()

    def record(self) -> dict:
        return self.server.record()

    def adopt(self, record: dict) -> str | None:
        return self.server.adopt(record)

    def confirm_stopped(self) -> StopCheck:
        return self.server.confirm_stopped()
