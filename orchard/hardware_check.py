"""Hardware-check driver for the hold-through-swap runbook (docs/runbooks/hardware-validation.md).

Run it as:

    python3 -m orchard.hardware_check --board <BDF of the board's first chip> \
        --gozer <gozer with `reset`> --env-script <tt-metal env script> --python <python with ttnn>

The three machine paths can also come from ORCHARD_GOZER, ORCHARD_ENV_SCRIPT and
ORCHARD_CHILD_PYTHON. --env-script and --python are needed only when --child-cmd is not given.
The driver has no built-in machine paths; a missing one stops it with a message that names the
flag and the variable.

One long-lived process (this one) owns one gozer lease for the whole session. Every model
process it starts is its child. It runs the runbook's checks under that lease and writes each
result to a run ledger. Whatever happens, a try/finally and the SIGINT/SIGTERM handlers stop the
child and release the lease before the process exits.

The order differs from the runbook's numbering on purpose. The 16 minute idle wait (H1) comes after
H2 to H4, so the lease is already older than 900 s when the board is checked after sitting idle.

    P   preflight, no lease yet
    A   acquire
    H2  a child of the driver holds the device and shows as HELD
    H3  reset refuses while the child holds the device, and works once it is gone
    H4  the device opens again after the reset
    H1  idle wait, then the lease must still be CLAIMED
    H6  release leaves the board FREE

H5 (container server) is not part of this driver.

Exit codes: 0 all pass, 1 a check failed or a stop condition hit, 2 preflight refused,
3 acquire failed, 4 internal error.

Safety rules in this driver:

- The refusal probe in H3 runs `gozer reset` with GOZER_RESET_CMD pointed at a stand-in script
  that only writes a marker file. If gozer fails to refuse, the stand-in runs and the chips are
  never reset under a live device open. The real resets use the normal environment.
- Every gozer call runs in its own session, so a Ctrl-C or SIGHUP aimed at the terminal's process
  group does not reach gozer or the reset command under it.
- `gozer reset` and `gozer release` always run to the end. A signal that arrives meanwhile is
  held back and delivered afterwards. A reset that outlives its timeout is left running and is
  never repeated.
- Cleanup runs with signals held off. It stops the child, kills anything left in the child's
  process group, and retries a release that exits 15. It never uses --force and never runs
  tt-smi -r.
- `tt-smi -s` is off by default (`--tt-smi none`). It opens every device on the box, so it must
  not run while another driver is active.

If the driver is killed with SIGKILL, no cleanup runs. The child's stdin closes. A child that has
finished opening the device then closes it and exits, the lease goes STALE once the owner pid is
dead, and gozer reaps it without a reset. A child that still holds the device keeps the lease
and gozer shows it as HELD-FOREIGN. In that case wait for the holder to exit, then run
`gozer release <lease-id>`. Never use --force and never run tt-smi -r by hand.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from orchard.ledger import Ledger

WHO = "orchard:hardware-check"
REASON = "validate hold-through-swap"
EVIDENCE_LIMIT = 2000          # characters of stdout or stderr kept per command
# The machine paths, when no flag gives them. The gozer must have `reset` (the tt-gozer branch
# binary); the gozer on PATH may not, so there is no PATH default.
GOZER_ENV, ENV_SCRIPT_ENV, PYTHON_ENV = "ORCHARD_GOZER", "ORCHARD_ENV_SCRIPT", "ORCHARD_CHILD_PYTHON"
DEFAULT_STATE_ROOT = "/tmp/tt-gozer"     # gozer's own default; the driver only reads under it
GUARDED_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
RELEASE_RETRIES = 3
RELEASE_RETRY_DELAY = 5.0

# What the real child runs. The device is opened and closed on the main thread.
CHILD_SNIPPET = (
    "import time, sys, ttnn; "
    "mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1)); "
    'print("OPENED", flush=True); '
    "sys.stdin.readline(); "
    "ttnn.close_mesh_device(mesh); "
    'print("CLOSED", flush=True)'
)

EXIT_PASS, EXIT_FAIL, EXIT_PREFLIGHT, EXIT_ACQUIRE, EXIT_INTERNAL = 0, 1, 2, 3, 4


class Abort(Exception):
    """A stop condition. The run ends at once, after cleanup, with exit 1."""


class AcquireFailed(Exception):
    """The lease could not be taken, or what was granted is not what we asked for."""


class Interrupted(BaseException):
    """SIGINT or SIGTERM arrived. BaseException so no `except Exception` swallows it."""


def default_child_argv(bdf: str, env_script: str, python: str, cache_dir: str = "cache",
                       logs_dir: str = "logs") -> list[str]:
    """The argv of the real child.

    `exec` makes python replace the shell, so the pid the driver holds is the process that has
    the device open. TT_VISIBLE_DEVICES is exported after the env script runs, because the
    script sets its own value for a different board.
    """
    # The env script sets one shared JIT cache and log path. Each run gets its own, so two
    # drivers on two boards never write into the same cache.
    script = (f"source {shlex.quote(env_script)}; "
              f"export TT_VISIBLE_DEVICES={shlex.quote(bdf)}; "
              f"export TT_METAL_CACHE={shlex.quote(str(cache_dir))} "
              f"TT_METAL_LOGS_PATH={shlex.quote(str(logs_dir))}; "
              f"mkdir -p {shlex.quote(str(cache_dir))} {shlex.quote(str(logs_dir))}; "
              f"exec {shlex.quote(python)} -c {shlex.quote(CHILD_SNIPPET)}")
    return ["bash", "-c", script]


def _trunc(text: str | None) -> str:
    text = text or ""
    return text if len(text) <= EVIDENCE_LIMIT else text[:EVIDENCE_LIMIT] + "...[truncated]"


class Child:
    """The model process the driver started, with a thread that collects its output."""

    def __init__(self, argv: list[str], env: dict):
        # start_new_session gives the child its own process group, so one pgrep -g can cover
        # its workers, and a Ctrl-C at the terminal reaches only the driver.
        self.argv = argv
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True, env=env)
        self.pid = self.proc.pid
        self.pgid = os.getpgid(self.pid)
        self.lines: list[str] = []
        self.q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.append(line.rstrip("\n"))
            self.q.put(line.rstrip("\n"))
        self.q.put(None)       # end of output

    def running(self) -> bool:
        return self.proc.poll() is None

    def output(self) -> str:
        return _trunc("\n".join(self.lines))


def release_reset_confirmed(history_root: str, lease_id: str, since: str | None = None):
    """Did the reset inside `gozer release` run and succeed? Read from gozer's history log.

    Returns True or False from the last `released` event for the lease, or None when the log or
    the event cannot be found. `gozer release` exits 0 even when its reset failed, so the exit
    code alone cannot answer this. With `since` (a UTC time in gozer's own format), events
    logged before it are ignored, so an old event for the same lease id cannot count.
    """
    try:
        lines = Path(history_root, "history.jsonl").read_text().splitlines()
    except OSError:
        return None
    found = None
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(ev, dict) and ev.get("event") == "released" and ev.get("lease_id") == lease_id
                and (since is None or str(ev.get("ts", "")) >= since)):
            found = bool(ev.get("reset_ran") and ev.get("reset_ok"))
    return found


EXPECTED_CHECKS = ["P", "A", "H2", "H3.refuse", "H3.stop", "H3.gone", "H3.reset", "H4",
                   "H4.stop", "H4.gone", "H4.reset", "H1", "H6.release", "H6.files", "H6.free"]


class Driver:
    def __init__(self, args, ledger: Ledger, clock, sleep, hook, out_dir=None):
        self.a = args
        self.led = ledger
        self.clock, self.sleep = clock, sleep
        self.hook = hook or (lambda name, **info: None)
        # Absolute, so a script written here still works when gozer runs from another directory.
        self.out_dir = Path(os.path.abspath(out_dir if out_dir else "."))
        self.pid = os.getpid()
        self.lease_id: str | None = None
        self.released = False
        self.lease_live = False        # true between a granted acquire and the release
        self.grant: dict = {}
        self.child: Child | None = None
        self.results: list[tuple[str, bool]] = []
        self.in_cleanup = False
        self.cleaned = False
        self.defer = False             # true while a reset or release runs: signals wait
        self.pending_signal: int | None = None
        self.abandoned: list[str] | None = None    # a gozer call left running after its timeout
        self.recovery_text: str | None = None
        self.last_stdout = ""
        self.t_acquire: float | None = None
        self.acquired_at: str | None = None
        self.board_serial: str | None = None
        self.dev_index: int | None = None

    # ---- ledger helpers -------------------------------------------------

    def notice(self, check: str, ok: bool, summary: str, **evidence) -> bool:
        evidence["summary"] = summary
        self.led.append("notice", None, check=check, ok=bool(ok), evidence=evidence)
        self.results.append((check, bool(ok)))
        return bool(ok)

    def safe_notice(self, *a, **k):
        """A notice that cannot raise. Used in cleanup, where a ledger error must not skip a release."""
        try:
            return self.notice(*a, **k)
        except Exception:
            return False

    def decision(self, what: str, **info):
        self.led.append("decision", None, decision=what, **info)

    def measure(self, name: str, value, unit: str = "s"):
        self.led.append("measurement", None, name=name, value=value, unit=unit, label="measured")

    # ---- signals --------------------------------------------------------

    @contextlib.contextmanager
    def protected(self):
        """Hold signals back while a reset or release runs, then deliver the first one.

        The handler records the signal and returns, so the call is never cut short. The caller
        records the result and then calls `deliver_pending`.
        """
        self.defer, self.pending_signal = True, None
        try:
            yield
        finally:
            self.defer = False

    def deliver_pending(self):
        """Raise the signal that arrived during a reset or release, now that its result is recorded."""
        if self.pending_signal is not None and not self.in_cleanup:
            sig, self.pending_signal = self.pending_signal, None
            raise Interrupted(f"signal {signal.Signals(sig).name}")

    # ---- running commands -----------------------------------------------

    def run_cmd(self, argv: list[str], timeout: float, check: str = "", env=None,
                protect: bool = False) -> dict:
        """Run a command and return its evidence. A timeout is a stop condition.

        The command gets its own session, so a Ctrl-C at the terminal reaches only the driver.
        With protect=True the call is never killed: signals wait (see `protected`) and a timeout
        leaves the command running.
        """
        t0 = time.monotonic()
        self.last_stdout = ""
        res = {"command": argv, "exit_code": None, "stdout": "", "stderr": ""}
        # For a protected call the deferral starts before Popen. A signal during fork and exec
        # then waits, and cannot leave a running gozer that nobody watches.
        guard = self.protected() if protect else contextlib.nullcontext()
        with guard:
            try:
                proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, start_new_session=True, env=env)
            except FileNotFoundError as exc:
                res["error"] = f"not found: {exc}"
                res["seconds"] = 0.0
                return res
            try:
                try:
                    out, err = proc.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    if protect:
                        # A reset may be mid-flight. Killing gozer would leave the reset command
                        # running with nobody watching. Give it one more full timeout.
                        out, err = proc.communicate(timeout=timeout)
                    else:
                        raise
            except subprocess.TimeoutExpired:
                res["error"] = f"timed out after {timeout} s"
                res["seconds"] = round(time.monotonic() - t0, 3)
                if protect:
                    self.abandoned = argv
                    res["note"] = "left running; the driver will not repeat it"
                else:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.communicate()
                self.notice(check or "TIMEOUT", False, f"step timed out: {' '.join(argv)}", **res)
                raise Abort(f"step timed out after {timeout} s: {' '.join(argv)}") from None
            except BaseException:
                if protect:
                    # Killing gozer would leave its reset command running unwatched. Leave it,
                    # and let cleanup know not to start a second one.
                    self.abandoned = argv
                else:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                raise
        # Evidence is cut at 2000 characters. Parsing needs the whole text.
        self.last_stdout = out
        res.update(exit_code=proc.returncode, stdout=_trunc(out), stderr=_trunc(err))
        res["seconds"] = round(time.monotonic() - t0, 3)
        return res

    def gozer(self, *args, timeout=None, check="", env=None, protect=False) -> dict:
        return self.run_cmd([self.a.gozer, *args], timeout or self.a.cmd_timeout, check,
                            env=env, protect=protect)

    def status(self, check="") -> tuple[dict, dict]:
        """`gozer status --json`: (evidence, parsed payload)."""
        res = self.gozer("status", "--json", check=check)
        if res["exit_code"] != 0:
            raise RuntimeError(f"gozer status failed: {res}")
        try:
            return res, json.loads(self.last_stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"gozer status gave no JSON: {res}") from exc

    # ---- reading status -------------------------------------------------

    def board_chips(self, payload: dict) -> list[dict]:
        chips = payload["chips"]
        mine = [c for c in chips if c["bdf"] == self.a.board]
        if not mine:
            return []
        self.board_serial = mine[0]["board"]
        self.dev_index = mine[0]["dev_index"]
        return [c for c in chips if c["board"] == self.board_serial]

    def is_ours(self, chip: dict) -> bool:
        return chip.get("who") == WHO and chip.get("pid") == self.pid

    def guarded_board(self, payload: dict) -> list[dict]:
        """Our board's chips, after the stop-condition checks.

        An unexpected lease on our board, our own lease vanishing while the driver is alive, or
        the child holding a device on another board ends the run. Every status read after the
        acquire goes through here.
        """
        chips = self.board_chips(payload)
        foreign = [c for c in chips if c.get("who") and not self.is_ours(c)]
        if foreign:
            raise Abort("unexpected lease on our board: "
                        + ", ".join(f"{c['bdf']} {c['who']} pid {c['pid']}" for c in foreign))
        if self.lease_live and not any(self.is_ours(c) for c in chips):
            raise Abort("our lease disappeared while the driver is alive")
        if self.child is not None:
            for c in payload["chips"]:
                if c["board"] != self.board_serial and self.child.pid in (c.get("pids_holding") or []):
                    raise Abort(f"the child holds a device on the other board ({c['bdf']})")
        return chips

    def grant_chips(self, chips: list[dict]) -> list[dict]:
        return [c for c in chips if c["bdf"] in self.grant.get("chips", [])]

    def all_claimed(self, chips: list[dict]) -> bool:
        """Every chip in the grant is ours and CLAIMED: no open device, no other owner."""
        ours = self.grant_chips(chips)
        return (len(ours) == len(self.grant.get("chips", [])) and bool(ours)
                and all(self.is_ours(c) and c["state"] == "CLAIMED" for c in ours))

    def other_leases(self, payload: dict) -> list[dict]:
        return [c for c in payload["chips"] if c.get("who") and not self.is_ours(c)]

    def wait_state(self, want, tries: int = 3, delay: float = 1.0):
        """Read status until the wanted state shows.

        gozer can show HELD-FOREIGN for a moment while a process starts or exits, so a
        mismatch is read again before it counts. Returns (evidence, first chip, board chips).
        """
        for i in range(tries):
            res, payload = self.status()
            chips = self.guarded_board(payload)
            first = next(c for c in chips if c["bdf"] == self.a.board)
            if want(first, chips) or i == tries - 1:
                return res, first, chips
            time.sleep(delay)

    # ---- the child ------------------------------------------------------

    def child_argv(self) -> list[str]:
        if self.a.child_cmd:
            argv = json.loads(self.a.child_cmd)
        else:
            argv = default_child_argv(self.a.board, self.a.env_script, self.a.python,
                                      self.out_dir / "cache", self.out_dir / "logs")
        return [s.replace("{bdf}", self.a.board).replace("{dev}", str(self.dev_index))
                for s in argv]

    def require_ready(self, check: str):
        """Before any device open: every granted chip is ours and CLAIMED, or the run stops."""
        res, first, chips = self.wait_state(lambda f, cs: self.all_claimed(cs))
        ok = self.all_claimed(chips)
        self.notice(f"{check}.ready", ok,
                    "every granted chip is ours and CLAIMED" if ok else
                    "board not ready: " + ", ".join(f"{c['bdf']} {c['state']} {c.get('who')}"
                                                    for c in chips), status=res)
        if not ok:
            raise Abort(f"board not ready for {check}")

    def start_child(self, check: str) -> float:
        """Start the child and wait for OPENED. A missing OPENED is a stop condition."""
        self.hook("before_start_child", driver=self, check=check)
        self.require_ready(check)
        env = dict(os.environ, TT_VISIBLE_DEVICES=self.a.board)
        argv = self.child_argv()
        t0 = time.monotonic()
        self.child = Child(argv, env)
        deadline = t0 + self.a.opened_timeout
        while True:
            try:
                line = self.child.q.get(timeout=max(0.0, min(0.2, deadline - time.monotonic())))
            except queue.Empty:
                line = ""
                if time.monotonic() >= deadline:
                    self.notice(check, False, "child never printed OPENED (timeout)",
                                command=argv, timeout_s=self.a.opened_timeout,
                                output=self.child.output())
                    raise Abort(f"child did not print OPENED within {self.a.opened_timeout} s")
            if line is None:
                self.child.proc.wait()
                self.notice(check, False, "child exited before OPENED", command=argv,
                            exit_code=self.child.proc.returncode, output=self.child.output())
                raise Abort("child exited before printing OPENED")
            if line.strip() == "OPENED":
                break
        seconds = round(time.monotonic() - t0, 3)
        self.measure(f"{check}_open_seconds", seconds)
        return seconds

    def stop_child(self, check: str) -> dict:
        """Stop the child: ask first, then SIGTERM, then SIGKILL. Safe to call twice."""
        c = self.child
        info = {"how": "already-gone"}
        if c is None:
            return info
        t0 = time.monotonic()
        if c.running():
            info["how"] = "quit"
            try:
                c.proc.stdin.write("quit\n")
                c.proc.stdin.flush()
                c.proc.stdin.close()
            except (BrokenPipeError, ValueError, OSError):
                pass
            try:
                c.proc.wait(timeout=self.a.stop_timeout)
            except subprocess.TimeoutExpired:
                self.decision("child ignored quit; sending SIGTERM", check=check, pid=c.pid)
                info["how"] = "SIGTERM"
                try:
                    os.kill(c.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    c.proc.wait(timeout=self.a.stop_timeout)
                except subprocess.TimeoutExpired:
                    self.decision("child ignored SIGTERM; sending SIGKILL to its group",
                                  check=check, pid=c.pid, pgid=c.pgid)
                    info["how"] = "SIGKILL"
                    try:
                        os.killpg(c.pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    c.proc.wait(timeout=10)
        c._thread.join(timeout=2)
        self.hook("after_stop", driver=self, check=check, how=info["how"])
        info.update(exit_code=c.proc.returncode, seconds=round(time.monotonic() - t0, 3),
                    output=c.output())
        return info

    def confirm_gone(self, check: str) -> bool:
        """The child and its group are gone and every granted chip is CLAIMED. A gate for the reset."""
        c = self.child
        ps = self.run_cmd(["ps", "-p", str(c.pid), "-o", "pid="], self.a.cmd_timeout)
        pg = self.run_cmd(["pgrep", "-g", str(c.pgid)], self.a.cmd_timeout)
        ps_gone = ps["exit_code"] == 1 and ps["stdout"].strip() == ""
        pg_gone = pg["exit_code"] == 1 and pg["stdout"].strip() == ""
        res, first, chips = self.wait_state(lambda f, cs: self.all_claimed(cs))
        claimed = self.all_claimed(chips)
        ok = ps_gone and pg_gone and claimed
        states = {c["bdf"]: c["state"] for c in self.grant_chips(chips)}
        return self.notice(check, ok,
                           "child gone and every granted chip CLAIMED" if ok else
                           f"child not confirmed gone (ps_gone={ps_gone}, group_gone={pg_gone}, "
                           f"chips={states})",
                           ps=ps, pgrep=pg, status=res, chip_state=first["state"],
                           chip_states=states)

    # ---- the checks -----------------------------------------------------

    def snapshot_docker(self, check: str, summary: str):
        if self.a.docker.lower() == "none":
            return
        d = self.run_cmd([self.a.docker, "ps", "--format",
                          "{{.ID}} {{.Image}} {{.Names}} {{.Ports}}"], self.a.cmd_timeout)
        # Recorded for the operator. gozer cannot see containers, so a person reads this.
        # It never decides anything here.
        self.notice(check, True, summary, **d)

    def gozer_env(self) -> dict:
        return {k: v for k, v in os.environ.items() if k.startswith("GOZER_")}

    def preflight(self) -> bool:
        # A GOZER_* variable points gozer at other state, or replaces the chip reset. Either one
        # would make every later check run against something else.
        env_set = self.gozer_env()
        if env_set and not self.a.allow_gozer_env:
            self.notice("P.env", False,
                        "GOZER_* variables are set: " + ", ".join(sorted(env_set))
                        + ". Unset them, or pass --allow-gozer-env (test only).",
                        gozer_env=env_set)
            return False
        # The gozer must have `reset`. The one on PATH may be older. Check before anything else.
        h = self.gozer("reset", "--help", check="P.gozer")
        if h["exit_code"] != 0:
            self.notice("P.gozer", False, "this gozer has no `reset` command", **h)
            return False
        self.notice("P.gozer", True, "gozer has `reset`", **h)
        res, payload = self.status("P")
        chips = self.board_chips(payload)
        if not chips:
            self.notice("P", False, f"board {self.a.board} is not in gozer status", status=res)
            return False
        busy = [c for c in chips if c["state"] != "FREE" or c.get("who")]
        ok = not busy
        self.notice("P", ok,
                    "every chip of the board is FREE" if ok else
                    "board is not free: " + ", ".join(f"{c['bdf']} {c['state']} {c.get('who')}"
                                                      for c in busy),
                    status=res, board_serial=self.board_serial)
        self.snapshot_docker("P.docker", "docker ps recorded")
        q = self.gozer("queue", "--json")
        queue_len = 0
        try:
            queue_len = len(json.loads(self.last_stdout).get("queue", []))
        except (json.JSONDecodeError, AttributeError):
            pass
        if queue_len:
            self.notice("P.queue", True, f"queue is not empty ({queue_len} entries); continuing", **q)
        else:
            self.notice("P.queue", True, "queue is empty", **q)
        return ok

    def acquire(self):
        self.hook("before_acquire", driver=self)
        # Block the signals while the lease is being taken, so one cannot land after gozer
        # granted it and before the driver knows its id. They are delivered on unblock.
        old = signal.pthread_sigmask(signal.SIG_BLOCK, set(GUARDED_SIGNALS))
        payload = None
        unreadable = False
        try:
            res = self.gozer("acquire", "--exact", self.a.board, "--chips", "1", "--who", WHO,
                             "--reason", REASON, "--owner-pid", str(self.pid), "--no-queue",
                             "--json", check="A")
            try:
                payload = json.loads(self.last_stdout)
            except (json.JSONDecodeError, TypeError):
                unreadable = res["exit_code"] is not None
            if res["exit_code"] == 0 and payload and payload.get("granted") and payload.get("lease_id"):
                self.lease_id = payload["lease_id"]
                self.grant = payload
                self.lease_live = True
                self.t_acquire = self.clock()
                # gozer's own timestamp format, so the history log can be compared by text.
                self.acquired_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)
        if unreadable and not self.lease_live:
            msg = (f"gozer acquire gave output the driver could not read. A lease owned by the "
                   f"driver pid {self.pid} may exist. It will be reaped by gozer once the driver "
                   "exits, and no reset runs for it.")
            print(msg, file=sys.stderr)
            self.safe_notice("A.malformed", False, msg, **res)
        if not self.lease_live:
            self.notice("A", False, f"acquire failed (exit {res['exit_code']})", **res)
            raise AcquireFailed("acquire failed")
        self.hook("after_acquire", driver=self)
        status_res, payload2 = self.status("A")
        chips = self.guarded_board(payload2)
        g = self.grant
        want_bdfs = {c["bdf"] for c in chips}
        want_devs = {c["dev_index"] for c in chips}
        # The grant must be exactly our board: no extra chip, one unit, no neighbours.
        shape_ok = (sorted(g.get("chips", [])) == sorted(want_bdfs)
                    and len(g.get("units", [])) == 1
                    and set(g.get("dev_indices", [])) == want_devs
                    and len(g.get("dev_indices", [])) == len(want_devs)
                    and not g.get("neighbours")
                    and g.get("owner_pid") == self.pid
                    and self.a.board in g.get("chips", []))
        ok = shape_ok and self.all_claimed(chips)
        self.notice("A", ok, f"lease {self.lease_id} granted; chips CLAIMED" if ok else
                    "lease granted but not as expected", acquire=res, grant=self.grant,
                    status=status_res, shape_ok=shape_ok)
        if not ok:
            raise AcquireFailed("granted lease does not match the request")

    def check_h2(self):
        t = self.start_child("H2")
        self.hook("after_h2_open", driver=self)
        res, first, _ = self.wait_state(lambda f, cs: f["state"] == "HELD")
        ps = self.run_cmd(["ps", "-eo", "pid,ppid,pgid,sid,args"], self.a.cmd_timeout)
        # Keep the driver's row, the child's row and anything in the child's process group.
        # The full listing is long, and the evidence field would cut it off.
        rows = [ln for ln in self.last_stdout.splitlines()
                if len(ln.split()) >= 3 and (ln.split()[0] == str(self.pid)
                                             or ln.split()[0] == str(self.child.pid)
                                             or ln.split()[2] == str(self.child.pgid))]
        state = first["state"]
        self.notice("H2", state == "HELD",
                    "child holds the device and chip is HELD" if state == "HELD" else
                    f"chip shows {state}, expected HELD",
                    chip_state=state, open_seconds=t, status=res, ps_rows=rows,
                    child_pid=self.child.pid, child_pgid=self.child.pgid)

    def refusal_probe(self):
        """`gozer reset` while the child holds the device. It must refuse and run nothing.

        This one call runs with GOZER_RESET_CMD pointed at a stand-in that only writes a marker
        and fails. If gozer does not refuse, the stand-in runs, the real chips are not reset
        under a live device open, and the run stops.
        """
        marker = self.out_dir / "refusal-probe-marker"
        stand_in = self.out_dir / "refusal-probe-reset.sh"
        stand_in.write_text(f'#!/bin/sh\necho "$@" >> {shlex.quote(str(marker))}\nexit 1\n')
        stand_in.chmod(0o755)
        probe_env = dict(os.environ, GOZER_RESET_CMD=str(stand_in))
        r = self.gozer("reset", self.lease_id, "--json", timeout=self.a.reset_timeout,
                       check="H3.refuse", env=probe_env, protect=True)
        text = r["stdout"] + r["stderr"]
        ran = marker.exists()
        ok = r["exit_code"] == 15 and "still open" in text and not ran
        self.notice("H3.refuse", ok,
                    "reset refused (exit 15, `still open`) and nothing ran" if ok else
                    f"refusal failed: exit {r['exit_code']}, stand-in reset ran: {ran}",
                    stand_in_ran=ran, **r)
        if not ok:
            raise Abort(f"the refusal failed: gozer reset exited {r['exit_code']}; "
                        f"stand-in reset ran: {ran}")
        self.deliver_pending()

    def do_reset(self, check: str, measure_name: str):
        """A real `gozer reset`. Anything but exit 0 and status `reset` stops the run."""
        t0 = time.monotonic()
        r = self.gozer("reset", self.lease_id, "--json", timeout=self.a.reset_timeout,
                       check=check, protect=True)
        elapsed = round(time.monotonic() - t0, 3)
        try:
            status = json.loads(self.last_stdout).get("status")
        except (json.JSONDecodeError, AttributeError):
            status = None
        if r["exit_code"] != 0 or status != "reset":
            self.notice(check, False, f"gozer reset exited {r['exit_code']} (status {status})", **r)
            raise Abort(f"gozer reset exited {r['exit_code']} (status {status})")
        self.hook("before_reset_status", driver=self, check=check)
        res, first, chips = self.wait_state(lambda f, cs: self.all_claimed(cs))
        claimed = self.all_claimed(chips)
        self.notice(check, claimed,
                    f"reset exited 0 in {elapsed} s; every granted chip CLAIMED" if claimed else
                    "reset ran but the chips are not all CLAIMED", status=res, **r)
        self.measure(measure_name, elapsed)
        if not claimed:
            raise Abort(f"chips not CLAIMED after {check}")
        self.hook("after_reset", driver=self, check=check)
        self.deliver_pending()

    def pause_after_open(self):
        """Wait after OPENED so a person can look at the child before the probe.

        The wait is in one second slices, so a signal is handled promptly.
        """
        secs = self.a.pause_after_open
        if secs <= 0:
            return
        pid = self.child.pid
        print(f"The child (pid {pid}) holds the device. Pausing {secs:g} s before the refusal "
              f"probe. You can check: ls -l /proc/{pid}/fd   and   gozer status", file=sys.stderr)
        left = secs
        while left > 0:
            step = min(1.0, left)
            self.sleep(step)
            left -= step
        self.hook("after_pause", driver=self)

    def check_h3(self):
        self.pause_after_open()
        self.refusal_probe()
        self.hook("after_refuse", driver=self)
        _, payload = self.status()
        self.guarded_board(payload)
        info = self.stop_child("H3.stop")
        self.notice("H3.stop", info.get("exit_code") == 0 and info["how"] == "quit",
                    f"child stopped ({info['how']})", **info)
        if not self.confirm_gone("H3.gone"):
            raise Abort("the child is not confirmed gone; no reset was run")
        self.do_reset("H3.reset", "reset_seconds")
        self.check_smi()

    def check_smi(self):
        if self.a.tt_smi.lower() == "none":
            self.decision("tt-smi snapshot skipped (--tt-smi none)")
            return
        r = self.run_cmd([self.a.tt_smi, "-s"], self.a.cmd_timeout)
        devices = None
        try:
            doc = json.loads(self.last_stdout)
            info = doc.get("device_info") if isinstance(doc, dict) else doc
            devices = len(info) if isinstance(info, list) else None
        except (json.JSONDecodeError, OSError, subprocess.SubprocessError):
            pass
        self.notice("H3.smi", r["exit_code"] == 0,
                    f"tt-smi -s exited {r['exit_code']}, {devices} devices listed", devices=devices, **r)

    def check_h4(self):
        t = self.start_child("H4")
        self.notice("H4", True, "device opened after the in-place reset", open_seconds=t,
                    output=self.child.output())
        info = self.stop_child("H4.stop")
        self.notice("H4.stop", info.get("exit_code") == 0 and info["how"] == "quit",
                    f"child stopped ({info['how']})", **info)
        if not self.confirm_gone("H4.gone"):
            raise Abort("the child is not confirmed gone after H4; no reset was run")
        self.do_reset("H4.reset", "reset2_seconds")

    def check_h1(self):
        want = self.a.idle_seconds
        # The age counts from the acquire. The clock and sleep are injected so a test does
        # not have to wait.
        while True:
            age = self.clock() - self.t_acquire
            if age >= want:
                break
            self.sleep(min(want - age, 30.0))
        self.hook("before_idle_status", driver=self, age=age)
        res, payload = self.status("H1")
        chips = self.guarded_board(payload)
        ours = self.grant_chips(chips)
        ok = bool(ours) and all(c["state"] == "CLAIMED" for c in ours)
        self.notice("H1", ok, f"lease {age:.0f} s old, chips {[c['state'] for c in ours]}",
                    lease_age_seconds=round(age, 3), status=res)
        self.measure("lease_age_at_idle_check_seconds", round(age, 3))
        others = self.other_leases(payload)
        if others:
            # reconcile reaps every stale lease on the box, including other agents' leases.
            whos = sorted({str(c["who"]) for c in others})
            self.decision(f"reconcile skipped: another lease is on the box ({', '.join(whos)})",
                          leases=[{"bdf": c["bdf"], "who": c["who"], "pid": c["pid"]} for c in others])
            return
        r = self.gozer("reconcile", "--json", check="H1.reconcile")
        res2, payload2 = self.status("H1.reconcile")
        chips2 = self.guarded_board(payload2)
        alive = any(self.is_ours(c) for c in chips2)
        self.notice("H1.reconcile", r["exit_code"] == 0 and alive,
                    "lease survived reconcile" if alive else "lease gone after reconcile",
                    reconcile=r, status=res2)

    def state_root(self) -> str:
        return os.environ.get("GOZER_ROOT") or DEFAULT_STATE_ROOT

    def history_root(self) -> str:
        return os.environ.get("GOZER_HISTORY_ROOT") or self.state_root()

    def check_h6(self):
        t0 = time.monotonic()
        r = self.gozer("release", self.lease_id, "--json", timeout=self.a.reset_timeout,
                       check="H6", protect=True)
        message = ""
        try:
            message = str(json.loads(self.last_stdout).get("message", ""))
        except (json.JSONDecodeError, AttributeError):
            pass
        if r["exit_code"] in (0, 13):
            # 13 means gozer no longer knows the lease. Either way there is nothing left to release.
            self.released = True
            self.lease_live = False
        self.hook("after_release", driver=self)
        # gozer release exits 0 even when its reset failed. Read the history log for the
        # reset result, and refuse the messages that say the reset did not happen.
        confirmed = release_reset_confirmed(self.history_root(), self.lease_id,
                                            since=self.acquired_at)
        bad_text = "NOT marked clean" in message or "not resetting" in message
        ok = r["exit_code"] == 0 and confirmed is True and not bad_text
        why = ("release exited 0 and its reset ran and succeeded" if ok else
               f"release exit {r['exit_code']}, reset confirmed in history: {confirmed}, "
               f"message: {message!r}")
        self.notice("H6.release", ok, why, history_reset_ok=confirmed, **r)
        self.measure("release_seconds", round(time.monotonic() - t0, 3))
        units = self.grant.get("units", [])
        root = Path(self.state_root())
        left = [str(p) for p in [root / "leases" / f"{self.lease_id}.json",
                                 *[root / "gate" / f"{u}.lock" for u in units]] if p.exists()]
        self.notice("H6.files", not left,
                    "no lease record or unit lock left under GOZER_ROOT" if not left else
                    f"left behind: {left}", left=left)
        res, payload = self.status()
        chips = self.board_chips(payload)
        free = bool(chips) and all(c["state"] == "FREE" and not c.get("who") for c in chips)
        pg = self.run_cmd(["pgrep", "-g", str(self.child.pgid)], self.a.cmd_timeout) if self.child else None
        group_empty = pg is None or (pg["exit_code"] == 1 and pg["stdout"].strip() == "")
        self.notice("H6.free", free and group_empty,
                    "board FREE and no process in the child's group" if free and group_empty else
                    f"board free={free}, group empty={group_empty}",
                    status=res, pgrep=pg)
        self.snapshot_docker("H6.docker", "docker ps recorded")
        self.deliver_pending()

    # ---- the run --------------------------------------------------------

    def run(self) -> int:
        code = EXIT_INTERNAL
        try:
            try:
                if not self.preflight():
                    code = EXIT_PREFLIGHT
                else:
                    self.acquire()
                    self.check_h2()
                    self.check_h3()
                    self.check_h4()
                    self.check_h1()
                    self.check_h6()
                    code = EXIT_PASS if all(ok for _, ok in self.results) else EXIT_FAIL
            except AcquireFailed as exc:
                self.decision("acquire failed; exiting 3", reason=str(exc))
                code = EXIT_ACQUIRE
            except Abort as exc:
                self.notice("ABORT", False, f"stop condition: {exc}", reason=str(exc))
                self.decision("aborting the run and cleaning up", reason=str(exc))
                code = EXIT_FAIL
            except Interrupted as exc:
                self.notice("ABORT", False, f"stop condition: {exc}", reason=str(exc))
                code = EXIT_FAIL
            except Exception:
                tb = traceback.format_exc()
                self.notice("ABORT", False, "internal error", reason="internal error", traceback=_trunc(tb))
                code = EXIT_INTERNAL
        finally:
            code = self.cleanup(code)
        return code

    def group_pids(self) -> str:
        """pgrep -g for the child's group: the pids still in it, or an empty string."""
        r = self.run_cmd(["pgrep", "-g", str(self.child.pgid)], self.a.cmd_timeout)
        return r["stdout"].strip() if r["exit_code"] == 0 else ""

    def cleanup_child(self):
        c = self.child
        if c is None:
            return
        if c.running():
            info = self.stop_child("cleanup.stop")
            forced = info["how"] in ("SIGTERM", "SIGKILL")
            self.safe_notice("cleanup.stop", not forced,
                             "child stopped in cleanup (quit)" if not forced else
                             f"child needed {info['how']}: board needs a reset", **info)
        # A worker can outlive the main pid and keep the device open. Look at the whole group.
        left = self.group_pids()
        if left:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(c.pgid, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and self.group_pids():
                time.sleep(0.1)
            self.safe_notice("cleanup.group", False,
                             "processes were left in the child's group; killed it: board needs a reset",
                             pgid=c.pgid, pids=left)

    def recovery_block(self) -> str:
        holders = ""
        try:
            _, payload = self.status()
            pids = sorted({p for c in self.board_chips(payload) for p in (c.get("pids_holding") or [])})
            holders = ", ".join(map(str, pids)) or "none seen"
        except Exception:
            holders = "unknown (gozer status failed)"
        return "\n".join([
            "!!! The driver could not release its lease.",
            f"    lease id: {self.lease_id}",
            f"    board: {self.a.board}",
            f"    pids holding the device: {holders}",
            "    Recovery:",
            "      1. Wait for the holder to exit (or stop it by its own tooling).",
            f"      2. Run: gozer release {self.lease_id}",
            "      Rules: never --force, never tt-smi -r by hand.",
        ])

    def cleanup_release(self, code: int) -> int:
        if not self.lease_id or self.released:
            return code
        if self.abandoned:
            # A reset or release is still running in gozer. A second one would reset the same
            # chips again, so the driver leaves the lease for the person to release.
            self.recovery_text = self.recovery_block() + (
                f"\n    gozer was still running: {' '.join(self.abandoned)}. Let it finish first.")
            print(self.recovery_text, file=sys.stderr)
            self.safe_notice("cleanup.release", False,
                             "release skipped: a gozer reset or release is still running",
                             abandoned=self.abandoned)
            return max(code, EXIT_FAIL) if code != EXIT_INTERNAL else code
        attempts = 0
        while True:
            r = self.gozer("release", self.lease_id, "--json", timeout=self.a.reset_timeout,
                           protect=True)
            if r["exit_code"] in (0, 13):
                self.released = True
                self.lease_live = False
                break
            if r["exit_code"] == 15 and attempts < RELEASE_RETRIES:
                # 15 means a device is still open. Give the holder a moment. Never --force.
                attempts += 1
                self.sleep(RELEASE_RETRY_DELAY)
                continue
            break
        bad = "NOT marked clean" in r["stdout"]
        # 13 means the lease is already gone, which is what cleanup wants.
        ok = r["exit_code"] in (0, 13) and not bad
        if not self.released:
            self.recovery_text = self.recovery_block()
            print(self.recovery_text, file=sys.stderr)
        self.safe_notice("cleanup.release", ok,
                         f"release in cleanup exited {r['exit_code']} after {attempts} retries"
                         + ("" if not bad else "; its reset failed: board needs a reset"),
                         retries=attempts, **r)
        if not ok and code == EXIT_PASS:
            code = EXIT_FAIL
        return code

    def end_snapshots(self):
        try:
            res = self.gozer("status", "--json")
            self.safe_notice("END.status", True, "gozer status at end", **res)
            self.snapshot_docker("END.docker", "docker ps at end")
        except BaseException as exc:
            self.safe_notice("END.status", False, f"end snapshot failed: {exc!r}")

    def cleanup(self, code: int) -> int:
        """Stop the child, then release the lease. Always runs, and runs once.

        Signals are held off from the first line, so nothing can cut cleanup short. Each step
        has its own guard: a failure in one step must not skip the release.
        """
        if self.cleaned:
            return code
        old = signal.pthread_sigmask(signal.SIG_BLOCK, set(GUARDED_SIGNALS))
        self.in_cleanup = True
        signal.pthread_sigmask(signal.SIG_SETMASK, old)    # the handler now ignores them
        try:
            self.cleanup_child()
        except BaseException as exc:
            self.safe_notice("cleanup.stop", False, f"stopping the child failed: {exc!r}")
            code = max(code, EXIT_FAIL) if code != EXIT_INTERNAL else code
        try:
            code = self.cleanup_release(code)
        except BaseException as exc:
            self.safe_notice("cleanup.release", False, f"release failed: {exc!r}")
            if code == EXIT_PASS:
                code = EXIT_FAIL
        self.end_snapshots()
        self.cleaned = True
        return code


def _one_line(text: str, limit: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "..."


def build_summary(entries: list[dict], code: int, recovery: str | None = None) -> str:
    start = entries[0]["data"] if entries and entries[0]["event"] == "run_start" else {}
    ver = start.get("gozer_version", {})
    lines = ["# Hardware check summary", "",
             f"gozer: {start.get('gozer_path')}  version: {_one_line(ver.get('stdout', ''))}",
             f"board: {start.get('options', {}).get('board')}", ""]
    by_id = {}
    for e in entries:
        d = e["data"]
        if e["event"] == "notice" and "check" in d:
            by_id[d["check"]] = d
            lines.append(f"{'PASS' if d['ok'] else 'FAIL'}  {d['check']:<14} "
                         f"{d.get('evidence', {}).get('summary', '')}")
    lines += ["", "Measurements:"]
    for e in entries:
        if e["event"] == "measurement":
            lines.append(f"  {e['data']['name']} = {e['data']['value']} {e['data']['unit']}")
    stops = [d["evidence"].get("reason") for d in by_id.values() if d["check"] == "ABORT"]
    not_run = [c for c in EXPECTED_CHECKS if c not in by_id] if stops else []
    lines += ["", "Stop condition: " + (stops[0] if stops else "none"),
              "Not run because of the stop: " + (", ".join(not_run) if not_run else "none")]
    decisions = [e["data"]["decision"] for e in entries if e["event"] == "decision"]
    skip = [d for d in decisions if d.startswith("reconcile skipped")]
    if "H1.reconcile" in by_id:
        lines.append("Reconcile: ran, " + by_id["H1.reconcile"]["evidence"]["summary"])
    elif skip:
        lines.append("Reconcile: skipped, " + skip[0][len("reconcile skipped: "):])
    else:
        lines.append("Reconcile: not reached")
    lines += ["", "H5 (container server) is not part of this driver.",
              "Detached: `gozer acquire --json` has no `detached` field. The grant's owner_pid "
              "is the driver pid, so gozer judges the lease by that pid.", ""]

    def snap(label, check):
        d = by_id.get(check)
        lines.append(f"{label}: " + (_one_line(d["evidence"].get("stdout", "")) or "(empty)"
                                     if d else "(not taken)"))
    snap("gozer status at start", "P")
    snap("docker ps at start", "P.docker")
    snap("gozer status at end", "END.status")
    snap("docker ps at end", "END.docker")
    lines += ["", "If the driver is killed with SIGKILL, no cleanup runs. The child's stdin closes. "
              "A child that has finished opening the device closes it and exits, the lease goes "
              "STALE and gozer reaps it without a reset. A child that still holds the device "
              "keeps the lease, shown as HELD-FOREIGN: wait for it to exit, then run "
              "`gozer release <lease-id>`. Never use --force. Never run tt-smi -r by hand."]
    if recovery:
        lines += ["", recovery]
    lines += ["", f"Exit code: {code}", ""]
    return "\n".join(lines)


def make_out_dir(board: str, base="runs/hardware-check", now=None) -> Path:
    """A new directory for one run: <base>/<UTC timestamp>-<board BDF>.

    The BDF is in the name so that two drivers started in the same second, one per board, never
    share a ledger. If the name is taken anyway (the same board twice in one second), a numeric
    suffix is added. mkdir without exist_ok is what claims the name, so there is no race.
    """
    stamp = time.strftime("%Y%m%dT%H%M%SZ", now or time.gmtime())
    name = f"{stamp}-{board.replace(':', '-')}"
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    n = 0
    while True:
        path = base / (name if n == 0 else f"{name}-{n}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            n += 1


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python3 -m orchard.hardware_check",
                                description="Run the hold-through-swap hardware checks on one board.")
    p.add_argument("--board", required=True, help="BDF of the board's first chip, e.g. 0000:03:00.0")
    p.add_argument("--out-dir", default=None,
                   help="default runs/hardware-check/<UTC timestamp>-<board BDF>/")
    p.add_argument("--gozer", default=os.environ.get(GOZER_ENV),
                   help=f"path to a gozer that has `reset` (the tt-gozer branch binary). Required; "
                        f"default ${GOZER_ENV}")
    p.add_argument("--tt-smi", default="none",
                   help="path to tt-smi to take a snapshot after the reset; default none. "
                        "tt-smi -s opens every device on the box. Do not use it while another "
                        "driver is active.")
    p.add_argument("--allow-gozer-env", action="store_true",
                   help="test-only: run even though GOZER_* variables are set in the environment")
    p.add_argument("--pause-after-open", type=float, default=0,
                   help="seconds to wait after H2's OPENED and before the refusal probe, so a "
                        "person can look at the child (default 0)")
    p.add_argument("--docker", default="docker", help="path to docker, or 'none' to skip docker ps")
    p.add_argument("--child-cmd", default=None,
                   help="JSON list: the child's argv. {bdf} and {dev} are filled in. "
                        "Default opens a 1x1 mesh with ttnn.")
    p.add_argument("--env-script", default=os.environ.get(ENV_SCRIPT_ENV),
                   help=f"shell script the default child sources to set up tt-metal (it must "
                        f"leave ttnn importable). Required without --child-cmd; default "
                        f"${ENV_SCRIPT_ENV}")
    p.add_argument("--python", default=os.environ.get(PYTHON_ENV),
                   help=f"python that can import ttnn, run by the default child. Required without "
                        f"--child-cmd; default ${PYTHON_ENV}")
    p.add_argument("--idle-seconds", type=float, default=960.0)
    p.add_argument("--opened-timeout", type=float, default=300.0)
    p.add_argument("--stop-timeout", type=float, default=30.0,
                   help="seconds to wait after quit, and again after SIGTERM")
    p.add_argument("--cmd-timeout", type=float, default=120.0, help="timeout for gozer, ps, docker")
    p.add_argument("--reset-timeout", type=float, default=600.0, help="timeout for reset and release")
    args = p.parse_args(argv)
    if not args.gozer:
        p.error(f"--gozer is required: pass the path to a gozer that has `reset` (the tt-gozer "
                f"branch binary), or set {GOZER_ENV}")
    if not args.child_cmd:
        missing = [f"{flag} (or {var})" for flag, var, value in
                   (("--env-script", ENV_SCRIPT_ENV, args.env_script),
                    ("--python", PYTHON_ENV, args.python)) if not value]
        if missing:
            p.error("the default child needs " + " and ".join(missing) + ": the tt-metal env "
                    "script to source and a python that can import ttnn. Or pass --child-cmd")
    return args


def main(argv=None, *, clock=time.monotonic, sleep=time.sleep, hook=None) -> int:
    args = parse_args(argv)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = make_out_dir(args.board)
    ledger = Ledger(out / "ledger.jsonl")
    drv = Driver(args, ledger, clock, sleep, hook, out_dir=out)

    def on_signal(signum, frame):
        # During cleanup the signal is ignored so the release cannot be cut short. During a
        # reset or release it waits until the call has finished.
        if drv.in_cleanup:
            return
        if drv.defer:
            drv.pending_signal = drv.pending_signal or signum
            return
        raise Interrupted(f"signal {signal.Signals(signum).name}")

    old = {}
    try:
        for s in GUARDED_SIGNALS:
            old[s] = signal.signal(s, on_signal)
    except ValueError:
        pass                     # not the main thread; the finally block still runs
    try:
        gz = shutil.which(args.gozer) or args.gozer
        ver = drv.gozer("--version")
        ledger.append("run_start", None, options=vars(args), gozer_path=gz, gozer_version=ver,
                      driver_pid=drv.pid, gozer_env=drv.gozer_env())
        if ver["exit_code"] != 0:
            drv.notice("ABORT", False, "gozer --version failed", reason="gozer --version failed", **ver)
            code = EXIT_INTERNAL
        else:
            try:
                code = drv.run()
            except BaseException:
                # A signal that landed in the gap before run()'s own cleanup. Cleanup is
                # idempotent, so running it here is safe.
                drv.cleanup(EXIT_FAIL)
                raise
        summary = build_summary(ledger.read(), code, drv.recovery_text)
        (out / "summary.md").write_text(summary)
        print(summary)
        return code
    except Exception:
        traceback.print_exc()
        return EXIT_INTERNAL
    finally:
        for s, h in old.items():
            signal.signal(s, h)
        ledger.close()


if __name__ == "__main__":
    sys.exit(main())
