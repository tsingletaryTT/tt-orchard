# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The `hardware_check` tool of a kernel-probe attempt: the policy around one chip run, and the small process that holds the lease.

The agent's shell cannot run gozer (the command runner refuses it and the shell has a no-chip device mask), so the probe gives the
agent a tool. This module holds what makes the tool safe to leave on. The probe passes two functions, `prepare` and `run`, which do
the work with the probe's own copy of the kernel, static gate and stall watcher (orchard/kernelprobe.py, `make_hardware_check`).

The lease is never held by the probe's process. Each call starts a short-lived wrapper (`python hwtool.py run ...`, below) in its own
session. The wrapper takes the lease with `gozer acquire --owner-pid <the wrapper's pid>`, waits on one queue ticket with `gozer
wait` when the board is busy, runs the shipped harness under the lease, releases it, and exits. If the wrapper dies by any signal,
gozer sees a dead owner and reaps the lease. The probe only starts the wrapper, waits for it with a hard timeout, kills its process
group on any exception, and then reads `gozer status` once. The probe repairs nothing: if that read shows a chip of ours still held, or
cannot be read, the tool stops (state `stopped`) and the probe ends with exit 5.

Three states: `idle` (no call is running), `running` (a call is running; a second call is answered at once) and `stopped` (the
check after a call found something held, or could not tell; every later call is refused, and so is every later attempt).

Rules, each tested in tests/test_agent_hw.py and tests/test_agent_hw_states.py:
- The tool takes at most a short `note`. Any other argument is an error and nothing runs.
- At most `max_calls` calls per attempt (a call that waits for a busy board counts, so the agent cannot poll for ever; a call
  refused before it reached the wrapper does not) and at most `lease_s` seconds of chip time in all. `cleanup_s` (gozer's term grace,
  which the stop and the chip reset can use in full) is reserved: a run is stopped at `call_timeout_s` or at the seconds left minus
  `cleanup_s`, whichever is less, and a call needs `min_call_s` seconds of run time left after that reserve.
- A busy board is waited for on one ticket for at most `wait_s` seconds; then the wrapper cancels the ticket and the agent is told
  the board is busy.
- Chip time is the wrapper's own: the seconds from its grant to the end of its release, taken from its own clock. A wrapper that
  was killed leaves no end stamp, and the probe then counts the whole run as an upper bound.
- After a hang, hardware checks are off for the rest of the attempt.
- Every call is recorded, and the probe writes the record into attempt.json and the report.

The choices (6 calls, 600 lease seconds, 240 s per call, 120 s of waiting, 210 s of cleanup reserve, 60 s minimum) are not measured
against a population of attempts. A whole call took about 47 s when measured (the device was open for about 0.5 s; the rest is the
lease and the reset that its release makes) (docs/superpowers/specs/2026-10-27-agent-hardware-check-design.md)."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

NOTE_MAX = 200                      # characters of the agent's note that are kept
TAIL_LINES = 15                     # lines of the run's output shown after a failure that is not a numeric one
TAIL_CHARS = 1200
VIOLATIONS_SHOWN = 5

# ---- the wrapper's constants -------------------------------------------------------------------------------------------------------
WRAP_BUSY = 10                      # the wrapper's exit status when the board stayed busy for the whole wait (the ticket was cancelled)
WRAP_NO_LEASE = 12                  # gozer would not give a lease, or gave one with no device list: the harness was not run
WRAP_TIMEOUT = 124                  # the harness was still running at the call timeout (counted from the grant) and was stopped
HEARTBEAT_S = 10.0                  # while waiting for a board the wrapper prints a new line this often, so the stall watcher sees progress
ACQUIRE_TIMEOUT_S = 60.0            # `gozer acquire` answers at once; one that takes this long is given up
STATUS_TIMEOUT_S = 30.0
WAIT_MARGIN_S = 30.0                # `gozer wait` is given its allowance plus this much before the wrapper gives up on the command itself
RELEASE_GRACE_S = 90.0              # the release resets the chips (about 42 s measured); the outer timeout leaves this much for it
RESET_FAILED_TEXT = ("NOT marked clean", "not resetting")      # `gozer release` exits 0 even when the reset failed; these say so
_NAME = re.compile(r"[0-9A-Za-z]+")


@dataclass(frozen=True)
class HwPolicy:
    """The limits of one attempt's hardware checks. `max_calls` 0 turns the tool off."""
    max_calls: int = 6
    lease_s: float = 600.0          # chip seconds in all: the seconds the wrapper held a lease, its release and reset included
    call_timeout_s: float = 240.0   # a run's backstop; the stall watcher usually stops a hang sooner
    wait_s: float = 120.0           # how long one call waits for a busy board
    min_call_s: float = 60.0        # seconds of run time that must be left, after the cleanup reserve, for a call to start
    cleanup_s: float = 210.0        # reserved for the stop (gozer's term grace, 120 s) and the separate release and reset (90 s)

    @property
    def enabled(self) -> bool:
        return self.max_calls > 0

    def as_dict(self) -> dict:
        return {"max_calls": self.max_calls, "lease_s": self.lease_s, "call_timeout_s": self.call_timeout_s,
                "wait_s": self.wait_s, "min_call_s": self.min_call_s,
                "cleanup_s": self.cleanup_s}


def schema(policy: HwPolicy) -> dict:
    """The tool as the model sees it. The description states the cost and the limit of this attempt's policy."""
    return {"type": "function", "function": {
        "name": "hardware_check",
        "description": ("Run your current kernel.py on a Tenstorrent chip and get the result of the probe's hardware check. "
                        "A whole call takes about 47 s (the kernel runs in under a second; the rest is the lease and the chip reset), and each attempt has "
                        f"a limited number of calls ({policy.max_calls} in this attempt; every answer says how many are "
                        "left). The simulator check after each write is much cheaper: use this for a kernel the "
                        "simulator cannot run (a FAIL: sim_gap), for a numeric check at hardware precision after the "
                        "simulator passes, or for a question about the hardware itself. If every board is busy it waits "
                        f"up to {policy.wait_s:.0f} s and then says so. The probe judges the copy of kernel.py it takes "
                        "now; nothing you say changes the verdict."),
        "parameters": {"type": "object",
                       "properties": {"note": {"type": "string", "description":
                                               f"optional, at most {NOTE_MAX} characters, kept in the record"}},
                       "required": [], "additionalProperties": False}}}


def outer_backstop_s(wait_s: float, call_timeout_s: float) -> float:
    """The probe's hard timeout on the whole wrapper: the sum of every phase it can be in, each with its own allowance. The acquire
    (ACQUIRE_TIMEOUT_S), the wait for a busy board (`wait_s` and its margin), the run after the grant (`call_timeout_s`, which the
    wrapper enforces itself from the grant) and the release (RELEASE_GRACE_S). The wrapper's own deadlines all end before this, so
    the outer timeout never cuts a phase short because an earlier phase used its time."""
    return ACQUIRE_TIMEOUT_S + max(wait_s, 0.0) + WAIT_MARGIN_S + call_timeout_s + RELEASE_GRACE_S


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float("inf")


# =====================================================================================================================================
# The probe's one look at gozer after a call
# =====================================================================================================================================


def _sh(argv, timeout):
    """(returncode, stdout, stderr) of one command. A timeout gives (None, "", reason). Looked up at call time so that the test suite
    can refuse it (tests/conftest.py): a real gozer command would take a lease or read the live board."""
    try:
        got = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout)
        return got.returncode, got.stdout, got.stderr
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout:.0f} s"
    except OSError as exc:
        return 127, "", str(exc)


def held_by_us(who: str, pids=(), *, gozer: str = "gozer", sh=None) -> str | None:
    """One `gozer status --json`. None when no chip is held under `who` (the tag this probe gave every wrapper) or by one of
    `pids` (the wrappers it started). Otherwise one sentence: the chips that are held, or that gozer could not be read. A read that
    fails counts as held: the check fails closed. Nothing is repaired here."""
    sh = sh or _sh
    try:
        code, out, err = sh([gozer, "status", "--json"], STATUS_TIMEOUT_S)
        data = json.loads(out) if code == 0 else None
    except Exception as exc:                                    # a runner that raises is a read that failed
        return f"cannot confirm that no chip is held: `gozer status` could not be read ({type(exc).__name__}: {exc})"
    chips = data.get("chips") if isinstance(data, dict) else None
    if not isinstance(chips, list):
        return ("cannot confirm that no chip is held: `gozer status` could not be read "
                f"({(err or out or f'exit {code}').strip()[:120]})")
    pids = {p for p in pids if isinstance(p, int)}

    def ours(row) -> bool:
        return isinstance(row, dict) and (row.get("who") == who or (row.get("pid") is not None and row.get("pid") in pids))
    held = sorted(str(c.get("bdf")) for c in chips if ours(c))
    queue = data.get("queue") if isinstance(data.get("queue"), list) else []
    tickets = sorted(str(q.get("ticket")) for q in queue if ours(q))          # a ticket left in the queue would be granted to nobody
    if not held and not tickets:
        return None
    parts = ([f"chip{'s' if len(held) != 1 else ''} {', '.join(held)} held"] if held else []) + (
        [f"ticket{'s' if len(tickets) != 1 else ''} {', '.join(tickets)} still queued"] if tickets else [])
    return (f"gozer still shows {' and '.join(parts)} by this probe's hardware call (who {who}); look with `gozer status` and use "
            "the gozer-gatekeeper skill")


# =====================================================================================================================================
# The wrapper: the only process that holds a lease
# =====================================================================================================================================


class _Stop(BaseException):
    """SIGTERM reached the wrapper. A BaseException so that no `except Exception` in the wrapper swallows it."""


def _json(text) -> dict:
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_atomic(path, data: dict) -> None:
    """Write the wrapper's record where the probe reads it. An error is ignored: the record is for the probe, and the wrapper's work
    (the release above all) goes on without it."""
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except OSError:
        pass


def _sha256_of(path) -> str | None:
    """The sha256 of a regular file (not a symlink), or None when it cannot be read as one."""
    import hashlib
    try:
        if os.path.islink(path) or not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def run_wrapper(*, who: str, reason: str, wait_s: float, result_path, command: list[str], gozer: str = "gozer",
                sh=None, spawn=None, now=time.time, out=None, pid: int | None = None,
                expect_sha256: str | None = None, kernel_file: str | None = None,
                call_timeout_s: float | None = None, mono=time.monotonic) -> int:
    """Take one chip, run `command` under the lease, release it, exit. Returns the exit status.

    The lease is requested with `gozer acquire --owner-pid <this process>`, so a wrapper that dies by any signal leaves a lease whose
    owner is dead, and gozer reaps it. A busy board gives a ticket; `gozer wait --timeout` is the one wait (it grants a detached lease,
    which gozer reaps by its own grace window if nothing holds the chips), and a wait that ends without a grant cancels the ticket.
    The release is in a `finally`, so a SIGTERM (it becomes `_Stop`) or a KeyboardInterrupt in the run still releases. The record
    in `result_path` is written after the grant and again at the end: it carries this process's own stamps, `granted_at` and
    `ended_at`, which are the chip time of the call."""
    sh = sh or _sh
    spawn = spawn or subprocess.Popen
    out = out or (lambda line: print(line, flush=True))
    pid = os.getpid() if pid is None else pid
    rec = {"pid": pid, "who": who, "started_at": now(), "granted_at": None, "ended_at": None, "lease_id": None, "chips": [],
           "outcome": None, "ticket": None, "cancel_error": None, "release_ok": None, "release_text": "", "returncode": None,
           "detached": None, "phase": "starting"}
    _write_atomic(result_path, rec)
    lease_id, ticket, status = None, None, 0
    try:
        code, text, err = sh([gozer, "acquire", "--json", "--chips", "1", "--exact", "1", "--who", who, "--reason", reason,
                              "--owner-pid", str(pid)], ACQUIRE_TIMEOUT_S)
        data = _json(text)
        grant = None
        if code == 0 and data.get("granted") is True and isinstance(data.get("lease_id"), str) and _NAME.fullmatch(data["lease_id"]):
            grant, rec["detached"] = data, False
        elif code == 10 and isinstance(data.get("ticket"), str) and _NAME.fullmatch(data["ticket"]):
            ticket = rec["ticket"] = data["ticket"]
            grant = _wait_for(ticket, wait_s, sh=sh, gozer=gozer, out=out)
            if grant is not None:
                ticket, rec["detached"] = None, True               # the ticket is spent: nothing left to cancel
            else:
                rec.update(outcome="busy")
                status = WRAP_BUSY
        else:
            rec.update(outcome="no_lease", release_text=((text or "") + (err or "")).strip()[:300] or f"gozer acquire exited {code}")
            status = WRAP_NO_LEASE
        if grant is not None:
            lease_id = rec["lease_id"] = grant["lease_id"]
            rec["chips"] = [c for c in grant.get("chips") or [] if isinstance(c, str)]
            rec["granted_at"], rec["phase"] = now(), "held"
            _write_atomic(result_path, rec)
            env = grant.get("env") if isinstance(grant.get("env"), dict) else {}
            devices = env.get("TT_VISIBLE_DEVICES") if isinstance(env.get("TT_VISIBLE_DEVICES"), str) else ",".join(rec["chips"])
            if not devices:
                rec.update(outcome="no_lease", release_text="gozer granted the lease with no device list, so the command was not run")
                status = WRAP_NO_LEASE
            elif expect_sha256 and _sha256_of(kernel_file) != expect_sha256:
                # The wait can last two minutes and the agent can write anywhere the operator can: the copy that was gated is
                # checked again, after the wait and before it runs.
                rec.update(outcome="changed", release_text="the kernel copy changed while the call waited for the board, so it was not run")
                status = WRAP_NO_LEASE
            else:
                status = _run_under_lease(command, devices, rec, spawn, call_timeout_s, mono)
    except _Stop:
        rec["outcome"], status = "interrupted", 143
    except KeyboardInterrupt:
        rec["outcome"], status = "interrupted", 130
    finally:
        # Every exit goes through here: give back what was taken. A second signal in this block is ignored (the first one is
        # what stopped the run), so the release is not cut short by the stop that is already in progress.
        previous = _ignore_term()
        try:
            if ticket is not None:
                rec["cancel_error"] = _cancel(ticket, sh, gozer)
            if lease_id is not None:
                _release(lease_id, rec, sh, gozer)
            rec["ended_at"], rec["phase"] = now(), "done"
            if rec["outcome"] is None:
                rec["outcome"] = "ran"
            _write_atomic(result_path, rec)
        finally:
            _restore_term(previous)
    if ticket is not None or rec["outcome"] == "busy":
        out(f"queued: ticket {rec['ticket']} was cancelled" if rec["cancel_error"] is None
            else f"queued: ticket {rec['ticket']} could not be cancelled: {rec['cancel_error']}")
    return status


def _wait_for(ticket: str, wait_s: float, *, sh, gozer: str, out) -> dict | None:
    """`gozer wait --timeout <wait_s>s TICKET`: the grant (a dict with lease_id, chips, env), or None when it ended without one. While
    it blocks, a new line is printed every HEARTBEAT_S, so a stall watcher on the wrapper's output sees progress."""
    seconds = max(int(wait_s + 0.999), 1)
    done = threading.Event()
    began = time.monotonic()
    beats = [0]

    def beat():
        while not done.wait(HEARTBEAT_S):
            beats[0] += 1                                  # a counter in the line: no two lines are alike, so the watcher sees no repeat
            out(f"waiting for the board: check {beats[0]}, {time.monotonic() - began:.0f} s of {seconds} s")
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        code, text, _ = sh([gozer, "wait", "--json", "--timeout", f"{seconds}s", ticket], seconds + 30.0)
    finally:
        done.set()
    data = _json(text)
    if code == 0 and data.get("granted") is True and isinstance(data.get("lease_id"), str) and _NAME.fullmatch(data["lease_id"]):
        return data
    return None


def _cancel(ticket: str, sh, gozer: str) -> str | None:
    """None when the ticket is gone (cancelled, or already consumed), else the reason it is not."""
    if not _NAME.fullmatch(ticket or ""):
        return f"not a ticket name: {ticket!r}"
    try:
        code, text, err = sh([gozer, "cancel", ticket], 60.0)
    except BaseException as exc:
        return f"{type(exc).__name__}: {exc}"
    if code in (0, 13):                                        # 13: no such ticket, so it is gone already
        return None
    return ((text or "") + (err or "")).strip()[:300] or f"gozer cancel exited {code}"


def _release(lease_id: str, rec: dict, sh, gozer: str) -> None:
    """`gozer release LEASE --json` (it resets the chips), with no timeout of its own: a reset must not be cut short, and the probe's
    hard timeout on the whole wrapper is the bound."""
    try:
        code, text, err = sh([gozer, "release", lease_id, "--json"], None)
    except BaseException as exc:
        rec.update(release_ok=False, release_text=f"{type(exc).__name__}: {exc}")
        return
    message = str(_json(text).get("message", "")) or (text or "").strip()
    if code == 0 and not any(t in message for t in RESET_FAILED_TEXT):
        rec["release_ok"] = True
    elif code == 13:                                           # no such lease: nothing is left to release
        rec["release_ok"] = True
    elif code == 0:
        rec.update(release_ok=False, release_text=f"the lease was released but its chip reset did not succeed: {message[:300]}")
    else:
        rec.update(release_ok=False, release_text=((text or "") + (err or "")).strip()[:300] or f"gozer release exited {code}")


def _stop_command(proc) -> None:
    """Ask the command to end, then kill it, with a second SIGTERM ignored meanwhile (the command is already being stopped)."""
    previous = _ignore_term()
    try:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except Exception:
            proc.kill()
            proc.wait()
    finally:
        _restore_term(previous)


def _run_under_lease(command, devices: str, rec: dict, spawn, call_timeout_s: float | None = None, mono=time.monotonic) -> int:
    """Run `command` with the lease's devices. The command is in the wrapper's own process group, so the probe's group kill reaches
    it. The call-timeout clock starts here, at the grant: the command is stopped at that deadline whatever the wait for the board took,
    and the stop is recorded as outcome `timeout` (exit status WRAP_TIMEOUT). The deadline is checked before every wait, so a late
    wake-up cannot extend it. On a stop (`_Stop` or KeyboardInterrupt) the command is asked to end and then killed, and the stop goes on."""
    env = dict(os.environ)
    env["TT_VISIBLE_DEVICES"] = devices
    proc = spawn(list(command), env=env)
    deadline = None if call_timeout_s is None else mono() + call_timeout_s
    try:
        while True:
            left = None if deadline is None else deadline - mono()
            if left is not None and left <= 0:
                _stop_command(proc)
                rec["outcome"] = "timeout"
                return WRAP_TIMEOUT
            try:
                code = proc.wait(timeout=left)
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        _stop_command(proc)
        raise
    rec["returncode"] = code
    return code if code >= 0 else 128 - code


def _ignore_term():
    try:
        return signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except (ValueError, OSError):                              # not the main thread (a test)
        return None


def _restore_term(previous) -> None:
    if previous is not None:
        try:
            signal.signal(signal.SIGTERM, previous)
        except (ValueError, OSError):
            pass


def wrapper_cli(argv: list[str], *, install_signals: bool = True, **inject) -> int:
    """`python hwtool.py run --who W --reason R --wait-s N --result PATH -- COMMAND...`. `inject` (sh, spawn, now, out, pid, gozer)
    goes to run_wrapper: the tests run the real wrapper in process with a fake gozer and a fake command."""
    if not argv or argv[0] != "run" or "--" not in argv:
        print("usage: hwtool.py run --who WHO --reason REASON --wait-s N --result PATH [--sha256 H --file PATH --call-timeout-s N] -- COMMAND...", file=sys.stderr)
        return 2
    cut = argv.index("--")
    opts, command = argv[1:cut], argv[cut + 1:]
    values = dict(zip(opts[::2], opts[1::2]))
    try:
        who, reason, result = values["--who"], values["--reason"], values["--result"]
        wait_s = float(values["--wait-s"])
    except (KeyError, ValueError):
        print("hwtool.py run: --who, --reason, --wait-s and --result are all required", file=sys.stderr)
        return 2
    if not command:
        return 2

    def stop(signum, frame):
        raise _Stop()
    if install_signals:
        signal.signal(signal.SIGTERM, stop)
    try:
        call_timeout = float(values["--call-timeout-s"]) if "--call-timeout-s" in values else None
    except ValueError:
        print("hwtool.py run: --call-timeout-s must be a number", file=sys.stderr)
        return 2
    return run_wrapper(who=who, reason=reason, wait_s=wait_s, result_path=result, command=command,
                       expect_sha256=values.get("--sha256"), kernel_file=values.get("--file"), call_timeout_s=call_timeout, **inject)


def read_wrapper_record(path) -> dict | None:
    """The wrapper's record, or None when it is missing or not readable."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# =====================================================================================================================================
# The tool
# =====================================================================================================================================


class HardwareCheck:
    """One attempt's hardware tool.

    prepare(k) -> (blocked, sha256): `blocked` is None when kernel.py was copied and passed the static gate, else the hw
    block that says why it cannot run (no chip is used).
    run(k, timeout_s, wait_s) -> (hw, event, text): one wrapper run of the prepared copy. `event` is None, "queued" (the board
    was busy for the whole wait; the wrapper cancelled the ticket and `hw["queued"]` says which) or "hang" (the stall watcher or the
    backstop stopped it). `hw` carries `lease_s` (the wrapper's own seconds from grant to the end of the release, or an upper
    bound when it has no end stamp), `waited_s`, `lease_id`, `wrapper_pid`, `release_error`, and may carry `device_s`, `numbers`
    and the judging fields; `text` is the run's output.
    check(pids) -> None or a sentence: the one look at gozer after a call (held_by_us). A sentence stops the tool."""

    def __init__(self, policy: HwPolicy, *, prepare, run, clock=time.monotonic, sleep=time.sleep,
                 stamp=_utc_stamp, on_record=None, check=None):
        self.policy, self.prepare, self.run = policy, prepare, run
        self.check = check                  # (set of wrapper pids) -> None, or a sentence about a chip that is still held
        self.stopped: str | None = None     # the sentence of the check that stopped the tool; None while it is not stopped
        self.pids: set[int] = set()         # the wrappers this attempt started
        self.on_record = on_record          # called with summary() each time a call is recorded; its errors are ignored
        self.clock, self.sleep, self.stamp = clock, sleep, stamp
        self._lock = threading.Lock()
        self.records: list[dict] = []
        self.counted = 0
        self.lease_used = 0.0
        self.bad_calls = 0
        self.hung = False

    def schema(self) -> dict:
        return schema(self.policy)

    @property
    def state(self) -> str:
        """`stopped`, `running` (a call is in progress) or `idle`."""
        if self.stopped:
            return "stopped"
        return "running" if self._lock.locked() else "idle"

    # ---- what the probe writes into attempt.json ----------------------------------------------------------------

    def summary(self) -> dict:
        out = {"enabled": self.policy.enabled, "policy": self.policy.as_dict(), "calls": [dict(r) for r in self.records],
               "calls_counted": self.counted, "bad_calls": self.bad_calls, "lease_s_used": round(self.lease_used, 1)}
        if self.stopped:
            out["stopped"] = self.stopped
        return out

    # ---- the call -----------------------------------------------------------------------------------------------

    def call(self, arguments) -> str:
        note = arguments.get("note", "") if isinstance(arguments, dict) else None
        if not isinstance(arguments, dict) or set(arguments) - {"note"} or not isinstance(note, str):
            self.bad_calls += 1
            return ("error: hardware_check takes no command and no path, only an optional `note` (a short string). "
                    "Nothing was run.")
        if not self._lock.acquire(blocking=False):
            return "error: a hardware check is already running; wait for its answer. Nothing was started."
        try:
            return self._call(note[:NOTE_MAX])
        finally:
            self._lock.release()

    def _left(self) -> str:
        return (f"budget: {max(self.policy.max_calls - self.counted, 0)} calls left, "
                f"{max(self.policy.lease_s - self.lease_used, 0):.0f} lease seconds left")

    def _record(self, **fields) -> dict:
        rec = {"at": self.stamp(), "k": None, "counted": False, "outcome": None, "category": None, "passed": False,
               "total_s": 0.0, "device_s": None, "lease_s": 0.0, "waited_s": 0.0, "queued_polls": 0,
               "cancelled": None, "timed_out": False, "note": "", "first_line": "", "pcc_min": None,
               "kernel_sha256": None, "lease_id": None}
        rec.update(fields)
        self.records.append(rec)
        self._persist()
        return rec

    def _persist(self) -> None:
        if self.on_record is not None:
            try:
                self.on_record(self.summary())
            except Exception:
                pass                                # keeping the record on disk must never end the call

    def _refuse(self, note: str, why: str) -> str:
        self._record(outcome="refused", note=note, first_line=why)
        return f"hardware_check refused: {why}"

    def _look(self) -> str:
        """The one check after a call that reached the wrapper. Fails closed: a check that raises is a sentence too. Returns the
        text to add to the answer ('' when nothing is held)."""
        try:
            sentence = self.check(set(self.pids)) if self.check is not None else None
        except Exception as exc:
            sentence = f"cannot confirm that no chip is held (the check raised {type(exc).__name__})"
        except BaseException:
            # An interrupt in the one look at gozer: it is not known whether a chip is held, so the tool stops, and the stop is
            # written before the interrupt goes on.
            self.stopped = "the check after the call was interrupted, so it is not known whether a chip is still held"
            self._persist()
            raise
        if not sentence:
            return ""
        self.stopped = sentence
        self._persist()
        return (f"\nSTOPPED: {sentence}. No further hardware call will be run in this attempt, and the probe will end the run. "
                "Use the simulator (the check after each write); do not try to clear the lease yourself.")

    def _call(self, note: str) -> str:
        p = self.policy
        if self.stopped:
            return self._refuse(note, f"the tool is stopped: {self.stopped}. No further hardware call is run. Use the simulator "
                                      "(the check after each write).")
        if self.hung:
            return self._refuse(note, "an earlier hardware run of this attempt hung, so hardware checks are off for the "
                                      "rest of it. Use the simulator (the check after each write).")
        if self.counted >= p.max_calls:
            return self._refuse(note, f"no hardware checks left: {self.counted} of {p.max_calls} used in this attempt. "
                                      "Use the simulator (the check after each write).")
        left = p.lease_s - self.lease_used - p.cleanup_s        # run time that is left once the release is reserved
        if left < p.min_call_s:
            return self._refuse(note, f"only {max(p.lease_s - self.lease_used, 0):.0f} lease seconds are left of "
                                      f"{p.lease_s:.0f}; {p.cleanup_s:.0f} are reserved for the cleanup (the stop and the "
                                      f"chip reset) and a call needs at least {p.min_call_s:.0f} to run. Use the "
                                      "simulator (the check after each write).")
        blocked, sha = self.prepare(self.counted + 1)
        if blocked is not None:
            return self._blocked(note, blocked, sha)
        self.counted += 1
        k = self.counted
        start = self.clock()
        finished = False
        try:
            timeout = min(p.call_timeout_s, p.lease_s - self.lease_used - p.cleanup_s)
            hw, event, text = self.run(k, timeout, max(p.wait_s, 0.0))
            finished = True
        finally:
            if not finished:                        # an interrupt or an error: the probe's runner has stopped the wrapper's group
                now = self.clock()
                held = max(now - start, 0.0)        # an upper bound: the wrapper may have held the board for all of it, or none
                self.lease_used += held
                self._record(k=k, counted=True, outcome="interrupted", note=note, kernel_sha256=sha,
                             total_s=round(held, 1), lease_s=round(held, 1), lease_upper_bound=True)
                self._look()                        # the one check; if it finds a chip held the tool is stopped
        charged: list[float] = []                   # the chip time this call was charged, so an interrupt does not charge it twice
        recorded = len(self.records)
        try:
            return self._finish(k, note, sha, start, hw, event, text, charged)
        except BaseException:
            # An interrupt that lands after the wrapper has exited (in the record, or in the status check). The finished call is
            # written to the record, with its chip time, before the interrupt goes on.
            if len(self.records) == recorded:
                held = charged[0] if charged else (float(hw["lease_s"]) if _number(hw.get("lease_s")) else max(self.clock() - start, 0.0))
                if not charged:
                    self.lease_used += held
                self._record(k=k, counted=True, outcome="interrupted", note=note, kernel_sha256=sha,
                             total_s=round(self.clock() - start, 1), lease_s=round(held, 1),
                             lease_id=hw.get("lease_id") if isinstance(hw.get("lease_id"), str) else None,
                             lease_upper_bound=True if hw.get("lease_upper_bound") else None)
            if not self.stopped:
                self._look()
            raise

    def _finish(self, k, note, sha, start, hw, event, text, charged) -> str:
        """The part of a call after the wrapper has exited: the record first, then the one look at gozer."""
        p = self.policy
        pid = hw.get("wrapper_pid")
        if isinstance(pid, int):
            self.pids.add(pid)                      # before the check: the check also matches chips held by this call's own wrapper
        total = round(self.clock() - start, 1)
        polls = 1 if event == "queued" else 0
        if event == "queued":
            q = hw.get("queued") if isinstance(hw.get("queued"), dict) else {}
            cancelled = bool(q.get("ticket")) and q.get("cancel_error") is None
            charged.append(0.0)
            self._record(k=k, counted=True, outcome="busy", category="queued", note=note, kernel_sha256=sha,
                         total_s=total, waited_s=total, queued_polls=polls, cancelled=cancelled,
                         first_line=str(hw.get("first_line") or ""))
            return self._busy_text(k, total, polls, cancelled, hw) + self._look()
        waited = float(hw["waited_s"]) if _number(hw.get("waited_s")) else 0.0
        elapsed = float(hw["lease_s"]) if _number(hw.get("lease_s")) else max(total - waited, 0.0)
        self.lease_used += elapsed
        charged.append(elapsed)
        device = hw.get("device_s") if _number(hw.get("device_s")) else None
        release_error = str(hw["release_error"]) if hw.get("release_error") else None
        lease_id = hw.get("lease_id") if isinstance(hw.get("lease_id"), str) else None
        upper = True if hw.get("lease_upper_bound") else None
        if event == "hang":
            self.hung = True
            self._record(k=k, counted=True, outcome="hang", category="hang", note=note, kernel_sha256=sha, total_s=total,
                         lease_s=round(elapsed, 1), waited_s=round(waited, 1), queued_polls=polls, timed_out=True,
                         release_error=release_error, lease_id=lease_id, lease_upper_bound=upper,
                         first_line=str(hw.get("first_line") or ""))
            return (f"hardware_check {k} of {p.max_calls}: HANG: the run was stopped ({hw.get('first_line')}) after "
                    f"{elapsed:.1f} s. The wrapper released the lease with `gozer release`, which resets the chip. Look with `gozer status` if in doubt (a lease marked stale is cleared "
                    "with `gozer reconcile`, the gozer-gatekeeper skill says how), and never reset a chip by hand. "
                    "Hardware checks are off for the rest of this attempt; the simulator still works."
                    f"{self._release_note(release_error)}\n{self._left()}" + self._look())
        passed = hw.get("passed") is True
        category = None if passed else hw.get("category")
        self._record(k=k, counted=True, outcome="pass" if passed else str(category), category=category, passed=passed,
                     note=note, kernel_sha256=sha, total_s=total, device_s=device, lease_s=round(elapsed, 1),
                     waited_s=round(waited, 1), queued_polls=polls, lease_id=lease_id, lease_upper_bound=upper,
                     release_error=release_error, first_line=str(hw.get("first_line") or ""),
                     pcc_min=hw.get("pcc_min") if _number(hw.get("pcc_min")) else None)
        answer = self._result_text(k, hw, text, total, waited, device, passed) + (
            "\n" + self._release_note(release_error).strip() if release_error else "")
        return answer + self._look()

    # ---- the answers --------------------------------------------------------------------------------------------

    @staticmethod
    def _release_note(error) -> str:
        if not error:
            return ""
        return (f" gozer could not release the lease ({error}); the lease may still be held: look with `gozer status`, "
                "and clear a stale one with `gozer reconcile`.")

    def _blocked(self, note: str, blocked: dict, sha) -> str:
        category = str(blocked.get("category"))
        self._record(outcome=category, category=category, note=note, kernel_sha256=sha,
                     first_line=str(blocked.get("first_line") or ""))
        lines = [f"hardware_check: FAIL: {category}: {blocked.get('first_line')}"]
        for v in [v for v in blocked.get("violations") or [] if isinstance(v, dict)][:VIOLATIONS_SHOWN]:
            lines.append(f"  {v.get('rule')} at line {v.get('line')}: {v.get('detail')}. {v.get('remedy')}")
        lines.append("The probe did not run this kernel, so no chip was used and this call was not counted.")
        lines.append(self._left())
        return "\n".join(lines)

    def _busy_text(self, k, total, polls, cancelled, hw) -> str:
        q = hw.get("queued") if isinstance(hw.get("queued"), dict) else {}
        tail = ("The queue ticket was cancelled." if cancelled else
                f"A ticket could not be cancelled ({q.get('cancel_error') or 'no ticket in gozer output'}; ticket "
                f"{q.get('ticket')}): look with `gozer status`.")
        return (f"hardware_check {k} of {self.policy.max_calls}: BUSY: the board was busy for {total:.0f} s "
                f"(waited on {polls} queue ticket). {tail} The chip was not used. Go on with the simulator (the check after each "
                f"write).\n{self._left()}")

    def _result_text(self, k, hw, text, total, waited, device, passed) -> str:
        p = self.policy
        if passed:
            head = (f"hardware_check {k} of {p.max_calls}: PASS (the probe judged this copy of kernel.py on the chip "
                    "with the hardware bars)")
        else:
            head = f"hardware_check {k} of {p.max_calls}: FAIL: {hw.get('category')}: {hw.get('first_line')}"
        lines = [head]
        sets = [s for s in hw.get("sets") or [] if isinstance(s, dict)]
        measures = []
        if _number(hw.get("pcc_min")):
            measures.append(f"pcc_min {hw['pcc_min']:.6g}")
        rng = hw.get("slope_range")
        if isinstance(rng, list) and len(rng) == 2 and all(_number(x) for x in rng):
            measures.append(f"slope {rng[0]:.4g} to {rng[1]:.4g}")
        if _number(hw.get("rel_l2_max")):
            measures.append(f"rel_l2_max {hw['rel_l2_max']:.3g}")
        if sets:
            measures.append(f"{len(sets)} input sets")
        if measures:
            lines.append(", ".join(measures))
        if not passed and hw.get("numbers"):
            lines.append(f"numbers: {hw['numbers']}")
        if not passed and hw.get("category") != "hw_numeric_mismatch" and text.strip():
            tail = "\n".join([l for l in text.splitlines() if l.strip()][-TAIL_LINES:])[-TAIL_CHARS:]
            lines.append("output (last lines):\n" + tail)
        dev = f"device {device:.1f} s" if device is not None else "device time not reported"
        wait = f", waited {waited:.1f} s for a lease" if waited >= 0.05 else ""
        lines.append(f"time: {total:.1f} s in all ({dev}; the rest is the lease and the chip reset){wait}")
        lines.append(self._left())
        return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(wrapper_cli(sys.argv[1:]))
