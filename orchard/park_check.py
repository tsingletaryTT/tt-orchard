"""Hardware check for park and restore (docs/runbooks/hardware-validation.md, "Park check").

Run it as: python3 -m orchard.park_check --board <BDF of the board's first chip>

It drives the real handoff code (orchard/handoff.py) through the real gozer adapter on one real
board. Two copies of orchard/fake_server.py stand in for the coder and the CPU stand-in. No model
is loaded and no process opens a device, so the chips stay CLAIMED and each `gozer reset` runs on
idle chips.

What it exercises: the adapter's status, acquire, reset and release against the real gozer and a
real board; the stop checks (ps, pgrep, ss, curl) against a real process; every park and restore
step and its ledger entry; the canary comparison. What it does not: a device held open
(orchard/hardware_check.py covers that), a container server, a real stand-in model, and a real
coder's restart time.

Expected: three resets of the board (park, restore, release), about 42 s each (spec section 3).

Signals: SIGINT and SIGTERM are held back while a reset or release runs, so neither is cut short;
a signal that arrives meanwhile takes effect when the call returns, and the cleanup (stop both
servers, release the lease) runs. They are held only around those calls. A blocked signal mask is
inherited by every child process, so a fake server started while signals were blocked would
ignore its SIGTERM. If the process is killed with SIGKILL, no cleanup runs. The lease is owned by
this pid, so gozer reaps it once the pid is dead (the fake servers hold no device). Stop any fake
server left behind by the pids in the ledger. Never use --force, and never run tt-smi -r by hand.

Exit codes: 0 pass, 1 a step failed or blocked, 2 preflight refused, 3 acquire failed.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sys
from pathlib import Path

from orchard.adapters import AdapterError
from orchard.adapters.gozer import GozerAdapter
from orchard.defaults import CHIPS_PER_BOARD, READY_POLL_S
from orchard.handoff import Blocked, Budgets, Handoff
from orchard.hardware_check import make_out_dir
from orchard.ledger import Ledger
from orchard.server import ServerControl, ServerError, ServerSpec, ServerStandIn

WHO = "orchard:park-check"
REASON = "validate park and restore"
CANARY_PROMPT = "park check: reply with the configured answer"
EXIT_PASS, EXIT_FAIL, EXIT_PREFLIGHT, EXIT_ACQUIRE = 0, 1, 2, 3
GUARDED = {signal.SIGINT, signal.SIGTERM}
FAKE_SERVER = Path(__file__).with_name("fake_server.py")


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python3 -m orchard.park_check",
                                description="Park and restore a fake coder on one real board.")
    p.add_argument("--board", required=True, help="BDF of the board's first chip, e.g. 0000:03:00.0")
    p.add_argument("--out-dir", default=None, help="default runs/park-check/<UTC time>-<BDF>/")
    p.add_argument("--gozer", default="gozer", help="a gozer that has `reset` (committed main, 0.3.2, has it)")
    p.add_argument("--port", type=int, default=20990, help="port of the fake coder")
    p.add_argument("--standin-port", type=int, default=20991, help="port of the fake stand-in")
    p.add_argument("--answer", default="the park check answer")
    p.add_argument("--ready-budget", type=float, default=60.0)
    p.add_argument("--ready-poll", type=float, default=READY_POLL_S)
    p.add_argument("--allow-gozer-env", action="store_true",
                   help="test only: run although GOZER_* variables are set")
    return p.parse_args(argv)


def fake_argv(port: int, answer: str) -> tuple[str, ...]:
    return (sys.executable, str(FAKE_SERVER), "--port", str(port), "--answer", answer,
            "--model", "fake")


@contextlib.contextmanager
def signals_held():
    """Hold SIGINT and SIGTERM back for the duration; they arrive when the block ends."""
    old = signal.pthread_sigmask(signal.SIG_BLOCK, GUARDED)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


class HeldAdapter:
    """The adapter, with signals held back around reset and release only.

    The mask is inherited by child processes, so it must not be held while a server starts.
    """

    def __init__(self, adapter):
        self._adapter = adapter
        self.owner_pid = adapter.owner_pid

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def reset(self, lease):
        with signals_held():
            self._adapter.reset(lease)

    def release(self, lease):
        with signals_held():
            self._adapter.release(lease)


def _interrupt(signum, frame):
    # SIGTERM's default action ends the process with no cleanup. Raising here runs the finally
    # blocks, which stop the servers and release the lease.
    raise KeyboardInterrupt(f"signal {signal.Signals(signum).name}")


def summary(entries: list[dict], code: int) -> str:
    lines = ["# Park check summary", ""]
    for e in entries:
        d = e["data"]
        if e["event"] in ("park", "restore"):
            lines.append(f"{e['event']:<8} {d.get('step')}")
        elif e["event"] == "measurement":
            lines.append(f"measured {d['name']} = {d['value']} {d['unit']}")
        elif e["event"] == "notice" and (d.get("blocked") or d.get("ok") is False):
            lines.append(f"STOP     {d.get('reason') or d.get('check')}")
    lines += ["", f"Exit code: {code}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.out_dir) if args.out_dir else make_out_dir(args.board, base="runs/park-check")
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out / "ledger.jsonl")
    old_term = signal.signal(signal.SIGTERM, _interrupt)
    try:
        ledger.append("run_start", None, options=vars(args), pid=os.getpid())
        code = _run(args, out, ledger)
        text = summary(ledger.read(), code)
        (out / "summary.md").write_text(text)
        print(text)
        return code
    finally:
        ledger.close()
        signal.signal(signal.SIGTERM, old_term)


def _run(args, out: Path, ledger) -> int:
    env_set = sorted(k for k in os.environ if k.startswith("GOZER_"))
    if env_set and not args.allow_gozer_env:
        # A GOZER_* variable points gozer at other state or replaces the reset command.
        ledger.append("notice", None, check="P.env", ok=False, gozer_env=env_set)
        return EXIT_PREFLIGHT
    adapter = HeldAdapter(GozerAdapter(gozer=args.gozer))
    try:
        chips = adapter.status()
    except AdapterError as exc:
        ledger.append("notice", None, check="P", ok=False, reason=str(exc))
        return EXIT_PREFLIGHT
    first = next((c for c in chips if c.bdf == args.board), None)
    if first is None:
        ledger.append("notice", None, check="P", ok=False, reason=f"{args.board} not in gozer status")
        return EXIT_PREFLIGHT
    busy = [c.bdf for c in chips if c.board == first.board and (c.state != "FREE" or c.who)]
    # A reset elsewhere makes every board look busy for about 42 s; two resets must not overlap.
    resetting = [c.bdf for c in chips if c.board != first.board and c.state == "BUSY-UNTRACKED"]
    if busy or resetting:
        ledger.append("notice", None, check="P", ok=False, busy=busy, other_board_busy=resetting)
        return EXIT_PREFLIGHT
    ledger.append("notice", None, check="P", ok=True, board=first.board)
    try:
        lease = adapter.acquire(CHIPS_PER_BOARD, WHO, REASON, exact=args.board)
    except AdapterError as exc:
        ledger.append("notice", None, check="A", ok=False, reason=str(exc))
        return EXIT_ACQUIRE
    ledger.append("decision", None, decision="lease taken", lease=lease.record())
    coder = ServerControl(ServerSpec("park-check/coder", "process", args.port, "fake",
                                     argv=fake_argv(args.port, args.answer)),
                          log_path=str(out / "coder.log"), ready_poll_s=args.ready_poll)
    standin = ServerStandIn(ServerControl(ServerSpec("park-check/standin", "process",
                                                     args.standin_port, "fake",
                                                     argv=fake_argv(args.standin_port, args.answer)),
                                          log_path=str(out / "standin.log"),
                                          ready_poll_s=args.ready_poll),
                            ready_budget_s=args.ready_budget)
    code = EXIT_FAIL
    try:
        coder.start(lease)
        coder.wait_ready(args.ready_budget)
        note = out / "handoff-note.json"
        note.write_text(json.dumps({"goal": "park check", "stage": "none", "evidence": [],
                                    "next_action": "restore the fake coder",
                                    "check_on_return": "the canary answer matches"}))
        h = Handoff(ledger=ledger, stage=None, adapter=adapter, server=coder, standin=standin,
                    lease=lease, canary_prompt=CANARY_PROMPT, note_path=note,
                    evidence_dir=out / "evidence", budgets=Budgets(cold_boot_s=args.ready_budget))
        h.park()
        ledger.append("decision", None, decision="no stage test in the park check; the board is "
                                                 "idle between the park reset and the restore reset")
        result = h.restore()
        code = EXIT_PASS if result is not None and result.match else EXIT_FAIL
    except (Blocked, AdapterError, ServerError) as exc:
        ledger.append("notice", None, check="ABORT", ok=False, reason=str(exc))
    finally:
        code = _cleanup(adapter, lease, coder, standin, ledger, code)
    return code


def _cleanup(adapter, lease, coder, standin, ledger, code: int) -> int:
    for name, stop in (("coder", coder.stop), ("standin", standin.stop)):
        try:
            stop()
        except Exception as exc:            # cleanup: record and continue to the release
            ledger.append("notice", None, check=f"cleanup.{name}", ok=False, reason=str(exc))
    try:
        adapter.release(lease)
        ledger.append("notice", None, check="cleanup.release", ok=True)
    except AdapterError as exc:
        ledger.append("notice", None, check="cleanup.release", ok=False, reason=str(exc),
                      recovery=f"once nothing holds the board, run: gozer release {lease.lease_id}; "
                               "never --force, never tt-smi -r by hand")
        return EXIT_FAIL
    return code


if __name__ == "__main__":
    sys.exit(main())
