"""Test doubles shared by the plan 3 tests. Nothing here touches hardware or the network."""
from __future__ import annotations

import json

from orchard.adapters import Lease
from orchard.commands import CommandResult

OWNER = 4242

# Shape copied from a real grant: runs/h5-20261002T201323Z/lease.json (owner pid changed).
GRANT = {"chips": ["0000:01:00.0", "0000:02:00.0"], "dev_indices": [0, 1],
         "env": {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}, "expanded": True,
         "granted": True, "lease_id": "fb9995", "neighbours": {}, "owner_pid": OWNER,
         "requested": 1, "units": ["0000046131924062"]}

LEASE = Lease("fb9995", ("0000:01:00.0", "0000:02:00.0"), (0, 1),
              {"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"}, ("0000046131924062",))


class FakeRun:
    """Stands in for orchard.commands.run_command.

    script maps a key to a list of answers, used in order; the last answer repeats. The key is
    argv[1] (the gozer subcommand) unless `key` is a function of argv. An answer is
    (returncode, stdout, stderr). stdout may be a dict, sent as JSON. returncode may be
    "timeout". A call with no scripted answer fails the test.
    """

    def __init__(self, script: dict | None = None, key=None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.key = key or (lambda argv: argv[1])
        self.calls: list[dict] = []

    def __call__(self, argv, timeout, *, env=None, kill_on_timeout=True):
        argv = [str(a) for a in argv]
        self.calls.append({"argv": argv, "timeout": timeout, "env": env,
                           "kill_on_timeout": kill_on_timeout})
        answers = self.script.get(self.key(argv))
        if not answers:
            raise AssertionError(f"unexpected call: {argv}")
        rc, out, err = answers.pop(0) if len(answers) > 1 else answers[0]
        if rc == "timeout":
            return CommandResult(tuple(argv), None, timed_out=True, left_running=not kill_on_timeout)
        return CommandResult(tuple(argv), rc, out if isinstance(out, str) else json.dumps(out), err)

    def argvs(self) -> list[list[str]]:
        return [c["argv"] for c in self.calls]


class FakeProc:
    """A Popen stand-in. Set .returncode to make it 'exit'."""

    def __init__(self, pid=4321):
        self.pid, self.returncode, self.polls = pid, None, 0

    def poll(self):
        self.polls += 1
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class FakeClock:
    """time.monotonic and time.sleep for tests: sleep advances the clock at once."""

    def __init__(self, now=0.0):
        self.now, self.sleeps = now, []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


# ---- the machine, for the handoff tests (Tasks 9 to 11) ------------------------------------------

from orchard.adapters import ChipState, LeaseLost, Queued, Refused, ResetFailed  # noqa: E402
from orchard.ledger import Ledger  # noqa: E402
from orchard.server import NotReady, ServerStarting, StopCheck  # noqa: E402

BOARD0 = ("0000:01:00.0", "0000:02:00.0")
BOARD1 = ("0000:03:00.0", "0000:04:00.0")
WHO = "orchard:test"
NOTE = {"goal": "bring up model X", "stage": 2, "evidence": ["evidence/pcc.json"],
        "next_action": "run the decoder test", "check_on_return": "PCC above 0.99"}
CANARY = "What is 2 + 2? Answer with one number."


class Crash(BaseException):
    """The supervisor process dies here. BaseException, so no `except Exception` catches it."""


class World:
    """The machine as the fake adapter, server and stand-in see it.

    One shared object, so a test can kill the supervisor and look at what is left. The coder is a
    tt-model container on board 0: while it runs, gozer shows its chips HELD-FOREIGN.
    """

    def __init__(self, owner_pid=100):
        self.owner_pid = owner_pid          # the live supervisor; any other owner pid is dead
        self.leases = {}                    # lease_id -> (Lease, owner pid)
        self.coder_running = False
        self.coder_starts = 0
        self.coder_answer = "4"
        self.coder_answer_after = None      # if set, what the coder says after a restart
        self.never_ready = False
        self.standin_running = False
        self.standin_answer = "4"
        self.standin_fails = False
        self.standin_survives_stop = False  # the stand-in ignores its stop
        self.standin_wait_raises = None     # raised by the stand-in's readiness wait, after its spawn
        self.stop_mesh_reset = False        # tt-model stop says it reset the mesh itself
        self.start_hangs_but_comes_up = False  # tt-model serve times out while the coder boots
        self.worker_left = False            # a vLLM worker that outlives the server
        self.leave_worker_on_stop = False
        self.crash_in_stop = False          # die inside tt-model stop, after the coder stopped
        self.refuse_resets = 0              # the next n resets are refused (exit 15)
        self.lose_lease_on_reset = False    # the next reset finds the lease taken (exit 18)
        self.reset_fails = False            # tt-smi ran and failed (exit 17)
        self.reset_unavailable = False      # the adapter has no reset command (permanent refusal)
        self.reset_hangs = False
        self.foreign_polls = 0              # the next n status reads show our chips HELD-FOREIGN
        self.queue_script = []              # answers for acquire(queue=True)/claim before a grant
        self.resets = []
        self.events = []
        self.n = 0


class FakeAdapter:
    def __init__(self, world, owner_pid=None):
        self.world = world
        self.owner_pid = owner_pid or world.owner_pid
        self.calls = []

    def _holding(self):
        return self.world.coder_running or self.world.worker_left

    def _grant(self):
        w = self.world
        # gozer reaps a lease whose owner is dead once no device is open.
        for lid, (_, owner) in list(w.leases.items()):
            if owner != w.owner_pid and not self._holding():
                del w.leases[lid]
        busy = {b for lease, _ in w.leases.values() for b in lease.chips}
        if busy & set(BOARD0):
            return None
        w.n += 1
        lease = Lease(f"L{w.n}", BOARD0, (0, 1), {"TT_VISIBLE_DEVICES": ",".join(BOARD0)}, ("B0",))
        w.leases[lease.lease_id] = (lease, self.owner_pid)
        return lease

    def _scripted(self):
        if self.world.queue_script:
            ans = self.world.queue_script.pop(0)
            if ans is not None:
                raise ans

    def acquire(self, chips, who, reason, *, queue=False, exact=None):
        self.calls.append(("acquire", chips, queue))
        self._scripted()
        lease = self._grant()
        if lease is None:
            raise Queued("t-busy") if queue else Refused("board 0 is leased")
        return lease

    def claim(self, ticket, chips, who, reason):
        self.calls.append(("claim", ticket))
        self._scripted()
        lease = self._grant()
        if lease is None:
            raise Queued(ticket)
        return lease

    def cancel(self, ticket):
        self.calls.append(("cancel", ticket))

    def release(self, lease):
        self.calls.append(("release", lease.lease_id))
        self.world.leases.pop(lease.lease_id, None)

    def reset(self, lease):
        self.calls.append(("reset", lease.lease_id))
        w = self.world
        entry = w.leases.get(lease.lease_id)
        if entry is None or entry[1] != self.owner_pid:
            raise LeaseLost(f"no lease {lease.lease_id} for pid {self.owner_pid}")
        if w.lose_lease_on_reset:
            w.lose_lease_on_reset = False
            raise LeaseLost(f"the reset ran, and afterwards lease {lease.lease_id} was taken")
        if w.reset_unavailable:
            raise Refused("no reset command configured", permanent=True)
        if w.reset_hangs:
            raise ResetFailed("gozer reset still running; left running", left_running=True)
        if w.reset_fails:
            raise ResetFailed("tt-smi -r failed")
        if w.refuse_resets > 0:
            w.refuse_resets -= 1
            raise Refused("device still open (HELD-FOREIGN)")
        if self._holding():
            raise Refused("device still open")
        w.resets.append(lease.lease_id)

    def status(self):
        w = self.world
        foreign = w.foreign_polls > 0
        if foreign:
            w.foreign_polls -= 1
        owner_of = {b: (lease, o) for lease, o in w.leases.values() for b in lease.chips}
        out = []
        for i, b in enumerate(BOARD0 + BOARD1):
            serial = "B0" if b in BOARD0 else "B1"
            holding = b in BOARD0 and self._holding()
            lo = owner_of.get(b)
            if lo is None:
                out.append(ChipState(b, "BUSY-UNTRACKED" if holding else "FREE", None, board=serial,
                                     dev_index=i, pids_holding=(777,) if holding else ()))
                continue
            _, owner = lo
            if holding or foreign:
                state = "HELD-FOREIGN"     # a container is never the supervisor's descendant
            else:
                state = "CLAIMED" if owner == w.owner_pid else "STALE"
            out.append(ChipState(b, state, WHO, board=serial, dev_index=i, lease_pid=owner,
                                 pids_holding=(777,) if holding else ()))
        return out


class FakeServer:
    def __init__(self, world):
        self.world, self.adopted = world, None

    def start(self, lease):
        self.world.coder_starts += 1
        self.world.events.append("coder_start")
        self.world.coder_running = True
        if self.world.start_hangs_but_comes_up:
            raise ServerStarting("tt-model serve still running; the server is coming up")

    def stop(self):
        w = self.world
        w.events.append("coder_stop")
        w.coder_running = False
        if w.leave_worker_on_stop:
            w.worker_left = True
        if w.crash_in_stop:
            w.crash_in_stop = False
            raise Crash("killed inside tt-model stop")
        return {"how": "tt-model stop", "mesh_reset": w.stop_mesh_reset}

    def confirm_stopped(self):
        # docker sees only the container; a leftover worker is invisible to it.
        ok = not self.world.coder_running
        return StopCheck(ok, {"docker_ps": ok}, {})

    def wait_ready(self, budget_s):
        if self.world.never_ready:
            raise NotReady(budget_s)
        return 20.3

    def ask(self, prompt):
        w = self.world
        if not w.coder_running:
            raise ConnectionRefusedError("coder is down")
        if w.coder_starts > 1 and w.coder_answer_after is not None:
            return w.coder_answer_after
        return w.coder_answer

    def record(self):
        return {"kind": "container", "port": 20000}

    def adopt(self, record):
        self.adopted = record


class FakeStandIn:
    def __init__(self, world):
        self.world = world

    def spawn(self):
        self.world.events.append("standin_start")
        self.world.standin_running = True

    def wait_ready(self):
        self.world.events.append("standin_wait_ready")
        if self.world.standin_wait_raises is not None:
            raise self.world.standin_wait_raises

    def ask(self, prompt):
        self.world.events.append("standin_ask")
        if self.world.standin_fails or not self.world.standin_running:
            raise ConnectionRefusedError("stand-in is down")
        return self.world.standin_answer

    def stop(self):
        self.world.events.append("standin_stop")
        if not self.world.standin_survives_stop:
            self.world.standin_running = False

    def confirm_stopped(self):
        ok = not self.world.standin_running
        return StopCheck(ok, {"process": ok}, {})

    def record(self):
        return {"kind": "process", "port": 8001, "pid": 5150, "pgid": 5150}

    def adopt(self, record):
        self.adopted = record


def make_handoff(tmp_path, world=None, *, ledger_cls=Ledger, ledger_kw=None, budgets=None):
    """A coder serving on board 0 under lease L1, and a Handoff ready to park it."""
    from orchard.handoff import Budgets, Handoff
    world = world or World()
    adapter = FakeAdapter(world)
    server = FakeServer(world)
    lease = adapter.acquire(2, WHO, "coder")
    server.start(lease)
    note = tmp_path / "note.json"
    note.write_text(json.dumps(NOTE))
    ledger = ledger_cls(tmp_path / "ledger.jsonl", **(ledger_kw or {}))
    clock = FakeClock()
    h = Handoff(ledger=ledger, stage=2, adapter=adapter, server=server, standin=FakeStandIn(world),
                lease=lease, canary_prompt=CANARY, note_path=note,
                evidence_dir=tmp_path / "evidence", budgets=budgets or Budgets(), clock=clock,
                sleep=clock.sleep)
    return h, world, ledger


def steps(ledger):
    return [(e["event"], e["data"]["step"]) for e in ledger.read() if e["event"] in ("park", "restore")]
