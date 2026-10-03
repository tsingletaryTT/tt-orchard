"""Fakes for the supervisor tests: a two-board machine, its coder, and a scripted bring-up.

`Machine` is the state a crash leaves behind: leases (with the pid that owns each), and whether the
coder container runs and on which chips. `MachineAdapter` behaves like gozer through the adapter
interface: a lease whose owner pid is dead is reaped at the next acquire once no device is open,
reset and release refuse while a device is open, and a chip of a running container shows
HELD-FOREIGN. `MachineCoder` behaves like a container coder. `bringup` is a model script that walks
stages 0 to 8 by writing the files each gate reads; it answers from the request alone, as a greedy
server would, so a restarted supervisor gets the same answers.
"""
from __future__ import annotations

import json
import re
import socket
import time

from fake_model import call, final, turn
from fakes import Crash, FakeClock
from orchard.adapters import ChipState, Lease, LeaseLost, Queued, Refused
from orchard.defaults import FIRST_BOOT_PROMPT
from orchard.ledger import Ledger
from orchard.server import NotReady, StopCheck

BOARDS = {"B0": ("0000:01:00.0", "0000:02:00.0"), "B1": ("0000:03:00.0", "0000:04:00.0")}
DEV = {c: i for i, c in enumerate(BOARDS["B0"] + BOARDS["B1"])}
CANARY_ANSWER = "42"


class CrashingLedger(Ledger):
    """A ledger whose process dies right after an append is on disk: the k-th append, or the first
    entry for which `crash_if(entry)` is true."""

    def __init__(self, path, crash_after=0, crash_if=None):
        super().__init__(path)
        self.crash_after, self.crash_if, self.appended = crash_after, crash_if, 0

    def append(self, *args, **kwargs):
        entry = super().append(*args, **kwargs)
        self.appended += 1
        if self.appended == self.crash_after or (self.crash_if and self.crash_if(entry)):
            raise Crash(f"killed after ledger event {entry['seq']}: {entry['event']}")
        return entry


class Machine:
    def __init__(self, owner_pid=100):
        self.owner_pid = owner_pid           # the live supervisor; every other owner pid is dead
        self.leases: dict[str, tuple[Lease, int]] = {}
        self.coder_running = False
        self.coder_chips: tuple[str, ...] = ()
        self.coder_starts = 0
        self.coder_answer = CANARY_ANSWER
        self.sanity_answer = "42"           # what the coder says to the first-boot arithmetic question
        self.asked: list[str] = []          # every prompt put to the coder, in order
        self.never_ready = False
        self.resets: list[tuple[str, str]] = []
        self.n = 0

    def held(self) -> set[str]:
        return set(self.coder_chips) if self.coder_running else set()


class MachineAdapter:
    def __init__(self, machine: Machine, owner_pid=None):
        self.m = machine
        self.owner_pid = owner_pid or machine.owner_pid
        self.calls: list[tuple] = []

    def _reap(self):
        for lid, (lease, owner) in list(self.m.leases.items()):
            if owner != self.m.owner_pid and not set(lease.chips) & self.m.held():
                del self.m.leases[lid]

    def acquire(self, chips, who, reason, *, queue=False, exact=None):
        self.calls.append(("acquire", chips, exact))
        self._reap()
        busy = {c for lease, _ in self.m.leases.values() for c in lease.chips} | self.m.held()
        free = [b for b, cs in BOARDS.items() if not set(cs) & busy and (exact is None or exact in cs)]
        need = max(1, chips // 2)
        if len(free) < need:
            if queue:
                raise Queued("t1")
            raise Refused("no free board")
        take = free[:need]
        got = tuple(c for b in take for c in BOARDS[b])
        self.m.n += 1
        lease = Lease(f"L{self.m.n}", got, tuple(DEV[c] for c in got),
                      {"TT_VISIBLE_DEVICES": ",".join(got)}, tuple(take))
        self.m.leases[lease.lease_id] = (lease, self.owner_pid)
        return lease

    def claim(self, ticket, chips, who, reason):
        return self.acquire(chips, who, reason, queue=True)

    def cancel(self, ticket):
        self.calls.append(("cancel", ticket))

    def release(self, lease):
        self.calls.append(("release", lease.lease_id))
        if lease.lease_id not in self.m.leases:
            raise LeaseLost(f"no lease {lease.lease_id}")
        if set(lease.chips) & self.m.held():
            raise Refused("device still open")
        del self.m.leases[lease.lease_id]
        self.m.resets.append(("release", lease.lease_id))

    def reset(self, lease):
        self.calls.append(("reset", lease.lease_id))
        entry = self.m.leases.get(lease.lease_id)
        if entry is None or entry[1] != self.owner_pid:
            raise LeaseLost(f"no lease {lease.lease_id} for pid {self.owner_pid}")
        if set(lease.chips) & self.m.held():
            raise Refused("device still open")
        self.m.resets.append(("reset", lease.lease_id))

    def status(self):
        owner_of = {c: o for lease, o in self.m.leases.values() for c in lease.chips}
        held, out = self.m.held(), []
        for board, chips in BOARDS.items():
            for c in chips:
                if c not in owner_of:
                    out.append(ChipState(c, "BUSY-UNTRACKED" if c in held else "FREE", None,
                                         board=board, dev_index=DEV[c]))
                    continue
                owner = owner_of[c]
                state = ("HELD-FOREIGN" if c in held else
                         "CLAIMED" if owner == self.m.owner_pid else "STALE")
                out.append(ChipState(c, state, "orchard", board=board, dev_index=DEV[c],
                                     lease_pid=owner))
        return out


class MachineCoder:
    """A container coder: it outlives the supervisor, and docker is the stop check."""

    def __init__(self, machine: Machine):
        self.m = machine

    def start(self, lease):
        self.m.coder_running, self.m.coder_chips = True, tuple(lease.chips)
        self.m.coder_starts += 1

    def stop(self):
        self.m.coder_running = False
        return {"how": "tt-model stop", "mesh_reset": False}

    def confirm_stopped(self):
        ok = not self.m.coder_running
        return StopCheck(ok, {"docker_ps": ok}, {})

    def wait_ready(self, budget_s):
        if self.m.never_ready:
            raise NotReady(budget_s)
        return 20.0

    def ask(self, prompt):
        if not self.m.coder_running:
            raise ConnectionRefusedError("the coder is down")
        self.m.asked.append(prompt)
        if prompt == FIRST_BOOT_PROMPT:
            return self.m.sanity_answer
        return self.m.coder_answer

    def record(self):
        return {"kind": "container", "target": "fake/qwen-coder", "port": 8000}

    def adopt(self, record):
        return None


# ---- the scripted bring-up ------------------------------------------------------------------------

def ev(n, *names):
    return [f"stages/{n}/evidence/{x}" for x in names or ("notes.txt",)]


DELTA = {"model": "Altworld/Hemmingway-1", "nearest_model": "Qwen/Qwen3.8-27B", "path": "weights-only",
         "differences": [{"area": a, "finding": "checked", "evidence": ev(0)}
                         for a in ("config", "tensors", "tensor_names", "tokenizer", "files",
                                   "generation_config")],
         "hazards": [{"area": a, "finding": "noted"} for a in ("tensor_cache", "drafter", "disk")]}


def note(n):
    return {"goal": "bring up Hemmingway-1", "stage": n, "evidence": ev(n),
            "next_action": "read the test output", "check_on_return": "the test exit code"}


def hw(n):
    return {"evidence/notes.txt": f"stage {n} plan",
            "hw_test.json": {"command": f"printenv > stages/{n}/evidence/devices.txt",
                             "deadline_s": 60},
            "handoff.json": note(n)}


def done(n):
    return ev(n, "hw-test-output.txt", "devices.txt")


FILES = {
    (0, "run"): {"evidence/notes.txt": "configs, tensors and tokenizers compared", "delta.json": DELTA},
    (1, "run"): {"evidence/notes.txt": "greedy decode matches",
                 "reference.json": {"verdict": "pass", "checks": [
                     {"name": n, "pass": True, "evidence": ev(1)}
                     for n in ("loads", "tokenizer round trip", "decodes forward", "matches the card")]}},
    (2, "prepare"): hw(2),
    (2, "finish"): {"result.json": {"pcc": 0.998, "argmax_match": True, "evidence": done(2)}},
    (3, "prepare"): hw(3),
    (3, "finish"): {"result.json": {"parity": True, "top1": 0.97, "evidence": done(3)}},
    (4, "prepare"): hw(4),
    (4, "finish"): {"result.json": {"configs": [{"chips": 2, "pass": True, "evidence": done(4)},
                                                {"chips": 1, "pass": True, "evidence": done(4)}]}},
    (5, "prepare"): hw(5),
    (5, "finish"): {"result.json": {"checks": {k: {"pass": True, "evidence": done(5)}
                                               for k in ("boots", "passkey", "canary")}}},
    (6, "prepare"): hw(6),
    (6, "finish"): {"result.json": {"numbers": [
        {"name": "decode", "value": 80.0, "unit": "tok/s/user", "label": "measured", "evidence": done(6)},
        {"name": "ttft", "value": None, "unit": "ms", "label": "TODO"}],
        "qualitative": {"evidence": done(6)}}},
    (8, "run"): {"bundle/RESULTS.md": "Stages 0 to 6 passed. Stage 7 was skipped.",
                 "bundle/RISKS.md": "ttft is TODO.",
                 "bundle/PUBLISH_COMMANDS.txt": "tt-model push example/hemmingway-1-p300\n"},
}


def where(request) -> tuple[int, str]:
    system = request["messages"][0]["content"]
    n = int(re.search(r"agent for stage (\d+) ", system).group(1))
    return n, re.search(r"^Phase: (\w+)$", system, re.M).group(1)


def bringup(request, overrides=None):
    """The scripted model. A request without tools is a canary, answered with CANARY_ANSWER."""
    if "tools" not in request:
        return final(CANARY_ANSWER)
    key = where(request)
    files = (overrides or {}).get(key, FILES[key])
    turns = [call("write_file", path=p, content=c if isinstance(c, str) else json.dumps(c))
             for p, c in files.items()] + [final(f"stage {key[0]} {key[1]} done")]
    return turns[min(turn(request), len(turns) - 1)]


def closed_port() -> int:
    """A local port with nothing listening on it."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def write_tiers(path, chips_endpoint: str, cpu_endpoint: str):
    """The operator's tier layout: large and small are the same model on 4 and 2 chips, and only
    the large one is serving (the small endpoint has nothing listening). The CPU tier is separate."""
    text = "\n".join([
        '[tiers.large]', 'role = "plan"', f'endpoint = "{chips_endpoint}"', 'model = "qwen-27b"',
        'placement = "chips"',
        '[tiers.small]', 'role = "run"', f'endpoint = "http://127.0.0.1:{closed_port()}/v1"',
        'model = "qwen-27b"', 'placement = "chips"',
        '[tiers.cpu]', 'role = "stand-in"', f'endpoint = "{cpu_endpoint}"', 'model = "cpu-model"',
        'placement = "cpu"',
        '[stages.0]', 'run = "large"', '[stages.1]', 'run = "small"',
        '[stages.2]', 'run = "small"', 'diagnose = "large"', '[stages.3]', 'run = "small"',
        'diagnose = "large"', '[stages.4]', 'run = "small"', 'plan = "large"', 'diagnose = "large"',
        '[stages.5]', 'run = "small"', '[stages.6]', 'run = "small"', '[stages.7]', 'run = "none"',
        '[stages.8]', 'run = "small"', '[escalation]', 'default = "large"', ""])
    path.write_text(text)
    return path


EXISTING = ("model-bringup", "functional-decoder", "full-model", "mesh-shrink", "multichip",
            "tt-device-usage", "vllm-integration", "qualitative-check", "benchmark-model")


def stub_skills(root):
    """Stand-ins for the installed tt-model-bringup skills, in the plugin's folder layout."""
    for name in EXISTING:
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "SKILL.md").write_text(f"# {name}\nStub for tests.\n")
    return root


def argv(tmp_path, tiers, chips_endpoint: str, chips=4):
    port = chips_endpoint.rsplit(":", 1)[1].split("/")[0]
    return ["run", "--model", "Altworld/Hemmingway-1", "--run-dir", str(tmp_path / "run"),
            "--tiers", str(tiers), "--coder-target", "fake/qwen-coder", "--coder-port", port,
            "--coder-chips", str(chips), "--input", "base=/models/base",
            "--skills-dir", str(stub_skills(tmp_path / "plugin-skills"))]


def plenty(path):
    from collections import namedtuple
    return namedtuple("U", "total used free")(0, 0, 10 ** 15)


class StuckClock(FakeClock):
    """A FakeClock that fails the test once the run has waited more than a simulated day. A run
    that waits for an operator nobody plays, or for a lease that never comes, fails here instead
    of hanging."""

    def sleep(self, s):
        super().sleep(s)
        if sum(self.sleeps) > 86400:
            raise AssertionError("the run waited more than a simulated day; it is stuck")


def clock():
    return StuckClock(time.time())
