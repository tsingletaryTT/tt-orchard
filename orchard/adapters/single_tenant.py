"""Lease adapter for a machine with no lease tool.

It assumes one tenant: this supervisor. It keeps nothing on disk. A lease is a record in memory of
which configured boards the supervisor is using. Leases cover whole boards (CHIPS_PER_BOARD chips
in the configured order), because UMD expands TT_VISIBLE_DEVICES to the whole board.

Where it refuses, and why:
- Building the adapter never refuses. After a supervisor crash the old coder can still hold a
  board, and the restarted supervisor must be able to build the adapter to recover.
- `acquire` refuses when a lease tool is present (`gozer` on PATH, or a gozer state directory).
  On such a machine other agents share the boards, and this adapter cannot see their leases.
  That refusal is permanent.
- `acquire` refuses a board while any process holds one of its device nodes.
- `reset` refuses permanently when no reset command was configured, because a reset on a machine
  whose tenants are unknown is a choice a person makes once.

Limits:
- The device scan reads /proc/<pid>/fd. An unprivileged user can read that only for its own
  processes, so a device held by another user's process (a root container) is invisible here, as
  it is to gozer (spec section 8, item 4).
- Release does not reset the chips, unlike gozer's release. The supervisor resets where it needs
  clean chips.
"""
from __future__ import annotations

import math
import os
import re
import shutil
from typing import Sequence

from orchard.adapters import ChipState, Lease, LeaseLost, Refused, ResetFailed
from orchard.commands import run_command
from orchard.defaults import CHIPS_PER_BOARD, RESET_TIMEOUT_S

DEV_LINK = re.compile(r"^/dev/tenstorrent/(\d+)$")
GOZER_STATE_DIRS = ("/tmp/tt-gozer",)        # gozer's default GOZER_ROOT


def device_holders(proc_root: str = "/proc") -> dict[int, list[int]]:
    """Device index to the pids that hold /dev/tenstorrent/<index> open."""
    out: dict[int, set[int]] = {}
    try:
        entries = os.listdir(proc_root)
    except FileNotFoundError:
        return {}
    for name in entries:
        if not name.isdigit():
            continue
        fd_dir = os.path.join(proc_root, name, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue                 # exited, or another user's process
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            m = DEV_LINK.match(target)
            if m:
                out.setdefault(int(m.group(1)), set()).add(int(name))
    return {k: sorted(v) for k, v in out.items()}


def _describe(held: dict[int, list[int]]) -> str:
    return ", ".join(f"pid {p} holds /dev/tenstorrent/{i}"
                     for i, pids in sorted(held.items()) for p in pids)


class SingleTenantAdapter:
    def __init__(self, chips: Sequence[tuple[str, int]], *, board: str = "local",
                 chips_per_board: int = CHIPS_PER_BOARD, proc_root: str = "/proc",
                 reset_argv: Sequence[str] | None = None, run=run_command,
                 reset_timeout: float = RESET_TIMEOUT_S, owner_pid: int | None = None,
                 which=shutil.which, gozer_state_dirs: Sequence[str] = GOZER_STATE_DIRS):
        chips = [(str(b), int(i)) for b, i in chips]
        if not chips or len(chips) % chips_per_board:
            raise ValueError(f"configure whole boards: a multiple of {chips_per_board} chips as "
                             f"(bdf, device index), got {len(chips)}")
        self.cpb = chips_per_board
        # Boards in configured order: local0, local1, ...
        self.boards = {f"{board}{k}": chips[k * chips_per_board:(k + 1) * chips_per_board]
                       for k in range(len(chips) // chips_per_board)}
        self.proc_root = proc_root
        self.reset_argv = list(reset_argv) if reset_argv else None
        self.run, self.reset_timeout = run, reset_timeout
        self.owner_pid = os.getpid() if owner_pid is None else owner_pid
        self.which, self.gozer_state_dirs = which, tuple(gozer_state_dirs)
        self.leases: dict[str, Lease] = {}
        self.in_flight: set[str] = set()
        self._n = 0

    def _no_lease_tool(self) -> None:
        found = self.which("gozer")
        if found:
            raise Refused(f"gozer is on PATH ({found}): this machine has a lease tool; use the "
                          "gozer adapter", permanent=True)
        for d in self.gozer_state_dirs:
            if os.path.isdir(d):
                raise Refused(f"a gozer state directory exists ({d}): this machine has a lease "
                              "tool; use the gozer adapter", permanent=True)

    def acquire(self, chips: int, who: str, reason: str, *, queue: bool = False,
                exact: str | None = None) -> Lease:
        self._no_lease_tool()
        taken = {u for lease in self.leases.values() for u in lease.units}
        free = [name for name in self.boards if name not in taken]
        held = device_holders(self.proc_root)
        if exact is not None:
            start = next((k for k, name in enumerate(free)
                          if exact in (b for b, _ in self.boards[name])), None)
            if start is None:
                raise Refused(f"{exact} is not on a free configured board")
            free = free[start:]
        else:
            # Without --exact, skip boards that some process holds open.
            free = [name for name in free if not any(i in held for _, i in self.boards[name])]
        need = math.ceil(chips / self.cpb)
        if need > len(free):
            raise Refused(f"asked for {chips} chips ({need} boards), {len(free)} free"
                          + (f"; {_describe(held)}" if held else ""))
        pick = free[:need]
        pairs = [pair for name in pick for pair in self.boards[name]]
        busy = {i: held[i] for _, i in pairs if i in held}
        if busy:
            raise Refused(_describe(busy))
        self._n += 1
        bdfs = tuple(b for b, _ in pairs)
        lease = Lease(lease_id=f"local-{self._n}", chips=bdfs,
                      dev_indices=tuple(i for _, i in pairs),
                      env={"TT_VISIBLE_DEVICES": ",".join(bdfs)}, units=tuple(pick))
        self.leases[lease.lease_id] = lease
        return lease

    def claim(self, ticket: str, chips: int, who: str, reason: str) -> Lease:
        raise Refused("the single-tenant adapter has no queue", permanent=True)

    def cancel(self, ticket: str) -> None:
        return None

    def release(self, lease: Lease) -> None:
        self.leases.pop(lease.lease_id, None)

    def reset(self, lease: Lease) -> None:
        if lease.lease_id not in self.leases:
            raise LeaseLost(f"no lease {lease.lease_id}")
        if lease.lease_id in self.in_flight:
            raise Refused(f"a reset of lease {lease.lease_id} is still running")
        if self.reset_argv is None:
            raise Refused("the single-tenant adapter has no reset command configured; pass "
                          "reset_argv when creating it", permanent=True)
        held = device_holders(self.proc_root)
        busy = {i: held[i] for i in lease.dev_indices if i in held}
        if busy:
            raise Refused("device still open: " + _describe(busy))
        res = self.run([*self.reset_argv, ",".join(lease.chips)], self.reset_timeout,
                       kill_on_timeout=False)
        if res.timed_out:
            self.in_flight.add(lease.lease_id)
            raise ResetFailed(f"{' '.join(res.argv)} still running after {self.reset_timeout} s; "
                              "left running", left_running=True)
        if res.returncode != 0:
            raise ResetFailed(f"{' '.join(res.argv)} exited {res.returncode}: "
                              f"{(res.stderr or res.stdout).strip()[:500]}")

    def status(self) -> list[ChipState]:
        held = device_holders(self.proc_root)
        leased = {u for lease in self.leases.values() for u in lease.units}
        out = []
        for name, pairs in self.boards.items():
            mine = name in leased
            for b, i in pairs:
                if i in held:
                    state = "HELD" if mine else "BUSY-UNTRACKED"
                else:
                    state = "CLAIMED" if mine else "FREE"
                out.append(ChipState(bdf=b, state=state, who="orchard" if mine else None,
                                     board=name, dev_index=i,
                                     lease_pid=self.owner_pid if mine else None,
                                     pids_holding=tuple(held.get(i, ()))))
        return out
