# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Lease adapters: how the supervisor takes, resets and gives back Tenstorrent boards.

This package owns the interface between the supervisor and whatever controls access to the chips
on a machine (spec section 4). `gozer.py` drives tt-gozer. `single_tenant.py` is for a machine
with no lease tool. Supervisor code depends only on the types in this file, so either adapter can
stand behind it.

A board is named by its serial, which is gozer's unit key at board grain. A chip is named by its
PCI address (BDF). Device indices appear only where a device node is meant (/dev/tenstorrent/N).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# The chip states `gozer status` reports (tt-gozer gozer/gatekeeper.py, reconcile).
STATES = frozenset({"HELD", "HELD-FOREIGN", "CLAIMED", "STALE", "BUSY-UNTRACKED", "FREE"})


@dataclass(frozen=True)
class Lease:
    lease_id: str
    chips: tuple[str, ...]
    dev_indices: tuple[int, ...]
    # A dict cannot be hashed. hash=False keeps the frozen dataclass hashable on its other fields.
    env: dict[str, str] = field(hash=False)
    units: tuple[str, ...]

    def record(self) -> dict:
        return {"lease_id": self.lease_id, "chips": list(self.chips),
                "dev_indices": list(self.dev_indices), "env": dict(self.env),
                "units": list(self.units)}

    @classmethod
    def from_record(cls, rec: dict) -> "Lease":
        return cls(lease_id=str(rec["lease_id"]), chips=tuple(rec["chips"]),
                   dev_indices=tuple(int(i) for i in rec["dev_indices"]),
                   env={str(k): str(v) for k, v in rec["env"].items()}, units=tuple(rec["units"]))


@dataclass(frozen=True)
class ChipState:
    bdf: str
    state: str
    who: str | None
    board: str = ""
    dev_index: int | None = None
    lease_pid: int | None = None          # the pid gozer judges the lease by
    pids_holding: tuple[int, ...] = ()    # processes with the device open, as gozer sees them

    def __post_init__(self):
        if self.state not in STATES:
            # A state this code does not know means the lease tool changed. Fail closed and let a
            # person read what it now reports.
            raise ValueError(f"unknown chip state {self.state!r}")


class AdapterError(Exception):
    """The lease tool did something the adapter cannot interpret."""


class Queued(AdapterError):
    """No chips now. The request waits in the queue under `ticket`."""

    def __init__(self, ticket: str, position: int | None = None):
        super().__init__(f"queued with ticket {ticket} (position {position})")
        self.ticket, self.position = ticket, position


class Refused(AdapterError):
    """The lease tool refused (gozer exit 12 for acquire, 15 for reset or release).

    `permanent` means no wait can change the answer (for example, an adapter with no reset
    command), so the caller blocks at once. gozer's exit 15 is never marked permanent: one of its
    causes is a device that is still open.
    """

    def __init__(self, reason: str, permanent: bool = False):
        super().__init__(reason)
        self.reason, self.permanent = reason, permanent


class ResetFailed(AdapterError):
    """The reset ran and failed, or is still running after its timeout (left_running)."""

    def __init__(self, detail: str, left_running: bool = False):
        super().__init__(detail)
        self.detail, self.left_running = detail, left_running


class LeaseLost(AdapterError):
    """The lease no longer exists, or another lease now holds its chips. Start no server."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class TicketGone(AdapterError):
    """The queue ticket no longer exists. gozer expires tickets after one hour."""

    def __init__(self, ticket: str):
        super().__init__(f"queue ticket {ticket} no longer exists")
        self.ticket = ticket


class LeaseAdapter(Protocol):
    owner_pid: int

    def acquire(self, chips: int, who: str, reason: str, *, queue: bool = False,
                exact: str | None = None) -> Lease: ...

    def claim(self, ticket: str, chips: int, who: str, reason: str) -> Lease: ...

    def cancel(self, ticket: str) -> None: ...

    def release(self, lease: Lease) -> None: ...

    def reset(self, lease: Lease) -> None: ...

    def status(self) -> list[ChipState]: ...


def boards_of(chips: list[ChipState]) -> dict[str, list[ChipState]]:
    """Group chip states by board serial, in the order the lease tool listed them."""
    out: dict[str, list[ChipState]] = {}
    for c in chips:
        out.setdefault(c.board, []).append(c)
    return out
