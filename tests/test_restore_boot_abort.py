# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 589 (review of PR 580): a coder that fails to boot during a restore gets the same ending as one that fails at the first start.

`Handoff.restore()` (and `recover_and_restore()`, which ends in it) waited for the coder and turned a stalled boot, an exited container or a
passed deadline into a bare block: the end of the container log and the exit code were dropped, the container was left running, and
unattended cleanup stopped it with the ordinary 120 s stop. The supervisor now hands the exception to the same `_abort_boot` it uses at
the first start (evidence, a 900 s stop, a reset through the lease, the device state). A Handoff built without that hook behaves as before."""
import pytest

from fakes import World, make_handoff
from orchard.handoff import Blocked
from orchard.server import BootStalled, CoderExited, NotReady


class Aborted(Exception):
    pass


def restore_with(tmp_path, exc, hook):
    world = World()
    h, world, ledger = make_handoff(tmp_path, world)
    h.park()
    h.boot_abort = hook
    original = h.server.wait_ready

    def failing(budget_s):
        raise exc
    h.server.wait_ready = failing
    return h


@pytest.mark.parametrize("exc", [BootStalled(900, 610, "loading layer 20"), CoderExited(50, "exited", 1, "RuntimeError"),
                                 NotReady(2700, "", "still loading")])
def test_a_boot_failure_in_a_restore_goes_to_the_boot_abort_with_the_lease_and_the_exception(tmp_path, exc):
    seen = []

    def hook(lease, e):
        seen.append((lease.lease_id, e))
        raise Blocked("the abort ran")
    h = restore_with(tmp_path, exc, hook)
    with pytest.raises(Blocked, match="the abort ran"):
        h.restore()
    assert seen == [(h.lease.lease_id, exc)] and seen[0][1].log_tail == exc.log_tail


def test_without_the_hook_a_stalled_boot_still_blocks_as_before(tmp_path):
    h = restore_with(tmp_path, BootStalled(900, 610, "x"), None)
    with pytest.raises(Blocked, match="exited before it was ready|did not grow"):
        h.restore()


def test_without_the_hook_a_missed_deadline_keeps_the_old_text(tmp_path):
    h = restore_with(tmp_path, NotReady(2700), None)
    with pytest.raises(Blocked, match="cold-boot budget"):
        h.restore()


def test_the_supervisor_gives_its_boot_abort_to_every_handoff_it_builds(tmp_path):
    """The wiring at the layer that decides: a Handoff built by the supervisor carries `_abort_boot`."""
    from orchard.supervisor import Supervisor
    sup = Supervisor.__new__(Supervisor)
    h0, world, ledger = make_handoff(tmp_path)
    sup.ledger, sup.adapter, sup.coder, sup.standin = ledger, h0.adapter, h0.server, h0.standin
    sup.run_dir, sup.budgets, sup.clock, sup.sleep = tmp_path, h0.budgets, h0.clock, h0.sleep
    built = sup._handoff(2, h0.lease)
    assert built.boot_abort == sup._abort_boot
