# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The wide phase (stage 2's single test on more boards than its stage's) runs again after the fabric handshake timeout (issue 487).

Evidence (issue 487, firmware 19.17.1.0): a 4-chip coder mesh failed with `Fabric Router Sync: Timeout after 10000 ms on Device 2` after
a coder had run on one board and the boards were reset at different times, and passed after both boards were reset again. Stage 4's
configurations retried after such a failure; the wide phase did not. These tests drive the real `_run_wide_test` and
`_wide_hardware_phase` with a fake test command and fake leases, and check the wiring: the resets, their order and the ledger entries."""
import json
from types import SimpleNamespace

import pytest

from orchard import supervisor
from orchard.adapters import ChipState
from orchard.supervisor import Supervisor

FABRIC = "Fabric Router Sync: Timeout after 10000 ms on Device 2\n"


def lease(lease_id, bdfs, devs):
    return SimpleNamespace(lease_id=lease_id, chips=list(bdfs), dev_indices=list(devs), units=(lease_id,))


# Board A holds the lowest PCI address and device indices 0 and 1; board B holds 2 and 3.
A = lease("A", ["0000:01:00.0", "0000:02:00.0"], [0, 1])
B = lease("B", ["0000:03:00.0", "0000:04:00.0"], [2, 3])


class Rig:
    def __init__(self, tmp_path, outputs):
        """`outputs`: one (returncode, timed_out, text) per try; the last is repeated."""
        self.tmp, self.outputs, self.tries = tmp_path, list(outputs), 0
        self.events, self.resets, self.timeline = [], [], []
        s = object.__new__(Supervisor)
        s.run_dir = tmp_path
        s.ledger = SimpleNamespace(append=self._append)
        s.test_adapter = SimpleNamespace(reset=self._reset)
        s._run_test = self._run_test
        self.s = s
        self.spec = SimpleNamespace(number=2, boards=1, skill="x")
        self.stage_dir = tmp_path / "stages" / "2"
        (self.stage_dir / "evidence").mkdir(parents=True)

    def _append(self, event, stage, **data):
        self.events.append((event, stage, data))
        return {"seq": len(self.events)}

    def _reset(self, lease):
        self.resets.append(lease.lease_id)
        self.timeline.append(("reset", lease.lease_id))

    def _run_test(self, spec, stage_dir, test, lease, *, boards=None, extra=None, output="evidence/hw-test-output.txt", **kw):
        code, timed_out, text = self.outputs[min(self.tries, len(self.outputs) - 1)]
        self.tries += 1
        self.timeline.append(("run", self.tries))
        (stage_dir / output).write_text(text)
        return {"returncode": code, "timed_out": timed_out, "seconds": 12.0, "chips": [], "command": "t", "output": {}}

    def run(self, leases):
        return self.s._run_wide_test(self.spec, self.stage_dir, {}, leases, boards=2)

    def retries(self):
        return [d for e, _, d in self.events if d.get("decision", "").startswith("wide test hit the fabric")]


def test_a_fabric_timeout_resets_both_boards_lowest_pci_last_and_runs_again(tmp_path):
    r = Rig(tmp_path, [(1, False, FABRIC), (0, False, "ok\n")])
    result = r.run([A, B])
    assert result["returncode"] == 0 and r.tries == 2
    assert r.resets == ["B", "A"]                            # the lease with the lowest PCI address is reset last
    assert r.timeline == [("run", 1), ("reset", "B"), ("reset", "A"), ("run", 2)]
    (d,) = r.retries()
    assert d["attempt"] == 1 and d["reset_order"] == ["B", "A"] and d["key"] == "pci_address"
    assert d["signature"] == "Fabric Router Sync: Timeout"
    assert "Fabric Router Sync" in (r.stage_dir / "evidence" / "hw-test-output-try1.txt").read_text()
    assert (r.stage_dir / "evidence" / "hw-test-output.txt").read_text() == "ok\n"


def test_the_order_follows_the_pci_address_not_the_list_order(tmp_path):
    r = Rig(tmp_path, [(1, False, FABRIC), (0, False, "ok\n")])
    r.run([B, A])
    assert r.resets == ["B", "A"] and r.retries()[0]["reset_order"] == ["B", "A"]


def test_at_most_two_retries_and_the_last_failure_is_the_result(tmp_path):
    r = Rig(tmp_path, [(1, False, FABRIC)])
    result = r.run([A, B])
    assert result["returncode"] == 1, "a fabric failure after the retries is still a failed test"
    assert r.tries == 3 and [d["attempt"] for d in r.retries()] == [1, 2]
    assert r.resets == ["B", "A", "B", "A"]


def test_the_second_retry_can_be_the_one_that_passes(tmp_path):
    r = Rig(tmp_path, [(1, False, FABRIC), (1, False, FABRIC), (0, False, "ok\n")])
    assert r.run([A, B])["returncode"] == 0 and r.tries == 3 and len(r.retries()) == 2


@pytest.mark.parametrize("code,timed_out,text", [
    (1, False, "server never healthy\n"),                    # another failure
    (1, False, "Fabric Router Sync: ok\nTimeout elsewhere\n"),   # the two phrases on different lines
    (None, False, FABRIC),                                   # a refused command never reached a chip
    (None, True, FABRIC),
    (1, True, FABRIC),                                       # a timeout is not a startup failure
    (0, False, FABRIC),                                      # a pass that logged the line
])
def test_only_a_failed_test_with_the_signature_is_retried(tmp_path, code, timed_out, text):
    r = Rig(tmp_path, [(code, timed_out, text)])
    r.run([A, B])
    assert r.tries == 1 and not r.resets and not r.retries()


def test_an_unreadable_output_is_not_a_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "FABRIC_SCAN_MAX", 4)
    r = Rig(tmp_path, [(1, False, FABRIC)])
    r.run([A, B])
    assert r.tries == 1 and not r.retries()


def test_a_single_lease_is_reset_and_retried_too(tmp_path):
    r = Rig(tmp_path, [(1, False, FABRIC), (0, False, "ok\n")])
    one = lease("W", ["0000:01:00.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"], [0, 1, 2, 3])
    assert r.run([one])["returncode"] == 0 and r.resets == ["W"]


def test_a_refused_reset_blocks_and_does_not_run_again(tmp_path):
    from orchard.adapters import AdapterError
    r = Rig(tmp_path, [(1, False, FABRIC), (0, False, "ok\n")])
    r.s.test_adapter = SimpleNamespace(reset=lambda lease: (_ for _ in ()).throw(AdapterError("refused")))
    r.s._block = lambda n, msg, **k: (_ for _ in ()).throw(RuntimeError(msg))
    with pytest.raises(RuntimeError, match="resetting the chips"):
        r.run([A, B])
    assert r.tries == 1


# ---- the wiring: the phase itself calls the retrying runner, with the coder parked and a further board ------------------------------

def test_the_wide_phase_with_a_parked_coder_retries_across_both_leases(tmp_path):
    r = Rig(tmp_path, [(1, False, FABRIC), (0, False, "ok\n")])
    s = r.s
    chips = [ChipState(c, "FREE", None, board=b, dev_index=i) for b, l in (("a", A), ("b", B)) for c, i in zip(l.chips, l.dev_indices)]
    s.coder_lease = SimpleNamespace(units=("a",))
    s.adapter = SimpleNamespace(status=lambda: chips)
    s._release_test_lease = lambda: True
    s._take_test_lease = lambda spec, shape, boards, exact: B
    h = SimpleNamespace(park=lambda: A, lease="restored")
    s._handoff = lambda n, lease: h
    s._cleanup_after_test = lambda after: None
    s._give_back_test_lease = lambda n, lease: r.timeline.append(("give back", lease.lease_id))
    s._restore_coder = lambda n, h: r.timeline.append(("restore",))
    s._run_wide_test = Supervisor._run_wide_test.__get__(s)
    s._wide_hardware_phase(r.spec, r.stage_dir, {}, 2)
    assert r.tries == 2 and r.resets == ["B", "A"]
    assert r.timeline[-2:] == [("give back", "B"), ("restore",)]
    assert s.coder_lease == "restored"


def test_the_wide_phase_on_free_boards_retries_on_its_single_lease(tmp_path):
    r = Rig(tmp_path, [(1, False, FABRIC), (0, False, "ok\n")])
    s = r.s
    chips = [ChipState(c, "FREE", None, board=b, dev_index=i) for b, l in (("a", A), ("b", B)) for c, i in zip(l.chips, l.dev_indices)]
    whole = lease("W", A.chips + B.chips, [0, 1, 2, 3])
    s.coder_lease = None
    s.adapter = SimpleNamespace(status=lambda: chips)
    s._release_test_lease = lambda: True
    s._take_test_lease = lambda spec, shape, boards, exact: whole
    s._cleanup_after_test = lambda after: None
    s._give_back_test_lease = lambda n, lease: None
    s._wide_hardware_phase(r.spec, r.stage_dir, {}, 2)
    assert r.tries == 2 and r.resets == ["W"]
