# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The lease whose chips have the lowest PCI address is reset last before a 4-chip test (issue 26).

Measured on 2026-11-03 on one two-board box, one 1x4 mesh open per trial with the fabric on (1-D, strict init), each trial
starting from a release and a fresh lease:
  one lease on all four chips, one `gozer reset`:                          0 of 11 trials failed;
  two leases, the lease with the lowest PCI address reset last:             0 of 12 failed;
  two leases, the lease with the lowest PCI address reset first:            6 of 12 failed, every time with `Fabric Router Sync:
                                                                            Timeout after 10000 ms on Device 0`.
On that box three keys give the same order: the lowest device index, the lowest PCI address, and the higher board serial first.
The trials cannot tell them apart. The rule uses the PCI address because it follows the physical slot, and a device index is a
position the driver assigns in probe order. No reboot and no second box has been tried.
When no lease carries a PCI address the order falls back to the device index, and the ledger says which key a reset used."""
import pytest

import run_fakes
from orchard import supervisor
from orchard.supervisor import EXIT_READY, reset_order
from run_fakes import BOARDS, DEV, FAKE_SWAP_TEST, MachineAdapter, bringup
from test_supervisor import Rig
from test_supervisor_fastretry import FLAKY, with_config

pytestmark = pytest.mark.usefixtures("stub_tools")

LOW, HIGH = BOARDS["B0"][0], BOARDS["B1"][0]          # the lowest PCI address of each board


class FakeLease:
    def __init__(self, chips, dev_indices):
        self.chips, self.dev_indices = tuple(chips), tuple(dev_indices)


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


@pytest.fixture
def lowest_bdf(monkeypatch):
    """lease id -> the lowest PCI address of the lease, for every lease the supervisor resets."""
    seen = {}
    real = MachineAdapter.reset

    def recording(self, lease):
        seen[lease.lease_id] = min(lease.chips)
        return real(self, lease)
    monkeypatch.setattr(MachineAdapter, "reset", recording)
    return seen


def four_chip_entries(rig):
    return [e for e in rig.entries() if e["stage"] == 4 and (e["data"].get("config") == 4 or e["event"] == "park")]


def keys_recorded(rig):
    return [e["data"] for e in rig.entries() if e["data"].get("decision") == "test chips reset order"]


def test_the_lease_with_the_lowest_pci_address_is_reset_last():
    low, high = FakeLease(["0000:01:00.0", "0000:02:00.0"], [0, 1]), FakeLease(["0000:03:00.0", "0000:04:00.0"], [2, 3])
    assert reset_order([low, high]) == ([high, low], "pci_address")
    assert reset_order([high, low]) == ([high, low], "pci_address")


def test_the_order_follows_the_pci_address_when_the_device_indices_run_the_other_way():
    low = FakeLease(["0000:01:00.0", "0000:02:00.0"], [2, 3])       # lowest PCI address, highest device indices
    high = FakeLease(["0000:03:00.0", "0000:04:00.0"], [0, 1])
    assert reset_order([low, high]) == ([high, low], "pci_address")
    assert reset_order([high, low]) == ([high, low], "pci_address")


def test_pci_addresses_compare_as_numbers_not_as_text():
    a = FakeLease(["0000:0B:00.0"], [0])                              # bus 0x0b is above bus 0x0a; as text "B" sorts below "a"
    b = FakeLease(["0000:0a:00.0"], [1])
    assert reset_order([a, b])[0] == [a, b]
    assert reset_order([b, a])[0] == [a, b]


def test_without_pci_addresses_the_order_falls_back_to_the_device_index_and_says_so():
    low, high = FakeLease([], [0, 1]), FakeLease([], [2, 3])
    assert reset_order([low, high]) == ([high, low], "device_index")
    odd = FakeLease(["0000:09:00.0", "not-an-address"], [2, 3])
    low = FakeLease(["0000:01:00.0"], [0, 1])
    assert reset_order([low, odd]) == ([odd, low], "device_index")      # one bad address: no mixed keys


def test_with_neither_key_the_list_order_is_kept():
    a, b = FakeLease([], []), FakeLease([], [])
    assert reset_order([a, b]) == ([a, b], "list_order")


def test_reset_order_does_not_change_the_list_it_is_given():
    leases = [FakeLease(["0000:01:00.0"], [0]), FakeLease(["0000:03:00.0"], [2])]
    ordered, _ = reset_order(leases)
    assert [l.chips for l in leases] == [("0000:01:00.0",), ("0000:03:00.0",)] and len(ordered) == 2


def test_a_startup_retry_resets_the_lease_with_the_lowest_pci_address_last(rig, lowest_bdf):
    """The park lease (the coder's board) comes first in the list the retry walks. It must still be reset last."""
    rig.script = lambda r: bringup(r, overrides=with_config(4, FLAKY.format(n=4) + FAKE_SWAP_TEST))
    assert rig.run() == EXIT_READY
    es = four_chip_entries(rig)
    first, second = [i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started"]
    between = [lowest_bdf[e["data"]["lease_id"]] for e in es[first:second] if e["data"].get("decision") == "test chips reset"]
    assert between == [HIGH, LOW]
    assert [k["key"] for k in keys_recorded(rig)] == ["pci_address"]


def test_the_retry_follows_the_pci_order_when_the_device_indices_are_reversed(rig, lowest_bdf, monkeypatch):
    """The fake machine hands out device indices in the opposite order to the PCI addresses."""
    monkeypatch.setattr(run_fakes, "DEV", {c: 3 - i for c, i in DEV.items()})
    rig.script = lambda r: bringup(r, overrides=with_config(4, FLAKY.format(n=4) + FAKE_SWAP_TEST))
    assert rig.run() == EXIT_READY
    es = four_chip_entries(rig)
    first, second = [i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started"]
    between = [lowest_bdf[e["data"]["lease_id"]] for e in es[first:second] if e["data"].get("decision") == "test chips reset"]
    assert between == [HIGH, LOW]


def test_with_the_coder_on_the_higher_pci_address_the_further_lease_is_reset_again_after_the_park(rig, lowest_bdf, monkeypatch):
    """The further lease holds the lowest PCI address. It was reset at take time and the park has since reset the coder's lease.
    It must be reset once more, so that the lowest PCI address is reset last."""
    monkeypatch.setattr(run_fakes, "BOARDS", {"B1": BOARDS["B1"], "B0": BOARDS["B0"]})
    assert rig.run() == EXIT_READY
    es = four_chip_entries(rig)
    park = next(i for i, e in enumerate(es) if e["event"] == "park")
    started = next(i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started")
    before = [e for e in es[:park] if e["data"].get("decision") == "test chips reset"]
    after = [e for e in es[park:started] if e["data"].get("decision") == "test chips reset"]
    assert len(before) == 1 and len(after) == 1
    assert before[0]["data"]["lease_id"] == after[0]["data"]["lease_id"]       # the further lease, twice
    assert lowest_bdf[after[0]["data"]["lease_id"]] == LOW
    assert [k["key"] for k in keys_recorded(rig)] == ["pci_address"]


def test_with_the_coder_on_the_lower_pci_address_the_order_is_already_right_and_nothing_is_reset_again(rig, lowest_bdf):
    assert rig.run() == EXIT_READY
    es = four_chip_entries(rig)
    park = next(i for i, e in enumerate(es) if e["event"] == "park")
    started = next(i for i, e in enumerate(es) if e["data"].get("decision") == "hardware test started")
    assert [e for e in es[park:started] if e["data"].get("decision") == "test chips reset"] == []
    assert len([e for e in es[:park] if e["data"].get("decision") == "test chips reset"]) == 1
    assert keys_recorded(rig) == []


def test_a_retry_without_pci_addresses_records_that_it_used_the_device_index(rig, monkeypatch):
    monkeypatch.setattr(supervisor, "lease_bdfs", lambda lease: ())
    rig.script = lambda r: bringup(r, overrides=with_config(4, FLAKY.format(n=4) + FAKE_SWAP_TEST))
    assert rig.run() == EXIT_READY
    assert [k["key"] for k in keys_recorded(rig)] == ["device_index"]
