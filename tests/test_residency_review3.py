# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""PR 435, the third Copilot review (the round limit): two more places where the estimate said more than it knew.

1. A mixture-of-experts record said every expert occupies chip memory, and its text labelled the expert bytes "on the chips", even under
   `--policy routed_experts=host`, where the experts stay in host memory and `device_bytes` holds none of them. The record now names the
   placement it used and the note and the text follow it.
2. The first boot's disk check demanded a converted-weight cache for every full port. A port's `load(model_dir, mesh_device)` may convert
   straight onto the chips and write no cache (scripts/minicpm3/chip_model.py does), and nothing a port declares exists when the check runs
   (stage 2 has not written it yet). The converted cache is now a notice. The download of the checkpoint (needed whatever the port does) and the
   host RAM stay blocking, and a caller that knows the port writes a cache says so (`cache_declared=True`) and gets the old block."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from orchard import blocked, residency as rs
from residency_headers import flash_next_tensors, write_snapshot
from test_residency_options import FIX, GIB, record


@pytest.fixture(scope="module")
def flash_headers(tmp_path_factory):
    cfg = json.loads((FIX / "qwen3.8-flash-next.json").read_text())
    return rs.read_headers(write_snapshot(tmp_path_factory.mktemp("flash3"), flash_next_tensors(cfg), shards=7, config=cfg))


@pytest.fixture(scope="module")
def on_chips(flash_headers):
    return rs.estimate(flash_headers, box_chips=4)


@pytest.fixture(scope="module")
def on_host(flash_headers):
    return rs.estimate(flash_headers, box_chips=4, policy={"routed_experts": "host"})


# ---- finding 2: the expert placement ----------------------------------------------------------------------------------------------


def test_experts_on_the_chips_are_reported_on_the_chips_as_before(on_chips):
    moe = on_chips["estimated"]["mixture_of_experts"]
    assert moe["placement"] == "device"
    assert moe["routed_expert_device_bytes"] == moe["routed_expert_stored_bytes"] > 0 and moe["routed_expert_host_bytes"] == 0
    assert "every expert" in moe["note"] and "resident" in moe["note"] and "on the chips" in moe["note"]


def test_experts_kept_on_the_host_are_reported_in_host_memory_and_not_on_the_chips(on_host, on_chips):
    moe = on_host["estimated"]["mixture_of_experts"]
    assert moe["present"] is True and moe["placement"] == "host"
    assert moe["routed_expert_device_bytes"] == 0 and moe["routed_expert_host_bytes"] == moe["routed_expert_stored_bytes"] > 0
    assert "host" in moe["note"] and "on the chips" not in moe["note"] and "no chip memory" in moe["note"]
    assert on_host["estimated"]["device_bytes"] < on_chips["estimated"]["device_bytes"]          # the experts are not in the device total


def test_the_text_says_where_the_experts_sit(on_host, on_chips):
    host_line = next(l for l in rs._text(on_host, []) if "mixture of experts" in l)
    assert "in host memory" in host_line and "on the chips" not in host_line
    chip_line = next(l for l in rs._text(on_chips, []) if "mixture of experts" in l)
    assert "on the chips" in chip_line and "in host memory" not in chip_line


def test_a_dense_model_still_has_no_mark(tmp_path):
    from residency_headers import dense_tensors
    cfg = json.loads((FIX / "qwen3.8-27b.json").read_text())
    rec = rs.estimate(rs.read_headers(write_snapshot(tmp_path, dense_tensors(cfg), shards=4, config=cfg)), box_chips=4)
    assert rec["estimated"]["mixture_of_experts"] == {"present": False}


# ---- finding 3: the converted cache is a notice unless a port declares it --------------------------------------------------------


def test_a_converted_cache_that_does_not_fit_is_a_notice_and_not_a_block_by_default():
    block, notice = rs.machine_check(record(disk=500 * GIB), free_disk_bytes=120 * GIB, mem_available_bytes=None)
    assert block is None
    assert notice and "500.0 GiB" in notice and "120.0 GiB" in notice and "converted" in notice and "not a block" in notice
    assert rs.machine_shortfall(record(disk=500 * GIB), free_disk_bytes=120 * GIB, mem_available_bytes=None) is None


def test_a_port_that_declares_a_cache_gets_the_disk_block_back():
    block, notice = rs.machine_check(record(disk=500 * GIB), free_disk_bytes=120 * GIB, mem_available_bytes=None, cache_declared=True)
    assert notice is None and "500.0 GiB" in block and "120.0 GiB" in block and "free disk" in block
    assert blocked.block_code({"reason": "blocked: " + block}) == "disk-full"
    assert rs.machine_shortfall(record(disk=500 * GIB), free_disk_bytes=120 * GIB, mem_available_bytes=None, cache_declared=True) == block


def test_a_download_that_does_not_fit_blocks_whatever_the_port_does():
    block, notice = rs.machine_check(record(disk=500 * GIB, download=400 * GIB), free_disk_bytes=120 * GIB, mem_available_bytes=None)
    assert notice is None and "download" in block and "400.0 GiB" in block and "120.0 GiB" in block and "free disk" in block
    assert blocked.block_code({"reason": "blocked: " + block}) == "disk-full"


def test_host_ram_blocks_whatever_the_port_declares():
    block, notice = rs.machine_check(record(ram=300 * GIB), free_disk_bytes=None, mem_available_bytes=100 * GIB)
    assert block and block.startswith("the model does not fit this box") and "host RAM" in block and notice is None


def test_a_cache_notice_is_quiet_when_the_disk_holds_the_first_boot():
    assert rs.machine_check(record(), free_disk_bytes=11 * GIB, mem_available_bytes=21 * GIB) == (None, None)


def test_the_notice_and_the_ram_block_come_together_when_both_apply():
    block, notice = rs.machine_check(record(disk=500 * GIB, ram=300 * GIB), free_disk_bytes=120 * GIB, mem_available_bytes=100 * GIB)
    assert block and "host RAM" in block and "free disk" not in block and notice and "converted-weight cache" in notice


# ---- the supervisor applies the split ---------------------------------------------------------------------------------------------


def _rec(disk, ram=1, download=0):
    return {"label": "estimate", "scan": "ran", "assumptions": [], "estimated": {"chips_needed": 1, "device_bytes": 1,
            "first_boot": {"snapshot_local": download == 0, "checkpoint_bytes_to_download": download, "cache_bytes": 1, "disk_bytes": disk,
                           "conversion_in_flight_bytes": 0, "host_ram_bytes": ram}}, "verdict": {"fits_box": True}}


def _check(tmp_path, monkeypatch, rec, readings):
    from orchard.supervisor import Supervisor
    monkeypatch.setattr(rs, "read_record", lambda run_dir: rec)
    monkeypatch.setattr(rs, "machine_facts", lambda path: readings)
    blocks, notes = [], []
    stub = SimpleNamespace(run_dir=tmp_path, lab=None, paths=SimpleNamespace(cache_root=tmp_path / "cache"), _box_chips=lambda spec: 4,
                           _block=lambda stage, why, **ev: blocks.append((why, ev)), _note=lambda event, **data: notes.append((event, data)))
    Supervisor._check_residency(stub, SimpleNamespace(boards=1, number=2, draft="full_port"), hardware_pending=True)
    return blocks, notes


def test_the_supervisor_does_not_block_stage_2_on_a_converted_cache_that_may_not_fit_and_records_a_notice(tmp_path, monkeypatch):
    blocks, notes = _check(tmp_path, monkeypatch, _rec(disk=500 * GIB), (120 * GIB, 100 * GIB))
    assert blocks == []
    [(event, data)] = notes
    assert event == "notice" and data["what"] == "the converted-weight cache may not fit" and "500.0 GiB" in data["detail"]
    assert data["disk_bytes_needed"] == 500 * GIB and data["free_disk_bytes"] == 120 * GIB


def test_the_supervisor_still_blocks_stage_2_on_a_download_that_does_not_fit(tmp_path, monkeypatch):
    blocks, notes = _check(tmp_path, monkeypatch, _rec(disk=500 * GIB, download=400 * GIB), (120 * GIB, 100 * GIB))
    assert len(blocks) == 1 and "download" in blocks[0][0] and blocked.block_code({"reason": "blocked: " + blocks[0][0]}) == "disk-full"
    assert notes == []


def test_the_supervisor_still_blocks_stage_2_on_host_ram(tmp_path, monkeypatch):
    blocks, notes = _check(tmp_path, monkeypatch, _rec(disk=1, ram=300 * GIB), (120 * GIB, 100 * GIB))
    assert len(blocks) == 1 and "host RAM" in blocks[0][0] and notes == []


# ---- the older finding: the Hub's totals are a block only when no host-resident table can be among them ---------------------------------


@pytest.mark.parametrize("host_table,status", [(True, "warn"), (None, "warn"), (False, "block")])
def test_the_totals_block_only_when_a_host_resident_table_is_ruled_out(host_table, status):
    from orchard import preflight as pf
    from test_residency_round3 import hub_payload
    hub = pf.parse_hub(hub_payload())
    hub.host_table = host_table
    c = pf.check_fit(hub, 4)
    assert c.status == status
    if status == "warn":
        assert "not a block" in c.detail and c.reason is None
