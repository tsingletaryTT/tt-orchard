# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A stage 2 port declares whether it writes a converted-weight cache, so the first-boot disk check can block on it (issue 451).

PR 435 made the converted cache a ledger notice, because a port's `load` may convert straight onto the chips and the check runs before the
port exists. The port now says so in `stages/2/port/converted_cache.json`; the supervisor reads it after the coder wrote the port and before
the hardware test. These tests cover the reader (`orchard/port_cache.py`), the disk sentence (`residency.cache_shortfall`), the supervisor's
check on a stub, and the whole stage 2 through the supervisor with a coder that writes the declaration."""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from orchard import blocked, port_cache, residency
from orchard.supervisor import Supervisor

GIB = 1 << 30


# ---- the reader ------------------------------------------------------------------------------------------------------------------

def declare(tmp_path, body):
    (tmp_path / "converted_cache.json").write_bytes(body if isinstance(body, bytes) else json.dumps(body).encode())
    return port_cache.read(tmp_path)


def test_no_file_is_absent_and_reads_nothing(tmp_path):
    assert port_cache.read(tmp_path) == {"state": "absent", "reason": None, "cache_dir": None, "sha256": None}
    assert port_cache.read(tmp_path / "no-such-dir")["state"] == "absent"


def test_a_declared_cache_and_a_declared_none_are_told_apart_and_the_hash_is_of_the_bytes_read(tmp_path):
    raw = b'{"writes_converted_cache": true, "cache_dir": "converted"}'
    got = declare(tmp_path, raw)
    assert got == {"state": "writes", "reason": None, "cache_dir": "converted", "sha256": hashlib.sha256(raw).hexdigest()}
    assert declare(tmp_path, {"writes_converted_cache": False})["state"] == "none"
    assert declare(tmp_path, {"writes_converted_cache": True})["cache_dir"] is None


@pytest.mark.parametrize("body", [
    {"writes_converted_cache": 1}, {"writes_converted_cache": "yes"}, {"writes_converted_cache": None}, {}, {"cache_dir": "x"},
    {"writes_converted_cache": True, "cache_dir": 5}, {"writes_converted_cache": True, "cache_dir": "x" * 201},
    {"writes_converted_cache": True, "extra": 1}, [True], "true", 7,
    b"not json", b'{"writes_converted_cache": true, "writes_converted_cache": false}', b'{"writes_converted_cache": NaN}', b"\xff\xfe",
    b'{"writes_converted_cache": true, "cache_dir": "\\ud800"}', b"{" * 5000, b" " * 5000,
])
def test_anything_that_is_not_the_declaration_is_unreadable_and_says_why(tmp_path, body):
    got = declare(tmp_path, body)
    assert got["state"] == "unreadable" and got["reason"].startswith("converted_cache.json")


def test_a_declaration_over_the_size_cap_is_refused_even_when_it_is_valid_json(tmp_path):
    got = declare(tmp_path, b'{"writes_converted_cache": true}' + b" " * 4097)
    assert got["state"] == "unreadable" and got["sha256"] is None
    assert declare(tmp_path, b'{"writes_converted_cache": true}' + b" " * 100)["state"] == "writes"


def test_a_link_a_fifo_and_a_directory_are_unreadable_not_followed(tmp_path):
    target = tmp_path / "elsewhere.json"
    target.write_text('{"writes_converted_cache": true}')
    port = tmp_path / "port"
    port.mkdir()
    os.symlink(target, port / "converted_cache.json")
    assert port_cache.read(port)["state"] == "unreadable"
    (port / "converted_cache.json").unlink()
    os.mkfifo(port / "converted_cache.json")
    assert port_cache.read(port)["state"] == "unreadable"
    (port / "converted_cache.json").unlink()
    (port / "converted_cache.json").mkdir()
    assert port_cache.read(port)["state"] == "unreadable"


# ---- the disk sentence -----------------------------------------------------------------------------------------------------------

def record(disk=500 * GIB, download=0):
    return {"label": "estimate", "scan": "ran", "assumptions": [], "estimated": {"chips_needed": 1, "device_bytes": 1,
            "first_boot": {"snapshot_local": download == 0, "checkpoint_bytes_to_download": download, "cache_bytes": 1, "disk_bytes": disk,
                           "conversion_in_flight_bytes": 0, "host_ram_bytes": 1}}, "verdict": {"fits_box": True}}


def test_a_declared_cache_on_a_disk_too_small_for_it_blocks_with_both_figures():
    why = residency.cache_shortfall(record(disk=500 * GIB), free_disk_bytes=100 * GIB)
    assert "500.0 GiB" in why and "100.0 GiB" in why and "converted weights" in why
    assert blocked.block_code({"reason": "blocked: " + why}) == "disk-full"


def test_the_sentence_is_the_one_machine_check_gives_for_a_declared_cache():
    why = residency.cache_shortfall(record(), free_disk_bytes=100 * GIB)
    assert why.startswith(residency.machine_check(record(), free_disk_bytes=100 * GIB, mem_available_bytes=None, cache_declared=True)[0].split(" (estimates")[0])


@pytest.mark.parametrize("rec,free", [(record(disk=50 * GIB), 100 * GIB), (record(), None), (None, 100 * GIB), ({"scan": "skipped"}, 100 * GIB),
                                      (record(download=400 * GIB), 100 * GIB), (record(disk=100 * GIB), 100 * GIB)])
def test_a_disk_that_holds_it_a_missing_reading_a_damaged_record_and_a_download_that_already_blocked_are_no_claim(rec, free):
    assert residency.cache_shortfall(rec, free_disk_bytes=free) is None


# ---- the supervisor's check on a stub --------------------------------------------------------------------------------------------

class _Ledger:
    def __init__(self):
        self.rows = []

    def append(self, event, stage, **data):
        self.rows.append((event, stage, data))


def stub(tmp_path, monkeypatch, *, free, lab=None, rec=None):
    reads, blocks, notes = [], [], []
    monkeypatch.setattr(residency, "read_record", lambda run_dir: rec if rec is not None else record())
    monkeypatch.setattr(residency, "machine_facts", lambda path: reads.append(path) or (free, 1 << 50))
    s = SimpleNamespace(run_dir=tmp_path, lab=lab, paths=SimpleNamespace(cache_root=tmp_path / "cache"), ledger=_Ledger(),
                        _block=lambda stage, why, **ev: blocks.append((stage, why, ev)), _note=lambda event, **d: notes.append((event, d)))
    return s, reads, blocks, notes


SPEC = SimpleNamespace(number=2, boards=1, draft="full_port")


def port_dir(tmp_path, body=None):
    stage = tmp_path / "stage"
    (stage / "port").mkdir(parents=True, exist_ok=True)
    if body is not None:
        (stage / "port" / "converted_cache.json").write_text(json.dumps(body))
    return stage


def test_a_port_that_declares_a_cache_the_disk_cannot_hold_blocks_naming_both_figures_and_the_hash(tmp_path, monkeypatch):
    s, reads, blocks, notes = stub(tmp_path, monkeypatch, free=100 * GIB)
    stage = port_dir(tmp_path, {"writes_converted_cache": True, "cache_dir": "conv"})
    Supervisor._check_port_cache(s, SPEC, stage)
    assert len(blocks) == 1 and reads == [tmp_path / "cache"]
    n, why, ev = blocks[0]
    assert n == 2 and "500.0 GiB" in why and "100.0 GiB" in why and ev["disk_bytes_needed"] == 500 * GIB and ev["free_disk_bytes"] == 100 * GIB
    assert ev["declaration_sha256"] == hashlib.sha256((stage / "port" / "converted_cache.json").read_bytes()).hexdigest()
    ledger = [d for e, n_, d in s.ledger.rows if d.get("decision") == "port cache declaration"]
    assert len(ledger) == 1 and ledger[0]["state"] == "writes" and ledger[0]["sha256"] == ev["declaration_sha256"]


def test_a_port_that_declares_a_cache_the_disk_holds_is_not_blocked(tmp_path, monkeypatch):
    s, reads, blocks, notes = stub(tmp_path, monkeypatch, free=1000 * GIB)
    Supervisor._check_port_cache(s, SPEC, port_dir(tmp_path, {"writes_converted_cache": True}))
    assert blocks == [] and notes == []


def test_a_port_that_declares_none_is_not_blocked_and_reads_no_disk(tmp_path, monkeypatch):
    s, reads, blocks, notes = stub(tmp_path, monkeypatch, free=1)
    Supervisor._check_port_cache(s, SPEC, port_dir(tmp_path, {"writes_converted_cache": False}))
    assert blocks == [] and reads == [] and notes == []
    assert [d["state"] for e, n, d in s.ledger.rows] == ["none"]


@pytest.mark.parametrize("body,state", [(None, "absent"), ({"writes_converted_cache": "yes"}, "unreadable")])
def test_a_port_that_declared_nothing_is_not_blocked_and_the_ledger_says_so(tmp_path, monkeypatch, body, state):
    s, reads, blocks, notes = stub(tmp_path, monkeypatch, free=1)
    Supervisor._check_port_cache(s, SPEC, port_dir(tmp_path, body))
    assert blocks == [] and reads == []
    assert [d["state"] for e, n, d in s.ledger.rows] == [state]
    assert len(notes) == 1 and notes[0][1]["what"] == "the port declared no converted-weight cache" and "not checked" in notes[0][1]["detail"]


def test_a_lab_run_reads_no_disk_of_this_box_for_a_declared_cache(tmp_path, monkeypatch):
    s, reads, blocks, notes = stub(tmp_path, monkeypatch, free=1, lab=object())
    Supervisor._check_port_cache(s, SPEC, port_dir(tmp_path, {"writes_converted_cache": True}))
    assert blocks == [] and reads == []
    assert [d["state"] for e, n, d in s.ledger.rows] == ["writes"]          # the declaration is still recorded


def test_an_unknown_reading_or_no_record_is_no_claim_for_a_declared_cache(tmp_path, monkeypatch):
    s, reads, blocks, notes = stub(tmp_path, monkeypatch, free=None)
    Supervisor._check_port_cache(s, SPEC, port_dir(tmp_path, {"writes_converted_cache": True}))
    assert blocks == []
    monkeypatch.setattr(residency, "read_record", lambda run_dir: None)
    Supervisor._check_port_cache(s, SPEC, port_dir(tmp_path, {"writes_converted_cache": True}))
    assert blocks == []


# ---- the whole stage 2 (the check is wired between the port and the hardware test) -----------------------------------------------

from test_supervisor_residency import SMALL, rig, with_model  # noqa: E402,F401  (rig is a fixture)
from test_supervisor_full_port import FILES, PREPARE, make, never_wait, started  # noqa: E402,F401
from orchard.supervisor import Control  # noqa: E402

pytestmark = pytest.mark.usefixtures("stub_tools")


def run_stage_2(rig, tmp_path, monkeypatch, declaration, free):
    with_model(rig, tmp_path, SMALL)
    extra = {} if declaration is None else {"port/converted_cache.json": declaration}
    make(rig, over={(2, "prepare"): {**PREPARE, **extra}})
    monkeypatch.setattr(residency, "machine_facts", lambda path: (free, 1 << 50))
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    return [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"]


def declarations(rig):
    return [d for d in rig.decisions() if d.get("decision") == "port cache declaration"]


def test_a_declared_cache_on_a_disk_too_small_for_it_blocks_before_the_hardware_test(rig, tmp_path, monkeypatch):
    pauses = run_stage_2(rig, tmp_path, monkeypatch, {"writes_converted_cache": True}, free=1 << 20)
    assert pauses and pauses[0].startswith("blocked: the first boot needs about") and blocked.block_code({"reason": pauses[0]}) == "disk-full"
    assert declarations(rig)[0]["state"] == "writes"
    assert not (rig.run_dir / "stages" / "2" / "test-result.json").exists()           # the hardware test did not run


def test_a_port_that_declares_none_goes_on_to_the_hardware_test_on_a_small_disk(rig, tmp_path, monkeypatch):
    pauses = run_stage_2(rig, tmp_path, monkeypatch, {"writes_converted_cache": False}, free=1 << 20)
    assert not any("first boot" in p for p in pauses)
    assert declarations(rig)[0]["state"] == "none" and (rig.run_dir / "stages" / "2" / "test-result.json").exists()


def test_a_port_with_no_declaration_goes_on_and_the_ledger_has_the_notice(rig, tmp_path, monkeypatch):
    pauses = run_stage_2(rig, tmp_path, monkeypatch, None, free=1 << 20)
    assert not any("first boot" in p for p in pauses)
    assert declarations(rig)[0]["state"] == "absent"
    assert any(e["event"] == "notice" and e["data"].get("what") == "the port declared no converted-weight cache" for e in rig.entries())


def test_the_stage_2_skill_tells_the_coder_the_file_name_and_both_forms():
    from pathlib import Path
    text = (Path(__file__).resolve().parent.parent / "orchard" / "skills" / "full-port-check.md").read_text()
    assert port_cache.FILE_NAME in text and '"writes_converted_cache": true' in text and '"writes_converted_cache": false' in text


def test_the_ledger_decision_has_no_reason_key_that_the_pause_readers_would_pick_up(tmp_path, monkeypatch):
    # readers of decisions take `reason` as the text of a pause or a block; this decision is neither
    s, reads, blocks, notes = stub(tmp_path, monkeypatch, free=1)
    Supervisor._check_port_cache(s, SPEC, port_dir(tmp_path))
    assert all("reason" not in d for e, n, d in s.ledger.rows)
