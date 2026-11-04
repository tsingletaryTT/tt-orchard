# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 on a container (v5.1) package (issue 60): the supervisor drafts the container config, copies the container script, and stops
what the test leaves behind. A v6 bundle that fits is used as before."""
import hashlib
import json
from pathlib import Path

import pytest

from orchard.supervisor import EXIT_READY, parse
from run_fakes import FILES, FakeContainers, argv, bringup
from swap_run_sh import valid_run_sh
from test_supervisor import Rig, Stop

pytestmark = pytest.mark.usefixtures("stub_tools")

NEAREST = "Qwen/Qwen3.8-27B"


def container_rig(tmp_path, *, bundle=False, listings=None, chips=2):
    r = Rig(tmp_path, chips=chips)
    op, root = tmp_path / "op", tmp_path / "tt-models"
    a = argv(tmp_path, r.tiers, r.chip_server.endpoint, chips=chips)
    a[a.index("--input") + 1] = f"base={tmp_path}/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/abc"
    a += ["--input", f"model={tmp_path}/hf/hub/models--Altworld--Hemmingway-1/snapshots/def", "--hf-home", f"{tmp_path}/hf",
          "--cache-root", f"{tmp_path}/cache", "--tt-model-root", str(root), "--operator-home", str(op)]
    r.args = parse(a)
    for key in ("base", "model"):
        Path(next(i for i in r.args.input if i.startswith(key + "=")).split("=", 1)[1]).mkdir(parents=True)
    mf = op / ".cache" / "tt-model" / "pulled" / "someone__qwen3.8-27b-p300" / "tt_kernel_manifest.json"
    mf.parent.mkdir(parents=True)
    mf.write_text(json.dumps({"schema_version": "5.1", "device_count": 2, "weights": {"repo_id": NEAREST}}))
    (op / ".cache" / "tt-model" / "installed.json").write_text(json.dumps(
        {"someone/qwen3.8-27b-p300": {"container": True, "image": "tt-model/qwen:abc", "profile": "default",
                                      "profiles": ["default"], "manifest": str(mf)}}))
    if bundle:
        b = root / "episod" / "qwen3.8-27b-dflash2-p300"
        b.mkdir(parents=True)
        (b / "tt_kernel_manifest.json").write_text(json.dumps(
            {"schema_version": "6", "device_count": 2, "weights": {"repo_id": NEAREST}}))
        (b / "run.sh").write_text(valid_run_sh(NEAREST))
    r.containers = FakeContainers(listings)
    return r


def drafted(r):
    return next(e for e in r.entries() if e["data"].get("decision") == "swap config drafted")


def test_stage_2_is_drafted_from_the_container_package_when_no_bundle_fits(tmp_path):
    r = container_rig(tmp_path)
    try:
        assert r.run() == EXIT_READY
        e = drafted(r)
        assert e["stage"] == 2 and e["data"]["package"] == "someone/qwen3.8-27b-p300" and e["data"]["bundle_dir"] is None
        assert e["data"]["copied"] == ["prepare_swap.py", "serve_and_compare.py", "prompt_suite.json", "serve_and_compare_container.py"]
        sd = r.run_dir / "stages" / "2"
        cfg = json.loads((sd / "swap_config.json").read_text())
        assert cfg["package"] == "someone/qwen3.8-27b-p300" and cfg["chips"] == 2 and "bundle_dir" not in cfg
        assert (sd / "serve_and_compare_container.py").is_file()
    finally:
        r.close()


def test_stage_2_uses_the_agreement_set_when_stage_1_wrote_its_reference_and_records_the_hash(tmp_path):
    r = container_rig(tmp_path, bundle=True)
    ref = r.run_dir / "stages" / "1" / "evidence" / "reference" / "agreement-reference.json"
    stage_1 = {**FILES[(1, "run")], "evidence/reference/agreement-reference.json": '{"entries": []}'}     # stage 1 writes the set's reference too
    r.script = lambda request: bringup(request, overrides={(1, "run"): stage_1})
    try:
        # The rig's fake hardware test writes no agreement record, so the gate (which now needs one, because the config asks for the set)
        # keeps failing stage 2 and the rig's simulated clock stops the run. What this test reads is what the supervisor wrote before that.
        with pytest.raises(AssertionError, match="stuck"):
            r.run()
        sd = r.run_dir / "stages" / "2"
        assert json.loads((sd / "swap_config.json").read_text())["agreement_set"] is True
        assert (sd / "agreement_core.py").is_file() and (sd / "agreement_prompts.json").is_file()
        recs = [e for e in r.entries() if e["data"].get("decision") == "agreement reference recorded"]
        assert recs and all(e["stage"] == 2 and e["data"]["sha256"] == hashlib.sha256(ref.read_bytes()).hexdigest() for e in recs)
    finally:
        r.close()


def test_stage_2_records_the_hash_of_stage_1s_reference_ids_before_any_test(tmp_path):
    """Issue 433: the gate recomputes the 32-token check from stage 1's generated ids, which the ledger binds by hash before the test."""
    r = container_rig(tmp_path, bundle=True)
    ids = r.run_dir / "stages" / "1" / "evidence" / "reference" / "generated-ids.json"
    stage_1 = {**FILES[(1, "run")], "evidence/reference/generated-ids.json": '{"generated_ids": [1, 2, 3]}'}
    r.script = lambda request: bringup(request, overrides={(1, "run"): stage_1})
    try:
        try:
            r.run()
        except AssertionError:
            pass                      # the rig's fake test writes no raw replies, so the gate may now refuse stage 2: the record is what this reads
        recs = [e for e in r.entries() if e["data"].get("decision") == "stage 1 reference ids recorded"]
        assert recs and all(e["stage"] == 2 and e["data"]["sha256"] == hashlib.sha256(ids.read_bytes()).hexdigest() for e in recs)
        drafted_at = next(e["seq"] for e in r.entries() if e["data"].get("decision") == "swap config drafted")
        assert min(e["seq"] for e in recs) < drafted_at
    finally:
        r.close()


def test_stage_2_records_no_reference_ids_hash_when_stage_1_wrote_none(tmp_path):
    r = container_rig(tmp_path, bundle=True)
    try:
        assert r.run() == EXIT_READY
        assert not [e for e in r.entries() if e["data"].get("decision") == "stage 1 reference ids recorded"]
    finally:
        r.close()


def test_stage_2_does_not_use_the_agreement_set_without_the_reference(tmp_path):
    r = container_rig(tmp_path, bundle=True)
    try:
        assert r.run() == EXIT_READY
        sd = r.run_dir / "stages" / "2"
        assert "agreement_set" not in json.loads((sd / "swap_config.json").read_text())
        assert not (sd / "agreement_core.py").exists()
        assert not [e for e in r.entries() if e["data"].get("decision") == "agreement reference recorded"]
    finally:
        r.close()


def test_a_bundle_that_fits_is_used_and_no_container_script_is_copied(tmp_path):
    r = container_rig(tmp_path, bundle=True)
    try:
        assert r.run() == EXIT_READY
        d = drafted(r)["data"]
        assert d["bundle_dir"].endswith("qwen3.8-27b-dflash2-p300") and "package" not in d
        assert d["copied"] == ["prepare_swap.py", "serve_and_compare.py", "prompt_suite.json"]
        assert not (r.run_dir / "stages" / "2" / "serve_and_compare_container.py").exists()
    finally:
        r.close()


def stage2_sweeps(r):
    return [e for e in r.entries() if e["stage"] == 2 and e["event"] == "notice"
            and e["data"].get("what") == "a hardware test left containers running; stopping them"]


def test_a_container_left_by_stage_2s_test_is_stopped_by_label(tmp_path):
    r = container_rig(tmp_path, listings=[["c1"], []])
    try:
        assert r.run() == EXIT_READY
        [note] = stage2_sweeps(r)
        assert note["data"]["containers"] == ["c1"]
        assert note["data"]["label"] == "orchard.test=" + hashlib.sha256(str(r.run_dir).encode()).hexdigest()[:12]
        assert ("remove", "c1") in r.containers.calls
    finally:
        r.close()


def test_a_container_that_survives_the_stop_blocks_stage_2_before_the_lease_is_given_back(tmp_path):
    r = container_rig(tmp_path, listings=[["c1"]])
    r.args.unattended = True                # a block ends the run instead of waiting for an operator
    try:
        assert r.run() != EXIT_READY
        blocked = [e for e in r.entries() if e["event"] == "notice" and e["data"].get("blocked") and e["stage"] == 2]
        assert blocked and "still running" in blocked[0]["data"]["reason"]
        assert not [e for e in r.entries() if e["data"].get("decision") == "test lease released" and e["stage"] == 2]
    finally:
        r.close()


def test_a_bundle_test_is_not_swept_at_stage_2(tmp_path):
    r = container_rig(tmp_path, bundle=True, listings=[["c1"], []])
    try:
        assert r.run() == EXIT_READY
        assert stage2_sweeps(r) == []
    finally:
        r.close()


@pytest.mark.parametrize("text,expected", [('{"package": "a/b", "profile": "p"}', True), ('{"bundle_dir": "/b"}', False),
                                           ('["package"]', False), ('"package"', False), ("not json", False), ("", False)])
def test_only_a_config_object_that_names_a_package_is_a_container_config(tmp_path, text, expected):
    from orchard.supervisor import Supervisor
    (tmp_path / "swap_config.json").write_text(text)
    assert Supervisor._container_config(tmp_path) is expected


def test_a_stage_with_no_swap_config_is_not_a_container_config(tmp_path):
    from orchard.supervisor import Supervisor
    assert Supervisor._container_config(tmp_path) is False


# ---- a container package that needs more chips than stage 2's board (issue 60 validation, 2026-10-25) -------------------

def wide_rig(tmp_path, *, device_count=4, **kw):
    r = container_rig(tmp_path, **kw)
    mf = tmp_path / "op" / ".cache" / "tt-model" / "pulled" / "someone__qwen3.8-27b-p300" / "tt_kernel_manifest.json"
    mf.write_text(json.dumps({"schema_version": "5.1", "device_count": device_count, "weights": {"repo_id": NEAREST}}))
    return r


def stage2_tests(r):
    return [e["data"] for e in r.entries() if e["stage"] == 2 and e["event"] == "decision"
            and e["data"].get("decision") == "hardware test started"]


def test_a_four_chip_container_is_tested_on_four_chips_in_stage_2(tmp_path):
    """Catalog container packages report 2 boards. The first real run of issue 60 drafted nothing, because stage 2 asked for a package
    on its own 2 chips. The package is now drafted with its own count and the test holds all four chips (the coder's board parked)."""
    r = wide_rig(tmp_path)
    try:
        assert r.run() == EXIT_READY
        assert drafted(r)["data"]["package"] == "someone/qwen3.8-27b-p300"
        cfg = json.loads((r.run_dir / "stages" / "2" / "swap_config.json").read_text())
        assert cfg["chips"] == 4
        started = stage2_tests(r)
        assert len(started) == 1 and len(started[0]["chips"]) == 4, started
        phase = [d for d in r.decisions() if d.get("decision") == "hardware phase" and d.get("action") == "park"]
        assert phase, "the coder's board was parked for the 4-chip test"
    finally:
        r.close()


def test_a_two_chip_container_is_still_tested_on_two_chips(tmp_path):
    r = wide_rig(tmp_path, device_count=2)
    try:
        assert r.run() == EXIT_READY
        started = stage2_tests(r)
        assert len(started) == 1 and len(started[0]["chips"]) == 2, started
    finally:
        r.close()


def test_the_further_board_goes_back_and_the_coder_is_restored_after_the_wide_test(tmp_path):
    r = wide_rig(tmp_path)
    try:
        assert r.run() == EXIT_READY
        ds = r.decisions()
        taken = [d for d in ds if d.get("decision") == "test lease taken" and d.get("config") == 4]
        assert taken, "the further board was leased for the 4-chip test"
        released = {d["lease_id"] for d in ds if d.get("decision") == "test lease released"}
        assert taken[0]["test_lease"]["lease_id"] in released
        # given back as stage 2's test ends, not left to the run's wind-down
        end2 = next(e["seq"] for e in r.entries() if e["event"] == "stage_end" and e["stage"] == 2)
        back = next(e["seq"] for e in r.entries() if e["data"].get("decision") == "test lease released"
                    and e["data"].get("lease_id") == taken[0]["test_lease"]["lease_id"])
        assert back < end2
    finally:
        r.close()


def test_a_package_that_needs_more_chips_than_the_box_has_blocks_with_the_numbers(tmp_path):
    r = wide_rig(tmp_path, device_count=8)
    def stop():
        raise Stop()
    r.on_sleep = stop                                  # the attended run pauses and would wait for the operator
    try:
        with pytest.raises(Stop):
            r.run()
        blocks = [e for e in r.entries() if e["event"] in ("block", "pause", "notice") and "8 chips" in json.dumps(e["data"])]
        assert any("this box has 4" in json.dumps(e["data"]) for e in blocks), [e["data"] for e in r.entries()][-3:]
    finally:
        r.close()


def test_a_four_chip_coder_hands_the_whole_lease_to_the_four_chip_test(tmp_path):
    """Review of PR 66: with the coder on both boards, parking supplies all four chips and there is no extra lease; the test
    was cut back to stage 2's two chips and the container rejected the device count."""
    r = wide_rig(tmp_path, chips=4)
    try:
        assert r.run() == EXIT_READY
        started = stage2_tests(r)
        assert len(started) == 1 and len(started[0]["chips"]) == 4, started
    finally:
        r.close()


def test_a_blocked_pre_test_reset_does_not_leave_the_further_board_claimed(tmp_path, monkeypatch):
    """Review of PR 66: the further board's lease stayed in test_lease after the reset blocked, so an attended resume saw the board
    claimed and waited for ever. The retained lease goes back before availability is read."""
    from orchard.adapters import Refused
    from orchard.supervisor import Blocked  # noqa: F401
    from test_supervisor import MachineAdapter
    r = wide_rig(tmp_path)
    calls = {"n": 0}
    real = MachineAdapter.reset

    def reset_once_refused(self, lease):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Refused("device still open")
        return real(self, lease)

    monkeypatch.setattr(MachineAdapter, "reset", reset_once_refused)
    pauses = {"n": 0}

    def on_sleep():
        pauses["n"] += 1
        if pauses["n"] > 400:
            raise Stop()
        if pauses["n"] == 1:
            from orchard.supervisor import Control
            Control(r.run_dir).write("resume")
    r.on_sleep = on_sleep
    try:
        assert r.run() == EXIT_READY
        started = stage2_tests(r)
        assert len(started) == 1 and len(started[0]["chips"]) == 4
    finally:
        r.close()


def test_aborting_after_a_blocked_pre_test_reset_gives_back_the_further_board(tmp_path, monkeypatch):
    """Review of PR 66: `_abort` released only the coder's lease, so a further board that a blocked reset left claimed
    stayed leased after the supervisor exited."""
    from orchard.adapters import Refused
    from orchard.supervisor import Control, EXIT_ABORTED
    from test_supervisor import MachineAdapter
    r = wide_rig(tmp_path)

    def refuse(self, lease):
        raise Refused("device still open")

    monkeypatch.setattr(MachineAdapter, "reset", refuse)
    ticks = {"n": 0}

    def abort_at_the_block():
        ticks["n"] += 1
        if ticks["n"] == 2:                      # the run is paused on the refused reset
            Control(r.run_dir).write("abort")
        assert ticks["n"] < 200
    r.on_sleep = abort_at_the_block
    try:
        assert r.run() == EXIT_ABORTED
        assert r.m.leases == {}
    finally:
        r.close()


def test_a_second_board_held_by_someone_else_is_waited_for_not_blocked(tmp_path, monkeypatch):
    """Review of PR 66: with the second board busy, decide_park says wait and the stage blocked at once. Stage 4 queues for the
    board while the coder keeps serving (reacquire waits for the lease); stage 2 does the same."""
    import orchard.supervisor as sup
    from orchard.handoff import ParkDecision
    real = sup.decide_park
    seen = []

    def busy_board(server_boards, chips, boards_needed):
        d = real(server_boards, chips, boards_needed)
        seen.append(d.action)
        return ParkDecision("wait", (), d.server_boards, boards_needed) if boards_needed == 2 else d

    monkeypatch.setattr(sup, "decide_park", busy_board)
    r = wide_rig(tmp_path)
    try:
        assert r.run() == EXIT_READY
        blocked = [e["data"] for e in r.entries() if e["event"] == "notice" and e["data"].get("blocked")]
        assert not [b for b in blocked if "cannot get its boards" in str(b.get("reason"))]
        assert len(stage2_tests(r)[0]["chips"]) == 4
    finally:
        r.close()


def test_stage_2_records_the_reference_ids_hash_even_when_the_config_could_not_be_drafted(tmp_path, monkeypatch):
    """Review of PR 444: the pin came after a successful draft, so a run whose bundle was ambiguous (swap_draft.draft gave None) reached the
    prepare agent and the hardware test with the reference ids unbound. The pin no longer waits for the draft."""
    from orchard import swap_draft
    monkeypatch.setattr(swap_draft, "draft", lambda **kw: (None, ["two bundles match"]))
    r = container_rig(tmp_path, bundle=True)
    ids = r.run_dir / "stages" / "1" / "evidence" / "reference" / "generated-ids.json"
    stage_1 = {**FILES[(1, "run")], "evidence/reference/generated-ids.json": '{"generated_ids": [1, 2, 3]}'}
    r.script = lambda request: bringup(request, overrides={(1, "run"): stage_1})
    try:
        try:
            r.run()
        except AssertionError:
            pass
        entries = r.entries()
        recs = [e for e in entries if e["data"].get("decision") == "stage 1 reference ids recorded"]
        assert recs and recs[0]["data"]["sha256"] == hashlib.sha256(ids.read_bytes()).hexdigest()
        assert any(e["data"].get("decision") == "swap config not drafted" for e in entries)
        first_step = next((e["seq"] for e in entries if e["data"].get("decision") == "agent step"
                           and e["stage"] == 2), None)
        assert first_step is None or recs[0]["seq"] < first_step
    finally:
        r.close()


def test_the_reference_ids_hash_is_recorded_once_and_a_retry_keeps_the_first(tmp_path):
    """The first record is the one the gate compares with: a retried draft must not record a changed file as the new truth."""
    from types import SimpleNamespace
    from orchard.ledger import Ledger
    from orchard.supervisor import Supervisor
    run = tmp_path / "run"
    ids = run / "stages" / "1" / "evidence" / "reference" / "generated-ids.json"
    ids.parent.mkdir(parents=True)
    ids.write_text('{"generated_ids": [1, 2, 3]}')
    led = Ledger(run / "ledger.jsonl")
    try:
        stub, spec = SimpleNamespace(run_dir=run, ledger=led), SimpleNamespace(number=2)
        Supervisor._pin_reference_ids(stub, spec)
        ids.write_text('{"generated_ids": [9, 9, 9]}')                    # the file changes after the first pin
        Supervisor._pin_reference_ids(stub, spec)
        recs = [e for e in led.read() if e["data"].get("decision") == "stage 1 reference ids recorded"]
        assert len(recs) == 1 and recs[0]["data"]["sha256"] == hashlib.sha256(b'{"generated_ids": [1, 2, 3]}').hexdigest()
    finally:
        led.close()


def test_pinning_with_no_reference_ids_file_records_nothing(tmp_path):
    from types import SimpleNamespace
    from orchard.ledger import Ledger
    from orchard.supervisor import Supervisor
    led = Ledger(tmp_path / "ledger.jsonl")
    try:
        Supervisor._pin_reference_ids(SimpleNamespace(run_dir=tmp_path, ledger=led), SimpleNamespace(number=2))
        assert not [e for e in led.read() if e["data"].get("decision") == "stage 1 reference ids recorded"]
    finally:
        led.close()
