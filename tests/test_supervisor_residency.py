# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Slice 4 of the full-port route: the residency estimate is written at stage 0 and read before a full-port stage 2 (issue 327). The chip, the
coder and the model are fakes; the supervisor, the stage machine and the estimator are the real ones. The model is a directory of
safetensors headers (tests/residency_headers.py): a model too large for the box's 4 chips, and one that fits."""
import json

import pytest

from orchard import blocked, narrate, residency, status
from orchard.supervisor import EXIT_BLOCKED, Control
from residency_headers import write_snapshot
from test_supervisor_full_port import make, never_wait, started
from test_supervisor import Rig

pytestmark = pytest.mark.usefixtures("stub_tools")

HUGE = [("model.layers.0.mlp.gate_proj.weight", "BF16", [1_000_000, 1_000_000])]          # 2 TB of weights: 35 chips at the defaults
SMALL = [("model.layers.0.mlp.gate_proj.weight", "BF16", [4096, 4096])]


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)          # the rig's box is two boards: four chips
    r.args.full_port = True
    yield r
    r.close()


def with_model(rig, tmp_path, tensors, name="model"):
    d = tmp_path / name
    if tensors is not None:
        write_snapshot(d, tensors, shards=1)
    d.mkdir(exist_ok=True)
    rig.args.input.append(f"model={d}")
    return d


def evidence(rig, what):
    return [e["data"] for e in rig.entries() if e["event"] == "evidence" and e["data"].get("what") == what]


def test_stage_0_writes_the_record_and_a_ledger_entry_that_names_the_box(rig, tmp_path):
    with_model(rig, tmp_path, SMALL)
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    rec = json.loads((rig.run_dir / "stages" / "0" / residency.FILE).read_text())
    assert rec["scan"] == "ran" and rec["verdict"] == {"box_chips": 4, "fits_box": True, "chips_needed": 1}
    (entry,) = evidence(rig, "residency estimate")
    assert entry["chips_needed"] == 1 and entry["fits_box"] is True and entry["hazards"] == [] and entry["path"].endswith("residency.json")


def test_a_model_the_box_cannot_hold_blocks_stage_2_with_the_numbers_and_starts_no_prepare_step(rig, tmp_path):
    with_model(rig, tmp_path, HUGE)
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    pauses = [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"]
    assert pauses and pauses[0].startswith("blocked: the model does not fit this box")
    assert "at least 35 chips" in pauses[0] and "this box has 4 chips" in pauses[0] and "choice" in pauses[0]
    assert not started(rig, 2)                                                         # no stage 2 step started: no prepare step
    stage2 = rig.run_dir / "stages" / "2"
    assert not stage2.exists() or not any(stage2.iterdir())
    notice = next(e["data"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked"))
    assert notice["evidence"]["need_chips"] == 35 and notice["evidence"]["box_chips"] == 4
    (entry,) = evidence(rig, "residency estimate")
    assert entry["fits_box"] is False and entry["hazards"] == ["model_does_not_fit_box"]


# Weights that need three chips at the defaults: 6.05e10 values at bfp8 is about 60 GiB, and (60 + 4) / 29 rounds up to 3.
THREE_CHIPS = [("model.layers.0.mlp.gate_proj.weight", "BF16", [246_000, 246_000])]


def test_a_model_the_box_holds_but_stage_2_s_lease_does_not_is_blocked_with_both_numbers(rig, tmp_path):
    with_model(rig, tmp_path, THREE_CHIPS)
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    rec = json.loads((rig.run_dir / "stages" / "0" / residency.FILE).read_text())
    assert rec["estimated"]["chips_needed"] == 3 and rec["verdict"]["fits_box"] is True       # four chips hold it
    pauses = [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"]
    assert pauses and pauses[0].startswith("blocked: the model does not fit stage 2's lease")
    assert "at least 3 chips" in pauses[0] and "stage 2's lease has 2 chips" in pauses[0]
    assert not started(rig, 2)
    notice = next(e["data"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked"))
    assert notice["evidence"]["stage_chips"] == 2 and notice["evidence"]["box_chips"] == 4
    assert blocked.block_code({"reason": pauses[0]}) == "model-does-not-fit"


class _Adapter:
    def __init__(self, boards=None, fail=False):
        self.boards, self.fail = boards, fail

    def status(self):
        if self.fail:
            raise RuntimeError("gozer is unavailable")
        from orchard.adapters import ChipState
        return [ChipState(f"c{b}{i}", "FREE", None, board=f"b{b}", dev_index=2 * b + i) for b in range(self.boards) for i in range(2)]


def _box(lab, test_adapter, adapter):
    import types
    from orchard.supervisor import Supervisor
    return Supervisor._box_chips(types.SimpleNamespace(lab=lab, test_adapter=test_adapter, adapter=adapter))


def test_the_box_is_counted_on_the_machine_the_tests_run_on():
    import types
    coder = _Adapter(boards=2)
    assert _box(None, coder, coder) == 4
    lab = types.SimpleNamespace(chips=None, adapter=_Adapter(boards=1))
    assert _box(lab, lab.adapter, coder) == 2                       # a lab with no stated capacity is counted on the lab, not on the coder's machine
    assert _box(types.SimpleNamespace(chips=8, adapter=_Adapter(boards=1)), None, coder) == 8
    assert _box(lab, _Adapter(fail=True), coder) is None             # a failed query is no claim
    assert _box(None, _Adapter(boards=0), coder) is None


def test_unattended_the_block_has_its_own_code_and_is_not_retried(rig, tmp_path):
    rig.args.unattended = True
    with_model(rig, tmp_path, HUGE)
    make(rig)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    block = [d for d in rig.decisions() if d["decision"] == "blocked"]
    assert [b["code"] for b in block] == ["model-does-not-fit"]
    assert "at least 35 chips" in (rig.run_dir / "BLOCKED.md").read_text()
    assert not started(rig, 2) and "model-does-not-fit" not in blocked.AUTO_RETRY and not blocked.retryable("model-does-not-fit", block[0]["reason"])


def test_the_code_is_classified_before_the_hardware_words_and_has_an_unblock_text():
    why = residency.shortfall({"scan": "ran", "label": "estimate", "estimated": {"chips_needed": 6, "device_bytes": 5 << 30}, "assumptions": []}, 4)
    assert status.classify_pause({"reason": f"blocked: {why}"})[0] == "blocked"
    assert blocked.block_code({"reason": f"blocked: {why}"}) == "model-does-not-fit"
    assert "model-does-not-fit" in blocked.REASONS and narrate.UNBLOCK["model-does-not-fit"]


def test_a_model_that_fits_goes_on_into_stage_2(rig, tmp_path):
    with_model(rig, tmp_path, SMALL)
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    assert started(rig, 2) and not any(str(d.get("reason", "")).startswith("blocked:") for d in rig.decisions())


def test_a_model_directory_without_headers_makes_no_claim(rig, tmp_path):
    with_model(rig, tmp_path, None)
    make(rig)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    rec = json.loads((rig.run_dir / "stages" / "0" / residency.FILE).read_text())
    assert rec["scan"] == "skipped" and started(rig, 2)


def test_only_a_full_port_stage_2_is_gated(tmp_path):
    r = Rig(tmp_path)                      # no --full-port: the generic stage 2 runs and the estimate is a record only
    try:
        with_model(r, tmp_path, HUGE)
        r.on_sleep = lambda: Control(r.run_dir).write("abort")
        r.run()
        assert not any(str(d.get("reason", "")).startswith("blocked: the model does not fit") for d in r.decisions())
        assert started(r, 2)
    finally:
        r.close()


# ---- the first boot against the machine (issue 327) --------------------------------------------------------------------------------

def run_with_readings(rig, tmp_path, monkeypatch, readings, tensors=SMALL):
    with_model(rig, tmp_path, tensors)
    make(rig)
    monkeypatch.setattr(residency, "machine_facts", lambda path: readings)
    rig.on_sleep = lambda: Control(rig.run_dir).write("abort")
    rig.run()
    return [d["reason"] for d in rig.decisions() if d.get("decision") == "pause"]


def test_too_little_free_disk_blocks_stage_2_with_a_disk_code_and_starts_no_prepare_step(rig, tmp_path, monkeypatch):
    pauses = run_with_readings(rig, tmp_path, monkeypatch, (1 << 20, 1 << 50))
    assert pauses and pauses[0].startswith("blocked: the first boot needs about") and "free disk" in pauses[0]
    assert blocked.block_code({"reason": pauses[0]}) == "disk-full"
    assert not started(rig, 2)
    notice = next(e["data"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked"))
    assert notice["evidence"]["free_disk_bytes"] == 1 << 20 and notice["evidence"]["disk_bytes_needed"] > 1 << 20


def test_too_little_host_ram_blocks_stage_2_as_a_model_that_does_not_fit(rig, tmp_path, monkeypatch):
    pauses = run_with_readings(rig, tmp_path, monkeypatch, (1 << 50, 1 << 20))
    assert pauses and pauses[0].startswith("blocked: the model does not fit this box: the first boot's conversion needs about")
    assert "host RAM" in pauses[0] and blocked.block_code({"reason": pauses[0]}) == "model-does-not-fit"
    assert not started(rig, 2)


def test_readings_that_hold_the_first_boot_or_are_unknown_block_nothing(rig, tmp_path, monkeypatch):
    pauses = run_with_readings(rig, tmp_path, monkeypatch, (1 << 50, 1 << 50))
    assert not any("first boot" in p for p in pauses) and started(rig, 2)


def test_unknown_readings_are_no_claim(rig, tmp_path, monkeypatch):
    pauses = run_with_readings(rig, tmp_path, monkeypatch, (None, None))
    assert not any("first boot" in p for p in pauses) and started(rig, 2)


def test_the_chip_shortfall_is_reported_before_the_machine_check(rig, tmp_path, monkeypatch):
    pauses = run_with_readings(rig, tmp_path, monkeypatch, (1, 1), tensors=HUGE)
    assert pauses and pauses[0].startswith("blocked: the model does not fit this box: the estimate needs at least")
    assert "precision" in pauses[0]                      # the options sentence rides along with the numbers


def test_a_lab_run_does_not_judge_the_first_boot_by_this_boxs_disk(tmp_path, monkeypatch):
    """`_check_residency` on a stub supervisor: with a lab the first boot happens on the lab, so this box's readings are not read at all."""
    from types import SimpleNamespace
    from orchard.supervisor import Supervisor
    rec = {"label": "estimate", "scan": "ran", "assumptions": [], "estimated": {"chips_needed": 1, "device_bytes": 1,
           "first_boot": {"snapshot_local": True, "checkpoint_bytes_to_download": 0, "cache_bytes": 1, "disk_bytes": 1 << 40,
                          "conversion_in_flight_bytes": 0, "host_ram_bytes": 1 << 40}}}
    monkeypatch.setattr(residency, "read_record", lambda run_dir: rec)
    reads, blocks = [], []
    monkeypatch.setattr(residency, "machine_facts", lambda path: reads.append(path) or (1, 1))

    def stub(lab):
        return SimpleNamespace(run_dir=tmp_path, lab=lab, paths=SimpleNamespace(cache_root=tmp_path / "cache"), _box_chips=lambda spec: 4,
                               _block=lambda stage, why, **ev: blocks.append(why))
    spec = SimpleNamespace(boards=1, number=2)
    Supervisor._check_residency(stub(object()), spec)
    assert blocks == [] and reads == []                                   # a lab: nothing read, nothing blocked
    Supervisor._check_residency(stub(None), spec)
    assert len(blocks) == 1 and reads == [tmp_path / "cache"] and "free disk" in blocks[0]       # no lab: the cache root's disk is read


# ---- a finish-only resume of stage 2 (review of PR 435, round 1) --------------------------------------------------------------------
# After the hardware test has ended, a resumed stage 2 only reads results: nothing is converted, nothing is downloaded. The converted cache
# the test made may fill the disk the first-boot check would demand again, so the machine check belongs to attempts that run the hardware
# phase (a first attempt, and a retry that must rerun a failed test), and a finish-only resume is not held to it.

def hardware_stub(tmp_path, *, marker: bool, failed: bool = False):
    from types import SimpleNamespace
    d = tmp_path / "stages" / "2"
    d.mkdir(parents=True)
    if marker:
        (d / "test-result.json").write_text("{}")
    return SimpleNamespace(run_dir=tmp_path, _test_failure=lambda spec, stage_dir: ({"problem": "the test failed"} if failed else None))


@pytest.mark.parametrize("resuming,marker,escalated,failed,expect", [
    (False, False, False, False, True),       # a fresh attempt runs the hardware phase
    (False, True, False, False, True),        # a marker left by an earlier attempt: open_stage_dir moves it aside, the test reruns
    (True, False, False, False, True),        # resumed before the test ended
    (True, True, False, False, False),        # resumed after a finished test: only the finish step runs
    (True, True, True, False, False),         # escalated, the marker holds a good result: still only the finish step
    (True, True, True, True, True),           # escalated, the marker holds a failed test: moved aside, the test reruns
    (True, True, False, True, False),         # not escalated, a failed test: the stage ends on it without a rerun
])
def test_the_hardware_phase_is_pending_unless_a_resumed_stage_only_reads_a_finished_test(tmp_path, resuming, marker, escalated, failed, expect):
    from types import SimpleNamespace
    from orchard.supervisor import Supervisor
    stub = hardware_stub(tmp_path, marker=marker, failed=failed)
    spec = SimpleNamespace(number=2, boards=1)
    assert Supervisor._hardware_pending(stub, spec, resuming=resuming, escalated=escalated) is expect


def _record():
    return {"label": "estimate", "scan": "ran", "assumptions": [], "estimated": {"chips_needed": 1, "device_bytes": 1,
            "first_boot": {"snapshot_local": True, "checkpoint_bytes_to_download": 0, "cache_bytes": 1, "disk_bytes": 1 << 40,
                           "conversion_in_flight_bytes": 0, "host_ram_bytes": 1 << 40}}, "verdict": {"fits_box": True}}


def test_a_check_for_a_finish_only_attempt_reads_no_machine_figure_and_blocks_nothing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from orchard.supervisor import Supervisor
    monkeypatch.setattr(residency, "read_record", lambda run_dir: _record())
    reads, blocks = [], []
    monkeypatch.setattr(residency, "machine_facts", lambda path: reads.append(path) or (1, 1))
    stub = SimpleNamespace(run_dir=tmp_path, lab=None, paths=SimpleNamespace(cache_root=tmp_path / "cache"), _box_chips=lambda spec: 4,
                           _block=lambda stage, why, **ev: blocks.append(why))
    spec = SimpleNamespace(boards=1, number=2)
    Supervisor._check_residency(stub, spec, hardware_pending=False)
    assert blocks == [] and reads == []
    Supervisor._check_residency(stub, spec, hardware_pending=True)
    assert len(blocks) == 1 and "free disk" in blocks[0] and reads == [tmp_path / "cache"]
    Supervisor._check_residency(stub, spec)                                                    # the default is the strict one
    assert len(blocks) == 2


def test_a_finish_only_attempt_is_still_held_to_the_chips_the_box_has(tmp_path, monkeypatch):
    """Only the machine check is lifted. The chip count is not about the cache, and the box the run resumes on is read again."""
    from types import SimpleNamespace
    from orchard.supervisor import Supervisor
    rec = _record()
    rec["estimated"]["chips_needed"] = 35
    rec["verdict"] = {"fits_box": False}
    monkeypatch.setattr(residency, "read_record", lambda run_dir: rec)
    blocks = []
    stub = SimpleNamespace(run_dir=tmp_path, lab=None, paths=SimpleNamespace(cache_root=tmp_path / "cache"), _box_chips=lambda spec: 4,
                           _block=lambda stage, why, **ev: blocks.append(why))
    Supervisor._check_residency(stub, SimpleNamespace(boards=1, number=2), hardware_pending=False)
    assert len(blocks) == 1 and "does not fit this box" in blocks[0]


def test_a_resume_after_the_hardware_test_is_not_blocked_by_the_disk_the_cache_now_fills(tmp_path, monkeypatch):
    """The wiring: a real supervisor crashes after stage 2's hardware test ended and before the stage's end was recorded, and resumes on a
    disk that has 1 MiB free. The first-boot check must not block it; the stage ends with a pass. The snapshot is the full-port tests' (a
    stage 2 that passes) with safetensors headers added, so stage 0 writes a residency record for the check to judge."""
    from fakes import Crash
    from test_supervisor_full_port import with_snapshot
    r = Rig(tmp_path)
    try:
        with_snapshot(r, tmp_path)
        r.args.full_port = True
        write_snapshot(tmp_path / "snapshot", SMALL, shards=1)
        make(r)
        monkeypatch.setattr(residency, "machine_facts", lambda path: (1 << 50, 1 << 50))
        agent_steps = []

        def crash_at_the_finish_step(e):          # stage 2's second agent step is the finish step, after the hardware test's evidence entry
            if e["event"] == "decision" and e["stage"] == 2 and e["data"].get("decision") == "agent step":
                agent_steps.append(e)
            return len(agent_steps) == 2
        with pytest.raises(Crash):
            r.run(crash_if=crash_at_the_finish_step)
        assert (r.run_dir / "stages" / "2" / "test-result.json").is_file()
        assert not [e for e in r.entries() if e["event"] == "stage_end" and e["stage"] == 2]
        assert json.loads((r.run_dir / "stages" / "0" / residency.FILE).read_text())["scan"] == "ran"     # there is a record to judge
        monkeypatch.setattr(residency, "machine_facts", lambda path: (1 << 20, 1 << 20))          # the converted cache fills the disk now
        r.on_sleep = lambda: Control(r.run_dir).write("abort")
        r.run(pid=200)
        pauses = [d["reason"] for d in r.decisions() if d.get("decision") == "pause"]
        assert not any("first boot" in p for p in pauses), pauses
        assert [e["data"]["result"] for e in r.entries() if e["event"] == "stage_end" and e["stage"] == 2] == ["pass"]
    finally:
        r.close()
