# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4 of the full-port route: the findings of the Copilot review of PR 489. Each test names the finding. The rig is the one of
tests/test_supervisor_full_port_stage4.py (fake chips, fake coder, the real supervisor)."""
import json

import pytest

from fakes import Crash
from orchard import full_port
from orchard.supervisor import EXIT_ABORTED, Control
from run_fakes import where
from test_supervisor_full_port_stage4 import abort_at_pause, entry, make4, rig, stage4, stage_4_tests  # noqa: F401  (rig is a fixture)
from test_supervisor_full_port import started, with_snapshot
from test_supervisor import Rig

pytestmark = pytest.mark.usefixtures("stub_tools")

PLAN_WRITTEN = lambda e: e["event"] == "evidence" and e["stage"] == 4 and e["data"].get("what") == "hardware test list"


def crash_after_plan(tmp_path, *, required=(1, 3, 4)):
    """A rig whose first run stops right after stage 4 wrote its plan, so the entry directories can be changed before the resume."""
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = required
    make4(r)
    abort_at_pause(r)
    with pytest.raises(Crash):
        r.run(crash_if=PLAN_WRITTEN)
    return r


def stage_end_results(r, n=4):
    return [d["result"] for d in r.ends(n)]


def refused(r) -> bool:
    """The first end of stage 4 names the refused path: the stage did not go on through the link."""
    ends = r.ends(4)
    return bool(ends) and ends[0]["result"] in ("escalate", "fail") and any("was refused" in x for x in ends[0].get("reasons", []))


# ---- a link in the entry's path (finding: the cleanup followed a symlinked configs/<N> or evidence directory) ----------------

def outside_decoys(tmp_path):
    """A directory outside the run holding a file under each name the cleanup removes, and its victim files' bytes."""
    out = tmp_path / "outside"
    (out / "evidence").mkdir(parents=True)
    names = ["covered.json", "test-result.json", "result.json", "evidence/port-check.json", "evidence/hw-test-output.txt"]
    for n in names:
        (out / n).write_text("precious\n")
    return out, {n: "precious\n" for n in names}


def untouched(out, want):
    return {n: (out / n).read_text() if (out / n).is_file() else None for n in want} == want


def test_a_symlinked_entry_directory_is_refused_and_nothing_outside_the_run_is_removed(tmp_path):
    r = crash_after_plan(tmp_path)
    try:
        out, want = outside_decoys(tmp_path)
        configs = stage4(r) / "configs"
        configs.mkdir(parents=True, exist_ok=True)
        (configs / "1").symlink_to(out, target_is_directory=True)
        r.run(pid=200)
        assert untouched(out, want), "the cleanup followed the link and removed or wrote a file outside the run"
        assert refused(r), "the first attempt must end with the refusal as its reason"
    finally:
        r.close()


def test_a_symlinked_configs_directory_is_refused_and_nothing_outside_the_run_is_removed(tmp_path):
    r = crash_after_plan(tmp_path)
    try:
        out, want = outside_decoys(tmp_path)
        (out / "1").mkdir()
        for n in want:
            (out / "1" / n).parent.mkdir(parents=True, exist_ok=True)
            (out / "1" / n).write_text("precious\n")
        configs = stage4(r) / "configs"
        if configs.is_dir() and not configs.is_symlink():
            configs.rmdir() if not any(configs.iterdir()) else None
        configs.symlink_to(out, target_is_directory=True)
        r.run(pid=200)
        assert untouched(out / "1", want)
        assert refused(r)
    finally:
        r.close()


def test_a_symlinked_evidence_directory_inside_an_entry_is_refused_and_its_target_is_kept(tmp_path):
    r = crash_after_plan(tmp_path)
    try:
        out, want = outside_decoys(tmp_path)
        edir = entry(r, 1)
        edir.mkdir(parents=True, exist_ok=True)
        (edir / "evidence").symlink_to(out / "evidence", target_is_directory=True)
        r.run(pid=200)
        assert untouched(out, {k: v for k, v in want.items() if k.startswith("evidence/")})
        assert refused(r)
    finally:
        r.close()


# ---- an abort or a pause is not a failure (finding: _prepare turned every status into a failed stage) -------------------------

def command_in_prepare(rig, word):
    """The operator writes `word` to the control file while stage 4's first prepare step is running."""
    inner = rig.script
    sent = []

    def script(request):
        if "tools" in request and where(request) == (4, "prepare") and not sent:
            sent.append(word)
            Control(rig.run_dir).write(word)
        return inner(request)
    rig.script = script


def abort_in_prepare(rig):
    command_in_prepare(rig, "abort")


def test_an_abort_during_an_entrys_prepare_step_ends_the_run_aborted_and_the_stage_does_not_fail_or_escalate(rig):
    make4(rig)
    abort_in_prepare(rig)
    assert rig.run() == EXIT_ABORTED
    ends = [e for e in rig.entries() if e["stage"] == 4 and e["event"] in ("stage_end", "escalate")]
    assert [(e["event"], (e["data"] or {}).get("result")) for e in ends] == [], f"an abort became a stage end: {ends}"


# ---- the deadline (finding: the entry ran under the plan's deadline and ignored the shorter one in hw_test.json) -------------

def test_an_entry_runs_under_the_shorter_deadline_of_its_hw_test_json(rig):
    make4(rig)                                  # the scripted coder writes deadline_s 60; the plan says 3600
    abort_at_pause(rig)
    rig.run()
    ran = stage_4_tests(rig)
    assert ran and [r["deadline_s"] for r in ran] == [60] * len(ran), [r["deadline_s"] for r in ran]


def test_an_operator_pause_during_an_entrys_prepare_step_pauses_the_run_and_the_stage_does_not_fail_or_escalate(rig):
    make4(rig)
    command_in_prepare(rig, "pause")
    abort_at_pause(rig)                         # the run waits for the operator; the test ends it there
    rig.run()
    pauses = [e for e in rig.entries() if e["stage"] == 4 and e["event"] == "decision" and e["data"].get("decision") == "pause"]
    assert [p["data"].get("reason") for p in pauses][:1] == ["operator"], pauses
    ends = [e for e in rig.entries() if e["stage"] == 4 and e["event"] in ("stage_end", "escalate")]
    assert ends == [], f"a pause became a stage end: {ends}"


# ---- the stage marker, the set-aside and prepared.json (second round of the review of PR 489) ----------------------------------

class _Ledger:
    """Just enough of the ledger for the helpers under test: it keeps what is appended."""

    def __init__(self):
        self.rows = []

    def append(self, event, stage, **data):
        self.rows.append((event, stage, data))


class _Sup:
    def __init__(self, run_dir):
        self.run_dir, self.ledger = run_dir, _Ledger()


def test_the_stage_marker_is_not_written_through_a_linked_stages_directory(tmp_path):
    # `_fail` and the `_Ended` handler run outside the stage's pin. The marker used to go through the plain path, so a link standing in
    # `stages/` sent it outside the run. A refusal is written to the ledger and does not raise: the stage keeps its own status.
    from orchard import full_port_stage4
    from orchard.full_port_entries import PortEntry
    run, out = tmp_path / "run", tmp_path / "outside"
    (out / "4").mkdir(parents=True)
    run.mkdir()
    (run / "stages").symlink_to(out, target_is_directory=True)
    sup = _Sup(run)
    full_port_stage4._write_stage_marker(sup, run / "stages" / "4", [])
    assert not (out / "4" / "test-result.json").exists(), "the marker followed the link out of the run"
    assert [r[2].get("decision") for r in sup.ledger.rows] == ["the stage's test-result.json was not written"]


def test_the_stage_marker_is_written_in_a_plain_run(tmp_path):
    from orchard import full_port_stage4
    run = tmp_path / "run"
    (run / "stages" / "4").mkdir(parents=True)
    sup = _Sup(run)
    full_port_stage4._write_stage_marker(sup, run / "stages" / "4", [])
    assert json.loads((run / "stages" / "4" / "test-result.json").read_text()) == {"entries": []}
    assert sup.ledger.rows == []


def test_the_set_aside_does_not_follow_a_linked_evidence_directory(tmp_path):
    from orchard import full_port_stage4, safeio
    run, out = tmp_path / "run", tmp_path / "outside"
    edir = run / "stages" / "4" / "configs" / "1"
    edir.mkdir(parents=True)
    out.mkdir()
    (out / "swap-check.json").write_text("precious\n")
    (edir / "evidence").symlink_to(out, target_is_directory=True)
    (edir / "result.json").write_text("{}")
    sup = _Sup(run)
    with safeio.pinned_root(run) as pin:
        with pytest.raises(safeio.SafeReadError):
            full_port_stage4._set_aside_results(sup, pin, edir)
    assert (out / "swap-check.json").read_text() == "precious\n"
    assert not list(out.glob("*.rejected-from-prepare")), "the rename went through the link"


def test_the_set_aside_moves_the_result_files_a_coder_left(tmp_path):
    from orchard import full_port_stage4, safeio
    run = tmp_path / "run"
    edir = run / "stages" / "4" / "configs" / "1"
    (edir / "evidence").mkdir(parents=True)
    (edir / "result.json").write_text("{}")
    (edir / "evidence" / "hw-test-output.txt").write_text("x")
    sup = _Sup(run)
    with safeio.pinned_root(run) as pin:
        moved = full_port_stage4._set_aside_results(sup, pin, edir)
    assert set(moved) == {"result.json", "evidence/hw-test-output.txt"}
    assert (edir / "result.json.rejected-from-prepare").is_file() and not (edir / "result.json").exists()


def test_a_prepared_json_the_coder_wrote_does_not_skip_the_prepare_step(tmp_path):
    # Only the supervisor's own ledger record lets a resume skip the prepare step; a coder could otherwise write the file itself.
    r = crash_after_plan(tmp_path)
    try:
        edir = entry(r, 1)
        edir.mkdir(parents=True, exist_ok=True)
        (edir / "prepared.json").write_text(json.dumps({"chips": 1, "seed_stage": 3}))
        r.run(pid=200)
        assert 1 in [c for c, _ in r.prepare_4_tasks], "the forged prepared.json skipped entry 1's prepare step"
        recorded = [e for e in r.entries() if e["stage"] == 4 and e["data"].get("what") == "entry prepare step done"]
        assert recorded, "the supervisor must record the prepared.json it writes"
    finally:
        r.close()


def test_an_evidence_directory_swapped_for_a_link_during_prepare_is_refused_by_the_set_aside(tmp_path):
    # The set-aside runs after the coder's prepare step. A coder that swaps `evidence/` for a link to a directory outside the run must not get
    # the supervisor to rename files there. This goes through the stage, so it fails when the stage stops using the anchored rename.
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (1, 3, 4)
    try:
        make4(r)
        abort_at_pause(r)
        out = tmp_path / "outside"
        out.mkdir()
        (out / "swap-check.json").write_text("precious\n")
        inner, done = r.script, []

        def script(request):
            reply = inner(request)
            if "tools" in request and where(request) == (4, "prepare") and not done:
                done.append(1)
                import shutil
                shutil.rmtree(entry(r, 1) / "evidence", ignore_errors=True)
                (entry(r, 1) / "evidence").symlink_to(out, target_is_directory=True)
            return reply
        r.script = script
        r.run(pid=200)
        assert done, "the swap never happened"
        assert (out / "swap-check.json").read_text() == "precious\n"
        assert not list(out.glob("*.rejected-from-prepare")), "the supervisor renamed a file outside the run"
    finally:
        r.close()


def test_a_prepared_json_changed_after_the_supervisor_wrote_it_does_not_skip_the_prepare_step_on_resume(tmp_path):
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (1, 3, 4)
    try:
        make4(r)
        abort_at_pause(r)
        with pytest.raises(Crash):
            r.run(crash_if=lambda e: e["event"] == "decision" and e["stage"] == 4 and e["data"].get("decision") == "port cache declaration")
        before = [c for c, _ in r.prepare_4_tasks].count(1)
        assert before, "entry 1 was prepared before the crash"
        (entry(r, 1) / "prepared.json").write_text(json.dumps({"chips": 1, "seed_stage": 3, "by": "the coder"}))
        assert r.run(pid=200) == EXIT_ABORTED
        assert [c for c, _ in r.prepare_4_tasks].count(1) > before, "the changed prepared.json skipped entry 1's prepare step on the resume"
    finally:
        r.close()
