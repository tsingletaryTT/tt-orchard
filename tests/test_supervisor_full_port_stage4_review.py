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

def abort_in_prepare(rig):
    inner = rig.script

    def script(request):
        if "tools" in request and where(request) == (4, "prepare"):
            Control(rig.run_dir).write("abort")
        return inner(request)
    rig.script = script


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
