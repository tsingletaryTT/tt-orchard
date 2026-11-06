# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 4 of the full-port route: findings of the second Copilot review of PR 489 ("a critical symlink-based write outside the run
directory and a crash-recovery flaw that can prevent a fresh retry"). The rig is the one of tests/test_supervisor_full_port_stage4.py."""
import json

import pytest

from fakes import Crash
from orchard.supervisor import EXIT_ABORTED
from test_supervisor_full_port_stage4 import abort_at_pause, entry, make4, rig, stage4, stage_4_tests  # noqa: F401  (rig is a fixture)
from test_supervisor_full_port import started, with_snapshot
from test_supervisor import Rig
from port_check_fakes import HONEST

pytestmark = pytest.mark.usefixtures("stub_tools")


def port_that_swaps_its_entry_for_a_link(out):
    """The honest port, plus code that runs when the test loads it: it moves its own entry directory aside and puts a link to `out` in its
    place. A port is the coder's code and runs inside the supervisor's test, so this is what a hostile one can do."""
    return HONEST.replace("def load(model_dir, mesh_device):\n",
        "def load(model_dir, mesh_device):\n"
        "    import os, pathlib\n"
        "    edir = pathlib.Path(__file__).resolve().parents[1]\n"
        f"    if not os.path.islink(edir) and edir.name == '1' and not os.path.exists({str(out / 'swapped')!r}):\n"
        f"        open({str(out / 'swapped')!r}, 'w').close()\n"
        "        os.rename(edir, edir.with_name('1.moved'))\n"
        f"        os.symlink({str(out)!r}, edir)\n", 1)


def test_an_entry_directory_swapped_for_a_link_by_the_port_during_its_test_takes_no_supervisor_file_outside(rig, tmp_path):
    out = tmp_path / "outside"
    out.mkdir()
    make4(rig, ports={1: port_that_swaps_its_entry_for_a_link(out)})
    abort_at_pause(rig)
    rig.run()
    names = sorted(p.name for p in out.rglob("*"))
    assert (out / "swapped").exists(), "the port never ran its swap, so this test checked nothing"
    assert "test-result.json" not in names, f"the supervisor wrote its marker through the link: {names}"
    assert "output-first-try.txt" not in names
    # The refusal must not leave the test lease held or the coder parked: every lease taken for a test is given back.
    ledger = rig.entries()
    end = next(i for i, e in enumerate(ledger) if e["event"] == "stage_end" and e["stage"] == 4)
    assert any(e["event"] == "decision" and e["stage"] == 4 and e["data"].get("decision") == "test lease released" for e in ledger[:end]), \
        "the entry's test ended in a refusal and its lease was not given back before the stage ended"


def test_the_plan_and_the_stage_result_are_written_through_the_run_directory_and_a_link_in_tests_is_refused(tmp_path):
    """`stages/4/tests` is the coder's to replace. The plan and the stage's result.json are written through a descriptor of the run directory,
    so a link there is refused and the file never lands in its target."""
    import types
    from orchard import full_port_entries, full_port_stage4, safeio
    run = tmp_path / "run"
    (run / "stages" / "4").mkdir(parents=True)
    out = tmp_path / "outside"
    out.mkdir()
    (run / "stages" / "4" / "tests").symlink_to(out, target_is_directory=True)
    sup = types.SimpleNamespace(run_dir=run)
    entries, problems = full_port_entries.plan_entries((1,), max_chips=4, budget_s=10 ** 6)
    assert not problems
    with pytest.raises(safeio.SafeReadError):
        full_port_stage4._write_json_pinned(sup, run / "stages" / "4" / full_port_entries.PLAN, full_port_entries.plan_data(entries))
    assert list(out.iterdir()) == [], "the plan was written through the link"
    # A plain directory works, and the file is the plan.
    (run / "stages" / "4" / "tests").unlink()
    path = full_port_stage4._write_json_pinned(sup, run / "stages" / "4" / full_port_entries.PLAN, full_port_entries.plan_data(entries))
    assert json.loads(path.read_text())["tests"][0]["chips"] == 1


PLAN_WRITTEN = lambda e: e["event"] == "evidence" and e["stage"] == 4 and e["data"].get("what") == "hardware test list"


def test_a_kill_between_the_plan_file_and_its_ledger_entry_is_recovered_by_writing_the_plan_again(tmp_path):
    """The plan is the supervisor's file. A kill after the file and before its ledger entry leaves a plan nothing vouches for; the resume
    must write it again and say so, or the stage can never pass and the retry starts from the same unvouched plan."""
    from orchard.ledger import Ledger
    r = Rig(tmp_path)
    with_snapshot(r, tmp_path)
    r.args.full_port = True
    r.args.required_chips = (1, 3, 4)
    try:
        make4(r)
        abort_at_pause(r)
        real = Ledger.append

        def dying(self, event, stage=None, **data):
            if event == "evidence" and stage == 4 and data.get("what") == "hardware test list" and not getattr(dying, "fired", False):
                dying.fired = True
                raise Crash("killed between the plan file and its ledger entry")
            return real(self, event, stage, **data)
        Ledger.append = dying
        try:
            with pytest.raises(Crash):
                r.run()
        finally:
            Ledger.append = real
        assert (stage4(r) / "tests" / "plan.json").is_file()
        assert r.run(pid=200) == EXIT_ABORTED
        assert [d["result"] for d in r.ends(4)] == ["pass"], "the resume must vouch for the plan it found, or write it again"
    finally:
        r.close()
