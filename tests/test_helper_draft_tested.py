# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 596: a helper draft that is placed at the start of a repair round is judged by the port check before the coder's repair step starts, the
ledger says how it fared in one decision, and the furthest port is kept when the helper is on in a coder-driven stage 2 (Olmo-Hybrid-7B run 3:
the helper's draft cost about 1.65 USD, the coder's step then stalled on top of it, and the draft was never run through the check).

The chips, the coder and the helper's endpoint are fakes; the supervisor, the stage machine, the gate and the check template are real (see
tests/test_supervisor_telegraph_in_repair.py)."""
import pytest

from orchard import port_repair, repair_help as rh
from orchard import supervisor as sv
from orchard.supervisor import EXIT_ABORTED
from port_check_fakes import HONEST
from test_supervisor import Rig  # noqa: F401
from test_supervisor_full_port import PREPARE, make
from test_supervisor_port_repair import BROKEN, repairs_by_attempt
from test_supervisor_telegraph_in_repair import PREFILL_BROKEN, PORT, decs, rig, start, stage_ends, wire_helper   # noqa: F401
from test_supervisor_telegraph_in_repair import pytestmark  # noqa: F401
from run_fakes import where  # noqa: F401
from orchard.ledger import Ledger  # noqa: F401

DRAFT_TESTED = getattr(rh, "DRAFT_TESTED", "helper draft tested")
NO_LINEAR = HONEST.replace("ttnn.linear(None, None)", "pass")                 # uploads nothing and calls no matmul-family function


def order(rig, *names):
    """The positions of the first decision of each name, in the ledger's order (None when absent)."""
    found = [d.get("decision") for d in rig.decisions()]
    return [found.index(n) if n in found else None for n in names]


# ---- the draft is judged before the coder's step ------------------------------------------------------------------------------------------------

def test_a_placed_draft_is_tested_before_the_coder_step_starts(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN])
    seen = start(rig, BROKEN, [HONEST + "# the coder\n"])
    assert rig.run() == EXIT_ABORTED
    assert stage_ends(rig) == ["pass"]
    placed, tested = order(rig, rh.PLACED, DRAFT_TESTED)
    assert placed is not None and tested is not None and placed < tested
    round2 = next(i for i, d in enumerate(rig.decisions()) if d.get("decision") == port_repair.DECISION and d.get("round") == 2)
    assert tested < round2                                                            # the draft's check ran before the coder's round began
    note = decs(rig, DRAFT_TESTED)[0]
    assert note["round"] == 1 and note["outcome"] == "failed" and note["op"] == "prefill" and "Sizes of tensors" in note["failure"]
    assert seen and "round 2 of" in seen[0]                                            # the coder's first step starts from the draft's failure


def test_a_draft_that_passes_the_check_ends_the_rounds_and_no_coder_step_runs(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [HONEST])
    seen = start(rig, BROKEN, [HONEST + "# the coder\n"])
    assert rig.run() == EXIT_ABORTED
    assert stage_ends(rig) == ["pass"]
    assert seen == []                                                                 # the coder's repair step never started
    note = decs(rig, DRAFT_TESTED)
    assert len(note) == 1 and note[0]["outcome"] == "passed" and note[0]["round"] == 1
    verdict = decs(rig, rh.PROGRESS)
    assert len(verdict) == 1 and verdict[0]["helped"] is True and verdict[0]["outcome"] == "passed"


def test_a_draft_that_makes_no_projection_call_is_refused_before_a_board_is_leased(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [NO_LINEAR])
    seen = start(rig, BROKEN, [HONEST + "# the coder\n"])
    assert rig.run() == EXIT_ABORTED
    note = decs(rig, DRAFT_TESTED)[0]
    assert note["outcome"] == "failed" and note["op"] == "device-ops" and "matmul-family" in note["failure"]
    pre = decs(rig, "device-ops pre-check failed before the board")
    assert pre, "the CPU pre-check did not judge the draft"
    placed, tested = order(rig, rh.PLACED, DRAFT_TESTED)
    assert placed < tested and seen and "round 2 of" in seen[0]


def test_the_ledger_decision_holds_no_machine_path_and_a_bounded_failure_line(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN])
    start(rig, BROKEN, [HONEST + "# the coder\n"])
    rig.run()
    note = decs(rig, DRAFT_TESTED)[0]
    assert len(note["failure"]) <= 300 and str(tmp_path) not in str(note) and "/home/" not in str(note)


# ---- when the draft cannot be tested --------------------------------------------------------------------------------------------------------------

def test_a_check_that_cannot_run_is_a_not_tested_decision_and_the_coder_goes_on_as_before(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN])
    seen = start(rig, BROKEN, [HONEST + "# the coder\n"])
    real = sv.Supervisor._pin_full_port

    def pin(self, spec, stage_dir, **k):
        if any((e.get("data") or {}).get("decision") == rh.PLACED for e in self.ledger.read()):
            return ["the check's files cannot be put back"]
        return real(self, spec, stage_dir, **k)
    monkeypatch.setattr(sv.Supervisor, "_pin_full_port", pin)
    rig.run()
    note = decs(rig, DRAFT_TESTED)
    assert len(note) == 1 and note[0]["outcome"] == "not tested" and "cannot be put back" in note[0]["reason"]
    assert seen and "has just replaced" in seen[0]                                    # the older flow: the coder's step starts on the placed draft


def test_no_draft_test_runs_when_nothing_was_placed(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN + "\n"], key=None)             # no key: the helper cannot be asked
    start(rig, BROKEN, [HONEST + "# the coder\n"])
    rig.run()
    assert decs(rig, DRAFT_TESTED) == []


# ---- the furthest port is kept when the helper is on -----------------------------------------------------------------------------------------------

def test_a_tested_draft_is_recorded_as_the_best_port_in_a_coder_driven_stage(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN])
    start(rig, BROKEN, [HONEST + "# the coder\n"])
    rig.run()
    best = decs(rig, rh.BEST)
    assert [b["op"] for b in best] == ["load", "prefill"]                                # the port it replaced, then the draft that got further
    assert best[-1]["rank"] > best[0]["rank"]


def test_the_next_attempt_starts_from_the_kept_draft_not_the_skeleton(rig, monkeypatch, tmp_path):
    wire_helper(monkeypatch, tmp_path, [PREFILL_BROKEN])
    make(rig, over={(2, "prepare"): {**PREPARE, PORT: BROKEN}})
    repairs_by_attempt(rig, {})                                                         # the coder's repair steps write nothing: attempt 1 fails
    rig.on_sleep = lambda: sv.Control(rig.run_dir).write("abort")
    rig.run()
    assert decs(rig, rh.BEST), "no best port was kept"
    restored = decs(rig, rh.RESTORED)
    assert restored and restored[0]["op"] == "prefill"


def test_the_round_after_a_failed_draft_test_belongs_to_the_coder_and_the_helper_is_not_asked_again_at_once(rig, monkeypatch, tmp_path):
    tel, wire = wire_helper(monkeypatch, tmp_path, [NO_LINEAR])                       # the failure stays at the same request: it would ask again
    seen = start(rig, BROKEN, [HONEST + "# the coder\n"])
    rig.run()
    asked = decs(rig, rh.ASKED)
    assert [a["round"] for a in asked][:1] == [1] and all(a["round"] != 2 or a["coder_failed"] for a in asked)
    assert seen and "round 2 of" in seen[0]


def test_draft_failed_in_round_reads_only_this_attempts_failed_decisions():
    def e(seq, event, **data):
        return {"seq": seq, "event": event, "stage": 2, "data": data}
    ledger = [e(1, "stage_start"), e(2, "decision", decision=rh.DRAFT_TESTED, round=1, outcome="failed"),
              e(3, "decision", decision=rh.DRAFT_TESTED, round=2, outcome="passed"), e(4, "decision", decision=rh.DRAFT_TESTED, round=3, outcome="not tested")]
    assert rh.draft_failed_in_round(ledger, 2, 1) is True
    assert rh.draft_failed_in_round(ledger, 2, 2) is False and rh.draft_failed_in_round(ledger, 2, 3) is False
    assert rh.draft_failed_in_round(ledger + [e(5, "stage_start")], 2, 1) is False        # a new attempt forgets it
    assert rh.start_trigger(ledger, 2, 2, {"op": "load"})[0] is False
