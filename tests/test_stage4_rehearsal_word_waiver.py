# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 570: regression tests for the word waiver in stage 4 of a full port (PR 569 added the waiver; its stage 4 tests covered only the
coherence waiver). In a rehearsal the free-run text of a random-weight model has too few different words. Stage 4 has one entry per required
chip count and each entry's check may waive the word count. These tests pin three things: the waived fields from several entries are
listed once, the supervisor writes exactly one word-waiver decision for the stage, and a stage that fails or is not vouched for by the
ledger writes none. The chips, the coder and the model are fakes; the supervisor, the gate and the check are real."""
import pytest

from orchard import full_port, full_port_entries, full_port_stage4, supervisor
from orchard.supervisor import EXIT_ABORTED, EXIT_BLOCKED
from port_check_fakes import WRONG
from test_supervisor_full_port import never_wait
from test_supervisor_full_port_stage4 import abort_at_pause, make4, rig, stage4   # noqa: F401  (rig is a fixture)

pytestmark = pytest.mark.usefixtures("stub_tools", "unbuilt_stages_5_6")


def word_decisions(rig, stage: int):
    return [e for e in rig.entries() if e["event"] == "decision" and e["stage"] == stage
            and e["data"].get("decision") == supervisor.REHEARSAL_WORDS_WAIVED]


def too_few_words(monkeypatch):
    """The fake chip's text has few different words only when the floor asks for more than it has (as in tests/test_supervisor_full_port.py)."""
    monkeypatch.setattr(full_port, "MIN_FREE_RUN_WORDS", 10 ** 6)


def test_low_word_text_in_several_entries_is_waived_once_and_the_ledger_says_so_once(rig, monkeypatch):
    too_few_words(monkeypatch)
    rig.args.full_port_rehearsal = True
    make4(rig)
    abort_at_pause(rig)
    assert rig.run() == EXIT_ABORTED
    assert [d["result"] for d in rig.ends(4)] == ["pass"]
    # Two entries (1 and 3 chips) ran a test and both had too few words; the third count is covered by stage 3.
    assert len(full_port_entries.load_plan(stage4(rig))[0]) == 3
    gate_view = full_port_entries.check_stage(stage4(rig), rig.run_dir, required=[1, 3, 4], rehearsal=True)
    reasons, _seen, waived = gate_view
    assert reasons == [] and waived.count(full_port.WORDS_WAIVED) == 1, waived
    decisions = word_decisions(rig, 4)
    assert len(decisions) == 1, [d["data"] for d in decisions]
    d = decisions[0]["data"]
    assert d["field"] == full_port.WORDS_WAIVED and d["by"] == "--full-port-rehearsal" and "every other stage 4 check still applied" in d["note"]


def test_a_failing_stage_4_records_no_word_waiver(rig, monkeypatch):
    """The 3-chip entry gets a port that gets every token wrong, so the stage fails on the recomputed agreement. The entry that passed still had
    low-word text, but a failed stage waived nothing."""
    too_few_words(monkeypatch)
    rig.args.full_port_rehearsal = True
    rig.args.unattended = True
    make4(rig, ports={3: WRONG})
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    results = [d["result"] for d in rig.ends(4)]
    assert results and "pass" not in results, results
    assert word_decisions(rig, 4) == []


def test_a_stage_the_ledger_does_not_vouch_for_records_no_word_waiver(rig, monkeypatch):
    """The gate passed (the text is waived) but one file is not the one the supervisor recorded: the stage fails and no decision is written."""
    too_few_words(monkeypatch)
    rig.args.full_port_rehearsal = True
    rig.args.unattended = True
    monkeypatch.setattr(full_port_stage4, "_vouched", lambda *a, **k: a[-1].append("planted: this file is not the one the supervisor recorded"))
    make4(rig)
    never_wait(rig)
    assert rig.run() == EXIT_BLOCKED
    results = [d["result"] for d in rig.ends(4)]
    assert results and "pass" not in results, results
    assert word_decisions(rig, 4) == []


class _Ledger:
    """Records what `vouch` appends. `read` is never called because the stubbed `_vouched` does not look at the ledger."""

    def __init__(self):
        self.appended = []

    def append(self, *args, **kwargs):
        self.appended.append((args, kwargs))

    def read(self):
        return []


def _vouch(tmp_path, monkeypatch, gate):
    from types import SimpleNamespace
    from orchard.stages import GateResult  # noqa: F401  (the gate is passed in)
    monkeypatch.setattr(full_port_stage4, "_vouched", lambda *a, **k: None)         # every file counts as vouched: only the waiver rule is under test
    ledger = _Ledger()
    sup = SimpleNamespace(run_dir=tmp_path, ledger=ledger, required_chips=())
    out = full_port_stage4.vouch(sup, SimpleNamespace(number=4), tmp_path, gate, waived_decision="coherence", words_decision="words")
    return out, ledger


def test_vouch_writes_the_word_decision_only_for_a_passing_gate(tmp_path, monkeypatch):
    """The unit under the supervisor tests: a gate that failed for some other reason leaves no decision even when the words were waived."""
    from orchard.stages import GateResult
    _out, ledger = _vouch(tmp_path, monkeypatch, GateResult(True, (), (), (full_port.WORDS_WAIVED,)))
    assert [kw["decision"] for _a, kw in ledger.appended] == ["words"]
    _out, ledger = _vouch(tmp_path, monkeypatch, GateResult(False, ("another check failed",), (), (full_port.WORDS_WAIVED,)))
    assert ledger.appended == []
