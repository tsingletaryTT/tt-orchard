# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A first attempt that reads and writes nothing gets one hint that names the file to write (issue 427).

Evidence: three tiny Qwen4 rehearsal runs, first attempts of stage 2 read for 12 to 26 turns in seconds, wrote no port, and were stopped by
the 20-turn nudge and an escalation. The guards here sit at the layers that fail: the detector, the ladder that delivers it (it must
use no rung), the supervisor's wiring (stage 2 full port only) and the hint text (it must name the file while the port is the skeleton)."""
import pytest

from orchard import port_facts, safeio
from orchard.defaults import FIRST_WRITE_TURNS
from orchard.ledger import Ledger
from orchard.watchdog import Event, FirstWriteDeadline, Ladder, NoFileWritten, Watchdog, replay


def resp(turn, agent="a"):
    return Event(ts=float(turn), agent=agent, kind="response", had_tool_call=True, turn=turn)


def wrote(turn, agent="a"):
    return Event(ts=turn + 0.5, agent=agent, kind="tool_result", tool="write_file", wrote=True, turn=turn)


def read(turn, agent="a"):
    return Event(ts=turn + 0.5, agent=agent, kind="tool_result", tool="read_file", wrote=None, turn=turn)


def stream(n, **kw):
    out = []
    for t in range(1, n + 1):
        out += [resp(t, **kw), read(t, **kw)]
    return out


def det(turns=FIRST_WRITE_TURNS):
    return FirstWriteDeadline(lambda n: f"HINT after {n}", turns)


def test_fires_once_at_the_deadline_with_the_supervisors_text():
    found = replay(stream(30), [det()])
    assert len(found) == 1
    f = found[0]
    assert f.ts == float(FIRST_WRITE_TURNS) and f.nudge == f"HINT after {FIRST_WRITE_TURNS}"
    assert f.nudge_only and f.detector == "first_write_deadline"


def test_does_not_fire_before_the_deadline():
    assert replay(stream(FIRST_WRITE_TURNS - 1), [det()]) == []


def test_a_write_resets_the_count():
    n = FIRST_WRITE_TURNS - 1
    evs = stream(n - 1) + [resp(n), wrote(n)] + [e for t in range(n + 1, n + FIRST_WRITE_TURNS) for e in (resp(t), read(t))]
    assert replay(evs, [det()]) == []


def test_a_refused_write_is_not_a_write():
    refused = Event(ts=1.0, agent="a", kind="tool_result", tool="write_file", wrote=False, turn=1)
    evs = [e for t in range(1, FIRST_WRITE_TURNS + 1) for e in (resp(t), refused)]
    assert len(replay(evs, [det()])) == 1


def test_the_first_response_of_a_new_step_starts_the_count_again():
    evs = stream(FIRST_WRITE_TURNS) + stream(FIRST_WRITE_TURNS - 1)    # the second step's turn 1 resets the once-per-step flag
    assert len(replay(evs, [det()])) == 1
    assert len(replay(stream(FIRST_WRITE_TURNS) + stream(FIRST_WRITE_TURNS), [det()])) == 2


def test_agents_are_counted_separately():
    evs = [e for t in range(1, FIRST_WRITE_TURNS) for e in (resp(t, "a"), resp(t, "b"))]
    assert replay(evs, [det()]) == []


def test_arguments_are_checked():
    with pytest.raises(ValueError):
        FirstWriteDeadline(lambda n: "x", 0)
    with pytest.raises(TypeError):
        FirstWriteDeadline("text")


class Act:
    def __init__(self):
        self.calls = []

    def nudge(self, agent, message):
        self.calls.append(("nudge", message))

    def escalate(self, agent, stage):
        self.calls.append(("escalate", stage))

    def pause(self, reason, evidence):
        self.calls.append(("pause", reason))


def test_through_the_ladder_the_hint_uses_no_rung_and_is_counted_in_the_ledger(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    act = Act()
    wd = Watchdog([det(), NoFileWritten(turns=FIRST_WRITE_TURNS + 8)], Ladder(act, led, {"a"}))
    for e in stream(FIRST_WRITE_TURNS + 3):
        wd.feed(e)
    assert act.calls == [("nudge", f"HINT after {FIRST_WRITE_TURNS}")]
    hints = [e for e in led.read() if e["data"].get("rung") == "hint"]
    assert [h["data"]["findings"][0]["detector"] for h in hints] == ["first_write_deadline"]
    # The 20-turn nudge rung is still whole afterwards: the hint took none of it.
    for e in stream(FIRST_WRITE_TURNS + 8)[2 * (FIRST_WRITE_TURNS + 3):]:
        wd.feed(e)
    assert [c[0] for c in act.calls] == ["nudge", "nudge"]
    assert [e["data"].get("rung") for e in led.read() if e["data"].get("watchdog")] == ["hint", "nudge"]


# ---- the hint text and the supervisor's wiring -----------------------------------------------------

def test_the_hint_names_the_file_and_its_functions_while_the_port_is_the_skeleton():
    t = port_facts.write_deadline_hint(12, port_is_skeleton=True, excerpts=True)
    for needle in ("stages/2/port/port_model.py", "load", "prefill", "decode_step", "close", "reference_excerpts.md", "port_facts.json", "12"):
        assert needle in t
    assert "reference_excerpts.md" not in port_facts.write_deadline_hint(12, port_is_skeleton=True, excerpts=False)


def test_the_hint_after_an_edit_does_not_claim_the_port_is_the_skeleton():
    t = port_facts.write_deadline_hint(12, port_is_skeleton=False, excerpts=True)
    assert "skeleton" not in t and "NotImplementedError" not in t


def test_the_task_deadline_and_the_hint_deadline_are_one_number():
    assert port_facts.FIRST_VERSION_TURNS == FIRST_WRITE_TURNS


class Spec:
    def __init__(self, number, draft):
        self.number, self.draft, self.budget_s = number, draft, 100.0


def make_supervisor(tmp_path):
    from orchard.supervisor import Supervisor
    sup = Supervisor.__new__(Supervisor)           # the wiring under test touches only the ledger, the actuator and the run directory
    sup.run_dir = tmp_path
    sup.ledger = Ledger(tmp_path / "ledger.jsonl")
    sup.actuator = Act()
    sup.clock = lambda: 0.0
    return sup


@pytest.mark.parametrize("number,draft,expected", [(2, "full_port", True), (2, None, False), (3, "full_port", False), (0, None, False)])
def test_only_the_full_port_stage_2_watchdog_has_the_deadline(tmp_path, number, draft, expected):
    sup = make_supervisor(tmp_path)
    wd = sup._watchdog(Spec(number, draft))
    assert any(isinstance(d, FirstWriteDeadline) for d in wd.detectors) is expected


def test_the_supervisors_hint_follows_the_port_file(tmp_path):
    sup = make_supervisor(tmp_path)
    port = tmp_path / "stages" / "2" / "port"
    port.mkdir(parents=True)
    skeleton = safeio.read_bytes(port_facts.SKELETON, max_bytes=port_facts.SKELETON_MAX)
    (port / "port_model.py").write_bytes(skeleton)
    assert "still the skeleton" in sup._write_deadline_hint(12)
    (port / "port_model.py").write_bytes(skeleton + b"\n# edited\n")
    assert "skeleton" not in sup._write_deadline_hint(12)
    (port / "port_model.py").unlink()
    assert "skeleton" not in sup._write_deadline_hint(12)         # a missing file is not claimed to be the skeleton
