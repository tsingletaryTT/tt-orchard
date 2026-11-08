# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issues 552, 553 and 554, found in the review of PR 542 (the wide phase's retry after a fabric handshake timeout).

552: the signature scan stopped at the first matching line, so a later line over the size limit was never checked and the test was
     retried although a full scan would have refused the output as too large.
553: every attempt wrote the resume marker (test-result.json), so a retry that stopped before the next attempt finished left a failed
     marker that resume accepted, and the hardware phase was skipped with the retry still owed.
554: the container sweep (the phase's `after_test`) ran only after the whole wide test, so a retry's chip resets happened with a
     leftover container still holding the chips."""
import json
from types import SimpleNamespace

import pytest

from orchard import supervisor
from orchard.adapters import AdapterError, ChipState
from orchard.supervisor import Blocked, Supervisor
from test_supervisor_wide_retry import A, B, FABRIC, Rig


# ---- 552 -------------------------------------------------------------------------------------------------------------------------------

def test_a_signature_line_followed_by_an_oversized_line_is_not_a_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "FABRIC_SCAN_LINE_MAX", 64)
    out = tmp_path / "out.txt"
    out.write_text(FABRIC + "x" * 200 + "\n")
    assert supervisor.fabric_timeout_in(out) is False


def test_the_whole_output_is_read_before_the_signature_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "FABRIC_SCAN_LINE_MAX", 64)
    r = Rig(tmp_path, [(1, False, FABRIC + "y" * 200 + "\n")])
    r.run([A, B])
    assert r.tries == 1 and not r.resets and not r.retries()


def test_a_signature_in_a_clean_output_still_counts(tmp_path):
    out = tmp_path / "out.txt"
    out.write_text("start\n" + FABRIC + "end\n")
    assert supervisor.fabric_timeout_in(out) is True


# ---- 554 -------------------------------------------------------------------------------------------------------------------------------

def test_the_sweep_runs_before_each_retrys_resets(tmp_path):
    r = Rig(tmp_path, [(1, False, FABRIC), (1, False, FABRIC), (0, False, "ok\n")])
    r.s._cleanup_after_test = Supervisor._cleanup_after_test
    result = r.s._run_wide_test(r.spec, r.stage_dir, {}, [A, B], boards=2, after_test=lambda: r.timeline.append(("sweep",)))
    assert result["returncode"] == 0
    assert r.timeline == [("run", 1), ("sweep",), ("reset", "B"), ("reset", "A"),
                          ("run", 2), ("sweep",), ("reset", "B"), ("reset", "A"), ("run", 3)]


def test_a_sweep_that_blocks_stops_the_retry_before_any_reset(tmp_path):
    r = Rig(tmp_path, [(1, False, FABRIC), (0, False, "ok\n")])
    r.s._cleanup_after_test = Supervisor._cleanup_after_test

    def sweep():
        raise Blocked("a container would not stop")

    with pytest.raises(Blocked):
        r.s._run_wide_test(r.spec, r.stage_dir, {}, [A, B], boards=2, after_test=sweep)
    assert r.tries == 1 and not r.resets


def test_the_phase_hands_its_cleanup_to_the_retrying_runner(tmp_path):
    """The wiring: `_wide_hardware_phase` passes `after_test` into `_run_wide_test`, so the retry can sweep."""
    seen = []
    s = object.__new__(Supervisor)
    s.run_dir = tmp_path
    s.ledger = SimpleNamespace(append=lambda *a, **k: {"seq": 1})
    s._release_test_lease = lambda: True
    chips = [ChipState(c, "FREE", None, board=b, dev_index=i) for b, l in (("a", A), ("b", B)) for c, i in zip(l.chips, l.dev_indices)]
    s.adapter = SimpleNamespace(status=lambda: chips)
    s.coder_lease = None
    s._take_test_lease = lambda spec, shape, boards, exact: A
    s._give_back_test_lease = lambda n, lease: None
    s._cleanup_after_test = lambda after: None
    s._run_wide_test = lambda *a, **k: seen.append(k.get("after_test"))
    marker = lambda: None
    spec = SimpleNamespace(number=2, boards=1, skill="x")
    s._wide_hardware_phase(spec, tmp_path, {}, 1, after_test=marker)
    assert seen == [marker]


# ---- 553 -------------------------------------------------------------------------------------------------------------------------------

class RealRunRig:
    """The real `_run_test` and `_run_wide_test` with a fake spawn: the marker file is what the test looks at."""

    def __init__(self, tmp_path, outputs, hook=None):
        self.tmp, self.outputs, self.tries, self.hook = tmp_path, list(outputs), 0, hook
        self.events, self.resets = [], []
        s = object.__new__(Supervisor)
        s.run_dir = tmp_path
        s.ledger = SimpleNamespace(append=self._append, read=lambda: [])
        s.test_adapter = SimpleNamespace(reset=lambda lease: self.resets.append(lease.lease_id))
        s._spawn = self._spawn
        s._helper_drive_on = lambda: False
        s.extra_env = {}
        self.s = s
        self.spec = SimpleNamespace(number=3, boards=1, skill="x", budget_s=100.0)
        self.stage_dir = tmp_path / "stages" / "3"
        (self.stage_dir / "evidence").mkdir(parents=True)
        self.marker = self.stage_dir / "test-result.json"

    def _append(self, event, stage, **data):
        self.events.append((event, stage, data))
        return {"seq": len(self.events)}

    def _spawn(self, n, command, chips, ids, deadline, out_path, env_extra=None, **record):
        if self.hook:
            self.hook(self.tries)
        code, text = self.outputs[min(self.tries, len(self.outputs) - 1)]
        self.tries += 1
        out_path.write_text(text)
        return {"returncode": code, "timed_out": False, "seconds": 1.0, "chips": [], "command": command, "output": {}}

    def run(self, leases):
        return self.s._run_wide_test(self.spec, self.stage_dir, {"command": "t", "deadline_s": 60}, leases, boards=2)

    def evidence(self):
        return [d for e, _, d in self.events if e == "evidence" and d.get("what") == "hardware test"]


def test_no_resume_marker_exists_while_a_retry_is_still_owed(tmp_path):
    seen = []
    r = RealRunRig(tmp_path, [(1, FABRIC), (0, "ok\n")], hook=lambda tries: seen.append(r.marker.exists()))
    result = r.run([A, B])
    assert result["returncode"] == 0
    assert seen == [False, False], "the second attempt started with the first attempt's failure published as the resume marker"
    assert json.loads(r.marker.read_text())["returncode"] == 0


def test_a_reset_that_blocks_between_attempts_leaves_no_marker(tmp_path):
    r = RealRunRig(tmp_path, [(1, FABRIC), (0, "ok\n")])
    r.s.test_adapter = SimpleNamespace(reset=lambda lease: (_ for _ in ()).throw(AdapterError("refused")))
    r.s._block = lambda n, msg, **k: (_ for _ in ()).throw(RuntimeError(msg))
    with pytest.raises(RuntimeError):
        r.run([A, B])
    assert not r.marker.exists(), "resume would take the intermediate failure as the test's result and skip the hardware phase"


def test_the_final_failure_is_published_once_with_one_evidence_entry(tmp_path):
    r = RealRunRig(tmp_path, [(1, FABRIC)])
    result = r.run([A, B])
    assert result["returncode"] == 1 and r.tries == 3
    assert json.loads(r.marker.read_text())["returncode"] == 1
    assert len(r.evidence()) == 1


def test_a_test_that_passes_first_time_is_published_as_before(tmp_path):
    r = RealRunRig(tmp_path, [(0, "ok\n")])
    r.run([A, B])
    assert json.loads(r.marker.read_text())["returncode"] == 0 and len(r.evidence()) == 1 and r.tries == 1
