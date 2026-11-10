# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 701: an agent-stuck block leaves the coder up, and the automatic retry uses it instead of booting another (local mode, no lab).

The first run ends blocked with the code `agent-stuck` (here a full-port pause whose block code is replaced, so the test needs no watchdog
script). With `--coder-keep-on-stall` the supervisor hands the coder's chips to a lease its container owns and writes the record
(orchard/kept_coder.py). The retry is a new supervisor process: it checks the coder with the canary and reuses it, and restarts it only
when the check fails. Any other block code, and a run without the flag, stop the coder as before."""
import pytest

from orchard import kept_coder, supervisor
from orchard.supervisor import EXIT_BLOCKED
from run_fakes import KEPT_PID, MachineAdapter, bringup
from test_supervisor import Rig, Stop

pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    r.args.unattended = True
    r.script = bringup
    r.calls = []

    class Recording(MachineAdapter):
        def release(self, lease, keep_running=False):
            r.calls.append(("release-keep" if keep_running else "release", lease.lease_id))
            return super().release(lease, keep_running=keep_running)
    r.adapter_cls = Recording
    sleeps = {"n": 0}

    def guard():
        sleeps["n"] += 1
        if sleeps["n"] > 5000:                       # an unattended run that waits for the operator would loop here for ever
            raise Stop("the run waited for the operator")
    r.on_sleep = guard
    yield r
    r.close()


def cache_of(rig):
    return rig.tmp / "cache"                          # RunPaths: the run directory's sibling `cache`


def decided(rig, what):
    return [e["data"] for e in rig.entries() if e["data"].get("decision") == what]


def as_code(monkeypatch, code):
    """The first run pauses once before stage 1 (stage 0 has run, so the coder is up) and the second once before stage 3, with `code` as the
    block code of both. The retry therefore runs stage 1, which needs the coder, and runs stages 1 and 2 (which need the coder at two points) and ends blocked again."""
    left = {1, 3}
    real_cap = supervisor.budget_cap

    def cap(p, now):
        if p.next_stage in left and (p.next_stage == 1 or 1 not in left):
            left.discard(p.next_stage)
            return f"test pause before stage {p.next_stage}"
        return real_cap(p, now)
    monkeypatch.setattr(supervisor, "budget_cap", cap)
    monkeypatch.setattr(supervisor, "block_code", lambda pause: code)
    monkeypatch.setattr(supervisor, "_test_block_code_active", True, raising=False)


def test_an_agent_stuck_block_leaves_the_coder_up_when_asked_to(rig, monkeypatch):
    rig.args.coder_keep_on_stall = True
    as_code(monkeypatch, "agent-stuck")
    assert rig.run() == EXIT_BLOCKED
    assert rig.m.coder_running and rig.m.coder_starts == 1
    left = decided(rig, "coder left up")
    assert left and left[0]["holder_pids"] == [KEPT_PID]
    assert decided(rig, "coder kept up across retry")
    rec = kept_coder.read(cache_of(rig))
    assert rec is not None and rec["reason"] == "agent-stuck" and rec["run_dir"] == str(rig.run_dir)


def test_the_retry_reuses_the_kept_coder_without_a_boot(rig, monkeypatch):
    rig.args.coder_keep_on_stall = True
    as_code(monkeypatch, "agent-stuck")
    rig.run()
    assert rig.run(pid=200) == EXIT_BLOCKED          # the second pause
    assert rig.m.coder_starts == 1, "the retry booted a second coder"
    assert len(decided(rig, "coder adopted")) == 1 and len(decided(rig, "coder reused after retry (canary ok)")) == 1, \
        "the coder is adopted once per retry, not at every stage"
    assert not decided(rig, "coder restarted (canary failed)")


def test_a_kept_coder_whose_canary_changed_is_stopped_and_booted_again(rig, monkeypatch):
    rig.args.coder_keep_on_stall = True
    as_code(monkeypatch, "agent-stuck")
    rig.run()
    first = kept_coder.read(cache_of(rig))["lease"]["lease_id"]
    rig.m.coder_answer = "a different answer"
    rig.run(pid=200)
    restarted = decided(rig, "coder restarted (canary failed)")
    assert restarted and "canary" in restarted[0]["reason"]
    assert rig.m.coder_starts == 2
    assert first not in rig.m.leases and ("release", first) in rig.calls, "the kept lease was not released before the new boot"
    assert not decided(rig, "coder reused after retry (canary ok)")


def test_a_kept_coder_that_stopped_serving_is_booted_again(rig, monkeypatch):
    rig.args.coder_keep_on_stall = True
    as_code(monkeypatch, "agent-stuck")
    rig.run()
    rig.m.coder_running = False                       # the container died while the supervisor was away
    rig.run(pid=200)
    assert decided(rig, "coder restarted (canary failed)") or decided(rig, "kept coder not used")
    assert rig.m.coder_starts == 2


@pytest.mark.parametrize("code", ["stage-failed", "needs-new-model-code", "hardware-unhealthy"])
def test_any_other_block_code_stops_the_coder_as_before(rig, monkeypatch, code):
    rig.args.coder_keep_on_stall = True
    as_code(monkeypatch, code)
    assert rig.run() == EXIT_BLOCKED
    assert not rig.m.coder_running and not decided(rig, "coder left up")
    assert kept_coder.read(cache_of(rig)) is None


def test_without_the_flag_an_agent_stuck_block_stops_the_coder(rig, monkeypatch):
    as_code(monkeypatch, "agent-stuck")
    assert rig.run() == EXIT_BLOCKED
    assert not rig.m.coder_running and not decided(rig, "coder left up")


def test_a_kept_coder_of_another_run_is_never_stopped_by_the_stall_path(rig, monkeypatch):
    rig.args.coder_keep_on_stall = True
    as_code(monkeypatch, "agent-stuck")
    rig.run()
    first = kept_coder.read(cache_of(rig))["lease"]["lease_id"]
    other = rig.tmp / "other-run"
    rig.run_dir, rig.args.run_dir = other, str(other)
    rig.m.coder_answer = "a different answer"
    try:
        rig.run(pid=200)
    except Exception:
        pass                                          # what the other run does about the busy port is not under test
    assert not decided(rig, "coder restarted (canary failed)"), "the stall path touched a coder that another run left up"
    assert first in rig.m.leases and rig.m.coder_running


def test_the_flag_is_off_by_default(tmp_path):
    from run_fakes import argv
    a = argv(tmp_path, tmp_path / "t.toml", "http://127.0.0.1:8000/v1")
    assert supervisor.parse(a).coder_keep_on_stall is False
    assert supervisor.parse(a + ["--coder-keep-on-stall"]).coder_keep_on_stall is True
