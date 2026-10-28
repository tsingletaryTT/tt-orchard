# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The supervisor stops only a coder container it started (issue 154).

A `tt-model serve` refused with "already running" means another process owns that container. The
cleanup after the blocked start must make no `tt-model stop` call. A coder this run started is
stopped, also after a crash and a resume that reads the ledger.
"""
import pytest

from fakes import LEASE, FakeClock, FakeRun
from run_fakes import MachineCoder
from orchard.server import ServerControl, ServerNotOwned, ServerSpec, StopCheck
from orchard.stages import coder_state
from test_supervisor import Rig, never_wait

SPEC = ServerSpec("episod/qwen-coder", "container", 8000, "qwen", profile="default")
SS_HEADER = "State  Recv-Q Send-Q Local Address:Port  Peer Address:Port Process\n"
DOCKER_LINE = "3c1d2e4f5a6b f0ed6056d85f tt-model-qwen-coder-default 0.0.0.0:8000->8000/tcp\n"


def key(argv):
    return "docker ps-q" if argv[:3] == ["docker", "ps", "-q"] else " ".join(argv[:2])


def server(serve, docker_ps=("", ""), **over):
    """A real ServerControl over scripted commands. `serve` is the answer to `tt-model serve`."""
    script = {"tt-model serve": [serve], "tt-model stop": [(0, "stopped 1 container(s)", "")],
              "docker ps": [(0, docker_ps[0], "")], "docker ps-q": [(0, "", "")],
              "ss -ltn": [(0, SS_HEADER, "")], "curl -sS": [(7, "", "refused")], **over}
    run = FakeRun(script, key=key)
    clock = FakeClock()
    return ServerControl(SPEC, run=run, clock=clock, sleep=clock.sleep), run


def stops(run):
    return [a for a in run.argvs() if a[:2] == ["tt-model", "stop"]]


def test_a_refused_start_marks_the_container_as_not_ours_and_stop_calls_nothing():
    ctl, run = server((1, "", "tt-model-qwen-coder-default is already running"), docker_ps=(DOCKER_LINE, ""))
    with pytest.raises(ServerNotOwned, match="another process owns it"):
        ctl.start(LEASE)
    assert ctl.owned is False and ctl.record()["owned"] is False
    ctl.stop()
    assert stops(run) == []


def test_a_start_that_fails_for_another_reason_is_not_called_foreign():
    ctl, run = server((1, "", "no such package"))
    with pytest.raises(Exception, match="no such package") as info:
        ctl.start(LEASE)
    assert not isinstance(info.value, ServerNotOwned)
    assert ctl.owned is None


def test_a_started_coder_is_stopped_through_the_recorded_ownership():
    ctl, run = server((0, "", ""))
    ctl.start(LEASE)
    record = ctl.record()
    assert record["owned"] is True
    # A fresh process reads the record (a resume after a crash) and still stops the container.
    other, run2 = server((0, "", ""))
    other.adopt(record)
    other.stop()
    assert len(stops(run2)) == 1


def test_a_not_owned_record_stays_not_owned_after_adopt():
    ctl, run = server((0, "", ""))
    ctl.adopt({"kind": "container", "target": SPEC.target, "port": 8000, "owned": False})
    ctl.stop()
    assert stops(run) == []


def test_a_record_without_the_field_is_still_stopped():
    # Ledgers written before the field existed keep working.
    ctl, run = server((0, "", ""))
    ctl.adopt({"kind": "container", "target": SPEC.target, "port": 8000})
    ctl.stop()
    assert len(stops(run)) == 1


def test_coder_state_reads_ownership_from_the_not_started_entry():
    entries = [
        {"event": "decision", "data": {"decision": "coder starting", "lease": {"lease_id": "L1"},
                                       "server": {"owned": None}}},
        {"event": "decision", "data": {"decision": "coder not started", "server": {"owned": False}}},
    ]
    assert coder_state(entries)[1] == {"owned": False}
    entries = [{"event": "decision", "data": {"decision": "coder container started",
                                              "server": {"owned": True}}}]
    assert coder_state(entries)[1] == {"owned": True}


# ---- the supervisor, end to end ---------------------------------------------------------------


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def test_the_blocked_run_leaves_the_other_processes_container_alone(rig):
    made = []

    def factory(machine):
        made.append(foreign_coder(machine))
        return made[0]

    rig.coder_cls = factory
    rig.args.unattended = True
    never_wait(rig)
    assert rig.run() == 5          # EXIT_BLOCKED
    notes = [e["data"] for e in rig.entries() if e["event"] == "notice" and e["data"].get("blocked")]
    assert any("another process owns it" in n.get("reason", "") for n in notes)
    assert stops(made[0].run) == []
    d = [x["decision"] for x in rig.decisions()]
    assert "stopping the coder" not in d
    # the lease went back, and with no reset: the foreign container may use those chips
    released = [x for x in rig.decisions() if x["decision"] == "hardware released"]
    assert released and released[-1]["reset"] is False and rig.m.leases == {}
    assert rig.m.resets == []


def test_a_coder_this_run_started_is_stopped_at_the_end(rig):
    rig.run()
    d = [x["decision"] for x in rig.decisions()]
    assert "coder container started" in d
    assert rig.m.coder_running is False


class HybridCoder(MachineCoder):
    """The fake machine for readiness and chips; the real ServerControl for ownership, the ledger
    record and the stop call, which goes to a scripted `tt-model`."""

    def __init__(self, machine, serve=(0, "", ""), docker_ps=("", "")):
        super().__init__(machine)
        self.ctl, self.run = server(serve, docker_ps=docker_ps)

    @property
    def owned(self):
        return self.ctl.owned

    @owned.setter
    def owned(self, value):
        self.ctl.owned = value

    def start(self, lease):
        self.ctl.start(lease)
        super().start(lease)

    def stop(self):
        out = self.ctl.stop()
        if not out.get("skipped"):
            super().stop()
        return out

    def confirm_stopped(self):
        # Stopped only when docker (scripted) and the fake machine both show nothing running.
        mine, theirs = self.ctl.confirm_stopped(), super().confirm_stopped()
        return StopCheck(mine.stopped and theirs.stopped, {**mine.checks, **theirs.checks}, {})

    def record(self):
        return self.ctl.record()

    def adopt(self, record):
        return self.ctl.adopt(record)


def test_a_resume_after_a_crash_stops_the_container_the_ledger_says_this_run_started(rig):
    from fakes import Crash
    first = []

    def one(machine):
        first.append(HybridCoder(machine))
        return first[-1]

    rig.coder_cls = one
    with pytest.raises(Crash):
        rig.run(pid=100, crash_if=lambda e: e["data"].get("decision") == "coder container started")
    started = [d for d in rig.decisions() if d["decision"] == "coder container started"]
    assert started[0]["server"]["owned"] is True
    assert stops(first[0].run) == []
    second = []

    def two(machine):
        second.append(HybridCoder(machine))
        return second[-1]

    rig.coder_cls = two
    rig.args.unattended = True
    never_wait(rig)
    rig.run(pid=200)
    assert len(stops(second[0].run)) >= 1          # the old container went down by the ledger's record


def test_a_resume_after_a_refused_start_never_stops_the_other_container(rig):
    made = []

    def factory(machine):
        made.append(HybridCoder(machine, serve=(1, "", "tt-model-qwen-coder-default is already running"),
                                  docker_ps=(DOCKER_LINE, "")))
        return made[-1]

    rig.coder_cls = factory
    rig.args.unattended = True
    never_wait(rig)
    assert rig.run(pid=100) == 5
    assert rig.run(pid=200) == 5                  # the operator resumes; the container still runs
    assert all(stops(c.run) == [] for c in made)
    blocks = [d for d in rig.decisions() if d["decision"] == "blocked"]
    assert len(blocks) == 2 and all("another process owns it" in b["reason"] for b in blocks)


def test_the_ledger_free_release_does_not_stop_or_reset_a_container_that_is_not_ours(rig):
    from orchard import supervisor
    from orchard.ledger import Ledger
    rig.coder_cls = foreign_coder
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        sup = supervisor.build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=100),
                               coder=rig.coder_cls(rig.m), versions={"tt_model": "test"},
                               clock=rig.clock, sleep=rig.sleep, disk_usage=lambda p: rig.usage(p),
                               home=rig.home, containers=rig.containers)
        sup.coder_lease = sup.adapter.acquire(2, "orchard:supervisor", "coder")
        with pytest.raises(ServerNotOwned):
            sup.coder.start(sup.coder_lease)
        assert sup._release_from_memory() is True
    assert stops(sup.coder.run) == [] and rig.m.leases == {} and rig.m.resets == []


# ---- the first review of PR 158 ---------------------------------------------------------------


def test_a_live_refusal_is_not_overwritten_by_an_older_ledger_record():
    ctl, run = server((1, "", "tt-model-qwen-coder-default is already running"), docker_ps=(DOCKER_LINE, ""))
    with pytest.raises(ServerNotOwned):
        ctl.start(LEASE)
    ctl.adopt({"kind": "container", "target": SPEC.target, "port": 8000, "owned": None})
    ctl.stop()
    assert ctl.owned is False and stops(run) == []


def test_a_new_start_attempt_forgets_an_earlier_refusal():
    ctl, run = server((0, "", ""))
    ctl.owned = False                       # an earlier attempt was refused
    ctl.start(LEASE)
    assert ctl.owned is True


def test_a_failed_retry_does_not_keep_an_earlier_refusal():
    ctl, run = server((1, "", "no such package"))
    ctl.owned = False
    with pytest.raises(Exception, match="no such package"):
        ctl.start(LEASE)
    assert ctl.owned is None


def test_the_pre_start_record_of_a_retry_does_not_carry_the_old_refusal(rig):
    from orchard import supervisor
    from orchard.ledger import Ledger
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        coder = HybridCoder(rig.m)
        coder.ctl.owned = False
        sup = supervisor.build(rig.args, led, adapter=rig.adapter_cls(rig.m, owner_pid=100), coder=coder,
                               versions={"tt_model": "test"}, clock=rig.clock, sleep=rig.sleep,
                               disk_usage=lambda p: rig.usage(p), home=rig.home, containers=rig.containers)
        sup._start_coder(None)
        starting = [e["data"] for e in led.read() if e["data"].get("decision") == "coder starting"]
    assert starting[0]["server"]["owned"] is None


def test_the_single_tenant_adapter_accepts_keep_running():
    from orchard.adapters.single_tenant import SingleTenantAdapter
    import inspect
    assert "keep_running" in inspect.signature(SingleTenantAdapter.release).parameters


def test_a_resume_that_still_holds_the_lease_does_not_adopt_the_other_processes_server(rig):
    """An attended run keeps its lease after a refused start. A resume must not pass the other
    container off as this run's coder because it answers."""
    from orchard import supervisor
    from orchard.ledger import Ledger
    rig.run_dir.mkdir(parents=True, exist_ok=True)
    rig.m.coder_running, rig.m.coder_chips = True, ()        # the other process's server answers
    with Ledger(rig.run_dir / "ledger.jsonl") as led:
        coder = HybridCoder(rig.m, serve=(1, "", "tt-model-qwen-coder-default is already running"),
                            docker_ps=(DOCKER_LINE, ""))
        adapter = rig.adapter_cls(rig.m, owner_pid=100)
        sup = supervisor.build(rig.args, led, adapter=adapter, coder=coder, versions={"tt_model": "test"},
                               clock=rig.clock, sleep=rig.sleep, disk_usage=lambda p: rig.usage(p),
                               home=rig.home, containers=rig.containers)
        lease = adapter.acquire(2, "orchard:supervisor", "coder")
        led.append("decision", None, decision="coder starting", lease=lease.record(),
                   server={**SPEC_RECORD, "owned": None})
        led.append("decision", None, decision="coder not started", lease_id=lease.lease_id,
                   server={**SPEC_RECORD, "owned": False}, reason="owned elsewhere")
        with pytest.raises(supervisor.Blocked):
            sup._ensure_coder()
    assert rig.m.asked == [] and stops(coder.run) == []


def test_a_refused_restore_is_recorded_so_recovery_respects_it(tmp_path):
    from fakes import make_handoff
    from orchard.handoff import progress
    h, world, ledger = make_handoff(tmp_path)
    h.park()

    def refused(lease):
        h.server.owned = False
        raise ServerNotOwned("the container belongs to someone else")

    world.coder_running = False
    h.server.start = refused
    h.server.record = lambda: {"kind": "container", "owned": h.server.owned}
    with pytest.raises(Exception, match="another|belongs|failed"):
        h.restore()
    assert progress(ledger.read()).server["owned"] is False


SPEC_RECORD = {"kind": "container", "target": SPEC.target, "port": 8000}


def foreign_coder(machine):
    """A coder whose serve is refused because another process's container of that name runs."""
    return HybridCoder(machine, serve=(1, "", "tt-model-qwen-coder-default is already running"),
                       docker_ps=(DOCKER_LINE, ""))
